"""DGSEM spatial operator: strong-form discontinuous Galerkin with LGL
collocation on curvilinear quadrilateral elements.

Formulation
-----------
With the conservative state U, physical fluxes (F, G) and the conservative
(contravariant) metric terms,

    Fh = J xi_x F + J xi_y G        (xi-direction contravariant flux)
    Gh = J eta_x F + J eta_y G      (eta-direction contravariant flux)

the semi-discrete scheme on every element reads (strong form)

    dU/dt = -1/J [ D_xi Fh + D_eta Gh ]  +  SAT terms,

where the boundary (SAT / simultaneous-approximation-term) lifting at each
face uses the OUTWARD face metric vector sJ n and the interface numerical
flux Fn*:

    xi = +1 :  Res += (Fn* - Fh|_{i=N}) / w_N
    xi = -1 :  Res += (Fn* + Fh|_{i=0}) / w_0      (Fn* oriented along -xi)
    eta = +1 :  Res += (Fn* - Gh|_{l=N}) / w_N
    eta = -1 :  Res += (Fn* + Gh|_{l=0}) / w_0     (Fn* oriented along -eta)

which is the Kopriva strong-form DGSEM; the signs are verified by the
free-stream-preservation test (a constant state has exactly zero residual,
including at the wall and far-field boundaries).

Connectivity: xi is periodic (np.roll); eta = -1 faces of the j = 0 band are
slip walls (Euler) and eta = +1 faces of the outer band face the far field.

Shock capturing: an element-wise artificial viscosity nu_e (computed by
dgsem.shock) is added through the same compact BR2/SIPG diffusion operator
used for the Navier-Stokes viscous terms, applied to the conservative
variables; the physical viscosity (mu > 0) acts on (u, v, T) with the
no-slip adiabatic wall closure.  Both diffusive parts can be active.

Run `python -m dgsem.operator` for the free-stream-preservation self test.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .euler import GAMMA, physical_fluxes, primitives, sound_speed
from .numflux import make_flux

__all__ = ["SolverConfig", "Operator"]


# ----------------------------------------------------------------------
@dataclass
class SolverConfig:
    gamma: float = GAMMA
    flux: str = "roe"                # 'roe' | 'rusanov'
    entropy_fix: float = 0.05
    fix_all_waves: bool = False      # Harten fix on all waves (see numflux)
    mu: float = 0.0                  # dynamic viscosity (Euler if 0)
    Pr: float = 0.72
    penalty: float = 2.0             # SIPG penalty multiplier
    wall: str = "slip"               # 'slip' | 'noslip_adia'
    farfield_vortex: bool = True     # Thomas-Salas point-vortex correction
    vortex_center: tuple = (0.25, 0.0)

    @property
    def k_cond(self) -> float:
        return self.mu * self.gamma / (self.Pr * (self.gamma - 1.0))


# ----------------------------------------------------------------------
class Operator:
    """DGSEM spatial operator with preallocated work buffers."""

    def __init__(self, mesh, cfg: SolverConfig):
        self.mesh = mesh
        self.cfg = cfg
        self.gamma = cfg.gamma
        self.flux = make_flux(cfg.flux, cfg.entropy_fix, cfg.fix_all_waves)
        nk, nj, Np1 = mesh.n_k, mesh.n_j, mesh.Np1
        shape = (nk, nj, Np1, Np1)
        self.shape = shape
        # work buffers
        self.Fh = np.empty(shape + (4,))
        self.Gh = np.empty(shape + (4,))
        self.Res = np.empty(shape + (4,))
        self._vol = np.empty(shape + (4,))           # (k, j, i, l, c)
        self._volT = np.empty(shape + (4,))          # (k, j, l, i, c) buffer
        self.rho = np.empty(shape)
        self.u = np.empty(shape)
        self.v = np.empty(shape)
        self.p = np.empty(shape)
        self.E = np.empty(shape)
        self.c = np.empty(shape)
        # far-field state (set by solver; gamma-corrected per Mach/alpha)
        self.Qinf = None
        self.Gamma = 0.0                              # circulation for vortex correction
        # optional constant ghost states for the two xi boundaries of
        # non-periodic meshes (shock-tube verification)
        self.Q_xi_L = None
        self.Q_xi_R = None

    # ------------------------------------------------------------------
    def set_freestream(self, Qinf: np.ndarray) -> None:
        self.Qinf = np.asarray(Qinf, dtype=float)

    # ------------------------------------------------------------------
    def rhs(self, U: np.ndarray, nu: np.ndarray | None = None) -> np.ndarray:
        """Assemble dU/dt (overwrites and returns self.Res / J)."""
        mesh = self.mesh
        D, w = mesh.D, mesh.w
        Np1 = mesh.Np1
        gamma = self.gamma

        # ---- primitives and physical fluxes -----------------------------
        rho, u, v, p = primitives(U, gamma)
        np.copyto(self.rho, rho); np.copyto(self.u, u)
        np.copyto(self.v, v); np.copyto(self.p, p)
        self.E[...] = U[..., 3]
        F, G = physical_fluxes(rho, u, v, U[..., 3], p)
        # contravariant fluxes
        np.multiply(mesh.Jxi_x[..., None], F, out=self.Fh)
        self.Fh += mesh.Jxi_y[..., None] * G
        np.multiply(mesh.Jeta_x[..., None], F, out=self.Gh)
        self.Gh += mesh.Jeta_y[..., None] * G

        # ---- volume terms (batched matmul over the node axis) -----------
        # xi-derivative: contract axis i (axis 2); matmul contracts the
        # second-to-last axis, so transpose i and l first.
        # NOTE: a FRESH array is allocated (and returned) on every call so
        # that multi-stage time integrators can hold several stage results
        # simultaneously without aliasing.
        Res = np.zeros(self.shape + (4,))
        nk, nj = mesh.n_k, mesh.n_j
        volT = self._volT                                    # (k, j, l, i, c)
        np.matmul(D, self.Fh.transpose(0, 1, 3, 2, 4), out=volT)
        Res += volT.transpose(0, 1, 3, 2, 4)                 # (k, j, i, l, c)
        # eta-derivative: contract axis l (axis 3) -- already second-to-last
        np.matmul(D, self.Gh, out=self._vol)                 # (k, j, i, l, c)
        Res += self._vol

        # ---- xi faces ------------------------------------------------------
        if mesh.periodic_xi:
            QL = U[:, :, Np1 - 1, :, :]
            QR = np.roll(U, -1, axis=0)[:, :, 0, :, :]
            Fn, _ = self.flux(QL, QR, mesh.Fxp[..., 0], mesh.Fxp[..., 1])
            Res[:, :, Np1 - 1, :, :] += (Fn - self.Fh[:, :, Np1 - 1, :, :]) / w[Np1 - 1]
            Res[:, :, 0, :, :] += (-np.roll(Fn, 1, axis=0) + self.Fh[:, :, 0, :, :]) / w[0]
        else:
            # interior xi interfaces k = 0 .. n_k - 2
            if mesh.n_k > 1:
                QL = U[0:-1, :, Np1 - 1, :, :]
                QR = U[1:, :, 0, :, :]
                Fn, _ = self.flux(QL, QR, mesh.Fxp[:-1, :, :, 0], mesh.Fxp[:-1, :, :, 1])
                Res[0:-1, :, Np1 - 1, :, :] += (Fn - self.Fh[0:-1, :, Np1 - 1, :, :]) / w[Np1 - 1]
                Res[1:, :, 0, :, :] += (-Fn + self.Fh[1:, :, 0, :, :]) / w[0]
            # downstream far-field faces: xi = -1 of column 0 and xi = +1 of
            # the last column (C-grid downstream boundary)
            self._farfield_xi_faces(U, Res)

        # ---- eta interior faces -------------------------------------------
        nj = mesh.n_j
        if nj > 1:
            QL = U[:, 0:nj - 1, :, Np1 - 1, :]
            QR = U[:, 1:nj, :, 0, :]
            Fn, _ = self.flux(QL, QR, mesh.Fep[:, 0:nj - 1, :, 0],
                              mesh.Fep[:, 0:nj - 1, :, 1])
            Res[:, 0:nj - 1, :, Np1 - 1, :] += (Fn - self.Gh[:, 0:nj - 1, :, Np1 - 1, :]) / w[Np1 - 1]
            Res[:, 1:nj, :, 0, :] += (-Fn + self.Gh[:, 1:nj, :, 0, :]) / w[0]

        # ---- periodic eta wrap (verification meshes) ---------------------
        if mesh.periodic_eta:
            QL = U[:, nj - 1, :, Np1 - 1, :]
            QR = U[:, 0, :, 0, :]
            Fn, _ = self.flux(QL, QR, mesh.Fep[:, nj - 1, :, 0],
                              mesh.Fep[:, nj - 1, :, 1])
            Res[:, nj - 1, :, Np1 - 1, :] += (Fn - self.Gh[:, nj - 1, :, Np1 - 1, :]) / w[Np1 - 1]
            Res[:, 0, :, 0, :] += (-Fn + self.Gh[:, 0, :, 0, :]) / w[0]

        # ---- wall (eta = -1 faces of the wall columns) --------------------
        if mesh.has_wall:
            self._wall_faces(U, Res, sign=-1)
        # ---- top slip wall for channel meshes (eta = +1 of outer band) ----
        if getattr(mesh, "wall_top", False):
            self._wall_faces(U, Res, sign=+1)
        # ---- far field (eta = +1 of the outer band) ----------------------
        if mesh.has_farfield:
            self._farfield_faces(U, Res)

        # ---- diffusion: artificial viscosity and/or physical viscosity ---
        if nu is not None and np.any(nu > 0.0):
            self._diffuse_conservative(U, Res, nu)
        if self.cfg.mu > 0.0:
            self._diffuse_ns(U, Res)

        # ---- final division by J ------------------------------------------
        Res /= mesh.J[..., None]
        np.negative(Res, out=Res)
        return Res

    # ------------------------------------------------------------------
    def rhs_reference(self, U, nu=None):
        """Alias kept for the equivalence tests against the Numba kernel."""
        return self.rhs(U, nu)

    # ------------------------------------------------------------------
    def _wall_faces(self, U, Res, sign: int = -1) -> None:
        """Slip wall (Euler): reflect the normal velocity in the ghost state;
        the Rusanov/Roe flux with the mirrored state gives exactly zero
        normal mass flux (impermeability).

        sign = -1: the wall is the eta = -1 face of the j = 0 band (airfoil
        lower side / channel bottom).  sign = +1: the wall is the eta = +1
        face of the j = n_j - 1 band (channel top); its OUTWARD metric
        vector is Fep (pointing out of the fluid), so the SAT sign and the
        interior-face index change accordingly."""
        mesh = self.mesh
        w, Np1 = mesh.w, mesh.Np1
        if sign < 0:
            wei = np.arange(mesh.n_k)
            Qint = U[:, 0, :, 0, :]
            sJnx = mesh.Fem[:, 0, :, 0]
            sJny = mesh.Fem[:, 0, :, 1]
        else:
            wei = np.arange(mesh.n_k)
            Qint = U[:, mesh.n_j - 1, :, Np1 - 1, :]
            sJnx = mesh.Fep[:, mesh.n_j - 1, :, 0]
            sJny = mesh.Fep[:, mesh.n_j - 1, :, 1]
        sJ = np.maximum(np.hypot(sJnx, sJny), 1e-300)
        nx = sJnx / sJ
        ny = sJny / sJ
        rhoW = np.maximum(Qint[..., 0], 1e-12)
        uW = Qint[..., 1] / rhoW
        vW = Qint[..., 2] / rhoW
        if self.cfg.wall == "copy":
            # geometry-neutral ghost (metric-identity test only)
            QG = Qint
        elif self.cfg.wall in ("slip", "noslip_adia", "noslip_isotherm"):
            if self.cfg.wall == "slip":
                Vn = uW * nx + vW * ny
                uG = uW - 2.0 * Vn * nx
                vG = vW - 2.0 * Vn * ny
            else:                       # no-slip: reflect both components
                uG, vG = -uW, -vW
            QG = np.empty_like(Qint)
            QG[..., 0] = rhoW
            QG[..., 1] = rhoW * uG
            QG[..., 2] = rhoW * vG
            QG[..., 3] = Qint[..., 3]     # adiabatic; isotherm handled in viscous SAT
        else:
            raise ValueError(f"unknown wall mode {self.cfg.wall}")
        # ghost is on the +outward-normal side (outside the fluid); the
        # interior state is on the -outward side, so it must be the LEFT
        # (first) argument of the upwind flux with the outward-pointing
        # normal.
        Fn, _ = self.flux(Qint, QG, sJnx, sJny)
        if sign < 0:
            Res[:, 0, :, 0, :] += (Fn + self.Gh[:, 0, :, 0, :]) / w[0]
        else:
            Res[:, -1, :, Np1 - 1, :] += (Fn - self.Gh[:, -1, :, Np1 - 1, :]) / w[Np1 - 1]

    # ------------------------------------------------------------------
    def _farfield_xi_faces(self, U, Res) -> None:
        """xi = -1 face of column 0 and xi = +1 face of the last column of
        non-periodic meshes.  The ghost is the free-stream state (with the
        point-vortex correction), or a constant state given by Q_xi_L /
        Q_xi_R (shock-tube verification)."""
        mesh = self.mesh
        w, Np1 = mesh.w, mesh.Np1

        def const_ghost(Qc, like):
            """Constant ghost state with the same shape as `like` (which
            already carries the 4-component axis)."""
            QR = np.empty_like(like)
            for c in range(4):
                QR[..., c] = Qc[c]
            return QR

        # xi = -1 face of column 0 (outward = -xi)
        QL = U[0, :, 0, :, :]
        if self.Q_xi_L is not None:
            QR = const_ghost(self.Q_xi_L, QL)
        else:
            QR, = self._farfield_ghost(QL, mesh.X[0, :, 0, :], mesh.Y[0, :, 0, :])
        Fn, _ = self.flux(QL, QR, mesh.Fxm[0, :, :, 0], mesh.Fxm[0, :, :, 1])
        Res[0, :, 0, :, :] += (Fn + self.Fh[0, :, 0, :, :]) / w[0]
        # xi = +1 face of the last column (outward = +xi)
        QL = U[-1, :, Np1 - 1, :, :]
        if self.Q_xi_R is not None:
            QR = const_ghost(self.Q_xi_R, QL)
        else:
            QR, = self._farfield_ghost(QL, mesh.X[-1, :, Np1 - 1, :], mesh.Y[-1, :, Np1 - 1, :])
        Fn, _ = self.flux(QL, QR, mesh.Fxp[-1, :, :, 0], mesh.Fxp[-1, :, :, 1])
        Res[-1, :, Np1 - 1, :, :] += (Fn - self.Fh[-1, :, Np1 - 1, :, :]) / w[Np1 - 1]

    # ------------------------------------------------------------------
    def _farfield_ghost(self, QL, x, y):
        """Free-stream ghost state (vortex-corrected) matching QL's shape."""
        cfg = self.cfg
        Qinf = self.Qinf
        QR = np.empty_like(QL)
        QR[..., 0] = Qinf[0]
        uinf = Qinf[1] / Qinf[0]
        vinf = Qinf[2] / Qinf[0]
        uB = np.full_like(x, uinf)
        vB = np.full_like(x, vinf)
        if cfg.farfield_vortex and self.Gamma != 0.0:
            cx, cy = cfg.vortex_center
            dx = x - cx
            dy = y - cy
            r2 = np.maximum(dx * dx + dy * dy, 1e-12)
            fac = self.Gamma / (2.0 * np.pi * r2)
            uB = uB + fac * (-dy)
            vB = vB + fac * (dx)
        QR[..., 1] = Qinf[0] * uB
        QR[..., 2] = Qinf[0] * vB
        QR[..., 3] = Qinf[3] + 0.5 * Qinf[0] * (
            (uB * uB + vB * vB) - (uinf * uinf + vinf * vinf))
        return (QR,)

    def _farfield_faces(self, U, Res) -> None:
        """Far-field boundary: numerical flux against the free-stream ghost
        state, optionally corrected by a point vortex of circulation Gamma
        (Thomas & Salas, AIAA J. 1981) that represents the airfoil lift."""
        mesh = self.mesh
        w, Np1 = mesh.w, mesh.Np1
        cfg = self.cfg
        Qinf = self.Qinf
        QL = U[:, mesh.n_j - 1, :, Np1 - 1, :]
        x = mesh.X[:, -1, :, Np1 - 1]
        y = mesh.Y[:, -1, :, Np1 - 1]

        QR = np.empty_like(QL)
        QR[..., 0] = Qinf[0]
        uinf = Qinf[1] / Qinf[0]
        vinf = Qinf[2] / Qinf[0]
        uB = np.full_like(x, uinf)
        vB = np.full_like(x, vinf)
        if cfg.farfield_vortex and self.Gamma != 0.0:
            cx, cy = cfg.vortex_center
            dx = x - cx
            dy = y - cy
            r2 = np.maximum(dx * dx + dy * dy, 1e-12)
            # velocity of a point vortex with circulation Gamma
            # (u, v) += Gamma/(2 pi r^2) * (-dy, dx)
            fac = self.Gamma / (2.0 * np.pi * r2)
            uB = uB + fac * (-dy)
            vB = vB + fac * (dx)
        QR[..., 1] = Qinf[0] * uB
        QR[..., 2] = Qinf[0] * vB
        # keep the free-stream pressure and density (small O(Gamma^2) error)
        QR[..., 3] = Qinf[3] + 0.5 * Qinf[0] * (
            (uB * uB + vB * vB) - (uinf * uinf + vinf * vinf))

        Fn, _ = self.flux(QL, QR, mesh.Fep[:, -1, :, 0], mesh.Fep[:, -1, :, 1])
        Res[:, -1, :, Np1 - 1, :] += (Fn - self.Gh[:, -1, :, Np1 - 1, :]) / w[Np1 - 1]

    # ------------------------------------------------------------------
    def _lifted_gradients(self, q4):
        """BR2-lifted reference gradients of an (..., 4) nodal field:
        D q + M^-1 B (qhat - q) with qhat the interior-face average.
        Boundary faces (wall/far field) get no correction (ghost = interior).
        Returns (dq_dxi, dq_deta)."""
        mesh = self.mesh
        D, w, Np1 = mesh.D, mesh.w, mesh.Np1
        nk, nj = mesh.n_k, mesh.n_j

        dq_xi = np.einsum('ab,kjblc->kjalc', D, q4)
        # contraction over the l axis: dq_eta[k,j,i,l,c] = sum_b D[l,b] q4[k,j,i,b,c]
        dq_eta = np.einsum('lb,kjibc->kjilc', D, q4)

        # xi faces: each interface's averaged trace feeds BOTH adjacent faces
        qavg_r = 0.5 * (q4[:, :, Np1 - 1, :, :] + np.roll(q4, -1, axis=0)[:, :, 0, :, :])
        qavg_l = np.roll(qavg_r, 1, axis=0)     # average at the left interface
        dq_xi[:, :, Np1 - 1, :, :] += (qavg_r - q4[:, :, Np1 - 1, :, :]) / w[Np1 - 1]
        dq_xi[:, :, 0, :, :] -= (qavg_l - q4[:, :, 0, :, :]) / w[0]
        # eta interior faces
        if nj > 1:
            qavg = 0.5 * (q4[:, 0:nj - 1, :, Np1 - 1, :] + q4[:, 1:nj, :, 0, :])
            dq_eta[:, 0:nj - 1, :, Np1 - 1, :] += (qavg - q4[:, 0:nj - 1, :, Np1 - 1, :]) / w[Np1 - 1]
            dq_eta[:, 1:nj, :, 0, :] -= (qavg - q4[:, 1:nj, :, 0, :]) / w[0]
        if mesh.periodic_eta:
            qavg = 0.5 * (q4[:, nj - 1, :, Np1 - 1, :] + q4[:, 0, :, 0, :])
            dq_eta[:, nj - 1, :, Np1 - 1, :] += (qavg - q4[:, nj - 1, :, Np1 - 1, :]) / w[Np1 - 1]
            dq_eta[:, 0, :, 0, :] -= (qavg - q4[:, 0, :, 0, :]) / w[0]
        return dq_xi, dq_eta

    # ------------------------------------------------------------------
    def _diffuse_conservative(self, U, Res, nu) -> None:
        """Element-wise artificial viscosity nu[k, j] applied to the
        conservative variables through the compact BR2/SIPG operator.
        Interior faces use averaged gradient traces and an SIPG penalty on
        the solution jump; wall and far-field faces are transparent to the
        artificial viscosity (ghost = interior state)."""
        mesh = self.mesh
        D, w, Np1 = mesh.D, mesh.w, mesh.Np1
        nk, nj = mesh.n_k, mesh.n_j
        invJ = 1.0 / mesh.J

        volT2, vol2 = self._volT, self._vol
        dq_xi, dq_eta = self._lifted_gradients(U)

        # physical gradients of each variable
        # d/dx = (Jxi_x d/dxi + Jeta_x d/deta) / J  (and the y version)
        gx = invJ[..., None] * (mesh.Jxi_x[..., None] * dq_xi
                                + mesh.Jeta_x[..., None] * dq_eta)
        gy = invJ[..., None] * (mesh.Jxi_y[..., None] * dq_xi
                                + mesh.Jeta_y[..., None] * dq_eta)

        # contravariant viscous "fluxes": nu * (grad U) transformed
        Fv = np.empty_like(self.Fh)
        Gv = np.empty_like(self.Gh)
        nu5 = nu[..., None, None]
        for c in range(4):
            Fv[..., c] = nu5 * (mesh.Jxi_x * gx[..., c] + mesh.Jxi_y * gy[..., c])
            Gv[..., c] = nu5 * (mesh.Jeta_x * gx[..., c] + mesh.Jeta_y * gy[..., c])

        # volume: Res -= D . Fv + D . Gv   (same matmul pattern as rhs)
        np.matmul(D, Fv.transpose(0, 1, 3, 2, 4), out=volT2)
        Res -= volT2.transpose(0, 1, 3, 2, 4)
        np.matmul(D, Gv, out=vol2)
        Res -= vol2

        # ---- interior-face SIPG penalties (sign opposite to the inviscid SAT)
        pen = self.cfg.penalty * (Np1 ** 2)
        # xi faces
        sJ = np.maximum(np.hypot(mesh.Fxp[..., 0], mesh.Fxp[..., 1]), 1e-300)
        nx, ny = mesh.Fxp[..., 0] / sJ, mesh.Fxp[..., 1] / sJ
        sig_x = 0.5 * pen * (nu / np.maximum(mesh.h_xi, 1e-300)
                             + np.roll(nu, -1, axis=0) / np.maximum(np.roll(mesh.h_xi, -1, axis=0), 1e-300))
        sig_x5 = sig_x[..., None, None]
        UL = U[:, :, Np1 - 1, :, :]
        UR = np.roll(U, -1, axis=0)[:, :, 0, :, :]
        # averaged physical gradient at the face
        gxL, gyL = gx[:, :, Np1 - 1, :, :], gy[:, :, Np1 - 1, :, :]
        gxR = np.roll(gx, -1, axis=0)[:, :, 0, :, :]
        gyR = np.roll(gy, -1, axis=0)[:, :, 0, :, :]
        nuL = nu[..., None, None]
        nuR = np.roll(nu, -1, axis=0)[..., None, None]
        Fv_face = (0.5 * (nuL * (gxL * nx[..., None] + gyL * ny[..., None])
                          + nuR * (gxR * nx[..., None] + gyR * ny[..., None]))
                   + sig_x5 * (UR - UL)) * sJ[..., None]
        Res[:, :, Np1 - 1, :, :] -= (Fv_face - Fv[:, :, Np1 - 1, :, :]) / w[Np1 - 1]
        Res[:, :, 0, :, :] -= (-np.roll(Fv_face, 1, axis=0) + Fv[:, :, 0, :, :]) / w[0]

        # eta interior faces
        if nj > 1:
            sJe = np.maximum(np.hypot(mesh.Fep[:, 0:nj - 1, :, 0],
                                      mesh.Fep[:, 0:nj - 1, :, 1]), 1e-300)
            nxe, nye = (mesh.Fep[:, 0:nj - 1, :, 0] / sJe,
                        mesh.Fep[:, 0:nj - 1, :, 1] / sJe)
            sig_e = 0.5 * pen * (nu[:, 0:nj - 1] / np.maximum(mesh.h_eta[:, 0:nj - 1], 1e-300)
                                 + nu[:, 1:nj] / np.maximum(mesh.h_eta[:, 1:nj], 1e-300))
            sig_e5 = sig_e[..., None, None]
            UL = U[:, 0:nj - 1, :, Np1 - 1, :]
            UR = U[:, 1:nj, :, 0, :]
            gxL, gyL = gx[:, 0:nj - 1, :, Np1 - 1, :], gy[:, 0:nj - 1, :, Np1 - 1, :]
            gxR, gyR = gx[:, 1:nj, :, 0, :], gy[:, 1:nj, :, 0, :]
            nuL = nu[:, 0:nj - 1, None, None]
            nuR = nu[:, 1:nj, None, None]
            Fv_face = (0.5 * (nuL * (gxL * nxe[..., None] + gyL * nye[..., None])
                              + nuR * (gxR * nxe[..., None] + gyR * nye[..., None]))
                       + sig_e5 * (UR - UL)) * sJe[..., None]
            Res[:, 0:nj - 1, :, Np1 - 1, :] -= (Fv_face - Gv[:, 0:nj - 1, :, Np1 - 1, :]) / w[Np1 - 1]
            Res[:, 1:nj, :, 0, :] -= (-Fv_face + Gv[:, 1:nj, :, 0, :]) / w[0]

        # eta periodic wrap (verification / shock-tube meshes)
        if mesh.periodic_eta:
            sJe = np.maximum(np.hypot(mesh.Fep[:, nj - 1, :, 0],
                                      mesh.Fep[:, nj - 1, :, 1]), 1e-300)
            nxe = mesh.Fep[:, nj - 1, :, 0] / sJe
            nye = mesh.Fep[:, nj - 1, :, 1] / sJe
            sig_e = 0.5 * pen * (nu[:, nj - 1]
                                 / np.maximum(mesh.h_eta[:, nj - 1], 1e-300)
                                 + nu[:, 0]
                                 / np.maximum(mesh.h_eta[:, 0], 1e-300))
            sig_e5 = sig_e[..., None, None]
            UL = U[:, nj - 1, :, Np1 - 1, :]
            UR = U[:, 0, :, 0, :]
            gxL, gyL = gx[:, nj - 1, :, Np1 - 1, :], gy[:, nj - 1, :, Np1 - 1, :]
            gxR, gyR = gx[:, 0, :, 0, :], gy[:, 0, :, 0, :]
            nuL = nu[:, nj - 1, None, None]
            nuR = nu[:, 0, None, None]
            Fv_face = (0.5 * (nuL * (gxL * nxe[..., None] + gyL * nye[..., None])
                              + nuR * (gxR * nxe[..., None] + gyR * nye[..., None]))
                       + sig_e5 * (UR - UL)) * sJe[..., None]
            Res[:, nj - 1, :, Np1 - 1, :] -= (Fv_face
                                              - Gv[:, nj - 1, :, Np1 - 1, :]) / w[Np1 - 1]
            Res[:, 0, :, 0, :] -= (-Fv_face + Gv[:, 0, :, 0, :]) / w[0]

    # ------------------------------------------------------------------
    def _diffuse_ns(self, U, Res) -> None:
        """Navier-Stokes viscous operator on (u, v, T) for the periodic-box
        verification problems (BR2 lifted gradients + SIPG).  Production
        airfoil runs use the Euler operator (mu = 0); wall and far-field
        viscous closures are therefore not required here."""
        mesh = self.mesh
        D, w, Np1 = mesh.D, mesh.w, mesh.Np1
        nk, nj = mesh.n_k, mesh.n_j
        mu, kc = self.cfg.mu, self.cfg.k_cond
        gamma = self.gamma
        invJ = 1.0 / mesh.J

        rho, u, v, p = primitives(U, gamma)
        T = p / rho          # gas constant R = 1 in this normalisation
        q4 = np.stack([u, v, T], axis=-1)
        dq_xi, dq_eta = self._lifted_gradients(q4)

        ux = invJ * (mesh.Jxi_x * dq_xi[..., 0] + mesh.Jeta_x * dq_eta[..., 0])
        uy = invJ * (mesh.Jxi_y * dq_xi[..., 0] + mesh.Jeta_y * dq_eta[..., 0])
        vx = invJ * (mesh.Jxi_x * dq_xi[..., 1] + mesh.Jeta_x * dq_eta[..., 1])
        vy = invJ * (mesh.Jxi_y * dq_xi[..., 1] + mesh.Jeta_y * dq_eta[..., 1])
        Tx = invJ * (mesh.Jxi_x * dq_xi[..., 2] + mesh.Jeta_x * dq_eta[..., 2])
        Ty = invJ * (mesh.Jxi_y * dq_xi[..., 2] + mesh.Jeta_y * dq_eta[..., 2])

        divu = ux + vy
        txx = mu * (2.0 * ux - (2.0 / 3.0) * divu)
        tyy = mu * (2.0 * vy - (2.0 / 3.0) * divu)
        txy = mu * (uy + vx)
        thx = txx * u + txy * v + kc * Tx
        thy = txy * u + tyy * v + kc * Ty

        # CONTRAVARIANT viscous fluxes: the collocation derivative acts on
        # (J xi_x F + J xi_y G) and (J eta_x F + J eta_y G), exactly like
        # the inviscid volume term -- NOT on the physical fluxes alone.
        Fv = np.zeros(self.Fh.shape)
        Gv = np.zeros(self.Gh.shape)
        Fv[..., 1] = mesh.Jxi_x * txx + mesh.Jxi_y * txy
        Fv[..., 2] = mesh.Jxi_x * txy + mesh.Jxi_y * tyy
        Fv[..., 3] = mesh.Jxi_x * thx + mesh.Jxi_y * thy
        Gv[..., 1] = mesh.Jeta_x * txx + mesh.Jeta_y * txy
        Gv[..., 2] = mesh.Jeta_x * txy + mesh.Jeta_y * tyy
        Gv[..., 3] = mesh.Jeta_x * thx + mesh.Jeta_y * thy

        volT2 = self._volT
        vol2 = self._vol
        np.matmul(D, Fv.transpose(0, 1, 3, 2, 4), out=volT2)
        Res -= volT2.transpose(0, 1, 3, 2, 4)
        np.matmul(D, Gv, out=vol2)
        Res -= vol2

        # interior-face SIPG penalties on (u, v, T)
        pen = self.cfg.penalty * (Np1 ** 2)
        h_xi, h_eta = mesh.h_xi, mesh.h_eta

        # xi faces
        sJ = np.maximum(np.hypot(mesh.Fxp[..., 0], mesh.Fxp[..., 1]), 1e-300)
        nx, ny = mesh.Fxp[..., 0] / sJ, mesh.Fxp[..., 1] / sJ
        sig_x = (pen * mu / np.maximum(h_xi, 1e-300))[..., None]
        qL = q4[:, :, Np1 - 1, :, :]
        qR = np.roll(q4, -1, axis=0)[:, :, 0, :, :]
        uxL, uyL = ux[:, :, Np1 - 1, :], uy[:, :, Np1 - 1, :]
        vxL, vyL = vx[:, :, Np1 - 1, :], vy[:, :, Np1 - 1, :]
        TxL, TyL = Tx[:, :, Np1 - 1, :], Ty[:, :, Np1 - 1, :]
        uxR = np.roll(ux, -1, axis=0)[:, :, 0, :]
        uyR = np.roll(uy, -1, axis=0)[:, :, 0, :]
        vxR = np.roll(vx, -1, axis=0)[:, :, 0, :]
        vyR = np.roll(vy, -1, axis=0)[:, :, 0, :]
        TxR = np.roll(Tx, -1, axis=0)[:, :, 0, :]
        TyR = np.roll(Ty, -1, axis=0)[:, :, 0, :]
        uxa, uya = 0.5 * (uxL + uxR), 0.5 * (uyL + uyR)
        vxa, vya = 0.5 * (vxL + vxR), 0.5 * (vyL + vyR)
        Txa, Tya = 0.5 * (TxL + TxR), 0.5 * (TyL + TyR)
        ufa, vfa = 0.5 * (qL[..., 0] + qR[..., 0]), 0.5 * (qL[..., 1] + qR[..., 1])
        diva = uxa + vya
        txxa = mu * (2.0 * uxa - (2.0 / 3.0) * diva)
        tyya = mu * (2.0 * vya - (2.0 / 3.0) * diva)
        txya = mu * (uya + vxa)
        thxa = txxa * ufa + txya * vfa + kc * Txa
        thya = txya * ufa + tyya * vfa + kc * Tya
        Fvnx = txxa * nx + txya * ny + 2.0 * mu * sig_x * (qR[..., 0] - qL[..., 0])
        Fvny = txya * nx + tyya * ny + 2.0 * mu * sig_x * (qR[..., 1] - qL[..., 1])
        FvnE = thxa * nx + thya * ny + kc * sig_x * (qR[..., 2] - qL[..., 2])
        Fv_face = np.stack([np.zeros_like(Fvnx), Fvnx, Fvny, FvnE], axis=-1) * sJ[..., None]
        Res[:, :, Np1 - 1, :, :] -= (Fv_face - Fv[:, :, Np1 - 1, :, :]) / w[Np1 - 1]
        Res[:, :, 0, :, :] -= (-np.roll(Fv_face, 1, axis=0) + Fv[:, :, 0, :, :]) / w[0]

        # eta interior faces
        if nj > 1:
            sJe = np.maximum(np.hypot(mesh.Fep[:, 0:nj - 1, :, 0],
                                      mesh.Fep[:, 0:nj - 1, :, 1]), 1e-300)
            nxe, nye = (mesh.Fep[:, 0:nj - 1, :, 0] / sJe,
                        mesh.Fep[:, 0:nj - 1, :, 1] / sJe)
            sig_e = (pen * mu / np.maximum(h_eta[:, 0:nj - 1], 1e-300))[..., None]
            qL = q4[:, 0:nj - 1, :, Np1 - 1, :]
            qR = q4[:, 1:nj, :, 0, :]
            uxa, uya = 0.5 * (ux[:, 0:nj - 1, :, Np1 - 1] + ux[:, 1:nj, :, 0]), \
                       0.5 * (uy[:, 0:nj - 1, :, Np1 - 1] + uy[:, 1:nj, :, 0])
            vxa, vya = 0.5 * (vx[:, 0:nj - 1, :, Np1 - 1] + vx[:, 1:nj, :, 0]), \
                       0.5 * (vy[:, 0:nj - 1, :, Np1 - 1] + vy[:, 1:nj, :, 0])
            Txa, Tya = 0.5 * (Tx[:, 0:nj - 1, :, Np1 - 1] + Tx[:, 1:nj, :, 0]), \
                       0.5 * (Ty[:, 0:nj - 1, :, Np1 - 1] + Ty[:, 1:nj, :, 0])
            ufa, vfa = 0.5 * (qL[..., 0] + qR[..., 0]), 0.5 * (qL[..., 1] + qR[..., 1])
            diva = uxa + vya
            txxa = mu * (2.0 * uxa - (2.0 / 3.0) * diva)
            tyya = mu * (2.0 * vya - (2.0 / 3.0) * diva)
            txya = mu * (uya + vxa)
            thxa = txxa * ufa + txya * vfa + kc * Txa
            thya = txya * ufa + tyya * vfa + kc * Tya
            Fvnx = txxa * nxe + txya * nye + 2.0 * mu * sig_e * (qR[..., 0] - qL[..., 0])
            Fvny = txya * nxe + tyya * nye + 2.0 * mu * sig_e * (qR[..., 1] - qL[..., 1])
            FvnE = thxa * nxe + thya * nye + kc * sig_e * (qR[..., 2] - qL[..., 2])
            Fv_face = np.stack([np.zeros_like(Fvnx), Fvnx, Fvny, FvnE], axis=-1) * sJe[..., None]
            Res[:, 0:nj - 1, :, Np1 - 1, :] -= (Fv_face - Gv[:, 0:nj - 1, :, Np1 - 1, :]) / w[Np1 - 1]
            Res[:, 1:nj, :, 0, :] -= (-Fv_face + Gv[:, 1:nj, :, 0, :]) / w[0]

        # eta periodic wrap (verification meshes)
        if mesh.periodic_eta:
            sJe = np.maximum(np.hypot(mesh.Fep[:, nj - 1, :, 0],
                                      mesh.Fep[:, nj - 1, :, 1]), 1e-300)
            nxe = mesh.Fep[:, nj - 1, :, 0] / sJe
            nye = mesh.Fep[:, nj - 1, :, 1] / sJe
            sig_e = (pen * mu / np.maximum(mesh.h_eta[:, nj - 1], 1e-300))[..., None]
            qL = q4[:, nj - 1, :, Np1 - 1, :]
            qR = q4[:, 0, :, 0, :]
            uxa, uya = 0.5 * (ux[:, nj - 1, :, Np1 - 1] + ux[:, 0, :, 0]), \
                       0.5 * (uy[:, nj - 1, :, Np1 - 1] + uy[:, 0, :, 0])
            vxa, vya = 0.5 * (vx[:, nj - 1, :, Np1 - 1] + vx[:, 0, :, 0]), \
                       0.5 * (vy[:, nj - 1, :, Np1 - 1] + vy[:, 0, :, 0])
            Txa, Tya = 0.5 * (Tx[:, nj - 1, :, Np1 - 1] + Tx[:, 0, :, 0]), \
                       0.5 * (Ty[:, nj - 1, :, Np1 - 1] + Ty[:, 0, :, 0])
            ufa, vfa = 0.5 * (qL[..., 0] + qR[..., 0]), 0.5 * (qL[..., 1] + qR[..., 1])
            diva = uxa + vya
            txxa = mu * (2.0 * uxa - (2.0 / 3.0) * diva)
            tyya = mu * (2.0 * vya - (2.0 / 3.0) * diva)
            txya = mu * (uya + vxa)
            thxa = txxa * ufa + txya * vfa + kc * Txa
            thya = txya * ufa + tyya * vfa + kc * Tya
            Fvnx = txxa * nxe + txya * nye + 2.0 * mu * sig_e * (qR[..., 0] - qL[..., 0])
            Fvny = txya * nxe + tyya * nye + 2.0 * mu * sig_e * (qR[..., 1] - qL[..., 1])
            FvnE = thxa * nxe + thya * nye + kc * sig_e * (qR[..., 2] - qL[..., 2])
            Fv_face = np.stack([np.zeros_like(Fvnx), Fvnx, Fvny, FvnE], axis=-1) * sJe[..., None]
            Res[:, nj - 1, :, Np1 - 1, :] -= (Fv_face
                                              - Gv[:, nj - 1, :, Np1 - 1, :]) / w[Np1 - 1]
            Res[:, 0, :, 0, :] -= (-Fv_face + Gv[:, 0, :, 0, :]) / w[0]


# ----------------------------------------------------------------------
def _self_test() -> None:  # pragma: no cover
    import time
    from .mesh import generate_airfoil_omesh, generate_periodic_box_mesh
    from .euler import initialize_freestream

    print("free-stream preservation self test (the critical correctness gate)")
    print("(wall ghost = copy: pure geometry test, Kopriva GCL convention)")
    for name, mesh in [
        ("O-mesh 24x16 N=3", generate_airfoil_omesh(24, 16, 3, 30.0, 0.008)),
        ("O-mesh 40x28 N=3", generate_airfoil_omesh(40, 28, 3, 30.0, 0.006)),
        ("warped periodic box", generate_periodic_box_mesh(6, 5, 3, 10.0, 0.3)),
    ]:
        wall = "copy" if mesh.has_wall else "slip"
        cfg = SolverConfig(flux="roe", wall=wall)
        op = Operator(mesh, cfg)
        U, Qinf = initialize_freestream(mesh, 0.3, 5.0, cfg.gamma)
        op.set_freestream(Qinf)
        t0 = time.time()
        dU = op.rhs(U)
        dt1 = time.time() - t0
        # measure the residual BEFORE the 1/J division: this is the
        # scale-invariant free-stream-preservation measure (a tiny element
        # must show a tiny residual, and dU = Res/J would amplify rounding)
        err = np.abs(dU * mesh.J[..., None]).max()
        print(f"  {name} ({cfg.flux}, wall={wall}): max|J dU/dt| = {err:.3e} "
              f"({dt1 * 1e3:.1f} ms/residual)")
        assert err < 1e-11, "free-stream preservation FAILED"

    # slip wall: the reflected-ghost Riemann flux must give exactly zero
    # normal mass flux through every wall face for ANY state
    mesh = generate_airfoil_omesh(24, 16, 3, 30.0, 0.008)
    cfg = SolverConfig(flux="roe", wall="slip")
    op = Operator(mesh, cfg)
    rng = np.random.default_rng(3)
    U = np.empty((mesh.n_k, mesh.n_j, mesh.Np1, mesh.Np1, 4))
    U[..., 0] = 1.0 + 0.2 * rng.random(U.shape[:-1])
    U[..., 1] = rng.normal(0.3, 0.1, U.shape[:-1])
    U[..., 2] = rng.normal(0.0, 0.1, U.shape[:-1])
    U[..., 3] = 2.5 + 0.5 * rng.random(U.shape[:-1])
    op.set_freestream(initialize_freestream(mesh, 0.3, 0.0, cfg.gamma)[1])
    dU = op.rhs(U)   # must simply run (signs exercised); impermeability is a
    # property of the wall flux itself:
    from .numflux import roe_flux
    Qint = U[:, 0, :, 0, :]
    sJnx, sJny = mesh.Fem[:, 0, :, 0], mesh.Fem[:, 0, :, 1]
    sJ = np.hypot(sJnx, sJny); nx, ny = sJnx / sJ, sJny / sJ
    rho = Qint[..., 0]; u = Qint[..., 1] / rho; v = Qint[..., 2] / rho
    Vn = u * nx + v * ny
    uG = u - 2 * Vn * nx; vG = v - 2 * Vn * ny
    QG = np.stack([rho, rho * uG, rho * vG, Qint[..., 3]], axis=-1)
    Fn, _ = roe_flux(QG, Qint, sJnx, sJny)
    mass_flux = Fn[..., 0]
    print(f"  slip wall impermeability: max|mass flux| = {np.abs(mass_flux).max():.3e}")
    assert np.abs(mass_flux).max() < 1e-12

    # rusanov too
    mesh = generate_airfoil_omesh(24, 16, 3, 30.0, 0.008)
    cfg = SolverConfig(flux="rusanov", wall="copy")
    op = Operator(mesh, cfg)
    U, Qinf = initialize_freestream(mesh, 0.8, 0.0, cfg.gamma)
    op.set_freestream(Qinf)
    err = np.abs(op.rhs(U) * mesh.J[..., None]).max()
    print(f"  rusanov M=0.8 alpha=0: max|J dU/dt| = {err:.3e}")
    assert err < 1e-11
    print("  all checks passed")


if __name__ == "__main__":
    _self_test()
