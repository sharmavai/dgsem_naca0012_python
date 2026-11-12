"""Numba-accelerated inviscid DGSEM right-hand side.

This module implements exactly the same semi-discrete inviscid operator as
:class:`dgsem.operator.Operator` (strong-form DGSEM, SAT interface terms,
Roe or Rusanov flux, slip walls, far-field boundary with the Thomas-Salas
point-vortex correction) as a single fused nopython kernel, and is used for
the production runs.  Cases with diffusion (artificial viscosity or the
Navier-Stokes terms) fall back to the NumPy reference operator; they only
occur in the small verification meshes, where NumPy speed is sufficient.

Equivalence with the reference operator is verified to ~1e-12 (relative) on
random states for every boundary configuration used in this project by the
self test (`python -m dgsem.operator_numba`).
"""
from __future__ import annotations

import numpy as np
from numba import njit

from .euler import GAMMA
from .operator import Operator, SolverConfig

__all__ = ["NumbaOperator", "rhs_inviscid"]


# ----------------------------------------------------------------------
@njit(cache=True, fastmath=True, error_model="numpy", inline="always")
def _flux_node(rL, ruL, rvL, REL, rR, ruR, rvR, RER,
               sJnx, sJny, gamma, flux_kind, efix, fix_all):
    """Interface flux (contravariant, sJ-weighted) at one face node.

    flux_kind: 0 = Roe (entropy fix on the acoustic waves, strength efix),
               1 = Rusanov (local Lax-Friedrichs).
    Mirrors dgsem.numflux.roe_flux / rusanov_flux exactly."""
    sJ = max(np.hypot(sJnx, sJny), 1e-300)
    nx = sJnx / sJ
    ny = sJny / sJ
    g1 = gamma - 1.0

    rLo = max(rL, 1e-12)
    rRo = max(rR, 1e-12)
    uL = ruL / rLo
    vL = rvL / rLo
    uR = ruR / rRo
    vR = rvR / rRo
    pL = max(g1 * (REL - 0.5 * rLo * (uL * uL + vL * vL)), 1e-12)
    pR = max(g1 * (RER - 0.5 * rRo * (uR * uR + vR * vR)), 1e-12)

    VnL = uL * nx + vL * ny
    VnR = uR * nx + vR * ny

    if flux_kind == 1:
        # Rusanov
        cL = np.sqrt(gamma * pL / rLo)
        cR = np.sqrt(gamma * pR / rRo)
        lam = max(abs(VnL) + cL, abs(VnR) + cR)
        f0 = 0.5 * (rLo * VnL + rRo * VnR) - 0.5 * lam * (rR - rL)
        f1 = (0.5 * (rLo * uL * VnL + pL * nx + rRo * uR * VnR + pR * nx)
              - 0.5 * lam * (ruR - ruL))
        f2 = (0.5 * (rLo * vL * VnL + pL * ny + rRo * vR * VnR + pR * ny)
              - 0.5 * lam * (rvR - rvL))
        f3 = (0.5 * ((REL + pL) * VnL + (RER + pR) * VnR)
              - 0.5 * lam * (RER - REL))
        return (f0 * sJ, f1 * sJ, f2 * sJ, f3 * sJ)

    # ---- Roe ----
    HL = (REL + pL) / rLo
    HR = (RER + pR) / rRo
    sL = np.sqrt(rLo)
    sR = np.sqrt(rRo)
    inv = 1.0 / (sL + sR)
    rt = sL * sR
    u_t = (sL * uL + sR * uR) * inv
    v_t = (sL * vL + sR * vR) * inv
    H_t = (sL * HL + sR * HR) * inv
    q2 = u_t * u_t + v_t * v_t
    c2 = max(g1 * (H_t - 0.5 * q2), 1e-12)
    c_t = np.sqrt(c2)
    Vn = u_t * nx + v_t * ny

    dr = rR - rL
    dru = ruR - ruL
    drv = rvR - rvL
    dE = RER - REL
    dp = g1 * (dE - u_t * dru - v_t * drv + 0.5 * q2 * dr)
    rtmax = rt if rt > 1e-12 else 1e-12
    dVn = ((dru - u_t * dr) * nx + (drv - v_t * dr) * ny) / rtmax
    dUt = (-(dru - u_t * dr) * ny + (drv - v_t * dr) * nx) / rtmax

    a1 = dr - dp / c2
    a2 = (dp - rt * c_t * dVn) / (2.0 * c2)
    a3 = rt * dUt
    a4 = (dp + rt * c_t * dVn) / (2.0 * c2)

    # entropy fix: on the acoustic waves, and (fix_all) on the entropy /
    # shear waves as well -- the latter is what damps the discrete wake
    # sheet; see the docstring of dgsem.numflux.roe_flux
    lam_e = abs(Vn)
    if efix > 0.0:
        delta = efix * (abs(Vn) + c_t)
        lm = Vn - c_t
        lp = Vn + c_t
        alm = abs(lm)
        alp = abs(lp)
        dmax = delta if delta > 1e-300 else 1e-300
        lam_m = (lm * lm + delta * delta) / (2.0 * dmax) if alm < delta else alm
        lam_p = (lp * lp + delta * delta) / (2.0 * dmax) if alp < delta else alp
        if fix_all:
            ale = abs(Vn)
            lam_e = (Vn * Vn + delta * delta) / (2.0 * dmax) if ale < delta else ale
    else:
        lam_m = abs(Vn - c_t)
        lam_p = abs(Vn + c_t)

    d1 = a1 * lam_e
    d2 = a2 * lam_m
    d3 = a3 * lam_e
    d4 = a4 * lam_p

    Ut_t = -u_t * ny + v_t * nx

    f0 = 0.5 * (rLo * VnL + rRo * VnR) - 0.5 * (d1 + d2 + d4)
    f1 = (0.5 * (rLo * uL * VnL + pL * nx + rRo * uR * VnR + pR * nx)
          - 0.5 * (d1 * u_t + d2 * (u_t - c_t * nx)
                   + d3 * (-ny) + d4 * (u_t + c_t * nx)))
    f2 = (0.5 * (rLo * vL * VnL + pL * ny + rRo * vR * VnR + pR * ny)
          - 0.5 * (d1 * v_t + d2 * (v_t - c_t * ny)
                   + d3 * nx + d4 * (v_t + c_t * ny)))
    f3 = (0.5 * ((REL + pL) * VnL + (RER + pR) * VnR)
          - 0.5 * (d1 * 0.5 * q2 + d2 * (H_t - Vn * c_t)
                   + d3 * Ut_t + d4 * (H_t + Vn * c_t)))
    return (f0 * sJ, f1 * sJ, f2 * sJ, f3 * sJ)


# ----------------------------------------------------------------------
@njit(cache=True, fastmath=True, error_model="numpy")
def rhs_inviscid(U, Fh, Gh, Res, D, w, J, Jxi_x, Jxi_y, Jeta_x, Jeta_y,
                 Fxp, Fep, Fem, kappa_x, kappa_y, gamma, flux_kind, efix,
                 periodic_eta, has_wall, wall_top, has_farfield,
                 Qinf, use_vortex, Gamma, fix_all):
    """Strong-form inviscid DGSEM residual; result written into Res.

    Layout: U, Fh, Gh, Res are (nk, nj, Np1, Np1, 4); J and the metric
    terms are (nk, nj, Np1, Np1); Fxp is (nk, nj, Np1, 2); Fep is
    (nk, nj, Np1, 2); Fem is (nk, Np1, 2); kappa_x/kappa_y are (nk, Np1)
    far-field vortex velocity factors (see NumbaOperator).  Xi is always
    periodic (the O-mesh seam, the channel and the verification boxes are
    all periodic in xi).
    """
    nk = U.shape[0]
    nj = U.shape[1]
    Np1 = U.shape[2]
    N = Np1 - 1

    # ---- physical + contravariant fluxes, volume terms -----------------
    for k in range(nk):
        for j in range(nj):
            for i in range(Np1):
                for l in range(Np1):
                    rho = U[k, j, i, l, 0]
                    rho = rho if rho > 1e-12 else 1e-12
                    u = U[k, j, i, l, 1] / rho
                    v = U[k, j, i, l, 2] / rho
                    E = U[k, j, i, l, 3]
                    p = (gamma - 1.0) * (E - 0.5 * rho * (u * u + v * v))
                    p = p if p > 1e-12 else 1e-12
                    Ep = E + p
                    # physical fluxes
                    Fx0 = rho * u
                    Fx1 = rho * u * u + p
                    Fx2 = rho * u * v
                    Fx3 = Ep * u
                    Gx0 = rho * v
                    Gx1 = rho * u * v
                    Gx2 = rho * v * v + p
                    Gx3 = Ep * v
                    ax = Jxi_x[k, j, i, l]
                    ay = Jxi_y[k, j, i, l]
                    bx = Jeta_x[k, j, i, l]
                    by = Jeta_y[k, j, i, l]
                    Fh[k, j, i, l, 0] = ax * Fx0 + ay * Gx0
                    Fh[k, j, i, l, 1] = ax * Fx1 + ay * Gx1
                    Fh[k, j, i, l, 2] = ax * Fx2 + ay * Gx2
                    Fh[k, j, i, l, 3] = ax * Fx3 + ay * Gx3
                    Gh[k, j, i, l, 0] = bx * Fx0 + by * Gx0
                    Gh[k, j, i, l, 1] = bx * Fx1 + by * Gx1
                    Gh[k, j, i, l, 2] = bx * Fx2 + by * Gx2
                    Gh[k, j, i, l, 3] = bx * Fx3 + by * Gx3

            for i in range(Np1):
                for l in range(Np1):
                    s0 = 0.0
                    s1 = 0.0
                    s2 = 0.0
                    s3 = 0.0
                    for m in range(Np1):
                        Dw = D[i, m]
                        s0 += Dw * Fh[k, j, m, l, 0]
                        s1 += Dw * Fh[k, j, m, l, 1]
                        s2 += Dw * Fh[k, j, m, l, 2]
                        s3 += Dw * Fh[k, j, m, l, 3]
                    t0 = 0.0
                    t1 = 0.0
                    t2 = 0.0
                    t3 = 0.0
                    for m in range(Np1):
                        Dw = D[l, m]
                        t0 += Dw * Gh[k, j, i, m, 0]
                        t1 += Dw * Gh[k, j, i, m, 1]
                        t2 += Dw * Gh[k, j, i, m, 2]
                        t3 += Dw * Gh[k, j, i, m, 3]
                    Res[k, j, i, l, 0] = s0 + t0
                    Res[k, j, i, l, 1] = s1 + t1
                    Res[k, j, i, l, 2] = s2 + t2
                    Res[k, j, i, l, 3] = s3 + t3

    # ---- xi-periodic faces (seam of the O-mesh / channel / box) --------
    for k in range(nk):
        kn = k + 1
        if kn == nk:
            kn = 0
        for j in range(nj):
            for l in range(Np1):
                f = _flux_node(U[k, j, N, l, 0], U[k, j, N, l, 1],
                               U[k, j, N, l, 2], U[k, j, N, l, 3],
                               U[kn, j, 0, l, 0], U[kn, j, 0, l, 1],
                               U[kn, j, 0, l, 2], U[kn, j, 0, l, 3],
                               Fxp[k, j, l, 0], Fxp[k, j, l, 1],
                               gamma, flux_kind, efix, fix_all)
                for c in range(4):
                    Res[k, j, N, l, c] += (f[c] - Fh[k, j, N, l, c]) / w[N]
                    Res[kn, j, 0, l, c] += (-f[c] + Fh[kn, j, 0, l, c]) / w[0]

    # ---- eta interior faces --------------------------------------------
    for k in range(nk):
        for j in range(nj - 1):
            for i in range(Np1):
                f = _flux_node(U[k, j, i, N, 0], U[k, j, i, N, 1],
                               U[k, j, i, N, 2], U[k, j, i, N, 3],
                               U[k, j + 1, i, 0, 0], U[k, j + 1, i, 0, 1],
                               U[k, j + 1, i, 0, 2], U[k, j + 1, i, 0, 3],
                               Fep[k, j, i, 0], Fep[k, j, i, 1],
                               gamma, flux_kind, efix, fix_all)
                for c in range(4):
                    Res[k, j, i, N, c] += (f[c] - Gh[k, j, i, N, c]) / w[N]
                    Res[k, j + 1, i, 0, c] += (-f[c] + Gh[k, j + 1, i, 0, c]) / w[0]

    # ---- periodic eta wrap (verification box meshes) --------------------
    if periodic_eta:
        for k in range(nk):
            for i in range(Np1):
                f = _flux_node(U[k, nj - 1, i, N, 0], U[k, nj - 1, i, N, 1],
                               U[k, nj - 1, i, N, 2], U[k, nj - 1, i, N, 3],
                               U[k, 0, i, 0, 0], U[k, 0, i, 0, 1],
                               U[k, 0, i, 0, 2], U[k, 0, i, 0, 3],
                               Fep[k, nj - 1, i, 0], Fep[k, nj - 1, i, 1],
                               gamma, flux_kind, efix, fix_all)
                for c in range(4):
                    Res[k, nj - 1, i, N, c] += (f[c] - Gh[k, nj - 1, i, N, c]) / w[N]
                    Res[k, 0, i, 0, c] += (-f[c] + Gh[k, 0, i, 0, c]) / w[0]

    # ---- slip wall at eta = -1 (airfoil surface / channel bottom) -------
    if has_wall:
        for k in range(nk):
            for i in range(Np1):
                rI = U[k, 0, i, 0, 0]
                rI = rI if rI > 1e-12 else 1e-12
                uI = U[k, 0, i, 0, 1] / rI
                vI = U[k, 0, i, 0, 2] / rI
                sJnx = Fem[k, i, 0]
                sJny = Fem[k, i, 1]
                sJ = np.hypot(sJnx, sJny)
                sJ = sJ if sJ > 1e-300 else 1e-300
                nx = sJnx / sJ
                ny = sJny / sJ
                Vn = uI * nx + vI * ny
                uG = uI - 2.0 * Vn * nx
                vG = vI - 2.0 * Vn * ny
                f = _flux_node(U[k, 0, i, 0, 0], U[k, 0, i, 0, 1],
                               U[k, 0, i, 0, 2], U[k, 0, i, 0, 3],
                               rI, rI * uG, rI * vG, U[k, 0, i, 0, 3],
                               sJnx, sJny, gamma, flux_kind, efix, fix_all)
                for c in range(4):
                    Res[k, 0, i, 0, c] += (f[c] + Gh[k, 0, i, 0, c]) / w[0]

    # ---- slip wall at eta = +1 (channel top) ----------------------------
    if wall_top:
        for k in range(nk):
            for i in range(Np1):
                rI = U[k, nj - 1, i, N, 0]
                rI = rI if rI > 1e-12 else 1e-12
                uI = U[k, nj - 1, i, N, 1] / rI
                vI = U[k, nj - 1, i, N, 2] / rI
                sJnx = Fep[k, nj - 1, i, 0]
                sJny = Fep[k, nj - 1, i, 1]
                sJ = np.hypot(sJnx, sJny)
                sJ = sJ if sJ > 1e-300 else 1e-300
                nx = sJnx / sJ
                ny = sJny / sJ
                Vn = uI * nx + vI * ny
                uG = uI - 2.0 * Vn * nx
                vG = vI - 2.0 * Vn * ny
                f = _flux_node(U[k, nj - 1, i, N, 0], U[k, nj - 1, i, N, 1],
                               U[k, nj - 1, i, N, 2], U[k, nj - 1, i, N, 3],
                               rI, rI * uG, rI * vG, U[k, nj - 1, i, N, 3],
                               sJnx, sJny, gamma, flux_kind, efix, fix_all)
                for c in range(4):
                    Res[k, nj - 1, i, N, c] += (f[c] - Gh[k, nj - 1, i, N, c]) / w[N]

    # ---- far field at eta = +1 (airfoil outer boundary) -----------------
    # The ghost state is the free stream plus the velocity of a point
    # vortex of circulation Gamma at (cx, cy): (u, v) += Gamma * (kx, ky)
    # with (kx, ky) = (-dy, dx) / (2 pi r^2), precomputed per boundary node.
    if has_farfield:
        rinf = Qinf[0]
        uinf = Qinf[1] / rinf
        vinf = Qinf[2] / rinf
        for k in range(nk):
            for i in range(Np1):
                if use_vortex and Gamma != 0.0:
                    uB = uinf + Gamma * kappa_x[k, i]
                    vB = vinf + Gamma * kappa_y[k, i]
                else:
                    uB = uinf
                    vB = vinf
                qB2 = uB * uB + vB * vB
                qI2 = uinf * uinf + vinf * vinf
                g0 = rinf
                g1 = rinf * uB
                g2 = rinf * vB
                g3 = Qinf[3] + 0.5 * rinf * (qB2 - qI2)
                f = _flux_node(U[k, nj - 1, i, N, 0], U[k, nj - 1, i, N, 1],
                               U[k, nj - 1, i, N, 2], U[k, nj - 1, i, N, 3],
                               g0, g1, g2, g3,
                               Fep[k, nj - 1, i, 0], Fep[k, nj - 1, i, 1],
                               gamma, flux_kind, efix, fix_all)
                for c in range(4):
                    Res[k, nj - 1, i, N, c] += (f[c] - Gh[k, nj - 1, i, N, c]) / w[N]

    # ---- final division by -J -------------------------------------------
    for k in range(nk):
        for j in range(nj):
            for i in range(Np1):
                for l in range(Np1):
                    Ji = -1.0 / J[k, j, i, l]
                    Res[k, j, i, l, 0] *= Ji
                    Res[k, j, i, l, 1] *= Ji
                    Res[k, j, i, l, 2] *= Ji
                    Res[k, j, i, l, 3] *= Ji
    return Res


# ----------------------------------------------------------------------
class NumbaOperator:
    """Drop-in replacement for :class:`dgsem.operator.Operator` for the
    inviscid operator (production airfoil runs).  Falls back to the NumPy
    reference operator whenever diffusion is active (artificial viscosity
    or physical viscosity), which only occurs on the small verification
    meshes.

    The far-field point-vortex velocity factors (kappa_x, kappa_y) are
    precomputed from the boundary node coordinates and the configured
    vortex center; the vortex circulation itself is a run-time attribute
    (self.Gamma) updated by the solver's Thomas-Salas feedback.
    """

    def __init__(self, mesh, cfg: SolverConfig):
        self.mesh = mesh
        self.cfg = cfg
        self.gamma = cfg.gamma
        self.flux_kind = 0 if cfg.flux == "roe" else 1
        self.efix = float(cfg.entropy_fix)
        self.fix_all = 1 if getattr(cfg, 'fix_all_waves', False) else 0
        self.Qinf = None
        self.Gamma = 0.0
        self._ref = None            # lazy NumPy fallback operator

        shape = (mesh.n_k, mesh.n_j, mesh.Np1, mesh.Np1)
        self._Fh = np.empty(shape + (4,))
        self._Gh = np.empty(shape + (4,))
        self._Res = np.empty(shape + (4,))

        # far-field vortex velocity factors (geometry only)
        cx, cy = cfg.vortex_center
        x = mesh.X[:, -1, :, -1]                     # (nk, Np1)
        y = mesh.Y[:, -1, :, -1]
        r2 = np.maximum((x - cx) ** 2 + (y - cy) ** 2, 1e-12)
        self._kappa_x = (-(y - cy)) / (2.0 * np.pi * r2)
        self._kappa_y = (x - cx) / (2.0 * np.pi * r2)

    # ------------------------------------------------------------------
    def set_freestream(self, Qinf: np.ndarray) -> None:
        self.Qinf = np.ascontiguousarray(np.asarray(Qinf, dtype=np.float64))

    # ------------------------------------------------------------------
    def rhs(self, U: np.ndarray, nu: np.ndarray | None = None) -> np.ndarray:
        mesh = self.mesh
        if nu is not None and np.any(nu > 0.0):
            return self._reference().rhs(U, nu)
        if self.cfg.mu > 0.0:
            return self._reference().rhs(U, nu)
        if not mesh.periodic_xi:
            return self._reference().rhs(U, nu)
        Res = self._Res
        # Qinf is only used when has_farfield; pass a dummy for periodic
        # verification meshes that never set a free stream
        Qinf_arg = (self.Qinf if self.Qinf is not None
                    else np.zeros(4))
        rhs_inviscid(U, self._Fh, self._Gh, Res,
                     mesh.D, mesh.w, mesh.J,
                     mesh.Jxi_x, mesh.Jxi_y, mesh.Jeta_x, mesh.Jeta_y,
                     mesh.Fxp, mesh.Fep, mesh.Fem[:, 0, :, :],
                     self._kappa_x, self._kappa_y,
                     self.gamma, self.flux_kind, self.efix,
                     mesh.periodic_eta, mesh.has_wall,
                     bool(getattr(mesh, "wall_top", False)),
                     mesh.has_farfield,
                     Qinf_arg,
                     bool(self.cfg.farfield_vortex), self.Gamma,
                     self.fix_all)
        # return a private copy so multi-stage integrators never alias
        return Res.copy()

    # ------------------------------------------------------------------
    def _reference(self):
        if self._ref is None:
            self._ref = Operator(self.mesh, self.cfg)
            self._ref.Qinf = self.Qinf
            self._ref.Gamma = self.Gamma
        self._ref.Qinf = self.Qinf
        self._ref.Gamma = self.Gamma
        return self._ref


# ----------------------------------------------------------------------
def _self_test() -> None:  # pragma: no cover
    """Equivalence of the Numba kernel and the NumPy reference operator on
    random states for every boundary configuration used in this project."""
    import time
    from .mesh import (generate_airfoil_omesh, generate_periodic_box_mesh,
                       generate_channel_mesh)
    from .euler import initialize_freestream

    rng = np.random.default_rng(7)
    t_compile = time.time()
    print("Numba operator self test (first call compiles, be patient)")

    def randomise(U, amp=0.35):
        V = U.copy()
        V += amp * (rng.random(V.shape) - 0.5) * np.abs(U).mean()
        # keep it physical: positive density / pressure
        V[..., 0] = np.maximum(V[..., 0], 0.3)
        Eint = V[..., 3] - 0.5 * (V[..., 1] ** 2 + V[..., 2] ** 2) / V[..., 0]
        V[..., 3] += np.maximum(0.2 - 2.0 * Eint, 0.0) * V[..., 0]
        return V

    cases = []
    # (mesh, cfg overrides)
    m = generate_airfoil_omesh(12, 8, 3, 25.0, 0.01)
    U, Qinf = initialize_freestream(m, 0.3, 5.0)
    cases.append(("airfoil O-mesh (wall + far field + vortex)", m, U, Qinf,
                  dict(farfield_vortex=True)))
    cases.append(("airfoil O-mesh (vortex off, rusanov)", m, U, Qinf,
                  dict(farfield_vortex=False, flux="rusanov")))
    cases.append(("airfoil O-mesh (entropy fix 0)", m, U, Qinf,
                  dict(entropy_fix=0.0)))
    m2 = generate_periodic_box_mesh(6, 5, 3, 10.0, 0.3)
    U2, Q2 = initialize_freestream(m2, 0.5, 0.0)
    cases.append(("periodic box", m2, U2, Q2, dict()))
    m3 = generate_channel_mesh(6, 4, 3, 2.0, 1.0)
    U3, Q3 = initialize_freestream(m3, 0.4, 0.0)
    cases.append(("channel (two slip walls)", m3, U3, Q3, dict()))

    worst = 0.0
    for name, mesh, U0, Qinf, over in cases:
        cfg = SolverConfig(**over)
        op_ref = Operator(mesh, cfg)
        op_nb = NumbaOperator(mesh, cfg)
        op_ref.set_freestream(Qinf)
        op_nb.set_freestream(Qinf)
        op_ref.Gamma = -0.09
        op_nb.Gamma = -0.09
        U = randomise(U0)
        R1 = op_ref.rhs(U)
        R2 = op_nb.rhs(U)
        err = np.abs(R1 - R2).max() / max(np.abs(R1).max(), 1e-300)
        worst = max(worst, err)
        print(f"  {name}: max rel diff {err:.3e}")
        assert err < 1e-11, name

    # free-stream preservation through the Numba kernel (channel: the
    # uniform state IS a solution with slip walls)
    m3 = generate_channel_mesh(8, 5, 3, 2.0, 1.0)
    U3, Q3 = initialize_freestream(m3, 0.4, 0.0)
    op_nb = NumbaOperator(m3, SolverConfig())
    op_nb.set_freestream(Q3)
    R = op_nb.rhs(U3)
    print(f"  channel free stream (slip walls): |R| = {np.abs(R).max():.3e}")
    assert np.abs(R).max() < 1e-12

    print(f"  worst equivalence error {worst:.3e}")
    print(f"  compile + tests: {time.time() - t_compile:.1f} s")
    print("  Numba operator self test passed")


if __name__ == "__main__":
    _self_test()
