"""Hess-Smith constant-source / constant-vortex panel method for the
incompressible potential flow around the NACA 0012, used to initialise the
compressible Euler march.

The panel solution (with the Kutta condition at the sharp trailing edge)
provides a near-steady initial condition, which removes the long impulsive
start transient that would otherwise dominate the explicit marching at low
Mach numbers.  At M <= 0.3 the incompressible potential solution differs
from the compressible steady Euler solution by only a few per cent, so the
subsequent march only has to relax a small perturbation.

Classical formulation: J. L. Hess and A. M. O. Smith, "Calculation of
non-lifting potential flow about arbitrary three-dimensional bodies",
J. Ship Res. 8 (1964); see also J. Katz and A. Plotkin, "Low-Speed
Aerodynamics", 2nd ed., Cambridge, 2001, Ch. 11.
"""
from __future__ import annotations

import numpy as np

from .mesh import Mesh

__all__ = ["PanelSolver", "potential_flow_state"]


class PanelSolver:
    """Constant-strength source panels + one global constant vortex, with
    the Kutta condition at a sharp trailing edge."""

    def __init__(self, xv: np.ndarray, yv: np.ndarray):
        # vertices, ordered from the trailing edge along the LOWER surface
        # to the leading edge and back along the UPPER surface (clockwise
        # when x points right and y up) -- the C-grid wall ordering.
        self.xv = np.asarray(xv, dtype=float)
        self.yv = np.asarray(yv, dtype=float)
        self.n = len(self.xv) - 1
        x, y = self.xv, self.yv
        self.xm = 0.5 * (x[:-1] + x[1:])               # control points
        self.ym = 0.5 * (y[:-1] + y[1:])
        dx = x[1:] - x[:-1]
        dy = y[1:] - y[:-1]
        self.len = np.hypot(dx, dy)
        self.tx, self.ty = dx / self.len, dy / self.len   # tangents
        self.nx, self.ny = -self.ty, self.tx              # normals (into flow)

        # influence coefficient matrices at the control points
        self._build_influence()

    def _panel_velocity(self, px, py, j):
        """Velocity influence (x, y) at point (px, py) of panel j with unit
        source strength (factor 1/2pi included) and with unit vortex
        strength.  Returns (u_s, v_s, u_v, v_v)."""
        x1, y1 = self.xv[j], self.yv[j]
        x2, y2 = self.xv[j + 1], self.yv[j + 1]
        dxp = px - x1
        dyp = py - y1
        lx = x2 - x1
        ly = y2 - y1
        L2 = lx * lx + ly * ly
        # r1, r2 squared
        r1 = dxp * dxp + dyp * dyp
        r2 = (px - x2) ** 2 + (py - y2) ** 2
        # angles
        th1 = np.arctan2(dyp, dxp)
        th2 = np.arctan2(py - y2, px - x2)
        dth = th2 - th1
        # wrap to (-pi, pi]; points exactly ON the panel take the flow-side
        # limit (+pi) of the subtended angle
        dth = np.mod(dth + np.pi, 2 * np.pi) - np.pi
        dth = np.where(dth <= -np.pi + 1e-12, np.pi, dth)
        # log term (with a small regularisation for on-panel points)
        with np.errstate(divide="ignore", invalid="ignore"):
            lr = 0.5 * np.log(np.maximum(r2, 1e-300) / np.maximum(r1, 1e-300))
        # the projection coefficients along the panel frame
        # (see Katz & Plotkin, Appendix C)
        f = (dxp * lx + dyp * ly) / L2
        g = (dyp * lx - dxp * ly) / L2
        ln_term = 0.5 * np.log(np.maximum((f - 1) ** 2 + g ** 2, 1e-300)
                               / np.maximum(f ** 2 + g ** 2, 1e-300))
        ang_term = np.arctan2(g, f ** 2 + g ** 2 - f) - np.arctan2(
            g, f ** 2 + g ** 2) if False else dth
        # velocity in panel frame (unit strength, 1/2pi absorbed):
        #   source: u_t = ln(r1/r2), u_n = (theta2 - theta1)
        #   vortex: u_t = (theta2 - theta1), u_n = ln(r2/r1)
        # (Katz & Plotkin, Appendix C; positive panel gamma corresponds to a
        #  clockwise circulation in the global frame.)
        us_t = -ln_term
        us_n = ang_term
        uv_t = ang_term
        uv_n = ln_term
        tx, ty = self.tx[j], self.ty[j]
        nx, ny = self.nx[j], self.ny[j]
        return (us_t * tx + us_n * nx, us_t * ty + us_n * ny,
                uv_t * tx + uv_n * nx, uv_t * ty + uv_n * ny)

    def _build_influence(self):
        n = self.n
        self.Asx = np.zeros((n, n)); self.Asy = np.zeros((n, n))
        self.Avx = np.zeros((n, n)); self.Avy = np.zeros((n, n))
        for j in range(n):
            usx, usy, uvx, uvy = self._panel_velocity(self.xm, self.ym, j)
            self.Asx[:, j] = usx
            self.Asy[:, j] = usy
            self.Avx[:, j] = uvx
            self.Avy[:, j] = uvy
        # self-influence: the control point lies exactly ON its own panel,
        # where the angle-subtended term is its principal value +pi (the
        # flow side); the angle wrap above would return -pi.
        for i in range(n):
            self.Asx[i, i] = np.pi * self.nx[i]
            self.Asy[i, i] = np.pi * self.ny[i]
            self.Avx[i, i] = np.pi * self.tx[i]
            self.Avy[i, i] = np.pi * self.ty[i]

    def solve(self, u_inf: float, v_inf: float):
        """Solve for the source strengths and the circulation with tangency
        at the control points and the Kutta condition."""
        n = self.n
        # tangency: (Vinf + sum_j sigma_j Us_j + gamma Uv_j) . n_i = 0
        A = np.zeros((n + 1, n + 1))
        b = np.zeros(n + 1)
        A[:n, :n] = self.Asx * self.nx[:, None] + self.Asy * self.ny[:, None]
        A[:n, n] = (self.Avx * self.nx[:, None]
                    + self.Avy * self.ny[:, None]).sum(axis=1)
        b[:n] = -(u_inf * self.nx + v_inf * self.ny)
        # Kutta: equal tangential velocity magnitudes at the first (lower
        # TE) and last (upper TE) panel control points, with the panel
        # tangents pointing in opposite streamwise senses:
        tl = np.array([self.tx[0], self.ty[0]])
        tu = np.array([self.tx[-1], self.ty[-1]])
        # the TE panel tangents are anti-parallel (both point away from the
        # TE along the contour), so equal TE velocities mean
        #     (V . t_lower) + (V . t_upper) = 0
        A[n, :n] = (self.Asx[0] * tl[0] + self.Asy[0] * tl[1]) \
            + (self.Asx[-1] * tu[0] + self.Asy[-1] * tu[1])
        A[n, n] = ((self.Avx[0] * tl[0] + self.Avy[0] * tl[1]).sum()
                   + (self.Avx[-1] * tu[0] + self.Avy[-1] * tu[1]).sum())
        b[n] = -((u_inf * tl[0] + v_inf * tl[1])
                 + (u_inf * tu[0] + v_inf * tu[1]))
        sol = np.linalg.solve(A, b)
        self.sigma = sol[:n]
        self.gamma = float(sol[n])
        return self.sigma, self.gamma

    def velocity(self, px, py):
        """Velocity field of the solved panel solution at arbitrary points."""
        u = np.full_like(np.asarray(px, dtype=float), 0.0)
        v = np.full_like(u, 0.0)
        u += self._u_inf
        v += self._v_inf
        for j in range(self.n):
            usx, usy, uvx, uvy = self._panel_velocity(px, py, j)
            u += self.sigma[j] * usx + self.gamma * uvx
            v += self.sigma[j] * usy + self.gamma * uvy
        return u, v

    def set_freestream(self, u_inf, v_inf):
        self._u_inf, self._v_inf = u_inf, v_inf


def _airfoil_vertices(n_panels: int = 240) -> tuple[np.ndarray, np.ndarray]:
    """NACA 0012 (sharp -0.1036 closure) contour vertices, C-grid wall
    ordering: TE -> lower surface -> LE -> upper surface -> TE, clustered
    towards the leading and trailing edges."""
    def yt(x):
        return 0.6 * (0.2969 * np.sqrt(np.maximum(x, 0.0)) - 0.1260 * x
                      - 0.3516 * x ** 2 + 0.2843 * x ** 3 - 0.1036 * x ** 4)
    # cosine clustering in x on each side
    n_side = n_panels // 2
    b = (np.arange(n_side + 1) / n_side) * np.pi
    x = 0.5 * (1 - np.cos(b))          # clustered at both ends
    # slightly relax the TE clustering (sharp TE, cosine is fine)
    yu = yt(x)
    xl = x
    yl = -yt(xl)
    # ordering: TE (1, 0) -> lower (x decreasing? we go TE->LE along lower)
    xs = np.concatenate([xl[1:][::-1], xl])
    ys = np.concatenate([yl[1:][::-1], yu])
    return xs, ys


def potential_flow_state(mesh: Mesh, Mach: float, alpha_deg: float,
                         gamma_gas: float = 1.4,
                         n_panels: int = 240) -> np.ndarray:
    """Initial DG state from the panel solution (isentropic density/energy),
    blended to the exact free stream in the far field."""
    a = np.radians(alpha_deg)
    u_inf, v_inf = Mach * np.cos(a), Mach * np.sin(a)
    ps = PanelSolver(*_airfoil_vertices(n_panels))
    ps.set_freestream(u_inf, v_inf)
    ps.solve(u_inf, v_inf)

    x = mesh.X.copy().ravel()
    y = mesh.Y.copy().ravel()
    # Points exactly on the panel vertices see a log-singular tangential
    # velocity from the individual panels; offset the wall-band nodes a
    # tiny distance into the fluid (the flow-side limit is finite).
    shp = mesh.X.shape
    wei = getattr(mesh, "wall_airfoil", None)
    if wei is None:
        wei = np.arange(mesh.n_k)
    fem = mesh.Fem[wei, 0, :, :]                 # (k, i, 2), into the body
    nrm = np.hypot(fem[..., 0], fem[..., 1])
    nrm = np.where(nrm < 1e-300, 1.0, nrm)
    offx = np.zeros(shp)
    offy = np.zeros(shp)
    offx[wei, 0, :, 0] = -1e-4 * fem[..., 0] / nrm
    offy[wei, 0, :, 0] = -1e-4 * fem[..., 1] / nrm
    x = (x + offx.ravel())
    y = (y + offy.ravel())
    u, v = ps.velocity(x, y)
    # blend to free stream far away (the panel field decays like 1/r; blend
    # for r > 15 chords to keep the far field exactly uniform)
    r = np.hypot(x - 0.5, y)
    w = np.clip((15.0 - r) / 5.0, 0.0, 1.0)
    u = u_inf + w * (u - u_inf)
    v = v_inf + w * (v - v_inf)

    q2 = u * u + v * v
    q_inf2 = u_inf * u_inf + v_inf * v_inf
    p_inf = 1.0 / gamma_gas
    # incompressible Bernoulli with the isentropic closure for rho, E
    p = p_inf + 0.5 * (q_inf2 - q2)
    rho_inf = 1.0
    rho = rho_inf * np.maximum(p / p_inf, 0.05) ** (1.0 / gamma_gas)
    E = p / (gamma_gas - 1.0) + 0.5 * rho * q2
    U = np.stack([rho, rho * u, rho * v, E], axis=-1)
    return U.reshape(mesh.X.shape + (4,)), ps


def _self_test() -> None:  # pragma: no cover
    import time
    from .mesh import generate_airfoil_omesh
    print("panel method self test")
    t0 = time.time()
    xv, yv = _airfoil_vertices(240)
    ps = PanelSolver(xv, yv)
    for alpha in (0.0, 5.0, 10.0):
        a = np.radians(alpha)
        ps.set_freestream(np.cos(a), np.sin(a))
        ps.solve(np.cos(a), np.sin(a))
        # lift from the Kutta-Joukowski value of gamma (sum over panels)
        # positive panel gamma = clockwise circulation of strength
        # 2*pi*gamma*L (the 1/2pi is absorbed in the coefficients):
        # Cl = L'/(0.5 rho U^2 c) = 4 pi gamma L / (U c)
        cl_panel = 4.0 * np.pi * ps.gamma * ps.len.sum()
        print(f"  alpha={alpha:5.1f}: panel Cl = {cl_panel:+.4f} "
              f"(thin-airfoil {2*np.pi*np.sin(a):+.4f})")
    mesh = generate_airfoil_omesh(28, 21, 3, 50.0, 0.30 / 21)
    U, ps2 = potential_flow_state(mesh, 0.15, 5.0)
    print(f"  field init: rho range {U[...,0].min():.3f}..{U[...,0].max():.3f}, "
          f"finite: {np.isfinite(U).all()}, [{time.time()-t0:.1f}s]")
    assert np.isfinite(U).all()
    print("  panel self test passed")


if __name__ == "__main__":
    _self_test()
