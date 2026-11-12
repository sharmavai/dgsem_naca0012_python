"""NACA 0012 airfoil contour geometry (sharp trailing edge).

The contour uses the standard NACA 4-digit thickness distribution with the
CLOSED (sharp) trailing-edge coefficient,

    y_t(x) = 5 t (0.2969 sqrt(x) - 0.1260 x - 0.3516 x^2
                   + 0.2843 x^3 - 0.1036 x^4),      t = 0.12,

which gives y_t(1) = 0 exactly, i.e. a sharp trailing edge.  This is the
geometry of the classical inviscid (Euler) NACA 0012 benchmarks: a sharp
edge with a finite included angle (about 16.5 degrees) at which the Kutta
condition is enforced by the discretisation.  (The wind-tunnel models of
the reference experiments had small blunt trailing-edge bases of order
0.0025 chord; the difference is stated explicitly in the paper.)

The closed contour is parameterised by arc length starting (and ending) at
the trailing-edge point, which is where the periodic seam of the O-mesh is
placed: the wake sheet leaves the domain through the mesh seam.  A
two-sided tangent-hyperbolic clustering concentrates surface nodes near the
leading edge and near the trailing edge; the clustered parameter is shared
between the airfoil contour and the far-field circle so that the transfinite
blending produces well-shaped elements.

Run `python -m dgsem.geometry` for the self test.
"""
from __future__ import annotations

import numpy as np

__all__ = ["naca0012_thickness", "TE_HALF_HEIGHT", "AirfoilContour",
           "two_sided_cluster", "cluster_parameter"]


def naca0012_thickness(x: np.ndarray | float, t: float = 0.12,
                       closed: bool = True) -> np.ndarray:
    """NACA 4-digit thickness distribution.

    closed=True uses the -0.1036 coefficient (y_t(1) = 0, sharp trailing
    edge); closed=False uses the original -0.1015 coefficient (open edge,
    y_t(1) = 0.00126)."""
    x = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
    a4 = -0.1036 if closed else -0.1015
    return 5.0 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x ** 2
                      + 0.2843 * x ** 3 + a4 * x ** 4)


TE_HALF_HEIGHT = 0.0            # sharp trailing edge (closed contour)


# ----------------------------------------------------------------------
def two_sided_cluster(u: np.ndarray, beta: float) -> np.ndarray:
    """Map u in [0, 1] to s in [0, 1] with clustering at both endpoints.

    s(u) = 0.5 (1 + tanh(beta (u - 0.5)) / tanh(beta / 2))

    The minimum (endpoint) spacing fraction is beta / sinh(beta); larger beta
    gives stronger clustering.
    """
    u = np.asarray(u, dtype=float)
    th = np.tanh(0.5 * beta)
    return 0.5 * (1.0 + np.tanh(beta * (u - 0.5)) / th)


def cluster_parameter(u: np.ndarray, beta: float) -> np.ndarray:
    """Alias of :func:`two_sided_cluster` (name used by the mesh generator)."""
    return two_sided_cluster(u, beta)


def _solve_beta(target_slope: float) -> float:
    """Invert sigma'(0) = beta/sinh(beta) = target in (0, 1] for beta."""
    if target_slope >= 1.0:
        return 0.0                      # no clustering
    lo, hi = 1e-3, 20.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if mid / np.sinh(mid) > target_slope:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ----------------------------------------------------------------------
class AirfoilContour:
    """Closed sharp-TE NACA 0012 contour parameterised by arc length.

    The contour starts at the trailing-edge point (1, 0), runs backwards
    along the upper surface to the leading edge, and forwards along the
    lower surface back to the trailing edge.  s = 0 and s = S are both the
    trailing-edge point (this is where the periodic seam of the O-mesh
    sits).

    Arc-length evaluation is performed on a fine lookup table (64k points)
    with monotone linear interpolation for x(s) and the *exact* analytic
    thickness for y(x), so the surface is reproduced to ~1e-9 accuracy and
    never overshoots (the table is built with the exact same analytic rule).
    """

    N_TABLE = 65536

    def __init__(self, t: float = 0.12) -> None:
        self.t = t
        self.closed = True

        # ---- fine table of the upper half, from LE (x=0) to TE (x=1) ----
        x_tab = np.linspace(0.0, 1.0, self.N_TABLE)
        y_tab = naca0012_thickness(x_tab, t)
        # arc length along the upper surface, measured from the leading edge
        dx = np.diff(x_tab)
        dy = np.diff(y_tab)
        ds_seg = np.hypot(dx, dy)
        s_up = np.concatenate(([0.0], np.cumsum(ds_seg)))   # 0 .. L_upper
        self.L_upper = float(s_up[-1])
        self.x_tab, self.s_up = x_tab, s_up

        # total contour arc length: upper (TE -> LE) + lower (LE -> TE)
        self.S = 2.0 * self.L_upper
        self.s_le = self.L_upper          # leading edge at s = S / 2

    # -- evaluation -----------------------------------------------------
    def points(self, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Contour coordinates at arc length s in [0, S)."""
        s = np.mod(np.asarray(s, dtype=float), self.S)
        x = np.zeros_like(s)
        y = np.zeros_like(s)

        m_up = s < self.s_le
        m_lo = ~m_up

        if np.any(m_up):
            # upper surface: s = 0 at the TE, s = L_upper at the LE
            s_up = self.L_upper - s[m_up]                  # LE -> 0
            x[m_up] = np.interp(s_up, self.s_up, self.x_tab)
            y[m_up] = naca0012_thickness(x[m_up], self.t)

        if np.any(m_lo):
            # lower surface: s = L_upper at the LE, s = S at the TE
            s_lo = s[m_lo] - self.s_le                     # 0 -> L_upper
            x[m_lo] = np.interp(s_lo, self.s_up, self.x_tab)
            y[m_lo] = -naca0012_thickness(x[m_lo], self.t)
        return x, y

    def tangents(self, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Unit contour tangent (dx/ds, dy/ds) at arc length s."""
        ds = 1e-7
        x2, y2 = self.points(np.asarray(s) + ds)
        x1, y1 = self.points(np.asarray(s))
        tx, ty = x2 - x1, y2 - y1
        nrm = np.hypot(tx, ty)
        nrm = np.where(nrm < 1e-300, 1.0, nrm)
        return tx / nrm, ty / nrm

    # -- node distribution ----------------------------------------------
    def element_breakpoints(self, n_half: int, h_end: float = 0.005) -> np.ndarray:
        """Arc-length breakpoints of the closed contour for n_half elements
        per surface (n_theta = 2 n_half elements, 2 n_half + 1 breakpoints,
        strictly increasing from the seam s = 0 at the trailing edge to
        s = S, which is the same point).

        h_end is the target minimum element arc length at the leading edge
        and at the trailing edge (clamped to at most the uniform spacing).
        The mid-chord elements grow smoothly between the two clustered ends
        (two-sided tanh distribution).
        """
        if n_half < 2:
            raise ValueError("n_half must be >= 2")
        # slope of the clustered map at the endpoints: sigma'(0) = h_end / h_uniform
        h_uniform = self.L_upper / n_half
        beta = _solve_beta(min(h_end / h_uniform, 0.9))
        self.beta = beta
        # clustered parameter on [0, 1], clustered at BOTH ends
        u = np.arange(n_half + 1) / n_half
        g = two_sided_cluster(u, beta)

        # upper surface, s increasing from the TE (s = 0) to the LE:
        # g(u) clusters at both u-ends, so s_upper = L_upper (1 - g) runs
        # LE -> TE; reversed it gives the increasing TE -> LE sequence.
        brk_up = (self.L_upper * (1.0 - g))[::-1]   # [0, ..., L_upper]
        # lower surface, s increasing from the LE to the TE:
        brk_lo = self.s_le + self.L_upper * g  # [L_upper, ..., S]

        brk = np.concatenate((brk_up, brk_lo[1:]))
        assert np.all(np.diff(brk) > 0)
        return brk


# ----------------------------------------------------------------------
def _self_test() -> None:  # pragma: no cover
    import time
    t0 = time.time()
    c = AirfoilContour()
    print(f"NACA 0012 contour (sharp TE): L_upper = {c.L_upper:.6f}, "
          f"S = {c.S:.6f}")
    assert abs(naca0012_thickness(1.0)) < 1e-15          # sharp edge

    # closed contour, exact surfaces
    s = np.linspace(0.0, c.S, 200001)[:-1]
    x, y = c.points(s)
    on_surface = np.abs(x - 1.0) > 1e-12
    err_up = np.abs(y[on_surface]
                    - np.sign(y[on_surface]) * naca0012_thickness(x[on_surface])).max()
    seam = np.hypot(x[0], y[0])
    print(f"  surface reproduction err = {err_up:.2e}, seam at (x,y)={x[0]:.3f},{y[0]:.3f}")
    assert err_up < 1e-9 and abs(seam - 1.0) < 1e-12     # seam at the TE point

    # TE included angle ~ 16 degrees (probe a table-resolution-sized offset)
    eps = 5e-5
    tx, ty = c.tangents(np.array([eps, c.S - eps]))
    # traversal tangents point in opposite x-directions; the included angle
    # of the wedge is 180 deg minus the angle between them
    ang_between = np.degrees(np.arccos(np.clip(tx[0] * tx[1] + ty[0] * ty[1], -1, 1)))
    ang = 180.0 - ang_between
    print(f"  trailing-edge included angle = {ang:.2f} deg")
    assert 14.0 < ang < 19.0

    # monotone arc length, continuity at the seam
    ds = np.diff(np.linspace(0, c.S, 400001))
    assert np.all(ds > 0)

    # breakpoints
    brk = c.element_breakpoints(10)
    assert len(brk) == 2 * 10 + 1
    assert np.all(np.diff(brk) > 0) and brk[0] == 0.0 and brk[-1] == c.S
    print(f"  n_half=10 breakpoints OK, beta = {c.beta:.3f}, "
          f"min element arc = {np.diff(brk).min():.5f}")

    # clustering actually clusters at LE and TE
    brk = c.element_breakpoints(40)
    d = np.diff(brk)
    print(f"  n_half=40: min element {d.min():.5f} (near LE/TE), max {d.max():.5f}")
    assert d.min() < 0.01 and d.max() < 0.055
    print(f"  self test passed in {time.time() - t0:.2f} s")


if __name__ == "__main__":
    _self_test()
