"""Body-fitted O-type mesh generation around the NACA 0012 airfoil.

Topology
--------
The computational domain is a rectangle in (xi, eta): xi is circumferential
and PERIODIC (the seam lies on the wake line y = 0 emanating from the sharp
trailing-edge point), eta runs from the airfoil contour (eta = -1 face of
the first radial element band) to the far-field circle (eta = +1 of the last).

Every element's eta = -1 face lies on the airfoil (slip wall boundary); every
element's eta = +1 face of the outermost band lies on the far-field circle.
There are no internal cuts, branch points or degenerate corners.

Construction
------------
1. The airfoil contour and the far-field circle are both parameterised by the
   same clustered arc-length parameter t in [0, 1), so element boundaries and
   nodes match one-to-one between the two boundary curves.
2. Radial node lines are geometrically stretched from a prescribed first-layer
   height h_wall at the airfoil to the circle, with the growth ratio solved
   per line (vectorised bisection) so that the last level lands exactly on the
   outer boundary.
3. Node positions follow the algebraic two-boundary (transfinite) blending
      X(t, sigma) = (1 - sigma) B(t) + sigma F(t).
4. Metric terms are computed by collocation differentiation of the node
   coordinates in conservative (curl) form,
      J xi_x = Y_eta,  J xi_y = -X_eta,  J eta_x = -Y_xi,  J eta_y = X_xi,
   which satisfies the discrete metric identities exactly (free-stream
   preservation / geometric conservation law; Kopriva, J. Sci. Comput. 2006).
   Face metric vectors are rows of the same arrays, so they are single-valued
   across interior faces for this conforming construction.

The module also provides a periodic warped-box mesh factory used by the
verification suite (isentropic vortex advection) to exercise the curvilinear
operator on a smooth non-Cartesian periodic geometry.

Run `python -m dgsem.mesh` for the self test (Jacobian positivity, geometric
conservation law, face matching, boundary accuracy).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .geometry import AirfoilContour, TE_HALF_HEIGHT
from .quadrature import derivative_matrix

__all__ = ["Mesh", "generate_airfoil_omesh", "generate_periodic_box_mesh",
           "generate_channel_mesh", "mirror_index_map"]


# ----------------------------------------------------------------------
@dataclass
class Mesh:
    """ DGSEM mesh arrays, layout (k, j, i, l) = (xi-element, eta-element,
    xi-GLL-node, eta-GLL-node).  Variables are appended as a last axis by the
    solver. """

    n_k: int                      # number of xi (circumferential) elements
    n_j: int                      # number of eta (radial) elements
    N: int                        # polynomial degree
    periodic_xi: bool = True
    periodic_eta: bool = False
    has_wall: bool = True         # eta = -1 faces of the j = 0 band
    has_farfield: bool = True     # eta = +1 faces of the j = n_j - 1 band
    wall_top: bool = False        # eta = +1 faces of the outer band are also
                                  # slip walls (channel test meshes)
    topology: str = "omesh"

    X: np.ndarray = field(default_factory=lambda: np.zeros(0))
    Y: np.ndarray = field(default_factory=lambda: np.zeros(0))
    J: np.ndarray = field(default_factory=lambda: np.zeros(0))
    Jxi_x: np.ndarray = field(default_factory=lambda: np.zeros(0))
    Jxi_y: np.ndarray = field(default_factory=lambda: np.zeros(0))
    Jeta_x: np.ndarray = field(default_factory=lambda: np.zeros(0))
    Jeta_y: np.ndarray = field(default_factory=lambda: np.zeros(0))
    Fxp: np.ndarray = field(default_factory=lambda: np.zeros(0))  # (k,j,l,2)
    Fxm: np.ndarray = field(default_factory=lambda: np.zeros(0))  # (k,j,l,2)
    Fep: np.ndarray = field(default_factory=lambda: np.zeros(0))  # (k,j,i,2)
    Fem: np.ndarray = field(default_factory=lambda: np.zeros(0))  # (k,j,i,2)
    h_xi: np.ndarray = field(default_factory=lambda: np.zeros(0))  # (k,j)
    h_eta: np.ndarray = field(default_factory=lambda: np.zeros(0))  # (k,j)
    volume: np.ndarray = field(default_factory=lambda: np.zeros(0))

    D: np.ndarray = field(default_factory=lambda: np.zeros(0))
    xi: np.ndarray = field(default_factory=lambda: np.zeros(0))
    w: np.ndarray = field(default_factory=lambda: np.zeros(0))

    # airfoil-specific extras
    contour: AirfoilContour | None = None
    R_far: float = 0.0
    t_of_xi: np.ndarray = field(default_factory=lambda: np.zeros(0))  # (n_k, N+1)

    # ---- derived helpers ------------------------------------------------
    @property
    def Np1(self) -> int:
        return self.N + 1

    def wall_faces(self) -> tuple[np.ndarray, np.ndarray]:
        """(x, y) coordinates and metric vectors of all eta = -1 faces of the
        j = 0 band (the airfoil surface).  Returns (points, sJn) with
        points shape (n_k, N+1, 2) and sJn shape (n_k, N+1, 2)."""
        pts = np.stack([self.X[:, 0, :, 0], self.Y[:, 0, :, 0]], axis=-1)
        return pts, self.Fem[:, 0, :, :]

    def metric_identities(self) -> tuple[float, float]:
        """Max-norm of the discrete metric identities
        D_xi Jxi_x + D_eta Jeta_x = 0 (and the y version), normalised by the
        largest metric magnitude (scale-free geometric-conservation measure)."""
        dx = np.einsum('ab,kjbl->kjal', self.D, self.Jxi_x) \
            + np.einsum('ba,kjia->kjib', self.D, self.Jeta_x)
        dy = np.einsum('ab,kjbl->kjal', self.D, self.Jxi_y) \
            + np.einsum('ba,kjia->kjib', self.D, self.Jeta_y)
        scale = max(np.abs(self.Jxi_x).max(), np.abs(self.Jxi_y).max(),
                    np.abs(self.Jeta_x).max(), np.abs(self.Jeta_y).max(), 1e-300)
        return float(np.abs(dx).max() / scale), float(np.abs(dy).max() / scale)

    def quality_report(self) -> dict:
        J = self.J
        # element-averaged Jacobian must be positive (volume)
        vol = 0.25 * np.einsum('i,l,kjil->kj', self.w, self.w, J)
        # aspect/skew indicators from the metric tensors
        g11 = self.Jxi_x ** 2 + self.Jxi_y ** 2
        g22 = self.Jeta_x ** 2 + self.Jeta_y ** 2
        g12 = self.Jxi_x * self.Jeta_x + self.Jxi_y * self.Jeta_y
        cos_skew = np.abs(g12) / np.sqrt(g11 * g22)
        return {
            "min_J": float(J.min()), "max_J": float(J.max()),
            "min_volume": float(vol.min()),
            "max_skew_cos": float(cos_skew.max()),
            "min_h_xi": float(self.h_xi.min()), "min_h_eta": float(self.h_eta.min()),
            "max_h_xi": float(self.h_xi.max()), "max_h_eta": float(self.h_eta.max()),
        }


def _growth_ratio(h0: np.ndarray, span: np.ndarray, n: int) -> np.ndarray:
    """Solve h0 (r^n - 1)/(r - 1) = span elementwise (bisection, vectorised).

    The solution r > 1 exists when span > h0 n (growing layers); for
    span < h0 n the solution is r < 1 (shrinking layers); span = h0 n gives
    r = 1 (uniform).  All three cases are bracketed.
    """
    h0 = np.asarray(h0, dtype=float)
    span = np.asarray(span, dtype=float)
    lo = np.full_like(span, 1e-9)
    hi = np.full_like(span, 10.0)
    # f(r) is monotonically increasing in r; f(1) = h0 n - span
    for _ in range(200):
        rm = 0.5 * (lo + hi)
        rm1 = np.where(np.abs(rm - 1.0) < 1e-12, 1.0 + 1e-12, rm)
        f = h0 * (rm1 ** n - 1.0) / (rm1 - 1.0) - span
        lo = np.where(f < 0, rm, lo)
        hi = np.where(f >= 0, rm, hi)
    return 0.5 * (lo + hi)


# ----------------------------------------------------------------------
def generate_airfoil_omesh(
        n_half: int = 40, n_r: int = 28, N: int = 3, R_far: float = 30.0,
        h_wall: float = 0.006, h_end: float | None = None,
        center: tuple[float, float] = (0.5, 0.0), verbose: bool = False,
        wake_curve: float = 0.0, wake_blend: float = 1.2) -> Mesh:
    """O-mesh around the NACA 0012 with N-degree elements.

    Parameters
    ----------
    n_half : elements per airfoil surface (2 n_half total circumferential;
             the periodic seam sits at the sharp trailing-edge point).
    n_r    : radial element layers.
    N      : polynomial degree (N+1 LGL nodes per direction).
    R_far  : far-field circle radius (chords), centred at `center`.
    h_wall : first radial layer height at the airfoil (chords).
    h_end  : minimum circumferential element arc near LE / TE
             (default: 0.005 * 40 / n_half to scale under refinement).
    wake_curve : coefficient a of the wake-alignment deformation
             y -> y - a * ln(1 + max(x-1, 0)) blended over a Gaussian band
             of half width `wake_blend` about y = 0.  a = Cl_est / (4 pi)
             with Cl_est the theoretical estimate of the lift coefficient
             aligns the periodic seam (the mesh wake line) with the
             expected trailing vortex sheet, which is then captured as a
             face-aligned contact.  a = 0 (default) disables the
             deformation (the airfoil surface is never affected: the
             deformation is identically zero for x <= 1).
    """
    if h_end is None:
        h_end = 0.005 * 40.0 / n_half
    contour = AirfoilContour()
    brk = contour.element_breakpoints(n_half, h_end)
    n_theta = len(brk) - 1
    D, xi, w = derivative_matrix(N)
    Np1 = N + 1

    # ---- global circumferential parameter t at all GLL nodes -------------
    # t_nodes[k, i]: parameter of element k, xi-node i (t in [0, 1)).
    t_nodes = np.zeros((n_theta, Np1))
    for k in range(n_theta):
        t_nodes[k, :] = (brk[k] + 0.5 * (xi + 1.0) * (brk[k + 1] - brk[k])) / contour.S

    # Reverse the circumferential direction so that (xi, eta) is a
    # right-handed frame: the contour is traversed counter-clockwise
    # (fluid on the left, eta pointing outward), giving J > 0.
    t_nodes = np.mod(t_nodes[::-1, ::-1], 1.0)
    # s at the nodes (arc length, for the airfoil curve)
    s_nodes = t_nodes * contour.S

    # ---- inner boundary (airfoil) at the node s-positions ----------------
    Xin = np.zeros((n_theta, Np1))
    Yin = np.zeros((n_theta, Np1))
    for k in range(n_theta):
        Xin[k, :], Yin[k, :] = contour.points(s_nodes[k, :])

    # ---- outer boundary (circle) at the same t ---------------------------
    theta = 2.0 * np.pi * t_nodes
    Xout = center[0] + R_far * np.cos(theta)
    Yout = center[1] + R_far * np.sin(theta)

    # ---- radial distribution: geometric stretch per (k, i) line ----------
    span = np.hypot(Xout - Xin, Yout - Yin)              # (n_theta, Np1)
    h0 = np.full_like(span, h_wall)
    r = _growth_ratio(h0, span, n_r)
    # physical radial levels d_l(k, i), l = 0..n_r  (d_0 = 0, d_n = span)
    d_lev = np.zeros((n_theta, Np1, n_r + 1))
    for lev in range(n_r + 1):
        d_lev[:, :, lev] = h0 * (r ** lev - 1.0) / (r - 1.0)
    d_lev[:, :, n_r] = span                       # land exactly on the circle

    # ---- node positions by transfinite blending ---------------------------
    m = Mesh(n_k=n_theta, n_j=n_r, N=N, periodic_xi=True, periodic_eta=False,
             has_wall=True, has_farfield=True, topology="omesh",
             D=D, xi=xi, w=w,
             contour=contour, R_far=R_far, t_of_xi=t_nodes)
    m.X = np.zeros((n_theta, n_r, Np1, Np1))
    m.Y = np.zeros((n_theta, n_r, Np1, Np1))
    eta_hat = 0.5 * (xi + 1.0)                     # node position in [0, 1] within a layer
    sig_lev = d_lev / span[:, :, None]             # radial fractions in [0, 1]
    for j in range(n_r):
        lo = sig_lev[:, :, j][:, :, None]          # (k, i, 1)
        hi = sig_lev[:, :, j + 1][:, :, None]
        sig = lo + eta_hat[None, None, :] * (hi - lo)          # (k, i, l)
        m.X[:, j, :, :] = (1.0 - sig) * Xin[:, :, None] + sig * Xout[:, :, None]
        m.Y[:, j, :, :] = (1.0 - sig) * Yin[:, :, None] + sig * Yout[:, :, None]

    # ---- optional wake alignment (seam follows the expected vortex sheet)
    if wake_curve != 0.0:
        xw = np.maximum(m.X - 1.0, 0.0)
        yw = -wake_curve * np.log1p(xw)               # wake dip (chords)
        blend = np.exp(-(m.Y / wake_blend) ** 2) * (xw / (xw + 1.0))
        m.Y = m.Y + blend * yw

    _finish_metrics(m)
    if verbose:
        q = m.quality_report()
        print(f"[mesh] n_theta={n_theta} n_r={n_r} N={N} "
              f"elements={n_theta * n_r} nodes={n_theta * n_r * Np1 * Np1}")
        print(f"[mesh] min(J)={q['min_J']:.4e} min(volume)={q['min_volume']:.4e} "
              f"max|cos skew|={q['max_skew_cos']:.3f}")
        print(f"[mesh] min h_xi={q['min_h_xi']:.4e} min h_eta={q['min_h_eta']:.4e}")
    return m


# ----------------------------------------------------------------------
def generate_periodic_box_mesh(
        n_x: int = 16, n_y: int = 16, N: int = 3, L: float = 10.0,
        warp: float = 0.1) -> Mesh:
    """Smoothly warped periodic quadrilateral mesh on [0, L]^2 (both
    directions periodic), used for the isentropic-vortex verification.

    The mapping
        x = xi  + warp * sin(2 pi xi / L) * cos(2 pi eta / L) * L/(2 pi)
        y = eta + warp * cos(2 pi xi / L) * sin(2 pi eta / L) * L/(2 pi)
    is periodic in both directions and has a positive Jacobian for
    warp < ~0.7, so it warps the mesh without folding it.
    """
    D, xi, w = derivative_matrix(N)
    Np1 = N + 1
    m = Mesh(n_k=n_x, n_j=n_y, N=N, periodic_xi=True, periodic_eta=True,
             has_wall=False, has_farfield=False, D=D, xi=xi, w=w)
    m.X = np.zeros((n_x, n_y, Np1, Np1))
    m.Y = np.zeros((n_x, n_y, Np1, Np1))
    for k in range(n_x):
        for j in range(n_y):
            a = L * (k + 0.5 * (xi[:, None] + 1.0)) / n_x       # (Np1, 1)
            b = L * (j + 0.5 * (xi[None, :] + 1.0)) / n_y       # (1, Np1)
            m.X[k, j, :, :] = a + warp * np.sin(2 * np.pi * a / L) * np.cos(
                2 * np.pi * b / L) * L / (2 * np.pi)
            m.Y[k, j, :, :] = b + warp * np.cos(2 * np.pi * a / L) * np.sin(
                2 * np.pi * b / L) * L / (2 * np.pi)
    _finish_metrics(m)
    return m


# ----------------------------------------------------------------------
def _finish_metrics(m: Mesh) -> None:
    """Collocation derivatives -> conservative metrics, face metric vectors,
    element sizes, volumes, and the positivity check."""
    D = m.D
    X, Y = m.X, m.Y
    # X_xi = D @ X (contract the i axis), X_eta = X @ D^T (contract the l axis)
    X_xi = np.einsum('ab,kjbl->kjal', D, X)
    Y_xi = np.einsum('ab,kjbl->kjal', D, Y)
    X_eta = np.einsum('ba,kjia->kjib', D, X)
    Y_eta = np.einsum('ba,kjia->kjib', D, Y)

    m.J = X_xi * Y_eta - X_eta * Y_xi
    if m.J.min() <= 0.0:
        idx = np.unravel_index(np.argmin(m.J), m.J.shape)
        raise RuntimeError(
            f"mesh has a non-positive Jacobian: min(J) = {m.J.min():.4e} "
            f"at element/node {idx}")

    m.Jxi_x, m.Jxi_y = Y_eta, -X_eta
    m.Jeta_x, m.Jeta_y = -Y_xi, X_xi

    # face metric vectors (outward, single-valued across conforming faces)
    m.Fxp = np.stack([m.Jxi_x[:, :, -1, :], m.Jxi_y[:, :, -1, :]], axis=-1)
    m.Fxm = np.stack([-m.Jxi_x[:, :, 0, :], -m.Jxi_y[:, :, 0, :]], axis=-1)
    m.Fep = np.stack([m.Jeta_x[:, :, :, -1], m.Jeta_y[:, :, :, -1]], axis=-1)
    m.Fem = np.stack([-m.Jeta_x[:, :, :, 0], -m.Jeta_y[:, :, :, 0]], axis=-1)

    # element sizes for the time-step and penalty estimates.
    # volume = quadrature of J over the reference element (sum w w J);
    # face lengths: sJ = ds/d(xi_ref) so sum w |F| = the true face length.
    m.volume = np.einsum('i,l,kjil->kj', m.w, m.w, m.J)
    Lxp = np.einsum('l,kjl->kj', m.w, np.hypot(m.Fxp[..., 0], m.Fxp[..., 1]))
    Lep = np.einsum('i,kji->kj', m.w, np.hypot(m.Fep[..., 0], m.Fep[..., 1]))
    # h_xi: element extent NORMAL to xi faces (the eta-direction size),
    # used for the xi-face viscous penalties; h_eta: normal to eta faces.
    m.h_xi = m.volume / np.maximum(Lep, 1e-300)
    m.h_eta = m.volume / np.maximum(Lxp, 1e-300)


# ----------------------------------------------------------------------
def _self_test() -> None:  # pragma: no cover
    import time
    t0 = time.time()
    print("O-mesh self test")
    for (n_half, n_r, N, R, h_wall) in [(24, 16, 3, 30.0, 0.008),
                                        (40, 28, 3, 30.0, 0.006),
                                        (40, 28, 4, 50.0, 0.005),
                                        (64, 40, 3, 30.0, 0.004)]:
        m = generate_airfoil_omesh(n_half, n_r, N, R, h_wall)
        q = m.quality_report()
        id_x, id_y = m.metric_identities()
        # airfoil boundary accuracy: wall nodes must sit on the analytic surface
        wx = m.X[:, 0, :, 0].ravel()
        wy = m.Y[:, 0, :, 0].ravel()
        on_side = np.abs(wx - 1.0) > 1e-12
        from .geometry import naca0012_thickness
        bnd_err = np.abs(np.abs(wy[on_side])
                         - naca0012_thickness(wx[on_side])).max()
        # periodic face matching: element k's xi=+1 face == element k+1's xi=-1 face
        gap = max(np.abs(m.X[:, :, -1, :] - np.roll(m.X, -1, axis=0)[:, :, 0, :]).max(),
                  np.abs(m.Y[:, :, -1, :] - np.roll(m.Y, -1, axis=0)[:, :, 0, :]).max())
        # face-metric single-valuedness: Fxp[k] == -Fxm[k+1]
        fm = max(np.abs(m.Fxp - np.roll(-m.Fxm, -1, axis=0)).max(),
                 np.abs(m.Fep[:, :-1] + m.Fem[:, 1:]).max())
        print(f"  n_half={n_half} n_r={n_r} N={N} R={R}: "
              f"min(J)={q['min_J']:.2e} GCL=({id_x:.1e},{id_y:.1e}) "
              f"bnd={bnd_err:.1e} gap={gap:.1e} facemetric={fm:.1e} "
              f"skew={q['max_skew_cos']:.3f}")
        assert q['min_J'] > 0 and id_x < 1e-12 and id_y < 1e-12
        assert bnd_err < 1e-9 and gap < 1e-13 and fm < 1e-13
        assert q['max_skew_cos'] < 0.9999

    # periodic box mesh
    m = generate_periodic_box_mesh(8, 8, 3, 10.0, 0.3)
    id_x, id_y = m.metric_identities()
    q = m.quality_report()
    print(f"  warped periodic box: min(J)={q['min_J']:.3e} "
          f"GCL=({id_x:.1e},{id_y:.1e})")
    assert q['min_J'] > 0 and id_x < 1e-12 and id_y < 1e-12
    print(f"  self test passed in {time.time() - t0:.2f} s")


# ----------------------------------------------------------------------
def generate_channel_mesh(n_x: int = 16, n_y: int = 8, N: int = 3,
                          Lx: float = 2.0, Ly: float = 1.0) -> Mesh:
    """Rectangular channel mesh on [0, Lx] x [0, Ly] used by the boundary
    condition and shock-tube verification tests.

    Topology: periodic in xi (x direction), slip walls at eta = -1 (y = 0)
    and eta = +1 (y = Ly).  There is no far-field boundary.  With the
    uniform free stream parallel to the walls (alpha = 0), the exact
    solution of the Euler equations with these slip walls IS the uniform
    state, so this mesh provides an exact test of the wall boundary
    condition (the mirror ghost must leave the state unchanged).
    """
    D, xi, w = derivative_matrix(N)
    Np1 = N + 1
    m = Mesh(n_k=n_x, n_j=n_y, N=N, periodic_xi=True, periodic_eta=False,
             has_wall=True, has_farfield=False, wall_top=True,
             D=D, xi=xi, w=w)
    m.X = np.zeros((n_x, n_y, Np1, Np1))
    m.Y = np.zeros((n_x, n_y, Np1, Np1))
    for k in range(n_x):
        m.X[k, :, :, :] = Lx * (k + 0.5 * (xi[:, None] + 1.0)) / n_x
    for j in range(n_y):
        m.Y[:, j, :, :] = Ly * (j + 0.5 * (xi[None, :] + 1.0)) / n_y
    _finish_metrics(m)
    return m


# ----------------------------------------------------------------------
def generate_sod_mesh(n_x: int = 64, n_y: int = 4, N: int = 3,
                      Lx: float = 2.0, Ly: float = 0.4) -> Mesh:
    """Rectangular mesh on [0, Lx] x [0, Ly] for the shock-tube verification:
    periodic in eta (the solution is y-invariant), non-periodic in xi.  The
    xi = -1 face of column 0 and the xi = +1 face of the last column receive
    constant ghost states (Operator.Q_xi_L / Q_xi_R)."""
    D, xi, w = derivative_matrix(N)
    Np1 = N + 1
    m = Mesh(n_k=n_x, n_j=n_y, N=N, periodic_xi=False, periodic_eta=True,
             has_wall=False, has_farfield=False, D=D, xi=xi, w=w)
    m.X = np.zeros((n_x, n_y, Np1, Np1))
    m.Y = np.zeros((n_x, n_y, Np1, Np1))
    for k in range(n_x):
        m.X[k, :, :, :] = Lx * (k + 0.5 * (xi[:, None] + 1.0)) / n_x
    for j in range(n_y):
        m.Y[:, j, :, :] = Ly * (j + 0.5 * (xi[None, :] + 1.0)) / n_y
    _finish_metrics(m)
    return m


# ----------------------------------------------------------------------
def mirror_index_map(mesh: Mesh) -> np.ndarray:
    """Element-index map k -> k' under reflection about y = 0 (O-mesh).

    For the airfoil O-mesh built from the symmetric NACA 0012 contour, the
    element columns pair up exactly: the column at circumferential parameter
    t maps to the column at 1 - t (with the node indices reversed in xi).
    The map is used by the mirror-symmetry tests (mesh symmetry and the
    antisymmetry of the +alpha / -alpha solutions)."""
    if getattr(mesh, "topology", "omesh") != "omesh" or mesh.contour is None:
        raise ValueError("mirror_index_map needs an O-mesh with a contour")
    # Element k covers the parameter interval [lo_k, hi_k] (the traversal
    # within an element may run either way).  Reflection about y = 0 maps
    # the contour parameter t -> 1 - t, so the mirrored element is the one
    # covering [1 - hi_k, 1 - lo_k] (mod 1).
    # unwrap elements whose interval crosses the seam (t = 0 ~ 1)
    tA, tB = mesh.t_of_xi[:, 0].copy(), mesh.t_of_xi[:, -1].copy()
    wrap = np.abs(tA - tB) > 0.5
    tA[wrap] += (tA[wrap] < 0.5)
    tB[wrap] += (tB[wrap] < 0.5)
    tA[wrap & (tA < tB)] = tA[wrap & (tA < tB)]  # no-op, keep clarity
    # after unwrapping both endpoints of wrapping elements exceed 1
    t0 = np.minimum(tA, tB)
    t1 = np.maximum(tA, tB)
    t_mid = 0.5 * (t0 + t1)
    t_mir = np.mod(1.0 - t_mid, 1.0)
    order = np.argsort(t_mid)
    smid = t_mid[order]
    pos = np.searchsorted(smid, t_mir)
    # nearest midpoints on both sides; pick the element whose interval
    # actually covers t_mir (or the nearer midpoint at the seam wrap)
    cand = np.stack([order[np.clip(pos - 1, 0, len(order) - 1)],
                     order[np.clip(pos, 0, len(order) - 1)]], axis=1)
    best = np.empty(len(t_mid), dtype=int)
    for k in range(len(t_mid)):
        c0, c1 = cand[k]
        d0 = abs(np.mod(t_mid[c0] - t_mir[k] + 0.5, 1.0) - 0.5)
        d1 = abs(np.mod(t_mid[c1] - t_mir[k] + 0.5, 1.0) - 0.5)
        best[k] = c0 if d0 <= d1 else c1
    kmap = best
    # verify: the mirrored interval midpoints must agree to well within the
    # larger of the two element extents
    err = np.abs(np.mod(t_mid[kmap] - t_mir + 0.5, 1.0) - 0.5)
    tol = 0.6 * np.maximum(t1 - t0, t1[kmap] - t0[kmap]).max()
    assert err.max() < max(tol, 1e-12), f"mirror map failed: {err.max()}"
    return kmap


if __name__ == "__main__":
    _self_test()
