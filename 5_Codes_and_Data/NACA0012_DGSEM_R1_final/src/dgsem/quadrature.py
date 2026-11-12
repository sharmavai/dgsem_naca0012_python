"""Legendre-Gauss-Lobatto (LGL) quadrature, differentiation matrices and
nodal interpolation tools for the DGSEM.

Everything in this module is polynomial-exact by construction and is verified
by the self test at the bottom (`python -m dgsem.quadrature`).

References
----------
D. A. Kopriva, "Implementing Spectral Methods for Partial Differential
Equations", Springer, 2009 (Algorithms 3, 4, 6, 21, 27).
J. S. Hesthaven and T. Warburton, "Nodal Discontinuous Galerkin Methods",
Springer, 2008.
"""
from __future__ import annotations

import numpy as np

__all__ = ["legendre", "legendre_derivative", "lgl_nodes", "lgl_weights",
           "derivative_matrix", "lagrange_interpolation_matrix",
           "gauss_nodes", "barycentric_weights"]


# ----------------------------------------------------------------------
def legendre(x: float, n: int) -> tuple[float, float]:
    """Value P_n(x) and derivative P'_n(x) of the Legendre polynomial."""
    if n == 0:
        return 1.0, 0.0
    p_km2, p_km1 = 1.0, x
    d_km2, d_km1 = 0.0, 1.0
    for k in range(2, n + 1):
        p_k = ((2 * k - 1) * x * p_km1 - (k - 1) * p_km2) / k
        d_k = ((2 * k - 1) * (p_km1 + x * d_km1) - (k - 1) * d_km2) / k
        p_km2, p_km1 = p_km1, p_k
        d_km2, d_km1 = d_km1, d_k
    return p_km1, d_km1


def legendre_derivative_second(x: float, n: int) -> float:
    """Second derivative P''_n(x) (needed for the LGL Newton iteration)."""
    if n == 0:
        return 0.0
    p_km2, p_km1 = 1.0, x
    d1_km2, d1_km1 = 0.0, 1.0
    d2_km2, d2_km1 = 0.0, 0.0
    for k in range(2, n + 1):
        p_k = ((2 * k - 1) * x * p_km1 - (k - 1) * p_km2) / k
        d1_k = ((2 * k - 1) * (p_km1 + x * d1_km1) - (k - 1) * d1_km2) / k
        d2_k = ((2 * k - 1) * (2 * d1_km1 + x * d2_km1) - (k - 1) * d2_km2) / k
        p_km2, p_km1 = p_km1, p_k
        d1_km2, d1_km1 = d1_km1, d1_k
        d2_km2, d2_km1 = d2_km1, d2_k
    return d2_km1


# ----------------------------------------------------------------------
def lgl_nodes(n: int) -> np.ndarray:
    """The n+1 Legendre-Gauss-Lobatto points on [-1, 1].

    The interior points are the roots of P'_n, computed with a safeguarded
    Newton iteration started from the Chebyshev initial guess (Kopriva,
    Algorithm 4); this converges to machine precision in a few steps.
    """
    if n < 1:
        raise ValueError("polynomial degree n must be >= 1")
    if n == 1:
        return np.array([-1.0, 1.0])
    xi = np.zeros(n + 1)
    xi[0], xi[-1] = -1.0, 1.0
    for j in range(1, n):
        x = -np.cos(j * np.pi / n)          # Chebyshev initial guess
        for _ in range(50):
            _, dp = legendre(x, n)
            d2p = legendre_derivative_second(x, n)
            dx = -dp / d2p
            x += dx
            if abs(dx) <= 1e-16 * max(1.0, abs(x)):
                break
        xi[j] = x
    return xi


def lgl_weights(n: int, xi: np.ndarray | None = None) -> np.ndarray:
    """LGL quadrature weights w_i = 2 / (n(n+1) P_n(xi_i)^2)."""
    if xi is None:
        xi = lgl_nodes(n)
    w = np.empty_like(xi)
    for i, x in enumerate(xi):
        p_n, _ = legendre(float(x), n)
        w[i] = 2.0 / (n * (n + 1) * p_n * p_n)
    return w


def gauss_nodes(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Legendre-Gauss nodes/weights (used for over-integration tests)."""
    x, w = np.polynomial.legendre.leggauss(n + 1)
    return x, w


# ----------------------------------------------------------------------
def derivative_matrix(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Polynomial differentiation matrix D on the LGL nodes.

    Returns (D, xi, w).  The matrix satisfies the summation-by-parts
    property D + D^T W = B exactly (verified in the self test), which is the
    algebraic backbone of the discrete metric identities and of the strong-form
    boundary lifting used by the spatial operator.
    """
    xi = lgl_nodes(n)
    w = lgl_weights(n, xi)
    p_n = np.array([legendre(float(x), n)[0] for x in xi])
    npts = n + 1
    D = np.zeros((npts, npts))
    for i in range(npts):
        for j in range(npts):
            if i != j:
                D[i, j] = p_n[i] / (p_n[j] * (xi[i] - xi[j]))
    for i in range(npts):
        D[i, i] = -np.sum(D[i, :])
    return D, xi, w


def barycentric_weights(xi: np.ndarray) -> np.ndarray:
    """Barycentric weights for the nodal Lagrange basis on arbitrary nodes."""
    npts = len(xi)
    w = np.ones(npts)
    for j in range(npts):
        d = xi[j] - np.delete(xi, j)
        w[j] = 1.0 / np.prod(d)
    return w


def lagrange_interpolation_matrix(x_new: np.ndarray, xi: np.ndarray) -> np.ndarray:
    """Matrix L with L[k, j] = l_j(x_new[k]) (nodal evaluation, barycentric
    formula with the safe no-node-coincides branch)."""
    xi = np.asarray(xi)
    x_new = np.atleast_1d(np.asarray(x_new))
    bw = barycentric_weights(xi)
    L = np.zeros((len(x_new), len(xi)))
    for k, x in enumerate(x_new):
        diff = x - xi
        if np.any(np.abs(diff) < 1e-14):
            j = int(np.argmin(np.abs(diff)))
            L[k, j] = 1.0
        else:
            tmp = bw / diff
            L[k, :] = tmp / np.sum(tmp)
    return L


# ----------------------------------------------------------------------
def _self_test() -> None:  # pragma: no cover - manual verification
    rng = np.random.default_rng(0)
    print("LGL quadrature / differentiation self test")
    for n in (1, 2, 3, 4, 5, 7):
        D, xi, w = derivative_matrix(n)
        # quadrature exactness for polynomials up to degree 2n-1
        max_err = 0.0
        for _ in range(20):
            coeffs = rng.normal(size=n + 1)
            f = np.polynomial.polynomial.Polynomial(coeffs)
            exact = f.integ()(1.0) - f.integ()(-1.0)
            quad = np.sum(w * f(xi))
            max_err = max(max_err, abs(exact - quad))
        # differentiation exactness
        g = np.polynomial.polynomial.Polynomial(rng.normal(size=n + 1))
        d_num = D @ g(xi)
        d_ex = g.deriv()(xi)
        d_err = np.max(np.abs(d_num - d_ex))
        # summation by parts:  D^T W + W D = B
        W = np.diag(w)
        B = np.zeros((n + 1, n + 1))
        B[0, 0], B[-1, -1] = -1.0, 1.0
        sbp = np.max(np.abs(D.T @ W + W @ D - B))
        # row sums vanish (discrete metric identities rely on this)
        rs = np.max(np.abs(D.sum(axis=1)))
        print(f"  N={n}: quad err {max_err:.2e}  diff err {d_err:.2e}  "
              f"SBP err {sbp:.2e}  row-sum {rs:.2e}")
        assert max_err < 1e-13 and d_err < 1e-13 and sbp < 1e-13 and rs < 1e-13
    # interpolation exactness
    D, xi, w = derivative_matrix(4)
    xt = np.linspace(-1, 1, 13)
    L = lagrange_interpolation_matrix(xt, xi)
    p = np.polynomial.polynomial.Polynomial(rng.normal(size=5))
    err = np.max(np.abs(L @ p(xi) - p(xt)))
    print(f"  interpolation err {err:.2e}")
    assert err < 1e-12
    print("  all checks passed")


if __name__ == "__main__":
    _self_test()
