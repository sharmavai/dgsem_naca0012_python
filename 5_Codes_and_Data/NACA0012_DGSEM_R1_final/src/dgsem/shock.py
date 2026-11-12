"""Modal shock sensor and element-wise artificial viscosity.

The sensor follows the Persson-Peraire idea: for each element the pressure is
projected onto the highest Legendre modal degree in each reference
direction; the normalised highest-mode energy fraction

    S_dir = (2N+1)/2 * a_N^2 / integral(p^2)

is small for well-resolved smooth solutions and becomes O(1) when the
element contains an unresolved (shock) feature.  The element sensor is the
maximum over the two directions, dilated once onto face neighbours.

The artificial viscosity ramps linearly-quadratically between the thresholds
(s0, s1) up to nu_max = c_nu * h_e * lambda_e (h_e: element size, lambda_e:
local spectral radius), and is applied to the conservative variables through
the BR2/SIPG operator in dgsem.operator.

Reference: P.-O. Persson and J. Peraire, "Sub-cell shock capturing for
discontinuous Galerkin methods", AIAA Paper 2006-112.
"""
from __future__ import annotations

import numpy as np

from .quadrature import legendre

__all__ = ["shock_sensor", "artificial_viscosity"]


def _legendre_values(xi: np.ndarray, n: int) -> np.ndarray:
    return np.array([legendre(float(x), n)[0] for x in xi])


def shock_sensor(U, mesh, gamma: float) -> np.ndarray:
    """Element shock sensor S in [0, ~1], shape (n_k, n_j)."""
    from .euler import pressure
    N = mesh.N
    p = pressure(U, gamma)                                    # (k,j,i,l)
    L_N = _legendre_values(mesh.xi, N)                        # (Np1,)
    w = mesh.w
    # quadrature weights tensor
    wxi = w[None, :]
    weta = w[:, None]
    denom = np.einsum('i,l,kjil->kj', w, w, p * p) + 1e-30

    # highest Legendre mode in the xi direction (integrate over eta)
    a_xi = np.einsum('i,l,kjil->kj', w * L_N, w, p)
    # highest Legendre mode in the eta direction (integrate over xi)
    a_eta = np.einsum('i,l,kjil->kj', w, w * L_N, p)

    fac = 0.5 * (2 * N + 1)
    S = np.maximum((fac * a_xi ** 2) / denom, (fac * a_eta ** 2) / denom)

    # one-layer dilation (max over face neighbours) for robustness
    S_d = S.copy()
    S_d = np.maximum(S_d, np.roll(S, 1, axis=0))
    S_d = np.maximum(S_d, np.roll(S, -1, axis=0))
    if mesh.n_j > 1:
        S_d[:, 1:] = np.maximum(S_d[:, 1:], S[:, :-1])
        S_d[:, :-1] = np.maximum(S_d[:, :-1], S[:, 1:])
    return S_d


def artificial_viscosity(U, mesh, gamma: float,
                         s0: float = 1.0e-3, s1: float = 5.0e-2,
                         c_nu: float = 0.35) -> tuple[np.ndarray, np.ndarray]:
    """Element-wise artificial viscosity (nu, sensor).

    The viscosity is a purely local function of the current state: no
    pseudo-time history is used, so the shock capturing cannot make the
    scheme history-dependent (a symmetric pair of runs stays exactly
    antisymmetric).  It is inactive (nu = 0) wherever the sensor stays
    below s0, which includes the entire subsonic airfoil flow of this
    paper; the mechanism is exercised by the Sod shock-tube test.
    """
    from .euler import primitives
    S = shock_sensor(U, mesh, gamma)

    rho, u, v, p = primitives(U, gamma)
    c = np.sqrt(gamma * p / rho)
    lam = (np.hypot(u, v) + c).max(axis=(2, 3))               # (k, j)
    h = np.minimum(mesh.h_xi, mesh.h_eta)
    nu_max = c_nu * h * lam

    ramp = np.clip((S - s0) / max(s1 - s0, 1e-30), 0.0, 1.0)
    ramp = ramp * ramp                                        # quadratic
    nu = nu_max * ramp
    return nu, S


def _self_test() -> None:  # pragma: no cover
    from .mesh import generate_airfoil_omesh
    from .euler import initialize_freestream
    mesh = generate_airfoil_omesh(12, 8, 3, 20.0, 0.01)
    U, _ = initialize_freestream(mesh, 0.8, 0.0)
    S = shock_sensor(U, mesh, 1.4)
    print(f"smooth-field sensor: max S = {S.max():.3e} (must be small)")
    assert S.max() < 1e-6
    # inject a jump: make one element's pressure oscillate at the highest mode
    U2 = U.copy()
    from .quadrature import legendre
    L = np.array([legendre(float(x), mesh.N)[0] for x in mesh.xi])
    U2[5, 3, :, :, 3] += 0.5 * L[None, :] * 0.0 + 0.5 * L[:, None] * 3.0
    S2 = shock_sensor(U2, mesh, 1.4)
    print(f"perturbed sensor: max S = {S2.max():.3e} (must be O(1) at k=5)")
    assert S2.max() > 0.1
    nu, _ = artificial_viscosity(U2, mesh, 1.4)
    print(f"artificial viscosity: max nu = {nu.max():.3e} at the perturbed element")
    assert nu[5, 3] > 0.0
    print("  shock sensor self test passed")


if __name__ == "__main__":
    _self_test()
