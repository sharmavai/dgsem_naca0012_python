"""Local time-step estimate for explicit time integration.

For the DGSEM with polynomial degree N, the convective stability restriction
is estimated with the contravariant spectral radii (Kopriva 2009, Sec. 4.7)

    a_xi  = |u xi_x + v xi_y| + c |grad(xi)|,     a_eta analogous,

    dt_e  = CFL / ((N+1)^2 max_nodes(a_xi + a_eta)),

evaluated per element (local time stepping).  When diffusion is active
(artificial viscosity or physical viscosity) the parabolic restriction

    dt_v,e = CFL_v h_e^2 / ((N+1)^4 nu_e)

is applied per element as well.
"""
from __future__ import annotations

import numpy as np

from .euler import GAMMA, primitives

__all__ = ["local_timestep"]


def local_timestep(U, mesh, gamma: float = GAMMA, cfl: float = 0.35,
                   nu: np.ndarray | None = None, cfl_visc: float = 0.2,
                   mu_eff: float = 0.0) -> np.ndarray:
    """Per-element stable time step, shape (n_k, n_j)."""
    rho, u, v, p = primitives(U, gamma)
    c = np.sqrt(gamma * p / rho)
    with np.errstate(divide="ignore", invalid="ignore"):
        xi_x = mesh.Jxi_x / mesh.J
        xi_y = mesh.Jxi_y / mesh.J
        eta_x = mesh.Jeta_x / mesh.J
        eta_y = mesh.Jeta_y / mesh.J
    a_xi = np.abs(u * xi_x + v * xi_y) + c * np.sqrt(xi_x ** 2 + xi_y ** 2)
    a_eta = np.abs(u * eta_x + v * eta_y) + c * np.sqrt(eta_x ** 2 + eta_y ** 2)
    lam = np.maximum(a_xi + a_eta, 1e-300).max(axis=(2, 3))       # (k, j)
    dt = cfl / (mesh.Np1 ** 2 * lam)

    if nu is not None and np.any(nu > 0.0):
        nu_eff = np.maximum(nu, mu_eff)
        h = np.minimum(mesh.h_xi, mesh.h_eta)
        dt_v = cfl_visc * h ** 2 / (mesh.Np1 ** 4 * np.maximum(nu_eff, 1e-300))
        dt = np.minimum(dt, dt_v)
    elif mu_eff > 0.0:
        h = np.minimum(mesh.h_xi, mesh.h_eta)
        dt_v = cfl_visc * h ** 2 / (mesh.Np1 ** 4 * mu_eff)
        dt = np.minimum(dt, dt_v)
    return dt
