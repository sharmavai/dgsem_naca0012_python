"""Compressible Euler / Navier-Stokes equations in two dimensions.

State and non-dimensionalisation
--------------------------------
Conservative variables  U = (rho, rho u, rho v, E)^T with the perfect-gas
closure

    p = (gamma - 1) (E - rho (u^2 + v^2) / 2),   gamma = 1.4.

The solver non-dimensionalises by free-stream density and speed of sound
(rho_inf = 1, a_inf = 1, chord c = 1), so

    p_inf = 1 / gamma,   |u_inf| = M_inf,   q_inf = M_inf^2 / 2,

and the pressure coefficient is Cp = (p - p_inf) / (M_inf^2 / 2).  Note
that p_inf = 1/gamma (not 1/(gamma M_inf^2)): with a_inf = 1 the sound
speed is one by construction, since a_inf^2 = gamma p_inf / rho_inf.
"""
from __future__ import annotations

import numpy as np

__all__ = ["GAMMA", "primitives", "physical_fluxes", "freestream_state",
           "sound_speed", "pressure", "initialize_freestream"]

GAMMA = 1.4


def primitives(U: np.ndarray, gamma: float = GAMMA) -> tuple:
    """(rho, u, v, p) from the conservative state, with floors for safety."""
    rho = np.maximum(U[..., 0], 1e-12)
    u = U[..., 1] / rho
    v = U[..., 2] / rho
    p = np.maximum((gamma - 1.0) * (U[..., 3] - 0.5 * rho * (u * u + v * v)),
                   1e-12)
    return rho, u, v, p


def pressure(U: np.ndarray, gamma: float = GAMMA) -> np.ndarray:
    rho = np.maximum(U[..., 0], 1e-12)
    u = U[..., 1] / rho
    v = U[..., 2] / rho
    return (gamma - 1.0) * (U[..., 3] - 0.5 * rho * (u * u + v * v))


def sound_speed(rho: np.ndarray, p: np.ndarray, gamma: float = GAMMA) -> np.ndarray:
    return np.sqrt(gamma * p / np.maximum(rho, 1e-12))


def physical_fluxes(rho, u, v, E, p):
    """Physical flux vectors F (x) and G (y); each (..., 4)."""
    F = np.empty(rho.shape + (4,))
    G = np.empty(rho.shape + (4,))
    F[..., 0] = rho * u
    F[..., 1] = rho * u * u + p
    F[..., 2] = rho * u * v
    F[..., 3] = (E + p) * u
    G[..., 0] = rho * v
    G[..., 1] = rho * u * v
    G[..., 2] = rho * v * v + p
    G[..., 3] = (E + p) * v
    return F, G


def freestream_state(Mach: float, alpha_deg: float, gamma: float = GAMMA) -> np.ndarray:
    """Free-stream conservative state in the solver's normalisation
    (rho_inf = 1, a_inf = 1, chord = 1):
        p_inf = 1/gamma,  |u_inf| = Mach.
    The dynamic pressure is q_inf = M^2 / 2, so Cp = (p - p_inf)/(M^2/2).
    """
    a = np.radians(alpha_deg)
    rho = 1.0
    p = 1.0 / gamma
    u = Mach * np.cos(a)
    v = Mach * np.sin(a)
    E = p / (gamma - 1.0) + 0.5 * rho * (u * u + v * v)
    return np.array([rho, rho * u, rho * v, E])


def initialize_freestream(mesh, Mach: float, alpha_deg: float,
                          gamma: float = GAMMA) -> tuple[np.ndarray, np.ndarray]:
    """Uniform free-stream solution on the mesh."""
    Qinf = freestream_state(Mach, alpha_deg, gamma)
    U = np.empty((mesh.n_k, mesh.n_j, mesh.Np1, mesh.Np1, 4))
    U[..., 0] = Qinf[0]
    U[..., 1] = Qinf[1]
    U[..., 2] = Qinf[2]
    U[..., 3] = Qinf[3]
    return U, Qinf
