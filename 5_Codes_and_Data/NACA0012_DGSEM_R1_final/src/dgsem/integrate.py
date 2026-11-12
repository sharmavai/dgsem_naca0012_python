"""Explicit time integration with local (per-element) time steps.

SSP-RK3 (Shu & Osher) and classical RK4 are provided.  With local time
stepping the scheme is time-inconsistent by construction and is used
exclusively to march to steady states, where the converged solution
satisfies R(U) = 0 independent of the time step.  The monitoring residual is
always the true (time-step-independent) residual R(U).

Both steppers allocate independent stage buffers: the spatial operator
returns a freshly allocated array from every rhs() call, so no stage result
can be silently overwritten by the next one.
"""
from __future__ import annotations

import numpy as np

__all__ = ["rk3_step", "rk4_step", "residual_norm"]


def _bc(dt: np.ndarray, U: np.ndarray) -> np.ndarray:
    """Broadcast the (k, j) time step over the node/variable axes."""
    return dt[..., None, None, None]


def rk3_step(U, op, dt, nu=None) -> np.ndarray:
    """One SSP-RK3 step with element-local dt (in-place on U).
    Returns the first-stage residual for monitoring."""
    d = _bc(dt, U)
    k1 = op.rhs(U, nu)
    U1 = U + d * k1
    k2 = op.rhs(U1, nu)
    U2 = 0.75 * U + 0.25 * (U1 + d * k2)
    k3 = op.rhs(U2, nu)
    U[...] = (1.0 / 3.0) * U + (2.0 / 3.0) * (U2 + d * k3)
    return k1


def rk4_step(U, op, dt, nu=None) -> np.ndarray:
    """One classical RK4 step with element-local dt (in-place on U).
    Returns the first-stage residual for monitoring."""
    d = _bc(dt, U)
    k1 = op.rhs(U, nu)
    Utmp = U + 0.5 * d * k1
    k2 = op.rhs(Utmp, nu)
    Utmp = U + 0.5 * d * k2
    k3 = op.rhs(Utmp, nu)
    Utmp = U + d * k3
    k4 = op.rhs(Utmp, nu)
    U[...] = U + (d / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return k1


def residual_norm(k1, mesh) -> float:
    """Mass-weighted RMS of the true residual (density component):
    R_rho = sqrt( int (d rho/dt)^2 dA / int dA )."""
    wJ = np.outer(mesh.w, mesh.w)[None, None, :, :] * mesh.J
    r2 = (k1[..., 0] ** 2 * wJ).sum() / wJ.sum()
    return float(np.sqrt(r2))
