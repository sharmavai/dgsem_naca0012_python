"""Positivity-preserving scaling limiter (Zhang & Shu type).

For each element whose nodal density or internal energy falls below the
floors, the nodal solution is scaled toward its (positive) element mean
until the floors hold.  The element mean is preserved exactly, so the limiter
is conservative and is applied only where needed.

Reference: X. Zhang and C.-W. Shu, "On maximum-principle-satisfying high
order schemes for scalar conservation laws", J. Comput. Phys. 229 (2010);
"On positivity-preserving high order discontinuous Galerkin schemes for
compressible Euler equations", J. Comput. Phys. 231 (2012).
"""
from __future__ import annotations

import numpy as np

__all__ = ["positivity_limiter"]


def positivity_limiter(U, mesh, gamma: float,
                       rho_min: float = 1e-4,
                       p_min: float = 1e-4) -> tuple[np.ndarray, int]:
    """Return (limited U, number of touched elements)."""
    w = mesh.w
    wgt = np.outer(w, w)[None, None, :, :]                    # (1,1,Np1,Np1)
    wJ = wgt * mesh.J
    wJ5 = wJ[..., None]
    wJ_tot = wJ.sum(axis=(2, 3), keepdims=True)
    U_mean = (wJ5 * U).sum(axis=(2, 3), keepdims=True) / wJ_tot[..., None]

    rho = U[..., 0]
    u = U[..., 1] / np.maximum(rho, 1e-12)
    v = U[..., 2] / np.maximum(rho, 1e-12)
    e_int = U[..., 3] - 0.5 * rho * (u * u + v * v)
    bad = (rho < rho_min) | (e_int < p_min / (gamma - 1.0))
    if not np.any(bad):
        return U, 0

    bad_elem = np.any(bad, axis=(2, 3))                       # (k, j)
    dev = U - U_mean
    lo = np.zeros_like(U_mean)
    hi = np.ones_like(U_mean)
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        Ut = U_mean + mid * dev
        r = Ut[..., 0]
        uu = Ut[..., 1] / np.maximum(r, 1e-12)
        vv = Ut[..., 2] / np.maximum(r, 1e-12)
        ei = Ut[..., 3] - 0.5 * r * (uu * uu + vv * vv)
        ok = np.all((r >= rho_min) & (ei >= p_min / (gamma - 1.0)),
                    axis=(2, 3))
        ok5 = ok[..., None, None, None]
        hi = np.where(ok5, mid, hi)
        lo = np.where(ok5, lo, mid)
    theta = 0.5 * (lo + hi)
    theta = np.where(bad_elem[..., None, None, None], theta, 1.0)
    return U_mean + theta * dev, int(bad_elem.sum())
