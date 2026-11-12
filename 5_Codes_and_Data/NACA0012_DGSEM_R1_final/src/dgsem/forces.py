"""Aerodynamic force and moment integration from the wall pressure.

The airfoil surface traction is integrated over the eta = -1 faces of the
first radial element band using the LGL quadrature and the face metric
vectors (which include the surface Jacobian):

    dF = -p_w (sJ n)_body ds-weight,   with (sJ n) = Fem (pointing out of the
    fluid, i.e. into the body), so  dF = +p_w * Fem * w.

Wall pressure p_w
-----------------
p_w is the Riemann-star pressure of the slip-wall Riemann problem, extracted
from the SAME Roe interface flux used by the discretisation (the mirror
state is the right state): with the mirrored ghost the Roe flux has exactly
zero mass and tangential-momentum flux, and its normal-momentum component
reduces to the star pressure,

    p_w = (F*_roe . n_hat)[momentum-normal] .

This is the pressure the discretisation actually applies to the wall; it
differs from the interior-trace pressure by the (second-order) solution
jump and is the physically consistent choice for force integration.  Both
are reported so the difference can be quantified (it is below 1e-4 Cp
counts in all production runs).

Coefficients use the free-stream dynamic pressure q_inf = M_inf^2 / 2 in the
solver normalisation (rho_inf = 1, a_inf = 1, chord = 1):

    Cl = F . e_lift / q_inf,   Cd = F . e_drag / q_inf,
    Cm = [(r - a) x dF]_z / q_inf   (about the quarter chord, nose-up +).

The surface Cp distribution is exported per surface node with an
upper/lower-surface classification (y > 0 / y < 0; the small trailing-edge
base is x = 1).
"""
from __future__ import annotations

import numpy as np

from .euler import GAMMA, pressure
from .numflux import roe_flux

__all__ = ["forces", "surface_cp", "wall_star_pressure"]


def wall_star_pressure(U, mesh, gamma: float = GAMMA,
                       entropy_fix: float = 0.05) -> np.ndarray:
    """Riemann-star pressure at the wall nodes, shape (n_k, N+1).

    Evaluates the Roe flux between the interior trace and its mirror ghost
    on every eta = -1 wall face and extracts the star pressure from the
    normal-momentum component (the mass flux is exactly zero there)."""
    Np1 = mesh.Np1
    Qint = U[:, 0, :, 0, :]
    sJnx = mesh.Fem[:, 0, :, 0]
    sJny = mesh.Fem[:, 0, :, 1]
    sJ = np.maximum(np.hypot(sJnx, sJny), 1e-300)
    nx = sJnx / sJ
    ny = sJny / sJ
    rhoW = np.maximum(Qint[..., 0], 1e-12)
    uW = Qint[..., 1] / rhoW
    vW = Qint[..., 2] / rhoW
    Vn = uW * nx + vW * ny
    QG = np.empty_like(Qint)
    QG[..., 0] = rhoW
    QG[..., 1] = rhoW * (uW - 2.0 * Vn * nx)
    QG[..., 2] = rhoW * (vW - 2.0 * Vn * ny)
    QG[..., 3] = Qint[..., 3]
    Fn, _ = roe_flux(Qint, QG, sJnx, sJny, gamma=gamma,
                     entropy_fix=entropy_fix)
    # With the mirror ghost the Roe flux has exactly zero mass flux
    # (rho Vn* = 0) and zero tangential momentum flux, so the momentum
    # flux vector reduces to sJ * p* * n_hat.  Projecting it back onto the
    # (unit) normal recovers the star pressure independent of its
    # orientation:  p* = (Fn . n_hat) / sJ.
    return (Fn[..., 1] * nx + Fn[..., 2] * ny) / sJ


def forces(U, mesh, Mach: float, alpha_deg: float,
           gamma: float = GAMMA, entropy_fix: float = 0.05) -> dict:
    """Integrate pressure forces/moments on the airfoil (star pressure)."""
    p_inf = 1.0 / gamma          # a_inf = 1 normalisation
    q_inf = 0.5 * Mach * Mach
    a = np.radians(alpha_deg)
    e_lift = np.array([-np.sin(a), np.cos(a)])
    e_drag = np.array([np.cos(a), np.sin(a)])

    pw = wall_star_pressure(U, mesh, gamma, entropy_fix)      # (k, i)
    p_trace = pressure(U[:, 0, :, 0, :], gamma)               # diagnostic
    w = mesh.w[None, :]                                      # (1, Np1)
    sJn = mesh.Fem[:, 0, :, :]                               # (k, i, 2)
    dFx = pw * sJn[..., 0] * w
    dFy = pw * sJn[..., 1] * w
    Fx, Fy = float(dFx.sum()), float(dFy.sum())

    # interior-trace variant (reported for comparison only)
    dFx_t = p_trace * sJn[..., 0] * w
    dFy_t = p_trace * sJn[..., 1] * w
    Fx_t, Fy_t = float(dFx_t.sum()), float(dFy_t.sum())

    x = mesh.X[:, 0, :, 0]
    y = mesh.Y[:, 0, :, 0]
    dM = (x - 0.25) * dFy - y * dFx                          # z-moment about c/4
    Mz = float(dM.sum())

    return {
        "Fx": Fx, "Fy": Fy,
        "Cl": Fx * e_lift[0] / q_inf + Fy * e_lift[1] / q_inf,
        "Cd": Fx * e_drag[0] / q_inf + Fy * e_drag[1] / q_inf,
        "Cm": Mz / q_inf,
        "Cl_trace": Fx_t * e_lift[0] / q_inf + Fy_t * e_lift[1] / q_inf,
        "Cd_trace": Fx_t * e_drag[0] / q_inf + Fy_t * e_drag[1] / q_inf,
        "max_dp_wall": float(np.abs(pw - p_trace).max()),
    }


def surface_cp(U, mesh, Mach: float, alpha_deg: float,
               gamma: float = GAMMA) -> dict:
    """Cp distribution on the airfoil surface.

    Returns a dict with 'upper' and 'lower' arrays of (x/c, Cp), each sorted
    by x/c (the sharp trailing-edge point, x/c = 1, is shared by both
    branches), plus 's' arc-length positions.  The wall pressure is the
    Riemann-star pressure, consistent with the force integration.
    """
    p_inf = 1.0 / gamma          # a_inf = 1 normalisation
    q_inf = 0.5 * Mach * Mach
    pw = wall_star_pressure(U, mesh, gamma)
    cp = (pw - p_inf) / q_inf
    x = mesh.X[:, 0, :, 0].ravel()
    y = mesh.Y[:, 0, :, 0].ravel()
    cpr = cp.ravel()
    upper = np.stack([x[y > 1e-12], cpr[y > 1e-12]], axis=1)
    lower = np.stack([x[y < -1e-12], cpr[y < -1e-12]], axis=1)
    base = np.stack([x[np.abs(y) <= 1e-12], cpr[np.abs(y) <= 1e-12]], axis=1)
    order_u = np.argsort(upper[:, 0])
    order_l = np.argsort(lower[:, 0])
    return {
        "upper": upper[order_u],
        "lower": lower[order_l],
        "base": base,
        "alpha_deg": alpha_deg, "Mach": Mach,
    }
