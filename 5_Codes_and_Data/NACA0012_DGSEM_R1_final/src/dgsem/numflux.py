"""Approximate Riemann solvers for the inviscid interface fluxes.

Both fluxes are evaluated in contravariant form: given the left/right states
and the face metric vector sJ n (whose norm sJ is the face Jacobian and whose
direction is the unit normal pointing from the left to the right state), they
return the vector  sJ * F*_n  with the correct orientation.

* Rusanov (local Lax--Friedrichs): robust, cheap.
* Roe with a Harten-type entropy fix applied to the ACOUSTIC waves only
  (the production choice).

Both satisfy F*(U, U) = F(U).n exactly (consistency, verified in the self
test).
"""
from __future__ import annotations

import numpy as np

from .euler import GAMMA

__all__ = ["rusanov_flux", "roe_flux", "make_flux"]


def _normal(sJnx, sJny):
    sJ = np.maximum(np.hypot(sJnx, sJny), 1e-300)
    return sJ, sJnx / sJ, sJny / sJ


def rusanov_flux(QL, QR, sJnx, sJny, gamma=GAMMA):
    """Contravariant Rusanov flux  sJ ({F.n} - 0.5 lam [U])."""
    sJ, nx, ny = _normal(sJnx, sJny)
    rL = np.maximum(QL[..., 0], 1e-12)
    rR = np.maximum(QR[..., 0], 1e-12)
    uL, vL = QL[..., 1] / rL, QL[..., 2] / rL
    uR, vR = QR[..., 1] / rR, QR[..., 2] / rR
    pL = np.maximum((gamma - 1.0) * (QL[..., 3] - 0.5 * rL * (uL * uL + vL * vL)), 1e-12)
    pR = np.maximum((gamma - 1.0) * (QR[..., 3] - 0.5 * rR * (uR * uR + vR * vR)), 1e-12)
    cL = np.sqrt(gamma * pL / rL)
    cR = np.sqrt(gamma * pR / rR)
    VnL = uL * nx + vL * ny
    VnR = uR * nx + vR * ny
    lam = np.maximum(np.abs(VnL) + cL, np.abs(VnR) + cR)

    FL = np.empty_like(QL)
    FR = np.empty_like(QR)
    FL[..., 0] = rL * VnL
    FL[..., 1] = rL * uL * VnL + pL * nx
    FL[..., 2] = rL * vL * VnL + pL * ny
    FL[..., 3] = (QL[..., 3] + pL) * VnL
    FR[..., 0] = rR * VnR
    FR[..., 1] = rR * uR * VnR + pR * nx
    FR[..., 2] = rR * vR * VnR + pR * ny
    FR[..., 3] = (QR[..., 3] + pR) * VnR
    F = 0.5 * (FL + FR) - 0.5 * lam[..., None] * (QR - QL)
    return F * sJ[..., None], lam


def roe_flux(QL, QR, sJnx, sJny, gamma=GAMMA, entropy_fix=0.05,
             fix_all_waves: bool = False):
    """Contravariant Roe flux with a Harten-type entropy fix.

    The fix replaces a wave speed lambda by
        |lambda|^ = (lambda^2 + delta^2) / (2 delta)   when |lambda| < delta,
    with delta = entropy_fix * (|Vn_roe| + c_roe).  entropy_fix = 0 disables
    the fix entirely.

    By default (fix_all_waves=False) the fix is applied ONLY to the two
    acoustic (nonlinear) waves, which are the only ones that can generate
    entropy-violating expansion shocks.  The entropy and shear waves keep
    their exact speeds |Vn|: fixing them would add spurious dissipation of
    O(delta) wherever |Vn| is small, i.e. precisely in the stagnation-point
    and wake regions, where the physical waves carry no dissipation at all.
    The sensitivity of the results to this choice (and to the value of
    entropy_fix) is quantified in the paper.
    """
    sJ, nx, ny = _normal(sJnx, sJny)
    g1 = gamma - 1.0

    rL = np.maximum(QL[..., 0], 1e-12)
    rR = np.maximum(QR[..., 0], 1e-12)
    uL, vL = QL[..., 1] / rL, QL[..., 2] / rL
    uR, vR = QR[..., 1] / rR, QR[..., 2] / rR
    pL = np.maximum(g1 * (QL[..., 3] - 0.5 * rL * (uL * uL + vL * vL)), 1e-12)
    pR = np.maximum(g1 * (QR[..., 3] - 0.5 * rR * (uR * uR + vR * vR)), 1e-12)
    HL = (QL[..., 3] + pL) / rL
    HR = (QR[..., 3] + pR) / rR

    sL = np.sqrt(rL)
    sR = np.sqrt(rR)
    inv = 1.0 / (sL + sR)
    rt = sL * sR
    r_t = rt
    u_t = (sL * uL + sR * uR) * inv
    v_t = (sL * vL + sR * vR) * inv
    H_t = (sL * HL + sR * HR) * inv
    q2 = u_t * u_t + v_t * v_t
    c2 = np.maximum(g1 * (H_t - 0.5 * q2), 1e-12)
    c_t = np.sqrt(c2)
    Vn = u_t * nx + v_t * ny

    # wave strengths from the jumps (Roe decomposition)
    dU = QR - QL
    dr = dU[..., 0]
    dru = dU[..., 1]
    drv = dU[..., 2]
    dE = dU[..., 3]
    dp = g1 * (dE - u_t * dru - v_t * drv + 0.5 * q2 * dr)
    dVn = ((dru - u_t * dr) * nx + (drv - v_t * dr) * ny) / np.maximum(r_t, 1e-12)
    dUt = (-(dru - u_t * dr) * ny + (drv - v_t * dr) * nx) / np.maximum(r_t, 1e-12)

    a1 = dr - dp / c2                                  # entropy wave
    a2 = (dp - r_t * c_t * dVn) / (2.0 * c2)           # left acoustic
    a3 = r_t * dUt                                     # shear wave
    a4 = (dp + r_t * c_t * dVn) / (2.0 * c2)           # right acoustic

    # entropy fix (acoustic waves by default; see docstring)
    if entropy_fix > 0.0:
        delta = entropy_fix * (np.abs(Vn) + c_t)

        def _fix(l):
            al = np.abs(l)
            return np.where(al < delta, (l * l + delta * delta)
                            / (2.0 * np.maximum(delta, 1e-300)), al)

        lam_e = np.abs(Vn)               # entropy + shear wave speed
        lam_m = _fix(Vn - c_t)           # left acoustic wave
        lam_p = _fix(Vn + c_t)           # right acoustic wave
        if fix_all_waves:
            lam_e = _fix(Vn)
    else:
        lam_e = np.abs(Vn)
        lam_m = np.abs(Vn - c_t)
        lam_p = np.abs(Vn + c_t)

    # dissipative part:  sum_k alpha_k |lambda_k| r_k with the correct
    # wave/eigenvector pairing:
    #   entropy : r = (1, u, v, q2/2),            lambda = Vn
    #   shear   : r = (0, -ny, nx, u_t tang.),    lambda = Vn
    #   acoust-L: r = (1, u-c nx, v-c ny, H-Vn c), lambda = Vn - c
    #   acoust-R: r = (1, u+c nx, v+c ny, H+Vn c), lambda = Vn + c
    Ut_t = -u_t * ny + v_t * nx
    d1 = a1 * lam_e
    d2 = a2 * lam_m
    d3 = a3 * lam_e
    d4 = a4 * lam_p

    F = np.empty_like(QL)
    FL = np.empty_like(QL)
    FR = np.empty_like(QR)
    VnL = uL * nx + vL * ny
    VnR = uR * nx + vR * ny
    FL[..., 0] = rL * VnL
    FL[..., 1] = rL * uL * VnL + pL * nx
    FL[..., 2] = rL * vL * VnL + pL * ny
    FL[..., 3] = (QL[..., 3] + pL) * VnL
    FR[..., 0] = rR * VnR
    FR[..., 1] = rR * uR * VnR + pR * nx
    FR[..., 2] = rR * vR * VnR + pR * ny
    FR[..., 3] = (QR[..., 3] + pR) * VnR
    F = 0.5 * (FL + FR)

    F[..., 0] -= 0.5 * (d1 + d2 + d4)
    F[..., 1] -= 0.5 * (d1 * u_t
                        + d2 * (u_t - c_t * nx)
                        + d3 * (-ny)
                        + d4 * (u_t + c_t * nx))
    F[..., 2] -= 0.5 * (d1 * v_t
                        + d2 * (v_t - c_t * ny)
                        + d3 * (nx)
                        + d4 * (v_t + c_t * ny))
    F[..., 3] -= 0.5 * (d1 * 0.5 * q2
                        + d2 * (H_t - Vn * c_t)
                        + d3 * Ut_t
                        + d4 * (H_t + Vn * c_t))
    return F * sJ[..., None], np.maximum(np.abs(Vn) + c_t, np.abs(uL * nx + vL * ny) + np.sqrt(gamma * pL / rL))


def make_flux(kind: str = "roe", entropy_fix: float = 0.05,
              fix_all_waves: bool = False):
    """Return a flux function f(QL, QR, sJnx, sJny) -> (Fn*sJ, lam)."""
    if kind == "roe":
        return lambda QL, QR, sJnx, sJny: roe_flux(
            QL, QR, sJnx, sJny, entropy_fix=entropy_fix,
            fix_all_waves=fix_all_waves)
    if kind == "rusanov":
        return rusanov_flux
    raise ValueError(f"unknown flux kind: {kind}")


# ----------------------------------------------------------------------
def _self_test() -> None:  # pragma: no cover
    rng = np.random.default_rng(42)
    print("numerical flux self test")
    shape = (7, 4)
    QL = np.stack([1 + 0.3 * rng.random(shape),
                   rng.normal(0.4, 0.1, shape),
                   rng.normal(0.0, 0.1, shape),
                   2.5 + 0.4 * rng.random(shape)], axis=-1)
    QR = QL + np.stack([0.2 * rng.random(shape), 0.2 * rng.random(shape),
                        0.2 * rng.random(shape), 0.3 * rng.random(shape)], axis=-1)
    ang = rng.uniform(0, 2 * np.pi, shape)
    sJ = 1.0 + rng.random(shape)
    sJnx, sJny = sJ * np.cos(ang), sJ * np.sin(ang)

    from .euler import physical_fluxes, primitives
    for name, flux in (("rusanov", rusanov_flux),
                       ("roe", lambda a, b, c, d: roe_flux(a, b, c, d)[0])):
        # consistency: F*(Q, Q) = F(Q).n * sJ
        Fn, _ = (rusanov_flux(QL, QL, sJnx, sJny) if name == "rusanov"
                 else (roe_flux(QL, QL, sJnx, sJny)[0], None))
        r, u, v, p = primitives(QL)
        E = QL[..., 3]
        F, G = physical_fluxes(r, u, v, E, p)
        _, nx, ny = _normal(sJnx, sJny)
        exact = (F * nx[..., None] + G * ny[..., None]) * sJ[..., None]
        err = np.abs(Fn - exact).max()
        print(f"  {name}: consistency err {err:.2e}")
        assert err < 1e-13

    # moving density contact (same u, p): Roe resolves it exactly,
    # Rusanov smears it.  Exact Godunov mass flux = F(QL) since Vn > 0.
    p0, u0 = 1.05, 0.5
    QLs = np.array([[1.0, 1.0 * u0, 0.0, p0 / 0.4 + 0.5 * u0 ** 2]])
    QRs = np.array([[0.5, 0.5 * u0, 0.0, p0 / 0.4 + 0.5 * 0.5 * u0 ** 2]])
    nx = np.array([1.0]); ny = np.array([0.0])
    Froe, _ = roe_flux(QLs, QRs, nx, ny)
    Frus, _ = rusanov_flux(QLs, QRs, nx, ny)
    exact_mass = 1.0 * u0
    print(f"  moving contact: |dF_roe(mass)| = {abs(Froe[0,0] - exact_mass):.2e}, "
          f"|dF_rusanov(mass)| = {abs(Frus[0,0] - exact_mass):.2e}")
    assert abs(Froe[0, 0] - exact_mass) < 1e-13
    assert abs(Frus[0, 0] - exact_mass) > 1e-2

    # entropy fix active region: flux finite and monotone-ish
    QLe = np.array([[1.0, 0.9, 0.0, 2.4]])     # sonic-ish
    QRe = np.array([[0.9, 0.4, 0.0, 2.2]])
    Fe, _ = roe_flux(QLe, QRe, nx, ny)
    assert np.all(np.isfinite(Fe))
    print("  entropy fix: finite flux OK")
    print("  all checks passed")


if __name__ == "__main__":
    _self_test()
