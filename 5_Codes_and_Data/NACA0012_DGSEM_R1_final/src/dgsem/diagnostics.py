"""Solution diagnostics: entropy error and field probes.

Entropy error
-------------
For the steady inviscid subsonic flows of this paper the entropy
s = ln(p / rho^gamma) must equal the free-stream value s_inf everywhere
except inside the numerical wake and the immediate trailing-edge region,
where the sharp-edge Kutta condition is enforced through the interface
dissipation of the scheme.  A domain-integrated entropy error

    E_s = [ int_Omega rho (s - s_inf)^2 dA ]^(1/2)

is therefore a direct measure of the spurious numerical dissipation of the
discretisation.  It is reported both over the whole domain and over the
near field |r - (0.25, 0)| < 5 chords (which contains the airfoil and the
first part of the wake, but not the stretched outer elements), and its
decay under mesh and polynomial refinement is tabulated in the paper.

All quantities use the solver normalisation (rho_inf = 1, a_inf = 1,
chord = 1, p_inf = 1/gamma), so s_inf = ln(p_inf) = -ln(gamma).
"""
from __future__ import annotations

import numpy as np

from .euler import GAMMA, primitives

__all__ = ["entropy_field", "entropy_error", "probe_fields"]


def entropy_field(U, gamma: float = GAMMA) -> tuple[np.ndarray, np.ndarray]:
    """Return (s - s_inf, s) with s = ln(p / rho^gamma)."""
    rho, u, v, p = primitives(U, gamma)
    s = np.log(p) - gamma * np.log(rho)
    s_inf = -np.log(gamma)          # rho_inf = 1, p_inf = 1/gamma
    return s - s_inf, s


def entropy_error(U, mesh, gamma: float = GAMMA,
                  near_field_r: float | None = None) -> dict:
    """L2 entropy error over the domain (and optionally the near field).

    near_field_r: if given, additionally integrate over the disc
    |(x, y) - (0.25, 0)| < near_field_r only."""
    ds, _ = entropy_field(U, gamma)
    wgt = np.outer(mesh.w, mesh.w)[None, None, :, :]
    integrand = mesh.J * primitives(U, gamma)[0] * ds * ds
    tot = float(np.einsum('kjil,kjil->', wgt, integrand))
    out = {"entropy_l2_domain": tot ** 0.5}
    if near_field_r is not None:
        r = np.hypot(mesh.X - 0.25, mesh.Y)
        mask = r < near_field_r
        tot_nf = float(np.einsum('kjil,kjil->', wgt * mask, integrand))
        out["entropy_l2_nearfield"] = tot_nf ** 0.5
    return out


def probe_fields(U, mesh, Mach: float, gamma: float = GAMMA) -> dict:
    """Mach number, Cp and entropy fields at the nodes (for plotting)."""
    rho, u, v, p = primitives(U, gamma)
    c = np.sqrt(gamma * p / rho)
    p_inf = 1.0 / gamma
    q_inf = 0.5 * Mach * Mach
    ds, _ = entropy_field(U, gamma)
    return {
        "X": mesh.X, "Y": mesh.Y,
        "mach": np.hypot(u, v) / c,
        "cp": (p - p_inf) / q_inf,
        "ds": ds,
        "p": p, "rho": rho, "u": u, "v": v,
    }
