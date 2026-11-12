"""Steady-state case driver: mesh, operator, time marching, monitoring,
checkpointing.

Design decisions fixed after the external review of the previous study
---------------------------------------------------------------------
* **Impulsive start.**  Every run starts from the uniform free stream.  No
  potential-flow (panel) initialisation and no continuation from a
  neighbouring angle of attack is used anywhere, so the circulation, and
  with it the lift, must develop from the trailing-edge Kutta condition
  through the discretisation itself.
* **No sponge, no wake dissipation.**  The wake of the O-mesh leaves the
  domain through the periodic seam and the far-field Riemann boundary;
  no artificial damping of any kind is applied in the interior or in the
  wake.
* **Strict convergence.**  A case counts as converged only when the RMS
  density residual has dropped by at least `res_drop_orders` decades from
  its transient maximum AND the drift of every force coefficient is below
  `force_tol` per 1000 steps.  The wall-clock budget is a safety net, not
  the convergence criterion.
* **Far-field circulation.**  The Thomas-Salas point-vortex correction is
  updated from the CURRENT lift coefficient every `vortex_update_every`
  steps (relaxed), never seeded from the panel solution.

The far-field vortex circulation is Gamma = -Cl * Mach / 2 in the solver
normalisation (a_inf = 1, chord = 1): clockwise for positive lift.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

from .diagnostics import entropy_error
from .euler import GAMMA, freestream_state, initialize_freestream, pressure
from .forces import forces
from .integrate import residual_norm, rk3_step, rk4_step
from .limiters import positivity_limiter
from .mesh import Mesh, generate_airfoil_omesh
from .operator import SolverConfig
from .operator_numba import NumbaOperator
from .timestep import local_timestep

__all__ = ["Case", "run_case", "MESH_FAMILY"]


# geometrically similar O-mesh family (refinement ratio ~1.4)
MESH_FAMILY = {
    "L1": dict(n_half=20, n_r=15),
    "L2": dict(n_half=28, n_r=21),
    "L3": dict(n_half=40, n_r=30),
}


# ----------------------------------------------------------------------
@dataclass
class Case:
    Mach: float = 0.30
    alpha_deg: float = 5.0

    # mesh (O-type, exact NACA 0012 contour, sharp trailing edge)
    n_half: int = 28
    n_r: int = 21
    N: int = 3
    R_far: float = 50.0
    h_wall: float | None = None      # default 0.30 / n_r
    wake_align: bool = True          # align the mesh seam with the expected
                                     # wake sheet (a = Cl_est / 4 pi)

    # numerics
    flux: str = "roe"                # 'roe' | 'rusanov'
    entropy_fix: float = 0.02
    fix_all_waves: bool = True       # small all-wave Harten fix (wake sheet)
    scheme: str = "rk4"              # 'rk3' | 'rk4'
    cfl: float = 1.5
    cfl_ramp_steps: int = 15000      # the impulsive-start transient is run
                                     # at a reduced CFL (0.15 -> cfl)

    # far field
    farfield_vortex: bool = True
    vortex_update_every: int = 400

    # outer relaxation sponge: the solution beyond sponge_r0 (in chords,
    # measured from the vortex centre) is relaxed toward the vortex-corrected
    # free stream at a rate ramping up to sponge_rate per pseudo-step.  The
    # region r < sponge_r0 (which contains the entire airfoil near field)
    # is untouched.  Set sponge_rate = 0 to disable.
    sponge_r0: float = 25.0
    sponge_rate: float = 0.0         # disabled by default; the sensitivity
                                     # study compares 0 and 0.01

    # convergence control
    max_steps: int = 150000
    res_drop_orders: float = 6.0
    force_tol: float = 2.0e-5        # max |dCl|,|dCd|,|dCm| per 1000 steps
    monitor_every: int = 100
    max_wall_s: float = 3600.0
    apply_limiter_every: int = 1   # identity unless floors are violated
    ckpt_every: int = 5000          # periodic checkpoint so an interrupted
                                    # window loses at most this many steps

    # output
    tag: str = ""
    out_dir: str = "results"
    save_field: bool = True
    resume: bool = True

    def label(self) -> str:
        a = (f"{self.alpha_deg:+g}").replace("+", "p").replace("-", "m")
        return self.tag or f"M{self.Mach:g}_a{a}"


# ----------------------------------------------------------------------
def _smooth_dt(dt: np.ndarray, passes: int = 2) -> np.ndarray:
    """Element-wise minimum smoothing of the local time step: each element
    is limited by the average of itself and its neighbours, which damps the
    local-stepping interface inconsistency.  The operation only ever
    DECREASES time steps, so it cannot destabilise the scheme.

    The xi smoothing uses the SYMMETRIC three-point average
    (d[k-1] + 2 d[k] + d[k+1]) / 4 in a single min(): applying the two
    neighbour directions sequentially (min over +1, then min over -1)
    would NOT commute with the mirror reflection k -> kmap, because the
    mirror reverses the neighbour order, and the sequential mins are
    order-sensitive.  The symmetric form is exactly mirror-covariant
    (verified by the antisymmetry test to ~1e-13 per step)."""
    d = dt.copy()
    for _ in range(passes):
        d = np.minimum(d, 0.25 * (np.roll(d, 1, axis=0) + 2.0 * d
                                  + np.roll(d, -1, axis=0)))
        dm = d.copy()
        dm[:, :-1] = np.minimum(d[:, :-1], 0.5 * (d[:, :-1] + d[:, 1:]))
        dm[:, 1:] = np.minimum(d[:, 1:], 0.5 * (d[:, 1:] + d[:, :-1]))
        d = dm
    return d


# ----------------------------------------------------------------------
def _apply_sponge(U, mesh, sponge, Qinf, op, gamma: float) -> None:
    """Relax the outer region toward the vortex-corrected free stream.

    The target state is the free stream plus the velocity induced by the
    current far-field circulation Gamma (consistent with the far-field
    boundary condition), so the sponge does not fight the circulation.  In
    the free stream (U = target) the update is exactly zero, so free-stream
    preservation is unaffected."""
    from .euler import freestream_state  # noqa: F401  (docstring only)
    cx, cy = op.cfg.vortex_center
    x, y = mesh.X, mesh.Y
    uinf = Qinf[1] / Qinf[0]
    vinf = Qinf[2] / Qinf[0]
    if op.cfg.farfield_vortex and op.Gamma != 0.0:
        r2 = np.maximum((x - cx) ** 2 + (y - cy) ** 2, 1e-12)
        fac = op.Gamma / (2.0 * np.pi * r2)
        uB = uinf + fac * (-(y - cy))
        vB = vinf + fac * (x - cx)
    else:
        uB = np.full_like(x, uinf)
        vB = np.full_like(x, vinf)
    Utar = np.empty_like(U)
    Utar[..., 0] = Qinf[0]
    Utar[..., 1] = Qinf[0] * uB
    Utar[..., 2] = Qinf[0] * vB
    Utar[..., 3] = Qinf[3] + 0.5 * Qinf[0] * (
        (uB * uB + vB * vB) - (uinf * uinf + vinf * vinf))
    sig = sponge[:, :, None, None, None]
    U -= sig * (U - Utar)


# ----------------------------------------------------------------------
def run_case(case: Case, verbose: bool = True) -> dict:
    """Run one case from an impulsive start to a converged steady state."""
    t_start = time.time()
    if case.h_wall is None:
        case.h_wall = 0.30 / case.n_r
    wake_curve = 0.0
    if case.wake_align:
        from .panel import PanelSolver, _airfoil_vertices
        _a = np.radians(case.alpha_deg)
        _ps = PanelSolver(*_airfoil_vertices(200))
        _ps.solve(np.cos(_a), np.sin(_a))
        cl_panel = 4.0 * np.pi * _ps.gamma * _ps.len.sum()
        cl_est = cl_panel / np.sqrt(max(1.0 - case.Mach ** 2, 1e-6))
        wake_curve = cl_est / (4.0 * np.pi)
    mesh = generate_airfoil_omesh(case.n_half, case.n_r, case.N,
                                  case.R_far, case.h_wall,
                                  wake_curve=wake_curve)
    cfg = SolverConfig(gamma=GAMMA, flux=case.flux,
                       entropy_fix=case.entropy_fix,
                       fix_all_waves=case.fix_all_waves,
                       farfield_vortex=case.farfield_vortex)
    op = NumbaOperator(mesh, cfg)
    Qinf = freestream_state(case.Mach, case.alpha_deg, GAMMA)
    op.set_freestream(Qinf)

    out = Path(case.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ckpt = out / f"ckpt_{case.label()}.npz"

    start_step, res_max = 0, 0.0
    hist_prior = {k: [] for k in ("step", "res", "Cl", "Cd", "Cm",
                                  "Gamma", "min_p", "time")}
    hist_file = out / f"history_{case.label()}.csv"
    if (case.resume and ckpt.exists()
            and not (out / f"diverged_{case.label()}").exists()):
        d = np.load(ckpt)
        U = d["U"]
        if not np.isfinite(U).all():
            U, _ = initialize_freestream(mesh, case.Mach, case.alpha_deg,
                                         GAMMA)
        else:
            start_step = int(d["step"])
            res_max = float(d["res_max"])
            op.Gamma = float(d["Gamma"])
            if verbose:
                print(f"  [{case.label()}] resuming from step {start_step}",
                      flush=True)
            # keep the full history across windows (rows from every
            # previous window are prepended when the CSV is rewritten)
            import csv as _csv
            if hist_file.exists():
                with open(hist_file) as fh:
                    rd = _csv.DictReader(fh)
                    for row in rd:
                        hist_prior["step"].append(int(row["step"]))
                        hist_prior["res"].append(float(row["residual"]))
                        hist_prior["Cl"].append(float(row["Cl"]))
                        hist_prior["Cd"].append(float(row["Cd"]))
                        hist_prior["Cm"].append(float(row["Cm"]))
                        hist_prior["Gamma"].append(float(row["Gamma"]))
                        hist_prior["min_p"].append(float(row["min_p"]))
                        hist_prior["time"].append(float(row["wall_s"]))
        if start_step == 0:
            U, _ = initialize_freestream(mesh, case.Mach, case.alpha_deg,
                                         GAMMA)
    else:
        U, _ = initialize_freestream(mesh, case.Mach, case.alpha_deg, GAMMA)

    step_fn = rk3_step if case.scheme == "rk3" else rk4_step

    # outer relaxation sponge weights (None disables the sponge entirely)
    if case.sponge_rate > 0.0:
        cx, cy = cfg.vortex_center
        r = np.hypot(mesh.X[:, :, :, :] - cx, mesh.Y[:, :, :, :] - cy)
        ramp = np.clip((r - case.sponge_r0)
                       / max(case.R_far - case.sponge_r0, 1e-12), 0.0, 1.0)
        sig = case.sponge_rate * ramp * ramp
        # element-wise max over the nodes (conservative choice)
        sponge = sig.max(axis=(2, 3))
    else:
        sponge = None

    hist = {"step": [], "res": [], "Cl": [], "Cd": [], "Cm": [],
            "Gamma": [], "min_p": [], "time": []}
    reason = "max_steps"
    res = np.nan
    fo = forces(U, mesh, case.Mach, case.alpha_deg, GAMMA)

    n = start_step
    n_limited = 0
    while n < start_step + case.max_steps:
        if time.time() - t_start > case.max_wall_s:
            reason = "wall_budget"
            break
        cfl_n = case.cfl * min(1.0, 0.10 + 0.90 * (n - start_step)
                               / max(case.cfl_ramp_steps, 1))
        dt = _smooth_dt(local_timestep(U, mesh, GAMMA, cfl_n))
        k1 = step_fn(U, op, dt)
        if n % case.apply_limiter_every == 0:
            U, nlim = positivity_limiter(U, mesh, GAMMA)
            n_limited += nlim
        if sponge is not None:
            _apply_sponge(U, mesh, sponge, Qinf, op, GAMMA)
        n += 1

        if n % case.monitor_every == 0:
            res = residual_norm(k1, mesh)
            res_max = max(res_max, res)
            fo = forces(U, mesh, case.Mach, case.alpha_deg, GAMMA)
            if (case.farfield_vortex
                    and n % case.vortex_update_every == 0):
                gam_target = -fo["Cl"] * case.Mach / 2.0
                op.Gamma += 0.5 * (gam_target - op.Gamma)
            hist["step"].append(n)
            hist["res"].append(res)
            hist["Cl"].append(fo["Cl"])
            hist["Cd"].append(fo["Cd"])
            hist["Cm"].append(fo["Cm"])
            hist["Gamma"].append(op.Gamma)
            hist["min_p"].append(float(pressure(U, GAMMA).min()))
            hist["time"].append(time.time() - t_start)
            if (case.ckpt_every > 0
                    and n % case.ckpt_every < case.monitor_every
                    and np.isfinite(U).all()):
                np.savez_compressed(
                    ckpt, U=U, X=mesh.X, Y=mesh.Y, J=mesh.J, Qinf=Qinf,
                    step=n, res_max=res_max, Gamma=op.Gamma,
                    case=json.dumps(asdict(case)))
            if verbose and n % max(case.monitor_every * 20, 1000) == 0:
                orders = np.log10(max(res_max, 1e-300) / max(res, 1e-300))
                print(f"  [{case.label()}] step {n:7d}  res {res:.2e} "
                      f"({orders:4.1f} ord)  Cl {fo['Cl']:+.5f}  "
                      f"Cd {fo['Cd']:+.5f}  Cm {fo['Cm']:+.5f}  "
                      f"G {op.Gamma:+.4f}  "
                      f"min(p) {pressure(U, GAMMA).min():.3f}", flush=True)
            if not np.isfinite(res) or np.isnan(U).any():
                reason = "diverged"
                break
            # strict convergence test
            if len(hist["step"]) >= 8 and n > start_step + 20000:
                orders = np.log10(max(res_max, 1e-300) / max(res, 1e-300))
                i0, i1 = -8, -1
                span = max(hist["step"][i1] - hist["step"][i0], 1)
                scale = 1000.0 / span
                drift = max(abs(hist["Cl"][i1] - hist["Cl"][i0]),
                            abs(hist["Cd"][i1] - hist["Cd"][i0]),
                            abs(hist["Cm"][i1] - hist["Cm"][i0])) * scale
                if orders >= case.res_drop_orders and drift < case.force_tol:
                    reason = "converged"
                    break

    fo = forces(U, mesh, case.Mach, case.alpha_deg, GAMMA)
    ent = entropy_error(U, mesh, GAMMA, near_field_r=5.0)
    orders = float(np.log10(max(res_max, 1e-300) / max(hist["res"][-1], 1e-300))) \
        if hist["res"] else 0.0

    result = {
        "case": asdict(case),
        "mesh": {"n_k": mesh.n_k, "n_j": mesh.n_j, "N": mesh.N,
                 "nodes": int(mesh.n_k * mesh.n_j * mesh.Np1 ** 2 * 4),
                 "elements": int(mesh.n_k * mesh.n_j),
                 "quality": mesh.quality_report(),
                 "metric_identities": mesh.metric_identities()},
        "history": hist,
        "forces": {k: fo[k] for k in
                   ("Cl", "Cd", "Cm", "Cl_trace", "Cd_trace",
                    "max_dp_wall")},
        "entropy": ent,
        "steps": n, "reason": reason, "residual_orders": orders,
        "Gamma": op.Gamma, "n_limited": n_limited,
        "wall_time_s": time.time() - t_start,
    }

    if case.save_field:
        if np.isfinite(U).all():
            np.savez_compressed(
                ckpt, U=U, X=mesh.X, Y=mesh.Y, J=mesh.J, Qinf=Qinf,
                step=n, res_max=res_max, Gamma=op.Gamma,
                case=json.dumps(asdict(case)))
        else:
            (out / f"diverged_{case.label()}").write_text(
                f"diverged at step {n}\n")
    import csv
    merged = {k: list(hist_prior[k]) + list(hist[k]) for k in hist}
    with open(hist_file, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["step", "residual", "Cl", "Cd", "Cm", "Gamma",
                     "min_p", "wall_s"])
        for i in range(len(merged["step"])):
            wr.writerow([merged["step"][i], merged["res"][i],
                         merged["Cl"][i], merged["Cd"][i], merged["Cm"][i],
                         merged["Gamma"][i], merged["min_p"][i],
                         merged["time"][i]])

    if verbose:
        print(f"  [{case.label()}] {reason}: {n} steps, "
              f"{result['wall_time_s']:.0f} s, res drop {orders:.1f} ord, "
              f"Cl {fo['Cl']:+.6f} Cd {fo['Cd']:+.6f} "
              f"Cm {fo['Cm']:+.6f}, Es {ent['entropy_l2_domain']:.2e}",
              flush=True)
    result["U"] = U
    result["mesh_obj"] = mesh
    return result


# ----------------------------------------------------------------------
def _self_test() -> None:  # pragma: no cover
    print("solver smoke test (tiny mesh, short run, impulsive start)")
    case = Case(Mach=0.3, alpha_deg=5.0, n_half=10, n_r=8, N=3,
                R_far=25.0, max_steps=1500, res_drop_orders=2.0,
                force_tol=1e-2, out_dir="/tmp/dgsem_smoke",
                max_wall_s=240.0, resume=False, save_field=False)
    r = run_case(case)
    print(f"  Cl = {r['forces']['Cl']:+.4f} after {r['steps']} steps "
          f"({r['reason']}), Es = {r['entropy']['entropy_l2_domain']:.2e}")
    assert r["forces"]["Cl"] > 0.3
    print("  smoke test passed")


if __name__ == "__main__":
    _self_test()
