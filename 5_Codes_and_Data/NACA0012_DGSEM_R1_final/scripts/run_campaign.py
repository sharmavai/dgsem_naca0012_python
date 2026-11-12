#!/usr/bin/env python3
"""Production campaign driver.

Runs the full matrix of the paper from impulsive starts (the p-refinement
runs at N >= 4 are initialised by prolongation of the converged N = 3
solution, as documented in the paper).  Every case checkpoints and resumes
automatically, so the campaign can be advanced in bounded wall-time chunks:

    python3 run_campaign.py            (process pending cases, ~8 min)
    python3 run_campaign.py status     (print the table)

Two workers can run concurrently (the script is claim-safe):
    python3 run_campaign.py & python3 run_campaign.py & wait
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dgsem.euler import freestream_state
from dgsem.mesh import generate_airfoil_omesh
from dgsem.quadrature import lgl_nodes, lagrange_interpolation_matrix
from dgsem.solver import Case, run_case

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results_new"
OUT.mkdir(exist_ok=True)

# ---------------------------------------------------------------- mesh family
L1 = dict(n_half=20, n_r=15)
L2 = dict(n_half=28, n_r=21)
L3 = dict(n_half=40, n_r=30)

BASE = dict(Mach=0.30, R_far=50.0, out_dir=str(OUT), save_field=True,
            max_steps=400000, res_drop_orders=5.0, force_tol=3.0e-5,
            max_wall_s=520.0, monitor_every=200)

# Ladson NASA TM-4074 Table IV measured angles at M = 0.30, R = 5.93e6.
# The 2.05 deg entry exists only in the R = 3.90e6 block of the same table
# and has no counterpart at 5.93e6: see data/experimental/SOURCES.md.
LADSON_A = [-4.37, -2.02, -0.01, 0.03, 2.05, 4.03, 6.08, 8.13, 10.12]

MANIFEST = []

# --- 1) angle-of-attack sweep (impulsive) --------------------------------
for a in LADSON_A:
    tag = f"sweep_a{a:+g}".replace("+", "p").replace("-", "m")
    MANIFEST.append(dict(tag=tag, kind="sweep",
                         case=dict(BASE, alpha_deg=a, **L2),
                         init="uniform"))

# --- 2) antisymmetry mirrors (impulsive) ----------------------------------
for a in (-8.13, -10.12):
    tag = f"sweep_a{a:+g}".replace("+", "p").replace("-", "m")
    MANIFEST.append(dict(tag=tag, kind="antisym",
                         case=dict(BASE, alpha_deg=a, **L2),
                         init="uniform"))

# --- 3) mesh refinement study at alpha = 5 (impulsive) --------------------
for name, fam in (("L1", L1), ("L2", L2), ("L3", L3)):
    MANIFEST.append(dict(tag=f"mesh_{name}_a5", kind="mesh",
                         case=dict(BASE, alpha_deg=5.0, **fam),
                         init="uniform"))

# --- 4) p refinement study at alpha = 5 on L2 ----------------------------
MANIFEST.append(dict(tag="p_N2_a5", kind="pstudy",
                     case=dict(BASE, alpha_deg=5.0, N=2, **L2),
                     init="uniform"))
for NN in (4, 5):
    MANIFEST.append(dict(tag=f"p_N{NN}_a5", kind="pstudy",
                         case=dict(BASE, alpha_deg=5.0, N=NN, **L2),
                         init=f"prolong:mesh_L2_a5:3:{NN}"))

# --- 5) sensitivity study at alpha = 5 on L2/N3 --------------------------
SENS = [
    ("sens_eps01", dict(entropy_fix=0.01)),
    ("sens_eps03", dict(entropy_fix=0.03)),
    ("sens_R25", dict(R_far=25.0)),
    ("sens_R100", dict(R_far=100.0)),
    ("sens_novortex", dict(farfield_vortex=False)),
    ("sens_rusanov", dict(flux="rusanov")),
    ("sens_sponge", dict(sponge_rate=0.01)),
]
for tag, over in SENS:
    MANIFEST.append(dict(tag=tag, kind="sens",
                         case=dict(BASE, alpha_deg=5.0, **dict(L2, **over)),
                         init="uniform"))


# ----------------------------------------------------------------------
def prolong_state(src_tag: str, N_from: int, N_to: int):
    """Prolong a converged state onto the same mesh at a higher degree."""
    d = np.load(OUT / f"ckpt_{src_tag}.npz")
    U = d["U"]
    xf = lgl_nodes(N_from)
    xt = lgl_nodes(N_to)
    V = lagrange_interpolation_matrix(xt, xf)
    return np.einsum('bi,al,kjilc->kjbac', V, V, U)


def init_state(spec: str):
    """Return (U0 or None, forced_resume_off)."""
    if spec == "uniform":
        return None
    if spec.startswith("prolong:"):
        _, src, nf, nt = spec.split(":")
        return prolong_state(src, int(nf), int(nt))
    raise ValueError(spec)


# ----------------------------------------------------------------------
def claim(tag: str) -> bool:
    p = OUT / f"claim_{tag}"
    try:
        p.mkdir()
        return True
    except FileExistsError:
        return False


def done(tag: str) -> bool:
    return (OUT / f"done_{tag}").exists()


def run_worker(budget_s: float):
    t0 = time.time()
    for item in MANIFEST:
        if done(item["tag"]):
            continue
        if not claim(item["tag"]):
            continue
        if time.time() - t0 > budget_s:
            (OUT / f"claim_{item['tag']}").rmdir()
            return
        U0 = None
        try:
            U0 = init_state(item["init"])
        except FileNotFoundError:
            # dependency not ready yet: unclaim, try later cases
            (OUT / f"claim_{item['tag']}").rmdir()
            continue
        case = Case(tag=item["tag"], resume=True, **item["case"])
        if U0 is not None:
            import dgsem.solver as S
            import dgsem.euler as E
            _orig = E.initialize_freestream
            _q = freestream_state(case.Mach, case.alpha_deg, 1.4)
            E.initialize_freestream = (
                lambda mesh, M, a, g: (U0.copy(), _q))
            S.initialize_freestream = E.initialize_freestream
            try:
                r = run_case(case, verbose=True)
            finally:
                E.initialize_freestream = _orig
                S.initialize_freestream = _orig
        else:
            r = run_case(case, verbose=True)
        info = {k: r[k] for k in ("reason", "steps", "residual_orders",
                                  "wall_time_s", "n_limited")}
        info["forces"] = r["forces"]
        info["entropy"] = r["entropy"]
        info["case"] = {k: v for k, v in r["case"].items()
                        if k not in ("out_dir",)}
        (OUT / f"done_{item['tag']}").write_text(json.dumps(info, indent=1,
                                                            default=str))
        print(f"[campaign] finished {item['tag']}: {r['reason']}",
              flush=True)


def status():
    rows = []
    for item in MANIFEST:
        t = item["tag"]
        f = OUT / f"done_{t}"
        if f.exists():
            d = json.loads(f.read_text())
            fo = d.get("forces", {})
            rows.append((t, d.get("reason", "?"),
                         f"{fo.get('Cl', float('nan')):+.4f}",
                         f"{fo.get('Cd', float('nan')):+.5f}",
                         str(d.get("steps", "?")),
                         f"{d.get('residual_orders', 0):.1f}"))
        elif (OUT / f"claim_{t}").exists() or (OUT / f"ckpt_{t}.npz").exists():
            rows.append((t, "PENDING/running", "", "", "", ""))
        else:
            rows.append((t, "-", "", "", "", ""))
    w = max(len(r[0]) for r in rows) + 2
    print(f"{'case'.ljust(w)}  {'reason':<14} {'Cl':>8} {'Cd':>9} "
          f"{'steps':>8} {'res.ord':>7}")
    for r in rows:
        print(f"{r[0].ljust(w)}  {r[1]:<14} {r[2]:>8} {r[3]:>9} "
              f"{r[4]:>8} {r[5]:>7}")
    n_done = sum(1 for r in rows if r[1] not in ("-", "PENDING/running"))
    print(f"{n_done}/{len(rows)} complete")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "status":
        status()
        raise SystemExit(0)
    budget = float(sys.argv[1]) if len(sys.argv) > 1 else 480.0
    run_worker(budget)
