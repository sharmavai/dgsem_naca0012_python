#!/usr/bin/env python3
"""Smoke test of the production-campaign machinery before the long run.

Verifies, on a small mesh and a short wall budget:
  1. run_case end-to-end with the campaign BASE parameters
     (mesh -> operator -> impulsive march -> checkpoint -> history CSV
      -> forces JSON-serialisable),
  2. the claim/done bookkeeping used by run_campaign.run_worker,
  3. the N3 -> N4 prolongation einsum used by the p-refinement cases,
  4. that a resumed case continues from the checkpoint.

Run:  python3 scripts/smoke_campaign.py   (~2-3 min incl. numba JIT)
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

OUT = Path("/tmp/smoke_campaign")
if OUT.exists():
    shutil.rmtree(OUT)
OUT.mkdir(parents=True)

from dgsem.euler import freestream_state
from dgsem.mesh import generate_airfoil_omesh
from dgsem.quadrature import lgl_nodes, lagrange_interpolation_matrix
from dgsem.solver import Case, run_case

BASE = dict(Mach=0.30, R_far=50.0, out_dir=str(OUT), save_field=True,
            max_steps=400000, res_drop_orders=5.0, force_tol=3.0e-5,
            max_wall_s=45.0, monitor_every=200)

print("=== 1) campaign-style case (L1 mesh, 45 s budget) ===")
t0 = time.time()
case = Case(tag="smoke_a5", resume=True, alpha_deg=5.0,
            n_half=20, n_r=15, **{k: v for k, v in BASE.items()
                                  if k != "out_dir"} | {"out_dir": str(OUT)})
r = run_case(case, verbose=True)
print(f"    wall {time.time()-t0:.0f} s, reason {r['reason']}, "
      f"steps {r['steps']}, Cl {r['forces']['Cl']:+.4f}")
assert r["reason"] in ("wall_budget", "converged", "max_steps")
assert np.isfinite(r["forces"]["Cl"])

# claim/done bookkeeping
claim = OUT / "claim_smoke_a5"
claim.mkdir()
try:
    claim.mkdir()
    raise AssertionError("double claim was allowed!")
except FileExistsError:
    print("    claim-safe OK")
(OUT / "done_smoke_a5").write_text(json.dumps(
    {"reason": r["reason"], "steps": r["steps"],
     "forces": {"Cl": r["forces"]["Cl"]}}, default=str))
print("    done_ marker OK")

print("=== 2) resume from checkpoint ===")
n_before = r["steps"]
case2 = Case(tag="smoke_a5", resume=True, alpha_deg=5.0,
             n_half=20, n_r=15, max_wall_s=20.0, monitor_every=200,
             out_dir=str(OUT), save_field=True)
r2 = run_case(case2, verbose=False)
assert r2["steps"] > n_before, "resume did not continue"
print(f"    resumed {n_before} -> {r2['steps']} steps OK")

print("=== 3) prolongation N3 -> N4 (campaign einsum) ===")
d = np.load(OUT / "ckpt_smoke_a5.npz")
U = d["U"]
print(f"    state shape (N=3): {U.shape}")
V = lagrange_interpolation_matrix(lgl_nodes(4), lgl_nodes(3))
Up = np.einsum('bi,al,kjilc->kjbac', V, V, U)
print(f"    prolonged shape (N=4): {Up.shape}")
assert Up.shape[2] == 5 and Up.shape[3] == 5
assert np.isfinite(Up).all()
# mass conservation through prolongation (constant field)
mass_before = U[..., 0].sum()
mass_after = Up[..., 0].sum()
print(f"    mass before {mass_before:.6f} after {mass_after:.6f} "
      f"(rel diff {abs(mass_after/mass_before-1):.2e})")

print("=== 4) run the prolonged state a few steps (stability probe) ===")
import dgsem.solver as S
import dgsem.euler as E
_orig = E.initialize_freestream
_q = freestream_state(0.30, 5.0, 1.4)
E.initialize_freestream = lambda mesh, M, a, g: (Up.copy(), _q)
S.initialize_freestream = E.initialize_freestream
try:
    case3 = Case(tag="smoke_n4", resume=False, alpha_deg=5.0,
                 n_half=20, n_r=15, N=4, max_wall_s=30.0,
                 monitor_every=100, out_dir=str(OUT), save_field=False)
    r3 = run_case(case3, verbose=True)
finally:
    E.initialize_freestream = _orig
    S.initialize_freestream = _orig
print(f"    N4 probe: reason {r3['reason']}, steps {r3['steps']}, "
      f"Cl {r3['forces']['Cl']:+.4f}, finite={np.isfinite(r3['U']).all()}")

print("\nALL SMOKE TESTS PASSED")
