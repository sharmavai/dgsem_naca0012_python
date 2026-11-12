#!/usr/bin/env python3
"""Quantify the impact of the symmetric time-step smoother fix on a
production case: re-run the reference case (alpha = 5, L2 mesh, N = 3)
from an impulsive start with the CURRENT (symmetric, mirror-covariant)
smoother and compare against the campaign result that was produced with
the previous sequential smoother."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from dgsem.solver import Case, run_case
from make_figures import cl_stats

OUT = ROOT / "results_new"

if __name__ == "__main__":
    tag = "mesh_L2_a5_v2"
    ck = OUT / f"ckpt_{tag}.npz"
    if not ck.exists():
        case = Case(tag=tag, resume=False, alpha_deg=5.0,
                    n_half=28, n_r=21, Mach=0.30, R_far=50.0,
                    max_steps=400000, res_drop_orders=5.0,
                    force_tol=3.0e-5, max_wall_s=500.0, monitor_every=200,
                    out_dir=str(OUT), save_field=True)
        r = run_case(case, verbose=True)
        info = {k: r[k] for k in ("reason", "steps", "residual_orders",
                                  "wall_time_s")}
        info["forces"] = r["forces"]
        info["case"] = {k: v for k, v in r["case"].items()
                        if k != "out_dir"}
        (OUT / f"done_{tag}").write_text(json.dumps(info, indent=1,
                                                    default=str))
    m_new, a_new = cl_stats(tag)
    m_old, a_old = cl_stats("mesh_L2_a5")
    print(f"old (sequential smoother): Cl {m_old:+.4f} +/- {a_old:.4f}")
    print(f"new (symmetric smoother): Cl {m_new:+.4f} +/- {a_new:.4f}")
    print(f"mean shift: {m_new - m_old:+.4f} "
          f"({100*(m_new-m_old)/abs(m_old):+.2f}%)")
