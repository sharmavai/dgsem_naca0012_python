#!/usr/bin/env python3
"""Manual retries of the two diverged campaign cases at a reduced CFL.

sweep_am8.13 and sens_rusanov diverged at the production CFL of 1.5
(impulsive-start transient).  This driver restarts them from scratch at
CFL 1.2 and writes the campaign-format done_ markers, so the figures and
tables pick them up like any other case.  The diverged_ markers are
preserved with a .cfl15 suffix for the record.
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from dgsem.solver import Case, run_case

OUT = ROOT / "results_new"

BASE = dict(Mach=0.30, R_far=50.0, out_dir=str(OUT), save_field=True,
            max_steps=400000, res_drop_orders=5.0, force_tol=3.0e-5,
            max_wall_s=500.0, monitor_every=200)

RETRIES = [
    # (tag, case overrides)
    ("sweep_am8.13", dict(alpha_deg=-8.13, n_half=28, n_r=21, cfl=1.2)),
    ("sens_rusanov", dict(alpha_deg=5.0, n_half=28, n_r=21, cfl=1.2,
                          flux="rusanov")),
]


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    for tag, over in RETRIES:
        if which != "all" and which not in tag:
            continue
        div = OUT / f"diverged_{tag}"
        if div.exists():
            shutil.move(str(div), str(OUT / f"diverged_{tag}.cfl15"))
            print(f"[retry] archived {div.name} -> diverged_{tag}.cfl15")
        stale_ckpt = OUT / f"ckpt_{tag}.npz"
        if stale_ckpt.exists():
            stale_ckpt.unlink()          # fresh impulsive start
        t0 = time.time()
        case = Case(tag=tag, resume=True, **dict(BASE, **over))
        r = run_case(case, verbose=True)
        info = {k: r[k] for k in ("reason", "steps", "residual_orders",
                                  "wall_time_s")}
        info["forces"] = r["forces"]
        info["entropy"] = r["entropy"]
        info["case"] = {k: v for k, v in r["case"].items()
                        if k != "out_dir"}
        info["retried_at_cfl"] = over["cfl"]
        (OUT / f"done_{tag}").write_text(json.dumps(info, indent=1,
                                                    default=str))
        print(f"[retry] {tag}: {r['reason']}, Cl "
              f"{r['forces']['Cl']:+.4f}, {time.time()-t0:.0f} s")


if __name__ == "__main__":
    main()
