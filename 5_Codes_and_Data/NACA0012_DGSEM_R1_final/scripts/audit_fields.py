#!/usr/bin/env python3
"""Recompute the flow-field statistics quoted in the manuscript directly from
the stored Gauss-node checkpoints (no figure pipeline in between).

    python scripts/audit_fields.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
RES = ROOT / "results_new"

from dgsem.euler import GAMMA  # noqa: E402


def stats(tag):
    d = np.load(RES / f"ckpt_{tag}.npz", allow_pickle=True)
    U = d["U"]
    case = json.loads(str(d["case"]))
    M = case["Mach"]
    rho = U[..., 0]
    u = U[..., 1] / rho
    v = U[..., 2] / rho
    E = U[..., 3]
    p = (GAMMA - 1.0) * (E - 0.5 * rho * (u * u + v * v))
    c = np.sqrt(GAMMA * p / rho)
    q = np.sqrt(u * u + v * v)
    mach = q / c
    p_inf = 1.0 / GAMMA
    ent = np.log(p / rho ** GAMMA)
    return dict(
        tag=tag, M=M,
        rho_min=rho.min() / 1.0, rho_max=rho.max(),
        p_min=p.min() / p_inf, p_max=p.max() / p_inf,
        cp_min=(p.min() - p_inf) / (0.5 * M * M),
        cp_max=(p.max() - p_inf) / (0.5 * M * M),
        mach_max=mach.max(), n_sup=int((mach > 1.0).sum()),
        n_nodes=mach.size,
        vel_max=q.max() / M,
        ent_min=ent.min(), ent_max=ent.max(),
        ent_l2=np.sqrt(np.mean(ent ** 2)),
        p=np,  # placeholder
    )


def main():
    print("flow-field statistics from the stored checkpoints "
          "(M_inf = 0.30, L2 56x21, N = 3)")
    print(f"{'case':<16} {'rho/rhi':>16} {'p/pinf':>16} {'Cp_min':>7} "
          f"{'M_max':>6} {'n(M>1)':>7} {'|V|/Vinf':>8} {'Es_L2':>9}")
    for tag in ("sweep_ap0.03", "sweep_ap4.03", "sweep_ap10.12",
                "mesh_L2_a5", "mesh_L3_a5", "p_N5_a5"):
        try:
            s = stats(tag)
        except FileNotFoundError:
            print(f"{tag:<16}  (checkpoint missing)")
            continue
        print(f"{s['tag']:<16} "
              f"{s['rho_min']:.3f}..{s['rho_max']:.3f} "
              f"{s['p_min']:.3f}..{s['p_max']:.3f} "
              f"{s['cp_min']:7.3f} {s['mach_max']:6.3f} "
              f"{s['n_sup']:7d}/{s['n_nodes']:<6d} "
              f"{s['vel_max']:8.3f} {s['ent_l2']:9.2e}")
        if s["n_sup"]:
            print(f"    -> supersonic nodes: {s['n_sup']} of {s['n_nodes']}, "
                  f"M_max = {s['mach_max']:.4f}, "
                  f"entropy range across field = "
                  f"[{s['ent_min']:+.3e}, {s['ent_max']:+.3e}]")


if __name__ == "__main__":
    main()
