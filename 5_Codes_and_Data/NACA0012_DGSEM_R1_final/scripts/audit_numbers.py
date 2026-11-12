#!/usr/bin/env python3
"""Recompute every headline number of the manuscript from the stored campaign
data and the verified reference table.

Run from the package root:

    python scripts/audit_numbers.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results_new"

# the eight angles at which Table IV (R = 5.93e6) has a measurement
LADSON_A = [-4.37, -2.02, -0.01, 0.03, 4.03, 6.08, 8.13, 10.12]
# the nine campaign sweep angles (alpha = 2.05 has no counterpart at 5.93e6)
SWEEP_A = [-4.37, -2.02, -0.01, 0.03, 2.05, 4.03, 6.08, 8.13, 10.12]
MIRROR_A = [-8.13, -10.12]


def load_history(tag):
    rows = list(csv.DictReader(open(RES / f"history_{tag}.csv")))
    return {k: np.array([float(r[k]) for r in rows])
            for k in ("step", "residual", "Cl", "Cd", "Cm", "Gamma")}


def stats(tag, frac=0.25):
    h = load_history(tag)
    i0 = max(0, int(len(h["Cl"]) * (1 - frac)))
    out = {}
    for k in ("Cl", "Cd", "Cm"):
        tail = h[k][i0:]
        out[k] = (tail.mean(), 0.5 * (tail.max() - tail.min()))
    return out


def done(tag):
    return json.loads((RES / f"done_{tag}").read_text())


def tag_of(a):
    return f"sweep_a{a:+g}".replace("+", "p").replace("-", "m")


def ladson():
    rows = list(csv.DictReader(open(ROOT / "data" / "experimental"
                                    / "ladson_force_data.csv")))
    return (np.array([float(r["alpha_deg"]) for r in rows]),
            np.array([float(r["CL"]) for r in rows]),
            np.array([float(r["CD"]) for r in rows]),
            np.array([float(r["CM"]) for r in rows]))


def main():
    print("=" * 78)
    print("AUDIT: numbers recomputed from results_new/ and the verified table")
    print("=" * 78)

    sweep = {}
    print("\n[1] sweep cases (mean, half wake-cycle amplitude)")
    for a in SWEEP_A + MIRROR_A:
        t = tag_of(a)
        s = stats(t)
        d = done(t)
        sweep[a] = s
        print(f"  {t:<16} a={a:+7.2f}  Cl={s['Cl'][0]:+.5f} "
              f"+/-{s['Cl'][1]:.5f}  Cd={s['Cd'][0]:+.5f}  "
              f"Cm={s['Cm'][0]:+.5f}  steps={d['steps']:>6}  "
              f"ord={d['residual_orders']:4.2f}  {d['reason']}")

    ae, cle, cde, cme = ladson()

    def fit(x, y):
        return np.polyfit(np.asarray(x), np.asarray(y), 1)

    # the fits use the SWEEP points only (the mirror runs are an independent
    # antisymmetry check and are never part of a least-squares fit)
    print("\n[2] lift-curve slopes (least squares over the sweep points)")
    sw8 = [a for a in SWEEP_A if abs(a) <= 8.13]
    f_dg8 = fit(sw8, [sweep[a]["Cl"][0] for a in sw8])
    print(f"  (a) DGSEM, 8 sweep angles |a|<=8.13 (includes 2.05): "
          f"slope = {f_dg8[0]:.6f} /deg, alpha0 = {-f_dg8[1] / f_dg8[0]:+.4f} deg")
    sw7 = [a for a in sw8 if abs(a - 2.05) > 1e-9]
    f_dg7 = fit(sw7, [sweep[a]["Cl"][0] for a in sw7])
    print(f"  (b) DGSEM, 7 sweep angles |a|<=8.13 (2.05 excluded): "
          f"slope = {f_dg7[0]:.6f} /deg")
    f_ex = fit([a for a in ae if abs(a) <= 8.13],
               [c for a, c in zip(ae, cle) if abs(a) <= 8.13])
    print(f"  (c) experiment, 7 measured angles |a|<=8.13: "
          f"slope = {f_ex[0]:.6f} /deg, alpha0 = {-f_ex[1] / f_ex[0]:+.4f} deg")
    print(f"  ratio (a)/(c) = {f_dg8[0] / f_ex[0]:.4f}  "
          f"offset = {100 * (f_dg8[0] / f_ex[0] - 1):+.2f}%")

    f_dg_all = fit(SWEEP_A, [sweep[a]["Cl"][0] for a in SWEEP_A])
    print(f"  (d) DGSEM, all 9 sweep angles |a|<=10.12: "
          f"slope = {f_dg_all[0]:.6f} /deg "
          f"(diff vs (a) = {100 * (f_dg_all[0] / f_dg8[0] - 1):+.2f}%)")
    f_ex_all = fit(ae, cle)
    print(f"      experiment, all 8 measured angles: slope = {f_ex_all[0]:.6f} /deg")

    print("\n[3] pointwise inviscid/viscous ratio at the measured angles")
    rats = []
    for a in LADSON_A:
        i = int(np.argmin(np.abs(ae - a)))
        c = sweep[a]["Cl"][0]
        r = c / cle[i]
        if abs(cle[i]) > 0.1:
            rats.append(r)
        print(f"  alpha={a:+7.2f}: Cl_dg={c:+.4f}  Cl_exp={cle[i]:+.4f}  "
              f"ratio={r:+.4f}  offset={100 * (r - 1):+7.2f}%"
              + ("   (near zero lift: ratio not meaningful)"
                 if abs(cle[i]) <= 0.1 else ""))
    rats = np.array(rats)
    print(f"  ratio range {rats.min():.4f}..{rats.max():.4f}")

    print("\n[4] mirror antisymmetry")
    for lo, hi in ((-10.12, 10.12), (-8.13, 8.13)):
        cm_, am_ = sweep[lo]["Cl"]
        cp_, ap_ = sweep[hi]["Cl"]
        print(f"  Cl({lo:+g})={cm_:+.5f}+/-{am_:.5f}  "
              f"Cl({hi:+g})={cp_:+.5f}+/-{ap_:.5f}  "
              f"asymmetry={100 * (cp_ + cm_) / ((cp_ - cm_) / 2):+.2f}%")

    print("\n[5] quarter-chord moment slope")
    m7 = [a for a in LADSON_A if abs(a) <= 8.13]
    f_cm = fit(m7, [sweep[a]["Cm"][0] for a in m7])
    f_cme = fit([a for a in ae if abs(a) <= 8.13],
                [c for a, c in zip(ae, cme) if abs(a) <= 8.13])
    print(f"  DGSEM dCm/da = {f_cm[0]:.5f} /deg   exp = {f_cme[0]:.5f} /deg")

    print("\n[6] drag level (Euler -> Cd ~ 0)")
    print(f"  max |Cd| over the sweep = "
          f"{max(abs(sweep[a]['Cd'][0]) for a in SWEEP_A):.2e}")

    print("\n[7] h-refinement (alpha = 5)")
    for t in ("mesh_L1_a5", "mesh_L2_a5", "mesh_L3_a5"):
        s = stats(t)["Cl"]
        print(f"  {t:<14} Cl={s[0]:.4f} +/- {s[1]:.4f}  "
              f"steps={done(t)['steps']:>6}")

    print("\n[8] p-refinement (alpha = 5)")
    for t in ("p_N2_a5", "mesh_L2_a5", "p_N4_a5", "p_N5_a5"):
        s = stats(t)["Cl"]
        print(f"  {t:<14} Cl={s[0]:.4f} +/- {s[1]:.4f}  "
              f"steps={done(t)['steps']:>6}  ord={done(t)['residual_orders']:4.2f}")

    print("\n[9] sensitivity (base = mesh_L2_a5)")
    base, base_amp = stats("mesh_L2_a5")["Cl"]
    print(f"  base Cl = {base:.4f} +/- {base_amp:.4f} "
          f"(+/- {100 * base_amp / base:.2f}%)")
    worst, worst_tag = 0.0, ""
    for t in ("sens_eps01", "sens_eps03", "sens_R25", "sens_R100",
              "sens_novortex", "sens_rusanov", "sens_sponge"):
        m, s = stats(t)["Cl"]
        rel = 100 * (m - base) / base
        if abs(rel) > worst:
            worst, worst_tag = abs(rel), t
        print(f"  {t:<14} Cl={m:.4f} +/- {s:.4f} "
              f"(+/-{100 * s / base:5.2f}% of base)  d={rel:+.2f}%")
    print(f"  largest |relative change| = {worst:.2f}%  ({worst_tag})")

    print("\n[10] wake-cycle half-amplitude as a fraction of |Cl|")
    fr = [(a, 100 * sweep[a]["Cl"][1] / abs(sweep[a]["Cl"][0]))
          for a in SWEEP_A if abs(sweep[a]["Cl"][0]) > 0.2]
    for a, v in sorted(fr, key=lambda z: z[1]):
        print(f"  alpha={a:+7.2f}: {v:5.2f}%")
    print(f"  lifted-case range = {min(v for _, v in fr):.2f}% .. "
          f"{max(v for _, v in fr):.2f}%")

    print("\n[11] entropy error (domain L2) at the sweep angles")
    for a in SWEEP_A:
        e = done(tag_of(a))["entropy"]
        print(f"  alpha={a:+7.2f}: Es_domain={e['entropy_l2_domain']:.3e}  "
              f"Es_near={e['entropy_l2_nearfield']:.3e}")

    print("\n[12] residual-drop range over the campaign")
    ords = [done(t)["residual_orders"] for t in
            [tag_of(a) for a in SWEEP_A + MIRROR_A]
            + ["mesh_L1_a5", "mesh_L2_a5", "mesh_L3_a5",
               "p_N2_a5", "p_N4_a5", "p_N5_a5"]]
    print(f"  min {min(ords):.2f} decades, max {max(ords):.2f} decades")
    steps = [done(t)["steps"] for t in
             [tag_of(a) for a in SWEEP_A + MIRROR_A]]
    print(f"  sweep step range {min(steps)}..{max(steps)}")

    print("\n[13] Gamma feedback check (should equal -Cl*M/2)")
    for a in (10.12, 4.03):
        h = load_history(tag_of(a))
        print(f"  alpha={a:+6.2f}: Gamma={h['Gamma'][-1]:+.5f}  "
              f"-Cl*M/2={-sweep[a]['Cl'][0] * 0.30 / 2:+.5f}")


if __name__ == "__main__":
    main()
