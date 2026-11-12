#!/usr/bin/env python3
"""Publication-quality figures for the REVISED NACA 0012 DGSEM study
(post-referee-review version).

Differences from the original figure set
----------------------------------------
* every plotted production number now comes from the revised campaign in
  ``results_new`` (impulsive starts, no sponge, no wake dissipation,
  strict convergence criteria, O-mesh with periodic wake seam),
* the lift curve runs at the EXACT measured angles of Ladson's Table IV
  (no interpolation against the experiment is needed),
* every DGSEM force coefficient is plotted as mean +/- the oscillation
  amplitude of the discrete wake-sheet limit cycle over the last quarter
  of the march (an honest uncertainty statement),
* new figures: mesh/p refinement studies, parameter sensitivity, and
  convergence histories with the strict stopping criteria,
* the transonic (M = 0.8) figure is DROPPED: the review rejected
  unconverged transonic results; the case is not part of the revised
  campaign (documented in docs/RESPONSE_TO_REVIEWERS.md).

Every figure is written as PNG (300 dpi), PDF (vector) and SVG (editable),
and the plotted data are exported to CSV alongside the images.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dgsem.euler import GAMMA, primitives
from dgsem.geometry import naca0012_thickness
from dgsem.forces import surface_cp
from dgsem.mesh import generate_airfoil_omesh

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results_new"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

MM = 1 / 25.4
SINGLE = 84 * MM
DOUBLE = 174 * MM

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 8,
    "axes.titlesize": 8.5,
    "axes.labelsize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "lines.linewidth": 1.0,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "axes.grid": False,
})

C_DG = "#0077BB"      # solver
C_DG2 = "#33BBEE"     # solver (secondary)
C_EXP = "#000000"     # experiment
C_MIR = "#EE7733"     # antisymmetry mirrors
C_OK = "#009988"      # verification
C_WARN = "#CC3311"    # warnings / diverged

MESH_FAMILY = {
    "L1": dict(n_half=20, n_r=15),
    "L2": dict(n_half=28, n_r=21),
    "L3": dict(n_half=40, n_r=30),
}

LADSON_A = [-4.37, -2.02, -0.01, 0.03, 2.05, 4.03, 6.08, 8.13, 10.12]
LADSON_MIRROR = [-8.13, -10.12]


# ======================================================================
# data loading helpers
# ======================================================================
def load_done(tag: str):
    f = RES / f"done_{tag}"
    if not f.exists():
        return None
    return json.loads(f.read_text())


def load_history(tag: str) -> dict:
    f = RES / f"history_{tag}.csv"
    rows = list(csv.DictReader(open(f))) if f.exists() else []
    out = {}
    for k, cast in (("step", int), ("residual", float), ("Cl", float),
                    ("Cd", float), ("Cm", float), ("Gamma", float),
                    ("min_p", float), ("wall_s", float)):
        out[k] = np.array([cast(r[k]) for r in rows]) if rows else np.array([])
    return out


def cl_stats(tag: str, frac: float = 0.25):
    """(mean, half-amplitude) of Cl over the last `frac` of the march."""
    h = load_history(tag)
    if len(h["Cl"]) == 0:
        return float("nan"), 0.0
    i0 = max(0, int(len(h["Cl"]) * (1.0 - frac)))
    tail = h["Cl"][i0:]
    return float(tail.mean()), float(0.5 * (tail.max() - tail.min()))


def load_ckpt(tag: str):
    f = RES / f"ckpt_{tag}.npz"
    if not f.exists():
        return None
    d = np.load(f, allow_pickle=True)
    case = json.loads(str(d["case"]))
    return d["U"], d["X"], d["Y"], case


def regen_mesh(case: dict):
    """Regenerate the mesh of a stored case (deterministic)."""
    wake_curve = 0.0
    if case.get("wake_align", True):
        from dgsem.panel import PanelSolver, _airfoil_vertices
        _a = np.radians(case["alpha_deg"])
        _ps = PanelSolver(*_airfoil_vertices(200))
        _ps.solve(np.cos(_a), np.sin(_a))
        cl_panel = 4.0 * np.pi * _ps.gamma * _ps.len.sum()
        cl_est = cl_panel / np.sqrt(max(1.0 - case["Mach"] ** 2, 1e-6))
        wake_curve = cl_est / (4.0 * np.pi)
    h_wall = case.get("h_wall") or 0.30 / case["n_r"]
    return generate_airfoil_omesh(case["n_half"], case["n_r"], case["N"],
                                  case["R_far"], h_wall,
                                  wake_curve=wake_curve)


def load_ladson():
    """The reference set: NASA TM-4074 Table IV, M = 0.30, R = 5.93e6, free
    transition (eight measured angles; the table has no 2.05 deg entry at
    this Reynolds number -- see data/experimental/SOURCES.md)."""
    f = ROOT / "data" / "experimental" / "ladson_force_data.csv"
    rows = [r for r in csv.DictReader(open(f))
            if abs(float(r["Mach"]) - 0.30) < 1e-9
            and abs(float(r["Re"]) / 1e6 - 5.93) < 1e-6
            and r["transition"] == "free"]
    assert len(rows) == 8, f"expected 8 reference rows, got {len(rows)}"
    return rows


def save(fig: plt.Figure, name: str) -> None:
    for ext in ("png", "pdf", "svg"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=300,
                    bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"  wrote figures/{name}.png/.pdf/.svg")


def export_csv(name: str, header: list, rows: list) -> None:
    with open(FIG / f"{name}_data.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(header)
        wr.writerows(rows)
    print(f"  wrote figures/{name}_data.csv")


def elem_lines(X, Y, Np1: int):
    """Element-boundary polylines for an O-mesh (k, j, i, l) layout."""
    lines = []
    nk, nj = X.shape[0], X.shape[1]
    for j in range(nj):
        for l in (0, Np1 - 1):
            xs = np.empty(nk * Np1)
            ys = np.empty(nk * Np1)
            for k in range(nk):
                xs[k * Np1:(k + 1) * Np1] = X[k, j, :, l]
                ys[k * Np1:(k + 1) * Np1] = Y[k, j, :, l]
            lines.append((xs, ys))
    for k in range(nk):
        for i in (0, Np1 - 1):
            lines.append((X[k, :, i, 0], Y[k, :, i, 0]))
    return lines


def naca0012_y(t):
    """Exact NACA 0012 half-thickness (t in [0, 1]).

    Delegates to the solver's own surface geometry (closed trailing
    edge) so the outline/silhouette drawn in the figures is EXACTLY the
    meshed wall.  (A previous revision used a garbled thickness formula
    here that drew a body ~15x too thick.)
    """
    return naca0012_thickness(t, t=0.12, closed=True)


# ======================================================================
# figure 1: the O-mesh family
# ======================================================================
def fig_mesh():
    m2 = generate_airfoil_omesh(28, 21, 3, 50.0, 0.30 / 21,
                                wake_curve=0.05)
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE, 58 * MM),
                             constrained_layout=True)

    # (a) full domain
    ax = axes[0]
    for xs, ys in elem_lines(m2.X, m2.Y, m2.Np1):
        ax.plot(xs, ys, color="0.75", lw=0.25, zorder=1)
    ax.add_patch(plt.Circle((0, 0), 50.0, fill=False, color="0.4",
                            lw=0.5, ls="--", zorder=2))
    ax.set_aspect("equal")
    ax.set_xlabel("$x/c$")
    ax.set_ylabel("$y/c$")
    ax.set_title("(a) O-mesh, far field $R = 50c$", pad=3)
    ax.set_xlim(-53, 53)
    ax.set_ylim(-53, 53)

    # (b) near field
    ax = axes[1]
    for xs, ys in elem_lines(m2.X, m2.Y, m2.Np1):
        ax.plot(xs, ys, color="0.75", lw=0.3, zorder=1)
    tc = np.linspace(0.0, 1.0, 400)
    yt = naca0012_y(tc)
    ax.plot(tc, yt, color=C_WARN, lw=1.4, zorder=4)
    ax.plot(tc, -yt, color=C_WARN, lw=1.4, zorder=4)
    ax.set_aspect("equal")
    ax.set_xlabel("$x/c$")
    ax.set_ylabel("$y/c$")
    ax.set_title("(b) near field (1176 elements, $N = 3$)", pad=3)
    ax.set_xlim(-0.6, 2.4)
    ax.set_ylim(-1.1, 1.1)

    # (c) refinement family at the trailing edge
    ax = axes[2]
    styles = {"L1": ("0.8", 0.4), "L2": ("0.62", 0.6), "L3": ("0.35", 0.8)}
    for name, kw in MESH_FAMILY.items():
        m = generate_airfoil_omesh(kw["n_half"], kw["n_r"], 3, 50.0,
                                   0.30 / kw["n_r"], wake_curve=0.05)
        col, lw = styles[name]
        for xs, ys in elem_lines(m.X, m.Y, m.Np1):
            ax.plot(xs, ys, color=col, lw=lw, zorder=1)
    ax.plot(tc, yt, color=C_WARN, lw=1.2, zorder=4)
    ax.plot(tc, -yt, color=C_WARN, lw=1.2, zorder=4)
    ax.set_aspect("equal")
    ax.set_xlabel("$x/c$")
    ax.set_ylabel("$y/c$")
    ax.set_title("(c) family $L1/L2/L3$ (ratio $\\approx 1.4$)", pad=3)
    ax.set_xlim(0.55, 1.75)
    ax.set_ylim(-0.45, 0.45)

    save(fig, "fig_mesh")
    rows = []
    for name, kw in MESH_FAMILY.items():
        m = generate_airfoil_omesh(kw["n_half"], kw["n_r"], 3, 50.0,
                                   0.30 / kw["n_r"], wake_curve=0.05)
        q = m.quality_report()
        idx, idy = m.metric_identities()
        rows.append([name, m.n_k * m.n_j, m.n_k * m.n_j * 16 * 4,
                     f"{q['min_J']:.2e}", f"{q['max_skew_cos']:.3f}",
                     f"{idx:.1e}", f"{idy:.1e}"])
    export_csv("fig_mesh", ["mesh", "elements", "dofs", "min_J",
                            "max_skew_cos", "metric_id_x", "metric_id_y"],
               rows)


# ======================================================================
# figure 2: verification suite
# ======================================================================
def fig_verification():
    V = json.loads((RES / "verification.json").read_text())
    fig, axes = plt.subplots(1, 4, figsize=(DOUBLE, 44 * MM),
                             constrained_layout=True)

    # (a) vortex h-convergence (N = 3, full period)
    ax = axes[0]
    vh = [r for r in V.get("vortex_full_period", []) if r["kind"] == "h"]
    if vh:
        h = np.array([10.0 / int(r["elems"].split("x")[0]) for r in vh])
        e = np.array([r["L2"] for r in vh])
        o = np.array([r["order"] for r in vh])
        ax.loglog(h, e, "o-", color=C_DG, ms=3.5, label="$N=3$ march")
        for i in range(1, len(h)):
            ax.annotate(f"{o[i]:.1f}", (h[i], e[i] * 1.7), fontsize=6,
                        ha="center", color=C_DG)
        ax.loglog(h, e[0] * (h / h[0]) ** 4.0, ":", color="0.5", lw=0.8,
                  label="$\\mathcal{O}(h^4)$ guide")
        ax.set_xlabel("$h$ ($L/n_e$, $L = 10$)")
        ax.set_ylabel("$\\|\\rho-\\rho_e\\|$")
        ax.set_xticks([0.05, 0.1, 0.2, 0.5, 1.0])
        ax.set_xticklabels(["0.05", "0.1", "0.2", "0.5", "1.0"])
        ax.minorticks_off()
        ax.legend(loc="lower right", handlelength=1.4)
    ax.set_title("(a) vortex advection", pad=3)

    # (b) vortex p-convergence (16 x 16)
    ax = axes[1]
    vp = [r for r in V.get("vortex_full_period", []) if r["kind"] == "p"]
    if vp:
        N = np.array([r["N"] for r in vp])
        e = np.array([r["L2"] for r in vp])
        ax.semilogy(N, e, "o-", color=C_OK, ms=3.5)
        for n, ee in zip(N, e):
            ax.annotate(f"{ee:.0e}", (n + 0.18, ee), fontsize=4.6,
                        ha="left", va="center", color=C_OK)
        ax.set_xlabel("polynomial degree $N$ (16$\\times$16)")
        ax.set_ylabel("$\\|\\rho-\\rho_e\\|$")
        ax.set_xticks(N)
        ax.set_xlim(0.4, 10.2)
    ax.set_title("(b) $p$-convergence", pad=3)

    # (c) diffusion MMS, marched
    ax = axes[2]
    mms = V.get("diffusion_mms_marched", [])
    for N, col in ((2, C_DG2), (3, C_DG), (4, C_OK)):
        rs = [r for r in mms if r["N"] == N]
        if not rs:
            continue
        ne = np.array([int(r["elems"].split("x")[0]) for r in rs])
        e = np.array([r["err"] for r in rs])
        o = np.array([r["order"] for r in rs])
        ax.loglog(ne, e, "o-", color=col, ms=3.5, label=f"$N={N}$")
        for i in range(1, len(ne)):
            ax.annotate(f"{o[i]:.1f}", (ne[i], e[i] * 1.8), fontsize=6,
                        ha="center", color=col)
    ax.set_xlabel("elements per direction")
    ax.set_ylabel("MMS error (marched)")
    ax.legend(loc="lower left", handlelength=1.4, framealpha=0.92)
    ax.set_title("(c) diffusion MMS", pad=3)

    # (d) Sod shock tube
    ax = axes[3]
    sod = V.get("sod", [])
    if sod:
        ne = np.array([r["elems"] for r in sod])
        e = np.array([r["L1_rho"] for r in sod])
        o = [r.get("L1_order") for r in sod]
        ax.loglog(ne, e, "s-", color=C_WARN, ms=4,
                  label="L1($\\rho$), AV + limiter")
        for i in range(1, len(ne)):
            if o[i]:
                ax.annotate(f"{o[i]:.1f}", (ne[i], e[i] * 1.6), fontsize=6,
                            ha="center", color=C_WARN)
        ax.set_xlabel("elements (N = 3)")
        ax.set_ylabel("L1 error")
        ax.legend(loc="lower left", handlelength=1.4)
    ax.set_title("(d) Sod tube", pad=3)

    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02,
                                    wspace=0.02, hspace=0.02)

    save(fig, "fig_verification")
    rows = []
    for r in V.get("vortex_full_period", []):
        o = r.get("order", float("nan"))
        rows.append(["vortex", r["kind"], r["N"], r["elems"],
                     f"{r['L2']:.6e}",
                     f"{o:.2f}" if o == o else ""])
    for r in mms:
        o = r.get("order", float("nan"))
        rows.append(["diffusion_mms", "", r["N"], r["elems"],
                     f"{r['err']:.6e}",
                     f"{o:.2f}" if o == o else ""])
    for r in sod:
        rows.append(["sod", "", 3, r["elems"], f"{r['L1_rho']:.6e}",
                     f"{r['L1_order']:.2f}" if r.get("L1_order") else ""])
    export_csv("fig_verification",
               ["test", "kind", "N", "elems", "error", "order"], rows)


# ======================================================================
# figure 3: the lift curve (headline validation figure)
# ======================================================================
def fig_cl_alpha():
    lad = {round(float(r["alpha_deg"]), 2):
           (float(r["CL"]), float(r["CD"]), float(r["CM"]))
           for r in load_ladson()}

    a_dg, cl_dg, cl_amp = [], [], []
    for a in LADSON_A:
        tag = f"sweep_a{a:+g}".replace("+", "p").replace("-", "m")
        if load_done(tag) is None:
            continue
        m, amp = cl_stats(tag)
        a_dg.append(a)
        cl_dg.append(m)
        cl_amp.append(amp)
    a_dg = np.array(a_dg, float)
    cl_dg = np.array(cl_dg)
    cl_amp = np.array(cl_amp)

    a_m, cl_m, clm_amp = [], [], []
    for a in LADSON_MIRROR:
        tag = f"sweep_a{a:+g}".replace("+", "p").replace("-", "m")
        if load_done(tag) is None:
            continue
        m, amp = cl_stats(tag)
        a_m.append(a)
        cl_m.append(m)
        clm_amp.append(amp)
    a_m = np.array(a_m, float)
    cl_m = np.array(cl_m)
    clm_amp = np.array(clm_amp)

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 62 * MM),
                             constrained_layout=True)

    # ---- (a) Cl(alpha) --------------------------------------------------
    ax = axes[0]
    ax.errorbar(a_dg, cl_dg, yerr=cl_amp, fmt="o", color=C_DG, ms=4,
                capsize=2, lw=1,
                label="DGSEM (Euler), $\\pm$ wake cycle")
    if len(a_m):
        ax.errorbar(a_m, cl_m, yerr=clm_amp, fmt="s", mfc="none",
                    color=C_MIR, ms=4, capsize=2, lw=1,
                    label="mirror ($-\\alpha$)")
    aexp = np.array(sorted(lad.keys()))
    clexp = np.array([lad[round(a, 2)][0] for a in aexp])
    ax.plot(aexp, clexp, "k^", ms=4, mfc="none",
            label="Ladson TM-4074 ($Re = 5.93\\times10^6$)")

    # linear fits (over the common range |alpha| <= 8.13)
    fit_dg = np.polyfit(a_dg[np.abs(a_dg) <= 8.13],
                        cl_dg[np.abs(a_dg) <= 8.13], 1)
    fit_ex = np.polyfit(aexp[np.abs(aexp) <= 8.13],
                        clexp[np.abs(aexp) <= 8.13], 1)
    xx = np.linspace(-10.5, 10.5, 10)
    ax.plot(xx, np.polyval(fit_dg, xx), color=C_DG, lw=0.8, ls="-")
    ax.plot(xx, np.polyval(fit_ex, xx), color="0.45", lw=0.8, ls="--")
    ax.annotate(f"slope DGSEM $= {fit_dg[0]:.4f}$/deg\n"
                f"slope exp. $= {fit_ex[0]:.4f}$/deg\n"
                f"ratio $= {fit_dg[0]/fit_ex[0]:.3f}$ (inviscid/viscous)",
                xy=(0.03, 0.97), xycoords="axes fraction", va="top",
                fontsize=7,
                bbox=dict(fc="white", ec="0.8", lw=0.5, pad=2.5))
    ax.set_xlabel("$\\alpha$ (deg)")
    ax.set_ylabel("$C_l$")
    ax.set_xlim(-11, 11)
    ax.set_xticks([-10, -7.5, -5, -2.5, 0, 2.5, 5, 7.5, 10])
    ax.axhline(0.0, color="0.85", lw=0.5, zorder=0)
    ax.legend(loc="lower right", handlelength=1.5, fontsize=6.0,
              framealpha=0.95)
    ax.set_title("(a) lift curve at the measured angles, $M = 0.30$",
                 pad=3)

    # ---- (b) Cm(alpha) --------------------------------------------------
    ax = axes[1]
    cm_dg, cm_amp = [], []
    for a in LADSON_A:
        tag = f"sweep_a{a:+g}".replace("+", "p").replace("-", "m")
        h = load_history(tag)
        if len(h["Cm"]) == 0:
            continue
        i0 = max(0, int(len(h["Cm"]) * 0.75))
        tail = h["Cm"][i0:]
        cm_dg.append(tail.mean())
        cm_amp.append(0.5 * (tail.max() - tail.min()))
    ax.errorbar(a_dg, cm_dg, yerr=cm_amp, fmt="o", color=C_DG, ms=4,
                capsize=2, lw=1, label="DGSEM (Euler)")
    cmexp = np.array([lad[round(a, 2)][2] for a in aexp])
    ax.plot(aexp, cmexp, "k^", ms=4, mfc="none", label="Ladson (viscous)")
    ax.axhline(0.0, color="0.85", lw=0.5, zorder=0)
    ax.set_xlabel("$\\alpha$ (deg)")
    ax.set_ylabel("$C_m$ (about $c/4$)")
    ax.set_xlim(-11, 11)
    ax.set_xticks([-10, -7.5, -5, -2.5, 0, 2.5, 5, 7.5, 10])
    ax.legend(loc="lower left", handlelength=1.5)
    ax.set_title("(b) quarter-chord moment", pad=3)

    save(fig, "fig_cl_alpha")
    rows = []
    for a, c, amp in zip(a_dg, cl_dg, cl_amp):
        tag = f"sweep_a{a:+g}".replace("+", "p").replace("-", "m")
        d = load_done(tag)
        e = lad.get(round(a, 2), (float("nan"),) * 3)
        rows.append(["DGSEM", a, f"{c:.5f}", f"{amp:.5f}",
                     f"{d['forces']['Cd']:.5f}", f"{d['steps']}",
                     f"{d['residual_orders']:.2f}", d["reason"]])
    for a, c, amp in zip(a_m, cl_m, clm_amp):
        rows.append(["DGSEM-mirror", a, f"{c:.5f}", f"{amp:.5f}",
                     "", "", "", ""])
    for a in aexp:
        cl, cd, cm = lad[round(a, 2)]
        rows.append(["experiment", a, f"{cl:.5f}", "", f"{cd:.5f}",
                     "", "", ""])
    export_csv("fig_cl_alpha",
               ["source", "alpha_deg", "Cl", "Cl_wake_amplitude", "Cd",
                "steps", "residual_orders", "reason"], rows)
    return fit_dg, fit_ex


# ======================================================================
# figure 4: surface pressure distributions
# ======================================================================
def fig_cp():
    angles = [0.03, 4.03, 10.12]
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE, 56 * MM),
                             constrained_layout=True)
    rows = []
    for ax, a in zip(axes, angles):
        tag = f"sweep_a{a:+g}".replace("+", "p").replace("-", "m")
        ck = load_ckpt(tag)
        if ck is None:
            ax.set_title(f"$\\alpha = {a}^\\circ$ (missing)", pad=3)
            continue
        U, X, Y, case = ck
        mesh = regen_mesh(case)
        cpd = surface_cp(U, mesh, case["Mach"], case["alpha_deg"], GAMMA)
        up, lo = cpd["upper"], cpd["lower"]
        ax.plot(up[:, 0], up[:, 1], color=C_DG, lw=1.1,
                label="upper surface")
        ax.plot(lo[:, 0], lo[:, 1], color=C_DG2, lw=1.1,
                label="lower surface")
        for x, c, surf in ((up[:, 0], up[:, 1], "upper"),
                           (lo[:, 0], lo[:, 1], "lower")):
            for xi, ci in zip(x, c):
                rows.append([a, surf, f"{xi:.6f}", f"{ci:.6f}"])
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-6.0, 1.1)
        ax.invert_yaxis()
        ax.set_xlabel("$x/c$")
        if ax is axes[0]:
            ax.set_ylabel("$c_p$")
            ax.legend(loc="lower right", handlelength=1.5)
        ax.set_title(f"$\\alpha = {a}^\\circ$", pad=3)
    save(fig, "fig_cp")
    export_csv("fig_cp", ["alpha_deg", "surface", "x/c", "Cp"], rows)


# ======================================================================
# figure 5: flow fields and the discrete wake sheet
# ======================================================================
def fig_fields():
    tag = "sweep_ap10.12"
    ck = load_ckpt(tag)
    if ck is None:
        tag = "sweep_ap8.13"
        ck = load_ckpt(tag)
    if ck is None:
        print("  [fig_fields] no sweep checkpoint available, skipped")
        return
    U, X, Y, case = ck
    rho, u, v, p = primitives(U, GAMMA)
    c = np.sqrt(GAMMA * p / rho)
    mach = np.hypot(u, v) / c
    qinf = 0.5 * case["Mach"] ** 2
    pinf = 1.0 / GAMMA
    cp = (p - pinf) / qinf

    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE, 60 * MM),
                             constrained_layout=True)
    tc = np.linspace(0.0, 1.0, 300)
    yt = naca0012_y(tc)

    # (a) Mach field, whole near field
    ax = axes[0]
    levels = np.linspace(0.0, 1.35, 28)
    for k in range(X.shape[0]):
        for j in range(X.shape[1]):
            ax.contourf(X[k, j].T, Y[k, j].T, mach[k, j].T, levels=levels,
                        cmap="viridis", extend="both")
    ax.fill_between(tc, -yt, yt, color="0.25", zorder=5)
    ax.set_aspect("equal")
    ax.set_xlim(-1.5, 4.5)
    ax.set_ylim(-2.2, 2.2)
    ax.set_xlabel("$x/c$")
    ax.set_ylabel("$y/c$")
    ax.set_title(f"(a) Mach, $\\alpha = {case['alpha_deg']:g}^\\circ$",
                 pad=3)

    # (b) Cp field near the airfoil
    ax = axes[1]
    levels = np.linspace(-3.2, 1.0, 28)
    for k in range(X.shape[0]):
        for j in range(X.shape[1]):
            ax.contourf(X[k, j].T, Y[k, j].T, cp[k, j].T, levels=levels,
                        cmap="RdBu_r", extend="both")
    ax.fill_between(tc, -yt, yt, color="0.9", zorder=5)
    ax.set_aspect("equal")
    ax.set_xlim(-0.5, 1.8)
    ax.set_ylim(-0.95, 0.95)
    ax.set_xlabel("$x/c$")
    ax.set_ylabel("$y/c$")
    ax.set_title("(b) $c_p$ field", pad=3)

    # (c) the wake sheet: entropy function s = ln(p/rho^gamma)
    ax = axes[2]
    s = np.log(p / rho ** GAMMA)
    lev = np.linspace(-0.12, 0.12, 25)
    for k in range(X.shape[0]):
        for j in range(0, X.shape[1]):
            ax.contourf(X[k, j].T, Y[k, j].T, s[k, j].T, levels=lev,
                        cmap="RdBu_r", extend="both")
    ax.fill_between(tc, -yt, yt, color="0.25", zorder=5)
    ax.set_aspect("equal")
    ax.set_xlim(0.7, 3.2)
    ax.set_ylim(-0.5, 0.5)
    ax.set_xlabel("$x/c$")
    ax.set_ylabel("$y/c$")
    ax.set_title("(c) wake sheet: $\\ln(p/\\rho^\\gamma)$", pad=3)

    save(fig, "fig_fields")
    rows = []
    j_mid = X.shape[1] // 2
    for k in range(X.shape[0]):
        for i in (0, X.shape[2] - 1):
            for l in range(X.shape[3]):
                rows.append([f"{X[k, j_mid, i, l]:.6f}",
                             f"{Y[k, j_mid, i, l]:.6f}",
                             f"{mach[k, j_mid, i, l]:.6f}",
                             f"{s[k, j_mid, i, l]:.6e}"])
    export_csv("fig_fields", ["x", "y", "Mach", "entropy"], rows)


# ======================================================================
# figure 6: mesh and p refinement studies
# ======================================================================
def fig_refinement():
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 58 * MM),
                             constrained_layout=True)
    rows = []

    # (a) mesh refinement (alpha = 5, N = 3)
    ax = axes[0]
    names, cl, amp, dof = [], [], [], []
    for name in ("L1", "L2", "L3"):
        tag = f"mesh_{name}_a5"
        d = load_done(tag)
        if d is None:
            continue
        m, a = cl_stats(tag)
        names.append(name)
        cl.append(m)
        amp.append(a)
        dof.append(d["case"]["n_half"] * 2 * d["case"]["n_r"] * 16 * 4)
        rows.append(["mesh", name, d["case"]["n_half"] * 2,
                     d["case"]["n_r"], f"{m:.5f}", f"{a:.5f}",
                     f"{d['forces']['Cl']:.5f}", f"{d['steps']}",
                     f"{d['residual_orders']:.2f}", d["reason"]])
    if names:
        dof = np.sqrt(np.array(dof, float))
        cl = np.array(cl)
        amp = np.array(amp)
        ax.errorbar(dof, cl, yerr=amp, fmt="o-", color=C_DG, ms=4,
                    capsize=2, label="$h$ family, $\\alpha = 5^\\circ$")
        for n, x, y in zip(names, dof, cl):
            ax.annotate(n, (x, y), xytext=(4, 5),
                        textcoords="offset points", fontsize=7)
    ax.set_xlabel("$\\sqrt{\\mathrm{DOF}}$")
    ax.set_ylabel("$C_l$")
    ax.legend(loc="best", handlelength=1.5)
    ax.set_title("(a) geometrically similar $h$ refinement", pad=3)

    # (b) p refinement (alpha = 5, L2 mesh)
    ax = axes[1]
    for tag, NN in (("p_N2_a5", 2), ("mesh_L2_a5", 3),
                    ("p_N4_a5", 4), ("p_N5_a5", 5)):
        d = load_done(tag)
        if d is None:
            continue
        m, a = cl_stats(tag)
        reason = d["reason"]
        col = C_WARN if reason == "diverged" else C_DG
        m_plt = m if reason != "diverged" else 0.0
        ax.errorbar([NN], [m_plt],
                    yerr=[a if reason != "diverged" else 0],
                    fmt="o-", color=col, ms=4, capsize=2,
                    label=("$p$ family, $\\alpha = 5^\\circ$"
                           if NN == 2 else None))
        if reason == "diverged":
            ax.annotate("diverged", (NN, 0.05), fontsize=6, color=C_WARN,
                        ha="center")
        rows.append(["p", NN, "", "", f"{m:.5f}", f"{a:.5f}",
                     f"{d['forces']['Cl']:.5f}", f"{d['steps']}",
                     f"{d['residual_orders']:.2f}", reason])
    ax.set_xlabel("polynomial degree $N$ ($L2$ mesh)")
    ax.set_ylabel("$C_l$")
    ax.set_xticks([2, 3, 4, 5])
    ax.legend(loc="best", handlelength=1.5)
    ax.set_title("(b) $p$ refinement; $N3 \\to N4/N5$ prolongated",
                 pad=3)

    save(fig, "fig_refinement")
    export_csv("fig_refinement",
               ["study", "label", "n_circ", "n_radial", "Cl_mean",
                "Cl_amplitude", "Cl_final_monitor", "steps",
                "residual_orders", "reason"], rows)


# ======================================================================
# figure 7: parameter sensitivity
# ======================================================================
def fig_sensitivity():
    base_tag = "mesh_L2_a5"
    d0 = load_done(base_tag)
    if d0 is None:
        print("  [fig_sensitivity] base case missing, skipped")
        return
    base, base_amp = cl_stats(base_tag)
    SENS = [("sens_eps01", "entropy fix 0.01"),
            ("sens_eps03", "entropy fix 0.03"),
            ("sens_R25", "far field $R = 25c$"),
            ("sens_R100", "far field $R = 100c$"),
            ("sens_novortex", "no vortex correction"),
            ("sens_rusanov", "Rusanov flux"),
            ("sens_sponge", "sponge 0.01")]
    labels, rel, rows = [], [], []
    for tag, lab in SENS:
        d = load_done(tag)
        if d is None:
            continue
        m, a = cl_stats(tag)
        labels.append(lab)
        rel.append(100.0 * (m - base) / abs(base))
        rows.append([lab, f"{m:.5f}", f"{a:.5f}",
                     f"{100.0*(m-base)/abs(base):+.3f}",
                     f"{d['reason']}", f"{d['steps']}"])
    if not labels:
        print("  [fig_sensitivity] no sensitivity cases finished")
        return

    fig, ax = plt.subplots(figsize=(SINGLE, 52 * MM),
                           constrained_layout=True)
    y = np.arange(len(labels))
    colors = [C_DG if abs(r) < 3.0 else C_WARN for r in rel]
    ax.barh(y, rel, color=colors, height=0.62)
    thr = 100.0 * base_amp / abs(base)
    ax.axvline(+thr, color="0.4", ls=":", lw=0.9)
    ax.axvline(-thr, color="0.4", ls=":", lw=0.9)
    ax.axvline(0.0, color="0.2", lw=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    for yi, r in zip(y, rel):
        ax.annotate(f"{r:+.2f}%", (r, yi), xytext=(3 if r >= 0 else -3, 0),
                    textcoords="offset points", va="center",
                    ha="left" if r >= 0 else "right", fontsize=6.5)
    ax.set_xlabel("relative $\\Delta C_l$ vs base configuration (\\%)")
    ax.set_title("Sensitivity at $\\alpha = 5^\\circ$; dotted lines: "
                 "$\\pm$ wake-cycle amplitude", pad=4)
    lo = min(min(rel) * 1.6, -thr * 1.8)
    hi = max(max(rel) * 1.6, thr * 1.8)
    ax.set_xlim(lo, hi)
    save(fig, "fig_sensitivity")
    export_csv("fig_sensitivity",
               ["variation", "Cl_mean", "Cl_amplitude", "rel_delta_pct",
                "reason", "steps"], rows)
    return base, base_amp, thr


# ======================================================================
# figure 8: convergence histories with the strict criteria
# ======================================================================
def fig_history():
    picks = ["sweep_ap0.03", "sweep_ap4.03", "sweep_ap10.12"]
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE, 52 * MM),
                             constrained_layout=True)
    rows = []
    for ax, tag in zip(axes, picks):
        h = load_history(tag)
        d = load_done(tag)
        if len(h["step"]) == 0:
            continue
        ax.semilogy(h["step"], h["residual"], color=C_DG, lw=0.9,
                    label="RMS $\\rho$ residual")
        ax.set_xlabel("time steps")
        ax.set_ylabel("residual")
        ax2 = ax.twinx()
        ax2.plot(h["step"], h["Cl"], color=C_WARN, lw=0.9, alpha=0.85)
        ax2.set_ylabel("$C_l$", color=C_WARN)
        ax2.tick_params(axis="y", colors=C_WARN)
        a = d["case"]["alpha_deg"] if d else float("nan")
        ax.set_title(f"$\\alpha = {a:g}^\\circ$ "
                     f"({d['reason']}, {d['residual_orders']:.1f} ord)",
                     pad=3)
        for s, r, cl in zip(h["step"], h["residual"], h["Cl"]):
            rows.append([tag, s, f"{r:.6e}", f"{cl:.6f}"])
    save(fig, "fig_history")
    export_csv("fig_history", ["case", "step", "residual", "Cl"], rows)


# ======================================================================
# summary table used by the paper
# ======================================================================
def write_summary():
    rows = []
    tags = ([f"sweep_a{a:+g}".replace("+", "p").replace("-", "m")
             for a in LADSON_A + LADSON_MIRROR]
            + ["mesh_L1_a5", "mesh_L2_a5", "mesh_L3_a5",
               "p_N2_a5", "p_N4_a5", "p_N5_a5"]
            + ["sens_eps01", "sens_eps03", "sens_R25", "sens_R100",
               "sens_novortex", "sens_rusanov", "sens_sponge"])
    for tag in tags:
        d = load_done(tag)
        if d is None:
            rows.append([tag, "", "", "", "", "", "", "", ""])
            continue
        m, amp = cl_stats(tag)
        rows.append([tag, f"{d['case']['alpha_deg']:g}",
                     f"{m:.5f}", f"{amp:.5f}",
                     f"{d['forces']['Cd']:+.5f}",
                     f"{d['forces']['Cm']:+.5f}",
                     f"{d['steps']}", f"{d['residual_orders']:.2f}",
                     d["reason"]])
    out = RES / "summary_forces.csv"
    with open(out, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["tag", "alpha_deg", "Cl_mean", "Cl_amplitude", "Cd",
                     "Cm", "steps", "residual_orders", "reason"])
        wr.writerows(rows)
    print(f"  wrote {out}")


# ======================================================================
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("which", nargs="*", default=["all"])
    args = p.parse_args()
    todo = args.which or ["all"]
    figs = {"mesh": fig_mesh, "verification": fig_verification,
            "cl_alpha": fig_cl_alpha, "cp": fig_cp, "fields": fig_fields,
            "refinement": fig_refinement, "sensitivity": fig_sensitivity,
            "history": fig_history, "summary": write_summary}
    for name in (todo if todo != ["all"] else list(figs)):
        if name in figs:
            print(f"[{name}]")
            figs[name]()
        else:
            print(f"unknown figure '{name}' (known: {list(figs)})")
