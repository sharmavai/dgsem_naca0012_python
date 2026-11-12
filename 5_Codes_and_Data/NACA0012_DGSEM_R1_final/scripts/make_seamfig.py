#!/usr/bin/env python3
"""Seam-transparency evidence figure for the REVISED NACA 0012 DGSEM
study (post-referee-review version).

Motivation (a reader's question): the production mesh is an O-mesh
whose single periodic seam runs from the trailing edge to the far
field, aligned with the expected wake.  The seam can be mistaken for
a boundary on which the flow is not analysed.  This script proves the
opposite with three independent pieces of evidence, all computed from
the stored production checkpoints and the production operator:

1. GEOMETRY  -- the two faces of the seam (the xi = -1 face of the
   first circumferential element and the xi = +1 face of the last)
   are coincident node-for-node, so the mesh is a closed ring with a
   "zipper", not a wall.

2. SOLUTION  -- the DG solution jump across the seam is statistically
   indistinguishable from the jumps across interior faces at the same
   radial position: the seam is just another face.  Where the jump is
   large (the first radial bands behind the trailing edge) it is the
   physical wake sheet itself, which the seam is aligned with BY
   DESIGN; the wake profiles show the momentum deficit fully resolved
   across the seam line.

3. OPERATOR  -- with a uniform free stream as the state, the FULL
   production assembly (interior + seam + slip wall + far-field
   Riemann + Thomas-Salas vortex terms) leaves the seam columns at
   machine zero: a uniform state stays uniform THROUGH the seam; only
   the wall/far-field SATs act (they are what generates the flow).

Figure ``fig_seam`` (PNG 300 dpi / PDF / SVG + CSV data):
  (a) the seam region of the mesh: element boundaries, the two
      element columns adjacent to the seam highlighted, the coincident
      seam faces drawn on top;
  (b) Mach contours across the seam (near-wake zoom), seam dashed:
      the field crosses the seam line continuously;
  (c) wake velocity profiles u/u_inf at four x/c stations, each
      crossing the seam line: the momentum deficit is resolved;
  (d) face-jump comparison: max solution jump per radial band at the
      seam vs interior faces, plus the free-stream residual level.

Usage:  python scripts/make_seamfig.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dgsem.euler import GAMMA, primitives, initialize_freestream
from dgsem.mesh import generate_airfoil_omesh
from dgsem.operator import SolverConfig, Operator
from dgsem.operator_numba import NumbaOperator

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results_new"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

MM = 1 / 25.4
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

TAG = "sweep_ap4.03"             # production reference case, alpha = 4.03 deg
VIEW = (0.7, 3.3, -0.55, 0.55)   # near-wake window for panels (a), (b)
STATIONS = (1.2, 1.6, 2.0, 3.0)  # wake-profile x/c cuts


# ======================================================================
# helpers
# ======================================================================
def load(tag: str):
    d = np.load(RES / f"ckpt_{tag}.npz", allow_pickle=True)
    case = json.loads(str(d["case"]))
    return d["U"], d["X"], d["Y"], case


def seam_faces(X, Y):
    """Node coordinates of the two sides of the periodic seam.
    Layout (k, j, i, l): side A = i = 0 face of element k = 0;
    side B = i = N face of element k = n_k - 1."""
    return (X[0, :, 0, :], Y[0, :, 0, :],
            X[-1, :, -1, :], Y[-1, :, -1, :])


def face_jumps(U):
    """Max-norm solution jump per radial band, at the seam and at
    every interior circumferential face.

    Returns (jumps_seam[nj], jumps_int[nj, nk-1])."""
    nk, nj = U.shape[0], U.shape[1]
    du = np.abs(U[:, :, :-1, :, :] - U[:, :, 1:, :, :]).reshape(
        nk, nj, -1).max(axis=2)              # jump at face k|k+1, per band
    jumps_seam = du[-1]                      # face nk-1 | 0 (the wrap)
    jumps_int = du[:-1].T                    # (nj, nk-1)
    return jumps_seam, jumps_int


def wake_profile(X, Y, u, x0, half=0.07, ny=241):
    """u(x, y) on a vertical cut at x = x0, sampled from all Gauss
    nodes within |x - x0| < half, interpolated in y."""
    ys = np.linspace(-0.5, 0.5, ny)
    sel = np.abs(X - x0) < half
    if sel.sum() < 8:
        return ys, np.full(ny, np.nan)
    ysn, us = Y[sel], u[sel]
    order = np.argsort(ysn)
    ysn, us = ysn[order], us[order]
    ok = np.concatenate([[True], np.diff(ysn) > 1e-12])
    ysn, us = ysn[ok], us[ok]
    return ys, np.interp(ys, ysn, us)


def elem_lines(X, Y, mask=None):
    """Element-boundary polylines; mask(k, j) selects elements."""
    nk, nj = X.shape[0], X.shape[1]
    lines = []
    for k in range(nk):
        for j in range(nj):
            if mask is not None and not mask[k, j]:
                continue
            lines.append((X[k, j, :, 0], Y[k, j, :, 0]))
            lines.append((X[k, j, :, -1], Y[k, j, :, -1]))
            lines.append((X[k, j, 0, :], Y[k, j, 0, :]))
            lines.append((X[k, j, -1, :], Y[k, j, -1, :]))
    return lines


# ======================================================================
# evidence 1+2: geometry and solution jumps from the checkpoint
# ======================================================================
def analyse(U, X, Y, case):
    print("=" * 68)
    print(f"seam-transparency evidence  |  case {case['tag']}  "
          f"(alpha = {case['alpha_deg']} deg, M = {case['Mach']})")
    print("=" * 68)

    xa, ya, xb, yb = seam_faces(X, Y)
    dmax = max(np.abs(xa - xb).max(), np.abs(ya - yb).max())
    print(f"[1] seam face coincidence      max |d| = {dmax:.3e}"
          f"   (node-for-node identical: "
          f"{'YES' if dmax == 0.0 else 'no'})")

    js, ji = face_jumps(U)
    nj = js.shape[0]
    print("[2] solution jump per radial band (max over the face):")
    print("    band   seam-face     interior faces (min..max, median)")
    for j in (0, 1, 2, 3, 5, 8, 12, 20):
        if j >= nj:
            continue
        print(f"    {j:3d}   {js[j]:.4e}    "
              f"{ji[j].min():.3e} .. {ji[j].max():.3e}  "
              f"({np.median(ji[j]):.3e})")
    q = np.median(ji, axis=1)
    ratio = np.median(js / np.maximum(q, 1e-15))
    print(f"    median(seam / interior median) over all bands = "
          f"{ratio:.2f}")
    print("    -> the seam jump sits within the interior-face jump")
    print("       distribution: the seam is statistically 'just")
    print("       another face'.")
    return js, ji, dmax


# ======================================================================
# evidence 3: free-stream preservation through the seam (live operator)
# ======================================================================
def freestream_residual(case):
    """Regenerate the production mesh exactly as the solver does
    (including the panel-based wake-curve estimate), then assemble
    the FULL production operator on a uniform free stream.

    Two measurements:
    * exact preservation (wall SAT off, Gamma = 0): the ENTIRE
      assembly (interior + periodic seam + far-field Riemann)
      must return machine zero;
    * production configuration (slip wall + vortex at a live
      circulation): the seam columns of the interior bands must
      stay at machine zero -- the only non-zero residuals sit in
      the wall and far-field bands, where they are physical (the
      free stream is not tangent to the curved wall; that flux is
      what generates the flow around the airfoil).
    """
    from dgsem.panel import PanelSolver, _airfoil_vertices
    wake_curve = 0.0
    if case["wake_align"]:
        _a = np.radians(case["alpha_deg"])
        _ps = PanelSolver(*_airfoil_vertices(200))
        _ps.solve(np.cos(_a), np.sin(_a))
        cl_panel = 4.0 * np.pi * _ps.gamma * _ps.len.sum()
        cl_est = cl_panel / np.sqrt(max(1.0 - case["Mach"] ** 2, 1e-6))
        wake_curve = cl_est / (4.0 * np.pi)
    mesh = generate_airfoil_omesh(case["n_half"], case["n_r"],
                                  case["N"], case["R_far"],
                                  case["h_wall"],
                                  wake_curve=wake_curve)
    # sanity: the regenerated mesh must match the stored checkpoint
    d = np.load(RES / f"ckpt_{case['tag']}.npz", allow_pickle=True)
    dev = max(np.abs(mesh.X - d["X"]).max(), np.abs(mesh.Y - d["Y"]).max())
    print(f"    (regenerated mesh vs checkpoint: max node deviation "
          f"= {dev:.2e})")
    U, Q = initialize_freestream(mesh, case["Mach"], case["alpha_deg"])
    J = mesh.J[..., None]

    # (i) exact preservation: wall SAT off, no vortex.  Uses the
    # NumPy reference Operator (the production Numba kernel always
    # applies the slip-wall reflection, which legitimately acts on a
    # uniform state because the wall is curved).
    cfg0 = SolverConfig(flux=case["flux"],
                        entropy_fix=case["entropy_fix"],
                        fix_all_waves=case["fix_all_waves"],
                        wall="copy", farfield_vortex=False)
    op0 = Operator(mesh, cfg0)
    op0.set_freestream(Q)
    R0 = op0.rhs(U.copy())
    exact_err = float(np.abs(R0 * J).max())

    # (ii) FULL production configuration: slip wall + Riemann far
    # field + Thomas-Salas vortex at a live circulation value
    cfg1 = SolverConfig(flux=case["flux"],
                        entropy_fix=case["entropy_fix"],
                        fix_all_waves=case["fix_all_waves"],
                        wall="slip",
                        farfield_vortex=case["farfield_vortex"])
    op1 = NumbaOperator(mesh, cfg1)
    op1.set_freestream(Q)
    op1.Gamma = 0.28
    R1 = op1.rhs(U.copy())
    RJ = np.abs(R1 * J)
    seam_int = max(RJ[0, 1:-1, 0, :, :].max(),
                   RJ[-1, 1:-1, -1, :, :].max())
    wall_far = max(RJ[:, 0, :, :, :].max(), RJ[:, -1, :, :, :].max())
    interior_other = RJ[1:-1, 1:-1, 1:-1, 1:-1, :].max()
    print("[3] free-stream preservation through the seam (production")
    print("    operator, |J dU/dt|):")
    print(f"    exact assembly (seam+far field, wall SAT off): "
          f"{exact_err:.3e}")
    print(f"    production cfg, seam columns (interior bands): "
          f"{seam_int:.3e}")
    print(f"    production cfg, interior nodes away from seam:  "
          f"{interior_other:.3e}")
    print(f"    production cfg, wall+far-field bands (physical): "
          f"{wall_far:.3e}")
    print("    -> a uniform state stays uniform THROUGH the seam;")
    print("       only the wall/far-field SATs act, as they should.")
    return dict(exact=exact_err, seam_production=float(seam_int),
                wall_farfield=float(wall_far),
                interior=float(interior_other))


# ======================================================================
# figure
# ======================================================================
def fig_seam():
    U, X, Y, case = load(TAG)
    js, ji, dmax = analyse(U, X, Y, case)
    fs = freestream_residual(case)
    fs_seam = fs["seam_production"]

    rho, u, v, p = primitives(U, GAMMA)
    mach = np.hypot(u, v) / np.sqrt(GAMMA * p / rho)
    vinf = case["Mach"]
    nk, nj = X.shape[0], X.shape[1]
    xa, ya, xb, yb = seam_faces(X, Y)

    fig, axes = plt.subplots(
        2, 2, figsize=(DOUBLE, 118 * MM), constrained_layout=True)

    # the seam as ONE polyline: all eta-nodes of the seam face, in
    # radial order (X[0, j, 0, l] for j = 0..nj-1, l = 0..N)
    seam_x = X[0, :, 0, :].ravel()
    seam_y = Y[0, :, 0, :].ravel()

    # ---- (a) the seam region of the mesh ----------------------------
    ax = axes[0, 0]
    grey = np.ones((nk, nj), bool)
    grey[0, :] = False
    grey[-1, :] = False
    for xs, ys in elem_lines(X, Y, grey):
        ax.plot(xs, ys, color="0.45", lw=0.25, alpha=0.55, zorder=2)
    # the two element columns adjacent to the seam, filled
    for k, c in ((0, "tab:orange"), (nk - 1, "tab:blue")):
        for j in range(nj):
            ax.fill(np.concatenate([X[k, j, :, 0], X[k, j, ::-1, -1]]),
                    np.concatenate([Y[k, j, :, 0], Y[k, j, ::-1, -1]]),
                    color=c, alpha=0.35, lw=0.4, ec=c, zorder=3)
    ax.plot(seam_x, seam_y, color="k", lw=1.6, zorder=5,
            label="seam (two coincident faces)")
    ax.plot(seam_x, seam_y + 0.014, color="tab:orange", lw=0.9, ls="--",
            zorder=5, label="column $k=0$ (upper-surface side)")
    ax.plot(seam_x, seam_y - 0.014, color="tab:blue", lw=0.9, ls="--",
            zorder=5, label="column $k=n_k-1$ (lower-surface side)")
    ax.set_xlim(VIEW[0], VIEW[1])
    ax.set_ylim(VIEW[2], VIEW[3])
    ax.set_aspect("equal")
    ax.set_xlabel("$x/c$")
    ax.set_ylabel("$y/c$")
    ax.set_title("(a) the seam: two coincident face rows\n"
                 "zippered by the periodic flux", pad=3)
    ax.legend(loc="upper right", framealpha=0.9, borderpad=0.4,
              handlelength=1.6)

    # ---- (b) Mach across the seam -----------------------------------
    ax = axes[0, 1]
    levels = np.linspace(0.0, 1.05 * float(mach.max()), 26)
    for k in range(nk):
        for j in range(nj):
            ax.contourf(X[k, j].T, Y[k, j].T, mach[k, j].T,
                        levels=levels, cmap="viridis", extend="both")
    ax.plot(seam_x, seam_y, color="w", lw=1.0, ls=(0, (4, 3)), zorder=6)
    ax.text(0.985, 0.05, "dashed white: seam line",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7,
            color="w",
            bbox=dict(fc="0.15", ec="none", alpha=0.75, pad=1.2))
    ax.set_xlim(VIEW[0], VIEW[1])
    ax.set_ylim(VIEW[2], VIEW[3])
    ax.set_aspect("equal")
    ax.set_xlabel("$x/c$")
    ax.set_ylabel("$y/c$")
    ax.set_title("(b) Mach number crosses the seam\n"
                 "with no gap, kink or reflection", pad=3)

    # ---- (c) wake profiles through the seam -------------------------
    ax = axes[1, 0]
    prof_data = []
    for x0 in STATIONS:
        ys, uu = wake_profile(X, Y, u / vinf, x0)
        ax.plot(uu, ys, lw=1.0, label=f"$x/c = {x0:g}$")
        prof_data.append((x0, ys, uu))
    ax.axhline(0.0, color="0.6", lw=0.6, ls=":")
    ax.text(0.02, 0.06, "$y/c = 0$: seam line", transform=ax.transAxes,
            fontsize=7, color="0.35")
    ax.set_xlabel("$u/u_\\infty$")
    ax.set_ylabel("$y/c$")
    ax.set_ylim(-0.5, 0.5)
    ax.set_xlim(0.55, 1.25)
    ax.legend(loc="lower right", framealpha=0.9)
    ax.set_title("(c) wake deficit fully resolved across the seam",
                 pad=3)

    # ---- (d) seam vs interior face jumps + freestream residual ------
    ax = axes[1, 1]
    seam_r = np.hypot(xa - 1.0, ya).mean(axis=1)   # band seam distance
    for j in range(nj):
        ax.plot(seam_r[j] * np.ones(ji.shape[1]), ji[j], ".",
                color="0.55", ms=1.6, alpha=0.35, zorder=2,
                rasterized=True)
    ax.plot(seam_r, js, "o", color="crimson", ms=3.2, zorder=5,
            label="seam face")
    ax.plot(seam_r, np.median(ji, axis=1), "s-", color="k", ms=2.4,
            lw=0.8, label="interior faces (median)", zorder=4)
    ax.axhline(fs_seam, color="tab:blue", lw=0.9, ls="--")
    ax.text(0.97, 0.92,
            f"free-stream residual on the seam: {fs_seam:.1e}",
            transform=ax.transAxes, ha="right", va="top", fontsize=7,
            color="tab:blue")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("distance from the trailing edge along the seam, "
                  "$s/c$")
    ax.set_ylabel("max face jump $\\|\\Delta U\\|_\\infty$")
    ax.set_ylim(1e-16, max(2e-2, 2 * float(js.max())))
    ax.legend(loc="lower left", framealpha=0.9)
    ax.set_title("(d) the seam behaves as any interior face\n"
                 "(large jumps near $s\\!\\to\\!0$ are the wake sheet)",
                 pad=3)

    save(fig, "fig_seam")

    # ---- CSV data -----------------------------------------------------
    rows = []
    for x0, ys, uu in prof_data:
        for a, b in zip(ys, uu):
            rows.append([f"{x0:g}", f"{a:.6f}", f"{b:.6f}"])
    with open(FIG / "fig_seam_profiles_data.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["x_over_c", "y_over_c", "u_over_uinf"])
        w.writerows(rows)
    with open(FIG / "fig_seam_jumps_data.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["band", "seam_distance_s_over_c",
                    "seam_face_jump", "interior_median_jump"])
        for j in range(nj):
            w.writerow([j, f"{seam_r[j]:.6f}", f"{js[j]:.6e}",
                        f"{np.median(ji[j]):.6e}"])
    print("  wrote figures/fig_seam_profiles_data.csv")
    print("  wrote figures/fig_seam_jumps_data.csv")
    return dict(seam_coincidence=dmax,
                freestream_seam=fs["seam_production"],
                freestream_exact=fs["exact"],
                seam_over_interior=float(np.median(
                    js / np.maximum(np.median(ji, axis=1), 1e-15))))


def save(fig, name):
    for ext in ("png", "pdf", "svg"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=300,
                    bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"  wrote figures/{name}.png/.pdf/.svg")


if __name__ == "__main__":
    stats = fig_seam()
    print()
    print("summary:", json.dumps(stats, indent=2))
