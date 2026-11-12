#!/usr/bin/env python3
"""Flow-field post-production figures for the REVISED NACA 0012 DGSEM
study (post-referee-review version).

This set complements ``make_figures.py`` with the full flow-field
post-production survey requested for the revision: the density,
static pressure, Mach number and velocity fields rendered directly
from the stored Gauss-node solution (the honest DGSEM picture: each
element is contoured individually, so inter-element jumps stay
visible), streamline patterns, the vorticity of the discrete wake
sheet, and element-resolved "flow on the mesh" views for the mesh-
and polynomial-refinement solutions.

Figures (each written as PNG 300 dpi / PDF vector / SVG editable,
with the plotted data exported to CSV alongside):

* ``fig_flow_density``  -- rho/rho_inf at alpha = 0.03, 4.03, 10.12 deg
* ``fig_flow_pressure`` -- p/p_inf     at the same three angles
* ``fig_flow_mach``     -- Mach number at the same three angles
                           (with the sonic line where it exists)
* ``fig_flow_velocity`` -- |V|/V_inf + streamlines, same three angles
* ``fig_flow_wake``     -- wake-sheet vorticity omega_c c/V_inf,
                           alpha = 4.03 and 10.12 deg, near wake
* ``fig_flow_mesh``     -- the solution ON the mesh: Mach with the
                           element boundaries overlaid for the
                           production L2 mesh, the fine L3 mesh and
                           the p-refined N = 5 solution (all alpha = 5)

Usage:  python scripts/make_flowfields.py [--only name,name,...]
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.ticker import MaxNLocator
from scipy.interpolate import griddata

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dgsem.euler import GAMMA, primitives
from dgsem.geometry import naca0012_thickness

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

# the three Ladson angles shown in the survey (deg)
SURVEY_ANGLES = [("sweep_ap0.03", 0.03), ("sweep_ap4.03", 4.03),
                 ("sweep_ap10.12", 10.12)]
# full near-field window (chords)
FULL_VIEW = (-1.5, 4.5, -2.2, 2.2)
# near-wake window for the vorticity figure
WAKE_VIEW = (0.6, 3.4, -0.7, 0.7)
# near-airfoil window for the on-the-mesh figure
MESH_VIEW = (-0.6, 1.8, -1.0, 1.0)


# ======================================================================
# helpers
# ======================================================================
def naca0012_y(t):
    """Exact NACA 0012 half-thickness (t in [0, 1]).

    Delegates to the solver's own surface geometry (closed trailing
    edge) so the silhouette drawn in every figure is EXACTLY the meshed
    wall.  (A previous revision used a garbled thickness formula here
    that drew a body ~15x too thick and masked the near field.)
    """
    return naca0012_thickness(t, t=0.12, closed=True)


_TC = np.linspace(0.0, 1.0, 400)
_YT = naca0012_y(_TC)
_AIRFOIL_POLY = MplPath(np.column_stack([
    np.concatenate([_TC, _TC[::-1]]),
    np.concatenate([_YT, -_YT[::-1]])]))


def _buffered_airfoil(grow_y=1.10, pad=0.006, pad_x=0.012):
    """Airfoil polygon inflated by a small margin, used to keep
    streamlines / vorticity samples a little away from the wall."""
    tc = np.linspace(-pad_x, 1.0 + pad_x, 400)
    yt = naca0012_y(np.clip(tc, 0.0, 1.0)) * grow_y + pad
    return MplPath(np.column_stack([
        np.concatenate([tc, tc[::-1]]),
        np.concatenate([yt, -yt[::-1]])]))


# small buffer: streamlines stop just outside the wall
_AIRFOIL_BUFF_S = _buffered_airfoil(1.08, 0.004, 0.008)
# larger buffer: hides the (physical) wall vorticity sheet in the
# wake figure so the discrete wake sheet itself stays the focus
_AIRFOIL_BUFF_L = _buffered_airfoil(1.35, 0.045, 0.05)


def load_ckpt(tag: str):
    f = RES / f"ckpt_{tag}.npz"
    if not f.exists():
        return None
    d = np.load(f, allow_pickle=True)
    case = json.loads(str(d["case"]))
    return d["U"], d["X"], d["Y"], case


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


def draw_airfoil(ax, color="0.25", zorder=5, edge="w", lw=0.7):
    ax.fill_between(_TC, -_YT, _YT, color=color, zorder=zorder,
                    edgecolor=edge, linewidth=lw)


def style_colorbar(cb, label):
    """Uniform colorbar style: small ticks, clean round numbers."""
    cb.set_label(label)
    cb.ax.tick_params(labelsize=6.0, width=0.5, length=2.2, pad=1.5)
    # the levels come from a contour, so matplotlib would otherwise place one
    # tick per level and the labels collide on a short colorbar; use four
    # evenly spaced ticks instead
    lo, hi = float(cb.vmin), float(cb.vmax)
    ticks = np.linspace(lo, hi, 4)
    cb.set_ticks(list(ticks))
    cb.set_ticklabels([f"{v:.2f}" for v in ticks])


def set_view(ax, view, xlabel="$x/c$", ylabel="$y/c$"):
    ax.set_aspect("equal")
    ax.set_xlim(view[0], view[1])
    ax.set_ylim(view[2], view[3])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def draw_field(ax, X, Y, f, levels, cmap, view=FULL_VIEW,
               iso_lines=None, airfoil_color="0.25"):
    """Per-element filled contours of a DG nodal field: each element is
    contoured on its own Gauss-node grid, so the representation on the
    mesh (including inter-element jumps) is shown as it is."""
    cs = None
    for k in range(X.shape[0]):
        for j in range(X.shape[1]):
            cs = ax.contourf(X[k, j].T, Y[k, j].T, f[k, j].T,
                             levels=levels, cmap=cmap, extend="both")
    if iso_lines:
        for k in range(X.shape[0]):
            for j in range(X.shape[1]):
                try:
                    ax.contour(X[k, j].T, Y[k, j].T, f[k, j].T,
                               levels=iso_lines, colors="w",
                               linewidths=0.55)
                except ValueError:
                    pass
    draw_airfoil(ax, color=airfoil_color)
    set_view(ax, view)
    return cs


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


def interp_grid(X, Y, fields, view, nx=281, ny=181,
                mask_poly=None):
    """Linear interpolation of nodal fields onto a regular Cartesian
    grid (used for streamlines and vorticity only).  Cells inside the
    mask polygon (the airfoil, optionally buffered) are NaN."""
    xg = np.linspace(view[0], view[1], nx)
    yg = np.linspace(view[2], view[3], ny)
    gx, gy = np.meshgrid(xg, yg)
    pts = np.column_stack([X.ravel(), Y.ravel()])
    out = []
    for f in fields:
        out.append(griddata(pts, f.ravel(), (gx, gy), method="linear"))
    poly = mask_poly if mask_poly is not None else _AIRFOIL_POLY
    inside = poly.contains_points(
        np.column_stack([gx.ravel(), gy.ravel()])).reshape(gx.shape)
    for a in out:
        a[inside] = np.nan
    return xg, yg, out


def panel_letter(i):
    return "abcdefghij"[i]


def nodal_rows(X, Y, vals, view=None, fmt="{:.6e}"):
    """CSV rows (x, y, v...) for all nodes, optionally clipped to a
    window."""
    x, y = X.ravel(), Y.ravel()
    keep = np.ones(x.shape, bool) if view is None else (
        (x >= view[0]) & (x <= view[1]) & (y >= view[2]) & (y <= view[3]))
    rows = []
    for xi, yi, vi in zip(x[keep], y[keep],
                          np.stack([v.ravel() for v in vals], 1)[keep]):
        rows.append([f"{xi:.6f}", f"{yi:.6f}"]
                    + [fmt.format(v) for v in vi])
    return rows


# ======================================================================
# figure: density survey
# ======================================================================
def fig_flow_density():
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE, 62 * MM),
                             constrained_layout=True)
    for i, (tag, alpha) in enumerate(SURVEY_ANGLES):
        ax = axes[i]
        U, X, Y, case = load_ckpt(tag)
        rho, _, _, _ = primitives(U, GAMMA)
        f = rho / 1.0                      # rho_inf = 1
        lo, hi = float(f.min()), float(f.max())
        if hi - lo < 1e-9:
            lo, hi = 1.0 - 1e-9, 1.0 + 1e-9
        levels = np.linspace(lo, hi, 26)
        cs = draw_field(ax, X, Y, f, levels, "viridis")
        style_colorbar(fig.colorbar(cs, ax=ax, shrink=0.92, pad=0.02,
                                    aspect=24),
                       r"$\rho/\rho_\infty$")
        ax.set_title(f"({panel_letter(i)}) $\\alpha = {alpha:g}^\\circ$", pad=2,
                     loc="left", fontsize=7.5)
        if tag == SURVEY_ANGLES[-1][0]:
            export_csv("fig_flow_density",
                       ["x", "y", "rho_over_rhoinf"],
                       nodal_rows(X, Y, [f], fmt="{:.6f}"))
    save(fig, "fig_flow_density")


# ======================================================================
# figure: static pressure survey
# ======================================================================
def fig_flow_pressure():
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE, 62 * MM),
                             constrained_layout=True)
    pinf = 1.0 / GAMMA
    for i, (tag, alpha) in enumerate(SURVEY_ANGLES):
        ax = axes[i]
        U, X, Y, case = load_ckpt(tag)
        _, _, _, p = primitives(U, GAMMA)
        f = p / pinf
        L = 1.05 * float(np.abs(f - 1.0).max())
        if L < 1e-9:
            L = 1e-9
        levels = np.linspace(1.0 - L, 1.0 + L, 25)
        cs = draw_field(ax, X, Y, f, levels, "RdBu_r")
        style_colorbar(fig.colorbar(cs, ax=ax, shrink=0.92, pad=0.02,
                                    aspect=24),
                       r"$p/p_\infty$")
        ax.set_title(f"({panel_letter(i)}) $\\alpha = {alpha:g}^\\circ$", pad=2,
                     loc="left", fontsize=7.5)
        if tag == SURVEY_ANGLES[-1][0]:
            export_csv("fig_flow_pressure",
                       ["x", "y", "p_over_pinf"],
                       nodal_rows(X, Y, [f], fmt="{:.6f}"))
    save(fig, "fig_flow_pressure")


# ======================================================================
# figure: Mach survey (with the sonic line where it exists)
# ======================================================================
def fig_flow_mach():
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE, 62 * MM),
                             constrained_layout=True)
    for i, (tag, alpha) in enumerate(SURVEY_ANGLES):
        ax = axes[i]
        U, X, Y, case = load_ckpt(tag)
        rho, u, v, p = primitives(U, GAMMA)
        mach = np.hypot(u, v) / np.sqrt(GAMMA * p / rho)
        hi = 1.03 * float(mach.max())
        levels = np.linspace(0.0, hi, 26)
        iso = [1.0] if mach.max() > 1.0 else None
        cs = draw_field(ax, X, Y, mach, levels, "viridis",
                        iso_lines=iso)
        style_colorbar(fig.colorbar(cs, ax=ax, shrink=0.92, pad=0.02,
                                    aspect=24), "$M$")
        ax.set_title(f"({panel_letter(i)}) $\\alpha = {alpha:g}^\\circ$", pad=2,
                     loc="left", fontsize=7.5)
        if iso and i == 2:
            ax.text(0.98, 0.97, "white: $M=1$",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=7,
                    bbox=dict(fc="white", ec="none", alpha=0.75,
                              pad=1.2))
        if tag == SURVEY_ANGLES[-1][0]:
            export_csv("fig_flow_mach", ["x", "y", "Mach"],
                       nodal_rows(X, Y, [mach], fmt="{:.6f}"))
    save(fig, "fig_flow_mach")


# ======================================================================
# figure: velocity magnitude + streamlines
# ======================================================================
def fig_flow_velocity():
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE, 62 * MM),
                             constrained_layout=True)
    for i, (tag, alpha) in enumerate(SURVEY_ANGLES):
        ax = axes[i]
        U, X, Y, case = load_ckpt(tag)
        _, u, v, _ = primitives(U, GAMMA)
        vinf = case["Mach"]                     # a_inf = 1
        spd = np.hypot(u, v) / vinf
        hi = 1.03 * float(spd.max())
        levels = np.linspace(0.0, hi, 26)
        cs = draw_field(ax, X, Y, spd, levels, "plasma")
        # streamlines from the linearly interpolated field
        # (buffered wall mask: lines stop just outside the surface)
        xg, yg, (ug, vg) = interp_grid(X, Y, [u / vinf, v / vinf],
                                       FULL_VIEW, nx=241, ny=161,
                                       mask_poly=_AIRFOIL_BUFF_S)
        s = 2
        ax.streamplot(xg[::s], yg[::s], ug[::s, ::s], vg[::s, ::s],
                      color="0.10", linewidth=0.40, density=1.1,
                      arrowsize=0.7, arrowstyle="-|>",
                      zorder=6)
        draw_airfoil(ax, color="0.10", zorder=7)
        style_colorbar(fig.colorbar(cs, ax=ax, shrink=0.92, pad=0.02,
                                    aspect=24),
                       r"$|\mathbf{V}|/V_\infty$")
        ax.set_title(f"({panel_letter(i)}) $\\alpha = {alpha:g}^\\circ$", pad=2,
                     loc="left", fontsize=7.5)
        if tag == SURVEY_ANGLES[-1][0]:
            export_csv("fig_flow_velocity",
                       ["x", "y", "vel_over_vinf"],
                       nodal_rows(X, Y, [spd], fmt="{:.6f}"))
    save(fig, "fig_flow_velocity")


# ======================================================================
# figure: wake-sheet vorticity (near wake)
# ======================================================================
def fig_flow_wake():
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 60 * MM),
                             constrained_layout=True)
    tags = [("sweep_ap4.03", 4.03), ("sweep_ap10.12", 10.12)]
    for i, (tag, alpha) in enumerate(tags):
        ax = axes[i]
        U, X, Y, case = load_ckpt(tag)
        _, u, v, _ = primitives(U, GAMMA)
        vinf = case["Mach"]
        xg, yg, (ug, vg) = interp_grid(X, Y, [u, v], WAKE_VIEW,
                                       nx=361, ny=181,
                                       mask_poly=_AIRFOIL_BUFF_L)
        dx = xg[1] - xg[0]
        dy = yg[1] - yg[0]
        om = (np.gradient(vg, dx, axis=1)
              - np.gradient(ug, dy, axis=0)) / vinf
        # robust colour scale from the WAKE region only (x > 1.05):
        # the wall vortex sheet near the body is physical but would
        # otherwise saturate the colour range and hide the wake sheet
        gx, gy = np.meshgrid(xg, yg)
        sel = (gx > 1.05) & np.isfinite(om)
        L = 1.1 * float(np.percentile(np.abs(om[sel]), 99.5)) if sel.any() \
            else 1.0
        pc = ax.pcolormesh(xg, yg, om, cmap="RdBu_r",
                           vmin=-L, vmax=L, shading="auto",
                           rasterized=True)
        draw_airfoil(ax, color="0.25", zorder=5)
        set_view(ax, WAKE_VIEW)
        style_colorbar(fig.colorbar(pc, ax=ax, shrink=0.92, pad=0.02,
                                    aspect=24),
                       r"$\omega_c\,c/V_\infty$")
        ax.set_title(f"({panel_letter(i)}) $\\alpha = {alpha:g}^\\circ$", pad=2,
                     loc="left", fontsize=7.5)
        if tag == tags[-1][0]:
            gx, gy = np.meshgrid(xg, yg)
            m = np.isfinite(om)
            rows = [[f"{a:.6f}", f"{b:.6f}", f"{c:.6e}" ]
                    for a, b, c in zip(gx[m], gy[m], om[m])]
            export_csv("fig_flow_wake", ["x", "y", "vorticity_nondim"],
                       rows)
    save(fig, "fig_flow_wake")


# ======================================================================
# figure: the flow ON the mesh (element boundaries overlaid)
# ======================================================================
def fig_flow_mesh():
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE, 66 * MM),
                             constrained_layout=True)
    tags = [("mesh_L2_a5", "L2 mesh, $N=3$"),
            ("mesh_L3_a5", "L3 mesh, $N=3$"),
            ("p_N5_a5", "L2 mesh, $N=5$")]
    for i, (tag, label) in enumerate(tags):
        ax = axes[i]
        U, X, Y, case = load_ckpt(tag)
        rho, u, v, p = primitives(U, GAMMA)
        mach = np.hypot(u, v) / np.sqrt(GAMMA * p / rho)
        hi = 1.03 * float(mach.max())
        levels = np.linspace(0.0, hi, 19)
        cs = draw_field(ax, X, Y, mach, levels, "viridis",
                        view=MESH_VIEW)
        for xs, ys in elem_lines(X, Y, X.shape[2]):
            ax.plot(xs, ys, color="k", lw=0.22, alpha=0.30, zorder=4)
        style_colorbar(fig.colorbar(cs, ax=ax, shrink=0.92, pad=0.02,
                                    aspect=24), "$M$")
        ax.set_title(label, pad=3)
        ax.text(0.98, 0.97, f"({panel_letter(i)}) $\\alpha = 5^\\circ$",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=8.5,
                bbox=dict(fc="white", ec="none", alpha=0.75, pad=1.2))
        if tag == tags[0][0]:
            export_csv("fig_flow_mesh",
                       ["mesh", "x", "y", "Mach"],
                       [[tags[0][1].split(" ")[0]] + r
                        for r in nodal_rows(X, Y, [mach],
                                            view=MESH_VIEW,
                                            fmt="{:.6f}")])
    save(fig, "fig_flow_mesh")


# ======================================================================
# main
# ======================================================================
FIGURES = {
    "density": fig_flow_density,
    "pressure": fig_flow_pressure,
    "mach": fig_flow_mach,
    "velocity": fig_flow_velocity,
    "wake": fig_flow_wake,
    "mesh": fig_flow_mesh,
}


def main():
    only = None
    if len(sys.argv) > 1:
        if sys.argv[1] == "--only":
            only = [s.strip() for s in sys.argv[2].split(",")]
        elif sys.argv[1] == "--list":
            print("available:", ", ".join(FIGURES))
            return
    print("flow-field post-production figures ->", FIG)
    for name, fn in FIGURES.items():
        if only and name not in only:
            continue
        print(f"[{name}]")
        fn()
    print("done.")


if __name__ == "__main__":
    main()
