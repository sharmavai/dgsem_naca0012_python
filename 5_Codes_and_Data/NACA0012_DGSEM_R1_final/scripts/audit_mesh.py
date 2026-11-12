#!/usr/bin/env python3
"""Independent audit of the mesh claims in the manuscript.

Checks, for every mesh of the production/refinement family and for the
variants used by the sensitivity study:
  * element count and degrees of freedom
  * discrete metric identities (normalised)
  * Jacobian positivity and minimum
  * wall-node agreement with the analytic NACA 0012 thickness
  * trailing-edge geometry (sharp or blunt?)
  * mirror symmetry of the un-aligned mesh, antisymmetry of the wake-curved one
  * first-layer height h_wall

    python scripts/audit_mesh.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dgsem.geometry import naca0012_thickness, TE_HALF_HEIGHT  # noqa: E402
from dgsem.mesh import (generate_airfoil_omesh,                  # noqa: E402
                       mirror_index_map)
from dgsem.panel import PanelSolver, _airfoil_vertices            # noqa: E402


def wake_curve(alpha, Mach):
    a = np.radians(alpha)
    ps = PanelSolver(*_airfoil_vertices(200))
    ps.solve(np.cos(a), np.sin(a))
    cl_panel = 4.0 * np.pi * ps.gamma * ps.len.sum()
    cl_est = cl_panel / np.sqrt(max(1.0 - Mach ** 2, 1e-6))
    return cl_est / (4.0 * np.pi)


def report(name, n_half, n_r, N, R, alpha, Mach=0.30):
    wc = wake_curve(alpha, Mach) if alpha != 0.0 else 0.0
    m = generate_airfoil_omesh(n_half, n_r, N, R, 0.30 / n_r, wake_curve=wc)
    q = m.quality_report()
    idx, idy = m.metric_identities()
    wx = m.X[:, 0, :, 0].ravel()
    wy = m.Y[:, 0, :, 0].ravel()
    side = np.abs(wx - 1.0) > 1e-12
    bnd = np.abs(np.abs(wy[side]) - naca0012_thickness(wx[side])).max()
    # first-layer radial height measured on the surface normal
    h1 = np.hypot(m.X[:, 0, :, 1] - m.X[:, 0, :, 0],
                  m.Y[:, 0, :, 1] - m.Y[:, 0, :, 0])
    print(f"{name:<22} n_k={m.n_k:>3} n_j={m.n_j:>3} N={N} "
          f"elem={m.n_k * m.n_j:>5} dof={m.n_k * m.n_j * m.Np1 ** 2 * 4:>7}  "
          f"GCL=({idx:.1e},{idy:.1e})  minJ={q['min_J']:.2e}  "
          f"wall={bnd:.1e}  h1={h1.min():.4f}..{h1.max():.4f}")
    return m


def main():
    print("=" * 100)
    print("MESH AUDIT")
    print("=" * 100)
    print(f"geometry module: TE_HALF_HEIGHT = {TE_HALF_HEIGHT}, "
          f"y_t(1.0) = {naca0012_thickness(1.0):.3e}, "
          f"max half-thickness = {naca0012_thickness(np.array([0.3]))[0]:.5f} "
          f"at x/c = 0.30")

    print("\n-- production + refinement family (alpha = 0, symmetric) --")
    for name, n_half, n_r in (("L1 40x15", 20, 15), ("L2 56x21", 28, 21),
                              ("L3 80x30", 40, 30)):
        report(name, n_half, n_r, 3, 50.0, 0.0)

    print("\n-- sensitivity meshes (alpha = 5 deg, wake-aligned) --")
    for name, R in (("R = 25c", 25.0), ("R = 50c (base)", 50.0),
                    ("R = 100c", 100.0)):
        report(name, 28, 21, 3, R, 5.0)

    print("\n-- p-family at alpha = 5 deg (same L2 mesh, different N) --")
    for N in (2, 3, 4, 5):
        report(f"N = {N}", 28, 21, N, 50.0, 5.0)

    print("\n-- mirror symmetry / wake-curve antisymmetry --")
    for wc, label in ((0.0, "symmetric (a = 0)"),
                      (0.05, "wake-curve +0.05"),
                      (-0.05, "wake-curve -0.05")):
        m = generate_airfoil_omesh(28, 21, 3, 50.0, 0.30 / 21, wake_curve=wc)
        kmap = mirror_index_map(m)
        # mirror map: element k -> k', xi reversed
        dX = np.abs(m.X[:, :, :, :][:, :, ::-1, :] - m.X[kmap, :, :, :]).max()
        dY = np.abs(m.Y[:, :, :, :][:, :, ::-1, :] + m.Y[kmap, :, :, :]).max()
        print(f"  {label:<18} max|X_err|={dX:.2e}  max|Y+y_err|={dY:.2e}")

    print("\n-- trailing edge probes --")
    m = generate_airfoil_omesh(28, 21, 3, 50.0, 0.30 / 21)
    wx = m.X[:, 0, :, 0]
    wy = m.Y[:, 0, :, 0]
    i_te = int(np.argmin(np.hypot(wx - 1.0, wy)))
    k_te, i_te2 = np.unravel_index(np.argmin(np.hypot(wx - 1.0, wy)), wx.shape)
    print(f"  closest wall node to (1,0): ({wx[k_te, i_te2]:.12f}, "
          f"{wy[k_te, i_te2]:.3e})")
    ys = np.unique(np.round(wy[np.abs(wx - 1.0) < 1e-9], 12))
    print(f"  distinct wall y-values at x = 1: {ys[:6]} ... n={len(ys)}")
    print(f"  => trailing edge is {'SHARP (single point)' if len(ys) <= 1 else 'BLUNT (base of ' + f'{ys.max() - ys.min():.2e}' + ')'}")


if __name__ == "__main__":
    main()
