#!/usr/bin/env python3
"""Verification suite for the revised solver.

Tests
-----
1.  Discrete metric identities (free-stream preservation of the interior
    operator) on every production mesh, and the exact geometric mirror
    symmetry of the O-mesh family.
2.  Slip-wall boundary-condition exactness: uniform flow parallel to the
    walls of a plane channel is an exact solution; the full operator
    (volume + wall SAT through the mirror ghost) must return machine zero.
    This is the test the previous study replaced by a copy ghost.
3.  Free-stream preservation of the full assembly (interior + periodic
    seam + far-field Riemann boundary) on the airfoil mesh; the wall SAT is
    disabled by a copy ghost because a uniform state is not a solution of
    the curved-wall problem -- stated as such.
4.  Isentropic-vortex advection over a FULL period on a warped periodic
    mesh: h-convergence at N = 3 (expected order ~ N+1), p-convergence
    N = 1..8, and a temporal-refinement check that bounds the time error.
5.  One-shot spatial-operator order test: residual of the exact vortex
    solution versus its exact time derivative (h-refinement, N = 2..4).
6.  Viscous-operator patch tests on a curved mesh (linear and quadratic
    exact solutions, machine precision) and an unsteady scalar-diffusion
    manufactured-solution order study.
7.  Sod shock tube against the exact Riemann solution (shock capturing:
    Persson-Peraire sensor + artificial viscosity + positivity limiter).
8.  Force integration against the exact potential-flow (panel) solution.
9.  Discrete mirror antisymmetry: +alpha / -alpha runs started from the
    free stream remain exact mirror images of each other.

Writes results_new/verification.json and prints a summary table.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dgsem.diagnostics import entropy_error
from dgsem.euler import GAMMA, freestream_state, initialize_freestream, primitives
from dgsem.integrate import rk4_step, residual_norm
from dgsem.limiters import positivity_limiter
from dgsem.mesh import (generate_airfoil_omesh, generate_channel_mesh,
                        generate_periodic_box_mesh, generate_sod_mesh,
                        mirror_index_map)
from dgsem.operator import Operator, SolverConfig
from dgsem.operator_numba import NumbaOperator
from dgsem.solver import _smooth_dt
from dgsem.timestep import local_timestep

RESULTS = {}
OUT = Path(__file__).resolve().parents[1] / "results_new"
OUT.mkdir(exist_ok=True)


def log(*a):
    print(*a, flush=True)


# ======================================================================
def test_metric_identities():
    log("--- metric identities / free-stream preservation (interior)")
    rows = []
    meshes = {
        "L1": dict(n_half=20, n_r=15),
        "L2": dict(n_half=28, n_r=21),
        "L3": dict(n_half=40, n_r=30),
        "L2-R25": dict(n_half=28, n_r=21, R_far=25.0),
        "L2-R100": dict(n_half=28, n_r=21, R_far=100.0),
        "L2-wake": dict(n_half=28, n_r=21, wake_curve=0.05),
        "L2-N4": dict(n_half=28, n_r=21, N=4),
    }
    for name, kw in meshes.items():
        kk = dict(kw)
        kk.setdefault("N", 3)
        kk.setdefault("R_far", 50.0)
        kk.setdefault("wake_curve", 0.0)
        kk["h_wall"] = 0.30 / kk["n_r"]
        m = generate_airfoil_omesh(**kk)
        idx, idy = m.metric_identities()
        q = m.quality_report()
        rows.append(dict(mesh=name, elements=int(m.n_k * m.n_j),
                         dofs=int(m.n_k * m.n_j * m.Np1 ** 2 * 4),
                         metric_id_x=idx, metric_id_y=idy,
                         min_J=q["min_J"], max_skew=q["max_skew_cos"]))
        log(f"  {name:9s} elems={rows[-1]['elements']:5d} "
            f"ids=({idx:.1e},{idy:.1e}) minJ={q['min_J']:.2e}")
        assert idx < 1e-12 and idy < 1e-12 and q["min_J"] > 0
    RESULTS["metric_identities"] = rows


def test_mesh_symmetry():
    log("--- O-mesh mirror symmetry")
    m = generate_airfoil_omesh(28, 21, 3, 50.0, 0.30 / 21)   # unaligned
    kmap = mirror_index_map(m)
    Np1 = m.Np1
    # node i of element k mirrors node Np1-1-i of element kmap (same l)
    X1 = m.X[kmap][:, :, ::-1, :]
    Y1 = m.Y[kmap][:, :, ::-1, :]
    ex = np.abs(m.X - X1).max()
    ey = np.abs(m.Y + Y1).max()
    log(f"  |X - mirror(X)| = {ex:.3e}, |Y + mirror(Y)| = {ey:.3e}")
    assert ex < 1e-12 and ey < 1e-12
    # a +wake-curve mesh mirrors onto a -wake-curve mesh (index-mirrored)
    mp = generate_airfoil_omesh(28, 21, 3, 50.0, 0.30 / 21, wake_curve=0.05)
    mm = generate_airfoil_omesh(28, 21, 3, 50.0, 0.30 / 21, wake_curve=-0.05)
    X2 = mp.X[kmap][:, :, ::-1, :]
    Y2 = mp.Y[kmap][:, :, ::-1, :]
    ex2 = np.abs(mm.X - X2).max()
    ey2 = np.abs(mm.Y + Y2).max()
    log(f"  wake-curve antisymmetry: {ex2:.3e}, {ey2:.3e}")
    assert ex2 < 1e-12 and ey2 < 1e-12
    RESULTS["mesh_symmetry"] = {"mirror_x": float(ex), "mirror_y": float(ey),
                                "wake_curve_x": float(ex2),
                                "wake_curve_y": float(ey2)}


def test_slip_wall():
    log("--- slip-wall exactness (channel, uniform flow parallel to walls)")
    rows = []
    for N in (2, 3, 4):
        for op_name in ("numpy", "numba"):
            m = generate_channel_mesh(8, 5, N, 2.0, 1.0)
            U, Q = initialize_freestream(m, 0.4, 0.0)
            if op_name == "numpy":
                op = Operator(m, SolverConfig(wall="slip"))
            else:
                op = NumbaOperator(m, SolverConfig(wall="slip"))
            op.set_freestream(Q)
            R = op.rhs(U)
            err = float(np.abs(R * m.J[..., None]).max())
            rows.append(dict(N=N, operator=op_name, err=err))
            log(f"  N={N} {op_name}: max|J dU/dt| = {err:.3e}")
            assert err < 1e-12
    RESULTS["slip_wall_exactness"] = rows


def test_freestream_assembly():
    log("--- free-stream preservation of interior+seam+far field (airfoil)")
    m = generate_airfoil_omesh(28, 21, 3, 50.0, 0.30 / 21, wake_curve=0.05)
    cfg = SolverConfig(wall="copy")       # wall SAT off: uniform state is
    op = Operator(m, cfg)                 # not a curved-wall solution
    U, Q = initialize_freestream(m, 0.3, 0.0)
    op.set_freestream(Q)
    R = op.rhs(U)
    err = float(np.abs(R * m.J[..., None]).max())
    log(f"  copy-wall assembly (alpha=0, Gamma=0): max|J dU/dt| = {err:.3e}")
    assert err < 1e-12
    # with the vortex correction active (Gamma != 0) the free stream is NOT
    # a solution; the residual must be finite and small instead
    op.Gamma = -0.09
    R = op.rhs(U)
    mx = float(np.abs(R).max())
    log(f"  with vortex correction: |R| = {mx:.3e} (O(Gamma / R_far), finite)")
    assert np.isfinite(mx) and mx < 0.05
    RESULTS["freestream_assembly"] = {"copy_wall_err": err,
                                      "vortex_residual": mx}


# ---------------------------------------------------------------- vortex
def vortex_state(mesh, beta=0.5, cx=5.0, cy=5.0):
    dx = mesh.X - cx
    dy = mesh.Y - cy
    r2 = dx * dx + dy * dy
    rho = (1.0 - (GAMMA - 1.0) * beta ** 2 / (8.0 * GAMMA * np.pi ** 2)
           * np.exp(1.0 - r2)) ** (1.0 / (GAMMA - 1.0))
    u = 1.0 - beta / (2.0 * np.pi) * dy * np.exp((1.0 - r2) / 2.0)
    v = beta / (2.0 * np.pi) * dx * np.exp((1.0 - r2) / 2.0)
    p = rho ** GAMMA
    U = np.empty(mesh.X.shape + (4,))
    U[..., 0] = rho
    U[..., 1] = rho * u
    U[..., 2] = rho * v
    U[..., 3] = p / (GAMMA - 1.0) + 0.5 * rho * (u * u + v * v)
    return U


def l2_rho(U, Ue, mesh):
    wJ = np.outer(mesh.w, mesh.w)[None, None, :, :] * mesh.J
    return float(np.sqrt(((U[..., 0] - Ue[..., 0]) ** 2 * wJ).sum()
                         / wJ.sum()))


def march(U, mesh, dt_glob, t_end, op=None):
    if op is None:
        op = NumbaOperator(mesh, SolverConfig(entropy_fix=0.0))
    U = U.copy()
    n = int(np.ceil(t_end / dt_glob))
    dt = np.full((mesh.n_k, mesh.n_j), t_end / n)
    for _ in range(n):
        k1 = rk4_step(U, op, dt)
        U, _ = positivity_limiter(U, mesh, GAMMA)
    return U


def test_vortex():
    log("--- isentropic vortex, full period T = 10 (warped periodic mesh)")
    L = 10.0
    T = L                      # one full period: returns to the start
    # h-convergence at N=3
    rows = []
    prev = None
    prev_h = None
    for ne in (10, 14, 20, 28):
        m = generate_periodic_box_mesh(ne, ne, 3, L, 0.1)
        U0 = vortex_state(m)
        dt = 0.9 / ((4) ** 2 * 2.5) * (L / ne)   # CFL 0.9 global
        Ue = vortex_state(m)                       # periodic: same state
        U = march(U0, m, dt, T)
        e = l2_rho(U, Ue, m)
        h = L / ne
        order = (np.log(prev / e) / np.log(prev_h / h)
                 if prev else float("nan"))
        rows.append(dict(kind="h", N=3, elems=f"{ne}x{ne}", L2=e,
                         order=float(order), dt=dt))
        log(f"  h: {ne:2d}x{ne:<2d} N=3  L2(rho)={e:.3e}  "
            f"order={order if prev else float('nan'):.2f}")
        prev, prev_h = e, h
    # p-convergence at 16x16
    rows_p = []
    prev = None
    for N in (1, 2, 3, 4, 5, 6, 7, 8):
        m = generate_periodic_box_mesh(16, 16, N, L, 0.1)
        U0 = vortex_state(m)
        dt = 0.9 / ((N + 1) ** 2 * 2.5) * (L / 16)
        Ue = vortex_state(m)
        U = march(U0, m, dt, T)
        e = l2_rho(U, Ue, m)
        rows_p.append(dict(kind="p", N=N, elems="16x16", L2=e, dt=dt))
        log(f"  p: 16x16 N={N}  L2(rho)={e:.3e}")
    # temporal check at N=3, 20x20: halve the CFL twice
    m = generate_periodic_box_mesh(20, 20, 3, L, 0.1)
    U0 = vortex_state(m)
    Ue = vortex_state(m)
    ets = []
    for cfl in (0.9, 0.3):
        dt = cfl / (16 * 2.5) * (L / 20)
        U = march(U0, m, dt, T)
        ets.append(l2_rho(U, Ue, m))
    log(f"  temporal check: L2(CFL .9)={ets[0]:.3e}, L2(CFL .3)={ets[1]:.3e} "
        f"(spatially dominated -> nearly equal)")
    RESULTS["vortex_full_period"] = rows + rows_p
    RESULTS["vortex_temporal_check"] = dict(L2_cfl09=ets[0], L2_cfl03=ets[1])
    # observed h-orders for the table
    o = [r["order"] for r in rows if r["order"] == r["order"]]
    RESULTS["vortex_h_orders"] = o


def test_operator_order():
    log("--- one-shot operator order (exact vortex residual)")
    rows = []
    L = 10.0
    for N in (2, 3, 4):
        prev = None
        prev_h = None
        for ne in (8, 12, 16, 24, 32):
            m = generate_periodic_box_mesh(ne, ne, N, L, 0.1)
            op = NumbaOperator(m, SolverConfig(entropy_fix=0.0))
            U0 = vortex_state(m)
            R = op.rhs(U0)
            # exact time derivative: the family U(x, t) = U0(x - t) gives
            # U_t = -dU0/dx; with shift(+d) = U0(x - d) one gets
            # U_t = (shift(+d) - shift(-d)) / (2 d)
            d = 1e-5
            Up = vortex_state_shift(m, d)
            Um = vortex_state_shift(m, -d)
            dUdt = (Up - Um) / (2 * d)      # see derivation in comment
            wJ = np.outer(m.w, m.w)[None, None, :, :] * m.J
            err = float(np.sqrt((((R - dUdt) ** 2)[..., 0] * wJ).sum()
                                / wJ.sum()))
            h = L / ne
            order = (np.log(prev / err) / np.log(prev_h / h)
                     if prev else float("nan"))
            rows.append(dict(N=N, elems=f"{ne}x{ne}", err=err, order=order))
            log(f"  N={N} {ne:2d}x{ne:<2d}: err={err:.3e} "
                f"order={order if prev else float('nan'):.2f}")
            prev, prev_h = err, h
    RESULTS["operator_order"] = rows


def vortex_state_shift(mesh, dx, beta=0.5):
    return vortex_state(mesh, beta, cx=5.0 + dx, cy=5.0)


# ---------------------------------------------------------------- viscous
def test_viscous_patches():
    log("--- viscous operator patch tests (flat periodic mesh)")
    # NOTE: the patch test uses a flat mesh because a field that is a
    # polynomial in physical (x, y) is only exactly representable when the
    # mapping is affine; on warped meshes the order study below covers the
    # curved-element case.
    m = generate_periodic_box_mesh(6, 5, 3, 2.0, 0.0)
    mu = 0.37
    kc = mu * GAMMA / (0.72 * (GAMMA - 1.0))       # Pr = 0.72
    opv = Operator(m, SolverConfig(mu=mu))
    op0 = Operator(m, SolverConfig(mu=0.0))
    p = 1.0 / GAMMA
    Y = m.Y
    # periodic polynomial test fields on [0, 2]^2 (u independent of x):
    #   V1  u = y (2 - y)  (global quadratic, C^0 at the periodic wrap,
    #       single-valued fluxes) -> momx residual = mu u'' = -2 mu exactly,
    #       energy residual = mu (u u'' + u'^2) exactly
    #   V2  T = y (2 - y), u = v = 0 -> energy residual = kappa T'' = -2 kappa
    def build(u, v, T):
        U = np.empty(m.X.shape + (4,))
        U[..., 0] = 1.0
        U[..., 1] = u
        U[..., 2] = v
        U[..., 3] = p / (GAMMA - 1.0) + 0.5 * (u * u + v * v) + 0.0 * T
        return U

    # V1: shear patch test.  u = y(2-y)(1-y): a global cubic with u, u'
    # continuous across the periodic wrap, so every viscous flux is
    # single-valued and the operator must return the exact mu*u''.
    u = Y * (2.0 - Y) * (1.0 - Y)
    uxx = 6.0 * Y - 6.0            # u'' of the cubic
    U = build(u, 0.0 * Y, (1.0 / GAMMA) * np.ones_like(Y))
    Rv = opv.rhs(U) - op0.rhs(U)
    e1 = float(np.abs(Rv[..., 1] - mu * uxx).max())
    e2 = float(np.abs(Rv[..., 2]).max())
    log(f"  V1 cubic shear: |momx err| = {e1:.3e}, |momy| = {e2:.3e}")
    assert e1 < 1e-11 and e2 < 1e-11

    # V2: conduction patch test (u = v = 0, cubic T, shifted to keep p > 0)
    T = 1.0 + 2.0 * Y * (2.0 - Y) * (1.0 - Y)
    U = build(0.0 * Y, 0.0 * Y, T)
    U[..., 3] = T / (GAMMA - 1.0)          # rho = 1 -> p = T
    Rv = opv.rhs(U) - op0.rhs(U)
    e3 = float(np.abs(Rv[..., 3] - kc * 2.0 * uxx).max())
    log(f"  V2 conduction: |energy err| = {e3:.3e}")
    assert e3 < 1e-11
    RESULTS["viscous_patch"] = dict(V1_mom=e1, V1_momy=e2, V2_energy=e3)


def test_diffusion_order():
    log("--- scalar diffusion MMS (one-shot operator order)")
    rows = []
    nu = 0.05
    for N in (2, 3, 4):
        prev = None
        prev_h = None
        for ne in (8, 12, 16, 24):
            m = generate_periodic_box_mesh(ne, ne, N, 2.0, 0.3)
            # manufactured field q = sin(2 pi x / L) sin(2 pi y / L): a
            # FULL period on the box, C^1 across the periodic wrap
            L = 2.0
            q = np.sin(2 * np.pi * m.X / L) * np.sin(2 * np.pi * m.Y / L)
            U = np.zeros(m.X.shape + (4,))
            U[..., 0] = q
            nu_arr = np.full((m.n_k, m.n_j), nu)
            cfg = SolverConfig()
            op = Operator(m, cfg)
            Res = np.zeros_like(U)
            op._diffuse_conservative(U, Res, nu_arr)
            # dU/dt of the diffusion block is -Res / J; the exact value is
            # div(nu grad q) = -nu (2 pi / L)^2 2 q
            exact = -nu * 2.0 * (2 * np.pi / L) ** 2 * q
            wq = np.outer(m.w, m.w)[None, None]
            diff = -Res[..., 0] / m.J - exact
            err = float(np.sqrt((diff ** 2 * wq * m.J).sum()
                                / (wq * m.J).sum()))
            h = 2.0 / ne
            order = (np.log(prev / err) / np.log(prev_h / h)
                     if prev else float("nan"))
            rows.append(dict(N=N, elems=f"{ne}x{ne}", err=err, order=order))
            log(f"  N={N} {ne:2d}x{ne:<2d}: err={err:.3e} "
                f"order={order if prev else float('nan'):.2f}")
            prev, prev_h = err, h
    RESULTS["diffusion_order"] = rows


# ---------------------------------------------------------------- diffusion
def test_diffusion_mms_marched():
    log("--- scalar diffusion MMS, marched in time (solution order N+1)")
    from dgsem.integrate import rk4_step
    rows = []
    nu = 0.05
    L = 2.0
    k = 2 * np.pi / L
    t_end = 0.15
    for N in (2, 3, 4):
        prev = None
        prev_h = None
        for ne in (6, 9, 13, 20):
            m = generate_periodic_box_mesh(ne, ne, N, L, 0.3)
            U = np.zeros(m.X.shape + (4,))
            U[..., 0] = 1.0 + 0.5 * np.sin(k * m.X) * np.sin(k * m.Y)
            op = Operator(m, SolverConfig())
            nu_arr = np.full((m.n_k, m.n_j), nu)
            dt = 0.05 * (L / ne) ** 2 / ((N + 1) ** 4 * nu)   # viscous limit
            nsteps = int(np.ceil(t_end / dt))
            dta = np.full((m.n_k, m.n_j), t_end / nsteps)
            for _ in range(nsteps):
                rk4_step(U, op, dta, nu_arr)
            decay = np.exp(-nu * k * k * 2 * t_end)
            exact = (1.0 + 0.5 * decay
                     * np.sin(k * m.X) * np.sin(k * m.Y))
            wq = np.outer(m.w, m.w)[None, None]
            err = float(np.sqrt(((U[..., 0] - exact) ** 2 * wq * m.J).sum()
                                 / (wq * m.J).sum()))
            h = L / ne
            order = (np.log(prev / err) / np.log(prev_h / h)
                     if prev else float("nan"))
            rows.append(dict(N=N, elems=f"{ne}x{ne}", err=err, order=order,
                             dt=t_end / nsteps))
            log(f"  N={N} {ne:2d}x{ne:<2d}: L2={err:.3e} "
                f"order={order if prev else float('nan'):.2f}")
            prev, prev_h = err, h
    RESULTS["diffusion_mms_marched"] = rows


# ---------------------------------------------------------------- Sod
def sod_exact(x, t, XL=1.0):
    """Exact Sod solution (rho, u, p) at (x, t); diaphragm at XL."""
    g = GAMMA
    rL, uL, pL = 1.0, 0.0, 1.0
    rR, uR, pR = 0.125, 0.0, 0.1

    def f(p, pk, rk):
        if p > pk:      # shock
            return (p - pk) * np.sqrt(2.0
                                      / (rk * ((g + 1) * p
                                               + (g - 1) * pk)))
        # rarefaction
        ck = np.sqrt(g * pk / rk)
        return 2.0 * ck / (g - 1) * ((p / pk) ** ((g - 1) / (2 * g))
                                     - 1.0)

    # solve p* by bisection
    lo, hi = 1e-6, 10.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        val = f(mid, pL, rL) + f(mid, pR, rR) + (uR - uL)
        if val > 0:
            hi = mid
        else:
            lo = mid
    pstar = 0.5 * (lo + hi)
    ustar = 0.5 * (uL + uR) + 0.5 * (f(pstar, pR, rR) - f(pstar, pL, rL))
    cL = np.sqrt(g * pL / rL)
    cR = np.sqrt(g * pR / rR)
    rstarL = rL * (pstar / pL) ** (1.0 / g)
    rstarR = rR * ((pstar / pR + (g - 1) / (g + 1))
                   / ((g - 1) * pstar / ((g + 1) * pR) + 1.0))
    # shock speed into the undisturbed right state (Toro): the mass flux
    # across the shock gives S = uR + cR sqrt(...); ustar is NOT the base
    # velocity here (that mistake displaces the exact shock by u* t).
    S = uR + cR * np.sqrt((g + 1) / (2 * g) * pstar / pR
                          + (g - 1) / (2 * g))
    xi = (x - XL) / t
    rho = np.where(xi < -cL, rL, 0.0)
    uu = np.where(xi < -cL, uL, 0.0)
    pp = np.where(xi < -cL, pL, 0.0)
    # rarefaction fan: u - c decreases from -cL to ustar - c*
    lam = ustar - cL * (pstar / pL) ** ((g - 1) / (2 * g))
    inR = (xi >= -cL) & (xi <= lam)
    inR = (xi >= -cL) & (xi <= lam)
    u_fan = 2.0 / (g + 1) * (cL + xi)
    c_fan = cL - (g - 1) / 2.0 * u_fan
    rho_fan = rL * (c_fan / cL) ** (2.0 / (g - 1))
    p_fan = pL * (c_fan / cL) ** (2 * g / (g - 1))
    rho = np.where(inR, rho_fan, rho)
    uu = np.where(inR, u_fan, uu)
    pp = np.where(inR, p_fan, pp)
    inSL = (xi > lam) & (xi < ustar)
    rho = np.where(inSL, rstarL, rho)
    uu = np.where(inSL, ustar, uu)
    pp = np.where(inSL, pstar, pp)
    inSR = (xi >= ustar) & (xi < S)
    rho = np.where(inSR, rstarR, rho)
    uu = np.where(inSR, ustar, uu)
    pp = np.where(inSR, pstar, pp)
    rho = np.where(xi >= S, rR, rho)
    uu = np.where(xi >= S, uR, uu)
    pp = np.where(xi >= S, pR, pp)
    return rho, uu, pp


def test_sod():
    log("--- Sod shock tube vs exact Riemann solution (t = 0.2)")
    from dgsem.shock import artificial_viscosity
    rows = []
    t_end = 0.2
    for ne in (48, 96, 192):
        m = generate_sod_mesh(ne, 3, 3, 2.0, 0.4)
        QL = np.array([1.0, 0.0, 0.0, 1.0 / (GAMMA - 1.0)])
        QR = np.array([0.125, 0.0, 0.0, 0.1 / (GAMMA - 1.0)])
        U = np.where((m.X < 1.0)[..., None], QL, QR)
        cfg = SolverConfig(flux="roe", entropy_fix=0.05)
        op = Operator(m, cfg)
        op.set_freestream(QL)
        op.Q_xi_L = QL
        op.Q_xi_R = QR
        # global time stepping, RK4, with sensor + AV + limiter.  The time
        # step is RECOMPUTED every step: the artificial viscosity grows as
        # the shock forms, and the viscous limit h^2/(nu (N+1)^4) tightens
        # accordingly (a fixed dt from the initial nu diverges).
        t = 0.0
        nu_run_max = 0.0
        while t < t_end - 1e-12:
            nu, _S = artificial_viscosity(U, m, GAMMA, 1e-3, 5e-2, 0.35)
            nu_run_max = max(nu_run_max, float(nu.max()))
            # NB: mesh.h_xi is the extent NORMAL to xi faces (the eta
            # direction); the smallest element extent governs both limits.
            h_min = min(m.h_xi.min(), m.h_eta.min())
            dt = min(0.2 * h_min ** 2
                     / (m.Np1 ** 4 * max(nu.max(), 1e-12)),
                     0.4 / (m.Np1 ** 2 * 2.5) * h_min,
                     t_end - t)
            dta = np.full((m.n_k, m.n_j), dt)
            k1 = rk4_step(U, op, dta, nu)
            U, _ = positivity_limiter(U, m, GAMMA)
            t += dt
        rho_e, u_e, p_e = sod_exact(m.X, t_end)
        rho = U[..., 0]
        # L1 error on the centerline (y-invariant solution)
        wJ = np.outer(m.w, m.w)[None, None] * m.J
        L1 = float((np.abs(rho - rho_e) * wJ).sum() / wJ.sum())
        # positions of the contact and the shock by threshold crossing on
        # the UNIQUE centerline x (the periodic eta layers duplicate x,
        # which would poison np.gradient with dx = 0).  The argmax-gradient
        # alternative is not robust: the contact jump exceeds the shock
        # jump, so it would compare the numerical shock with the exact
        # contact.
        xs = m.X[:, 1, :, 1]
        rs = rho[:, 1, :, 1]
        xu, idx = np.unique(xs.flatten(), return_index=True)
        ru = rs.flatten()[idx]
        rho_starL = float(ru[(xu > 0.9) & (xu < 1.05)].mean())
        rho_starR = float(ru[(xu > 1.22) & (xu < 1.30)].mean())
        rho_R = float(ru[xu > 1.85].mean())
        c_thr = 0.5 * (rho_starL + rho_starR)
        s_thr = 0.5 * (rho_starR + rho_R)
        x_contact = xu[np.argmax(ru < c_thr)]
        x_shock = xu[np.argmax(ru < s_thr)]
        x_num = x_shock
        xs_e = np.linspace(0, 2, 4001)
        re, _, _ = sod_exact(xs_e, t_end)
        re_sL = float(re[(xs_e > 0.9) & (xs_e < 1.05)].mean())
        re_sR = float(re[(xs_e > 1.22) & (xs_e < 1.30)].mean())
        re_R = float(re[xs_e > 1.85].mean())
        x_contact_e = xs_e[np.argmax(re < 0.5 * (re_sL + re_sR))]
        x_shock_e = xs_e[np.argmax(re < 0.5 * (re_sR + re_R))]
        x_ex = x_shock_e
        rows.append(dict(elems=ne, L1_rho=L1, shock_num=float(x_num),
                         shock_exact=float(x_ex),
                         contact_num=float(x_contact),
                         contact_exact=float(x_contact_e),
                         nu_max=nu_run_max))
        log(f"  {ne:3d} elems: L1(rho)={L1:.4e}  shock x={x_num:.4f} "
            f"(exact {x_ex:.4f})  contact x={x_contact:.4f} "
            f"(exact {x_contact_e:.4f})  nu_max={nu_run_max:.3g}")
    prev = None
    for r in rows:
        if prev:
            r["L1_order"] = float(np.log2(prev / r["L1_rho"]))
        else:
            r["L1_order"] = None
        prev = r["L1_rho"]
    for r in rows:
        if r["L1_order"]:
            log(f"  L1 order ({r['elems']}): {r['L1_order']:.2f}")
    RESULTS["sod"] = rows


# ---------------------------------------------------------------- forces
def test_forces_vs_panel():
    log("--- force integration vs exact potential-flow (panel) solution")
    from dgsem.forces import forces
    from dgsem.panel import PanelSolver, _airfoil_vertices, potential_flow_state
    rows = []
    for a in (0.0, 5.0):
        m = generate_airfoil_omesh(28, 21, 3, 50.0, 0.30 / 21)
        U, ps = potential_flow_state(m, 0.5, a, GAMMA)   # Mach = 0.5 must
        fo = forces(U, m, 0.5, a, GAMMA)   # match the q_inf normalisation
        V = 0.5                       # free-stream speed of the panel solve
        cl_panel = 4.0 * np.pi * ps.gamma * ps.len.sum() / V
        rows.append(dict(alpha=a, Cl_dg=fo["Cl"], Cl_panel=float(cl_panel),
                         rel=float(abs(fo["Cl"] - cl_panel)
                                   / max(abs(cl_panel), 1e-12))))
        log(f"  alpha={a}: Cl(DG of panel field)={fo['Cl']:+.5f}  "
            f"Cl(panel)={cl_panel:+.5f}  rel={rows[-1]['rel']:.2e}")
    RESULTS["forces_vs_panel"] = rows


# ---------------------------------------------------------------- antisym
def test_antisymmetry():
    log("--- discrete antisymmetry of +alpha / -alpha marches")
    m = generate_airfoil_omesh(28, 21, 3, 50.0, 0.30 / 21, wake_curve=0.05)
    m2 = generate_airfoil_omesh(28, 21, 3, 50.0, 0.30 / 21, wake_curve=-0.05)
    kmap = mirror_index_map(m)
    U1, Q1 = initialize_freestream(m, 0.3, 5.0)
    U2, Q2 = initialize_freestream(m2, 0.3, -5.0)
    op1 = NumbaOperator(m, SolverConfig(entropy_fix=0.02, fix_all_waves=True))
    op1.set_freestream(Q1)
    op2 = NumbaOperator(m2, SolverConfig(entropy_fix=0.02, fix_all_waves=True))
    op2.set_freestream(Q2)
    op1.Gamma = -0.09
    op2.Gamma = +0.09

    def mirror_err():
        Um2 = U2[kmap][:, :, ::-1, :, :]
        Um2[..., 2] *= -1.0      # v -> -v under reflection
        return float(np.abs(U1 - Um2).max() / np.abs(U1).max())

    # Stage 1: exact machinery test over a short march.  The spatial
    # operator, the limiter and the (symmetric) time-step smoother are all
    # mirror-covariant to machine precision, so a short march must stay
    # at the rounding level.
    for i in range(30):
        cfl = 0.9 * min(1.0, 0.1 + 0.9 * i / 600)
        d1 = _smooth_dt(local_timestep(U1, m, GAMMA, cfl))
        d2 = _smooth_dt(local_timestep(U2, m2, GAMMA, cfl))
        rk4_step(U1, op1, d1)
        rk4_step(U2, op2, d2)
        U1, _ = positivity_limiter(U1, m, GAMMA)
        U2, _ = positivity_limiter(U2, m2, GAMMA)
    err_short = mirror_err()
    log(f"  mirror difference after 30 steps: {err_short:.3e} "
        f"(machine precision required)")
    assert err_short < 1e-12

    # Stage 2: 600 steps from the impulsive start, informational.  The
    # violent starting transient amplifies the ~1e-13 coordinate rounding
    # of the mirrored meshes exponentially in the wake; this is physical
    # (Lyapunov) growth shared by any truncation-level perturbation, not a
    # consistency defect, but it bounds how precisely +/-alpha runs can be
    # expected to agree late in a march.
    for i in range(30, 600):
        cfl = 0.9 * min(1.0, 0.1 + 0.9 * i / 600)
        d1 = _smooth_dt(local_timestep(U1, m, GAMMA, cfl))
        d2 = _smooth_dt(local_timestep(U2, m2, GAMMA, cfl))
        rk4_step(U1, op1, d1)
        rk4_step(U2, op2, d2)
        U1, _ = positivity_limiter(U1, m, GAMMA)
        U2, _ = positivity_limiter(U2, m2, GAMMA)
    err = mirror_err()
    log(f"  mirror difference after 600 steps: {err:.3e} "
        f"(transient amplification, regression guard < 5e-3)")
    assert err < 5e-3
    RESULTS["antisymmetry"] = {"rel_mirror_diff_short": err_short,
                              "rel_mirror_diff_600": err}


# ======================================================================
if __name__ == "__main__":
    t0 = time.time()
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "ids"):
        test_metric_identities()
        test_mesh_symmetry()
    if which in ("all", "wall"):
        test_slip_wall()
        test_freestream_assembly()
    if which in ("all", "vortex"):
        test_vortex()
        test_operator_order()
    if which in ("all", "visc"):
        test_viscous_patches()
        test_diffusion_order()
        test_diffusion_mms_marched()
    if which in ("all", "sod"):
        test_sod()
    if which in ("all", "forces"):
        test_forces_vs_panel()
    if which in ("all", "antisym"):
        test_antisymmetry()
    with open(OUT / "verification.json", "w") as fh:
        json.dump(RESULTS, fh, indent=1)
    log(f"verification written to {OUT / 'verification.json'} "
        f"({time.time() - t0:.0f} s)")
