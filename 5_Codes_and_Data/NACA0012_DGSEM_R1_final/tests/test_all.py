"""Run every module self test in sequence (the CI entry point).

The runner is deliberately interpreter-agnostic: each module is executed in a
fresh subprocess with the package's ``src`` directory placed on ``sys.path``
explicitly, so the suite also works with embedded / isolated Python builds
that do not add the working directory or ``PYTHONPATH`` to the import path.

    python tests/test_all.py
"""
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"

mods = ["dgsem.quadrature", "dgsem.geometry", "dgsem.mesh",
        "dgsem.numflux", "dgsem.shock", "dgsem.panel",
        "dgsem.euler", "dgsem.limiters", "dgsem.timestep",
        "dgsem.integrate", "dgsem.forces", "dgsem.diagnostics",
        "dgsem.operator", "dgsem.operator_numba"]

BOOT = ("import sys, runpy; sys.path.insert(0, {src!r}); "
        "runpy.run_module({mod!r}, run_name='__main__')")


def run(mod, timeout=None):
    code = BOOT.format(src=str(SRC), mod=mod)
    return subprocess.run([sys.executable, "-c", code], cwd=SRC,
                          capture_output=True, text=True, timeout=timeout)


def main():
    fails = 0
    for m in mods:
        r = run(m)
        ok = r.returncode == 0
        print(f"{'PASS' if ok else 'FAIL'}  {m}")
        if not ok:
            fails += 1
            print(r.stdout[-1200:], r.stderr[-1200:])
    # solver smoke test (tiny mesh, short impulsive run) last: slowest
    try:
        r = run("dgsem.solver", timeout=1800)
        ok = r.returncode == 0
    except subprocess.TimeoutExpired:
        ok = False
        r = None
    print(f"{'PASS' if ok else 'FAIL'}  dgsem.solver (smoke)")
    if not ok:
        fails += 1
        if r is not None:
            print(r.stdout[-1200:], r.stderr[-1200:])
    if fails:
        print(f"{fails} module self test(s) FAILED")
        return 1
    print("ALL MODULE SELF-TESTS PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
