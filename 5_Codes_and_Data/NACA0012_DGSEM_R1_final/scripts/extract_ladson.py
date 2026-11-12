"""Reference data: NACA 0012 force coefficients from NASA TM-4074, Table IV.

Transcription provenance
------------------------
C. L. Ladson, *Effects of independent variation of Mach and Reynolds numbers
on the low-speed aerodynamic characteristics of the NACA 0012 airfoil
section*, NASA TM-4074 (1988), **Table IV** ("Force Coefficients at M = 0.30
for Free Transition"), page 14 of the report (PDF page index 13).

Table IV is printed as four stacked blocks, one per Reynolds number:
R = 3.90, 5.93, 8.96 and 11.90 x 10^6.  The present study uses the
**R = 5.93 x 10^6** block.  Its angle column is

    -4.37, -2.02, -0.01, 0.03, 4.03, 6.08, 8.13, 10.12, 11.13, ...

Note that the block contains **no 2.05 deg entry**: the 2.05 deg row exists
only in the R = 3.90 x 10^6 block (C_l = 0.2216, C_d = 0.00698,
C_m = 0.0011), which is a different Reynolds number.  An earlier version of
this package carried the 2.05 deg value from that lower-Reynolds block
inside the R = 5.93e6 set, and labelled the whole set R = 5.97e6.  Both are
corrected here: the production reference file
(``ladson_force_data.csv``) contains the eight R = 5.93e6 rows only, and the
R = 3.90e6 block is transcribed separately in
``ladson_tableIV_R3.90e6.csv`` for provenance.

Running this script re-writes both CSVs from the transcription below and
verifies them row by row against the values read off the scan.
"""
from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "experimental"

# R = 5.93e6 block, Table IV (the production reference set).
TABLE_IV_R593 = [
    (-4.37, -0.4894, 0.00648, -0.0015),
    (-2.02, -0.2316, 0.00630, -0.0005),
    (-0.01, -0.0079, 0.00635, -0.0002),
    (0.03, -0.0026, 0.00659, -0.0002),
    (4.03, 0.4452, 0.00728, 0.0015),
    (6.08, 0.6762, 0.00711, 0.0031),
    (8.13, 0.9039, 0.00822, 0.0057),
    (10.12, 1.1251, 0.01200, 0.0094),
]

# R = 3.90e6 block, same table, for provenance only (not used in the paper).
TABLE_IV_R390 = [
    (-4.11, -0.4479, 0.00727, -0.0026),
    (-1.96, -0.2191, 0.00651, -0.0011),
    (0.04, 0.0001, 0.00657, 0.0000),
    (0.05, 0.0013, 0.00657, 0.0001),
    (2.05, 0.2216, 0.00698, 0.0011),
    (4.19, 0.4561, 0.00672, 0.0028),
    (6.20, 0.6781, 0.00743, 0.0048),
    (8.06, 0.8821, 0.01030, 0.0074),
    (10.15, 1.1089, 0.01270, 0.0112),
    (11.11, 1.2014, 0.01374, 0.0139),
    (12.10, 1.2840, 0.01569, 0.0167),
    (13.16, 1.3468, 0.02122, 0.0221),
    (14.13, 1.3466, 0.03118, 0.0292),
    (15.33, 1.0964, 0.17380, -0.0656),
    (16.46, 1.1197, 0.24760, -0.0995),
    (17.09, 1.1578, 0.25661, -0.0878),
    (18.61, 1.1636, 0.27777, -0.0868),
]

HEADER = ["source", "Mach", "Re", "transition", "alpha_deg", "CL", "CD", "CM"]


def _write(path: Path, rows, label: str, Re: str) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(HEADER)
        for a, cl, cd, cm in rows:
            w.writerow([label, 0.30, Re, "free", a, cl, cd, cm])
    print(f"wrote {path.relative_to(ROOT)}  ({len(rows)} rows, {Re})")


def _check(path: Path, rows, label: str, Re: str) -> None:
    """Re-read the CSV and compare against the transcription numerically."""
    with open(path) as fh:
        got = list(csv.DictReader(fh))
    assert len(got) == len(rows), (path, len(got), len(rows))
    for r, (a, cl, cd, cm) in zip(got, rows):
        assert r["source"] == label
        assert abs(float(r["Mach"]) - 0.30) < 1e-12
        assert r["Re"] == Re
        assert abs(float(r["alpha_deg"]) - a) < 1e-12
        for key, ref in (("CL", cl), ("CD", cd), ("CM", cm)):
            assert abs(float(r[key]) - ref) < 1e-12, (path, r, key)
    print(f"verified {path.name}: {len(rows)} rows match the transcription")


def main() -> None:
    f1 = OUT / "ladson_force_data.csv"
    f2 = OUT / "ladson_tableIV_R3.90e6.csv"
    _write(f1, TABLE_IV_R593, "NASA TM 4074 Table IV (R=5.93e6)", "5.93e6")
    _write(f2, TABLE_IV_R390, "NASA TM 4074 Table IV (R=3.90e6)", "3.90e6")
    _check(f1, TABLE_IV_R593, "NASA TM 4074 Table IV (R=5.93e6)", "5.93e6")
    _check(f2, TABLE_IV_R390, "NASA TM 4074 Table IV (R=3.90e6)", "3.90e6")

    # the production set must not contain a 2.05 deg row (see module docstring)
    assert not any(abs(a - 2.05) < 1e-9 for a, *_ in TABLE_IV_R593)
    print("OK: production reference set has no 2.05 deg row "
          "(that row exists only at R = 3.90e6)")


if __name__ == "__main__":
    main()
