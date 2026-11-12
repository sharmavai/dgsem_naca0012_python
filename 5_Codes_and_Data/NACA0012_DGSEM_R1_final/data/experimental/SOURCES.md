# Experimental reference data: sources and provenance

## `ladson_force_data.csv` — the reference set used in the paper

C. L. Ladson, *Effects of independent variation of Mach and Reynolds
numbers on the low-speed aerodynamic characteristics of the NACA 0012
airfoil section*, NASA TM-4074 (1988).

* **Table IV**, "Force Coefficients at M = 0.30 for Free Transition",
  report page 14 (PDF page index 13), **R = 5.93 x 10^6 block**.
* Eight measured angles: -4.37, -2.02, -0.01, 0.03, 4.03, 6.08, 8.13,
  10.12 deg, with C_l, C_d and C_m (about the quarter chord).
* Transcription: direct reading of the scanned table page, reproduced
  and re-checked by `scripts/extract_ladson.py`.  The rendered page used
  for the check is `raw/ladson_tableIV_p14_300dpi.png`.

**Note on the 2.05 deg point.** Table IV prints four Reynolds-number
blocks (R = 3.90, 5.93, 8.96 and 11.90 x 10^6).  The 2.05 deg row exists
**only in the R = 3.90 x 10^6 block**; the R = 5.93 x 10^6 block has no
2.05 deg entry.  The revised DGSEM campaign still computes alpha = 2.05 deg
(it is one of the campaign's sweep angles), but that case has **no
point-to-point experimental counterpart** and is therefore excluded from
the experimental comparison and from the experimental least-squares fit.
It is retained in the campaign because it is a valid computed point.

## `ladson_tableIV_R3.90e6.csv` — provenance only

The complete R = 3.90 x 10^6 block of the same Table IV, transcribed for
traceability (it is the block that contains the 2.05 deg row).  It is
**not** used anywhere in the paper; in particular it is not mixed into the
R = 5.93 x 10^6 reference set.

## Earlier version of this data set (superseded)

The first release of this package carried (i) a Reynolds number label of
5.97 x 10^6 instead of 5.93 x 10^6, (ii) the 2.05 deg row taken from the
R = 3.90 x 10^6 block and silently inserted into the R = 5.93 x 10^6 set,
and (iii) a block labelled "Table I, M = 0.15, R = 5.97e6" whose values are
not Table I of TM-4074 at all (TM-4074 Table I is printed at
R = 2.00 x 10^6 and R = 3.94 x 10^6).  All three are corrected: the
production set now contains only the eight verified R = 5.93 x 10^6 rows,
and the unused M = 0.15 block has been removed from the package.

## Other raw material in `raw/`

* `ladson_tm4074.pdf` — the report.
* `ladson_tm4074.txt` — the NTRS OCR text layer (used only as a
  cross-check; the printed table is the authority).
* `ladson_tableIV_p14_300dpi.png` — rendered page carrying Table IV.
* `harris_tm81927.pdf`, `harris_tm81927.txt`, `harris_fig5k_M080_300dpi.png`
  — Harris (1981), cited for the transonic condition that the revised
  study does **not** attempt.
* `rm3726.pdf`, `rm3726.txt`, `rm3726_fig4_cp_300dpi.png`,
  `rm3726_fig5_nose_300dpi.png` — Gregory & O'Reilly (1970), cited for the
  low-speed baseline; not used numerically in this study.
