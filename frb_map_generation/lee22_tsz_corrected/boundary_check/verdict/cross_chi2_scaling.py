"""How much would removing the group_particles boundary bug move the y x DM chi^2 of Section 2.7?

Reads the stored annulus-binned model w(theta) of each (y, DM) pair (tsz_x_frbdm_corrected_annuli.csv, written by
lee22_tsz_corrected/plot_corrected_cross.py) and the real Takahashi+25 jackknife covariance through the same
fullsky_tsz_dm_comparison.observations(), recomputes chi^2 (must equal the stored value), then rescales the model by a
uniform factor s.  Removing the bug raises the Lee22 y by <= 0.27 % (Y_cyl(<4 R200c)) and the Lee22 DM by <= 0.33 %
(N_e in the R200c sphere) (propagate_fit_bias.txt), so s = 1.006 bounds the product; s = 1.003 is the y-only or
DM-only bound.  A uniform scale is conservative (the bias is concentrated at large radii).
Run from the repository root: python frb_map_generation/lee22_tsz_corrected/boundary_check/verdict/cross_chi2_scaling.py
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path("/home/cbllover/HalfDome")
sys.path.insert(0, str(REPO / "frb_map_generation"))
import fullsky_tsz_dm_comparison as fs  # noqa: E402

ANN = REPO / "frb_map_generation/outputs/lee22_corrected_20260930/analysis/tsz_x_frbdm_corrected_annuli.csv"
CHI = REPO / "frb_map_generation/outputs/lee22_corrected_20260930/analysis/tsz_x_frbdm_corrected_chi2.csv"
OUT = REPO / "frb_map_generation/outputs/lee22_boundary_check_20261007/verdict/cross_chi2_scaling.json"
PAIRS = [("b12", "lee22_noconc_phys_sphere1"), ("lee22p_m1", "lee22_noconc_phys_sphere1"),
         ("lee22p_m1_b12hi", "lee22_noconc_phys_sphere1"), ("lee22p_m1_calib", "lee22_noconc_phys_sphere1_calib")]
SCALES = [1.003, 1.006, 1.010]

rows = list(csv.DictReader(open(ANN)))
stored = {(r["survey"], r["y"], r["dm"]): float(r["chi2"]) for r in csv.DictReader(open(CHI))}
res = []
for plane in fs.SURVEYS:
    obs = fs.observations(plane)
    use = obs["lo"] >= fs.PAPER_CUT[plane] - 1e-9
    inv = np.linalg.inv(obs["covariance"][np.ix_(use, use)])
    chi2 = lambda m: float((m[use] - obs["w"][use]) @ inv @ (m[use] - obs["w"][use]))  # noqa: E731
    for y, dm in PAIRS:
        m = np.array([float(r["w_pc_cm3"]) for r in rows if (r["survey"], r["y"], r["dm"]) == (plane, y, dm)])
        assert len(m) == len(obs["w"]), (plane, y, dm, len(m))
        c0 = chi2(m)
        d = dict(survey=plane, y=y, dm=dm, chi2_recomputed=c0, chi2_stored=stored[(plane, y, dm)])
        for s in SCALES:
            d[f"dchi2_s{s}"] = chi2(s * m) - c0
        res.append(d)
        print(f"{plane:6s} {y:16s} {dm:34s} chi2 {c0:7.3f} (stored {stored[(plane, y, dm)]:7.3f}) "
              + " ".join(f"dchi2(s={s}) {d[f'dchi2_s{s}']:+.3f}" for s in SCALES))
json.dump(res, open(OUT, "w"), indent=1)
