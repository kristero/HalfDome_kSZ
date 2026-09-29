"""Workstream A: integrated electron-pressure ratios Lee22 / Battaglia12 (both P_e, physical, same P200 and R200c):
Y_sph(<R) ∝ int_0^R P_e r^2 dr, for R = R500c-ish (0.65 R200c) and R200c, as painted and with the M1 (1+z)^2 factor.
Run: LD_LIBRARY_PATH=/home/cbllover/miniconda3/envs/halfdome/lib /home/cbllover/miniconda3/envs/halfdome/bin/python pressure_ratio_summary.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import quad

sys.path.insert(0, "/home/cbllover/HalfDome/frb_map_generation/lee22_normalization_forensics/C_pressure_consistency")
import lee22_models as LM  # noqa: E402

OUT = Path("/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_tsz_verification_20260929/A_xgpaint_audit")
H = LM.HALFDOME.h
res = {}
print("log10M  z   Ysph(<R200c) Lee22/B12  (as painted, x(1+z)^2)   Ysph(<4R200c) ratio (painted, x(1+z)^2)")
for lm in (13.0, 13.5, 14.0, 14.5, 15.0):
    for z in (0.0, 0.25, 0.5, 1.0, 1.5, 2.0):
        m = 10 ** lm
        def Y(f, R):
            return sum(quad(lambda x: f(x) * x * x, a, b, epsrel=1e-10, limit=200)[0]
                       for a, b in ((1e-6, 0.01), (0.01, 0.1), (0.1, 1.0), (1.0, R)) if b <= R or a < R)
        b12 = lambda x: LM.b12_Pe_over_P200(x, m, z)
        lee = lambda x: LM.lee_Pbar(x, m * H, z, which="noc", shape_clip_hinv=10 ** 14.8)
        r1 = Y(lee, 1.0) / Y(b12, 1.0)
        r4 = Y(lee, 4.0) / Y(b12, 4.0)
        res[f"logM{lm}_z{z}"] = {"Ysph_R200c_lee22_over_b12": r1, "Ysph_R200c_lee22x1pz2_over_b12": r1 * (1 + z) ** 2,
                                 "Ysph_4R200c_lee22_over_b12": r4, "Ysph_4R200c_lee22x1pz2_over_b12": r4 * (1 + z) ** 2,
                                 "in_lee22_mass_range": bool(1e13 <= m * H <= 10 ** 14.8)}
        print(f"{lm:5.1f} {z:4.2f}   {r1:6.3f}  {r1*(1+z)**2:6.3f}          {r4:6.3f}  {r4*(1+z)**2:6.3f}")
json.dump(res, open(OUT / "pressure_ratio_summary.json", "w"), indent=1)
