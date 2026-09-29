#!/usr/bin/env python3
"""CLASS-SZ's tabulated y_ell(M, z) (x_outSZ = 4 sphere) at default and high precision (N_samp_fftw 4096 -> 8192)
against the exact 3-D transform of halo_model.py, at M200c = 1e13, 1e14, 1e15 Msun, z = 0.1, 0.5, 1, 2,
ell = 100, 300, 1000, 3000 (added in the 2026-09-29 continuation).
Output: OUT/profiles/yell_class_sz_precision_check.json
Run: LD_LIBRARY_PATH=<halfdome env>/lib python yell_precision_check.py
"""
import json, sys, warnings
import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/cbllover/HalfDome/frb_map_generation/lee22_tsz_verification/B_class_sz")
import common as C  # noqa: E402
import halo_model as HM  # noqa: E402

if __name__ == "__main__":
    ing = HM.Ingredients()
    out = {}
    for run, tag, model in (("b12_paint_all", "b12", "b12"), ("b12_paint_all_hiprec", "b12", "b12"),
                            ("lee22_paint_low_hiprec", "lee22_low", "lee22")):
        d = np.load(C.OUT + f"/class_sz/{run}.npz")
        M, Z, ell, yc = d["test_M_msun"], d["test_z"], d["test_ell"], d[f"yell_{tag}"]
        rs = []
        for i, m in enumerate(M):
            if tag == "lee22_low" and m * C.PAINT["h"] >= C.LEE_CLIP_HINV:
                continue
            for j, z in enumerate(Z):
                y = HM.y_ell_grid(ing, model, np.array([z]), np.log(np.array([m * C.PAINT["h"]])), ell)[0, 0]
                rs.append(yc[i, j] / y)
        rs = np.array(rs)
        out[run] = dict(min=float(np.nanmin(rs)), max=float(np.nanmax(rs)), mean_by_ell=np.nanmean(rs, 0).tolist(),
                        ell=ell.tolist())
        print(run, "CLASS-SZ/exact y_ell range", round(np.nanmin(rs), 4), round(np.nanmax(rs), 4), "mean by ell",
              np.round(np.nanmean(rs, 0), 4))
    json.dump(out, open(C.OUT + "/profiles/yell_class_sz_precision_check.json", "w"), indent=1)
