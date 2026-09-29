#!/usr/bin/env python3
"""Rough size of the point-sampling aliasing in the painted NSIDE 4096 y maps (added in the 2026-09-29 continuation).

The maps are point samples of y(theta) at pixel centres (no pixel-area average), so their C_ell carries no pixel window
but does contain power aliased from ell above the HEALPix band limit. Order-of-magnitude model: the aliased power is
white, C_alias = Omega_pix * sum_{ell > L_N} (2 ell + 1)/(4 pi) C_ell, for L_N = 8192 (map2alm lmax) and 12288 (3 NSIDE).
C_ell above 8192 is the Python Tinker08 1-halo term (CLASS-SZ ingredients, XGPaint-like 4 R200c projected aperture)
evaluated to ell = 1e5 (2-halo negligible there).
Output: OUT/analysis/aliasing_estimate.json
Run: LD_LIBRARY_PATH=<halfdome env>/lib OMP_NUM_THREADS=1 python aliasing_estimate.py
"""
import json, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, "/home/cbllover/HalfDome/frb_map_generation/lee22_tsz_verification/B_class_sz")
import common as C  # noqa: E402
import halo_model as HM  # noqa: E402

OUTD = Path(C.OUT) / "analysis"; OUTD.mkdir(parents=True, exist_ok=True)
NSIDE = 4096
OMEGA_PIX = 4 * np.pi / (12 * NSIDE ** 2)

if __name__ == "__main__":
    ing = HM.Ingredients()
    ell_hi = np.geomspace(6000.0, 1.0e5, 22)
    res = {"nside": NSIDE, "omega_pix_sr": OMEGA_PIX, "ell_hi": ell_hi.tolist(), "models": {}}
    for model in ("b12", "lee22"):
        t0 = time.time()
        r = HM.halo_model_cl(ing, model, kind="aperture", X=4.0, ell=ell_hi, n_z=6, n_m=6)
        c1 = r["cl_1h"]
        L = np.arange(6000, 100001)
        cl_int = np.exp(np.interp(np.log(L), np.log(ell_hi), np.log(c1)))
        out = {"cl_1h_hi": c1.tolist(), "seconds": time.time() - t0}
        for LN in (8192, 12288):
            s = L > LN
            var = float(np.sum((2 * L[s] + 1) / (4 * np.pi) * cl_int[s]))
            out[f"C_alias_white_LN{LN}"] = var * OMEGA_PIX
            out[f"variance_above_LN{LN}"] = var
        res["models"][model] = out
        print(model, {k: v for k, v in out.items() if k.startswith("C_alias")}, f"{out['seconds']:.0f} s", flush=True)
    # fraction of the stored map spectra (band means, raw) that the white floor would represent
    fs = np.load(C.FULLSKY_SPECTRA, allow_pickle=True)
    frac = {}
    for key, model in (("cl_y_b12", "b12"), ("cl_y_lee22p", "lee22")):
        cl = fs[key]
        frac[key] = {str(l): {str(LN): res["models"][model][f"C_alias_white_LN{LN}"] / float(cl[int(0.9 * l):int(1.1 * l) + 1].mean())
                              for LN in (8192, 12288)} for l in (300, 1000, 2000, 3000, 5000, 8000)}
    res["alias_over_map_band"] = frac
    json.dump(res, open(OUTD / "aliasing_estimate.json", "w"), indent=1)
    print(json.dumps(frac, indent=1))
