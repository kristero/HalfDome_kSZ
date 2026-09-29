#!/usr/bin/env python3
"""Per-halo integrated Compton-Y (4 R200c projected aperture, as painted) of Lee22-as-painted relative to Battaglia12,
from the tabulated y_ell grids at the lowest multipole (ell = 10, where y_ell -> Y_ap / D_A^2 for all but the nearest
halos); added in the 2026-09-29 continuation. Shows how the Lee22 P0 extrapolation (M^0.80 above Mcut vs B12 M^0.154)
makes the rare halos above the fit range (10^14.8 h^-1 Msun) up to ~2-3x brighter than B12.
Output: OUT/analysis/per_halo_Y_ratio.json
Run: LD_LIBRARY_PATH=<halfdome env>/lib python per_halo_Y_ratio.py
"""
import json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, "/home/cbllover/HalfDome/frb_map_generation/lee22_tsz_verification/B_class_sz")
import common as C  # noqa: E402

HMD = Path(C.OUT) / "halo_model"
if __name__ == "__main__":
    gb = np.load(HMD / "ygrid_hm_b12_aperture.npz"); gl = np.load(HMD / "ygrid_hm_lee22_aperture.npz")
    z, lnm = gb["z"], gb["lnm"]
    M = np.exp(lnm)
    out = {"note": "ratio of y_ell at ell = 10 (= Y within the 4 R200c aperture, full LOS); M200c in Msun/h, h = 0.68",
           "rows": []}
    for zt in (0.03, 0.1, 0.3, 0.6, 1.0, 2.0):
        j = int(np.argmin(np.abs(z - zt)))
        for mt in (1e13, 3e13, 1e14, 3e14, C.LEE_CLIP_HINV, 1e15, 2e15):
            i = int(np.argmin(np.abs(np.log(M / mt))))
            r = float(gl["y"][j, i, 0] / gb["y"][j, i, 0])
            out["rows"].append(dict(z=float(z[j]), M200c_msunh=float(M[i]), Y_lee22_over_b12=r))
            print(f"z={z[j]:.3f} M={M[i]:.2e} Msun/h  Y_Lee22/Y_B12 = {r:.3f}")
    json.dump(out, open(Path(C.OUT) / "analysis" / "per_halo_Y_ratio.json", "w"), indent=1)
