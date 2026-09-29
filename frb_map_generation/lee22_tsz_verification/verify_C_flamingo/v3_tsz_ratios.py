"""Independent recomputation of workstream C's tSZ D_ell ratio table from the existing spectra (no anafast, no class_sz).

Bands here: +-10% windows around each target ell (different from the workstream's geometric bands), D_ell mean.
FLAMINGO lc0/lc1 divided by pixwin^2 (NSIDE 4096, from the products npz, cross-checked against healpy.pixwin).
Also: truncation-only ratio sphere4/projected4 from the radius-series run (same painter, same cosmology), which is what
the 'known truncation-geometry effect' should be, versus the B12 sphere4 (radius series, h = 0.6774) / B12 production
(h = 0.68) ratio that workstream C attributes to truncation geometry.
Output: OUTV/v3_tsz_ratios.json
"""
import json
from pathlib import Path

import healpy as hp
import numpy as np

HD = Path("/home/cbllover/HalfDome/flamingo_pressure_calibration/halfdome_tsz_comparison/products/comparison_products.npz")
FS = Path("/home/cbllover/HalfDome/frb_map_generation/outputs/tsz_dm_fullsky_20260918/spectra/fullsky_spectra.npz")
RS = Path("/home/cbllover/HalfDome/truncation_comparison/radius_series/output_local/raw_cl_radius_series.csv")
OUTV = Path("/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_tsz_verification_20260929/verify_C_flamingo")
ELLS = [100, 300, 1000, 2000, 3000, 5000]


def band(cl, l):
    ell = np.arange(len(cl), dtype=float)
    s = slice(int(round(0.9 * l)), int(round(1.1 * l)) + 1)
    return float(np.mean(ell[s] * (ell[s] + 1) * cl[s] / (2 * np.pi)))


def main():
    P = np.load(HD)
    S = np.load(FS, allow_pickle=True)
    pw = hp.pixwin(4096, lmax=8192)
    out = {"pixwin_products_vs_healpy_maxabs": float(np.max(np.abs(P["pixwin"] - pw))), "pixwin_at": {l: float(pw[l]) for l in ELLS}}
    lc0 = P["cl_flamingo_lc0"] / pw**2
    lc1 = P["cl_flamingo_lc1"] / pw**2
    rs = np.genfromtxt(RS, delimiter=",", names=True)
    tab = {}
    for l in ELLS:
        ref = band(lc0, l)
        conv = band(P["cl_halfdome_b12"], l) / band(S["cl_y_b12"], l)
        rec = {"Dl_lc0_1e12": 1e12 * ref, "lc1/lc0": band(lc1, l) / ref,
               "B12_sphere4/lc0": band(P["cl_halfdome_b12"], l) / ref,
               "FLfit/lc0": band(P["cl_halfdome_flamingo"], l) / ref,
               "B12prod/lc0": band(S["cl_y_b12"], l) / ref,
               "Lee22prod/lc0 (raw, production painter)": band(S["cl_y_lee22p"], l) / ref,
               "Lee22prod/lc0 (x sphere4/production B12)": band(S["cl_y_lee22p"], l) * conv / ref,
               "Lee22calib/lc0 (x conv)": band(S["cl_y_lee22p_calib"], l) * conv / ref,
               "B12_sphere4/B12_production": conv,
               "truncation_only_sphere4/projected4 (radius series)": float(np.mean(rs["y_sphere4"][int(0.9 * l):int(1.1 * l) + 1]
                                                                                  * 1) / np.mean(rs["y_projected4"][int(0.9 * l):int(1.1 * l) + 1])),
               "radius_series_sphere4_vs_products_cl_halfdome_b12": band(P["cl_halfdome_b12"], l) /
                   float(np.mean([li * (li + 1) / (2 * np.pi) * rs["y_sphere4"][li] for li in range(int(round(0.9 * l)), int(round(1.1 * l)) + 1)])),
               "radius_series_projected4 / B12 production (painter cosmology h 0.6774 vs 0.68)":
                   float(np.mean([li * (li + 1) / (2 * np.pi) * rs["y_projected4"][li] for li in range(int(round(0.9 * l)), int(round(1.1 * l)) + 1)]))
                   / band(S["cl_y_b12"], l)}
        tab[l] = rec
    out["table"] = tab
    OUTV.mkdir(parents=True, exist_ok=True)
    (OUTV / "v3_tsz_ratios.json").write_text(json.dumps(out, indent=1) + "\n")
    keys = list(tab[ELLS[0]])
    for k in keys:
        print(f"{k:80s}", " ".join(f"{tab[l][k]:7.4f}" for l in ELLS))
    print("pixwin check", out["pixwin_products_vs_healpy_maxabs"], out["pixwin_at"])


if __name__ == "__main__":
    main()
