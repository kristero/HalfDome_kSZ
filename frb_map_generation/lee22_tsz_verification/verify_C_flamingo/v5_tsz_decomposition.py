"""Second verifier: tSZ ratio table with the workstream's own bands (unique(round(geomspace(30,8000,31)))) and a
decomposition of its B12 'sphere-4R200c / production' conversion factor (0.975-0.981), which the workstream attributes to
"the known truncation-geometry effect".

  sphere4 / production = [sphere4 / projected4](same painter, truncation only) x [projected4(radius series) / production]
The radius-series painter used h = 0.6774, Omega_b = 0.0486, Omega_c = 0.2603 (provenance.txt); production used 0.68 /
0.049 / 0.261 (fullsky_spectra metadata). The radius series spectra are anafast niter 0; the production spectra map2alm
iter 3 (both raw, no pixel window).
Also: independent Lee22 / lc0 values and the halo-model transfer product.
Output: OUTV/v5_tsz_decomposition.json
"""
import json
from pathlib import Path

import numpy as np

HD = Path("/home/cbllover/HalfDome/flamingo_pressure_calibration/halfdome_tsz_comparison/products/comparison_products.npz")
FULL = Path("/home/cbllover/HalfDome/frb_map_generation/outputs/tsz_dm_fullsky_20260918/spectra/fullsky_spectra.npz")
RS = Path("/home/cbllover/HalfDome/truncation_comparison/radius_series/output_local/raw_cl_radius_series.csv")
BUD = Path("/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_normalization_forensics_20260928/F_sz_budget/budget_results.json")
OUTV = Path("/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_tsz_verification_20260929/verify_C_flamingo")
EDGES = np.unique(np.round(np.geomspace(30, 8000, 31)).astype(int))
CEN = np.sqrt(EDGES[:-1] * EDGES[1:])


def band(cl, ell):
    dl = ell * (ell + 1) * cl / (2 * np.pi)
    return np.array([dl[lo:hi].mean() for lo, hi in zip(EDGES[:-1], EDGES[1:])])


def main():
    P = np.load(HD)
    S = np.load(FULL)
    rs = np.genfromtxt(RS, delimiter=",", names=True)
    ell = np.arange(len(P["cl_flamingo_lc0"]), dtype=float)
    assert np.array_equal(P["ell"], ell) and np.array_equal(rs["ell"][: len(ell)], ell)
    import healpy as hp
    pw = hp.pixwin(4096, lmax=len(ell) - 1)
    lc0 = band(P["cl_flamingo_lc0"] / pw**2, ell)
    lc1 = band(P["cl_flamingo_lc1"] / pw**2, ell)
    b12s = band(P["cl_halfdome_b12"], ell)
    flf = band(P["cl_halfdome_flamingo"], ell)
    b12p = band(S["cl_y_b12"], ell)
    lee = band(S["cl_y_lee22p"], ell)
    leec = band(S["cl_y_lee22p_calib"], ell)
    s4 = band(rs["y_sphere4"][: len(ell)], ell)
    p4 = band(rs["y_projected4"][: len(ell)], ell)
    bud = json.loads(BUD.read_text())
    row = [r for r in bud["pressure_corrections"] if r["label"] == "1 (1+z)^2"][0]
    lh = np.array([30, 100, 300, 1000, 2000, 3000, 5000, 8000.0])
    tr = np.exp(np.interp(np.log(CEN), np.log(lh), np.log([row[f"validation_tsz_ratio_to_literal_l{int(l)}"] for l in lh])))
    out = {"products_cl_halfdome_b12_equals_radius_series_sphere4_maxrel": float(np.nanmax(np.abs(P["cl_halfdome_b12"][2:] / rs["y_sphere4"][2:len(ell)] - 1))),
           "pixwin_products_vs_healpy_maxabs": float(np.max(np.abs(P["pixwin"] - pw))), "rows": []}
    for L in (100, 300, 1000, 2000, 3000, 5000):
        k = int(np.argmin(np.abs(CEN - L)))
        conv = b12s[k] / b12p[k]
        rec = {"ell": L, "band": f"{EDGES[k]}-{EDGES[k + 1]}", "B12/lc0": b12s[k] / lc0[k], "FLfit/lc0": flf[k] / lc0[k],
               "Lee22/lc0": lee[k] * conv / lc0[k], "Lee22calib/lc0": leec[k] * conv / lc0[k],
               "Lee22a2_transfer/lc0": lee[k] * conv * tr[k] / lc0[k], "lc1/lc0": lc1[k] / lc0[k],
               "sphere4/production": conv, "truncation_only_sphere4/projected4": s4[k] / p4[k],
               "painter_inputs_projected4_rs/production": p4[k] / b12p[k], "Lee22/B12_same_painter": lee[k] / b12p[k]}
        out["rows"].append(rec)
        print({a: (round(b, 3) if isinstance(b, float) else b) for a, b in rec.items()})
    print({k: v for k, v in out.items() if k != "rows"})
    (OUTV / "v5_tsz_decomposition.json").write_text(json.dumps(out, indent=1, default=float) + "\n")


if __name__ == "__main__":
    main()
