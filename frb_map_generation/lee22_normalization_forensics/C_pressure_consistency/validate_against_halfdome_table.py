#!/usr/bin/env python3
"""Validate the numpy Lee22 Table-8 formula (lee22_models.lee_nbar) against the production XGPaint
tabulation outputs/analytic_halo_pdf_20260925/tables/halfdome_sphere_profiles.h5
(lee22_noconc_xgpnative; literal = xgpnative / 0.6016316602183194).

rho_crit(z) is backed out of the stored R200c and M200c so that the check is independent of the
radiation/neutrino terms in XGPaint's Cosmology.jl background.
"""
import json
from pathlib import Path

import h5py
import numpy as np
from scipy import integrate

import lee22_models as L

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
TABLE = REPO / "frb_map_generation/outputs/analytic_halo_pdf_20260925/tables/halfdome_sphere_profiles.h5"
OUT = REPO / "frb_map_generation/outputs/lee22_normalization_forensics_20260928/C_pressure_consistency"
XGP_OVER_LIT = 0.6016316602183194
H = 0.68
FB = 0.049 / 0.31


def main():
    with h5py.File(TABLE, "r") as f:
        logm = f["log10_m200c_msun"][:]
        zs = f["redshift"][:]
        xs = f["x_r200c"][:]
        g = f["lee22_noconc_xgpnative/chord_mean_dm_pc_cm3"]
        r200 = f["lee22_noconc_xgpnative/r200c_mpc"]
        worst = 0.0
        rows = []
        for im in (300, 350, 400, 461, 500, 560, 600, 650, 699):   # log10 M = 12.0 ... 15.995
            for iz in (0, 20, 50, 99):
                m = 10 ** logm[im]
                z = zs[iz]
                R = r200[im, iz] * L.MPC
                rho_c = 3 * m * L.MSUN / (4 * np.pi * 200 * R ** 3)
                n200 = 200 * rho_c * FB / (L.X_H * L.M_P)
                for ix in (5, 60, 100, 150, 199):
                    x = xs[ix]
                    lmax = np.sqrt(1 - x * x)
                    ne = lambda l: L.lee_nbar(np.sqrt(x * x + l * l), m * H, z) * n200
                    val, _ = integrate.quad(ne, 0, lmax, epsabs=0, epsrel=1e-11, limit=200)
                    gm = R / L.PC / (1 + z) * val / lmax
                    gt = g[im, iz, ix] / XGP_OVER_LIT
                    rel = gm / gt - 1
                    worst = max(worst, abs(rel))
                    rows.append((float(logm[im]), float(z), float(x), float(gt), float(gm), float(rel)))
    res = {"table": str(TABLE), "n_points": len(rows), "max_abs_relative_difference": worst,
           "note": "numpy Table-8 literal (Mcut pivot, shape clip 10^14.8 h^-1) vs XGPaint table / 0.6016316602183194"}
    (OUT / "validation_vs_halfdome_table.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    for r in rows[:: max(1, len(rows) // 12)]:
        print("logM=%.3f z=%.3f x=%.4f table=%.6e numpy=%.6e rel=%.2e" % r)


if __name__ == "__main__":
    main()
