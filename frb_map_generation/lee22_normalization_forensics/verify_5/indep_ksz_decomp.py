#!/usr/bin/env python3
"""verify_5: where does the kSZ excess of the TNG-calibrated literal Lee22 over B16 come from?
Catalogue-weighted 1-halo sums (indep_halo_sums_tables.npz): 3R200c vs 1R200c spheres; calibrated
window (1e13/h <= M200c < 10^14.8/h, z <= 2) vs extrapolated halos; ratio of electron counts."""
import json
from pathlib import Path

import numpy as np

OUT = Path("/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_normalization_forensics_20260928/verify_5")
H = 0.68
TCMB_UK, C_KMS = 2.7255e6, 299792.458
C0, K = 0.2841790493880425, 2.483190184608004
XGP = 0.6016316602183194


def main():
    d = np.load(OUT / "indep_halo_sums_tables.npz")
    ells = d["ells"]; V2 = d["V2"]; M = d["M"]; z = d["z"]; N = d["N"]
    fac = ells * (ells + 1) / (2 * np.pi)
    wv = V2 / C_KMS ** 2 * TCMB_UK ** 2
    C = C0 * (1 + z) ** K
    win = (M >= 1e13 / H) & (M < 10 ** 14.8 / H) & (z <= 2.0)
    win_konietzka = (M >= 1e13) & (M < 1e14)

    def Dk(tau, w=1.0, sel=None):
        ww = wv * w * (1.0 if sel is None else sel)
        return fac * (ww[:, None] * tau ** 2).sum(0) / (4 * np.pi)

    rows = {}
    for i, l in enumerate(ells):
        r = {}
        for tag, tl, tb in (("3r", d["tau_lit"], d["tau_b16"]), ("1r", d["tau_lit_1r"], d["tau_b16_1r"])):
            r["litC_over_b16_" + tag] = float(Dk(tl, C ** 2)[i] / Dk(tb)[i])
            r["lit_over_b16_" + tag] = float(Dk(tl)[i] / Dk(tb)[i])
            r["litC_over_b16_" + tag + "_window"] = float(Dk(tl, C ** 2, win)[i] / Dk(tb, 1.0, win)[i])
            r["litC_over_b16_" + tag + "_konietzka_window"] = float(Dk(tl, C ** 2, win_konietzka)[i] / Dk(tb, 1.0, win_konietzka)[i])
            r["window_share_litC_" + tag] = float(Dk(tl, C ** 2, win)[i] / Dk(tl, C ** 2)[i])
            r["window_share_b16_" + tag] = float(Dk(tb, 1.0, win)[i] / Dk(tb)[i])
        r["D_litC_3r_uK2"] = float(Dk(d["tau_lit"], C ** 2)[i]); r["D_b16_3r_uK2"] = float(Dk(d["tau_b16"])[i])
        r["D_lit_3r_uK2"] = float(Dk(d["tau_lit"])[i]); r["D_xgp_3r_uK2"] = float(Dk(d["tau_lit"])[i] * XGP ** 2)
        rows[int(l)] = r
    # per-halo electron counts (l -> 0 limit of tau_l times D_A^2/sigma_T): ratios at representative bins
    ne = {}
    t0 = d["tau_lit"][:, 0]; t1 = d["tau_lit_1r"][:, 0]; b0 = d["tau_b16"][:, 0]; b1 = d["tau_b16_1r"][:, 0]
    for zc in (0.05, 0.2, 0.5):
        for lmh in (13.0, 13.5, 14.0, 14.5):
            k = np.argmin((np.log10(z) - np.log10(zc)) ** 2 / 0.01 ** 2 + (np.log10(M) - (lmh - np.log10(H))) ** 2 / 0.02 ** 2)
            Ck = C0 * (1 + z[k]) ** K
            ne["z{}_log10Mh{}".format(zc, lmh)] = dict(z=float(z[k]), log10M=float(np.log10(M[k])),
                                                       litC_over_b16_Ne_1R=float(Ck * t1[k] / b1[k]),
                                                       litC_over_b16_Ne_3R=float(Ck * t0[k] / b0[k]),
                                                       lit_Ne3R_over_Ne1R=float(t0[k] / t1[k]), b16_Ne3R_over_Ne1R=float(b0[k] / b1[k]))
    res = dict(rows=rows, electron_counts_l30_limit=ne,
               note="1-halo, catalogue-weighted, literal x C_TNG(z) = 0.2842(1+z)^2.483 vs XGPaint B16; ratios of D_ell")
    (OUT / "indep_ksz_decomp.json").write_text(json.dumps(res, indent=1))
    keys = ("lit_over_b16_3r", "litC_over_b16_3r", "litC_over_b16_1r", "litC_over_b16_3r_window", "litC_over_b16_1r_window",
            "litC_over_b16_3r_konietzka_window", "litC_over_b16_1r_konietzka_window", "window_share_litC_3r", "window_share_b16_3r",
            "D_litC_3r_uK2", "D_xgp_3r_uK2", "D_b16_3r_uK2")
    print("l " + " ".join(keys))
    for l, r in rows.items():
        print(l, " ".join("{:.3f}".format(r[k]) for k in keys))
    for k, v in ne.items():
        print(k, {a: round(b, 3) for a, b in v.items()})


if __name__ == "__main__":
    main()
