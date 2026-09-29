#!/usr/bin/env python3
"""verify_5: analysis of the independent catalogue-weighted 1-halo sums (indep_halo_sums_tables.npz)."""
import json
from pathlib import Path

import numpy as np

OUT = Path("/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_normalization_forensics_20260928/verify_5")
H = 0.68
CUT = 10 ** 14.8 / H
TCMB_UK, C_KMS = 2.7255e6, 299792.458
C0, K = 0.2841790493880425, 2.483190184608004


def main():
    d = np.load(OUT / "indep_halo_sums_tables.npz")
    ells = d["ells"]; N = d["N"]; V2 = d["V2"]; M = d["M"]; z = d["z"]
    fac = ells * (ells + 1) / (2 * np.pi)
    hi = M >= CUT; lo = ~hi
    res = {"ells": ells.tolist()}

    def Dy(y, sel=None, w=None):
        ww = N if w is None else N * w
        if sel is not None:
            ww = ww * sel
        return fac * (ww[:, None] * y ** 2).sum(0) / (4 * np.pi)

    yb, yl = d["y_b12"], d["y_lee22p"]
    t = {}
    t["D_b12_all"] = Dy(yb); t["D_b12_low"] = Dy(yb, lo); t["D_b12_high"] = Dy(yb, hi)
    t["D_l22_all"] = Dy(yl); t["D_l22_low"] = Dy(yl, lo); t["D_l22_high"] = Dy(yl, hi)
    z2 = (1 + z) ** 4                                  # (1+z)^2 on y -> (1+z)^4 on |y|^2
    t["D_l22z2_all"] = Dy(yl, w=z2); t["D_l22z2_low"] = Dy(yl, lo, w=z2); t["D_l22z2_high"] = Dy(yl, hi, w=z2)
    t["D_hybrid_1h"] = t["D_l22_low"] + t["D_b12_high"]
    rt = {}
    for i, l in enumerate(ells):
        rt[int(l)] = dict(all=t["D_l22_all"][i] / t["D_b12_all"][i], low=t["D_l22_low"][i] / t["D_b12_low"][i],
                          high=t["D_l22_high"][i] / t["D_b12_high"][i],
                          frac_l22_high=t["D_l22_high"][i] / t["D_l22_all"][i],
                          frac_b12_high=t["D_b12_high"][i] / t["D_b12_all"][i],
                          hybrid_over_b12=t["D_hybrid_1h"][i] / t["D_b12_all"][i],
                          z2corr_all=t["D_l22z2_all"][i] / t["D_b12_all"][i],
                          z2corr_low=t["D_l22z2_low"][i] / t["D_b12_low"][i])
    res["tsz_1h_catalogue"] = rt
    # shape-only test at fixed (M, z): per-bin |y_l|^2 ratio for in-range bins near fixed z
    shape = {}
    for zc in (0.1, 0.3, 0.6):
        for lmc in (13.5, 14.0, 14.5):
            k = np.argmin((np.log10(z) - np.log10(zc)) ** 2 / 0.01 ** 2 + (np.log10(M) - (lmc - np.log10(H))) ** 2 / 0.02 ** 2)
            r = (yl[k] / yb[k]) ** 2
            shape["z{}_log10Mh{}".format(zc, lmc)] = dict(z=float(z[k]), log10M=float(np.log10(M[k])),
                                                          ratio_l300=float(r[ells == 300][0]), ratio_l1000=float(r[ells == 1000][0]),
                                                          ratio_l3000=float(r[ells == 3000][0]), ratio_l8000=float(r[ells == 8000][0]))
    res["tsz_fixed_Mz_power_ratio_lee22p_over_b12"] = shape

    # kSZ
    tl, tb = d["tau_lit"], d["tau_b16"]
    wv = V2 / C_KMS ** 2 * TCMB_UK ** 2
    C = C0 * (1 + z) ** K
    Cf = C0 * (1 + np.minimum(z, 2.0)) ** K

    def Dk(tau, w=1.0):
        return fac * ((wv * w)[:, None] * tau ** 2).sum(0) / (4 * np.pi)

    k = {"D_lit": Dk(tl), "D_b16": Dk(tb), "D_litC": Dk(tl, C ** 2), "D_litCf": Dk(tl, Cf ** 2),
         "D_xgp": Dk(tl) * 0.6016316602183194 ** 2}
    rk = {}
    order = np.argsort(z)
    for i, l in enumerate(ells):
        contrib = wv * tl[:, i] ** 2
        cz = np.cumsum(contrib[order]) / contrib.sum()
        zs = z[order]
        wC2 = (contrib * C ** 2).sum() / contrib.sum()
        cb = wv * tb[:, i] ** 2
        czb = np.cumsum(cb[order]) / cb.sum()
        rk[int(l)] = dict(D_lit=k["D_lit"][i], D_b16=k["D_b16"][i], D_litC=k["D_litC"][i], D_litCf=k["D_litCf"][i],
                          lit_over_b16=k["D_lit"][i] / k["D_b16"][i], xgp_over_b16=k["D_xgp"][i] / k["D_b16"][i],
                          litC_over_b16=k["D_litC"][i] / k["D_b16"][i], litCf_over_b16=k["D_litCf"][i] / k["D_b16"][i],
                          lit_over_litC=k["D_lit"][i] / k["D_litC"][i],
                          z_median_lit=float(np.interp(0.5, cz, zs)), frac_lit_z_lt_0p5=float(np.interp(0.5, zs, cz)),
                          frac_lit_z_lt_0p1=float(np.interp(0.1, zs, cz)),
                          z_median_b16=float(np.interp(0.5, czb, zs)),
                          powerweighted_C2=float(wC2), effective_lit_over_TNG_electrons=float(1 / np.sqrt(wC2)),
                          frac_litC_from_z_gt_2=float((contrib * C ** 2)[z > 2].sum() / (contrib * C ** 2).sum()))
    res["ksz_1h_catalogue"] = rk
    res["C_TNG_at"] = {str(zz): C0 * (1 + zz) ** K for zz in (0.0, 0.05, 0.079, 0.1, 0.188, 0.39, 0.5, 1.0, 2.0)}
    (OUT / "indep_analysis.json").write_text(json.dumps(res, indent=1))
    print("tSZ 1-halo (catalogue): l  all  low  high  fracL22high  fracB12high  hybrid/B12  z2corr_all  z2corr_low")
    for l, r in rt.items():
        print(l, " ".join("{:.3f}".format(r[x]) for x in ("all", "low", "high", "frac_l22_high", "frac_b12_high",
                                                            "hybrid_over_b12", "z2corr_all", "z2corr_low")))
    print("fixed-(M,z) |y_l|^2 ratio Lee22p/B12:")
    for kk, v in shape.items():
        print(kk, {a: round(b, 3) for a, b in v.items()})
    print("kSZ 1-halo (catalogue): l  D_lit  D_b16  lit/B16  xgp/B16  litC/B16  litCf/B16  lit/litC  zmed_lit  f(z<0.5)  f(z<0.1)  zmed_b16  lit/TNG_eff  fz>2(litC)")
    for l, r in rk.items():
        print(l, " ".join("{:.3f}".format(r[x]) for x in ("D_lit", "D_b16", "lit_over_b16", "xgp_over_b16", "litC_over_b16",
                                                            "litCf_over_b16", "lit_over_litC", "z_median_lit", "frac_lit_z_lt_0p5",
                                                            "frac_lit_z_lt_0p1", "z_median_b16", "effective_lit_over_TNG_electrons",
                                                            "frac_litC_from_z_gt_2")))


if __name__ == "__main__":
    main()
