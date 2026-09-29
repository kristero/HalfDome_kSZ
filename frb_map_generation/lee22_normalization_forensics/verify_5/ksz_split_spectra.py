#!/usr/bin/env python3
"""verify_5: spectra of the repainted literal-Lee22 kSZ maps (redshift split + C_TNG(z) variants).

Same estimator as the published comparison (healpy map2alm lmax 6000, 3 iterations, monopole removed).
Checks: sum of the five z-bin maps vs 2.7627 x published xgpnative spectrum.
Outputs: per-band D_ell (2l+1 weighted), z-budget (symmetric split of cross terms), C_TNG/B16 ratios."""
import json
import time
from pathlib import Path

import h5py
import healpy as hp
import numpy as np

R = Path("/home/cbllover/HalfDome/frb_map_generation")
OUT = R / "outputs/lee22_normalization_forensics_20260928/verify_5"
PUB = R / "outputs/lee22_pdf_ksz_comparison_20260922/ksz"
LMAX = 6000
TCMB = 2.7255e6
ZB = ("lit_z0p00_0p10", "lit_z0p10_0p25", "lit_z0p25_0p50", "lit_z0p50_1p00", "lit_z1p00_inf")
ZEDGE = np.array([0.0, 0.1, 0.25, 0.5, 1.0, 3.9])
BANDS = ((300, 327), (500, 550), (1000, 1050), (1500, 1550), (2000, 2050), (2950, 3050), (4000, 4100), (4950, 5050))


def main():
    t0 = time.time()
    path = OUT / "ksz_literal_zsplit_ctng_nside4096.h5"
    alm = {}
    means = {}
    with h5py.File(path, "r") as f:
        names = list(f.keys())
        for name in names:
            m = f[name][:]
            means[name] = float(m.mean())
            alm[name] = hp.map2alm(m - m.mean(), lmax=LMAX, iter=3, pol=False)
            del m
            print(name, "alm", round(time.time() - t0, 1), flush=True)
    tot = sum(alm[n] for n in ZB)
    ell = np.arange(LMAX + 1)
    cl = {"lit_total": hp.alm2cl(tot), "litC_tng_extrap": hp.alm2cl(alm["litC_tng_extrap"]),
          "litC_tng_frozen_z2": hp.alm2cl(alm["litC_tng_frozen_z2"])}
    cross = np.empty((5, 5, LMAX + 1))
    for i in range(5):
        for j in range(i, 5):
            cross[i, j] = cross[j, i] = hp.alm2cl(alm[ZB[i]], alm[ZB[j]])
    pub = np.load(PUB / "map_spectra.npz")
    cl_b16 = pub["cl_full_battaglia16"]; cl_xgp = pub["cl_full_lee22_previous"]; cl_lit_pub = pub["cl_full_lee22_paper"]
    np.savez_compressed(OUT / "ksz_split_cl.npz", ell=ell, cross=cross, cl_b16_pub=cl_b16, cl_lit_pub=cl_lit_pub,
                        **{"cl_" + k: v for k, v in cl.items()})

    def D(c, lo, hi):
        l = ell[lo:hi]; w = 2 * l + 1
        return float((w * l * (l + 1) / (2 * np.pi) * c[lo:hi] * TCMB ** 2).sum() / w.sum())

    rows = []
    for lo, hi in BANDS:
        r = dict(band=[lo, hi], D_lit=D(cl["lit_total"], lo, hi), D_lit_pub=D(cl_lit_pub, lo, hi),
                 D_b16_pub=D(cl_b16, lo, hi), D_xgp_pub=D(cl_xgp, lo, hi),
                 D_litC_extrap=D(cl["litC_tng_extrap"], lo, hi), D_litC_frozen=D(cl["litC_tng_frozen_z2"], lo, hi))
        r["check_lit_over_pub"] = r["D_lit"] / r["D_lit_pub"]
        r["lit_over_b16"] = r["D_lit"] / r["D_b16_pub"]
        r["xgp_over_b16"] = r["D_xgp_pub"] / r["D_b16_pub"]
        r["litC_extrap_over_b16"] = r["D_litC_extrap"] / r["D_b16_pub"]
        r["litC_frozen_over_b16"] = r["D_litC_frozen"] / r["D_b16_pub"]
        r["litC_extrap_over_lit"] = r["D_litC_extrap"] / r["D_lit"]
        # z budget: symmetric split of the cross terms
        Dij = np.array([[D(cross[i, j], lo, hi) for j in range(5)] for i in range(5)])
        share = Dij.sum(1) / Dij.sum()
        auto = np.diag(Dij) / Dij.sum()
        r["zbin_share_symsplit"] = share.tolist()
        r["zbin_share_auto_only"] = auto.tolist()
        r["cross_fraction"] = float(1 - np.trace(Dij) / Dij.sum())
        cum = np.concatenate([[0], np.cumsum(share)])
        r["frac_z_lt_0p5"] = float(cum[3]); r["frac_z_lt_0p25"] = float(cum[2]); r["frac_z_lt_0p1"] = float(cum[1])
        r["z_median_linear_interp"] = float(np.interp(0.5, cum, ZEDGE))
        rows.append(r)
    res = dict(means=means, bands=rows, elapsed_s=time.time() - t0,
               note="D in uK^2, (2l+1)-weighted band means; B16/xgpnative/literal-published from map_spectra.npz")
    (OUT / "ksz_split_results.json").write_text(json.dumps(res, indent=1))
    for r in rows:
        print(r["band"], "chk %.4f" % r["check_lit_over_pub"], "lit/B16 %.2f" % r["lit_over_b16"],
              "litC/B16 %.3f frozen %.3f" % (r["litC_extrap_over_b16"], r["litC_frozen_over_b16"]),
              "D_litC %.3f" % r["D_litC_extrap"], "share", np.round(r["zbin_share_symsplit"], 3),
              "zmed %.3f" % r["z_median_linear_interp"], "cross %.3f" % r["cross_fraction"])


if __name__ == "__main__":
    main()
