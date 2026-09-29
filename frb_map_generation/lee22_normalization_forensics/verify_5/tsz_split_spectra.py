#!/usr/bin/env python3
"""verify_5: spectra of the tSZ mass-split maps (claim C6 tSZ refutation route).

Maps (all NSIDE 4096 RING, XGPaint projected 4R200c aperture, h=0.68 paint contract, all 85,224,251 halos):
  L_full : existing Lee22 no-c pressure map (tsz_dm_cross_updated_20260917/inputs/...)
  L_high : Lee22 pressure, M200c >= 10^14.8/0.68 Msun   (painted here, same painter + same cache)
  B_high : Battaglia12, M >= cut                       (painted here)
  B_low  : Battaglia12, M <  cut                       (painted here)
  B_ref  : existing Battaglia12 full map (tsz_frb_1m_z1_r200c_apertures, 2026-09-09) -> consistency check
Derived (SHT is linear, so alm combinations are exact):
  L_low = L_full - L_high ; B = B_low + B_high ; hybrid = L_low + B_high.
"""
import json
import sys
import time
from pathlib import Path

import healpy as hp
import numpy as np

R = Path("/home/cbllover/HalfDome/frb_map_generation")
OUT = R / "outputs/lee22_normalization_forensics_20260928/verify_5"
MAPS = dict(
    L_full=R / "outputs/tsz_dm_cross_updated_20260917/inputs/lee22_noconc_pressure_compton_y_allz_nside4096_m200c_r200cx4.fits",
    L_high=OUT / "maps/lee22p_high.fits",
    B_high=OUT / "maps/b12_high.fits",
    B_low=OUT / "maps/b12_low.fits",
    B_ref=R / "outputs/tsz_frb_1m_z1_r200c_apertures/maps/battaglia12_fiducial_halfdome_compton_y_allz_nside4096_m200c_r200cx4.fits",
)
LMAX = 8192
BANDS = (30, 100, 300, 1000, 2000, 3000, 5000, 8000)


def main():
    t0 = time.time()
    checks = {}
    alm = {}
    means = {}
    # consistency checks at the map level (one or two maps in memory at a time)
    Lf = hp.read_map(str(MAPS["L_full"]), dtype=np.float64)
    Lh = hp.read_map(str(MAPS["L_high"]), dtype=np.float64)
    Llow = Lf - Lh
    checks["L_low_min"] = float(Llow.min())
    checks["L_low_min_over_Lfull_max"] = float(Llow.min() / Lf.max())
    checks["n_pix_L_low_negative_below_-1e-15"] = int((Llow < -1e-15).sum())
    checks["mean_y"] = dict(L_full=float(Lf.mean()), L_high=float(Lh.mean()), L_low=float(Llow.mean()))
    del Llow
    for key, m in (("L_full", Lf), ("L_high", Lh)):
        means[key] = float(m.mean())
        alm[key] = hp.map2alm(m - m.mean(), lmax=LMAX, iter=3, pol=False)
        print(key, "alm done", round(time.time() - t0, 1), flush=True)
    del Lf, Lh
    Bl = hp.read_map(str(MAPS["B_low"]), dtype=np.float64)
    Bh = hp.read_map(str(MAPS["B_high"]), dtype=np.float64)
    Bref = hp.read_map(str(MAPS["B_ref"]), dtype=np.float64)
    Bsum = Bl + Bh
    d = Bsum - Bref
    checks["B_sum_vs_B_ref"] = dict(max_abs_diff=float(np.abs(d).max()), max_abs_ref=float(np.abs(Bref).max()),
                                    rel_L2=float(np.sqrt((d ** 2).sum() / (Bref ** 2).sum())),
                                    mean_sum=float(Bsum.mean()), mean_ref=float(Bref.mean()))
    del d, Bsum
    checks["mean_y"].update(B_low=float(Bl.mean()), B_high=float(Bh.mean()), B_ref=float(Bref.mean()))
    for key, m in (("B_low", Bl), ("B_high", Bh), ("B_ref", Bref)):
        means[key] = float(m.mean())
        alm[key] = hp.map2alm(m - m.mean(), lmax=LMAX, iter=3, pol=False)
        print(key, "alm done", round(time.time() - t0, 1), flush=True)
    del Bl, Bh, Bref

    a = alm
    comb = dict(L=a["L_full"], Lh=a["L_high"], Ll=a["L_full"] - a["L_high"], Bh=a["B_high"], Bl=a["B_low"],
                B=a["B_low"] + a["B_high"], Bref=a["B_ref"], hyb=a["L_full"] - a["L_high"] + a["B_high"])
    cl = {k: hp.alm2cl(v) for k, v in comb.items()}
    cl["Ll_x_Lh"] = hp.alm2cl(comb["Ll"], comb["Lh"])
    cl["Bl_x_Bh"] = hp.alm2cl(comb["Bl"], comb["Bh"])
    np.savez_compressed(OUT / "tsz_split_cl.npz", ell=np.arange(LMAX + 1), **{"cl_" + k: v for k, v in cl.items()})

    # published painted spectra for cross-checking
    ann = np.load(R / "outputs/tsz_dm_cross_updated_20260917/rays/annular_y_samples_lee22_pressure.npz")["cl_yy"]
    b16npz = np.load(R / "outputs/takahashi_cross_comparison_20260913/spectra/battaglia16.npz")["cl_yy"]

    def band(c, l):
        s = slice(int(l * 0.9), int(l * 1.1) + 1)
        return float(c[s].mean())

    rows = {}
    for l in BANDS:
        r = {k: band(v, l) for k, v in cl.items()}
        r["ratio_L_over_B"] = r["L"] / r["B"]
        r["ratio_hybrid_over_B"] = r["hyb"] / r["B"]
        r["ratio_Llow_over_Blow"] = r["Ll"] / r["Bl"]
        r["ratio_Lhigh_over_Bhigh"] = r["Lh"] / r["Bh"]
        r["frac_L_power_high_auto"] = r["Lh"] / r["L"]
        r["frac_L_power_high_symsplit"] = (r["Lh"] + r["Ll_x_Lh"]) / r["L"]
        r["frac_B_power_high_auto"] = r["Bh"] / r["B"]
        r["frac_B_power_high_symsplit"] = (r["Bh"] + r["Bl_x_Bh"]) / r["B"]
        r["check_L_vs_published_annular_cl"] = r["L"] / band(ann, l)
        r["check_B_vs_published_B_ref_cl"] = r["B"] / r["Bref"]
        r["check_B_vs_cluster_takahashi_cl"] = r["B"] / band(b16npz, l)
        r["published_ratio_L_over_B"] = band(ann, l) / band(b16npz, l)
        rows[l] = r
    res = dict(checks=checks, means=means, bands=rows, band_definition="mean C_l over int(0.9 l)..int(1.1 l) (as F)",
               lmax=LMAX, iter=3, elapsed_s=time.time() - t0)
    (OUT / "tsz_split_results.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(checks, indent=1))
    print("l  L/B  hyb/B  Llow/Blow  Lhigh/Bhigh  fracLhigh(auto,sym)  fracBhigh(auto,sym)  chkL chkB chkBclu")
    for l, r in rows.items():
        print(l, *["{:.3f}".format(r[k]) for k in ("ratio_L_over_B", "ratio_hybrid_over_B", "ratio_Llow_over_Blow",
                                                    "ratio_Lhigh_over_Bhigh", "frac_L_power_high_auto",
                                                    "frac_L_power_high_symsplit", "frac_B_power_high_auto",
                                                    "frac_B_power_high_symsplit", "check_L_vs_published_annular_cl",
                                                    "check_B_vs_published_B_ref_cl", "check_B_vs_cluster_takahashi_cl")])


if __name__ == "__main__":
    main()
