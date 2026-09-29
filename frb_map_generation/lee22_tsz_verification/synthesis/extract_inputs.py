#!/usr/bin/env python
"""Synthesis step 1: collect the workstream products the synthesis figures need into one cache file.

Nothing heavy is recomputed. Everything comes from the saved products of workstreams A-D:
  * CAMELS IllustrisTNG-model (workstream D): per-halo tables halo_tables/{precomputed,direct}_halos.npz.
    Models (B12, Lee22 printed = as painted, Lee22 x (1+z)^2, Lee22 density variants, B16) are evaluated at each
    halo's own M200c and z with D's profile_tools (which wraps the validated lee22_models.py) and averaged over the
    same haloes, exactly as analyze_stacks.py does. Errors: delete-one-box jackknife over the 27 CV boxes.
  * FLAMINGO L2p8_m9 (workstream C): the 2026-09-27 preliminary pressure stacks, rebuilt with C's
    pressure_comparison.analyse() (same haloes, same R100c-truncated shells, models shell-averaged on the same shells).
    Per-stack integrated quantities are read at plot time from C's pressure_metrics_raw.csv / density_metrics.csv.
  * Spectra: B's curves.npz (maps, CLASS-SZ and catalogue-weighted halo models on integer ell) and the FLAMINGO
    lightcone spectra in halfdome_tsz_comparison/products/comparison_products.npz.

Conventions: x = r/R200c, R200c physical; M200c in physical Msun (no h) for bins; Lee22 fits take h^-1 Msun
(CAMELS: M x 0.6774 as in D; FLAMINGO: M x 0.681 as in C); P200 = 200 G M200 rho_c(z) f_b/(2 R200) with each
simulation's own f_b and physical rho_c(z); P_e = n_e k T (B12: 0.5176 P_th).  CAMELS pressure = 0.5176 P_th
(P_e(x_e)/(0.5176 P_th) = 1.001-1.002 on the direct boxes).  CAMELS gas fraction f_gas = M_gas(<R200c)/(f_b M200c)
(all gas; full-ionisation electron equivalent).

Output: OUT/synthesis_inputs.npz  (OUT = outputs/lee22_tsz_verification_20260929/synthesis)
Run: PYTHONDONTWRITEBYTECODE=1 LD_LIBRARY_PATH=/home/cbllover/miniconda3/envs/halfdome/lib \
     /home/cbllover/miniconda3/envs/halfdome/bin/python extract_inputs.py
"""
import os
import sys
import time

import numpy as np

sys.dont_write_bytecode = True
ROOT = "/home/cbllover/HalfDome/frb_map_generation"
VER = f"{ROOT}/outputs/lee22_tsz_verification_20260929"
OUT = f"{VER}/synthesis"
os.makedirs(OUT, exist_ok=True)

# ------------------------------------------------------------------ CAMELS (workstream D)
sys.path.insert(0, f"{ROOT}/lee22_tsz_verification/D_tng_camels")
import camels_common as cc  # noqa: E402
import profile_tools as pt  # noqa: E402

t0 = time.time()
P = dict(np.load(f"{VER}/D_tng_camels/halo_tables/precomputed_halos.npz"))
D = dict(np.load(f"{VER}/D_tng_camels/halo_tables/direct_halos.npz"))
XG = P["xg"]
LMP = np.log10(P["M200"])
LMD = np.log10(D["M200"])
ZOLD = {int(o): float(np.mean(P["z"][P["old"] == o])) for o in sorted(set(P["old"]))}
MU_XH = cc.MU_E / cc.X_H          # f_e = (mu_e/X_H) 3 int nbar x^2 dx for a Lee22-form n_e/n200
out = {"xg": XG}


def jk_ratio(num, den, boxes):
    """ratio of halo means num/den (arrays (nh,) or (nh, nx)) and delete-one-box jackknife fractional error."""
    r = np.nanmean(num, 0) / np.nanmean(den, 0)
    ub = sorted(set(boxes))
    jk = np.array([np.nanmean(num[boxes != b], 0) / np.nanmean(den[boxes != b], 0) for b in ub])
    n = len(ub)
    e = np.sqrt((n - 1) / n * np.nansum((jk - np.nanmean(jk, 0)) ** 2, 0)) if n >= 3 else np.full_like(r, np.nan)
    return r, e / r


def jk_mean(v, boxes):
    m = np.nanmean(v, 0)
    ub = sorted(set(boxes))
    jk = np.array([np.nanmean(v[boxes != b], 0) for b in ub])
    n = len(ub)
    e = np.sqrt((n - 1) / n * np.nansum((jk - np.nanmean(jk, 0)) ** 2, 0)) if n >= 3 else np.full_like(m, np.nan)
    return m, e


CAM_BINS = [(13.0, 13.5), (13.5, 14.0), (14.0, 15.0)]
# (1) profile stacks at the target snapshots (fig 1)
for zt, old in cc.TARGETS.items():
    for lo, hi in CAM_BINS:
        s = (P["old"] == old) & (LMP >= lo) & (LMP < hi)
        if s.sum() < 3:
            continue
        idx = np.flatnonzero(s)
        mods = {}
        for k in idx:
            d = pt.model_curves(XG, P["M200"][k], P["z"][k])
            for key in ("P_lee_printed", "P_lee_a2", "P_b12"):
                mods.setdefault(key, []).append(d[key])
        boxes = P["box"][s]
        m, e = jk_mean(P["Pbar"][s], boxes)
        tag = f"cam_z{zt:g}_m{lo:g}"
        out[tag + "_P"] = m
        out[tag + "_Perr"] = e
        out[tag + "_nh"] = s.sum()
        out[tag + "_z"] = ZOLD[old]
        out[tag + "_logM_mean"] = np.mean(LMP[s])
        for key, v in mods.items():
            v = np.array(v)
            out[f"{tag}_{key}"] = np.nanmean(v, 0)
            # model / data ratio with the data's jackknife (model evaluated on the same haloes)
            r, re = jk_ratio(v, P["Pbar"][s], boxes)
            out[f"{tag}_{key}_over_data"] = r
            out[f"{tag}_{key}_over_data_err"] = re
print(f"CAMELS profile stacks done ({time.time() - t0:.0f} s)")

# (2) integrated normalisations at all 24 snapshots (fig 2, fig 4)
olds = sorted(set(P["old"]))
out["cam_trend_z"] = np.array([ZOLD[o] for o in olds])
for lo, hi in CAM_BINS[:2]:
    keys = ["Y_data_over_printed", "Y_B12_over_printed", "fgas_over_printed_fe", "fB16_over_printed_fe",
            "fgas", "fe_printed", "fe_prod", "fe_tngcal", "fgas_B16", "Yhat", "Y_printed", "Y_B12"]
    res = {k: np.full(len(olds), np.nan) for k in keys}
    err = {k: np.full(len(olds), np.nan) for k in keys}
    nh = np.zeros(len(olds), int)
    for i, o in enumerate(olds):
        s = (P["old"] == o) & (LMP >= lo) & (LMP < hi)
        nh[i] = s.sum()
        if s.sum() < 3:
            continue
        mi = [pt.model_integrals(P["M200"][k], P["z"][k]) for k in np.flatnonzero(s)]
        Yp = np.array([d["P_lee_printed"] for d in mi])
        Yb = np.array([d["P_b12"] for d in mi])
        fe_p = MU_XH * np.array([d["n_lee_printed"] for d in mi])
        fe_prod = MU_XH * np.array([d["n_lee_production"] for d in mi])
        fe_tng = MU_XH * np.array([d["n_lee_tngcal"] for d in mi])
        fb16 = np.array([d["rho_b16"] for d in mi]) / 200.0
        boxes = P["box"][s]
        for k, (num, den) in {"Y_data_over_printed": (P["Yhat"][s], Yp), "Y_B12_over_printed": (Yb, Yp),
                              "fgas_over_printed_fe": (P["fgas"][s], fe_p), "fB16_over_printed_fe": (fb16, fe_p)}.items():
            r, e = jk_ratio(num, den, boxes)
            res[k][i], err[k][i] = r, e
        for k, v in {"fgas": P["fgas"][s], "fe_printed": fe_p, "fe_prod": fe_prod, "fe_tngcal": fe_tng, "fgas_B16": fb16,
                     "Yhat": P["Yhat"][s], "Y_printed": Yp, "Y_B12": Yb}.items():
            m, e = jk_mean(v, boxes)
            res[k][i], err[k][i] = m, e
    for k in keys:
        out[f"cam_trend_m{lo:g}_{k}"] = res[k]
        out[f"cam_trend_m{lo:g}_{k}_err"] = err[k]
    out[f"cam_trend_m{lo:g}_nh"] = nh
print(f"CAMELS trends done ({time.time() - t0:.0f} s)")

# (3) Lee22-bookkeeping emulation on the directly processed CAMELS cells (4 boxes)
#     Pbar_emul / nbar_emul = what y_prof.cpp / ne_prof.cpp would have produced on the same cells (comoving (ckpc/h)^3
#     shells, physical rho_crit(z), comoving R200 in P200). Normalisation = 3 int_{0.04}^{1} q x^2 dx on the x grid
#     (the emulated profiles start at x = 0.04),
#     for the emulated profile and for the printed Lee22 model on the same haloes and the same grid.
sl = (XG >= 0.04 - 1e-9) & (XG <= 1.0 + 1e-9)
Zd = np.round(D["z"], 2)
for lo, hi in CAM_BINS[:2]:
    zz_l, rp, rn, nn = [], [], [], []
    for zz in sorted(set(Zd)):
        s = (Zd == zz) & (LMD >= lo) & (LMD < hi)
        if s.sum() < 2:
            continue
        mp, mn = [], []
        for k in np.flatnonzero(s):
            d = pt.model_curves(XG, D["M200"][k], D["z"][k])
            mp.append(d["P_lee_printed"])
            mn.append(d["n_lee_printed"])
        integ = lambda a: 3 * np.trapz(np.nanmean(a, 0)[sl] * XG[sl] ** 2, XG[sl])  # noqa: E731
        zz_l.append(float(np.mean(D["z"][s])))
        rp.append(integ(D["Pbar_emul"][s]) / integ(np.array(mp)))
        rn.append(integ(D["nbar_emul"][s]) / integ(np.array(mn)))
        nn.append(int(s.sum()))
    out[f"emul_m{lo:g}_z"] = np.array(zz_l)
    out[f"emul_m{lo:g}_P_over_printed"] = np.array(rp)
    out[f"emul_m{lo:g}_n_over_printed"] = np.array(rn)
    out[f"emul_m{lo:g}_nh"] = np.array(nn)
print(f"CAMELS emulation done ({time.time() - t0:.0f} s)")

# ------------------------------------------------------------------ FLAMINGO stacks (workstream C)
sys.path.insert(0, f"{ROOT}/lee22_tsz_verification/C_flamingo")
import pressure_comparison as pc  # noqa: E402

_tag, _rows, _trends, per_stack = pc.analyse(False)
FL_BINS = [13.2, 13.6, 14.4]
for p in per_stack:
    st = p["st"]
    lo = round(st["log10_m_lo"], 1)
    if lo not in FL_BINS or st["z"] > 2.01:
        continue
    tag = f"fl_z{round(st['z'], 1):g}_m{lo:g}"
    ok = p["ok"]
    out[tag + "_x"] = st["x_mid"][ok]
    out[tag + "_P"] = st["ratio"][ok]
    out[tag + "_Perr"] = st["err"][ok]
    out[tag + "_nh"] = st["n_haloes"]
    out[tag + "_logM_median"] = st["log10_m_median"]
    for name in ("B12", "Lee22", "Lee22_a2"):
        out[f"{tag}_{name}"] = p["models"][name][ok]
print(f"FLAMINGO stacks done ({time.time() - t0:.0f} s)")

# ------------------------------------------------------------------ spectra (workstreams B, C)
B = np.load(f"{VER}/B_class_sz/analysis/curves.npz")
Pp = np.load("/home/cbllover/HalfDome/flamingo_pressure_calibration/halfdome_tsz_comparison/products/comparison_products.npz")
assert np.array_equal(B["ell"], Pp["ell"])
out["ell"] = B["ell"]
for k in ("map_B12", "map_L22", "map_L22calib", "th_CS_b12_paint_all_hiprec", "th_CSX_lee22", "th_CSX_lee22m1",
          "th_CAT_b12", "th_CAT_lee22", "th_CAT_lee22m1", "th_CS_lee22_paint_calib", "th_CS_lee22m1_paint_calib"):
    out["cl_" + k] = B[k]
pw2 = Pp["pixwin"] ** 2
out["cl_flamingo_lc0"] = Pp["cl_flamingo_lc0"] / pw2      # FLAMINGO maps are pixel averages: divide by pixwin^2
out["cl_flamingo_lc1"] = Pp["cl_flamingo_lc1"] / pw2
np.savez_compressed(f"{OUT}/synthesis_inputs.npz", **out)
print(f"wrote {OUT}/synthesis_inputs.npz with {len(out)} arrays ({time.time() - t0:.0f} s)")
