#!/usr/bin/env python
"""Steps 2b-4: TNG-physics (CAMELS CV, 27 boxes) stacks of n_e/n200 and P_e/P200 vs x = r/R200c, gas fractions and
Y(<R200c), compared with Lee22 (printed / production / TNG-calibrated / M1-corrected), Battaglia12 and Battaglia16;
redshift trend of the normalisation relative to the printed Lee22 fits; emulation of Lee22's bookkeeping.

Inputs : halo_tables/precomputed_halos.npz, halo_tables/direct_halos.npz (build_halo_tables.py)
Outputs: tables/*.csv, stacks.npz, fig_D2..D6*.png, analysis_summary.txt (all in the D_tng_camels output dir)
Run: LD_LIBRARY_PATH=/home/cbllover/miniconda3/envs/halfdome/lib /home/cbllover/miniconda3/envs/halfdome/bin/python analyze_stacks.py

Stacking: arithmetic mean over halos of the per-halo dimensionless profile at each x (median also stored); every
model is evaluated at each halo's own M200c and z and averaged over the same halos.  Errors: delete-one-box
jackknife over the 27 CV boxes (the boxes are independent realisations).
Density data = full-ionisation n_e from the precomputed gas density, times the ElectronAbundance correction
f_xe(x, z) = <n_e(x_e)>/<n_e(full ion)> measured on the directly computed boxes (all M >= 10^12.5 Msun).
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.optimize import curve_fit  # noqa: E402

import camels_common as cc  # noqa: E402
import profile_tools as pt  # noqa: E402

LM = cc.LM
OUT = cc.OUTDIR
TAB = os.path.join(OUT, "tables")
os.makedirs(TAB, exist_ok=True)
P = dict(np.load(os.path.join(OUT, "halo_tables", "precomputed_halos.npz")))
D = dict(np.load(os.path.join(OUT, "halo_tables", "direct_halos.npz")))
XG = P["xg"]
LOGM_P = np.log10(P["M200"])
LOGM_D = np.log10(D["M200"])
MB = cc.MASS_BINS
COL = cc.COLORS
ZOLD = {int(o): float(np.mean(P["z"][P["old"] == o])) for o in sorted(set(P["old"]))}   # mean exact header z per snapshot
summary = []


def say(s=""):
    print(s, flush=True)
    summary.append(s)


def ix(x):
    return int(np.argmin(np.abs(XG - x)))


# ------------------------------------------------------------------ models per halo (cached)
def models_for(M, z):
    out = {}
    for k in range(len(M)):
        d = pt.model_curves(XG, M[k], z[k])
        for key, v in d.items():
            out.setdefault(key, []).append(v)
    return {k: np.array(v) for k, v in out.items()}


def model_integrals_for(M, z):
    out = {}
    for k in range(len(M)):
        d = pt.model_integrals(M[k], z[k])
        for key, v in d.items():
            out.setdefault(key, []).append(v)
    return {k: np.array(v) for k, v in out.items()}


say("computing model curves for %d precomputed halo-snapshots" % len(P["M200"]))
PM = models_for(P["M200"], P["z"])
PMI = model_integrals_for(P["M200"], P["z"])
DM = models_for(D["M200"], D["z"]) if len(D["M200"]) else {}

# ------------------------------------------------------------------ ElectronAbundance correction from direct runs
# group the direct halos by snapshot number (header z of snapshot 052 differs between boxes: 1.4778 vs 1.4793)
Zd = np.array([np.round(np.mean(D["z"][D["new"] == n]), 3) for n in D["new"]]) if len(D["M200"]) else np.array([])
fxe = {}
say("\n== ElectronAbundance correction f_xe = <n_e(x_e)>/<n_e(full ion)> (direct boxes, M200c >= 10^12.5 Msun)")
for zz in sorted(set(Zd)):
    s = (Zd == zz)
    num = np.nanmean(D["nbar_xe"][s], 0)
    den = np.nanmean(D["nbar_fullion"][s], 0)
    fxe[zz] = num / den
    boxes = sorted(set(D["box"][s]))
    say(f"  z={zz:.3f} ({s.sum()} halos, boxes {boxes}): " +
        ", ".join(f"x={x}: {fxe[zz][ix(x)]:.3f}" for x in cc.X_QUOTE) +
        f"; P_e(x_e)/(0.5176 P_th) at x=0.3: {np.nanmean(D['Pbar_xe'][s], 0)[ix(0.3)] / np.nanmean(D['Pbar_th'][s], 0)[ix(0.3)]:.4f}")


def fxe_at(z):
    """nearest direct redshift's correction (the direct runs sample z = 0, 0.47, 1.05, 1.48, 2.00)"""
    if not fxe:
        return np.ones(len(XG))
    keys = np.array(sorted(fxe))
    k = keys[np.argmin(np.abs(keys - z))]
    f = fxe[k].copy()
    return np.where(np.isfinite(f), f, 1.0)


# ------------------------------------------------------------------ method check: precomputed-derived vs direct, same halos
say("\n== method check: precomputed (coarse ckpc/h bins, slope-corrected, interpolated) vs direct fine-x-bin profiles, same halos"
    " (boxes CV_0, CV_1, CV_3; CV_2 raw data are a different realisation, see validation/)")
keyP = {(int(b), round(float(z), 2), int(i)): k for k, (b, z, i) in enumerate(zip(P["box"], P["z"], P["ID"]))}
# CV_2 is excluded: the raw CV_2 now hosted (2023-01-07, Arepo 2021, seed 3999666 in
# CosmoAstroSeed_IllustrisTNG_L25n256_CV.txt) is a different realisation from the one the 2022 precomputed CV_2
# profiles were made from (validate_precomputed.py: different halo catalogue), so group IDs do not match.
pairs = [(keyP[(int(b), round(float(z), 2), int(i))], k) for k, (b, z, i) in enumerate(zip(D["box"], D["z"], D["ID"]))
         if (int(b), round(float(z), 2), int(i)) in keyP and int(b) != 2]
if pairs:
    kp = np.array([p[0] for p in pairs])
    kd = np.array([p[1] for p in pairs])
    for name, a_, b_ in (("nbar(full ion)", P["nbar"][kp], D["nbar_fullion"][kd]), ("Pbar(0.5176 Pth)", P["Pbar"][kp], D["Pbar_th"][kd]),
                         ("fgas(<R200)", P["fgas"][kp][:, None], D["fgas"][kd][:, None]), ("Yhat(<R200)", P["Yhat"][kp][:, None], D["Yhat"][kd][:, None])):
        if a_.shape[1] == 1:
            r = np.nanmean(a_[:, 0]) / np.nanmean(b_[:, 0])
            say(f"  {name:18s}: mean precomputed / mean direct = {r:.4f} ({len(kp)} halos)")
        else:
            r = np.nanmean(a_, 0) / np.nanmean(b_, 0)
            say(f"  {name:18s}: stacked ratio " + ", ".join(f"x={x}: {r[ix(x)]:.3f}" for x in [0.05] + cc.X_QUOTE + [2.0]) +
                f" ({len(kp)} halos)")
    # explicit Lee22-bookkeeping emulation vs analytic a^3, a^2
    for zz in sorted(set(Zd)):
        s = Zd == zz
        a = 1 / (1 + zz)
        rn = np.nanmean(D["nbar_emul"][s], 0) / np.nanmean(D["nbar_xe"][s], 0)
        rp = np.nanmean(D["Pbar_emul"][s], 0) / np.nanmean(D["Pbar_xe"][s], 0)
        sl = (XG >= 0.1) & (XG <= 1.5)
        say(f"  emulated/true at z={zz:.2f}: density {np.nanmean(rn[sl]):.4f} (a^3 = {a ** 3:.4f}), "
            f"pressure {np.nanmean(rp[sl]):.4f} (a^2 = {a ** 2:.4f})  [mean over 0.1<x<1.5]")


# ------------------------------------------------------------------ stacking helpers
def jk_mean_ratio(vals, mod, boxes, ixs=None):
    """ratio of halo means (data/model) with delete-one-box jackknife error (fractional)."""
    num = np.nanmean(vals, 0)
    den = np.nanmean(mod, 0)
    r = num / den
    ub = sorted(set(boxes))
    if len(ub) < 3:
        return r, np.full_like(r, np.nan)
    jk = []
    for b in ub:
        s = boxes != b
        jk.append(np.nanmean(vals[s], 0) / np.nanmean(mod[s], 0))
    jk = np.array(jk)
    n = len(ub)
    err = np.sqrt((n - 1) / n * np.nansum((jk - np.nanmean(jk, 0)) ** 2, 0))
    return r, err


def sel_P(z_old, lo, hi):
    return (P["old"] == z_old) & (LOGM_P >= lo) & (LOGM_P < hi)


def density_data(s):
    z = P["z"][s][0]
    return P["nbar"][s] * fxe_at(z)[None, :]


# ------------------------------------------------------------------ Table: log ratios at quote points, target redshifts
rows = ["z_target,z_snap,logM_lo,logM_hi,nhalo,nbox,mean_logM,quantity,model,x,data_mean,data_jk_frac_err,data_median,"
        "model_mean,log10_data_over_model,log10_err"]
nmodels = {"lee_printed": "n_lee_printed", "lee_production_x0.6016": "n_lee_production", "lee_x_Cz_tngcal": "n_lee_tngcal",
           "lee_x_(1+z)^3": "n_lee_a3", "b16_xgpaint_ne(0.9)": "n_b16_xgp"}
pmodels = {"lee_printed(=painted)": "P_lee_printed", "lee_x_(1+z)^2": "P_lee_a2", "b12_Pe": "P_b12"}
stacks = {}
say("\n== stacks at target redshifts (log10 data/model at x = 0.1, 0.3, 0.5, 1; jackknife over boxes)")
for zt, old in cc.TARGETS.items():
    zsnap = ZOLD[old]
    for lo, hi in MB:
        s = sel_P(old, lo, hi)
        nh = int(s.sum())
        if nh < 3:
            say(f"  z={zsnap:.3f} logM {lo}-{hi}: {nh} halos - skipped")
            continue
        boxes = P["box"][s]
        nb = len(set(boxes))
        dn = density_data(s)
        dp = P["Pbar"][s]
        dr = P["rhob"][s]
        stacks[(zt, lo)] = dict(z=zsnap, n=dn, P=dp, rho=dr, M=P["M200"][s], boxes=boxes,
                                models={k: PM[k][s] for k in PM}, nh=nh, nb=nb)
        line = f"  z={zsnap:.3f} logM {lo}-{hi}: {nh} halos in {nb} boxes, <logM>={np.mean(LOGM_P[s]):.2f}"
        say(line)
        for qname, data, models in (("n_e/n200", dn, nmodels), ("P_e/P200", dp, pmodels),
                                    ("rho_gas/(fb rho_c)", dr, {"b16_gas": "rho_b16"})):
            for mname, mkey in models.items():
                r, e = jk_mean_ratio(data, PM[mkey][s], boxes)
                med = np.nanmedian(data, 0)
                for x in cc.X_QUOTE:
                    k = ix(x)
                    rows.append(f"{zt},{zsnap},{lo},{hi},{nh},{nb},{np.mean(LOGM_P[s]):.3f},{qname},{mname},{x},"
                                f"{np.nanmean(data[:, k]):.5e},{e[k]:.4f},{med[k]:.5e},{np.nanmean(PM[mkey][s][:, k]):.5e},"
                                f"{np.log10(r[k]):.4f},{e[k] / np.log(10):.4f}")
                say(f"     {qname:18s} / {mname:24s}: " + "  ".join(f"{np.log10(r[ix(x)]):+.3f}" for x in cc.X_QUOTE))
with open(os.path.join(TAB, "log_ratios_target_z.csv"), "w") as f:
    f.write("\n".join(rows) + "\n")

# ------------------------------------------------------------------ redshift trend relative to printed Lee22
say("\n== redshift trend: <data>/<Lee22 printed> vs (1+z), all snapshots z = 0-2; exponent k of (1+z)^k (box jackknife)")
olds = sorted(set(P["old"]))
zs = np.array([ZOLD[o] for o in olds])
AMP_SL = (XG >= 0.04) & (XG <= 1.34)
trend_rows = ["quantity,logM_lo,logM_hi,x,z,nhalo,ratio,jk_frac_err"]
fit_rows = ["quantity,logM_lo,logM_hi,x,k_exponent,k_jk_err,ratio_at_z0_fit,ratio_z0_jk_err,zmax_used,n_z,k_without_CV2,"
            "ratio_z0_without_CV2"]
trend = {}


def ratio_series(qdata, mkey, lo, hi, xsel, boxes_excl=None):
    """ratio(z) of halo-mean data to halo-mean model; xsel = index or 'amp' (geometric mean over 0.04-1.34)."""
    out = np.full(len(olds), np.nan)
    nh = np.zeros(len(olds), int)
    for i, o in enumerate(olds):
        s = sel_P(o, lo, hi)
        if boxes_excl is not None:
            s &= P["box"] != boxes_excl
        if s.sum() < 3:
            continue
        data = qdata(s)
        mod = PM[mkey][s]
        if xsel == "amp":
            r = np.nanmean(data, 0)[AMP_SL] / np.nanmean(mod, 0)[AMP_SL]
            out[i] = np.exp(np.nanmean(np.log(r)))
        else:
            out[i] = np.nanmean(data[:, xsel]) / np.nanmean(mod[:, xsel])
        nh[i] = s.sum()
    return out, nh


def fit_k(r, zmax=2.1):
    ok = np.isfinite(r) & (r > 0) & (zs <= zmax)
    if ok.sum() < 3:
        return np.nan, np.nan
    c = np.polyfit(np.log10(1 + zs[ok]), np.log10(r[ok]), 1)
    return c[0], 10 ** c[1]


qfun = {"n_e/n200": (lambda s: density_data_all(s), "n_lee_printed"),
        "P_e/P200": (lambda s: P["Pbar"][s], "P_lee_printed")}


def density_data_all(s):
    zz = P["z"][s]
    f = np.array([fxe_at(z) for z in zz])
    return P["nbar"][s] * f


for qname, (qf, mkey) in qfun.items():
    for lo, hi in MB:
        for xs in ["amp"] + [ix(x) for x in (0.1, 0.3, 0.5, 1.0)]:
            xl = "amp(0.04-1.34)" if xs == "amp" else f"{XG[xs]:.2f}"
            r, nh = ratio_series(qf, mkey, lo, hi, xs)
            k, c0 = fit_k(r)
            jk_k, jk_c = [], []
            for b in range(27):
                rb, _ = ratio_series(qf, mkey, lo, hi, xs, boxes_excl=b)
                kb, cb = fit_k(rb)
                jk_k.append(kb)
                jk_c.append(cb)
            jk_k, jk_c = np.array(jk_k), np.array(jk_c)
            ek = np.sqrt(26 / 27 * np.nansum((jk_k - np.nanmean(jk_k)) ** 2))
            ec = np.sqrt(26 / 27 * np.nansum((jk_c - np.nanmean(jk_c)) ** 2)) / c0
            trend[(qname, lo, xl)] = (r, nh, k, ek, c0, ec)
            for i in range(len(olds)):
                trend_rows.append(f"{qname},{lo},{hi},{xl},{zs[i]},{nh[i]},{r[i]:.5f},")
            fit_rows.append(f"{qname},{lo},{hi},{xl},{k:.3f},{ek:.3f},{c0:.4f},{ec:.4f},2.0,{np.isfinite(r).sum()},"
                            f"{jk_k[2]:.3f},{jk_c[2]:.4f}")
            if xs == "amp" or XG[xs] in (0.3, 0.5):
                say(f"  {qname} logM {lo}-{hi} x={xl:15s}: ratio = {c0:.3f} (1+z)^({k:.2f} +- {ek:.2f});  "
                    f"ratio at z=0,1,2 = {r[olds.index(33)]:.3f}, {r[olds.index(18)]:.3f}, {r[olds.index(10)]:.3f}; "
                    f"halos z=0/1/2: {nh[olds.index(33)]}/{nh[olds.index(18)]}/{nh[olds.index(10)]}; without CV_2: "
                    f"{jk_c[2]:.3f} (1+z)^{jk_k[2]:.2f}")
with open(os.path.join(TAB, "redshift_trend_vs_lee22_printed.csv"), "w") as f:
    f.write("\n".join(trend_rows) + "\n")
with open(os.path.join(TAB, "redshift_trend_powerlaw_fits.csv"), "w") as f:
    f.write("\n".join(fit_rows) + "\n")

# ------------------------------------------------------------------ direct-only cross-check of the redshift trend
say("\n== cross-check with the DIRECT profiles only (4 boxes CV_0-CV_3, ElectronAbundance n_e and n_e k_B T, fine x bins):")
say("   amplitude <data>/<Lee22 printed> (geometric mean over 0.04 <= x <= 1.34) at the 5 direct snapshots, power-law fit in 1+z")
dtrend_rows = ["quantity,logM_lo,logM_hi,z,nhalo,nbox,ratio_direct,ratio_precomputed_all27,k_direct,c0_direct"]
DT = {}
if len(D["M200"]):
    zd_list = sorted(set(Zd))
    for qname, dkey, mkey, pkey in (("n_e/n200", "nbar_xe", "n_lee_printed", "n_e/n200"), ("P_e/P200", "Pbar_xe", "P_lee_printed", "P_e/P200")):
        for lo, hi in MB:
            rr, nn, nbx = [], [], []
            for zz in zd_list:
                sd = (Zd == zz) & (LOGM_D >= lo) & (LOGM_D < hi)
                if sd.sum() < 3:
                    rr.append(np.nan); nn.append(int(sd.sum())); nbx.append(len(set(D["box"][sd])))
                    continue
                q = np.nanmean(D[dkey][sd], 0)[AMP_SL] / np.nanmean(DM[mkey][sd], 0)[AMP_SL]
                rr.append(float(np.exp(np.nanmean(np.log(q)))))
                nn.append(int(sd.sum()))
                nbx.append(len(set(D["box"][sd])))
            rr = np.array(rr)
            zz_arr = np.array(zd_list)
            ok = np.isfinite(rr)
            kd, cd = (np.polyfit(np.log10(1 + zz_arr[ok]), np.log10(rr[ok]), 1) if ok.sum() >= 3 else (np.nan, np.nan))
            cd = 10 ** cd if np.isfinite(cd) else np.nan
            rp = trend[(pkey, lo, "amp(0.04-1.34)")][0]
            DT[(qname, lo)] = (zz_arr, rr, nn, kd, cd)
            for i, zz in enumerate(zd_list):
                ip = int(np.argmin(np.abs(zs - zz)))
                dtrend_rows.append(f"{qname},{lo},{hi},{zz},{nn[i]},{nbx[i]},{rr[i]:.4f},{rp[ip]:.4f},{kd:.3f},{cd:.4f}")
            if ok.sum() >= 2:
                say(f"  {qname} logM {lo}-{hi}: direct ratio at z=" + ", ".join(f"{zz:.2f}: {r:.3f} ({n})" for zz, r, n in zip(zd_list, rr, nn)) +
                    (f"  -> {cd:.3f} (1+z)^{kd:.2f}" if np.isfinite(kd) else "") +
                    f";  precomputed 27-box fit {trend[(pkey, lo, 'amp(0.04-1.34)')][4]:.3f} (1+z)^{trend[(pkey, lo, 'amp(0.04-1.34)')][2]:.2f}")
with open(os.path.join(TAB, "redshift_trend_direct_only.csv"), "w") as f:
    f.write("\n".join(dtrend_rows) + "\n")

# ------------------------------------------------------------------ F_n and pressure normalisation after the M1 correction
say("\n== normalisation after removing M1: truth / (printed x (1+z)^3) [density] and truth / (printed x (1+z)^2) [pressure]")
say("   (amplitude = geometric mean of the stacked ratio over 0.04 <= x <= 1.34; F_n = printed x (1+z)^3 / truth)")
fn_rows = ["quantity,logM_lo,logM_hi,z,nhalo,truth_over_M1corrected_amp,jk_frac_err,F_n_or_F_P"]
for qname, (qf, mkey) in qfun.items():
    p = 3 if qname.startswith("n") else 2
    for lo, hi in MB:
        r, nh, k, ek, c0, ec = trend[(qname, lo, "amp(0.04-1.34)")]
        rc = r / (1 + zs) ** p
        # jackknife error per z from the box-deleted series
        for zt, old in cc.TARGETS.items():
            i = olds.index(old)
            jk = []
            for b in range(27):
                rb, _ = ratio_series(qf, mkey, lo, hi, "amp", boxes_excl=b)
                jk.append(rb[i] / (1 + zs[i]) ** p)
            jk = np.array(jk)
            e = np.sqrt(26 / 27 * np.nansum((jk - np.nanmean(jk)) ** 2)) / rc[i] if np.isfinite(rc[i]) else np.nan
            fn_rows.append(f"{qname},{lo},{hi},{zs[i]},{nh[i]},{rc[i]:.4f},{e:.4f},{1 / rc[i]:.3f}")
        say(f"  {qname} logM {lo}-{hi}: truth/(printed x (1+z)^{p}) at z=0,0.47,1.05,1.48,2.00 = " +
            ", ".join(f"{rc[olds.index(o)]:.3f}" for o in cc.TARGETS.values()) +
            f"  -> F = " + ", ".join(f"{1 / rc[olds.index(o)]:.2f}" for o in cc.TARGETS.values()))
with open(os.path.join(TAB, "normalisation_after_M1.csv"), "w") as f:
    f.write("\n".join(fn_rows) + "\n")

# ------------------------------------------------------------------ gas fractions and Y inside R200c
say("\n== inside R200c: f_gas/f_b (precomputed gas; electron-equivalent from direct), Yhat = Y_sph/(P200 4pi/3 R200^3)")
gy_rows = ["z_target,z_snap,logM_lo,logM_hi,nhalo,fgas_mean,fgas_jk,fe_direct_mean,n_direct,lee_printed_fe,lee_production_fe,"
           "lee_tngcal_fe,lee_a3_fe,b16_fgas,Yhat_mean,Yhat_jk,lee_printed_Y,lee_a2_Y,b12_Y"]
mu_xh = cc.MU_E / cc.X_H
for zt, old in cc.TARGETS.items():
    for lo, hi in MB:
        s = sel_P(old, lo, hi)
        if s.sum() < 3:
            continue
        boxes = P["box"][s]
        fg, efg = jk_mean_ratio(P["fgas"][s][:, None], np.ones((s.sum(), 1)), boxes)
        yh, eyh = jk_mean_ratio(P["Yhat"][s][:, None], np.ones((s.sum(), 1)), boxes)
        sd = (np.abs(D["z"] - ZOLD[old]) < 0.02) & (LOGM_D >= lo) & (LOGM_D < hi)
        fe_d = np.mean(D["fe"][sd]) if sd.sum() else np.nan
        m = {k: np.mean(PMI[k][s]) for k in PMI}
        gy_rows.append(f"{zt},{ZOLD[old]},{lo},{hi},{s.sum()},{fg[0]:.4f},{efg[0]:.4f},{fe_d:.4f},{sd.sum()},"
                       f"{mu_xh * m['n_lee_printed']:.4f},{mu_xh * m['n_lee_production']:.4f},{mu_xh * m['n_lee_tngcal']:.4f},"
                       f"{mu_xh * m['n_lee_a3']:.4f},{m['rho_b16'] / 200:.4f},{yh[0]:.4f},{eyh[0]:.4f},"
                       f"{m['P_lee_printed']:.4f},{m['P_lee_a2']:.4f},{m['P_b12']:.4f}")
        say(f"  z={ZOLD[old]:.2f} logM {lo}-{hi} ({s.sum()}): f_gas/f_b = {fg[0]:.3f}+-{efg[0] * fg[0]:.3f} "
            f"(direct f_e {fe_d:.3f}, n={sd.sum()});  Lee22 f_e printed {mu_xh * m['n_lee_printed']:.2f}, prod {mu_xh * m['n_lee_production']:.2f}, "
            f"C(z) {mu_xh * m['n_lee_tngcal']:.2f}, (1+z)^3 {mu_xh * m['n_lee_a3']:.2f}; B16 {m['rho_b16'] / 200:.2f} | "
            f"Yhat = {yh[0]:.3f}+-{eyh[0] * yh[0]:.3f}; Lee22 printed {m['P_lee_printed']:.3f}, x(1+z)^2 {m['P_lee_a2']:.3f}, B12 {m['P_b12']:.3f}")
with open(os.path.join(TAB, "gas_fraction_and_Y_R200c.csv"), "w") as f:
    f.write("\n".join(gy_rows) + "\n")

# ------------------------------------------------------------------ Lee22-form fits to the emulated stacks
say("\n== Lee22 functional form (gamma=-0.3, alpha=1) fitted to the EMULATED stacks (truth x a^3 / x a^2), 0.04<=x<=1.34")
fit_rows = ["quantity,z,logM_lo,logM_hi,nhalo,mean_Mh_hinv,A_fit,xc_fit,beta_fit,A_printed,xc_printed,beta_printed,"
            "amp_ratio_frozen_shape"]


def gnfw_log(lx, lA, lxc, beta):
    x = np.exp(lx)
    xc = np.exp(lxc)
    return lA + (-0.3) * np.log(x / xc) - beta * np.log1p(x / xc)


for (zt, lo), st in sorted(stacks.items()):
    if lo >= 14.0:
        continue
    a = 1 / (1 + st["z"])
    mh = np.mean(st["M"]) * cc.H_TNG
    for qname, data, p, parfun, mkey in (("n_e/n200", st["n"], 3, LM.lee_n_noc_params, "n_lee_printed"),
                                         ("P_e/P200", st["P"], 2, LM.lee_P_noc_params, "P_lee_printed")):
        y = np.nanmean(data, 0) * a ** p
        mod = np.nanmean(st["models"][mkey], 0)
        sl = AMP_SL & np.isfinite(y) & (y > 0)
        A0, xc0, b0 = parfun(mh, st["z"])
        amp = np.exp(np.nanmean(np.log(y[sl] / mod[sl])))
        try:
            # bounded: 0.05 < xc < 50, 0.5 < beta < 40 (the gamma=-0.3, alpha=1 form is degenerate for small halos)
            p0 = [np.log(A0 * amp), np.log(np.clip(xc0, 0.06, 45)), np.clip(b0, 0.6, 39)]
            popt, _ = curve_fit(gnfw_log, np.log(XG[sl]), np.log(y[sl]), p0=p0, maxfev=20000,
                                bounds=([-50, np.log(0.05), 0.5], [50, np.log(50.0), 40.0]))
            Af, xcf, bf = np.exp(popt[0]), np.exp(popt[1]), popt[2]
        except Exception:  # noqa: BLE001
            Af = xcf = bf = np.nan
        fit_rows.append(f"{qname},{st['z']},{lo},{lo + 0.5},{st['nh']},{mh:.4e},{Af:.4g},{xcf:.4g},{bf:.4g},{A0:.4g},{xc0:.4g},"
                        f"{b0:.4g},{amp:.4f}")
        say(f"  {qname} z={st['z']:.2f} logM {lo}: emulated/printed (shape frozen) = {amp:.3f};  free fit A={Af:.3g} xc={xcf:.3g} "
            f"beta={bf:.3g}  vs printed A={A0:.3g} xc={xc0:.3g} beta={b0:.3g}")
with open(os.path.join(TAB, "lee22_form_fits_to_emulated_stacks.csv"), "w") as f:
    f.write("\n".join(fit_rows) + "\n")

np.savez_compressed(os.path.join(OUT, "stacks.npz"), xg=XG,
                    **{f"{q}_z{zt}_m{lo}": np.nanmean(st[q], 0) for (zt, lo), st in stacks.items() for q in ("n", "P", "rho")},
                    **{f"nh_z{zt}_m{lo}": st["nh"] for (zt, lo), st in stacks.items()})

# ================================================================== FIGURES
ZT = list(cc.TARGETS)
MB3 = MB[:3]


def band(ax, x, data, boxes, color):
    m = np.nanmean(data, 0)
    ub = sorted(set(boxes))
    jk = np.array([np.nanmean(data[boxes != b], 0) for b in ub])
    n = len(ub)
    e = np.sqrt((n - 1) / n * np.nansum((jk - np.nanmean(jk, 0)) ** 2, 0)) if n > 2 else np.zeros_like(m)
    ax.fill_between(x, np.clip(m - e, 1e-30, None), m + e, color=color, alpha=0.25, lw=0)
    ax.plot(x, m, color=color, lw=2)


def profile_figure(q, fname, ylabel, models, title, weight=1.0):
    fig, axs = plt.subplots(len(ZT), len(MB3), figsize=(12.5, 15), sharex=True, constrained_layout=True)
    for i, zt in enumerate(ZT):
        for j, (lo, hi) in enumerate(MB3):
            ax = axs[i, j]
            st = stacks.get((zt, lo))
            if st is None:
                ax.text(0.5, 0.5, f"z = {cc.snap_z(0, cc.TARGETS[zt]):.2f}, {lo} < log M200c/Msun < {hi}:\n< 3 halos in 27 boxes",
                        transform=ax.transAxes, ha="center", va="center", fontsize=9)
                ax.set_axis_off()          # (set_xticks would clear the ticks of all sharex panels)
                if i > 0:
                    axs[i - 1, j].tick_params(labelbottom=True)
                    axs[i - 1, j].set_xlabel(r"$x = r/R_{200c}$")
                continue
            w = XG ** weight
            band(ax, XG, st[q] * w, st["boxes"], "black")
            for mkey, (c, ls, lab) in models.items():
                ax.plot(XG, np.nanmean(st["models"][mkey], 0) * w, color=c, ls=ls, lw=1.6, label=lab)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.axvspan(0.03, 0.04, color="0.9", lw=0)
            ax.axvspan(1.34, 3.0, color="0.9", lw=0)
            ax.set_title(f"z = {st['z']:.2f}, {lo} < log M200c/Msun < {hi}: {st['nh']} halos / {st['nb']} boxes",
                         fontsize=8.5)
            if i == len(ZT) - 1:
                ax.set_xlabel(r"$x = r/R_{200c}$")
            if j == 0:
                ax.set_ylabel(ylabel)
    h, l = axs[0, 0].get_legend_handles_labels()
    from matplotlib.patches import Patch
    h = [plt.Line2D([], [], color="black", lw=2)] + h + [Patch(color="0.9")]
    l = ["CAMELS TNG CV (27 boxes), mean +- box jackknife"] + l + ["outside the Lee22 fit range 0.04 < x < 1.34"]
    fig.legend(h, l, loc="lower center", ncol=4 if len(h) > 6 else 3, fontsize=9, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(title, y=1.06)
    fig.savefig(os.path.join(OUT, fname), dpi=110, bbox_inches="tight")
    plt.close(fig)


profile_figure("n", "fig_D2_density_stacks.png", r"$x^2\, n_e/n_{200}$",
               {"n_lee_printed": (COL["lee_print"], "-", "Lee22 printed (Table 8, no-c)"),
                "n_lee_production": (COL["lee_prod"], "-", "Lee22 production (x0.6016)"),
                "n_lee_tngcal": (COL["lee_corr"], "-", "Lee22 x C(z) (TNG-calibrated)"),
                "n_lee_a3": (COL["lee_corr"], "--", "Lee22 x (1+z)^3 (M1)"),
                "n_b16_xgp": (COL["b12"], "-", "Battaglia16 (XGPaint n_e, 0.9 ionised)")},
               r"Electron density: $n_e/n_{200}$, $n_{200}=200\rho_c(z) f_b/(X_H m_p)$ (physical); "
               r"CAMELS $n_e$ = full-ionisation gas $\times f_{x_e}$ (ElectronAbundance)", weight=2.0)
profile_figure("P", "fig_D3_pressure_stacks.png", r"$x^3\, P_e/P_{200}$",
               {"P_lee_printed": (COL["lee_print"], "-", "Lee22 printed (Table 7, no-c) = painted"),
                "P_lee_a2": (COL["lee_corr"], "--", "Lee22 x (1+z)^2 (M1)"),
                "P_b12": (COL["b12"], "-", "Battaglia12 x 0.5176 (P_e)")},
               r"Electron pressure: $P_e/P_{200}$, $P_{200}=200 G M_{200}\rho_c f_b/(2R_{200})$ (physical); "
               r"CAMELS $P_e = 0.5176\,P_{th}$", weight=3.0)

# --- D4: redshift trend
fig, axs = plt.subplots(2, 3, figsize=(13.5, 8), constrained_layout=True, sharex=True)
for i, (qname, p, lab) in enumerate((("n_e/n200", 3, "density"), ("P_e/P200", 2, "pressure"))):
    for j, (lo, hi) in enumerate(MB3):
        ax = axs[i, j]
        for xl, mk, c in (("amp(0.04-1.34)", "o", "black"), ("0.30", "s", "0.45"), ("1.00", "^", COL["other"])):
            r, nh, k, ek, c0, ec = trend[(qname, lo, xl)]
            ok = np.isfinite(r)
            ax.plot(1 + zs[ok], r[ok], mk, color=c, ms=4,
                    label=(f"x = {float(xl):.1f}: " if xl != "amp(0.04-1.34)" else "amplitude (0.04-1.34): ") +
                    f"${c0:.2f}\\,(1+z)^{{{k:.2f}\\pm{ek:.2f}}}$")
            if xl == "amp(0.04-1.34)":
                zz = np.linspace(0, 2, 50)
                ax.plot(1 + zz, c0 * (1 + zz) ** k, color="black", lw=1)
        zz = np.linspace(0, 2, 50)
        r0 = trend[(qname, lo, "amp(0.04-1.34)")][4]
        if p == 3:
            ax.plot(1 + zz, 0.2841790493880425 * (1 + zz) ** 2.483190184608004, color=COL["lee_corr"], lw=1.8,
                    label="$C(z) = 0.284\\,(1+z)^{2.48}$ (TNG FRB DM)")
            ax.plot(1 + zz, r0 * (1 + zz) ** 3, color=COL["lee_corr"], ls="--", lw=1.8, label="M1: const $\\times\\,(1+z)^3$")
            ax.axhline(0.6016, color=COL["lee_prod"], lw=1.5, label="production 0.6016")
        else:
            ax.plot(1 + zz, (1 + zz) ** 2, color=COL["lee_corr"], ls="--", lw=1.8, label="M1: $(1+z)^2$")
            ax.plot(1 + zz, r0 * (1 + zz) ** 2, color=COL["lee_corr"], ls=":", lw=1.5, label="const $\\times\\,(1+z)^2$")
        ax.axhline(1, color=COL["lee_print"], lw=1.2, label="printed Lee22")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks([1, 1.5, 2, 3])
        ax.set_xticklabels(["1", "1.5", "2", "3"])
        ax.set_title(f"{lab}, {lo} < log M200c/Msun < {hi}", fontsize=10)
        ax.legend(fontsize=6.5, loc="upper left")
        if i == 1:
            ax.set_xlabel("1 + z")
        if j == 0:
            ax.set_ylabel(f"CAMELS TNG / Lee22 printed ({'n_e/n200' if p == 3 else 'P_e/P200'})")
fig.suptitle("Redshift dependence of the TNG-physics normalisation relative to the printed Lee22 fits "
             "(CAMELS CV, 27 boxes, 24 snapshots z=0-2)")
fig.savefig(os.path.join(OUT, "fig_D4_redshift_trend.png"), dpi=120)
plt.close(fig)

# --- D5: gas fraction and Yhat vs mass
fig, axs = plt.subplots(2, 3, figsize=(13.5, 8), constrained_layout=True)
mm = np.logspace(12.5, 14.3, 30)
for j, old in enumerate((33, 18, 10)):
    z = ZOLD[old]
    s = (P["old"] == old)
    sd = np.abs(D["z"] - z) < 0.02
    mi = [pt.model_integrals(m, z) for m in mm]
    ax = axs[0, j]
    ax.plot(P["M200"][s], P["fgas"][s], ".", color="0.6", ms=3, label="CAMELS halos (gas)")
    if sd.sum():
        ax.plot(D["M200"][sd], D["fe"][sd], "x", color="black", ms=4, label="direct boxes: electron-equiv. f_e")
    for lo, hi in MB:
        ss = s & (LOGM_P >= lo) & (LOGM_P < hi)
        if ss.sum() >= 3:
            ax.errorbar(10 ** np.mean(LOGM_P[ss]), np.mean(P["fgas"][ss]),
                        yerr=np.std(P["fgas"][ss]) / np.sqrt(ss.sum()), fmt="o", color="black", ms=6)
    for key, c, ls, lab in (("n_lee_printed", COL["lee_print"], "-", "Lee22 printed"),
                            ("n_lee_production", COL["lee_prod"], "-", "Lee22 production"),
                            ("n_lee_tngcal", COL["lee_corr"], "-", "Lee22 x C(z)"),
                            ("n_lee_a3", COL["lee_corr"], "--", "Lee22 x (1+z)^3")):
        ax.plot(mm, [mu_xh * d[key] for d in mi], color=c, ls=ls, label=lab)
    ax.plot(mm, [d["rho_b16"] / 200 for d in mi], color=COL["b12"], label="Battaglia16 (gas)")
    ax.axhline(1, color="0.5", lw=0.8, ls=":")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(0.03, 80)
    ax.set_title(f"z = {z:.2f}: gas / electron fraction inside R200c", fontsize=10)
    ax.set_xlabel(r"$M_{200c}$ [$M_\odot$]")
    if j == 0:
        ax.set_ylabel(r"$M_{gas}(<R_{200c})/(f_b M_{200c})$")
        ax.legend(fontsize=7, loc="upper left", ncol=2)
    ax = axs[1, j]
    ax.plot(P["M200"][s], P["Yhat"][s], ".", color="0.6", ms=3, label="CAMELS halos")
    if sd.sum():
        ax.plot(D["M200"][sd], D["Yhat"][sd], "x", color="black", ms=4, label="direct boxes (x_e)")
    for lo, hi in MB:
        ss = s & (LOGM_P >= lo) & (LOGM_P < hi)
        if ss.sum() >= 3:
            ax.errorbar(10 ** np.mean(LOGM_P[ss]), np.mean(P["Yhat"][ss]),
                        yerr=np.std(P["Yhat"][ss]) / np.sqrt(ss.sum()), fmt="o", color="black", ms=6)
    for key, c, ls, lab in (("P_lee_printed", COL["lee_print"], "-", "Lee22 printed (= painted)"),
                            ("P_lee_a2", COL["lee_corr"], "--", "Lee22 x (1+z)^2"),
                            ("P_b12", COL["b12"], "-", "Battaglia12 x 0.5176")):
        ax.plot(mm, [d[key] for d in mi], color=c, ls=ls, label=lab)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(0.01, 8)
    ax.set_xlabel(r"$M_{200c}$ [$M_\odot$]")
    ax.set_title(f"z = {z:.2f}: " + r"$\hat Y = Y_{sph}(<R_{200c})/(P_{200}\,\frac{4\pi}{3}R_{200c}^3)$", fontsize=10)
    if j == 0:
        ax.set_ylabel(r"$\hat Y = 3\int_0^1 (P_e/P_{200}) x^2 dx$")
        ax.legend(fontsize=7, loc="upper left", ncol=2)
fig.suptitle("Inside R200c: CAMELS TNG gas (electron) fraction and electron-pressure integral vs Lee22 / Battaglia")
fig.savefig(os.path.join(OUT, "fig_D5_gas_fraction_Y.png"), dpi=120)
plt.close(fig)

# --- D6: emulation of Lee22 bookkeeping vs printed
from matplotlib.ticker import FuncFormatter, NullFormatter  # noqa: E402
fig, axs = plt.subplots(1, 2, figsize=(13, 5.6), constrained_layout=True)
for i, (qname, p, mkey) in enumerate((("n_e/n200", 3, "n_lee_printed"), ("P_e/P200", 2, "P_lee_printed"))):
    ax = axs[i]
    for (lo, hi), mk in zip(MB, ("o", "s", "^", "v")):
        r, nh, k, ek, c0, ec = trend[(qname, lo, "amp(0.04-1.34)")]
        re = r / (1 + zs) ** p
        ok = np.isfinite(re)
        if ok.sum() < 2:
            continue
        kk = np.polyfit(np.log10(1 + zs[ok]), np.log10(re[ok]), 1)[0] if ok.sum() > 2 else np.nan
        ax.plot(1 + zs[ok], re[ok], mk + "-", color="black", ms=4, lw=0.8,
                label=f"logM {lo}-{hi}: 27 boxes, truth x a^{p}; slope {kk:+.2f}")
        # explicit group_particles emulation on the directly computed boxes, same mass bin
        key = "nbar_emul" if p == 3 else "Pbar_emul"
        first = True
        for zz in sorted(set(Zd)):
            sd = (Zd == zz) & (LOGM_D >= lo) & (LOGM_D < hi)
            if sd.sum() < 2:
                continue
            rr = np.nanmean(D[key][sd], 0)[AMP_SL] / np.nanmean(DM[mkey][sd], 0)[AMP_SL]
            ax.plot(1 + zz, np.exp(np.nanmean(np.log(rr))), mk, mfc="none", mec=COL["other"], mew=1.8, ms=10,
                    label=(f"logM {lo}-{hi}: explicit emulation, direct boxes" if first else None))
            first = False
    ax.axhline(1, color=COL["lee_print"], lw=1.2, label="printed Lee22")
    if p == 3:
        ax.axhline(0.6016, color=COL["lee_prod"], lw=1.2, label="production 0.6016")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([1, 1.5, 2, 3])
    ax.set_xticklabels(["1", "1.5", "2", "3"])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: f"{v:g}" if f"{v:.0e}"[0] in "2358" else ""))
    ax.set_xlabel("1 + z")
    ax.set_ylabel(f"emulated Lee22 bookkeeping / printed Lee22 ({qname})")
    ax.set_title(("density: comoving (ckpc/h)^3 shells / physical n200  (= truth x a^3)" if p == 3 else
                  "pressure: comoving shells, comoving R200 in P200  (= truth x a^2)"), fontsize=10)
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2)
fig.suptitle("Does Lee22's bookkeeping applied to TNG-model (CAMELS) cells reproduce the printed fits?\n"
             "amplitude = geometric mean over 0.04-1.34 R200c; lines: 27 boxes, truth x a^3 (a^2); "
             "open symbols: explicit emulation, 4 direct boxes", fontsize=11)
fig.savefig(os.path.join(OUT, "fig_D6_lee22_emulation.png"), dpi=120)
plt.close(fig)

# --- D7: electron / star-forming corrections from the direct computation
if len(D["M200"]):
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.4), constrained_layout=True)
    cmap = plt.get_cmap("viridis")
    zl = sorted(set(Zd))
    for k, zz in enumerate(zl):
        s = (Zd == zz)
        c = cmap(k / max(len(zl) - 1, 1))
        lab = f"z = {zz:.2f} ({s.sum()} halos, {len(set(D['box'][s]))} boxes)"
        axs[0].plot(XG, np.nanmean(D["nbar_xe"][s], 0) / np.nanmean(D["nbar_fullion"][s], 0), color=c, label=lab)
        axs[1].plot(XG, np.nanmean(D["nbar_xe_nosf"][s], 0) / np.nanmean(D["nbar_xe"][s], 0), color=c, label=lab)
        axs[2].plot(XG, np.nanmean(D["Pbar_xe"][s], 0) / np.nanmean(D["Pbar_th"][s], 0), color=c, label=lab)
    for ax, t in zip(axs, (r"$n_e(x_e)$ / $n_e$(full ionisation) = $f_{x_e}$",
                           r"$n_e$(non-star-forming cells) / $n_e$(all)",
                           r"$P_e = n_e k_B T$ / $(0.5176\,P_{th})$")):
        ax.set_xscale("log")
        ax.axhline(1, color="0.5", lw=0.8)
        ax.set_xlabel(r"$x = r/R_{200c}$")
        ax.set_title(t, fontsize=10)
    axs[0].set_ylim(0.6, 1.1)
    axs[1].set_ylim(0.3, 1.1)
    axs[2].set_ylim(0.98, 1.02)
    axs[0].set_ylabel(r"ratio of halo-mean profiles ($M_{200c}\geq10^{12.5}M_\odot$)")
    axs[0].legend(fontsize=7)
    fig.suptitle("Direct computation from raw CAMELS snapshots: what the precomputed (gas-only) profiles miss")
    fig.savefig(os.path.join(OUT, "fig_D7_electron_corrections.png"), dpi=120)
    plt.close(fig)

with open(os.path.join(OUT, "analysis_summary.txt"), "w") as f:
    f.write("\n".join(summary) + "\n")
print("done")
