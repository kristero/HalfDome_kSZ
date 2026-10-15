#!/usr/bin/env python3
"""Independent re-derivation of the FRB halo-DM numbers of lee22_calibration.html version 4 (verify_v4, 2026-10-01).

Reads the Konietzka TNG300 arrays directly with numpy (not through tng_windows.py), tests the window identity under both readings,
builds the decades itself, and recomputes mean DM per hit / hit fractions from the raw TNG values, from the HalfDome summary CSVs
(exact per-ray moments) and from the HDF5 histograms (binned), plus the analytic npz statistics. Writes only to
outputs/lee22_verify_v4_20261001/frb_decades/. Nothing is uploaded."""
import csv
import json
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path("/home/cbllover/HalfDome")
FRB = ROOT / "frb_map_generation"
TNG = ROOT / "frb_catalog_comparison_outputs/haloDM-20260819T104507Z-1-001/haloDM"
OUT = FRB / "outputs/lee22_verify_v4_20261001/frb_decades"
OUT.mkdir(parents=True, exist_ok=True)
RAYDIRS = {
    ("b16", "idark"): FRB / "outputs/lee22_tngcal_20260929/rays",
    ("lee22_xgpnative", "idark"): FRB / "outputs/lee22_tngcal_20260929/rays",
    ("lee22_tngcal", "idark"): FRB / "outputs/lee22_tngcal_20260929/rays",
    ("lee22_phys", "idark"): FRB / "outputs/lee22_corrected_20260930/rays",
}
ZSER = FRB / "outputs/lee22_pdf_zseries_20261001/rays"
ANA = FRB / "outputs/lee22_corrected_20260930/analytic/analysis"
ZS = (0.1, 0.5, 1.0, 1.5, 2.0, 3.0)
NRAY = 120000

zgrid = np.load(TNG / "redshifts_DMhalo.npy")
NAMES = ["all", "mass10to12", "mass10to13", "mass10to14", "mass10to15", "mass11to14", "mass12to14", "mass12to15", "mass13to14",
         "mass9to13", "mass9to14"]
ARR = {n: np.load(TNG / ("DMhalo_r200_%s.npy" % n), mmap_mode="r") for n in NAMES}


def t(name, zs):
    i = int(np.flatnonzero(np.isclose(zgrid, zs))[0])
    return np.asarray(ARR[name][i], float)


log = open(OUT / "report.txt", "w")


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    log.write(s + "\n")


# ---------------- (1) identity tests at every catalogue redshift ----------------
say("== (1) window identity, all 55 z_s ==")
worst_new = 0.0; worst_nom_min = np.inf; worst_sub = 0.0; worst_nom2_min = np.inf
for zs in zgrid:
    a = {n: t(n, zs) for n in NAMES}
    new = np.abs(a["mass10to14"] - a["mass10to12"] - a["mass13to14"])          # [1e10,1e13)+[1e13,1e15) = [1e10,1e15)
    nom = np.abs(a["mass10to14"] - a["mass10to12"] - a["mass12to14"])          # nominal: [1e10,1e12)+[1e12,1e14) = [1e10,1e14)
    nom2 = np.abs((a["mass10to13"] - a["mass10to12"]) - (a["mass12to14"] - a["mass13to14"]))  # nominal: both = [1e12,1e13)
    sub = (a["mass12to14"] - a["mass13to14"]) - a["mass10to12"]               # new: [1e12,1e13) subset of [1e10,1e13) -> <= 0
    worst_new = max(worst_new, new.max())
    worst_nom_min = min(worst_nom_min, np.mean(nom > 1e-6))
    worst_nom2_min = min(worst_nom2_min, np.mean(nom2 > 1e-6))
    worst_sub = max(worst_sub, sub.max())
    if np.isclose(zs, 0.1) or np.isclose(zs, 1.0) or np.isclose(zs, 3.0):
        say("z_s %.1f: max|10to14-10to12-13to14| = %.2e; nominal 10to14-10to12-12to14 fails (>1e-6) on %.1f %% of rays, median |res| of failing %.2f;"
            " nominal 10to13-10to12 = 12to14-13to14 fails on %.1f %%; max[(12to14-13to14) - 10to12] = %.2e"
            % (zs, new.max(), 100 * np.mean(nom > 1e-6), np.median(nom[nom > 1e-6]), 100 * np.mean(nom2 > 1e-6), sub.max()))
say("ALL z_s: max new-identity residual %.2e; nominal identity fails on >= %.1f %% of rays at every z_s; nominal #2 fails on >= %.1f %%;"
    " max subset violation %.2e" % (worst_new, 100 * worst_nom_min, 100 * worst_nom2_min, worst_sub))

# genuine small values: is the 1e-9 clip safe?
for zs in (0.1, 1.0, 3.0):
    raw = np.concatenate([t(n, zs) for n in ("mass10to12", "mass13to14", "mass10to13")])
    pos = raw[raw > 0]
    d13 = t("mass10to13", zs) - t("mass10to12", zs)
    d14 = t("mass10to14", zs) - t("mass10to13", zs)
    say("z_s %.1f: smallest genuine positive raw value %.3e; decade 1e13-14 diffs in (0,1e-9]: %d, in (-inf,0): %d, |neg| max %.1e;"
        " 1e14-15 diffs in (0,1e-9]: %d, neg %d; diffs in (1e-9, 1e-3): %d / %d"
        % (zs, pos.min(), np.sum((d13 > 0) & (d13 <= 1e-9)), np.sum(d13 < 0), (np.abs(d13[d13 < 0]).max() if np.any(d13 < 0) else 0),
           np.sum((d14 > 0) & (d14 <= 1e-9)), np.sum(d14 < 0), np.sum((d13 > 1e-9) & (d13 < 1e-3)), np.sum((d14 > 1e-9) & (d14 < 1e-3))))


def clip(v):
    return np.where(v > 1e-9, v, 0.0)


def tng_dec(zs):
    """[1e13,1e14) built by me: under 'AtoB'=[10^A,10^(B+1)), 10to13=[1e10,1e14), 10to12=[1e10,1e13)."""
    return clip(t("mass10to13", zs) - t("mass10to12", zs))


def tng_12_14(zs):
    """[1e12,1e14) = 12to14 [1e12,1e15) - (10to14-10to13) [1e14,1e15); alternatively 10to13 - [1e10,1e12) where [1e10,1e12)
    = 10to14 - 12to14. Both built, compared; clipped."""
    v1 = t("mass12to14", zs) - clip(t("mass10to14", zs) - t("mass10to13", zs))
    v2 = t("mass10to13", zs) - (t("mass10to14", zs) - t("mass12to14", zs))
    return v1, v2


# ---------------- (2) TNG means per hit ----------------
edges = None


def ray_file(tag, zs, where):
    folder = RAYDIRS[(tag, "idark")] if where == "idark" else (ZSER if where == "local" else where)
    return folder / ("zsrc%s_nside4096_nrays120000_allhalos_sphere1p0_m200c_%s_seed42" % (("%.1f" % zs).replace(".", "p"), tag))


def ray_stats(path, win):
    with h5py.File(path / "halfdome_uniform_fixedz_foreground_mass_histograms.h5", "r") as h:
        labels = [x.decode() if isinstance(x, bytes) else str(x) for x in h["window_label"][:]]
        e = np.asarray(h["pdf_bin_edges_pc_cm3"][:], float)
        cnt = np.asarray(h["pdf_count"][:], np.int64)
        attrs = {k: (v.decode() if isinstance(v, bytes) else v) for k, v in h.attrs.items()}
    if cnt.shape[0] != len(labels):
        cnt = cnt.T
    c = cnt[labels.index(win)].astype(float)
    cen = np.sqrt(e[1:] * e[:-1])
    with open(path / "halfdome_uniform_fixedz_foreground_mass_histograms_summary.csv") as fh:
        row = [r for r in csv.DictReader(fh) if r["label"] == win][0]
    hit = 1 - float(row["zero_fraction"])
    mean_all = float(row["dm_mean_pc_cm3"]); std_all = float(row["dm_std_pc_cm3"])
    # per-hit second moment from all-ray moments: E[x^2] = std^2 + mean^2 (population std assumed)
    m2 = (std_all ** 2 + mean_all ** 2) / hit
    mph = mean_all / hit
    sd_hit = np.sqrt(max(m2 - mph ** 2, 0))
    return dict(e=e, c=c, cen=cen, hit=hit, mph_exact=mph, sd_hit=sd_hit, nhit=hit * NRAY, mph_binned=float((c * cen).sum() / c.sum()),
                n_in_hist=c.sum(), attrs=attrs, zscale=str(attrs.get("provenance_lee2022_cache_signature", attrs.get("provenance_dm_model_family", "")))[-60:],
                julia=attrs.get("provenance_julia_version"))


def tng_stats(v, e):
    cen = np.sqrt(e[1:] * e[:-1])
    hit = np.mean(v != 0)
    hv = v[v != 0]
    c, _ = np.histogram(v, bins=e)
    return dict(hit=hit, mph_exact=hv.mean(), sd_hit=hv.std(), nhit=hv.size, mph_binned=float((c * cen).sum() / c.sum()), c=c.astype(float))


rows = []
say("\n== (2) mean DM per hit, 10^13-10^14 ==")
say("z_s  | TNG decade exact/binned hit% | '13to14' exact/binned hit% | nominal-ratio | models: exact mean/hit (binned) ratio-to-TNG(exact) hit%")
TAGS = ("b16", "lee22_xgpnative", "lee22_tngcal", "lee22_phys")
res = {}
for zs in ZS:
    where = "idark" if zs <= 1.0 else "local"
    ref = ray_stats(ray_file("b16", zs, where), "m1e13_to_1e14")
    e = ref["e"]
    td = tng_stats(tng_dec(zs), e)
    ta = tng_stats(t("mass13to14", zs), e)
    res[(zs, "TNG")] = td; res[(zs, "TNGarr")] = ta
    line = "%.1f | %.2f/%.2f %.2f%% | %.2f/%.2f %.2f%% | %.3f |" % (zs, td["mph_exact"], td["mph_binned"], 100 * td["hit"], ta["mph_exact"],
                                                                 ta["mph_binned"], 100 * ta["hit"], ta["mph_exact"] / td["mph_exact"])
    for tag in TAGS:
        if tag == "lee22_tngcal" and zs > 1.0:
            continue
        r = ray_stats(ray_file(tag, zs, where), "m1e13_to_1e14")
        res[(zs, tag)] = r
        line += " %s %.2f (%.2f) %.4f %.2f%% |" % (tag, r["mph_exact"], r["mph_binned"], r["mph_exact"] / td["mph_exact"], 100 * r["hit"])
        rows.append(dict(z_s=zs, model=tag, where=where, julia=r["julia"], hit_pct=100 * r["hit"], mph_exact=r["mph_exact"],
                         mph_binned=r["mph_binned"], ratio_exact=r["mph_exact"] / td["mph_exact"],
                         ratio_binned=r["mph_binned"] / td["mph_binned"], tng_mph_exact=td["mph_exact"], tng_mph_binned=td["mph_binned"],
                         tng_hit_pct=100 * td["hit"], tng_arr_mph=ta["mph_exact"], tng_arr_hit_pct=100 * ta["hit"]))
    say(line)
    # abundance ratio from hit fractions (Poisson)
    say("   Poisson hits/ray TNG %.4f HalfDome %.4f -> ratio %.3f; raw hit ratio %.3f; model signatures: %s" % (
        -np.log(1 - td["hit"]), -np.log(1 - ref["hit"]), np.log(1 - td["hit"]) / np.log(1 - ref["hit"]), td["hit"] / ref["hit"],
        "; ".join("%s:%s:%s" % (tg, res[(zs, tg)]["julia"], res[(zs, tg)]["zscale"][-34:]) for tg in TAGS if (zs, tg) in res)))

with open(OUT / "mean_per_hit_m1e13_1e14.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

say("\nratios (exact) corrected:", ", ".join("%.3f" % (res[(z, "lee22_phys")]["mph_exact"] / res[(z, "TNG")]["mph_exact"]) for z in ZS))
say("ratios (exact) pipeline:", ", ".join("%.3f" % (res[(z, "lee22_xgpnative")]["mph_exact"] / res[(z, "TNG")]["mph_exact"]) for z in ZS))
say("ratios (exact) B16:", ", ".join("%.3f" % (res[(z, "b16")]["mph_exact"] / res[(z, "TNG")]["mph_exact"]) for z in ZS))
say("ratios (exact) C(z):", ", ".join("%.3f" % (res[(z, "lee22_tngcal")]["mph_exact"] / res[(z, "TNG")]["mph_exact"]) for z in ZS[:3]))
say("C(z) vs '13to14' array:", ", ".join("%.3f" % (res[(z, "lee22_tngcal")]["mph_exact"] / res[(z, "TNGarr")]["mph_exact"]) for z in ZS[:3]))
# statistical error of the TNG and ray means
for z in ZS:
    td = res[(z, "TNG")]; r = res[(z, "lee22_phys")]
    say("z_s %.1f: TNG mean/hit err %.2f %% ; corrected rays err %.2f %%" % (z, 100 * td["sd_hit"] / td["mph_exact"] / np.sqrt(td["nhit"]),
                                                                         100 * r["sd_hit"] / r["mph_exact"] / np.sqrt(r["nhit"])))

# ---------------- analytic (TNG cosmology) ----------------
say("\n== analytic npz (TNG cosmology, Tinker08) and validation (HalfDome abundance) ==")
AM = {"b16": "b16", "lee22_xgpnative": "lee22_noconc_xgpnative", "lee22_tngcal": "lee22_noconc_tngcal", "lee22_phys": "lee22_noconc_phys"}
for zs in (0.1, 0.5, 1.0):
    for win in ("m1e13_to_1e14", "m1e12_to_1e14"):
        if win == "m1e12_to_1e14" and zs != 1.0:
            continue
        parts = []
        for tag, am in AM.items():
            d = np.load(ANA / ("tng_%s_%s_zs%s.npz" % (am, win, zs)))
            st = json.loads(str(d["stats_clustered"])); run = json.loads(str(d["run"]))
            v = np.load(ANA / ("validation_%s_catalog_%s_zs%s.npz" % (am, win, zs)))
            sv = json.loads(str(v["stats_clustered"]))
            r = ray_stats(ray_file(tag, zs, "idark"), win)
            if win == "m1e13_to_1e14":
                tm = res[(zs, "TNG")]["mph_exact"]
            else:
                v1, v2 = tng_12_14(zs)
                tm = v1[v1 != 0].mean()
            parts.append("%s an %.1f (hit %.1f%%, %s/%s) an/TNG %.3f | valid %.1f vs rays %.1f (%.3f) hit %.1f vs %.1f" % (
                tag, st["mean_dm_hit"], 100 * st["hit_fraction"], run["cosmology"], run["abundance"], st["mean_dm_hit"] / tm,
                sv["mean_dm_hit"], r["mph_exact"], sv["mean_dm_hit"] / r["mph_exact"], 100 * sv["hit_fraction"], 100 * r["hit"]))
        say("z_s %.1f %s:\n   " % (zs, win) + "\n   ".join(parts))

# ---------------- Fig. 4 right panel: 1e12-1e14 at z_s = 1 ----------------
say("\n== 1e12-1e14 at z_s = 1 ==")
v1, v2 = tng_12_14(1.0)
say("two constructions agree to %.2e; hit (v!=0) %.2f %%, hit (|v|>1e-9) %.2f %%, negatives %d, |v|<=1e-9 nonzero %d; mean/hit %.2f"
    % (np.max(np.abs(v1 - v2)), 100 * np.mean(v1 != 0), 100 * np.mean(np.abs(v1) > 1e-9), np.sum(v1 < 0), np.sum((v1 != 0) & (np.abs(v1) <= 1e-9)),
       v1[np.abs(v1) > 1e-9].mean()))
# tng_windows-style (unclipped outer subtraction): emulate exactly
tw = t("mass12to14", 1.0) - clip(t("mass10to14", 1.0) - t("mass10to13", 1.0))
say("tng_windows-style m1e12_to_1e14: nonzero %.3f %%, in (0,1e-9] or negative: %d" % (100 * np.mean(tw != 0), np.sum((tw != 0) & (tw <= 1e-9))))
for tag in ("lee22_phys", "b16", "lee22_xgpnative", "lee22_tngcal"):
    r = ray_stats(ray_file(tag, 1.0, "idark"), "m1e12_to_1e14")
    say("  %s rays: hit %.2f %% mean/hit %.2f" % (tag, 100 * r["hit"], r["mph_exact"]))

# ---------------- low-DM deficit ----------------
say("\n== low-DM side: corrected vs TNG decade, P(DM<10 | hit) and per-bin residuals below 10 ==")
for zs in ZS:
    td = res[(zs, "TNG")]; e = res[(zs, "b16")]["e"]; cen = np.sqrt(e[1:] * e[:-1])
    vals = tng_dec(zs); hv = vals[vals != 0]
    for tag in ("lee22_phys", "b16", "lee22_xgpnative"):
        r = res[(zs, tag)]
        # pdf conditional on in-histogram counts (as in the figures)
        pr = r["c"] / r["c"].sum(); pt = td["c"] / td["c"].sum()
        lo = cen < 10
        frac_r = pr[lo].sum(); frac_t = pt[lo].sum()
        rob = (r["c"] >= 10) & (td["c"] >= 10) & lo
        band = rob & (cen > 2)
        res_bins = pr[band] / pt[band] - 1
        if tag == "lee22_phys":
            for lo_, hi_ in ((2, 5), (5, 10), (10, 20)):
                m = (cen >= lo_) & (cen < hi_)
                say("   z_s %.1f corrected P(%g<=DM<%g) ratio to TNG %.3f" % (zs, lo_, hi_, pr[m].sum() / pt[m].sum()))
        say("z_s %.1f %-16s P(DM<10|hit) model %.4f TNG %.4f ratio %.3f; robust bins 2-10: n=%d median resid %+.0f %% (IQR %+.0f..%+.0f)"
            % (zs, tag, frac_r, frac_t, frac_r / frac_t, band.sum(), 100 * np.median(res_bins) if band.any() else np.nan,
               100 * np.percentile(res_bins, 25) if band.any() else np.nan, 100 * np.percentile(res_bins, 75) if band.any() else np.nan))

# ---------------- idark vs local at z_s = 1 ----------------
say("\n== idark vs local rerun, corrected, z_s = 1 ==")
vdir = FRB / "outputs/lee22_pdf_zseries_20261001/validation/zsrc1p0_nside4096_nrays120000_allhalos_sphere1p0_m200c_lee22_phys_seed42"
for win in ("m1e13_to_1e14", "m1e12_to_1e14", "m1e10_to_1e16", "m1e14_to_1e16"):
    a = ray_stats(ray_file("lee22_phys", 1.0, "idark"), win)
    b = ray_stats(vdir, win)
    err = np.hypot(a["sd_hit"] / np.sqrt(a["nhit"]) / a["mph_exact"], b["sd_hit"] / np.sqrt(b["nhit"]) / b["mph_exact"])
    say("%s: idark %.3f (julia %s) local %.3f (julia %s): local/idark - 1 = %+.2f %% +- %.2f %% (stat); binned %+.2f %%; hit %.2f vs %.2f %%"
        % (win, a["mph_exact"], a["julia"], b["mph_exact"], b["julia"], 100 * (b["mph_exact"] / a["mph_exact"] - 1), 100 * err,
           100 * (b["mph_binned"] / a["mph_binned"] - 1), 100 * a["hit"], 100 * b["hit"]))

# ---------------- Sec. 3 cross-check numbers (consistency only) ----------------
say("\n== Sec. 3 / 'Not the finite box' cross-check ==")
for zs in (0.1, 0.5, 1.0):
    d1415 = clip(t("mass10to14", zs) - t("mass10to13", zs))
    d1516 = clip(t("mass10to15", zs) - t("mass10to14", zs))
    r = ray_stats(ray_file("lee22_phys", zs, "idark"), "m1e14_to_1e16")
    r2 = ray_stats(ray_file("lee22_phys", zs, "idark"), "m1e15_to_1e16")
    approx = 1 - (1 - r["hit"]) / (1 - r2["hit"])
    say("z_s %.1f: TNG 1e14-15 decade hit %.3f %%, 10to15-10to14 nonzero %.3f %%; HalfDome 1e14-16 %.2f %%, 1e15-16 %.3f %%, 1e14-15 approx %.2f %%"
        % (zs, 100 * np.mean(d1415 > 0), 100 * np.mean(d1516 > 0), 100 * r["hit"], 100 * r2["hit"], 100 * approx))
log.close()
