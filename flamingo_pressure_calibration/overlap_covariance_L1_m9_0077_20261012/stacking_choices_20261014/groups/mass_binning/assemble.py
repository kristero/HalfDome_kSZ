"""Mass-binning group: merge the part results into results.json and the web records into web_variants.json."""
import json

import numpy as np

from mbcommon import (BAND_LABELS, BANDS, CAL_LABELS, CENSUS_BINS, EDGES_FIT, FIELDS_WEB, GLOBAL_SEED, GROUP, N_BOOT,
                      OBS_KEEP, WORK, band_ratios, dump, impacts, load, ref_key, sc, sig, strict_mask, wmean_boot)

G = "mass_binning"
rs = json.loads((WORK / "res_schemes.json").read_text())
rn = json.loads((WORK / "res_norm.json").read_text())
rz = json.loads((WORK / "res_samplesize.json").read_text())
rt = json.loads((WORK / "res_topbins.json").read_text())
rr = json.loads((WORK / "res_rebin_test.json").read_text())
ri = json.loads((WORK / "res_interp_test.json").read_text())

KEY_TEMPLATE = "{field}|geometric|removed|all|all|fit|{P200|rhoc_fb|T200|K200} (T_e, K: estimator 'weighted')"
BASE_SETTINGS = {"key_template": KEY_TEMPLATE, "statistic": "mean", "rule": "strict", "min_count": 20,
                 "softening_factor": 4.0, "hsml_factor": 2.0, "scheme": "M200c_0.2", "sample": "all haloes of the bin"}


def settings(**kw):
    s = dict(BASE_SETTINGS)
    s.update(kw)
    return s


SCHEME_META = {
    "M200c_0.1": ("Mass bins: width", "0.1 dex in M200c",
                  "Bins of 0.1 dex in M200c (29 bins, about 150 haloes each below 10^14.5) instead of 0.2 dex."),
    "M200c_0.3": ("Mass bins: width", "0.3 dex in M200c",
                  "Bins of 0.3 dex in M200c (10 bins) instead of 0.2 dex; the top bin 15.2-15.5 holds 13 haloes."),
    "M200c_0.5": ("Mass bins: width", "0.5 dex in M200c",
                  "Bins of 0.5 dex in M200c (6 bins) instead of 0.2 dex; each stack mixes haloes over a factor 3 "
                  "in mass."),
    "M500c_0.2": ("Mass bins: binning variable", "0.2 dex in M500c",
                  "Haloes binned by M500c (0.2 dex) instead of M200c; profiles still in R200c / P200 units. "
                  "logm is the mean log M500c."),
    "M500c_0.5": ("Mass bins: binning variable", "0.5 dex in M500c (B24-like)",
                  "B24-like 0.5-dex M500c bins (13.0-15.5) instead of 0.2-dex M200c bins; profiles still in "
                  "R200c / P200 units. logm is the mean log M500c."),
    "Lee22_h": ("Mass bins: h^-1 units (Lee22)", "0.2 dex in h^-1 Msun (Lee22)",
                "Lee22's 0.2-dex bins in M200c [h^-1 Msun] (h = 0.681). In Msun every edge lies 0.167 dex above its "
                "label (e.g. 13.4-13.6 h^-1 Msun = 13.567-13.767 Msun), so its nodes sit 0.03-0.07 dex from the "
                "reference nodes; logm is in h^-1 Msun. Lee22's >29-halo rule would drop 15.0-15.2 (17) and "
                "15.2-15.4 (2)."),
}

variants = [{"id": "ref", "dimension": "Reference", "label": "Reference (0.2 dex M200c)",
             "description": "Inclusion-weighted mean of each halo's own P/P200 (n_e, rho, T_e, K in their own "
                            "self-similar units) at r/R200c, 0.2-dex M200c bins, all haloes, strict resolution.",
             "fields": list(FIELDS_WEB), "scheme": "M200c_0.2", "grid": "fit", "settings": settings()}]
for S, (dim, lab, desc) in SCHEME_META.items():
    variants.append({"id": f"{G}:{S}", "dimension": dim, "label": lab, "description": desc,
                     "fields": list(FIELDS_WEB), "scheme": S, "grid": "fit", "settings": settings(scheme=S)})
variants += [
    {"id": f"{G}:binmean_norm", "dimension": "Normalisation: own halo vs bin mean",
     "label": "Bin-mean P200, R200c (0.2 dex)",
     "description": "Profiles stacked at fixed physical radius in units of the bin's mean-mass R200c and divided by "
                    "the mean-mass P200 (T200, K200) instead of each halo's own R200c and P200.",
     "fields": list(FIELDS_WEB), "scheme": "M200c_0.2", "grid": "fit",
     "settings": settings(key_template=KEY_TEMPLATE + " rescaled from x200 to bin-mean units")},
    {"id": f"{G}:binmean_norm_0.5", "dimension": "Normalisation: own halo vs bin mean",
     "label": "Bin-mean P200, R200c (0.5 dex)",
     "description": "As 'bin-mean P200, R200c' but in 0.5-dex M200c bins; compare with the 0.5-dex own-normalised "
                    "variant (same haloes).",
     "fields": list(FIELDS_WEB), "scheme": "M200c_0.5", "grid": "fit",
     "settings": settings(scheme="M200c_0.5", key_template=KEY_TEMPLATE + " rescaled from x200 to bin-mean units")},
]
for N in (25, 50, 100):
    variants.append({"id": f"{G}:N{N}", "dimension": "Sample size per bin", "label": f"{N} random haloes per bin",
                     "description": f"One random subsample of {N} haloes per 0.2-dex bin (bins with more than {N} "
                                    "haloes; inclusion weights kept); err is that subsample's own bootstrap error and "
                                    "the strict rule is applied to the subsample.",
                     "fields": list(FIELDS_WEB), "scheme": "M200c_0.2", "grid": "fit",
                     "settings": settings(sample=f"{N} random haloes per bin (realisation 0)")})
variants += [
    {"id": f"{G}:pool_15.1", "dimension": "Sparse top bins", "label": "Pooled >= 10^15.1 (27 haloes)",
     "description": "The two top bins (21 + 6 haloes) stacked together as one bin 15.1-15.5.",
     "fields": list(FIELDS_WEB), "scheme": "M200c_0.2", "grid": "fit", "settings": settings(sample="pooled 15.1-15.5")},
    {"id": f"{G}:pool_14.9", "dimension": "Sparse top bins", "label": "Pooled >= 10^14.9 (154 haloes)",
     "description": "The three top bins (127 + 21 + 6 haloes) stacked together as one bin 14.9-15.5.",
     "fields": list(FIELDS_WEB), "scheme": "M200c_0.2", "grid": "fit", "settings": settings(sample="pooled 14.9-15.5")},
    {"id": f"{G}:trend_top", "dimension": "Sparse top bins", "label": "Trend fit 14.3-15.1, extrapolated",
     "description": "Top-bin profiles predicted by a per-shell fit of ln F linear in log M200c over the four bins "
                    "14.3-15.1 (1,665 haloes; bootstrap-variance weights). N is the top bin's own halo count, n_used "
                    "the node haloes entering the fit per shell; no per-halo percentiles.",
     "fields": list(FIELDS_WEB), "scheme": "M200c_0.2", "grid": "fit",
     "settings": settings(sample="trend fit over the bins 14.3-15.1 (1,665 haloes)")},
]

# ------------------------------------------------------------------------------------------------- web records
records = []
for fn in ("web_schemes.json", "web_norm.json", "web_samplesize.json", "web_topbins.json"):
    records += json.loads((WORK / fn).read_text())
ids = {v["id"] for v in variants}
assert {r["variant"] for r in records} <= ids, {r["variant"] for r in records} - ids
web = {"group": G, "variants": variants, "records": records}
dump(web, GROUP / "web_variants.json")
size = (GROUP / "web_variants.json").stat().st_size
print("web_variants.json", size / 1e6, "MB", len(records), "records")

# ----------------------------------------------------------------------------- per-variant comparisons (results)
per_variant = {}
CP = rs["scheme_impacts_continuous_painting"]
per_variant["ref"] = {
    "comparator": "itself",
    "impacts_vs_truth": {f: {"impact": CP["M200c_0.2"][f]["vs_truth"]} for f in ("P", "E")},
    "impact_method": "continuous painting through the 15 reference nodes vs the true bin means (the reference's own "
                     "painting error)"}
for S in SCHEME_META:
    per_variant[f"{G}:{S}"] = {
        "comparator": "reference (M200c_0.2) interpolated to the bin's mean log M200c (ln F linear in log M)",
        "bands": {f: rs["scheme_vs_reference_same_mass"][f][S] for f in FIELDS_WEB},
        "impacts": {f: {"impact": CP[S][f]["vs_reference_painted_same_way"]} for f in ("P", "E")},
        "impacts_vs_truth": {f: {"impact": CP[S][f]["vs_truth"]} for f in ("P", "E")},
        "impacts_at_reference_nodes": rs["scheme_impacts_smooth_painting"][S],
        "impact_method": "impacts: continuous painting (every calibrated halo at its own log M200c through the "
                         "scheme's nodes) relative to the reference painted the same way (metric C); "
                         "impacts_vs_truth: the same vs the true bin means (metric B); impacts_at_reference_nodes: the "
                         "scheme interpolated to the reference's node masses vs the reference stacks (metric A, "
                         "exact for the reference by construction)"}
for S, vid in (("M200c_0.2", "binmean_norm"), ("M200c_0.5", "binmean_norm_0.5")):
    per_variant[f"{G}:{vid}"] = {
        "comparator": "own-normalised stack of the same haloes" + (" (the reference)" if S == "M200c_0.2" else
                                                                   " (variant mass_binning:M200c_0.5)"),
        "bands": {f: [{"bin": e["bin"], "mean_logm200c": e["mean_logm200c"], "N": e["N"], "bands": e["fields"][f]}
                      for e in rn["bins"][S]] for f in FIELDS_WEB},
        "impacts": {f: {"impact": rn["impacts"][S][f]} for f in ("P", "E")},
        "bin_details": [{k: e[k] for k in ("bin", "s_range", "sd_ln_s", "Pbar_arith_over_geo")} for e in rn["bins"][S]]}

# subsample web variants: realisation-0 stacks vs the full bins (paired global bootstrap) and their impacts
rows_npz = np.load(WORK / "samplesize_web_rows.npz")
bins = sc.mass_bins("M200c_0.2")
for N in (25, 50, 100):
    bands_f = {}
    cal_v, cal_b, cal_rv, cal_rb = {f: [] for f in ("P", "E")}, {f: [] for f in ("P", "E")}, \
        {f: [] for f in ("P", "E")}, {f: [] for f in ("P", "E")}
    for f in FIELDS_WEB:
        F = load(ref_key(f))
        out = []
        for b in bins:
            vf, bf, _ = wmean_boot(F, b["rows"])
            key = f"N{N}__{b['label']}"
            if key in rows_npz.files:
                sub = b["rows"][rows_npz[key]]
                vs, bs, _ = wmean_boot(F, sub)
                out.append({"bin": b["label"], "N": int(N), "bands": band_ratios(vs, vf, bs, bf, strict_mask(b["rows"]))})
            else:
                vs, bs = vf, bf
            if f in ("P", "E") and b["label"] in CAL_LABELS:
                cal_v[f].append(vs)
                cal_b[f].append(bs)
                cal_rv[f].append(vf)
                cal_rb[f].append(bf)
        bands_f[f] = out
    imp = {}
    for f in ("P", "E"):
        imp[f] = {"impact": impacts(np.array(cal_v[f]), np.array(cal_rv[f]), np.moveaxis(np.array(cal_b[f]), 0, 1),
                                    np.moveaxis(np.array(cal_rb[f]), 0, 1), f),
                  "realisation_scatter_pct": {k: rz["observable_scatter"][str(N)][f][k] for k in OBS_KEEP[f]}}
    per_variant[f"{G}:N{N}"] = {
        "comparator": "full bin (same 0.2-dex bin, all haloes); full-bin strict mask",
        "bands": bands_f, "impacts": imp,
        "impact_method": "realisation 0 in every calibrated bin with more than N haloes (others full); "
                         "realisation_scatter_pct: SD over 400 independent realisations (see sample_size)"}
# top-bin variants
for pk, imp_key in (("pool_15.1", "pool_15.1_step"), ("pool_14.9", "pool_14.9_step")):
    per_variant[f"{G}:{pk}"] = {
        "comparator": "each pooled bin's own stack (the reference)",
        "bands": {f: [{"bin": lab, "bands": rt["pooled"][pk][f]["vs_bins"][lab]} for lab in rt["pooled"][pk][f]["vs_bins"]]
                  for f in FIELDS_WEB},
        "impacts": {f: {"impact": rt["impacts"][imp_key][f]} for f in ("P", "E")},
        "impact_method": "step: every pooled bin painted with the pooled stack"}
per_variant[f"{G}:pool_15.1"]["impacts_smooth"] = {f: rt["impacts"]["pool_15.1_smooth"][f] for f in ("P", "E")}
per_variant[f"{G}:pool_14.9"]["impacts_smooth"] = {f: rt["impacts"]["pool_14.9_smooth"][f] for f in ("P", "E")}
for pk, ck in (("pool_15.1", "pool_15.1_node"), ("pool_14.9", "pool_14.9_node")):
    per_variant[f"{G}:{pk}"]["impacts_continuous"] = rt["continuous_painting"][ck]
per_variant[f"{G}:trend_top"] = {
    "comparator": "the measured top-bin stacks (reference)",
    "bands": {f: [{"bin": lab, "bands": rt["trend"][f][lab]["trend_over_measured"]} for lab in ("15.1-15.3", "15.3-15.5")]
              for f in FIELDS_WEB},
    "impacts": {f: {"impact": rt["impacts"]["trend_14.3-15.1"][f]} for f in ("P", "E")},
    "impact_method": "both top bins painted with the trend prediction (shells with < 2 resolved nodes keep the measured "
                     "value)",
    "impacts_continuous": rt["continuous_painting"]["trend_14.3-15.1_nodes"]}


# ---------------------------------------------------------------------------- derived summary tables (for the text)
def nneed(sigma1, p):
    return None if sigma1 is None else int(np.ceil((sigma1 / p)**2))


nneeded = {}
for lab, d in rz["analytic_sigma1"].items():
    nneeded[lab] = {"N": d["N"], "population": d["population"]}
    for f in ("P", "E", "G", "T_e", "K"):
        nneeded[lab][f] = {bl: ({"sigma1": d[f][bl], "N_1pct": nneed(d[f][bl], 0.01), "N_3pct": nneed(d[f][bl], 0.03),
                                 "N_5pct": nneed(d[f][bl], 0.05),
                                 "err_current_pct": sig(100 * d[f][bl] / np.sqrt(d["N_eff"])),
                                 "err_complete_pct": sig(100 * d[f][bl] / np.sqrt(d["population"]))}
                                if d[f][bl] is not None else None) for bl in BAND_LABELS}
# subsampling-based sigma1 (median over N = 25, 50, 100) for the detail bins
nneeded_sub = {}
for lab, d in rz["precision"].items():
    nneeded_sub[lab] = {}
    for f in ("P", "E", "G", "T_e", "K"):
        nneeded_sub[lab][f] = {}
        for bl in BAND_LABELS:
            vals = [d[str(N)][f][bl]["sigma1"] for N in (25, 50, 100) if d.get(str(N), {}).get(f, {}).get(bl)]
            vsh = [d[str(N)][f][bl]["sigma1_shell_median"] for N in (25, 50, 100)
                   if d.get(str(N), {}).get(f, {}).get(bl)]
            if not vals:
                nneeded_sub[lab][f][bl] = None
                continue
            s1, s1s = float(np.median(vals)), float(np.median(vsh))
            nneeded_sub[lab][f][bl] = {"sigma1_band": sig(s1), "N_1pct": nneed(s1, .01), "N_3pct": nneed(s1, .03),
                                       "N_5pct": nneed(s1, .05), "sigma1_shell": sig(s1s),
                                       "N_3pct_single_shell": nneed(s1s, .03)}


def pulls(scheme, field, lo=13.5, hi=15.1):
    """RMS of (ratio - 1)/err over calibrated-range bins and resolved bands, and the count of |pull| > 3."""
    pl = []
    for e in rs["scheme_vs_reference_same_mass"][field][scheme]:
        if not (lo <= e["mean_logm200c"] < hi):
            continue
        for bl, v in e["bands"].items():
            if v and v["err"]:
                pl.append((v["ratio"] - 1) / v["err"])
    pl = np.array(pl)
    return {"n": int(len(pl)), "rms_pull": sig(np.sqrt(np.mean(pl**2))) if len(pl) else None,
            "n_abs_gt_3": int((np.abs(pl) > 3).sum()), "mean_pull": sig(np.mean(pl)) if len(pl) else None}


pull_tab = {S: {f: pulls(S, f) for f in ("P", "E", "G", "T_e", "K")} for S in SCHEME_META}


def wide_summary(S, field, lo, hi):
    """wide_vs_fine 'vs_0p1_at_same_mean_mass' over bins with lo <= mean log M200c < hi (all bands with an error)."""
    dev, pl, used = [], [], []
    for e in rs["wide_vs_fine"][field][S]:
        if not (lo <= e["mean_logm200c"] < hi):
            continue
        used.append(e["bin"])
        for bl, v in e["vs_0p1_at_same_mean_mass"].items():
            if v and v["err"]:
                dev.append(abs(v["ratio"] - 1))
                pl.append((v["ratio"] - 1) / v["err"])
    dev, pl = np.array(dev), np.array(pl)
    addv = [abs(v["ratio"] - 1) for e in rs["wide_vs_fine"][field][S] if lo <= e["mean_logm200c"] < hi
            for v in e["vs_weighted_avg_of_subbins"].values() if v]
    return {"bins": used, "n_band_values": int(len(dev)), "median_abs_dev_pct": sig(100 * np.median(dev)),
            "max_abs_dev_pct": sig(100 * dev.max()), "rms_pull": sig(np.sqrt(np.mean(pl**2))),
            "n_abs_pull_gt_3": int((np.abs(pl) > 3).sum()),
            "additivity_max_abs_dev": sig(max(addv)) if addv else None}


wide_sum = {"selection_note": "bins of the wide scheme whose inclusion-weighted mean log M200c lies in the range; "
                              "'calibrated' = [13.5, 15.5), 'all' = [12.5, 15.5); every band with a bootstrap error "
                              "(shells resolved in the wide bin and both bracketing 0.1-dex bins); bins whose 0.1-dex "
                              "comparator is extrapolated carry 'extrapolated': true in analyses.wide_vs_fine.",
            "calibrated": {S: {f: wide_summary(S, f, 13.5, 15.5) for f in ("P", "E", "T_e", "K")}
                           for S in ("M200c_0.2", "M200c_0.3", "M200c_0.5")},
            "all": {S: {f: wide_summary(S, f, 12.5, 15.5) for f in ("P", "E", "T_e", "K")}
                    for S in ("M200c_0.2", "M200c_0.3", "M200c_0.5")}}


def rng_of(pairs):
    pairs = [(lab, v) for lab, v in pairs if v is not None]
    lo = min(pairs, key=lambda t: t[1])
    hi = max(pairs, key=lambda t: t[1])
    return {"min": lo[1], "min_bin": lo[0], "max": hi[1], "max_bin": hi[0]}


ranges = {}
for f in ("P", "E"):
    for bl, pk in (("0.5-1", "N_3pct"), ("2-3", "N_3pct"), ("3-4", "N_5pct")):
        ranges[f"subsampling_sigma1|{f}|{bl}"] = rng_of([(lab, d[f][bl]["sigma1_band"]) for lab, d in nneeded_sub.items()
                                                       if d[f][bl]])
        ranges[f"subsampling_{pk}|{f}|{bl}"] = rng_of([(lab, d[f][bl][pk]) for lab, d in nneeded_sub.items()
                                                     if d[f][bl]])
        ranges[f"analytic_sigma1_calibrated|{f}|{bl}"] = rng_of([(lab, nneeded[lab][f][bl]["sigma1"])
                                                                for lab in CAL_LABELS if nneeded[lab][f][bl]])
m5 = rs["m500_vs_m200"]["per_M200c_bin"]
ranges["sd_dlogm_200c_500c|all15"] = rng_of([(e["bin"], e["sd_dlogm"]) for e in m5])
ranges["corr_dlogm_logc|all15"] = rng_of([(e["bin"], e["corr_with_log_c200c"]) for e in m5])
ranges["sd_dlogm_200c_500c|calibrated"] = rng_of([(e["bin"], e["sd_dlogm"]) for e in m5 if e["bin"] in CAL_LABELS])
ranges["corr_dlogm_logc|calibrated"] = rng_of([(e["bin"], e["corr_with_log_c200c"]) for e in m5
                                               if e["bin"] in CAL_LABELS])
ranges["corr_dlogm_logc|13.5-15.1"] = rng_of([(e["bin"], e["corr_with_log_c200c"]) for e in m5
                                              if e["bin"] in CAL_LABELS[:8]])
for bl in ("2-3", "3-4"):
    for N in ("10", "25", "50"):
        ranges[f"subsample_median_bias_P|{bl}|N{N}"] = rng_of(
            [(lab, d[N]["P"][bl]["median_bias"]) for lab, d in rz["precision"].items() if d.get(N, {}).get("P", {}).get(bl)])
mt = rs["mass_trend"]
ranges["sd_logm_within_0.2dex|calibrated"] = rng_of([(e["bin"], e["sd_logm_within"]) for e in mt["P"]
                                                     if e["bin"] in CAL_LABELS])
for f in ("P", "E"):
    ranges[f"trend_var_share|{f}|calibrated_13.5-15.1"] = rng_of(
        [(f"{e['bin']}|{bl}", v) for e in mt[f] if e["bin"] in CAL_LABELS[:8] and e.get("trend_var_share")
         for bl, v in e["trend_var_share"].items() if v is not None])
osc = rz["observable_scatter"]
amp_sqrtN = {N: {o: sig(osc[N][f][o]["sd_pct"] * np.sqrt(int(N))) for f, o in (("P", "tSZ_yy_total_ell1000"),
                                                                              ("P", "tSZ_yy_total_ell100"),
                                                                              ("E", "DM_mean"), ("E", "kSZ_1h_ell3000"))}
             for N in ("10", "25", "50", "100")}
bc = rs["bootstrap_consistency"]
boot_sum = {}
for design in ("core_design_over_stack_bin", "census_design_over_stack_bin"):
    for grp, labs in (("sampled_12.5-14.5", [b["label"] for b in sc.mass_bins("M200c_0.2")][:10]),
                      ("census_14.5-15.5", [b["label"] for b in sc.mass_bins("M200c_0.2")][10:])):
        vals = [bc[f][lab][bl][design] for f in ("P", "E", "G", "T_e", "K") for lab in labs for bl in BAND_LABELS
                if bc[f][lab].get(bl)]
        v05 = [bc[f][lab]["0.5-1"][design] for f in ("P", "E", "G", "T_e", "K") for lab in labs
               if bc[f][lab].get("0.5-1")]
        boot_sum[f"{design}|{grp}"] = {"all_bands_min": sig(min(vals)), "all_bands_median": sig(np.median(vals)),
                                       "all_bands_max": sig(max(vals)), "band_0.5-1_min": sig(min(v05)),
                                       "band_0.5-1_max": sig(max(v05))}
    boot_sum[f"{design}|top_bins_per_field_0.5-1"] = {lab: {f: bc[f][lab]["0.5-1"][design] for f in ("P", "E")
                                                            if bc[f][lab].get("0.5-1")}
                                                      for lab in ("14.9-15.1", "15.1-15.3", "15.3-15.5")}

results = {
    "group": G,
    "reference": {"key": "sc.make_key(field, grid='fit') (T_e, K: estimator 'weighted')", "statistic": "mean",
                  "rule": "strict (min_count 20, softening_factor 4, hsml_factor 2, min_edge 0.04)",
                  "scheme": "M200c_0.2", "sample": "all haloes of the bin", "n_boot": N_BOOT},
    "conventions": {
        "bands_R200c": [list(b) for b in BANDS], "band_labels": list(BAND_LABELS),
        "band_statistic": "median over the band's resolved fit shells (volume-weighted radius in band) of the ratio",
        "paired_bootstrap": f"one global bootstrap ({N_BOOT} draws, seed {GLOBAL_SEED}) shared by every bin of every "
                            "scheme, so comparisons across different bin sets are paired halo by halo.  Design "
                            "'census': strata mass_bin x class_main in the randomly sampled bins 12.5-14.5, plain "
                            "within-bin resampling in the complete-census bins 14.5-15.5 (mass_bin indices "
                            f"{[int(c) for c in CENSUS_BINS]}).  The web records keep the core's per-bin stratified "
                            "bootstrap (sc.stack_bin defaults), whose 1-3-halo class strata understate the errors of "
                            "the top bins (see analyses.bootstrap_consistency).",
        "impacts": "per cent dO/O, sc.impact_from_stacks (first order, Test F importance Jacobians) on the 10 "
                   "calibrated 0.2-dex M200c bins 13.5-15.5 x 35 fit shells; err = paired-bootstrap SD; per_bin_pct = "
                   "contribution of each calibrated bin",
        "observables": {f: list(v) for f, v in OBS_KEEP.items()}},
    "variants": per_variant,
    "analyses": {
        "scheme_vs_reference_same_mass": rs["scheme_vs_reference_same_mass"],
        "scheme_pull_summary_13.5_15.1": pull_tab,
        "scheme_impacts_smooth_painting": rs["scheme_impacts_smooth_painting"],
        "scheme_impacts_continuous_painting": rs["scheme_impacts_continuous_painting"],
        "interpolant_test": ri,
        "wide_vs_fine_summary": wide_sum,
        "bootstrap_consistency": rs["bootstrap_consistency"],
        "bootstrap_consistency_summary": boot_sum,
        "rescaling_test": rr,
        "summary_ranges": ranges,
        "observable_scatter_times_sqrtN_pct": amp_sqrtN,
        "wide_vs_fine": rs["wide_vs_fine"],
        "mass_trend": rs["mass_trend"],
        "m500_vs_m200": rs["m500_vs_m200"],
        "painting_assignment_estimate": rs["painting_assignment_estimate"],
        "normalisation_own_vs_binmean": rn,
        "sample_size": {k: rz[k] for k in ("settings", "precision", "resolved_range", "scatter_precision",
                                            "covariance", "observable_scatter", "analytic_sigma1")},
        "N_needed_analytic": nneeded,
        "N_needed_subsampling": nneeded_sub,
        "top_bins": rt,
    },
    "notes": {**{f"schemes.{k}": v for k, v in rs["notes"].items()},
              **{f"sample_size.{k}": v for k, v in rz["notes"].items()},
              **{f"top_bins.{k}": v for k, v in rt["notes"].items()},
              **{f"normalisation.{k}": v for k, v in rn["notes"].items()},
              "N_needed_analytic": "sigma1 = per-halo fractional SD (reliability-weighted SD / mean, band median); "
                                   "N for precision p = (sigma1/p)^2; err_current = sigma1/sqrt(N_eff); err_complete = "
                                   "sigma1/sqrt(population of the bin) (sum of inclusion weights)",
              "N_needed_subsampling": "sigma1 from the subsampling experiment (median over N = 25, 50, 100, FPC-corrected); "
                                      "band = band-median statistic; single_shell = per-shell precision"},
}
dump(results, GROUP / "results.json")
print("results.json", (GROUP / "results.json").stat().st_size / 1e6, "MB")
