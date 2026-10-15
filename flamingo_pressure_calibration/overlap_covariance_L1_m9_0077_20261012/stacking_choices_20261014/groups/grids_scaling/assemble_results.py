"""Merge work/main_results.json and work/fits_results.json into results.json, adding the within-bin scatter of the
R500c->R200c conversion factors (and their correlation with the profile), and the truncation results quoted from
Tests C, F and G (not recomputed; source files named)."""
from __future__ import annotations

import json
import pickle

import numpy as np

import gs_common as C

sc = C.sc
WORK = C.HERE / "work"
M = json.loads((WORK / "main_results.json").read_text())
FR = json.loads((WORK / "fits_results.json").read_text())
TAB = sc.halo_table()
FIT_E = sc.edges("fit")
FB_ = float(sc.load_index()["baryon_fraction"])


def wcorr(a, b, w):
    ma, mb = np.average(a, weights=w), np.average(b, weights=w)
    cov = np.average((a - ma) * (b - mb), weights=w)
    return float(cov / np.sqrt(np.average((a - ma)**2, weights=w) * np.average((b - mb)**2, weights=w)))


# ---------------------------------------------------------------- conversion-factor scatter and shape correlation
lnq = np.log(TAB["r500c"] / TAB["r200c"])
lnf = np.log(TAB["p500_B24_pa"] / TAB["p200_pa"])
P = np.asarray(sc.per_halo(sc.make_key("P", grid="fit")))
E = np.asarray(sc.per_halo(sc.make_key("E", grid="fit")))
shells = {"0.20-0.23": (0.199, 0.2335), "0.44-0.52": (0.442, 0.5185), "1.34-1.5": (1.34, 1.5), "2.5-3": (2.5, 3.0)}
conv = {}
for b in C.calibrated(C.bins()):
    r, w = b["rows"], b["weights"]
    d = {"sd_ln_R500c_over_R200c": float(np.sqrt(np.average((lnq[r] - np.average(lnq[r], weights=w))**2, weights=w))),
         "sd_ln_Pe500_over_P200": float(np.sqrt(np.average((lnf[r] - np.average(lnf[r], weights=w))**2, weights=w))),
         "corr_lnq_lnf": wcorr(lnq[r], lnf[r], w), "corr_lnq_with_lnP_shell": {}, "corr_lnq_with_lnE_shell": {}}
    for name, (a, bb) in shells.items():
        k = int(np.argmin(np.abs(FIT_E - a)))
        ok = (P[r, k] > 0) & (E[r, k] > 0)
        d["corr_lnq_with_lnP_shell"][name] = wcorr(lnq[r][ok], np.log(P[r, k][ok]), w[ok])
        d["corr_lnq_with_lnE_shell"][name] = wcorr(lnq[r][ok], np.log(E[r, k][ok]), w[ok])
    conv[b["label"]] = d

# ---------------------------------------------------------------- quoted truncation results (not recomputed)
quoted = {
    "note": "Quoted from the study's reports (not recomputed here).",
    "Test_C_sigmoid (C_sigmoid/REPORT.md)": {
        "hard_vs_0.05dex_logistic_at_4R_pressure_Cl_change": "<= 0.21% for ell = 100-2e4 (<= 0.07% flux-matched); "
                                                              "box P(k) <= 0.35%",
        "wider_windows_or_cuts_at_1-2R": "move C_ell by 5-20% through flux redistribution, not ringing",
        "electrons_edge_sensitivity": "about 5x pressure: DM-field C_ell -1.1% at ell = 1000 (0.05-dex logistic at 4R)",
        "ringing": "single-halo transforms only, +-2% at k ~ 10/Mpc for a 2R cut, +-0.2% at 4R; none in C_ell",
        "recommendation": "hard spherical cut at 4R; a 0.05-dex flux-matched logistic if a smooth edge is needed"},
    "Test_F_observables (F_observables/REPORT.md)": {
        "family_RMS_errors_Rfit_x_Rcut_objective_a (tSZ/kSZ/DM/tSZxDM, %)": {
            "(2,2)": "6.5/27.2/22.1/13.9", "(2,4) current": "4.1/3.2/2.2/3.9", "(3,3)": "3.8/12.5/10.1/6.3",
            "(4,2)": "5.1/25.7/22.0/11.9", "(4,3)": "2.8/10.1/9.5/4.4", "(4,4)": "2.2/1.1/1.3/1.8"},
        "pareto_front": "only (R_fit, R_cut) = (4, 4); chosen by all 8 CV folds",
        "cut_costs_at_Rfit4": "3R cut: kSZ -10%, DM -9.5%; 2R cut: tSZ -5%, kSZ -26%, DM -22% (family RMS |dO/O| "
                              "of the objective-(a) cells (R_fit, R_cut) = (4, 3): 2.8/10.1/9.5/4.4 and (4, 2): "
                              "5.1/25.7/22.0/11.9 for tSZ/kSZ/DM/tSZxDM, exact engine, fitted models, RMS over the "
                              "observables of each family; F_observables/REPORT.md section 2)",
        "signal_shares_inside_1R/1-2R/2-4R": "tSZ 82/15/4% (ell 100), 96/4/0% (ell 3000); kSZ 1h 46/30/25% (ell 100); "
                                            "mean DM 32/31/37%; DM Q16 5/31/64%",
        "importance_still_rising_at_4R": "DM and low-ell kSZ still gain signal per unit ln r at 4R"},
    "Test_G_physical_outer (G_physical_outer/REPORT.md)": {
        "both_fields_prefer": "R_fit = R_cut = 4 R200c (objective a, S1 scatter)",
        "electrons": "4R cut needed: a 3R cut costs 8-10% in kSZ/DM, a 2R cut 20-24%",
        "pressure": "R_fit = 3 or 4 equivalent (0.44-0.52% tSZ with S1), but R_cut must stay at 4R",
        "physical_outer_terms": "splashback-like transition x_t = 2.1-2.4 R200c (electrons); linear 2-halo term "
                                "matches 3-4R electrons within +-10% at 10^13.9-14.9"},
}

# ---------------------------------------------------------------- derived summaries quoted in the report
def rng(v):
    v = [x for x in v if x is not None and np.isfinite(x)]
    return [float(min(v)), float(max(v))] if v else None


CALL = M["calibrated_bins"]
WELL = [l for l in CALL if float(l.split("-")[1]) <= 15.1 + 1e-9]
der = {}

# (a) B12/B16 grid sensitivity
gs = {}
for case, cd in FR["cases"].items():
    for g, comp in cd["vs_fit"].items():
        pb = comp["per_bin"]
        d = {}
        for X in ("<1", "<2", "<4"):
            vals = [pb[l]["dln_model_integral"][X] for l in pb]
            errs = [pb[l]["dln_model_integral_err"][X] for l in pb]
            d[f"dln_model_integral{X}_range"] = rng(vals)
            d[f"max_abs_dln_model_integral{X}"] = float(np.nanmax(np.abs(vals)))
            sig = [abs(v) / e if e and e > 0 else np.nan for v, e in zip(vals, errs)]
            d[f"significance{X}_range_sigma"] = rng(sig)
            d[f"per_bin{X}"] = {l: {"dln": v, "err": e, "sigma": s_} for l, v, e, s_ in zip(pb, vals, errs, sig)}
        d["max_abs_dln_param"] = comp["summary"]["max_abs_dln_param"]
        d["max_abs_dln_param_over_err"] = comp["summary"]["max_abs_dln_param_over_err"]
        d["impact_cut4R"] = {o: {"dO_over_O": v["dO_over_O"], "err": v["err"]} for o, v in comp["impact_cut4R"].items()}
        d["max_abs_impact"] = float(max(abs(v["dO_over_O"]) for v in comp["impact_cut4R"].values()))
        d["n_shells_range"] = rng([cd["grids"][g][l]["n_shells"] for l in cd["grids"][g]])
        d["n_shells_range_fit_grid"] = rng([cd["grids"]["fit"][l]["n_shells"] for l in cd["grids"]["fit"]])
        gs[f"{case}:{g}"] = d
der["grid_fit_sensitivity"] = gs
low = [v for k, v in gs.items() if "Rfit4:" not in k and "@" not in k]     # '@' = diagnostic sliver case
der["grid_fit_sensitivity_Rfit_le_2"] = {
    "max_abs_dln_model_integral<4": float(max(v["max_abs_dln_model_integral<4"] for v in low)),
    "max_abs_dln_model_integral<2": float(max(v["max_abs_dln_model_integral<2"] for v in low)),
    "max_abs_observable_impact": float(max(v["max_abs_impact"] for v in low)),
    "argmax_integral<4": max(((k, v["max_abs_dln_model_integral<4"]) for k, v in gs.items()
                              if "Rfit4:" not in k and "@" not in k), key=lambda t: t[1])[0],
    "note": "R_fit = 1.34 and 2 cases on x200, old and lee22; R_fit = 1.34 is exact on fit, x200 and lee22, 1.3508 on old "
            "(no 1.34 edge); the diagnostic 'x200@1.3508' case is excluded"}

# (a2) one boundary sliver: x200 fitted to 1.3508 (adds the 1.34-1.3508 shell) minus x200 fitted to exactly 1.34
FD = pickle.load(open(WORK / "figdata_fits.pkl", "rb"))
FDL = FD["cal"]
slv = {"sliver_R200c": [1.34, float(sc.edges("x200")[np.argmin(np.abs(sc.edges("x200") - 1.3508))])]}
slv["sliver_width_dex"] = float(np.log10(slv["sliver_R200c"][1] / 1.34))
for field in ("P", "E"):
    case = f"{field}_Rfit1.34"
    d = FD["figd"][(case, "x200@1.3508")]["dI"] - FD["figd"][(case, "x200")]["dI"]      # [bin, 1 + rep, X]
    per = {}
    for i, l in enumerate(FDL):
        val, err = float(d[i, 0, 2]), float(np.nanstd(d[i, 1:, 2], ddof=1))
        per[l] = {"dln": val, "err": err, "sigma": abs(val) / err if err > 0 else None,
                  "sliver_strict_resolved": FR["cases"][case]["grids"]["x200@1.3508"][l]["n_shells"]
                  > FR["cases"][case]["grids"]["x200"][l]["n_shells"]}
    k = max(per, key=lambda l: abs(per[l]["dln"]))
    imp_s = FR["cases"][case]["vs_fit"]["x200@1.3508"]["impact_cut4R"]
    imp_x = FR["cases"][case]["vs_fit"]["x200"]["impact_cut4R"]
    slv[field] = {"per_bin_dln_model_integral<4": per, "max_abs_dln<4": abs(per[k]["dln"]), "argmax_bin": k,
                  "impact_cut4R_with_sliver_minus_exact": {o: imp_s[o]["dO_over_O"] - imp_x[o]["dO_over_O"]
                                                           for o in imp_x}}
der["sliver_effect_x200_Rfit1.34"] = slv

# (a3) objective-(a) per-shell sigma = max(SD/mean, 5%) on the strict shells inside 4 R200c (SD = halo-to-halo)
sig = {}
for field in ("P", "E"):
    sig[field] = {}
    for b in C.calibrated(C.bins()):
        sig[field][b["label"]] = {}
        for g in ("fit", "x200"):
            st = sc.stack_bin(sc.make_key(field, grid=g), b, "mean", "strict", n_boot=C.N_BOOT)
            e = sc.edges(g)
            m = st["resolved"] & (e[1:] <= 4 + 1e-12) & (st["value"] > 0)
            sg = np.maximum(st["sd"] / st["value"], 0.05)[m]
            ii = np.flatnonzero(m)
            sig[field][b["label"]][g] = {"min": float(sg.min()), "max": float(sg.max()),
                                         "max_over_min": float(sg.max() / sg.min()),
                                         "innermost_shell_sigma": float(sg[0]), "outermost_shell_sigma": float(sg[-1]),
                                         "innermost_edge": float(e[ii[0]])}
    sig[field + "_range_fit_grid"] = {"min": float(min(v["fit"]["min"] for v in sig[field].values())),
                                      "max": float(max(v["fit"]["max"] for v in sig[field].values())),
                                      "max_over_min_range": rng([v["fit"]["max_over_min"] for v in sig[field].values()])}
der["objective_a_sigma_strict_shells_to_4R"] = sig


# (b) band-ratio ranges over the calibrated bins (min, max of the per-bin band medians)
br = {}
for vid, vr in M["variants"].items():
    br[vid] = {}
    for f, byb in vr["band_ratios"].items():
        br[vid][f] = {bl: rng([byb[l][bl]["ratio"] for l in CALL]) for bl in M["bands_R200c"]}
        br[vid][f]["all_bands"] = rng([byb[l][bl]["ratio"] for l in CALL for bl in M["bands_R200c"]])
der["band_ratio_ranges_calibrated_bins"] = br

# (c) R500c/R200c conversion factors: scatter and correlations, 13.5-15.1 and all 10 calibrated bins
fit_first = {l: M["resolution_masks"]["fit"][l]["first_resolved_edge"] for l in CALL}
cf = {}
for name, labs in (("13.5-15.1", WELL), ("13.5-15.5 (all 10 calibrated)", CALL)):
    cf[name] = {"sd_ln_R500c_over_R200c": rng([conv[l]["sd_ln_R500c_over_R200c"] for l in labs]),
                "sd_ln_Pe500_over_P200": rng([conv[l]["sd_ln_Pe500_over_P200"] for l in labs]),
                "corr_lnq_lnf": rng([conv[l]["corr_lnq_lnf"] for l in labs])}
    for sh in shells:
        cf[name][f"corr_lnq_lnP_{sh}"] = rng([conv[l]["corr_lnq_with_lnP_shell"][sh] for l in labs])
        cf[name][f"corr_lnq_lnE_{sh}"] = rng([conv[l]["corr_lnq_with_lnE_shell"][sh] for l in labs])
cf["first_strict_resolved_fit_edge_per_bin"] = fit_first
cf["bins_where_0.20-0.23_shell_is_unresolved"] = [l for l in CALL if fit_first[l] > 0.199 + 1e-6]
der["conversion_factors"] = cf

# (d) P_e,500/P200 against the exact proportionality to (R500c/R200c)^2
q = TAB["r500c"] / TAB["r200c"]
fP = TAB["p500_B24_pa"] / TAB["p200_pa"]
const = 6.25 * (0.159 / FB_) * (0.59 / 1.14)
dev = fP / (const * q**2) - 1
der["Pe500_over_P200_vs_q2"] = {
    "constant_(500/200)^2_(f_B/f_b)_(mu/mu_e)": const, "max_abs_dev": float(np.max(np.abs(dev))),
    "median_dev": float(np.median(dev)),
    "within_bin_weighted_corr_lnf_lnq_range_calibrated": rng([conv[l]["corr_lnq_lnf"] for l in CALL])}

# (e) self-similarity comparisons (common bins)
ss = M["self_similarity"]["P"]
xr = ss["R200c, P200 (ref)"]["x_R200c_equivalent"]
a_ = ss["R200c, P200 (ref)"]["common_bins_spread_sd_lnF"]
b_ = ss["R500c, P_e,500 (B24)"]["common_bins_spread_sd_lnF"]
c_ = ss["R500c, P500 (M500/3e14)^0.12 (A10 full)"]["common_bins_spread_sd_lnF"]
nz = ss["R200c, P200 (ref)"]["median_bootstrap_fractional_error_of_stack"]
der["self_similarity_P"] = {
    "x_R200c_equivalent": xr,
    "spread_ref_minus_R500cB24": [None if (u is None or v is None) else u - v for u, v in zip(a_, b_)],
    "max_abs_spread_difference_ref_vs_R500cB24": float(np.nanmax([abs(u - v) for u, v in zip(a_, b_)
                                                                  if u is not None and v is not None])),
    "spread_ratio_A10M_over_R500cB24": [None if (u is None or v is None) else u / v for u, v in zip(c_, b_)],
    "spread_over_stack_noise_ref": [None if (u is None or v is None) else u / v for u, v in zip(a_, nz)],
    "common_bins_n": ss["R200c, P200 (ref)"]["common_bins_n"]}

# (f) cumulative growth at 4R and the last-shell share (calibrated bins)
cg = {}
for f in ("P", "E", "G"):
    cu = M["cumulative_fractions_fit_grid"][f]
    cg[f] = {"dlnI_dlnx_3.634-4_per_bin": {l: cu[l]["dlnI_dlnx_3.634-4"]["value"] for l in CALL},
             "dlnI_dlnx_3.634-4_range": rng([cu[l]["dlnI_dlnx_3.634-4"]["value"] for l in CALL]),
             "dlnI_dlnx_3-4_range": rng([cu[l]["dlnI_dlnx_3-4"]["value"] for l in CALL]),
             "share_in_last_shell_range": rng([cu[l]["share_in_last_shell_3.634-4"]["value"] for l in CALL]),
             "profile_log_slope_3.15_to_3.8_range": rng([cu[l]["profile_log_slope_3.15_to_3.8"] for l in CALL])}
der["growth_at_4R"] = cg

# (g) truncation at 2R: partial (calibrated) vs all engine bins, quadratic estimates
tr = {}
for f in ("P", "E"):
    for o, v in M["truncation_first_order"][f]["2"].items():
        tr[o] = {k: v[k] for k in v if k not in ("per_bin",)}
der["truncation_2R"] = tr

# (h) extraction reach: fractions with 6 R500c > 4 R200c
der["fraction_6R500c_gt_4R200c"] = {
    "per_bin_inclusion_weighted": {l: M["extraction_reach"][l]["fraction_6R500c_gt_4R200c"] for l in CALL},
    "range_13.5-15.1": rng([M["extraction_reach"][l]["fraction_6R500c_gt_4R200c"] for l in WELL]),
    "unweighted_sample": M["normalisation_checks"]["fraction of haloes with 6 R500c > 4 R200c (unweighted sample of 4588)"],
    "population_12.5-15.5": M["normalisation_checks"][
        "fraction with 6 R500c > 4 R200c, inclusion-weighted population 12.5-15.5"],
    "population_13.5-15.5": M["normalisation_checks"][
        "fraction with 6 R500c > 4 R200c, inclusion-weighted population 13.5-15.5"],
    "mean_6R500c_over_R200c_range_calibrated": rng([M["extraction_reach"][l]["mean_6R500c_over_R200c"] for l in CALL])}

# (i) remap-method sensitivity of the R500c variants
rm = {}
for kind, bym in M["remap_method_sensitivity"].items():
    if kind == "note":
        continue
    rm[kind] = {}
    for meth, d in bym.items():
        rm[kind][meth] = {f: {"impact_dO_over_O_err_missing_share": {o: [v["dO_over_O"], v["err"], v["missing_share"]]
                                                                     for o, v in d[f]["impact"].items()},
                              "band_ratio_range_calibrated": d[f].get("band_ratio_range_calibrated")} for f in d}
der["remap_method_sensitivity"] = rm
# (j) strict-resolved shells inside / beyond R200c on the fit and x200 grids (the implicit weight of a diagonal fit)
sc_counts = {}
for b in C.calibrated(C.bins()):
    row = {}
    for g in ("fit", "x200"):
        e = sc.edges(g)
        m = sc.resolution_mask(b["rows"], g, "strict")
        row[g] = {"inside_1R": int((m & (e[1:] <= 1 + 1e-9)).sum()), "beyond_1R": int((m & (e[:-1] >= 1 - 1e-9)).sum()),
                  "first_resolved_edge": float(e[np.flatnonzero(m)[0]])}
        row[g]["inner_to_outer"] = row[g]["inside_1R"] / row[g]["beyond_1R"]
    row["inner_to_outer_relative_change_x200_vs_fit"] = row["x200"]["inner_to_outer"] / row["fit"]["inner_to_outer"] - 1
    sc_counts[b["label"]] = row
der["strict_shell_counts_fit_vs_x200"] = sc_counts
der["inner_to_outer_relative_change_range"] = rng([v["inner_to_outer_relative_change_x200_vs_fit"]
                                                   for v in sc_counts.values()])

# (k) correlations at 0.20-0.23 R200c over the bins where that shell IS strict-resolved
unres = set(cf["bins_where_0.20-0.23_shell_is_unresolved"])
der["conversion_factors"]["corr_lnq_lnP_0.20-0.23_resolved_bins_only"] = {
    "13.5-15.1": {"bins": [l for l in WELL if l not in unres],
                  "range": rng([conv[l]["corr_lnq_with_lnP_shell"]["0.20-0.23"] for l in WELL if l not in unres])},
    "13.5-15.5 (all 10 calibrated)": {"bins": [l for l in CALL if l not in unres],
                                      "range": rng([conv[l]["corr_lnq_with_lnP_shell"]["0.20-0.23"]
                                                    for l in CALL if l not in unres])}}

# (l) coverage of the fit grid by 6 R500c: per halo (x500_halo, 6 x min q) and at the bin mean (x500_paint, 6 x mean q)
cov = {}
qh = TAB["r500c"] / TAB["r200c"]
for b in C.calibrated(C.bins()):
    r, w = b["rows"], b["weights"]
    out_ = {}
    for name, reach in (("x500_halo_6_min_q", 6 * float(np.min(qh[r]))),
                        ("x500_paint_6_mean_q", 6 * float(np.average(qh[r], weights=w)))):
        covered = FIT_E[1:] <= reach + 1e-9
        k = int(np.flatnonzero(~covered)[0])
        out_[name] = {"reach_R200c": reach, "first_uncovered_shell": [float(FIT_E[k]), float(FIT_E[k + 1])]}
    cov[b["label"]] = out_
der["fit_grid_coverage_by_6R500c"] = cov
der["fit_grid_coverage_by_6R500c_summary"] = {
    "x500_halo_first_uncovered_inner_edge_per_bin": {l: v["x500_halo_6_min_q"]["first_uncovered_shell"][0]
                                                     for l, v in cov.items()},
    "x500_paint_first_uncovered_inner_edge_per_bin": {l: v["x500_paint_6_mean_q"]["first_uncovered_shell"][0]
                                                      for l, v in cov.items()},
    "x500_paint_reach_range": rng([v["x500_paint_6_mean_q"]["reach_R200c"] for v in cov.values()])}
fe = sc.edges("fit")
xe = sc.edges("x200")
der["grid_widths_dex"] = {"fit_min_max": [float(np.diff(np.log10(fe[1:])).min()), float(np.diff(np.log10(fe[1:])).max())],
                          "x200_min_max": [float(np.diff(np.log10(xe[1:])).min()), float(np.diff(np.log10(xe[1:])).max())],
                          "fit_min_dlnx": float(np.diff(np.log(fe[1:])).min()),
                          "x200_min_dlnx": float(np.diff(np.log(xe[1:])).min())}
der["note"] = "Ranges are [min, max] over the calibrated bins 13.5-15.5 unless the key names another bin set."

results = dict(M)
results["grid_fits"] = FR
results["conversion_factor_scatter_calibrated_bins"] = conv
results["quoted_truncation_tests"] = quoted
results["derived_summaries"] = der
results["files"] = {"web_variants": "web_variants.json", "figures": sorted(p.name for p in (C.HERE / "figures").glob("*.png")),
                    "scripts": ["gs_common.py", "run_grids_scaling.py", "run_grid_fits.py", "assemble_results.py",
                                "make_figures.py", "report_tables.py"]}
(C.HERE / "results.json").write_text(json.dumps(C.jsonify(results), indent=1))
print("results.json", (C.HERE / "results.json").stat().st_size / 1e6, "MB")
for lab, d in conv.items():
    print(lab, {k: (round(v, 3) if isinstance(v, float) else {kk: round(vv, 2) for kk, vv in v.items()})
                for k, v in d.items()})
