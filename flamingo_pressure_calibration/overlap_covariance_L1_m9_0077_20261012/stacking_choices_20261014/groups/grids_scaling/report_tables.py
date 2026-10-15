"""Print the report tables of the 'grids_scaling' group from results.json (stdout only; writes nothing).

Run:  <halfdome env>/bin/python -B report_tables.py > <scratch>/tables.txt
Every number in the group's report text is read from results.json by this script or by the key named next to it.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = json.loads((HERE / "results.json").read_text())
D = R["derived_summaries"]
CAL = R["calibrated_bins"]
G = "grids_scaling"


def pct(x, n=2, sign=True):
    return "-" if x is None else (f"{100 * x:+.{n}f}" if sign else f"{100 * x:.{n}f}")


def rr(r, n=3):
    return "-" if r is None else f"{r[0]:.{n}f}-{r[1]:.{n}f}"


def h(t):
    print(f"\n==== {t}")


# ---------------------------------------------------------------- 1. rebinning
h("rebinning")
for sect in ("per_halo", "stack", "ratio_fields_after_stacking"):
    for k, v in R["rebinning"][sect].items():
        print(f"{sect} | {k} | {v:.2g}" if isinstance(v, float) else f"{sect} | {k} | {v}")
for f, v in R["rebinning"]["median_after_stacking"].items():
    print("median", f, {k: round(100 * x, 2) for k, x in v["per_bin"].items()})
h("remap accuracy (max / median |dev| over resolved shells of the calibrated bins)")
for k, v in R["rebinning"]["remap_accuracy"].items():
    print(f"| {k} | {100 * v['max_abs_dev']:.1f}% | {100 * v['median_abs_dev']:.2f}% | {v['worst_bin']} "
          f"{v['worst_shell_R200c'][0]:.3g}-{v['worst_shell_R200c'][1]:.3g} |")
h("grid widths")
print(D["grid_widths_dex"])

# ---------------------------------------------------------------- 2. resolution masks
h("first strict-resolved edge (R200c) and holes")
for g in ("fit", "old", "x200", "x500", "b24"):
    rm = R["resolution_masks"][g]
    print(g, {l: (round(rm[l]["first_resolved_edge_R200c"], 3), len(rm[l]["holes"])) for l in CAL})
h("strict shell counts inside/beyond 1R (fit | x200)")
for l, v in D["strict_shell_counts_fit_vs_x200"].items():
    print(l, v["fit"]["inside_1R"], v["fit"]["beyond_1R"], f"{v['fit']['inner_to_outer']:.2f}", "|",
          v["x200"]["inside_1R"], v["x200"]["beyond_1R"], f"{v['x200']['inner_to_outer']:.2f}")

# ---------------------------------------------------------------- 3. grid fits
h("grid-fit sensitivity (max |dln param| (sigma); dln I(<4) range (sigma range); dln I(<2) range)")
for k, v in D["grid_fit_sensitivity"].items():
    mp = max(v["max_abs_dln_param"].items(), key=lambda t: t[1])
    ms = max(v["max_abs_dln_param_over_err"].values())
    print(f"| {k} | {100 * mp[1]:.1f}% ({mp[0]}), <= {ms:.1f} sigma | I<4 {pct(v['dln_model_integral<4_range'][0])}.."
          f"{pct(v['dln_model_integral<4_range'][1])}% ({v['significance<4_range_sigma'][0]:.1f}-"
          f"{v['significance<4_range_sigma'][1]:.1f} sigma) | I<2 {pct(v['dln_model_integral<2_range'][0])}.."
          f"{pct(v['dln_model_integral<2_range'][1])}% | shells {v['n_shells_range']} vs fit {v['n_shells_range_fit_grid']} |")
    print("     impacts:", {o: f"{pct(x['dO_over_O'])}+-{pct(x['err'], sign=False)}" for o, x in v["impact_cut4R"].items()})
print("R_fit <= 2:", D["grid_fit_sensitivity_Rfit_le_2"])
print("parity", R["grid_fits"]["parity_TestF"])
for c in ("E_Rfit4:x200", "P_Rfit4:x200"):
    print(c, {l: (round(100 * x["dln"], 2), round(x["sigma"], 1)) for l, x in D["grid_fit_sensitivity"][c]["per_bin<4"].items()})
    print(c, "<2:", {l: round(100 * x["dln"], 2) for l, x in D["grid_fit_sensitivity"][c]["per_bin<2"].items()})
for k, v in D["grid_fit_sensitivity"].items():
    print(k, "per-parameter max |dln| %:", {n: round(100 * x, 2) for n, x in v["max_abs_dln_param"].items()},
          "max sigma:", {n: round(x, 1) for n, x in v["max_abs_dln_param_over_err"].items()})
h("one boundary sliver (x200 fitted to 1.3508 minus x200 fitted to exactly 1.34), d ln model integral <4R")
sl = D["sliver_effect_x200_Rfit1.34"]
print("sliver", sl["sliver_R200c"], f"{sl['sliver_width_dex']:.4f} dex")
for f in ("P", "E"):
    print(f, {l: (pct(x["dln"]), None if x["sigma"] is None else round(x["sigma"], 1), x["sliver_strict_resolved"])
              for l, x in sl[f]["per_bin_dln_model_integral<4"].items()})
    print(f, "max", pct(sl[f]["max_abs_dln<4"]), "at", sl[f]["argmax_bin"], "| impacts with - without sliver:",
          {o: pct(x) for o, x in sl[f]["impact_cut4R_with_sliver_minus_exact"].items()})
h("objective (a) sigma = max(SD/mean, 5%) on strict shells to 4R (fit grid): min / max / max/min, innermost edge")
sg = D["objective_a_sigma_strict_shells_to_4R"]
for f in ("P", "E"):
    print(f, sg[f + "_range_fit_grid"])
    for l, v in sg[f].items():
        print("  ", l, f"{v['fit']['min']:.3f} / {v['fit']['max']:.3f} / {v['fit']['max_over_min']:.1f}",
              f"inner {v['fit']['innermost_shell_sigma']:.3f} outer {v['fit']['outermost_shell_sigma']:.3f}")
print("inner-to-outer strict-shell ratio, relative change x200 vs fit:",
      {l: round(100 * v["inner_to_outer_relative_change_x200_vs_fit"], 1) for l, v in D["strict_shell_counts_fit_vs_x200"].items()},
      "range", [round(100 * x, 1) for x in D["inner_to_outer_relative_change_range"]])

# ---------------------------------------------------------------- 4. normalisation
h("normalisation checks")
for k, v in R["normalisation_checks"].items():
    print(k, "|", v)
print(D["Pe500_over_P200_vs_q2"])
h("fit-grid coverage by 6 R500c (x500_halo: 6 x min q per bin; x500_paint: 6 x bin-mean q)")
for l, v in D["fit_grid_coverage_by_6R500c"].items():
    print(l, {k: (round(x["reach_R200c"], 3), [round(y, 3) for y in x["first_uncovered_shell"]]) for k, x in v.items()})
print(D["fit_grid_coverage_by_6R500c_summary"])
print("corr(ln q, ln P) at 0.20-0.23 R200c, resolved bins only:",
      D["conversion_factors"]["corr_lnq_lnP_0.20-0.23_resolved_bins_only"])

# ---------------------------------------------------------------- 5. self-similarity
h("self-similarity (common bins)")
for f in ("P", "E", "T_e", "K"):
    for name, s in R["self_similarity"][f].items():
        x = s["x_R200c_equivalent"]
        print(f"{f} | {name} | spread " + " ".join("-" if v is None else f"{v:.3f}" for v in s["common_bins_spread_sd_lnF"])
              + " | within " + " ".join("-" if v is None else f"{v:.3f}" for v in s["common_bins_within_bin_sd_ln_median"])
              + " | n " + " ".join(str(v) for v in s["common_bins_n"]))
        print(f"     slope(13.5-15.1) " + " ".join("-" if v is None else f"{v:+.2f}" for v in s["dlnF_dlog10M200c_13.5-15.1"])
              + " | noise " + " ".join("-" if v is None else f"{v:.3f}" for v in s["median_bootstrap_fractional_error_of_stack"]))
print("x =", R["self_similarity"]["P"]["R200c, P200 (ref)"]["x_R200c_equivalent"])
print(D["self_similarity_P"])
h("counts context (median particles per shell)")
for g, v in R["counts_context_median_per_shell"].items():
    print(g, v["13.5-13.7"])
h("conversion factors")
print(json.dumps(D["conversion_factors"], indent=0))

# ---------------------------------------------------------------- 6. R500c variants
h("band-ratio ranges over calibrated bins")
for vid in (f"{G}:x500_paint", f"{G}:x500_P200", f"{G}:fit_P500", f"{G}:x500_A10M", f"{G}:b24_B24", f"{G}:x500_halo",
            f"{G}:lee22", f"{G}:old", f"{G}:x200"):
    for f, bb in D["band_ratio_ranges_calibrated_bins"][vid].items():
        print(f"| {vid} {f} | " + " | ".join(rr(bb[b]) for b in R["bands_R200c"]) + f" | all {rr(bb['all_bands'], 4)}")
h("impacts of R500c / normalisation variants (%, paired SD, missing share)")
for vid in (f"{G}:x500_paint", f"{G}:x500_B24", f"{G}:x500_A10", f"{G}:x500_A10M", f"{G}:x500_ss500", f"{G}:x500_P200",
            f"{G}:fit_P500", f"{G}:b24_B24", f"{G}:x500_halo", f"{G}:lee22", f"{G}:old", f"{G}:x200"):
    for f, imp in R["variants"][vid]["impact"].items():
        print(vid, f, {o: f"{pct(x['dO_over_O'])}+-{pct(x['err'], sign=False)} (miss {x['missing_share']:.3f})"
                       for o, x in imp.items()})
h("remap-method sensitivity")
for kind, bym in D["remap_method_sensitivity"].items():
    for m, v in bym.items():
        for f, x in v.items():
            print(kind, m, f, {o: f"{pct(a[0])}+-{pct(a[1], sign=False)} (miss {a[2]:.3f})"
                               for o, a in x["impact_dO_over_O_err_missing_share"].items()})

# ---------------------------------------------------------------- 7. radial range
h("cumulative fractions (% beyond x of the value inside 4R)")
for f in ("P", "E"):
    for l in ("13.5-13.7", "14.1-14.3", "14.5-14.7", "14.9-15.1", "15.3-15.5"):
        c = R["cumulative_fractions_fit_grid"][f][l]
        print(f, l, " / ".join(f"{100 * c[f'beyond_{x}']['value']:.1f}+-{100 * c[f'beyond_{x}']['err']:.1f}"
                               for x in ("1", "1.34", "2", "3")),
              "| per-halo beyond 1 q16/50/84", [round(v, 2) for v in c["per_halo_beyond_1_q16_q50_q84"]])
print(json.dumps(D["growth_at_4R"], indent=0))
print("R500c units", {f: {l: (round(v["I(<R500c)/I(<5R500c)"]["value"], 3), round(v["I(<R500c)/I(<5R500c)"]["err"], 3))
                          for l, v in R["cumulative_R500c_units_M500c_bins"][f].items()} for f in ("P", "E")},
      "A10", R["cumulative_R500c_units_M500c_bins"]["A10_universal_shape_I(1)/I(5)"])
print(D["fraction_6R500c_gt_4R200c"])
h("truncation (first order, calibrated bins; all engine bins; quadratic estimates)")
for f in ("P", "E"):
    for cut, v in R["truncation_first_order"][f].items():
        print(f, cut, {o: (pct(x["dO_over_O"], 1), pct(x["dO_over_O_first_order_cut_all_engine_bins"], 1),
                           pct(x.get("quadratic_estimate_single_amplitude_calibrated_bins"), 1),
                           pct(x.get("quadratic_estimate_independent_bins_calibrated_bins"), 1))
                       for o, x in v.items()})
for o, v in D["truncation_2R"].items():
    print(o, "calibrated_share", round(v["calibrated_share"], 3))

# ---------------------------------------------------------------- 8. clumping
h("clumping (band medians, resolved shells)")
for l in ("13.5-13.7", "14.5-14.7", "14.9-15.1"):
    c = R["clumping"][l]
    for nm, bands in c["band_medians_resolved"].items():
        print(l, nm, {b: (None if v is None else round(v, 3)) for b, v in bands.items()})
    print(l, "identity", c["identity_G_QM/G_geo_eq_C_rho/f_V_per_halo_max_abs_dev"])
