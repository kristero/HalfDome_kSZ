"""L stage 'numbers': every number quoted in the L1 report section, with its source file and field.

Writes tables/L1_report_bins.csv (the per-bin table of the report) and results/report_numbers.json
({name: {value, source, field}}); verify_l.py re-resolves each entry from its source (gate G8: a check of this
registry against the result files, not of the report prose).
Also writes tables/L1_report_tables.json (added 2026-10-16, review round 2): the report's numeric tables (L1.0 global
radius and D0 shares beyond x_t, L1.1 per-bin radii, L1.2 extrapolated shares, global-vs-per-bin terms, L1.4 shell
shares and FLAMINGO/HalfDome ratios) as markdown rows formatted from the JSON (3 decimals, f'{v:.3f}'); the report
pastes these rows verbatim, and verify_l.py gate G12 re-formats them independently and diffs them against the report
text (env L_REPORT_TEXT), so prose tables and result files cannot diverge.
"""
from __future__ import annotations

import csv

import l_common as LC
import l_global as LG
import numpy as np


def get(d, path):
    for k in path:
        d = d[k]
    return d


def main():
    files = {"radii": LC.RESULTS / "radii.json", "global": LC.RESULTS / "global.json",
             "codex": LC.RESULTS / "codex.json", "stacks": LC.RESULTS / "stacks_summary.json",
             "inputs": LC.INPUT_META, "extrap": LC.RESULTS / "extrap.json"}
    J = {k: LC.jload(v) for k, v in files.items() if v.exists()}
    nums = {}

    def put(name, src, path, scale=1.0):
        try:
            v = get(J[src], path)
        except (KeyError, IndexError, TypeError):
            return                                  # not available in this mode (smoke: two bins only)
        nums[name] = {"value": None if v is None else float(v) * scale, "source": str(files[src].relative_to(LC.HERE)),
                      "field": "/".join(map(str, path)), "scale": scale}

    rad = J["radii"]
    rows = []
    for lab, d in rad["bins"].items():
        r = d["rules"]
        sb = r["SB"]["E"]
        row = {"bin": lab, "mean_logm200c": d["mean_logm200c"], "N": J["stacks"]["bins"][lab]["N"],
               "R200c_over_R200m": d["R200c_over_R200m"], "R200c_over_R500c": d["R200c_over_R500c"]}
        for rule in ("B32", "B32cb", "B50", "C68", "Req", "2H2"):
            for f in ("P", "E"):
                s = r[rule][f]
                row[f"{rule}_{f}"] = s["value"]
                row[f"{rule}_{f}_q16"] = s["q16"]
                row[f"{rule}_{f}_q84"] = s["q84"]
        for f in ("P", "E"):
            row[f"B32_{f}_R200m"] = r["B32"][f]["R200m"]["value"]
            row[f"B32_{f}_R500c"] = r["B32"][f]["R500c"]["value"]
            row[f"B32_{f}_inside_a2ii"] = r["B32"][f]["inside_a2ii"]["value"]
            row[f"B32_{f}_inside_hb"] = r["B32"][f]["inside_hb"]["value"]
            row[f"B32_{f}_non_monotonic"] = r["B32"][f].get("non_monotonic")
            row[f"B32u_{f}"] = r["B32u"][f]["value"]
            row[f"B32lo_{f}"] = r["B32lo"][f]["value"]
            row[f"B32hi_{f}"] = r["B32hi"][f]["value"]
        row["SB_E"] = sb["value"]
        row["SB_E_shallow"] = sb.get("shallow")
        row["SB_E_R200m"] = sb["R200m"]["value"]
        row["SB_P"] = r["SB"]["P"]["value"]
        row["SB_P_shallow"] = r["SB"]["P"].get("shallow")
        row["MDK15"] = r["MDK15"]["E"]["value"]
        row["MDK15_R200m"] = r["MDK15"]["E"]["R200m"]["value"]
        row["SH_K0"] = r["SH_K0"]["E"]["value"]
        row["SH_TM"] = r["SH_TM"]["E"]["value"]
        row["SH_TM_status"] = r["SH_TM"]["E"]["status"]
        d3 = J["global"]["d3_convergence"]["own_bin"].get("dm_mean@scalar", {}).get(lab, {})
        row["D3_dm_mean_R90"] = d3.get("R90")
        row["D3_dm_mean_R95"] = d3.get("R95")
        row["C90_E"] = r["C90"]["E"]["value"]
        row["C90_P"] = r["C90"]["P"]["value"]
        row["C95_P"] = r["C95"]["P"]["value"]
        rows.append(row)
    with open(LC.TABLES / "L1_report_bins.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        for r_ in rows:
            w.writerow({k: (f"{v:.4g}" if isinstance(v, float) else v) for k, v in r_.items()})
    # ---- scalar numbers of the text
    G = J["global"]["global"]
    for rule in ("B32", "B32cb", "B32u", "B50", "C68", "C90", "Req", "SB", "2H2", "MDK15"):
        for f in ("P", "E"):
            for unit in ("R", "R200m", "R500c"):
                for k in ("value", "q16", "q84", "cv"):
                    if k in G[rule][f][unit]["family"]:
                        put(f"global_{rule}_{f}_{unit}_{k}", "global", ["global", rule, f, unit, "family", k])
            if "max_over_min_12.9-15.3" in G[rule][f]["R"]["family"]:
                put(f"global_{rule}_{f}_maxmin", "global", ["global", rule, f, "R", "family", "max_over_min_12.9-15.3"])
            for unit in ("R200m", "R500c"):
                put(f"global_{rule}_{f}_{unit}_maxmin", "global",
                    ["global", rule, f, unit, "family", "max_over_min_12.9-15.3"])
            for o in LG.FAMILY[f]:                   # every family observable (single-observable weights)
                put(f"global_{rule}_{f}_R_{o}", "global", ["global", rule, f, "R", o, "value"])
        for unit in ("R", "R200m", "R500c"):
            put(f"global_{rule}_common_{unit}", "global", ["global", rule, "common", unit])
    for tag, res in J["global"]["additive_terms"].items():
        for rule in ("B32", "B50", "C68", "Req", "SB", "2H2", "B32cb"):
            if rule in res:
                put(f"T_{tag}_{rule}", "global", ["additive_terms", tag, rule, "T_perbin"])
                put(f"Tglob_{tag}_{rule}", "global", ["additive_terms", tag, rule, "T_global"])
                if "perbin_over_D3" in res[rule]:
                    put(f"overD3_{tag}_{rule}", "global", ["additive_terms", tag, rule, "perbin_over_D3"])
    for k, v in J["global"].get("weight_mass_shares", {}).items():
        for kk in v:
            put(f"wshare_{k}_{kk}", "global", ["weight_mass_shares", k, kk])
    for lab, d in J["global"].get("band_bound_shares", {}).get("bins", {}).items():
        for kk, vals in d.items():
            for i in range(len(vals)):
                put(f"band_{lab}_{kk}_{i}", "global", ["band_bound_shares", "bins", lab, kk, i])
    for f in ("P", "E"):
        put(f"a2ii_vs_all_1.5R_{f}", "global", ["a2ii_vs_all_inside_1.5R", "max_1_minus_a2ii_over_all", f])
    for o, v in J["global"]["d3_convergence"]["global"].items():
        for k in ("R50", "R90", "R95"):
            put(f"D3global_{o}_{k}", "global", ["d3_convergence", "global", o, k])
    for rule in ("B32",):
        for f in ("P", "E"):
            put(f"tw_{rule}_{f}", "global", ["truncated_target_weights", rule, f, "R_weighted"])
    # added 2026-10-16 after review: observable-optimal single cut, sensitivity of the global radius, per-shell
    # fraction ranges, B32 bootstrap half-widths, per-bin B32 radii
    for rule, d in J["global"].get("observable_optimal", {}).items():
        if not isinstance(d, dict) or rule in ("grid",):
            continue
        for fam, r in d.items():
            for k in ("x_opt", "rms_at_opt", "max_abs_dev_at_opt", "x_signal_weighted", "rms_at_signal_weighted"):
                put(f"opt_{rule}_{fam}_{k}", "global", ["observable_optimal", rule, fam, k])
            for k in ("0", "1"):
                put(f"opt_{rule}_{fam}_flat{k}", "global", ["observable_optimal", rule, fam, "rms_within_0.001_of_min", int(k)])
            for t in r.get("dev_at_opt", {}):
                put(f"opt_{rule}_{fam}_devopt_{t}", "global", ["observable_optimal", rule, fam, "dev_at_opt", t])
                put(f"opt_{rule}_{fam}_devsw_{t}", "global", ["observable_optimal", rule, fam, "dev_at_signal_weighted", t])
    for rule, d in J["global"].get("global_sensitivity", {}).items():
        if not isinstance(d, dict):
            continue
        for f in ("P", "E"):
            if f not in d:
                continue
            for lab, r in d[f]["leave_one_bin_out"].items():
                for k in ("value", "q16", "q84", "weight_of_left_out_bin"):
                    put(f"loo_{rule}_{f}_{lab}_{k}", "global", ["global_sensitivity", rule, f, "leave_one_bin_out", lab, k])
            for k in ("single_observable_min", "single_observable_max"):
                put(f"sens_{rule}_{f}_{k}", "global", ["global_sensitivity", rule, f, k])
        for k in ("min", "max"):
            put(f"sens_{rule}_both_single_{k}", "global", ["global_sensitivity", rule, "both_fields_single_observable", k])
    for lab, d in J["global"].get("per_shell_fraction_ranges", {}).get("bins", {}).items():
        for kk, v in d.items():
            if v is not None:
                for i in (0, 1):
                    put(f"psr_{lab}_{kk}_{i}", "global", ["per_shell_fraction_ranges", "bins", lab, kk, i])
    hw = J["global"].get("B32_bootstrap_halfwidths", {})
    for lab, d in hw.get("bins", {}).items():
        for f in d:
            put(f"hw_B32_{lab}_{f}", "global", ["B32_bootstrap_halfwidths", "bins", lab, f])
    for k in ("below_15.3_min", "below_15.3_max", "below_15.3_median"):
        put(f"hw_B32_{k}", "global", ["B32_bootstrap_halfwidths", k])
    for lab, d in rad["bins"].items():
        for rule in ("B32", "B32cb", "B50", "C68", "Req", "C90", "SB", "2H2", "2H1", "MDK15", "CLg", "SH_K0", "SH_TM", "SH_K"):
            for f in ("P", "E"):
                for k in ("value", "q16", "q84"):
                    put(f"rad_{lab}_{rule}_{f}_{k}", "radii", ["bins", lab, "rules", rule, f, k])
            for f in ("P", "E"):
                for k in ("depth",):
                    put(f"rad_{lab}_{rule}_{f}_{k}", "radii", ["bins", lab, "rules", rule, f, k])
        for k in ("R200c_over_R200m", "R200c_over_R500c"):
            put(f"rad_{lab}_{k}", "radii", ["bins", lab, k])
    cx = J.get("codex")
    if cx:
        for fld in ("tsz", "ksz", "dm"):
            for b in ("77", "325", "1025", "3025"):
                for k, idx in (("1R", 4), ("1.5R", 5), ("2R", 6), ("3R", 7)):
                    put(f"HD_{fld}_{b}_T{k}", "codex", ["halfdome_maps", fld, b, "T_at_edges", idx])
                put(f"HD_{fld}_{b}_TB32", "codex", ["halfdome_maps", fld, b, "T_at_rule_radius_ESTIMATE", "B32"])
                for i, sh in enumerate(("0-1", "1-1.5", "1.5-2", "2-3", "3-4")):
                    put(f"HD_{fld}_{b}_share_{sh}", "codex", ["halfdome_maps", fld, b, "common_shell_fractions", i])
            for e in ("100", "1000", "3000"):
                for k in ("1", "1.5", "2", "3"):
                    put(f"ENG_{fld}_{e}_T{k}", "codex", ["engine", fld, e, "T_at", k])
                put(f"ENG_{fld}_{e}_TB32", "codex", ["engine", fld, e, "T_at_rule_radius", "B32"])
                for i, sh in enumerate(("0-1", "1-1.5", "1.5-2", "2-3", "3-4")):
                    put(f"ENG_{fld}_{e}_share_{sh}", "codex", ["engine", fld, e, "band_importance_shares", i])
            for sim in ("L1_m9", "L2p8_m9"):
                for k in ("ratio_4R", "ratio_B32_ESTIMATE", "ratio_Req_ESTIMATE", "ratio_cut_1R"):
                    put(f"FL_{fld}_{sim}_{k}", "codex", ["flamingo_over_halfdome_1025", fld, sim, k])
            for g in ("mass_fractions", "redshift_fractions"):
                if g in cx["halfdome_maps"][fld]:
                    for b in ("77", "1025", "3025"):
                        for i, v in enumerate(cx["halfdome_maps"][fld][g][b]):
                            put(f"HD_{fld}_{g}_{b}_{i}", "codex", ["halfdome_maps", fld, g, b, i])
        for sim in ("L1_m9", "L2p8_m9"):
            p = cx["pilot"]["codex"][sim]
            put(f"pilot_{sim}_logm", "codex", ["pilot", "codex", sim, "log10_m200c"])
            for k in ("B32", "B50", "B32_central"):
                put(f"pilot_{sim}_{k}_R", "codex", ["pilot", "codex", sim, k, "R"])
                put(f"pilot_{sim}_{k}_Rfirst", "codex", ["pilot", "codex", sim, k, "R_first"])
            for R in ("1", "2", "3", "4"):
                put(f"pilot_{sim}_cum{R}", "codex", ["pilot", "codex", sim, "cumulative_hb_over_all", R])
            for k in ("f_hb_max", "x_of_max", "f_central_max"):
                put(f"pilot_{sim}_bump_{k}", "codex", ["pilot", "codex", sim, "bump_beyond_first_B32", k])
            for i in (0, 1):
                put(f"pilot_{sim}_bump_xrange{i}", "codex", ["pilot", "codex", sim, "bump_beyond_first_B32",
                                                             "x_range_f_hb_ge_0.32", i])
            put(f"pilot_{sim}_B32_ncross", "codex", ["pilot", "codex", sim, "B32", "n_cross"])
        for lab in cx["pilot"]["ours_pressure"]:
            for R in ("1", "2", "3", "4"):
                put(f"ours_{lab}_cum{R}_hb", "codex", ["pilot", "ours_pressure", lab, "stack_cumulative", R, "hb"])
                put(f"ours_{lab}_cum{R}_q16", "codex", ["pilot", "ours_pressure", lab, "per_halo_cumulative_hb", R, "q16"])
                put(f"ours_{lab}_cum{R}_q84", "codex", ["pilot", "ours_pressure", lab, "per_halo_cumulative_hb", R, "q84"])
                put(f"ours_{lab}_cum{R}_min", "codex", ["pilot", "ours_pressure", lab, "per_halo_cumulative_hb", R, "min"])
                put(f"ours_{lab}_cum{R}_max", "codex", ["pilot", "ours_pressure", lab, "per_halo_cumulative_hb", R, "max"])
    # per-bin unit conversions, inside fractions and halo counts (radii.json, stacks_summary.json)
    for lab, d in rad["bins"].items():
        put(f"N_{lab}", "stacks", ["bins", lab, "N"])
        put(f"logm_{lab}", "radii", ["bins", lab, "mean_logm200c"])
        for rule in ("B32", "B50", "C68", "Req", "B32cb"):
            for f in ("P", "E"):
                for k in ("R200m", "R500c", "inside_a2ii", "inside_hb"):
                    put(f"rad_{lab}_{rule}_{f}_{k}", "radii", ["bins", lab, "rules", rule, f, k, "value"])
            for f in ("P", "E"):
                put(f"rad_{lab}_{rule}_{f}_ncross", "radii", ["bins", lab, "rules", rule, f, "n_cross"])
                put(f"rad_{lab}_{rule}_{f}_nonmono_boot", "radii", ["bins", lab, "rules", rule, f,
                                                                    "non_monotonic_fraction_boot"])
        for rule in ("B32u", "B32lo", "B32hi", "C90", "C95", "SB_G", "2H1", "Req_all"):
            for f in ("P", "E"):
                put(f"rad_{lab}_{rule}_{f}_value", "radii", ["bins", lab, "rules", rule, f, "value"])
    # added 2026-10-16 after the user's clarification: the extrapolated region (extrap.json)
    ex = J.get("extrap")
    if ex:
        put("x_t", "extrap", ["x_t"])
        for f, d in ex.get("family_weighted", {}).items():
            for which, r in d.items():
                for k in ("s_a2ii", "s_hb", "s_nest_hi", "s_pub"):
                    for q in ("value", "q16", "q84"):
                        put(f"ex_fw_{f}_{which}_{k}_{q}", "extrap", ["family_weighted", f, which, k, q])
                for k in ("n_bins_pub_in_bracket", "n_bins_pub_in_bracket_all", "weight_in_bracket"):
                    put(f"ex_fw_{f}_{which}_{k}", "extrap", ["family_weighted", f, which, k])
                for k in ("t_pub_range", "t_pub_range_ge13.5", "t_pub_range_lt13.5", "s_a2ii_range", "s_hb_range",
                          "s_all_range", "s_pub_range"):
                    for i in (0, 1):
                        put(f"ex_fw_{f}_{which}_{k}_{i}", "extrap", ["family_weighted", f, which, k, i])
        for lab, d in ex["bins"].items():
            for f in ("P", "E"):
                for which in ("global", "perbin"):
                    for k in ("s_a2ii", "s_hb", "s_all", "s_nest_hi", "s_pub", "t_pub", "t_pub_all", "n_pub",
                              "pub_out_over_a2ii_out"):
                        for q in ("value", "q16", "q84"):
                            put(f"ex_{lab}_{f}_{which}_{k}_{q}", "extrap", ["bins", lab, f, which, k, q])
                    for k in ("pub_in_bracket", "pub_in_bracket_all", "pub_in_bracket_fraction_boot"):
                        put(f"ex_{lab}_{f}_{which}_{k}", "extrap", ["bins", lab, f, which, k])
        ol = ex.get("observable_level")
        if ol:
            for t in ol["D0_share_beyond_xt"]:
                put(f"ex_D0beyond_{t}", "extrap", ["observable_level", "D0_share_beyond_xt", t])
            for which in ("global", "perbin"):
                for k in ("lower_over_D0", "inside_over_D0", "share_beyond_upper", "share_beyond_lower", "D3_over_D0",
                          "lower_over_D3"):
                    put(f"ex_dm_{which}_{k}", "extrap", ["observable_level", "mean_dm_bracket", which, k])
                for lab in ol["mean_dm_bracket"].get(which, {}).get("per_bin", {}):
                    for k in ("share_beyond_upper", "share_beyond_lower", "inside_D3_over_inside_D0"):
                        put(f"ex_dm_{which}_{lab}_{k}", "extrap", ["observable_level", "mean_dm_bracket", which,
                                                                   "per_bin", lab, k])
        for fld, d in ex.get("maps_share_beyond_xt", {}).get("values", {}).items():
            for band, r in d.items():
                for k in ("halfdome_B12B16_share_beyond_xt_ESTIMATE", "engine_D0_share_beyond_xt"):
                    put(f"ex_map_{fld}_{band}_{k}", "extrap", ["maps_share_beyond_xt", "values", fld, band, k])
    cxj = J.get("codex")
    if cxj:
        for fld in ("tsz", "ksz", "dm"):
            for sim in ("L1_m9", "L2p8_m9"):
                put(f"FL_{fld}_{sim}_ratio_cut_1.5R", "codex", ["flamingo_over_halfdome_1025", fld, sim, "ratio_cut_1.5R"])
            for b_ in ("77", "1025"):
                for i in range(9):
                    put(f"HD_{fld}_{b_}_Tedge{i}", "codex", ["halfdome_maps", fld, b_, "T_at_edges", i])
                for i in range(8):
                    put(f"HD_{fld}_{b_}_add{i}", "codex", ["halfdome_maps", fld, b_, "additive_shell_fractions", i])
        for sim, d in cxj.get("redshift_shares_maps", {}).items():
            for fld, v in d.items():
                for b_ in ("77", "1025", "3025"):
                    for i in range(len(v["bands"][b_])):
                        put(f"zmap_{sim}_{fld}_{b_}_{i}", "codex", ["redshift_shares_maps", sim, fld, "bands", b_, i])
    sm = J["stacks"]["pressure_means_ESTIMATE"]
    nums["P_mean_pa"] = {"value": sm["P_mean_pa"], "source": "results/stacks_summary.json",
                         "field": "pressure_means_ESTIMATE/P_mean_pa", "scale": 1.0}
    nums["b_P_effective"] = {"value": sm["b_P_effective"], "source": "results/stacks_summary.json",
                             "field": "pressure_means_ESTIMATE/b_P_effective", "scale": 1.0}
    # added 2026-10-16 (review round 2): the yy1h fallback bracket (ESTIMATE) and the common-x_t inside shares
    if ex and "observable_level" in ex:
        for e_, d in ex["observable_level"].get("yy1h_fallback_ESTIMATE", {}).items():
            if isinstance(d, dict):
                for k in d:
                    put(f"ex_yy1hfb_{e_}_{k}", "extrap", ["observable_level", "yy1h_fallback_ESTIMATE", e_, k])
    LC.jdump(nums, LC.RESULTS / "report_numbers.json")
    if not LC.SMOKE:
        LC.jdump(report_tables(J), LC.TABLES / "L1_report_tables.json")
    LC.log(f"numbers: {len(nums)} scalars, {len(rows)} table rows")


L12_BINS = ("12.7-12.9", "13.1-13.3", "13.5-13.7", "13.9-14.1", "14.5-14.7", "14.9-15.1", "15.1-15.3")
TERM_LABEL = {"ymean": "mean y", "yy1h@1000": "tSZ 1-halo, ell = 1000", "ksz1h@100": "kSZ 1-halo, ell = 100",
              "dm_mean": "mean DM"}


def report_tables(J):
    """The report's numeric tables as markdown rows (header, separator, rows), every cell f'{v:.3f}' of a JSON field."""
    f3 = lambda v: f"{v:.3f}"                                                       # noqa: E731
    rng = lambda d: f"{d['value']:.3f} [{d['q16']:.3f}, {d['q84']:.3f}]"            # noqa: E731
    T = {}
    G = J["global"]["global"]["B32"]
    t = ["| | R200c | R200m | R500c |", "|---|---|---|---|"]
    for f in ("P", "E"):
        t.append(f"| {f} | {rng(G[f]['R']['family'])} | {f3(G[f]['R200m']['family']['value'])} | "
                 f"{f3(G[f]['R500c']['family']['value'])} |")
    t.append(f"| common | {f3(G['common']['R'])} | {f3(G['common']['R200m'])} | {f3(G['common']['R500c'])} |")
    T["L1.0_global_radius"] = t
    D = J["extrap"]["observable_level"]["D0_share_beyond_xt"]
    T["L1.0_D0_share_beyond_xt"] = [
        "| term | share from beyond x_t |", "|---|---|",
        f"| mean y | {f3(D['ymean'])} |",
        f"| tSZ 1-halo, ell = 100 / 1000 / 3000 | {' / '.join(f3(D[f'yy1h@{e}']) for e in (100, 1000, 3000))} |",
        f"| mean DM | {f3(D['dm_mean'])} |",
        f"| DM variance | {f3(D['dm_var'])} |",
        f"| kSZ 1-halo, ell = 100 / 1000 / 3000 | {' / '.join(f3(D[f'ksz1h@{e}']) for e in (100, 1000, 3000))} |",
        f"| DM-field C_ell 1-halo, ell = 1000 | {f3(D['dmcl1h@1000'])} |"]
    t = ["| bin | N | B32 P | B32 E | B32cb P / E | B50 P / E | C68 P / E | inside B32, P / E |",
         "|---|---|---|---|---|---|---|---|"]
    for lab, d in J["radii"]["bins"].items():
        r = d["rules"]
        t.append(f"| {lab} | {J['stacks']['bins'][lab]['N']} | {rng(r['B32']['P'])} | {rng(r['B32']['E'])} | "
                 f"{f3(r['B32cb']['P']['value'])} / {f3(r['B32cb']['E']['value'])} | "
                 f"{f3(r['B50']['P']['value'])} / {f3(r['B50']['E']['value'])} | "
                 f"{f3(r['C68']['P']['value'])} / {f3(r['C68']['E']['value'])} | "
                 f"{f3(r['B32']['P']['inside_a2ii']['value'])} / {f3(r['B32']['E']['inside_a2ii']['value'])} |")
    T["L1.1_per_bin_radii"] = t
    ex = J["extrap"]
    t = ["| bin | P: A2-ii | P: all gas | P: bound | P: B12 shape | t_B12 | E: A2-ii | E: all gas | E: bound | "
         "E: B16 shape | t_B16 |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for lab in L12_BINS:
        c = []
        for f in ("P", "E"):
            g = ex["bins"][lab][f]["global"]
            c += [f3(g[k]["value"]) for k in ("s_a2ii", "s_all", "s_hb", "s_pub", "t_pub")]
        t.append(f"| {lab} | " + " | ".join(c) + " |")
    c = []
    for f in ("P", "E"):
        g = ex["family_weighted"][f]["global"]
        c += [f3(g["s_a2ii"]["value"]), "–", f3(g["s_hb"]["value"]), f3(g["s_pub"]["value"]), "–"]
    t.append("| family-weighted | " + " | ".join(c) + " |")
    T["L1.2_extrapolated_shares"] = t
    A = J["global"]["additive_terms"]
    rP, rE = A["ymean"]["B32"]["R_global"], A["dm_mean"]["B32"]["R_global"]
    t = [f"| term | field radius (P {rP:.4f} / E {rE:.4f}) | common x_t = {ex['x_t']:.4f} | per-bin B32 radii |",
         "|---|---|---|---|"]
    for tag, lab_ in TERM_LABEL.items():
        t.append(f"| {lab_} | {f3(A[tag]['B32']['T_global'])} | {f3(1 - D[tag])} | {f3(A[tag]['B32']['T_perbin'])} |")
    T["L1.2_global_vs_perbin"] = t
    cx = J["codex"]
    t = ["| | HalfDome B12/B16 (ell ≈ 77) | engine, FLAMINGO A2-ii (ell = 100) |", "|---|---|---|"]
    for fld, nm in (("tsz", "tSZ"), ("ksz", "kSZ"), ("dm", "DM")):
        t.append(f"| {nm} | {' / '.join(f3(v) for v in cx['halfdome_maps'][fld]['77']['common_shell_fractions'])} | "
                 f"{' / '.join(f3(v) for v in cx['engine'][fld]['100']['band_importance_shares'])} |")
    T["L1.4_shell_shares"] = t
    t = ["| | L1_m9 | L2p8 |", "|---|---|---|"]
    for fld, nm in (("tsz", "tSZ"), ("ksz", "kSZ"), ("dm", "DM")):
        r = cx["flamingo_over_halfdome_1025"][fld]
        t.append(f"| {nm} | {f3(r['L1_m9']['ratio_4R'])} | {f3(r['L2p8_m9']['ratio_4R'])} |")
    T["L1.4_flamingo_over_halfdome"] = t
    return T


if __name__ == "__main__":
    main()
