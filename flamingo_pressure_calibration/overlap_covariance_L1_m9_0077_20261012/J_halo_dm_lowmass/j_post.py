"""Local post-processing of Test J (reads the fetched results/; writes post/j_post.json and post/J_*.csv).

Every number is an exact function of the fetched arrays (no new physics), except the L1_m8 cost, which is an ESTIMATE
from the COSMA inventory (research/cosma/cosma_inventory.json) and the audit's measured transfer rates.
Conventions: a DM statistic of an evaluation is the moment form (mean, variance, DM-field C_ell, y x DM C_ell) of its
scatter-inclusive truth or the quantile of its per-halo DM PDF (z_s = 1); a quantile at or below 0.125 pc cm^-3 (the
PDF grid step) lies inside the atom of sightlines that cross no halo boundary ('atom').
Usage: python -B j_post.py [--smoke]
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFS = ("D0", "D1", "D2", "D3")
DEF_LABEL = {"D0": "A2-ii to 4 R200c", "D1": "A2-ii, sphere r <= R200c", "D2": "A2-ii, sphere r <= 2 R200c",
             "D3": "host-bound, to 4 R200c"}
CUT = {"D0": 4.0, "D1": 1.0, "D2": 2.0, "D3": 4.0}
Q_NAMES = ("q16", "q50", "q84", "q975", "q999")
SEL_ELL = (100, 300, 1000, 3000)
ATOM = 0.125
T_CMB_UK = 2.7255e6
FIRST_ORDER = ROOT / "stacking_choices_20261014" / "groups" / "selection" / "results.json"
COSMA = ROOT / "stacking_choices_20261014" / "research" / "cosma" / "cosma_inventory.json"


def load(res):
    def js(n):
        p = res / n
        return json.loads(p.read_text()) if p.exists() else None

    def npz(n):
        p = res / n
        return dict(np.load(p, allow_pickle=False)) if p.exists() else None
    return {"prep": js("prepare.json"), "stacks": npz("stacks.npz"), "truth": npz("truth.npz"),
            "ts": js("truth_summary.json"), "sens": npz("sens.npz"), "ss": js("sens_summary.json"),
            "parity": js("parity.json"), "fbpar": js("fallback_parity.json"), "basis": js("basis.json")}


def stat(R, tag, name):
    ts = R["ts"]
    if name in Q_NAMES:
        s = ts["pdf"].get(f"{tag}|truth")
        return None if s is None else float(s["quantiles"][Q_NAMES.index(name)])
    if name in ("p_atom", "cdf_at_1pc"):
        s = ts["pdf"].get(f"{tag}|truth")
        return None if s is None else float(s["p_first_bin" if name == "p_atom" else "cdf_at_1pc"])
    ev = ts["evaluations"].get(f"{tag}|truth")
    return None if ev is None else ev.get(name)


STATS = ("dm_mean", "dm_var", "q16", "q50", "q84", "q975", "q999", "p_atom", "dmcltot_100", "dmcltot_300",
         "dmcltot_1000", "dmcltot_3000", "ydmtot_100", "ydmtot_1000", "ydmtot_3000")
KSZ = tuple(f"{k}_{e}" for k in ("ksz1h", "ksz2h", "ksztot") for e in SEL_ELL)
RSTATS = tuple(s for s in STATS if s != "p_atom")          # ratios are not formed for the atom probability


def ratio(a, b):
    if a is None or b is None or b == 0:
        return None
    return float(a / b)


def write_csv(path, rows):
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    res = HERE / ("results_smoke" if a.smoke else "results")
    out_dir = HERE / ("post_smoke" if a.smoke else "post")
    out_dir.mkdir(exist_ok=True)
    R = load(res)
    ts, ss, prep = R["ts"], R["ss"], R["prep"]
    tr = R["truth"]
    jbins = prep["j_bins"]
    labels = ["12.5-12.7", "12.7-12.9", "12.9-13.1", "13.1-13.3", "13.3-13.5", "13.5-13.7", "13.7-13.9", "13.9-14.1",
              "14.1-14.3", "14.3-14.5", "14.5-14.7", "14.7-14.9", "14.9-15.1", "15.1-15.3", "15.3-15.5", ">=15.5"]
    P = {"generated_from": str(res)}
    # ------------------------------------------------------------------ definitions (C14, T1), C10, C0
    rows = []
    tabs = {}
    for cfg in ("C14", "C10", "C0"):
        for d in DEFS:
            tag = f"{cfg}|{d}|T1" if cfg != "C0" else f"C0|cut{CUT[d]:g}|-"
            rec = {s: stat(R, tag, s) for s in STATS + KSZ}
            tabs[f"{cfg}|{d}"] = rec
            rows.append(dict(configuration=cfg, definition=d, label=DEF_LABEL[d], **rec))
    write_csv(out_dir / "J_definitions_table.csv", rows)
    P["definitions"] = tabs
    P["definitions_ratio_to_D0_C14"] = {d: {s: ratio(tabs[f"C14|{d}"][s], tabs["C14|D0"][s]) for s in RSTATS + KSZ}
                                        for d in DEFS}
    P["measured_over_fallback_C14_over_C10"] = {d: {s: ratio(tabs[f"C14|{d}"][s], tabs[f"C10|{d}"][s])
                                                    for s in RSTATS + KSZ} for d in DEFS}
    P["measured_over_published_C14_over_C0"] = {d: {s: ratio(tabs[f"C14|{d}"][s], tabs[f"C0|{d}"][s])
                                                    for s in RSTATS + KSZ} for d in DEFS}
    P["fallback_C10_ratio_to_D0"] = {d: {s: ratio(tabs[f"C10|{d}"][s], tabs["C10|D0"][s]) for s in RSTATS}
                                     for d in DEFS}
    # atoms: quantiles at or below the grid step sit in the no-crossing atom
    P["quantiles_in_zero_atom"] = {f"{cfg}|{d}": [q for q in Q_NAMES if (tabs[f"{cfg}|{d}"][q] or 0) <= ATOM]
                                   for cfg in ("C14", "C10", "C0") for d in DEFS}
    P["crossings"] = ts.get("crossings")
    # ------------------------------------------------------------------ per mass bin (C14) and measured vs fallback per bin
    pb = {}
    for d in DEFS:
        tag = f"C14|{d}|T1"
        m = tr[f"{tag}|perbin__dm_mean"]
        v = tr[f"{tag}|perbin__dm_var"]
        fb_m = tr[f"{tag}|fbperbin__dm_mean"]
        fb_v = tr[f"{tag}|fbperbin__dm_var"]
        tot_m = float(tr[f"{tag}|truth__dm_mean"])
        tot_v = float(tr[f"{tag}|truth__dm_var"])
        pb[d] = {"bins": [labels[b] for b in jbins], "dm_mean": m.tolist(), "dm_mean_share": (m / tot_m).tolist(),
                 "dm_var": v.tolist(), "dm_var_share": (v / tot_v).tolist(),
                 "fallback_dm_mean_share": float(fb_m.sum() / tot_m), "fallback_dm_var_share": float(fb_v.sum() / tot_v)}
    for d in DEFS:
        pb[d]["dm_mean_ratio_to_D0"] = (np.asarray(pb[d]["dm_mean"]) / np.asarray(pb["D0"]["dm_mean"])).tolist()
        pb[d]["dm_var_ratio_to_D0"] = (np.asarray(pb[d]["dm_var"]) / np.asarray(pb["D0"]["dm_var"])).tolist()
    # per-bin measured (C14) vs published fallback (C10) for the newly calibrated bins
    low = [b for b in jbins if b < 5]
    mvf = {}
    for d in DEFS:
        c14 = tr[f"C14|{d}|T1|perbin__dm_mean"]
        c10fb = tr[f"C10|{d}|T1|fbperbin__dm_mean"]
        c14v = tr[f"C14|{d}|T1|perbin__dm_var"]
        c10fbv = tr[f"C10|{d}|T1|fbperbin__dm_var"]
        mvf[d] = {labels[b]: {"dm_mean_measured": float(c14[jbins.index(b)]), "dm_mean_fallback": float(c10fb[b]),
                              "ratio_mean": float(c14[jbins.index(b)] / c10fb[b]) if c10fb[b] else None,
                              "dm_var_measured": float(c14v[jbins.index(b)]), "dm_var_fallback": float(c10fbv[b]),
                              "ratio_var": float(c14v[jbins.index(b)] / c10fbv[b]) if c10fbv[b] else None}
                  for b in low}
        mvf[d]["sum_low_bins"] = {
            "dm_mean_measured": float(sum(c14[jbins.index(b)] for b in low)),
            "dm_mean_fallback": float(sum(c10fb[b] for b in low)),
            "share_of_C10_total_mean_from_low_bins_fallback": float(sum(c10fb[b] for b in low)
                                                                    / float(tr[f"C10|{d}|T1|truth__dm_mean"])),
            "share_of_C14_total_mean_from_low_bins_measured": float(sum(c14[jbins.index(b)] for b in low)
                                                                    / float(tr[f"C14|{d}|T1|truth__dm_mean"]))}
    P["per_bin_C14"] = pb
    P["per_bin_measured_vs_fallback"] = mvf
    rows = []
    for d in DEFS:
        for i, b in enumerate(jbins):
            rows.append(dict(definition=d, mass_bin=labels[b], dm_mean=pb[d]["dm_mean"][i],
                             dm_mean_share=pb[d]["dm_mean_share"][i], dm_mean_ratio_to_D0=pb[d]["dm_mean_ratio_to_D0"][i],
                             dm_var=pb[d]["dm_var"][i], dm_var_share=pb[d]["dm_var_share"][i],
                             dm_var_ratio_to_D0=pb[d]["dm_var_ratio_to_D0"][i]))
    write_csv(out_dir / "J_per_bin.csv", rows)
    # ------------------------------------------------------------------ treatments and subset (C14)
    tsens = {}
    flags = []
    for d in DEFS:
        for t in ("T2", "T3", "S82"):
            tag = f"C14|{d}|{t}"
            if f"{tag}|truth" not in ts["evaluations"]:
                continue
            rr = {}
            for s in RSTATS:
                a_, b_ = stat(R, tag, s), stat(R, f"C14|{d}|T1", s)
                if s in Q_NAMES and (b_ is not None and b_ <= ATOM) and (a_ is not None and a_ <= ATOM):
                    rr[s] = 0.0
                    continue
                r_ = ratio(a_, b_)
                rr[s] = None if r_ is None else r_ - 1
                if r_ is not None and abs(r_ - 1) > 0.01 and s != "p_atom":
                    flags.append({"definition": d, "variant": t, "statistic": s, "change": r_ - 1})
            tsens[f"{d}|{t}"] = rr
    P["treatment_sensitivity_rel_change"] = tsens
    P["treatment_changes_above_1pct"] = flags
    # ------------------------------------------------------------------ kSZ exact (Q3) and first-order check
    ksz = {cfg: {k: ratio(stat(R, f"{cfg}|D3|T1", k), stat(R, f"{cfg}|D0|T1", k)) for k in KSZ} for cfg in ("C14", "C10")}
    P["kSZ_D3_over_D0"] = ksz
    P["kSZ_values_C14"] = {d: {k: stat(R, f"C14|{d}|T1", k) for k in KSZ} for d in ("D0", "D3")}
    P["kSZ_Dl_uK2_C14"] = {d: {k: (stat(R, f"C14|{d}|T1", k) * int(k.split("_")[1]) * (int(k.split("_")[1]) + 1)
                                   / (2 * np.pi) * T_CMB_UK**2) for k in KSZ} for d in ("D0", "D3")}
    fo = json.loads(FIRST_ORDER.read_text())["text_support"]["impact_significance"]["selection:host_bound"]
    exact = {"kSZ_1h_ell100": ratio(stat(R, "C10|D3|T1", "ksz1h_100"), stat(R, "C10|ALL|T1", "ksz1h_100")),
             "kSZ_1h_ell1000": ratio(stat(R, "C10|D3|T1", "ksz1h_1000"), stat(R, "C10|ALL|T1", "ksz1h_1000")),
             "kSZ_1h_ell3000": ratio(stat(R, "C10|D3|T1", "ksz1h_3000"), stat(R, "C10|ALL|T1", "ksz1h_3000")),
             "DM_mean": ratio(stat(R, "C10|D3|T1", "dm_mean"), stat(R, "C10|ALL|T1", "dm_mean")),
             "DM_var": ratio(stat(R, "C10|D3|T1", "dm_var"), stat(R, "C10|ALL|T1", "dm_var")),
             "DM_q16": ratio(stat(R, "C10|D3|T1", "q16"), stat(R, "C10|ALL|T1", "q16")),
             "DM_q50": ratio(stat(R, "C10|D3|T1", "q50"), stat(R, "C10|ALL|T1", "q50")),
             "DM_q84": ratio(stat(R, "C10|D3|T1", "q84"), stat(R, "C10|ALL|T1", "q84")),
             "DM_Cl_total_ell1000": ratio(stat(R, "C10|D3|T1", "dmcltot_1000"), stat(R, "C10|ALL|T1", "dmcltot_1000"))}
    P["first_order_vs_exact_C10_hostbound_over_allgas"] = {
        k: {"first_order_pct": fo[k]["value_pct"], "exact_pct": None if v is None else 100 * (v - 1)}
        for k, v in exact.items() if k in fo}
    # ------------------------------------------------------------------ sens digests
    if ss is not None:
        dig = {}
        for d in DEFS:
            dd = {}
            for name in ("dm_mean@scalar", "dm_var@scalar", "dmcltot@1000", "ydmtot_E@1000", "ksz1h@1000", "ksz1h@100",
                         "ksz1h@3000", "dm_q16@scalar", "dm_q50@scalar", "dm_q84@scalar", "dm_q975@scalar",
                         "dm_q999@scalar"):
                key = f"{d}__{name}"
                dd[name] = {"bands": ss["bands"].get(key), "peak": ss["peaks"].get(key),
                            "tolerance": ss["tolerance"].get(key),
                            "shares": {k: v for k, v in (ss["shares"].get(key) or {}).items()}}
            dd["radii"] = {k: v for k, v in ss["radii"][d].items()
                           if k.split("@")[0] in ("dm_mean", "dm_var", "dmcltot", "ksz1h", "ksz2h", "ksztot",
                                                  "ydmtot_Ecut", "dm_q16", "dm_q50", "dm_q84", "dm_q975", "dm_q999")}
            dd["fallback_share"] = ss.get("fallback_share", {}).get(d)
            dig[d] = dd
        P["sens"] = dig
        P["sens_checks"] = {"sum_rule_moments_max": max(v for k, v in ss["sum_rules"].items() if "dm_q" not in k),
                            "sum_rule_quantiles": {k: v for k, v in ss["sum_rules"].items() if "dm_q" in k}}
    # ------------------------------------------------------------------ J1a digest (low-mass stacks vs fallback)
    P["J1a"] = {lab: prep["J1a_by_bin"][lab] for lab in prep["J1a_by_bin"]}
    P["lightcone"] = prep["lightcone"]
    # ------------------------------------------------------------------ L1_m8 cost (ESTIMATE)
    P["L1_m8_cost_ESTIMATE"] = l1m8_cost(prep)
    # ------------------------------------------------------------------ derived quantities quoted by the report (j_report.py)
    P["derived_for_report"] = derived_for_report(R, P, flags)
    (out_dir / "j_post.json").write_text(json.dumps(P, indent=1, default=float) + "\n")
    print("wrote", out_dir / "j_post.json")


def derived_for_report(R, P, flags):
    """Small exact functions of the fetched arrays that the report quotes (no new physics)."""
    st, ts = R["stacks"], R["ts"]
    labels = ["12.5-12.7", "12.7-12.9", "12.9-13.1", "13.1-13.3", "13.3-13.5"]
    e = st["x_edges"]
    vol = 4 * np.pi / 3 * (e[1:]**3 - e[:-1]**3)
    k1 = int(np.flatnonzero(np.isclose(e, 1.0))[0])
    inner = {}
    for b in R["prep"]["low_bins"]:
        E = st["E_a2ii_T1"][b]
        inner[labels[b]] = {
            "x_outer": float(e[1]),
            "stacked_count_a2ii": float(st["stacked_count_a2ii"][b, 0]),
            "stacked_count_hb": float(st["stacked_count_hb"][b, 0]),
            "E_over_B16": float(st["ratio_E_a2ii_T1_over_B16"][b, 0]),
            "P_over_B12": float(st["ratio_P_a2ii_T1_over_B12"][b, 0]),
            "share_of_Ne_lt_R200c": float(vol[0] * E[0] / np.sum(vol[:k1] * E[:k1])),
            "softening_limit_4eps_over_R": float(st["softening_limit"][b])}
    # largest relative change of any DM statistic under T2 / T3 / S82 (C14, every definition)
    worst = {"abs_change": 0.0}
    by_variant = {t: {"abs_change": 0.0} for t in ("T2", "T3", "S82")}
    for key, rr in P["treatment_sensitivity_rel_change"].items():
        t = key.split("|")[1]
        for s, v in rr.items():
            if v is not None and abs(v) > worst["abs_change"]:
                worst = {"abs_change": abs(v), "change": v, "case": key, "statistic": s}
            if v is not None and abs(v) > by_variant[t]["abs_change"]:
                by_variant[t] = {"abs_change": abs(v), "change": v, "case": key, "statistic": s}
    # D1 Mecke quantile maps (atom at DM = 0): sums of the maps for Q50 ... Q99.9 (Q16 lies in the atom); J1's
    # original half-atom formula (truth_summary.json) and, after the 2026-10-16 revision, the whole-atom maps
    msr = ts["mecke_sum_rule"]
    d1 = [1 + v for k in ("C14|D1|T1|truth_subsample", "C14|D1|T1|truthmean") for v in msr[k][1:]]
    mu_p = HERE / "post" / "mecke_update.json"
    d1w = None
    if mu_p.exists():
        mw = json.loads(mu_p.read_text())["mecke_sum_rule_minus_1"]["D1"]
        d1w = [1 + v for k in ("whole|C14|D1|T1|truth_subsample", "whole|C14|D1|T1|truthmean") for v in mw[k][1:]]
    return {"innermost_shell_low_bins": inner,
            "treatment_largest_change": worst, "treatment_largest_change_by_variant": by_variant,
            "treatment_changes_above_1pct_count": len(flags),
            "shell_volume_ratio_4to5R_over_0to4R": (5.0**3 - 4.0**3) / 4.0**3,
            "D1_mecke_map_sum_q50_to_q999_half_atom_J1": {"min": float(min(d1)), "max": float(max(d1))},
            "D1_mecke_map_sum_q50_to_q999_whole_atom": None if d1w is None else {"min": float(min(d1w)),
                                                                                "max": float(max(d1w))},
            "p_atom_definition": "probability in the first cell of the DM-PDF grid, P(DM < 0.0625 pc cm^-3) "
                                 "(grid step 0.125, mid-cell convention of f_pdf / engine.pdf_summary)",
            "f_b_engine_over_f_b_study": float(R["parity"]["chord"]["f_b_engine_(lightcone_analytic.FB)"]
                                               / R["parity"]["chord"]["f_b_study"])}


def l1m8_cost(prep):
    """Download cost of FLAMINGO L1_m8 (8x better gas mass resolution) for the low-mass bins: ESTIMATE ONLY.
    Inputs: COSMA inventory (full snapshot z = 0: gas particle count and SWIFT bytes; reduced z = 0 header gas count),
    the audit's bytes per particle (68 B full-snapshot fields incl. int64 membership; 64 B reduced) and its measured /
    quoted transfer rates (0.31 MB/s local link, 8.67 MB/s one timed COSMA read, ~12 MB/s idark).  The 4 R200c cells
    of a halo sample are assumed to cover the same fraction of the box as for L1_m9 in the audit's case (d) (1,625
    haloes: 18.7%; 4,375 haloes: 31%) - an assumption: L1_m8's top-level cell layout was not checked."""
    inv = json.loads(COSMA.read_text())
    full = inv["full_snapshots"]["counts"]["L1_m8/L1_m8"]
    i0 = full["z"].index(0.0)
    ngas = float(full["ngas"][i0])
    swift_tb = float(full["swift_bytes"][i0]) / 1e12
    red = inv["reduced_snapshots"]["gas_counts_from_headers"]
    red_m8 = None
    for k, v in red.items():
        if "L1_m8" in k:
            vv = v["L1_m8"] if isinstance(v, dict) and "L1_m8" in v else v
            if isinstance(vv, dict) and "z" in vv:
                j = vv["z"].index(0.0) if 0.0 in vv["z"] else len(vv["z"]) - 1
                for key in ("ngas", "gas", "n_gas", "PartType0"):
                    if key in vv:
                        red_m8 = float(vv[key][j])
                        break
    rates = {"local_0.31_MBps": 0.31e6, "cosma_timed_8.67_MBps": 8.67e6, "idark_12_MBps": 12e6}
    cases = {}
    for name, frac, haloes in (("low-mass bins 12.7-13.5, 300 haloes/bin (1,200 haloes; cell fraction as the audit's "
                                "1,625-halo case)", 0.187, 1200),
                               ("all painted bins, 300/bin + all >10^14.5 (cell fraction as the audit's 4,375-halo case)",
                                0.31, 4375)):
        gb = frac * ngas * 68 / 1e9
        cases[name] = {"gas_rows": frac * ngas, "GB": gb,
                       "hours": {k: gb * 1e9 / r / 3600 for k, r in rates.items()}}
    out = {"L1_m8_z0_gas_particles": ngas, "L1_m8_z0_snapshot_TB": swift_tb, "bytes_per_particle_full": 68,
           "gas_particle_mass_ratio_m9_over_m8": 8.0, "cases_full_snapshot": cases,
           "reduced_z0": {"gas_rows_header": red_m8 if red_m8 else 1.265e9,
                          "source": "inventory" if red_m8 else "cosma_service_audit.md table (1,265M at z = 0)",
                          "GB_at_64B": (red_m8 if red_m8 else 1.265e9) * 64 / 1e9,
                          "limits": "only haloes >= 10^13 Msun (0.05-dex bins, 200 per bin) and only r < R100c ~ "
                                    "1.35 R200c (ESTIMATE of R100c/R200c): this covers the WHOLE D1 extent "
                                    "(r <= R200c) and 1-1.35 R200c of D0/D2/D3 for haloes of 10^13-10^13.5 (bins "
                                    "13.1-13.3, 13.3-13.5 and the upper half of 12.9-13.1), so it can test whether "
                                    "N_e(<R200c) and the split of the gas across R200c are resolution-converged there; "
                                    "it does not cover 12.7-13.0 and not D0/D2/D3 beyond ~1.35 R200c"},
           "resolution_gain_ESTIMATE": "8x more gas particles per shell at fixed mass (median per fine shell inside R200c "
                                       "x8); SPH smoothing length h ~ (m/rho)^(1/3) halves at fixed density, so the "
                                       "hydrodynamic-resolution radius x_h (median 2h/R200c) roughly halves",
           "x_h_L1_m9": {lab: prep["J1a_by_bin"][lab]["x_h"] for lab in prep["J1a_by_bin"]},
           "on_maas": False, "downsampled_L1_m8": "not served by COSMA"}
    return out


if __name__ == "__main__":
    main()
