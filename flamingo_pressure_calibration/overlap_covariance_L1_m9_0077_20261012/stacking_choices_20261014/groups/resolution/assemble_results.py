"""Merge work/results_main.json and work/redshift.json into results.json (group 'resolution')."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import res_common as R  # noqa: E402
import numpy as np  # noqa: E402

sc = R.sc
W = R.GROUP / "work"
M = json.loads((W / "results_main.json").read_text())
Z = json.loads((W / "redshift.json").read_text())
LABELS = [b["label"] for b in sc.mass_bins(R.SCHEME)]


def summary(sid):
    m = M["masks"][sid]
    mr = M["masks"]["ref"]
    imp = M["impacts"][sid]

    def shift(l):
        a, b = m[l].get("first_shell_index"), mr[l].get("first_shell_index")
        return None if a is None or b is None else a - b

    out = {"first_edge": {l: m[l]["first_edge"] for l in LABELS},
           "first_shell_index": {l: m[l].get("first_shell_index") for l in LABELS},
           "shell_shift_vs_ref": {l: shift(l) for l in LABELS},
           "n_kept_to_4R": {l: m[l]["n_kept"] for l in LABELS},
           "n_kept_to_2R": {l: m[l]["n_kept_to_2R"] for l in LABELS},
           "binding": {l: m[l]["binding"] for l in LABELS},
           "innermost_count_median": {l: (m[l].get("innermost_counts") or {}).get("median") for l in LABELS},
           "innermost_count_min": {l: (m[l].get("innermost_counts") or {}).get("min") for l in LABELS}}
    for tag in imp:
        if tag == "missing_fit_bins":
            continue
        out[f"impact_{tag}"] = {o: {"dO_over_O": v["dO_over_O"], "err": v.get("err"),
                                    "missing_share": v.get("missing_share")}
                                for f in ("P", "E") for o, v in imp[tag][f].items()}
    out["missing_fit_bins"] = imp.get("missing_fit_bins")
    return out


def report_numbers():
    """Ranges quoted in the report text, computed here so that every number is traceable."""
    cal = list(R.CAL_LABELS)
    ps = {s["id"]: summary(s["id"]) for s in R.SETTINGS}
    ref = ps["ref"]
    tab = sc.halo_table()
    out = {}
    lab_h = [l for l in LABELS if l in ref["binding"] and ref["binding"][l] and "smoothing_length" in ref["binding"][l]]
    out["bins_where_h_test_binds"] = lab_h
    out["innermost_count_min_range_h_binding_bins"] = [min(ref["innermost_count_min"][l] for l in lab_h),
                                                        max(ref["innermost_count_min"][l] for l in lab_h)]
    out["innermost_count_median_range_h_binding_bins"] = [min(ref["innermost_count_median"][l] for l in lab_h),
                                                           max(ref["innermost_count_median"][l] for l in lab_h)]
    out["innermost_count_min_by_bin"] = ref["innermost_count_min"]
    out["bins_with_innermost_count_min_ge_32"] = [l for l in LABELS if (ref["innermost_count_min"][l] or 0) >= 32]
    for short in ("mc0", "mc5", "mc10", "mc32", "mc50", "mc100", "mc200", "mc500", "hf0", "hf1", "hf3", "me0", "me0.1",
                  "sf0", "sf2", "sf3", "sf6", "frac95", "frac84", "per_halo_f50"):
        sid = R.SETTING_BY_SHORT[short]["id"]
        sh = ps[sid]["shell_shift_vs_ref"]
        fe = ps[sid]["first_edge"]
        diff = [l for l in LABELS if fe[l] != ref["first_edge"][l]]
        out[f"shell_shift_{short}"] = {
            "calibrated_by_bin": [sh[l] for l in cal],
            "calibrated_range": [min(v for v in (sh[l] for l in cal) if v is not None),
                                 max(v for v in (sh[l] for l in cal) if v is not None)]
            if any(sh[l] is not None for l in cal) else None,
            "bins_with_different_mask": diff}
    for short in ("mconly0", "mconly5", "mconly10", "mconly20"):
        sid = R.SETTING_BY_SHORT[short]["id"]
        nk = ps[sid]["n_kept_to_4R"]
        out[f"n_kept_{short}"] = {"calibrated_range": [min(nk[l] for l in cal), max(nk[l] for l in cal)],
                                  "all_bins_range": [min(nk.values()), max(nk.values())]}
    e4 = 4 * tab["eps_over_r200c"]
    calrows = np.concatenate([sc.find_bin(R.SCHEME, l)["rows"] for l in cal])
    out["four_eps_over_R200c_range_all"] = [float(e4.min()), float(e4.max())]
    out["four_eps_over_R200c_range_calibrated"] = [float(e4[calrows].min()), float(e4[calrows].max())]
    # largest paired 1-sigma error of the 4R-fit impacts (vs reference fit) outside the per-halo settings
    errs = []
    for sid, d in ps.items():
        if "per_halo" in sid or sid == "ref":
            continue
        for o, v in d["impact_fit4R_vs_ref_fit"].items():
            if v["err"] is not None:
                errs.append((v["err"], sid, o))
    errs.sort(reverse=True)
    out["largest_impact_errors_non_per_halo_fit4R_vs_ref_fit"] = [
        {"err": e, "setting": sid, "observable": o} for e, sid, o in errs[:8]]
    return out


def phase_check():
    """Share of the cold / star-forming phase (SFR > 0 or T < 1e5 K) in the stacked n_e and P per shell (inner
    shells explain why unmasked cores rise in n_e but not in P)."""
    out = {}
    for b in sc.mass_bins(R.SCHEME):
        d = {}
        for f in ("E", "P"):
            allv = sc.stack_bin(sc.make_key(f, phase="all"), b, n_boot=0)["value"]
            cold = sc.stack_bin(sc.make_key(f, phase="cold_sf"), b, n_boot=0)["value"]
            with np.errstate(divide="ignore", invalid="ignore"):
                d[f"{f}_cold_sf_share"] = np.where(allv > 0, cold / allv, np.nan)
        out[b["label"]] = d
    return out


def grid_gaps():
    """Shells beyond the first strict-kept edge that the strict rule rejects (gaps), per grid and bin, with the number
    of haloes failing each criterion there and the shell width in dex (narrow sliver shells fail the count test)."""
    out = {}
    for grid in ("fit", "old", "lee22", "x200", "x500", "b24"):
        e = sc.edges(grid)
        res = {}
        for b in sc.mass_bins(R.SCHEME):
            d = R.resolution_detail(b["rows"], grid, dict(R.REF))
            k = d["resolved"].all(axis=0)
            idx = np.flatnonzero(k)
            if not len(idx):
                continue
            gaps = [i for i in range(idx[0], len(k)) if not k[i]]
            res[b["label"]] = [{"x_in": float(e[i]), "x_out": float(e[i + 1]),
                                "width_dex": float(np.log10(e[i + 1] / e[i])),
                                "n_fail_count": int((~d["c_count"][:, i]).sum()),
                                "n_fail_smoothing_length": int((~d["c_hsml"][:, i]).sum()),
                                "n_fail_geometric": int((~d["geometric"][:, i]).sum())} for i in gaps]
        out[grid] = res
    return out


def per_halo_bias_by_fraction():
    """Per-halo / reference ratio aggregated over the calibrated bins' partially resolved shells (0 < f < 0.999),
    grouped by the inclusion-weighted fraction f of haloes resolved in the shell."""
    edges = [0.0, 0.05, 0.2, 0.4, 0.6, 0.8, 0.9, 0.999]
    out = {}
    for f, per_bin in M["per_halo_bias"].items():
        fr, ra = [], []
        for lab in R.CAL_LABELS:
            d = per_bin[lab]
            for a, b in zip(d["fraction_kept"], d["ratio"]):
                if a is not None and b is not None and 0 < a < 0.999:
                    fr.append(a)
                    ra.append(b)
        fr, ra = np.asarray(fr), np.asarray(ra)
        rows = {}
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = (fr >= lo) & (fr < hi)
            rows[f"{lo:g}-{hi:g}"] = {"n_shells": int(sel.sum()),
                                      "median_ratio": float(np.median(ra[sel])) if sel.any() else None,
                                      "min_ratio": float(ra[sel].min()) if sel.any() else None,
                                      "max_ratio": float(ra[sel].max()) if sel.any() else None}
        out[f] = rows
    return out


def main():
    settings = [{k: s[k] for k in ("id", "short", "dimension", "label", "description", "opts")} for s in R.SETTINGS]
    per_setting = {}
    for s in R.SETTINGS:
        sid = s["id"]
        per_setting[sid] = {"summary": summary(sid), "masks": M["masks"][sid],
                            "stack_band_ratios": M["stack_band_ratios"].get(sid), "fits": M["fits"][sid],
                            "impacts": M["impacts"][sid]}
    res = {
        "group": R.GROUP_NAME, "created": time.strftime("%Y-%m-%d %H:%M"),
        "reference": {"key": "field|geometric|removed|all|all|fit|<default norm> (weighted estimator for T_e, K)",
                      "statistic": "mean (inclusion-weighted)", "rule": "strict", "min_count": 20,
                      "softening_factor": 4.0, "hsml_factor": 2.0, "min_edge": 0.04, "scheme": R.SCHEME,
                      "n_boot": R.N_BOOT, "seeds": "stackcore default (20261012 + 1000 x bin index), stratified"},
        "conventions": {
            "bands_R200c": R.BAND_LABELS,
            "band_assignment": "a shell belongs to the band containing its volume-weighted radius 0.75(b^4-a^4)/(b^3-a^3)",
            "stack_band_ratio": "median over the shells of the band that the variant keeps of variant/reference stack; "
                                "err = SD over the 200 paired bootstrap replicates of the same median",
            "mask_only_settings": "strict / frac / diagnostic / none settings do not change the stacked values (every "
                                  "halo enters every shell), so their stack ratios are exactly 1; they act through "
                                  "the shells passed to the fit",
            "fits": "B12 (P) and B16 (E), analysis.fitting.objective_a (diagonal, sigma = max(SD/mean, 5%), analytic "
                    "Jacobian, radii234 bounds), on the setting's kept shells with outer edge <= 2 or 4 R200c; each "
                    "of the 200 paired replicates refitted with the full-sample sigma; shells with a single "
                    "contributing halo (sd = NaN) get a 100% error",
            "integrals": "int_lt1 = sum over shells with outer edge <= 1 R200c of F_s 4pi/3 dx^3 (Y(<R200c) for P, "
                         "N_e(<R200c) for E, dimensionless units); int_lt4 likewise to 4 R200c; 'vs_truth' divides by "
                         "the same sum of the reference stack over ALL shells (exact particle sums, resolution "
                         "independent); 'vs_ref' divides by the reference-mask fit",
            "impacts": "stackcore.impact_from_stacks on the 10 calibrated bins 13.5-15.5 x 35 fit shells (first "
                       "order, F_observables importance Jacobians). 'stack': variant stack on its kept shells vs "
                       "the reference stack (cells not kept = missing = no change). 'fitXR_vs_ref_fit': fitted "
                       "model with the setting's inner limit vs the reference-mask fit. 'fitXR_vs_truth': fitted "
                       "model vs the measured reference stack on all shells (painting error, first order). err = "
                       "SD over the 200 paired replicates",
            "softening": Z["softening"],
            "none_inner": "P_zero_fraction / E_zero_fraction: fraction of the bin's haloes with P (n_e) = 0 in the "
                          "shell, i.e. no gas other than recently AGN-heated gas (empty shells included); the fraction "
                          "with no gas particle at all is counts_profiles.<bin>.count_zero_fraction (also in "
                          "inner_decomposition)",
            "inner_decomposition": "per fit shell: stacked P, n_e and their hot (phase xray: SFR <= 0, T >= 1e5 K, not "
                                   "AGN) and cold / star-forming phases divided by the reference B12 / B16 fit to "
                                   "2 R200c (reference mask) extrapolated inward; E_excess_hot_share = (n_e,xray - "
                                   "model) / (n_e - model) where n_e > model; P_agn_kept_over_removed = stack with "
                                   "recently AGN-heated gas kept / reference; count_median, xh_median = median gas "
                                   "particles and median x_in / h over the bin's haloes",
            "missing_fit_bins": "per_setting.*.impacts.missing_fit_bins: calibrated bins where the setting leaves fewer "
                                "than MIN_SHELLS = 6 shells inside the outer fit limit, so no fit exists; impacts treat "
                                "those bins as 'no change' and missing_share gives their share of the observable's "
                                "importance",
        },
        "settings": settings,
        "per_setting": per_setting,
        "resolved_fraction_per_shell": M["resolved_fraction"],
        "counts_profiles": M["counts_profiles"],
        "per_halo_bias": M["per_halo_bias"],
        "per_halo_bias_by_fraction": per_halo_bias_by_fraction(),
        "none_inner": M["none_inner"],
        "inner_decomposition": M["inner_decomposition"],
        "report_numbers": report_numbers(),
        "none_inner_phase_check": phase_check(),
        "grid_dependence": M["grid_dependence"],
        "grid_gaps_strict": grid_gaps(),
        "n_dependence_of_strict": M["n_dependence"],
        "redshift": Z,
        "fit_edges_R200c": [float(x) for x in sc.edges(R.GRID)],
        "files": {"web": "web_variants.json", "figures": sorted(p.name for p in (R.GROUP / "figures").glob("*.png")),
                  "scripts": ["res_common.py", "run_resolution.py", "run_redshift.py", "make_figures.py",
                              "make_web.py", "assemble_results.py"]},
        "issues": [
            "stackcore.resolution_detail with hsml_factor = 0 still rejects empty halo-shells (NaN smoothing length "
            "compares False), so min_count = 0 behaves as min_count = 1 there; worked around locally "
            "(res_common.resolution_detail treats hsml_factor = 0 as 'criterion off'); no effect for min_count >= 1.",
            "analysis.fitting.objective_a aborts (non-finite residuals) when a kept shell has a single contributing "
            "halo (reliability-weighted SD = NaN); happens only with per-halo masks; worked around by a 100% error "
            "for such shells.",
        ],
        "wall_seconds_main": M["wall_seconds"],
    }
    R.write_json(R.GROUP / "results.json", res, n=5)
    print("results.json", (R.GROUP / "results.json").stat().st_size / 1e6, "MB")


if __name__ == "__main__":
    main()
