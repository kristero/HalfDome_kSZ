"""Digest of Test H's fetched results for the report (local, read-only post-processing; no new physics).

Collects, from results/{truth,parity,prepare,sens,beyond,binning,params}*.json|npz and the tables written by
report_numbers.py, the numbers quoted in the report section, and writes
  results/tables/H_digest.json      (nested: every quoted number with the file/field it came from)
  results/tables/H_digest.txt       (the same, as readable text)
  results/tables/H_report_*.csv     (compact tables: truncation radii, beyond-4R, binning, parameters)
Derived quantities (labelled as such):
  coherent parameter sensitivity  G_O[k] = sum_c g_O[c, k]: d ln O for the same d ln theta_k in every calibrated bin
  family tolerance                0.01 / sqrt(lambda_max) of the Fisher-type matrix I_fam[c]: the shift of ln theta
                                  along the best-constrained direction that changes the family's RMS by 1%
Usage: python -B report_digest.py [--smoke]
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CAL_LABELS = ["13.5-13.7", "13.7-13.9", "13.9-14.1", "14.1-14.3", "14.3-14.5", "14.5-14.7", "14.7-14.9", "14.9-15.1",
              "15.1-15.3", "15.3-15.5"]


def wcsv(path, rows):
    cols = []
    for r in rows:
        for c in r:
            if c not in cols:
                cols.append(c)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: (f"{v:.5g}" if isinstance(v, float) else v) for c, v in r.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    res = HERE / ("results_smoke" if a.smoke else "results")
    tab = res / "tables"
    rj = lambda n: json.loads((res / n).read_text())                    # noqa: E731
    D = {"source_dir": str(res)}
    # ------------------------------------------------------------------ truth, parity, resolution effect
    ts = rj("truth_summary.json")
    par = rj("parity.json")
    prep = rj("prepare.json")
    D["prepare"] = {"n_haloes": prep.get("n_haloes"), "haloes_per_calibrated_bin": prep.get("haloes_per_calibrated_bin"),
                    "rebinned_moments_vs_F_max_rel": max(prep["rebinned_moments_vs_test_F_max_rel"].values()),
                    "_src": "prepare.json"}
    D["parity"] = {k: par[k] for k in ("deposit_identity_max_abs", "moments_max_rel", "importance_max_abs_over_max",
                                        "pdf_max_rel", "importance_quantiles_max_abs_over_max", "passed")}
    D["parity"]["mecke_max"] = max(max(v.values()) for v in par["mecke"].values())
    D["parity"]["_src"] = "parity.json"
    D["fine_vs_F_truth"] = ts["fine_vs_F_truth"]
    D["basis_shape_sensitivity"] = ts["basis_shape_sensitivity"]
    D["pdf_vs_moments"] = ts.get("pdf_vs_moments")
    D["mecke_sum_rule"] = ts.get("mecke_sum_rule")
    obs = ts["observables"]
    keys = ["yytot_100", "yytot_1000", "yytot_3000", "ymean", "ksz1h_100", "ksz1h_1000", "ksz1h_3000", "dm_mean",
            "dm_var", "ydmtot_1000", "dmcltot_1000"]
    D["a2ii_over_main"] = {k: obs["truth_a2ii"][k] / obs["truth_main"][k] - 1 for k in keys if k in obs["truth_a2ii"]}
    D["a2ii_over_main"].update({f"dm_{q}": ts["pdf"]["truth_a2ii"]["quantiles"][p] / ts["pdf"]["truth_main"]["quantiles"][p] - 1
                                for p, q in enumerate(("q16", "q50", "q84", "q975", "q999"))})
    D["scatter_boost_truth_over_truthmean_a2ii"] = {k: obs["truth_a2ii"][k] / obs["truthmean_a2ii"][k] - 1 for k in keys
                                                   if k in obs["truth_a2ii"]}
    D["calibrated_fraction_a2ii"] = {k: obs["truthcal_a2ii"][k] / obs["truth_a2ii"][k] for k in keys
                                     if k in obs["truthcal_a2ii"]}
    # ------------------------------------------------------------------ H3
    ss = rj("sens_summary.json")
    kn = json.loads((tab / "H_key_numbers.json").read_text())
    rad = {}
    for sel in ("a2ii", "main"):
        for k, v in kn.items():
            if k.startswith(f"{sel}|"):
                rad[k] = v
    D["H3_radii"] = rad
    D["H3_radii_src"] = "tables/H_key_numbers.json (report_numbers.py from sens.npz trunc_all__*)"
    D["H3_bands"] = {k: v for k, v in ss["bands"].items()}
    D["H3_peaks"] = ss["peaks"]
    D["H3_calibrated_share"] = {k: v["calibrated"] for k, v in ss["shares"].items()}
    D["H3_self_fraction"] = ss["self_fraction"]
    D["H3_unresolved_share"] = kn.get("unresolved_calibrated_share")
    D["H3_first_resolved_edge"] = kn.get("first_resolved_inner_edge_by_bin")
    D["H3_tolerance"] = ss["tolerance"]
    D["H3_sum_rules_max"] = {
        "moments": max(v for k, v in ss["sum_rules"].items() if not isinstance(v, dict) and "dm_q" not in k),
        "quantiles_subsample": max(v for k, v in ss["sum_rules"].items() if not isinstance(v, dict) and "dm_q" in k
                                   and k.startswith("imp__")),
        "quantiles_truthmean": max(v for k, v in ss["sum_rules"].items() if not isinstance(v, dict) and "dm_q" in k
                                   and k.startswith("impdet__"))}
    D["H3_per_halo_quantile_losses"] = {sel: {k: v for k, v in ss["radii"][sel].items() if k.endswith("per_halo_loss")}
                                        for sel in ("a2ii", "main")}
    # per-bin R90 table (own contribution)
    rows = list(csv.DictReader(open(tab / "H_report_radii_by_bin.csv")))
    D["H3_R90_by_bin"] = {}
    for r in rows:
        D["H3_R90_by_bin"].setdefault(f"{r['observable']}@{r['ell'] or 'scalar'}", {})[CAL_LABELS[int(r["bin"])]] = \
            {k: float(r[k]) if r[k] not in ("", "nan") else None for k in ("R50", "R90", "R95", "R99", "R99conv")}
    # ------------------------------------------------------------------ H4
    bs = rj("beyond_summary.json")
    D["H4_changes"] = {}
    for sel in ("a2ii", "main"):
        for model, d in bs["changes"].get(sel, {}).items():
            D["H4_changes"][f"{sel}|{model}"] = {R: {k: v for k, v in d[R].items()} for R in ("R5", "R6", "R8", "R12")
                                                 if R in d}
    D["H4_dm_quantiles"] = bs["changes"].get("dm_quantiles")
    D["H4_double_counting"] = bs["double_counting"]
    D["H4_models"] = bs["models"]
    B = dict(np.load(res / "beyond.npz"))
    D["H4_budget_added_median"] = {}
    for k, v in B.items():
        if k.startswith("budget__") and any(k.endswith(f"_R{r}") for r in ("5", "6", "8", "12")):
            D["H4_budget_added_median"][k] = {"median": float(np.median(v)), "min": float(np.min(v)),
                                              "max": float(np.max(v))}
    # ------------------------------------------------------------------ H5
    bn = rj("binning_summary.json")
    D["H5_reference_checks"] = {k: bn.get(k) for k in ("reference_check_mean_profiles_max_rel",
                                                        "reference_check_observables_max_rel_vs_truthmean",
                                                        "grouping_check_max_abs_quantile_diff")}
    D["H5_schemes"] = {s: [(b["lo"], b["hi"], b["N"]) for b in v["bins"]] for s, v in bn["schemes"].items()}
    rows5 = []
    D["H5_family_rms"] = {}
    for lab, e in bn["errors"].items():
        sc, mode, rc, scat = lab.split("|")
        D["H5_family_rms"][lab] = {"vs_truth": e["family_rms_vs_truth"], "vs_truth_mean": e["family_rms_vs_truth_mean"],
                                   "dm_quantiles_included": e["dm_quantiles_included"]}
        rows5.append(dict(scheme=sc, mode=mode, R_cut=float(rc), scatter=scat,
                          **{f"{f}_vs_truth_pct": 100 * v for f, v in e["family_rms_vs_truth"].items()},
                          **{f"{f}_vs_mean_pct": 100 * v for f, v in e["family_rms_vs_truth_mean"].items()},
                          dmq_included=e["dm_quantiles_included"]))
    wcsv(tab / "H_report_binning.csv", rows5)
    # worst single-observable errors (det vs truth mean, R_cut 4) per scheme / mode
    D["H5_max_abs_error_det_vs_mean_Rc4"] = {}
    for sc in bn["schemes"]:
        for mode in ("piecewise", "interp"):
            e = bn["errors"][f"{sc}|{mode}|4|det"]["vs_truth_mean"]
            k = max(e, key=lambda q: abs(e[q]))
            D["H5_max_abs_error_det_vs_mean_Rc4"][f"{sc}|{mode}"] = {"observable": k, "error": e[k]}
    # ------------------------------------------------------------------ H6
    ps = rj("params_summary.json")
    P = dict(np.load(res / "params.npz"))
    D["H6_fits"] = {k: {"parameters": v["parameters"], "chi2_per_dof": v["chi2_per_dof"], "rms_percent": v["rms_percent"],
                        "accepted": v["accepted"], "n_shells": v["n_shells"],
                        "influence_by_region_abs": {reg: np.abs(np.asarray(h)).tolist()
                                                    for reg, h in v["influence_by_region"].items()}}
                    for k, v in ps["fits"].items()}
    D["H6_painting_errors"] = ps["painting_errors"]
    coh = {}
    for k, v in P.items():
        if k.startswith("g__"):
            _, rf, field, lab = k.split("__")
            coh[f"{rf}|{field}|{lab}"] = dict(zip(("p1", "p2", "p3"), v.sum(axis=0).tolist()))
    D["H6_coherent_sensitivity"] = coh
    D["H6_coherent_sensitivity_note"] = ("derived: sum over calibrated bins of d ln O / d ln theta_k (params.npz g__*); "
                                         "parameters P0, x_c, beta (P) or rho0, alpha, beta (E)")
    fis = {}
    for k, rows_ in ps["fisher"].items():
        fis[k] = {b: {"eigenvalues": r["eigenvalues"], "tol_1pct": r["tolerance_lnparam_1pct_along_eigvecs"],
                      "best": r["most_constrained_combination"], "worst": r["least_constrained_combination"],
                      "fit_best": r["fit_best_combination"],
                      "alignment": r["alignment_obs_best_vs_fit_eigvecs_desc"]} for b, r in rows_.items()}
    D["H6_fisher"] = fis
    D["H6_transfer"] = ps["transfer"]
    D["H6_transfer_fd"] = ps["transfer_finite_difference_check"]
    D["H6_bootstrap"] = ps["bootstrap"]
    rows6 = []
    for key, reg in ps["transfer"].items():
        rf, field, lab = key.split("|")
        rows6.append(dict(R_fit=rf, field=field, observable=lab,
                          **{f"{r}_{t}": v[t] for r, v in reg.items() for t in ("direct", "transfer")}))
    wcsv(tab / "H_report_transfer.csv", rows6)
    (tab / "H_digest.json").write_text(json.dumps(D, indent=1, default=float) + "\n")
    # ------------------------------------------------------------------ text digest
    lines = []

    def p(*x):
        lines.append(" ".join(str(y) for y in x))
    p("== parity", json.dumps({k: v for k, v in D["parity"].items()}, default=float))
    p("== fine vs F truth (sub-shell resolution effect)", json.dumps(D["fine_vs_F_truth"], default=float)[:3000])
    p("== basis shape sensitivity", json.dumps(D["basis_shape_sensitivity"], default=float)[:3000])
    p("== A2-ii / all gas - 1", json.dumps(D["a2ii_over_main"], default=float))
    p("== scatter boost (truth / truth mean - 1), A2-ii", json.dumps(D["scatter_boost_truth_over_truthmean_a2ii"], default=float))
    p("== calibrated fraction, A2-ii", json.dumps(D["calibrated_fraction_a2ii"], default=float))
    p("== H3 radii (calibrated-signal fraction; R_X first crossing, R_Xconv converged; losses 1-T at cuts)")
    for k, v in rad.items():
        if isinstance(v, dict) and "R50" in v:
            p(f"  {k:32s}", " ".join(f"{q}={v[q]:.3g}" for q in ("R50", "R80", "R90", "R95", "R99", "R90conv", "R95conv",
                                                                  "R99conv") if q in v and v[q] is not None),
              "| loss", " ".join(f"{c}={100 * v[f'loss_total_{c}R']:.2f}%" for c in ("1", "1.5", "2", "3", "3.6342")
                                 if f"loss_total_{c}R" in v), f"Cmax={v.get('C_max', float('nan')):.3f}")
        elif isinstance(v, dict):
            p(f"  {k:32s}", " ".join(f"{q}={100 * x:.2f}%" for q, x in v.items()))
    p("== H3 bands (importance, A2-ii)")
    for k, v in D["H3_bands"].items():
        if k.startswith("a2ii__"):
            p(f"  {k:28s}", " ".join(f"{b}:{100 * x:.1f}" for b, x in v.items()),
              f"| cal {100 * D['H3_calibrated_share'][k]:.1f}% | peak x {D['H3_peaks'][k]['x_peak_density']:.3g}")
    p("== H3 bands (importance, all gas)")
    for k, v in D["H3_bands"].items():
        if k.startswith("main__"):
            p(f"  {k:28s}", " ".join(f"{b}:{100 * x:.1f}" for b, x in v.items()),
              f"| cal {100 * D['H3_calibrated_share'][k]:.1f}% | peak x {D['H3_peaks'][k]['x_peak_density']:.3g}")
    p("== H3 self fraction", json.dumps(D["H3_self_fraction"], default=float))
    p("== H3 unresolved calibrated share (A2-ii)", json.dumps({k: v for k, v in (D["H3_unresolved_share"] or {}).items()
                                                               if k.startswith("a2ii")}, default=float)[:2500])
    p("== H3 tolerance (min per 0.1 dex, A2-ii)")
    for k, v in D["H3_tolerance"].items():
        if k.startswith("a2ii__"):
            pc = lambda t: "inf" if t is None else f"{100 * t:.3g}%"                       # noqa: E731
            p(f"  {k:28s} min {pc(v['min_tol_per_0.1dex'])} at x={v['at_x']:.3g};",
              " ".join(f"{r}:{pc(t)}" for r, t in v["tol_per_0.1dex_at"].items()))
    p("== H3 per-bin radii (own contribution)")
    for k, v in D["H3_R90_by_bin"].items():
        p(f"  {k:16s}", " ".join(f"{b}:{(x['R90'] or float('nan')):.2f}" for b, x in v.items()))
    p("== H3 per-halo quantile losses", json.dumps(D["H3_per_halo_quantile_losses"], default=float))
    p("== H4 changes (moments)")
    for k, v in D["H4_changes"].items():
        for R, d in v.items():
            p(f"  {k:34s} {R:4s}", " ".join(f"{q}:{100 * x:+.2f}%" for q, x in d.items()
                                             if q in ("yytot_100", "yytot_1000", "yytot_3000", "ymean", "ksz1h_100",
                                                      "ksz1h_1000", "ksz1h_3000", "ksz2h_100", "dm_mean", "dm_var",
                                                      "dmcltot_100", "dmcltot_1000", "ydmtot_1000")))
    p("== H4 DM quantiles", json.dumps(D["H4_dm_quantiles"], default=float)[:4000])
    p("== H4 double counting", json.dumps(D["H4_double_counting"], default=float))
    p("== H4 models", json.dumps(D["H4_models"], default=float)[:3000])
    p("== H4 budgets", json.dumps(D["H4_budget_added_median"], default=float)[:3000])
    p("== H5 reference checks", json.dumps(D["H5_reference_checks"], default=float))
    p("== H5 schemes", json.dumps(D["H5_schemes"], default=float)[:2000])
    p("== H5 family RMS (%), tSZ/kSZ/DM/tSZxDM")
    for lab, v in D["H5_family_rms"].items():
        f = lambda d: " ".join(f"{100 * d.get(q, float('nan')):.3f}" for q in ("tSZ", "kSZ", "DM", "tSZxDM"))  # noqa: E731
        p(f"  {lab:34s} vs truth {f(v['vs_truth'])} | vs mean {f(v['vs_truth_mean'])} | dmq {v['dm_quantiles_included']}")
    p("== H5 max abs error det vs mean Rc4", json.dumps(D["H5_max_abs_error_det_vs_mean_Rc4"], default=float))
    p("== H6 fits")
    for k, v in D["H6_fits"].items():
        p(f"  {k}: chi2/dof", np.round(v["chi2_per_dof"], 3).tolist(), "rms%", np.round(v["rms_percent"], 2).tolist())
        p("     params", np.round(np.asarray(v["parameters"]), 4).tolist())
        p("     |influence| by region", {r: np.round(np.asarray(h).sum(axis=1) / 3, 3).tolist()
                                         for r, h in v["influence_by_region_abs"].items()})
    p("== H6 painting errors", json.dumps(D["H6_painting_errors"], default=float)[:6000])
    p("== H6 coherent sensitivity")
    for k, v in coh.items():
        p(f"  {k:36s}", " ".join(f"{q}={x:+.3f}" for q, x in v.items()))
    p("== H6 fisher (per bin: eigenvalues, tolerance, best combination)")
    for k, v in fis.items():
        for b, r in v.items():
            p(f"  {k:18s} {b}: ev", np.round(r["eigenvalues"], 5).tolist(), "tol", np.round(r["tol_1pct"], 4).tolist(),
              "best", {q: round(x, 2) for q, x in r["best"].items()}, "align", np.round(r["alignment"], 3).tolist())
    p("== H6 transfer vs direct")
    for k, v in D["H6_transfer"].items():
        p(f"  {k:30s}", " ".join(f"{r}: T {x['transfer']:.4f} D {x['direct']:.4f};" for r, x in v.items()))
    p("== H6 FD check", json.dumps(D["H6_transfer_fd"], default=float))
    bt = D["H6_bootstrap"]
    for k in ("P_Rf4", "P_Rf2", "E_Rf4", "E_Rf2"):
        if k in bt:
            p(f"== H6 bootstrap {k} sigma_lnparam", np.round(np.asarray(bt[k]["sigma_lnparam"]), 4).tolist(),
              "n_ok", bt[k]["n_ok"])
            p("   corr", [np.round(np.asarray(c), 2).tolist() for c in bt[k]["corr"]])
    for k in ("observables_Rf4", "observables_Rf2", "observables_stack_no_fit"):
        if k in bt:
            p(f"== H6 bootstrap {k} SD %", " ".join(f"{q}:{100 * x['sd']:.3f}" for q, x in bt[k].items()))
    (tab / "H_digest.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
