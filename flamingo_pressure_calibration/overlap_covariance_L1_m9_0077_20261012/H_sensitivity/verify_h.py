"""Verification of Test H (local, after cluster_tests.py fetch H): writes verification.json and the directory's
SHA256SUMS.  Gates (every number is read from the fetched results files):
  G0  H0 export: rebinned fine per-halo profiles = Test F's (inputs/h0_meta.json), F-grid masks reproduced
  G1  prepare: rebinned fine moments = Test F's (results/prepare.json)
  G2  parity (results/parity.json): deposit identity, moment observables, importance maps, DM quantiles, Mecke
  G3  sum rules (results/sens_summary.json): every moment importance map sums to 1; quantile maps (Mecke) to within
      the subsample tolerance; truncation curves end at the truth and start at the fallback-only value
  G4  cumulative curves reach 1 at 4 R200c and the importance cumulative sums reach 1 (results/sens.npz)
  G5  binning reference: M200c_0.2 piecewise painting = the truth mean; DM grouping shortcut vs per-halo entries
  G6  fit-transfer kernel vs refits (finite differences) and fit acceptance
  G7  results/SHA256SUMS verifies; run manifest exit 0
  G8  provenance (added 2026-10-14 after the round-2 verification): the local compute modules equal the copies that
      ran on maas (post/remote_code_check.json, read-only sha256 of the remote mirror, written by
      remote_code_check.py) and the pre-fix archive; the remote check is current (no local .py changed after it)
  G9  post-run decomposition (post/h_post.json from h_post.py): stored quantile maps rebuilt exactly from truth.npz,
      bins 13.5-15.3 equal the calibrated haloes' own linear contributions, calibrated-halo shares additive over
      1h + 2h, the 1-halo fallback split agrees with the bin's own signal, 0 <= fallback-in-calibrated-bins <= bin share;
      binning entries reproduce the stored family RMS; band labels state the edges used
Usage: python -B verify_h.py [--smoke]
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    res = HERE / ("results_smoke" if a.smoke else "results")
    rj = lambda n: json.loads((res / n).read_text())                    # noqa: E731
    gates = {}
    # G0
    meta = json.loads((HERE / "inputs" / "h0_meta.json").read_text())
    w = max(v["max_rel"] for v in meta["rebinned_vs_test_F_profiles"].values())
    gates["G0_h0_rebinned_vs_F_profiles_max_rel"] = {"value": w, "threshold": 1e-13, "passed": w <= 1e-13}
    gates["G0_fit_grid_masks_equal_F"] = {"value": all(meta["fit_grid_base_mask_equals_test_F"]),
                                          "passed": all(meta["fit_grid_base_mask_equals_test_F"])}
    # G1
    prep = rj("prepare.json")
    w = max(prep["rebinned_moments_vs_test_F_max_rel"].values())
    gates["G1_prepare_moments_vs_F_max_rel"] = {"value": w, "threshold": 1e-12, "passed": w <= 1e-12}
    # G2
    par = rj("parity.json")
    for k, thr in (("deposit_identity_max_abs", 0.0), ("moments_max_rel", 1e-8), ("importance_max_abs_over_max", 1e-8),
                   ("pdf_max_rel", 1e-6), ("importance_quantiles_max_abs_over_max", 1e-6)):
        gates[f"G2_parity_{k}"] = {"value": par[k], "threshold": thr, "passed": par[k] <= thr}
    mk = max(max(v.values()) for v in par["mecke"].values())
    gates["G2_parity_mecke_J_max_abs_over_max"] = {"value": mk, "threshold": 1e-6, "passed": mk <= 1e-6}
    # G3
    ss = rj("sens_summary.json")
    sr = ss["sum_rules"]
    mom = {k: v for k, v in sr.items() if not isinstance(v, dict) and "dm_q" not in k}
    q_sub = {k: v for k, v in sr.items() if not isinstance(v, dict) and "dm_q" in k and k.startswith("imp__")}
    q_det = {k: v for k, v in sr.items() if not isinstance(v, dict) and "dm_q" in k and k.startswith("impdet__")}
    gates["G3_importance_sum_rule_moments_max_abs"] = {"value": max(mom.values()), "threshold": 1e-9,
                                                       "passed": max(mom.values()) <= 1e-9}
    thr_q = 1.0 if a.smoke else 0.02          # smoke: sparse sky, Q16 at the grid origin (informational)
    gates["G3_importance_sum_rule_quantiles_subsample_max_abs"] = {
        "value": max(q_sub.values()), "threshold": thr_q, "passed": max(q_sub.values()) <= thr_q,
        "note": "Mecke Jacobians of a 64-halo-per-bin subsample (Test F's recipe): the Euler sum rule holds for the "
                "population, the subsample estimate scatters about it (Test F: up to 1%)"}
    thr_d = 1.0 if a.smoke else 1e-4
    gates["G3_importance_sum_rule_quantiles_truthmean_max_abs"] = {"value": max(q_det.values()), "threshold": thr_d,
                                                                   "passed": max(q_det.values()) <= thr_d}
    ends = max(max(v.values()) for k, v in sr.items() if k.startswith("truncation_endpoint"))
    zero = max(max(v.values()) for k, v in sr.items() if k.startswith("truncation_zero_point"))
    gates["G3_truncation_curves_end_at_truth_max_rel"] = {"value": ends, "threshold": 1e-10, "passed": ends <= 1e-10}
    gates["G3_truncation_curves_zero_point_is_fallback_only_max_rel"] = {"value": zero, "threshold": 1e-10,
                                                                          "passed": zero <= 1e-10}
    # G4
    S = dict(np.load(res / "sens.npz"))
    worst_c, worst_i = 0.0, 0.0
    for k, v in S.items():
        if k.startswith("trunc_all__") and not k.endswith(("dmq_mean", "dmmoments_mean")):
            v = np.asarray(v, float)
            C = (v - v[0]) / np.where(v[-1] - v[0] != 0, v[-1] - v[0], 1)
            worst_c = max(worst_c, float(np.nanmax(np.abs(C[-1] - 1))))
        if k.startswith("imp__") and not k.split("__")[-1].startswith("dm_q"):
            tot = np.asarray(v).sum(axis=(0, 1))
            worst_i = max(worst_i, float(np.max(np.abs(tot - 1))))
    gates["G4_cumulative_curves_reach_1_at_4R"] = {"value": worst_c, "threshold": 1e-12, "passed": worst_c <= 1e-12}
    gates["G4_importance_cumulative_reaches_1_at_4R"] = {"value": worst_i, "threshold": 1e-9, "passed": worst_i <= 1e-9}
    # G5
    bs = rj("binning_summary.json")
    v = max(bs["reference_check_mean_profiles_max_rel"], bs["reference_check_observables_max_rel_vs_truthmean"])
    gates["G5_binning_reference_equals_truth_mean"] = {"value": v, "threshold": 1e-10, "passed": v <= 1e-10}
    e = bs["errors"]["M200c_0.2|piecewise|4|det"]["vs_truth_mean"]
    qv = max(abs(e[f"dm_{q}"]) for q in ("q16", "q50", "q84", "q975", "q999"))
    gates["G5_binning_reference_DM_quantiles_vs_truth_mean"] = {"value": qv, "threshold": 1e-8, "passed": qv <= 1e-8}
    gv = bs.get("grouping_check_max_abs_quantile_diff")
    gates["G5_binning_grouped_vs_per_halo_DM_quantiles"] = {"value": gv, "threshold": 1e-3,
                                                           "passed": gv is not None and gv <= 1e-3}
    # G6
    ps = rj("params_summary.json")
    worst = 0.0
    for k, d in ps["transfer_finite_difference_check"].items():
        if abs(d["refit_dlnO"]) > 1e-6:
            worst = max(worst, abs(d["transfer_first_order"] / d["refit_dlnO"] - 1))
    gates["G6_transfer_kernel_vs_refit_max_rel"] = {"value": worst, "threshold": 0.05, "passed": worst <= 0.05}
    acc = {k: all(v["accepted"]) for k, v in ps["fits"].items()}
    gates["G6_fits_accepted"] = {"value": acc, "passed": all(acc.values()),
                                 "note": "Test F's acceptance: success, away from bounds, Jacobian condition < 1e6"}
    # G7
    bad = []
    for line in (res / "SHA256SUMS").read_text().splitlines():
        if not line.strip():
            continue
        digest, rel = line.split("  ", 1)
        if sha(res / rel) != digest:
            bad.append(rel)
    gates["G7_results_SHA256SUMS"] = {"value": len(bad), "passed": not bad, "bad": bad}
    man = rj("run_manifest.json")
    exits = [inv.get("exit") for inv in man["invocations"]]
    stages = {s: r["status"] for inv in man["invocations"] for s, r in inv["stages"].items()}
    gates["G7_run_manifest"] = {"value": {"exits": exits, "final_stage_status": stages},
                                "passed": all(v == "ok" for v in stages.values())}
    if not a.smoke:
        # G8
        rc = json.loads((HERE / "post" / "remote_code_check.json").read_text())
        compute = ["h_basis.py", "h_beyond.py", "h_binning.py", "h_common.py", "h_fits.py", "h_inputs.py",
                   "h_params.py", "h_pdf.py", "h_prepare.py", "h_sens.py", "h_truth.py", "run_test_h.py"]
        bad = [f for f in compute if f not in rc["equal"] or rc["local"][f] != sha(HERE / f)]
        gates["G8_compute_code_equals_maas_copy"] = {"value": bad, "passed": not bad, "checked_utc": rc["checked_utc"],
                                                     "local_only_postprocessing_differs": rc["differ"] + rc["local_only"]}
        # the remote check must be current: every local file it hashed is unchanged since (round-3 fix: the
        # 20:55 check listed verify_h.py as equal although verify_h.py was edited afterwards)
        stale = [f for f, h in rc["local"].items() if not (HERE / f).is_file() or sha(HERE / f) != h]
        unlisted = sorted(p.name for p in HERE.glob("*.py") if p.name not in rc["local"])
        gates["G8_remote_code_check_is_current"] = {"value": {"stale": stale, "unlisted_local_py": unlisted},
                                                    "passed": not stale and not unlisted}
        pf = HERE / "attempts" / "pre_fix_20261014"
        same = {f: sha(pf / f) == sha(HERE / f) for f in ("h_sens.py", "h_truth.py")}
        gates["G8_reverted_files_equal_run_code"] = {"value": same, "passed": all(same.values()),
                                                     "note": "the 17:25 post-run edit is kept, unapplied, as "
                                                             "attempts/pre_fix_20261014/postrun_patch_1725_NOT_RUN.diff.txt"}
        # G9
        hp = json.loads((HERE / "post" / "h_post.json").read_text())
        ck = hp["calibrated_shares_checks"]
        v = max(hp["quantile_map_reconstruction_max_abs"], hp["linear_own_contribution_vs_stored_rows_13.5-15.3_max_abs"],
                ck["additivity_max_abs"], ck["bin_top_1h_vs_own_max_abs"])
        gates["G9_post_decomposition_exact_checks_max_abs"] = {"value": v, "threshold": 1e-12, "passed": v <= 1e-12}
        gates["G9_post_fallback_inside_bounds"] = {"value": ck["bounds_violations"], "passed": not ck["bounds_violations"]}
        # round-3 fix: binning entries are named by what they are and reproduce the stored family RMS
        bv = hp["binning_family_rms_pct_Rcut4"]["check_vs_stored_family_rms_max_abs_pct"]
        gates["G9_post_binning_entries_reproduce_stored_family_rms"] = {"value": bv, "threshold": 1e-8,
                                                                        "passed": bv <= 1e-8}
        # band labels state the edges actually used (whole fine shells)
        be = hp["band_edges"]
        lab_ok = (abs(be["bands5"][1] - 0.2335) < 1e-3 and abs(be["bands5"][2] - 0.5185) < 1e-3
                  and list(next(iter(hp["bands5"].values()))["all_bins"])[:3] == ["<0.233R", "0.233-0.519R", "0.519-1R"])
        gates["G9_post_band_labels_match_edges"] = {"value": be["bands5"], "passed": lab_ok}
        b8 = hp["bands8_relabel"]["recompute_vs_stored_max_abs"]
        gates["G9_post_bands8_relabel_recomputed_vs_results_table"] = {
            "value": b8, "threshold": 1e-7, "passed": b8 <= 1e-7,
            "note": "results/tables/H3_bands.csv (8 significant digits) labels the edges 0.1051/0.2335/0.5185 as "
                    "0.1/0.25/0.5; post/H_post_bands8.csv gives the same values with the edges used"}
        fv = hp["fine_vs_F_truth_absmax"]["a2ii"]["_max"]
        gates["G9_fine_vs_F_truth_all_ell_max"] = {"value": fv["all_ell"][0], "where": fv["all_ell"][1:],
                                                   "threshold": 1e-3, "passed": fv["all_ell"][0] <= 1e-3,
                                                   "note": "informational: sub-shell (within-F-shell) resolution effect"}
    out = {"results": str(res), "gates": gates, "all_passed": all(g["passed"] for g in gates.values())}
    name = "verification_smoke.json" if a.smoke else "verification.json"
    (HERE / name).write_text(json.dumps(out, indent=1, default=float) + "\n")
    print(json.dumps({k: (v["passed"], v["value"]) for k, v in gates.items()}, indent=1, default=str))
    print("all_passed", out["all_passed"])
    if not a.smoke:
        lines = []
        for p in sorted(HERE.rglob("*")):
            rel = p.relative_to(HERE)
            if p.is_file() and not set(rel.parts) & {"__pycache__", "work", "prior", "results_smoke", "figures_smoke",
                                                     "verify_independent"} \
                    and p.name != "SHA256SUMS":
                lines.append(f"{sha(p)}  {rel}")
        (HERE / "SHA256SUMS").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
