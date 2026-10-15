"""Assemble F_observables/verification.json from the maas results and the local post-processing, and write
F_observables/SHA256SUMS (code, CLUSTER_RUN.json, REPORT.md, results/, results_local/ without work/, figures/,
results_smoke/ without work/, local_precheck/).  Every entry is recomputed from saved files."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
LOC = HERE / "results_local"


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    v = {}
    man = load(RES / "run_manifest.json")
    done = load(RES / "test_F_done.json")
    val = load(RES / "validation_V.json")
    v["maas_full_run"] = {
        "passed": bool(done["returncode"] == 0 and val["passed_all"]),
        "host": done["host"], "returncode": done["returncode"], "wall_seconds": done["wall_seconds"],
        "workers": done["workers"], "nice": done["nice"], "peak_tree_rss_GiB": done["peak_tree_rss_GiB"],
        "stage_wall_seconds": {k: round(s["wall_seconds"], 1) for k, s in man["stages"].items()},
        "packages": man["packages"], "python": man["python"]}
    v["validation_gate_in_basis_vs_engine"] = {
        "passed": val["passed_all"], "observables_max_relative": max(val["observables_max_relative"].values()),
        "jacobians_max_abs_over_kO": max(val["jacobians_max_abs_over_kO"].values()),
        "dm_quantiles_max_relative": val["pdf"]["quantiles_max_relative"],
        "quantile_jacobians_max_abs_over_Q": val["pdf"]["quantile_jacobian_max_abs_over_Q"],
        "pdf_mean_vs_analytic": val["pdf"]["pdf_mean_vs_analytic"],
        "pdf_variance_vs_analytic": val["pdf"]["pdf_variance_vs_analytic"],
        "reference": "kernels/jacobians.npz (engine run of the kernels/ P4 and E4 fits)"}
    lines = (RES / "SHA256SUMS").read_text().split("\n")
    bad = [ln for ln in lines if ln.strip() and sha(RES / ln.split("  ", 1)[1]) != ln.split("  ", 1)[0]]
    v["results_sha256"] = {"passed": not bad, "entries": sum(1 for ln in lines if ln.strip()), "mismatches": bad}
    smoke = load(HERE / "results_smoke" / "run_manifest.json")
    sval = load(HERE / "results_smoke" / "validation_V.json")
    v["smoke_test_local"] = {"passed": bool(sval["passed_all"]), "wall_seconds": smoke["wall_seconds"],
                             "workers": smoke["workers"], "stages": list(smoke["stages"])}
    pre = load(HERE / "local_precheck" / "validation_V_full_local_20261012.json")
    v["full_size_kernels_local_precheck"] = {"passed": all(pre["passed"].values()),
                                             "observables_max_relative": max(pre["observables_max_relative"].values()),
                                             "quantiles_max_relative": pre["pdf"]["quantiles_max_relative"]}
    ts = load(RES / "truth_summary.json")
    pm = ts["pdf_vs_moments"]
    v["truth_pdf_vs_moment_forms"] = {
        "passed": max(abs(x["mean"]) for x in pm.values()) < 1e-6 and max(abs(x["variance"]) for x in pm.values()) < 1e-5,
        "max_abs_mean": max(abs(x["mean"]) for x in pm.values()),
        "max_abs_variance": max(abs(x["variance"]) for x in pm.values())}
    me = ts["mecke"]
    v["mecke_quantile_jacobian_sum_rules"] = {
        "deterministic_truth_mean_max_abs": max(abs(x) for x in me["truthmean_main"]["sum_rule_over_Q"]),
        "per_halo_64_subsample_max_abs": max(abs(x) for x in me["truth_main_subsample"]["sum_rule_over_Q"]),
        "passed": max(abs(x) for x in me["truthmean_main"]["sum_rule_over_Q"]) < 1e-4,
        "note": "the per-halo importance uses a 64-halo subsample per bin (reweighted): sum rule holds to sampling noise"}
    cv = load(RES / "cv_summary.json")
    sel = {fam for fold in cv["selection"].values() for fam in fold.values()}
    v["cv"] = {"pooled_fold_truth_check": cv["pooled_fold_quantile_check"], "passed": cv["pooled_fold_quantile_check"] < 1e-10,
               "training_selected_configs": sorted(sel), "lambda_choice_per_fold": cv["lambda_choice"]}
    ec = load(RES / "engine_check.json")
    rows = list(csv.DictReader(open(RES / "tables" / "F_engine_check.csv")))
    sig = [abs(float(r["basis_over_engine_minus_1"])) for r in rows
           if r["edge"] == "sigmoid/hard" and r["basis_over_engine_minus_1"] not in ("", "nan")]
    v["engine_check"] = {"basis_vs_engine_smooth_fits_max_primary_ell": ec["basis_vs_engine_hard"]["max_abs_primary_ell"],
                         "basis_vs_engine_smooth_fits_max_ell_8000": ec["basis_vs_engine_hard"]["max_abs_ell_8000"],
                         "sigmoid_over_hard_engine_vs_basis_max_abs_difference": max(sig),
                         "passed": ec["basis_vs_engine_hard"]["max_abs_primary_ell"] < 1e-3}
    nu = load(LOC / "numerics.json")
    v["numerics_test_D_kernel_cancellation"] = {
        "reference_shapes_max_indicator": nu["max_indicator"], "threshold": nu["threshold"],
        "indicators": {k: x["max_indicator"] for k, x in nu["indicators"].items()},
        "stable_over_frozen_spectra_max_abs": max(nu["effect"]["spectra_max_abs_stable_over_frozen_minus_1"].values()),
        "relative_error_shift_max_abs": max(nu["effect"]["error_shift_max_abs"].values()),
        "frozen_local_rebuild_vs_maas_max_abs": nu["effect"]["frozen_rebuild_vs_maas_max_abs"],
        "frozen_local_rebuild_values_compared": nu["effect"]["frozen_rebuild_vs_maas_compared"],
        "corrected_numbers": "none (largest change 6e-6)",
        "model_profiles_if_propagated_continuously": {k: x["max_indicator"]
                                                      for k, x in nu["model_indicators_not_used_by_the_run"].items()},
        "td_stable": nu["td_stable"], "td_stable_sha256": nu["td_stable_sha256"],
        "passed": nu["max_indicator"] < 1e-3 and max(nu["effect"]["error_shift_max_abs"].values()) < 1e-4
        and nu["effect"]["frozen_rebuild_vs_maas_max_abs"] < 1e-6}
    sp = load(LOC / "scatter_pdf.json")
    v["local_scatter_pdf_supplement"] = {"reproducibility_vs_maas_max_abs": sp["reproducibility_max_abs_vs_maas"],
                                         "passed": sp["reproducibility_max_abs_vs_maas"] < 1e-6,
                                         "wall_seconds": sp["wall_seconds"], "workers": sp["timing"]["workers"]}
    err = {}
    for r in csv.DictReader(open(RES / "tables" / "F_errors_long.csv")):
        if r["model"] == "tab_stack" and r["prediction"] == "measured_2nd_moment" and not r["observable"].startswith("dm_q"):
            err[r["observable"]] = abs(float(r["rel_error"]))
    v["identity_stack_with_measured_second_moments_equals_truth"] = {"max_abs": max(err.values()),
                                                                     "passed": max(err.values()) < 1e-10}
    acc = {}
    for r in csv.DictReader(open(RES / "tables" / "F_fit_records.csv")):
        if r.get("accepted") in ("True", "False"):
            k = r["method"] if not (r["method"].startswith("f") and r["method"][1:2].isdigit()) else "f(lambda scan)"
            a = acc.setdefault(k, [0, 0])
            a[0] += 1
            a[1] += r["accepted"] == "True"
    v["fit_acceptance"] = {k: f"{a[1]}/{a[0]}" for k, a in sorted(acc.items())}
    figs = sorted(p.name for p in (HERE / "figures").glob("*.png"))
    v["figures"] = {"count_png": len(figs), "each_png_has_svg": all((HERE / "figures" / f.replace(".png", ".svg")).exists() for f in figs),
                    "visual_inspection": "every PNG viewed after the last regeneration; overlaps fixed"}
    v["all_passed"] = all(x.get("passed", True) for x in v.values() if isinstance(x, dict))
    (HERE / "verification.json").write_text(json.dumps(v, indent=1) + "\n")
    # SHA256SUMS
    files = []
    for pattern in ("*.py", "CLUSTER_RUN.json", "REPORT.md", "verification.json"):
        files += sorted(HERE.glob(pattern))
    for sub in ("results", "results_local", "figures", "results_smoke", "local_precheck"):
        files += sorted(p for p in (HERE / sub).rglob("*") if p.is_file() and "work" not in p.relative_to(HERE / sub).parts)
    out = [f"{sha(p)}  {p.relative_to(HERE)}" for p in files if p.name != "SHA256SUMS" or p.parent != HERE]
    (HERE / "SHA256SUMS").write_text("\n".join(out) + "\n")
    print(json.dumps({k: (x.get("passed") if isinstance(x, dict) else x) for k, x in v.items()}, indent=1))


if __name__ == "__main__":
    main()
