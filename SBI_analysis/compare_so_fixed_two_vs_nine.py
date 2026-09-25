#!/usr/bin/env python3
"""Descriptive comparisons of fixed-nuisance two-parameter and nine-parameter SBI.

Read completed metric caches only. Never pair unrelated Sobol rows between
experiments or represent the preliminary nine-parameter subset as complete.
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PARAMS = ["P0", "xc", "beta", "alpha_m_P0", "alpha_m_xc", "alpha_m_beta",
          "alpha_z_P0", "alpha_z_xc", "alpha_z_beta"]
LABELS = dict(zip(PARAMS, [r"$P_0$", r"$x_{\rm c}$", r"$\beta$",
    r"$\alpha_{m,P_0}$", r"$\alpha_{m,x_{\rm c}}$", r"$\alpha_{m,\beta}$",
    r"$\alpha_{z,P_0}$", r"$\alpha_{z,x_{\rm c}}$", r"$\alpha_{z,\beta}$"]))
TARGETS = ["P0", "beta"]
STYLES = {
    "two_bins40": ("2-param: 40 bins", "#0072B2", "o", "-"),
    "two_moped": ("2-param: MOPED (2)", "#D55E00", "s", "-"),
    "nine_legacy": ("9-param: earlier 40-bin sweep", "#525252", "^", "--"),
    "nine_bins40": ("9-param: 40 bins, preliminary", "#009E73", "D", "None"),
    "nine_moped": ("9-param: MOPED (9), preliminary", "#CC79A7", "P", "None"),
}


def read_json(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_shared(root):
    with np.load(root / "shared.npz", allow_pickle=False) as z:
        keys = ("theta", "low", "high", "param_names", "test_indices", "pool_indices")
        return {key: z[key] for key in keys}


def disjoint_training(root, test_indices):
    with np.load(root / "shared.npz", allow_pickle=False) as z:
        fit, validation = z["fit_indices"], z["validation_indices"]
        require(not np.intersect1d(fit, validation).size, f"Fit/validation overlap: {root}")
        require(not np.intersect1d(np.concatenate([fit, validation]), test_indices).size,
                f"Training/test overlap: {root}")
        return len(fit), len(validation)


def validate_rows(frame, names, test_indices, truth, low, high):
    """Validate each run on its own test population, not another experiment's IDs."""
    required = ["method", "n_train", "test_index", "param", "truth", "mean", "std", "pull"]
    require(set(required) <= set(frame), "Missing metric columns")
    require(not frame.empty, "Empty metrics")
    require(set(frame.param) == set(names), "Missing or unexpected parameters")
    require(not frame.duplicated(["method", "n_train", "test_index", "param"]).any(),
            "Duplicated posterior metrics")
    numeric = frame[["n_train", "test_index", "truth", "mean", "std", "pull"]].to_numpy(float)
    require(np.isfinite(numeric).all() and (frame["std"] > 0).all(), "Invalid metrics/std")
    for key in ("n_train", "test_index"):
        require(np.equal(frame[key], frame[key].astype(np.int64)).all(), f"Noninteger {key}")
    for _, sub in frame.groupby(["method", "n_train", "param"]):
        require(set(sub.test_index) == set(test_indices), "Missing or inconsistent test rows")
    param_index = {name: j for j, name in enumerate(names)}
    columns = frame.param.map(param_index).to_numpy(int)
    np.testing.assert_allclose(frame.truth, truth[frame.test_index.to_numpy(int), columns],
                               rtol=1e-6, atol=1e-8)
    frame = frame.copy()
    frame["prior_low"] = np.asarray(low)[columns]
    frame["prior_high"] = np.asarray(high)[columns]
    error = frame["mean"] - frame.truth
    np.testing.assert_allclose(frame.pull, error / frame["std"], rtol=1e-6, atol=1e-7)
    stored = "error_prior" if "error_prior" in frame else "normalized_error_prior"
    np.testing.assert_allclose(frame[stored], error / (frame.prior_high-frame.prior_low),
                               rtol=1e-6, atol=1e-7)
    return frame


def load_two(root):
    config = read_json(root / "experiment.json")
    complete = read_json(root / "summary/summary_complete.json")
    require(complete["experiment_id"] == config["experiment_id"], "Stale two-param summary")
    require(not complete["selective_omission"], "Incomplete two-param test sample")
    require(config["fixed_noise_diagnostic"], "This comparison requires fixed-noise labeling")
    shared = load_shared(root)
    np.testing.assert_array_equal(shared["param_names"], TARGETS)
    frame = pd.read_csv(root / "summary/per_profile_metrics.csv")
    require(set(frame.method) == {"bins40", "moped"}, "Wrong two-param methods")
    require(set(frame.n_train) == set(config["sizes"]), "Missing two-param sizes")
    require(complete["n_test"] == config["n_test"] == len(shared["test_indices"]), "Wrong test count")
    audit = []
    for n in config["sizes"]:
        location = root / f"N{n}"
        n_fit, n_validation = disjoint_training(location, shared["test_indices"])
        require(n_fit+n_validation == n, "Wrong size split")
        for method in ("bins40", "moped"):
            run = location / method
            training = read_json(run / "training_complete.json")
            done = read_json(run / "run_complete.json")
            require(done["experiment_id"] == training["experiment_id"] == f"{config['experiment_id']}:N{n}",
                    f"Stale run: {run}")
            require(training["weights_selection"] == "best_validation_snapshot", "Wrong model snapshot")
            require((run / "density_estimator.pkl").stat().st_size > 0, "Missing saved estimator")
            audit.append(dict(n_train=n, method=method, n_fit=n_fit, n_validation=n_validation,
                              epochs=training["epochs_trained"],
                              early_stopping=training["converged_by_early_stopping"]))
    frame = validate_rows(frame, TARGETS, shared["test_indices"], shared["theta"], shared["low"], shared["high"])
    frame["series"] = "two_" + frame.method
    return frame, config, shared, audit


def load_nine(root, allow_preliminary):
    config = read_json(root / "experiment.json")
    require(allow_preliminary, "Nine-param reference is preliminary; pass --allow-preliminary-nine explicitly")
    summary = root / "summary_preliminary"
    marker = read_json(summary / "preliminary_summary_complete.json")
    require(marker["experiment_id"] == config["experiment_id"], "Stale nine-param reference")
    require(marker["preliminary"] is True, "Unexpected nine-param summary status")
    shared = load_shared(root)
    np.testing.assert_array_equal(shared["param_names"], PARAMS)
    fit, validation = disjoint_training(root, shared["test_indices"])
    require(fit+validation == config["n_train"], "Wrong nine-param training count")
    selected = np.load(summary / "compared_test_indices.npy", allow_pickle=False)
    require(set(selected) <= set(shared["test_indices"]), "Invalid preliminary test indices")
    require(len(selected) == marker["n_compared"] and len(shared["test_indices"]) == marker["n_requested"],
            "Preliminary sample count mismatch")
    frame = pd.read_csv(summary / "per_profile_metrics.csv")
    frame = frame[frame.method.isin(["bins40", "moped"])].copy()
    require(set(frame.method) == {"bins40", "moped"}, "Missing nine-param method")
    frame["n_train"] = config["n_train"]
    frame = validate_rows(frame, PARAMS, selected, shared["theta"], shared["low"], shared["high"])
    frame["series"] = "nine_" + frame.method
    return frame, config, marker


def load_legacy(path):
    frame = pd.read_csv(path)
    frame = frame[frame.case == "masked_baseline_noise_cross_deproj0"].copy()
    require(not frame.empty, "Missing baseline deproj0 legacy sweep")
    require((frame.analysis_target == "last_n").all(), "Not held-out profile diagnostics")
    require(set(frame.param) == set(PARAMS), "Incomplete nine-param legacy parameters")
    frame = frame.rename(columns={"theta_true": "truth", "posterior_mean": "mean", "posterior_std": "std"})
    frame["method"], frame["series"] = "bins40", "nine_legacy"
    require(not frame.duplicated(["n_train", "test_index", "param"]).any(), "Repeated legacy metrics")
    require(np.isfinite(frame[["truth", "mean", "std", "prior_low", "prior_high"]].to_numpy()).all(),
            "Invalid legacy metrics")
    require((frame["std"] > 0).all(), "Invalid legacy posterior width")
    tests = set(frame.test_index)
    truth_by_row = frame.groupby(["test_index", "param"]).truth.nunique()
    require((truth_by_row == 1).all(), "Legacy truth labels change across training sizes")
    audit = []
    for n, sub in frame.groupby("n_train"):
        require(sub.groupby("param").test_index.apply(set).apply(lambda x: x == tests).all(),
                "Legacy test populations differ across runs")
        require(sub.run_dir.nunique() == 1, "Ambiguous legacy run")
        run = Path(sub.run_dir.iloc[0])
        indices = np.load(run / "train_indices.npy", allow_pickle=False)
        require(len(indices) == n and not np.intersect1d(indices, list(tests)).size,
                f"Legacy train/test leakage: {run}")
        metadata = read_json(run / "run_metadata.json")
        audit.append(dict(n_train=int(n), n_test=len(tests), training_test_overlap=0,
                          weights_selection=metadata.get("weights_selection", "not recorded")))
    error = frame["mean"]-frame.truth
    np.testing.assert_allclose(frame.error_over_prior_range, error/(frame.prior_high-frame.prior_low),
                               rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(frame.pull, error/frame["std"], rtol=1e-6, atol=1e-7)
    return frame, audit


def statistics(frame, reference_widths):
    records = []
    for (series, n, param), sub in frame.groupby(["series", "n_train", "param"], sort=True):
        error = sub["mean"].to_numpy()-sub.truth.to_numpy()
        native_width = (sub.prior_high-sub.prior_low).to_numpy()
        require(np.all(native_width > 0), "Nonpositive prior width")
        require(np.allclose(native_width, native_width[0], rtol=1e-10), "Prior changed within a run")
        reference = reference_widths.get(param, native_width[0])
        r = (float(np.corrcoef(sub.truth, sub["mean"])[0, 1])
             if sub.truth.std() > 0 and sub["mean"].std() > 0 else np.nan)
        records.append(dict(series=series, n_train=int(n), param=param, n_test=len(sub),
            pearson_r=r, rmse=float(np.sqrt(np.mean(error**2))),
            rmse_prior_native=float(np.sqrt(np.mean((error/native_width)**2))),
            rmse_prior_common=float(np.sqrt(np.mean((error/reference)**2))),
            rmse_std=float(np.sqrt(np.mean((error/sub["std"].to_numpy())**2))),
            prior_low=float(sub.prior_low.iloc[0]), prior_high=float(sub.prior_high.iloc[0]),
            reference_width=float(reference)))
    return pd.DataFrame(records)


def save_plot(fig, output, name):
    fig.tight_layout(rect=(0, 0, 1, .94))
    for extension in ("png", "jpg", "pdf"):
        fig.savefig(output / f"{name}.{extension}", dpi=250, bbox_inches="tight")
    plt.close(fig)


def comparison_curves(stats, output):
    for key, ylabel in (("pearson_r", "Pearson correlation coefficient"),
                        ("rmse_prior_common", "RMSE / common prior range"),
                        ("rmse_std", "RMS standardized error")):
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.9))
        for ax, param in zip(axes, TARGETS):
            for series, (label, color, marker, linestyle) in STYLES.items():
                sub = stats[(stats.series == series) & (stats.param == param)].sort_values("n_train")
                if sub.empty:
                    continue
                ax.plot(sub.n_train.to_numpy(), sub[key].to_numpy(), color=color, marker=marker,
                        ls=linestyle, ms=7 if series.startswith("nine_") and series != "nine_legacy" else 3,
                        lw=1.2, label=label)
            ax.set(xscale="log", xlabel="Training dataset size (includes validation)",
                   ylabel=ylabel, title=LABELS[param])
            if key != "pearson_r" and (stats[key] > 0).all():
                ax.set_yscale("log")
            ax.grid(alpha=.2)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=7, frameon=False)
        fig.suptitle("Descriptive comparison: different test populations and nuisance parameters\n"
                     "2-param: fixed noise, 1000 tests; 9-param sweep: 100 tests; compression: 489/495 tests",
                     fontsize=8)
        fig.tight_layout(rect=(0, .18, 1, .85))
        for extension in ("png", "jpg", "pdf"):
            fig.savefig(output / f"two_vs_nine_{key}_vs_dataset_size.{extension}", dpi=250, bbox_inches="tight")
        plt.close(fig)


def nine_panels(stats, output):
    for key, ylabel in (("pearson_r", "Pearson r"), ("rmse_prior_native", "RMSE / saved prior range")):
        fig, axes = plt.subplots(3, 3, figsize=(8, 7.2))
        for ax, param in zip(axes.flat, PARAMS):
            for series in ("nine_legacy", "nine_bins40", "nine_moped"):
                sub = stats[(stats.series == series) & (stats.param == param)].sort_values("n_train")
                label, color, marker, linestyle = STYLES[series]
                if len(sub):
                    ax.plot(sub.n_train.to_numpy(), sub[key].to_numpy(), color=color,
                            marker=marker, ls=linestyle, ms=4, label=label)
            ax.set(xscale="log", title=LABELS[param], ylabel=ylabel)
            if key != "pearson_r":
                ax.set_ylim(bottom=0)
            ax.grid(alpha=.2)
        for ax in axes[-1]:
            ax.set_xlabel("Training dataset size")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=1, fontsize=7, frameon=False)
        fig.suptitle("Nine-parameter results; sweep: 100 tests, compression reference: preliminary 489/495\n"
                     "Different training recipes and test populations; not a paired comparison", fontsize=8)
        fig.tight_layout(rect=(0, .1, 1, .92))
        for extension in ("png", "jpg", "pdf"):
            fig.savefig(output / f"nine_parameter_{key}_vs_dataset_size.{extension}", dpi=250, bbox_inches="tight")
        plt.close(fig)


def true_mean_plots(frame, stats, output):
    for method in ("bins40", "moped"):
        fig, axes = plt.subplots(2, 2, figsize=(7.5, 6.4))
        for row, series in enumerate(("two_"+method, "nine_"+method)):
            selected = frame[frame.series == series]
            n = int(selected.n_train.max())
            for ax, param in zip(axes[row], TARGETS):
                sub = selected[(selected.n_train == n) & (selected.param == param)]
                stat = stats[(stats.series == series) & (stats.n_train == n) & (stats.param == param)].iloc[0]
                label, color, _, _ = STYLES[series]
                ax.scatter(sub.truth.to_numpy(), sub["mean"].to_numpy(), s=5, alpha=.3, color=color,
                           rasterized=True)
                lo, hi = sub.prior_low.iloc[0], sub.prior_high.iloc[0]
                ax.plot([lo, hi], [lo, hi], "k:", lw=.8)
                ax.set(xlabel="True "+LABELS[param], ylabel="Posterior mean "+LABELS[param],
                       title=f"{label}\nN={n:,}, n_test={len(sub)}, r={stat.pearson_r:.5f}")
                ax.grid(alpha=.2)
        fig.suptitle("Largest available training sets; different nuisance parameters and test populations", fontsize=8)
        save_plot(fig, output, f"two_vs_nine_true_vs_mean_{method}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--two-root", type=Path, required=True)
    p.add_argument("--nine-root", type=Path, required=True)
    p.add_argument("--nine-convergence-csv", type=Path)
    p.add_argument("--allow-preliminary-nine", action="store_true")
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    marker = args.output_dir / "comparison_complete.json"
    marker.unlink(missing_ok=True)
    two, two_config, two_shared, two_audit = load_two(args.two_root)
    nine, nine_config, preliminary = load_nine(args.nine_root, args.allow_preliminary_nine)
    frames, legacy_audit = [two, nine], []
    if args.nine_convergence_csv:
        legacy, legacy_audit = load_legacy(args.nine_convergence_csv)
        frames.append(legacy)
    reference = dict(zip(TARGETS, two_shared["high"]-two_shared["low"]))
    stats = statistics(pd.concat(frames, ignore_index=True), reference)
    # Independently reproduce the completed two-parameter summary from its row metrics.
    saved = pd.read_csv(args.two_root / "summary/per_parameter_summary.csv")
    saved["series"] = "two_"+saved.method
    joined = saved.merge(stats, on=["series", "n_train", "param"], validate="one_to_one", suffixes=("_saved", ""))
    for original, computed in (("pearson_r_saved", "pearson_r"), ("rmse_prior", "rmse_prior_native"), ("rmse_std_saved", "rmse_std")):
        np.testing.assert_allclose(joined[original], joined[computed], rtol=2e-6, atol=1e-8)
    all_rows = pd.concat(frames, ignore_index=True)
    stats.to_csv(args.output_dir / "comparison_metrics.csv", index=False)
    all_rows.to_csv(args.output_dir / "comparison_profile_metrics.csv", index=False)
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 8,
                         "axes.titlesize": 8, "axes.labelsize": 8, "pdf.fonttype": 42})
    comparison_curves(stats, args.output_dir)
    nine_panels(stats, args.output_dir)
    true_mean_plots(all_rows, stats, args.output_dir)
    latest = stats[stats.param.isin(TARGETS)].copy()
    latest = latest[latest.n_train == latest.groupby("series").n_train.transform("max")]
    latest.to_csv(args.output_dir / "largest_runs_comparison.csv", index=False)
    scope = dict(complete=True, new_sampling=False, common_prior_widths=reference,
        two_config=two_config, nine_config=nine_config, nine_preliminary=preliminary,
        two_run_checks=two_audit, legacy_run_checks=legacy_audit,
        source_sha256={str(path): digest(path) for path in (
            args.two_root / "summary/per_profile_metrics.csv",
            args.nine_root / "summary_preliminary/per_profile_metrics.csv",
            *([args.nine_convergence_csv] if args.nine_convergence_csv else []))},
        warnings=["Different Sobol designs and test populations: no cross-experiment row pairing.",
            "Two parameters vary with seven fixed; nine-param runs vary nuisance parameters too.",
            "Two-param fixed-noise diagnostic is not independent-noise calibration.",
            "Nine-param compression omits six difficult profiles; metrics can be optimistic.",
            "Legacy nine-param sweep has no recorded best-validation snapshot selection.",
            "Training sizes include validation; training recipes differ between experiments.",
            "Common-range normalization does not make the different training priors identical."],
        formulas=dict(rmse_prior_common="sqrt(mean_i(((posterior_mean_i-truth_i)/common_prior_width)^2))",
                      rmse_std="sqrt(mean_i(((posterior_mean_i-truth_i)/posterior_std_i)^2))",
                      pearson="Pearson correlation of truth and posterior mean across own test set"))
    marker.write_text(json.dumps(scope, indent=2, allow_nan=False)+"\n")
    print(latest[["series", "n_train", "param", "n_test", "pearson_r", "rmse_prior_common", "rmse_std"]].to_string(index=False))
    print("Saved:", args.output_dir)


if __name__ == "__main__":
    main()
