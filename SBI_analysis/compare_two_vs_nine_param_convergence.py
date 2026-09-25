#!/usr/bin/env python3
"""Compare two-parameter and nine-parameter SBI convergence diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CASE = "masked_baseline_noise_cross_deproj0"
TARGET_PARAMS = ("P0", "beta")
PARAM_LABELS = {"P0": r"$P_0$", "beta": r"$\beta$"}
ARCHITECTURE_LABELS = {
    "two": "2-parameter NPE",
    "nine": "9-parameter NPE",
}
COLORS = {"two": "#0072B2", "nine": "#D55E00"}
MARKERS = {"two": "o", "nine": "s"}


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_two_metrics() -> Path:
    return (
        project_root()
        / "SBI_analysis"
        / "outputs"
        / "2_param_analysis"
        / "maf"
        / "heldout_metrics_all_runs.csv"
    )


def default_nine_metrics() -> Path:
    return (
        project_root()
        / "SBI_analysis"
        / "convergence_tests"
        / "last100_param_metrics.csv"
    )


def default_output_dir() -> Path:
    return (
        project_root()
        / "SBI_analysis"
        / "outputs"
        / "2_param_analysis"
        / "maf"
        / "comparison_with_9_parameter"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--two-param-metrics",
        type=Path,
        default=default_two_metrics(),
        help="Per-profile metrics from the two-parameter evaluations.",
    )
    parser.add_argument(
        "--nine-param-metrics",
        type=Path,
        default=default_nine_metrics(),
        help="Per-parameter metrics from the nine-parameter evaluations.",
    )
    parser.add_argument("--case", default=CASE)
    parser.add_argument(
        "--holdout-last-n",
        type=int,
        default=100,
        help="Use the final N profile indices shared by both analyses.",
    )
    parser.add_argument(
        "--true-mean-n-train",
        type=int,
        default=None,
        help="Training size for true-vs-mean panels; default is largest shared size.",
    )
    parser.add_argument("--output-dir", type=Path, default=default_output_dir())
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("jpg", "png", "pdf"),
        default=("jpg", "png"),
    )
    return parser.parse_args()


def require_columns(frame: pd.DataFrame, required: Iterable[str], source: Path) -> None:
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"{source} lacks required columns: {sorted(missing)}")


def numeric(frame: pd.DataFrame, columns: Iterable[str], source: Path) -> None:
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if not np.isfinite(frame[column].to_numpy(dtype=float)).all():
            raise ValueError(f"{source} contains non-finite values in {column!r}")


def load_two_parameter_metrics(path: Path) -> pd.DataFrame:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    require_columns(
        frame,
        (
            "n_train",
            "dataset_index",
            "param",
            "theta_true",
            "posterior_mean",
            "posterior_std",
            "normalized_error_prior",
            "pull",
        ),
        path,
    )
    frame = frame[frame["param"].astype(str).isin(TARGET_PARAMS)].copy()
    frame["architecture"] = "two"
    frame["dataset_index"] = pd.to_numeric(
        frame["dataset_index"], errors="raise"
    ).astype(np.int64)
    frame["n_train"] = pd.to_numeric(frame["n_train"], errors="raise").astype(
        np.int64
    )
    numeric(
        frame,
        (
            "theta_true",
            "posterior_mean",
            "posterior_std",
            "normalized_error_prior",
            "pull",
        ),
        path,
    )
    return frame[
        [
            "architecture",
            "n_train",
            "dataset_index",
            "param",
            "theta_true",
            "posterior_mean",
            "posterior_std",
            "normalized_error_prior",
            "pull",
        ]
    ]


def load_nine_parameter_metrics(path: Path, case: str) -> pd.DataFrame:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    require_columns(
        frame,
        (
            "n_train",
            "test_index",
            "param",
            "theta_true",
            "posterior_mean",
            "posterior_std",
            "error_over_prior_range",
            "pull",
        ),
        path,
    )
    if "case" in frame.columns:
        frame = frame[frame["case"].astype(str) == case].copy()
        if frame.empty:
            raise ValueError(f"No rows for case={case!r} in {path}")
    frame["architecture"] = "nine"
    frame["dataset_index"] = pd.to_numeric(
        frame["test_index"], errors="raise"
    ).astype(np.int64)
    frame["n_train"] = pd.to_numeric(frame["n_train"], errors="raise").astype(
        np.int64
    )
    frame["normalized_error_prior"] = frame["error_over_prior_range"]
    numeric(
        frame,
        (
            "theta_true",
            "posterior_mean",
            "posterior_std",
            "normalized_error_prior",
            "pull",
        ),
        path,
    )
    return frame[
        [
            "architecture",
            "n_train",
            "dataset_index",
            "param",
            "theta_true",
            "posterior_mean",
            "posterior_std",
            "normalized_error_prior",
            "pull",
        ]
    ]


def complete_indices(frame: pd.DataFrame, params: Sequence[str]) -> Set[int]:
    selected = frame[frame["param"].isin(params)]
    if selected.empty:
        return set()
    counts = selected.groupby(["n_train", "dataset_index"])["param"].nunique()
    complete = counts[counts == len(params)].reset_index()
    index_sets = [
        set(group["dataset_index"].astype(int))
        for _, group in complete.groupby("n_train")
    ]
    if not index_sets:
        return set()
    return set.intersection(*index_sets)


def validate_selected_rows(
    frame: pd.DataFrame,
    selected_indices: Sequence[int],
    required_params: Sequence[str],
    label: str,
) -> None:
    selected = frame[
        frame["dataset_index"].isin(selected_indices)
        & frame["param"].isin(required_params)
    ]
    duplicated = selected.duplicated(["n_train", "dataset_index", "param"])
    if duplicated.any():
        example = selected.loc[
            duplicated, ["n_train", "dataset_index", "param"]
        ].iloc[0]
        raise ValueError(f"Duplicate {label} metric row: {example.to_dict()}")

    expected = len(selected_indices) * len(required_params)
    counts = selected.groupby("n_train").size()
    bad = counts[counts != expected]
    if not bad.empty:
        raise ValueError(
            f"Incomplete {label} rows after profile matching: {bad.to_dict()}, "
            f"expected {expected} rows per training size"
        )

    truth_spread = (
        selected.groupby(["dataset_index", "param"])["theta_true"]
        .agg(lambda values: float(np.max(values) - np.min(values)))
        .max()
    )
    if float(truth_spread) > 1.0e-10:
        raise ValueError(
            f"{label} theta_true changes across training sizes; "
            f"maximum spread={truth_spread}"
        )


def validate_cross_analysis_truth(
    two: pd.DataFrame,
    nine: pd.DataFrame,
    selected_indices: Sequence[int],
) -> float:
    def reference(frame: pd.DataFrame) -> pd.DataFrame:
        return (
            frame[
                frame["dataset_index"].isin(selected_indices)
                & frame["param"].isin(TARGET_PARAMS)
            ][["dataset_index", "param", "theta_true"]]
            .drop_duplicates(["dataset_index", "param"])
        )

    joined = reference(two).merge(
        reference(nine),
        on=["dataset_index", "param"],
        suffixes=("_two", "_nine"),
        validate="one_to_one",
    )
    expected = len(selected_indices) * len(TARGET_PARAMS)
    if len(joined) != expected:
        raise ValueError(
            f"Truth-alignment comparison has {len(joined)} rows; expected {expected}"
        )
    difference = np.abs(
        joined["theta_true_two"].to_numpy(dtype=float)
        - joined["theta_true_nine"].to_numpy(dtype=float)
    )
    maximum = float(difference.max(initial=0.0))
    if not np.allclose(
        joined["theta_true_two"].to_numpy(dtype=float),
        joined["theta_true_nine"].to_numpy(dtype=float),
        rtol=1.0e-7,
        atol=1.0e-10,
    ):
        raise ValueError(
            "Two- and nine-parameter analyses use different theta_true values; "
            f"maximum absolute difference={maximum}"
        )
    return maximum


def pearson_r(truth: np.ndarray, mean: np.ndarray) -> float:
    if len(truth) < 2 or np.std(truth) == 0.0 or np.std(mean) == 0.0:
        return float("nan")
    return float(np.corrcoef(truth, mean)[0, 1])


def mean_profile_rms(subset: pd.DataFrame, column: str) -> float:
    """RMS over parameters per profile, followed by the test-profile mean."""
    squared = subset.assign(_squared=subset[column].to_numpy(dtype=float) ** 2)
    per_profile = np.sqrt(
        squared.groupby("dataset_index", sort=False)["_squared"].mean()
    )
    return float(per_profile.mean())


def metric_row(
    subset: pd.DataFrame,
    architecture: str,
    n_train: int,
    scope: str,
) -> Dict[str, object]:
    unique_profiles = int(subset["dataset_index"].nunique())
    unique_params = int(subset["param"].nunique())
    row: Dict[str, object] = {
        "architecture": architecture,
        "architecture_label": ARCHITECTURE_LABELS[architecture],
        "n_train": int(n_train),
        "scope": scope,
        "n_test": unique_profiles,
        "n_parameters_in_metric": unique_params,
        "rmse_over_prior_range": mean_profile_rms(
            subset, "normalized_error_prior"
        ),
        "rmse_over_posterior_std": mean_profile_rms(subset, "pull"),
        "pearson_r": np.nan,
    }
    if scope in TARGET_PARAMS:
        row["pearson_r"] = pearson_r(
            subset["theta_true"].to_numpy(dtype=float),
            subset["posterior_mean"].to_numpy(dtype=float),
        )
    return row


def summarize_metrics(
    two: pd.DataFrame,
    nine: pd.DataFrame,
    selected_indices: Sequence[int],
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for architecture, frame in (("two", two), ("nine", nine)):
        selected = frame[frame["dataset_index"].isin(selected_indices)]
        for n_train, run in selected.groupby("n_train", sort=True):
            for param in TARGET_PARAMS:
                rows.append(
                    metric_row(
                        run[run["param"] == param],
                        architecture,
                        int(n_train),
                        param,
                    )
                )
            rows.append(
                metric_row(
                    run[run["param"].isin(TARGET_PARAMS)],
                    architecture,
                    int(n_train),
                    "P0_beta",
                )
            )
            if architecture == "nine":
                rows.append(metric_row(run, architecture, int(n_train), "all_9"))
    return pd.DataFrame(rows).sort_values(
        ["architecture", "n_train", "scope"]
    ).reset_index(drop=True)


def paper_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "Computer Modern Roman",
                "Times New Roman",
                "DejaVu Serif",
            ],
            "mathtext.fontset": "cm",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.0,
            "savefig.bbox": "tight",
        }
    )


def save_figure(
    figure: plt.Figure,
    stem: Path,
    formats: Sequence[str],
    dpi: int,
) -> None:
    for suffix in formats:
        path = stem.with_suffix("." + suffix)
        figure.savefig(path, dpi=dpi)
        print(f"Saved {path}")
    plt.close(figure)


def finish_convergence_axis(axis: plt.Axes, ylabel: str) -> None:
    axis.set_xscale("log")
    axis.set_xlabel("Training set size")
    axis.set_ylabel(ylabel)
    axis.grid(True, which="both", alpha=0.25, lw=0.5)


def plot_correlation(
    summary: pd.DataFrame,
    output_dir: Path,
    formats: Sequence[str],
    dpi: int,
) -> None:
    figure, axes = plt.subplots(
        1, 2, figsize=(18.0 / 2.54, 6.4 / 2.54), sharey=True
    )
    for axis, param in zip(axes, TARGET_PARAMS):
        for architecture in ("two", "nine"):
            selected = summary[
                (summary["architecture"] == architecture)
                & (summary["scope"] == param)
            ].sort_values("n_train")
            axis.plot(
                selected["n_train"].to_numpy(dtype=float),
                selected["pearson_r"].to_numpy(dtype=float),
                color=COLORS[architecture],
                marker=MARKERS[architecture],
                ms=3.5,
                label=ARCHITECTURE_LABELS[architecture],
            )
        axis.set_title(PARAM_LABELS[param])
        finish_convergence_axis(axis, r"Pearson correlation, $r$")
        axis.set_ylim(0.0, 1.02)
    axes[0].legend(frameon=False)
    figure.tight_layout()
    save_figure(
        figure,
        output_dir / "pearson_r_P0_beta_two_vs_nine",
        formats,
        dpi,
    )


def plot_metric(
    summary: pd.DataFrame,
    column: str,
    ylabel: str,
    stem: str,
    output_dir: Path,
    formats: Sequence[str],
    dpi: int,
    reference: Optional[float] = None,
) -> None:
    figure, axis = plt.subplots(figsize=(9.2 / 2.54, 6.8 / 2.54))
    for architecture in ("two", "nine"):
        selected = summary[
            (summary["architecture"] == architecture)
            & (summary["scope"] == "P0_beta")
        ].sort_values("n_train")
        axis.plot(
            selected["n_train"].to_numpy(dtype=float),
            selected[column].to_numpy(dtype=float),
            color=COLORS[architecture],
            marker=MARKERS[architecture],
            ms=3.5,
            label=ARCHITECTURE_LABELS[architecture] + r", $P_0+\beta$",
        )

    all_nine = summary[
        (summary["architecture"] == "nine")
        & (summary["scope"] == "all_9")
    ].sort_values("n_train")
    axis.plot(
        all_nine["n_train"].to_numpy(dtype=float),
        all_nine[column].to_numpy(dtype=float),
        color="0.25",
        marker="^",
        ms=3.0,
        ls=":",
        label="9-parameter NPE, all nine",
    )
    if reference is not None:
        axis.axhline(reference, color="0.35", lw=0.7, ls="--")
    finish_convergence_axis(axis, ylabel)
    axis.set_ylim(bottom=0.0)
    axis.legend(frameon=False, fontsize=6.4)
    figure.tight_layout()
    save_figure(figure, output_dir / stem, formats, dpi)


def choose_true_mean_size(
    two: pd.DataFrame,
    nine: pd.DataFrame,
    requested: Optional[int],
) -> int:
    two_sizes = set(two["n_train"].astype(int))
    nine_sizes = set(nine["n_train"].astype(int))
    shared = sorted(two_sizes & nine_sizes)
    if not shared:
        raise ValueError("There are no shared training sizes for true-vs-mean plots")
    if requested is None:
        return shared[-1]
    if requested not in shared:
        raise ValueError(
            f"--true-mean-n-train={requested} is not shared. "
            f"Available shared sizes: {shared}"
        )
    return requested


def plot_true_vs_mean(
    two: pd.DataFrame,
    nine: pd.DataFrame,
    selected_indices: Sequence[int],
    n_train: int,
    output_dir: Path,
    formats: Sequence[str],
    dpi: int,
) -> None:
    frames = {"two": two, "nine": nine}
    figure, axes = plt.subplots(
        2, 2, figsize=(12.0 / 2.54, 11.0 / 2.54), squeeze=False
    )
    for row, param in enumerate(TARGET_PARAMS):
        panel_data = []
        for architecture in ("two", "nine"):
            selected = frames[architecture][
                (frames[architecture]["n_train"] == n_train)
                & (frames[architecture]["param"] == param)
                & (frames[architecture]["dataset_index"].isin(selected_indices))
            ]
            panel_data.append(selected)
        combined_truth = np.concatenate(
            [item["theta_true"].to_numpy(dtype=float) for item in panel_data]
        )
        combined_mean = np.concatenate(
            [item["posterior_mean"].to_numpy(dtype=float) for item in panel_data]
        )
        low = float(min(combined_truth.min(), combined_mean.min()))
        high = float(max(combined_truth.max(), combined_mean.max()))
        padding = 0.04 * (high - low) if high > low else 1.0
        limits = (low - padding, high + padding)

        for column, (architecture, selected) in enumerate(
            zip(("two", "nine"), panel_data)
        ):
            axis = axes[row, column]
            truth = selected["theta_true"].to_numpy(dtype=float)
            mean = selected["posterior_mean"].to_numpy(dtype=float)
            correlation = pearson_r(truth, mean)
            axis.scatter(
                truth,
                mean,
                s=9,
                alpha=0.55,
                color=COLORS[architecture],
                edgecolors="none",
            )
            axis.plot(limits, limits, color="black", lw=0.8, ls="--")
            axis.set_xlim(limits)
            axis.set_ylim(limits)
            axis.set_aspect("equal", adjustable="box")
            axis.grid(True, alpha=0.2, lw=0.5)
            axis.text(
                0.04,
                0.95,
                rf"$r={correlation:.3f}$",
                transform=axis.transAxes,
                ha="left",
                va="top",
                fontsize=7,
            )
            if row == 0:
                axis.set_title(ARCHITECTURE_LABELS[architecture])
            if column == 0:
                axis.set_ylabel(rf"Posterior mean {PARAM_LABELS[param]}")
            if row == len(TARGET_PARAMS) - 1:
                axis.set_xlabel(rf"True {PARAM_LABELS[param]}")
    figure.suptitle(
        rf"Matched held-out profiles, $N_{{\rm train}}={n_train:,}$",
        fontsize=8,
    )
    figure.tight_layout()
    save_figure(
        figure,
        output_dir / f"true_vs_mean_P0_beta_N{n_train}_two_vs_nine",
        formats,
        dpi,
    )


def main() -> int:
    args = parse_args()
    if args.holdout_last_n <= 1:
        raise ValueError("--holdout-last-n must exceed one")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    two = load_two_parameter_metrics(args.two_param_metrics)
    nine = load_nine_parameter_metrics(args.nine_param_metrics, args.case)

    shared_indices = sorted(
        complete_indices(two, TARGET_PARAMS)
        & complete_indices(nine, TARGET_PARAMS)
    )
    if len(shared_indices) < args.holdout_last_n:
        raise ValueError(
            f"Only {len(shared_indices)} complete profile indices are shared; "
            f"{args.holdout_last_n} were requested"
        )
    selected_indices = shared_indices[-args.holdout_last_n :]

    validate_selected_rows(two, selected_indices, TARGET_PARAMS, "two-parameter")
    validate_selected_rows(nine, selected_indices, TARGET_PARAMS, "nine-parameter")
    nine_params = tuple(sorted(nine["param"].astype(str).unique()))
    validate_selected_rows(nine, selected_indices, nine_params, "nine-parameter/all")
    maximum_truth_difference = validate_cross_analysis_truth(
        two, nine, selected_indices
    )

    two_selected = two[two["dataset_index"].isin(selected_indices)].copy()
    nine_selected = nine[nine["dataset_index"].isin(selected_indices)].copy()
    combined = pd.concat((two_selected, nine_selected), ignore_index=True)
    summary = summarize_metrics(two_selected, nine_selected, selected_indices)
    true_mean_n_train = choose_true_mean_size(
        two_selected, nine_selected, args.true_mean_n_train
    )

    combined.to_csv(output_dir / "matched_profile_metrics.csv", index=False)
    summary.to_csv(output_dir / "two_vs_nine_summary_metrics.csv", index=False)
    np.save(
        output_dir / "matched_dataset_indices.npy",
        np.asarray(selected_indices, dtype=np.int64),
    )

    paper_style()
    plot_correlation(summary, output_dir, args.formats, args.dpi)
    plot_metric(
        summary,
        "rmse_over_prior_range",
        r"$\left\langle\sqrt{\left\langle"
        r"[(\bar{\theta}-\theta_{\rm true})/"
        r"\Delta\theta_{\rm prior}]^2"
        r"\right\rangle_{\theta}}\right\rangle_{\rm test}$",
        "rmse_over_prior_range_two_vs_nine",
        output_dir,
        args.formats,
        args.dpi,
    )
    plot_metric(
        summary,
        "rmse_over_posterior_std",
        r"$\left\langle\sqrt{\left\langle"
        r"[(\bar{\theta}-\theta_{\rm true})/"
        r"\sigma_{\rm post}]^2"
        r"\right\rangle_{\theta}}\right\rangle_{\rm test}$",
        "rmse_over_posterior_std_two_vs_nine",
        output_dir,
        args.formats,
        args.dpi,
        reference=1.0,
    )
    plot_true_vs_mean(
        two_selected,
        nine_selected,
        selected_indices,
        true_mean_n_train,
        output_dir,
        args.formats,
        args.dpi,
    )

    provenance = {
        "two_parameter_metrics": str(args.two_param_metrics.expanduser().resolve()),
        "nine_parameter_metrics": str(
            args.nine_param_metrics.expanduser().resolve()
        ),
        "case": args.case,
        "selected_profile_count": len(selected_indices),
        "selected_profile_index_min": int(min(selected_indices)),
        "selected_profile_index_max": int(max(selected_indices)),
        "maximum_cross_analysis_theta_true_difference": maximum_truth_difference,
        "two_parameter_training_sizes": sorted(
            two_selected["n_train"].astype(int).unique().tolist()
        ),
        "nine_parameter_training_sizes": sorted(
            nine_selected["n_train"].astype(int).unique().tolist()
        ),
        "true_vs_mean_training_size": true_mean_n_train,
        "metric_definitions": {
            "pearson_r": "Pearson correlation across matched held-out profiles.",
            "rmse_over_prior_range": (
                "For each profile, sqrt(mean over parameters in scope of "
                "((posterior_mean-theta_true)/prior_width)^2), followed by "
                "the mean over matched test profiles."
            ),
            "rmse_over_posterior_std": (
                "For each profile, sqrt(mean over parameters in scope of "
                "((posterior_mean-theta_true)/posterior_std)^2), followed by "
                "the mean over matched test profiles. This is never "
                "mean(RMSE)/mean(posterior_std)."
            ),
            "P0_beta": "Like-for-like pooled metric over P0 and beta.",
            "all_9": "Context curve pooling all nine inferred parameters.",
        },
    }
    (output_dir / "comparison_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"Matched profile indices: {selected_indices[0]}..{selected_indices[-1]}")
    print(f"Maximum theta_true difference: {maximum_truth_difference:.3e}")
    print(f"True-vs-mean training size: {true_mean_n_train}")
    print(f"Comparison outputs: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
