#!/usr/bin/env python3
"""Create A&A-style convergence figures for two-parameter SO NPE runs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evaluate_so_two_param_npe import (
    BATTAGLIA12,
    COLORS,
    LABELS,
    MODE_LABELS,
    PARAMS,
    paper_style,
    write_csv,
    write_json,
)
from plot_battaglia12_so_getdist import plot_getdist


MODES = ("asinh",)


DEFAULT_ROOT = Path(
    "/lustre/work/kristero10/adrian_two_param_nsf_convergence_baseline_deproj0"
)
DEFAULT_SIZES = (
    256,
    512,
    1024,
    2048,
    4096,
    8192,
    16384,
    32768,
    65536,
    98304,
    131072,
    196608,
    262144,
    327680,
    393216,
    458752,
    523288,
)
DEFAULT_CORNER_SIZES = (256, 32768, 523288)


def scalar_string(value: Any, default: str = "") -> str:
    array = np.asarray(value)
    return str(array.reshape(()).item()) if array.size == 1 else default


def parse_ints(value: str) -> list[int]:
    return [
        int(part.replace("_", ""))
        for part in str(value).replace(",", " ").split()
        if part
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--dataset-sizes",
        default=",".join(str(value) for value in DEFAULT_SIZES),
    )
    parser.add_argument(
        "--corner-sizes",
        default=",".join(str(value) for value in DEFAULT_CORNER_SIZES),
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument(
        "--expected-density-estimator",
        choices=("maf", "nsf"),
        default="nsf",
    )
    parser.add_argument("--expected-hidden-features", type=int, default=64)
    parser.add_argument("--expected-num-transforms", type=int, default=6)
    return parser.parse_args()


def load_outputs(
    run_root: Path,
    sizes: list[int],
    allow_missing: bool,
    expected_density_estimator: str,
    expected_hidden_features: int,
    expected_num_transforms: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    metric_frames: list[pd.DataFrame] = []
    summary_frames: list[pd.DataFrame] = []
    validation_rows: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for mode in MODES:
        for n_train in sizes:
            run_dir = run_root / mode / f"N{n_train}"
            completion = run_dir / "evaluation" / "evaluation_complete.json"
            metrics_path = run_dir / "evaluation" / "heldout_metrics.csv"
            summary_path = run_dir / "evaluation" / "heldout_summary.csv"
            metadata_path = run_dir / "run_metadata.json"
            required = (completion, metrics_path, summary_path, metadata_path)
            absent = [str(path) for path in required if not path.is_file()]
            if absent:
                missing.append(
                    {"mode": mode, "n_train": n_train, "missing": absent}
                )
                continue

            metrics = pd.read_csv(metrics_path)
            metrics["mode"] = mode
            metrics["n_train"] = n_train
            metric_frames.append(metrics)

            summary = pd.read_csv(summary_path)
            summary["mode"] = mode
            summary["n_train"] = n_train
            summary_frames.append(summary)

            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            actual_contract = (
                str(metadata.get("density_estimator", "")).lower(),
                int(metadata.get("hidden_features", -1)),
                int(metadata.get("num_transforms", -1)),
            )
            expected_contract = (
                str(expected_density_estimator).lower(),
                int(expected_hidden_features),
                int(expected_num_transforms),
            )
            if actual_contract != expected_contract:
                raise ValueError(
                    f"Estimator contract mismatch in {metadata_path}: "
                    f"actual={actual_contract}, expected={expected_contract}"
                )
            validation_rows.append(
                {
                    "mode": mode,
                    "n_train": n_train,
                    "best_validation_performance": metadata.get(
                        "best_validation_performance"
                    ),
                    "density_estimator": metadata.get("density_estimator"),
                    "hidden_features": metadata.get("hidden_features"),
                    "num_transforms": metadata.get("num_transforms"),
                    "internal_z_score_x": metadata.get("internal_z_score_x"),
                }
            )

    if missing and not allow_missing:
        preview = "\n".join(
            f"  {item['mode']} N={item['n_train']}: {item['missing'][0]}"
            for item in missing[:12]
        )
        raise FileNotFoundError(
            f"{len(missing)} convergence runs are incomplete. First missing outputs:\n{preview}"
        )
    if not metric_frames or not summary_frames:
        raise FileNotFoundError(f"No completed convergence evaluations under {run_root}")
    return (
        pd.concat(metric_frames, ignore_index=True),
        pd.concat(summary_frames, ignore_index=True),
        pd.DataFrame(validation_rows),
        missing,
    )


def finish_axis(axis: Any, ylabel: str, correlation: bool = False) -> None:
    axis.set_xscale("log")
    axis.set_xlabel("Training set size")
    axis.set_ylabel(ylabel)
    axis.grid(True, alpha=0.25, lw=0.5)
    if correlation:
        axis.set_ylim(0.0, 1.02)


def plot_correlation(summary: pd.DataFrame, output: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(9.0 / 2.54, 7.0 / 2.54))
    line_styles = {"P0": "-", "beta": "--"}
    markers = {"P0": "o", "beta": "s"}
    for mode in MODES:
        for param in PARAMS:
            sub = summary[(summary["mode"] == mode) & (summary["param"] == param)].sort_values("n_train")
            ax.plot(
                sub["n_train"].to_numpy(dtype=float),
                sub["pearson_r"].to_numpy(dtype=float),
                color=COLORS[mode],
                ls=line_styles[param],
                marker=markers[param],
                ms=3.0,
                lw=0.9,
                label=rf"{MODE_LABELS[mode]}, {LABELS[param]}",
            )
    finish_axis(ax, r"Pearson correlation, $r$", correlation=True)
    ax.legend(frameon=False, fontsize=6.2, ncol=2)
    fig.tight_layout()
    fig.savefig(output, dpi=dpi)
    plt.close(fig)


def plot_metric_panels(
    summary: pd.DataFrame,
    column: str,
    ylabel: str,
    output: Path,
    dpi: int,
    reference: float | None = None,
) -> None:
    panel_params = (*PARAMS, "all")
    fig, axes = plt.subplots(
        1, 3, figsize=(18.0 / 2.54, 6.2 / 2.54), sharey=False
    )
    for axis, param in zip(axes, panel_params):
        for mode in MODES:
            sub = summary[(summary["mode"] == mode) & (summary["param"] == param)].sort_values("n_train")
            if sub.empty or column not in sub:
                continue
            axis.plot(
                sub["n_train"].to_numpy(dtype=float),
                pd.to_numeric(sub[column], errors="coerce").to_numpy(dtype=float),
                marker="o",
                ms=3.0,
                lw=0.9,
                color=COLORS[mode],
                label=MODE_LABELS[mode],
            )
        if reference is not None:
            axis.axhline(reference, color="black", lw=0.7, ls="--")
        title = "Combined" if param == "all" else LABELS[param]
        axis.set_title(title)
        finish_axis(axis, ylabel)
    axes[0].legend(frameon=False, fontsize=6.5)
    fig.tight_layout()
    fig.savefig(output, dpi=dpi)
    plt.close(fig)


def plot_validation(validation: pd.DataFrame, output: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(9.0 / 2.54, 6.5 / 2.54))
    for mode in MODES:
        sub = validation[validation["mode"] == mode].sort_values("n_train")
        ax.plot(
            sub["n_train"].to_numpy(dtype=float),
            pd.to_numeric(
                sub["best_validation_performance"], errors="coerce"
            ).to_numpy(dtype=float),
            marker="o",
            ms=3.0,
            lw=0.9,
            color=COLORS[mode],
            label=MODE_LABELS[mode],
        )
    finish_axis(ax, "Best validation performance")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=dpi)
    plt.close(fig)


def selected_true_mean_plot(
    metrics: pd.DataFrame,
    mode: str,
    sizes: list[int],
    output: Path,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(
        len(PARAMS),
        len(sizes),
        figsize=(18.0 / 2.54, 10.5 / 2.54),
        squeeze=False,
    )
    artists = []
    maximum = 1.0
    for row_index, param in enumerate(PARAMS):
        for column_index, n_train in enumerate(sizes):
            axis = axes[row_index, column_index]
            sub = metrics[
                (metrics["mode"] == mode)
                & (metrics["param"] == param)
                & (metrics["n_train"] == n_train)
            ]
            if sub.empty:
                axis.set_visible(False)
                continue
            truth = sub["theta_true"].to_numpy()
            mean = sub["posterior_mean"].to_numpy()
            low = min(float(truth.min()), float(mean.min()))
            high = max(float(truth.max()), float(mean.max()))
            artist = axis.hexbin(
                truth,
                mean,
                gridsize=30,
                mincnt=1,
                cmap="viridis",
                linewidths=0,
            )
            artists.append(artist)
            if artist.get_array().size:
                maximum = max(maximum, float(artist.get_array().max()))
            axis.plot([low, high], [low, high], color="black", lw=0.7, ls="--")
            axis.set(xlim=(low, high), ylim=(low, high))
            axis.set_title(rf"$N={n_train:,}$")
            axis.set_xlabel(rf"True {LABELS[param]}")
            if column_index == 0:
                axis.set_ylabel(rf"Posterior mean {LABELS[param]}")
    for artist in artists:
        artist.set_clim(1.0, maximum)
    if artists:
        colorbar = fig.colorbar(artists[-1], ax=axes, pad=0.015)
        colorbar.set_label("Profiles per bin")
    fig.suptitle(MODE_LABELS[mode], fontsize=8)
    fig.savefig(output, dpi=dpi)
    plt.close(fig)


def load_battaglia_samples(
    run_root: Path,
    mode: str,
    sizes: list[int],
) -> list[dict[str, Any]]:
    sample_sets = []
    for n_train in sizes:
        run_dir = run_root / mode / f"N{n_train}"
        path = run_dir / "battaglia12_posterior_samples.npy"
        contract_path = run_dir / "battaglia12_conditioning_contract.npz"
        if not path.is_file():
            raise FileNotFoundError(
                f"Representative Battaglia12 samples are missing: {path}"
            )
        if not contract_path.is_file():
            raise FileNotFoundError(
                "Battaglia12 samples lack a validated conditioning contract: "
                f"{contract_path}. Refresh this corner run."
            )
        with np.load(contract_path, allow_pickle=True) as contract:
            source = scalar_string(contract["observation_source"])
            transformed = np.asarray(
                contract["transformed_observation"], dtype=np.float32
            )
        if not source.startswith("validated_baseline_deproj0:"):
            raise ValueError(
                f"Battaglia12 samples have the wrong observation source: {source!r}"
            )
        if transformed.ndim != 1 or not np.all(np.isfinite(transformed)):
            raise ValueError(f"Invalid transformed observation in {contract_path}")
        samples = np.asarray(np.load(path), dtype=np.float64)
        if samples.ndim != 2 or samples.shape[1] != len(PARAMS):
            raise ValueError(f"Invalid Battaglia12 samples: {path}, {samples.shape}")
        sample_sets.append({"label": rf"$N={n_train:,}$", "samples": samples})
    return sample_sets


def constraint_rows(
    run_root: Path,
    corner_sizes: list[int],
) -> list[dict[str, Any]]:
    rows = []
    for mode in MODES:
        for n_train, item in zip(
            corner_sizes, load_battaglia_samples(run_root, mode, corner_sizes)
        ):
            samples = item["samples"]
            for index, param in enumerate(PARAMS):
                quantiles = np.quantile(
                    samples[:, index], [0.025, 0.16, 0.5, 0.84, 0.975]
                )
                rows.append(
                    {
                        "mode": mode,
                        "n_train": n_train,
                        "param": param,
                        "truth": BATTAGLIA12[index],
                        "mean": samples[:, index].mean(),
                        "std": samples[:, index].std(ddof=1),
                        "q025": quantiles[0],
                        "q16": quantiles[1],
                        "median": quantiles[2],
                        "q84": quantiles[3],
                        "q975": quantiles[4],
                    }
                )
    return rows


def plot_sbc_cdf(
    metrics: pd.DataFrame,
    n_train: int,
    output: Path,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(
        1, 2, figsize=(18.0 / 2.54, 7.0 / 2.54), sharex=True, sharey=True
    )
    grid = np.linspace(0.0, 1.0, 401)
    for axis, param in zip(axes, PARAMS):
        n_test = len(
            metrics[
                (metrics["mode"] == MODES[0])
                & (metrics["param"] == param)
                & (metrics["n_train"] == n_train)
            ]
        )
        epsilon = math.sqrt(math.log(2.0 / 0.05) / (2.0 * n_test))
        axis.fill_between(
            grid,
            np.maximum(0.0, grid - epsilon),
            np.minimum(1.0, grid + epsilon),
            color="0.85",
            label="95% DKW band",
        )
        axis.plot(grid, grid, color="black", lw=0.7, ls="--", label="Uniform")
        for mode in MODES:
            ranks = np.sort(
                metrics[
                    (metrics["mode"] == mode)
                    & (metrics["param"] == param)
                    & (metrics["n_train"] == n_train)
                ]["rank_fraction"].to_numpy()
            )
            empirical = np.arange(1, len(ranks) + 1) / len(ranks)
            axis.step(
                ranks,
                empirical,
                where="post",
                lw=0.9,
                color=COLORS[mode],
                label=MODE_LABELS[mode],
            )
        axis.set_title(LABELS[param])
        axis.set_xlabel("SBC rank fraction")
        axis.grid(True, alpha=0.2, lw=0.5)
    axes[0].set_ylabel("Empirical cumulative density")
    axes[0].legend(frameon=False, fontsize=6.2)
    fig.tight_layout()
    fig.savefig(output, dpi=dpi)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    run_root = args.run_root.expanduser().resolve()
    output = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else run_root / "summary"
    )
    output.mkdir(parents=True, exist_ok=True)
    sizes = parse_ints(args.dataset_sizes)
    corner_sizes = parse_ints(args.corner_sizes)
    if not sizes or len(corner_sizes) != 3:
        raise ValueError("Dataset sizes must be non-empty and corner sizes must contain three values.")
    if any(value not in sizes for value in corner_sizes):
        raise ValueError(f"Corner sizes {corner_sizes} must be present in {sizes}.")

    paper_style()
    metrics, summary, validation, missing = load_outputs(
        run_root,
        sizes,
        args.allow_missing,
        args.expected_density_estimator,
        args.expected_hidden_features,
        args.expected_num_transforms,
    )
    metrics.to_csv(output / "heldout_metrics_all_runs.csv", index=False)
    summary.to_csv(output / "convergence_metrics.csv", index=False)
    validation.to_csv(output / "validation_performance.csv", index=False)

    plot_correlation(
        summary, output / "correlation_vs_dataset_size.jpg", args.dpi
    )
    plot_metric_panels(
        summary,
        "rmse_over_prior",
        r"RMSE / prior width",
        output / "prior_normalized_rmse_vs_dataset_size.jpg",
        args.dpi,
    )
    plot_metric_panels(
        summary,
        "pull_rmse",
        r"RMSE / posterior std",
        output / "rmse_over_posterior_std_vs_dataset_size.jpg",
        args.dpi,
        reference=1.0,
    )
    plot_metric_panels(
        summary,
        "mean_posterior_std_over_prior",
        r"Mean posterior std / prior width",
        output / "posterior_std_over_prior_vs_dataset_size.jpg",
        args.dpi,
    )
    plot_validation(
        validation, output / "validation_performance_vs_dataset_size.jpg", args.dpi
    )

    skipped_optional_outputs = []
    for mode in MODES:
        selected_true_mean_plot(
            metrics,
            mode,
            corner_sizes,
            output / f"true_vs_mean_min_mid_max_{mode}.jpg",
            args.dpi,
        )
        try:
            battaglia_sample_sets = load_battaglia_samples(
                run_root, mode, corner_sizes
            )
        except (FileNotFoundError, ValueError) as exc:
            if not args.allow_missing:
                raise
            message = f"Skipped optional Battaglia12 corner for {mode}: {exc}"
            skipped_optional_outputs.append(message)
            print(f"WARNING: {message}")
        else:
            plot_getdist(
                battaglia_sample_sets,
                list(PARAMS),
                BATTAGLIA12,
                output / f"battaglia12_P0_beta_corner_min_mid_max_{mode}.jpg",
                filled_last_only=True,
                dpi=args.dpi,
            )

    plot_sbc_cdf(
        metrics,
        max(sizes),
        output / "sbc_rank_cdf_max_dataset_size.jpg",
        args.dpi,
    )

    try:
        constraints = constraint_rows(run_root, corner_sizes)
    except (FileNotFoundError, ValueError) as exc:
        if not args.allow_missing:
            raise
        message = f"Skipped optional Battaglia12 constraint table: {exc}"
        skipped_optional_outputs.append(message)
        print(f"WARNING: {message}")
    else:
        write_csv(output / "battaglia12_constraints_min_mid_max.csv", constraints)
    write_json(
        output / "convergence_summary.json",
        {
            "run_root": run_root,
            "dataset_sizes": sizes,
            "corner_sizes": corner_sizes,
            "modes": MODES,
            "target_parameters": PARAMS,
            "density_estimator": args.expected_density_estimator,
            "hidden_features": args.expected_hidden_features,
            "num_transforms": args.expected_num_transforms,
            "internal_z_score_x": "none",
            "heldout_last_n": 1000,
            "missing_runs": missing,
            "skipped_optional_outputs": skipped_optional_outputs,
            "primary_rmse_definition": (
                "sqrt(mean(((posterior_mean - theta_true) / prior_width)^2))"
            ),
        },
    )
    print(
        f"Completed two-parameter "
        f"{args.expected_density_estimator.upper()} convergence summary: "
        f"{output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
