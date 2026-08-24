#!/usr/bin/env python3
"""Evaluate and compare raw and asinh P0/beta NPEs on the last 1000 rows."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import chisquare, kstest, pearsonr, spearmanr

from plot_battaglia12_so_getdist import plot_getdist
from run_so_sbc import (
    apply_x_transform,
    load_posterior,
    load_x_transform,
    sample_posterior_at_x,
)

DEFAULT_DATASET = Path(
    "/home/kristero10/HalfDome_kSZ/SBI_analysis/data_for_cluster/"
    "adrian_so_sbi_cases_ell80_7979_dataset_row_sobolrow/"
    "so_masked_baseline_noise_cross_deproj0_ell80_7979_sbi_run.npz"
)
DEFAULT_RUN_ROOT = Path(
    "/lustre/work/kristero10/adrian_two_param_npe_baseline_deproj0"
)
PARAMS = ("P0", "beta")
LABELS = {"P0": r"$P_0$", "beta": r"$\beta$"}
MODES = ("none", "asinh")
MODE_LABELS = {"none": "No transformation", "asinh": r"$\mathrm{asinh}(x/s)$"}
COLORS = {"none": "#1f77b4", "asinh": "#d62728"}
BATTAGLIA12 = np.array([18.1, 4.35], dtype=float)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare raw and asinh two-parameter NPEs on held-out rows."
    )
    parser.add_argument("--prepared-dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--last-n-test", type=int, default=1000)
    parser.add_argument("--num-posterior-samples", type=int, default=2000)
    parser.add_argument("--rank-bins", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--reuse-samples", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def paper_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
        }
    )


def load_data(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=True) as data:
        theta = np.asarray(data["theta"], dtype=np.float64)
        x = np.asarray(data["x"], dtype=np.float32)
        names = [str(value) for value in data["param_names"]]
        low = np.asarray(data["prior_low"], dtype=np.float64)
        high = np.asarray(data["prior_high"], dtype=np.float64)
    if theta.ndim != 2 or x.ndim != 2 or theta.shape[0] != x.shape[0]:
        raise ValueError(f"Invalid theta/x shapes: {theta.shape}, {x.shape}")
    indices = np.array([names.index(name) for name in PARAMS], dtype=np.int64)
    return {
        "theta": theta,
        "x": x,
        "names": names,
        "indices": indices,
        "low": low[indices],
        "high": high[indices],
    }


def validate_models(
    run_root: Path,
    target_indices: np.ndarray,
    train: np.ndarray,
    heldout: np.ndarray,
) -> dict[str, Any]:
    report = {}
    for mode in MODES:
        run_dir = run_root / mode
        metadata_path = run_dir / "run_metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(metadata_path)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        saved_train = np.asarray(np.load(run_dir / "train_indices.npy"), dtype=np.int64)
        saved_test = np.asarray(np.load(run_dir / "heldout_test_indices.npy"), dtype=np.int64)
        saved_targets = np.asarray(np.load(run_dir / "target_param_indices.npy"), dtype=np.int64)
        transform = load_x_transform(run_dir)
        transform_train = np.asarray(transform.get("train_indices", []), dtype=np.int64)
        checks = {
            "mode": metadata.get("x_rescale_mode") == mode,
            "target_names": metadata.get("target_param_names") == list(PARAMS),
            "target_indices": np.array_equal(saved_targets, target_indices),
            "train_indices": np.array_equal(saved_train, train),
            "heldout_indices": np.array_equal(saved_test, heldout),
            "transform_train_indices": np.array_equal(transform_train, train),
            "overlap_count": int(np.intersect1d(saved_train, saved_test).size),
            "density_estimator": (run_dir / "density_estimator.pkl").is_file(),
            "posterior": (run_dir / "posterior.pkl").is_file(),
        }
        failed = [
            key
            for key, value in checks.items()
            if (key == "overlap_count" and value != 0)
            or (key != "overlap_count" and not value)
        ]
        if failed:
            raise ValueError(f"{mode} validation failed: {failed}; {checks}")
        report[mode] = checks
    if not np.array_equal(
        np.load(run_root / "none" / "train_indices.npy"),
        np.load(run_root / "asinh" / "train_indices.npy"),
    ):
        raise ValueError("The two transformations used different training rows")
    return report


def heldout_samples(
    mode: str,
    run_dir: Path,
    x_eval: np.ndarray,
    output_path: Path,
    count: int,
    seed: int,
    device: str,
    reuse: bool,
) -> np.ndarray:
    shape = (x_eval.shape[0], count, len(PARAMS))
    if reuse and output_path.is_file():
        samples = np.load(output_path, mmap_mode="r")
        if samples.shape != shape:
            raise ValueError(f"Cannot reuse {output_path}: {samples.shape} != {shape}")
        print(f"Reusing {output_path}")
        return samples

    posterior = load_posterior(run_dir)
    transform = load_x_transform(run_dir)
    np.random.seed(seed)
    torch.manual_seed(seed)
    output = np.lib.format.open_memmap(
        output_path, mode="w+", dtype=np.float32, shape=shape
    )
    for index, x_obs in enumerate(x_eval):
        samples = sample_posterior_at_x(
            posterior,
            apply_x_transform(x_obs, transform),
            count,
            device,
        )
        if samples.shape != (count, len(PARAMS)):
            raise ValueError(
                f"{mode} returned {samples.shape} at held-out row {index}; "
                f"expected {(count, len(PARAMS))}"
            )
        output[index] = samples.astype(np.float32)
        if index == 0 or (index + 1) % 10 == 0:
            print(f"{mode}: sampled {index + 1}/{x_eval.shape[0]}", flush=True)
    output.flush()
    del output, posterior
    return np.load(output_path, mmap_mode="r")


def metric_rows(
    mode: str,
    samples: np.ndarray,
    truth: np.ndarray,
    dataset_indices: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
) -> list[dict[str, Any]]:
    width = high - low
    rows = []
    for local_index, dataset_index in enumerate(dataset_indices):
        current = np.asarray(samples[local_index], dtype=np.float64)
        means = current.mean(axis=0)
        stds = current.std(axis=0, ddof=1)
        qs = np.quantile(current, [0.025, 0.16, 0.5, 0.84, 0.975], axis=0)
        for param_index, param in enumerate(PARAMS):
            theta_true = float(truth[local_index, param_index])
            mean = float(means[param_index])
            std = float(stds[param_index])
            error = mean - theta_true
            rank = int(np.count_nonzero(current[:, param_index] < theta_true))
            rows.append(
                {
                    "mode": mode,
                    "test_local_index": local_index,
                    "dataset_index": int(dataset_index),
                    "param": param,
                    "param_index": param_index,
                    "theta_true": theta_true,
                    "posterior_mean": mean,
                    "posterior_median": float(qs[2, param_index]),
                    "posterior_std": std,
                    "error": error,
                    "normalized_error_prior": error / width[param_index],
                    "pull": error / std,
                    "posterior_std_over_prior": std / width[param_index],
                    "q025": float(qs[0, param_index]),
                    "q16": float(qs[1, param_index]),
                    "q84": float(qs[3, param_index]),
                    "q975": float(qs[4, param_index]),
                    "covered_68": int(qs[1, param_index] <= theta_true <= qs[3, param_index]),
                    "covered_95": int(qs[0, param_index] <= theta_true <= qs[4, param_index]),
                    "rank": rank,
                    "rank_fraction": (rank + 0.5) / (current.shape[0] + 1.0),
                    "num_posterior_samples": current.shape[0],
                }
            )
    return rows


def summaries(frame: pd.DataFrame, bins: int) -> list[dict[str, Any]]:
    rows = []
    for (mode, param), sub in frame.groupby(["mode", "param"], sort=True):
        error = sub["error"].to_numpy()
        normalized = sub["normalized_error_prior"].to_numpy()
        pull = sub["pull"].to_numpy()
        ranks = sub["rank_fraction"].to_numpy()
        counts, _ = np.histogram(ranks, bins=bins, range=(0.0, 1.0))
        rows.append(
            {
                "mode": mode,
                "param": param,
                "n_test": len(sub),
                "bias": error.mean(),
                "bias_over_prior": normalized.mean(),
                "rmse": np.sqrt(np.mean(error**2)),
                "rmse_over_prior": np.sqrt(np.mean(normalized**2)),
                "mean_posterior_std_over_prior": sub["posterior_std_over_prior"].mean(),
                "pull_mean": pull.mean(),
                "pull_rmse": np.sqrt(np.mean(pull**2)),
                "coverage_68": sub["covered_68"].mean(),
                "coverage_95": sub["covered_95"].mean(),
                "pearson_r": pearsonr(sub["theta_true"], sub["posterior_mean"])[0],
                "spearman_r": spearmanr(sub["theta_true"], sub["posterior_mean"])[0],
                "sbc_rank_mean": ranks.mean(),
                "sbc_ks_pvalue": kstest(ranks, "uniform").pvalue,
                "sbc_hist_chi2_pvalue": chisquare(counts).pvalue,
            }
        )
    for mode, sub in frame.groupby("mode", sort=True):
        normalized = sub["normalized_error_prior"].to_numpy()
        pull = sub["pull"].to_numpy()
        rows.append(
            {
                "mode": mode,
                "param": "all",
                "n_test": len(sub),
                "bias_over_prior": normalized.mean(),
                "rmse_over_prior": np.sqrt(np.mean(normalized**2)),
                "mean_posterior_std_over_prior": sub["posterior_std_over_prior"].mean(),
                "pull_mean": pull.mean(),
                "pull_rmse": np.sqrt(np.mean(pull**2)),
                "coverage_68": sub["covered_68"].mean(),
                "coverage_95": sub["covered_95"].mean(),
            }
        )
    return [{key: jsonable(value) for key, value in row.items()} for row in rows]


def plot_correlations(frame: pd.DataFrame, output: Path, dpi: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(18.0 / 2.54, 8.0 / 2.54))
    for ax, param in zip(axes, PARAMS):
        values = []
        for mode in MODES:
            sub = frame[(frame["mode"] == mode) & (frame["param"] == param)]
            true = sub["theta_true"].to_numpy()
            mean = sub["posterior_mean"].to_numpy()
            values.extend([true, mean])
            correlation = pearsonr(true, mean)[0]
            ax.scatter(
                true,
                mean,
                s=7,
                alpha=0.22,
                linewidths=0,
                color=COLORS[mode],
                label=rf"{MODE_LABELS[mode]}, $r={correlation:.3f}$",
            )
        lo = min(np.min(value) for value in values)
        hi = max(np.max(value) for value in values)
        ax.plot([lo, hi], [lo, hi], color="black", lw=0.8, ls="--")
        ax.set(xlim=(lo, hi), ylim=(lo, hi))
        ax.set_xlabel(rf"True {LABELS[param]}")
        ax.set_ylabel(rf"Posterior mean {LABELS[param]}")
        ax.grid(True, alpha=0.25, lw=0.5)
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=dpi)
    plt.close(fig)


def plot_density_correlations(frame: pd.DataFrame, output: Path, dpi: int) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(18.0 / 2.54, 12.0 / 2.54))
    artists = []
    max_count = 1.0
    for row_index, mode in enumerate(MODES):
        for column_index, param in enumerate(PARAMS):
            ax = axes[row_index, column_index]
            sub = frame[(frame["mode"] == mode) & (frame["param"] == param)]
            true = sub["theta_true"].to_numpy()
            mean = sub["posterior_mean"].to_numpy()
            lo = min(true.min(), mean.min())
            hi = max(true.max(), mean.max())
            artist = ax.hexbin(
                true, mean, gridsize=32, mincnt=1, cmap="viridis", linewidths=0
            )
            artists.append(artist)
            if artist.get_array().size:
                max_count = max(max_count, float(artist.get_array().max()))
            ax.plot([lo, hi], [lo, hi], color="black", lw=0.8, ls="--")
            ax.set(xlim=(lo, hi), ylim=(lo, hi))
            ax.set_title(rf"{MODE_LABELS[mode]}, {LABELS[param]}")
            ax.set_xlabel(rf"True {LABELS[param]}")
            ax.set_ylabel(rf"Posterior mean {LABELS[param]}")
    for artist in artists:
        artist.set_clim(1.0, max_count)
    colorbar = fig.colorbar(artists[-1], ax=axes, pad=0.02)
    colorbar.set_label("Profiles per hexagonal bin")
    fig.savefig(output, dpi=dpi)
    plt.close(fig)


def plot_metrics(summary: pd.DataFrame, output: Path, dpi: int) -> None:
    items = (
        ("rmse_over_prior", "RMSE / prior width"),
        ("pull_rmse", "Pull RMSE"),
        ("mean_posterior_std_over_prior", "Mean posterior std / prior width"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(18.0 / 2.54, 6.5 / 2.54))
    x = np.arange(len(PARAMS))
    for ax, (column, label) in zip(axes, items):
        for mode, offset in (("none", -0.1), ("asinh", 0.1)):
            sub = summary[(summary["mode"] == mode) & summary["param"].isin(PARAMS)]
            sub = sub.set_index("param").loc[list(PARAMS)]
            ax.plot(
                x + offset,
                sub[column],
                marker="o",
                lw=1.0,
                color=COLORS[mode],
                label=MODE_LABELS[mode],
            )
        if column == "pull_rmse":
            ax.axhline(1.0, color="black", lw=0.8, ls="--")
        ax.set_xticks(x)
        ax.set_xticklabels([LABELS[param] for param in PARAMS])
        ax.set_ylabel(label)
        ax.grid(True, axis="y", alpha=0.25, lw=0.5)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=dpi)
    plt.close(fig)


def plot_sbc_histograms(frame: pd.DataFrame, bins: int, output: Path, dpi: int) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(18.0 / 2.54, 11.5 / 2.54), sharex=True)
    for row_index, mode in enumerate(MODES):
        for column_index, param in enumerate(PARAMS):
            ax = axes[row_index, column_index]
            ranks = frame[
                (frame["mode"] == mode) & (frame["param"] == param)
            ]["rank_fraction"].to_numpy()
            expected = len(ranks) / bins
            sigma = math.sqrt(len(ranks) * (1.0 / bins) * (1.0 - 1.0 / bins))
            ax.axhspan(
                max(0.0, expected - 1.96 * sigma),
                expected + 1.96 * sigma,
                color="0.85",
            )
            ax.hist(
                ranks,
                bins=bins,
                range=(0.0, 1.0),
                color=COLORS[mode],
                alpha=0.78,
            )
            ax.axhline(expected, color="black", lw=0.8, ls="--")
            ax.set_title(rf"{MODE_LABELS[mode]}, {LABELS[param]}")
            ax.set_ylabel("Count")
            ax.grid(True, axis="y", alpha=0.2, lw=0.5)
    for ax in axes[-1]:
        ax.set_xlabel("SBC rank fraction")
    fig.tight_layout()
    fig.savefig(output, dpi=dpi)
    plt.close(fig)


def plot_sbc_cdf(frame: pd.DataFrame, output: Path, dpi: int) -> None:
    fig, axes = plt.subplots(
        1, 2, figsize=(18.0 / 2.54, 7.5 / 2.54), sharex=True, sharey=True
    )
    grid = np.linspace(0.0, 1.0, 401)
    n_test = len(frame[(frame["mode"] == "none") & (frame["param"] == PARAMS[0])])
    epsilon = math.sqrt(math.log(2.0 / 0.05) / (2.0 * n_test))
    for ax, param in zip(axes, PARAMS):
        ax.fill_between(
            grid,
            np.maximum(0.0, grid - epsilon),
            np.minimum(1.0, grid + epsilon),
            color="0.85",
            label="95% DKW band",
        )
        ax.plot(grid, grid, color="black", lw=0.8, ls="--", label="Uniform")
        for mode in MODES:
            ranks = np.sort(
                frame[(frame["mode"] == mode) & (frame["param"] == param)][
                    "rank_fraction"
                ].to_numpy()
            )
            empirical = np.arange(1, len(ranks) + 1) / len(ranks)
            ax.step(
                ranks,
                empirical,
                where="post",
                lw=1.0,
                color=COLORS[mode],
                label=MODE_LABELS[mode],
            )
        ax.set_title(LABELS[param])
        ax.set_xlabel("SBC rank fraction")
        ax.grid(True, alpha=0.2, lw=0.5)
    axes[0].set_ylabel("Empirical cumulative density")
    axes[0].legend(frameon=False, fontsize=6.5)
    fig.tight_layout()
    fig.savefig(output, dpi=dpi)
    plt.close(fig)


def plot_coverage(frame: pd.DataFrame, output: Path, dpi: int) -> None:
    fig, axes = plt.subplots(
        1, 2, figsize=(18.0 / 2.54, 7.5 / 2.54), sharex=True, sharey=True
    )
    levels = np.linspace(0.05, 0.99, 40)
    n_test = len(frame[(frame["mode"] == "none") & (frame["param"] == PARAMS[0])])
    band = 1.96 * np.sqrt(levels * (1.0 - levels) / n_test)
    for ax, param in zip(axes, PARAMS):
        ax.fill_between(
            levels,
            np.maximum(0.0, levels - band),
            np.minimum(1.0, levels + band),
            color="0.85",
            label="95% binomial band",
        )
        ax.plot(levels, levels, color="black", lw=0.8, ls="--", label="Ideal")
        for mode in MODES:
            ranks = frame[
                (frame["mode"] == mode) & (frame["param"] == param)
            ]["rank_fraction"].to_numpy()
            empirical = [
                np.mean(np.abs(2.0 * ranks - 1.0) <= level) for level in levels
            ]
            ax.plot(
                levels,
                empirical,
                color=COLORS[mode],
                lw=1.0,
                label=MODE_LABELS[mode],
            )
        ax.set_title(LABELS[param])
        ax.set_xlabel("Nominal central credibility")
        ax.grid(True, alpha=0.2, lw=0.5)
    axes[0].set_ylabel("Empirical coverage")
    axes[0].legend(frameon=False, fontsize=6.5)
    fig.tight_layout()
    fig.savefig(output, dpi=dpi)
    plt.close(fig)


def battaglia_outputs(run_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sample_sets = []
    rows = []
    for mode in MODES:
        path = run_root / mode / "battaglia12_posterior_samples.npy"
        samples = np.asarray(np.load(path), dtype=np.float64)
        if samples.ndim != 2 or samples.shape[1] != len(PARAMS):
            raise ValueError(f"Invalid Battaglia12 samples: {path}, {samples.shape}")
        sample_sets.append({"label": MODE_LABELS[mode], "samples": samples})
        for index, param in enumerate(PARAMS):
            qs = np.quantile(samples[:, index], [0.025, 0.16, 0.5, 0.84, 0.975])
            rows.append(
                {
                    "mode": mode,
                    "param": param,
                    "truth": BATTAGLIA12[index],
                    "mean": samples[:, index].mean(),
                    "std": samples[:, index].std(ddof=1),
                    "q025": qs[0],
                    "q16": qs[1],
                    "median": qs[2],
                    "q84": qs[3],
                    "q975": qs[4],
                    "n_samples": samples.shape[0],
                }
            )
    return sample_sets, rows


def plot_validation(run_root: Path, output: Path, dpi: int) -> None:
    values = []
    for mode in MODES:
        metadata = json.loads(
            (run_root / mode / "run_metadata.json").read_text(encoding="utf-8")
        )
        value = metadata.get("best_validation_performance")
        values.append(np.nan if value is None else float(value))
    fig, ax = plt.subplots(figsize=(9.0 / 2.54, 6.5 / 2.54))
    ax.bar(range(2), values, color=[COLORS[mode] for mode in MODES], width=0.62)
    ax.set_xticks(range(2))
    ax.set_xticklabels([MODE_LABELS[mode] for mode in MODES])
    ax.set_ylabel("Best validation performance")
    ax.grid(True, axis="y", alpha=0.25, lw=0.5)
    fig.tight_layout()
    fig.savefig(output, dpi=dpi)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    if args.last_n_test <= 0 or args.num_posterior_samples <= 1:
        raise ValueError("Invalid held-out or posterior sample count")
    dataset_path = args.prepared_dataset.expanduser().resolve()
    run_root = args.run_root.expanduser().resolve()
    output = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else run_root / "comparison_last1000"
    )
    output.mkdir(parents=True, exist_ok=True)
    paper_style()
    torch.set_num_threads(max(1, int(os.environ.get("TORCH_NUM_THREADS", "1"))))

    data = load_data(dataset_path)
    n_rows = data["theta"].shape[0]
    n_train = n_rows - args.last_n_test
    if n_train <= 0:
        raise ValueError("The held-out count exceeds the dataset")
    train = np.arange(n_train, dtype=np.int64)
    heldout = np.arange(n_train, n_rows, dtype=np.int64)
    checks = validate_models(run_root, data["indices"], train, heldout)
    preflight = {
        "prepared_dataset": dataset_path,
        "run_root": run_root,
        "n_rows": n_rows,
        "n_train": n_train,
        "heldout_last_n": args.last_n_test,
        "heldout_first": heldout[0],
        "heldout_last": heldout[-1],
        "target_param_names": PARAMS,
        "target_param_indices": data["indices"],
        "model_checks": checks,
    }
    write_json(output / "evaluation_preflight.json", preflight)
    print(json.dumps(jsonable(preflight), indent=2, sort_keys=True))
    if args.validate_only:
        print("Evaluation validation-only check passed; no sampling was performed.")
        return 0

    x_eval = np.ascontiguousarray(data["x"][heldout], dtype=np.float32)
    truth = np.ascontiguousarray(
        data["theta"][np.ix_(heldout, data["indices"])], dtype=np.float64
    )
    rows = []
    sample_paths = {}
    for mode in MODES:
        path = output / f"heldout_posterior_samples_{mode}.npy"
        samples = heldout_samples(
            mode,
            run_root / mode,
            x_eval,
            path,
            args.num_posterior_samples,
            args.seed,
            args.device,
            args.reuse_samples,
        )
        sample_paths[mode] = str(path)
        rows.extend(
            metric_rows(mode, samples, truth, heldout, data["low"], data["high"])
        )
        del samples

    frame = pd.DataFrame(rows)
    summary_rows = summaries(frame, args.rank_bins)
    frame.to_csv(output / "heldout_metrics.csv", index=False)
    write_csv(output / "heldout_summary.csv", summary_rows)
    summary_frame = pd.DataFrame(summary_rows)

    plot_correlations(frame, output / "true_vs_mean_asinh_vs_none.jpg", args.dpi)
    plot_density_correlations(
        frame, output / "true_vs_mean_density_asinh_vs_none.jpg", args.dpi
    )
    plot_metrics(
        summary_frame,
        output / "normalized_test_metrics_asinh_vs_none.jpg",
        args.dpi,
    )
    plot_sbc_histograms(
        frame,
        args.rank_bins,
        output / "sbc_rank_histograms_asinh_vs_none.jpg",
        args.dpi,
    )
    plot_sbc_cdf(frame, output / "sbc_rank_cdf_asinh_vs_none.jpg", args.dpi)
    plot_coverage(frame, output / "coverage_asinh_vs_none.jpg", args.dpi)

    sample_sets, constraint_rows = battaglia_outputs(run_root)
    plot_getdist(
        sample_sets,
        list(PARAMS),
        BATTAGLIA12,
        output / "battaglia12_P0_beta_getdist_asinh_vs_none.jpg",
        filled_last_only=True,
        dpi=args.dpi,
    )
    write_csv(output / "battaglia12_constraints.csv", constraint_rows)
    plot_validation(
        run_root,
        output / "best_validation_performance_asinh_vs_none.jpg",
        args.dpi,
    )
    write_json(
        output / "comparison_summary.json",
        {
            **preflight,
            "num_posterior_samples_per_test": args.num_posterior_samples,
            "sample_paths": sample_paths,
            "metric_summary": summary_rows,
            "battaglia12_constraints": constraint_rows,
            "rank_fraction_definition": "(count(samples < truth) + 0.5) / (S + 1)",
            "sbc_note": (
                "The held-out simulations are Sobol prior-predictive design points; "
                "the rank diagnostics are an empirical SBC approximation."
            ),
        },
    )
    print(f"Completed two-parameter comparison: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
