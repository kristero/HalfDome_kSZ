#!/usr/bin/env python3
"""Plot Pearson-correlation evolution for all nine SBI parameters.

This is a plotting-only local analysis. It reads the per-profile metrics CSV
written by the cluster evaluation and does not import torch or sbi.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PARAMETERS = [
    "P0",
    "xc",
    "beta",
    "alpha_m_P0",
    "alpha_m_xc",
    "alpha_m_beta",
    "alpha_z_P0",
    "alpha_z_xc",
    "alpha_z_beta",
]

LABELS = {
    "P0": r"$P_0$",
    "xc": r"$x_{\rm c}$",
    "beta": r"$\beta$",
    "alpha_m_P0": r"$\alpha_{m,P_0}$",
    "alpha_m_xc": r"$\alpha_{m,x_{\rm c}}$",
    "alpha_m_beta": r"$\alpha_{m,\beta}$",
    "alpha_z_P0": r"$\alpha_{z,P_0}$",
    "alpha_z_xc": r"$\alpha_{z,x_{\rm c}}$",
    "alpha_z_beta": r"$\alpha_{z,\beta}$",
}

COLORS = [
    "#1f77b4",
    "#d62728",
    "#2ca02c",
    "#9467bd",
    "#ff7f0e",
    "#17becf",
    "#8c564b",
    "#e377c2",
    "#4d4d4d",
]


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_csv = (
        repo_root
        / "SBI_analysis"
        / "convergence_tests"
        / "last100_param_metrics.csv"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-csv", type=Path, default=default_csv)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: correlation_evolution_9param beside the input CSV.",
    )
    parser.add_argument(
        "--case",
        default="masked_baseline_noise_cross_deproj0",
        help="Case selected when the CSV contains multiple noise cases.",
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--full-correlation-range",
        action="store_true",
        help="Show y=[-1,1]. The default publication plot uses y=[0,1].",
    )
    return parser.parse_args()


def pearson_r(truth: np.ndarray, estimate: np.ndarray) -> float:
    valid = np.isfinite(truth) & np.isfinite(estimate)
    truth = truth[valid]
    estimate = estimate[valid]
    if len(truth) < 2 or np.std(truth) == 0.0 or np.std(estimate) == 0.0:
        return float("nan")
    return float(np.corrcoef(truth, estimate)[0, 1])


def fisher_interval(r: float, count: int) -> tuple[float, float]:
    """Approximate two-sided 95% interval from Fisher's z transform."""
    if count <= 3 or not np.isfinite(r):
        return float("nan"), float("nan")
    z = np.arctanh(np.clip(r, -0.999999, 0.999999))
    dz = 1.959963984540054 / np.sqrt(count - 3)
    return float(np.tanh(z - dz)), float(np.tanh(z + dz))


def load_metrics(path: Path, case: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"n_train", "param", "theta_true", "posterior_mean"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    if "case" in frame.columns:
        available = sorted(frame["case"].dropna().astype(str).unique())
        if case not in available:
            raise ValueError(f"case={case!r} is absent. Available: {available}")
        frame = frame[frame["case"].astype(str) == case].copy()

    for column in ("n_train", "theta_true", "posterior_mean"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["n_train", "param", "theta_true", "posterior_mean"]
    )
    frame["n_train"] = frame["n_train"].astype(int)
    frame = frame[frame["param"].isin(PARAMETERS)].copy()

    keys = ["n_train", "param"]
    if "test_index" in frame.columns:
        keys.append("test_index")
    if frame.duplicated(keys, keep=False).any():
        raise ValueError(
            "Duplicate held-out rows found. Do not concatenate repeated "
            "evaluation tables before running this script."
        )
    return frame


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for n_train in sorted(frame["n_train"].unique()):
        for param in PARAMETERS:
            selected = frame[
                (frame["n_train"] == n_train) & (frame["param"] == param)
            ]
            truth = selected["theta_true"].to_numpy(dtype=float)
            mean = selected["posterior_mean"].to_numpy(dtype=float)
            valid = np.isfinite(truth) & np.isfinite(mean)
            r = pearson_r(truth, mean)
            low, high = fisher_interval(r, int(valid.sum()))
            rows.append(
                {
                    "n_train": int(n_train),
                    "param": param,
                    "n_profiles": int(valid.sum()),
                    "pearson_r": r,
                    "pearson_95_low": low,
                    "pearson_95_high": high,
                }
            )
    return pd.DataFrame(rows)


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
        }
    )


def plot_panels(
    summary: pd.DataFrame,
    output: Path,
    dpi: int,
    full_range: bool,
) -> None:
    fig, axes = plt.subplots(
        3,
        3,
        sharex=True,
        sharey=True,
        figsize=(18.0 / 2.54, 16.5 / 2.54),
    )
    for index, (axis, param) in enumerate(zip(axes.flat, PARAMETERS)):
        selected = summary[summary["param"] == param].sort_values("n_train")
        x = selected["n_train"].to_numpy(dtype=float)
        y = selected["pearson_r"].to_numpy(dtype=float)
        low = selected["pearson_95_low"].to_numpy(dtype=float)
        high = selected["pearson_95_high"].to_numpy(dtype=float)
        valid = np.isfinite(x) & np.isfinite(y)
        band = valid & np.isfinite(low) & np.isfinite(high)

        axis.plot(
            x[valid],
            y[valid],
            marker="o",
            ms=3.0,
            lw=1.0,
            color=COLORS[index],
        )
        axis.fill_between(
            x[band],
            low[band],
            high[band],
            color=COLORS[index],
            alpha=0.15,
        )
        axis.set_xscale("log")
        axis.set_ylim((-1.0, 1.0) if full_range else (0.0, 1.0))
        axis.set_title(LABELS[param], pad=3)
        axis.grid(True, which="both", alpha=0.24, lw=0.5)

    for axis in axes[-1, :]:
        axis.set_xlabel(r"Training set size, $N_{\rm train}$")
    for axis in axes[:, 0]:
        axis.set_ylabel(r"Pearson $r$")
    fig.tight_layout()
    fig.savefig(output, dpi=dpi)
    plt.close(fig)


def plot_combined(
    summary: pd.DataFrame,
    output: Path,
    dpi: int,
    full_range: bool,
) -> None:
    fig, axis = plt.subplots(figsize=(11.0 / 2.54, 8.0 / 2.54))
    for index, param in enumerate(PARAMETERS):
        selected = summary[summary["param"] == param].sort_values("n_train")
        axis.plot(
            selected["n_train"].to_numpy(dtype=float),
            selected["pearson_r"].to_numpy(dtype=float),
            marker="o",
            ms=2.7,
            lw=0.9,
            color=COLORS[index],
            label=LABELS[param],
        )

    axis.set_xscale("log")
    axis.set_ylim((-1.0, 1.0) if full_range else (0.0, 1.0))
    axis.set_xlabel(r"Training set size, $N_{\rm train}$")
    axis.set_ylabel(
        r"Pearson $r(\theta_{\rm true},\bar{\theta}_{\rm post})$"
    )
    axis.grid(True, which="both", alpha=0.24, lw=0.5)
    axis.legend(ncol=3, frameon=False, columnspacing=0.9, handlelength=1.8)
    fig.tight_layout()
    fig.savefig(output, dpi=dpi)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    metrics_path = args.metrics_csv.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else metrics_path.parent / "correlation_evolution_9param"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = load_metrics(metrics_path, args.case)
    summary = summarize(metrics)
    summary_path = output_dir / "pearson_correlation_by_parameter_and_ntrain.csv"
    panel_path = output_dir / "pearson_correlation_evolution_9param_panels.jpg"
    combined_path = output_dir / "pearson_correlation_evolution_9param_combined.jpg"

    summary.to_csv(summary_path, index=False)
    negative = summary[summary["pearson_r"] < 0.0]
    if not negative.empty and not args.full_correlation_range:
        print("WARNING: negative correlations are outside the [0,1] plot range:")
        print(
            negative[["n_train", "param", "pearson_r"]].to_string(index=False)
        )

    configure_style()
    plot_panels(summary, panel_path, args.dpi, args.full_correlation_range)
    plot_combined(summary, combined_path, args.dpi, args.full_correlation_range)

    counts = summary.pivot(
        index="n_train",
        columns="param",
        values="n_profiles",
    )
    print(f"Read: {metrics_path}")
    print(f"Training sizes: {sorted(metrics['n_train'].unique())}")
    print("Held-out profile counts per point:")
    print(counts.to_string())
    print("Saved:")
    print(f"  {summary_path}")
    print(f"  {panel_path}")
    print(f"  {combined_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
