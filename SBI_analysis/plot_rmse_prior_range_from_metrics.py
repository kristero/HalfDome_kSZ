#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


CASE_LABELS = {
    "no_noise": r"no noise",
    "goal_deproj0": r"goal deproj0",
    "baseline_deproj0": r"baseline deproj0",
    "goal_deproj2": r"goal deproj2",
    "baseline_deproj2": r"baseline deproj2",
    "unmasked_no_noise": r"unmasked no noise",
    "masked_no_noise": r"no noise",
    "masked_goal_noise_cross_deproj0": r"goal deproj0",
    "masked_baseline_noise_cross_deproj0": r"baseline deproj0",
    "masked_goal_noise_cross_deproj2": r"goal deproj2",
    "masked_baseline_noise_cross_deproj2": r"baseline deproj2",
}

CASE_COLORS = {
    "no_noise": "#222222",
    "goal_deproj0": "#1f77b4",
    "baseline_deproj0": "#d62728",
    "goal_deproj2": "#2ca02c",
    "baseline_deproj2": "#9467bd",
    "unmasked_no_noise": "#666666",
    "masked_no_noise": "#222222",
    "masked_goal_noise_cross_deproj0": "#1f77b4",
    "masked_baseline_noise_cross_deproj0": "#d62728",
    "masked_goal_noise_cross_deproj2": "#2ca02c",
    "masked_baseline_noise_cross_deproj2": "#9467bd",
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_path(path: str | Path, root: Path) -> Path:
    path = Path(path).expanduser()
    return path if path.is_absolute() else root / path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def sem(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 1:
        return 0.0
    return float(np.nanstd(arr, ddof=1) / math.sqrt(arr.size))


def parse_cases(raw: str) -> list[str]:
    return [part for part in str(raw).replace(",", " ").split() if part]


def find_case_dataset(case: str, dataset_dir: Path, index_json: Path | None) -> Path:
    if index_json is not None and index_json.is_file():
        index = json.loads(index_json.read_text(encoding="utf-8"))
        case_entry = index.get("cases", {}).get(case)
        if case_entry and case_entry.get("path"):
            path = Path(case_entry["path"]).expanduser()
            if path.is_file():
                return path
    matches = sorted(dataset_dir.glob(f"so_{case}_*_sbi_run.npz"))
    if not matches:
        raise FileNotFoundError(f"Could not find case dataset for {case} in {dataset_dir}")
    if len(matches) > 1:
        raise ValueError(f"Multiple case datasets for {case} in {dataset_dir}: {matches}")
    return matches[0]


def load_prior_widths(cases: list[str], dataset_dir: Path, index_json: Path | None) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for case in cases:
        dataset_path = find_case_dataset(case, dataset_dir, index_json)
        with np.load(dataset_path, allow_pickle=True) as data:
            names = [str(v) for v in data["param_names"]]
            prior_low = np.asarray(data["prior_low"], dtype=float)
            prior_high = np.asarray(data["prior_high"], dtype=float)
        out[case] = {
            name: float(hi - lo)
            for name, lo, hi in zip(names, prior_low, prior_high)
        }
    return out


def summarize_from_param_rows(
    param_rows: list[dict[str, str]],
    prior_widths: dict[str, dict[str, float]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, str], list[float]] = defaultdict(list)
    for row in param_rows:
        case = str(row["case"])
        n_train = int(float(row["n_train"]))
        test_index = str(row["test_index"])
        param = str(row["param"])
        error = float(row["error"])
        width = float(row.get("prior_width") or prior_widths[case][param])
        if width > 0.0:
            grouped[(case, n_train, test_index)].append(error / width)

    profile_values: dict[tuple[str, int], list[float]] = defaultdict(list)
    for (case, n_train, _test_index), values in grouped.items():
        arr = np.asarray(values, dtype=float)
        profile_values[(case, n_train)].append(float(np.sqrt(np.nanmean(arr**2))))

    summary_rows: list[dict[str, Any]] = []
    for (case, n_train), values in sorted(profile_values.items(), key=lambda item: (item[0][0], item[0][1])):
        summary_rows.append(
            {
                "case": case,
                "n_train": int(n_train),
                "n_test": int(len(values)),
                "mean_rmse_over_prior_range": float(np.nanmean(values)),
                "mean_rmse_over_prior_range_err": sem(values),
            }
        )
    return summary_rows


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 8,
            "axes.linewidth": 0.7,
            "savefig.bbox": "tight",
            "savefig.dpi": 300,
        }
    )


def plot_summary(summary_rows: list[dict[str, Any]], cases: list[str], output_path: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(8.8 / 2.54, 6.2 / 2.54))
    for case in cases:
        rows = sorted([row for row in summary_rows if row["case"] == case], key=lambda row: int(row["n_train"]))
        if not rows:
            continue
        ax.errorbar(
            [int(row["n_train"]) for row in rows],
            [float(row["mean_rmse_over_prior_range"]) for row in rows],
            yerr=[float(row["mean_rmse_over_prior_range_err"]) for row in rows],
            marker="o",
            ms=3.4,
            lw=1.0,
            capsize=0.0,
            color=CASE_COLORS.get(case),
            label=CASE_LABELS.get(case, case),
        )

    ax.set_xscale("log")
    ax.set_xlabel(r"Training set size")
    ax.set_ylabel(r"$\sqrt{\langle[(\bar{\theta}-\theta_{\rm true})/\Delta\theta_{\rm prior}]^2\rangle}$")
    ax.grid(True, which="both", alpha=0.25, lw=0.5)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    print(f"Saved {output_path}")


def parse_args() -> argparse.Namespace:
    root = repo_root()
    parser = argparse.ArgumentParser(
        description="Plot RMSE normalized by prior range from analyze_so_sbi_dataset_size.py param_metrics CSV."
    )
    parser.add_argument("--param-metrics-csv", required=True)
    parser.add_argument(
        "--case-dataset-dir",
        default=str(root / "SBI_analysis" / "data_for_cluster" / "adrian_so_sbi_cases_ell80_7979_dataset_row"),
    )
    parser.add_argument("--case-index-json", default="")
    parser.add_argument("--cases", default="", help="Comma/space-separated cases. Default uses cases found in CSV order.")
    parser.add_argument("--output", default="")
    parser.add_argument("--summary-csv", default="")
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = repo_root()
    param_csv = resolve_path(args.param_metrics_csv, root)
    if not param_csv.is_file():
        raise FileNotFoundError(f"Param metrics CSV not found: {param_csv}")

    param_rows = read_csv(param_csv)
    if not param_rows:
        raise ValueError(f"No rows in {param_csv}")

    if args.cases:
        cases = parse_cases(args.cases)
    else:
        cases = []
        for row in param_rows:
            case = str(row["case"])
            if case not in cases:
                cases.append(case)

    dataset_dir = resolve_path(args.case_dataset_dir, root)
    index_json = resolve_path(args.case_index_json, root) if args.case_index_json else dataset_dir / "case_dataset_index.json"
    prior_widths = load_prior_widths(cases, dataset_dir, index_json)
    summary_rows = summarize_from_param_rows(param_rows, prior_widths)

    output_path = resolve_path(args.output, root) if args.output else param_csv.parent / "jpg" / "all_cases_rmse_over_prior_range_vs_dataset_size.jpg"
    summary_csv = resolve_path(args.summary_csv, root) if args.summary_csv else param_csv.with_name(param_csv.stem.replace("_param_metrics", "") + "_rmse_over_prior_range_summary.csv")

    apply_style()
    write_csv(
        summary_csv,
        summary_rows,
        ["case", "n_train", "n_test", "mean_rmse_over_prior_range", "mean_rmse_over_prior_range_err"],
    )
    plot_summary(summary_rows, cases, output_path, args.dpi)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
