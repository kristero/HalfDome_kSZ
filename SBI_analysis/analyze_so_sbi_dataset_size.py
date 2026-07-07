#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_CASES = [
    "no_noise",
    "goal_deproj0",
    "baseline_deproj0",
    "goal_deproj2",
    "baseline_deproj2",
]
DEFAULT_DATASET_SIZES = [256, 512, 1024, 2048, 4096, 8192, 16384, 32600]

PARAM_LABELS = {
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

CASE_LABELS = {
    "no_noise": r"no noise",
    "goal_deproj0": r"goal, deproj. 0",
    "baseline_deproj0": r"baseline, deproj. 0",
    "goal_deproj2": r"goal, deproj. 2",
    "baseline_deproj2": r"baseline, deproj. 2",
}

CASE_COLORS = {
    "no_noise": "#222222",
    "goal_deproj0": "#1f77b4",
    "baseline_deproj0": "#d62728",
    "goal_deproj2": "#2ca02c",
    "baseline_deproj2": "#9467bd",
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_path(path: str | Path, root: Path) -> Path:
    path = Path(path).expanduser()
    return path if path.is_absolute() else root / path


def parse_int_list(value: str | list[int]) -> list[int]:
    if isinstance(value, list):
        return [int(v) for v in value]
    parts = [part for part in str(value).replace(";", ",").replace(" ", ",").split(",") if part]
    return [int(float(part.replace("_", ""))) for part in parts]


def n_sort_key(path: Path) -> int:
    match = re.search(r"N(\d+)$", path.name)
    return int(match.group(1)) if match else 999999999


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def to_float(value: Any) -> float:
    return float(value)


def to_int(value: Any) -> int:
    return int(float(value))


def apply_paper_style(usetex: bool = False) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "text.usetex": bool(usetex),
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.minor.width": 0.5,
            "ytick.minor.width": 0.5,
            "savefig.bbox": "tight",
            "savefig.dpi": 300,
        }
    )


def load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def load_posterior(run_dir: Path) -> Any:
    inference_path = run_dir / "inference.pkl"
    density_path = run_dir / "density_estimator.pkl"
    if inference_path.exists() and density_path.exists():
        inference = load_pickle(inference_path)
        density_estimator = load_pickle(density_path)
        return inference.build_posterior(density_estimator)

    posterior_path = run_dir / "posterior.pkl"
    if posterior_path.exists():
        return load_pickle(posterior_path)

    raise FileNotFoundError(f"No posterior.pkl or inference.pkl+density_estimator.pkl found in {run_dir}")


def sample_posterior_at_x(posterior: Any, x_obs: np.ndarray, num_samples: int, device: str) -> np.ndarray:
    import torch

    x_t = torch.as_tensor(np.asarray(x_obs, dtype=np.float32), dtype=torch.float32, device=device)
    try:
        samples = posterior.sample((int(num_samples),), x=x_t, show_progress_bars=False)
    except TypeError:
        try:
            samples = posterior.sample((int(num_samples),), x=x_t)
        except TypeError:
            posterior_x = posterior.set_default_x(x_t)
            if posterior_x is None:
                posterior_x = posterior
            try:
                samples = posterior_x.sample((int(num_samples),), show_progress_bars=False)
            except TypeError:
                samples = posterior_x.sample((int(num_samples),))

    if torch.is_tensor(samples):
        samples = samples.detach().cpu().numpy()
    return np.asarray(samples, dtype=np.float64)


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


def find_run_dirs(run_root: Path, case: str, dataset_sizes: list[int], allow_missing: bool) -> dict[int, Path]:
    case_root = run_root / case
    if not case_root.is_dir():
        if allow_missing:
            return {}
        raise FileNotFoundError(f"Case output directory not found: {case_root}")

    found: dict[int, Path] = {}
    duplicates: dict[int, list[Path]] = defaultdict(list)
    for path in sorted(case_root.glob("**/N*"), key=n_sort_key):
        if not path.is_dir():
            continue
        match = re.fullmatch(r"N(\d+)", path.name)
        if match is None:
            continue
        n_train = int(match.group(1))
        if n_train not in dataset_sizes:
            continue
        if not ((path / "posterior.pkl").exists() or ((path / "inference.pkl").exists() and (path / "density_estimator.pkl").exists())):
            continue
        if n_train in found:
            duplicates[n_train].extend([found[n_train], path])
        else:
            found[n_train] = path

    if duplicates:
        msg = "\n".join(f"N={n}: {sorted(set(map(str, paths)))}" for n, paths in sorted(duplicates.items()))
        raise ValueError(f"Duplicate completed run directories under {case_root}:\n{msg}")

    missing = [n for n in dataset_sizes if n not in found]
    if missing and not allow_missing:
        raise FileNotFoundError(f"Missing completed runs for case={case}: {missing}")
    return dict(sorted(found.items()))


def sem(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size <= 1:
        return 0.0
    return float(np.nanstd(arr, ddof=1) / math.sqrt(arr.size))


def summarize_profile_rows(profile_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in profile_rows:
        key = (str(row["case"]), to_int(row["n_train"]))
        grouped[key]["mse"].append(to_float(row["mse"]))
        grouped[key]["rmse"].append(to_float(row["rmse"]))
        grouped[key]["rmse_over_std"].append(to_float(row["rmse_over_std"]))
        grouped[key]["mean_posterior_std"].append(to_float(row["mean_posterior_std"]))
        grouped[key]["mean_abs_pull"].append(to_float(row["mean_abs_pull"]))

    rows: list[dict[str, Any]] = []
    for (case, n_train), values in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        rows.append(
            {
                "case": case,
                "n_train": n_train,
                "n_test": len(values["rmse"]),
                "mean_mse": float(np.nanmean(values["mse"])),
                "mean_mse_err": sem(values["mse"]),
                "mean_rmse": float(np.nanmean(values["rmse"])),
                "mean_rmse_err": sem(values["rmse"]),
                "mean_rmse_over_std": float(np.nanmean(values["rmse_over_std"])),
                "mean_rmse_over_std_err": sem(values["rmse_over_std"]),
                "mean_posterior_std": float(np.nanmean(values["mean_posterior_std"])),
                "mean_posterior_std_err": sem(values["mean_posterior_std"]),
                "mean_abs_pull": float(np.nanmean(values["mean_abs_pull"])),
                "mean_abs_pull_err": sem(values["mean_abs_pull"]),
            }
        )
    return rows


def scalar_string(value: Any, default: str = "") -> str:
    arr = np.asarray(value)
    if arr.shape == ():
        return str(arr.item())
    if arr.size == 1:
        return str(arr.reshape(()).item())
    return default


def evaluation_set_from_dataset(
    data: Any,
    analysis_target: str,
    last_n_test: int,
) -> tuple[np.ndarray, np.ndarray, list[str], str]:
    theta = np.asarray(data["theta"], dtype=np.float64)
    x = np.asarray(data["x"], dtype=np.float32)

    if analysis_target == "obs":
        if "obs" not in data.files or "obs_theta" not in data.files:
            raise KeyError("analysis-target=obs requires obs and obs_theta in the case dataset.")
        obs_source = scalar_string(data["obs_source"], "obs") if "obs_source" in data.files else "obs"
        return (
            np.asarray(data["obs"], dtype=np.float32).reshape(1, -1),
            np.asarray(data["obs_theta"], dtype=np.float64).reshape(1, -1),
            [obs_source],
            obs_source,
        )

    if analysis_target != "last_n":
        raise ValueError(f"Unsupported analysis_target={analysis_target!r}")

    if "test_indices" in data.files and np.asarray(data["test_indices"]).size:
        test_indices = np.asarray(data["test_indices"], dtype=np.int64)
    else:
        test_indices = np.arange(theta.shape[0] - int(last_n_test), theta.shape[0], dtype=np.int64)

    if int(last_n_test) > 0:
        test_indices = test_indices[-int(last_n_test) :]
    if test_indices.size == 0:
        raise ValueError("No test indices available for analysis-target=last_n")

    return (
        np.asarray(x[test_indices], dtype=np.float32),
        np.asarray(theta[test_indices], dtype=np.float64),
        [str(int(idx)) for idx in test_indices],
        f"last{int(test_indices.size)}",
    )


def compute_metrics(args: argparse.Namespace, root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    run_root = resolve_path(args.run_root, root)
    dataset_dir = resolve_path(args.case_dataset_dir, root)
    index_json = resolve_path(args.case_index_json, root) if args.case_index_json else dataset_dir / "case_dataset_index.json"
    dataset_sizes = parse_int_list(args.dataset_sizes)

    param_rows: list[dict[str, Any]] = []
    profile_rows: list[dict[str, Any]] = []

    for case in args.cases:
        dataset_path = find_case_dataset(case, dataset_dir, index_json)
        print(f"\nCase {case}: dataset={dataset_path}")
        with np.load(dataset_path, allow_pickle=True) as data:
            param_names = [str(v) for v in data["param_names"]]
            x_eval, theta_eval, eval_labels, eval_tag = evaluation_set_from_dataset(
                data,
                args.analysis_target,
                args.last_n_test,
            )

        run_dirs = find_run_dirs(run_root, case, dataset_sizes, args.allow_missing)
        print(f"  Using N values: {list(run_dirs)}")
        print(f"  Analysis target: {args.analysis_target} ({eval_tag}), n_eval={len(eval_labels)}")

        for n_train, run_dir in run_dirs.items():
            print(f"  N={n_train}: {run_dir}")
            posterior = load_posterior(run_dir)

            for eval_idx, eval_label in enumerate(eval_labels):
                samples = sample_posterior_at_x(posterior, x_eval[eval_idx], args.num_posterior_samples, args.device)
                n_params = min(samples.shape[1], theta_eval.shape[1], len(param_names))
                samples = samples[:, :n_params]
                theta_true = theta_eval[eval_idx, :n_params]

                mean = np.nanmean(samples, axis=0)
                std = np.nanstd(samples, axis=0, ddof=1)
                std = np.where(std > 0.0, std, np.nan)
                error = mean - theta_true
                error_over_std = error / std
                mse = float(np.nanmean(error**2))

                profile_rows.append(
                    {
                        "case": case,
                        "n_train": int(n_train),
                        "test_index": str(eval_label),
                        "analysis_target": args.analysis_target,
                        "mse": mse,
                        "rmse": float(np.sqrt(mse)),
                        "rmse_over_std": float(np.sqrt(np.nanmean(error_over_std**2))),
                        "mean_posterior_std": float(np.nanmean(std)),
                        "mean_abs_pull": float(np.nanmean(np.abs(error_over_std))),
                    }
                )

                for j in range(n_params):
                    param_rows.append(
                        {
                            "case": case,
                            "n_train": int(n_train),
                            "test_index": str(eval_label),
                            "analysis_target": args.analysis_target,
                            "param": param_names[j],
                            "param_index": int(j),
                            "theta_true": float(theta_true[j]),
                            "posterior_mean": float(mean[j]),
                            "posterior_std": float(std[j]),
                            "error": float(error[j]),
                            "pull": float(error_over_std[j]),
                            "abs_pull": float(abs(error_over_std[j])),
                            "num_posterior_samples": int(args.num_posterior_samples),
                            "run_dir": str(run_dir),
                        }
                    )

            del posterior

    summary_rows = summarize_profile_rows(profile_rows)
    return param_rows, profile_rows, summary_rows


def group_rows(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    return grouped


def plot_profile_metric(
    profile_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    case: str,
    metric: str,
    summary_metric: str,
    err_metric: str,
    ylabel: str,
    output_path: Path,
    hline: float | None = None,
    dpi: int = 300,
) -> None:
    case_profile = [row for row in profile_rows if row["case"] == case]
    case_summary = sorted([row for row in summary_rows if row["case"] == case], key=lambda row: to_int(row["n_train"]))
    color = CASE_COLORS.get(case, "#1f77b4")

    fig, ax = plt.subplots(figsize=(8.8 / 2.54, 6.2 / 2.54))
    by_test = group_rows(case_profile, ("test_index",))
    for _, rows in sorted(by_test.items()):
        rows = sorted(rows, key=lambda row: to_int(row["n_train"]))
        ax.plot(
            [to_int(row["n_train"]) for row in rows],
            [to_float(row[metric]) for row in rows],
            color=color,
            alpha=0.08,
            lw=0.6,
            zorder=1,
        )

    ax.errorbar(
        [to_int(row["n_train"]) for row in case_summary],
        [to_float(row[summary_metric]) for row in case_summary],
        yerr=[to_float(row[err_metric]) for row in case_summary],
        color=color,
        marker="o",
        ms=3.2,
        lw=1.2,
        capsize=2.5,
        label=CASE_LABELS.get(case, case),
        zorder=3,
    )

    if hline is not None:
        ax.axhline(hline, color="black", lw=0.8, ls="--", alpha=0.75)
    ax.set_xscale("log")
    ax.set_xlabel(r"Training set size")
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", alpha=0.25, lw=0.5)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_case_comparison(
    summary_rows: list[dict[str, Any]],
    cases: list[str],
    summary_metric: str,
    err_metric: str,
    ylabel: str,
    output_path: Path,
    hline: float | None = None,
    dpi: int = 300,
) -> None:
    fig, ax = plt.subplots(figsize=(8.8 / 2.54, 6.2 / 2.54))
    for case in cases:
        rows = sorted([row for row in summary_rows if row["case"] == case], key=lambda row: to_int(row["n_train"]))
        if not rows:
            continue
        ax.errorbar(
            [to_int(row["n_train"]) for row in rows],
            [to_float(row[summary_metric]) for row in rows],
            yerr=[to_float(row[err_metric]) for row in rows],
            marker="o",
            ms=3.0,
            lw=1.0,
            capsize=2.0,
            color=CASE_COLORS.get(case),
            label=CASE_LABELS.get(case, case),
        )
    if hline is not None:
        ax.axhline(hline, color="black", lw=0.8, ls="--", alpha=0.75)
    ax.set_xscale("log")
    ax.set_xlabel(r"Training set size")
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", alpha=0.25, lw=0.5)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_true_vs_mean(
    param_rows: list[dict[str, Any]],
    case: str,
    n_train: int,
    output_path: Path,
    dpi: int = 300,
) -> None:
    rows = [row for row in param_rows if row["case"] == case and to_int(row["n_train"]) == int(n_train)]
    if not rows:
        return

    param_indices = sorted({to_int(row["param_index"]) for row in rows})
    n_params = len(param_indices)
    n_cols = 3
    n_rows = int(math.ceil(n_params / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18.0 / 2.54, max(6.0, 5.4 * n_rows) / 2.54))
    axes = np.asarray(axes).reshape(-1)
    color = CASE_COLORS.get(case, "#1f77b4")

    for ax, param_index in zip(axes, param_indices):
        sub = [row for row in rows if to_int(row["param_index"]) == param_index]
        x_true = np.asarray([to_float(row["theta_true"]) for row in sub], dtype=float)
        y_mean = np.asarray([to_float(row["posterior_mean"]) for row in sub], dtype=float)
        param = str(sub[0]["param"])
        lo = float(np.nanmin([np.nanmin(x_true), np.nanmin(y_mean)]))
        hi = float(np.nanmax([np.nanmax(x_true), np.nanmax(y_mean)]))
        pad = 0.06 * (hi - lo) if hi > lo else 0.1 * max(abs(hi), 1.0)
        lo -= pad
        hi += pad

        ax.scatter(x_true, y_mean, s=11, alpha=0.55, color=color, edgecolor="none")
        ax.plot([lo, hi], [lo, hi], color="black", lw=0.8, ls=":")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_title(PARAM_LABELS.get(param, param), pad=2.0)
        ax.set_xlabel(r"True")
        ax.set_ylabel(r"Posterior mean")
        ax.grid(True, alpha=0.25, lw=0.5)

    for ax in axes[n_params:]:
        ax.axis("off")

    fig.suptitle(rf"{CASE_LABELS.get(case, case)}, $N={int(n_train):,}$", y=0.995, fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def make_plots(
    param_rows: list[dict[str, Any]],
    profile_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    args: argparse.Namespace,
    root: Path,
) -> None:
    output_dir = resolve_path(args.output_dir, root)
    plot_dir = output_dir / "jpg"
    cases = list(args.cases)
    dataset_sizes = parse_int_list(args.dataset_sizes)
    tag = args.analysis_tag or ("battaglia12" if args.analysis_target == "obs" else f"last{int(args.last_n_test)}")

    for case in cases:
        plot_profile_metric(
            profile_rows,
            summary_rows,
            case,
            metric="mse",
            summary_metric="mean_mse",
            err_metric="mean_mse_err",
            ylabel=r"$\langle(\bar{\theta}-\theta_{\rm true})^2\rangle$",
            output_path=plot_dir / f"{case}_{tag}_mse_vs_dataset_size.jpg",
            dpi=args.dpi,
        )
        plot_profile_metric(
            profile_rows,
            summary_rows,
            case,
            metric="rmse",
            summary_metric="mean_rmse",
            err_metric="mean_rmse_err",
            ylabel=r"$\sqrt{\langle(\bar{\theta}-\theta_{\rm true})^2\rangle}$",
            output_path=plot_dir / f"{case}_{tag}_rmse_vs_dataset_size.jpg",
            dpi=args.dpi,
        )
        plot_profile_metric(
            profile_rows,
            summary_rows,
            case,
            metric="rmse_over_std",
            summary_metric="mean_rmse_over_std",
            err_metric="mean_rmse_over_std_err",
            ylabel=r"$\sqrt{\langle[(\bar{\theta}-\theta_{\rm true})/\sigma_{\rm post}]^2\rangle}$",
            output_path=plot_dir / f"{case}_{tag}_rmse_over_std_vs_dataset_size.jpg",
            hline=1.0,
            dpi=args.dpi,
        )

        for n_train in dataset_sizes:
            plot_true_vs_mean(
                param_rows,
                case,
                n_train,
                plot_dir / f"{case}_{tag}_true_vs_posterior_mean_N{int(n_train)}.jpg",
                dpi=args.dpi,
            )

    plot_case_comparison(
        summary_rows,
        cases,
        summary_metric="mean_mse",
        err_metric="mean_mse_err",
        ylabel=r"$\langle(\bar{\theta}-\theta_{\rm true})^2\rangle$",
        output_path=plot_dir / f"all_cases_{tag}_mse_vs_dataset_size.jpg",
        dpi=args.dpi,
    )
    plot_case_comparison(
        summary_rows,
        cases,
        summary_metric="mean_rmse",
        err_metric="mean_rmse_err",
        ylabel=r"$\sqrt{\langle(\bar{\theta}-\theta_{\rm true})^2\rangle}$",
        output_path=plot_dir / f"all_cases_{tag}_rmse_vs_dataset_size.jpg",
        dpi=args.dpi,
    )
    plot_case_comparison(
        summary_rows,
        cases,
        summary_metric="mean_rmse_over_std",
        err_metric="mean_rmse_over_std_err",
        ylabel=r"$\sqrt{\langle[(\bar{\theta}-\theta_{\rm true})/\sigma_{\rm post}]^2\rangle}$",
        output_path=plot_dir / f"all_cases_{tag}_rmse_over_std_vs_dataset_size.jpg",
        hline=1.0,
        dpi=args.dpi,
    )


def parse_args() -> argparse.Namespace:
    root = repo_root()
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate SO-noise SBI dataset-size sweeps on the stored observation or last held-out profiles and "
            "make A&A-style JPG diagnostics."
        )
    )
    parser.add_argument(
        "--run-root",
        default=str(root / "SBI_analysis" / "outputs" / "cluster_outputs" / "SBI_SO_noise_dataset_size_analysis"),
        help="Root containing case/group*/N* SBI outputs.",
    )
    parser.add_argument(
        "--case-dataset-dir",
        default=str(root / "SBI_analysis" / "data_for_cluster" / "so_noise_sbi_cases_ell80_7979"),
        help="Directory containing case-specific NPZ datasets and case_dataset_index.json.",
    )
    parser.add_argument("--case-index-json", default="")
    parser.add_argument("--output-dir", default="", help="Diagnostics output directory. Default depends on --analysis-target.")
    parser.add_argument("--cases", nargs="+", default=DEFAULT_CASES)
    parser.add_argument(
        "--dataset-sizes",
        default=",".join(str(v) for v in DEFAULT_DATASET_SIZES),
        help="Comma-separated N values to analyze.",
    )
    parser.add_argument(
        "--analysis-target",
        choices=("obs", "last_n"),
        default="obs",
        help="Default obs analyzes the stored obs/obs_theta in each case NPZ, which is Battaglia12 in the default preparation.",
    )
    parser.add_argument(
        "--analysis-tag",
        default="",
        help="Short string used in output filenames. Default: battaglia12 for obs, otherwise last<N>.",
    )
    parser.add_argument("--last-n-test", type=int, default=100)
    parser.add_argument("--num-posterior-samples", type=int, default=50000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--reuse-metrics", action="store_true", help="Read existing CSV metrics instead of resampling posteriors.")
    parser.add_argument("--usetex", action="store_true", help="Use external LaTeX for matplotlib text if available.")
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = repo_root()
    if not args.output_dir:
        default_tag = args.analysis_tag or ("battaglia12" if args.analysis_target == "obs" else f"last{int(args.last_n_test)}")
        args.output_dir = str(resolve_path(args.run_root, root) / f"diagnostics_{default_tag}")
    output_dir = resolve_path(args.output_dir, root)
    output_dir.mkdir(parents=True, exist_ok=True)

    apply_paper_style(args.usetex)

    csv_tag = args.analysis_tag or ("battaglia12" if args.analysis_target == "obs" else f"last{int(args.last_n_test)}")
    param_csv = output_dir / f"{csv_tag}_param_metrics.csv"
    profile_csv = output_dir / f"{csv_tag}_profile_metrics.csv"
    summary_csv = output_dir / f"{csv_tag}_summary.csv"

    if args.reuse_metrics and param_csv.is_file() and profile_csv.is_file() and summary_csv.is_file():
        print(f"Reusing metrics from {output_dir}")
        param_rows = read_csv(param_csv)
        profile_rows = read_csv(profile_csv)
        summary_rows = read_csv(summary_csv)
    else:
        param_rows, profile_rows, summary_rows = compute_metrics(args, root)
        write_csv(
            param_csv,
            param_rows,
            [
                "case",
                "n_train",
                "test_index",
                "analysis_target",
                "param",
                "param_index",
                "theta_true",
                "posterior_mean",
                "posterior_std",
                "error",
                "pull",
                "abs_pull",
                "num_posterior_samples",
                "run_dir",
            ],
        )
        write_csv(
            profile_csv,
            profile_rows,
            [
                "case",
                "n_train",
                "test_index",
                "analysis_target",
                "mse",
                "rmse",
                "rmse_over_std",
                "mean_posterior_std",
                "mean_abs_pull",
            ],
        )
        write_csv(
            summary_csv,
            summary_rows,
            [
                "case",
                "n_train",
                "n_test",
                "mean_mse",
                "mean_mse_err",
                "mean_rmse",
                "mean_rmse_err",
                "mean_rmse_over_std",
                "mean_rmse_over_std_err",
                "mean_posterior_std",
                "mean_posterior_std_err",
                "mean_abs_pull",
                "mean_abs_pull_err",
            ],
        )
        print(f"Wrote metrics to {output_dir}")

    make_plots(param_rows, profile_rows, summary_rows, args, root)
    print(f"Wrote JPG plots to {output_dir / 'jpg'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
