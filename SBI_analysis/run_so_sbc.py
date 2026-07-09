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
    "no_noise": "no noise",
    "goal_deproj0": "goal, deproj. 0",
    "baseline_deproj0": "baseline, deproj. 0",
    "goal_deproj2": "goal, deproj. 2",
    "baseline_deproj2": "baseline, deproj. 2",
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
    raw = str(value or "").strip()
    if not raw or raw.lower() == "auto":
        return []
    parts = [part for part in raw.replace(";", ",").replace(" ", ",").split(",") if part]
    return [int(float(part.replace("_", ""))) for part in parts]


def n_sort_key(path: Path) -> int:
    match = re.search(r"N(\d+)$", path.name)
    return int(match.group(1)) if match else 999999999


def scalar_string(value: Any, default: str = "") -> str:
    arr = np.asarray(value)
    if arr.shape == ():
        return str(arr.item())
    if arr.size == 1:
        return str(arr.reshape(()).item())
    return default


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)


def apply_paper_style() -> None:
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
    samples_np = np.asarray(samples, dtype=np.float64)
    if samples_np.ndim == 1:
        samples_np = samples_np.reshape(1, -1)
    elif samples_np.ndim > 2:
        samples_np = samples_np.reshape(-1, samples_np.shape[-1])
    if samples_np.ndim != 2 or samples_np.shape[0] == 0:
        raise ValueError(f"Posterior sampling returned unsupported shape: {samples_np.shape}")
    return samples_np


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
    requested = set(int(v) for v in dataset_sizes) if dataset_sizes else None

    for path in sorted(case_root.glob("**/N*"), key=n_sort_key):
        if not path.is_dir():
            continue
        match = re.fullmatch(r"N(\d+)", path.name)
        if match is None:
            continue
        n_train = int(match.group(1))
        if requested is not None and n_train not in requested:
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

    if requested is not None:
        missing = [n for n in sorted(requested) if n not in found]
        if missing and not allow_missing:
            raise FileNotFoundError(f"Missing completed runs for case={case}: {missing}")
    return dict(sorted(found.items()))


def load_x_transform(run_dir: Path) -> dict[str, Any]:
    transform_path = run_dir / "x_transform.npz"
    if not transform_path.is_file():
        return {"mode": "none", "path": ""}
    with np.load(transform_path, allow_pickle=True) as data:
        out = {key: np.asarray(data[key]).copy() for key in data.files}
    out["mode"] = scalar_string(out.get("mode", "none"), "none")
    out["path"] = str(transform_path)
    return out


def apply_x_transform(x_values: np.ndarray, transform: dict[str, Any]) -> np.ndarray:
    values = np.asarray(x_values, dtype=np.float32)
    mode = str(transform.get("mode", "none")).strip().lower().replace("-", "_")
    if mode in {"", "none", "raw"}:
        return np.ascontiguousarray(values, dtype=np.float32)
    if mode in {"asinh", "asinh_median_abs"}:
        scale = np.asarray(transform["scale"], dtype=np.float32)
        return np.ascontiguousarray(np.arcsinh(values / scale), dtype=np.float32)
    if mode == "standardize":
        mean = np.asarray(transform["mean"], dtype=np.float32)
        std = np.asarray(transform["std"], dtype=np.float32)
        return np.ascontiguousarray((values - mean) / std, dtype=np.float32)
    if mode == "asinh_standardize":
        scale = np.asarray(transform["scale"], dtype=np.float32)
        mean = np.asarray(transform["mean"], dtype=np.float32)
        std = np.asarray(transform["std"], dtype=np.float32)
        return np.ascontiguousarray((np.arcsinh(values / scale) - mean) / std, dtype=np.float32)
    raise ValueError(f"Unsupported x transform mode in {transform.get('path', '<memory>')}: {mode!r}")


def choose_sbc_indices(data: Any, last_n_test: int, max_sbc: int, seed: int) -> np.ndarray:
    theta = np.asarray(data["theta"], dtype=np.float32)
    if "test_indices" in data.files and np.asarray(data["test_indices"]).size:
        indices = np.asarray(data["test_indices"], dtype=np.int64)
    else:
        indices = np.arange(theta.shape[0] - int(last_n_test), theta.shape[0], dtype=np.int64)
    if int(last_n_test) > 0:
        indices = indices[-int(last_n_test) :]
    if int(max_sbc) > 0 and indices.size > int(max_sbc):
        rng = np.random.default_rng(int(seed))
        indices = np.sort(rng.choice(indices, size=int(max_sbc), replace=False).astype(np.int64))
    if indices.size == 0:
        raise ValueError("No SBC test indices selected")
    return indices


def sbc_for_run(
    *,
    case: str,
    n_train: int,
    run_dir: Path,
    posterior: Any,
    x_eval: np.ndarray,
    theta_eval: np.ndarray,
    eval_indices: np.ndarray,
    param_names: list[str],
    num_posterior_samples: int,
    device: str,
    save_samples_dir: Path | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    transform = load_x_transform(run_dir)
    mode = str(transform.get("mode", "none"))
    rank_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    ranks_by_param: dict[int, list[int]] = {j: [] for j in range(len(param_names))}
    rank_fractions_by_param: dict[int, list[float]] = {j: [] for j in range(len(param_names))}

    for local_i, test_index in enumerate(eval_indices):
        x_condition = apply_x_transform(x_eval[local_i], transform)
        samples = sample_posterior_at_x(posterior, x_condition, num_posterior_samples, device)
        sample_count = int(samples.shape[0])
        n_params = min(samples.shape[1], theta_eval.shape[1], len(param_names))
        samples = samples[:, :n_params]
        theta_true = theta_eval[local_i, :n_params]

        if save_samples_dir is not None:
            save_samples_dir.mkdir(parents=True, exist_ok=True)
            np.save(
                save_samples_dir / f"{case}_N{int(n_train)}_idx{int(test_index)}_posterior_samples.npy",
                np.asarray(samples, dtype=np.float32),
            )

        for j in range(n_params):
            values = samples[:, j]
            truth = float(theta_true[j])
            rank = int(np.count_nonzero(values < truth))
            rank_fraction = float(rank / float(sample_count))
            ranks_by_param[j].append(rank)
            rank_fractions_by_param[j].append(rank_fraction)
            rank_rows.append(
                {
                    "case": case,
                    "n_train": int(n_train),
                    "run_dir": str(run_dir),
                    "test_index": int(test_index),
                    "param": param_names[j],
                    "param_index": int(j),
                    "theta_true": truth,
                    "rank": rank,
                    "rank_fraction": rank_fraction,
                    "num_posterior_samples": sample_count,
                    "x_rescale_mode": mode,
                    "posterior_mean": float(np.nanmean(values)),
                    "posterior_std": float(np.nanstd(values, ddof=1)),
                }
            )

    for j, param in enumerate(param_names):
        ranks = np.asarray(ranks_by_param[j], dtype=float)
        frac = np.asarray(rank_fractions_by_param[j], dtype=float)
        if ranks.size == 0:
            continue
        summary_rows.append(
            {
                "case": case,
                "n_train": int(n_train),
                "run_dir": str(run_dir),
                "param": param,
                "param_index": int(j),
                "n_sbc": int(ranks.size),
                "num_posterior_samples": int(num_posterior_samples),
                "rank_mean_fraction": float(np.nanmean(frac)),
                "rank_std_fraction": float(np.nanstd(frac, ddof=1)) if ranks.size > 1 else 0.0,
                "rank_min": int(np.nanmin(ranks)),
                "rank_max": int(np.nanmax(ranks)),
                "x_rescale_mode": mode,
            }
        )
    return rank_rows, summary_rows


def plot_sbc_rank_histograms(
    rank_rows: list[dict[str, Any]],
    *,
    case: str,
    n_train: int,
    param_names: list[str],
    output_path: Path,
    bins: int,
    dpi: int,
) -> None:
    rows = [row for row in rank_rows if row["case"] == case and int(row["n_train"]) == int(n_train)]
    if not rows:
        print(f"Skipping SBC plot for case={case}, N={int(n_train)}: no rank rows.")
        return

    n_cols = 3
    n_rows = int(math.ceil(len(param_names) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18.0 / 2.54, max(6.0, 5.4 * n_rows) / 2.54))
    axes = np.asarray(axes).reshape(-1)
    color = CASE_COLORS.get(case, "#1f77b4")

    for ax, param in zip(axes, param_names):
        sub = [row for row in rows if row["param"] == param]
        fractions = np.asarray([float(row["rank_fraction"]) for row in sub], dtype=float)
        fractions = fractions[np.isfinite(fractions)]
        if fractions.size == 0:
            ax.text(0.5, 0.5, "no ranks", ha="center", va="center", transform=ax.transAxes)
            ax.axis("off")
            continue
        counts, edges, _ = ax.hist(fractions, bins=int(bins), range=(0.0, 1.0), color=color, alpha=0.75)
        expected = fractions.size / float(bins)
        ax.axhline(expected, color="black", lw=0.8, ls=":", alpha=0.75)
        ax.set_title(PARAM_LABELS.get(param, param), pad=2.0)
        ax.set_xlim(0.0, 1.0)
        ax.set_xlabel(r"SBC rank fraction")
        ax.set_ylabel("count")
        ax.grid(True, axis="y", alpha=0.25, lw=0.5)

    for ax in axes[len(param_names) :]:
        ax.axis("off")

    fig.suptitle(rf"{CASE_LABELS.get(case, case)}, $N={int(n_train):,}$", y=0.995, fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    print(f"Saved {output_path}")


def plot_sbc_rank_mean_vs_n(
    summary_rows: list[dict[str, Any]],
    *,
    case: str,
    param_names: list[str],
    output_path: Path,
    dpi: int,
) -> None:
    rows = [row for row in summary_rows if row["case"] == case]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(11.0 / 2.54, 7.0 / 2.54))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(param_names), 1)))
    for j, param in enumerate(param_names):
        sub = sorted([row for row in rows if row["param"] == param], key=lambda row: int(row["n_train"]))
        if not sub:
            continue
        ax.plot(
            [int(row["n_train"]) for row in sub],
            [float(row["rank_mean_fraction"]) for row in sub],
            marker="o",
            ms=3.0,
            lw=1.0,
            color=colors[j],
            label=PARAM_LABELS.get(param, param),
        )
    ax.axhline(0.5, color="black", lw=0.8, ls=":", alpha=0.75)
    ax.set_xscale("log")
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel(r"Training set size")
    ax.set_ylabel(r"Mean SBC rank fraction")
    ax.set_title(CASE_LABELS.get(case, case), pad=3.0)
    ax.grid(True, which="both", alpha=0.25, lw=0.5)
    ax.legend(ncols=3, frameon=False, fontsize=6)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    print(f"Saved {output_path}")


def run_sbc(args: argparse.Namespace, root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    run_root = resolve_path(args.run_root, root)
    dataset_dir = resolve_path(args.case_dataset_dir, root)
    index_json = resolve_path(args.case_index_json, root) if args.case_index_json else dataset_dir / "case_dataset_index.json"
    dataset_sizes = parse_int_list(args.dataset_sizes)
    output_dir = resolve_path(args.output_dir, root)
    save_samples_dir = output_dir / "posterior_samples" if args.save_posterior_samples else None

    all_rank_rows: list[dict[str, Any]] = []
    all_summary_rows: list[dict[str, Any]] = []
    params_by_case: dict[str, list[str]] = {}

    for case in args.cases:
        dataset_path = find_case_dataset(case, dataset_dir, index_json)
        print(f"\nCase {case}: dataset={dataset_path}")
        with np.load(dataset_path, allow_pickle=True) as data:
            theta = np.asarray(data["theta"], dtype=np.float64)
            x = np.asarray(data["x"], dtype=np.float32)
            param_names = [str(v) for v in data["param_names"]]
            test_indices = choose_sbc_indices(
                data,
                last_n_test=int(args.last_n_test),
                max_sbc=int(args.max_sbc),
                seed=int(args.seed),
            )
        params_by_case[case] = param_names
        x_eval = np.ascontiguousarray(x[test_indices], dtype=np.float32)
        theta_eval = np.ascontiguousarray(theta[test_indices], dtype=np.float64)

        run_dirs = find_run_dirs(run_root, case, dataset_sizes, args.allow_missing)
        print(f"  Using N values: {list(run_dirs)}")
        print(f"  SBC simulations: {len(test_indices)} from indices {int(test_indices[0])}..{int(test_indices[-1])}")

        for n_train, run_dir in run_dirs.items():
            print(f"  N={n_train}: {run_dir}")
            posterior = load_posterior(run_dir)
            rank_rows, summary_rows = sbc_for_run(
                case=case,
                n_train=n_train,
                run_dir=run_dir,
                posterior=posterior,
                x_eval=x_eval,
                theta_eval=theta_eval,
                eval_indices=test_indices,
                param_names=param_names,
                num_posterior_samples=int(args.num_posterior_samples),
                device=args.device,
                save_samples_dir=save_samples_dir,
            )
            all_rank_rows.extend(rank_rows)
            all_summary_rows.extend(summary_rows)
            del posterior

    for case in args.cases:
        param_names = params_by_case.get(case, [])
        for n_train in sorted({int(row["n_train"]) for row in all_rank_rows if row["case"] == case}):
            plot_sbc_rank_histograms(
                all_rank_rows,
                case=case,
                n_train=n_train,
                param_names=param_names,
                output_path=output_dir / "jpg" / f"{case}_N{int(n_train)}_sbc_rank_histograms.jpg",
                bins=int(args.rank_bins),
                dpi=int(args.dpi),
            )
        plot_sbc_rank_mean_vs_n(
            all_summary_rows,
            case=case,
            param_names=param_names,
            output_path=output_dir / "jpg" / f"{case}_sbc_mean_rank_fraction_vs_n.jpg",
            dpi=int(args.dpi),
        )

    return all_rank_rows, all_summary_rows


def parse_args() -> argparse.Namespace:
    root = repo_root()
    parser = argparse.ArgumentParser(
        description=(
            "Run SBC diagnostics on saved SO SBI density estimators/posteriors. "
            "Uses held-out simulations from the prepared case datasets and each run's saved x_transform.npz."
        )
    )
    parser.add_argument(
        "--run-root",
        default=str(root / "SBI_analysis" / "outputs" / "cluster_outputs" / "SBI_SO_noise_dataset_size_ell80_7979_battaglia12_asinh"),
    )
    parser.add_argument(
        "--case-dataset-dir",
        default=str(root / "SBI_analysis" / "data_for_cluster" / "so_noise_sbi_cases_ell80_7979_battaglia12"),
    )
    parser.add_argument("--case-index-json", default="")
    parser.add_argument("--output-dir", default="", help="Default: <run-root>/sbc_last<N>_S<samples>.")
    parser.add_argument("--cases", nargs="+", default=DEFAULT_CASES)
    parser.add_argument("--dataset-sizes", default=",".join(str(v) for v in DEFAULT_DATASET_SIZES))
    parser.add_argument("--last-n-test", type=int, default=100)
    parser.add_argument("--max-sbc", type=int, default=100, help="Maximum held-out simulations used for SBC. Use <=0 for all selected.")
    parser.add_argument("--num-posterior-samples", type=int, default=2000)
    parser.add_argument("--rank-bins", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--save-posterior-samples", action="store_true")
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = repo_root()
    if not args.output_dir:
        args.output_dir = str(
            resolve_path(args.run_root, root)
            / f"sbc_last{int(args.last_n_test)}_max{int(args.max_sbc)}_S{int(args.num_posterior_samples)}"
        )
    output_dir = resolve_path(args.output_dir, root)
    output_dir.mkdir(parents=True, exist_ok=True)
    apply_paper_style()

    config = vars(args).copy()
    write_json(output_dir / "sbc_config.json", config)

    rank_rows, summary_rows = run_sbc(args, root)
    write_csv(
        output_dir / "sbc_ranks.csv",
        rank_rows,
        [
            "case",
            "n_train",
            "run_dir",
            "test_index",
            "param",
            "param_index",
            "theta_true",
            "rank",
            "rank_fraction",
            "num_posterior_samples",
            "x_rescale_mode",
            "posterior_mean",
            "posterior_std",
        ],
    )
    write_csv(
        output_dir / "sbc_summary.csv",
        summary_rows,
        [
            "case",
            "n_train",
            "run_dir",
            "param",
            "param_index",
            "n_sbc",
            "num_posterior_samples",
            "rank_mean_fraction",
            "rank_std_fraction",
            "rank_min",
            "rank_max",
            "x_rescale_mode",
        ],
    )
    print(f"Wrote SBC outputs to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
