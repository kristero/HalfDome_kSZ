#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import shutil
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

PARAM_COLORS = [
    "#1f77b4",
    "#d62728",
    "#2ca02c",
    "#9467bd",
    "#ff7f0e",
    "#17becf",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
]


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


def to_float(value: Any) -> float:
    return float(value)


def to_int(value: Any) -> int:
    return int(float(value))


def apply_style() -> None:
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
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "savefig.bbox": "tight",
            "savefig.dpi": 300,
        }
    )


def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(mask) < 2:
        return float("nan")
    x = x[mask]
    y = y[mask]
    if np.nanstd(x) == 0.0 or np.nanstd(y) == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def find_param_csvs(run_root: Path, cases: list[str], tag: str) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for case in cases:
        candidates = [
            run_root / f"diagnostics_{tag}_{case}" / f"{tag}_param_metrics.csv",
            run_root / f"diagnostics_{tag}" / f"{tag}_param_metrics.csv",
            run_root / "diagnostics_last100_split" / case / f"{tag}_param_metrics.csv",
        ]
        for path in candidates:
            if path.is_file():
                found[case] = path
                break

    # If a combined diagnostics CSV exists, all cases can read from it.
    combined = run_root / f"diagnostics_{tag}" / f"{tag}_param_metrics.csv"
    if combined.is_file():
        for case in cases:
            found.setdefault(case, combined)

    missing = [case for case in cases if case not in found]
    if missing:
        searched = "\n".join(str(run_root / f"diagnostics_{tag}_{case}" / f"{tag}_param_metrics.csv") for case in missing)
        raise FileNotFoundError(f"Missing param metrics CSVs for cases {missing}. Searched examples:\n{searched}")
    return found


def load_case_rows(csv_path: Path, case: str) -> list[dict[str, str]]:
    rows = read_csv(csv_path)
    if rows and "case" in rows[0]:
        rows = [row for row in rows if row.get("case") == case]
    return rows


def rows_for(rows: list[dict[str, str]], n_train: int, param: str) -> list[dict[str, str]]:
    return [
        row
        for row in rows
        if to_int(row["n_train"]) == int(n_train) and str(row["param"]) == str(param)
    ]


def param_order(rows: list[dict[str, str]]) -> list[str]:
    pairs = sorted({(to_int(row["param_index"]), str(row["param"])) for row in rows})
    return [param for _, param in pairs]


def plot_final_true_vs_mean(
    *,
    case: str,
    rows: list[dict[str, str]],
    final_n: int,
    output_dir: Path,
    dpi: int,
) -> list[dict[str, Any]]:
    params = param_order(rows)
    if not params:
        raise ValueError(f"No parameter rows found for case={case}")

    n_cols = 3
    n_rows = int(math.ceil(len(params) / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(18.0 / 2.54, max(6.0, 5.4 * n_rows) / 2.54),
    )
    axes = np.asarray(axes).reshape(-1)
    color = CASE_COLORS.get(case, "#1f77b4")
    summary_rows: list[dict[str, Any]] = []

    for ax, param in zip(axes, params):
        sub = rows_for(rows, final_n, param)
        x_true = np.asarray([to_float(row["theta_true"]) for row in sub], dtype=float)
        y_mean = np.asarray([to_float(row["posterior_mean"]) for row in sub], dtype=float)
        y_std = np.asarray([to_float(row["posterior_std"]) for row in sub], dtype=float)
        mask = np.isfinite(x_true) & np.isfinite(y_mean)
        x_true = x_true[mask]
        y_mean = y_mean[mask]
        y_std = y_std[mask]
        r = pearson_r(x_true, y_mean)

        summary_rows.append(
            {
                "case": case,
                "n_train": int(final_n),
                "param": param,
                "n_points": int(x_true.size),
                "pearson_r": r,
            }
        )

        if x_true.size == 0:
            ax.text(0.5, 0.5, "no points", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(PARAM_LABELS.get(param, param), pad=2.0)
            ax.axis("off")
            continue

        finite_std = np.isfinite(y_std) & (y_std > 0.0)
        if np.any(finite_std):
            ax.errorbar(
                x_true[finite_std],
                y_mean[finite_std],
                yerr=y_std[finite_std],
                fmt="none",
                ecolor=color,
                alpha=0.10,
                elinewidth=0.45,
                capsize=0.0,
                zorder=1,
            )
        ax.scatter(x_true, y_mean, s=13, color=color, alpha=0.72, edgecolor="none", zorder=2)

        lo = float(np.nanmin(np.concatenate([x_true, y_mean])))
        hi = float(np.nanmax(np.concatenate([x_true, y_mean])))
        pad = 0.06 * (hi - lo) if hi > lo else 0.1 * max(abs(hi), 1.0)
        lo -= pad
        hi += pad

        ax.plot([lo, hi], [lo, hi], color="black", lw=0.8, ls=":", zorder=0)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_title(rf"{PARAM_LABELS.get(param, param)}  $r={r:.2f}$", pad=2.0)
        ax.set_xlabel(r"True")
        ax.set_ylabel(r"Posterior mean")
        ax.grid(True, alpha=0.25, lw=0.5)

    for ax in axes[len(params) :]:
        ax.axis("off")

    fig.suptitle(rf"{CASE_LABELS.get(case, case)}, $N={int(final_n):,}$", y=0.995, fontsize=8)
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / f"{case}_true_vs_mean_correlations_N{int(final_n)}.jpg"
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    print(f"Saved {out}")
    return summary_rows


def plot_true_vs_mean_frame(
    *,
    case: str,
    rows: list[dict[str, str]],
    n_train: int,
    output_path: Path,
    dpi: int,
) -> None:
    params = param_order(rows)
    n_cols = 3
    n_rows = int(math.ceil(len(params) / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(18.0 / 2.54, max(6.0, 5.4 * n_rows) / 2.54),
    )
    axes = np.asarray(axes).reshape(-1)
    color = CASE_COLORS.get(case, "#1f77b4")

    for ax, param in zip(axes, params):
        sub = rows_for(rows, n_train, param)
        x_true = np.asarray([to_float(row["theta_true"]) for row in sub], dtype=float)
        y_mean = np.asarray([to_float(row["posterior_mean"]) for row in sub], dtype=float)
        mask = np.isfinite(x_true) & np.isfinite(y_mean)
        x_true = x_true[mask]
        y_mean = y_mean[mask]
        r = pearson_r(x_true, y_mean)

        if x_true.size == 0:
            ax.text(0.5, 0.5, "no points", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(PARAM_LABELS.get(param, param), pad=2.0)
            ax.axis("off")
            continue

        ax.scatter(x_true, y_mean, s=13, color=color, alpha=0.72, edgecolor="none", zorder=2)

        lo = float(np.nanmin(np.concatenate([x_true, y_mean])))
        hi = float(np.nanmax(np.concatenate([x_true, y_mean])))
        pad = 0.06 * (hi - lo) if hi > lo else 0.1 * max(abs(hi), 1.0)
        lo -= pad
        hi += pad

        ax.plot([lo, hi], [lo, hi], color="black", lw=0.8, ls=":", zorder=0)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_title(rf"{PARAM_LABELS.get(param, param)}  $r={r:.2f}$", pad=2.0)
        ax.set_xlabel(r"True")
        ax.set_ylabel(r"Posterior mean")
        ax.grid(True, alpha=0.25, lw=0.5)

    for ax in axes[len(params) :]:
        ax.axis("off")

    fig.suptitle(rf"{CASE_LABELS.get(case, case)}, $N={int(n_train):,}$", y=0.995, fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def make_true_vs_mean_gif(
    *,
    case: str,
    rows: list[dict[str, str]],
    dataset_sizes: list[int],
    output_dir: Path,
    dpi: int,
    duration: float,
) -> Path | None:
    try:
        import imageio.v2 as imageio
    except ModuleNotFoundError:
        print("imageio is not installed; skipping GIF creation.")
        return None

    frame_dir = output_dir / "gif_frames" / case
    if frame_dir.exists():
        shutil.rmtree(frame_dir)
    frame_dir.mkdir(parents=True, exist_ok=True)

    frame_paths = []
    for n_train in dataset_sizes:
        has_rows = any(to_int(row["n_train"]) == int(n_train) for row in rows)
        if not has_rows:
            print(f"Skipping GIF frame for case={case}, N={int(n_train)}: no rows.")
            continue
        frame_path = frame_dir / f"{case}_true_vs_mean_N{int(n_train):06d}.png"
        plot_true_vs_mean_frame(
            case=case,
            rows=rows,
            n_train=int(n_train),
            output_path=frame_path,
            dpi=dpi,
        )
        frame_paths.append(frame_path)

    if not frame_paths:
        print(f"No GIF frames written for case={case}; skipping GIF.")
        return None

    gif_path = output_dir / f"{case}_true_vs_mean_by_dataset_size.gif"
    frames = [imageio.imread(path) for path in frame_paths]
    imageio.mimsave(gif_path, frames, duration=float(duration), loop=0)
    print(f"Saved {gif_path}")
    return gif_path


def plot_correlation_vs_n(
    *,
    case: str,
    rows: list[dict[str, str]],
    dataset_sizes: list[int],
    output_dir: Path,
    dpi: int,
) -> list[dict[str, Any]]:
    params = param_order(rows)
    fig, ax = plt.subplots(figsize=(11.0 / 2.54, 7.0 / 2.54))
    summary_rows: list[dict[str, Any]] = []

    for i, param in enumerate(params):
        values = []
        n_values = []
        for n_train in dataset_sizes:
            sub = rows_for(rows, n_train, param)
            x_true = np.asarray([to_float(row["theta_true"]) for row in sub], dtype=float)
            y_mean = np.asarray([to_float(row["posterior_mean"]) for row in sub], dtype=float)
            r = pearson_r(x_true, y_mean)
            values.append(r)
            n_values.append(int(n_train))
            summary_rows.append(
                {
                    "case": case,
                    "n_train": int(n_train),
                    "param": param,
                    "n_points": int(np.count_nonzero(np.isfinite(x_true) & np.isfinite(y_mean))),
                    "pearson_r": r,
                }
            )

        ax.plot(
            n_values,
            values,
            marker="o",
            ms=3.0,
            lw=1.0,
            color=PARAM_COLORS[i % len(PARAM_COLORS)],
            label=PARAM_LABELS.get(param, param),
        )

    ax.axhline(1.0, color="black", lw=0.7, ls=":", alpha=0.6)
    ax.axhline(0.0, color="black", lw=0.7, ls="--", alpha=0.35)
    ax.set_xscale("log")
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel(r"Training set size")
    ax.set_ylabel(r"Pearson correlation $r(\theta_{\rm true}, \bar{\theta}_{\rm post})$")
    ax.set_title(CASE_LABELS.get(case, case), pad=3.0)
    ax.grid(True, which="both", alpha=0.25, lw=0.5)
    ax.legend(ncols=3, frameon=False, fontsize=6)
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / f"{case}_correlation_vs_dataset_size_by_param.jpg"
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    print(f"Saved {out}")
    return summary_rows


def parse_args() -> argparse.Namespace:
    root = repo_root()
    parser = argparse.ArgumentParser(
        description=(
            "Make true-vs-posterior-mean correlation plots from existing SO last-N diagnostics CSVs."
        )
    )
    parser.add_argument(
        "--run-root",
        default=str(
            root
            / "SBI_analysis"
            / "outputs"
            / "cluster_outputs"
            / "SBI_SO_noise_dataset_size_ell80_7979_battaglia12_asinh"
        ),
    )
    parser.add_argument("--tag", default="last100")
    parser.add_argument("--cases", nargs="+", default=DEFAULT_CASES)
    parser.add_argument(
        "--dataset-sizes",
        default=",".join(str(v) for v in DEFAULT_DATASET_SIZES),
    )
    parser.add_argument(
        "--final-n",
        type=int,
        default=32600,
        help="Training size used for the true-vs-mean scatter panels. Use 32000 if that is your saved N.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Default: <run-root>/diagnostics_<tag>_correlations",
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--gif-dpi", type=int, default=140)
    parser.add_argument("--gif-duration", type=float, default=0.9, help="Seconds per GIF frame.")
    parser.add_argument("--no-gif", action="store_true", help="Disable true-vs-mean GIF creation.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = repo_root()
    run_root = resolve_path(args.run_root, root)
    output_dir = resolve_path(args.output_dir, root) if args.output_dir else run_root / f"diagnostics_{args.tag}_correlations"
    output_dir.mkdir(parents=True, exist_ok=True)
    apply_style()

    dataset_sizes = parse_int_list(args.dataset_sizes)
    csvs = find_param_csvs(run_root, list(args.cases), args.tag)

    all_final_rows: list[dict[str, Any]] = []
    all_corr_rows: list[dict[str, Any]] = []
    for case in args.cases:
        rows = load_case_rows(csvs[case], case)
        if not rows:
            print(f"Warning: no rows for case={case} in {csvs[case]}")
            continue
        all_final_rows.extend(
            plot_final_true_vs_mean(
                case=case,
                rows=rows,
                final_n=args.final_n,
                output_dir=output_dir,
                dpi=args.dpi,
            )
        )
        all_corr_rows.extend(
            plot_correlation_vs_n(
                case=case,
                rows=rows,
                dataset_sizes=dataset_sizes,
                output_dir=output_dir,
                dpi=args.dpi,
            )
        )
        if not args.no_gif:
            make_true_vs_mean_gif(
                case=case,
                rows=rows,
                dataset_sizes=dataset_sizes,
                output_dir=output_dir,
                dpi=args.gif_dpi,
                duration=args.gif_duration,
            )

    write_csv(
        output_dir / f"{args.tag}_final_N{int(args.final_n)}_correlations.csv",
        all_final_rows,
        ["case", "n_train", "param", "n_points", "pearson_r"],
    )
    write_csv(
        output_dir / f"{args.tag}_correlation_vs_dataset_size.csv",
        all_corr_rows,
        ["case", "n_train", "param", "n_points", "pearson_r"],
    )
    print(f"Wrote correlation plots to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
