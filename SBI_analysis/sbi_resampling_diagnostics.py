from __future__ import annotations

import csv
import math
import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_LATEX_LABELS = [
    r"P_0",
    r"x_{\rm c}",
    r"\beta",
    r"\alpha_{m,P_0}",
    r"\alpha_{m,x_{\rm c}}",
    r"\alpha_{m,\beta}",
    r"\alpha_{z,P_0}",
    r"\alpha_{z,x_{\rm c}}",
    r"\alpha_{z,\beta}",
]


def to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def safe_label(value: str | Path) -> str:
    text = str(value)
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("_")[:120] or "xo"


def load_x_o(source: str | Path, key: str | None = None) -> np.ndarray:
    source = Path(source).expanduser()
    if not source.exists():
        raise FileNotFoundError(f"x_o source does not exist: {source}")

    if source.suffix == ".npy":
        arr = np.load(source, allow_pickle=False)
    elif source.suffix == ".npz":
        with np.load(source, allow_pickle=True) as data:
            if key is None:
                preferred = ("obs", "x_o", "xo", "x_obs", "prepared_obs", "observation")
                matches = [name for name in preferred if name in data.files]
                if not matches:
                    raise KeyError(
                        f"No observation key found in {source}. "
                        f"Available keys: {data.files}. Pass X_O_KEY explicitly."
                    )
                key = matches[0]
            arr = data[key]
    else:
        raise ValueError(f"Unsupported x_o file type: {source.suffix}. Use .npy or .npz.")

    arr = np.asarray(arr, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"x_o loaded from {source} is empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"x_o loaded from {source} has non-finite values")
    return arr


def resolve_run_by_n(runs: list[dict[str, Any]], n_train: int) -> dict[str, Any]:
    matches = [run for run in runs if int(run["n_train"]) == int(n_train)]
    if not matches:
        available = sorted({int(run["n_train"]) for run in runs})
        raise KeyError(f"No run found for n_train={n_train}. Available: {available}")
    if len(matches) > 1:
        matches = sorted(matches, key=lambda run: str(run["run_dir"]))
    return matches[0]


def run_label(run: dict[str, Any]) -> str:
    group = run.get("group", "")
    job = run.get("job_id", "")
    suffix = f", {group}" if group else ""
    if job:
        suffix += f", {job}"
    return f"N={int(run['n_train'])}{suffix}"


def sample_saved_nn_posterior(
    run: dict[str, Any],
    x_o: np.ndarray,
    *,
    num_samples: int,
    device: str = "cpu",
    seed: int | None = 12345,
    output_dir: str | Path | None = None,
    output_label: str = "xo",
) -> tuple[np.ndarray, Path | None]:
    import torch

    posterior_path = Path(run["run_dir"]) / "posterior.pkl"
    if not posterior_path.exists():
        raise FileNotFoundError(f"Missing saved posterior object: {posterior_path}")

    x_o = np.asarray(x_o, dtype=np.float32).reshape(-1)
    expected_x_dim = run.get("metadata", {}).get("x_dim")
    if expected_x_dim is not None and int(expected_x_dim) != int(x_o.size):
        raise ValueError(
            f"x_o has length {x_o.size}, but {run_label(run)} expects x_dim={expected_x_dim}. "
            "Use an observation vector made with the same binning/data-vector as the training run."
        )

    if seed is not None:
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))

    with posterior_path.open("rb") as handle:
        posterior = pickle.load(handle)

    x_t = torch.as_tensor(x_o, dtype=torch.float32, device=device)
    with torch.no_grad():
        try:
            samples = posterior.sample((int(num_samples),), x=x_t)
        except TypeError:
            if not hasattr(posterior, "set_default_x"):
                raise
            posterior_with_x = posterior.set_default_x(x_t)
            if posterior_with_x is None:
                posterior_with_x = posterior
            samples = posterior_with_x.sample((int(num_samples),))

    samples_np = np.asarray(to_numpy(samples), dtype=np.float32)
    if samples_np.ndim != 2:
        raise ValueError(f"Expected posterior samples with shape (n_samples, n_params), got {samples_np.shape}")

    save_path = None
    if output_dir is not None:
        output_dir = Path(output_dir).expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)
        save_path = output_dir / f"resampled_{safe_label(output_label)}_N{int(run['n_train'])}.npy"
        np.save(save_path, samples_np)

    return samples_np, save_path


def summarize_posterior_samples(
    samples: np.ndarray,
    *,
    n_train: int,
    param_names: list[str],
    theta_true: np.ndarray | None = None,
    label: str = "",
) -> list[dict[str, Any]]:
    samples = np.asarray(samples, dtype=float)
    if samples.ndim != 2:
        raise ValueError(f"Expected samples shape (n_samples, n_params), got {samples.shape}")

    n_params = min(samples.shape[1], len(param_names))
    theta_true = None if theta_true is None else np.asarray(theta_true, dtype=float).reshape(-1)
    rows = []
    for index, parameter in enumerate(param_names[:n_params]):
        values = samples[:, index]
        mean = float(np.mean(values))
        median = float(np.median(values))
        std = float(np.std(values, ddof=1))
        true = np.nan if theta_true is None or index >= theta_true.size else float(theta_true[index])
        rows.append(
            {
                "n_train": int(n_train),
                "label": label,
                "parameter": parameter,
                "true": true,
                "mean": mean,
                "median": median,
                "std": std,
                "mean_minus_true": float(mean - true) if np.isfinite(true) else np.nan,
                "median_minus_true": float(median - true) if np.isfinite(true) else np.nan,
                "pull_mean": float((mean - true) / std) if std > 0 and np.isfinite(true) else np.nan,
                "pull_median": float((median - true) / std) if std > 0 and np.isfinite(true) else np.nan,
                "q16": float(np.percentile(values, 16)),
                "q84": float(np.percentile(values, 84)),
                "q2.5": float(np.percentile(values, 2.5)),
                "q97.5": float(np.percentile(values, 97.5)),
            }
        )
    return rows


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def plot_resampled_mean_offsets(
    summary_rows: list[dict[str, Any]],
    param_names: list[str],
    *,
    metric: str = "mean_minus_true",
    ylabel: str = "posterior mean - true",
    xscale: str = "log",
    output_path: str | Path | None = None,
):
    import matplotlib.pyplot as plt

    ncols = 3
    nrows = int(math.ceil(len(param_names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.05 * nrows), sharex=True)
    axes = np.asarray(axes).ravel()

    for ax, parameter in zip(axes, param_names):
        rows = [row for row in summary_rows if row["parameter"] == parameter and np.isfinite(row.get(metric, np.nan))]
        rows = sorted(rows, key=lambda row: row["n_train"])
        if rows:
            x = np.array([row["n_train"] for row in rows], dtype=float)
            y = np.array([row[metric] for row in rows], dtype=float)
            ax.plot(x, y, marker="o", linewidth=1.8, markersize=4.5)
        ax.axhline(0.0, color="black", linewidth=0.9, alpha=0.65)
        ax.set_title(parameter)
        ax.set_xscale(xscale)
        ax.set_xlabel("number of datapoints")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", alpha=0.25)

    for ax in axes[len(param_names) :]:
        ax.axis("off")

    fig.tight_layout()
    if output_path is not None:
        output_path = Path(output_path).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved {output_path}")
    return fig


def load_existing_or_resampled_samples(
    n_train: int,
    runs: list[dict[str, Any]],
    resampled_samples_by_n: dict[int, np.ndarray] | None = None,
    *,
    prefer_resampled: bool = True,
) -> tuple[np.ndarray, str]:
    if prefer_resampled and resampled_samples_by_n is not None and int(n_train) in resampled_samples_by_n:
        return np.asarray(resampled_samples_by_n[int(n_train)]), f"resampled N={int(n_train)}"
    run = resolve_run_by_n(runs, int(n_train))
    samples_path = Path(run["posterior_samples_path"])
    if not samples_path.exists():
        samples_path = Path(run["run_dir"]) / "posterior_samples.npy"
    if not samples_path.exists():
        raise FileNotFoundError(f"No posterior samples found for {run_label(run)}")
    return np.load(samples_path, mmap_mode="r"), f"saved N={int(n_train)}"


def plot_getdist_triangle(
    samples_by_label: dict[str, np.ndarray],
    *,
    theta_true: np.ndarray | None = None,
    labels: list[str] | None = None,
    output_path: str | Path | None = None,
    width_inch: float = 18.0 / 2.54,
    filled: bool = True,
):
    import matplotlib.pyplot as plt
    from getdist import MCSamples, plots

    arrays = [np.asarray(to_numpy(samples)) for samples in samples_by_label.values()]
    if not arrays:
        raise ValueError("samples_by_label is empty")
    n_params = min(array.shape[1] for array in arrays)
    if labels is None:
        labels = DEFAULT_LATEX_LABELS
    n_params = min(n_params, len(labels))

    names = [f"p{i}" for i in range(n_params)]
    gd_samples = []
    legend_labels = []
    for legend_label, samples in samples_by_label.items():
        samples_np = np.asarray(to_numpy(samples))[:, :n_params]
        gd = MCSamples(samples=samples_np, names=names, labels=labels[:n_params], label=legend_label)
        gd.updateSettings(
            {
                "smooth_scale_1D": 0.3,
                "smooth_scale_2D": 0.3,
                "fine_bins": 2048,
                "fine_bins_2D": 1024,
            }
        )
        gd_samples.append(gd)
        legend_labels.append(legend_label)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
        }
    )

    g = plots.get_subplot_plotter(width_inch=width_inch)
    g.settings.axes_fontsize = 7
    g.settings.lab_fontsize = 8
    g.settings.legend_fontsize = 8
    g.settings.alpha_filled_add = 0.55
    g.settings.linewidth = 1.0
    g.settings.num_plot_contours = 2
    g.settings.figure_legend_frame = False
    g.settings.scaling = False

    marker_values = None
    if theta_true is not None:
        theta_true = np.asarray(theta_true, dtype=float).reshape(-1)
        marker_values = {names[index]: float(theta_true[index]) for index in range(min(n_params, theta_true.size))}

    g.triangle_plot(
        gd_samples,
        params=names,
        filled=filled,
        legend_labels=legend_labels,
        contour_lws=[0.9 for _ in gd_samples],
        diag1d_kwargs={"linewidth": 1.0},
        markers=marker_values,
        marker_args={
            "color": "black",
            "lw": 0.8,
            "ls": "--",
        },
    )

    if output_path is not None:
        output_path = Path(output_path).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved {output_path}")
    return g
