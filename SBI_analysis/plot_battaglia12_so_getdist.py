#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BATTAGLIA12_THETA_BY_NAME = {
    "P0": 18.1,
    "xc": 0.497,
    "beta": 4.35,
    "alpha_m_P0": 0.154,
    "alpha_m_xc": -0.00865,
    "alpha_m_beta": 0.0393,
    "alpha_z_P0": -0.758,
    "alpha_z_xc": 0.731,
    "alpha_z_beta": 0.415,
}

BATTAGLIA12_FILENAMES = {
    "no_noise": (
        "halfdome_fullsky_masked_no_noise_cl_m200c_nside4096_base_cosmo_fid_"
        "gaussbeam_2p0arcmin_so_fsky0p4_apo60p0arcmin_seed12345_deproj0_lmax7979.npy"
    ),
    "goal_deproj0": (
        "halfdome_fullsky_masked_goal_noise_cross_cl_m200c_nside4096_base_cosmo_fid_"
        "gaussbeam_2p0arcmin_so_fsky0p4_apo60p0arcmin_seed12345_deproj0_lmax7979.npy"
    ),
    "baseline_deproj0": (
        "halfdome_fullsky_masked_baseline_noise_cross_cl_m200c_nside4096_base_cosmo_fid_"
        "gaussbeam_2p0arcmin_so_fsky0p4_apo60p0arcmin_seed12345_deproj0_lmax7979.npy"
    ),
    "goal_deproj2": (
        "halfdome_fullsky_masked_goal_noise_cross_cl_m200c_nside4096_base_cosmo_fid_"
        "gaussbeam_2p0arcmin_so_fsky0p4_apo60p0arcmin_seed12345_deproj2_lmax7979.npy"
    ),
    "baseline_deproj2": (
        "halfdome_fullsky_masked_baseline_noise_cross_cl_m200c_nside4096_base_cosmo_fid_"
        "gaussbeam_2p0arcmin_so_fsky0p4_apo60p0arcmin_seed12345_deproj2_lmax7979.npy"
    ),
}

CASE_TO_BATTAGLIA_KEY = {
    "no_noise": "no_noise",
    "masked_no_noise": "no_noise",
    "goal_deproj0": "goal_deproj0",
    "masked_goal_noise_cross_deproj0": "goal_deproj0",
    "baseline_deproj0": "baseline_deproj0",
    "masked_baseline_noise_cross_deproj0": "baseline_deproj0",
    "goal_deproj2": "goal_deproj2",
    "masked_goal_noise_cross_deproj2": "goal_deproj2",
    "baseline_deproj2": "baseline_deproj2",
    "masked_baseline_noise_cross_deproj2": "baseline_deproj2",
}

LABEL_BY_NAME = {
    "P0": r"P_0",
    "xc": r"x_{\rm c}",
    "beta": r"\beta",
    "alpha_m_P0": r"\alpha_{m,P_0}",
    "alpha_m_xc": r"\alpha_{m,x_{\rm c}}",
    "alpha_m_beta": r"\alpha_{m,\beta}",
    "alpha_z_P0": r"\alpha_{z,P_0}",
    "alpha_z_xc": r"\alpha_{z,x_{\rm c}}",
    "alpha_z_beta": r"\alpha_{z,\beta}",
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_path(path: str | Path, root: Path) -> Path:
    path = Path(path).expanduser()
    return path if path.is_absolute() else root / path


def scalar_string(value: Any, default: str = "") -> str:
    arr = np.asarray(value)
    if arr.shape == ():
        return str(arr.item())
    if arr.size == 1:
        return str(arr.reshape(()).item())
    return default


def parse_int_list(value: str) -> list[int]:
    return [int(float(part.replace("_", ""))) for part in str(value).replace(",", " ").split() if part]


def n_sort_key(path: Path) -> int:
    match = re.fullmatch(r"N(\d+)", path.name)
    return int(match.group(1)) if match else 999999999


def load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def load_posterior(run_dir: Path) -> Any:
    inference_path = run_dir / "inference.pkl"
    density_path = run_dir / "density_estimator.pkl"
    if inference_path.is_file() and density_path.is_file():
        inference = load_pickle(inference_path)
        density_estimator = load_pickle(density_path)
        return inference.build_posterior(density_estimator)

    posterior_path = run_dir / "posterior.pkl"
    if posterior_path.is_file():
        return load_pickle(posterior_path)

    raise FileNotFoundError(f"No posterior.pkl or inference.pkl+density_estimator.pkl found in {run_dir}")


def find_run_dir(run_root: Path, case: str, n_train: int) -> Path:
    case_root = run_root / case
    if not case_root.is_dir():
        raise FileNotFoundError(f"Case output directory not found: {case_root}")

    matches = []
    for path in sorted(case_root.glob("**/N*"), key=n_sort_key):
        if path.is_dir() and path.name == f"N{int(n_train)}":
            if (path / "posterior.pkl").is_file() or ((path / "inference.pkl").is_file() and (path / "density_estimator.pkl").is_file()):
                matches.append(path)
    if not matches:
        raise FileNotFoundError(f"Could not find completed run for case={case}, N={int(n_train)} under {case_root}")
    if len(matches) > 1:
        raise ValueError(f"Multiple completed runs for case={case}, N={int(n_train)}: {matches}")
    return matches[0]


def available_n_values(run_root: Path, case: str) -> list[int]:
    case_root = run_root / case
    if not case_root.is_dir():
        return []
    values: list[int] = []
    for path in sorted(case_root.glob("**/N*"), key=n_sort_key):
        if not path.is_dir():
            continue
        match = re.fullmatch(r"N(\d+)", path.name)
        if match is None:
            continue
        if (path / "posterior.pkl").is_file() or ((path / "inference.pkl").is_file() and (path / "density_estimator.pkl").is_file()):
            values.append(int(match.group(1)))
    return sorted(set(values))


def choose_true_vs_mean_n(run_root: Path, case: str, requested_n_values: list[int], explicit_n: int) -> int:
    if int(explicit_n) > 0:
        return int(explicit_n)
    if len(set(requested_n_values)) >= 2:
        return sorted(set(requested_n_values))[-2]
    available = available_n_values(run_root, case)
    if len(available) < 2:
        raise ValueError(
            f"Need at least two completed N values to select the second-largest run for case={case}. "
            f"Found: {available}"
        )
    return available[-2]


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


def bin_weights(ell_values: np.ndarray, weighting: str) -> np.ndarray:
    ell_values = np.asarray(ell_values, dtype=np.float64)
    weighting = str(weighting or "2ell_plus_1").lower()
    if weighting in {"uniform", "none", "flat"}:
        return np.ones_like(ell_values, dtype=np.float64)
    if weighting == "ell":
        return ell_values
    if weighting in {"2ell_plus_1", "modes", "mode_count"}:
        return 2.0 * ell_values + 1.0
    raise ValueError(f"Unsupported bin weighting: {weighting!r}")


def make_dl_bin_matrix(ell_unbinned: np.ndarray, bin_ell_min: np.ndarray, bin_ell_max: np.ndarray, weighting: str) -> np.ndarray:
    ell_unbinned = np.asarray(ell_unbinned, dtype=np.float64).reshape(-1)
    matrix = np.zeros((ell_unbinned.size, len(bin_ell_min)), dtype=np.float64)
    for i, (lo, hi) in enumerate(zip(bin_ell_min, bin_ell_max)):
        idx = np.flatnonzero((ell_unbinned >= float(lo)) & (ell_unbinned <= float(hi)))
        if idx.size == 0:
            raise ValueError(f"No ell values found for bin {lo}-{hi}")
        weights = bin_weights(ell_unbinned[idx], weighting)
        matrix[idx, i] = weights / np.sum(weights)
    dl_factor = ell_unbinned * (ell_unbinned + 1.0) / (2.0 * np.pi)
    return dl_factor[:, None] * matrix


def read_profile(path: Path) -> np.ndarray:
    arr = np.load(path)
    arr = np.asarray(arr)
    if arr.ndim == 1:
        return np.ascontiguousarray(arr, dtype=np.float32)
    if arr.ndim == 2 and 1 in arr.shape:
        return np.ascontiguousarray(arr.reshape(-1), dtype=np.float32)
    if arr.ndim == 2 and arr.shape[1] >= 2:
        return np.ascontiguousarray(arr[:, -1], dtype=np.float32)
    raise ValueError(f"Cannot interpret profile array from {path}: shape={arr.shape}")


def battaglia12_theta(param_names: list[str]) -> np.ndarray:
    return np.asarray([BATTAGLIA12_THETA_BY_NAME[name] for name in param_names], dtype=np.float32)


def battaglia12_obs_from_case_dataset(
    case: str,
    dataset_path: Path,
    battaglia12_dir: Path,
    battaglia_profile_path: Path | None,
    bin_weighting: str,
) -> tuple[np.ndarray, np.ndarray, list[str], str]:
    with np.load(dataset_path, allow_pickle=True) as data:
        param_names = [str(v) for v in data["param_names"]]
        theta_true = battaglia12_theta(param_names)

        if battaglia_profile_path is None and "obs_source" in data.files and scalar_string(data["obs_source"]) == "battaglia12":
            return (
                np.asarray(data["obs"], dtype=np.float32).reshape(-1),
                theta_true,
                param_names,
                scalar_string(data["obs_profile_path"], "stored battaglia12 obs"),
            )

        if "ell_unbinned" not in data.files:
            raise KeyError(
                f"{dataset_path} does not contain ell_unbinned, so I cannot bin an external Battaglia12 C_ell profile. "
                "Use a case dataset with obs_source=battaglia12 or provide one containing ell_unbinned/bin_ell_min/bin_ell_max."
            )
        ell_unbinned = np.asarray(data["ell_unbinned"], dtype=np.float32)
        bin_ell_min = np.asarray(data["bin_ell_min"], dtype=np.float32)
        bin_ell_max = np.asarray(data["bin_ell_max"], dtype=np.float32)

    if battaglia_profile_path is None:
        battaglia_key = CASE_TO_BATTAGLIA_KEY.get(case)
        if battaglia_key is None:
            raise KeyError(f"No default Battaglia12 profile mapping for case={case!r}; pass --battaglia-profile-path.")
        battaglia_profile_path = battaglia12_dir / BATTAGLIA12_FILENAMES[battaglia_key]

    if not battaglia_profile_path.is_file():
        raise FileNotFoundError(f"Battaglia12 C_ell profile not found: {battaglia_profile_path}")

    cl = read_profile(battaglia_profile_path)
    if cl.size != ell_unbinned.size:
        raise ValueError(f"{battaglia_profile_path} has length {cl.size}; expected {ell_unbinned.size}.")
    dl_bin_matrix = make_dl_bin_matrix(ell_unbinned, bin_ell_min, bin_ell_max, bin_weighting)
    obs = np.asarray(cl, dtype=np.float64) @ dl_bin_matrix
    return np.ascontiguousarray(obs, dtype=np.float32), theta_true, param_names, str(battaglia_profile_path)


def plot_dell_density_with_reference(
    *,
    dataset_path: Path,
    reference_dell: np.ndarray,
    output_path: Path,
    title: str,
    max_rows: int,
    n_y_bins: int,
    percentile_range: tuple[float, float],
    dpi: int,
) -> None:
    from matplotlib.colors import LogNorm

    with np.load(dataset_path, allow_pickle=True) as data:
        x = np.asarray(data["x"], dtype=np.float32)
        ell = np.asarray(data["ell"], dtype=np.float32) if "ell" in data.files else np.arange(x.shape[1], dtype=np.float32)

    if x.ndim != 2:
        raise ValueError(f"Expected dataset x to be 2D, got {x.shape} in {dataset_path}")
    if ell.size != x.shape[1]:
        raise ValueError(f"ell length {ell.size} does not match x dimension {x.shape[1]} in {dataset_path}")

    if int(max_rows) > 0 and int(max_rows) < x.shape[0]:
        rng = np.random.default_rng(12345)
        row_idx = np.sort(rng.choice(x.shape[0], size=int(max_rows), replace=False))
        x_plot = np.asarray(x[row_idx], dtype=np.float32)
    else:
        x_plot = x

    reference_dell = np.asarray(reference_dell, dtype=np.float32).reshape(-1)
    if reference_dell.size != x.shape[1]:
        raise ValueError(f"Reference D_ell length {reference_dell.size} does not match x dimension {x.shape[1]}")

    finite_values = x_plot[np.isfinite(x_plot)]
    if finite_values.size == 0:
        raise ValueError(f"No finite D_ell values found in {dataset_path}")

    p_low, p_high = percentile_range
    y_low, y_high = np.nanpercentile(finite_values, [float(p_low), float(p_high)])
    ref_finite = reference_dell[np.isfinite(reference_dell)]
    if ref_finite.size:
        y_low = min(float(y_low), float(np.nanmin(ref_finite)))
        y_high = max(float(y_high), float(np.nanmax(ref_finite)))
    if not np.isfinite(y_low) or not np.isfinite(y_high) or y_low == y_high:
        center = 0.0 if not np.isfinite(y_low) else float(y_low)
        pad = max(abs(center), 1.0) * 0.1
        y_low = center - pad
        y_high = center + pad

    y_edges = np.linspace(float(y_low), float(y_high), int(n_y_bins) + 1, dtype=np.float64)
    density = np.zeros((int(n_y_bins), x.shape[1]), dtype=np.float64)
    for i in range(x.shape[1]):
        vals = np.asarray(x_plot[:, i], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        hist, _ = np.histogram(vals, bins=y_edges, density=True)
        density[:, i] = hist

    nonzero = density[density > 0.0]
    if nonzero.size:
        vmin = max(float(np.nanpercentile(nonzero, 1.0)), np.finfo(float).tiny)
        vmax = float(np.nanpercentile(nonzero, 99.5))
        norm = LogNorm(vmin=vmin, vmax=max(vmax, vmin * 1.01))
    else:
        norm = None

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "savefig.bbox": "tight",
        }
    )

    fig, ax = plt.subplots(figsize=(8.8 / 2.54, 6.2 / 2.54))
    mesh = ax.pcolormesh(
        ell,
        y_edges,
        density,
        shading="auto",
        cmap="magma",
        norm=norm,
    )
    ax.plot(ell, reference_dell, color="cyan", lw=1.15, label="Battaglia12", zorder=3)
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(r"$D_\ell$")
    ax.set_title(title, pad=2.0)
    ax.grid(True, alpha=0.18, lw=0.4)
    ax.legend(frameon=False, loc="best")
    colorbar = fig.colorbar(mesh, ax=ax, pad=0.02)
    colorbar.set_label(r"Density")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=int(dpi))
    plt.close(fig)
    print(f"Saved {output_path}")


def load_x_transform(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "x_transform.npz"
    if not path.is_file():
        return {"mode": "none", "path": ""}
    with np.load(path, allow_pickle=True) as data:
        out = {key: np.asarray(data[key]).copy() for key in data.files}
    out["mode"] = scalar_string(out.get("mode", "none"), "none")
    out["path"] = str(path)
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
    raise ValueError(f"Unsupported x transform mode: {mode}")


def sample_posterior_at_x(posterior: Any, x_obs: np.ndarray, num_samples: int, device: str) -> np.ndarray:
    import torch

    x_t = torch.as_tensor(np.asarray(x_obs, dtype=np.float32), dtype=torch.float32, device=device)
    posterior_x = posterior
    if hasattr(posterior, "set_default_x"):
        try:
            maybe_posterior = posterior.set_default_x(x_t)
            if maybe_posterior is not None:
                posterior_x = maybe_posterior
        except Exception:
            posterior_x = posterior

    try:
        samples = posterior_x.sample((int(num_samples),), x=x_t, show_progress_bars=False)
    except TypeError:
        try:
            samples = posterior_x.sample((int(num_samples),), x=x_t)
        except TypeError:
            try:
                samples = posterior_x.sample((int(num_samples),), show_progress_bars=False)
            except TypeError:
                samples = posterior_x.sample((int(num_samples),))
    if torch.is_tensor(samples):
        samples = samples.detach().cpu().numpy()
    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim == 1:
        samples = samples.reshape(1, -1)
    elif samples.ndim > 2:
        samples = samples.reshape(-1, samples.shape[-1])
    return samples


def plot_getdist(
    sample_sets: list[dict[str, Any]],
    param_names: list[str],
    theta_true: np.ndarray,
    output_path: Path,
    filled_last_only: bool,
    dpi: int,
) -> None:
    from getdist import MCSamples, plots

    n_params = min([samples["samples"].shape[1] for samples in sample_sets] + [len(param_names), theta_true.size])
    names = [f"p{i}" for i in range(n_params)]
    labels = [LABEL_BY_NAME.get(name, name) for name in param_names[:n_params]]

    gd_samples = []
    for item in sample_sets:
        gd = MCSamples(
            samples=np.asarray(item["samples"], dtype=float)[:, :n_params],
            names=names,
            labels=labels,
            label=item["label"],
        )
        gd.updateSettings(
            {
                "smooth_scale_1D": 0.3,
                "smooth_scale_2D": 0.3,
                "fine_bins": 2048,
                "fine_bins_2D": 1024,
            }
        )
        gd_samples.append(gd)

    colors = [
        "#1f77b4",
        "#d62728",
        "#2ca02c",
        "#9467bd",
        "#ff7f0e",
        "#17becf",
        "#8c564b",
        "#e377c2",
        "#222222",
    ]

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
        }
    )

    g = plots.get_subplot_plotter(width_inch=18.0 / 2.54)
    g.settings.axes_fontsize = 7
    g.settings.lab_fontsize = 8
    g.settings.legend_fontsize = 7
    g.settings.alpha_filled_add = 0.32
    g.settings.linewidth = 1.0
    g.settings.num_plot_contours = 2
    g.settings.figure_legend_frame = False
    g.settings.scaling = False

    line_args = [{"color": colors[i % len(colors)], "lw": 1.0} for i in range(len(gd_samples))]
    if filled_last_only and len(gd_samples) > 1:
        g.triangle_plot(
            gd_samples[:-1],
            params=names,
            filled=False,
            legend_labels=[item["label"] for item in sample_sets[:-1]],
            contour_colors=colors[: len(gd_samples) - 1],
            line_args=line_args[:-1],
            diag1d_kwargs={"linestyle": "--", "linewidth": 0.9},
            markers=theta_true[:n_params],
            marker_args={"color": "black", "lw": 0.8, "ls": ":"},
        )
        g.triangle_plot(
            [gd_samples[-1]],
            params=names,
            filled=True,
            legend_labels=[sample_sets[-1]["label"]],
            contour_colors=[colors[(len(gd_samples) - 1) % len(colors)]],
            line_args=[{"color": colors[(len(gd_samples) - 1) % len(colors)], "lw": 1.35}],
            diag1d_kwargs={"linestyle": "-", "linewidth": 1.2},
            markers=theta_true[:n_params],
            marker_args={"color": "black", "lw": 0.8, "ls": ":"},
        )
    else:
        g.triangle_plot(
            gd_samples,
            params=names,
            filled=True,
            legend_labels=[item["label"] for item in sample_sets],
            contour_colors=colors[: len(gd_samples)],
            line_args=line_args,
            markers=theta_true[:n_params],
            marker_args={"color": "black", "lw": 0.8, "ls": ":"},
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi)
    plt.close("all")
    print(f"Saved {output_path}")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def last_n_dataset_profiles(dataset_path: Path, last_n: int) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    with np.load(dataset_path, allow_pickle=True) as data:
        x = np.asarray(data["x"], dtype=np.float32)
        theta = np.asarray(data["theta"], dtype=np.float32)
        param_names = [str(v) for v in data["param_names"]]
    if int(last_n) <= 0:
        raise ValueError("--true-vs-mean-last-n must be positive")
    if int(last_n) > x.shape[0]:
        raise ValueError(f"Requested last_n={last_n}, but dataset has only {x.shape[0]} rows")
    indices = np.arange(x.shape[0] - int(last_n), x.shape[0], dtype=np.int64)
    return (
        np.ascontiguousarray(x[indices], dtype=np.float32),
        np.ascontiguousarray(theta[indices], dtype=np.float32),
        param_names,
        indices,
    )


def train_eval_overlap(transform: dict[str, Any], eval_indices: np.ndarray) -> np.ndarray:
    if "train_indices" not in transform:
        return np.empty(0, dtype=np.int64)
    train_indices = np.asarray(transform["train_indices"], dtype=np.int64).reshape(-1)
    eval_indices = np.asarray(eval_indices, dtype=np.int64).reshape(-1)
    if train_indices.size == 0 or eval_indices.size == 0:
        return np.empty(0, dtype=np.int64)
    return np.intersect1d(train_indices, eval_indices, assume_unique=False)


def require_no_train_eval_overlap(
    *,
    transform: dict[str, Any],
    eval_indices: np.ndarray,
    run_dir: Path,
    allow_overlap: bool,
    diagnostic_name: str,
) -> None:
    overlap = train_eval_overlap(transform, eval_indices)
    if overlap.size == 0:
        return
    message = (
        f"{diagnostic_name} would evaluate {overlap.size} rows that were used for training in {run_dir}. "
        f"Examples: {overlap[:10].tolist()}. Rerun training with SBI_EXCLUDE_LAST_N_FROM_TRAINING "
        "at least as large as the diagnostic last-N value, or pass --allow-train-eval-overlap "
        "only if you intentionally want an in-training diagnostic."
    )
    if allow_overlap:
        print(f"Warning: {message}")
    else:
        raise ValueError(message)


def compute_true_vs_mean_rows(
    *,
    posterior: Any,
    x_values: np.ndarray,
    theta_values: np.ndarray,
    indices: np.ndarray,
    param_names: list[str],
    transform: dict[str, Any],
    num_samples: int,
    device: str,
    case: str,
    n_train: int,
    progress_every: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for local_i, dataset_index in enumerate(indices):
        if local_i == 0 or (local_i + 1) % int(progress_every) == 0 or local_i + 1 == len(indices):
            print(f"  true-vs-mean {local_i + 1}/{len(indices)}")
        x_condition = apply_x_transform(x_values[local_i], transform)
        samples = sample_posterior_at_x(posterior, x_condition, int(num_samples), device)
        n_params = min(samples.shape[1], theta_values.shape[1], len(param_names))
        mean = np.nanmean(samples[:, :n_params], axis=0)
        std = np.nanstd(samples[:, :n_params], axis=0, ddof=1)
        truth = theta_values[local_i, :n_params]
        for j in range(n_params):
            rows.append(
                {
                    "case": case,
                    "n_train": int(n_train),
                    "dataset_index": int(dataset_index),
                    "param": param_names[j],
                    "param_index": int(j),
                    "theta_true": float(truth[j]),
                    "posterior_mean": float(mean[j]),
                    "posterior_std": float(std[j]),
                    "error": float(mean[j] - truth[j]),
                    "num_posterior_samples": int(samples.shape[0]),
                }
            )
    return rows


def compute_true_vs_mean_reference_rows(
    *,
    posterior: Any,
    x_obs: np.ndarray,
    theta_true: np.ndarray,
    param_names: list[str],
    transform: dict[str, Any],
    num_samples: int,
    device: str,
    case: str,
    n_train: int,
    reference_label: str,
) -> list[dict[str, Any]]:
    print(f"  true-vs-mean reference: {reference_label}")
    x_condition = apply_x_transform(x_obs, transform)
    samples = sample_posterior_at_x(posterior, x_condition, int(num_samples), device)
    n_params = min(samples.shape[1], theta_true.size, len(param_names))
    mean = np.nanmean(samples[:, :n_params], axis=0)
    std = np.nanstd(samples[:, :n_params], axis=0, ddof=1)
    truth = np.asarray(theta_true[:n_params], dtype=float)

    rows: list[dict[str, Any]] = []
    for j in range(n_params):
        rows.append(
            {
                "case": case,
                "n_train": int(n_train),
                "dataset_index": reference_label,
                "reference_label": reference_label,
                "is_reference": 1,
                "param": param_names[j],
                "param_index": int(j),
                "theta_true": float(truth[j]),
                "posterior_mean": float(mean[j]),
                "posterior_std": float(std[j]),
                "error": float(mean[j] - truth[j]),
                "num_posterior_samples": int(samples.shape[0]),
            }
        )
    return rows


def plot_true_vs_mean_rows(
    rows: list[dict[str, Any]],
    param_names: list[str],
    output_path: Path,
    title: str,
    dpi: int,
    reference_rows: list[dict[str, Any]] | None = None,
) -> None:
    if not rows:
        print(f"No rows for true-vs-mean plot: {output_path}")
        return

    reference_rows = reference_rows or []
    all_rows = rows + reference_rows
    n_params = min(len(param_names), max(int(row["param_index"]) for row in all_rows) + 1)
    n_cols = 3
    n_rows = int(np.ceil(n_params / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18.0 / 2.54, max(6.0, 5.4 * n_rows) / 2.54))
    axes = np.asarray(axes).reshape(-1)

    for j in range(n_params):
        ax = axes[j]
        sub = [row for row in rows if int(row["param_index"]) == j]
        true = np.asarray([float(row["theta_true"]) for row in sub], dtype=float)
        mean = np.asarray([float(row["posterior_mean"]) for row in sub], dtype=float)
        std = np.asarray([float(row["posterior_std"]) for row in sub], dtype=float)
        finite = np.isfinite(true) & np.isfinite(mean)
        true = true[finite]
        mean = mean[finite]
        std = std[finite]
        param = param_names[j]
        ref_sub = [row for row in reference_rows if int(row["param_index"]) == j]
        ref_true = np.asarray([float(row["theta_true"]) for row in ref_sub], dtype=float)
        ref_mean = np.asarray([float(row["posterior_mean"]) for row in ref_sub], dtype=float)
        ref_std = np.asarray([float(row["posterior_std"]) for row in ref_sub], dtype=float)
        ref_finite = np.isfinite(ref_true) & np.isfinite(ref_mean)
        ref_true = ref_true[ref_finite]
        ref_mean = ref_mean[ref_finite]
        ref_std = ref_std[ref_finite]

        if true.size == 0 and ref_true.size == 0:
            ax.axis("off")
            continue

        limits_payload = []
        if true.size:
            limits_payload.extend([true, mean])
        if ref_true.size:
            limits_payload.extend([ref_true, ref_mean])
        lo = float(np.nanmin(np.concatenate(limits_payload)))
        hi = float(np.nanmax(np.concatenate(limits_payload)))
        pad = 0.06 * (hi - lo) if hi > lo else 0.1 * max(abs(hi), 1.0)
        lo -= pad
        hi += pad

        finite_std = np.isfinite(std) & (std > 0.0)
        if np.any(finite_std):
            ax.errorbar(
                true[finite_std],
                mean[finite_std],
                yerr=std[finite_std],
                fmt="none",
                ecolor="#1f77b4",
                alpha=0.09,
                elinewidth=0.45,
                capsize=0.0,
                zorder=1,
            )
        ax.scatter(true, mean, s=10, alpha=0.62, color="#1f77b4", edgecolor="none", zorder=2)
        if ref_true.size:
            ref_finite_std = np.isfinite(ref_std) & (ref_std > 0.0)
            if np.any(ref_finite_std):
                ax.errorbar(
                    ref_true[ref_finite_std],
                    ref_mean[ref_finite_std],
                    yerr=ref_std[ref_finite_std],
                    fmt="none",
                    ecolor="#d62728",
                    alpha=0.65,
                    elinewidth=0.75,
                    capsize=1.8,
                    zorder=4,
                )
            ax.scatter(
                ref_true,
                ref_mean,
                s=28,
                marker="D",
                color="#d62728",
                edgecolor="black",
                linewidth=0.35,
                label="Battaglia12" if j == 0 else None,
                zorder=5,
            )
        ax.plot([lo, hi], [lo, hi], color="black", lw=0.8, ls=":", zorder=3)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_title(f"${LABEL_BY_NAME.get(param, param)}$", pad=2.0)
        ax.set_xlabel(r"True")
        ax.set_ylabel(r"Posterior mean")
        ax.grid(True, alpha=0.25, lw=0.5)
        if j == 0 and ref_true.size:
            ax.legend(frameon=False, loc="best", fontsize=6.5)

    for ax in axes[n_params:]:
        ax.axis("off")

    fig.suptitle(title, y=0.995, fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    print(f"Saved {output_path}")


def parse_args() -> argparse.Namespace:
    root = repo_root()
    parser = argparse.ArgumentParser(
        description="Sample saved SO SBI posteriors at the Battaglia12 SO profile and make a GetDist triangle."
    )
    parser.add_argument(
        "--run-root",
        default=str(root / "SBI_analysis" / "outputs" / "cluster_outputs" / "SBI_Adrian_SO_dataset_size_ell80_7979_dataset_row_sobolrow_asinh"),
    )
    parser.add_argument(
        "--case-dataset-dir",
        default=str(root / "SBI_analysis" / "data_for_cluster" / "adrian_so_sbi_cases_ell80_7979_dataset_row_sobolrow"),
    )
    parser.add_argument("--case-index-json", default="")
    parser.add_argument("--case", default="masked_no_noise")
    parser.add_argument("--n-train", default="523788", help="One or more N values, comma or space separated.")
    parser.add_argument(
        "--battaglia12-dir",
        default=str(root / "tSZ_visuals" / "outputs" / "so_noise_battaglia12_fiducial_local"),
    )
    parser.add_argument("--battaglia-profile-path", default="", help="Explicit C_ell profile override.")
    parser.add_argument("--bin-weighting", default="2ell_plus_1")
    parser.add_argument("--num-posterior-samples", type=int, default=100000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--output-name", default="")
    parser.add_argument("--reuse-samples", action="store_true")
    parser.add_argument("--filled-last-only", action="store_true")
    parser.add_argument("--skip-dell-density", action="store_true")
    parser.add_argument("--only-dell-density", action="store_true")
    parser.add_argument("--dell-density-max-rows", type=int, default=0, help="Rows to use for D_ell density; 0 uses all rows.")
    parser.add_argument("--dell-density-y-bins", type=int, default=220)
    parser.add_argument("--dell-density-percentile-low", type=float, default=0.5)
    parser.add_argument("--dell-density-percentile-high", type=float, default=99.5)
    parser.add_argument("--skip-true-vs-mean", action="store_true")
    parser.add_argument("--true-vs-mean-last-n", type=int, default=500)
    parser.add_argument(
        "--true-vs-mean-n-train",
        type=int,
        default=0,
        help="N value for true-vs-mean diagnostic. Default 0 uses the second-largest completed/requested N.",
    )
    parser.add_argument("--true-vs-mean-num-posterior-samples", type=int, default=5000)
    parser.add_argument("--true-vs-mean-progress-every", type=int, default=50)
    parser.add_argument(
        "--skip-battaglia12-true-vs-mean-reference",
        action="store_true",
        help="Do not add the Battaglia12 observation as a red reference point in the true-vs-mean plot.",
    )
    parser.add_argument(
        "--allow-train-eval-overlap",
        action="store_true",
        help="Allow true-vs-mean rows to overlap with train_indices saved in x_transform.npz.",
    )
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = repo_root()
    run_root = resolve_path(args.run_root, root)
    dataset_dir = resolve_path(args.case_dataset_dir, root)
    index_json = resolve_path(args.case_index_json, root) if args.case_index_json else dataset_dir / "case_dataset_index.json"
    battaglia12_dir = resolve_path(args.battaglia12_dir, root)
    battaglia_profile_path = resolve_path(args.battaglia_profile_path, root) if args.battaglia_profile_path else None
    n_values = parse_int_list(args.n_train)
    if not n_values:
        raise ValueError("--n-train must contain at least one value")

    output_dir = resolve_path(args.output_dir, root) if args.output_dir else run_root / args.case / "battaglia12_getdist"
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = find_case_dataset(args.case, dataset_dir, index_json)
    obs, theta_true, param_names, obs_source = battaglia12_obs_from_case_dataset(
        args.case,
        dataset_path,
        battaglia12_dir,
        battaglia_profile_path,
        args.bin_weighting,
    )
    print(f"case: {args.case}")
    print(f"dataset: {dataset_path}")
    print(f"Battaglia12 obs source: {obs_source}")
    print(f"obs shape: {obs.shape}")
    print(f"truth: {theta_true}")

    if not args.skip_dell_density:
        plot_dell_density_with_reference(
            dataset_path=dataset_path,
            reference_dell=obs,
            output_path=output_dir / f"{args.case}_dell_density_battaglia12_reference.jpg",
            title=rf"{args.case}: $D_\ell$ density",
            max_rows=int(args.dell_density_max_rows),
            n_y_bins=int(args.dell_density_y_bins),
            percentile_range=(
                float(args.dell_density_percentile_low),
                float(args.dell_density_percentile_high),
            ),
            dpi=int(args.dpi),
        )
    if args.only_dell_density:
        return 0

    sample_sets = []
    for n_train in n_values:
        run_dir = find_run_dir(run_root, args.case, n_train)
        sample_path = output_dir / f"{args.case}_N{int(n_train)}_battaglia12_posterior_samples.npy"
        if args.reuse_samples and sample_path.is_file():
            samples = np.load(sample_path)
        else:
            posterior = load_posterior(run_dir)
            transform = load_x_transform(run_dir)
            x_condition = apply_x_transform(obs, transform)
            samples = sample_posterior_at_x(
                posterior,
                x_condition,
                int(args.num_posterior_samples),
                args.device,
            )
            np.save(sample_path, np.asarray(samples, dtype=np.float32))
            print(f"Saved posterior samples: {sample_path}")
        sample_sets.append(
            {
                "label": rf"$N={int(n_train):,}$",
                "samples": np.asarray(samples, dtype=np.float64),
            }
        )

    output_name = args.output_name or f"{args.case}_battaglia12_getdist_N{'_'.join(str(n) for n in n_values)}.jpg"
    plot_getdist(
        sample_sets=sample_sets,
        param_names=param_names,
        theta_true=theta_true,
        output_path=output_dir / output_name,
        filled_last_only=bool(args.filled_last_only),
        dpi=int(args.dpi),
    )

    if not args.skip_true_vs_mean:
        tvm_n_train = choose_true_vs_mean_n(
            run_root,
            args.case,
            n_values,
            int(args.true_vs_mean_n_train),
        )
        tvm_run_dir = find_run_dir(run_root, args.case, tvm_n_train)
        tvm_output_dir = output_dir / f"true_vs_mean_last{int(args.true_vs_mean_last_n)}_N{int(tvm_n_train)}"
        tvm_csv = tvm_output_dir / f"{args.case}_N{int(tvm_n_train)}_last{int(args.true_vs_mean_last_n)}_true_vs_mean.csv"
        tvm_plot = tvm_output_dir / f"{args.case}_N{int(tvm_n_train)}_last{int(args.true_vs_mean_last_n)}_true_vs_mean.jpg"

        print("")
        print(
            f"Computing true-vs-mean diagnostic for last {int(args.true_vs_mean_last_n)} "
            f"dataset rows using N={int(tvm_n_train)}"
        )
        x_last, theta_last, tvm_param_names, indices = last_n_dataset_profiles(
            dataset_path,
            int(args.true_vs_mean_last_n),
        )
        tvm_posterior = load_posterior(tvm_run_dir)
        tvm_transform = load_x_transform(tvm_run_dir)
        require_no_train_eval_overlap(
            transform=tvm_transform,
            eval_indices=indices,
            run_dir=tvm_run_dir,
            allow_overlap=bool(args.allow_train_eval_overlap),
            diagnostic_name="true-vs-mean",
        )
        tvm_rows = compute_true_vs_mean_rows(
            posterior=tvm_posterior,
            x_values=x_last,
            theta_values=theta_last,
            indices=indices,
            param_names=tvm_param_names,
            transform=tvm_transform,
            num_samples=int(args.true_vs_mean_num_posterior_samples),
            device=args.device,
            case=args.case,
            n_train=int(tvm_n_train),
            progress_every=int(args.true_vs_mean_progress_every),
        )
        tvm_reference_rows: list[dict[str, Any]] = []
        if not args.skip_battaglia12_true_vs_mean_reference:
            tvm_reference_rows = compute_true_vs_mean_reference_rows(
                posterior=tvm_posterior,
                x_obs=obs,
                theta_true=theta_true,
                param_names=tvm_param_names,
                transform=tvm_transform,
                num_samples=int(args.true_vs_mean_num_posterior_samples),
                device=args.device,
                case=args.case,
                n_train=int(tvm_n_train),
                reference_label="Battaglia12",
            )
        write_csv(
            tvm_csv,
            tvm_rows + tvm_reference_rows,
            [
                "case",
                "n_train",
                "dataset_index",
                "reference_label",
                "is_reference",
                "param",
                "param_index",
                "theta_true",
                "posterior_mean",
                "posterior_std",
                "error",
                "num_posterior_samples",
            ],
        )
        plot_true_vs_mean_rows(
            tvm_rows,
            tvm_param_names,
            tvm_plot,
            title=rf"{args.case}, $N={int(tvm_n_train):,}$, last {int(args.true_vs_mean_last_n)} profiles",
            dpi=int(args.dpi),
            reference_rows=tvm_reference_rows,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
