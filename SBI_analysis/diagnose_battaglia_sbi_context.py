#!/usr/bin/env python3
"""Compare a Battaglia12 observation with the NPE training context distribution."""

from __future__ import annotations

import argparse
import inspect
import json
import pickle
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def scalar_string(value: Any, default: str = "") -> str:
    array = np.asarray(value)
    return str(array.reshape(()).item()) if array.size == 1 else default


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_observation(path: Path) -> np.ndarray:
    path = path.expanduser().resolve()
    if path.suffix == ".npy":
        values = np.load(path)
    elif path.suffix == ".npz":
        with np.load(path, allow_pickle=True) as data:
            for key in ("x_binned_dell", "binned_dell", "x", "obs"):
                if key in data.files:
                    values = data[key]
                    break
            else:
                raise KeyError(
                    f"{path} has no x_binned_dell, binned_dell, x, or obs key."
                )
    else:
        raise ValueError(f"Observation must be NPY or NPZ, got {path}.")
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"Observation contains non-finite values: {path}")
    return values


def load_transform(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "x_transform.npz"
    with np.load(path, allow_pickle=True) as data:
        result = {key: np.asarray(data[key]) for key in data.files}
    result["mode"] = scalar_string(result.get("mode", "none"), "none")
    result["path"] = path
    return result


def apply_transform(values: np.ndarray, transform: dict[str, Any]) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    mode = str(transform["mode"]).lower().replace("-", "_")
    if mode in {"none", "raw"}:
        return values
    if mode == "standardize":
        return (values - transform["mean"]) / transform["std"]
    if mode in {"asinh", "asinh_standardize"}:
        transformed = np.arcsinh(values / transform["scale"])
        if mode == "asinh_standardize":
            transformed = (transformed - transform["mean"]) / transform["std"]
        return np.asarray(transformed, dtype=np.float32)
    raise ValueError(f"Unsupported x transform mode: {mode!r}")


def load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def raw_flow_samples(run_dir: Path, context: np.ndarray, count: int) -> np.ndarray:
    posterior_path = run_dir / "posterior.pkl"
    estimator_path = run_dir / "density_estimator.pkl"
    # Prefer the estimator itself: loading posterior.pkl imports the full SBI
    # posterior stack, including optional ArviZ/Numba dependencies not needed here.
    if estimator_path.is_file():
        estimator = load_pickle(estimator_path)
    elif posterior_path.is_file():
        posterior = load_pickle(posterior_path)
        estimator = getattr(
            posterior,
            "posterior_estimator",
            getattr(posterior, "_posterior_estimator", None),
        )
        if estimator is None:
            raise AttributeError("Saved posterior has no density estimator.")
    else:
        raise FileNotFoundError(
            f"No posterior.pkl or density_estimator.pkl under {run_dir}."
        )

    context_t = torch.as_tensor(context[None, :], dtype=torch.float32)
    parameters = inspect.signature(estimator.sample).parameters
    with torch.no_grad():
        if "context" in parameters:
            samples = estimator.sample(int(count), context=context_t)
        elif "condition" in parameters:
            samples = estimator.sample(torch.Size([int(count)]), condition=context_t)
        else:
            raise TypeError("Unsupported density-estimator sample interface.")

    if torch.is_tensor(samples):
        samples = samples.detach().cpu().numpy()
    samples = np.asarray(samples, dtype=np.float64)
    return samples.reshape(-1, samples.shape[-1])


def nearest_distance(query: np.ndarray, reference: np.ndarray, chunk: int = 256) -> np.ndarray:
    output = np.empty(query.shape[0], dtype=np.float64)
    reference_norm = np.sum(reference * reference, axis=1)
    for start in range(0, query.shape[0], chunk):
        block = query[start : start + chunk]
        distance2 = (
            np.sum(block * block, axis=1)[:, None]
            + reference_norm[None, :]
            - 2.0 * block @ reference.T
        )
        output[start : start + len(block)] = np.sqrt(
            np.maximum(np.min(distance2, axis=1), 0.0)
        )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-dataset", type=Path, required=True)
    parser.add_argument("--observation", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--context-sample-size", type=int, default=100_000)
    parser.add_argument("--flow-samples", type=int, default=10_000)
    parser.add_argument("--profile-lines", type=int, default=8)
    parser.add_argument("--pca-components", type=int, default=20)
    parser.add_argument("--seed", type=int, default=271828)
    parser.add_argument("--covariance-shrinkage", type=float, default=0.02)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    prepared_path = args.prepared_dataset.expanduser().resolve()
    run_dir = args.run_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with np.load(prepared_path, allow_pickle=True) as data:
        needed = {"x", "ell_binned", "prior_low", "prior_high"}
        missing = sorted(needed.difference(data.files))
        if missing:
            raise KeyError(f"Prepared dataset lacks keys: {missing}")
        x_all = np.asarray(data["x"], dtype=np.float32)
        ell = np.asarray(data["ell_binned"], dtype=np.float64)
        prior_low = np.asarray(data["prior_low"], dtype=np.float64)
        prior_high = np.asarray(data["prior_high"], dtype=np.float64)
        product = scalar_string(data["product"]) if "product" in data.files else ""

    observation = load_observation(args.observation)
    if observation.shape != (x_all.shape[1],):
        raise ValueError(
            f"Observation shape {observation.shape} does not match x {x_all.shape}."
        )

    transform = load_transform(run_dir)
    train_indices = np.asarray(
        transform.get("train_indices", np.arange(x_all.shape[0])),
        dtype=np.int64,
    ).reshape(-1)
    if train_indices.size == 0:
        raise ValueError("Saved x transform has no training indices.")

    rng = np.random.default_rng(args.seed)
    sample_count = min(int(args.context_sample_size), train_indices.size)
    selected = rng.choice(train_indices, size=sample_count, replace=False)
    raw_context = np.asarray(x_all[selected], dtype=np.float32)
    context = np.asarray(apply_transform(raw_context, transform), dtype=np.float64)
    observation_t = np.asarray(
        apply_transform(observation, transform),
        dtype=np.float64,
    )

    mean = context.mean(axis=0)
    std = np.maximum(context.std(axis=0, ddof=1), 1e-12)
    z = (observation_t - mean) / std
    percentiles = np.mean(context <= observation_t[None, :], axis=0)

    standardized = (context - mean) / std
    covariance = np.cov(standardized, rowvar=False)
    shrinkage = float(args.covariance_shrinkage)
    covariance = (
        (1.0 - shrinkage) * covariance
        + shrinkage * np.diag(np.diag(covariance))
    )
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    floor = max(float(eigenvalues.max()) * 1e-8, 1e-10)
    inverse = (eigenvectors * (1.0 / np.maximum(eigenvalues, floor))) @ eigenvectors.T
    observation_standardized = (observation_t - mean) / std
    observation_d2 = float(observation_standardized @ inverse @ observation_standardized)
    training_d2 = np.einsum("ni,ij,nj->n", standardized, inverse, standardized)
    mahalanobis_percentile = float(np.mean(training_d2 <= observation_d2))

    order = np.argsort(eigenvalues)[::-1]
    components = min(int(args.pca_components), context.shape[1])
    basis = eigenvectors[:, order[:components]]
    scales = np.sqrt(np.maximum(eigenvalues[order[:components]], floor))
    whitened = standardized @ basis / scales
    observation_whitened = observation_standardized @ basis / scales
    reference_count = min(5000, whitened.shape[0] // 2)
    query_count = min(2000, whitened.shape[0] - reference_count)
    permutation = rng.permutation(whitened.shape[0])
    reference = whitened[permutation[:reference_count]]
    queries = whitened[
        permutation[reference_count : reference_count + query_count]
    ]
    training_nn = nearest_distance(queries, reference)
    observation_nn = float(
        nearest_distance(observation_whitened[None, :], whitened)[0]
    )
    nearest_percentile = float(np.mean(training_nn <= observation_nn))

    raw_samples = raw_flow_samples(run_dir, observation_t, args.flow_samples)
    if raw_samples.shape[1] != prior_low.size:
        raise ValueError(
            f"Flow returned {raw_samples.shape}; prior has {prior_low.size} parameters."
        )
    finite = np.all(np.isfinite(raw_samples), axis=1)
    within = finite & np.all(
        (raw_samples >= prior_low[None, :])
        & (raw_samples <= prior_high[None, :]),
        axis=1,
    )
    raw_prior_fraction = float(np.mean(within))

    profile_indices = rng.choice(
        np.arange(raw_context.shape[0]),
        size=min(int(args.profile_lines), raw_context.shape[0]),
        replace=False,
    )
    quantiles = np.quantile(raw_context, [0.05, 0.5, 0.95], axis=0)

    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "savefig.bbox": "tight",
    })
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(18.0 / 2.54, 12.0 / 2.54),
        sharex=False,
    )
    axis = axes[0]
    axis.fill_between(
        ell,
        quantiles[0],
        quantiles[2],
        color="#4C78A8",
        alpha=0.16,
        label="training 5--95%",
    )
    axis.plot(ell, quantiles[1], color="#4C78A8", lw=1.1, label="training median")
    for index in profile_indices:
        axis.plot(ell, raw_context[index], color="#4C78A8", lw=0.55, alpha=0.22)
    axis.plot(
        ell,
        observation,
        color="black",
        lw=1.5,
        label="Battaglia12",
        zorder=10,
    )
    axis.axhline(0.0, color="0.4", lw=0.6)
    axis.set_xlabel(r"$\ell$")
    axis.set_ylabel(r"$D_\ell$")
    axis.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axis.grid(alpha=0.2)
    axis.legend(frameon=False, ncol=2)

    axes[1].hist(
        training_d2,
        bins=60,
        density=True,
        color="#4C78A8",
        alpha=0.55,
        label="training contexts",
    )
    axes[1].axvline(
        observation_d2,
        color="black",
        lw=1.4,
        label=rf"Battaglia12: percentile {mahalanobis_percentile:.3f}",
    )
    axes[1].set_xlabel(r"shrinkage Mahalanobis distance squared")
    axes[1].set_ylabel("density")
    axes[1].grid(alpha=0.2)
    axes[1].legend(frameon=False)

    figure.tight_layout()
    figure_path = output_dir / "battaglia12_context_diagnostics.jpg"
    figure.savefig(figure_path, dpi=300)
    plt.close(figure)

    report = {
        "prepared_dataset": prepared_path,
        "product": product,
        "observation": args.observation.expanduser().resolve(),
        "run_dir": run_dir,
        "transform_mode": transform["mode"],
        "training_rows_total": int(train_indices.size),
        "training_rows_diagnostic_sample": int(sample_count),
        "context_dimension": int(context.shape[1]),
        "max_abs_marginal_z": float(np.max(np.abs(z))),
        "marginal_percentile_min": float(percentiles.min()),
        "marginal_percentile_max": float(percentiles.max()),
        "mahalanobis_distance_squared": observation_d2,
        "mahalanobis_percentile": mahalanobis_percentile,
        "pca_components_for_nearest_neighbor": components,
        "nearest_neighbor_distance": observation_nn,
        "nearest_neighbor_percentile": nearest_percentile,
        "flow_raw_samples": int(raw_samples.shape[0]),
        "flow_finite_fraction": float(np.mean(finite)),
        "flow_raw_prior_fraction": raw_prior_fraction,
        "diagnostic_flags": {
            "marginal_context_within_5_sigma": bool(np.max(np.abs(z)) <= 5.0),
            "multivariate_context_between_0p1_and_99p9_percentiles": bool(
                0.001 <= mahalanobis_percentile <= 0.999
            ),
            "nearest_neighbor_below_99p9_percentile": bool(
                nearest_percentile <= 0.999
            ),
            "flow_raw_prior_fraction_at_least_1_percent": bool(
                raw_prior_fraction >= 0.01
            ),
        },
        "outputs": {"figure": figure_path},
    }
    report_path = output_dir / "battaglia12_context_diagnostics.json"
    write_json(report_path, report)

    print(f"Prepared product: {product}")
    print(f"Transform: {transform['mode']}")
    print(f"Max |marginal z|: {report['max_abs_marginal_z']:.3f}")
    print(f"Mahalanobis percentile: {mahalanobis_percentile:.6f}")
    print(f"Nearest-neighbor percentile: {nearest_percentile:.6f}")
    print(f"Raw flow fraction inside prior: {raw_prior_fraction:.6%}")
    print(f"Saved: {figure_path}")
    print(f"Saved: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
