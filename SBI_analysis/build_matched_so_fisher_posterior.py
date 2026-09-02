#!/usr/bin/env python3
"""Build an observation-matched linear Fisher posterior with a hard box prior."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


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
                raise KeyError(f"No binned observation key found in {path}.")
    else:
        raise ValueError(f"Observation must be NPY or NPZ: {path}")
    return np.asarray(values, dtype=np.float64).reshape(-1)


def symmetric_inverse(matrix: np.ndarray) -> np.ndarray:
    matrix = 0.5 * (matrix + matrix.T)
    values, vectors = np.linalg.eigh(matrix)
    floor = max(float(np.max(np.abs(values))) * 1e-12, 1e-300)
    if np.any(values <= 0.0):
        raise ValueError(
            f"Matrix is not positive definite; minimum eigenvalue={values.min():.6e}."
        )
    return (vectors * (1.0 / np.maximum(values, floor))) @ vectors.T


def systematic_resample(
    samples: np.ndarray,
    weights: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    positions = (rng.random() + np.arange(count)) / count
    cumulative = np.cumsum(weights)
    cumulative[-1] = 1.0
    indices = np.searchsorted(cumulative, positions, side="right")
    return np.asarray(samples[indices], dtype=np.float64)


def build_matched_posterior(
    fisher_root: Path,
    sensitivity_dir: Path,
    observation_path: Path,
    output_dir: Path,
    covariance_scope: str = "conditional_noise",
    proposal_draws: int = 1_000_000,
    posterior_samples: int = 200_000,
    minimum_importance_ess: int = 5_000,
    seed: int = 271828,
) -> dict[str, Any]:
    fisher_root = fisher_root.expanduser().resolve()
    sensitivity_dir = sensitivity_dir.expanduser().resolve()
    observation_path = observation_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    table = pd.read_csv(sensitivity_dir / "all_parameter_sensitivities.csv")
    selected = table[
        (table["case"] == "baseline")
        & (table["covariance_scope"] == covariance_scope)
    ].copy()
    if selected.empty:
        raise ValueError(
            f"No baseline/{covariance_scope} rows in all_parameter_sensitivities.csv."
        )

    names = selected["parameter"].astype(str).tolist()
    fiducial = selected["fiducial"].to_numpy(dtype=np.float64)
    prior_low = selected["prior_low"].to_numpy(dtype=np.float64)
    prior_high = selected["prior_high"].to_numpy(dtype=np.float64)
    prior_width = prior_high - prior_low

    derivatives = np.asarray(
        np.load(fisher_root / "analysis" / "derivatives_richardson.npy"),
        dtype=np.float64,
    )
    fiducial_dell = np.asarray(
        np.load(fisher_root / "analysis" / "fiducial_battaglia12_dell.npy"),
        dtype=np.float64,
    ).reshape(-1)
    covariance_path = (
        sensitivity_dir
        / "baseline"
        / covariance_scope
        / "covariance_binned_dell.npy"
    )
    covariance = np.asarray(np.load(covariance_path), dtype=np.float64)
    observation = load_observation(observation_path)

    expected_shape = (len(names), fiducial_dell.size)
    if derivatives.shape != expected_shape:
        raise ValueError(f"Derivative shape {derivatives.shape} != {expected_shape}.")
    if covariance.shape != (fiducial_dell.size, fiducial_dell.size):
        raise ValueError(f"Covariance has incompatible shape {covariance.shape}.")
    if observation.shape != fiducial_dell.shape:
        raise ValueError(
            f"Observation shape {observation.shape} != {fiducial_dell.shape}."
        )

    precision = symmetric_inverse(covariance)
    residual = observation - fiducial_dell
    fisher = derivatives @ precision @ derivatives.T
    fisher = 0.5 * (fisher + fisher.T)
    fisher_eigenvalues = np.linalg.eigvalsh(fisher)
    fisher_threshold = max(float(fisher_eigenvalues.max()) * 1e-8, 0.0)
    fisher_rank = int(np.count_nonzero(fisher_eigenvalues > fisher_threshold))

    # This Gaussian is only an importance proposal. Its variance matches each
    # uniform prior's variance; importance weights remove it from the result.
    gaussian_prior_sigma = prior_width / np.sqrt(12.0)
    gaussian_prior_precision = np.diag(1.0 / gaussian_prior_sigma**2)
    proposal_precision = fisher + gaussian_prior_precision
    proposal_covariance = symmetric_inverse(proposal_precision)
    score = derivatives @ precision @ residual
    proposal_mean = fiducial + proposal_covariance @ score

    rng = np.random.default_rng(seed)
    proposal_draws = int(proposal_draws)
    posterior_samples = int(posterior_samples)
    if proposal_draws < posterior_samples:
        raise ValueError("proposal_draws must be at least posterior_samples.")

    accepted_batches: list[np.ndarray] = []
    total_drawn = 0
    while total_drawn < proposal_draws:
        count = min(100_000, proposal_draws - total_drawn)
        batch = rng.multivariate_normal(
            proposal_mean,
            proposal_covariance,
            size=count,
        )
        inside = np.all(
            (batch >= prior_low[None, :])
            & (batch <= prior_high[None, :]),
            axis=1,
        )
        if np.any(inside):
            accepted_batches.append(batch[inside])
        total_drawn += count

    if not accepted_batches:
        raise RuntimeError("The Gaussian importance proposal produced no in-prior draws.")
    candidates = np.concatenate(accepted_batches, axis=0)

    standardized = (
        candidates - fiducial[None, :]
    ) / gaussian_prior_sigma[None, :]
    log_weights = 0.5 * np.sum(standardized**2, axis=1)
    log_weights -= np.max(log_weights)
    weights = np.exp(log_weights)
    weights /= np.sum(weights)
    importance_ess = float(1.0 / np.sum(weights**2))
    if importance_ess < int(minimum_importance_ess):
        raise RuntimeError(
            f"Importance ESS={importance_ess:.1f} is below "
            f"{minimum_importance_ess}. Increase --proposal-draws."
        )

    samples = systematic_resample(candidates, weights, posterior_samples, rng)
    stem = f"matched_fisher_baseline_{covariance_scope}_hard_uniform"
    samples_path = output_dir / f"{stem}_samples.npy"
    bundle_path = output_dir / f"{stem}.npz"
    summary_path = output_dir / f"{stem}_summary.json"
    np.save(samples_path, samples)
    np.savez_compressed(
        bundle_path,
        samples=samples,
        param_names=np.asarray(names),
        fiducial=fiducial,
        prior_low=prior_low,
        prior_high=prior_high,
        observation=observation,
        fiducial_dell=fiducial_dell,
        residual=residual,
        fisher_matrix=fisher,
        proposal_mean=proposal_mean,
        proposal_covariance=proposal_covariance,
        posterior_mean=samples.mean(axis=0),
        posterior_covariance=np.cov(samples, rowvar=False),
        covariance_scope=np.asarray(covariance_scope),
        prior=np.asarray("hard bounded uniform"),
    )

    summary = {
        "complete": True,
        "physics_choice": (
            "baseline conditional_noise: the mask and signal realization are fixed "
            "while independent SO noise seeds vary"
        ),
        "covariance_scope": covariance_scope,
        "prior": "hard bounded uniform, identical to SBI",
        "fisher_root": fisher_root,
        "sensitivity_dir": sensitivity_dir,
        "observation": observation_path,
        "covariance": covariance_path,
        "fisher_rank": fisher_rank,
        "fisher_eigenvalue_threshold": fisher_threshold,
        "proposal_draws": proposal_draws,
        "proposal_inside_prior": int(candidates.shape[0]),
        "proposal_inside_prior_fraction": float(candidates.shape[0] / proposal_draws),
        "importance_ess": importance_ess,
        "posterior_samples": posterior_samples,
        "posterior_mean": samples.mean(axis=0),
        "posterior_std": samples.std(axis=0, ddof=1),
        "outputs": {"samples": samples_path, "bundle": bundle_path},
    }
    write_json(summary_path, summary)
    print(f"Fisher rank: {fisher_rank}/{len(names)}")
    print(
        "Gaussian-proposal in-prior fraction: "
        f"{summary['proposal_inside_prior_fraction']:.3%}"
    )
    print(f"Importance ESS: {importance_ess:.1f}")
    print(f"Saved: {samples_path}")
    print(f"Saved: {bundle_path}")
    print(f"Saved: {summary_path}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fisher-root", type=Path, required=True)
    parser.add_argument("--sensitivity-dir", type=Path, required=True)
    parser.add_argument("--observation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--covariance-scope",
        choices=("conditional_noise", "gaussian_total"),
        default="conditional_noise",
    )
    parser.add_argument("--proposal-draws", type=int, default=1_000_000)
    parser.add_argument("--posterior-samples", type=int, default=200_000)
    parser.add_argument("--minimum-importance-ess", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=271828)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    build_matched_posterior(
        fisher_root=args.fisher_root,
        sensitivity_dir=args.sensitivity_dir,
        observation_path=args.observation,
        output_dir=args.output_dir,
        covariance_scope=args.covariance_scope,
        proposal_draws=args.proposal_draws,
        posterior_samples=args.posterior_samples,
        minimum_importance_ess=args.minimum_importance_ess,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
