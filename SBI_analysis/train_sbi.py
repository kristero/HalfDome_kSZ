#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from sbi_utils import (
    cl_to_log10_dl,
    dataset_to_log10_dl,
    estimate_lightcone_sigma,
    load_combined_dataset,
    load_config,
    load_emulator,
    load_observed_vector,
    predict_emulator_cl,
    prior_bounds,
    sample_uniform_prior,
    save_pickle,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Battaglia-parameter SBI posterior using sbi SNPE.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--observed-spectrum-path", default="")
    parser.add_argument("--output-dir", default="")
    return parser.parse_args()


def make_training_simulations(config, dataset, transformed, sigma, rng):
    simulator_cfg = config.get("simulator", {})
    n_sims = int(simulator_cfg.get("num_simulations", 20000))
    batch_size = int(simulator_cfg.get("batch_size", 512))
    add_noise = bool(simulator_cfg.get("add_noise", True))
    include_dataset = bool(simulator_cfg.get("include_dataset_simulations", True))

    artifact = load_emulator(config["emulator_artifact_path"])
    x_columns = dataset["x_columns"]
    if list(artifact["x_columns"]) != list(x_columns):
        raise ValueError(f"Emulator x_columns={artifact['x_columns']} do not match dataset x_columns={x_columns}")
    if not np.allclose(np.asarray(artifact["ell"], dtype=float), dataset["ell"]):
        raise ValueError("Emulator ell grid does not match selected dataset ell grid.")

    low, high = prior_bounds(config, x_columns)
    theta = sample_uniform_prior(rng, n_sims, low, high)
    cl = predict_emulator_cl(artifact, theta, batch_size=batch_size)
    x = cl_to_log10_dl(cl, dataset["ell"]).astype(np.float32)
    if add_noise:
        x += rng.normal(0.0, sigma, size=x.shape).astype(np.float32)

    if include_dataset:
        theta = np.vstack([theta, dataset["theta"].astype(np.float32)])
        x = np.vstack([x, transformed["combined"].astype(np.float32)])

    return theta.astype(np.float32), x.astype(np.float32), artifact


def train_snpe(config, theta, x, low, high):
    import torch
    from sbi.inference import SNPE
    from sbi.utils import BoxUniform

    sbi_cfg = config.get("sbi", {})
    device = sbi_cfg.get("device", "cpu")
    low_t = torch.as_tensor(low, dtype=torch.float32, device=device)
    high_t = torch.as_tensor(high, dtype=torch.float32, device=device)
    try:
        prior = BoxUniform(low=low_t, high=high_t, device=device)
    except TypeError:
        prior = BoxUniform(low=low_t, high=high_t)
    density_estimator_name = sbi_cfg.get("density_estimator", "maf")
    try:
        inference = SNPE(prior=prior, density_estimator=density_estimator_name, device=device)
    except TypeError:
        inference = SNPE(prior=prior, density_estimator=density_estimator_name)

    theta_t = torch.as_tensor(theta, dtype=torch.float32, device=device)
    x_t = torch.as_tensor(x, dtype=torch.float32, device=device)
    density_estimator = inference.append_simulations(theta_t, x_t).train(
        training_batch_size=int(sbi_cfg.get("training_batch_size", 256)),
        max_num_epochs=int(sbi_cfg.get("max_num_epochs", 300)),
        validation_fraction=float(sbi_cfg.get("validation_fraction", 0.1)),
    )
    posterior = inference.build_posterior(density_estimator)
    return prior, inference, density_estimator, posterior


def to_jsonable(value):
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def save_emulator_validation(output_dir, dataset, transformed, artifact):
    batch_size = 512
    cl_pred = predict_emulator_cl(artifact, dataset["theta"], batch_size=batch_size)
    x_pred = cl_to_log10_dl(cl_pred, dataset["ell"]).astype(np.float32)
    residual = x_pred - transformed["combined"].astype(np.float32)
    output_dir = Path(output_dir)
    np.savez_compressed(
        output_dir / "emulator_dataset_validation.npz",
        theta=dataset["theta"].astype(np.float32),
        ell=dataset["ell"].astype(np.float32),
        x_true=transformed["combined"].astype(np.float32),
        x_pred=x_pred,
        residual=residual,
        x_columns=np.asarray(dataset["x_columns"]),
    )
    write_json(
        output_dir / "emulator_dataset_validation.json",
        {
            "median_abs_log10_dl_residual": float(np.median(np.abs(residual))),
            "p84_abs_log10_dl_residual": float(np.percentile(np.abs(residual), 84)),
            "max_abs_log10_dl_residual": float(np.max(np.abs(residual))),
        },
    )

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    p16, p50, p84 = np.percentile(residual, [16, 50, 84], axis=0)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.fill_between(dataset["ell"], p16, p84, alpha=0.25, label="16-84%")
    ax.plot(dataset["ell"], p50, color="black", lw=1.5, label="median")
    ax.axhline(0.0, color="tab:red", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("ell")
    ax.set_ylabel("emulator - dataset [log10(D_l)]")
    ax.set_title("Emulator residual on generated dataset")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "emulator_dataset_residual_log10_dl.png", dpi=180)
    plt.close(fig)


def save_basic_plots(output_dir, ell, sigma, theta, x, x_columns, observed=None, posterior_samples=None, artifact=None):
    output_dir = Path(output_dir)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(ell, sigma)
    ax.set_xscale("log")
    ax.set_xlabel("ell")
    ax.set_ylabel("sigma[log10(D_l)]")
    ax.set_title("Empirical lightcone scatter")
    fig.tight_layout()
    fig.savefig(plots_dir / "noise_sigma_log10_dl.png", dpi=180)
    plt.close(fig)

    ncols = 3
    nrows = int(np.ceil(len(x_columns) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.0 * nrows), constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)
    for idx, name in enumerate(x_columns):
        axes[idx].hist(theta[:, idx], bins=40, histtype="step", color="black")
        axes[idx].set_title(name)
    for ax in axes[len(x_columns) :]:
        ax.axis("off")
    fig.savefig(plots_dir / "training_parameter_coverage.png", dpi=180)
    plt.close(fig)

    q16, q50, q84 = np.percentile(x, [16, 50, 84], axis=0)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.fill_between(ell, q16, q84, alpha=0.25, label="training 16-84%")
    ax.plot(ell, q50, color="black", lw=1.5, label="training median")
    if observed is not None:
        ax.plot(ell, observed, color="tab:red", lw=1.5, label="observed")
    ax.set_xscale("log")
    ax.set_xlabel("ell")
    ax.set_ylabel("log10(D_l)")
    ax.set_title("SBI training data vector coverage")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "training_data_vector_coverage.png", dpi=180)
    plt.close(fig)

    if posterior_samples is not None:
        try:
            import corner

            fig = corner.corner(posterior_samples, labels=x_columns, show_titles=True)
            fig.savefig(plots_dir / "posterior_corner.png", dpi=180)
            plt.close(fig)
        except ModuleNotFoundError:
            pass

    if posterior_samples is not None and observed is not None and artifact is not None:
        n_pp = min(512, posterior_samples.shape[0])
        rng = np.random.default_rng(12345)
        choose = rng.choice(posterior_samples.shape[0], size=n_pp, replace=False)
        cl_pp = predict_emulator_cl(artifact, posterior_samples[choose], batch_size=256)
        x_pp = cl_to_log10_dl(cl_pp, ell)
        p16, p50, p84 = np.percentile(x_pp, [16, 50, 84], axis=0)
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.fill_between(ell, p16, p84, alpha=0.25, label="posterior predictive 16-84%")
        ax.plot(ell, p50, color="black", lw=1.5, label="posterior predictive median")
        ax.plot(ell, observed, color="tab:red", lw=1.5, label="observed")
        ax.set_xscale("log")
        ax.set_xlabel("ell")
        ax.set_ylabel("log10(D_l)")
        ax.set_title("Posterior predictive spectra")
        ax.legend()
        fig.tight_layout()
        fig.savefig(plots_dir / "posterior_predictive_log10_dl.png", dpi=180)
        plt.close(fig)


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.observed_spectrum_path:
        config["observed_spectrum_path"] = args.observed_spectrum_path

    output_dir = Path(config["output_dir"]).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "run_config.json", config)

    rng = np.random.default_rng(int(config.get("random_seed", 1234)))
    dataset = load_combined_dataset(
        config["combined_dataset_path"],
        int(config.get("ell_min", 2)),
        config.get("ell_max", None),
    )
    transformed = dataset_to_log10_dl(dataset)
    noise_cfg = config.get("noise", {})
    sigma = estimate_lightcone_sigma(
        transformed["y100"],
        transformed["y102"],
        target=noise_cfg.get("target", "single_lightcone"),
        sigma_floor=float(noise_cfg.get("sigma_floor", 1.0e-4)),
        scale=float(noise_cfg.get("scale", 1.0)),
    ).astype(np.float32)

    theta_train, x_train, artifact = make_training_simulations(config, dataset, transformed, sigma, rng)
    low, high = prior_bounds(config, dataset["x_columns"])

    np.savez_compressed(
        output_dir / "sbi_noise_and_dataset_summary.npz",
        theta_dataset=dataset["theta"].astype(np.float32),
        x_dataset=transformed["combined"].astype(np.float32),
        x_y100=transformed["y100"].astype(np.float32),
        x_y102=transformed["y102"].astype(np.float32),
        ell=dataset["ell"].astype(np.float32),
        sigma_log10_dl=sigma,
        x_columns=np.asarray(dataset["x_columns"]),
        prior_low=low,
        prior_high=high,
    )

    if bool(config.get("simulator", {}).get("save_training_data", True)):
        np.savez_compressed(
            output_dir / "sbi_training_simulations.npz",
            theta=theta_train,
            x=x_train,
            ell=dataset["ell"].astype(np.float32),
            x_columns=np.asarray(dataset["x_columns"]),
        )

    observed = None
    observed_path = config.get("observed_spectrum_path", "")
    if observed_path:
        observed = load_observed_vector(
            observed_path,
            dataset["ell"],
            input_kind=config.get("observed_input_kind", "auto"),
        ).astype(np.float32)
        np.save(output_dir / "observed_log10_dl.npy", observed)

    prior, inference, density_estimator, posterior = train_snpe(config, theta_train, x_train, low, high)
    save_pickle(output_dir / "sbi_prior.pkl", prior)
    save_pickle(output_dir / "sbi_inference.pkl", inference)
    save_pickle(output_dir / "sbi_density_estimator.pkl", density_estimator)
    save_pickle(output_dir / "sbi_posterior.pkl", posterior)
    write_json(output_dir / "sbi_training_summary.json", to_jsonable(getattr(inference, "_summary", {})))
    save_emulator_validation(output_dir, dataset, transformed, artifact)

    posterior_samples = None
    if observed is not None:
        import torch

        n_samples = int(config.get("sbi", {}).get("posterior_samples", 50000))
        posterior_samples = posterior.sample(
            (n_samples,),
            x=torch.as_tensor(observed, dtype=torch.float32, device=config.get("sbi", {}).get("device", "cpu")),
            show_progress_bars=True,
        ).detach().cpu().numpy()
        np.save(output_dir / "posterior_samples.npy", posterior_samples)
        header = ",".join(dataset["x_columns"])
        np.savetxt(output_dir / "posterior_samples.csv", posterior_samples, delimiter=",", header=header, comments="")

    summary = {
        "combined_dataset_path": str(config["combined_dataset_path"]),
        "emulator_artifact_path": str(config["emulator_artifact_path"]),
        "n_dataset_points": int(dataset["theta"].shape[0]),
        "n_training_simulations": int(theta_train.shape[0]),
        "n_ell": int(dataset["ell"].size),
        "ell_min": float(dataset["ell"][0]),
        "ell_max": float(dataset["ell"][-1]),
        "x_columns": dataset["x_columns"],
        "noise_mode": noise_cfg.get("mode", "paired_lightcone_diagonal"),
        "noise_target": noise_cfg.get("target", "single_lightcone"),
        "median_sigma_log10_dl": float(np.median(sigma)),
        "max_sigma_log10_dl": float(np.max(sigma)),
        "observed_spectrum_path": observed_path,
        "posterior_samples": 0 if posterior_samples is None else int(posterior_samples.shape[0]),
    }
    write_json(output_dir / "sbi_run_summary.json", summary)
    save_basic_plots(
        output_dir,
        dataset["ell"],
        sigma,
        theta_train,
        x_train,
        dataset["x_columns"],
        observed=observed,
        posterior_samples=posterior_samples,
        artifact=artifact,
    )
    print(f"Wrote SBI outputs to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
