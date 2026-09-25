#!/usr/bin/env python3
"""Audit a new MOPED observation against paired spectra and a noise ensemble.

No simulation, retraining or observation substitution. Posterior sampling is
optional and explicitly restricted to the fixed-noise diagnostic.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from export_so_moped_bundle import sha256
from so_moped_local import (
    load_bundle, load_checked_model, plot_corner, posterior_samples,
    prepare_context, sampling_diagnostics,
)
from so_sbi_compression import (
    FIDUCIAL, PARAM_NAMES, check_pair, load_dataset, metrics_from_samples,
    project, save_npz, write_json,
)


def read_observation(folder, contract, transform):
    with np.load(folder / "observation.npz", allow_pickle=False) as z:
        saved = dict(z)
    complete = json.loads((folder / "simulation_complete.json").read_text())
    raw = Path(complete["raw_profile"])
    if sha256(raw) != complete["raw_sha256"]:
        raise ValueError(f"Raw observation hash changed: {raw}")
    x, context = prepare_context(raw, contract, transform)
    np.testing.assert_array_equal(x, saved["x"])
    np.testing.assert_array_equal(context, saved["context"])
    np.testing.assert_allclose(saved["theta"], complete["request"]["theta"], rtol=0, atol=0)
    return saved, complete, raw, x, context


def analyze_arrays(noisy, clean, observation, clean_observation):
    residual = np.asarray(noisy, dtype=np.float64) - np.asarray(clean, dtype=np.float64)
    mean, std = residual.mean(axis=0), residual.std(axis=0, ddof=1)
    if np.any(std <= 0):
        raise ValueError("Paired training residual has a degenerate bin")
    new_residual = np.asarray(observation, dtype=np.float64) - clean_observation
    return dict(mean=mean, std=std, residual=residual,
                new_residual=new_residual, new_residual_z=(new_residual-mean)/std,
                mean_over_scatter=mean/std)


def main(argv=None):
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observation-dir", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, default=base / "outputs/compression_bestval_20260909/moped_bundle")
    parser.add_argument("--dataset-dir", type=Path, default=base / "data_for_cluster/adrian_so_sbi_cases_ell80_7979_dataset_row_sobolrow")
    parser.add_argument("--ensemble-dir", type=Path, default=base / "data_for_cluster/ensemble_battaglia12")
    parser.add_argument("--reference-rows", type=int, default=10000)
    parser.add_argument("--raw-samples", type=int, default=20000)
    parser.add_argument("--compare-observation-dir", type=Path)
    parser.add_argument("--fixed-noise-posterior", action="store_true",
                        help="Sample only a seed-12345 observation, labelled conditional")
    parser.add_argument("--posterior-samples", type=int, default=10000)
    args = parser.parse_args(argv)
    if args.reference_rows < 2 or args.raw_samples < 1:
        raise ValueError("Invalid diagnostic sample counts")
    output = args.observation_dir / "diagnostics"
    output.mkdir(exist_ok=True)
    contract, transform, manifest = load_bundle(args.bundle)
    saved, complete, raw, x, context = read_observation(args.observation_dir, contract, transform)
    settings = dict(arg.split("=", 1) for arg in complete["request"]["command"][5:])
    if args.fixed_noise_posterior and (
        settings["noise_seed"] != "12345" or settings["mask_seed"] != "12345"
    ):
        raise ValueError("The fixed-noise posterior option requires noise and mask seed 12345")
    clean_paths = list(raw.parent.glob("*masked_no_noise_cl*.npy"))
    if len(clean_paths) != 1:
        raise ValueError("Need the paired clean spectrum from this simulation")
    clean_x, _ = prepare_context(clean_paths[0], contract, transform)

    import torch
    torch.set_num_threads(2)
    estimator = load_checked_model(args.bundle, contract)
    pilot = sampling_diagnostics(estimator, context, contract, args.raw_samples)
    write_json(output / "sampling_preflight.json", pilot)

    noisy = load_dataset(args.dataset_dir / "so_masked_baseline_noise_cross_deproj0_ell80_7979_sbi_run.npz")
    clean = load_dataset(args.dataset_dir / "so_masked_no_noise_ell80_7979_sbi_run.npz")
    check_pair(noisy, clean)
    # Restrict the diagnostic references to the eligible training pool, not holdout.
    in_prior = np.all((noisy["theta"] >= contract["low"]) &
                      (noisy["theta"] <= contract["high"]), axis=1)
    pool = np.flatnonzero(in_prior[:len(noisy["x"]) - 500])
    indices = np.random.default_rng(73).choice(pool, min(args.reference_rows, len(pool)), replace=False)
    residual = analyze_arrays(noisy["x"][indices], clean["x"][indices], x, clean_x)
    training_context = project(noisy["x"][indices], transform)
    summary = dict(experiment_id=manifest["experiment_id"],
        observation_dir=str(args.observation_dir), preprocessing_reproduces_saved_context=True,
        paired_data_alignment_passed=True, local_cluster_model_log_prob_passed=True,
        noise_seed=int(settings["noise_seed"]), mask_seed=int(settings["mask_seed"]),
        reference_rows=len(indices), context=context.tolist(),
        context_training_min=training_context.min(0).tolist(),
        context_training_max=training_context.max(0).tolist(),
        raw_prior_acceptance=pilot["acceptance"],
        observation_parameters=np.asarray(saved["theta"]).tolist(),
        max_abs_residual_z=float(np.max(np.abs(residual["new_residual_z"]))),
        max_abs_training_residual_mean_over_scatter=float(np.max(np.abs(residual["mean_over_scatter"]))),
        caution="Training residual scatter spans changing theta, not repeated noise at fixed theta.")

    ensemble_x = []
    for path in sorted(args.ensemble_dir.glob("noise_seed*/prepared/*sbi_observation.npz")):
        with np.load(path, allow_pickle=False) as z:
            if str(z["product"].item()) != "masked_baseline_noise_cross_deproj0":
                raise ValueError(f"Wrong product in {path}")
            np.testing.assert_allclose(z["theta"].reshape(-1), FIDUCIAL, rtol=1e-6)
            np.testing.assert_array_equal(z["ell_binned"], contract["ell_binned"])
            ensemble_x.append(np.asarray(z["x_binned_dell"], dtype=np.float64).reshape(-1))
    ensemble_x = np.asarray(ensemble_x)
    if len(ensemble_x) >= 3:
        ensemble_context = project(ensemble_x, transform)
        ensemble_std = ensemble_x.std(axis=0, ddof=1)
        summary["ensemble"] = dict(count=len(ensemble_x),
            coordinate_std=ensemble_context.std(0, ddof=1).tolist(),
            max_absolute_coordinates=np.max(np.abs(ensemble_context), axis=0).tolist(),
            std_over_training_residual_std=(ensemble_std / residual["std"]).tolist())
    else:
        ensemble_context = np.empty((0, len(context)))

    save_npz(output / "noise_contract_arrays.npz", ell=contract["ell_binned"],
             indices=indices, context=context, reference_context=training_context,
             ensemble_context=ensemble_context, ensemble_x=ensemble_x,
             x=x, clean_x=clean_x, **residual)
    write_json(output / "observation_diagnosis.json", summary)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 9})
    ell = contract["ell_binned"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for row in residual["residual"][:60]:
        axes[0].plot(ell, row/1e-14, color="0.55", alpha=.15, lw=.6)
    axes[0].plot(ell, residual["mean"]/1e-14, color="black", label="Training residual mean")
    axes[0].plot(ell, residual["new_residual"]/1e-14, color="#cc3311", label="New observation residual")
    axes[0].set_xlabel(r"$\ell$")
    axes[0].set_ylabel(r"$(D_\ell^{\rm noisy}-D_\ell^{\rm clean})/10^{-14}$")
    axes[0].legend(fontsize=7)
    axes[1].plot(ell, np.abs(residual["mean"]), color="black", label="Absolute training residual mean")
    axes[1].plot(ell, residual["std"], color="#4477aa", label="Training residual scatter")
    if len(ensemble_x) >= 3:
        axes[1].plot(ell, ensemble_x.std(0, ddof=1), color="#228833",
                     label=f"{len(ensemble_x)}-seed fixed-theta scatter")
    axes[1].set_yscale("log"); axes[1].set_xlabel(r"$\ell$")
    axes[1].set_ylabel(r"$D_\ell$ residual scale"); axes[1].legend(fontsize=7)
    components = np.arange(1, len(context)+1)
    low, high = np.quantile(training_context, [.01, .99], axis=0)
    axes[2].fill_between(components, low, high, alpha=.35, color="#4477aa", label="Training 1-99%")
    if len(ensemble_context):
        low, high = np.quantile(ensemble_context, [.01, .99], axis=0)
        axes[2].fill_between(components, low, high, alpha=.2, color="#228833", label="Noise ensemble 1-99%")
    axes[2].plot(components, context, "o-", color="#cc3311", label="New observation")
    axes[2].set_xlabel("MOPED component"); axes[2].set_ylabel("Standardized compressed coordinate")
    axes[2].legend(fontsize=7)
    for ax in axes:
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(output / "noise_contract_diagnosis.png", dpi=200)
    fig.savefig(output / "noise_contract_diagnosis.pdf")
    plt.close(fig)

    if args.compare_observation_dir:
        other, other_complete, _, other_x, other_context = read_observation(
            args.compare_observation_dir, contract, transform)
        np.testing.assert_array_equal(saved["theta"], other["theta"])
        other_settings = dict(arg.split("=", 1) for arg in other_complete["request"]["command"][5:])
        # Seeds and runtime paths may differ; physical settings and source hashes must not.
        allowed = {"noise_seed", "output_dir", "cache_dir"}
        if {k: v for k, v in settings.items() if k not in allowed} != {
            k: v for k, v in other_settings.items() if k not in allowed
        }:
            raise ValueError("Compared simulation physical settings differ")
        for key in ("sources", "noise_sha256", "catalogue"):
            if complete["request"][key] != other_complete["request"][key]:
                raise ValueError(f"Compared simulation provenance differs: {key}")
        other_pilot = sampling_diagnostics(estimator, other_context, contract, args.raw_samples)
        summary["comparison"] = dict(
            observation_dir=str(args.compare_observation_dir),
            noise_seed=int(other_settings["noise_seed"]),
            raw_prior_acceptance=other_pilot["acceptance"],
            context=other_context.tolist(), matched_physics_and_source_hashes=True)
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
        for row in noisy["x"][indices[:60]]:
            axes[0].plot(ell, row, color="0.6", alpha=.10, lw=.5)
        axes[0].plot(ell, x, color="#228833", label="Fixed-noise Battaglia12")
        axes[0].plot(ell, other_x, "--", color="#cc3311", label="Independent-noise Battaglia12")
        axes[0].set(xlabel=r"$\ell$", ylabel=r"$D_\ell$")
        axes[0].set_yscale("symlog", linthresh=1e-14)
        # Subtract each observation's own clean spectrum to expose the noise pattern.
        other_raw = Path(other_complete["raw_profile"])
        other_clean = list(other_raw.parent.glob("*masked_no_noise_cl*.npy"))
        if len(other_clean) != 1:
            raise ValueError("Missing comparison observation's paired clean spectrum")
        other_clean_x, _ = prepare_context(other_clean[0], contract, transform)
        axes[1].plot(ell, residual["mean"]/1e-14, color="black", label="Training residual mean")
        axes[1].plot(ell, residual["new_residual"]/1e-14, color="#228833", label="Fixed noise")
        axes[1].plot(ell, (other_x-other_clean_x)/1e-14, "--", color="#cc3311", label="Independent noise")
        axes[1].set(xlabel=r"$\ell$", ylabel=r"$(D_\ell^{\rm noisy}-D_\ell^{\rm clean})/10^{-14}$")
        low, high = np.quantile(training_context, [.01, .99], axis=0)
        axes[2].fill_between(components, low, high, color="0.6", alpha=.3, label="Training 1-99%")
        axes[2].plot(components, context, "o-", color="#228833", label="Fixed noise")
        axes[2].plot(components, other_context, "o--", color="#cc3311", label="Independent noise")
        axes[2].set(xlabel="MOPED component", ylabel="Standardized compressed coordinate")
        for ax in axes:
            ax.grid(alpha=.2)
            ax.legend(fontsize=7)
        fig.suptitle("Fixed-noise diagnostic: same pressure truth, mask seed, beam and NPE")
        fig.tight_layout()
        for extension in ("png", "pdf"):
            fig.savefig(output / f"fixed_vs_independent_noise.{extension}", dpi=200)
        plt.close(fig)

    write_json(output / "observation_diagnosis.json", summary)
    if args.fixed_noise_posterior:
        samples, proposals, acceptance = posterior_samples(
            estimator, context, contract, count=args.posterior_samples, diagnostics_dir=output)
        np.save(args.observation_dir / "posterior_samples.npy", samples)
        import pandas as pd
        metric = metrics_from_samples(samples, saved["theta"], contract["low"], contract["high"])
        pd.DataFrame(dict(param=PARAM_NAMES, truth=saved["theta"], posterior_mean=metric["mean"],
                          posterior_std=metric["std"], pull=metric["pull"],
                          error_over_prior=metric["normalized_error_prior"])).to_csv(
                              output / "posterior_metrics.csv", index=False)
        plot_corner(samples, saved["theta"], contract, output,
                    label="Fixed-noise diagnostic: seed 12345 (not noise-marginalized)")
        plt.close("all")
        write_json(output / "fixed_noise_sampling_summary.json", dict(
            experiment_id=manifest["experiment_id"], raw_profile=str(raw),
            raw_sha256=sha256(raw), samples_sha256=sha256(args.observation_dir / "posterior_samples.npy"),
            theta_true=saved["theta"].tolist(), noise_seed=12345, mask_seed=12345,
            n_samples=len(samples), proposals=proposals, raw_prior_acceptance=acceptance,
            sampling_seed=47, conditional_on_fixed_noise=True, independent_noise_validated=False))
    print(json.dumps(summary, indent=2))
    print("Diagnostics:", output)


if __name__ == "__main__":
    main()
