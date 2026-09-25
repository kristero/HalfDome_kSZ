#!/usr/bin/env python3
"""Apply the unchanged 40-bin MOPED/SBI bundle and compare simulation outputs."""
import argparse
import csv
import json
import os
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import toml


LABELS = {"HalfDome": "HalfDome Battaglia12 (fresh control)",
          "L1_m9": "FLAMINGO fiducial", "fgas-8sigma": r"FLAMINGO $f_{gas}-8\sigma$",
          "Mstar-1sigma": r"FLAMINGO $M_\star-1\sigma$"}
COLORS = {"HalfDome": "#222222", "L1_m9": "#0072B2",
          "fgas-8sigma": "#D55E00", "Mstar-1sigma": "#009E73"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--helpers", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    args = parser.parse_args()
    # Use the same verified project helpers as the historical comparison.
    sys.path.insert(0, str(args.helpers))
    from plot_so_nine_fisher_moped524k import checked_bundle, checked_fixed_posterior
    from so_moped_local import load_checked_model, prepare_context, posterior_samples
    from so_sbi_compression import write_json, project
    from export_so_moped_bundle import sha256
    import torch
    torch.set_num_threads(int(os.environ.get("APP_THREADS", "26")))
    torch.set_num_interop_threads(1)

    root = args.campaign
    output = root / "comparison"
    output.mkdir(exist_ok=True)
    contract, transform, manifest, training = checked_bundle(args.bundle)
    model = load_checked_model(args.bundle, contract)
    old_samples, old_raw, old_x, old_context, old_report, old_sources = checked_fixed_posterior(
        args.reference, contract, transform, manifest)
    cache = toml.load(root / "cache/complete.toml")
    ell = np.asarray(contract["ell_binned"])
    noise_x, _ = prepare_context(root / "cache/masked_noise_cross_cl.npy", contract, transform)
    all_results, observations, samples = {}, {}, {}

    for name in LABELS:
        directory = root / "results" / name
        map_meta = toml.load(directory / "complete.toml")
        for key in ("cache_file_sha256", "mask_pixel_sha256", "noise1_pixel_sha256", "noise2_pixel_sha256"):
            if map_meta[key] != cache[key]:
                raise ValueError("Different mask/noise for " + name)
        clean, _ = prepare_context(directory / "masked_clean_cl.npy", contract, transform)
        unmasked, _ = prepare_context(directory / "unmasked_clean_cl.npy", contract, transform)
        noisy, context = prepare_context(directory / "masked_noisy_cross_cl.npy", contract, transform)
        np.testing.assert_array_equal(context, project(noisy, transform))
        observations[name] = dict(clean=clean, unmasked=unmasked, noisy=noisy,
                                  noise_noise=noise_x, signal_noise=noisy-clean-noise_x,
                                  context=context)
        np.savez(directory / "observation.npz", ell_binned=ell, **observations[name],
                 param_names=contract["param_names"], experiment_id=manifest["experiment_id"])
        result = dict(map=map_meta, status="pending", context=context.tolist(),
                      max_absolute_moped_coordinate=float(np.abs(context).max()),
                      conditional_on_fixed_noise=True,
                      physics_interpretation="Effective HalfDome/Battaglia inference, not FLAMINGO input-parameter recovery")
        try:
            array, proposals, acceptance = posterior_samples(
                model, context, contract, count=10000, seed=20260914,
                max_proposals=1000000, seconds=300, pilot_count=20000,
                diagnostics_dir=directory / "diagnostics")
            if array.shape != (10000, 9) or not np.isfinite(array).all():
                raise ValueError("Invalid posterior sample array")
            if np.any(array < contract["low"]) or np.any(array > contract["high"]):
                raise ValueError("Samples outside original prior")
            np.save(directory / "posterior_samples.npy", array)
            samples[name] = array
            result.update(status="posterior_sampled", proposals=int(proposals),
                          acceptance=float(acceptance), sample_count=len(array),
                          mean=array.mean(axis=0).tolist(), std=array.std(axis=0, ddof=1).tolist(),
                          q16=np.quantile(array, .16, axis=0).tolist(),
                          median=np.median(array, axis=0).tolist(),
                          q84=np.quantile(array, .84, axis=0).tolist())
        except RuntimeError as error:
            # A failed support probe is a result for this external observation,
            # and must not suppress spectra or the remaining feedback models.
            result.update(status="posterior_unavailable", error=str(error))
        diagnostic = directory / "diagnostics/sampling_preflight.json"
        if diagnostic.exists():
            result["sampling_preflight"] = json.loads(diagnostic.read_text())
        all_results[name] = result
        write_json(directory / "inference_result.json", result)
        print(name + ": " + result["status"], flush=True)

    hd = observations["HalfDome"]
    reference_comparison = dict(
        saved_reference=str(args.reference), saved_reference_report=old_report,
        fractional_binned_difference=(hd["noisy"]-old_x).tolist(),
        relative_l2_difference=float(np.linalg.norm(hd["noisy"]-old_x)/np.linalg.norm(old_x)),
        max_absolute_context_difference=float(np.max(np.abs(hd["context"]-old_context))),
        old_context=old_context.tolist(), new_context=hd["context"].tolist(),
        note="Raw signed differences are retained; only the L2 metric is relative. All FLAMINGO maps share the fresh control's exact cached noise arrays.")
    reference_comparison["binned_difference"] = reference_comparison.pop("fractional_binned_difference")
    write_json(output / "halfdome_reference_check.json", reference_comparison)

    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 11})
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    for name, obs in observations.items():
        color, label = COLORS[name], LABELS[name]
        axes[0, 0].plot(ell, obs["clean"], color=color, label=label)
        axes[0, 1].plot(ell, obs["noisy"], color=color, label=label)
        axes[1, 0].plot(ell, 100*(obs["clean"]/hd["clean"]-1), color=color)
        axes[1, 1].plot(ell, obs["noisy"]-hd["noisy"], color=color)
    axes[0, 0].set_yscale("log")
    axes[0, 1].set_yscale("symlog", linthresh=1e-13)
    axes[0, 0].set_ylabel(r"Clean masked $D_\ell^{yy}$")
    axes[0, 1].set_ylabel(r"Noisy masked cross $D_\ell^{yy}$")
    axes[1, 0].set_ylabel("Clean difference from HalfDome [%]")
    axes[1, 1].set_ylabel(r"Noisy $D_\ell-D_{\ell,\mathrm{HalfDome}}$")
    for ax in axes.ravel():
        ax.set_xscale("log")
        ax.grid(alpha=.2)
    for ax in axes[1]:
        ax.axhline(0, color="grey", lw=.7)
        ax.set_xlabel(r"$\ell$")
    axes[0, 0].legend(fontsize=9)
    fig.suptitle("FLAMINGO versus HalfDome: identical beam, mask and fixed SO noise\n"
                 "2 arcmin; mask support 0.4; baseline deproj0; seed 12345; original 40 bins")
    fig.tight_layout()
    fig.savefig(output / "spectra_comparison.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    fiducial = observations["L1_m9"]
    for name, obs in observations.items():
        axes[0].plot(np.arange(1, 10), obs["context"], "o-", color=COLORS[name], label=LABELS[name])
        axes[1].plot(ell, obs["signal_noise"], color=COLORS[name])
        if name != "HalfDome":
            axes[2].plot(ell, 100*(obs["clean"]/fiducial["clean"]-1), color=COLORS[name])
    reference = contract["reference_context"]
    axes[0].fill_between(np.arange(1, 10), reference.min(axis=0), reference.max(axis=0),
                         alpha=.15, color="grey", label="Saved training-reference range")
    axes[0].set_yscale("symlog", linthresh=1)
    axes[0].set_xlabel("Saved standardized MOPED coordinate")
    axes[0].set_ylabel("Compressed observation")
    axes[0].legend(fontsize=7)
    axes[1].set_xlabel(r"$\ell$")
    axes[1].set_ylabel(r"Signal-noise cross terms in $D_\ell$")
    axes[2].set_xlabel(r"$\ell$")
    axes[2].set_ylabel("Clean difference from FLAMINGO fiducial [%]")
    for ax in axes:
        ax.grid(alpha=.2)
    for ax in axes[1:]:
        ax.set_xscale("log")
    fig.tight_layout()
    fig.savefig(output / "moped_and_feedback_diagnostics.png", dpi=180)
    plt.close(fig)

    # Show all nine marginal distributions when posterior sampling succeeds.
    # The historical reference is explicitly distinct from the fresh control.
    fig, axes = plt.subplots(3, 3, figsize=(12, 9))
    for index, ax in enumerate(axes.ravel()):
        ax.hist(old_samples[:, index], bins=50, density=True, histtype="step",
                color="grey", ls="--", label="Historical HalfDome reference")
        for name, array in samples.items():
            ax.hist(array[:, index], bins=50, density=True, histtype="step", color=COLORS[name], label=LABELS[name])
        ax.set_xlabel(str(contract["param_names"][index]))
        ax.set_ylabel("Posterior density")
    axes[0, 0].legend(fontsize=7)
    missing = [name for name, result in all_results.items() if result["status"] != "posterior_sampled"]
    fig.suptitle("Unchanged MOPED/SBI posterior marginals\nNo posterior available: " + (", ".join(missing) or "none"), fontsize=12)
    fig.tight_layout()
    fig.savefig(output / "posterior_marginals.png", dpi=180)
    plt.close(fig)

    with (output / "spectra_binned.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["model", "ell", "clean_masked_dl", "noisy_cross_dl", "unmasked_beamed_dl", "noise_noise_dl", "signal_noise_dl"])
        for name, obs in observations.items():
            for index, multipole in enumerate(ell):
                writer.writerow([name, multipole] + [obs[key][index] for key in ("clean", "noisy", "unmasked", "noise_noise", "signal_noise")])
    with (output / "posterior_summary.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["model", "parameter", "status", "mean", "std", "q16", "median", "q84"])
        for name, result in all_results.items():
            for index, parameter in enumerate(contract["param_names"]):
                writer.writerow([name, parameter, result["status"]] +
                                [result[key][index] if key in result else "" for key in ("mean", "std", "q16", "median", "q84")])
    report = dict(status="comparison_completed", experiment_id=manifest["experiment_id"],
                  model_sha256=manifest["files"]["density_estimator.pkl"], training=training,
                  original_prior_low=contract["low"].tolist(), original_prior_high=contract["high"].tolist(),
                  models=all_results, cache=cache, reference_comparison=reference_comparison,
                  helpers_sha256={p.name: sha256(p) for p in args.helpers.glob("*.py")},
                  limitations=["Fixed-noise conditional model; not noise-marginalized SO constraints",
                               "Different signal phases, cosmology, and gas models can cause out-of-distribution contexts",
                               "FLAMINGO L1 maps are lensed and integrated to z=3; no empirical corrections applied",
                               "Saved training-reference ranges are heuristic diagnostics, not hard support bounds"])
    write_json(output / "comparison_summary.json", report)
    files = {p.relative_to(root).as_posix(): sha256(p) for p in output.iterdir() if p.is_file()}
    for path in (root / "results").rglob("*"):
        if path.is_file():
            files[path.relative_to(root).as_posix()] = sha256(path)
    write_json(output / "complete.json", dict(status="comparison_completed", files=files,
               posterior_available=list(samples), posterior_unavailable=missing))
    print("Comparison complete; posterior availability: " + str(list(samples)), flush=True)


if __name__ == "__main__":
    main()
