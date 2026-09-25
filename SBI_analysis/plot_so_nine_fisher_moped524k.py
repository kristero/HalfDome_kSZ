#!/usr/bin/env python3
"""Overlay the historical nine-parameter MOPED NPE on the saved Fisher pilot.

The historical NPE used fixed noise. Applying it to the Fisher observation is
a diagnostic of a distribution mismatch, not a matched inference comparison.
Each method retains its own saved binning and prior; existing plots are kept.
Use --fixed-noise-posterior only for an explicitly selected existing conditional
posterior. It is labelled as a different observation, never a sampling fallback.
"""
import argparse
import json
import os
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from export_so_moped_bundle import sha256
from so_moped_local import load_checked_model, posterior_samples, prepare_context
from so_sbi_compression import FIDUCIAL, PARAM_NAMES, project, save_npz, write_json

LABELS = [r"P_0", r"x_{\rm c}", r"\beta", r"\alpha_{m,P_0}",
          r"\alpha_{m,x_{\rm c}}", r"\alpha_{m,\beta}", r"\alpha_{z,P_0}",
          r"\alpha_{z,x_{\rm c}}", r"\alpha_{z,\beta}"]


def checked_bundle(bundle):
    manifest = json.loads((bundle / "manifest.json").read_text())
    if manifest["method"] != "moped":
        raise ValueError("Expected the exported nine-parameter MOPED model")
    for name, digest in manifest["files"].items():
        if Path(name).name != name or sha256(bundle / name) != digest:
            raise ValueError(f"Missing or changed model bundle member: {name}")
    with np.load(bundle / "observation_contract.npz", allow_pickle=False) as data:
        contract = dict(data)
    with np.load(bundle / "transform.npz", allow_pickle=False) as data:
        transform = dict(data)
    np.testing.assert_array_equal(contract["param_names"], PARAM_NAMES)
    if str(contract["experiment_id"]) != manifest["experiment_id"]:
        raise ValueError("Model experiment IDs differ")
    np.testing.assert_allclose(project(contract["reference_x"], transform),
        contract["reference_context"], rtol=1e-5, atol=2e-5)
    training = json.loads((bundle / "training_complete.json").read_text())
    if training["x_dim"] != 9 or training["weights_selection"] != "best_validation_snapshot":
        raise ValueError("Expected nine coordinates and best validation weights")
    metadata = json.loads(str(contract["metadata_json"]))
    if metadata["n_rows"] != 524288:
        raise ValueError("Expected the estimator from the 524288-row dataset")
    return contract, transform, manifest, training


def load_fisher(root):
    complete = json.loads((root / "complete.json").read_text())
    for name, digest in complete["artifacts"].items():
        if Path(name).name != name or sha256(root / name) != digest:
            raise ValueError(f"Missing or changed Fisher artifact: {name}")
    with np.load(root / "fisher_arrays.npz", allow_pickle=False) as data:
        fisher = dict(data)
    np.testing.assert_array_equal(fisher["param_names"], PARAM_NAMES)
    np.testing.assert_allclose(fisher["fiducial"], FIDUCIAL, rtol=0, atol=1e-14)
    summary = json.loads((root / "analysis_summary.json").read_text())
    seed = int(fisher["observation_noise_seed"])
    matches = [Path(p) for p in summary["provenance"] if "/raw/" in p and
               f"noiseseed{seed}_" in p and "masked_baseline_noise_cross_cl" in p]
    if len(matches) != 1 or sha256(matches[0]) != summary["provenance"][str(matches[0])]:
        raise ValueError("Cannot verify the Fisher observation's raw spectrum")
    return fisher, summary, matches[0]


def checked_fixed_posterior(root, contract, transform, manifest):
    """Verify the saved conditional posterior and its original observation."""
    report_path = root / "diagnostics/fixed_noise_sampling_summary.json"
    report = json.loads(report_path.read_text())
    simulation_path = root / "simulation_complete.json"
    simulation = json.loads(simulation_path.read_text())
    if report["experiment_id"] != manifest["experiment_id"]:
        raise ValueError("Fixed-noise posterior belongs to another model experiment")
    if not report["conditional_on_fixed_noise"] or report["noise_seed"] != 12345:
        raise ValueError("Expected the validated seed-12345 conditional posterior")
    np.testing.assert_allclose(report["theta_true"], FIDUCIAL, rtol=0, atol=1e-14)
    np.testing.assert_allclose(simulation["request"]["theta"], FIDUCIAL, rtol=0, atol=1e-14)
    settings = dict(part.split("=", 1) for part in simulation["request"]["command"] if "=" in part)
    for key, value in {"noise_seed": "12345", "mask_seed": "12345", "nside": "4096",
                       "gaussian_beam_fwhm_arcmin": "2.0"}.items():
        if settings[key] != value:
            raise ValueError(f"Unexpected fixed-noise simulation setting: {key}")
    raw = root / "raw" / Path(report["raw_profile"]).name
    sample_path = root / "posterior_samples.npy"
    if sha256(raw) != report["raw_sha256"] or report["raw_sha256"] != simulation["raw_sha256"]:
        raise ValueError("Fixed-noise observation checksum differs")
    if sha256(sample_path) != report["samples_sha256"]:
        raise ValueError("Fixed-noise posterior sample checksum differs")
    binned, context = prepare_context(raw, contract, transform)
    observation_path = root / "observation.npz"
    with np.load(observation_path, allow_pickle=False) as observation:
        np.testing.assert_array_equal(observation["param_names"], PARAM_NAMES)
        np.testing.assert_allclose(observation["theta"], FIDUCIAL, rtol=0, atol=1e-14)
        np.testing.assert_allclose(observation["x"], binned, rtol=2e-6, atol=0)
        np.testing.assert_allclose(observation["context"], context, rtol=2e-4, atol=2e-5)
        if str(observation["experiment_id"]) != manifest["experiment_id"]:
            raise ValueError("Fixed observation experiment differs")
    samples = np.load(sample_path, allow_pickle=False)
    if samples.shape != (report["n_samples"], 9):
        raise ValueError("Fixed posterior sample count differs")
    sources = [sample_path, raw, observation_path, report_path, simulation_path]
    return samples, raw, binned, context, report, sources


def make_plot(output, fisher_root, fisher, contract, moped, moped_seed):
    from getdist import MCSamples, plots
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm"})
    names = [f"p{i}" for i in range(9)]
    arrays = [np.load(fisher_root / "fiducial_forecast_samples.npy", allow_pickle=False),
              np.load(fisher_root / "observed_posterior_samples.npy", allow_pickle=False), moped]
    bounds = [(fisher["prior_low"], fisher["prior_high"])] * 2 + [(contract["low"], contract["high"])]
    roots = []
    for array, (low, high) in zip(arrays, bounds):
        selected = np.random.default_rng(82).choice(len(array), min(20000, len(array)), replace=False)
        ranges = {name: (float(low[j]), float(high[j])) for j, name in enumerate(names)}
        roots.append(MCSamples(samples=array[selected], names=names, labels=LABELS, ranges=ranges,
            settings={"smooth_scale_1D": .3, "smooth_scale_2D": .4, "fine_bins_2D": 128}))
    colors = ["#0072B2", "#D55E00", "#7B3294"]
    g = plots.get_subplot_plotter(width_inch=13)
    g.settings.axes_fontsize = 9
    g.settings.lab_fontsize = 11
    g.settings.legend_fontsize = 11
    g.settings.figure_legend_frame = False
    seed = int(fisher["observation_noise_seed"])
    g.triangle_plot(roots, filled=[False, True, False], markers=FIDUCIAL,
        contour_colors=colors,
        legend_labels=["Fiducial Fisher + hard prior", f"Observed Fisher + hard prior (seed {seed})",
                       f"MOPED NPE: 524k dataset, fixed noise (seed {moped_seed})"],
        line_args=[dict(color=colors[0], ls="--"), dict(color=colors[1]),
                   dict(color=colors[2], lw=1.8)],
        marker_args=dict(color="black", ls=":", lw=.7))
    observation_note = (f"Same observation (noise seed {seed})" if seed == moped_seed else
                        f"Different noise realizations: Fisher {seed}, MOPED {moped_seed}")
    g.fig.suptitle("Battaglia12, nine parameters: Fisher pilot and historical MOPED NPE\n"
        + observation_note + "; diagnostic comparison of different noise models",
        y=1.025, fontsize=14)
    for extension in ("png", "pdf"):
        g.fig.savefig(output / f"battaglia12_fisher_moped524k_corner.{extension}",
                      dpi=180, bbox_inches="tight")
    plt.close(g.fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fisher", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=20000, help="Sample count for fresh same-observation inference")
    parser.add_argument("--fixed-noise-posterior", type=Path,
                        help="Explicitly overlay this verified existing seed-12345 posterior instead")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260915)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Use a fresh output directory; previous plots and samples are preserved")
    started = time.monotonic()
    import torch
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    contract, transform, manifest, training = checked_bundle(args.bundle)
    fisher, fisher_summary, raw = load_fisher(args.fisher)
    # Rebin the SAME raw observation using the estimator's original float32
    # preprocessing. Feeding the new pilot bins into the old transform is wrong.
    binned, context = prepare_context(raw, contract, transform)
    estimator = load_checked_model(args.bundle, contract)
    cached = None
    extra_sources = []
    moped_seed = int(fisher["observation_noise_seed"])
    if args.fixed_noise_posterior is not None:
        samples, raw, binned, context, cached, extra_sources = checked_fixed_posterior(
            args.fixed_noise_posterior, contract, transform, manifest)
        moped_seed = int(cached["noise_seed"])
    interpretation = ("Different observations: independent-noise Fisher and fixed-noise conditional NPE; diagnostic only"
                      if cached is not None else
                      "Historical fixed-noise NPE applied to independent-noise observation; diagnostic only")
    args.output.mkdir(parents=True)
    request = dict(pbs_job_id=os.environ.get("PBS_JOBID"), model_manifest=manifest,
        training=training, observation_noise_seed=int(fisher["observation_noise_seed"]),
        raw_spectrum=str(raw), raw_sha256=sha256(raw),
        requested_samples=len(samples) if cached is not None else args.samples,
        seed=cached["sampling_seed"] if cached is not None else args.seed,
        same_raw_observation_as_fisher=cached is None, moped_observation_noise_seed=moped_seed,
        reused_verified_posterior=cached is not None,
        fisher_bins=[fisher["bin_ell_min"].tolist(), fisher["bin_ell_max"].tolist()],
        moped_bins=[contract["bin_ell_min"].tolist(), contract["bin_ell_max"].tolist()],
        fisher_prior=[fisher["prior_low"].tolist(), fisher["prior_high"].tolist()],
        moped_prior=[contract["low"].tolist(), contract["high"].tolist()],
        interpretation=interpretation,
        provenance={str(p): sha256(p) for p in [Path(__file__), args.fisher / "complete.json",
            args.bundle / "manifest.json", Path(__file__).with_name("so_moped_local.py"),
            Path(__file__).with_name("prepare_validate_battaglia12_sbi_observation.py"),
            Path(__file__).with_name("run_so_sbi_compression_comparison.py")] + extra_sources})
    write_json(args.output / "request.json", request)
    save_npz(args.output / "moped_observation.npz", binned_dell=binned, context=context,
        truth=FIDUCIAL, prior_low=contract["low"], prior_high=contract["high"],
        param_names=np.array(PARAM_NAMES))
    if cached is not None:
        proposals, acceptance = cached["proposals"], cached["raw_prior_acceptance"]
        print("Verified and reusing the explicitly selected fixed-noise posterior", flush=True)
    else:
        print("Verified 524k MOPED model and original preprocessing; probing observation support", flush=True)
        try:
            samples, proposals, acceptance = posterior_samples(estimator, context, contract,
                count=args.samples, seed=args.seed, max_proposals=1000000, seconds=300,
                pilot_count=20000, diagnostics_dir=args.output)
        except Exception as exc:
            write_json(args.output / "incomplete.json", dict(status="sampling_failed", error=str(exc)))
            raise
    if samples.shape != (request["requested_samples"], 9) or not np.isfinite(samples).all():
        raise ValueError("Invalid posterior sample array")
    if not np.all((samples >= contract["low"]) & (samples <= contract["high"])):
        raise ValueError("Samples outside the model's saved prior")
    np.save(args.output / "moped524k_posterior_samples.npy", samples)
    print(f"Accepted {len(samples)} samples; raw prior acceptance={acceptance:.3%}", flush=True)
    make_plot(args.output, args.fisher, fisher, contract, samples, moped_seed)
    summary = dict(status="completed_diagnostic_overlay", calibrated_comparison=False,
        pbs_job_id=os.environ.get("PBS_JOBID"), samples=len(samples), proposals=proposals,
        reused_verified_posterior=cached is not None, moped_observation_noise_seed=moped_seed,
        acceptance=acceptance, mean=samples.mean(axis=0).tolist(),
        std=samples.std(axis=0, ddof=1).tolist(), elapsed_seconds=time.monotonic()-started,
        limitations=[request["interpretation"], "Each method retains its own saved binning and prior.",
            "The MOPED checkpoint reached the epoch cap; early-stopping convergence was not established.",
            "The Fisher result uses 16 noise realizations and strong covariance regularization."])
    write_json(args.output / "summary.json", summary)
    write_json(args.output / "complete.json", dict(status=summary["status"],
        artifacts={p.name: sha256(p) for p in args.output.iterdir() if p.is_file()}))
    print(f"Completed: {args.output}", flush=True)


if __name__ == "__main__":
    main()
