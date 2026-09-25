#!/usr/bin/env python3
"""Compare unbinned-to-MOPED with 40-bin MOPED and direct 40-bin inference.

All methods use the historical fixed-noise 524k dataset, original priors and
identical train/validation/test rows. No map synthesis or noise replacement.
"""
import argparse
import json
from pathlib import Path
import pickle
import shutil
import time

import numpy as np

from export_so_moped_bundle import sha256
from prepare_validate_battaglia12_sbi_observation import align_to_ell, read_profile
from run_so_sbi_compression_comparison import train, evaluate, load_npz, write_csv, LABELS
from so_moped_local import prepare_context, posterior_samples
from so_sbi_compression import (FIDUCIAL, PARAM_NAMES, array_digest, check_pair,
    load_dataset, metrics_from_samples, pearson_columns, project, save_npz,
    split_rows, write_json)
from so_unbinned_moped import (fit_unbinned, open_raw, project_observation,
    require, validate_rebin, write_projected)

METHODS = ("bins40", "moped40", "unbinned_moped")
DISPLAY = {"bins40": "40-bin NPE", "moped40": "40 bins -> MOPED",
           "unbinned_moped": "Unbinned -> MOPED"}
COLORS = {"bins40": "#0072B2", "moped40": "#D55E00", "unbinned_moped": "#7B3294"}
DEFAULT_BASELINE = Path("/lustre/work/kristero10/adrian_9param_compression_baseline_deproj0_bestval_20260909")
DEFAULT_OBSERVATION = Path("/lustre/work/kristero10/so9_fisher_moped524k_20260913/fixed_noise_observation")


def prepared_dataset(path):
    result = load_dataset(path)
    with np.load(path, allow_pickle=False) as data:
        result["ell_unbinned"] = data["ell_unbinned"]
    return result


def prepare(args):
    require(not args.root.exists(), "Preparation requires a fresh output root")
    started = time.monotonic()
    baseline = json.loads((args.baseline_root / "experiment.json").read_text())
    noisy_path = args.dataset or Path(baseline["dataset"])
    clean_path = args.clean_dataset or Path(baseline["clean_dataset"])
    noisy, clean = prepared_dataset(noisy_path), prepared_dataset(clean_path)
    check_pair(noisy, clean)
    require(len(noisy["theta"]) == args.expected_rows == baseline["n_rows"], "Unexpected dataset size")
    digest = array_digest(noisy["theta"], noisy["x"], clean["x"], noisy["sobol_global_row"],
        noisy["prior_low"], noisy["prior_high"], noisy["bin_ell_min"], noisy["bin_ell_max"])
    require(digest == baseline["input_digest"], "Prepared data differ from the baseline training input")
    shared = load_npz(args.baseline_root / "shared.npz")
    np.testing.assert_array_equal(shared["theta"], noisy["theta"])
    np.testing.assert_array_equal(shared["low"], noisy["prior_low"])
    np.testing.assert_array_equal(shared["high"], noisy["prior_high"])
    np.testing.assert_array_equal(shared["sobol_global_row"], noisy["sobol_global_row"])
    split = split_rows(noisy["theta"], shared["low"], shared["high"],
        baseline["holdout_last_n"], baseline["validation_fraction"], baseline["seed"])
    for name, indices in split.items():
        np.testing.assert_array_equal(indices, shared[f"{name}_indices"])
    for method in ("bins40", "moped"):
        record = json.loads((args.baseline_root / method / "training_complete.json").read_text())
        require(record["experiment_id"] == baseline["experiment_id"], "Baseline checkpoint experiment differs")
        require(record.get("converged_by_early_stopping") or
                record.get("weights_selection") == "best_validation_snapshot", "Baseline best weights are unverified")
    source_paths = [args.baseline_root / "experiment.json", args.baseline_root / "shared.npz",
                    noisy_path, clean_path]
    raw_data, ell, raw_paths = [], None, []
    for dataset, raw_override, meta_override in ((noisy, args.raw_noisy, args.noisy_metadata),
                                                (clean, args.raw_clean, args.clean_metadata)):
        metadata = json.loads(str(dataset["metadata_json"]))
        path = raw_override or Path(metadata["source_cl_path"])
        meta = meta_override or Path(metadata["source_metadata_path"])
        raw, current_ell = open_raw(path, meta, dataset, args.ell_min, args.ell_max)
        if ell is not None:
            np.testing.assert_array_equal(ell, current_ell)
        ell = current_ell
        raw_data.append(raw)
        raw_paths.append(path)
        source_paths.extend([path, meta])
    args.root.mkdir(parents=True)
    write_json(args.root / "preparation_started.json", dict(
        baseline_root=str(args.baseline_root), raw_files=[str(p) for p in raw_paths],
        ell_min=args.ell_min, ell_max=args.ell_max, n_rows=args.expected_rows))
    rebin_checks = []
    for raw, dataset in zip(raw_data, (noisy, clean)):
        print("Checking all unbinned rows against", str(dataset["product"]), flush=True)
        rebin_checks.append(validate_rebin(raw, ell, dataset, args.block_rows))
    print("Raw/40-bin row and spectral checks passed", flush=True)
    transform, diagnostics = fit_unbinned(*raw_data, ell, shared["theta"],
        shared["fit_indices"], shared["low"], shared["high"],
        local_n=baseline["moped_local_n"], shrinkage=baseline["covariance_shrinkage"],
        rcond=baseline["moped_rcond"], feature_block=args.feature_block)
    transform = write_projected(raw_data[0], ell, transform, shared["fit_indices"],
        args.root / "unbinned_moped_x.npy", args.block_rows)
    save_npz(args.root / "unbinned_moped_transform.npz", **transform)
    save_npz(args.root / "unbinned_moped_diagnostics.npz", **diagnostics)
    save_npz(args.root / "shared.npz", **shared)
    imported = {}
    for source_method, target_method in (("bins40", "bins40"), ("moped", "moped40")):
        source = args.baseline_root / source_method
        destination = args.root / target_method
        destination.mkdir()
        record = json.loads((source / "training_complete.json").read_text())
        for source_name, destination_path in (
            (source / "density_estimator.pkl", destination / "density_estimator.pkl"),
            (source / "training_complete.json", destination / "source_training_complete.json"),
            (args.baseline_root / f"{source_method}_transform.npz", args.root / f"{target_method}_transform.npz"),
            (args.baseline_root / f"{source_method}_x.npy", args.root / f"{target_method}_x.npy")):
            shutil.copy2(source_name, destination_path)
            source_paths.append(source_name)
        existing_transform = load_npz(args.root / f"{target_method}_transform.npz")
        existing_x = np.load(args.root / f"{target_method}_x.npy", mmap_mode="r")
        reference = shared["fit_indices"][:512]
        np.testing.assert_allclose(project(noisy["x"][reference], existing_transform),
                                   existing_x[reference], rtol=2e-5, atol=2e-5)
        imported[target_method] = record
    sources = [Path(__file__).resolve()] + [Path(__file__).with_name(name).resolve() for name in
        ("so_unbinned_moped.py", "so_sbi_compression.py", "sbi_for_cluster.py",
         "run_so_sbi_compression_comparison.py", "so_moped_local.py")]
    print("Hashing input spectra and baseline checkpoints for provenance", flush=True)
    provenance = {str(path.resolve()): sha256(path) for path in source_paths + sources}
    config = dict(baseline)
    config.pop("experiment_id")
    config.update(methods=list(METHODS), baseline_root=str(args.baseline_root.resolve()),
        dataset=str(noisy_path.resolve()), clean_dataset=str(clean_path.resolve()),
        source_experiment_id=baseline["experiment_id"], input_digest=digest,
        ell_min=int(ell[0]), ell_max=int(ell[-1]), unbinned_dimensions=len(ell),
        unbinned_moped_components=int(transform["matrix"].shape[1]),
        unbinned_moped_fisher_relative_error=float(diagnostics["fisher_relative_error"]),
        unbinned_derivative_relative_change_half=diagnostics["derivative_relative_change_half"].tolist(),
        unbinned_clean_fit_rmse_over_noise_max=float(diagnostics["clean_fit_rmse_over_noise_std"].max()),
        local_dataset_indices=diagnostics["local_dataset_indices"].tolist(),
        rebin_checks=rebin_checks, source_sha256=provenance,
        observation_directory=str(args.observation.resolve()),
        noise_assumption="Historical fixed-noise dataset; conditional comparison, not independent-noise calibration",
        synthetic_smoke=bool(args.synthetic_smoke), preparation_seconds=time.monotonic()-started)
    config["experiment_id"] = array_digest(np.frombuffer(json.dumps(config, sort_keys=True).encode(), dtype=np.uint8))
    for method, record in imported.items():
        write_json(args.root / method / "training_complete.json", dict(record,
            experiment_id=config["experiment_id"], method=method,
            source_experiment_id=record["experiment_id"], training_reused=True))
    write_json(args.root / "experiment.json", config)
    artifacts = [p for p in args.root.glob("*.np*")] + [args.root / "experiment.json"]
    write_json(args.root / "prepare_complete.json", dict(experiment_id=config["experiment_id"],
        artifacts={p.name: sha256(p) for p in artifacts}))
    print("Prepared:", args.root, flush=True)


def fixed_observation(root, method, config, shared):
    """Use the same verified fixed-noise Battaglia12 spectrum for all methods."""
    run = root / method / "fiducial"
    marker = run / "complete.json"
    if marker.exists():
        saved = json.loads(marker.read_text())
        require(saved["experiment_id"] == config["experiment_id"], "Stale fiducial posterior")
        require(sha256(run / "posterior_samples.npy") == saved["samples_sha256"], "Changed posterior samples")
        return
    observation_root = Path(config["observation_directory"])
    receipt = json.loads((observation_root / "simulation_complete.json").read_text())
    np.testing.assert_allclose(receipt["request"]["theta"], FIDUCIAL, rtol=0, atol=1e-14)
    settings = dict(item.split("=", 1) for item in receipt["request"]["command"] if "=" in item)
    require(settings["noise_seed"] == settings["mask_seed"] == "12345", "Observation noise/mask contract differs")
    raw_path = observation_root / "raw" / Path(receipt["raw_profile"]).name
    require(sha256(raw_path) == receipt["raw_sha256"], "Observation spectrum checksum differs")
    transform = load_npz(root / f"{method}_transform.npz")
    if method == "unbinned_moped":
        ell = transform["ell"]
        spectrum = align_to_ell(read_profile(raw_path), ell)
        context = project_observation(spectrum, ell, transform)
    else:
        with np.load(Path(config["dataset"]), allow_pickle=False) as dataset:
            contract = {key: dataset[key] for key in ("ell_unbinned", "bin_ell_min", "bin_ell_max", "metadata_json")}
        _, context = prepare_context(raw_path, contract, transform)
    values = np.load(root / f"{method}_x.npy", mmap_mode="r")
    contract = dict(low=shared["low"], high=shared["high"],
                    reference_context=np.asarray(values[shared["fit_indices"][:512]]))
    run.mkdir(parents=True, exist_ok=True)
    save_npz(run / "observation.npz", context=context, truth=FIDUCIAL,
             param_names=np.asarray(PARAM_NAMES), raw_sha256=np.asarray(sha256(raw_path)))
    with (root / method / "density_estimator.pkl").open("rb") as stream:
        estimator = pickle.load(stream).cpu().eval()
    samples, proposals, acceptance = posterior_samples(estimator, context, contract,
        count=10000, seed=config["seed"]+100000, max_proposals=1000000,
        seconds=300, pilot_count=20000, diagnostics_dir=run)
    np.save(run / "posterior_samples.npy", samples)
    write_json(marker, dict(experiment_id=config["experiment_id"], n_samples=len(samples),
        proposals=proposals, acceptance=acceptance, noise_seed=12345,
        samples_sha256=sha256(run / "posterior_samples.npy"), raw_sha256=sha256(raw_path)))


def save_figure(fig, directory, stem):
    import matplotlib.pyplot as plt
    for suffix in ("png", "pdf"):
        fig.savefig(directory / f"{stem}.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def corner(root, output, shared, samples, truth, title, stem):
    from getdist import MCSamples, plots
    names = [f"p{i}" for i in range(9)]
    ranges = {name: (float(shared["low"][i]), float(shared["high"][i])) for i, name in enumerate(names)}
    chains = [MCSamples(samples=samples[method], names=names, labels=LABELS, ranges=ranges,
        label=DISPLAY[method], settings={"smooth_scale_1D": .3, "smooth_scale_2D": .4}) for method in METHODS]
    plot = plots.get_subplot_plotter(width_inch=13)
    plot.settings.axes_fontsize, plot.settings.lab_fontsize = 9, 11
    plot.settings.legend_fontsize = 11
    plot.settings.figure_legend_frame = False
    plot.triangle_plot(chains, filled=[False, False, True], markers=truth,
        contour_colors=[COLORS[m] for m in METHODS],
        line_args=[dict(color=COLORS[m], ls="--" if m == "bins40" else "-") for m in METHODS],
        marker_args=dict(color="black", ls=":", lw=.7))
    plot.fig.suptitle(title, y=1.02, fontsize=14)
    save_figure(plot.fig, output, stem)


def summarize(root, config, shared):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output = root / "summary"
    output.mkdir(exist_ok=True)
    records, all_metrics, fiducial_samples, training = [], {}, {}, {}
    example = int(shared["example_index"])
    heldout_samples = {}
    for method in METHODS:
        run = root / method
        status = json.loads((run / "evaluation/evaluation_complete.json").read_text())
        require(status["experiment_id"] == config["experiment_id"] and
                status["n_test"] == config["n_test"], "Incomplete or different test evaluation")
        training[method] = json.loads((run / "training_complete.json").read_text())
        values = []
        for idx in shared["test_indices"]:
            saved = load_npz(run / f"evaluation/profiles/row{idx}.npz")
            require(str(saved["experiment_id"]) == config["experiment_id"] and
                    str(saved["method"]) == method, "Stale evaluation samples")
            np.testing.assert_array_equal(saved["truth"], shared["theta"][idx])
            require(len(saved["samples"]) == config["posterior_samples"], "Wrong posterior count")
            values.append(metrics_from_samples(saved["samples"], saved["truth"], shared["low"], shared["high"]))
            if int(idx) == example:
                heldout_samples[method] = saved["samples"]
        combined = {key: np.stack([v[key] for v in values]) for key in values[0]}
        all_metrics[method] = combined
        correlations = pearson_columns(combined["truth"], combined["mean"])
        for j, name in enumerate(PARAM_NAMES):
            records.append(dict(method=method, parameter=name, n_test=len(values),
                rmse_prior=float(np.sqrt(np.mean(combined["normalized_error_prior"][:, j]**2))),
                rms_pull=float(np.sqrt(np.mean(combined["pull"][:, j]**2))),
                mean_std_prior=float(combined["std_over_prior"][:, j].mean()),
                coverage68=float(combined["coverage68"][:, j].mean()),
                coverage95=float(combined["coverage95"][:, j].mean()), pearson_r=float(correlations[j])))
        fiducial = json.loads((run / "fiducial/complete.json").read_text())
        require(fiducial["experiment_id"] == config["experiment_id"], "Different fiducial posterior")
        sample_path = run / "fiducial/posterior_samples.npy"
        require(sha256(sample_path) == fiducial["samples_sha256"], "Changed fiducial samples")
        fiducial_samples[method] = np.load(sample_path, allow_pickle=False)
        metrics_from_samples(fiducial_samples[method], FIDUCIAL, shared["low"], shared["high"])
    write_csv(output / "per_parameter_metrics.csv", records)
    save_npz(output / "heldout_metrics.npz", test_indices=shared["test_indices"],
        sobol_global_row=shared["sobol_global_row"][shared["test_indices"]],
        param_names=np.asarray(PARAM_NAMES), experiment_id=np.asarray(config["experiment_id"]),
        **{f"{method}_{key}": value for method, metrics in all_metrics.items()
           for key, value in metrics.items()})
    aggregate = [dict(method=m, rmse_prior=float(np.sqrt(np.mean(v["normalized_error_prior"]**2))),
        rms_pull=float(np.sqrt(np.mean(v["pull"]**2))), coverage68=float(v["coverage68"].mean()),
        coverage95=float(v["coverage95"].mean()), mean_std_prior=float(v["std_over_prior"].mean()))
        for m, v in all_metrics.items()]
    write_csv(output / "aggregate_metrics.csv", aggregate)
    fiducial_table = []
    for method, samples in fiducial_samples.items():
        quantiles = np.quantile(samples, [.16, .5, .84], axis=0)
        for j, name in enumerate(PARAM_NAMES):
            fiducial_table.append(dict(method=method, parameter=name, truth=FIDUCIAL[j],
                mean=float(samples[:, j].mean()), std=float(samples[:, j].std(ddof=1)),
                q16=quantiles[0, j], median=quantiles[1, j], q84=quantiles[2, j]))
    write_csv(output / "battaglia12_parameter_intervals.csv", fiducial_table)
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 10})
    notice = "SYNTHETIC SOFTWARE SMOKE TEST" if config["synthetic_smoke"] else "Historical fixed-noise dataset"
    title = f"{notice}; {config['n_train']:,} training/validation rows; {config['n_test']} shared test rows"
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, key, label in zip(axes.flat, ("rmse_prior", "rms_pull", "mean_std_prior", "coverage68", "coverage95", "pearson_r"),
        ("RMSE / prior width", "RMS standardized error", "Mean posterior std / prior width",
         "68% interval coverage", "95% interval coverage", "Pearson correlation")):
        for method in METHODS:
            ax.plot(range(9), [r[key] for r in records if r["method"] == method], "o-", label=DISPLAY[method], color=COLORS[method])
        if key in ("rms_pull", "coverage68", "coverage95"):
            ax.axhline({"rms_pull": 1, "coverage68": .68, "coverage95": .95}[key], color="gray", ls=":")
        ax.set_xticks(range(9))
        ax.set_xticklabels([f"${v}$" for v in LABELS], rotation=45, ha="right")
        ax.set_ylabel(label)
        ax.grid(alpha=.2)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    save_figure(fig, output, "unbinned_vs_40bin_metrics")
    corner(root, output, shared, fiducial_samples, FIDUCIAL,
        f"Battaglia12: unbinned MOPED versus 40-bin inference\n{notice}; same observation, noise seed 12345",
        "battaglia12_unbinned_vs_40bin_corner")
    corner(root, output, shared, heldout_samples, shared["theta"][example],
        f"Shared held-out row {example} (not exactly Battaglia12)\n{title}", "heldout_unbinned_vs_40bin_corner")
    write_json(output / "comparison_summary.json", dict(experiment_id=config["experiment_id"],
        methods=list(METHODS), aggregate=aggregate, synthetic_smoke=config["synthetic_smoke"],
        training=training, noise_assumption=config["noise_assumption"],
        limitations=["Conditional fixed-noise comparison; not independent-noise calibration.",
            "MOPED uses a local quadratic derivative fit and regularized residual covariance.",
            "Original and unbinned standardizations operate before compression; this compares full pipelines.",
            "Check early-stopping status and held-out coverage before interpreting narrower contours."]))
    write_json(output / "complete.json", dict(status="completed_unbinned_moped_comparison",
        artifacts={p.name: sha256(p) for p in output.iterdir() if p.is_file() and p.name != "complete.json"}))
    print(json.dumps(aggregate, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "train", "evaluate", "summarize"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--clean-dataset", type=Path)
    parser.add_argument("--raw-noisy", type=Path)
    parser.add_argument("--raw-clean", type=Path)
    parser.add_argument("--noisy-metadata", type=Path)
    parser.add_argument("--clean-metadata", type=Path)
    parser.add_argument("--observation", type=Path, default=DEFAULT_OBSERVATION)
    parser.add_argument("--expected-rows", type=int, default=524288)
    parser.add_argument("--ell-min", type=int, default=80)
    parser.add_argument("--ell-max", type=int, default=7979)
    parser.add_argument("--feature-block", type=int, default=64)
    parser.add_argument("--block-rows", type=int, default=2048)
    parser.add_argument("--method", choices=METHODS, default="unbinned_moped")
    parser.add_argument("--synthetic-smoke", action="store_true")
    parser.add_argument("--max-proposals", type=int,
                        help="Evaluation-only proposal budget; may extend the reference limit")
    parser.add_argument("--sampling-seconds", type=float,
                        help="Evaluation-only time budget per held-out row")
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(args)
        return
    require((args.root / "prepare_complete.json").exists(), "Preparation is incomplete")
    config = json.loads((args.root / "experiment.json").read_text())
    shared = load_npz(args.root / "shared.npz")
    if args.stage == "train":
        require(args.method == "unbinned_moped", "The two 40-bin checkpoints are reused, not overwritten")
        train(args.root, args.method, config, shared)
    elif args.stage == "evaluate":
        runtime_path = args.root / args.method / "evaluation_runtime.json"
        runtime = json.loads(runtime_path.read_text()) if runtime_path.exists() else {}
        require(not runtime or runtime["experiment_id"] == config["experiment_id"],
                "Evaluation runtime belongs to another experiment")
        effective = dict(config)
        for key in ("max_proposals", "sampling_seconds"):
            requested = getattr(args, key)
            effective[key] = requested if requested is not None else runtime.get(key, config[key])
            require(effective[key] >= config[key] and np.isfinite(effective[key]),
                    "Evaluation budgets must be finite and at least the reference limits")
        write_json(runtime_path, dict(experiment_id=config["experiment_id"],
            max_proposals=effective["max_proposals"], sampling_seconds=effective["sampling_seconds"],
            reference_max_proposals=config["max_proposals"],
            reference_sampling_seconds=config["sampling_seconds"],
            posterior_samples=config["posterior_samples"],
            explanation="Bounded rejection sampling of the same model/prior; no change to posterior target"))
        # Produce fiducial diagnostics even if one or more held-out rows fail.
        errors = []
        for action in (evaluate, fixed_observation):
            try:
                action(args.root, args.method, effective, shared)
            except Exception as exc:
                errors.append(f"{action.__name__}: {exc}")
        if errors:
            write_json(args.root / args.method / "stage_failures.json", dict(errors=errors))
            raise RuntimeError("; ".join(errors))
        write_json(args.root / args.method / "stage_failures.json", dict(
            errors=[], status="evaluation_completed", experiment_id=config["experiment_id"]))
    else:
        summarize(args.root, config, shared)


if __name__ == "__main__":
    main()
