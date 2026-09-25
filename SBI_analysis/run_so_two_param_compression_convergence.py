#!/usr/bin/env python3
"""Matched P0/beta 40-bin versus MOPED convergence, with train-only compression.

Input is a completed NPZ from combine_so_two_param_baseline_deproj0.py.
Precomputed full-dataset x_moped is deliberately not used.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from so_sbi_compression import (
    FIDUCIAL, PARAM_NAMES, array_digest, asinh_coordinates, finish_transform,
    fit_asinh, metrics_from_samples, moped_basis, pearson_columns, project,
    quadratic_design, save_npz, shrunk_covariance, write_json,
)

METHODS = ("bins40", "moped")
TARGETS = ("P0", "beta")
TRUTH = FIDUCIAL[[0, 2]]
PRODUCT = "masked_baseline_noise_cross_deproj0"
DEFAULT_DATA = Path("/lustre/work/kristero10/two_param_compression_convergence_32k_v1/prepared/dataset.npz")
DEFAULT_ROOT = Path("/lustre/work/kristero10/two_param_compression_convergence_32k_v1")
DEFAULT_SIZES = "256,512,1024,2048,4096,8192,16384,24576,max"


def read_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return dict(z)


def find_datasets(path):
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Dataset directory is not accessible: {path}")
    paths = sorted(set(path.glob("*.npz")) | set(path.glob("prepared/*.npz"))
                   | set(path.glob("*/prepared/*.npz")))
    matches = []
    for candidate in paths:
        with np.load(candidate, allow_pickle=False) as z:
            if {"theta", "theta_full", "x", "x_no_noise", "noise_seed"} <= set(z.files):
                matches.append(candidate)
    if not matches:
        raise FileNotFoundError(
            f"No paired prepared two-parameter NPZ found under {path}. "
            "Raw spectra or a Sobol parameter design alone are not prepared data. "
            f"Top-level entries: {[p.name for p in sorted(path.iterdir())][:40]}")
    return matches


def load_data(path, fixed_noise=False):
    """Require explicit paired spectra, seed labels and the saved prior."""
    blocks, reports = [], []
    common = ("prior_low", "prior_high", "param_names", "full_param_names",
              "ell_binned", "ell_unbinned", "bin_ell_min", "bin_ell_max")
    rows = ("theta", "theta_full", "x", "x_no_noise", "sobol_global_row",
            "noise_seed", "mask_seed")
    for source in find_datasets(path):
        d = read_npz(source)
        required = set(common + rows + ("product", "metadata_json"))
        if required - set(d):
            raise ValueError(f"{source}: missing {sorted(required - set(d))}")
        meta = json.loads(str(d["metadata_json"].item()))
        if str(d["product"].item()) != PRODUCT:
            raise ValueError(f"{source}: wrong product")
        if tuple(d["param_names"].astype(str)) != TARGETS:
            raise ValueError(f"{source}: expected only P0 and beta")
        if tuple(d["full_param_names"].astype(str)) != PARAM_NAMES:
            raise ValueError(f"{source}: wrong full parameter order")
        n = len(d["theta"])
        if d["theta"].shape != (n, 2) or d["theta_full"].shape != (n, 9):
            raise ValueError("Wrong parameter dimensions")
        if d["x"].shape != (n, 40) or d["x_no_noise"].shape != (n, 40):
            raise ValueError("Require paired noisy/clean signed 40-bin D_ell")
        if not all(np.isfinite(d[k]).all() for k in ("theta", "theta_full", "x", "x_no_noise")):
            raise ValueError("Nonfinite data")
        np.testing.assert_array_equal(d["theta"], d["theta_full"][:, [0, 2]])
        fixed = [1, 3, 4, 5, 6, 7, 8]
        np.testing.assert_allclose(d["theta_full"][:, fixed],
            np.broadcast_to(FIDUCIAL[fixed], (n, 7)), rtol=2e-6, atol=1e-8)
        if not meta.get("complete", False):
            raise ValueError(f"{source}: generation is not marked complete")
        if ("weighted mean of linear D_ell" not in meta.get("statistic", "")
                or meta.get("bin_weighting") != "2ell_plus_1"):
            raise ValueError("Unverified data statistic or bin weighting")
        if not meta.get("same_mask_all_rows", False):
            raise ValueError("A fixed mask must be recorded")
        if not fixed_noise and not meta.get("independent_noise_all_rows", False):
            raise ValueError("Independent row noise and fixed mask must be recorded")
        if not meta.get("beam_applied_to_signal") or meta.get("beam_fwhm_arcmin") != 2.0:
            raise ValueError("Expected the existing 2 arcmin signal beam")
        for key in ("sobol_global_row", "noise_seed", "mask_seed"):
            if d[key].shape != (n,) or d[key].dtype.kind not in "iu":
                raise ValueError(f"Invalid row labels: {key}")
        if not np.all(d["mask_seed"] == 12345):
            raise ValueError("Expected mask seed 12345")
        expected = (np.full(n, 12345) if fixed_noise else
                    int(meta["noise_seed_base"]) + d["sobol_global_row"])
        np.testing.assert_array_equal(d["noise_seed"], expected)
        low, high = d["prior_low"], d["prior_high"]
        if low.shape != (2,) or high.shape != (2,) or not np.all(high > low):
            raise ValueError("Missing or invalid saved prior bounds")
        if not np.all((TRUTH > low) & (TRUTH < high)):
            raise ValueError("Battaglia12 is outside the supplied prior")
        if blocks:
            for key in common:
                np.testing.assert_array_equal(blocks[0][key], d[key], err_msg=f"Blocks differ: {key}")
        reports.append(dict(path=str(source.resolve()), n_rows=n, source_metadata=meta))
        blocks.append(d)
    result = {key: blocks[0][key] for key in common}
    result.update({key: np.concatenate([b[key] for b in blocks]) for key in rows})
    for key in (("sobol_global_row",) if fixed_noise else ("sobol_global_row", "noise_seed")):
        if len(np.unique(result[key])) != len(result[key]):
            raise ValueError(f"Repeated {key}: overlapping input blocks or reused noise")
    if np.any(result["sobol_global_row"] < 1):
        raise ValueError("Sobol labels must be one-based positive integers")
    order = np.argsort(result["sobol_global_row"], kind="stable")
    result.update({key: result[key][order] for key in rows})
    for key in ("obs", "obs_theta", "obs_theta_full", "obs_source", "obs_noise_seed", "obs_mask_seed"):
        if key not in blocks[0]:
            raise ValueError(f"Battaglia12 observation missing: {key}")
        result[key] = blocks[0][key]
        for block in blocks[1:]:
            np.testing.assert_array_equal(result[key], block[key])
    np.testing.assert_allclose(result["obs_theta_full"], FIDUCIAL, rtol=2e-6)
    np.testing.assert_allclose(result["obs_theta"], TRUTH, rtol=2e-6)
    if result["obs"].shape != (40,) or not np.isfinite(result["obs"]).all():
        raise ValueError("Invalid Battaglia12 D_ell observation")
    obs_seed = int(result["obs_noise_seed"])
    if fixed_noise and obs_seed != 12345:
        raise ValueError("Fixed-noise comparison requires the same observation seed")
    if not fixed_noise and obs_seed in result["noise_seed"]:
        raise ValueError("Observation noise seed overlaps the independent-noise training set")
    if int(result["obs_mask_seed"]) != 12345:
        raise ValueError("Observation mask differs")
    result["input_digest"] = array_digest(*(result[k] for k in common + rows),
        result["obs"], result["obs_theta_full"], result["obs_noise_seed"])
    return result, reports


def parse_sizes(specification, maximum):
    values = sorted(set(maximum if token.strip() == "max" else int(token)
                        for token in specification.split(",")))
    if not values or min(values) < 128 or max(values) > maximum:
        raise ValueError(f"Training sizes must be within 128..{maximum}; got {values}")
    return values


def fit_two_moped(noisy, clean, theta, width, local_n=2048, shrinkage=.05):
    """Local mean derivatives and paired-noise covariance, using fit rows only."""
    delta = (theta - TRUTH) / width
    nearest = np.argsort(np.linalg.norm(delta, axis=1), kind="stable")[:min(local_n, len(theta))]
    design = quadratic_design(delta[nearest])
    if len(nearest) < 80:
        raise ValueError("Need at least 80 optimization rows for 40-bin covariance")
    clean_coeff, _, rank, _ = np.linalg.lstsq(design, clean[nearest], rcond=None)
    if rank != 6:
        raise ValueError("Two-parameter quadratic mean fit is rank deficient")
    residual = noisy[nearest] - clean[nearest]
    bias_coeff = np.linalg.lstsq(design, residual, rcond=None)[0]
    residual_centered = (residual - design @ bias_coeff) * np.sqrt(
        (len(nearest)-1)/(len(nearest)-rank))
    covariance = shrunk_covariance(residual_centered, shrinkage)
    mean_coeff = clean_coeff + bias_coeff
    derivatives = mean_coeff[1:3].T
    basis = moped_basis(derivatives, covariance, 1e-8)
    if basis["matrix"].shape != (40, 2):
        raise ValueError("MOPED has fewer than two resolved mean-sensitivity directions")
    half = len(nearest)//2
    half_coeff = np.linalg.lstsq(design[:half], noisy[nearest[:half]], rcond=None)[0]
    change = np.linalg.norm(half_coeff[1:3].T-derivatives, axis=0) / np.maximum(
        np.linalg.norm(derivatives, axis=0), 1e-30)
    basis.update(center=mean_coeff[0], derivatives=derivatives, covariance=covariance,
        local_indices=nearest, derivative_relative_change_half=change,
        clean_fit_rmse_over_noise_std=np.sqrt(np.mean(
            (clean[nearest]-design@clean_coeff)**2, axis=0))/np.sqrt(np.diag(covariance)),
        regression_condition=np.asarray(np.linalg.cond(design)))
    np.testing.assert_allclose(basis["compressed_covariance"], np.eye(2), atol=1e-7)
    np.testing.assert_allclose(basis["fisher"], basis["compressed_fisher"], rtol=1e-6, atol=1e-8)
    return basis


def setup(args):
    d, reports = load_data(args.data, args.fixed_noise)
    n = len(d["theta"])
    if not 3 <= args.holdout < n-128:
        raise ValueError("Invalid held-out count")
    low, high = d["prior_low"], d["prior_high"]
    supported = np.all((d["theta"] >= low) & (d["theta"] <= high), axis=1)
    test = np.arange(n-args.holdout, n)[supported[-args.holdout:]]
    pool = np.random.default_rng(args.seed).permutation(np.flatnonzero(supported[:n-args.holdout]))
    if len(test) < 3:
        raise ValueError("Too few in-prior held-out rows")
    sizes = parse_sizes(args.sizes, len(pool))
    config = dict(algorithm_version=1, input_digest=d["input_digest"], n_rows=n,
        n_excluded=int((~supported).sum()), n_test=len(test), holdout=args.holdout,
        sizes=sizes, seed=args.seed, validation_fraction=.1, hidden_features=64,
        num_transforms=6, training_batch_size=args.batch_size,
        stop_after_epochs=args.patience, max_num_epochs=args.epochs,
        posterior_samples=args.samples, max_proposals=args.max_proposals,
        sampling_seconds=args.sampling_seconds, moped_local_n=args.moped_local_n,
        covariance_shrinkage=.05, methods=list(METHODS), density_estimator="maf",
        prior_policy="exclude", priors=dict(low=low.tolist(), high=high.tolist()),
        transform="train-only asinh(x/s), standardization, optional two-component MOPED",
        corner_samples=args.corner_samples, data_sources=reports,
        fixed_noise_diagnostic=args.fixed_noise,
        corner_source="Battaglia12, fixed noise" if args.fixed_noise else "Battaglia12, independent noise")
    config["experiment_id"] = array_digest(np.frombuffer(
        json.dumps(config, sort_keys=True).encode(), dtype=np.uint8))
    marker = args.root / "experiment.json"
    if marker.exists():
        if json.loads(marker.read_text()) != config:
            raise ValueError("Changed dataset/configuration: choose a new output root")
        saved = read_npz(args.root / "shared.npz")
        for key in ("pool_indices", "test_indices"):
            expected = pool if key == "pool_indices" else test
            np.testing.assert_array_equal(saved[key], expected, err_msg="Stale saved split")
        for key, value in d.items():
            if key != "input_digest":
                np.testing.assert_array_equal(saved[key], value, err_msg=f"Stale input: {key}")
        print("Reusing setup:", args.root)
        return
    example = test[np.argmin(np.linalg.norm((d["theta"][test]-TRUTH)/(high-low), axis=1))]
    save_npz(args.root / "shared.npz", **{k: v for k, v in d.items() if k != "input_digest"},
        pool_indices=pool, test_indices=test, example_index=np.asarray(example),
        excluded_indices=np.flatnonzero(~supported), low=low, high=high)
    write_json(marker, config)
    write_json(args.root / "dataset_audit.json", dict(
        n_rows=n, varying_parameters=list(TARGETS), fixed_parameters="Battaglia12",
        unique_noise_seeds=len(np.unique(d["noise_seed"])), same_mask=True, training_pool=len(pool),
        n_test=len(test), n_excluded=config["n_excluded"],
        noise_policy=config["corner_source"]+"; not a pixel-by-pixel RNG audit",
        sizes=sizes, input_digest=d["input_digest"]))
    print(json.dumps(config, indent=2))


def prepare_size(root, size, config, data):
    location = root / f"N{size}"
    marker = location / "experiment.json"
    local = dict(config, n_train=size, n_fit=int(.9*size),
        training_batch_size=min(config["training_batch_size"], max(16, size//4)))
    local["experiment_id"] = f"{config['experiment_id']}:N{size}"
    if marker.exists():
        if json.loads(marker.read_text()) != local:
            raise ValueError(f"Stale size preparation: {location}")
        for name in ("shared.npz", "bins40_x.npy", "moped_x.npy",
                     "bins40_transform.npz", "moped_transform.npz", "moped_diagnostics.npz"):
            if not (location / name).is_file():
                raise FileNotFoundError(location / name)
        return local
    pool = data["pool_indices"][:size]
    fit, validation = pool[:local["n_fit"]], pool[local["n_fit"]:]
    if np.intersect1d(pool, data["test_indices"]).size:
        raise ValueError("Training/test overlap")
    base = fit_asinh(data["x"][fit])
    values = asinh_coordinates(data["x"][fit], base)
    moped = fit_two_moped(values, asinh_coordinates(data["x_no_noise"][fit], base),
        data["theta"][fit], data["high"]-data["low"],
        config["moped_local_n"], config["covariance_shrinkage"])
    moped["local_dataset_indices"] = fit[moped["local_indices"]]
    save_npz(location / "moped_diagnostics.npz", **moped)
    for method, matrix, center in (("bins40", np.eye(40), np.zeros(40)),
                                  ("moped", moped["matrix"], moped["center"])):
        transform = finish_transform(base, values, matrix, center)
        save_npz(location / f"{method}_transform.npz", **transform)
        np.save(location / f"{method}_x.npy", project(data["x"], transform))
    save_npz(location / "shared.npz", theta=data["theta"], low=data["low"], high=data["high"],
        sobol_global_row=data["sobol_global_row"], param_names=data["param_names"],
        pool_indices=pool, fit_indices=fit, validation_indices=validation,
        test_indices=data["test_indices"], example_index=data["example_index"],
        obs=data["obs"], obs_theta=data["obs_theta"])
    write_json(marker, local)
    print(f"Prepared N={size}: fit={len(fit)}, validation={len(validation)}; "
          f"MOPED half-neighborhood changes={moped['derivative_relative_change_half']}", flush=True)


def run_one(args):
    from run_so_sbi_compression_comparison import train, evaluate, bounded_samples
    import torch
    location = args.root / f"N{args.n_train}"
    config = json.loads((location / "experiment.json").read_text())
    shared = read_npz(location / "shared.npz")
    train(location, args.method, config, shared)
    evaluate(location, args.method, config, shared)
    # A corner failure cannot erase the completed held-out metrics.
    run = location / args.method
    transform = read_npz(location / f"{args.method}_transform.npz")
    context = project(shared["obs"], transform)
    truth = shared["obs_theta"]
    path = run / "corner_samples.npz"
    if path.exists():
        saved = read_npz(path)
        if (str(saved["experiment_id"].item()) != config["experiment_id"]
                or str(saved["method"].item()) != args.method
                or len(saved["samples"]) != config["corner_samples"]
                or not np.array_equal(saved["context"], context)
                or not np.array_equal(saved["truth"], truth)):
            raise ValueError("Stale corner sample cache")
        metrics_from_samples(saved["samples"], truth, shared["low"], shared["high"])
    else:
        with (run / "density_estimator.pkl").open("rb") as stream:
            estimator = pickle.load(stream).cpu().eval()
        torch.manual_seed(config["seed"] + 991)
        samples, proposals, acceptance = bounded_samples(
            estimator, context, shared["low"], shared["high"],
            dict(config, posterior_samples=config["corner_samples"]))
        save_npz(path, samples=samples, context=context, truth=truth,
            observation_source=np.asarray(config["corner_source"]), method=np.asarray(args.method),
            experiment_id=np.asarray(config["experiment_id"]),
            proposals=np.asarray(proposals), acceptance=np.asarray(acceptance))
    write_json(run / "run_complete.json", dict(experiment_id=config["experiment_id"],
        method=args.method, n_train=args.n_train))


def compute_statistics(truth, mean, std, width):
    if np.any(std <= 0) or np.any(width <= 0):
        raise ValueError("Invalid normalization")
    error, pull = (mean-truth)/width, (mean-truth)/std
    if not np.isfinite(error).all() or not np.isfinite(pull).all():
        raise ValueError("Invalid metrics")
    return dict(pearson=pearson_columns(truth, mean),
        rmse_prior=np.sqrt(np.mean(error**2, axis=0)),
        rmse_std=np.sqrt(np.mean(pull**2, axis=0)),
        aggregate_rmse_prior=float(np.sqrt(np.mean(error**2))),
        aggregate_rmse_std=float(np.sqrt(np.mean(pull**2))))


def summarize(root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from getdist import MCSamples, plots
    config = json.loads((root / "experiment.json").read_text())
    data = read_npz(root / "shared.npz")
    output = root / "summary"
    output.mkdir(exist_ok=True)
    required = [(n, m) for n in config["sizes"] for m in METHODS]
    missing = [f"N{n}/{m}" for n, m in required
        if not (root / f"N{n}/{m}/run_complete.json").is_file()]
    if missing:
        write_json(output / "summary_incomplete.json", dict(missing=missing))
        raise RuntimeError(f"Missing completed runs: {missing}. Rerun those jobs; no rows are dropped.")
    records, aggregate, profile_rows, collected = [], [], [], {}
    for n, method in required:
        location = root / f"N{n}"
        expected_id = f"{config['experiment_id']}:N{n}"
        complete = json.loads((location / method / "run_complete.json").read_text())
        if complete["experiment_id"] != expected_id:
            raise ValueError("Stale run completion")
        values = np.load(location / f"{method}_x.npy", mmap_mode="r")
        metrics = []
        for idx in data["test_indices"]:
            row = read_npz(location / method / f"evaluation/profiles/row{idx}.npz")
            if (str(row["experiment_id"].item()) != expected_id
                    or str(row["method"].item()) != method
                    or len(row["samples"]) != config["posterior_samples"]):
                raise ValueError("Stale posterior row")
            np.testing.assert_array_equal(row["truth"], data["theta"][idx])
            np.testing.assert_array_equal(row["context"], values[idx])
            metric = metrics_from_samples(row["samples"], row["truth"], data["low"], data["high"])
            metrics.append(metric)
            for j, name in enumerate(TARGETS):
                profile_rows.append(dict(n_train=n, method=method, test_index=int(idx), param=name,
                    truth=float(metric["truth"][j]), mean=float(metric["mean"][j]),
                    std=float(metric["std"][j]), error_prior=float(metric["normalized_error_prior"][j]),
                    pull=float(metric["pull"][j])))
        truth, mean, std = [np.stack([m[k] for m in metrics]) for k in ("truth", "mean", "std")]
        stats = compute_statistics(truth, mean, std, data["high"]-data["low"])
        collected[n, method] = (truth, mean, std, stats)
        aggregate.append(dict(n_train=n, method=method, n_test=len(truth),
            rmse_prior=stats["aggregate_rmse_prior"], rmse_std=stats["aggregate_rmse_std"]))
        for j, name in enumerate(TARGETS):
            records.append(dict(n_train=n, method=method, param=name, n_test=len(truth),
                pearson_r=float(stats["pearson"][j]), rmse_prior=float(stats["rmse_prior"][j]),
                rmse_std=float(stats["rmse_std"][j])))
    frame, total = pd.DataFrame(records), pd.DataFrame(aggregate)
    frame.to_csv(output / "per_parameter_summary.csv", index=False)
    total.to_csv(output / "aggregate_summary.csv", index=False)
    pd.DataFrame(profile_rows).to_csv(output / "per_profile_metrics.csv", index=False)
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 9})
    colors = dict(bins40="#0072B2", moped="#D55E00")
    labels = dict(bins40="40 bins", moped="MOPED (2)")

    def save(fig, name):
        if config["fixed_noise_diagnostic"]:
            fig.suptitle("Fixed-noise diagnostic; not independent-noise uncertainties", fontsize=8)
        fig.tight_layout()
        for extension in ("png", "jpg", "pdf"):
            fig.savefig(output / f"{name}.{extension}", dpi=300, bbox_inches="tight")
        plt.close(fig)

    for key, ylabel in (("rmse_prior", "RMSE / prior range"),
                        ("rmse_std", "RMS standardized error"),
                        ("pearson_r", "Pearson correlation coefficient")):
        fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.1))
        for j, param in enumerate(TARGETS):
            for method in METHODS:
                sub = frame[(frame.param == param) & (frame.method == method)].sort_values("n_train")
                axes[j].plot(sub.n_train.to_numpy(), sub[key].to_numpy(), "o-",
                    color=colors[method], label=labels[method], ms=3)
            axes[j].set(xscale="log", xlabel="Training dataset size", ylabel=ylabel,
                        title=(r"$P_0$" if j == 0 else r"$\beta$"))
            axes[j].grid(alpha=.2)
            axes[j].legend()
        save(fig, f"{key}_vs_dataset_size")
    fig, ax = plt.subplots(figsize=(5, 3.5))
    for method in METHODS:
        sub = total[total.method == method].sort_values("n_train")
        ax.plot(sub.n_train.to_numpy(), sub.rmse_prior.to_numpy(), "o-", color=colors[method],
                label=labels[method])
    ax.set(xscale="log", xlabel="Training dataset size",
        ylabel=r"$\sqrt{\langle[(\bar\theta-\theta_{\rm true})/\Delta\theta_{\rm prior}]^2\rangle}$")
    ax.grid(alpha=.2)
    ax.legend()
    save(fig, "aggregate_rmse_prior_range_vs_dataset_size")
    maximum = max(config["sizes"])
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 6))
    for i, method in enumerate(METHODS):
        truth, mean, std, stats = collected[maximum, method]
        for j, param in enumerate(TARGETS):
            ax = axes[i, j]
            ax.errorbar(truth[:, j], mean[:, j], yerr=std[:, j], fmt=".", ms=2,
                alpha=.25, color=colors[method], lw=.4)
            limits = [data["low"][j], data["high"][j]]
            ax.plot(limits, limits, "k:", lw=.8)
            ax.set(xlim=limits, xlabel="True "+(r"$P_0$" if j == 0 else r"$\beta$"),
                ylabel="Posterior mean", title=f"{labels[method]}: r={stats['pearson'][j]:.3f}")
            ax.grid(alpha=.2)
    save(fig, "true_vs_mean_max_dataset")
    true = data["obs_theta"]
    selected = sorted(set((config["sizes"][0], config["sizes"][len(config["sizes"])//2], maximum)))
    for n in selected:
        samples = [read_npz(root / f"N{n}/{m}/corner_samples.npz") for m in METHODS]
        for method, s in zip(METHODS, samples):
            if str(s["experiment_id"].item()) != f"{config['experiment_id']}:N{n}":
                raise ValueError("Stale corner posterior")
            np.testing.assert_array_equal(s["truth"], true)
            np.testing.assert_array_equal(s["context"],
                project(data["obs"], read_npz(root / f"N{n}/{method}_transform.npz")))
            metrics_from_samples(s["samples"], true, data["low"], data["high"])
        ranges = {p: [float(data["low"][j]), float(data["high"][j])] for j, p in enumerate(TARGETS)}
        gd = [MCSamples(samples=s["samples"], names=list(TARGETS), labels=[r"P_0", r"\beta"],
            ranges=ranges, settings={"smooth_scale_1D": .3, "smooth_scale_2D": .3}) for s in samples]
        g = plots.get_subplot_plotter(width_inch=5)
        g.settings.figure_legend_frame = False
        g.settings.axes_fontsize = 8
        g.settings.lab_fontsize = 10
        g.settings.legend_fontsize = 9
        g.settings.scaling = False
        g.triangle_plot(gd, filled=[False, True], contour_colors=[colors[m] for m in METHODS],
            legend_labels=[labels[m] for m in METHODS], markers=true,
            marker_args={"color": "black", "ls": ":"},
            line_args=[{"color": colors[m], "ls": "--"} for m in METHODS])
        g.fig.suptitle(f"N={n:,}; {config['corner_source']}", fontsize=9, y=1.03)
        for extension in ("png", "jpg", "pdf"):
            g.export(str(output / f"corner_bins40_vs_moped_N{n}.{extension}"), dpi=300)
        plt.close(g.fig)
    write_json(output / "summary_complete.json", dict(
        experiment_id=config["experiment_id"], sizes=config["sizes"], n_test=config["n_test"],
        metric="sqrt(mean over profiles and parameters of ((mean-truth)/prior_width)^2)",
        corner_source=config["corner_source"], corner_truth=true.tolist(), corner_sizes=selected,
        selective_omission=False))
    print(total.to_string(index=False))
    print("Saved summary:", output)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("audit", "setup", "prepare", "run", "summarize"))
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--sizes", default=DEFAULT_SIZES)
    p.add_argument("--holdout", type=int, default=1000)
    p.add_argument("--method", choices=METHODS)
    p.add_argument("--n-train", type=int)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--patience", type=int, default=60)
    p.add_argument("--samples", type=int, default=2000)
    p.add_argument("--corner-samples", type=int, default=10000)
    p.add_argument("--max-proposals", type=int, default=200000)
    p.add_argument("--sampling-seconds", type=float, default=120.)
    p.add_argument("--moped-local-n", type=int, default=2048)
    p.add_argument("--fixed-noise", action="store_true",
                   help="Explicit conditional diagnostic; never independent-noise constraints")
    return p


def main():
    args = parser().parse_args()
    if args.stage == "audit":
        d, sources = load_data(args.data, args.fixed_noise)
        print(json.dumps(dict(n_rows=len(d["theta"]), priors=[d["prior_low"].tolist(),
            d["prior_high"].tolist()], sources=sources, unique_noise_seeds=len(np.unique(
                d["noise_seed"])), input_digest=d["input_digest"]), indent=2))
    elif args.stage == "setup":
        setup(args)
    elif args.stage == "prepare":
        config = json.loads((args.root / "experiment.json").read_text())
        data = read_npz(args.root / "shared.npz")
        for n in config["sizes"]:
            prepare_size(args.root, n, config, data)
    elif args.stage == "run":
        if args.n_train is None or args.method is None:
            raise ValueError("run requires --n-train and --method")
        run_one(args)
    else:
        summarize(args.root)


if __name__ == "__main__":
    main()
