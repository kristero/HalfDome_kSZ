#!/usr/bin/env python3
"""Prepare, train/evaluate, and plot the full-data nine-parameter compression test."""

from __future__ import annotations

import argparse
import contextlib
import csv
from functools import lru_cache
import inspect
import json
import pickle
import re
import sys
import time
from pathlib import Path

import numpy as np

from so_sbi_compression import (
    FIDUCIAL, METHODS, PARAM_NAMES, array_digest, asinh_coordinates, check_pair,
    finish_transform, fit_asinh, fit_moped, fit_pca, load_dataset,
    metrics_from_samples, pearson_columns, project, save_npz, split_rows, write_json,
)


PROJECT = Path("/home/kristero10/HalfDome_kSZ")
DATA_ROOT = PROJECT / "SBI_analysis/data_for_cluster/adrian_so_sbi_cases_ell80_7979_dataset_row_sobolrow"
DEFAULT_ROOT = Path("/lustre/work/kristero10/adrian_9param_compression_baseline_deproj0")
COLORS = ("#252525", "#0072B2", "#D55E00")
LABELS = (r"P_0", r"x_{\rm c}", r"\beta", r"\alpha_{m,P_0}",
          r"\alpha_{m,x_{\rm c}}", r"\alpha_{m,\beta}",
          r"\alpha_{z,P_0}", r"\alpha_{z,x_{\rm c}}", r"\alpha_{z,\beta}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run", "summarize", "check-runtime"))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--dataset", type=Path,
                        default=DATA_ROOT / "so_masked_baseline_noise_cross_deproj0_ell80_7979_sbi_run.npz")
    parser.add_argument("--clean-dataset", type=Path,
                        default=DATA_ROOT / "so_masked_no_noise_ell80_7979_sbi_run.npz")
    parser.add_argument("--holdout-last-n", type=int, default=500)
    parser.add_argument("--pca-components", type=int, default=9)
    parser.add_argument("--moped-local-n", type=int, default=20000)
    parser.add_argument("--covariance-shrinkage", type=float, default=.05)
    parser.add_argument("--moped-rcond", type=float, default=1e-6)
    parser.add_argument("--hidden-features", type=int, default=64)
    parser.add_argument("--num-transforms", type=int, default=6)
    parser.add_argument("--training-batch-size", type=int, default=1024)
    parser.add_argument("--stop-after-epochs", type=int, default=20)
    parser.add_argument("--max-num-epochs", type=int, default=200)
    parser.add_argument("--validation-fraction", type=float, default=.1)
    parser.add_argument("--posterior-samples", type=int, default=2000)
    parser.add_argument("--max-proposals", type=int, default=200000)
    parser.add_argument("--sampling-seconds", type=float, default=120.)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-corner", action="store_true",
                        help="Summary plots need only numpy/scipy/matplotlib; skip optional GetDist.")
    return parser.parse_args()


def load_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


@lru_cache(maxsize=1)
def configure_runtime_threads():
    from sbi_for_cluster import configure_torch_threads
    configure_torch_threads()


def config_values(args):
    fields = ("holdout_last_n", "pca_components", "moped_local_n", "covariance_shrinkage",
              "moped_rcond", "hidden_features", "num_transforms", "training_batch_size",
              "stop_after_epochs", "max_num_epochs", "validation_fraction",
              "posterior_samples", "max_proposals", "sampling_seconds", "seed")
    config = {key: getattr(args, key) for key in fields}
    for key in ("hidden_features", "num_transforms", "training_batch_size",
                "stop_after_epochs", "max_num_epochs", "posterior_samples"):
        if config[key] < 2:
            raise ValueError(f"{key} must be at least two")
    if config["max_proposals"] < config["posterior_samples"] or config["sampling_seconds"] <= 0:
        raise ValueError("Sampling limits cannot be smaller than the requested sample count")
    config.update(algorithm_version=1, density_estimator="maf", prior_policy="exclude",
                  transform="asinh_then_training_standardization", methods=list(METHODS))
    return config


def prepare(args):
    root = args.output_root
    root.mkdir(parents=True, exist_ok=True)
    noisy, clean = load_dataset(args.dataset), load_dataset(args.clean_dataset)
    check_pair(noisy, clean)
    config = config_values(args)
    config["input_digest"] = array_digest(
        noisy["theta"], noisy["x"], clean["x"], noisy["sobol_global_row"],
        noisy["prior_low"], noisy["prior_high"], noisy["bin_ell_min"], noisy["bin_ell_max"],
    )
    marker = root / "experiment.json"
    if marker.exists():
        old = json.loads(marker.read_text())
        if any(old.get(key) != value for key, value in config.items()):
            raise ValueError("Input/configuration differs from this experiment. Use a new output root.")
        required = [root / "shared.npz", root / "moped_diagnostics.npz", root / "pca_diagnostics.npz"]
        required += [root / f"{method}_transform.npz" for method in METHODS]
        required += [root / f"{method}_x.npy" for method in METHODS]
        if not all(path.is_file() for path in required):
            raise ValueError("Preparation marker exists but its files are missing. Use a new output root.")
        print("Reusing completed preparation:", marker, flush=True)
        return

    theta, x = noisy["theta"], noisy["x"]
    low, high = noisy["prior_low"], noisy["prior_high"]
    split = split_rows(theta, low, high, args.holdout_last_n, args.validation_fraction, args.seed)
    fit, test = split["fit"], split["test"]
    print(f"Rows: {len(theta):,}; excluded outside saved prior: {len(split['excluded']):,}", flush=True)
    print(f"N_train (including validation): {len(split['pool']):,}; optimization: {len(fit):,}; "
          f"held-out: {len(test):,}/{args.holdout_last_n}", flush=True)
    base = fit_asinh(x[fit])
    fit_values = asinh_coordinates(x[fit], base)
    pca = fit_pca(fit_values, args.pca_components)
    moped = fit_moped(fit_values, asinh_coordinates(clean["x"][fit], base), theta[fit],
                      high - low, args.moped_local_n, args.covariance_shrinkage, args.moped_rcond)
    moped["local_dataset_indices"] = fit[moped["local_indices"]]
    diagnostics = dict(moped)
    save_npz(root / "moped_diagnostics.npz", **diagnostics)
    save_npz(root / "pca_diagnostics.npz", **pca)
    for method, basis in (("bins40", dict(matrix=np.eye(40), center=np.zeros(40))),
                          ("pca", pca), ("moped", moped)):
        transform = finish_transform(base, fit_values, basis["matrix"], basis["center"])
        save_npz(root / f"{method}_transform.npz", **transform)
        destination = root / f"{method}_x.npy"
        temporary = destination.with_suffix(".npy.tmp")
        with temporary.open("wb") as stream:
            np.save(stream, project(x, transform))
        temporary.replace(destination)
    example = int(test[np.argmin(np.linalg.norm((theta[test] - FIDUCIAL) / (high - low), axis=1))])
    save_npz(root / "shared.npz", theta=theta, low=low, high=high,
             param_names=noisy["param_names"], sobol_global_row=noisy["sobol_global_row"],
             ell_binned=noisy["ell_binned"], example_index=np.asarray(example),
             **{f"{key}_indices": value for key, value in split.items()})
    fisher_error = float(np.linalg.norm(moped["fisher"] - moped["compressed_fisher"])
                         / max(np.linalg.norm(moped["fisher"]), 1e-30))
    config.update(dataset=str(args.dataset.resolve()), clean_dataset=str(args.clean_dataset.resolve()),
                  n_rows=len(theta), n_train=len(split["pool"]), n_fit=len(fit),
                  n_validation=len(split["validation"]), n_test=len(test),
                  excluded_rows=len(split["excluded"]), example_index=example,
                  pca_retained_variance=float(pca["explained_variance_fraction"][:args.pca_components].sum()),
                  moped_components=moped["matrix"].shape[1],
                  moped_fisher_relative_error=fisher_error,
                  moped_derivative_relative_change_half=moped["derivative_relative_change_half"].tolist(),
                  covariance_condition=float(np.linalg.cond(moped["covariance"])))
    warnings = []
    if np.max(moped["clean_fit_rmse_over_noise_std"]) > 1:
        warnings.append("Local clean quadratic fit errors exceed paired noise scatter in some bins. "
                        "Treat MOPED as an approximate compression, not a precision Fisher forecast.")
    if np.max(moped["derivative_relative_change_half"]) > .1:
        warnings.append("At least one derivative changes by >10% with half the local rows.")
    config["scientific_warnings"] = warnings
    config["experiment_id"] = array_digest(np.frombuffer(json.dumps(config, sort_keys=True).encode(), dtype=np.uint8))
    write_json(marker, config)
    print(f"PCA variance retained: {config['pca_retained_variance']:.5f}; "
          f"MOPED components: {config['moped_components']}; local Fisher error: {fisher_error:.3g}")
    print("MOPED half-neighborhood derivative changes:", config["moped_derivative_relative_change_half"])
    for warning in warnings:
        print("WARNING:", warning)
    print("Prepared experiment:", root)


def restore_best_validation_weights(estimator, inference):
    """Select SBI's best recorded snapshot even when training hits its epoch cap.

    SBI 0.22 returns the final network at the cap; its normal early-stopping
    path alone restores the best weights. Keep this version-specific access
    isolated and fail clearly if a future SBI version changes the contract.
    """
    import torch
    best = getattr(inference, "_best_model_state_dict", None)
    if not best:
        raise RuntimeError("SBI did not expose a best-validation state dictionary; "
                           "refusing to label final-epoch weights as the best model")
    current = estimator.state_dict()
    differs = set(current) != set(best) or any(
        not torch.equal(current[key], best[key]) for key in current
    )
    estimator.load_state_dict(best, strict=True)
    return dict(weights_selection="best_validation_snapshot",
                returned_weights_differed_from_best=differs)


def train(root, method, config, shared):
    import torch
    from sbi_for_cluster import (
        SBI_NPE, ForwardingCapture, build_prior_from_bounds,
        parse_validation_losses_from_training_output, save_training_output_and_validation_losses,
    )
    try:
        from sbi.neural_nets import posterior_nn
    except ImportError:
        from sbi.utils.get_nn_models import posterior_nn

    run = root / method
    run.mkdir(exist_ok=True)
    marker = run / "training_complete.json"
    if marker.exists():
        status = json.loads(marker.read_text())
        if status["experiment_id"] != config["experiment_id"]:
            raise ValueError("Training belongs to another experiment")
        if not (run / "density_estimator.pkl").is_file():
            raise FileNotFoundError(run / "density_estimator.pkl")
        if (not status.get("converged_by_early_stopping", False)
                and status.get("weights_selection") != "best_validation_snapshot"):
            print("WARNING: this legacy capped run may contain final-epoch weights, "
                  "not its reported best-validation weights. Reuse leaves it unchanged; "
                  "a corrected training comparison needs a new output root.", flush=True)
        print("Reusing trained density estimator:", run, flush=True)
        return
    configure_runtime_threads()
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    pool = shared["pool_indices"]
    theta = torch.as_tensor(shared["theta"][pool], dtype=torch.float32)
    x = torch.as_tensor(np.load(root / f"{method}_x.npy", mmap_mode="r")[pool].copy())
    prior = build_prior_from_bounds(dict(prior_low=shared["low"], prior_high=shared["high"]), "cpu")

    class FixedSplitNPE(SBI_NPE):
        def get_dataloaders(self, *positional, **keywords):
            # Keep sbi's loaders/training loop, but use the already saved split.
            original = super().get_dataloaders
            signature = inspect.signature(original)
            if "resume_training" not in signature.parameters:
                raise RuntimeError("This sbi version cannot reuse explicit train/validation indices")
            bound = signature.bind(*positional, **keywords)
            bound.arguments["resume_training"] = True
            self.train_indices = torch.arange(config["n_fit"])
            self.val_indices = torch.arange(config["n_fit"], len(pool))
            loaders = original(*bound.args, **bound.kwargs)
            if (not np.array_equal(np.asarray(self.train_indices), np.arange(config["n_fit"]))
                    or not np.array_equal(np.asarray(self.val_indices), np.arange(config["n_fit"], len(pool)))):
                raise RuntimeError("sbi changed the requested common train/validation split")
            return loaders

    builder = posterior_nn(model="maf", hidden_features=config["hidden_features"],
                           num_transforms=config["num_transforms"], z_score_x="none")
    inference = FixedSplitNPE(prior=prior, density_estimator=builder, device="cpu")
    capture = ForwardingCapture(sys.stdout)
    started = time.monotonic()
    with contextlib.redirect_stdout(capture):
        estimator = inference.append_simulations(theta, x).train(
            training_batch_size=config["training_batch_size"],
            validation_fraction=config["validation_fraction"],
            stop_after_epochs=config["stop_after_epochs"],
            max_num_epochs=config["max_num_epochs"], show_train_summary=True,
        )
    seconds = time.monotonic() - started
    output = capture.getvalue()
    validation = parse_validation_losses_from_training_output(output)
    save_training_output_and_validation_losses(run, output, validation)
    selection = restore_best_validation_weights(estimator, inference)
    print("Saved-weight selection:", selection, flush=True)
    # Save the learned estimator before any potentially slow posterior evaluation.
    estimator.eval()
    for name, value in (("density_estimator.pkl", estimator), ("prior.pkl", prior)):
        path = run / name
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("wb") as stream:
            pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
        temporary.replace(path)
    torch.save(estimator.state_dict(), run / "density_estimator_state_dict.pt")
    import sbi
    epochs = re.findall(r"Epochs trained:\s*(\d+)", output)
    write_json(marker, dict(experiment_id=config["experiment_id"], method=method,
                           training_seconds=seconds, n_train=len(pool), n_fit=config["n_fit"],
                           x_dim=x.shape[1], best_validation_performance=validation,
                           epochs_trained=int(epochs[-1]) if epochs else None,
                           converged_by_early_stopping="successfully converged" in output.lower(),
                           python=sys.version, sbi=str(sbi.__version__), torch=str(torch.__version__),
                           **selection))


def bounded_samples(estimator, context, low, high, config):
    """Direct NPE sampling with the same BoxUniform restriction as DirectPosterior.

    No inference.pkl, MCMC fallback, boundary clipping, or unbounded rejection.
    The time budget is checked between small network calls, not inside a call.
    """
    import torch
    parameters = inspect.signature(estimator.sample).parameters
    x = torch.tensor(np.asarray(context), dtype=torch.float32).reshape(1, -1)
    batches, accepted, proposals, inside_total = [], 0, 0, 0
    start = time.monotonic()
    while accepted < config["posterior_samples"]:
        if proposals >= config["max_proposals"] or time.monotonic() - start > config["sampling_seconds"]:
            raise RuntimeError(f"Direct sampling budget reached: accepted={accepted}, proposals={proposals}, "
                               f"raw acceptance={inside_total / max(proposals, 1):.6%}")
        count = min(2048, config["max_proposals"] - proposals)
        with torch.no_grad():
            if "context" in parameters:
                raw = estimator.sample(count, context=x)
            elif "condition" in parameters:
                raw = estimator.sample(torch.Size([count]), condition=x)
            else:
                raise TypeError("Unsupported saved density-estimator sample interface")
        raw = raw.detach().cpu().numpy().reshape(-1, len(low))
        if len(raw) != count or not np.isfinite(raw).all():
            raise ValueError("Invalid raw density samples")
        proposals += len(raw)
        valid = raw[np.all((raw >= low) & (raw <= high), axis=1)]
        inside_total += len(valid)
        kept = valid[:config["posterior_samples"] - accepted]
        batches.append(kept)
        accepted += len(kept)
    return np.concatenate(batches), proposals, inside_total / proposals


def evaluate(root, method, config, shared):
    import torch
    configure_runtime_threads()
    run = root / method
    rows_dir = run / "evaluation/profiles"
    rows_dir.mkdir(parents=True, exist_ok=True)
    with (run / "density_estimator.pkl").open("rb") as stream:
        estimator = pickle.load(stream)
    estimator.eval()
    values = np.load(root / f"{method}_x.npy", mmap_mode="r")
    low, high, theta = (shared[k] for k in ("low", "high", "theta"))
    failures = []
    for number, idx in enumerate(shared["test_indices"], 1):
        path = rows_dir / f"row{idx}.npz"
        if path.exists():
            saved = load_npz(path)
            if str(saved["experiment_id"].item()) != config["experiment_id"]:
                raise ValueError(f"Stale posterior checkpoint: {path}")
            if str(saved["method"].item()) != method or not np.array_equal(saved["context"], values[idx]):
                raise ValueError(f"Different method or conditioning vector in checkpoint: {path}")
            if not np.array_equal(saved["truth"], theta[idx]):
                raise ValueError(f"Stale truth vector: {path}")
            if len(saved["samples"]) != config["posterior_samples"]:
                raise ValueError(f"Wrong posterior sample count: {path}")
            metrics_from_samples(saved["samples"], theta[idx], low, high)
            continue
        torch.manual_seed(config["seed"] + int(idx))
        try:
            samples, proposals, acceptance = bounded_samples(estimator, values[idx], low, high, config)
            metrics = metrics_from_samples(samples, theta[idx], low, high)
            save_npz(path, samples=samples, proposals=np.asarray(proposals),
                     acceptance=np.asarray(acceptance), test_index=np.asarray(idx),
                     sobol_global_row=shared["sobol_global_row"][idx],
                     method=np.asarray(method), context=values[idx],
                     experiment_id=np.asarray(config["experiment_id"]), **metrics)
            print(f"{method}: held-out {number}/{len(shared['test_indices'])}, row={idx}, "
                  f"acceptance={acceptance:.2%}", flush=True)
        except RuntimeError as exc:
            failures.append(dict(index=int(idx), error=str(exc)))
            write_json(run / "evaluation/sampling_failures.json", failures)
            print(f"{method}: row {idx} FAILED: {exc}", flush=True)
    write_json(run / "evaluation/sampling_failures.json", failures)
    if failures:
        raise RuntimeError(f"{method}: {len(failures)} failed rows. Completed checkpoints retained; "
                           "no selective-subset summary will be reported.")
    write_json(run / "evaluation/evaluation_complete.json",
               dict(experiment_id=config["experiment_id"], n_test=len(shared["test_indices"])))


def save_figure(fig, output, name):
    for suffix in ("png", "jpg"):
        fig.savefig(output / f"{name}.{suffix}", dpi=300, bbox_inches="tight")


def write_csv(path, rows):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(args, config, shared):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = args.output_root
    out = root / "summary"
    out.mkdir(exist_ok=True)
    # Validate every expected row first. Never compare different successful subsets.
    all_metrics, all_records, stats = {}, [], []
    for method in METHODS:
        run = root / method
        marker = run / "evaluation/evaluation_complete.json"
        if not marker.exists() or json.loads(marker.read_text())["experiment_id"] != config["experiment_id"]:
            raise FileNotFoundError(f"Incomplete or stale evaluation: {marker}. Rerun stage 'run'.")
        collected = []
        for idx in shared["test_indices"]:
            saved = load_npz(run / f"evaluation/profiles/row{idx}.npz")
            if str(saved["experiment_id"].item()) != config["experiment_id"]:
                raise ValueError("Posterior checkpoint is from another experiment")
            if (str(saved["method"].item()) != method
                    or not np.array_equal(saved["truth"], shared["theta"][idx])
                    or len(saved["samples"]) != config["posterior_samples"]):
                raise ValueError("Posterior method, truth, or sample count mismatch")
            metric = metrics_from_samples(saved["samples"], shared["theta"][idx], shared["low"], shared["high"])
            collected.append(metric)
            for j, name in enumerate(PARAM_NAMES):
                all_records.append(dict(method=method, test_index=int(idx), param=name,
                                        **{key: float(value[j]) for key, value in metric.items()}))
        values = {key: np.stack([item[key] for item in collected]) for key in collected[0]}
        values["pearson"] = pearson_columns(values["truth"], values["mean"])
        all_metrics[method] = values
        train_meta = json.loads((run / "training_complete.json").read_text())
        if not train_meta.get("converged_by_early_stopping", False):
            print(f"WARNING: {method} has no recorded early-stopping convergence; inspect its training summary.")
        for j, name in enumerate(PARAM_NAMES):
            stats.append(dict(method=method, param=name, n_train=config["n_train"], n_test=config["n_test"],
                              pearson_r=float(values["pearson"][j]),
                              rmse=float(np.sqrt(np.mean(values["error"][:, j]**2))),
                              rmse_prior=float(np.sqrt(np.mean(values["normalized_error_prior"][:, j]**2))),
                              rmse_std=float(np.sqrt(np.mean(values["pull"][:, j]**2))),
                              mean_std_prior=float(values["std_over_prior"][:, j].mean()),
                              coverage68=float(values["coverage68"][:, j].mean()),
                              coverage95=float(values["coverage95"][:, j].mean()),
                              training_seconds=train_meta["training_seconds"]))
    for row in stats:
        baseline = next(r for r in stats if r["method"] == "bins40" and r["param"] == row["param"])
        row["rmse_prior_improvement_percent"] = (
            100 * (1 - row["rmse_prior"] / baseline["rmse_prior"])
            if baseline["rmse_prior"] > 0 else float("nan"))
    write_csv(out / "per_profile_metrics.csv", all_records)
    write_csv(out / "per_parameter_summary.csv", stats)
    aggregate = []
    for method in METHODS:
        v = all_metrics[method]
        aggregate.append(dict(method=method, n_train=config["n_train"], n_test=config["n_test"],
                              rmse_prior=float(np.sqrt(np.mean(v["normalized_error_prior"]**2))),
                              rmse_std=float(np.sqrt(np.mean(v["pull"]**2))),
                              mean_std_prior=float(v["std_over_prior"].mean()),
                              coverage68=float(v["coverage68"].mean()),
                              coverage95=float(v["coverage95"].mean())))
    write_csv(out / "aggregate_summary.csv", aggregate)
    print("\nMethod      RMSE/prior     RMS(error/std)     mean(std/prior)    68% coverage")
    for row in aggregate:
        print(f"{row['method']:8s}    {row['rmse_prior']:.5f}          {row['rmse_std']:.5f}"
              f"             {row['mean_std_prior']:.5f}           {row['coverage68']:.3f}")
    for warning in config.get("scientific_warnings", []):
        print("WARNING:", warning)

    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 8,
                         "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7})
    display = {"bins40": "40 bins", "pca": f"PCA ({config['pca_components']})",
               "moped": f"MOPED ({config['moped_components']})"}
    positions = np.arange(9)
    fig, axes = plt.subplots(1, 3, figsize=(18 / 2.54, 7 / 2.54))
    for ax, key, title in zip(axes, ("rmse_prior", "rmse_std", "mean_std_prior"),
                             ("RMSE / prior range", "RMS standardized error", "Mean std / prior range")):
        ax.bar(np.arange(3), [row[key] for row in aggregate], color=COLORS, width=.65)
        ax.set_xticks(np.arange(3))
        ax.set_xticklabels([display[m] for m in METHODS], rotation=20)
        ax.set_title(title)
        ax.grid(axis="y", alpha=.2)
    fig.tight_layout()
    save_figure(fig, out, "aggregate_metrics_comparison")
    plt.close(fig)
    for key, ylabel, filename in (
        ("pearson_r", r"Pearson $r(\theta_{\rm true},\bar\theta)$", "correlation_comparison"),
        ("rmse_prior", r"$\sqrt{\langle[(\bar\theta-\theta_{\rm true})/\Delta\theta_{\rm prior}]^2\rangle}$", "rmse_prior_range_comparison"),
        ("rmse_std", r"$\sqrt{\langle[(\bar\theta-\theta_{\rm true})/\sigma_{\rm post}]^2\rangle}$", "rmse_posterior_std_comparison"),
        ("mean_std_prior", r"$\langle\sigma_{\rm post}/\Delta\theta_{\rm prior}\rangle$", "posterior_width_comparison"),
        ("coverage68", "Empirical 68% marginal coverage", "coverage_comparison"),
    ):
        fig, ax = plt.subplots(figsize=(18 / 2.54, 9 / 2.54))
        for k, method in enumerate(METHODS):
            y = np.asarray([row[key] for row in stats if row["method"] == method])
            ax.plot(positions + (k - 1) * .16, y, "o-", color=COLORS[k], label=display[method], ms=3)
            if key == "pearson_r":
                for xx, yy in zip(positions + (k - 1) * .16, y):
                    ax.annotate(f"{yy:.2f}", (xx, yy), xytext=(0, 4 + 8*k),
                                textcoords="offset points", ha="center", fontsize=5.5, color=COLORS[k])
        if key == "coverage68":
            ax.axhline(.68, color="0.5", ls=":", lw=.8)
        if key == "rmse_std":
            ax.axhline(1, color="0.5", ls=":", lw=.8)
        ax.set_xticks(positions)
        ax.set_xticklabels([f"${label}$" for label in LABELS])
        ax.set_ylabel(ylabel)
        ax.set_title(f"Nine parameters; N = {config['n_train']:,}; shared test rows = {config['n_test']}")
        ax.grid(alpha=.2)
        ax.margins(y=.25)
        ax.legend(fontsize=7)
        fig.tight_layout()
        save_figure(fig, out, filename)
        plt.close(fig)

    for selected, filename in ((METHODS, "true_vs_mean_comparison"),
                                *[((method,), f"true_vs_mean_{method}") for method in METHODS]):
        fig, axes = plt.subplots(3, 3, figsize=(18 / 2.54, 18 / 2.54))
        for j, ax in enumerate(axes.flat):
            ax.plot([shared["low"][j], shared["high"][j]],
                    [shared["low"][j], shared["high"][j]], "k:", lw=.8)
            for method in selected:
                k, v = METHODS.index(method), all_metrics[method]
                ax.scatter(v["truth"][:, j], v["mean"][:, j], s=5, alpha=.28,
                           color=COLORS[k], rasterized=True)
                ax.text(.03, .97 - .085*k if len(selected) > 1 else .97,
                        f"{display[method]}: r = {v['pearson'][j]:.3f}", color=COLORS[k],
                        transform=ax.transAxes, va="top", fontsize=6)
            ax.set_xlabel(f"True ${LABELS[j]}$")
            ax.set_ylabel(f"Posterior mean ${LABELS[j]}$")
            ax.grid(alpha=.2)
        fig.tight_layout()
        save_figure(fig, out, filename)
        plt.close(fig)

    pca = load_npz(root / "pca_diagnostics.npz")
    moped = load_npz(root / "moped_diagnostics.npz")
    fig, axes = plt.subplots(1, 2, figsize=(18 / 2.54, 7 / 2.54))
    axes[0].plot(np.arange(1, 41), np.cumsum(pca["explained_variance_fraction"]), "o-", ms=2)
    axes[0].axvline(config["pca_components"], color="0.5", ls=":")
    axes[0].set(xlabel="PCA components", ylabel="Cumulative variance fraction")
    axes[1].semilogy(np.arange(1, 10), moped["singular_values"], "o-")
    axes[1].set(xlabel="MOPED sensitivity mode", ylabel="Whitened derivative singular value")
    fig.tight_layout()
    save_figure(fig, out, "compression_diagnostics")
    plt.close(fig)

    corner_created = False
    if not args.skip_corner:
        try:
            from getdist import MCSamples, plots
        except ImportError:
            print("GetDist unavailable; metric plots saved. Install getdist and rerun summarize for the corner.")
        else:
            idx = int(shared["example_index"])
            names = [f"p{j}" for j in range(9)]
            ranges = {name: (float(shared["low"][j]), float(shared["high"][j])) for j, name in enumerate(names)}
            roots = []
            for method in METHODS:
                saved = load_npz(root / method / f"evaluation/profiles/row{idx}.npz")
                roots.append(MCSamples(samples=saved["samples"], names=names, labels=list(LABELS),
                                       ranges=ranges, label=display[method],
                                       settings={"smooth_scale_1D": .4, "smooth_scale_2D": .4}))
            g = plots.get_subplot_plotter(width_inch=22 / 2.54)
            g.settings.scaling = False
            g.settings.axes_fontsize, g.settings.lab_fontsize, g.settings.legend_fontsize = 7, 8, 8
            g.settings.alpha_filled_add = .35
            g.settings.figure_legend_frame = False
            g.triangle_plot(roots, params=names, filled=[False, False, True],
                            contour_colors=list(COLORS),
                            line_args=[dict(color=color, ls="--", lw=1) for color in COLORS],
                            markers=shared["theta"][idx], marker_args=dict(color="black", ls=":"),
                            legend_loc="upper right")
            for legend in g.fig.legends:
                legend.set_bbox_to_anchor((.93, .91), transform=g.fig.transFigure)
            g.fig.text(.60, .97, f"Shared held-out row {idx}, not Battaglia12",
                       ha="center", va="top", fontsize=9)
            save_figure(g.fig, out, "heldout_constraints_corner")
            plt.close(g.fig)
            corner_created = True
    write_json(out / "summary_complete.json", dict(experiment_id=config["experiment_id"],
                                                   corner_created=corner_created))
    print("Summary plots and CSVs:", out)


def main():
    args = parse_args()
    if args.stage == "check-runtime":
        # Test the actual imports used here. Newer sbi does not need ArviZ/Numba.
        import torch
        import sbi
        from sbi_for_cluster import SBI_NPE
        print(json.dumps(dict(python=sys.executable, numpy=np.__version__,
                              torch=str(torch.__version__), sbi=str(sbi.__version__),
                              inference_class=SBI_NPE.__name__), indent=2))
        return 0
    if args.stage == "prepare":
        prepare(args)
        return 0
    config = json.loads((args.output_root / "experiment.json").read_text())
    shared = load_npz(args.output_root / "shared.npz")
    if args.stage == "run":
        if args.method is None:
            raise ValueError("stage run requires --method bins40, pca, or moped")
        train(args.output_root, args.method, config, shared)
        evaluate(args.output_root, args.method, config, shared)
    else:
        failure = args.output_root / "summary/summary_failure.json"
        try:
            summarize(args, config, shared)
        except Exception as error:
            write_json(failure, dict(experiment_id=config["experiment_id"],
                                    error_type=type(error).__name__, error=str(error)))
            raise
        else:
            failure.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
