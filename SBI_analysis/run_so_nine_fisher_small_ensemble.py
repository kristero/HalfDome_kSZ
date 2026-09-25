#!/usr/bin/env python3
"""Preliminary nine-parameter Fisher analysis using existing disjoint noise splits.

No map synthesis or halo painting occurs. Rebin the saved clean derivative
spectra and select a genuinely disjoint subset of the old Battaglia12 ensemble.
This is not a completed comparison to the forthcoming independent-noise NPEs.
"""
import argparse
import json
import os
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import solve_triangular

from export_so_moped_bundle import sha256
from generate_so_fisher_variations import BATTAGLIA12, PARAMETER_NAMES
from so_nine_fisher import (conditional_covariance, fisher_matrix, fisher_modes,
    hard_prior_posterior, moped_weights, require, write_csv)
from so_sbi_compression import write_json, save_npz

LABELS = [r"P_0", r"x_{\rm c}", r"\beta", r"\alpha_{m,P_0}", r"\alpha_{m,x_{\rm c}}",
          r"\alpha_{m,\beta}", r"\alpha_{z,P_0}", r"\alpha_{z,x_{\rm c}}", r"\alpha_{z,\beta}"]


def split_seeds(noise_seed):
    """The existing baseline/deproj0 simulator's actual split seed formula."""
    return (int(noise_seed) + 10101, int(noise_seed) + 10102)


def select_disjoint(records, observation_seed, count):
    require(count >= 4, "At least four noise realizations are required for this pilot")
    observation = next(record for record in records if record["seed"] == observation_seed)
    used = set(split_seeds(observation_seed))
    selected, rejected = [], []
    for record in sorted(records, key=lambda r: r["seed"]):
        if record["seed"] == observation_seed:
            continue
        splits = set(split_seeds(record["seed"]))
        if used & splits:
            rejected.append(record["seed"])
            continue
        selected.append(record)
        used.update(splits)
    require(len(selected) >= count, f"Only {len(selected)} disjoint covariance realizations are available")
    return observation, selected[:count], rejected, len(selected)


def bin_operator(config):
    require(config["ell_min"] == 80 and config["ell_max"] == 7979,
            "Saved spectra cover ell=80..7979")
    ell = np.arange(config["ell_min"], config["ell_max"] + 1)
    groups = ell // int(config["delta_ell"])
    unique = np.unique(groups)
    operator = np.zeros((7980, len(unique)))
    centers, low, high = [], [], []
    for column, group in enumerate(unique):
        selected = ell[groups == group]
        weights = 2 * selected + 1
        operator[selected, column] = weights / weights.sum() * selected * (selected + 1) / (2 * np.pi)
        centers.append(np.average(selected, weights=weights))
        low.append(selected.min())
        high.append(selected.max())
    require(operator.shape == (7980, 40), "Expected the collaborator's 40-bin statistic")
    return operator, np.array(centers), np.array(low), np.array(high)


def load_inputs(root, config):
    manifest_path = root / "fisher_variations_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    np.testing.assert_array_equal(manifest["parameter_names"], PARAMETER_NAMES)
    np.testing.assert_allclose(manifest["fiducial"], BATTAGLIA12, rtol=0, atol=1e-14)
    require(len(manifest["rows"]) == 37 and manifest["rows"][0]["label"] == "fiducial",
            "Expected fiducial followed by 36 controlled variations")
    require([row["row_1based"] for row in manifest["rows"]] == list(range(1, 38)),
            "Manifest rows must be ordered and contiguous")
    operator, centers, bin_min, bin_max = bin_operator(config)
    profiles, sources = [], [manifest_path]
    for row in manifest["rows"]:
        folder = root / "variations" / f"row{row['row_1based']:03d}_{row['label']}"
        found = list(folder.glob("*masked_no_noise_cl*.npy"))
        require(len(found) == 1, f"Missing/ambiguous derivative: {folder}")
        spectrum = np.load(found[0], allow_pickle=False)
        require(spectrum.shape == (7980,) and np.isfinite(spectrum).all(), "Invalid clean spectrum")
        profiles.append(spectrum @ operator)
        sources.append(found[0])
    profiles = np.asarray(profiles)
    derivatives = {}
    for fraction, label in ((.01, "small"), (.02, "large")):
        columns = []
        for parameter in PARAMETER_NAMES:
            rows = [r for r in manifest["rows"] if r["parameter"] == parameter and
                    np.isclose(r["step_fraction"], fraction)]
            require(len(rows) == 2, "Missing central difference pair")
            minus = next(r for r in rows if r["sign"] == -1)
            plus = next(r for r in rows if r["sign"] == 1)
            require(plus["step_absolute"] > 0 and plus["step_absolute"] == minus["step_absolute"],
                    "Central difference steps differ")
            columns.append((profiles[plus["row_1based"]-1] - profiles[minus["row_1based"]-1]) /
                           (2 * plus["step_absolute"]))
        derivatives[label] = np.asarray(columns).T
    derivatives["richardson"] = (4 * derivatives["small"] - derivatives["large"]) / 3
    ensemble_root = root / "battaglia12_baseline_deproj0_ensemble"
    records = []
    for folder in sorted(ensemble_root.glob("noise_seed*")):
        report_path = folder / "prepared/battaglia12_baseline_deproj0_validation.json"
        report = json.loads(report_path.read_text())
        require(report["all_checks_passed"], f"Invalid observation: {folder}")
        contract = report["simulator_contract"]
        require(contract["mask_seed"] == 12345 and contract["deprojection"] == 0 and
                contract["gaussian_beam_fwhm_arcmin"] == 2.0 and contract["mask_fsky"] == .4 and
                contract["mask_apodization_arcmin"] == 60.0 and
                contract["product"] == "masked_baseline_noise_cross_deproj0", "Ensemble contract differs")
        raw = folder / "raw" / Path(report["raw_battaglia12_profile"]).name
        spectrum = np.load(raw, allow_pickle=False)
        require(spectrum.shape == (7980,) and np.isfinite(spectrum).all(), "Invalid noisy spectrum")
        with np.load(folder / "prepared/battaglia12_masked_baseline_noise_cross_deproj0_sbi_observation.npz") as bundle:
            np.testing.assert_allclose(bundle["theta"], BATTAGLIA12, rtol=1e-6)
            np.testing.assert_array_equal(bundle["param_names"], PARAMETER_NAMES)
        records.append(dict(seed=contract["noise_seed"], x=spectrum @ operator))
        sources.extend([report_path, raw])
    require(len({r["seed"] for r in records}) == len(records), "Duplicate noise-root labels")
    return profiles[0], derivatives, records, (centers, bin_min, bin_max), sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fisher-root", type=Path, default=Path("/lustre/work/kristero10/adrian_fisher_baseline_deproj0"))
    parser.add_argument("--config", type=Path, required=True, help="Collaborator config.json supplies authoritative priors/binning")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-noise", type=int, default=16)
    parser.add_argument("--observation-seed", type=int, default=20001)
    parser.add_argument("--seed", type=int, default=20260914)
    args = parser.parse_args()
    started = time.monotonic()
    require(not args.output.exists(), "Use a fresh output directory")
    config = json.loads(args.config.read_text())
    low, high = np.array(config["prior_low"]), np.array(config["prior_high"])
    width = high-low
    np.testing.assert_allclose(config["fiducial"], BATTAGLIA12, rtol=0, atol=1e-14)
    require(np.all((BATTAGLIA12 > low) & (BATTAGLIA12 < high)), "Fiducial outside saved prior")
    fiducial, derivatives, records, bins, sources = load_inputs(args.fisher_root, config)
    observation, selected, rejected, available = select_disjoint(records, args.observation_seed, args.n_noise)
    ensemble = np.asarray([r["x"] for r in selected])
    covariance, shrinkage = conditional_covariance(ensemble)
    jacobian = derivatives["richardson"]
    fisher = fisher_matrix(jacobian*width, covariance)
    eigenvalues, eigenvectors, keep = fisher_modes(fisher)
    weights, singular, moped_error = moped_weights(jacobian*width, covariance)
    args.output.mkdir(parents=True)
    covariance_seeds = [r["seed"] for r in selected]
    save_npz(args.output / "fisher_arrays.npz", param_names=np.array(PARAMETER_NAMES),
        fiducial=BATTAGLIA12, prior_low=low, prior_high=high, fiducial_dell=fiducial,
        observation=observation["x"], ensemble=ensemble, covariance=covariance,
        derivatives=jacobian, derivative_small=derivatives["small"], derivative_large=derivatives["large"],
        fisher_normalized=fisher, fisher_theta=fisher/np.outer(width, width),
        fisher_eigenvalues=eigenvalues, fisher_eigenvectors=eigenvectors, resolved_modes=keep,
        moped_weights=weights, ell_binned=bins[0], bin_ell_min=bins[1], bin_ell_max=bins[2],
        covariance_noise_seeds=covariance_seeds, observation_noise_seed=np.asarray(args.observation_seed))
    print(f"Using {args.n_noise} disjoint covariance realizations; observation={args.observation_seed}", flush=True)
    print(f"Raw sample covariance rank <= {args.n_noise-1}; OAS shrinkage={shrinkage:.6f}; Fisher rank={keep.sum()}/9", flush=True)
    samples, sampling = {}, {}
    rows = []
    for i, (label, residual) in enumerate((("fiducial_forecast", np.zeros(40)),
                                        ("observed_posterior", observation["x"]-fiducial))):
        values, info = hard_prior_posterior(jacobian, covariance, residual, low, high, BATTAGLIA12, seed=args.seed+i)
        _, repeat = hard_prior_posterior(jacobian, covariance, residual, low, high, BATTAGLIA12, seed=args.seed+i+50)
        change = np.abs(np.array(info["weighted_std"])/repeat["weighted_std"]-1)
        shift = np.abs(np.array(info["weighted_mean"])-repeat["weighted_mean"])/info["weighted_std"]
        require(change.max()<.05 and shift.max()<.05, "Importance runs disagree; increase integration budget")
        info.update(repeat_ess=repeat["importance_ess"], repeat_max_relative_std_change=float(change.max()),
                    repeat_max_mean_shift_over_std=float(shift.max()))
        samples[label], sampling[label] = values, info
        np.save(args.output/f"{label}_samples.npy", values)
        quantiles = np.quantile(values, [.025, .16, .5, .84, .975], axis=0)
        for j, name in enumerate(PARAMETER_NAMES):
            rows.append(dict(method=label, parameter=name, fiducial=BATTAGLIA12[j],
                mean=values[:, j].mean(), std=values[:, j].std(ddof=1),
                std_over_prior=values[:, j].std(ddof=1)/width[j],
                q025=quantiles[0, j], q16=quantiles[1, j], median=quantiles[2, j], q84=quantiles[3, j], q975=quantiles[4, j]))
    write_csv(args.output/"parameter_constraints.csv", rows)
    chol = np.linalg.cholesky(covariance)
    reference_derivative = np.linalg.norm(solve_triangular(chol, jacobian, lower=True), axis=0)
    stability = []
    for label in ("small", "large"):
        relative = np.linalg.norm(solve_triangular(chol, derivatives[label]-jacobian, lower=True), axis=0)/reference_derivative
        stability.extend(dict(derivative=label, parameter=p, noise_weighted_relative_change=e) for p, e in zip(PARAMETER_NAMES, relative))
    write_csv(args.output/"derivative_stability.csv", stability)
    # The following Gaussian prior-moment widths are diagnostics of C only.
    # The reported parameter table instead uses the exact hard-prior samples.
    reference = np.sqrt(np.diag(np.linalg.inv(fisher+12*np.eye(9))))
    sensitivity = []
    for n in sorted({max(4, args.n_noise//2), args.n_noise}):
        c, s = conditional_covariance(ensemble[:n])
        f = fisher_matrix(jacobian*width, c)
        ratio = np.sqrt(np.diag(np.linalg.inv(f+12*np.eye(9))))/reference
        sensitivity.append(dict(n_noise=n, shrinkage=s, max_prior_moment_sigma_change=float(np.max(np.abs(ratio-1)))))
    rng = np.random.default_rng(args.seed+200)
    bootstrap = []
    for _ in range(200):
        c, _ = conditional_covariance(ensemble[rng.integers(args.n_noise, size=args.n_noise)])
        f = fisher_matrix(jacobian*width, c)
        bootstrap.append(np.sqrt(np.diag(np.linalg.inv(f+12*np.eye(9))))/reference)
    write_csv(args.output/"covariance_sensitivity.csv", sensitivity)
    np.save(args.output/"bootstrap_prior_moment_sigma_ratios.npy", bootstrap)
    summary = dict(status="completed_small_ensemble_fisher_pilot", preliminary=True, n_noise=args.n_noise,
        map_generation_performed=False, halo_painting_performed=False, npe_training_performed=False,
        pbs_job_id=os.environ.get("PBS_JOBID"), observation_noise_seed=args.observation_seed,
        covariance_noise_seeds=covariance_seeds, available_disjoint_covariance_count=available,
        rejected_overlapping_noise_roots=rejected,
        existing_ensemble_roots=len(records), existing_split_count=2*len(records),
        existing_unique_splits=len({s for r in records for s in split_seeds(r["seed"])}),
        observation_excluded_from_covariance=True, all_used_noise_splits_disjoint=True,
        covariance_raw_rank_bound=args.n_noise-1, oas_shrinkage=shrinkage,
        fisher_rank=int(keep.sum()), fisher_rank_threshold=1e-8, fisher_rank_coordinates="prior-width units",
        moped_fisher_relative_error=moped_error, sampling=sampling,
        bootstrap_prior_moment_sigma_ratio_16_84=np.quantile(bootstrap, [.16,.84], axis=0).tolist(),
        limitations=["16-realization default is a shrinkage-dependent pilot, not covariance convergence.",
            "Saved spectra use the earlier painter; exact agreement with the collaborator's corrected painter is unverified.",
            "Fixed signal and mask; no changing-sky/halo variance and no covariance-derivative Fisher term.",
            "MOPED comparison is the local Fisher identity, not a trained-NPE comparison.",
            "The original 64 adjacent-seed outputs contain reused noise splits and are not 64 independent draws."],
        provenance={str(p):sha256(p) for p in sources+[args.config,Path(__file__),Path(__file__).with_name("so_nine_fisher.py")]})
    print("Rendering Fisher contours and small-ensemble diagnostics", flush=True)
    plot_results(args.output, samples, low, high, covariance, eigenvalues, rows, np.asarray(bootstrap), args.n_noise)
    summary["elapsed_seconds"] = time.monotonic()-started
    write_json(args.output/"analysis_summary.json", summary)
    write_json(args.output/"complete.json", dict(status=summary["status"], preliminary=True,
        artifacts={p.name:sha256(p) for p in args.output.iterdir() if p.is_file()}))
    print(f"Completed: {args.output}", flush=True)


def plot_results(output, samples, low, high, covariance, eigenvalues, rows, bootstrap, n_noise):
    from getdist import MCSamples, plots
    plt.rcParams.update({"font.family":"serif", "mathtext.fontset":"cm", "font.size":10})
    names = [f"p{i}" for i in range(9)]
    ranges = {name:(low[i], high[i]) for i, name in enumerate(names)}
    roots = []
    for label in ("fiducial_forecast", "observed_posterior"):
        array = samples[label]
        selected = np.random.default_rng(82).choice(len(array), 20000, replace=False)
        roots.append(MCSamples(samples=array[selected], names=names, labels=LABELS, ranges=ranges,
            settings={"smooth_scale_1D":.3, "smooth_scale_2D":.4, "fine_bins_2D":128}))
    g = plots.get_subplot_plotter(width_inch=13)
    g.settings.axes_fontsize = 9
    g.settings.lab_fontsize = 11
    g.settings.legend_fontsize = 11
    g.settings.figure_legend_frame = False
    g.triangle_plot(roots, filled=[False, True], markers=BATTAGLIA12,
        contour_colors=["#0072B2", "#D55E00"],
        legend_labels=["Fiducial Fisher + hard prior", "Observed Fisher + hard prior"],
        line_args=[dict(color="#0072B2", ls="--"), dict(color="#D55E00")],
        marker_args=dict(color="black", ls=":", lw=.7))
    g.fig.suptitle(f"PRELIMINARY: Battaglia12, nine parameters, {n_noise} disjoint noise realizations", y=1.01)
    save_plot(g.fig, output, "battaglia12_fisher_9param_pilot")
    fig, axes = plt.subplots(1,3,figsize=(15,4.5))
    correlation = covariance/np.sqrt(np.outer(np.diag(covariance),np.diag(covariance)))
    im = axes[0].imshow(correlation,vmin=-1,vmax=1,cmap="RdBu_r",origin="lower")
    axes[0].set(title="OAS noise correlation",xlabel="Bin",ylabel="Bin")
    fig.colorbar(im,ax=axes[0],shrink=.8)
    axes[1].semilogy(np.arange(1,10),np.maximum(eigenvalues,1e-20),"o-")
    axes[1].axhline(eigenvalues[0]*1e-8,color="gray",ls=":",label="Rank threshold")
    axes[1].set(title="Fisher eigenvalues",xlabel="Mode",ylabel="Eigenvalue in prior-width units")
    axes[1].legend(fontsize=8)
    q = np.quantile(bootstrap,[.16,.5,.84],axis=0)
    axes[2].errorbar(np.arange(9),q[1],yerr=[q[1]-q[0],q[2]-q[1]],fmt="o")
    axes[2].axhline(1,color="gray",ls=":")
    axes[2].set_xticks(np.arange(9))
    axes[2].set_xticklabels([f"${s}$" for s in LABELS], rotation=50, ha="right")
    axes[2].set(title="Covariance bootstrap: prior-moment diagnostic",ylabel="Sigma / original sigma")
    fig.suptitle(f"PRELIMINARY: {n_noise} realizations for 40 bins; regularization required")
    fig.tight_layout()
    save_plot(fig, output, "small_ensemble_covariance_diagnostics")


def save_plot(fig, output, stem):
    for extension in ("png","pdf"):
        fig.savefig(output/f"{stem}.{extension}",dpi=180,bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
