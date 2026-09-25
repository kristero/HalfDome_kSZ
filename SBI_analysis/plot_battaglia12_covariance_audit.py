#!/usr/bin/env python3
"""Verify archived inputs and plot the available FIXED-NOISE Battaglia12 SBI.

This does not label the independent-noise Fisher pilot as a matched SBI
comparison. The separate covariance audit records why that overlay is invalid.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from export_so_moped_bundle import sha256
from generate_so_fisher_variations import BATTAGLIA12, PARAMETER_NAMES
from run_so_nine_independent_comparison import LABELS
from so_fisher_compression import project_likelihood
from so_nine_fisher import conditional_covariance, read_npz, require, write_csv
from so_sbi_compression import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    base = Path(__file__).resolve().parent / "outputs"
    parser.add_argument("--sbi-root", type=Path, default=base / "unbinned_moped_524k_20260914")
    parser.add_argument("--fisher-root", type=Path, default=base / "so9_fisher_noise16_20260913/results")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Use a fresh output directory")
    manifest = json.loads((args.sbi_root / "export_manifest.json").read_text())
    provenance = {}

    def checked(relative):
        path = args.sbi_root / relative
        digest = sha256(path)
        require(digest == manifest["artifacts"][relative], f"Changed SBI artifact: {relative}")
        provenance[str(path)] = digest
        return path

    contract = read_npz(checked("inference_contract.npz"))
    low, high = contract["low"], contract["high"]
    np.testing.assert_array_equal(contract["param_names"], PARAMETER_NAMES)
    methods = ("bins40", "moped40", "unbinned_moped")
    labels = ("40 bins", "40 bins → MOPED", "Unbinned → MOPED")
    colors = ("#0072B2", "#D55E00", "#7B3294")
    samples, raw_hashes = {}, []
    for method in methods:
        folder = f"{method}/fiducial"
        done = json.loads(checked(f"{folder}/complete.json").read_text())
        path = checked(f"{folder}/posterior_samples.npy")
        require(sha256(path) == done["samples_sha256"], "Changed posterior samples")
        values = np.load(path)
        require(values.shape == (done["n_samples"], 9) and np.isfinite(values).all(), "Invalid posterior")
        require(np.all((values >= low) & (values <= high)), "Posterior outside saved prior")
        observation = read_npz(checked(f"{folder}/observation.npz"))
        np.testing.assert_allclose(observation["truth"], BATTAGLIA12, rtol=1e-6)
        require(str(observation["raw_sha256"]) == done["raw_sha256"], "Observation identity differs")
        raw_hashes.append(done["raw_sha256"])
        samples[method] = values
    require(len(set(raw_hashes)) == 1, "SBI methods use different observations")

    fisher_path = args.fisher_root / "fisher_arrays.npz"
    marker = json.loads((args.fisher_root / "complete.json").read_text())
    require(sha256(fisher_path) == marker["artifacts"][fisher_path.name], "Changed Fisher inputs")
    provenance[str(fisher_path)] = sha256(fisher_path)
    cal = read_npz(fisher_path)
    covariance, shrinkage = conditional_covariance(cal["ensemble"])
    np.testing.assert_allclose(covariance, cal["covariance"], rtol=1e-12)
    seeds = cal["covariance_noise_seeds"]
    split_seeds = np.concatenate([seeds + 10101, seeds + 10102,
                                 np.asarray(cal["observation_noise_seed"]) + np.array([10101, 10102])])
    require(len(np.unique(split_seeds)) == len(split_seeds), "Reused Fisher noise splits")
    jacobian = cal["derivatives"] * (cal["prior_high"] - cal["prior_low"])
    residual = cal["observation"] - cal["fiducial_dell"]
    full = project_likelihood(jacobian, covariance, residual, np.eye(40))
    compressed = project_likelihood(jacobian, covariance, residual, cal["moped_weights"])
    score_error = np.linalg.norm(full["score"] - compressed["score"]) / np.linalg.norm(full["score"])
    require(score_error < 1e-8 and compressed["relative_fisher_change"] < 1e-8, "Failed MOPED identity")
    args.output.mkdir(parents=True)
    rows = []
    for method, values in samples.items():
        q = np.quantile(values, [.025, .16, .5, .84, .975], axis=0)
        for j, parameter in enumerate(PARAMETER_NAMES):
            rows.append(dict(method=method, parameter=parameter, fiducial=BATTAGLIA12[j],
                mean=values[:, j].mean(), std=values[:, j].std(ddof=1),
                q025=q[0, j], q16=q[1, j], median=q[2, j], q84=q[3, j], q975=q[4, j]))
    write_csv(args.output / "battaglia12_intervals.csv", rows)
    make_plots(args.output, samples, methods, labels, colors, low, high)
    write_json(args.output / "audit.json", dict(
        matched_fisher_sbi=False, sbi_same_observation=True, raw_observation_sha256=raw_hashes[0],
        sbi_noise_model="reused fixed noise maps", fisher_noise_model="independent split pairs at fixed sky",
        fisher_noise_count=len(seeds), fisher_oas_shrinkage=shrinkage,
        fisher_moped_relative_error=float(compressed["relative_fisher_change"]),
        fisher_moped_observed_score_relative_error=float(score_error),
        fisher_first_bin=[int(cal["bin_ell_min"][0]), int(cal["bin_ell_max"][0])],
        sbi_first_bin=[80, 279],
        same_prior_bounds=bool(np.array_equal(low, cal["prior_low"]) and np.array_equal(high, cal["prior_high"])),
        pca_posterior_available_locally=False, new_training_performed=False,
        limitations=["Fixed-noise SBI widths are not independent-noise forecast errors.",
                     "Fisher pilot covariance uses only 16 draws and strong shrinkage.",
                     "PCA estimator is on the unavailable cluster; no PCA posterior was fabricated.",
                     "Fisher and SBI differ in noise, observation, bin edges and prior bounds."],
        input_sha256=provenance, source_sha256={Path(__file__).name:sha256(Path(__file__))}))
    write_json(args.output / "complete.json", dict(complete=True, scope="archived SBI diagnostic and covariance audit",
        artifacts={p.name:sha256(p) for p in args.output.iterdir() if p.is_file()}))
    print(args.output)


def make_plots(output, samples, methods, labels, colors, low, high):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from getdist import MCSamples, plots
    plt.rcParams.update({"font.family":"serif", "mathtext.fontset":"cm", "font.size":16})
    names = [f"p{i}" for i in range(9)]
    ranges = {name:(low[i], high[i]) for i, name in enumerate(names)}
    roots = [MCSamples(samples=samples[m], names=names, labels=LABELS, ranges=ranges,
                      settings={"smooth_scale_1D":.3, "smooth_scale_2D":.4, "fine_bins_2D":128})
             for m in methods]
    plot = plots.get_subplot_plotter(width_inch=19)
    plot.settings.scaling = False
    plot.settings.axes_fontsize = 13
    plot.settings.lab_fontsize = 17
    plot.settings.legend_fontsize = 17
    plot.settings.figure_legend_frame = False
    plot.triangle_plot(roots, filled=True, contour_colors=colors, legend_labels=labels,
                       markers=BATTAGLIA12, marker_args={"color":"black", "ls":":", "lw":1})
    plot.fig.suptitle("Battaglia12 · fixed-noise SBI", fontsize=21, y=1.01)
    for ext in ("png", "pdf"):
        plot.fig.savefig(output / f"battaglia12_sbi_corner.{ext}", dpi=180, bbox_inches="tight")
    plt.close(plot.fig)
    fig, axes = plt.subplots(3, 3, figsize=(15, 11))
    for j, ax in enumerate(axes.flat):
        for i, (method, color) in enumerate(zip(methods, colors)):
            q = np.quantile(samples[method][:, j], [.025, .16, .5, .84, .975])
            ax.plot(q[[0, 4]], [i, i], color=color, lw=2)
            ax.plot(q[[1, 3]], [i, i], color=color, lw=6)
            ax.plot(q[2], i, "o", color=color, ms=7)
        ax.axvline(BATTAGLIA12[j], color="black", ls=":", lw=1.5)
        ax.set(yticks=range(3), yticklabels=labels if j % 3 == 0 else [""]*3,
               xlabel=f"${LABELS[j]}$", ylim=(-.6, 2.6))
        ax.tick_params(labelsize=13)
        ax.grid(axis="x", alpha=.2)
    fig.suptitle("Battaglia12 · fixed-noise SBI", fontsize=21)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(output / f"battaglia12_sbi_intervals.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
