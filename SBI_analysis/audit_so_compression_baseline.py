#!/usr/bin/env python3
"""Compare saved old/new metrics on identical rows; no new posterior sampling."""
from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd

from so_sbi_compression import PARAM_NAMES, write_json


def matched_metrics(old, new, n_train):
    old = old.loc[old.n_train == n_train].copy().rename(columns={
        "theta_true": "truth", "posterior_mean": "mean", "posterior_std": "std",
        "error_over_prior_range": "normalized_error_prior",
    })
    if "case" in old and old["case"].nunique() != 1:
        raise ValueError("Select one noise product before comparison")
    old["method"] = "old_bins40"
    frames = [old] + [new.loc[new.method == m].copy() for m in ("bins40", "pca", "moped")]
    complete_ids = []
    for frame in frames:
        if frame.empty or frame.duplicated(["test_index", "param"]).any():
            raise ValueError("Missing method or duplicate parameter/observation pairs")
        if set(frame.param) != set(PARAM_NAMES):
            raise ValueError("All nine parameters are required")
        counts = frame.groupby("test_index").size()
        complete_ids.append(set(counts[counts == 9].index))
    common = sorted(set.intersection(*complete_ids))
    if len(common) < 3:
        raise ValueError("Fewer than three common complete observations")
    frames = [f.loc[f.test_index.isin(common)].sort_values(["test_index", "param"])
              for f in frames]
    truth = frames[0].truth.to_numpy()
    for f in frames:
        np.testing.assert_allclose(f.truth, truth, rtol=0, atol=1e-7)
        values = f[["truth", "mean", "std", "pull", "normalized_error_prior"]].to_numpy()
        if not np.isfinite(values).all() or (f["std"] <= 0).any():
            raise ValueError("Invalid saved metrics")
        error = f["mean"] - f.truth
        np.testing.assert_allclose(f.pull, error / f["std"], rtol=1e-6, atol=1e-7)
        widths = frames[0].prior_width.to_numpy()
        np.testing.assert_allclose(f.normalized_error_prior, error / widths,
                                   rtol=1e-6, atol=1e-7)
    joined = pd.concat(frames, ignore_index=True)
    rows = []
    for (method, param), sub in joined.groupby(["method", "param"], sort=False):
        r = np.corrcoef(sub.truth.to_numpy(), sub["mean"].to_numpy())[0, 1]
        rows.append(dict(method=method, param=param, n_test=len(sub), pearson_r=r,
                         rmse_prior=np.sqrt(np.mean(sub.normalized_error_prior**2)),
                         rmse_std=np.sqrt(np.mean(sub.pull**2))))
    return pd.DataFrame(rows), joined, common


def main():
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-metrics", type=Path, default=base / "convergence_tests/last100_param_metrics.csv")
    parser.add_argument("--summary", type=Path, default=base / "outputs/compression_bestval_20260909/summary")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--n-train", type=int, default=523788)
    args = parser.parse_args()
    summary, profiles, indices = matched_metrics(pd.read_csv(args.old_metrics),
        pd.read_csv(args.summary / "per_profile_metrics.csv"), args.n_train)
    output = args.output or args.summary.parent / "baseline_audit"
    output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output / "matched_parameter_metrics.csv", index=False)
    profiles.to_csv(output / "matched_profile_metrics.csv", index=False)
    np.save(output / "matched_test_indices.npy", indices)
    write_json(output / "audit.json", dict(n_common=len(indices), old_metrics=str(args.old_metrics),
        new_summary=str(args.summary), old_n_train=args.n_train, truth_equal=True,
        normalization_checks_passed=True, preliminary=True,
        warning="Successful-profile intersection; omitted difficult profiles can bias results optimistically."))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 9})
    colors = {"old_bins40": "black", "bins40": "#cc6677", "pca": "#4477aa", "moped": "#228833"}
    labels = [r"$P_0$", r"$x_c$", r"$\beta$", r"$\alpha_{m,P_0}$", r"$\alpha_{m,x_c}$",
              r"$\alpha_{m,\beta}$", r"$\alpha_{z,P_0}$", r"$\alpha_{z,x_c}$", r"$\alpha_{z,\beta}$"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, key, ylabel in zip(axes, ("pearson_r", "rmse_prior", "rmse_std"),
                               ("Pearson r", "RMSE / prior range", "RMS(error / posterior std)")):
        for method, color in colors.items():
            sub = summary[summary.method == method].set_index("param").loc[list(PARAM_NAMES)]
            ax.plot(np.arange(9), sub[key].to_numpy(), "o-", color=color, label=method)
        ax.set_xticks(np.arange(9)); ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel(ylabel); ax.grid(alpha=.25)
    axes[0].legend(fontsize=8)
    title = f"PRELIMINARY: identical {len(indices)} successful observations; not a controlled training comparison"
    fig.suptitle(title, color="firebrick", fontsize=10)
    fig.tight_layout()
    fig.savefig(output / "old_vs_compression_matched_rows.png", dpi=200)
    fig.savefig(output / "old_vs_compression_matched_rows.pdf")
    plt.close(fig)
    print(summary[summary.param.isin(["P0", "beta"])].to_string(index=False))
    print("Saved:", output)


if __name__ == "__main__":
    main()
