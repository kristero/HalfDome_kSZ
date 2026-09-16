#!/usr/bin/env python3
"""Summarize numerical radius-table validation, retaining failures in the report."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = {"production_changed": False, "density_normalization_changed": False,
              "relative_error_target": 0.01, "relative_error_dm_threshold_pc_cm3": 1e-5,
              "profile_validation": {}}
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    for level in [1, 2]:
        path = args.output_dir/"radius_scaled_validation_r{}.csv".format(level)
        if not path.exists():
            continue
        table = np.genfromtxt(str(path), delimiter=",", names=True, dtype=None, encoding="utf8")
        for i, name in enumerate(["battaglia16", "lee2022"]):
            points = table[table["profile"] == name]
            meaningful = points["direct_dm"] > 1e-5
            err = np.abs(points["new_relative_error"][meaningful])
            old = np.abs(points["old_relative_error"][meaningful])
            record = {"n_test_points": len(points), "maximum_old_relative_error": float(old.max()),
                      "maximum_new_relative_error": float(err.max()),
                      "percentile99_new_relative_error": float(np.percentile(err, 99)),
                      "column_accuracy_target_passed_on_test_points": bool(np.all(err < 0.01)),
                      "maximum_absolute_error_pc_cm3": float(np.max(np.abs(points["radius_cache_dm"]-points["direct_dm"])))}
            report["profile_validation"][name+"_refinement"+str(level)] = record
            ax = axes[i]
            if level == 1:
                ax.scatter(points["redshift"], 100*points["old_relative_error"], s=7, alpha=.2, color="0.4", label="Legacy angular table")
            ax.scatter(points["redshift"], 100*points["new_relative_error"], s=8, alpha=.4,
                       color="#e89630" if level == 1 else "#2088a0", label="Radius table: refinement "+str(level))
            ax.set_xscale("log")
            ax.set_yscale("symlog", linthresh=1)
            ax.set_title(name+": interpolation error only")
            ax.set_xlabel("Halo redshift")
            ax.set_ylabel("100 (interpolated / direct - 1) [%]")
            ax.grid(alpha=.2)
    for ax in axes:
        ax.axhspan(-1, 1, color="green", alpha=.08, label="Proposed +/-1% target")
        ax.axhline(0, color="black", lw=.5)
        ax.legend(fontsize=8)
    fig.suptitle("Radius-scaled cache prototype: actual nearby halos and held-out mass/z/radius points")
    fig.savefig(str(args.output_dir/"radius_scaled_cache_validation.png"), dpi=180, bbox_inches="tight")
    plt.close(fig)
    budget = np.genfromtxt(str(args.output_dir/"ionized_gas_budget_and_los_checks.csv"), delimiter=",", names=True)
    example = budget[(budget["log10_mass_msun"]==14)&(budget["redshift"]==0)][0]
    report["normalization_warning"] = {
        "lee22_gas_equivalent_over_fbM_at_1e14_z0": float(example["lee_gas_equivalent_over_fbM_r0_to_1"]),
        "status": "Unchanged fitted normalization; authors reference evaluation still needed",
        "maximum_los_split_relative_difference": float(np.max(np.abs(budget["lee_los_split_over_default"]-1)))}
    (args.output_dir/"cluster_radius_validation_summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
