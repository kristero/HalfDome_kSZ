#!/usr/bin/env python3
"""Verify completed artifacts and summarize spectra and SBI rejection diagnostics.

This reads only small saved results. It does not rerun the network or modify
the completed campaign, its samples, or its original figures.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MODELS = ("HalfDome", "L1_m9", "fgas-8sigma", "Mstar-1sigma")
LABELS = ("HalfDome control", "FLAMINGO fiducial", "FLAMINGO fgas-8sigma",
          "FLAMINGO Mstar-1sigma")
COLORS = ("#222222", "#0072B2", "#D55E00", "#009E73")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.results.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    campaign = json.loads((root / "campaign_complete.json").read_text())
    manifest_path = root / "comparison/complete.json"
    manifest = json.loads(manifest_path.read_text())
    assert digest(manifest_path) == campaign["completion_sha256"]
    for name, expected in manifest["files"].items():
        path = (root / name).resolve()
        path.relative_to(root)
        assert digest(path) == expected, "Changed result: " + name
    report = json.loads((root / "comparison/comparison_summary.json").read_text())
    control = report["reference_comparison"]
    assert control["relative_l2_difference"] == 0
    assert control["max_absolute_context_difference"] == 0
    low = np.asarray(report["original_prior_low"])
    high = np.asarray(report["original_prior_high"])
    posterior = np.load(root / "results/HalfDome/posterior_samples.npy")
    assert posterior.shape == (10000, 9) and np.isfinite(posterior).all()
    assert ((posterior >= low) & (posterior <= high)).all()
    for name in MODELS:
        observation = np.load(root / "results" / name / "observation.npz")
        assert observation["clean"].shape == (40,)
        assert observation["context"].shape == (9,)
        assert all(np.isfinite(observation[key]).all()
                   for key in ("clean", "noisy", "context"))
        for key in ("mask_pixel_sha256", "noise1_pixel_sha256", "noise2_pixel_sha256"):
            assert report["models"][name]["map"][key] == report["cache"][key]

    with (root / "comparison/spectra_binned.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    spectra = {name: [r for r in rows if r["model"] == name] for name in MODELS}
    ell = np.array([float(r["ell"]) for r in spectra["HalfDome"]])
    comparisons = []
    for target in (1000, 3000, 5000, 7000):
        index = int(np.argmin(abs(ell - target)))
        hd = float(spectra["HalfDome"][index]["clean_masked_dl"])
        fiducial = float(spectra["L1_m9"][index]["clean_masked_dl"])
        for name in MODELS:
            value = float(spectra[name][index]["clean_masked_dl"])
            comparisons.append(dict(model=name, requested_ell=target,
                actual_bin_ell=float(ell[index]), clean_masked_dl=value,
                percent_from_halfdome=100*(value/hd-1),
                percent_from_flamingo_fiducial=100*(value/fiducial-1)))
    with (output / "selected_scale_comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)

    # Show unconstrained flow proposals, explicitly distinct from posteriors.
    # Each parameter is normalized by its original prior width; [0,1] is allowed.
    fig, ax = plt.subplots(figsize=(13, 6.5))
    x = np.arange(9)
    ax.axhspan(0, 1, color="grey", alpha=.15, label="Original prior interval")
    acceptance = {}
    for offset, name, label, color in zip(np.linspace(-.24, .24, 4), MODELS, LABELS, COLORS):
        info = report["models"][name]
        probe = info["sampling_preflight"]
        parameters = probe["per_parameter"]
        median = np.array([p["prior_fraction_median"] for p in parameters])
        q05 = np.array([p["prior_fraction_q05"] for p in parameters])
        q95 = np.array([p["prior_fraction_q95"] for p in parameters])
        ax.errorbar(x+offset, median, yerr=np.array([median-q05, q95-median]),
                    fmt="o", ms=4, capsize=3, color=color, label=label)
        acceptance[name] = dict(accepted=probe["accepted_count"], total=probe["raw_count"],
                               fraction=probe["acceptance"], posterior_status=info["status"])
        if "zero_count_95pct_upper_bound" in probe:
            acceptance[name]["binomial_95pct_upper_bound"] = probe["zero_count_95pct_upper_bound"]
    ax.set_xticks(x)
    ax.set_xticklabels([r"$P_0$", r"$x_c$", r"$\beta$", r"$\alpha_{m,P_0}$",
        r"$\alpha_{m,x_c}$", r"$\alpha_{m,\beta}$", r"$\alpha_{z,P_0}$",
        r"$\alpha_{z,x_c}$", r"$\alpha_{z,\beta}$"], fontsize=12)
    ax.set_ylabel(r"Proposal parameter as fraction of prior: $(\theta-L)/(H-L)$")
    ax.set_title("Unconstrained network proposals: rejection diagnostic, not posterior intervals\n"
                 "Medians and 5th-95th percentiles of 20,000 draws per simulation")
    ax.grid(axis="y", alpha=.2)
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.12), ncol=3, frameon=False)
    fig.tight_layout(rect=(0, .05, 1, 1))
    fig.savefig(output / "inference_support.png", dpi=180)
    plt.close(fig)

    stages = {}
    for path in sorted((root / "logs").glob("*.status.json")):
        record = json.loads(path.read_text())
        assert record["exit_code"] == 0, "Nonzero science stage: " + path.name
        stages[path.name[:-len(".status.json")]] = {
            "elapsed_seconds": record["elapsed_seconds"],
            "peak_rss_GiB": record["peak_rss_kbytes"]/1024**2}
    summary = dict(status="verified", job_id=campaign["job_id"],
                   completed_artifacts_verified=len(manifest["files"]),
                   exact_binned_control_match=True, exact_reprojected_control_match=True,
                   stages=stages, acceptance=acceptance, selected_scales=comparisons)
    (output / "verification_and_metrics.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps(dict(status=summary["status"], job_id=summary["job_id"],
                         artifacts=summary["completed_artifacts_verified"],
                         acceptance=acceptance), indent=2))


if __name__ == "__main__":
    main()
