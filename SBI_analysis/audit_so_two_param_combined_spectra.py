#!/usr/bin/env python3
"""Read-only raw-file audit before training on the supplied 32k SO dataset.

Seed tags and residual similarity are evidence, not an original RNG-log audit.
This tool never infers theta values from an unrelated Sobol design.
"""
import argparse
import collections
import csv
import json
import re
from pathlib import Path

import numpy as np


def inspect_layout(path):
    labels = {}
    multipoles = None
    for line in path.read_text().splitlines():
        parts = line.split()
        if not parts or parts[0].startswith("#"):
            continue
        if parts[0] == "n_ell":
            multipoles = int(parts[1])
        elif parts[0].isdigit():
            labels[int(parts[0])] = parts[1]
    if labels.get(1) != "masked_no_noise" or labels.get(2) != "masked_baseline_noise_cross_deproj0":
        raise ValueError("Unknown product-row layout; refusing to guess")
    if multipoles != 7980 or len(labels) != 6:
        raise ValueError("Expected six C_ell products with ell=0..7979")
    return labels


def discover(root):
    files = sorted(root.glob("row*/*combined_cl*.npy"))
    records = []
    pattern = re.compile(r"sobol_battaglia_sobol_p0_beta_32768_(\d+)_row(\d+)_")
    for path in files:
        match = pattern.search(path.name)
        folder = re.fullmatch(r"row(\d+)", path.parent.name)
        seed = re.search(r"_seed(\d+)_", path.name)
        if not (match and folder and seed):
            raise ValueError(f"Unrecognized row identity or seed tag: {path}")
        split, local = map(int, match.groups())
        row = int(folder.group(1))
        if not 1 <= local <= 128 or (split-1)*128+local != row:
            raise ValueError(f"Folder/split/local row mapping disagrees: {path}")
        records.append(dict(row=row, split=split, local=local, seed_tag=int(seed.group(1)),
                            path=str(path)))
    ids = [r["row"] for r in records]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate spectrum outputs for a row")
    return sorted(records, key=lambda r: r["row"])


def binned_dell(cl):
    ell = np.arange(80, 7980)
    dell = np.asarray(cl)[..., ell] * (ell*(ell+1)/(2*np.pi))
    groups = ell//200
    values, centers = [], []
    for group in np.unique(groups):
        mask = groups == group
        weight = 2*ell[mask]+1
        values.append(np.average(dell[..., mask], weights=weight, axis=-1))
        centers.append(np.average(ell[mask], weights=weight))
    return np.asarray(centers), np.stack(values, axis=-1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path,
        default=Path("/lustre/work/kristero10/two_param_P0_beta_32k/y100"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-rows", type=int, default=64)
    args = parser.parse_args()
    layout = inspect_layout(args.data / "combined_cl_layout.txt")
    records = discover(args.data)
    if not records or args.sample_rows < 3:
        raise ValueError("Need spectra and at least three diagnostic rows")
    indices = np.unique(np.linspace(0, len(records)-1,
                        min(args.sample_rows, len(records)), dtype=int))
    clean, noisy = [], []
    for i in indices:
        path = Path(records[i]["path"])
        spectra = np.load(path, mmap_mode="r", allow_pickle=False)
        if spectra.shape != (6, 7980) or not np.isfinite(spectra).all():
            raise ValueError(f"Invalid raw spectrum: {path}")
        ell, values = binned_dell(spectra[[1, 2]])
        clean.append(values[0])
        noisy.append(values[1])
    clean, noisy = np.asarray(clean), np.asarray(noisy)
    residual = noisy-clean
    high = residual[:, ell >= 6000]
    norm = np.linalg.norm(high, axis=1, keepdims=True)
    if np.any(norm == 0):
        raise ValueError("Zero high-ell noise residual")
    cosines = (high/norm) @ (high/norm).T
    cosine_values = cosines[np.triu_indices(len(indices), 1)]
    seeds = dict(collections.Counter(r["seed_tag"] for r in records))
    expected = set(range(1, 32769))
    observed = {r["row"] for r in records}
    report = dict(data=str(args.data), n_files=len(records), expected_rows=32768,
        missing_rows=sorted(expected-observed), unexpected_rows=sorted(observed-expected),
        folder_split_local_mapping_valid=True, split_rows=128,
        seed_tag_counts=seeds, sampled_rows=[records[i]["row"] for i in indices],
        high_ell_residual_cosine_quantiles=np.quantile(cosine_values, [0, .5, 1]).tolist(),
        product_layout=layout, statistic="signed mode-count weighted D_ell, Delta ell=200",
        theta_values_verified=False, original_noise_seed_logs_verified=False,
        warning="All labels share a seed and residuals may share noise. "
                "Do not assume independent row noise without generation provenance.",
        required_before_training="Exact generation P0/beta CSV or theta metadata, and chosen noise policy")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "audit.json").write_text(json.dumps(report, indent=2)+"\n")
    with (args.output / "row_index.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    np.savez_compressed(args.output / "residual_diagnostics.npz",
        ell=ell, clean=clean, noisy=noisy, residual=residual,
        rows=np.asarray(report["sampled_rows"]), cosine_matrix=cosines)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for x in residual:
        axes[0].plot(ell, x/1e-14, alpha=.2, lw=.7, color="#0072B2")
    axes[0].plot(ell, residual.mean(0)/1e-14, color="black", lw=1, label="Mean")
    axes[0].set(xlabel=r"$\ell$", ylabel=r"$(D_\ell^{noisy}-D_\ell^{clean})/10^{-14}$")
    axes[0].legend()
    axes[1].hist(cosine_values, bins=30, color="#D55E00")
    axes[1].set(xlabel="High-ell residual cosine similarity", ylabel="Profile pairs")
    fig.suptitle("32k raw-data noise audit (not posterior constraints)")
    fig.tight_layout()
    fig.savefig(args.output / "shared_noise_diagnostic.png", dpi=200)
    plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
