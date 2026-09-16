#!/usr/bin/env python3
"""Test unique-ray noise subtraction against complete, unchanged cluster maps.

This is a fixed-sky sampling experiment, not new halo painting. All foreground
halos remain in each input. The known full-map mean isolates the exact finite-
population correction from the extra bias of estimating a mean from a catalogue.
Only PNG/NPZ/CSV/JSON diagnostics are written; no original products are replaced.
Compatible with the cluster's Python 3.6/SZ environment.
"""
import argparse
import csv
import gc
import json
import platform
from pathlib import Path

import healpy as hp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from audit_lee22_sparse_power import bands, test_finite_population_identity


def read_provenance(path):
    values = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    expected = {"nside": "4096", "profile_mass_definition": "M200c",
                "aperture_radius_definition": "R200c", "ordering": "RING"}
    for key, value in expected.items():
        if values.get(key) != value:
            raise ValueError("{}: {} is not {}".format(path, key, value))
    if float(values["source_redshift"]) != 1.0 or float(values["aperture_r200c_multiplier"]) != 3.0:
        raise ValueError("Expected source z=1 and projected aperture 3R200c")
    if values["catalog_rows_scanned"] != values["catalog_total_rows"]:
        raise ValueError("Input map used a truncated catalogue")
    if "complete resolved" not in values.get("mass_selection", ""):
        raise ValueError("Input map is not the complete resolved mass range")
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--battaglia16-map", type=Path, required=True)
    parser.add_argument("--lee22-map", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mc-seeds", type=int, default=16)
    parser.add_argument("--lmax", type=int, default=8192)
    parser.add_argument("--n-rays", type=int, default=1000000)
    args = parser.parse_args()
    if args.n_rays != 1000000 or args.mc_seeds < 2:
        raise ValueError("Use one million rays and at least two independent seeds")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    test_finite_population_identity()
    edges = np.unique(np.rint(np.geomspace(2, args.lmax+1, 56)).astype(int))
    lower, upper = edges[:-1], edges[1:]-1
    ell = np.array([np.average(np.arange(lo, hi+1), weights=2*np.arange(lo, hi+1)+1)
                    for lo, hi in zip(lower, upper)])
    factor = ell*(ell+1)/(2*np.pi)
    output = {"ell": ell, "ell_min": lower, "ell_max": upper}
    report = {"hostname": platform.node(), "python": platform.python_version(),
              "numpy": np.__version__, "healpy": hp.__version__,
              "n_rays": args.n_rays, "seeds_per_profile": args.mc_seeds,
              "mean_policy": "known complete-map mean; not a catalogue-only estimator",
              "beam": "none", "instrumental_noise": "none", "models": {}}
    common = None
    table = []
    for label, path in [("battaglia16", args.battaglia16_map), ("lee22", args.lee22_map)]:
        provenance = read_provenance(path.with_name(path.stem+"_provenance.txt"))
        identity = tuple(provenance[key] for key in ("catalogue", "halos_selected", "selected_mass_min_msun", "selected_mass_max_msun", "selected_redshift_min", "selected_redshift_max"))
        if common is not None and common != identity:
            raise ValueError("Map halo populations do not match")
        common = identity
        print("Reading complete sky:", path, flush=True)
        sky = hp.read_map(str(path), dtype=np.float64, verbose=False)
        if hp.get_nside(sky) != 4096 or not np.all(np.isfinite(sky)) or np.min(sky)<0:
            raise ValueError("Expected a nonnegative, complete NSIDE4096 map")
        mean = float(np.mean(sky))
        sky -= mean
        variance = float(np.mean(sky*sky))
        print("Computing complete-map reference:", label, flush=True)
        dense_alm = hp.map2alm(sky, lmax=args.lmax, iter=0, pol=False)
        dense_cl = hp.alm2cl(dense_alm)
        del dense_alm
        dense = bands(dense_cl, lower, upper)
        p, n = len(sky), args.n_rays
        a = p*(n-1)/(n*(p-1))
        correction = (p-n)/(p-1)
        old, fixed, shot, centered_approx = [], [], [], []
        for seed in range(args.mc_seeds):
            pixels = np.random.default_rng(20260910+seed).choice(p, size=n, replace=False)
            q = sky[pixels]
            sparse = np.zeros(p, dtype=np.float64)
            sparse[pixels] = (p/n)*q
            alm = hp.map2alm(sparse, lmax=args.lmax, iter=0, pol=False)
            cl = hp.alm2cl(alm)
            noise = 4*np.pi/n*np.mean(q*q)
            old.append(bands(cl-noise, lower, upper))
            fixed.append(bands((cl-correction*noise)/a, lower, upper))
            shot.append(noise)
            # Measure the additional sample-mean issue explicitly for the first
            # four seeds. This remains labelled approximate, not an exact repair.
            if seed < 4:
                q_centered = q-np.mean(q)
                sparse[pixels] = (p/n)*q_centered
                del alm
                alm = hp.map2alm(sparse, lmax=args.lmax, iter=0, pol=False)
                cl_centered = hp.alm2cl(alm)
                noise_centered = 4*np.pi/n*np.mean(q_centered*q_centered)
                sample_factor = n*(p-n)/(p*(n-1))
                centered_approx.append(bands((cl_centered-sample_factor*noise_centered)/a, lower, upper))
            del sparse, alm
            gc.collect()
            print("{}: completed seed {}/{}".format(label, seed+1, args.mc_seeds), flush=True)
        old, fixed = np.asarray(old), np.asarray(fixed)
        std = np.std(fixed, axis=0, ddof=1)
        average = np.mean(fixed, axis=0)
        report["models"][label] = {
            "path": str(path), "provenance": provenance, "mean_dm": mean,
            "variance_dm": variance, "last_band_dense_dl": float(factor[-1]*dense[-1]),
            "last_band_old_mean_dl": float(factor[-1]*np.mean(old[:, -1])),
            "last_band_corrected_mean_dl": float(factor[-1]*average[-1]),
            "last_band_sampling_std_dl": float(factor[-1]*std[-1]),
            "last_band_mean_standard_error_dl": float(factor[-1]*std[-1]/np.sqrt(args.mc_seeds)),
        }
        output[label+"_dense_cl"] = dense_cl
        output[label+"_dense_bands"] = dense
        output[label+"_legacy_samples"] = old
        output[label+"_finite_samples"] = fixed
        output[label+"_sample_mean_approx"] = np.asarray(centered_approx)
        output[label+"_shot_cl"] = np.asarray(shot)
        for j, l in enumerate(ell):
            table.append(dict(profile=label, ell=l, dense_dl=factor[j]*dense[j],
                              legacy_mean_dl=factor[j]*np.mean(old[:, j]),
                              finite_mean_dl=factor[j]*average[j],
                              sampling_std_dl=factor[j]*std[j],
                              mean_standard_error_dl=factor[j]*std[j]/np.sqrt(args.mc_seeds)))
        del sky
        gc.collect()
        # Checkpoint after each model so a wall-time interruption is recoverable.
        np.savez(args.output_dir/"cluster_sampling_samples.npz", **output)
        (args.output_dir/"cluster_sampling_summary.json").write_text(json.dumps(report, indent=2))
    with (args.output_dir/"cluster_sampling_bands.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    fig, axes = plt.subplots(2, 3, figsize=(17, 9), constrained_layout=True)
    for j, label in enumerate(["battaglia16", "lee22"]):
        dense = output[label+"_dense_bands"]
        samples = output[label+"_finite_samples"]
        avg, spread = samples.mean(axis=0), samples.std(axis=0, ddof=1)
        sem = spread/np.sqrt(args.mc_seeds)
        axes[0, j].plot(ell, factor*dense, color="black", label="Complete-map reference")
        axes[0, j].plot(ell, factor*output[label+"_legacy_samples"].mean(axis=0), color="#d74b57", label="Poisson subtraction: mean")
        axes[0, j].errorbar(ell, factor*avg, yerr=factor*sem, color="#287b9b", lw=1, label="Finite population: mean +/- SEM")
        axes[0, j].set_yscale("symlog", linthresh=2000)
        axes[0, j].set_ylabel(r"$D_\ell$ [(pc cm$^{-3}$)$^2$]")
        axes[0, j].set_title(label+": signed estimates")
        axes[0, j].legend(fontsize=8)
        axes[1, j].plot(ell, 100*(avg/dense-1), color="#287b9b")
        axes[1, j].fill_between(ell, 100*((avg-sem)/dense-1), 100*((avg+sem)/dense-1), color="#287b9b", alpha=.2)
        axes[1, j].axhline(0, color="black", lw=.7)
        axes[1, j].set_ylabel("Mean difference from full sky [%]")
        # Autoscale the linear percentage axis: large high-ell residuals are
        # diagnostic and must not be hidden by a fixed display range.
        for ax in axes[:, j]:
            ax.set_xscale("log")
            ax.set_xlim(500, args.lmax)
            ax.grid(alpha=.2)
            ax.set_xlabel(r"Multipole $\ell$")
        axes[0, 2].loglog(ell, factor*dense, label=label)
        axes[1, 2].semilogx(ell, spread/dense, label=label)
    axes[0, 2].set_title("Underlying complete-sky spectra")
    axes[0, 2].set_ylabel(r"$D_\ell$ [(pc cm$^{-3}$)$^2$]")
    axes[1, 2].set_title("Scatter of a single million-ray realization")
    axes[1, 2].set_ylabel("Sampling standard deviation / full-sky signal")
    axes[1, 2].axhline(1, color="black", ls="--", lw=.8)
    for ax in axes[:, 2]:
        ax.legend(fontsize=9)
        ax.set_xlabel(r"Multipole $\ell$")
        ax.grid(alpha=.2)
    fig.suptitle("Cluster sampling test: all foreground halos, z(source)=1, NSIDE4096, 3R200c")
    fig.savefig(args.output_dir/"cluster_sampling_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("Sampling experiment completed", flush=True)


if __name__ == "__main__":
    main()
