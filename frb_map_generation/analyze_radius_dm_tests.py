#!/usr/bin/env python3
"""Full-map and finite-population diagnostics; Python 3.6 compatible.

This is a controlled numerical test, not an estimator for a 71-source survey.
The sampling test uses the known full-sky mean, and retains negative estimates.
"""
import argparse
import gc
import json
from pathlib import Path

import healpy as hp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from audit_lee22_sparse_power import bands, test_finite_population_identity


def provenance(path):
    result = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            result[key] = value
    return result


def analyze_map(args):
    meta = provenance(args.map.with_name(args.map.stem + "_provenance.txt"))
    expected = {"nside": "4096", "source_redshift": "1.0",
                "aperture_r200c_multiplier": "3.0", "profile_mass_definition": "M200c",
                "dm_cache_coordinate": "radius", "ordering": "RING"}
    for key, value in expected.items():
        if meta.get(key) != value:
            raise ValueError("Wrong map metadata: {}={}".format(key, meta.get(key)))
    if meta["catalog_rows_scanned"] != meta["catalog_total_rows"]:
        raise ValueError("Truncated catalogue is not a full-halo test")
    test_finite_population_identity()
    sky = hp.read_map(str(args.map), dtype=np.float64, verbose=False)
    if hp.get_nside(sky) != 4096 or not np.all(np.isfinite(sky)) or np.min(sky) < 0:
        raise ValueError("Invalid complete DM map")
    mean = float(np.mean(sky))
    sky -= mean
    edges = np.unique(np.rint(np.geomspace(2, args.lmax + 1, 56)).astype(int))
    lower, upper = edges[:-1], edges[1:] - 1
    ell = np.array([np.average(np.arange(lo, hi+1), weights=2*np.arange(lo, hi+1)+1)
                    for lo, hi in zip(lower, upper)])
    print("Computing dense reference:", args.label, flush=True)
    alm = hp.map2alm(sky, lmax=args.lmax, iter=0, pol=False)
    cl = hp.alm2cl(alm)
    del alm
    dense = bands(cl, lower, upper)
    p, n = len(sky), 1000000
    amplitude = p*(n-1)/(n*(p-1))
    finite_factor = (p-n)/(p-1)
    samples, legacy = [], []
    for seed in range(args.samples):
        pixels = np.random.default_rng(20260913+seed).choice(p, size=n, replace=False)
        values = sky[pixels]
        sparse = np.zeros(p, dtype=np.float64)
        sparse[pixels] = p/n*values
        alm = hp.map2alm(sparse, lmax=args.lmax, iter=0, pol=False)
        raw = hp.alm2cl(alm)
        poisson = 4*np.pi/n*np.mean(values*values)
        samples.append(bands((raw-finite_factor*poisson)/amplitude, lower, upper))
        legacy.append(bands(raw-poisson, lower, upper))
        del alm, sparse
        gc.collect()
        print("{}: million-ray seed {}/{}".format(args.label, seed+1, args.samples), flush=True)
    samples = np.asarray(samples)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    np.savez(output/(args.label + ".npz"), ell=ell, ell_min=lower, ell_max=upper,
             dense_cl=cl, dense_bands=dense, finite_samples=samples,
             legacy_samples=np.asarray(legacy), mean_dm=mean)
    report = {"label": args.label, "map": str(args.map), "provenance": meta,
              "mean_dm": mean, "n_rays": n, "seeds": args.samples,
              "mean_policy": "known full-map mean; numerical test only",
              "noise": "no instrumental noise", "beam": "none",
              "last_band_dense_cl": float(dense[-1]),
              "last_band_sampling_mean_cl": float(samples[:, -1].mean()),
              "last_band_sampling_std_cl": float(samples[:, -1].std(ddof=1)),
              "last_band_sem_cl": float(samples[:, -1].std(ddof=1)/np.sqrt(args.samples))}
    (output/(args.label + ".json")).write_text(json.dumps(report, indent=2))
    factor = ell*(ell+1)/(2*np.pi)
    avg, sd = samples.mean(axis=0), samples.std(axis=0, ddof=1)
    sem = sd/np.sqrt(args.samples)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    axes[0, 0].plot(ell, factor*dense, color="black", label="Complete-map reference")
    axes[0, 0].errorbar(ell, factor*avg, yerr=factor*sem, label="1M rays: mean +/- SEM", color="#287b9b")
    axes[0, 0].set_yscale("symlog", linthresh=max(1, float(np.max(factor*dense))/100))
    axes[0, 0].set_ylabel(r"$D_\ell$ [(pc cm$^{-3}$)$^2$]")
    axes[0, 0].legend(fontsize=9)
    axes[1, 0].plot(ell, 100*(avg/dense-1))
    axes[1, 0].fill_between(ell, 100*((avg-sem)/dense-1), 100*((avg+sem)/dense-1), alpha=.2)
    axes[1, 0].set_ylabel("Difference from full map [%]")
    axes[1, 0].axhline(0, color="black", lw=.7)
    axes[0, 1].loglog(ell, factor*dense, color="black")
    axes[0, 1].set_title("Complete sky: no ray shot noise")
    axes[0, 1].set_ylabel(r"$D_\ell$ [(pc cm$^{-3}$)$^2$]")
    axes[1, 1].plot(ell, sd/dense)
    axes[1, 1].axhline(1, color="black", ls="--")
    axes[1, 1].set_ylabel("One-catalogue scatter / full-map signal")
    for ax in axes.flat:
        ax.set_xscale("log")
        ax.set_xlabel(r"Multipole $\ell$")
        ax.grid(alpha=.2)
    fig.suptitle(args.label + ": all foreground halos, source z=1, NSIDE4096")
    fig.savefig(output/(args.label + "_sampling.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_profiles(output):
    table = np.genfromtxt(output/"preferred_profile_grid.csv", delimiter=",", names=True, dtype=None, encoding="utf-8")
    models = [("battaglia16", "Battaglia16", "#253b47"),
              ("lee22_legacy", "Lee22 no concentration (legacy)", "#d95367"),
              ("lee22_preferred_duffy", "Lee22 preferred + Duffy08", "#3485a4")]
    for z in (0.0, 1.0):
        fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
        for col, mass in enumerate((13.0, 14.0, 15.0)):
            for key, label, color in models:
                rows = table[(table["profile"] == key) & (table["redshift"] == z) & (table["log10_mass_msun"] == mass)]
                x = rows["radius_r200c"]
                axes[0, col].loglog(x, rows["ne_cm3"], color=color, label=label)
                keep = x < 3
                axes[1, col].loglog(x[keep], rows["spherical3_column_pc_cm3"][keep], color=color)
            axes[0, col].set_title(r"$\log_{10}(M_{200c}/M_\odot)=$" + str(mass))
            for ax in axes[:, col]:
                ax.axvspan(.001, .04, color="grey", alpha=.10)
                ax.axvspan(1.34, 5, color="grey", alpha=.10)
                ax.set_xlim(.001, 3)
                ax.set_xlabel(r"$R/R_{200c}$")
                ax.grid(alpha=.2)
        axes[0, 0].set_ylabel(r"Electron density [cm$^{-3}$]")
        axes[1, 0].set_ylabel(r"DM, spherical $3R_{200c}$ [pc cm$^{-3}$]")
        axes[0, 0].legend(fontsize=8)
        fig.suptitle("Density and finite projection at halo z={}; shaded radii outside Lee22 fit".format(z))
        fig.savefig(output/("preferred_profiles_z{}.png".format(int(z))), dpi=160, bbox_inches="tight")
        plt.close(fig)
    best = table[(table["profile"] == "lee22_preferred_duffy") & (table["radius_r200c"] == table["radius_r200c"][0])]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    for z in (0.0, .5, 1.0, 2.0, 4.0):
        rows = best[best["redshift"] == z]
        axes[0].plot(rows["log10_mass_msun"], rows["outer_slope"], label="halo z={}".format(z))
        axes[1].plot(rows["log10_mass_msun"], rows["los_1e5_over_1e4"])
        axes[2].plot(rows["log10_mass_msun"], rows["gas_equivalent_fraction_r200c"])
    axes[0].axhline(1, color="black", ls="--", label="Below: infinite column diverges")
    axes[0].set_ylabel(r"Outer density slope: $n_e\propto r^{-s}$")
    axes[0].legend(fontsize=8)
    axes[1].set_ylabel("Column(LOS=1e5 R200c) / column(1e4 R200c)")
    axes[2].set_ylabel(r"Gas-equivalent mass(<R200c) / $(f_b M_{200c})$")
    for ax in axes:
        ax.set_xlabel(r"$\log_{10}(M_{200c}/M_\odot)$")
        ax.grid(alpha=.2)
    fig.suptitle("Preferred Lee22 extrapolation diagnostics: no gas renormalization")
    fig.savefig(output/"preferred_extrapolation_diagnostics.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def compare_outputs(output):
    labels = ["battaglia16_projected3", "lee22_legacy_projected3",
              "battaglia16_spherical3", "lee22_legacy_spherical3", "lee22_preferred_spherical3"]
    available = {label: np.load(output/(label+".npz")) for label in labels if (output/(label+".npz")).exists()}
    if not available:
        return
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    for col, kind in enumerate(("projected3", "spherical3")):
        reference = available.get("battaglia16_" + kind)
        if reference is None:
            continue
        for label, data in available.items():
            if not label.endswith(kind):
                continue
            ell = data["ell"]
            np.testing.assert_array_equal(ell, reference["ell"])
            factor = ell*(ell+1)/(2*np.pi)
            axes[0, col].loglog(ell, factor*data["dense_bands"], label=label.replace("_", " "))
            axes[1, col].semilogx(ell, 100*(data["dense_bands"]/reference["dense_bands"]-1))
        axes[0, col].set_title(kind + ": complete maps")
        axes[0, col].set_ylabel(r"$D_\ell$ [(pc cm$^{-3}$)$^2$]")
        axes[0, col].legend(fontsize=8)
        axes[1, col].set_ylabel("Difference from matching Battaglia16 [%]")
        axes[1, col].axhline(0, color="black", lw=.7)
    for ax in axes.flat:
        ax.grid(alpha=.2)
        ax.set_xlabel(r"Multipole $\ell$")
    fig.savefig(output/"corrected_full_map_comparison.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", type=Path)
    parser.add_argument("--label")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--lmax", type=int, default=8192)
    parser.add_argument("--profiles-only", action="store_true")
    args = parser.parse_args()
    if args.profiles_only:
        plot_profiles(args.output_dir)
    elif args.map:
        if not args.label or args.samples < 2:
            parser.error("Map analysis requires --label and at least two samples")
        analyze_map(args)
    else:
        compare_outputs(args.output_dir)
