#!/usr/bin/env python3
"""Read-only Lee22/B16 spectral audit using existing maps and one-million rays.

The production products are never replaced. Test outputs go to a separate
directory. Optional Monte Carlo draws sample the *existing complete sky maps*,
not a subset of halos: every map retains the original foreground population.

For n distinct pixels drawn from P pixels and q=DM-its known full-sky mean,
the exact sampling expectation of the zero-filled harmonic estimate is

    E[C_sparse] = a C_full + N_finite,
    a = P(n-1)/(n(P-1)),
    N_finite = (4*pi/n) ((P-n)/(P-1)) Var_full(DM).

This follows from the two inclusion probabilities n/P and n(n-1)/[P(P-1)]
and the spherical-harmonic addition theorem. It is not the Poisson formula
for independent positions with replacement. The oracle comparison uses the
known full-map variance/mean to test this identity; it is NOT a proposed
observable estimator when the underlying sky is unknown.
"""

import argparse
import csv
import gc
import json
import itertools
from pathlib import Path

import h5py
import healpy as hp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import eval_legendre

ROOT = Path(__file__).resolve().parent
RUN = ROOT / "outputs/tsz_frb_1m_z1_r200c_apertures"
DENSE = ROOT / "outputs/power_spectra"
MAP_NAMES = {
    "battaglia16": "battaglia16_halfdome_full_halo_dm_zsrc1p0_nside4096_m200c_r200cx3p0.fits",
    "lee22": "lee22_noconcentration_halfdome_full_halo_dm_zsrc1p0_nside4096_m200c_r200cx3p0.fits",
}


def bands(cl, lower, upper):
    ell = np.arange(len(cl), dtype=float)
    return np.array([np.average(cl[int(lo):int(hi)+1],
                               weights=2*ell[int(lo):int(hi)+1]+1)
                     for lo, hi in zip(lower, upper)])


def test_finite_population_identity():
    """Exhaust all 792 samples on a 12-pixel toy sky; no MC uncertainty."""
    p, n = 12, 5
    q = np.array([0., 1., 2., 4., 1., 0., 5., 2., 0., 1., 3., 8.])
    q -= np.mean(q)
    vectors = np.asarray(hp.pix2vec(1, np.arange(p)))
    a = p*(n-1)/(n*(p-1))
    inclusion_correction = (p-n)/(p-1)
    for ell in [1, 2, 3]:
        kernel = eval_legendre(ell, np.clip(vectors.T@vectors, -1, 1))
        full_cl = 4*np.pi/p**2 * (q@kernel@q)
        estimates = []
        for indices in itertools.combinations(range(p), n):
            selected = np.array(indices)
            values = q[selected]
            observed = 4*np.pi/n**2 * (values@kernel[np.ix_(selected, selected)]@values)
            sample_noise = 4*np.pi/n * np.mean(values**2)
            estimates.append((observed-inclusion_correction*sample_noise)/a)
        np.testing.assert_allclose(np.mean(estimates), full_cl, atol=1e-12, rtol=1e-12)
    print("PASS: finite-population formula, exhaustive 12-pixel test", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/lee22_power_audit_20260909")
    parser.add_argument("--mc-seeds", type=int, default=3, help="Independent one-million-ray draws per full map; 0 skips.")
    parser.add_argument("--lmax", type=int, default=8192)
    parser.add_argument("--self-test", action="store_true", help="Only run the exhaustive analytic estimator test.")
    args = parser.parse_args()
    test_finite_population_identity()
    if args.self_test:
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dense_path = DENSE / "analysis/lee22_vs_battaglia16_lmax8192_niter0_pixwin0.npz"
    dense = np.load(dense_path)
    if args.lmax != int(dense["lmax"]):
        raise ValueError("Audit requires the same lmax as the saved dense spectra.")
    with h5py.File(RUN / "spectra/tsz_frb_1m_z1_r200c_aperture_spectra.h5") as h5:
        group = h5["dm_cross_log_binned"]
        lower, upper = group["ell_min"][:], group["ell_max"][:]
        ell = group["ell_effective"][:]
        saved = {name: group[name][:] for name in group if name.startswith("cl_dm_")}
    factor = ell*(ell+1)/(2*np.pi)
    summary = {"scope": "3R200c, NSIDE4096, z_source1; original maps/rays unchanged", "models": {}}
    curves, rows = {}, []
    for profile, map_name in MAP_NAMES.items():
        print("Reading full map:", profile, flush=True)
        map_path = DENSE / map_name
        cache_prefix = "b16" if profile == "battaglia16" else "lee22"
        stat = map_path.stat()
        if stat.st_size != int(dense[cache_prefix+"_size"]) or stat.st_mtime_ns != int(dense[cache_prefix+"_mtime_ns"]):
            raise ValueError("Saved full-map spectrum is stale: " + str(map_path))
        full = hp.read_map(map_path, dtype=np.float64)
        p = full.size
        mu, variance = float(np.mean(full)), float(np.var(full))
        disk_profile = "lee2022" if profile == "lee22" else profile
        ray_path = RUN / f"ray_catalogues/{disk_profile}_r3_zsrc1p0_nside4096_nfrb1000000_seed42.h5"
        with h5py.File(ray_path) as h5:
            pixels = h5["frb_pixel_ring_1based"][:].astype(np.int64)-1
            dm = np.asarray(h5["dm_pc_cm3"][:]).reshape(-1)
        n = dm.size
        assert n == 1_000_000 and np.unique(pixels).size == n
        delta = full[pixels]-dm
        stats = {
            "n_rays": n, "n_pixels": p, "sampling_fraction": n/p,
            "full_map_mean_dm": mu, "full_map_variance": variance,
            "ray_vs_dense_max_abs_dm": float(np.max(np.abs(delta))),
            "ray_vs_dense_rms_dm": float(np.sqrt(np.mean(delta**2))),
            "ray_vs_dense_median_relative": float(np.median(np.abs(delta)/np.maximum(dm, 1e-30))),
        }
        a = p*(n-1)/(n*(p-1))
        noise_factor = (p-n)/(p-1)
        oracle_noise = 4*np.pi/n * variance * noise_factor
        observed = saved[f"cl_dm_observed_{profile}_r3"]
        old = saved[f"cl_dm_corrected_{profile}_r3"]
        # Approximate sample-only finite-population repair. Sample mean removal
        # introduces extra terms of order 1/n, so do not label it exact.
        sample_noise_scale = n*(p-n)/(p*(n-1))
        sample_finite = (observed - sample_noise_scale*saved[f"cl_dm_shot_{profile}_r3"])/a
        full_cl = dense["cl_battaglia16" if profile == "battaglia16" else "cl_lee22"]
        full_bands = bands(full_cl, lower, upper)
        mc_old, mc_finite, mc_oracle = [], [], []
        for seed in range(args.mc_seeds):
            rng = np.random.default_rng(20260909+seed)
            selection = rng.choice(p, size=n, replace=False)
            # Known full mean isolates finite-population theory from estimated-
            # mean effects. Same random indices in both models aid comparison.
            q = full[selection]-mu
            sparse = np.zeros(p, dtype=np.float64)
            sparse[selection] = q*(p/n)
            print(profile, "sampling seed", seed, flush=True)
            alm = hp.map2alm(sparse, lmax=args.lmax, iter=0, pol=False, use_pixel_weights=False)
            cl = hp.alm2cl(alm)
            del sparse, alm
            sample_shot = 4*np.pi/n * np.mean(q*q)
            mc_old.append(bands(cl-sample_shot, lower, upper))
            # With the known full mean, mean(q_sample**2) is unbiased for the
            # population variance. Subtracting its own diagonal term also
            # cancels realization-to-realization shot-amplitude fluctuations.
            mc_finite.append(bands((cl-noise_factor*sample_shot)/a, lower, upper))
            mc_oracle.append(bands((cl-oracle_noise)/a, lower, upper))
            gc.collect()
        stats.update({
            "oracle_finite_population_noise_cl": oracle_noise,
            "saved_poisson_shot_cl": float(saved[f"cl_dm_shot_{profile}_r3"][0]),
            "last_bin_ell": float(ell[-1]),
            "last_bin_dense_dl": float(factor[-1]*full_bands[-1]),
            "last_bin_legacy_dl": float(factor[-1]*old[-1]),
            "last_bin_sample_finite_dl": float(factor[-1]*sample_finite[-1]),
            "legacy_negative_bands": int(np.sum(old<0)),
        })
        if mc_finite:
            stats["mc_last_bin_finite_dl"] = (np.asarray(mc_finite)[:, -1]*factor[-1]).tolist()
            stats["mc_last_bin_legacy_dl"] = (np.asarray(mc_old)[:, -1]*factor[-1]).tolist()
            stats["mc_last_bin_fixed_variance_dl"] = (np.asarray(mc_oracle)[:, -1]*factor[-1]).tolist()
        summary["models"][profile] = stats
        curves[profile] = dict(dense=full_bands, legacy=old, sample_finite=sample_finite,
                               shot=saved[f"cl_dm_shot_{profile}_r3"], mc_finite=np.asarray(mc_finite),
                               mc_legacy=np.asarray(mc_old), mc_fixed_variance=np.asarray(mc_oracle))
        for j, l in enumerate(ell):
            rows.append(dict(profile=profile, ell=l, dense_dl=factor[j]*full_bands[j],
                             legacy_dl=factor[j]*old[j], sample_finite_dl=factor[j]*sample_finite[j],
                             shot_dl=factor[j]*saved[f"cl_dm_shot_{profile}_r3"][j]))
        print(json.dumps(stats, indent=2), flush=True)
        del full
        gc.collect()
    with (args.output_dir / "sparse_power_audit.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    with (args.output_dir / "sparse_vs_dense_bands.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez(args.output_dir / "audit_curves.npz", ell=ell, **{
        f"{profile}_{key}": value for profile, items in curves.items() for key, value in items.items()
    })
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.6), layout="constrained")
    for profile, color in [("battaglia16", "#28749a"), ("lee22", "#d24863")]:
        c = curves[profile]
        axes[0].loglog(ell, factor*c["dense"], color=color, label=profile+" complete map")
    axes[0].set_title("Complete-sky maps: no sparse-ray noise")
    axes[0].legend(fontsize=9)
    for ax, (profile, color) in zip(axes[1:], [("battaglia16", "#28749a"), ("lee22", "#d24863")]):
        c = curves[profile]
        ax.plot(ell, factor*c["dense"], color="black", lw=2, label="Complete-map reference")
        ax.plot(ell, factor*c["legacy"], color=color, marker=".", label="Saved Poisson subtraction")
        ax.plot(ell, factor*c["sample_finite"], color=color, ls="--", label="Finite-population diagnostic")
        for k, mc in enumerate(c["mc_finite"]):
            ax.plot(ell, factor*mc, color="0.55", alpha=.45, lw=.8,
                    label="Independent samples (known sky mean)" if k == 0 else None)
        ax.axhline(0, color="black", lw=.6)
        ax.set_xscale("log")
        ax.set_yscale("symlog", linthresh=2000)
        ax.set_xlim(700, 8500)
        ax.set_title(profile+": negative bands retained")
        ax.legend(fontsize=8, loc="lower left")
    for ax in axes:
        ax.set_xlabel(r"Multipole $\ell$")
        ax.set_ylabel(r"$D_\ell^{\rm DM}$ [$(\mathrm{pc\,cm^{-3}})^2$]")
        ax.grid(alpha=.2, which="both")
    for ext in ["png", "pdf"]:
        fig.savefig(args.output_dir / ("sparse_vs_dense_power_audit."+ext), dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
