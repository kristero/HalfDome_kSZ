#!/usr/bin/env python3
"""Finite-source realizations of the tSZ x FRB-DM cross-correlation, and a second Compton-y map.

Two additions to compare_updated_sightlines.py:

1. `sample-y`: annular Compton-y samples at the 100k source positions for any full-sky y map
   (here the Lee22 no-concentration pressure map), the same filters and 12 annuli as the 2026-09-14
   Battaglia12 samples (10' Planck beam, 1.6' ACT beam, unbeamed; NSIDE 4096, lmax 8192).
2. `analyze` / `plot`: for each source plane, N disjoint realizations that use exactly the observed
   number of FRBs (71 Planck, 31 ACT), one random ray per observed redshift. Each realization's
   cross-correlation is
       w_b = (1/G) sum_g (D_{g} - <D>_g) (Y_{g,b} - <Y>_{g,b}),
   with the stratum means taken from all 100k rays (the analogue of the DM-z relation and of the
   random-position y mean of the observational estimator). The mean over realizations and its
   scatter show the small-scale sampling noise of a 71/31-source measurement on this fixed sky.
"""
import argparse
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from compare_halfdome_takahashi import read_rows, write_rows, sha256, gaussian_beam
from compare_takahashi_sightlines import stratified_covariance, decode_text, EDGES, FILTERS, annulus_window
from compare_updated_sightlines import (PREVIOUS, INPUTS, OUTPUT, PLANES, PAPER_CUT, BLUE, ORANGE, RC, column, style_axis)

Y_SOURCES = {  # key: (annular-sample file, legend)
    "b12": (PREVIOUS / "rays/annular_y_samples.npz", "Battaglia12 pressure"),
    "lee22p": (OUTPUT / "rays/annular_y_samples_lee22_pressure.npz", "Lee22 no-c pressure"),
}
DM_MODELS = {"b16_sphere1": ("Battaglia16", BLUE, "o"), "lee22_noconc_sphere1": ("Lee22", ORANGE, "s")}


def sample_y(args):
    import healpy as hp
    target = OUTPUT / "rays/annular_y_samples_lee22_pressure.npz"
    if target.exists():
        raise FileExistsError(str(target))
    with h5py.File(str(PREVIOUS / "rays/source_positions.h5"), "r") as h5:
        pixels = h5["pixel_ring_zero_based"][:]
        nside = int(h5.attrs["nside"])
    digest = sha256(args.ymap)
    y = hp.read_map(str(args.ymap), dtype=np.float64)
    if hp.get_nside(y) != nside or not np.all(np.isfinite(y)) or np.any(y < 0):
        raise ValueError("Invalid Compton-y map")
    y_mean = float(np.mean(y))
    y -= y_mean
    print("map2alm lmax={} ...".format(args.lmax), flush=True)
    alm = hp.map2alm(y, lmax=args.lmax, iter=3, pol=False)
    cl_yy = hp.alm2cl(alm)
    del y
    arrays = {"lower_arcmin": EDGES[:-1], "upper_arcmin": EDGES[1:], "cl_yy": cl_yy}
    ell = np.arange(args.lmax + 1)
    for name, beam in FILTERS:
        values = np.empty((len(pixels), 12))
        for b, (lo, hi) in enumerate(zip(EDGES[:-1], EDGES[1:])):
            window = annulus_window(args.lmax, lo, hi) * gaussian_beam(ell, beam)
            window[0] = 0.0
            filtered = hp.alm2map(hp.almxfl(alm, window), nside, pol=False)
            values[:, b] = filtered[pixels]
            del filtered
            print("{} annulus {}/12".format(name, b + 1), flush=True)
        arrays[name] = values
    arrays["metadata_json"] = np.asarray(json.dumps({
        "tsz_map": str(args.ymap), "tsz_map_sha256": digest, "y_mean": y_mean, "lmax": args.lmax, "nside": nside,
        "positions_sha256": sha256(PREVIOUS / "rays/source_positions.h5"),
        "operation": "annular full-sky y average at each individual source position; y monopole subtracted",
        "noise": "none", "mask": "none"}))
    np.savez_compressed(str(target), **arrays)
    print("Saved " + str(target), flush=True)


def realizations(dm, y, groups, n_real, rng):
    """dm: (n, nmodel); y: (n, 12); groups: (n,) stratum ids. Returns (n_real, nmodel, 12)."""
    strata = np.unique(groups)
    dm_centered = np.empty_like(dm)
    y_centered = np.empty_like(y)
    chosen = np.empty((n_real, len(strata)), dtype=int)
    for k, g in enumerate(strata):
        idx = np.flatnonzero(groups == g)
        if len(idx) < n_real:
            raise ValueError("Stratum too small for disjoint realizations")
        dm_centered[idx] = dm[idx] - dm[idx].mean(axis=0)
        y_centered[idx] = y[idx] - y[idx].mean(axis=0)
        chosen[:, k] = rng.permutation(idx)[:n_real]
    out = np.empty((n_real, dm.shape[1], y.shape[1]))
    for i in range(n_real):
        rays = chosen[i]
        out[i] = np.einsum("gp,gb->pb", dm_centered[rays], y_centered[rays]) / len(strata)
    return out, chosen


def analyze(args):
    (OUTPUT / "analysis").mkdir(exist_ok=True)
    rng = np.random.RandomState(args.seed)
    labels = list(DM_MODELS)
    rows, arrays, summary = [], {}, {"seed": args.seed, "n_realizations": args.n_real, "y_sources": {}}
    with h5py.File(str(PREVIOUS / "rays/source_positions.h5"), "r") as pos, \
            h5py.File(str(OUTPUT / "rays/individual_dm_updated.h5"), "r") as dmf:
        digest = sha256(PREVIOUS / "rays/source_positions.h5")
        if decode_text(dmf.attrs["source_positions_sha256"]) != digest:
            raise ValueError("DM positions mismatch")
        for ykey, (path, legend) in Y_SOURCES.items():
            if not path.exists():
                print("skipping {} ({} missing)".format(ykey, path))
                continue
            with np.load(str(path)) as yf:
                meta = json.loads(str(yf["metadata_json"].item()))
                if meta["positions_sha256"] != digest:
                    raise ValueError("y positions mismatch for " + ykey)
                summary["y_sources"][ykey] = {"file": str(path), "tsz_map_sha256": meta["tsz_map_sha256"], "legend": legend}
                for plane in ("planck", "act"):
                    groups = pos[plane + "/observed_source_index"][:]
                    dm = np.column_stack([dmf[plane + "/" + m][:] for m in labels])
                    y = yf[plane][:]
                    full, cov = stratified_covariance(dm, y, groups)
                    err = np.sqrt(np.maximum(0, np.diag(cov))).reshape(full.shape)
                    real, chosen = realizations(dm, y, groups, args.n_real, rng)
                    # large disjoint set for the distribution of a single 71/31-source measurement
                    many, _ = realizations(dm, y, groups, args.n_diag, np.random.RandomState(args.seed + 1))
                    arrays["{}__{}__realizations".format(ykey, plane)] = real
                    arrays["{}__{}__chosen_rays".format(ykey, plane)] = chosen
                    arrays["{}__{}__diagnostic_realizations".format(ykey, plane)] = many
                    for p, m in enumerate(labels):
                        for b in range(12):
                            d = many[:, p, b]
                            rows.append(dict(y_source=ykey, plane=plane, model=m, theta_lower_arcmin=EDGES[b],
                                theta_upper_arcmin=EDGES[b + 1], theta_arcmin=np.sqrt(EDGES[b] * EDGES[b + 1]),
                                cross_100k=full[p, b], sigma_100k=err[p, b],
                                realization_mean=real[:, p, b].mean(), realization_std=real[:, p, b].std(ddof=1),
                                realization_min=real[:, p, b].min(), realization_max=real[:, p, b].max(),
                                diag_n=len(d), diag_mean=d.mean(), diag_std=d.std(ddof=1), diag_median=np.median(d),
                                diag_p16=np.percentile(d, 16), diag_p84=np.percentile(d, 84),
                                diag_fraction_above_100k=np.mean(d > full[p, b])))
    write_rows(OUTPUT / "analysis/finite_source_realizations.csv", rows)
    np.savez_compressed(str(OUTPUT / "analysis/finite_source_realizations.npz"), **arrays)
    (OUTPUT / "analysis/finite_source_realizations.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def observed(plane):
    obs = read_rows(INPUTS / "digitized/takahashi_fig13_approximate.csv")
    obs = [r for r in obs if ("ACT" in r["series"]) == (plane == "act")]
    return (column(obs, "theta_plotted_arcmin"), column(obs, "w_yDM_pc_cm3"),
            np.vstack([column(obs, "error_lower_pc_cm3"), column(obs, "error_upper_pc_cm3")]))


def plot_main(rows, ykey, reference_key, out):
    """100k curves and the N-realization means for one y map; the other y map dotted for reference."""
    plt.rcParams.update(RC)
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 7.4))
    n_real = None
    for ax, plane in zip(axes, ("planck", "act")):
        name, beam, count = PLANES[plane]
        x_obs, w_obs, e_obs = observed(plane)
        ax.errorbar(x_obs, w_obs / 1e-5, yerr=e_obs / 1e-5, fmt="o", color="black", ms=6, capsize=2.5, lw=1.3,
                    label="Takahashi+25", zorder=6)
        for m, (legend, color, marker) in DM_MODELS.items():
            sel = [r for r in rows if r["y_source"] == ykey and r["plane"] == plane and r["model"] == m]
            x = column(sel, "theta_arcmin")
            ax.errorbar(x, column(sel, "cross_100k") / 1e-5, yerr=column(sel, "sigma_100k") / 1e-5, color=color, lw=3.0,
                        marker=marker, ms=6, capsize=2, label=legend + " DM, 100k sightlines", zorder=4)
            ax.errorbar(x * 1.06, column(sel, "realization_mean") / 1e-5,
                        yerr=column(sel, "realization_std") / 1e-5 / np.sqrt(10), color=color, ls="none", marker=marker,
                        ms=8, mfc="white", mew=1.8, capsize=3, elinewidth=1.4,
                        label=legend + " DM, mean of 10 x {} FRBs".format(count), zorder=5)
            if reference_key is not None:
                ref = [r for r in rows if r["y_source"] == reference_key and r["plane"] == plane and r["model"] == m]
                if ref:
                    ax.plot(column(ref, "theta_arcmin"), column(ref, "cross_100k") / 1e-5, color=color,
                            ls=(0, (1.2, 1.6)), lw=1.8, label=legend + " DM, " + Y_SOURCES[reference_key][1] + " y", zorder=3)
        ax.axvspan(1, PAPER_CUT[plane], color=".5", alpha=.12, zorder=0)
        style_axis(ax, 1)
        ax.set_title("{}: {}, {} FRB redshifts".format(name, beam, count), pad=10)
    axes[0].set_ylabel(r"$w_{y\,\mathrm{DM}}(\theta)\ \ [10^{-5}\ \mathrm{pc\,cm^{-3}}]$")
    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    order = [l for l in labels if l.startswith("Battaglia16")] + [l for l in labels if l.startswith("Lee22")] + ["Takahashi+25"]
    fig.legend([by_label[l] for l in order], order, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.0), columnspacing=1.2, handlelength=2.6)
    fig.suptitle("Compton-y from the {}".format(Y_SOURCES[ykey][1]), y=.985 - .13, fontsize=17)
    fig.subplots_adjust(left=.07, right=.985, top=.74, bottom=.12, wspace=.2)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(str(out / "plots" / ("takahashi_fig13_y_{}.{}".format(ykey, ext))), dpi=200 if ext == "png" else None)
    plt.close(fig)


def plot_realizations(rows, arrays, ykey, out):
    """Only the individual realizations (thin lines) and the observations."""
    plt.rcParams.update(RC)
    models = list(DM_MODELS)
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 11.5), sharex="col")
    for i, m in enumerate(models):
        legend, color, marker = DM_MODELS[m]
        for j, plane in enumerate(("planck", "act")):
            ax = axes[i, j]
            name, beam, count = PLANES[plane]
            real = arrays["{}__{}__realizations".format(ykey, plane)][:, i, :]
            x = np.sqrt(EDGES[:-1] * EDGES[1:])
            for k in range(real.shape[0]):
                ax.plot(x, real[k] / 1e-5, color=color, lw=1.2, alpha=.65,
                        label="single realization, {} FRBs".format(count) if k == 0 else None, zorder=3)
            sel = [r for r in rows if r["y_source"] == ykey and r["plane"] == plane and r["model"] == m]
            ax.fill_between(x, column(sel, "diag_p16") / 1e-5, column(sel, "diag_p84") / 1e-5, color=color, alpha=.13, lw=0,
                            label="16-84% of 1000 realizations", zorder=1)
            ax.plot(x, column(sel, "diag_median") / 1e-5, color=color, lw=2.0, ls=(0, (4, 2)),
                    label="median of 1000 realizations", zorder=3.5)
            ax.plot(x, column(sel, "cross_100k") / 1e-5, color=".15", lw=2.6, label="100k sightlines (= ensemble mean)", zorder=4)
            x_obs, w_obs, e_obs = observed(plane)
            ax.errorbar(x_obs, w_obs / 1e-5, yerr=e_obs / 1e-5, fmt="o", color="black", ms=6, capsize=2.5, lw=1.3,
                        label="Takahashi+25", zorder=6)
            ax.axvspan(1, PAPER_CUT[plane], color=".5", alpha=.12, zorder=0)
            style_axis(ax, 1)
            ax.set_title("{} DM;  {}: {}, {} FRBs".format(legend, name, beam, count), pad=8)
            if j == 0:
                ax.set_ylabel(r"$w_{y\,\mathrm{DM}}(\theta)\ \ [10^{-5}\ \mathrm{pc\,cm^{-3}}]$")
            if i == 0:
                ax.set_xlabel("")
            if i == 0 and j == 0:
                ax.legend(loc="upper right", frameon=False)
    fig.suptitle("10 realizations with the observed number of FRBs;  Compton-y from the {}".format(Y_SOURCES[ykey][1]),
                 fontsize=17, y=.985)
    fig.subplots_adjust(left=.07, right=.985, top=.92, bottom=.07, hspace=.22, wspace=.2)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(str(out / "plots" / ("takahashi_fig13_realizations_y_{}.{}".format(ykey, ext))), dpi=200 if ext == "png" else None)
    plt.close(fig)


def plot_cl_yy(out):
    """Auto power spectra of the two Compton-y maps (monopole removed; no beam)."""
    lee = Y_SOURCES["lee22p"][0]
    if not lee.exists():
        return
    with np.load(str(lee)) as f, np.load(str(PREVIOUS.parent / "takahashi_cross_comparison_20260913/spectra/battaglia16.npz")) as g:
        cl_lee, cl_b12 = f["cl_yy"], g["cl_yy"]
    ell = np.arange(len(cl_lee))
    plt.rcParams.update(RC)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))
    keep = ell >= 10
    fac = ell * (ell + 1) / (2 * np.pi)
    axes[0].loglog(ell[keep], (fac * cl_b12)[keep], color=".15", lw=2.2, label="Battaglia12 pressure")
    axes[0].loglog(ell[keep], (fac * cl_lee)[keep], color=ORANGE, lw=2.2, ls="--", label="Lee22 no-c pressure")
    axes[0].set_ylabel(r"$\ell(\ell+1)\,C_\ell^{yy}/2\pi$")
    axes[0].legend(frameon=False)
    axes[1].semilogx(ell[keep], (cl_lee / cl_b12)[keep], color=ORANGE, lw=2.2)
    axes[1].axhline(1, color=".5", lw=.9)
    axes[1].set_ylabel(r"$C_\ell^{yy}$ ratio  Lee22 / Battaglia12")
    ratio = cl_lee / np.maximum(cl_b12, 1e-300)
    smooth = np.convolve(ratio[keep], np.ones(51) / 51, mode="same")
    axes[1].set_ylim(0, 1.15 * np.nanmax(smooth[25:-25]))
    for ax in axes:
        ax.set_xlabel(r"$\ell$")
        ax.grid(alpha=.15, which="both")
        ax.tick_params(direction="in", which="both", top=True, right=True)
    fig.suptitle("HalfDome full-lightcone Compton-y maps (NSIDE 4096, XGPaint 4R200c aperture, no beam)", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, .95))
    for ext in ("png", "pdf"):
        fig.savefig(str(out / "plots" / ("cl_yy_lee22_pressure_vs_b12." + ext)), dpi=180 if ext == "png" else None)
    plt.close(fig)


def plot(args):
    rows = read_rows(OUTPUT / "analysis/finite_source_realizations.csv")
    arrays = dict(np.load(str(OUTPUT / "analysis/finite_source_realizations.npz")))
    available = sorted({r["y_source"] for r in rows})
    (OUTPUT / "plots").mkdir(exist_ok=True)
    for ykey in available:
        other = [k for k in available if k != ykey]
        plot_main(rows, ykey, other[0] if other else None, OUTPUT)
        plot_realizations(rows, arrays, ykey, OUTPUT)
    plot_cl_yy(OUTPUT)
    print("Saved figures for y sources: " + ", ".join(available))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("sample-y", "analyze", "plot", "all"))
    parser.add_argument("--ymap", type=Path, help="full-sky Compton-y FITS map for sample-y")
    parser.add_argument("--lmax", type=int, default=8192)
    parser.add_argument("--n-real", type=int, default=10)
    parser.add_argument("--n-diag", type=int, default=1000, help="disjoint realizations for the distribution diagnostic")
    parser.add_argument("--seed", type=int, default=20260917)
    args = parser.parse_args()
    if args.stage == "sample-y":
        sample_y(args)
    if args.stage in ("analyze", "all"):
        analyze(args)
    if args.stage in ("plot", "all"):
        plot(args)


if __name__ == "__main__":
    main()
