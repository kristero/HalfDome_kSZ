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
    "lee22p_calib": (OUTPUT / "rays/annular_y_samples_lee22_pressure_calib.npz", "Lee22 no-c pressure, calibrated range"),
}
DM_MODELS = {"b16_sphere1": ("Battaglia16", BLUE, "o"), "lee22_noconc_sphere1": ("Lee22", ORANGE, "s")}
GREEN = "#009E73"
# matched (pressure, density) pairs for the selected-realization figures: key -> (y key, DM model, legend, colour)
PAIRS = {
    "battaglia": ("b12", "b16_sphere1", "Battaglia12 y × Battaglia16 DM", BLUE),
    "lee22": ("lee22p", "lee22_noconc_sphere1", "Lee22 y × Lee22 DM, all halos", ORANGE),
    "lee22_calib": ("lee22p_calib", "lee22_noconc_sphere1_calib", "Lee22 y × Lee22 DM, calibrated range only", GREEN),
}
PAIRS_OUT = Path("frb_map_generation/outputs/tsz_dm_fullsky_20260918")
TAKAHASHI25_REAL = Path("frb_map_generation/outputs/takahashi25_real_observations")
TAKAHASHI25_SURVEY_KEY = {"planck": "planck_milca", "act": "act"}
SYMLOG_LINTHRESH = 10.0  # 1e-5 pc cm^-3; linear part of the selected-realization axes
# distinct colour + marker for each shown realization (chosen away from the model colours)
BEST_STYLE = [("#4B0082", "o"), ("#8B4513", "s"), ("#FF1493", "^"), ("#556B2F", "v"), ("#00A5A5", "D"), ("#9ACD32", "p"),
              ("#708090", "*"), ("#DAA520", "h")]
OUTLIER_STYLE = [("#B2182B", "-", "X"), ("#7F0000", (0, (4, 1.5)), "P")]
FILTERS_FWHM = {"planck": 10.0, "act": 1.6}


def sample_y(args):
    import healpy as hp
    target = Y_SOURCES[args.ykey][0]
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
        "y_key": args.ykey, "legend": Y_SOURCES[args.ykey][1],
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
    """Real Takahashi+25 measurement, the same 12 EDGES-matching bins as observed_bins, for the
    plot_main overlay (no covariance/chi2 use here, display only)."""
    obs = observed_bins(plane)
    return obs["x"], obs["w"], obs["err"]


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


# ---------------------------------------------------------------------------------------------
# Matched-pair realizations with the observed numbers of FRBs: mean of 1000, the highest outliers
# and the realizations that fit the Takahashi points best (output under PAIRS_OUT).
# ---------------------------------------------------------------------------------------------
def observed_bins(plane):
    """Real Takahashi+25 w_yDM(theta) measurement and jackknife covariance, restricted to the 12
    bins that match this repository's fixed EDGES annuli (author bin indices 1-12; author bin 0,
    theta ~ 0.75' < 1', falls outside EDGES's range and is not used here). See
    prepare_takahashi25_real_observations.py, which verifies indices 1-12 equal EDGES bins 0-11."""
    survey_key = TAKAHASHI25_SURVEY_KEY[plane]
    with np.load(str(TAKAHASHI25_REAL / (survey_key + ".npz"))) as f:
        lo, hi = f["theta_lo_arcmin"][1:], f["theta_hi_arcmin"][1:]
        if not (np.allclose(lo, EDGES[:-1]) and np.allclose(hi, EDGES[1:])):
            raise ValueError("Observed annuli differ from the simulation annuli")
        x = f["theta_mean_arcmin"][1:]
        w = f["w_yDM_pc_cm3"][1:]
        sigma = f["sigma_pc_cm3"][1:]
        covariance = f["covariance_pc2_cm6"][1:, 1:]
    use = EDGES[:-1] >= PAPER_CUT[plane] - 1e-9
    inv_covariance = np.linalg.inv(covariance[np.ix_(use, use)])
    return dict(x=x, w=w, err=np.vstack([sigma, sigma]), sigma=sigma, covariance=covariance,
                inv_covariance=inv_covariance, use=use)


def chi_square(curves, obs):
    """curves: (..., 12). Generalized chi^2 against the real Takahashi+25 measurement, using the
    full jackknife covariance (inverted once, in observed_bins) of the bins above the paper's
    angular cut -- a real goodness-of-fit weighting, not the earlier diagonal/symmetrized-error
    approximation forced by the plot-digitized data."""
    use = obs["use"]
    delta = curves[..., use] - obs["w"][use]
    return np.einsum("...i,ij,...j->...", delta, obs["inv_covariance"], delta)


def analyze_pairs(args):
    (PAIRS_OUT / "analysis").mkdir(parents=True, exist_ok=True)
    digest = sha256(PREVIOUS / "rays/source_positions.h5")
    rows, arrays, summary = [], {}, {"seed_diagnostic": args.seed + 1, "n_realizations": args.n_diag,
                                     "n_outliers": args.n_outliers, "n_best": args.n_best, "pairs": {}}
    with h5py.File(str(PREVIOUS / "rays/source_positions.h5"), "r") as pos, \
            h5py.File(str(OUTPUT / "rays/individual_dm_updated.h5"), "r") as dmf:
        if decode_text(dmf.attrs["source_positions_sha256"]) != digest:
            raise ValueError("DM positions mismatch")
        for pair, (ykey, model, legend, _) in PAIRS.items():
            path = Y_SOURCES[ykey][0]
            if not path.exists():
                print("skipping pair {} ({} missing)".format(pair, path), flush=True)
                continue
            with np.load(str(path)) as yf:
                meta = json.loads(str(yf["metadata_json"].item()))
                if meta["positions_sha256"] != digest:
                    raise ValueError("y positions mismatch for " + ykey)
                summary["pairs"][pair] = {"y_samples": str(path), "tsz_map_sha256": meta["tsz_map_sha256"], "dm_model": model,
                                          "legend": legend, "planes": {}}
                for plane in ("planck", "act"):
                    groups = pos[plane + "/observed_source_index"][:]
                    dm = dmf[plane + "/" + model][:][:, None]
                    y = yf[plane][:]
                    full, cov = stratified_covariance(dm, y, groups)
                    full = full[0]
                    # same generator state as the 1000 diagnostic realizations of `analyze`: identical ray draws
                    many, chosen = realizations(dm, y, groups, args.n_diag, np.random.RandomState(args.seed + 1))
                    many = many[:, 0, :]
                    obs = observed_bins(plane)
                    chi2 = chi_square(many, obs)
                    peak = np.max(many[:, obs["use"]], axis=1)
                    outliers = np.argsort(peak)[::-1][:args.n_outliers]
                    rest = np.setdiff1d(np.arange(len(many)), outliers)
                    best = rest[np.argsort(chi2[rest])[:args.n_best]]
                    mean = many.mean(axis=0)
                    key = "{}__{}".format(pair, plane)
                    arrays[key + "__realizations"] = many
                    arrays[key + "__chosen_rays"] = chosen
                    arrays[key + "__chi2"] = chi2
                    arrays[key + "__outliers"] = outliers
                    arrays[key + "__best"] = best
                    arrays[key + "__cross_100k"] = full
                    summary["pairs"][pair]["planes"][plane] = dict(
                        chi2_mean_of_realizations=float(chi_square(mean, obs)), chi2_100k=float(chi_square(full, obs)),
                        chi2_median_realization=float(np.median(chi2)), chi2_best=float(chi2[best[0]]),
                        fraction_realizations_better_than_mean=float(np.mean(chi2 < chi_square(mean, obs))),
                        outlier_indices=outliers.tolist(), outlier_peaks_1e_5=(peak[outliers] / 1e-5).tolist(),
                        outlier_chi2=chi2[outliers].tolist(), best_indices=best.tolist(), best_chi2=chi2[best].tolist(),
                        n_annuli_used=int(obs["use"].sum()))
                    for kind, indices in (("mean", [-1]), ("cross_100k", [-2]), ("outlier", outliers), ("best", best)):
                        for i in indices:
                            curve = mean if i == -1 else (full if i == -2 else many[i])
                            row = dict(pair=pair, plane=plane, kind=kind, realization=int(i), chi2=float(chi_square(curve, obs)),
                                       peak_1e_5=float(np.max(curve[obs["use"]]) / 1e-5))
                            row.update({"w_{:.2f}".format(np.sqrt(EDGES[b] * EDGES[b + 1])): curve[b] for b in range(12)})
                            rows.append(row)
    write_rows(PAIRS_OUT / "analysis/finite_source_pairs_selected.csv", rows)
    np.savez_compressed(str(PAIRS_OUT / "analysis/finite_source_pairs.npz"), **arrays)
    (PAIRS_OUT / "analysis/finite_source_pairs_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({p: v["planes"] for p, v in summary["pairs"].items()}, indent=1))


def draw_selected(ax, arrays, pair, plane, n_diag):
    legend, color = PAIRS[pair][2], PAIRS[pair][3]
    key = "{}__{}".format(pair, plane)
    many, outliers, best = arrays[key + "__realizations"], arrays[key + "__outliers"], arrays[key + "__best"]
    x = np.sqrt(EDGES[:-1] * EDGES[1:])
    name, beam, count = PLANES[plane]
    for j, i in enumerate(best):
        ax.plot(x, many[i] / 1e-5, color=".45", lw=1.3, alpha=.75, zorder=2,
                label="{} best-fitting of {}".format(len(best), n_diag) if j == 0 else None)
    for j, i in enumerate(outliers):
        ax.plot(x, many[i] / 1e-5, color="#B2182B", lw=1.6, ls=(0, (4, 1.5)) if j else "-", alpha=.9, zorder=3,
                label="{} highest of {}".format(len(outliers), n_diag) if j == 0 else None)
    ax.plot(x, many.mean(axis=0) / 1e-5, color=color, lw=3.2, zorder=4, label="mean of {} realizations".format(n_diag))
    # the realizations use 31k/71k of the 100k rays; the full 100k estimate is the better ensemble mean
    ax.plot(x, arrays[key + "__cross_100k"] / 1e-5, color=".15", ls=(0, (1.5, 1.5)), lw=1.8, zorder=3.5,
            label="ensemble mean, 100k sightlines")
    obs = observed_bins(plane)
    ax.errorbar(obs["x"], obs["w"] / 1e-5, yerr=obs["err"] / 1e-5, fmt="o", color="black", ms=6, capsize=2.5, lw=1.3,
                label="Takahashi+25", zorder=6)
    ax.axvspan(1, PAPER_CUT[plane], color=".5", alpha=.12, zorder=0)
    style_axis(ax, 1)
    # the highest realizations exceed the data by 2-4 orders of magnitude: linear axis up to
    # SYMLOG_LINTHRESH (data, mean, best fits), logarithmic above (outliers)
    ax.set_yscale("symlog", linthresh=SYMLOG_LINTHRESH, linscale=1.4)
    top = max(np.max(many[outliers]) / 1e-5, np.max(obs["w"] + obs["err"][1]) / 1e-5) * 2.0
    bottom = min(-4.0, 1.3 * min(np.min(many[best]) / 1e-5, np.min(obs["w"] - obs["err"][0]) / 1e-5))
    ax.set_ylim(bottom, top)
    ax.axhline(SYMLOG_LINTHRESH, color=".6", lw=.8, ls=(0, (2, 3)), zorder=1)
    ax.text(0.985, SYMLOG_LINTHRESH, "log scale above ", transform=ax.get_yaxis_transform(), ha="right", va="bottom",
            fontsize=11, color=".4")
    ax.set_title("{}: {}, {} FRBs".format(name, beam, count), pad=8)


def plot_selected(args):
    arrays = dict(np.load(str(PAIRS_OUT / "analysis/finite_source_pairs.npz")))
    summary = json.loads((PAIRS_OUT / "analysis/finite_source_pairs_summary.json").read_text())
    n_diag = summary["n_realizations"]
    (PAIRS_OUT / "plots").mkdir(exist_ok=True)
    plt.rcParams.update(RC)
    for stem, pairs, title in (("battaglia", ["battaglia"], "Battaglia12 pressure × Battaglia16 density"),
                               ("lee22", ["lee22", "lee22_calib"], "Lee22 pressure × Lee22 density")):
        pairs = [p for p in pairs if p in summary["pairs"]]
        if not pairs:
            continue
        nrow = len(pairs)
        fig, axes = plt.subplots(nrow, 2, figsize=(15.5, 6.3 * nrow + 1.2), squeeze=False)
        for i, pair in enumerate(pairs):
            for j, plane in enumerate(("planck", "act")):
                draw_selected(axes[i, j], arrays, pair, plane, n_diag)
                if j == 0:
                    axes[i, j].set_ylabel(r"$w_{y\,\mathrm{DM}}(\theta)\ \ [10^{-5}\ \mathrm{pc\,cm^{-3}}]$")
                if i < nrow - 1:
                    axes[i, j].set_xlabel("")
            if nrow > 1:
                axes[i, 0].text(-0.13, 1.13, PAIRS[pair][2], transform=axes[i, 0].transAxes, fontsize=16, fontweight="bold",
                                ha="left", va="bottom")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        fig.legend([by_label[l] for l in labels], labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0),
                   columnspacing=1.8, handlelength=2.8)
        fig.suptitle("{}:  realizations with the observed number of FRBs".format(title), y=.875 if nrow == 1 else .93,
                     fontsize=16)
        top = .77 if nrow == 1 else .85
        fig.subplots_adjust(left=.07, right=.985, top=top, bottom=.11 if nrow == 1 else .06, hspace=.42, wspace=.2)
        for ext in ("png", "pdf", "svg"):
            fig.savefig(str(PAIRS_OUT / "plots" / ("realizations_selected_{}.{}".format(stem, ext))), dpi=200 if ext == "png" else None)
        plt.close(fig)
    print("Saved selected-realization figures to " + str(PAIRS_OUT / "plots"))


def draw_selected_distinct(ax_w, ax_p, arrays, pair, plane, n_diag):
    """Like draw_selected, with every shown realization in its own colour/marker and a bottom panel with the
    percentage difference from the observations, (w - obs) / |obs|, inside -100..100 %."""
    from fullsky_tsz_dm_comparison import draw_percent, residual_axis
    legend, color = PAIRS[pair][2], PAIRS[pair][3]
    key = "{}__{}".format(pair, plane)
    many, outliers, best = arrays[key + "__realizations"], arrays[key + "__outliers"], arrays[key + "__best"]
    full = arrays[key + "__cross_100k"]
    x = np.sqrt(EDGES[:-1] * EDGES[1:])
    name, beam, count = PLANES[plane]
    obs = observed_bins(plane)
    denominator = np.abs(obs["w"])
    percent = lambda curve: 100 * (curve - obs["w"]) / denominator
    band = np.minimum(100 * obs["sigma"] / denominator, 100)
    ax_p.fill_between(x, -band, band, color=".7", alpha=.35, lw=0, label="observed ±1σ", zorder=1)
    for j, i in enumerate(best):
        c, marker = BEST_STYLE[j % len(BEST_STYLE)]
        ax_w.plot(x, many[i] / 1e-5, color=c, lw=1.5, marker=marker, ms=5.5, alpha=.95, zorder=3, label="best fit {}".format(j + 1))
        draw_percent(ax_p, x, percent(many[i]), c, marker, lw=1.2, ms=5, annotate=False, alpha=.95, zorder=3)
    for j, i in enumerate(outliers):
        c, ls, marker = OUTLIER_STYLE[j % len(OUTLIER_STYLE)]
        ax_w.plot(x, many[i] / 1e-5, color=c, lw=2.0, ls=ls, marker=marker, ms=6.5, zorder=3.2,
                  label="highest of {}".format(n_diag) if j == 0 else "{}. highest".format(j + 1))
        draw_percent(ax_p, x, percent(many[i]), c, marker, ls=ls, lw=1.4, ms=6, annotate=False, zorder=3.2)
    mean = many.mean(axis=0)
    ax_w.plot(x, mean / 1e-5, color=color, lw=3.4, zorder=4, label="mean of {} realizations".format(n_diag))
    draw_percent(ax_p, x, percent(mean), color, None, lw=3.0, annotate=True, zorder=4)
    ax_w.plot(x, full / 1e-5, color=".15", ls=(0, (1.5, 1.5)), lw=1.8, zorder=3.5, label="ensemble mean, 100k sightlines")
    draw_percent(ax_p, x, percent(full), ".15", None, ls=(0, (1.5, 1.5)), lw=1.8, annotate=False, zorder=3.5)
    ax_w.errorbar(obs["x"], obs["w"] / 1e-5, yerr=obs["err"] / 1e-5, fmt="o", color="black", ms=6, capsize=2.5, lw=1.3,
                  label="Takahashi+25", zorder=6)
    for ax in (ax_w, ax_p):
        ax.axvspan(1, PAPER_CUT[plane], color=".5", alpha=.12, zorder=0)
    style_axis(ax_w, 1)
    ax_w.set_xlabel("")
    ax_w.set_yscale("symlog", linthresh=SYMLOG_LINTHRESH, linscale=1.4)
    top = max(np.max(many[outliers]) / 1e-5, np.max(obs["w"] + obs["err"][1]) / 1e-5) * 2.0
    bottom = min(-4.0, 1.3 * min(np.min(many[best]) / 1e-5, np.min(obs["w"] - obs["err"][0]) / 1e-5))
    ax_w.set_ylim(bottom, top)
    ax_w.axhline(SYMLOG_LINTHRESH, color=".6", lw=.8, ls=(0, (2, 3)), zorder=1)
    ax_w.text(0.985, SYMLOG_LINTHRESH, "log scale above ", transform=ax_w.get_yaxis_transform(), ha="right", va="bottom",
              fontsize=11, color=".4")
    ax_w.set_title("{}: {:g}′ Gaussian beam on y, {} FRBs".format(name, FILTERS_FWHM[plane], count), pad=8)
    residual_axis(ax_p, "")
    ax_p.set_xscale("log")
    ax_p.set_xlim(1, 1000)
    ax_p.set_xlabel(r"$\theta$ [arcmin]")


def plot_selected_residuals(args):
    arrays = dict(np.load(str(PAIRS_OUT / "analysis/finite_source_pairs.npz")))
    summary = json.loads((PAIRS_OUT / "analysis/finite_source_pairs_summary.json").read_text())
    n_diag = summary["n_realizations"]
    (PAIRS_OUT / "plots").mkdir(exist_ok=True)
    plt.rcParams.update(RC)
    for stem, pairs, title in (("battaglia", ["battaglia"], "Battaglia12 pressure × Battaglia16 density"),
                               ("lee22", ["lee22", "lee22_calib"], "Lee22 pressure × Lee22 density")):
        pairs = [p for p in pairs if p in summary["pairs"]]
        if not pairs:
            continue
        nrow = len(pairs)
        # one block per pair: w panel, residual panel; a spacer row between blocks carries the pair label
        spacer = nrow > 1  # a label row above every pair block
        ratios = []
        for i in range(nrow):
            ratios += ([0.5] if spacer else []) + [3.0, 1.5]
        fig = plt.figure(figsize=(15.5, 9.6 * nrow + 1.9))
        gs = fig.add_gridspec(len(ratios), 2, height_ratios=ratios, hspace=.1, wspace=.2,
                              left=.075, right=.985, top=.85 if nrow == 1 else .91, bottom=.075 if nrow == 1 else .04)
        first_w = {}
        for i, pair in enumerate(pairs):
            r0 = (3 if spacer else 2) * i + (1 if spacer else 0)
            for j, plane in enumerate(("planck", "act")):
                ax_w = fig.add_subplot(gs[r0, j], sharex=first_w.get(j))
                first_w.setdefault(j, ax_w)
                ax_p = fig.add_subplot(gs[r0 + 1, j], sharex=ax_w)
                draw_selected_distinct(ax_w, ax_p, arrays, pair, plane, n_diag)
                plt.setp(ax_w.get_xticklabels(), visible=False)
                if j == 0:
                    ax_w.set_ylabel(r"$w_{y\,\mathrm{DM}}(\theta)\ \ [10^{-5}\ \mathrm{pc\,cm^{-3}}]$")
                    ax_p.set_ylabel("realization − observed\n[% of |observed|]")
                if i == 0 and j == 0:
                    legend_axes = (ax_w, ax_p)
            if spacer:
                label_ax = fig.add_subplot(gs[r0 - 1, :])
                label_ax.axis("off")
                label_ax.text(0.0, 0.2, PAIRS[pair][2], transform=label_ax.transAxes, fontsize=16, fontweight="bold",
                              ha="left", va="center")
        handles, labels = legend_axes[0].get_legend_handles_labels()
        h2, l2 = legend_axes[1].get_legend_handles_labels()
        by_label = dict(zip(labels + l2, handles + h2))
        order = ["Takahashi+25", "observed ±1σ", "mean of {} realizations".format(n_diag), "ensemble mean, 100k sightlines",
                 "highest of {}".format(n_diag), "2. highest"] + ["best fit {}".format(k + 1) for k in range(len(BEST_STYLE))]
        order = [l for l in order if l in by_label]
        fig.legend([by_label[l] for l in order], order, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.0),
                   columnspacing=1.6, handlelength=2.6)
        fig.suptitle("{}:  realizations with the observed number of FRBs;  triangles at the panel edge: beyond ±100 %".format(title),
                     y=.905 if nrow == 1 else .95, fontsize=15.5)
        for ext in ("png", "pdf", "svg"):
            fig.savefig(str(PAIRS_OUT / "plots" / ("realizations_selected_{}_residuals.{}".format(stem, ext))), dpi=200 if ext == "png" else None)
        plt.close(fig)
    print("Saved selected-realization residual figures to " + str(PAIRS_OUT / "plots"))


def plot_selected_simple(args, n_best=3):
    """Reduced version: only the Takahashi points and the n_best best-fitting realizations, with the percentage
    panel below; large labels, minimal text."""
    from fullsky_tsz_dm_comparison import draw_percent, residual_axis
    arrays = dict(np.load(str(PAIRS_OUT / "analysis/finite_source_pairs.npz")))
    summary = json.loads((PAIRS_OUT / "analysis/finite_source_pairs_summary.json").read_text())
    (PAIRS_OUT / "plots").mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 19, "axes.labelsize": 22, "axes.titlesize": 22, "xtick.labelsize": 18,
                         "ytick.labelsize": 18, "legend.fontsize": 19, "axes.linewidth": 1.2})
    short = {"battaglia": "Battaglia", "lee22": "Lee22, all halos", "lee22_calib": "Lee22, calibrated range"}
    for stem, pairs in (("battaglia", ["battaglia"]), ("lee22", ["lee22", "lee22_calib"])):
        pairs = [q for q in pairs if q in summary["pairs"]]
        if not pairs:
            continue
        nrow = len(pairs)
        ratios = []
        for i in range(nrow):
            ratios += ([0.8] if nrow > 1 else []) + [3.0, 1.4]
        fig = plt.figure(figsize=(16, 8.6 * nrow + 1.2))
        gs = fig.add_gridspec(len(ratios), 2, height_ratios=ratios, hspace=.08, wspace=.16,
                              left=.08, right=.985, top=.88 if nrow == 1 else .95, bottom=.09 if nrow == 1 else .05)
        legend_handles = None
        for i, pair in enumerate(pairs):
            r0 = (3 if nrow > 1 else 2) * i + (1 if nrow > 1 else 0)
            if nrow > 1:
                lab = fig.add_subplot(gs[r0 - 1, :])
                lab.axis("off")
                lab.text(0.0, 0.62, short[pair], transform=lab.transAxes, fontsize=23, fontweight="bold", ha="left", va="center")
            for j, plane in enumerate(("planck", "act")):
                ax_w = fig.add_subplot(gs[r0, j])
                ax_p = fig.add_subplot(gs[r0 + 1, j], sharex=ax_w)
                key = "{}__{}".format(pair, plane)
                many, best = arrays[key + "__realizations"], arrays[key + "__best"][:n_best]
                x = np.sqrt(EDGES[:-1] * EDGES[1:])
                obs = observed_bins(plane)
                denominator = np.abs(obs["w"])
                band = np.minimum(100 * obs["sigma"] / denominator, 100)
                ax_p.fill_between(x, -band, band, color=".72", alpha=.4, lw=0, zorder=1)
                for k, idx in enumerate(best):
                    c, marker = BEST_STYLE[k]
                    ax_w.plot(x, many[idx] / 1e-5, color=c, lw=2.6, marker=marker, ms=8.5, zorder=3, label="best fit {}".format(k + 1))
                    draw_percent(ax_p, x, 100 * (many[idx] - obs["w"]) / denominator, c, marker, lw=2.0, ms=7.5, annotate=False, zorder=3)
                ax_w.errorbar(obs["x"], obs["w"] / 1e-5, yerr=obs["err"] / 1e-5, fmt="o", color="black", ms=8, capsize=3, lw=1.6,
                              label="Takahashi+25", zorder=6)
                for ax in (ax_w, ax_p):
                    ax.axvspan(1, PAPER_CUT[plane], color=".5", alpha=.12, zorder=0)
                    ax.set_xscale("log")
                    ax.set_xlim(1, 1000)
                    ax.grid(alpha=.15, which="both")
                    ax.tick_params(direction="in", which="both", top=True, right=True, length=6, width=1.1)
                ax_w.axhline(0, color=".55", lw=.9)
                plt.setp(ax_w.get_xticklabels(), visible=False)
                ax_w.set_title("{}, {:g}′ beam".format("Planck" if plane == "planck" else "ACT", FILTERS_FWHM[plane]), pad=10)
                residual_axis(ax_p, "")
                if i == nrow - 1:
                    ax_p.set_xlabel(r"$\theta$ [arcmin]")
                if j == 0:
                    ax_w.set_ylabel(r"$w_{y\,\mathrm{DM}}\ \ [10^{-5}\ \mathrm{pc\,cm^{-3}}]$")
                    ax_p.set_ylabel("Δ [%]")
                if legend_handles is None:
                    legend_handles = ax_w.get_legend_handles_labels()
        handles, labels = legend_handles
        order = ["Takahashi+25"] + ["best fit {}".format(k + 1) for k in range(n_best)]
        by_label = dict(zip(labels, handles))
        fig.legend([by_label[l] for l in order], order, loc="upper center", ncol=len(order), frameon=False,
                   bbox_to_anchor=(0.5, 1.0), columnspacing=2.4, handlelength=2.8)
        if nrow == 1:
            fig.text(.5, .905, short[pairs[0]], ha="center", va="bottom", fontsize=23, fontweight="bold")
        for ext in ("png", "pdf", "svg"):
            fig.savefig(str(PAIRS_OUT / "plots" / ("realizations_best{}_{}.{}".format(n_best, stem, ext))), dpi=200 if ext == "png" else None)
        plt.close(fig)
    print("Saved best-{} realization figures to {}".format(n_best, PAIRS_OUT / "plots"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("sample-y", "analyze", "plot", "all", "analyze-pairs", "plot-selected", "plot-simple"))
    parser.add_argument("--ymap", type=Path, help="full-sky Compton-y FITS map for sample-y")
    parser.add_argument("--ykey", default="lee22p", choices=tuple(k for k in Y_SOURCES if k != "b12"),
                        help="which Y_SOURCES entry sample-y writes")
    parser.add_argument("--n-outliers", type=int, default=2, help="highest realizations shown in plot-selected")
    parser.add_argument("--n-best", type=int, default=8, help="best-fitting realizations shown in plot-selected")
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
    if args.stage == "analyze-pairs":
        analyze_pairs(args)
    if args.stage == "plot-selected":
        plot_selected(args)
        plot_selected_residuals(args)
        plot_selected_simple(args)
    if args.stage == "plot-simple":
        plot_selected_simple(args)


if __name__ == "__main__":
    main()
