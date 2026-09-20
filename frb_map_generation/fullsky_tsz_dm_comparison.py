#!/usr/bin/env python3
"""Full-sky (map x map) tSZ x FRB-DM comparison with Takahashi et al. 2025, updated implementations.

The "cluster" method of 2026-09-13 (compare_halfdome_takahashi.py): every pixel of a full-sky,
observed-redshift-weighted halo-DM map is a source, so <y DM> over the sky is the ensemble mean of
the stratified sightline estimator; no 100k sightlines, no finite-source sampling noise.

Maps (all NSIDE 4096, full HalfDome lightcone, physical M200c):
  Compton-y   b12          Battaglia12 pressure (XGPaint 4 R200c projected aperture)
              lee22p       Lee22 no-c pressure (Table 7), XGPaint mapping, same aperture
              lee22p_calib the same, only halos inside the Lee22 calibration ranges
                           (1e13-10^14.8 h^-1 Msun, z <= 2)
  DM          b16_sphere1, lee22_noconc_sphere1, lee22_noconc_sphere1_calib, each weighted with the
              Planck (71) and ACT (31) observed-redshift kernels (paint_halfdome_kernel_weighted_dm_maps.jl)
Pairs: Battaglia12 y x Battaglia16 DM; Lee22 y x Lee22 DM; Lee22 y x Lee22 DM with both restricted
to the calibration ranges.

Stages
  spectra : map2alm (lmax 8192, 3 iterations) of the 3 y and 6 DM maps; all auto- and cross-spectra
  check   : kernel-weighted map means against the weighted mean of the 100k individual sightlines
  plot    : Takahashi Fig. 13 comparison (Planck 10', ACT 1.6' beams on y) and the three-column
            power-spectrum figure (y auto, DM auto, y x DM)
"""
import argparse
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from compare_halfdome_takahashi import (read_rows, write_rows, sha256, read_provenance, gaussian_beam,
                                        angular_correlation, annular_correlation)
from compare_takahashi_sightlines import EDGES
from compare_updated_sightlines import INPUTS, OUTPUT as UPDATED, PLANES, PAPER_CUT, RC, BLUE, ORANGE, column, style_axis

OUT = Path("frb_map_generation/outputs/tsz_dm_fullsky_20260918")
PREVIOUS_RAYS = Path("frb_map_generation/outputs/takahashi_100k_20260914/rays/source_positions.h5")
GREEN = "#009E73"
Y_MAPS = {
    "b12": (Path("frb_map_generation/outputs/tsz_frb_1m_z1_r200c_apertures/maps/"
                 "battaglia12_fiducial_halfdome_compton_y_allz_nside4096_m200c_r200cx4.fits"), "Battaglia12 y"),
    "lee22p": (UPDATED / "inputs/lee22_noconc_pressure_compton_y_allz_nside4096_m200c_r200cx4.fits", "Lee22 y"),
    "lee22p_calib": (OUT / "maps/lee22_noconc_pressure_compton_y_calib_zmax2p0_nside4096_m200c_r200cx4.fits",
                     "Lee22 y, calibrated range"),
}
DM_MODELS = ("b16_sphere1", "lee22_noconc_sphere1", "lee22_noconc_sphere1_calib")
SURVEYS = ("planck", "act")
# pair key: (y key, DM model, legend, colour, linestyle, marker)
PAIRS = {
    "battaglia": ("b12", "b16_sphere1", "Battaglia12 y × Battaglia16 DM", BLUE, "-", "o"),
    "lee22": ("lee22p", "lee22_noconc_sphere1", "Lee22 y × Lee22 DM", ORANGE, "-", "s"),
    "lee22_calib": ("lee22p_calib", "lee22_noconc_sphere1_calib", "Lee22 y × Lee22 DM, calibrated range only", GREEN, "--", "D"),
}
KERNEL_STYLE = {"planck": ("-", "Planck FRB redshifts, 71"), "act": ((0, (1.6, 1.6)), "ACT FRB redshifts, 31")}
BEAMS = {"planck": 10.0, "act": 1.6}  # Gaussian FWHM [arcmin] applied to y only
SURVEY_LABEL = {"planck": "Planck: 71 FRB redshifts, 10′ beam on y", "act": "ACT: 31 FRB redshifts, 1.6′ beam on y"}
SPECTRA = OUT / "spectra/fullsky_spectra.npz"
TAKAHASHI25_REAL = Path("frb_map_generation/outputs/takahashi25_real_observations")
TAKAHASHI25_SURVEY_KEY = {"planck": "planck_milca", "act": "act"}


def dm_map_path(model, survey):
    return OUT / "maps" / "{}_{}.fits".format(model, survey)


def load_centered(path, nside_expected=4096):
    import healpy as hp
    m = hp.read_map(str(path), dtype=np.float64)
    if hp.get_nside(m) != nside_expected or not np.all(np.isfinite(m)) or np.any(m < 0):
        raise ValueError("Invalid full-sky map: " + str(path))
    mean = float(np.mean(m))
    m -= mean
    return m, mean


def spectra(args):
    import healpy as hp
    if SPECTRA.exists() and not args.overwrite:
        raise FileExistsError(str(SPECTRA))
    SPECTRA.parent.mkdir(parents=True, exist_ok=True)
    lmax = args.lmax
    alms, arrays, meta = {}, {"ell": np.arange(lmax + 1)}, {"lmax": lmax, "map2alm_iterations": args.iter,
                                                             "maps": {}, "pixel_window": "not deconvolved"}
    for key, (path, legend) in Y_MAPS.items():
        prov = read_provenance(path.with_name(path.stem + "_provenance.txt"))
        if prov["catalog_rows_scanned"] != prov["catalog_total_rows"] or prov["beam"] != "none":
            raise ValueError("Incomplete or beamed y map: " + str(path))
        m, mean = load_centered(path)
        print("map2alm y {} (mean y {:.4e})".format(key, mean), flush=True)
        alms["y_" + key] = hp.map2alm(m, lmax=lmax, iter=args.iter, pol=False)
        del m
        meta["maps"]["y_" + key] = {"path": str(path), "sha256": sha256(path), "mean": mean, "legend": legend, "provenance": prov}
    kernel_digest = {s: sha256(OUT / "kernels" / (s + "_sources.csv")) for s in SURVEYS}
    for model in DM_MODELS:
        for survey in SURVEYS:
            path = dm_map_path(model, survey)
            prov = read_provenance(path.with_name(path.stem + "_provenance.txt"))
            if (prov["catalog_rows_scanned"] != prov["catalog_total_rows"] or prov["source_kernel_sha256"] != kernel_digest[survey]
                    or int(prov["source_count"]) != PLANES[survey][2] or prov["profile_label"] != model):
                raise ValueError("DM map provenance mismatch: " + str(path))
            m, mean = load_centered(path)
            print("map2alm DM {} {} (mean DM {:.3f} pc/cm3)".format(model, survey, mean), flush=True)
            alms["dm_{}_{}".format(model, survey)] = hp.map2alm(m, lmax=lmax, iter=args.iter, pol=False)
            del m
            meta["maps"]["dm_{}_{}".format(model, survey)] = {"path": str(path), "sha256": sha256(path), "mean": mean, "provenance": prov}
    for key, alm in alms.items():
        arrays["cl_" + key] = hp.alm2cl(alm)
    for ykey in Y_MAPS:
        for model in DM_MODELS:
            for survey in SURVEYS:
                cross = hp.alm2cl(alms["y_" + ykey], alms["dm_{}_{}".format(model, survey)])
                cross[0] = 0.0
                auto_y, auto_d = arrays["cl_y_" + ykey], arrays["cl_dm_{}_{}".format(model, survey)]
                if not np.all(np.isfinite(cross)) or np.any(cross[1:] ** 2 > auto_y[1:] * auto_d[1:] * (1 + 1e-9)):
                    raise ValueError("Invalid cross spectrum {} x {} {}".format(ykey, model, survey))
                arrays["cl_y_{}__dm_{}_{}".format(ykey, model, survey)] = cross
    arrays["metadata_json"] = np.asarray(json.dumps(meta))
    np.savez_compressed(str(SPECTRA), **arrays)
    print("Saved " + str(SPECTRA), flush=True)


def check(args):
    """Kernel-weighted map mean = expected DM of a random sightline with the observed redshift
    distribution; the 100k individual sightlines (one stratum per observed redshift, equal stratum
    weights) estimate the same number with finite sampling error."""
    rows = []
    with h5py.File(str(PREVIOUS_RAYS), "r") as pos, h5py.File(str(UPDATED / "rays/individual_dm_updated.h5"), "r") as dmf:
        for model in DM_MODELS:
            for survey in SURVEYS:
                prov = read_provenance(dm_map_path(model, survey).with_name(model + "_" + survey + "_provenance.txt"))
                w = pos[survey + "/analysis_weight"][:]
                w = w / w.sum()
                dm = dmf[survey + "/" + model][:]
                mean_rays = float(np.sum(w * dm))
                # variance of the weighted mean (independent rays)
                err = float(np.sqrt(np.sum(w ** 2 * (dm - mean_rays) ** 2)))
                rows.append(dict(model=model, survey=survey, map_mean_pc_cm3=float(prov["map_mean_pc_cm3"]),
                                 rays_weighted_mean_pc_cm3=mean_rays, rays_standard_error_pc_cm3=err,
                                 difference_in_sigma=(float(prov["map_mean_pc_cm3"]) - mean_rays) / err,
                                 map_max_pc_cm3=float(prov["map_max_pc_cm3"]), halos_selected=prov["halos_selected"]))
    write_rows(OUT / "analysis/map_mean_vs_100k_rays.csv", rows)
    for r in rows:
        print("{model:28s} {survey:6s} map mean {map_mean_pc_cm3:8.3f}  rays {rays_weighted_mean_pc_cm3:8.3f} +- "
              "{rays_standard_error_pc_cm3:.3f}  ({difference_in_sigma:+.2f} sigma)".format(**r))


def observations(plane):
    """Real Takahashi+25 w_yDM(theta) measurement and jackknife covariance (Eq. 27's first term
    minus its second term), replacing the earlier plot-digitized Fig. 13 approximation. See
    prepare_takahashi25_real_observations.py; author bin 0 (theta ~ 0.75', below this figure's
    1 arcmin lower limit) is dropped -- its marker would be off-axis anyway, but its (large,
    noisy) error bar was otherwise dragging the y-autoscale down (e.g. ACT bin 0 has sigma
    ~1.7x |w|, pulling the axis to w - sigma ~ -13e-5 with nothing visibly plotted there)."""
    with np.load(str(TAKAHASHI25_REAL / (TAKAHASHI25_SURVEY_KEY[plane] + ".npz"))) as obs:
        sigma = obs["sigma_pc_cm3"][1:]
        return dict(x=obs["theta_mean_arcmin"][1:], w=obs["w_yDM_pc_cm3"][1:], err=np.vstack([sigma, sigma]),
                    lo=obs["theta_lo_arcmin"][1:], hi=obs["theta_hi_arcmin"][1:],
                    covariance=obs["covariance_pc2_cm6"][1:, 1:])


def load_spectra():
    with np.load(str(SPECTRA), allow_pickle=False) as f:
        data = {k: f[k] for k in f.files}
    meta = json.loads(str(data.pop("metadata_json").item()))
    return data, meta


def plot_takahashi(data, meta):
    plt.rcParams.update(RC)
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 7.2))
    theta = np.geomspace(1, 1000, 500)
    rows = []
    for ax, plane in zip(axes, SURVEYS):
        name, beam_label, count = PLANES[plane]
        beam = 10.0 if plane == "planck" else 1.6
        obs = observations(plane)
        ax.errorbar(obs["x"], obs["w"] / 1e-5, yerr=obs["err"] / 1e-5, fmt="o", color="black", ms=6, capsize=2.5, lw=1.3,
                    label="Takahashi+25", zorder=6)
        for pair, (ykey, model, legend, color, ls, marker) in PAIRS.items():
            cl = data["cl_y_{}__dm_{}_{}".format(ykey, model, plane)]
            smooth = angular_correlation(cl, theta, beam)
            binned = annular_correlation(cl, obs["lo"], obs["hi"], beam)
            ax.plot(theta, smooth / 1e-5, color=color, ls=ls, lw=2.8 if pair != "lee22_calib" else 2.4, label=legend, zorder=4)
            ax.plot(np.sqrt(obs["lo"] * obs["hi"]), binned / 1e-5, marker=marker, ls="none", color=color, ms=6.5,
                    mfc="white" if pair == "lee22_calib" else color, mew=1.6, zorder=5)
            sigma = 0.5 * (obs["err"][0] + obs["err"][1])
            for i in range(len(binned)):
                rows.append(dict(survey=plane, pair=pair, theta_lower_arcmin=obs["lo"][i], theta_upper_arcmin=obs["hi"][i],
                                 theta_arcmin=np.sqrt(obs["lo"][i] * obs["hi"][i]), fullsky_w_pc_cm3=binned[i],
                                 observed_w_pc_cm3=obs["w"][i], observed_sigma_pc_cm3=sigma[i],
                                 model_minus_observed_in_sigma=(binned[i] - obs["w"][i]) / sigma[i],
                                 below_paper_angular_cut=bool(obs["lo"][i] < PAPER_CUT[plane])))
        ax.axvspan(1, PAPER_CUT[plane], color=".5", alpha=.12, zorder=0)
        style_axis(ax, 1)
        ax.set_title("{}: {}, {} FRB redshifts".format(name, beam_label, count), pad=10)
    axes[0].set_ylabel(r"$w_{y\,\mathrm{DM}}(\theta)\ \ [10^{-5}\ \mathrm{pc\,cm^{-3}}]$")
    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    order = [PAIRS[p][2] for p in PAIRS] + ["Takahashi+25"]
    fig.legend([by_label[l] for l in order], order, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.0),
               columnspacing=2.0, handlelength=2.8)
    fig.text(.5, .845, "Full-sky halo-only prediction, gas inside $R_{200c}$;  markers: annulus means", ha="center", va="bottom",
             fontsize=14, color=".25")
    fig.subplots_adjust(left=.07, right=.985, top=.765, bottom=.125, wspace=.2)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(str(OUT / "plots" / ("fullsky_takahashi_fig13." + ext)), dpi=200 if ext == "png" else None)
    plt.close(fig)
    write_rows(OUT / "analysis/fullsky_takahashi_annuli.csv", rows)
    return rows


def log_bin(ell, cl, lmin=30, per_decade=12):
    edges = np.unique(np.round(np.logspace(np.log10(lmin), np.log10(ell[-1] + 1), int(per_decade * np.log10((ell[-1] + 1) / lmin)) + 1)).astype(int))
    dl = ell * (ell + 1) * cl / (2 * np.pi)
    centres, values = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (ell >= lo) & (ell < hi)
        if not np.any(sel):
            continue
        w = 2.0 * ell[sel] + 1.0
        centres.append(np.exp(np.sum(w * np.log(ell[sel])) / np.sum(w)))
        values.append(np.sum(w * dl[sel]) / np.sum(w))
    return np.array(centres), np.array(values)


def plot_spectra(data, meta):
    plt.rcParams.update(RC)
    ell = data["ell"].astype(float)
    nside = 4096
    fig, axes = plt.subplots(1, 3, figsize=(21, 7.0))
    rows = []
    for pair, (ykey, model, legend, color, _, _) in PAIRS.items():
        lc, dl = log_bin(ell, data["cl_y_" + ykey])
        axes[0].plot(lc, dl, color=color, ls="-", lw=2.6, label=Y_MAPS[ykey][1])
        for i in range(len(lc)):
            rows.append(dict(pair=pair, spectrum="yy", kernel="", ell=lc[i], D_ell=dl[i]))
        for survey in SURVEYS:
            ls, _ = KERNEL_STYLE[survey]
            lc, dl = log_bin(ell, data["cl_dm_{}_{}".format(model, survey)])
            axes[1].plot(lc, dl, color=color, ls=ls, lw=2.6)
            for i in range(len(lc)):
                rows.append(dict(pair=pair, spectrum="DMDM", kernel=survey, ell=lc[i], D_ell=dl[i]))
            lc, dl = log_bin(ell, data["cl_y_{}__dm_{}_{}".format(ykey, model, survey)])
            axes[2].plot(lc, dl, color=color, ls=ls, lw=2.6)
            for i in range(len(lc)):
                rows.append(dict(pair=pair, spectrum="yDM", kernel=survey, ell=lc[i], D_ell=dl[i]))
    titles = [r"tSZ auto:  $\ell(\ell+1)C_\ell^{yy}/2\pi$",
              r"FRB DM auto:  $\ell(\ell+1)C_\ell^{\rm DM\,DM}/2\pi$  [pc$^2$ cm$^{-6}$]",
              r"tSZ × FRB DM:  $\ell(\ell+1)C_\ell^{y\,\rm DM}/2\pi$  [pc cm$^{-3}$]"]
    for ax, title in zip(axes, titles):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(30, ell[-1])
        ax.axvspan(nside, ell[-1], color=".5", alpha=.1, zorder=0)
        ax.set_xlabel(r"multipole $\ell$")
        ax.set_title(title, pad=10, fontsize=15)
        ax.grid(alpha=.15, which="both")
        ax.tick_params(direction="in", which="both", top=True, right=True, length=5)
    model_handles = [Line2D([], [], color=PAIRS[p][3], lw=2.6) for p in PAIRS]
    model_labels = ["Battaglia12 pressure, Battaglia16 density", "Lee22 pressure, Lee22 density",
                    "Lee22 pressure and density, calibrated range only"]
    kernel_handles = [Line2D([], [], color=".25", ls=KERNEL_STYLE[s][0], lw=2.4) for s in SURVEYS]
    kernel_labels = ["DM map weighted with the " + KERNEL_STYLE[s][1] for s in SURVEYS]
    axes[0].legend(model_handles, model_labels, loc="lower left", frameon=False, fontsize=12.5)
    axes[1].legend(kernel_handles, kernel_labels, loc="lower left", frameon=False, fontsize=12.5)
    fig.suptitle("Full-sky HalfDome maps, NSIDE 4096, no beam, monopoles removed;  shaded: $\\ell > N_{\\rm side}$",
                 fontsize=15, y=.99)
    fig.subplots_adjust(left=.05, right=.99, top=.86, bottom=.12, wspace=.24)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(str(OUT / "plots" / ("fullsky_power_spectra_three_column." + ext)), dpi=200 if ext == "png" else None)
    plt.close(fig)
    write_rows(OUT / "analysis/fullsky_binned_spectra.csv", rows)


def summary(data, meta, annuli):
    ell = data["ell"].astype(float)
    out = {"maps": {k: {"sha256": v["sha256"], "mean": v["mean"]} for k, v in meta["maps"].items()}, "annuli": {}, "spectra_ratios": {}}
    for plane in SURVEYS:
        for pair in PAIRS:
            sel = [r for r in annuli if r["survey"] == plane and r["pair"] == pair and not r["below_paper_angular_cut"]]
            out["annuli"]["{}_{}".format(pair, plane)] = {
                "theta_arcmin": [round(r["theta_arcmin"], 2) for r in sel],
                "w_1e-5": [round(r["fullsky_w_pc_cm3"] / 1e-5, 3) for r in sel],
                "model_minus_obs_sigma": [round(r["model_minus_observed_in_sigma"], 2) for r in sel]}
    for l0 in (100, 300, 1000, 3000):
        i = int(np.argmin(np.abs(ell - l0)))
        sl = slice(max(i - 25, 2), i + 25)
        band = lambda key: float(np.mean(data[key][sl]))
        out["spectra_ratios"]["ell~{}".format(l0)] = {
            "yy_lee22_over_b12": band("cl_y_lee22p") / band("cl_y_b12"),
            "yy_lee22calib_over_lee22": band("cl_y_lee22p_calib") / band("cl_y_lee22p"),
            "dd_lee22_over_b16_planck": band("cl_dm_lee22_noconc_sphere1_planck") / band("cl_dm_b16_sphere1_planck"),
            "dd_lee22calib_over_lee22_planck": band("cl_dm_lee22_noconc_sphere1_calib_planck") / band("cl_dm_lee22_noconc_sphere1_planck"),
            "yd_lee22_over_b_planck": band("cl_y_lee22p__dm_lee22_noconc_sphere1_planck") / band("cl_y_b12__dm_b16_sphere1_planck"),
            "yd_lee22calib_over_lee22_planck": band("cl_y_lee22p_calib__dm_lee22_noconc_sphere1_calib_planck") / band("cl_y_lee22p__dm_lee22_noconc_sphere1_planck"),
            "yd_act_over_planck_kernel_b": band("cl_y_b12__dm_b16_sphere1_act") / band("cl_y_b12__dm_b16_sphere1_planck")}
    (OUT / "analysis/fullsky_summary.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out["spectra_ratios"], indent=1))




# ---------------------------------------------------------------------------------------------
# Versions with a percentage-difference panel (-100 to 100 %) and the survey beams made explicit
# ---------------------------------------------------------------------------------------------
def residual_axis(ax, ylabel):
    ax.axhline(0, color=".4", lw=.9, zorder=2)
    ax.set_ylim(-100, 100)
    ax.set_yticks([-100, -50, 0, 50, 100])
    ax.set_ylabel(ylabel)
    ax.grid(alpha=.15, which="both")
    ax.tick_params(direction="in", which="both", top=True, right=True, length=5)


def draw_percent(ax, x, percent, color, marker, ls="-", lw=1.8, label=None, ms=6.5, mfc=None, annotate=True, zorder=4, alpha=1.0,
                 text_slot=0):
    """Percentage differences inside [-100, 100]; a value outside is drawn as an open triangle at the
    edge (with the value written next to it when annotate is True) so that nothing is silently lost."""
    x = np.asarray(x, dtype=float)
    percent = np.asarray(percent, dtype=float)
    inside = np.abs(percent) <= 100
    shown = np.where(inside, percent, np.nan)
    ax.plot(x, shown, color=color, ls=ls, lw=lw, marker=marker, ms=ms, mfc=mfc or color, mew=1.3, label=label, zorder=zorder, alpha=alpha)
    for xi, p in zip(x[~inside], percent[~inside]):
        edge = 92.0 if p > 0 else -92.0
        ax.plot([xi], [edge], marker="^" if p > 0 else "v", ms=8, color=color, mfc="white", mew=1.5, ls="none", zorder=zorder + 1, alpha=alpha)
        if annotate:
            # one text row per curve (text_slot) so that several clipped values at the same angle stay readable
            offset = -15 - 11 * text_slot if p > 0 else 7 + 11 * text_slot
            ax.annotate("{:+.0f}".format(p), (xi, edge), textcoords="offset points", xytext=(0, offset), ha="center",
                        fontsize=9.5, color=color, zorder=zorder + 1)


def plot_takahashi_residuals(data, meta):
    """Figure 1 with a bottom panel: (model - observed) / |observed| in percent, per annulus; grey band = observed 1 sigma."""
    plt.rcParams.update(RC)
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 9.8), sharex="col", gridspec_kw={"height_ratios": [3, 1.5], "hspace": .06, "wspace": .2})
    theta = np.geomspace(1, 1000, 500)
    for j, plane in enumerate(SURVEYS):
        name, beam_label, count = PLANES[plane]
        beam = BEAMS[plane]
        obs = observations(plane)
        x = np.sqrt(obs["lo"] * obs["hi"])
        sigma = obs["err"].mean(axis=0)
        top, bot = axes[0, j], axes[1, j]
        top.errorbar(obs["x"], obs["w"] / 1e-5, yerr=obs["err"] / 1e-5, fmt="o", color="black", ms=6, capsize=2.5, lw=1.3,
                     label="Takahashi+25", zorder=6)
        band = np.minimum(100 * sigma / np.abs(obs["w"]), 100)
        bot.fill_between(x, -band, band, color=".7", alpha=.35, lw=0, label="observed ±1σ", zorder=1)
        for k, (pair, (ykey, model, legend, color, ls, marker)) in enumerate(PAIRS.items()):
            cl = data["cl_y_{}__dm_{}_{}".format(ykey, model, plane)]
            smooth = angular_correlation(cl, theta, beam)
            binned = annular_correlation(cl, obs["lo"], obs["hi"], beam)
            calib = pair == "lee22_calib"
            top.plot(theta, smooth / 1e-5, color=color, ls=ls, lw=2.8 if not calib else 2.4, label=legend, zorder=4)
            top.plot(x, binned / 1e-5, marker=marker, ls="none", color=color, ms=6.5, mfc="white" if calib else color, mew=1.6, zorder=5)
            draw_percent(bot, x, 100 * (binned - obs["w"]) / np.abs(obs["w"]), color, marker, ls=ls, lw=1.6,
                         mfc="white" if calib else None, text_slot=k)
        for ax in (top, bot):
            ax.axvspan(1, PAPER_CUT[plane], color=".5", alpha=.12, zorder=0)
        style_axis(top, 1)
        top.set_xlabel("")
        top.set_title("{}: {:g}′ Gaussian beam on y, {} FRB redshifts".format(name, beam, count), pad=10)
        residual_axis(bot, "model − observed\n[% of |observed|]" if j == 0 else "")
        bot.set_xscale("log")
        bot.set_xlim(1, 1000)
        bot.set_xlabel(r"$\theta$ [arcmin]")
    axes[0, 0].set_ylabel(r"$w_{y\,\mathrm{DM}}(\theta)\ \ [10^{-5}\ \mathrm{pc\,cm^{-3}}]$")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    h2, l2 = axes[1, 0].get_legend_handles_labels()
    by_label = dict(zip(labels + l2, handles + h2))
    order = [PAIRS[p][2] for p in PAIRS] + ["Takahashi+25", "observed ±1σ"]
    fig.legend([by_label[l] for l in order], order, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0),
               columnspacing=2.0, handlelength=2.8)
    fig.text(.5, .875, "Full-sky halo-only prediction, gas inside $R_{200c}$;  markers: annulus means;  "
             "triangles at the panel edge: values beyond ±100 %", ha="center", va="bottom", fontsize=13.5, color=".25")
    fig.subplots_adjust(left=.075, right=.985, top=.815, bottom=.085)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(str(OUT / "plots" / ("fullsky_takahashi_fig13_residuals." + ext)), dpi=200 if ext == "png" else None)
    plt.close(fig)


def plot_spectra_beamed(data, meta):
    """Three columns with the survey beams on the tSZ side (Planck 10′ with the Planck kernel, ACT 1.6′ with
    the ACT kernel); bottom panels: percentage difference relative to the Lee22 x Lee22 all-halo pair, and
    the beam suppression alone (grey)."""
    plt.rcParams.update(RC)
    ell = data["ell"].astype(float)
    nside = 4096
    beam = {s: gaussian_beam(ell, BEAMS[s]) for s in SURVEYS}
    ref_y, ref_dm = PAIRS["lee22"][0], PAIRS["lee22"][1]
    fig, axes = plt.subplots(2, 3, figsize=(21, 9.6), sharex="col", gridspec_kw={"height_ratios": [3, 1.6], "hspace": .06, "wspace": .24})
    rows = []
    for pair, (ykey, model, legend, color, _, _) in PAIRS.items():
        for survey in SURVEYS:
            ls = KERNEL_STYLE[survey][0]
            spectra = {
                "yy": (data["cl_y_" + ykey] * beam[survey] ** 2, data["cl_y_" + ref_y] * beam[survey] ** 2),
                "DMDM": (data["cl_dm_{}_{}".format(model, survey)], data["cl_dm_{}_{}".format(ref_dm, survey)]),
                "yDM": (data["cl_y_{}__dm_{}_{}".format(ykey, model, survey)] * beam[survey],
                        data["cl_y_{}__dm_{}_{}".format(ref_y, ref_dm, survey)] * beam[survey]),
            }
            for col, key in enumerate(("yy", "DMDM", "yDM")):
                lc, dl = log_bin(ell, spectra[key][0])
                _, dref = log_bin(ell, spectra[key][1])
                axes[0, col].plot(lc, dl, color=color, ls=ls, lw=2.5)
                if pair != "lee22":
                    axes[1, col].plot(lc, 100 * (dl / dref - 1), color=color, ls=ls, lw=2.2)
                for i in range(len(lc)):
                    rows.append(dict(pair=pair, spectrum=key, survey=survey, beam_fwhm_arcmin=BEAMS[survey] if key != "DMDM" else 0.0,
                                     ell=lc[i], D_ell=dl[i], percent_vs_lee22_all=100 * (dl[i] / dref[i] - 1)))
    for survey in SURVEYS:
        ls = KERNEL_STYLE[survey][0]
        keep = ell >= 30
        axes[1, 0].plot(ell[keep], 100 * (beam[survey][keep] ** 2 - 1), color=".55", ls=ls, lw=1.3, zorder=1)
        axes[1, 2].plot(ell[keep], 100 * (beam[survey][keep] - 1), color=".55", ls=ls, lw=1.3, zorder=1)
    titles = [r"tSZ auto with beam:  $\ell(\ell+1)C_\ell^{yy}B_\ell^2/2\pi$",
              r"FRB DM auto, no beam:  $\ell(\ell+1)C_\ell^{\rm DM\,DM}/2\pi$  [pc$^2$ cm$^{-6}$]",
              r"tSZ × FRB DM with beam:  $\ell(\ell+1)C_\ell^{y\,\rm DM}B_\ell/2\pi$  [pc cm$^{-3}$]"]
    for col, title in enumerate(titles):
        top, bot = axes[0, col], axes[1, col]
        top.set_yscale("log")
        if col != 1:  # the beams suppress the high-l tSZ power by many decades; show five
            peak = max(np.nanmax(line.get_ydata()) for line in top.get_lines())
            top.set_ylim(peak * 10 ** -5.2, peak * 3)
        top.set_title(title, pad=10, fontsize=14.5)
        for ax in (top, bot):
            ax.set_xscale("log")
            ax.set_xlim(30, ell[-1])
            ax.axvspan(nside, ell[-1], color=".5", alpha=.1, zorder=0)
            ax.grid(alpha=.15, which="both")
            ax.tick_params(direction="in", which="both", top=True, right=True, length=5)
        residual_axis(bot, "relative to Lee22 × Lee22,\nall halos [%]" if col == 0 else "")
        bot.set_xlabel(r"multipole $\ell$")
    model_handles = [Line2D([], [], color=PAIRS[p][3], lw=2.5) for p in PAIRS]
    model_labels = ["Battaglia12 pressure, Battaglia16 density", "Lee22 pressure, Lee22 density",
                    "Lee22 pressure and density, calibrated range only"]
    survey_handles = [Line2D([], [], color=".25", ls=KERNEL_STYLE[s][0], lw=2.4) for s in SURVEYS]
    survey_labels = [SURVEY_LABEL[s] for s in SURVEYS]
    axes[0, 0].legend(model_handles, model_labels, loc="lower left", frameon=False, fontsize=12)
    axes[0, 1].legend(survey_handles, survey_labels, loc="lower left", frameon=False, fontsize=12)
    axes[1, 0].legend([Line2D([], [], color=".55", lw=1.3)], ["beam alone: $B_\\ell^2 - 1$ (left), $B_\\ell - 1$ (right)"],
                      loc="lower left", frameon=False, fontsize=11)
    fig.suptitle("Full-sky HalfDome maps, NSIDE 4096, monopoles removed; Gaussian beams on y only;  shaded: $\\ell > N_{\\rm side}$",
                 fontsize=15, y=.985)
    fig.subplots_adjust(left=.055, right=.99, top=.9, bottom=.085)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(str(OUT / "plots" / ("fullsky_power_spectra_three_column_beamed." + ext)), dpi=200 if ext == "png" else None)
    plt.close(fig)
    write_rows(OUT / "analysis/fullsky_binned_spectra_beamed.csv", rows)


def plot(args):
    (OUT / "plots").mkdir(exist_ok=True)
    data, meta = load_spectra()
    annuli = plot_takahashi(data, meta)
    plot_spectra(data, meta)
    plot_takahashi_residuals(data, meta)
    plot_spectra_beamed(data, meta)
    summary(data, meta, annuli)
    print("Saved figures to " + str(OUT / "plots"))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=("spectra", "check", "plot"))
    parser.add_argument("--lmax", type=int, default=8192)
    parser.add_argument("--iter", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    {"spectra": spectra, "check": check, "plot": plot}[args.stage](args)


if __name__ == "__main__":
    main()
