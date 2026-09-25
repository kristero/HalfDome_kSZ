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
import medlock_bp_spectra as medlock

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


MEDLOCK_LS = (0, (6, 2, 1.5, 2))
MEDLOCK_MARKER = "h"
_MEDLOCK_CL = {}
_MEDLOCK_CACHE = {}


def medlock_curve_name(name):
    """Readable Medlock curve name for tables: the "__1" fiducial as "fiducial", others "<param>x<mult>"."""
    return "fiducial" if name == medlock.FIDUCIAL else name.replace("__", "x")


def medlock_prediction(plane, theta, obs):
    """Medlock BP fiducial and one-at-a-time variations (medlock_bp_spectra.py), with the survey beam on y and the
    same Legendre estimator as the HalfDome pairs -- but not the same source kernel (the BP spectra are taken to have
    all FRBs at z = 2) nor the same halo boundary/mass range/cosmology: smooth w(theta), the min-max envelope over
    every curve, and exact annulus means in the observed bins. Cached per (plane, beam, theta grid, bins)."""
    if "cl" not in _MEDLOCK_CL:
        _MEDLOCK_CL["cl"] = medlock.all_cl()[0]
    beam = BEAMS[plane]
    key = (plane, beam, np.asarray(theta).tobytes(), np.asarray(obs["lo"]).tobytes(), np.asarray(obs["hi"]).tobytes())
    if key not in _MEDLOCK_CACHE:
        cls = _MEDLOCK_CL["cl"]
        smooth_all = {n: angular_correlation(cl, theta, beam) for n, cl in cls.items()}
        smooth = np.array(list(smooth_all.values()))
        binned = {n: annular_correlation(cl, obs["lo"], obs["hi"], beam) for n, cl in cls.items()}
        _MEDLOCK_CACHE[key] = dict(smooth=smooth_all[medlock.FIDUCIAL], smooth_all=smooth_all,
                                   band=(smooth.min(axis=0), smooth.max(axis=0)),
                                   binned=binned[medlock.FIDUCIAL], binned_all=binned)
    return _MEDLOCK_CACHE[key]


def draw_medlock(ax, theta, x, pred):
    ax.fill_between(theta, pred["band"][0] / 1e-5, pred["band"][1] / 1e-5, color=medlock.COLOR, alpha=.16, lw=0,
                    label=medlock.BAND_LABEL, zorder=2)
    ax.plot(theta, pred["smooth"] / 1e-5, color=medlock.COLOR, ls=MEDLOCK_LS, lw=2.6, label=medlock.LABEL, zorder=4)
    ax.plot(x, pred["binned"] / 1e-5, marker=MEDLOCK_MARKER, ls="none", color=medlock.COLOR, ms=7.5, mew=1.4, zorder=5)


def chi_square_table(data):
    """Generalized chi^2 (real Takahashi+25 jackknife covariance inverted directly -- no Hartlap-type debiasing, the
    number of jackknife regions is not in the delivered files -- bins above the paper's angular cut) for w = 0, the
    three HalfDome pairs, the Medlock BP fiducial and every Medlock variation. Model-to-model differences are the
    meaningful quantity; absolute PTEs would need the debiasing factor."""
    theta = np.geomspace(1, 1000, 500)
    rows = []
    for plane in SURVEYS:
        obs = observations(plane)
        use = obs["lo"] >= PAPER_CUT[plane] - 1e-9
        inv = np.linalg.inv(obs["covariance"][np.ix_(use, use)])
        chi2 = lambda model: float((model[use] - obs["w"][use]) @ inv @ (model[use] - obs["w"][use]))
        rows.append(dict(survey=plane, model="null_w0", legend="w = 0", chi2=chi2(np.zeros_like(obs["w"])),
                         n_bins=int(use.sum())))
        for pair, (ykey, model, legend, *_) in PAIRS.items():
            binned = annular_correlation(data["cl_y_{}__dm_{}_{}".format(ykey, model, plane)], obs["lo"], obs["hi"], BEAMS[plane])
            rows.append(dict(survey=plane, model=pair, legend=legend, chi2=chi2(binned), n_bins=int(use.sum())))
        for name, binned in medlock_prediction(plane, theta, obs)["binned_all"].items():
            rows.append(dict(survey=plane, model="medlock_bp_" + medlock_curve_name(name),
                             legend="Medlock BP " + medlock_curve_name(name), chi2=chi2(binned), n_bins=int(use.sum())))
    write_rows(OUT / "analysis/takahashi25_chi2_halfdome_and_medlock.csv", rows)
    return rows


def load_spectra():
    with np.load(str(SPECTRA), allow_pickle=False) as f:
        data = {k: f[k] for k in f.files}
    meta = json.loads(str(data.pop("metadata_json").item()))
    return data, meta


def plot_takahashi(data, meta):
    plt.rcParams.update(RC)
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 7.8))
    theta = np.geomspace(1, 1000, 500)
    rows, medlock_rows = [], []
    for ax, plane in zip(axes, SURVEYS):
        name, beam_label, count = PLANES[plane]
        beam = BEAMS[plane]
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
        x = np.sqrt(obs["lo"] * obs["hi"])
        pred = medlock_prediction(plane, theta, obs)
        draw_medlock(ax, theta, x, pred)
        sigma = 0.5 * (obs["err"][0] + obs["err"][1])
        for i in range(len(x)):
            medlock_rows.append(dict(survey=plane, theta_lower_arcmin=obs["lo"][i], theta_upper_arcmin=obs["hi"][i],
                                     theta_arcmin=x[i], observed_w_pc_cm3=obs["w"][i], observed_sigma_pc_cm3=sigma[i],
                                     below_paper_angular_cut=bool(obs["lo"][i] < PAPER_CUT[plane]),
                                     **{"medlock_" + medlock_curve_name(n) + "_w_pc_cm3": v[i] for n, v in pred["binned_all"].items()}))
        ax.axvspan(1, PAPER_CUT[plane], color=".5", alpha=.12, zorder=0)
        style_axis(ax, 1)
        ax.set_title("{}: {}, {} FRBs".format(name, beam_label, count), pad=10)
    axes[0].set_ylabel(r"$w_{y\,\mathrm{DM}}(\theta)\ \ [10^{-5}\ \mathrm{pc\,cm^{-3}}]$")
    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    order = [PAIRS[p][2] for p in PAIRS] + [medlock.LABEL, medlock.BAND_LABEL, "Takahashi+25"]
    fig.legend([by_label[l] for l in order], order, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.0),
               columnspacing=3.0, handlelength=2.8)
    fig.text(.5, .84, "HalfDome: halo-only, gas inside $R_{200c}$, observed FRB redshifts;  Medlock: BP halo model, all FRBs at "
             "$z = 2$ (assumed);  markers: annulus means", ha="center", va="bottom", fontsize=13.5, color=".25")
    fig.subplots_adjust(left=.07, right=.985, top=.77, bottom=.115, wspace=.2)
    write_rows(OUT / "analysis/medlock_bp_takahashi_annuli.csv", medlock_rows)
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
                 text_slot=0, edge_shared=False):
    """Percentage differences inside [-100, 100]; a value outside is drawn as an open triangle at the
    edge (with the value written next to it when annotate is True) so that nothing is silently lost.
    text_slot: annotation row, per curve or per point; edge_shared: per point, another curve's triangle sits at the
    same spot, so a neutral triangle is drawn and the coloured text carries the identity. Non-finite values are skipped."""
    x = np.asarray(x, dtype=float)
    percent = np.asarray(percent, dtype=float)
    finite = np.isfinite(percent)
    inside = finite & (np.abs(percent) <= 100)
    shown = np.where(inside, percent, np.nan)
    slots = np.broadcast_to(np.asarray(text_slot), x.shape)
    shared = np.broadcast_to(np.asarray(edge_shared, dtype=bool), x.shape)
    ax.plot(x, shown, color=color, ls=ls, lw=lw, marker=marker, ms=ms, mfc=mfc or color, mew=1.3, label=label, zorder=zorder,
            alpha=alpha, clip_on=False)
    out = finite & ~inside
    for xi, p, slot, common in zip(x[out], percent[out], slots[out], shared[out]):
        edge = 92.0 if p > 0 else -92.0
        ax.plot([xi], [edge], marker="^" if p > 0 else "v", ms=8, color=".3" if common else color, mfc="white", mew=1.5,
                ls="none", zorder=zorder + 1, alpha=alpha)
        if annotate:
            # one text row per clipped curve so that several clipped values at the same angle stay readable
            offset = -14 - 10 * slot if p > 0 else 6 + 10 * slot
            text = "{:+.0f}".format(p) if abs(round(p)) > 100 else "{:+.1f}".format(p)
            ax.annotate(text, (xi, edge), textcoords="offset points", xytext=(0, offset), ha="center",
                        fontsize=9, color=color, zorder=zorder + 1)


MEDLOCK_PARAMS = (("epsilon", r"$\epsilon$"), ("fstar", r"$f_\star$"), ("Sstar", r"$S_\star$"),
                  ("A_nt", r"$A_{\rm nt}$"), ("B_nt", r"$B_{\rm nt}$"), ("gamma_nt", r"$\gamma_{\rm nt}$"))


def plot_medlock_variations(data, meta):
    """Small multiples: each Medlock BP gas parameter varied alone (x multiplier, all others fiducial), with the
    survey beam on y, against the real Takahashi+25 points. Rows: Planck, ACT; light to dark = low to high."""
    plt.rcParams.update(RC)
    theta = np.geomspace(1, 1000, 500)
    fig, axes = plt.subplots(2, len(MEDLOCK_PARAMS), figsize=(27, 10.5), sharex=True, sharey="row")
    cmap = plt.get_cmap("Purples")
    for j, plane in enumerate(SURVEYS):
        name, _, count = PLANES[plane]
        obs = observations(plane)
        pred = medlock_prediction(plane, theta, obs)
        for i, (param, symbol) in enumerate(MEDLOCK_PARAMS):
            ax = axes[j, i]
            variants = {float(n.split("__")[1]): n for n in pred["smooth_all"] if n.split("__")[0] == param}
            variants[1.0] = medlock.FIDUCIAL
            mults = sorted(variants)
            for m in mults:
                fiducial = m == 1.0
                # shade follows the multiplier itself (x0.25 ... x4 on one log scale), not its rank in this panel
                shade = cmap(0.55 + 0.4 * (np.log(m) - np.log(0.25)) / (np.log(4) - np.log(0.25)))
                ax.plot(theta, pred["smooth_all"][variants[m]] / 1e-5, color=medlock.COLOR if fiducial else shade,
                        lw=3.2 if fiducial else 2.2, ls=MEDLOCK_LS if fiducial else "-",
                        label="×1 (fiducial)" if fiducial else "×{:g}".format(m), zorder=4 if fiducial else 3)
            ax.errorbar(obs["x"], obs["w"] / 1e-5, yerr=obs["err"] / 1e-5, fmt="o", color="black", ms=5.5, capsize=2,
                        lw=1.2, zorder=6)
            ax.axvspan(1, PAPER_CUT[plane], color=".5", alpha=.12, zorder=0)
            style_axis(ax, 1)
            if j == 0:
                ax.set_title(symbol, fontsize=22, pad=8)
                ax.legend(loc="upper right", fontsize=12, frameon=False, handlelength=3.2, labelspacing=.3)
                ax.set_xlabel("")
            if i == 0:
                ax.set_ylabel("{} data, {} FRBs\n".format(name, count) + r"$w_{y\,\mathrm{DM}}\ [10^{-5}\ \mathrm{pc\,cm^{-3}}]$")
    fig.suptitle("Medlock BP halo model, all FRBs at $z = 2$ (assumed): one gas parameter varied at a time;  "
                 "black: Takahashi+25;  survey beam on y", fontsize=17, y=.995)
    fig.subplots_adjust(left=.05, right=.99, top=.9, bottom=.08, wspace=.15, hspace=.12)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(str(OUT / "plots" / ("medlock_bp_parameter_variations_takahashi." + ext)), dpi=200 if ext == "png" else None)
    plt.close(fig)


def shared_edges(percents):
    """True where another curve is clipped at the same angle with the same sign (its edge triangle would sit on top)."""
    percents = np.asarray(percents, dtype=float)
    shared = np.zeros(percents.shape, dtype=bool)
    for sign in (1, -1):
        clipped = sign * percents > 100
        shared |= clipped & (clipped.sum(axis=0) > 1)
    return shared


def compact_text_slots(percents):
    """Per-point annotation rows for several curves in one percentage panel: at each angle only the
    curves clipped beyond +-100 % (separately for each sign) get consecutive rows, in curve order."""
    percents = np.asarray(percents, dtype=float)
    slots = np.zeros(percents.shape, dtype=int)
    for sign in (1, -1):
        clipped = sign * percents > 100
        slots = np.where(clipped, np.cumsum(clipped, axis=0) - 1, slots)
    return slots


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
        percents = []   # (percent, colour, marker, linestyle, marker face)
        for pair, (ykey, model, legend, color, ls, marker) in PAIRS.items():
            cl = data["cl_y_{}__dm_{}_{}".format(ykey, model, plane)]
            smooth = angular_correlation(cl, theta, beam)
            binned = annular_correlation(cl, obs["lo"], obs["hi"], beam)
            calib = pair == "lee22_calib"
            top.plot(theta, smooth / 1e-5, color=color, ls=ls, lw=2.8 if not calib else 2.4, label=legend, zorder=4)
            top.plot(x, binned / 1e-5, marker=marker, ls="none", color=color, ms=6.5, mfc="white" if calib else color, mew=1.6, zorder=5)
            percents.append((100 * (binned - obs["w"]) / np.abs(obs["w"]), color, marker, ls, "white" if calib else None))
        pred = medlock_prediction(plane, theta, obs)
        draw_medlock(top, theta, x, pred)
        percents.append((100 * (pred["binned"] - obs["w"]) / np.abs(obs["w"]), medlock.COLOR, MEDLOCK_MARKER, MEDLOCK_LS, None))
        stack = [p[0] for p in percents]
        for (percent, color, marker, ls, mfc), slots, shared in zip(percents, compact_text_slots(stack), shared_edges(stack)):
            draw_percent(bot, x, percent, color, marker, ls=ls, lw=1.6, mfc=mfc, text_slot=slots, edge_shared=shared)
        for ax in (top, bot):
            ax.axvspan(1, PAPER_CUT[plane], color=".5", alpha=.12, zorder=0)
        style_axis(top, 1)
        top.set_xlabel("")
        top.set_title("{}: {:g}′ Gaussian beam on y, {} FRBs".format(name, beam, count), pad=10)
        residual_axis(bot, "model − observed\n[% of |observed|]" if j == 0 else "")
        bot.set_xscale("log")
        bot.set_xlim(1, 1000)
        bot.set_xlabel(r"$\theta$ [arcmin]")
    axes[0, 0].set_ylabel(r"$w_{y\,\mathrm{DM}}(\theta)\ \ [10^{-5}\ \mathrm{pc\,cm^{-3}}]$")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    h2, l2 = axes[1, 0].get_legend_handles_labels()
    by_label = dict(zip(labels + l2, handles + h2))
    order = [PAIRS[p][2] for p in PAIRS] + [medlock.LABEL, medlock.BAND_LABEL, "Takahashi+25", "observed ±1σ"]
    fig.legend([by_label[l] for l in order], order, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0),
               columnspacing=2.0, handlelength=2.8)
    fig.text(.5, .875, "HalfDome: halo-only in $R_{200c}$, observed FRB redshifts;  Medlock BP: all FRBs at $z = 2$ (assumed);  "
             "markers: annulus means;  edge triangles: beyond ±100 %", ha="center", va="bottom", fontsize=12.5, color=".25")
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
    plot_medlock_variations(data, meta)
    plot_spectra_beamed(data, meta)
    summary(data, meta, annuli)
    for r in chi_square_table(data):
        if not r["model"].startswith("medlock_bp_") or r["model"] == "medlock_bp_fiducial":
            print("chi2 {survey:6s} {model:32s} {chi2:9.2f} / {n_bins} bins".format(**r))
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
