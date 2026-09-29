#!/usr/bin/env python3
"""tSZ x FRB-DM cross-correlation w_yDM(theta) with the TNG-calibrated Lee22 density, against the real Takahashi+25
measurement (Planck MILCA, 71 FRBs, 10' beam; ACT, 31 FRBs, 1.6' beam) and the previous full-sky pairs.

Same method as fullsky_tsz_dm_comparison.py (kernel-weighted full-sky DM maps for the observed FRB redshifts, map2alm
lmax 8192, w = sum (2l+1)/4pi C_l B_l P_l with the survey Gaussian beam on y; markers = exact annulus means; chi^2 with the
real jackknife covariance over the bins above the paper's angular cut, no Hartlap factor). Spectra: previous pairs from
outputs/tsz_dm_fullsky_20260918/spectra/fullsky_spectra.npz, new pairs from outputs/lee22_tngcal_20260929/spectra/
tngcal_spectra.npz (idark, 2026-09-29). Run from the repository root."""
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import fullsky_tsz_dm_comparison as fs  # noqa: E402

OUT = Path("frb_map_generation/outputs/lee22_tngcal_20260929")
NEW = OUT / "spectra/tngcal_spectra.npz"
PAIRS = {  # key: (spectra source, y key, DM model, legend, colour, linestyle, marker)
    "battaglia": ("old", "b12", "b16_sphere1", "Battaglia12 y × Battaglia16 DM", "#0072B2", "-", "o"),
    "lee22": ("old", "lee22p", "lee22_noconc_sphere1", "Lee22 y × Lee22 DM ×0.6016 (previous)", "#E69F00", (0, (6, 3)), "s"),
    "lee22_tngcal": ("new", "lee22p", "lee22_noconc_tngcal_sphere1", "Lee22 y × Lee22 DM ×$C(z)$ (TNG-calibrated)", "#009E73", "-", "D"),
    "b12_lee22_tngcal": ("new", "b12", "lee22_noconc_tngcal_sphere1", "Battaglia12 y × Lee22 DM ×$C(z)$", "#CC79A7", "-.", "^"),
}
EXTRA = {  # tabulated only
    "lee22_calib": ("old", "lee22p_calib", "lee22_noconc_sphere1_calib", "Lee22 y × Lee22 DM ×0.6016, calibrated range"),
    "lee22_tngcal_calib": ("new", "lee22p_calib", "lee22_noconc_tngcal_sphere1_calib", "Lee22 y × Lee22 DM ×C(z), calibrated range"),
    "b16_with_lee22p": ("old", "lee22p", "b16_sphere1", "Lee22 y × Battaglia16 DM"),
}


def load_new():
    with np.load(str(NEW), allow_pickle=False) as f:
        data = {k: f[k] for k in f.files}
    meta = json.loads(str(data.pop("metadata_json").item()))
    return data, meta


def main():
    old, _ = fs.load_spectra()
    new, meta_new = load_new()
    src = {"old": old, "new": new}
    theta = np.geomspace(1, 1000, 500)
    rows, chi = [], []
    with plt.rc_context(fs.RC):
        fig, axes = plt.subplots(1, 2, figsize=(15.5, 7.6))
        for ax, plane in zip(axes, fs.SURVEYS):
            name, beam_label, count = fs.PLANES[plane]
            beam = fs.BEAMS[plane]
            obs = fs.observations(plane)
            use = obs["lo"] >= fs.PAPER_CUT[plane] - 1e-9
            inv = np.linalg.inv(obs["covariance"][np.ix_(use, use)])
            chi2 = lambda m: float((m[use] - obs["w"][use]) @ inv @ (m[use] - obs["w"][use]))
            ax.errorbar(obs["x"], obs["w"] / 1e-5, yerr=obs["err"] / 1e-5, fmt="o", color="black", ms=6, capsize=2.5, lw=1.3,
                        label="Takahashi+25", zorder=7)
            chi.append(dict(survey=plane, pair="null_w0", legend="w = 0", chi2=chi2(np.zeros_like(obs["w"])), n_bins=int(use.sum())))
            lines = ["$\\chi^2$ ({} bins):  $w=0$  {:.1f}".format(int(use.sum()), chi[-1]["chi2"])]
            for key, (s, ykey, model, legend, col, ls, mk) in list(PAIRS.items()) + [(k, v + (None, None, None)) for k, v in EXTRA.items()]:
                cl = src[s]["cl_y_{}__dm_{}_{}".format(ykey, model, plane)]
                binned = fs.annular_correlation(cl, obs["lo"], obs["hi"], beam)
                c2 = chi2(binned)
                chi.append(dict(survey=plane, pair=key, legend=legend, chi2=c2, n_bins=int(use.sum())))
                for i in range(len(binned)):
                    rows.append(dict(survey=plane, pair=key, theta_lower_arcmin=obs["lo"][i], theta_upper_arcmin=obs["hi"][i],
                                     w_pc_cm3=binned[i], observed_w_pc_cm3=obs["w"][i],
                                     below_paper_cut=bool(obs["lo"][i] < fs.PAPER_CUT[plane])))
                if key not in PAIRS:
                    continue
                smooth = fs.angular_correlation(cl, theta, beam)
                ax.plot(theta, smooth / 1e-5, color=col, ls=ls, lw=2.9 if key == "lee22_tngcal" else 2.3, label=legend, zorder=5)
                ax.plot(np.sqrt(obs["lo"] * obs["hi"]), binned / 1e-5, marker=mk, ls="none", color=col, ms=6.5, mew=1.4,
                        mfc="white" if key in ("lee22", "b12_lee22_tngcal") else col, zorder=6)
                lines.append("{:.1f}  {}".format(c2, {"battaglia": "B12 × B16", "lee22": "Lee22 × Lee22 (prev.)",
                                                       "lee22_tngcal": "Lee22 × Lee22 C(z)", "b12_lee22_tngcal": "B12 × Lee22 C(z)"}[key]))
            ax.text(.97, .97, "\n".join(lines), transform=ax.transAxes, ha="right", va="top", fontsize=11.5, color=".2",
                    bbox=dict(facecolor="white", edgecolor=".85", boxstyle="round,pad=.4"), zorder=8)
            ax.axvspan(1, fs.PAPER_CUT[plane], color=".5", alpha=.12, zorder=0)
            fs.style_axis(ax, 1)
            ax.set_title("{}: {}, {} FRBs".format(name, beam_label, count), pad=10)
        axes[0].set_ylabel(r"$w_{y\,\mathrm{DM}}(\theta)\ \ [10^{-5}\ \mathrm{pc\,cm^{-3}}]$")
        h, l = axes[0].get_legend_handles_labels()
        by = dict(zip(l, h))
        order = [v[3] for v in PAIRS.values()] + ["Takahashi+25"]
        fig.legend([by[o] for o in order], order, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(.5, 1.0), handlelength=2.8,
                   columnspacing=2.2)
        fig.text(.5, .872, "HalfDome halo-only DM (gas inside $R_{200c}$) weighted by the observed FRB redshifts;  "
                 "$C(z) = 0.284\\,(1+z)^{2.48}$ (TNG-calibrated density)\nmarkers: annulus means;  shaded: below the paper's "
                 "angular cut (excluded from $\\chi^2$);  y maps unchanged (Battaglia12 / Lee22 Table 7 pressure)",
                 ha="center", va="center", fontsize=12.2, color=".3", linespacing=1.4)
        fig.subplots_adjust(left=.07, right=.985, top=.775, bottom=.11, wspace=.2)
        (OUT / "plots").mkdir(parents=True, exist_ok=True)
        for ext in ("png", "pdf"):
            fig.savefig(str(OUT / "plots" / ("tsz_x_frbdm_lee22_tngcal_takahashi." + ext)), dpi=190 if ext == "png" else None)
        plt.close(fig)
    (OUT / "analysis").mkdir(parents=True, exist_ok=True)
    for path, data in ((OUT / "analysis/tsz_x_frbdm_lee22_tngcal_annuli.csv", rows), (OUT / "analysis/tsz_x_frbdm_lee22_tngcal_chi2.csv", chi)):
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
    for r in chi:
        print("{survey:7s} {pair:20s} chi2 = {chi2:7.2f} ({n_bins} bins)  {legend}".format(**r))
    for k, v in meta_new["maps"].items():
        if k.startswith("dm_"):
            print("{}: mean {:.3f} pc/cm3".format(k, v["mean"]))


if __name__ == "__main__":
    main()
