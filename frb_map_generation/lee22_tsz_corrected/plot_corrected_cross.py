#!/usr/bin/env python3
"""tSZ x FRB-DM w_yDM(theta) with the corrected Lee22 pressure and density maps (idark, 2026-09-30) against the real
Takahashi+25 measurement (Planck MILCA 71 FRBs, 10' beam; ACT 31 FRBs, 1.6' beam). Same method as
fullsky_tsz_dm_comparison.py (map2alm lmax 8192, w = sum (2l+1)/4pi C_l B_l P_l with the survey beam on y, exact annulus
means, chi^2 with the real jackknife covariance above the paper's angular cut, no Hartlap factor). All five y maps are the
2026-09-30 repaints with exact low-z profiles. Writes plots/tsz_x_frbdm_corrected_lee22_takahashi.{png,pdf} and the
chi^2 of every y x DM combination. Run from the repository root."""
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

OUT = Path("frb_map_generation/outputs/lee22_corrected_20260930")
Y = {"b12": "Battaglia12", "lee22p": "Lee22 as printed", "lee22p_m1": "Lee22×(1+z)²", "lee22p_m1_b12hi": "hybrid (Lee22×(1+z)², B12 above fit)",
     "lee22p_m1_calib": "Lee22×(1+z)², calibrated range"}
DM = {"b16_sphere1": "Battaglia16", "lee22_noconc_sphere1": "Lee22 ×0.6016", "lee22_noconc_sphere1_calib": "Lee22 ×0.6016, calib.",
      "lee22_noconc_tngcal_sphere1": "Lee22 ×C(z)", "lee22_noconc_tngcal_sphere1_calib": "Lee22 ×C(z), calib.",
      "lee22_noconc_a3fn_sphere1": "Lee22 ×(1+z)³/5.4", "lee22_noconc_a3fn_sphere1_calib": "Lee22 ×(1+z)³/5.4, calib.",
      "lee22_noconc_phys_sphere1": "Lee22 ×[h(1+z)]³X²", "lee22_noconc_phys_sphere1_calib": "Lee22 ×[h(1+z)]³X², calib."}
PLOT = (  # y, DM, short label, colour, linestyle, marker, open marker
    ("b12", "b16_sphere1", "Battaglia12 y × Battaglia16 DM", "#0072B2", "-", "o", False),
    ("lee22p", "lee22_noconc_sphere1", "Lee22 y × Lee22 DM ×0.6016 (previous)", "#E69F00", (0, (6, 3)), "s", True),
    ("lee22p_m1", "lee22_noconc_phys_sphere1", "Lee22×(1+z)² y × Lee22 DM ×[h(1+z)]³X²", "#009E73", "-", "D", False),
    ("lee22p_m1_b12hi", "lee22_noconc_phys_sphere1", "hybrid y × Lee22 DM ×[h(1+z)]³X²", "#CC79A7", "-.", "^", False),
    ("lee22p_m1_calib", "lee22_noconc_phys_sphere1_calib", "calibrated ranges only (both corrected)", "#56B4E9", (0, (2, 2)), "v", True),
)


def main():
    with np.load(str(OUT / "spectra/corrected_spectra.npz"), allow_pickle=False) as f:
        S = {k: f[k] for k in f.files}
    meta = json.loads(str(S.pop("metadata_json").item()))
    theta = np.geomspace(1, 1000, 500)
    chi, rows = [], []
    with plt.rc_context(fs.RC):
        fig, axes = plt.subplots(1, 2, figsize=(15.5, 7.8))
        for ax, plane in zip(axes, fs.SURVEYS):
            name, beam_label, count = fs.PLANES[plane]
            beam = fs.BEAMS[plane]
            obs = fs.observations(plane)
            use = obs["lo"] >= fs.PAPER_CUT[plane] - 1e-9
            inv = np.linalg.inv(obs["covariance"][np.ix_(use, use)])
            chi2 = lambda m: float((m[use] - obs["w"][use]) @ inv @ (m[use] - obs["w"][use]))
            ax.errorbar(obs["x"], obs["w"] / 1e-5, yerr=obs["err"] / 1e-5, fmt="o", color="black", ms=6, capsize=2.5, lw=1.3,
                        label="Takahashi+25", zorder=7)
            chi.append(dict(survey=plane, y="none", dm="none", chi2=chi2(np.zeros_like(obs["w"])), n_bins=int(use.sum())))
            for ykey in Y:
                for dkey in DM:
                    cl = S["cl_y_{}__dm_{}_{}".format(ykey, dkey, plane)]
                    binned = fs.annular_correlation(cl, obs["lo"], obs["hi"], beam)
                    chi.append(dict(survey=plane, y=ykey, dm=dkey, chi2=chi2(binned), n_bins=int(use.sum())))
                    for i in range(len(binned)):
                        rows.append(dict(survey=plane, y=ykey, dm=dkey, theta_lower_arcmin=obs["lo"][i], theta_upper_arcmin=obs["hi"][i],
                                         w_pc_cm3=binned[i], observed_w_pc_cm3=obs["w"][i], below_paper_cut=bool(obs["lo"][i] < fs.PAPER_CUT[plane])))
            lines = ["$\\chi^2$ ({} bins):  $w=0$  {:.1f}".format(int(use.sum()), chi2(np.zeros_like(obs["w"])))]
            for ykey, dkey, lab, col, ls, mk, open_ in PLOT:
                cl = S["cl_y_{}__dm_{}_{}".format(ykey, dkey, plane)]
                smooth = fs.angular_correlation(cl, theta, beam)
                binned = fs.annular_correlation(cl, obs["lo"], obs["hi"], beam)
                ax.plot(theta, smooth / 1e-5, color=col, ls=ls, lw=2.6, label=lab, zorder=5)
                ax.plot(np.sqrt(obs["lo"] * obs["hi"]), binned / 1e-5, marker=mk, ls="none", color=col, ms=6.5, mew=1.4,
                        mfc="white" if open_ else col, zorder=6)
                lines.append("{:.1f}  {}".format(chi2(binned), lab.split(" (")[0].replace("calibrated ranges: ", "calib.: ")))
            ax.text(.97, .97, "\n".join(lines), transform=ax.transAxes, ha="right", va="top", fontsize=10.8, color=".2",
                    bbox=dict(facecolor="white", edgecolor=".85", boxstyle="round,pad=.4"), zorder=8)
            ax.axvspan(1, fs.PAPER_CUT[plane], color=".5", alpha=.12, zorder=0)
            fs.style_axis(ax, 1)
            ax.set_title("{}: {}, {} FRBs".format(name, beam_label, count), pad=10)
        axes[0].set_ylabel(r"$w_{y\,\mathrm{DM}}(\theta)\ \ [10^{-5}\ \mathrm{pc\,cm^{-3}}]$")
        h, l = axes[0].get_legend_handles_labels()
        by = dict(zip(l, h))
        order = [p[2] for p in PLOT] + ["Takahashi+25"]
        fig.legend([by[o] for o in order], order, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(.5, 1.0), handlelength=2.8,
                   columnspacing=1.8, fontsize=12)
        fig.text(.5, .855, "y: Lee22 pressure ×(1+z)² (bookkeeping correction); hybrid = Battaglia12 above $10^{14.8}h^{-1}M_\\odot$;  "
                 "DM: halo gas inside $R_{200c}$ weighted by the observed FRB redshifts\ncorrected density: Lee22 × $[h_{TNG}(1+z)]^3X_H^2$ "
                 "(derived, $h_{TNG}$ = 0.6774);  markers: annulus means;  shaded: below the paper's angular cut (excluded from $\\chi^2$)",
                 ha="center", va="center", fontsize=11.8, color=".3", linespacing=1.4)
        fig.subplots_adjust(left=.07, right=.985, top=.77, bottom=.11, wspace=.2)
        (OUT / "plots").mkdir(parents=True, exist_ok=True)
        for ext in ("png", "pdf"):
            fig.savefig(str(OUT / "plots" / ("tsz_x_frbdm_corrected_lee22_takahashi." + ext)), dpi=190 if ext == "png" else None)
        plt.close(fig)
    (OUT / "analysis").mkdir(parents=True, exist_ok=True)
    for path, data in ((OUT / "analysis/tsz_x_frbdm_corrected_annuli.csv", rows), (OUT / "analysis/tsz_x_frbdm_corrected_chi2.csv", chi)):
        with open(str(path), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
    # compact chi^2 table: rows y, columns DM, "Planck / ACT"
    table = {}
    for r in chi:
        table.setdefault((r["y"], r["dm"]), {})[r["survey"]] = r["chi2"]
    print("chi^2 Planck / ACT; w=0: {:.1f} / {:.1f}".format(table[("none", "none")]["planck"], table[("none", "none")]["act"]))
    print("{:34s}".format("y \\ DM") + "".join("{:>22s}".format(DM[d][:21]) for d in DM))
    for ykey in Y:
        print("{:34s}".format(Y[ykey][:33]) + "".join("{:>22s}".format("{:.1f} / {:.1f}".format(table[(ykey, d)]["planck"], table[(ykey, d)]["act"])) for d in DM))
    for k, v in meta["maps"].items():
        if k.startswith("dm_lee22_noconc_phys"):
            print("{}: mean {:.3f} pc/cm3".format(k, v["mean"]))


if __name__ == "__main__":
    main()
