#!/usr/bin/env python3
"""Lee22 paper Fig. 2 (electron pressure) and Fig. 5 (electron density) at z = 0 against our implementation and the TNG
model (CAMELS). Paper curves: vector paths extracted from the arXiv source PDFs (2205.01710v1, figures/fig_2.pdf and
fig_4.pdf; extract_lee22_fig2_fig5.py): TNG mean profile of the lowest-concentration quartile and the no-concentration fit
(Tables 7/8). Our curves: lee22_models.py (Table 7/8 transcription used in production) at the bin-centre mass, B12 P_e and
B16 n_e (XGPaint). CAMELS: 27 CV boxes at z = 0, halos in the same M200c bins (h = 0.6774 for the paper's h^-1 Msun),
gas-only precomputed profiles x the electron fraction f_xe(x) measured on the direct boxes' z = 0 halos above 10^13.2 Msun; mean over halos, box jackknife.
At z = 0 the group_particles bookkeeping factor a^3 (a^2) is exactly 1, so these panels isolate the constant factors."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "lee22_normalization_forensics/C_pressure_consistency"))
import lee22_models as lm  # noqa: E402

CURVES = ROOT / "outputs/lee22_normalization_forensics_20260928/C_pressure_consistency/lee22_fig2_fig5_vector_curves.csv"
CAM = ROOT / "outputs/lee22_tsz_verification_20260929/D_tng_camels/halo_tables"
OUT = ROOT / "outputs/lee22_corrected_20260930/doc"
H_TNG = 0.6774
BINS = ((13.2, 13.4), (13.8, 14.0))   # log10 M200c / (h^-1 Msun), the paper's panels
FN, CZ0 = 1.0 / (0.6774 ** 3 * 0.76 ** 2), 0.2841790493880425   # derived F_n = 1/(h_TNG^3 X_H^2) = 5.57
INK, PAPER, OURS, B12C, A3C, CZC = "#111111", "#8c8c8c", "#E69F00", "#0072B2", "#CC79A7", "#009E73"
RC = {"font.family": "DejaVu Sans", "font.size": 13, "axes.labelsize": 14, "axes.titlesize": 13.5, "xtick.labelsize": 11.5,
      "ytick.labelsize": 11.5, "legend.fontsize": 11.5, "xtick.direction": "in", "ytick.direction": "in",
      "savefig.facecolor": "white", "pdf.fonttype": 42}


def paper(quantity, panel, curve):
    d = np.genfromtxt(CURVES, delimiter=",", names=True, dtype=None, encoding="utf-8")
    s = (d["quantity"] == quantity) & (d["panel"] == panel) & (d["curve"] == curve)
    x, v = d["x_r_over_R200"][s].astype(float), d["value"][s].astype(float)
    o = np.argsort(x)
    return x[o], v[o]


def camels_stack(which, lo, hi):
    P = dict(np.load(str(CAM / "precomputed_halos.npz")))
    D = dict(np.load(str(CAM / "direct_halos.npz")))
    xg = P["xg"]
    # electron fraction from the direct boxes, restricted to group masses (a pooled f_xe is dominated by
    # 10^12.5-13 halos and biases group densities low by ~15 %; verifier of workstream D)
    d0 = (np.abs(D["z"]) < 0.01) & (D["M200"] >= 10 ** 13.2)
    fxe = np.nansum(D["nbar_xe"][d0], 0) / np.nansum(D["nbar_fullion"][d0], 0)
    print("f_xe from %d direct z=0 halos with M200c >= 10^13.2 Msun; f_xe(0.1, 0.5, 1) =" % d0.sum(),
          np.round(np.interp([0.1, 0.5, 1.0], xg, fxe), 3))
    m_lo, m_hi = 10 ** lo / H_TNG, 10 ** hi / H_TNG
    s = (np.abs(P["z"]) < 0.01) & (P["M200"] >= m_lo) & (P["M200"] < m_hi)
    prof = P["nbar"][s] * fxe[None, :] if which == "n" else P["Pbar"][s]
    boxes = P["box"][s]
    mean = np.nanmean(prof, 0)
    ub = np.unique(boxes)
    jk = np.array([np.nanmean(prof[boxes != b], 0) for b in ub])
    err = np.sqrt((len(ub) - 1) / len(ub) * np.nansum((jk - jk.mean(0)) ** 2, 0))
    return xg, mean, err, int(s.sum()), len(ub), float(np.median(P["M200"][s]))


def main():
    x = np.geomspace(0.03, 2.5, 300)
    with plt.rc_context(RC):
        fig, axes = plt.subplots(2, 2, figsize=(13.2, 10.6), sharex=True)
        for row, (q, qq, lab) in enumerate((("P", "x3_Pe_over_P200", r"$x^3\,P_e/P_{200}$"), ("n", "x3_ne_over_n200", r"$x^3\,n_e/n_{200}$"))):
            for col, (lo, hi) in enumerate(BINS):
                ax = axes[row, col]
                xs, vs = paper(qq, col, "simulation")
                ax.plot(xs, vs, color=PAPER, lw=5, alpha=.55, solid_capstyle="round", zorder=2)
                xf, vf = paper(qq, col, "without_c_bpl")
                ax.plot(xf, vf, color=OURS, lw=3.2, ls=(0, (5, 3)), zorder=4)
                mh = 10 ** (0.5 * (lo + hi))           # bin centre, h^-1 Msun
                m_msun = mh / H_TNG
                ours = lm.lee_Pbar(x, mh, 0.0) if q == "P" else lm.lee_nbar(x, mh, 0.0)
                ax.plot(x, x ** 3 * ours, color=OURS, lw=1.6, zorder=5)
                if q == "P":
                    ax.plot(x, x ** 3 * lm.b12_Pe_over_P200(x, m_msun, 0.0), color=B12C, lw=2.0, zorder=3)
                else:
                    ax.plot(x, x ** 3 * lm.b16_nbar(x, m_msun, 0.0), color=B12C, lw=2.0, zorder=3)
                    ax.plot(x, x ** 3 * ours / FN, color=A3C, lw=2.2, ls="-.", zorder=4)
                    ax.plot(x, x ** 3 * ours * CZ0, color=CZC, lw=2.2, ls=(0, (2, 1.5)), zorder=4)
                xg, mean, err, nh, nb, mmed = camels_stack(q, lo, hi)
                ok = np.isfinite(mean) & (xg >= 0.03) & (xg <= 2.5)
                ax.errorbar(xg[ok], xg[ok] ** 3 * mean[ok], yerr=xg[ok] ** 3 * err[ok], fmt="o", ms=5.5, color=INK, mfc="white",
                            mew=1.4, capsize=2, lw=1.1, zorder=6)
                # ratio of the paper's TNG curve and of CAMELS at x = 0.3-1 (the constant factors of this section)
                sel = (xs > 0.3) & (xs < 1.0)
                xm = np.exp(np.mean(np.log(xs[sel])))
                r_paper_fit = np.exp(np.mean(np.log(vs[sel] / (xs[sel] ** 3 * (lm.lee_Pbar(xs[sel], mh, 0.0) if q == "P" else lm.lee_nbar(xs[sel], mh, 0.0))))))
                cs = ok & (xg > 0.3) & (xg < 1.0)
                r_cam_fit = np.exp(np.mean(np.log(mean[cs] / (lm.lee_Pbar(xg[cs], mh, 0.0) if q == "P" else lm.lee_nbar(xg[cs], mh, 0.0)))))
                ax.text(0.03, 0.96, "{}\nCAMELS: {} halos, {} boxes\nat $0.3<x<1$:  paper TNG / fit = {:.2f}\n"
                        "                  CAMELS / fit = {:.2f}".format(
                            r"$10^{%.1f}$–$10^{%.1f}\,h^{-1}M_\odot$" % (lo, hi), nh, nb, r_paper_fit, r_cam_fit),
                        transform=ax.transAxes, va="top", fontsize=11, color=".25", linespacing=1.35)
                ax.set_xscale("log"); ax.set_xlim(0.03, 2.5); ax.grid(color=".9", lw=.6)
                ax.tick_params(which="both", top=True, right=True)
                if col == 0:
                    ax.set_ylabel(lab)
                if row == 1:
                    ax.set_xlabel(r"$x = r/R_{200c}$")
                ax.set_ylim(0, None)
            axes[row, 0].set_title(("Electron pressure (Lee22 Fig. 2)" if q == "P" else "Electron density (Lee22 Fig. 5)"), loc="left", pad=6)
        h = [Line2D([], [], color=PAPER, lw=5, alpha=.55, label="Lee22 figure: their TNG measurement (lowest-$c$ quartile)"),
             Line2D([], [], color=OURS, lw=3.2, ls=(0, (5, 3)), label="Lee22 figure: no-$c$ fit (Tables 7/8)"),
             Line2D([], [], color=OURS, lw=1.6, label="our implementation of Tables 7/8 (production code)"),
             Line2D([], [], color=INK, marker="o", mfc="white", ls="none", label="TNG model, CAMELS (27 boxes, all halos)"),
             Line2D([], [], color=B12C, lw=2.0, label="Battaglia12 $P_e$ / Battaglia16 $n_e$ (XGPaint)"),
             Line2D([], [], color=A3C, lw=2.2, ls="-.", label="density fit × $h^3X_H^2$ (derived, = 1/5.57 at z = 0)"),
             Line2D([], [], color=CZC, lw=2.2, ls=(0, (2, 1.5)), label="density fit × $C(0)$ = 0.284")]
        fig.legend(handles=h, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.0), handlelength=2.8)
        fig.subplots_adjust(left=.08, right=.985, top=.86, bottom=.07, hspace=.12, wspace=.14)
        OUT.mkdir(parents=True, exist_ok=True)
        for ext in ("png", "pdf"):
            fig.savefig(str(OUT / ("doc_fig_paper_profiles_z0." + ext)), dpi=160 if ext == "png" else None)
        plt.close(fig)


if __name__ == "__main__":
    main()
