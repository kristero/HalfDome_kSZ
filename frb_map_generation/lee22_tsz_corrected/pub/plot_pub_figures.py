#!/usr/bin/env python3
"""Publication versions of the Lee22-correction figures (2026-10-01): minimal labelling, vector output, journal font sizes.

  pub_halo_pdf_zseries      halo-DM PDFs of 10^13-10^14 Msun halos at z_s = 0.1-3: TNG300 (Konietzka; true decade
                            DM(10to13) - DM(10to12), tng_windows.py) against HalfDome rays with Battaglia16, the Lee22 XGPaint
                            pipeline (density x 0.6016) and the corrected Lee22 density (x [h(1+z)]^3 X_H^2)
  pub_halo_pdf_analytic     z_s = 0.1, 0.5, 1: HalfDome rays (solid) and the analytic halo model in the TNG cosmology (dashed)
  pub_takahashi             w_yDM(theta) against Takahashi+25 (Planck MILCA, ACT) for three y x DM pairs

Histogram bins with fewer than MIN_COUNT rays are not drawn (they trace the one-count floor). Numbers for captions are written to
captions.txt. Run from the repository root with the halfdome env (LD_LIBRARY_PATH=<env>/lib)."""
import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, NullFormatter
import numpy as np
from scipy.stats import chi2 as chi2_dist

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parents[1]))
from frb_map_generation.compare_tng_halfdome_direct import DirectComparison  # noqa: E402
from tng_windows import identity_residual, tng_exact  # noqa: E402
import fullsky_tsz_dm_comparison as fs  # noqa: E402

OUT = ROOT / "frb_map_generation/outputs/lee22_pub_figures_20261001"
FMG = ROOT / "frb_map_generation/outputs"
RAYS = {("b16", "idark"): FMG / "lee22_tngcal_20260929/rays", ("lee22_xgpnative", "idark"): FMG / "lee22_tngcal_20260929/rays",
        ("lee22_phys", "idark"): FMG / "lee22_corrected_20260930/rays"}
LOCAL = FMG / "lee22_pdf_zseries_20261001/rays"
ANALYTIC = FMG / "lee22_corrected_20260930/analytic/analysis"
WIN, NRAY, MIN_COUNT = "m1e13_to_1e14", 120000, 5
# Okabe-Ito colours, distinct line styles as a second cue
MOD = (("b16", "b16", "Battaglia16", "#0072B2", (0, (5, 1.5, 1, 1.5))),
       ("lee22_xgpnative", "lee22_noconc_xgpnative", "Lee22 XGPaint pipeline", "#E69F00", (0, (4, 2))),
       ("lee22_phys", "lee22_noconc_phys", "Lee22 corrected", "#009E73", "-"))
TNG_KW = dict(color="k", lw=1.6)
RC = {"font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix", "font.size": 8.5,
      "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8.5, "axes.linewidth": 0.6,
      "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True, "xtick.major.width": 0.6,
      "ytick.major.width": 0.6, "xtick.minor.width": 0.4, "ytick.minor.width": 0.4, "lines.linewidth": 1.2,
      "savefig.dpi": 300, "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02}


def rays(tag, zs):
    folder = RAYS[(tag, "idark")] if zs <= 1.0 else LOCAL
    path = folder / "zsrc{}_nside4096_nrays120000_allhalos_sphere1p0_m200c_{}_seed42".format(("%.1f" % zs).replace(".", "p"), tag)
    with h5py.File(path / "halfdome_uniform_fixedz_foreground_mass_histograms.h5", "r") as h:
        labels = [v.decode() if isinstance(v, bytes) else str(v) for v in h["window_label"][:]]
        edges = np.asarray(h["pdf_bin_edges_pc_cm3"][:], float)
        cnt = np.asarray(h["pdf_count"][:], np.int64)
        assert abs(float(h.attrs["provenance_source_redshift"]) - zs) < 1e-9
    cnt = cnt if cnt.shape[0] == len(labels) else cnt.T
    return edges, cnt[labels.index(WIN)].astype(float)


def pdf(counts, edges):
    """p(DM | DM > 0) per unit DM, with bins below MIN_COUNT masked."""
    p = counts / counts.sum() / np.diff(edges)
    return np.where(counts >= MIN_COUNT, p, np.nan)


def mean_per_hit(counts, edges):
    return float((counts * np.sqrt(edges[1:] * edges[:-1])).sum() / counts.sum())


def decorate_pdf(top, bot, first_col, last_row, tcnt, edges):
    """Axes fitted to the TNG PDF (bins with >= MIN_COUNT rays), so every panel is filled; tick labels on every panel."""
    cen = np.sqrt(edges[1:] * edges[:-1]); pt = pdf(tcnt, edges); ok = np.isfinite(pt) & (pt > 0)
    xlo, xhi = cen[ok].min() / 1.12, cen[ok].max() * 1.12
    ylo, yhi = pt[ok].min() / 1.6, pt[ok].max() * 1.8
    for ax in (top, bot):
        ax.set_xscale("log"); ax.set_xlim(xlo, xhi)
        ax.xaxis.set_minor_formatter(NullFormatter())
    top.set_yscale("log"); top.set_ylim(ylo, yhi); top.yaxis.set_major_locator(LogLocator(numticks=8))
    top.yaxis.set_minor_formatter(NullFormatter())
    top.tick_params(labelbottom=False)
    bot.axhline(0, color="0.5", lw=0.6, zorder=1); bot.set_ylim(-80, 80); bot.set_yticks([-50, 0, 50])
    if first_col:
        top.set_ylabel(r"$p(\mathrm{DM}\,|\,\mathrm{DM}>0)$  [pc$^{-1}$ cm$^{3}$]"); bot.set_ylabel(r"$\Delta$ [%]")
    else:
        bot.tick_params(labelleft=False)
    if last_row:
        bot.set_xlabel(r"DM  [pc cm$^{-3}$]")


def residual(model_counts, tng_counts, edges):
    pm, pt = model_counts / model_counts.sum(), tng_counts / tng_counts.sum()
    ok = (model_counts >= 10) & (tng_counts >= 10)
    return np.where(ok, 100 * (pm / np.where(pt > 0, pt, np.nan) - 1), np.nan)


def fig_zseries(direct, captions):
    zss = (0.1, 0.5, 1.0, 1.5, 2.0, 3.0)
    with plt.rc_context(RC):
        fig = plt.figure(figsize=(7.1, 5.2))
        outer = fig.add_gridspec(2, 3, wspace=0.2, hspace=0.2, left=0.085, right=0.995, bottom=0.075, top=0.885)
        captions.append("pub_halo_pdf_zseries: mean DM per hit / TNG (z_s = " + ", ".join("%g" % z for z in zss) + ")")
        ratios = {m[2]: [] for m in MOD}
        for j, zs in enumerate(zss):
            g = outer[j // 3, j % 3].subgridspec(2, 1, height_ratios=(3, 1.1), hspace=0.0)
            top, bot = fig.add_subplot(g[0]), None
            bot = fig.add_subplot(g[1], sharex=top)
            assert identity_residual(direct, zs) < 1e-10
            edges, _ = rays("b16", zs)
            cen = np.sqrt(edges[1:] * edges[:-1])
            _, tcnt, _ = direct.histogram_from_values(tng_exact(direct, WIN, zs), edges)
            tcnt = np.asarray(tcnt, float)
            for tag, _, lab, col, ls in MOD:
                e2, c = rays(tag, zs)
                np.testing.assert_allclose(e2, edges, rtol=1e-12)
                top.plot(cen, pdf(c, edges), color=col, ls=ls, lw=1.2, zorder=4)
                bot.plot(cen, residual(c, tcnt, edges), color=col, ls=ls, lw=1.0, zorder=4)
                ratios[lab].append(mean_per_hit(c, edges) / mean_per_hit(tcnt, edges))
            top.plot(cen, pdf(tcnt, edges), zorder=5, **TNG_KW)
            top.text(0.05, 0.06, r"$z_s = %g$" % zs, transform=top.transAxes, va="bottom")
            decorate_pdf(top, bot, j % 3 == 0, j // 3 == 1, tcnt, edges)
        for lab, r in ratios.items():
            captions.append("  %-24s %s" % (lab, "  ".join("%.2f" % x for x in r)))
        h = [Line2D([], [], label="TNG300", **TNG_KW)] + [Line2D([], [], color=c, ls=ls, lw=1.2, label=l) for _, _, l, c, ls in MOD]
        fig.suptitle(r"Halos with $10^{13} \leq M_{200c} < 10^{14}\,M_\odot$", y=0.995, fontsize=10)
        fig.legend(handles=h, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.54, 0.965), handlelength=2.6, columnspacing=1.6)
        save(fig, "pub_halo_pdf_zseries")


def analytic_pdf(model, zs):
    d = np.load(str(ANALYTIC / ("tng_{}_{}_zs{}.npz".format(model, WIN, zs))))
    return d["edges"], d["pdf_clustered"]


def fig_analytic(direct, captions):
    zss = (0.1, 0.5, 1.0)
    use = [m for m in MOD if m[0] != "lee22_xgpnative"]
    with plt.rc_context(RC):
        fig = plt.figure(figsize=(7.1, 3.05))
        outer = fig.add_gridspec(1, 3, wspace=0.2, left=0.085, right=0.995, bottom=0.14, top=0.8)
        captions.append("pub_halo_pdf_analytic: mean DM per hit / TNG, rays | analytic (z_s = 0.1, 0.5, 1)")
        txt = {m[2]: [] for m in use}
        for j, zs in enumerate(zss):
            g = outer[0, j].subgridspec(2, 1, height_ratios=(3, 1.1), hspace=0.0)
            top = fig.add_subplot(g[0]); bot = fig.add_subplot(g[1], sharex=top)
            edges, _ = rays("b16", zs)
            cen = np.sqrt(edges[1:] * edges[:-1])
            _, tcnt, _ = direct.histogram_from_values(tng_exact(direct, WIN, zs), edges)
            tcnt = np.asarray(tcnt, float)
            tp = tcnt / tcnt.sum() / np.diff(edges)
            for tag, amodel, lab, col, ls in use:
                _, c = rays(tag, zs)
                ae, ap = analytic_pdf(amodel, zs)
                np.testing.assert_allclose(ae, edges, rtol=1e-9)
                top.plot(cen, pdf(c, edges), color=col, lw=1.2, zorder=4)
                top.plot(cen, np.where(ap > 1e-12, ap, np.nan), color=col, lw=1.0, ls=(0, (3, 1.5)), zorder=4)
                bot.plot(cen, residual(c, tcnt, edges), color=col, lw=1.0)
                ok = tcnt >= 10
                bot.plot(cen, np.where(ok, 100 * (ap / np.where(tp > 0, tp, np.nan) - 1), np.nan), color=col, lw=0.9, ls=(0, (3, 1.5)))
                am = float((ap * np.diff(edges) * cen).sum() / (ap * np.diff(edges)).sum())
                txt[lab].append("%.2f | %.2f" % (mean_per_hit(c, edges) / mean_per_hit(tcnt, edges), am / mean_per_hit(tcnt, edges)))
            top.plot(cen, pdf(tcnt, edges), zorder=5, **TNG_KW)
            top.text(0.05, 0.06, r"$z_s = %g$" % zs, transform=top.transAxes, va="bottom")
            decorate_pdf(top, bot, j == 0, True, tcnt, edges)
        for lab, r in txt.items():
            captions.append("  %-24s %s" % (lab, "   ".join(r)))
        h = [Line2D([], [], label="TNG300", **TNG_KW)] + [Line2D([], [], color=c, lw=1.2, label=l) for _, _, l, c, _ in use]
        h += [Line2D([], [], color="0.4", lw=1.2, label="HalfDome rays"), Line2D([], [], color="0.4", lw=1.0, ls=(0, (3, 1.5)), label="halo model")]
        fig.suptitle(r"Halos with $10^{13} \leq M_{200c} < 10^{14}\,M_\odot$", y=0.995, fontsize=10)
        fig.legend(handles=h, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.54, 0.935), handlelength=2.6, columnspacing=1.4)
        save(fig, "pub_halo_pdf_analytic")


def fig_takahashi(captions):
    with np.load(str(FMG / "lee22_corrected_20260930/spectra/corrected_spectra.npz"), allow_pickle=False) as f:
        S = {k: f[k] for k in f.files if k.startswith("cl_")}
    pairs = (("b12", "b16_sphere1", "Battaglia12 / Battaglia16", MOD[0][3], MOD[0][4]),
             ("lee22p", "lee22_noconc_sphere1", "Lee22 XGPaint pipeline", MOD[1][3], MOD[1][4]),
             ("lee22p_m1", "lee22_noconc_phys_sphere1", "Lee22 corrected", MOD[2][3], MOD[2][4]))
    hartlap = {"planck": (71 - 8 - 2) / 70, "act": (31 - 11 - 2) / 30}   # leave-one-FRB-out jackknife, 71 / 31 FRBs
    theta = np.geomspace(1, 1000, 400)
    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.7), sharey=False)
        for ax, plane, title in zip(axes, ("planck", "act"), ("Planck", "ACT")):
            obs = fs.observations(plane)
            use = obs["lo"] >= fs.PAPER_CUT[plane] - 1e-9
            inv = np.linalg.inv(obs["covariance"][np.ix_(use, use)])
            chi2 = lambda m: float((m[use] - obs["w"][use]) @ inv @ (m[use] - obs["w"][use]))
            n = int(use.sum())
            line = "%s (%d bins, Hartlap %.3f):  w=0 %.1f" % (title, n, hartlap[plane], chi2(np.zeros_like(obs["w"])))
            ax.axvspan(1, fs.PAPER_CUT[plane], color="0.92", lw=0, zorder=0)
            for yk, dk, lab, col, ls in pairs:
                cl = S["cl_y_{}__dm_{}_{}".format(yk, dk, plane)]
                ax.plot(theta, fs.angular_correlation(cl, theta, fs.BEAMS[plane]) / 1e-5, color=col, ls=ls, lw=1.2, zorder=4)
                c2 = chi2(fs.annular_correlation(cl, obs["lo"], obs["hi"], fs.BEAMS[plane]))
                line += ";  %s %.1f -> %.1f (p = %.3f)" % (lab.split(" $")[0].split(" y")[0], c2, hartlap[plane] * c2,
                                                           chi2_dist.sf(hartlap[plane] * c2, n))
            ax.errorbar(obs["x"], obs["w"] / 1e-5, yerr=obs["err"] / 1e-5, fmt="o", color="k", ms=3, capsize=1.5, lw=0.8, zorder=6)
            ax.axhline(0, color="0.5", lw=0.6, zorder=1)
            ax.set_xscale("log"); ax.set_xlim(1, 1000)
            ax.set_xlabel(r"$\theta$  [arcmin]")
            ax.text(0.95, 0.92, title, transform=ax.transAxes, ha="right", va="top")
            captions.append(line)
        axes[0].set_ylabel(r"$w_{y\mathrm{DM}}(\theta)$  [$10^{-5}$ pc cm$^{-3}$]")
        h = [Line2D([], [], color="k", marker="o", ms=3, ls="none", label="Takahashi+25")]
        h += [Line2D([], [], color=c, ls=ls, lw=1.2, label=l) for _, _, l, c, ls in pairs]
        fig.legend(handles=h, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.53, 1.04), handlelength=2.6, columnspacing=1.6)
        fig.subplots_adjust(left=0.08, right=0.995, bottom=0.16, top=0.87, wspace=0.18)
        save(fig, "pub_takahashi")


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(str(OUT / ("%s.%s" % (name, ext))))
    plt.close(fig)


def main():
    direct = DirectComparison(ROOT)
    captions = []
    fig_zseries(direct, captions)
    fig_analytic(direct, captions)
    fig_takahashi(captions)
    (OUT / "captions.txt").write_text("\n".join(captions) + "\n")
    print("\n".join(captions))


if __name__ == "__main__":
    main()
