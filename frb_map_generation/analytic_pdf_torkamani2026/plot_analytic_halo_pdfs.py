#!/usr/bin/env python3
"""Figures: analytical halo-model DM PDFs (Torkamani et al. 2026 formalism, run_analytic_halo_pdfs.py)
against the IllustrisTNG within-R200 catalogue and the HalfDome ray-traced sphere products, z_s = 1.

Same windows, 300 log bins over 0.1 - 3e4 pc cm^-3 and normalization over rays inside that range as
the published comparison (make_publication_comparisons.halo_pdf_publication_panels). Percent panels:
per bin where the simulation sample holds >= 10 rays (thick), adjacent bins merged until it does
elsewhere (thin steps); linear axis clipped to +-100 %.
Run from the repository root: python frb_map_generation/analytic_pdf_torkamani2026/plot_analytic_halo_pdfs.py
"""
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, MaxNLocator
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from frb_map_generation.compare_tng_halfdome_direct import DirectComparison  # noqa: E402
from frb_map_generation.make_publication_comparisons import (  # noqa: E402
    load_spherical_1r200c, percent_difference_extended, merged_percent_difference, ROBUST_BIN_COUNT)

OUT = HERE.parent / "outputs" / "analytic_halo_pdf_20260925"
ANALYSIS = OUT / "analysis"
COL = {"b16": "#0072B2", "lee22_noconc_xgpnative": "#D55E00", "bfc": "#009E73"}
NAME = {"b16": "Battaglia16", "lee22_noconc_xgpnative": "Lee22 no-c", "bfc": "BFC (TNG-calibrated)"}
HD_KEYS = {"b16": "b16", "lee22_noconc_xgpnative": "lee22_noconc_xgpnative"}
HD_WINDOW = {"all": "m1e10_to_1e16"}
ANALYTIC_LS = (0, (5.5, 2.2))
RC = {"font.family": "DejaVu Sans", "font.size": 16, "axes.labelsize": 18, "axes.titlesize": 19,
      "xtick.labelsize": 14.5, "ytick.labelsize": 14.5, "legend.fontsize": 14.5, "axes.linewidth": 1.1,
      "xtick.direction": "in", "ytick.direction": "in", "pdf.fonttype": 42, "svg.fonttype": "none",
      "savefig.facecolor": "white"}


def load_analytic(name):
    d = np.load(ANALYSIS / (name + ".npz"))
    return dict(edges=d["edges"], pdf=d["pdf_clustered"], pdf_u=d["pdf_unclustered"],
                stats=json.loads(str(d["stats_clustered"])), stats_u=json.loads(str(d["stats_unclustered"])),
                nbar=float(d["nbar"]), run=json.loads(str(d["run"])))


def model_vs_counts(model_pdf, ref_pdf, ref_counts, edges):
    """Per-bin 100 (model/ref - 1) where the reference holds >= ROBUST_BIN_COUNT rays, and merged steps
    (reference counts accumulated low to high DM until >= ROBUST_BIN_COUNT) for the whole range."""
    width = np.diff(edges)
    robust = ref_counts >= ROBUST_BIN_COUNT
    per_bin = np.where(robust & (ref_pdf > 0), 100 * (model_pdf / np.where(ref_pdf > 0, ref_pdf, 1) - 1), np.nan)
    pm, pr = model_pdf * width, ref_pdf * width
    medges, vals = [], []
    i, n = 0, len(ref_counts)
    while i < n:
        j, c = i, int(ref_counts[i])
        while c < ROBUST_BIN_COUNT and j + 1 < n:
            j += 1; c += int(ref_counts[j])
        if c >= ROBUST_BIN_COUNT:
            if not medges:
                medges.append(edges[i])
            elif medges[-1] != edges[i]:
                medges.append(np.nan); vals.append(np.nan); medges.append(edges[i])
            medges.append(edges[j + 1]); vals.append(100 * (pm[i:j + 1].sum() / pr[i:j + 1].sum() - 1))
        i = j + 1
    return per_bin, robust, np.asarray(medges), np.asarray(vals)


def draw_steps(ax, medges, vals, color, ls, lw=1.1, alpha=.75):
    for k in range(len(vals)):
        if np.isfinite(vals[k]):
            v = np.clip(vals[k], -100, 100)
            ax.hlines(v, medges[k], medges[k + 1], color=color, lw=lw, alpha=alpha, ls=ls)
            if k + 1 < len(vals) and np.isfinite(vals[k + 1]):
                ax.vlines(medges[k + 1], v, np.clip(vals[k + 1], -100, 100), color=color, lw=.8, alpha=alpha * .8)
            if vals[k] > 100:
                ax.plot(np.sqrt(medges[k] * medges[k + 1]), 96, marker="^", ms=6, color=color, mec="white", mew=.5,
                        ls="none", clip_on=False)
            if vals[k] < -100:
                ax.plot(np.sqrt(medges[k] * medges[k + 1]), -96, marker="v", ms=6, color=color, mec="white", mew=.5,
                        ls="none", clip_on=False)


def axis(ax, percent=False, ylim=100):
    ax.set_xscale("log"); ax.set_xlim(.1, 5000)
    ax.grid(which="major", color=".86", lw=.65)
    ax.tick_params(which="major", length=6); ax.tick_params(which="minor", length=3)
    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=6))
    if percent:
        ax.axhline(0, color=".45", lw=1)
        ax.set_ylim(-ylim, ylim)
        ax.yaxis.set_major_locator(MaxNLocator(5, symmetric=True))


def tng_hist(direct, label, edges):
    pdf, counts, zero = direct.histogram_from_values(direct.tng_values(label, 1.0), edges)
    return pdf, counts, zero


PANELS_MAIN = (("All halos", "all"), (r"$10^{13}$ – $10^{14}\,M_\odot$", "m1e13_to_1e14"))
PANELS_EXTRA = ((r"$10^{10}$ – $10^{12}\,M_\odot$", "m1e10_to_1e12"), (r"$10^{10}$ – $10^{13}\,M_\odot$", "m1e10_to_1e13"),
                (r"$10^{12}$ – $10^{14}\,M_\odot$", "m1e12_to_1e14"))


def main_figure(direct, hd, rows_out, stats_out):
    n = len(PANELS_MAIN)
    with plt.rc_context(RC):
        fig = plt.figure(figsize=(7.6 * n + 0.8, 12.4))
        outer = fig.add_gridspec(1, n, wspace=.2, left=.08, right=.985, bottom=.06, top=.835)
        for j, (title, win) in enumerate(PANELS_MAIN):
            g = outer[0, j].subgridspec(3, 1, height_ratios=(2.7, 1.35, 1.2), hspace=.07)
            top = fig.add_subplot(g[0]); mid = fig.add_subplot(g[1], sharex=top); bot = fig.add_subplot(g[2], sharex=top)
            hdw = HD_WINDOW.get(win, win)
            edges = hd["b16"][hdw]["edges"]
            tpdf, tcnt, tzero = tng_hist(direct, win, edges)
            centers = np.sqrt(edges[1:] * edges[:-1])
            keep = tpdf > 0
            top.plot(centers[keep], tpdf[keep], color=".1", lw=3.3, zorder=5)
            stats_out.append(dict(window=win, product="TNG within R200", **tng_stats(direct, win)))
            for key in ("b16", "lee22_noconc_xgpnative"):
                prod = hd[key][hdw]
                np.testing.assert_allclose(prod["edges"], edges, rtol=1e-12)
                k2 = prod["pdf"] > 0
                top.plot(prod["centers"][k2], prod["pdf"][k2], color=COL[key], lw=1.9, alpha=.8)
                delta = percent_difference_extended(prod["pdf"], tpdf, prod["counts"], tcnt, 1)
                robust = (prod["counts"] >= ROBUST_BIN_COUNT) & (tcnt >= ROBUST_BIN_COUNT)
                me, mv = merged_percent_difference(prod["counts"], tcnt, edges)
                draw_steps(mid, me, mv, COL[key], "-", lw=.9, alpha=.45)
                mid.plot(prod["centers"], np.where(robust, delta, np.nan), color=COL[key], lw=1.6, alpha=.6)
            for key in ("b16", "lee22_noconc_xgpnative", "bfc"):
                an = load_analytic("tng_{}_{}".format(key, win))
                np.testing.assert_allclose(an["edges"], edges, rtol=1e-12)
                k3 = an["pdf"] > 1e-12
                top.plot(centers[k3], an["pdf"][k3], color=COL[key], lw=3.0, ls=ANALYTIC_LS, zorder=6)
                per, robust, me, mv = model_vs_counts(an["pdf"], tpdf, tcnt, edges)
                draw_steps(mid, me, mv, COL[key], ANALYTIC_LS, lw=1.2, alpha=.85)
                mid.plot(centers, per, color=COL[key], lw=2.8, ls=ANALYTIC_LS)
                stats_out.append(dict(window=win, product="analytic " + NAME[key] + " (Planck15, Tinker08, clustered)",
                                      nbar=an["nbar"], **an["stats"]))
                for b in range(len(edges) - 1):
                    rows_out.append(dict(figure="main", window=win, comparison="analytic_vs_tng", model=key,
                                         dm_low=edges[b], dm_high=edges[b + 1], model_pdf=an["pdf"][b], model_pdf_unclustered=an["pdf_u"][b],
                                         reference_pdf=tpdf[b], reference_count=int(tcnt[b]), percent=per[b], robust=bool(robust[b])))
                # validation: the same halos as HalfDome (HalfDome catalogue abundance) against the HalfDome rays
                if key in HD_KEYS:
                    val = load_analytic("validation_{}_catalog_{}".format(key, win))
                    prod = hd[key][hdw]
                    per_v, rob_v, me_v, mv_v = model_vs_counts(val["pdf"], prod["pdf"], prod["counts"], edges)
                    per_vu, _, _, _ = model_vs_counts(val["pdf_u"], prod["pdf"], prod["counts"], edges)
                    draw_steps(bot, me_v, mv_v, COL[key], ANALYTIC_LS, lw=1.1, alpha=.8)
                    bot.plot(centers, per_v, color=COL[key], lw=2.6, ls=ANALYTIC_LS)
                    bot.plot(centers, per_vu, color=COL[key], lw=1.3, ls=(0, (1, 1.6)), alpha=.9)
                    stats_out.append(dict(window=win, product="analytic " + NAME[key] + " (HalfDome halos: catalogue abundance, clustered)",
                                          nbar=val["nbar"], **val["stats"]))
                    stats_out.append(dict(window=win, product="analytic " + NAME[key] + " (HalfDome halos: catalogue abundance, unclustered)",
                                          nbar=val["nbar"], **val["stats_u"]))
                    stats_out.append(dict(window=win, product="HalfDome rays " + NAME[key], **hd_stats(prod)))
                    for b in range(len(edges) - 1):
                        rows_out.append(dict(figure="main", window=win, comparison="analytic_same_halos_vs_halfdome", model=key,
                                             dm_low=edges[b], dm_high=edges[b + 1], model_pdf=val["pdf"][b],
                                             model_pdf_unclustered=val["pdf_u"][b], reference_pdf=prod["pdf"][b],
                                             reference_count=int(prod["counts"][b]), percent=per_v[b], robust=bool(rob_v[b])))
            top.set_title(title, pad=10)
            top.set_yscale("log"); axis(top); axis(mid, True); axis(bot, True, 60)
            top.set_ylim(3e-8, 3e-2)
            top.yaxis.set_major_locator(LogLocator(base=10, numticks=6))
            top.tick_params(labelbottom=False); mid.tick_params(labelbottom=False)
            bot.set_xlabel(r"DM [pc cm$^{-3}$]")
            if win == "all":
                top.text(.03, .05, "analytic: $10^{9}$–$10^{16}\\,M_\\odot$ (TNG's halos)\nHalfDome rays: $\\geq 7.3\\times10^{12}\\,M_\\odot$",
                         transform=top.transAxes, fontsize=12.5, va="bottom", color=".25")
                bot.text(.03, .06, "analytic restricted to $\\geq 7.3\\times10^{12}\\,M_\\odot$", transform=bot.transAxes,
                         fontsize=11.5, color=".3")
            if j == 0:
                top.set_ylabel(r"$p(\mathrm{DM}\mid\mathrm{DM}>0)$")
                mid.set_ylabel(r"vs TNG [%]")
                bot.set_ylabel("analytic vs\nHalfDome [%]")
            else:
                for ax in (top, mid, bot):
                    ax.tick_params(labelleft=False)
        handles = [Line2D([], [], color=".1", lw=3.3, label="IllustrisTNG (within $R_{200}$)"), Line2D([], [], color="none", label=" ")]
        handles += [Line2D([], [], color=COL[k], lw=1.9, alpha=.8, label="HalfDome rays: " + NAME[k]) for k in HD_KEYS]
        handles += [Line2D([], [], color=COL[k], lw=3.0, ls=ANALYTIC_LS, label="Analytic: " + NAME[k]) for k in ("b16", "lee22_noconc_xgpnative", "bfc")]
        handles += [Line2D([], [], color=".35", lw=1.3, ls=(0, (1, 1.6)), label="Analytic, no halo clustering (bottom row)")]
        fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(.5, .998),
                   handlelength=3.4, columnspacing=1.6, fontsize=13.5)
        fig.text(.5, .9, "Halo-only DM to $z_s = 1$, gas inside each halo's $R_{200c}$ sphere.  Analytic: Torkamani+26 halo model "
                 "(Tinker08 abundance, halo clustering), Planck15 for TNG rows, HalfDome's own halos in the bottom row",
                 ha="center", fontsize=12.2, color=".25")
        save(fig, "analytic_halo_pdf_vs_tng_halfdome")


def extra_figure(direct, hd, rows_out, stats_out):
    n = len(PANELS_EXTRA)
    with plt.rc_context(RC):
        fig = plt.figure(figsize=(6.6 * n + 0.8, 9.2))
        outer = fig.add_gridspec(1, n, wspace=.18, left=.075, right=.985, bottom=.085, top=.79)
        for j, (title, win) in enumerate(PANELS_EXTRA):
            g = outer[0, j].subgridspec(2, 1, height_ratios=(2.6, 1.4), hspace=.07)
            top = fig.add_subplot(g[0]); mid = fig.add_subplot(g[1], sharex=top)
            edges = hd["b16"]["m1e13_to_1e14"]["edges"]
            centers = np.sqrt(edges[1:] * edges[:-1])
            tpdf, tcnt, _ = tng_hist(direct, win, edges)
            keep = tpdf > 0
            top.plot(centers[keep], tpdf[keep], color=".1", lw=3.3, zorder=5)
            stats_out.append(dict(window=win, product="TNG within R200", **tng_stats(direct, win)))
            if win == "m1e12_to_1e14":
                for key in HD_KEYS:
                    prod = hd[key][win]
                    k2 = prod["pdf"] > 0
                    top.plot(prod["centers"][k2], prod["pdf"][k2], color=COL[key], lw=1.9, alpha=.8)
                    delta = percent_difference_extended(prod["pdf"], tpdf, prod["counts"], tcnt, 1)
                    robust = (prod["counts"] >= ROBUST_BIN_COUNT) & (tcnt >= ROBUST_BIN_COUNT)
                    mid.plot(prod["centers"], np.where(robust, delta, np.nan), color=COL[key], lw=1.6, alpha=.6)
            for key in ("b16", "lee22_noconc_xgpnative", "bfc"):
                an = load_analytic("tng_{}_{}".format(key, win))
                k3 = an["pdf"] > 1e-12
                top.plot(centers[k3], an["pdf"][k3], color=COL[key], lw=3.0, ls=ANALYTIC_LS, zorder=6)
                per, robust, me, mv = model_vs_counts(an["pdf"], tpdf, tcnt, edges)
                draw_steps(mid, me, mv, COL[key], ANALYTIC_LS, lw=1.2, alpha=.85)
                mid.plot(centers, per, color=COL[key], lw=2.8, ls=ANALYTIC_LS)
                stats_out.append(dict(window=win, product="analytic " + NAME[key] + " (Planck15, Tinker08, clustered)",
                                      nbar=an["nbar"], **an["stats"]))
                for b in range(len(edges) - 1):
                    rows_out.append(dict(figure="extra", window=win, comparison="analytic_vs_tng", model=key,
                                         dm_low=edges[b], dm_high=edges[b + 1], model_pdf=an["pdf"][b], model_pdf_unclustered=an["pdf_u"][b],
                                         reference_pdf=tpdf[b], reference_count=int(tcnt[b]), percent=per[b], robust=bool(robust[b])))
            top.set_title(title, pad=10)
            top.set_yscale("log"); axis(top); axis(mid, True)
            top.set_ylim(3e-8, 3e-2); top.yaxis.set_major_locator(LogLocator(base=10, numticks=6))
            top.tick_params(labelbottom=False)
            mid.set_xlabel(r"DM [pc cm$^{-3}$]")
            if j == 0:
                top.set_ylabel(r"$p(\mathrm{DM}\mid\mathrm{DM}>0)$"); mid.set_ylabel(r"vs TNG [%]")
            else:
                top.tick_params(labelleft=False); mid.tick_params(labelleft=False)
        handles = [Line2D([], [], color=".1", lw=3.3, label="IllustrisTNG (within $R_{200}$)")]
        handles += [Line2D([], [], color=COL[k], lw=3.0, ls=ANALYTIC_LS, label="Analytic: " + NAME[k]) for k in ("b16", "lee22_noconc_xgpnative", "bfc")]
        handles += [Line2D([], [], color=COL[k], lw=1.9, alpha=.8, label="HalfDome rays: " + NAME[k]) for k in HD_KEYS]
        fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(.5, .995), handlelength=3.4,
                   columnspacing=1.8, fontsize=13.5)
        fig.text(.5, .865, "Windows below the HalfDome halo floor ($7.3\\times10^{12}\\,M_\\odot$): only the analytic model reaches them "
                 "(HalfDome rays shown for $10^{12}$–$10^{14}$, where they hold only halos above the floor)",
                 ha="center", fontsize=12, color=".25")
        save(fig, "analytic_halo_pdf_vs_tng_low_mass_windows")


def tng_stats(direct, win):
    v = direct.tng_values(win, 1.0)
    hit = v[v > 0]
    out = dict(hit_fraction=float((v > 0).mean()), mean_dm_hit=float(hit.mean()), median_dm_hit=float(np.median(hit)))
    for t in (1, 3, 10, 30):
        out["frac_hit_below_{}".format(t)] = float((hit < t).mean())
    return out


def hd_stats(prod):
    c = prod["counts"].astype(float); e = prod["edges"]
    cdf = np.concatenate([[0], np.cumsum(c)]) / c.sum()
    out = dict(hit_fraction=1 - prod["zero_fraction"], mean_dm_hit=float((c * np.sqrt(e[1:] * e[:-1])).sum() / c.sum()),
               median_dm_hit=float(np.interp(.5, cdf, e)))
    for t in (1, 3, 10, 30):
        out["frac_hit_below_{}".format(t)] = float(np.interp(t, e, cdf))
    return out


def save(fig, stem):
    (OUT / "plots").mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / "plots" / (stem + "." + ext), dpi=190 if ext == "png" else None, bbox_inches="tight", pad_inches=.15)
    plt.close(fig)
    print("saved", OUT / "plots" / (stem + ".png"))


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)


def main():
    direct = DirectComparison(ROOT)
    hd = {k: load_spherical_1r200c(ROOT, v) for k, v in HD_KEYS.items()}
    rows, stats = [], []
    main_figure(direct, hd, rows, stats)
    extra_figure(direct, hd, rows, stats)
    write_csv(ANALYSIS / "analytic_halo_pdf_bins.csv", rows)
    seen, uniq = set(), []
    for s in stats:
        key = (s["window"], s["product"])
        if key not in seen:
            seen.add(key); uniq.append(s)
    write_csv(ANALYSIS / "analytic_halo_pdf_summary.csv", uniq)
    for s in uniq:
        print("{:14s} {:78s} hit={:.4f} mean={:7.2f} median={:7.2f} <3:{:.4f} <10:{:.4f} <30:{:.4f}".format(
            s["window"], s["product"], s["hit_fraction"], s["mean_dm_hit"], s["median_dm_hit"],
            s["frac_hit_below_3"], s["frac_hit_below_10"], s["frac_hit_below_30"]))


if __name__ == "__main__":
    main()
