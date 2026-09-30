#!/usr/bin/env python3
"""Figure of the analytic TNG300 reproduction (fig_analytic_tng_repro.{png,pdf}).
Top: (a) total LSS DM at z_s = 0.7 (Torkamani+26's validation case), (b) halo-only 'all' at z_s = 1, (c) 1e13-1e14
array under the reinterpreted window [1e13, 9e14] at z_s = 1, (d) mean DM per ray by mass decade at z_s = 1.
Bottom: percent differences to TNG300 for (a)-(c), and (h) model/TNG mean of the reinterpreted window vs z_s.
Black = TNG300 (Konietzka); Okabe-Ito: #0072B2 Battaglia16, #009E73 corrected Lee22 (solid F_n = 5.4, dashed 5.57),
#CC79A7 BFC / Torkamani, #E69F00 previous Lee22 (x0.6016)."""
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.lines import Line2D
import numpy as np

import repro_lib as R
from analyze_repro import lin_cdf, cdf_moments, macquart_cdf, F_MEDLOCK

RUNS = R.OUT / "analysis" / "runs"
COL = {"b16": "#0072B2", "lee22_corr_fn5p40": "#009E73", "lee22_corr_fn5p57": "#009E73", "bfc": "#CC79A7",
       "lee22_prev_xgpnative": "#E69F00", "tng": "black", "macquart": "0.45"}
LS = {"b16": "-", "lee22_corr_fn5p40": "-", "lee22_corr_fn5p57": (0, (5, 2.2)), "bfc": "-", "lee22_prev_xgpnative": "-"}
ORDER = ("lee22_corr_fn5p40", "lee22_corr_fn5p57", "b16", "bfc", "lee22_prev_xgpnative")
LABEL = {"b16": "Battaglia16", "lee22_corr_fn5p40": r"corrected Lee22, $F_n$ = 5.4",
         "lee22_corr_fn5p57": r"corrected Lee22, $F_n = 1/(h^3X_H^2)$ = 5.57", "bfc": "BFC, Torkamani+26 Table 1",
         "lee22_prev_xgpnative": "previous Lee22 (x0.6016)"}


def tng_hist(values, edges):
    v = values[values > 0]
    c, _ = np.histogram(v, edges)
    return c / (c.sum() * np.diff(edges)), c


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.labelsize": 12, "axes.titlesize": 12,
                         "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
                         "pdf.fonttype": 42})
    fig = plt.figure(figsize=(18.5, 8.8))
    gs = fig.add_gridspec(2, 4, height_ratios=(2.2, 1.25), hspace=0.07, wspace=0.26, left=0.05, right=0.99, top=0.865, bottom=0.08)
    axT = [fig.add_subplot(gs[0, j]) for j in range(4)]
    axB = [fig.add_subplot(gs[1, j], sharex=axT[j]) for j in range(3)]
    axB.append(fig.add_subplot(gs[1, 3].subgridspec(2, 1, height_ratios=(0.30, 1), hspace=0)[1]))

    # (a) total LSS, z_s = 0.7
    z = 0.7
    dm = R.tng("lss_total", z)
    mu, sd = dm.mean(), dm.std()
    edges = np.linspace(0.3 * mu, 2.2 * mu, 77)
    cen = 0.5 * (edges[1:] + edges[:-1])
    cnt, _ = np.histogram(dm, edges)
    tp = cnt / (dm.size * np.diff(edges))
    ok = cnt >= 10
    ax, bx = axT[0], axB[0]
    ax.step(cen, tp, where="mid", color="k", lw=2.0)
    d = np.load(RUNS / "lss_bfc5r200_taper25_zs0.7.npz")
    g, c = lin_cdf(d)
    mh, _ = cdf_moments(g, c)
    mp = np.diff(np.interp(edges - (mu - mh), g, c)) / np.diff(edges)
    ax.plot(cen, mp, color=COL["bfc"], lw=2.2)
    fig6 = np.genfromtxt(R.BFC_FIG6, delimiter=",", names=True)
    ax.plot(fig6["dm_pc_cm3"], fig6["pdf_per_pc_cm3"], color=COL["bfc"], lw=1.6, ls=(0, (1.2, 1.6)))
    mq = np.diff(macquart_cdf(edges, mu, F_MEDLOCK / np.sqrt(z))) / np.diff(edges)
    ax.plot(cen, mq, color=COL["macquart"], lw=1.6, ls=(0, (6, 2)))
    for arr, style in ((mp, dict(color=COL["bfc"], lw=2.2)), (mq, dict(color=COL["macquart"], lw=1.6, ls=(0, (6, 2))))):
        bx.plot(cen, np.where(ok, 100 * (arr / np.where(tp > 0, tp, 1) - 1), np.nan), **style)
    f6 = np.interp(cen, fig6["dm_pc_cm3"], fig6["pdf_per_pc_cm3"], left=np.nan, right=np.nan)
    bx.plot(cen, np.where(ok, 100 * (f6 / np.where(tp > 0, tp, 1) - 1), np.nan), color=COL["bfc"], lw=1.6, ls=(0, (1.2, 1.6)))
    ax.set_xlim(250, 1500); ax.set_ylim(0, max(tp.max(), np.nanmax(mq)) * 1.08)
    ax.set_ylabel(r"$p$(DM) [(pc cm$^{-3}$)$^{-1}$]")
    ax.set_title("(a) total cosmic DM, $z_s$ = 0.7 (Torkamani's test)", loc="left")
    ax.text(0.97, 0.95, "BFC halos to 5 $R_{200c}$,\n1e8-1e16 $M_\\odot/h$, shifted\nto the TNG300 mean", transform=ax.transAxes,
            ha="right", va="top", fontsize=9.5, color="0.25")
    bx.set_xlabel(r"DM$_{\rm LSS}$ [pc cm$^{-3}$]")
    bx.axvspan(mu - sd, mu + sd, color="0.93", zorder=0)

    # (b), (c): halo-only
    panels = ((1, "all", "all", "tng300_full", "(b) halo-only, all halos, $z_s$ = 1"),
              (2, "reinterp", "m1e13_to_1e14", "tng300", "(c) '1e13-1e14' read as [1e13, 9e14], $z_s$ = 1"))
    for j, target, label, ab, title in panels:
        ax, bx = axT[j], axB[j]
        tv = R.tng(label, 1.0)
        e = R.PDF_EDGES
        cc = np.sqrt(e[1:] * e[:-1])
        tp2, c2 = tng_hist(tv, e)
        ax.step(cc, tp2, where="mid", color="k", lw=2.0)
        for m in ORDER:
            d = np.load(RUNS / f"halo_{m}_{target}_{ab}_zs1.0.npz")
            p = d["pdf_clustered"]
            ax.plot(cc, p, color=COL[m], ls=LS[m], lw=1.9)
            ok2 = c2 >= 10
            bx.plot(cc, np.where(ok2, 100 * (p / np.where(tp2 > 0, tp2, 1) - 1), np.nan), color=COL[m], ls=LS[m], lw=1.7)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(1, 3000) if target == "all" else ax.set_xlim(3, 3000)
        ax.set_ylim(1e-6, 0.05) if target == "all" else ax.set_ylim(3e-6, 0.02)
        ax.set_title(title, loc="left")
        bx.set_xlabel(r"DM$_{\rm halo}$ (gas inside $R_{200c}$) [pc cm$^{-3}$]")
        note = "TNG300 abundance" if target == "all" else "TNG300 hydro abundance"
        ax.text(0.97, 0.97, note, transform=ax.transAxes, fontsize=9.5, color="0.25", va="top", ha="right")
    for bx in axB[:3]:
        bx.axhline(0, color="0.5", lw=0.8)
        bx.set_ylim(-100, 100)
        bx.grid(color="0.9", lw=0.6)
    axB[0].set_ylabel("model / TNG300 - 1 [%]")
    for ax in axT[:3]:
        ax.tick_params(labelbottom=False)
        ax.grid(color="0.9", lw=0.6)

    # (d) mean DM per ray by mass decade, z_s = 1
    md = json.loads((R.OUT / "analysis" / "mass_decomposition.json").read_text())["decades"]
    keys = [("1e9-1e10", "9-10"), ("1e10-1e11", "10-11"), ("1e11-1e12", "11-12"), ("1e12-1e13", "12-13"),
            ("1e13-9e14 reinterpreted", "13-\n14.95"), ("1e14-1e15 nominal", "14-15"), ("9e14-9e15 reinterpreted", "14.95-\n15.95")]
    ax = axT[3]
    A = R.low_mass_abundance()
    xs = np.arange(len(keys))
    for i, (k, _) in enumerate(keys):
        ax.plot(i, md[k]["tng"]["1.0"]["mean_all"], marker="_", ms=22, mew=3, color="k", zorder=5)
    for jm, m in enumerate(ORDER):
        ys = []
        for i, (k, _) in enumerate(keys):
            lo = md[k]["window_msun"][0]
            if lo < 1e13:
                fac = A[int(round(np.log10(lo)))]
                v = md[k]["model"][f"{m}|tinker08"]["1.0"]["mean_all"] * fac
            else:
                v = md[k]["model"][f"{m}|tng300"]["1.0"]["mean_all"]
            ys.append(v)
        off = (jm - 2) * 0.12
        ax.plot(xs + off, ys, ls="none", marker="o" if m != "lee22_corr_fn5p57" else "D", ms=6.5 if m != "lee22_corr_fn5p57" else 5.5,
                mfc=COL[m] if m != "lee22_corr_fn5p57" else "white", mec=COL[m], mew=1.6)
    ax.set_yscale("log"); ax.set_ylim(0.02, 300)
    ax.set_xticks(xs); ax.set_xticklabels([k[1] for k in keys], fontsize=9)
    ax.set_xlim(-0.5, len(keys) - 0.5)
    ax.axvspan(4.5, 6.5, color="0.94", zorder=0)
    ax.text(5.5, 0.03, "cluster window:\nnominal | x9", ha="center", va="bottom", fontsize=9, color="0.3")
    ax.set_ylabel("mean DM over all rays [pc cm$^{-3}$]")
    ax.set_title("(d) mean DM by mass window, $z_s$ = 1", loc="left")
    ax.set_xlabel(r"$\log_{10} M_{200c}$ [$M_\odot$] window", fontsize=10.5, labelpad=1)
    ax.grid(axis="y", color="0.9", lw=0.6)
    ax.text(0.02, 0.97, "black bars: TNG300 (window differences)\nmodels: TNG300 abundance", transform=ax.transAxes, fontsize=9, color="0.25", va="top")

    # (h) redshift dependence of the mean, reinterpreted window
    rows = list(csv.DictReader(open(R.OUT / "analysis" / "redshift_means.csv")))
    zz = np.array([float(r["z_s"]) for r in rows])
    bx = axB[3]
    for m in ORDER:
        rat = np.array([float(r[f"{m}|reinterp_tng300|mean"]) / float(r["tng_13to14_mean"]) for r in rows])
        bx.plot(zz, rat, color=COL[m], ls=LS[m], lw=1.8, marker="o", ms=3.5)
    bx.axhline(1, color="0.5", lw=0.8)
    bx.set_yscale("log"); bx.set_ylim(0.6, 3.5)
    bx.set_yticks([0.7, 1, 1.5, 2, 3]); bx.set_yticklabels(["0.7", "1", "1.5", "2", "3"]); bx.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    bx.set_xlabel("source redshift $z_s$"); bx.set_ylabel("model / TNG300 mean")
    bx.set_title("(h) [1e13, 9e14] window: mean vs $z_s$", loc="left", fontsize=10.5, pad=3)
    bx.grid(color="0.9", lw=0.6)

    handles = [Line2D([], [], color="k", lw=2.0, label="IllustrisTNG300 (Konietzka+25 rays)")]
    handles += [Line2D([], [], color=COL[m], ls=LS[m], lw=2.0, label=LABEL[m]) for m in ORDER]
    handles += [Line2D([], [], color=COL["bfc"], lw=1.6, ls=(0, (1.2, 1.6)), label="Torkamani+26 Fig. 6 (published curve)"),
                Line2D([], [], color=COL["macquart"], lw=1.6, ls=(0, (6, 2)), label="Macquart form, F = 0.182 (Medlock+24)")]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, fontsize=10.5, bbox_to_anchor=(0.5, 0.995),
               columnspacing=1.6, handlelength=2.6)
    for ext in ("png", "pdf"):
        fig.savefig(R.OUT / f"fig_analytic_tng_repro.{ext}", dpi=150)
    print("wrote", R.OUT / "fig_analytic_tng_repro.png")


if __name__ == "__main__":
    main()
