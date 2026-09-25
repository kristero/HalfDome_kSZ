#!/usr/bin/env python3
"""The closed-form DM_cosmic PDFs used by Medlock et al. 2024 (arXiv:2403.02313, eqs. 7-8, from
Macquart et al. 2020) and the log-normal approximation tested by Torkamani et al. 2026 (Sec. 4.4),
against the IllustrisTNG300 total large-scale-structure DM (Konietzka's DMall_IGM_halo catalogue,
120,000 rays), plus the published BFC analytical curve at z = 0.7 (Torkamani Fig. 6, digitized).

These functions describe the total cosmic DM (IGM + halos), not the halo-only component, so they
are compared with TNG's total LSS DM only; HalfDome has no IGM component and cannot enter here.

  Macquart form   p(Delta) = A Delta^-beta exp[-(Delta^-alpha - C0)^2 / (2 alpha^2 sigma^2)],
                  Delta = DM / <DM>, alpha = beta = 3, C0 and A such that <Delta> = 1 and the PDF is
                  normalized; sigma = sigma_DM = F z^-1/2 (Medlock eq. 8).
                  (i)  F = 0.182, Medlock+24's CAMELS-IllustrisTNG fiducial value (measured at z = 0.5);
                  (ii) F from TNG300 itself, sigma_DM = std(DM)/mean(DM) at that z (Medlock's recipe);
                  (iii) sigma fitted to the TNG300 rays by maximum likelihood (the best this form can do).
  log-normal      with TNG300's mean and variance.
<DM> is TNG300's mean at each z in every case, so only the shapes are compared.
"""
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
from scipy import optimize

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from frb_map_generation.compare_tng_halfdome_direct import DirectComparison  # noqa: E402

OUT = HERE.parent / "outputs" / "analytic_halo_pdf_20260925"
F_MEDLOCK_TNG = 0.182
REDSHIFTS = (0.5, 0.7, 1.0)
BFC_DIGITIZED = ROOT / "frb_catalog_comparison_outputs" / "analytical_bfc_torkamani2026" / "figure6_bfc_z0p7_digitized.csv"
COL = dict(tng=".1", medlock="#CC79A7", macq_tng="#0072B2", macq_ml="#56B4E9", lognormal="#E69F00", bfc="#009E73")
LS = dict(medlock="-", macq_tng="-", macq_ml=(0, (1.2, 1.4)), lognormal="-", bfc=(0, (5, 2)))


def macquart_pdf(delta, sigma, alpha=3.0, beta=3.0):
    """Normalized Macquart+20 p(Delta) with <Delta> = 1 (C0 solved numerically)."""
    grid = np.linspace(1e-3, 12, 120001)

    def moments(c0):
        with np.errstate(over="ignore", under="ignore"):
            p = grid ** -beta * np.exp(-(grid ** -alpha - c0) ** 2 / (2 * alpha ** 2 * sigma ** 2))
        norm = np.trapz(p, grid)
        return p / norm, np.trapz(grid * p, grid) / norm

    c0 = optimize.brentq(lambda c: moments(c)[1] - 1.0, -5, 5, xtol=1e-12)
    p, _ = moments(c0)
    return np.interp(delta, grid, p, left=0, right=0), c0


def lognormal_pdf(dm, mean, std):
    s2 = np.log(1 + (std / mean) ** 2)
    mu = np.log(mean) - s2 / 2
    return np.exp(-(np.log(dm) - mu) ** 2 / (2 * s2)) / (dm * np.sqrt(2 * np.pi * s2))


def bin_average(fn, edges, n_sub=40):
    x = np.linspace(edges[0], edges[-1], (len(edges) - 1) * n_sub + 1)
    y = fn(x)
    cum = np.concatenate([[0], np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(x))])
    return np.diff(np.interp(edges, x, cum)) / np.diff(edges)


def main():
    direct = DirectComparison(ROOT)
    bfc = np.genfromtxt(BFC_DIGITIZED, delimiter=",", names=True)
    rows, summary = [], []
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 15, "axes.labelsize": 17, "axes.titlesize": 18,
                         "xtick.labelsize": 13.5, "ytick.labelsize": 13.5, "xtick.direction": "in", "ytick.direction": "in",
                         "pdf.fonttype": 42, "svg.fonttype": "none"}):
        fig = plt.figure(figsize=(19, 8.6))
        outer = fig.add_gridspec(1, len(REDSHIFTS), wspace=.16, left=.06, right=.99, bottom=.09, top=.77)
        for j, z in enumerate(REDSHIFTS):
            dm = direct.tng_values("lss_total", z)
            mean, std = dm.mean(), dm.std()
            sig_tng = std / mean
            edges = np.linspace(0.3 * mean, 2.2 * mean, 77)
            centers = 0.5 * (edges[1:] + edges[:-1])
            counts, _ = np.histogram(dm, edges)
            tpdf = counts / (len(dm) * np.diff(edges))           # normalized over all rays
            models = {}
            sig_m = F_MEDLOCK_TNG / np.sqrt(z)
            _, c0m = macquart_pdf(np.array([1.0]), sig_m)
            models["medlock"] = (bin_average(lambda x: macquart_pdf(x / mean, sig_m)[0] / mean, edges),
                                 "Macquart form, $F = 0.182$ (Medlock+24, CAMELS-TNG)")
            _, c0t = macquart_pdf(np.array([1.0]), sig_tng)
            models["macq_tng"] = (bin_average(lambda x: macquart_pdf(x / mean, sig_tng)[0] / mean, edges),
                                  "Macquart form, $F$ from TNG300 ($\\sigma_{\\rm DM}$ = std/mean)")
            sample = dm[::4] / mean
            nll = lambda sg: -np.sum(np.log(np.maximum(macquart_pdf(sample, sg)[0], 1e-300)))
            sig_ml = optimize.minimize_scalar(nll, bounds=(0.05, 1.5), method="bounded", options=dict(xatol=1e-4)).x
            models["macq_ml"] = (bin_average(lambda x: macquart_pdf(x / mean, sig_ml)[0] / mean, edges),
                                 "Macquart form, sigma maximum-likelihood fit to TNG300")
            models["lognormal"] = (bin_average(lambda x: lognormal_pdf(x, mean, std), edges), "log-normal, TNG300 mean and variance")
            if np.isclose(z, 0.7):
                inside = (edges[:-1] >= bfc["dm_pc_cm3"][0]) & (edges[1:] <= bfc["dm_pc_cm3"][-1])
                bfc_b = np.full(len(centers), np.nan)
                bfc_b[inside] = bin_average(lambda x: np.interp(x, bfc["dm_pc_cm3"], bfc["pdf_per_pc_cm3"]), edges)[inside]
                models["bfc"] = (bfc_b, "BFC analytical model (Torkamani+26 Fig. 6, digitized)")
            g = outer[0, j].subgridspec(2, 1, height_ratios=(2.5, 1.3), hspace=.06)
            top = fig.add_subplot(g[0]); bot = fig.add_subplot(g[1], sharex=top)
            top.step(centers, tpdf, where="mid", color=COL["tng"], lw=2.6)
            robust = counts >= 10
            for key, (pdf, label) in models.items():
                ls = LS[key]
                top.plot(centers, pdf, color=COL[key], lw=2.4, ls=ls)
                pct = np.where(robust & np.isfinite(pdf), 100 * (pdf / np.where(tpdf > 0, tpdf, 1) - 1), np.nan)
                bot.plot(centers, np.clip(pct, -100, 100), color=COL[key], lw=2.2, ls=ls)
                core = robust & (np.abs(centers - mean) < std)
                summary.append(dict(z=z, model=key, label=label, tng_mean=mean, tng_std=std, sigma_dm_tng=sig_tng,
                                    sigma_param=dict(medlock=sig_m, macq_tng=sig_tng, macq_ml=sig_ml).get(key, np.nan),
                                    F_tng=sig_tng * np.sqrt(z), max_abs_percent_within_1sigma=float(np.nanmax(np.abs(pct[core]))),
                                    rms_percent_robust=float(np.sqrt(np.nanmean(pct[robust] ** 2)))))
                for b in range(len(centers)):
                    rows.append(dict(z=z, model=key, dm_low=edges[b], dm_high=edges[b + 1], tng_pdf=tpdf[b], tng_count=int(counts[b]),
                                     model_pdf=pdf[b], percent=pct[b]))
            top.set_yscale("log"); top.set_ylim(tpdf[tpdf > 0].min() * .5, tpdf.max() * 2.5)
            top.set_xlim(edges[0], edges[-1])
            top.set_title("$z_s = {}$:  TNG300 $\\langle$DM$\\rangle$ = {:.0f}, $F_{{\\rm TNG300}}$ = {:.3f}".format(z, mean, sig_tng * np.sqrt(z)),
                          fontsize=15.5, pad=8)
            top.tick_params(labelbottom=False)
            bot.axhline(0, color=".45", lw=1); bot.set_ylim(-100, 100); bot.yaxis.set_major_locator(MaxNLocator(5, symmetric=True))
            for ax in (top, bot):
                ax.grid(color=".88", lw=.6)
            bot.set_xlabel(r"DM$_{\rm LSS}$ [pc cm$^{-3}$]")
            if j == 0:
                top.set_ylabel(r"$p(\mathrm{DM}_{\rm LSS})$ [(pc cm$^{-3}$)$^{-1}$]"); bot.set_ylabel("vs TNG300 [%]")
        handles = [Line2D([], [], color=COL["tng"], lw=2.6, label="IllustrisTNG300 total LSS DM (Konietzka)")]
        labels = {"medlock": "Macquart form, $F = 0.182$ (Medlock+24 CAMELS-TNG fiducial)",
                  "macq_tng": "Macquart form, $F$ from TNG300's own std/mean (same recipe)",
                  "macq_ml": "Macquart form, $\\sigma$ fitted to TNG300 (max. likelihood)",
                  "lognormal": "log-normal with TNG300's mean and variance",
                  "bfc": "BFC analytical model, Torkamani+26 Fig. 6 (digitized, $z = 0.7$)"}
        handles += [Line2D([], [], color=COL[k], lw=2.4, ls=LS[k], label=v) for k, v in labels.items()]
        fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(.5, .995), fontsize=13, handlelength=3)
        fig.text(.5, .85, "Closed-form DM$_{\\rm cosmic}$ PDFs describe the total (IGM + halo) DM, so they are compared with TNG's total LSS DM;"
                 " $\\langle$DM$\\rangle$ is TNG300's own at each $z$, so only shapes are tested", ha="center", fontsize=12, color=".25")
        for ext in ("png", "pdf", "svg"):
            fig.savefig(OUT / "plots" / ("analytic_functions_total_lss_vs_tng." + ext), dpi=190 if ext == "png" else None,
                        bbox_inches="tight", pad_inches=.15)
        plt.close(fig)
    for path, data in ((OUT / "analysis" / "analytic_functions_total_lss_bins.csv", rows),
                       (OUT / "analysis" / "analytic_functions_total_lss_summary.csv", summary)):
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
    for s in summary:
        print("z={z:.1f} {model:10s} F_TNG300={F_tng:.3f} sigma_TNG={sigma_dm_tng:.3f} sigma_param={sigma_param:.3f} max|%| within 1 sigma={max_abs_percent_within_1sigma:6.1f} "
              "rms%={rms_percent_robust:6.1f}".format(**s))


if __name__ == "__main__":
    main()
