#!/usr/bin/env python3
"""Halo-DM PDFs in the '10^13-10^14 Msun' window at six source redshifts (2026-10-01): IllustrisTNG (Konietzka within-R200
catalogue) against HalfDome rays with Battaglia16, Lee22 x 0.6016 (unedited XGPaint ne2d, previous production) and Lee22
corrected (x [h_TNG (1+z)]^3 X_H^2, derived, no empirical factor). HalfDome: 120,000 NSIDE-4096 rays per z_s, gas inside each
halo's R200c sphere, seed 42, the same rays for the three models at a given z_s: z_s <= 1 from idark (Julia 1.6.2;
lee22_tngcal_20260929 and lee22_corrected_20260930), z_s >= 1.5 from run_pdf_zseries_local.sh (Julia 1.12.2, whose RNG draws
other directions; a local rerun at z_s = 1 agrees statistically, validation/). The TNG catalogue has z_s = 0.1-5.5 in steps of 0.1; HalfDome's lightcone ends at z = 3.86, so six z_s are shown.
Writes plots/halo_pdf_zseries_m1e13_1e14.{png,pdf} and analysis/halo_pdf_zseries_stats.csv. Run from the repository root:
LD_LIBRARY_PATH=<env>/lib python frb_map_generation/lee22_tsz_corrected/plot_pdf_zseries.py"""
import csv
import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from frb_map_generation.compare_tng_halfdome_direct import DirectComparison  # noqa: E402
from frb_map_generation.make_publication_comparisons import percent_difference_extended, ROBUST_BIN_COUNT  # noqa: E402

OUT = ROOT / "frb_map_generation/outputs/lee22_pdf_zseries_20261001"
IDARK = {"b16": ROOT / "frb_map_generation/outputs/lee22_tngcal_20260929/rays",
         "lee22_xgpnative": ROOT / "frb_map_generation/outputs/lee22_tngcal_20260929/rays",
         "lee22_phys": ROOT / "frb_map_generation/outputs/lee22_corrected_20260930/rays"}
ZS = (0.1, 0.5, 1.0, 1.5, 2.0, 3.0)
WIN = "m1e13_to_1e14"
NRAY = 120000
MODELS = (  # ray tag, colour, line style, width, label, short
    ("b16", "#17becf", "-.", 2.4, "Battaglia16 (XGPaint)", "B16"),
    ("lee22_xgpnative", "#ff7f0e", (0, (5, 3)), 2.4, "Lee22 via unedited XGPaint (×0.6016, previous)", "×0.6016"),
    ("lee22_phys", "#9467bd", "-", 2.8, "Lee22 corrected, ×$[h(1+z)]^3X_H^2$ (derived)", "corrected"),
)
RC = {"font.family": "DejaVu Sans", "font.size": 13, "axes.labelsize": 14.5, "axes.titlesize": 14.5, "xtick.labelsize": 11.5,
      "ytick.labelsize": 11.5, "legend.fontsize": 12.5, "xtick.direction": "in", "ytick.direction": "in",
      "savefig.facecolor": "white", "pdf.fonttype": 42}


def rays(folder, tag, zs):
    path = folder / "zsrc{}_nside4096_nrays120000_allhalos_sphere1p0_m200c_{}_seed42".format(("%.1f" % zs).replace(".", "p"), tag)
    text = lambda v: v.decode() if isinstance(v, bytes) else str(v)
    with h5py.File(path / "halfdome_uniform_fixedz_foreground_mass_histograms.h5", "r") as h:
        attrs = {k: text(v) for k, v in h.attrs.items()}
        labels = [text(v) for v in h["window_label"][:]]
        edges = np.asarray(h["pdf_bin_edges_pc_cm3"][:], float)
        dens = np.asarray(h["pdf_density_per_pc_cm3"][:], float); cnt = np.asarray(h["pdf_count"][:], np.int64)
    if dens.shape[0] != len(labels):
        dens, cnt = dens.T, cnt.T
    assert abs(float(attrs["provenance_source_redshift"]) - zs) < 1e-9 and attrs["provenance_halo_boundary"] == "spherical"
    assert int(attrs["provenance_catalog_streamed_halos"]) == int(attrs["provenance_catalog_total_halos"])
    assert int(attrs["provenance_frb_seed"]) == 42
    i = labels.index(WIN)
    return dict(edges=edges, pdf=dens[i], counts=cnt[i], julia=attrs.get("provenance_julia_version", "?"))


def stats(counts, edges):
    c = counts.astype(float); cen = np.sqrt(edges[1:] * edges[:-1])
    cdf = np.cumsum(c) / c.sum()
    return dict(hit_percent=100 * c.sum() / NRAY, mean_per_hit=float((c * cen).sum() / c.sum()),
                median_per_hit=float(np.exp(np.interp(0.5, cdf, np.log(cen)))))


def main():
    direct = DirectComparison(ROOT)
    rows = []
    with plt.rc_context(RC):
        fig = plt.figure(figsize=(19, 12.6))
        outer = fig.add_gridspec(2, 3, wspace=.06, hspace=.2, left=.055, right=.99, bottom=.06, top=.865)
        for j, zs in enumerate(ZS):
            g = outer[j // 3, j % 3].subgridspec(2, 1, height_ratios=(2.5, 1.2), hspace=.05)
            top = fig.add_subplot(g[0]); bot = fig.add_subplot(g[1], sharex=top)
            src = (lambda tag: IDARK[tag]) if zs <= 1.0 else (lambda tag: OUT / "rays")
            ref = rays(src(MODELS[0][0]), MODELS[0][0], zs)
            edges = ref["edges"]; cen = np.sqrt(edges[1:] * edges[:-1])
            tpdf, tcnt, tzero = direct.histogram_from_values(direct.tng_values(WIN, zs), edges)
            k = tpdf > 0
            top.plot(cen[k], tpdf[k], color=".08", lw=3.4, zorder=6)
            tst = stats(tcnt, edges); tst["hit_percent"] = 100 * (1 - tzero)
            rows.append(dict(z_s=zs, model="IllustrisTNG (Konietzka)", source="TNG300 catalogue", **tst, mean_over_tng=1.0))
            lines = ["mean DM per hit [pc cm$^{-3}$]", "TNG {:.0f}".format(tst["mean_per_hit"])]
            for tag, col, ls, lw, lab, short in MODELS:
                p = rays(src(tag), tag, zs)
                np.testing.assert_allclose(p["edges"], edges, rtol=1e-12)
                assert p["julia"] == ("1.6.2" if zs <= 1.0 else "1.12.2"), (tag, zs, p["julia"])
                st = stats(p["counts"], edges)
                rows.append(dict(z_s=zs, model=lab, source="HalfDome rays ({})".format("idark" if zs <= 1.0 else "local"), **st,
                                 mean_over_tng=st["mean_per_hit"] / tst["mean_per_hit"]))
                kk = p["pdf"] > 0
                top.plot(cen[kk], p["pdf"][kk], color=col, ls=ls, lw=lw, zorder=5)
                robust = (p["counts"] >= ROBUST_BIN_COUNT) & (tcnt >= ROBUST_BIN_COUNT)
                delta = percent_difference_extended(p["pdf"], tpdf, p["counts"], tcnt, 1)
                bot.plot(cen, np.where(robust, delta, np.nan), color=col, ls=ls, lw=lw - .3)
                lines.append("{} {:.0f}  ({:+.0f}%)".format(short, st["mean_per_hit"], 100 * (st["mean_per_hit"] / tst["mean_per_hit"] - 1)))
            lines.append("rays hit: TNG {:.1f}%, HalfDome {:.1f}%".format(tst["hit_percent"], st["hit_percent"]))
            top.text(.035, .045, "\n".join(lines), transform=top.transAxes, fontsize=10.6, va="bottom", color=".2", linespacing=1.3,
                     bbox=dict(facecolor="white", edgecolor=".85", boxstyle="round,pad=.35", alpha=.9))
            top.set_title(r"$z_s = %g$" % zs + ("   (halos at $z>2$: Lee22 extrapolated)" if zs > 2 else ""), pad=7)
            for ax in (top, bot):
                ax.set_xscale("log"); ax.set_xlim(.3, 5000); ax.grid(color=".9", lw=.6); ax.tick_params(which="both", top=True, right=True)
            top.set_yscale("log"); top.set_ylim(3e-7, 4e-2); top.yaxis.set_major_locator(LogLocator(numticks=6))
            top.tick_params(labelbottom=False)
            bot.axhline(0, color=".35", lw=1); bot.set_ylim(-100, 100); bot.set_yticks([-100, -50, 0, 50, 100])
            if j // 3 == 1:
                bot.set_xlabel(r"DM [pc cm$^{-3}$]")
            else:
                bot.tick_params(labelbottom=False)
            if j % 3 == 0:
                top.set_ylabel(r"$p(\mathrm{DM}\mid\mathrm{DM}>0)$"); bot.set_ylabel("vs TNG [%]")
            else:
                top.tick_params(labelleft=False); bot.tick_params(labelleft=False)
        h = [Line2D([], [], color=".08", lw=3.4, label="IllustrisTNG300 (Konietzka catalogue, within $R_{200}$)")]
        h += [Line2D([], [], color=c, ls=ls, lw=lw, label=lab) for _, c, ls, lw, lab, _ in MODELS]
        fig.legend(handles=h, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(.5, 1.0), handlelength=3.2, columnspacing=2.5)
        fig.text(.5, .918, "Halo DM from halos of $10^{13}$–$10^{14}\\,M_\\odot$ only; HalfDome: 120,000 rays per $z_s$, gas inside each halo's "
                 "$R_{200c}$ sphere.\nCaveat: the TNG window behaves as if it held halos up to $\\sim9\\times10^{14}M_\\odot$ "
                 "(its $10^{14}$–$10^{15}$ window is nearly empty), so its high-DM tail is not like-for-like",
                 ha="center", va="center", fontsize=11.3, color=".3")
        (OUT / "plots").mkdir(parents=True, exist_ok=True)
        for ext in ("png", "pdf"):
            fig.savefig(str(OUT / "plots" / ("halo_pdf_zseries_m1e13_1e14." + ext)), dpi=160 if ext == "png" else None)
        plt.close(fig)
    (OUT / "analysis").mkdir(parents=True, exist_ok=True)
    with open(str(OUT / "analysis/halo_pdf_zseries_stats.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    for r in rows:
        print("{z_s:>4} {model:50s} {source:40s} hit {hit_percent:6.2f}%  mean/hit {mean_per_hit:7.1f}  median {median_per_hit:6.1f}"
              "  mean/TNG {mean_over_tng:5.2f}".format(**r))


if __name__ == "__main__":
    main()
