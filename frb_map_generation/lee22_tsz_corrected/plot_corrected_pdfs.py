#!/usr/bin/env python3
"""Halo-DM PDFs of the corrected Lee22 densities against IllustrisTNG (Konietzka within-R200 catalogue) and the analytic
halo model (Torkamani et al. 2026) evaluated with the same densities (2026-09-30).
HalfDome rays: 120k NSIDE-4096 rays, seed 42, gas inside each halo's R200c sphere (idark; Battaglia16, Lee22 x0.6016 and
x C(z) from 2026-09-29, Lee22 x (1+z)^3/F_n from 2026-09-30). Analytic: clustered compound-Poisson PDF in the TNG
(Planck15) cosmology with Tinker08 abundance down to 1e9 Msun ("tng"), and with the HalfDome catalogue abundance
("validation", which must reproduce the rays). Run from the repository root:
python frb_map_generation/lee22_tsz_corrected/plot_corrected_pdfs.py"""
import csv
import json
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

OUT = ROOT / "frb_map_generation/outputs/lee22_corrected_20260930"
OLD_RAYS = ROOT / "frb_map_generation/outputs/lee22_tngcal_20260929/rays"
MODELS = (  # key, ray dir, ray tag, analytic model, colour, label
    ("xgpnative", OLD_RAYS, "lee22_xgpnative", "lee22_noconc_xgpnative", "#E69F00", "Lee22 × 0.6016 (previous production)"),
    ("tngcal", OLD_RAYS, "lee22_tngcal", "lee22_noconc_tngcal", "#56B4E9", "Lee22 × $C(z)$ (empirical, superseded)"),
    ("phys", OUT / "rays", "lee22_phys", "lee22_noconc_phys", "#009E73", "Lee22 × $[h(1+z)]^3X_H^2$ (derived, final)"),
)
REF = ("b16", OLD_RAYS, "b16", "b16", "#0072B2", "Battaglia16")
PANELS = ((0.1, "m1e13_to_1e14", r"$z_s=0.1$,  window '$10^{13}$–$10^{14}\,M_\odot$'"),
          (0.5, "m1e13_to_1e14", r"$z_s=0.5$,  window '$10^{13}$–$10^{14}\,M_\odot$'"),
          (1.0, "m1e13_to_1e14", r"$z_s=1$,  window '$10^{13}$–$10^{14}\,M_\odot$'"),
          (1.0, "m1e12_to_1e14", r"$z_s=1$,  window '$10^{12}$–$10^{14}\,M_\odot$'"))
RC = {"font.family": "DejaVu Sans", "font.size": 13, "axes.labelsize": 14.5, "axes.titlesize": 14, "xtick.labelsize": 11.5,
      "ytick.labelsize": 11.5, "legend.fontsize": 12, "xtick.direction": "in", "ytick.direction": "in",
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
    return {lab: dict(edges=edges, pdf=dens[i], counts=cnt[i]) for i, lab in enumerate(labels)}, attrs


def analytic(group, model, window, zs):
    name = ("validation_{}_catalog_{}_zs{}" if group == "validation" else "tng_{}_{}_zs{}").format(model, window, zs)
    d = np.load(str(OUT / "analytic/analysis" / (name + ".npz")))
    return dict(edges=d["edges"], pdf=d["pdf_clustered"], stats=json.loads(str(d["stats_clustered"])))


def stats(counts, edges):
    c = counts.astype(float); cen = np.sqrt(edges[1:] * edges[:-1])
    return dict(hit_percent=100 * c.sum() / 120000.0, mean_per_hit=float((c * cen).sum() / c.sum()))


def main():
    direct = DirectComparison(ROOT)
    rows = []
    with plt.rc_context(RC):
        fig = plt.figure(figsize=(21, 8.8))
        outer = fig.add_gridspec(1, len(PANELS), wspace=.13, left=.05, right=.99, bottom=.1, top=.76)
        for j, (zs, win, title) in enumerate(PANELS):
            g = outer[0, j].subgridspec(2, 1, height_ratios=(2.5, 1.35), hspace=.06)
            top = fig.add_subplot(g[0]); bot = fig.add_subplot(g[1], sharex=top)
            ref, _ = rays(REF[1], REF[2], zs)
            edges = ref[win]["edges"]; cen = np.sqrt(edges[1:] * edges[:-1])
            tpdf, tcnt, tzero = direct.histogram_from_values(direct.tng_values(win, zs), edges)
            k = tpdf > 0
            top.plot(cen[k], tpdf[k], color=".1", lw=3.2, zorder=6)
            tst = dict(hit_percent=100 * (1 - tzero), mean_per_hit=float((tcnt * cen).sum() / tcnt.sum()))
            rows.append(dict(z_s=zs, window=win, model="TNG (Konietzka)", source="TNG300", **tst))
            txt = ["mean DM per hit: TNG %.0f" % tst["mean_per_hit"], "        rays / analytic-TNG"]
            for key, folder, tag, amodel, col, lab in (REF,) + MODELS:
                p, _ = rays(folder, tag, zs)
                p = p[win]
                np.testing.assert_allclose(p["edges"], edges, rtol=1e-12)
                st = stats(p["counts"], edges)
                at = analytic("tng", amodel, win, zs); av = analytic("validation", amodel, win, zs)
                np.testing.assert_allclose(at["edges"], edges, rtol=1e-9)
                rows.append(dict(z_s=zs, window=win, model=lab, source="HalfDome rays", **st))
                rows.append(dict(z_s=zs, window=win, model=lab, source="analytic, HalfDome catalogue abundance",
                                 hit_percent=100 * av["stats"]["hit_fraction"], mean_per_hit=av["stats"]["mean_dm_hit"]))
                rows.append(dict(z_s=zs, window=win, model=lab, source="analytic, TNG cosmology + Tinker08",
                                 hit_percent=100 * at["stats"]["hit_fraction"], mean_per_hit=at["stats"]["mean_dm_hit"]))
                if key == "b16":
                    continue
                kk = p["pdf"] > 0
                top.plot(cen[kk], p["pdf"][kk], color=col, lw=2.6, zorder=5)
                ka = at["pdf"] > 1e-12
                top.plot(cen[ka], at["pdf"][ka], color=col, lw=1.8, ls=(0, (4, 2)), zorder=4)
                robust = (p["counts"] >= ROBUST_BIN_COUNT) & (tcnt >= ROBUST_BIN_COUNT)
                delta = percent_difference_extended(p["pdf"], tpdf, p["counts"], tcnt, 1)
                bot.plot(cen, np.where(robust, delta, np.nan), color=col, lw=2.2)
                da = np.where(tcnt >= ROBUST_BIN_COUNT, 100 * (at["pdf"] / np.where(tpdf > 0, tpdf, np.nan) - 1), np.nan)
                bot.plot(cen, da, color=col, lw=1.5, ls=(0, (4, 2)))
                short = {"xgpnative": "×0.6016", "tngcal": "×C(z)", "phys": "×[h(1+z)]³X²"}[key]
                txt.append("%-11s %4.0f / %4.0f" % (short, st["mean_per_hit"], at["stats"]["mean_dm_hit"]))
            txt.append("hit %%: TNG %.1f, rays %.1f, an. %.1f" % (tst["hit_percent"], st["hit_percent"], 100 * at["stats"]["hit_fraction"]))
            top.text(.03, .04, "\n".join(txt), transform=top.transAxes, fontsize=9.8, va="bottom", color=".25", family="DejaVu Sans Mono",
                     linespacing=1.25)
            top.set_title(title, pad=8)
            for ax in (top, bot):
                ax.set_xscale("log"); ax.set_xlim(.3, 5000); ax.grid(color=".9", lw=.6); ax.tick_params(which="both", top=True, right=True)
            top.set_yscale("log"); top.set_ylim(3e-7, 4e-2); top.yaxis.set_major_locator(LogLocator(numticks=6))
            top.tick_params(labelbottom=False)
            bot.axhline(0, color=".4", lw=1); bot.set_ylim(-100, 100); bot.set_yticks([-100, -50, 0, 50, 100])
            bot.set_xlabel(r"DM [pc cm$^{-3}$]")
            if j == 0:
                top.set_ylabel(r"$p(\mathrm{DM}\mid\mathrm{DM}>0)$"); bot.set_ylabel("vs TNG [%]")
            else:
                top.tick_params(labelleft=False); bot.tick_params(labelleft=False)
        h = [Line2D([], [], color=".1", lw=3.2, label="IllustrisTNG (Konietzka, within $R_{200}$)")]
        h += [Line2D([], [], color=c, lw=2.6, label=lab) for _, _, _, _, c, lab in MODELS]
        h += [Line2D([], [], color=".45", lw=2.6, label="solid: HalfDome rays"),
              Line2D([], [], color=".45", lw=1.8, ls=(0, (4, 2)), label="dashed: analytic halo model (TNG cosmology, Tinker08 to $10^9 M_\\odot$)")]
        fig.legend(handles=h, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(.5, .995), handlelength=3.0)
        fig.text(.5, .855, "Gas inside each halo's $R_{200c}$ sphere; HalfDome: the same 120,000 rays at every $z_s$, catalogue complete above "
                 "$7.3\\times10^{12}M_\\odot$ (hence the low hit fraction at $10^{12}$–$10^{14}$); analytic: Torkamani+26 compound-Poisson "
                 "PDF with clustering, same densities\nThe TNG catalogue's '$10^{13}$–$10^{14}$' window behaves as if it held halos up to "
                 "$\\sim9\\times10^{14}M_\\odot$ (its $10^{14}$–$10^{15}$ window is almost empty): its high-DM tail is not a like-for-like target", ha="center", va="center", fontsize=11.5, color=".3", linespacing=1.4)
        (OUT / "plots").mkdir(parents=True, exist_ok=True)
        for ext in ("png", "pdf"):
            fig.savefig(str(OUT / "plots" / ("halo_pdf_corrected_lee22_rays_analytic_tng." + ext)), dpi=170 if ext == "png" else None,
                        bbox_inches="tight")
        plt.close(fig)
    (OUT / "analysis").mkdir(parents=True, exist_ok=True)
    with open(str(OUT / "analysis/halo_pdf_corrected_lee22_stats.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    for r in rows:
        print("{z_s:>4} {window:14s} {model:48s} {source:40s} hit {hit_percent:6.2f}%  mean/hit {mean_per_hit:7.1f}".format(**r))


if __name__ == "__main__":
    main()
