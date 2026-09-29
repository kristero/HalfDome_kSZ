#!/usr/bin/env python3
"""Halo-DM PDFs with the TNG-calibrated Lee22 against IllustrisTNG (Konietzka within-R200 catalogue) and the previous
models, from the idark ray products of 2026-09-29 (120k HEALPix NSIDE-4096 rays, seed 42, gas inside the R200c sphere,
the same rays for every source redshift and model). Panels: 1e13-1e14 Msun at z_s = 0.1, 0.5, 1 and all halos at z_s = 1.
Percent panels: 100 (model - TNG)/TNG per bin where both samples hold >= 10 rays (thick), merged bins elsewhere (thin).
Run from the repository root:  python frb_map_generation/lee22_tng_correction/plot_tngcal_pdfs.py"""
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
from frb_map_generation.make_publication_comparisons import (  # noqa: E402
    percent_difference_extended, merged_percent_difference, ROBUST_BIN_COUNT)

OUT = ROOT / "frb_map_generation/outputs/lee22_tngcal_20260929"
RAYS = OUT / "rays"
LOCAL_Z1 = {"b16": "zsrc1p0_nside4096_nrays120000_allhalos_sphere1p0_m200c_b16_seed42",
            "lee22_xgpnative": "zsrc1p0_nside4096_nrays120000_allhalos_sphere1p0_m200c_lee22_noconc_xgpnative_seed42"}
MODELS = (("b16", "#0072B2", "-.", 2.0, "Battaglia16"),
          ("lee22_xgpnative", "#E69F00", (0, (6, 3)), 2.2, "Lee22 × 0.6016 (previous production)"),
          ("lee22_tngcal", "#009E73", "-", 2.8, "Lee22 × $C(z)$ (TNG-calibrated)"))
PANELS = ((0.1, "m1e13_to_1e14", "m1e13_to_1e14", r"$z_s=0.1$,  $10^{13}$–$10^{14}\,M_\odot$"),
          (0.5, "m1e13_to_1e14", "m1e13_to_1e14", r"$z_s=0.5$,  $10^{13}$–$10^{14}\,M_\odot$"),
          (1.0, "m1e13_to_1e14", "m1e13_to_1e14", r"$z_s=1$,  $10^{13}$–$10^{14}\,M_\odot$"),
          (1.0, "m1e12_to_1e14", "m1e12_to_1e14", r"$z_s=1$,  $10^{12}$–$10^{14}\,M_\odot$ (extrapolated)"))
RC = {"font.family": "DejaVu Sans", "font.size": 13, "axes.labelsize": 14.5, "axes.titlesize": 14,
      "xtick.labelsize": 11.5, "ytick.labelsize": 11.5, "legend.fontsize": 12.5, "xtick.direction": "in",
      "ytick.direction": "in", "savefig.facecolor": "white", "pdf.fonttype": 42}


def run_dir(model, zs):
    return RAYS / "zsrc{}_nside4096_nrays120000_allhalos_sphere1p0_m200c_{}_seed42".format(("%.1f" % zs).replace(".", "p"), model)


def load(path, expect):
    text = lambda v: v.decode() if isinstance(v, bytes) else str(v)
    with h5py.File(path / "halfdome_uniform_fixedz_foreground_mass_histograms.h5", "r") as h:
        attrs = {k: text(v) for k, v in h.attrs.items()}
        labels = [text(v) for v in h["window_label"][:]]
        edges = np.asarray(h["pdf_bin_edges_pc_cm3"][:], float)
        density = np.asarray(h["pdf_density_per_pc_cm3"][:], float)
        counts = np.asarray(h["pdf_count"][:], np.int64)
    if density.shape[0] != len(labels):
        density, counts = density.T, counts.T
    for k, v in expect.items():
        if attrs.get(k) != v:
            raise ValueError("provenance {} = {!r}, expected {!r} ({})".format(k, attrs.get(k), v, path))
    if int(attrs["provenance_catalog_streamed_halos"]) != int(attrs["provenance_catalog_total_halos"]):
        raise ValueError("incomplete catalogue pass: " + str(path))
    return {lab: dict(edges=edges, pdf=density[i], counts=counts[i], zero=1 - counts[i].sum() / 120000.) for i, lab in enumerate(labels)}, attrs


def expectation(model, zs):
    base = {"provenance_halo_boundary": "spherical", "provenance_halo_boundary_sphere_r200c": "1.0", "n_rays": "120000",
            "provenance_frb_seed": "42", "provenance_nside": "4096"}
    if model == "b16":
        base["provenance_dm_profile"] = "battaglia16"
    elif model == "lee22_xgpnative":
        base.update(provenance_dm_profile="lee2022_xgpaint", provenance_lee2022_normalization="xgpaint_ne2d",
                    provenance_lee2022_redshift_scaling="physical", provenance_lee2022_n0_pivot="mcut")
    else:
        base.update(provenance_dm_profile="lee2022_xgpaint", provenance_lee2022_normalization="literal",
                    provenance_lee2022_redshift_scaling="tng_calibrated", provenance_lee2022_n0_pivot="mcut")
    return base


def stats(counts, edges, zero):
    c = counts.astype(float); cen = np.sqrt(edges[1:] * edges[:-1]); cdf = np.concatenate([[0], np.cumsum(c)]) / c.sum()
    return dict(hit_percent=100 * (1 - zero), mean_per_hit=float((c * cen).sum() / c.sum()), median_per_hit=float(np.interp(.5, cdf, edges)))


def main():
    direct = DirectComparison(ROOT)
    products = {}
    for model, *_ in MODELS:
        for zs in (0.1, 0.5, 1.0):
            prod, attrs = load(run_dir(model, zs), expectation(model, zs))
            if abs(float(attrs["provenance_source_redshift"]) - zs) > 1e-9:
                raise ValueError("source redshift mismatch " + str(run_dir(model, zs)))
            products[(model, zs)] = prod
    # reproducibility: idark z_s = 1 products against the local ones used in the earlier figures
    repro = []
    for model, run in LOCAL_Z1.items():
        local, _ = load(ROOT / "frb_map_generation/outputs" / run, {})
        for w in ("m1e13_to_1e14", "m1e12_to_1e14"):
            sa = stats(local[w]["counts"], local[w]["edges"], local[w]["zero"])
            sb = stats(products[(model, 1.0)][w]["counts"], local[w]["edges"], products[(model, 1.0)][w]["zero"])
            repro.append((model, w, sa, sb))
    rows = []
    with plt.rc_context(RC):
        fig = plt.figure(figsize=(21, 8.4))
        outer = fig.add_gridspec(1, len(PANELS), wspace=.13, left=.05, right=.99, bottom=.1, top=.8)
        for j, (zs, tlab, hlab, title) in enumerate(PANELS):
            g = outer[0, j].subgridspec(2, 1, height_ratios=(2.5, 1.35), hspace=.06)
            top = fig.add_subplot(g[0]); bot = fig.add_subplot(g[1], sharex=top)
            edges = products[("b16", zs)][hlab]["edges"]
            cen = np.sqrt(edges[1:] * edges[:-1])
            tpdf, tcnt, tzero = direct.histogram_from_values(direct.tng_values(tlab, zs), edges)
            k = tpdf > 0
            top.plot(cen[k], tpdf[k], color=".1", lw=3.2, zorder=6)
            rows.append(dict(z_s=zs, window=hlab, model="TNG (Konietzka)", **stats(tcnt, edges, tzero)))
            for model, col, ls, lw, lab in MODELS:
                p = products[(model, zs)][hlab]
                np.testing.assert_allclose(p["edges"], edges, rtol=1e-12)
                k = p["pdf"] > 0
                top.plot(cen[k], p["pdf"][k], color=col, ls=ls, lw=lw, zorder=5 if model == "lee22_tngcal" else 4)
                delta = percent_difference_extended(p["pdf"], tpdf, p["counts"], tcnt, 1)
                robust = (p["counts"] >= ROBUST_BIN_COUNT) & (tcnt >= ROBUST_BIN_COUNT)
                me, mv = merged_percent_difference(p["counts"], tcnt, edges)
                for i in range(len(mv)):
                    if np.isfinite(mv[i]):
                        bot.hlines(np.clip(mv[i], -100, 100), me[i], me[i + 1], color=col, lw=1.0, alpha=.6, ls=ls)
                bot.plot(cen, np.where(robust, delta, np.nan), color=col, ls=ls, lw=lw)
                st = stats(p["counts"], edges, p["zero"])
                rows.append(dict(z_s=zs, window=hlab, model=lab, **st,
                                 median_abs_percent_robust=float(np.nanmedian(np.abs(np.where(robust, delta, np.nan))))))
            ts = rows[-4]["mean_per_hit"]
            txt = ("rays hit: TNG %.1f%%, HalfDome %.1f%%\nmean DM per hit\nTNG %.0f" % (rows[-4]["hit_percent"], rows[-1]["hit_percent"], ts)
                   + "".join("\n%s %.0f" % (short, r["mean_per_hit"]) for short, r in zip(("B16", "Lee22 ×0.6016", "Lee22 ×C(z)"), rows[-3:])))
            top.text(.03, .04, txt, transform=top.transAxes, fontsize=11, va="bottom", color=".25", linespacing=1.25)
            top.set_title(title, pad=8)
            for ax in (top, bot):
                ax.set_xscale("log"); ax.set_xlim(.1, 5000); ax.grid(color=".9", lw=.6); ax.tick_params(which="both", top=True, right=True)
            top.set_yscale("log"); top.set_ylim(3e-7, 4e-2); top.yaxis.set_major_locator(LogLocator(numticks=6))
            top.tick_params(labelbottom=False)
            bot.axhline(0, color=".4", lw=1); bot.set_ylim(-100, 100); bot.set_yticks([-100, -50, 0, 50, 100])
            bot.set_xlabel(r"DM [pc cm$^{-3}$]")
            if j == 0:
                top.set_ylabel(r"$p(\mathrm{DM}\mid\mathrm{DM}>0)$"); bot.set_ylabel("vs TNG [%]")
            else:
                top.tick_params(labelleft=False); bot.tick_params(labelleft=False)
        handles = [Line2D([], [], color=".1", lw=3.2, label="IllustrisTNG (Konietzka, within $R_{200}$)")]
        handles += [Line2D([], [], color=c, ls=l, lw=w, label=lab) for _, c, l, w, lab in MODELS]
        fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(.5, .995), handlelength=3.2)
        fig.text(.5, .885, "HalfDome halo-only DM, gas inside each halo's $R_{200c}$ sphere, the same 120,000 rays at every source redshift "
                 "(idark, 2026-09-29).  $C(z)=0.284\\,(1+z)^{2.48}$ was fitted to TNG's mean DM of $10^{13}$–$10^{14}\\,M_\\odot$\npercent panels: thick = bins with $\\geq$10 rays in both samples, thin = merged bins",
                 ha="center", va="center", fontsize=12, color=".3", linespacing=1.4)
        OUT.joinpath("plots").mkdir(parents=True, exist_ok=True)
        for ext in ("png", "pdf"):
            fig.savefig(OUT / "plots" / ("halo_pdf_lee22_tngcal_vs_tng." + ext), dpi=170 if ext == "png" else None, bbox_inches="tight")
        plt.close(fig)
    OUT.joinpath("analysis").mkdir(parents=True, exist_ok=True)
    with open(OUT / "analysis/halo_pdf_lee22_tngcal_stats.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(dict.fromkeys(k for r in rows for k in r))); w.writeheader(); w.writerows(rows)
    for r in rows:
        print("{z_s:>4} {window:14s} {model:40s} hit {hit_percent:6.2f}%  mean/hit {mean_per_hit:7.1f}  median {median_per_hit:6.1f}".format(**r)
              + ("  median|%%| {:5.1f}".format(r["median_abs_percent_robust"]) if "median_abs_percent_robust" in r else ""))
    # Julia 1.6.2 (idark) and 1.12 (local) draw different directions from MersenneTwister(42): statistical, not bitwise, check
    for m, w, a, b in repro:
        print("reproducibility z_s=1 {} {}: hit {:.2f}% / {:.2f}%, mean/hit {:.2f} / {:.2f} (local / idark)".format(
            m, w, a["hit_percent"], b["hit_percent"], a["mean_per_hit"], b["mean_per_hit"]))


if __name__ == "__main__":
    main()
