#!/usr/bin/env python
"""Figures of group 'selection' (reads results.json and web_variants.json written by compute_selection.py).

Palette: the dataviz reference categorical order (blue, orange, aqua, yellow, magenta, green, violet, red), a
single-hue ordinal blue ramp (steps 250-700) plus one marker per bin for ordered mass bins (exploratory bin muted
grey, dashed), and a blue-grey-red diverging ramp; series identity is always backed by a legend and line style.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)
R = json.loads((HERE / "results.json").read_text())
W = json.loads((HERE / "web_variants.json").read_text())
V = R["variants"]
X = np.array(R["x_fit"], float)
EDGES = np.array(R["edges_fit"], float)
BANDS = list(R["bands"])
BINS = {b["label"]: b for b in R["bins"]}

CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
STYLES = ["-", "--", "-.", ":", (0, (5, 1, 1, 1)), (0, (3, 1, 1, 1, 1, 1)), (0, (8, 2)), (0, (1, 1))]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
# Ordered mass bins: the exploratory bin (index 0) is muted grey and dashed; the five calibrated bins use the widest
# ordinal blue steps (250/400/500/600/700) plus a distinct marker each, so identity never rests on shade alone.
SEQ6 = ["#898781", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"]
MASS_MARK = ["", "o", "s", "^", "D", "v"]
DIV = LinearSegmentedColormap.from_list("bgr", ["#104281", "#3987e5", "#cde2fb", "#f0efec", "#f6c4c3", "#e34948",
                                                "#9b2423"])
MASS6 = ["12.9-13.1", "13.5-13.7", "13.9-14.1", "14.3-14.5", "14.7-14.9", "15.1-15.3"]
MASS3 = ["13.5-13.7", "14.1-14.3", "14.7-14.9"]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 10.5, "axes.labelsize": 10,
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "legend.fontsize": 8.5, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "savefig.facecolor": "#fcfcfb", "lines.linewidth": 1.8})


def first_resolved(lab):
    e = BINS[lab]["strict_first_edge"]
    return e


def mass_label(lab):
    return rf"$10^{{{lab.split('-')[0]}}}$-$10^{{{lab.split('-')[1]}}}$" + r" M$_\odot$"


def arr(a):
    return np.array([np.nan if v is None else v for v in a], float)


def split_plot(ax, x, y, resolved, color, label, ls="-", lw=1.8, marker=None, markevery=None):
    """Resolved shells solid (given style), unresolved shells as a thin dotted continuation."""
    resolved = np.asarray(resolved, bool)
    y_r = np.where(resolved, y, np.nan)
    ax.plot(x, y, color=color, ls=":", lw=1.0, alpha=0.8)
    ax.plot(x, y_r, color=color, ls=ls, lw=lw, label=label, marker=marker or None, ms=4.5 if marker else None,
            markevery=markevery, markeredgecolor="#fcfcfb" if marker else None, markeredgewidth=0.6)


def mass_kw(k):
    """Line style of the k-th entry of MASS6 (colour + marker + dash for the exploratory bin)."""
    return {"ls": "--" if k == 0 else "-", "marker": MASS_MARK[k] or None, "markevery": (k % 3, 3)}


def xaxis(ax, lo=0.015, hi=4.2, xlabel=True):
    ax.set_xscale("log")
    ticks = [t for t in (0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 4) if lo <= t <= hi]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:g}" for t in ticks])
    ax.minorticks_off()
    ax.set_xlim(lo, hi)
    ax.set_xlabel(r"$r/R_{200c}$" if xlabel else "")


def bottom_legend(fig, ax, ncol=6, y=0.045, title=None):
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=ncol, bbox_to_anchor=(0.5, y), fontsize=8.5, title=title,
               title_fontsize=8.5)


def note(fig, text, y=0.005):
    fig.text(0.01, y, text, fontsize=7.8, color=INK2, ha="left", va="bottom")


# ============================================================================ 1. AGN-heated share
def fig_agn():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), gridspec_kw={"width_ratios": [1, 1, 0.9]})
    for ax, field, name in zip(axes[:2], ("P", "E"), ("electron pressure", "free-electron density")):
        for k, lab in enumerate(MASS6):
            d = R["agn_share_of_total"][field][lab]
            y = arr(d["shell"]) * 100
            y = np.where(y > 0, y, np.nan)
            split_plot(ax, X, y, d["resolved"], SEQ6[k], mass_label(lab) + (" (exploratory)" if k == 0 else ""),
                       **mass_kw(k))
        xaxis(ax)
        ax.set_yscale("log")
        ax.set_ylim(1e-6, 100)
        ax.minorticks_off()
        ax.set_ylabel(f"recently AGN-heated share of stacked {field} [%]")
        ax.set_title(f"({'ab'[field == 'E']}) Share of the stacked {name}")
    bottom_legend(fig, axes[0], title=None)
    ax = axes[2]
    lm = []
    for lab, b in BINS.items():
        lm.append(b["logm"])
    lm = np.array(lm)
    for j, (field, ann, lab_txt) in enumerate([("P", "0-1", "P, inside R200c"), ("P", "0-4", "P, inside 4 R200c"),
                                               ("E", "0-1", "n_e, inside R200c"), ("E", "0-4", "n_e, inside 4 R200c")]):
        y = np.array([R["agn_share_of_total"][field][lab]["integral"][ann]["value"] for lab in BINS]) * 100
        e = np.array([R["agn_share_of_total"][field][lab]["integral"][ann]["err"] or 0 for lab in BINS]) * 100
        ax.errorbar(lm, y, yerr=e, color=CAT[j // 2], marker=MARKERS[j % 2], ls=STYLES[j % 2], ms=5, lw=1.4,
                    capsize=0, label=lab_txt)
    ax.set_yscale("log")
    ax.axvspan(12.45, 13.5, color="#f0efec", zorder=0)
    ax.text(12.55, 2e-4, "exploratory\n(< 10$^{13.5}$)", fontsize=8, color=INK2)
    ax.set_xlabel(r"inclusion-weighted mean log$_{10}$ M$_{200c}$ [M$_\odot$]")
    ax.set_ylabel("share of the volume integral [%]")
    ax.set_title("(c) Integrated share (Y or N$_e$ inside the radius)")
    ax.legend(loc="upper right", fontsize=7.8)
    note(fig, "Share = AGN-only stack / (reference + AGN-only stack), inclusion-weighted means, paired bootstrap "
              "(200). Dotted segments: shells failing the strict resolution mask; gaps: no recently heated particle "
              "in any halo of the bin at that radius (share exactly 0).")
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(FIG / "agn_heated_share.png", dpi=150)
    plt.close(fig)


# ============================================================================ 2. phase
def fig_phase():
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.6))
    for ax, field, ttl in zip(axes[:3], ("E", "G", "P"), ("(a) free electrons", "(b) gas mass", "(c) pressure")):
        for k, lab in enumerate(MASS6):
            d = R["cold_sf_share_of_total"][field][lab]
            y = arr(d["shell"]) * 100
            if field == "P":
                y = np.where(y > 0, y, np.nan)
            split_plot(ax, X, y, d["resolved"], SEQ6[k], mass_label(lab) + (" (exploratory)" if k == 0 else ""), **mass_kw(k))
        xaxis(ax)
        ax.set_yscale("log")
        ax.set_ylim((1e-6, 30) if field == "P" else (0.05, 100))
        ax.set_ylabel(f"cold/star-forming share of stacked {field} [%]")
        ax.set_title(f"{ttl}: share of T < 1e5 K or SFR > 0 gas")
    ax = axes[3]
    for k, lab in enumerate(MASS6):
        d = V["selection:xray"]["shell_ratio"]["T_e"][lab]
        y = (arr(d["ratio"]) - 1) * 100
        split_plot(ax, X, y, d["resolved"], SEQ6[k], mass_label(lab) + (" (exploratory)" if k == 0 else ""), **mass_kw(k))
    xaxis(ax)
    ax.set_ylim(-1, 12)
    ax.set_ylabel(r"$T_e$(X-ray phase) / $T_e$(all) $-$ 1 [%]")
    ax.set_title("(d) electron-weighted temperature, X-ray phase")
    note(fig, "Cold/SF phase = star-forming or T < 1e5 K gas, not recently AGN-heated; reference = all gas with "
              "AGN-heated removed = X-ray phase + cold/SF phase. Dotted: not strict-resolved; dashed: exploratory bin.")
    bottom_legend(fig, axes[0])
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(FIG / "gas_phase.png", dpi=150)
    plt.close(fig)


# ============================================================================ 3. electron counting
def fig_electrons():
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))
    for ax, vid, ttl, yl in ((axes[0], "selection:e_fullion", "(a) fully ionised H+He vs SWIFT n$_e$",
                              (-1, 5)),
                             (axes[1], "selection:e_gas_muxgp", r"(b) gas density / $\mu_{\rm XGP}$ vs SWIFT n$_e$",
                              (-12, 2))):
        for k, lab in enumerate(MASS6):
            d = V[vid]["shell_ratio"]["E"][lab]
            y = (arr(d["ratio"]) - 1) * 100
            split_plot(ax, X, y, d["resolved"], SEQ6[k], mass_label(lab) + (" (exploratory)" if k == 0 else ""), **mass_kw(k))
        xaxis(ax)
        ax.set_ylim(*yl)
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_ylabel("stacked n$_e$ ratio to the reference $-$ 1 [%]")
        ax.set_title(ttl)
    bottom_legend(fig, axes[0])
    ax = axes[2]
    lm = np.array([b["logm"] for b in BINS.values()])
    for j, (vid, txt) in enumerate((("selection:e_fullion", "fully ionised H+He"),
                                    ("selection:e_gas_muxgp", r"gas / $\mu_{\rm XGP}$"),
                                    ("selection:xray", "X-ray phase only (cold/SF dropped)"))):
        y = np.array([V[vid]["integral_ratio"]["E"][lab]["0-4"]["dev"] for lab in BINS]) * 100
        e = np.array([V[vid]["integral_ratio"]["E"][lab]["0-4"]["err"] for lab in BINS]) * 100
        ax.errorbar(lm, y, yerr=e, color=CAT[j], marker=MARKERS[j], ls=STYLES[j], ms=5, lw=1.4, label=txt)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.axvspan(12.45, 13.5, color="#f0efec", zorder=0)
    ax.set_xlabel(r"inclusion-weighted mean log$_{10}$ M$_{200c}$ [M$_\odot$]")
    ax.set_ylabel(r"N$_e(<4R_{200c})$ ratio to the reference $-$ 1 [%]")
    ax.set_title("(c) integrated electrons inside 4 R200c")
    ax.legend(loc="center right", fontsize=8)
    note(fig, r"Reference: SWIFT ElectronNumberDensities. $\mu_{\rm XGP} = (m_e + 2m_p/(1+0.76))/0.9 = 1.263\,m_p$ "
              "(XGPaint). Dotted: not strict-resolved; dashed: exploratory bin.")
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(FIG / "electron_counting.png", dpi=150)
    plt.close(fig)


# ============================================================================ 4. owner split
def fig_owner_split():
    comps = [("central_bound", "bound to the central"), ("own_satellites", "bound to own satellites"),
             ("unbound", "unbound (diffuse)"), ("bound_to_other_haloes", "bound to other haloes")]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)
    for i, field in enumerate(("P", "E")):
        for j, lab in enumerate(MASS3):
            ax = axes[i, j]
            o = R["owner_split"][field][lab]
            ys = [np.clip(np.nan_to_num(arr(o[c]["shell"])), 0, None) * 100 for c, _ in comps]
            ax.stackplot(X, *ys, colors=CAT[:4], labels=[t for _, t in comps], edgecolor="#fcfcfb", linewidth=0.8)
            xaxis(ax, 0.02, 4.0, xlabel=(i == 1))
            ax.set_ylim(0, 100)
            fe = first_resolved(lab)
            if fe:
                ax.axvline(fe, color="#ffffff", ls="--", lw=1.2)
            ax.set_title(f"{'pressure' if field == 'P' else 'free electrons'}, {mass_label(lab)}")
            if j == 0:
                ax.set_ylabel(f"share of stacked {field} [%]")
            if i == 0:
                ax.set_xlabel("")
    bottom_legend(fig, axes[0, 0], ncol=4, y=0.03)
    note(fig, "White dashed line: first shell passing the strict resolution mask. HBT+ bound membership: central-bound = central_bound; own satellites = host_bound - central_bound; "
              "unbound = no_other - host_bound; other haloes = all - no_other (all recent-AGN removed).")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(FIG / "owner_split.png", dpi=150)
    plt.close(fig)


# ============================================================================ 5. ownership ratios
def fig_ownership_ratios(which, fname, sel, ylog, ylim):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)
    for i, field in enumerate(("P", "E")):
        for j, lab in enumerate(MASS3):
            ax = axes[i, j]
            for k, (s, txt) in enumerate(sel):
                d = V["selection:" + s]["shell_ratio"][field][lab]
                y = arr(d["ratio"])
                e = arr(d["err"])
                res = np.asarray(d["resolved"], bool)
                split_plot(ax, X, y, res, CAT[k], txt, ls=STYLES[k])
                ax.fill_between(X, np.where(res, y - e, np.nan), np.where(res, y + e, np.nan), color=CAT[k],
                                alpha=0.18, lw=0)
            ax.axhline(1, color=MUTED, lw=0.8)
            xaxis(ax, 0.03, 4.0, xlabel=(i == 1))
            if ylog:
                ax.set_yscale("log")
            ax.set_ylim(*ylim)
            ax.set_title(f"{'pressure' if field == 'P' else 'free electrons'}, {mass_label(lab)}")
            if j == 0:
                ax.set_ylabel(f"stacked {field}: selection / all gas")
    axes[0, 0].legend(loc="lower left", fontsize=8, frameon=True, framealpha=0.9)
    note(fig, f"{which}. Bands: paired-bootstrap 1-sigma of the ratio (200 replicates); dotted: shells failing the "
              "strict mask. Reference = all gas, recent-AGN removed.")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(FIG / fname, dpi=150)
    plt.close(fig)


# ============================================================================ 6. heatmaps of band ratios
OWN_ROWS = ["central_bound", "host_bound", "no_satellites_or_other", "no_other", "rm_ov_mu0.03", "rm_ov_mu0.1",
            "rm_ov_mu0.3", "rm_ov_mu0.1_keepsat", "rm_ov_mu0.1_reach1R", "A2i", "A2ii", "A2ii_A2i", "exc_all_k0.5",
            "exc_all_k1", "exc_all_k2", "exc_nbr_k1"]


def fig_heatmap(field):
    fig, axes = plt.subplots(1, 3, figsize=(17, 7.2), sharey=True)
    norm = TwoSlopeNorm(vmin=-30, vcenter=0, vmax=30)
    for ax, lab in zip(axes, MASS3):
        M = np.full((len(OWN_ROWS), len(BANDS)), np.nan)
        Emat = np.full_like(M, np.nan)
        for r_, s in enumerate(OWN_ROWS):
            for c_, bd in enumerate(BANDS):
                d = V["selection:" + s]["band_ratio"][field][lab][bd]
                if d:
                    M[r_, c_] = d["dev"] * 100
                    Emat[r_, c_] = d["err"] * 100 if d["err"] is not None else np.nan
        im = ax.imshow(np.clip(M, -30, 30), cmap=DIV, norm=norm, aspect="auto")
        for r_ in range(M.shape[0]):
            for c_ in range(M.shape[1]):
                if np.isfinite(M[r_, c_]):
                    v = M[r_, c_]
                    txt = "0" if abs(v) < 0.05 else (f"{v:+.1f}" if abs(v) < 99.5 else f"{v:+.0f}")
                    ax.text(c_, r_, txt, ha="center", va="center", fontsize=7.2,
                            color="#ffffff" if abs(v) > 20 else INK)
                else:
                    ax.text(c_, r_, "n/r", ha="center", va="center", fontsize=6.5, color=MUTED)
        ax.set_xticks(range(len(BANDS)))
        ax.set_xticklabels(BANDS, rotation=0, fontsize=8)
        ax.set_xlabel(r"radial band [$R_{200c}$]")
        ax.set_yticks(range(len(OWN_ROWS)))
        ax.set_yticklabels([V["selection:" + s]["label"] for s in OWN_ROWS], fontsize=8.2)
        ax.set_title(f"{mass_label(lab)} (N = {BINS[lab]['N']})")
        ax.grid(False)
    cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.01)
    cb.set_label(f"stacked {field}: selection / all gas $-$ 1 [%] (colour clipped at $\\pm$30)")
    fig.suptitle(f"Ownership: median shell ratio per radial band, {'pressure' if field == 'P' else 'free electrons'} "
                 "(n/r = no strict-resolved shell)", fontsize=11)
    fig.savefig(FIG / f"ownership_band_heatmap_{field}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ============================================================================ 7. sample (overlap classes)
def fig_sample():
    sel = [("clean", "CLEAN (no overlapper)"), ("overlap_in1R", "overlapper inside 1R"),
           ("overlap_annulus", "overlapper in 1-2R only")]
    fig, axes = plt.subplots(3, 3, figsize=(15, 11), sharex=True)
    for i, field in enumerate(("P", "E", "T_e")):
        for j, lab in enumerate(MASS3):
            ax = axes[i, j]
            for k, (s, txt) in enumerate(sel):
                d = V["selection:" + s]["shell_ratio"][field][lab]
                y = arr(d["ratio"])
                e = arr(d["err"])
                res = np.asarray(d["resolved"], bool)
                split_plot(ax, X, y, res, CAT[k], txt, ls=STYLES[k])
                ax.fill_between(X, np.where(res, y - e, np.nan), np.where(res, y + e, np.nan), color=CAT[k],
                                alpha=0.18, lw=0)
            ax.axhline(1, color=MUTED, lw=0.8)
            xaxis(ax, 0.03, 4.0, xlabel=(i == 2))
            ax.set_yscale("log")
            ax.set_yticks([0.6, 0.8, 1, 1.25, 1.6, 2, 3] if field != "T_e" else [0.8, 0.9, 1, 1.1, 1.25, 1.5, 1.8])
            ax.set_ylim((0.55, 3.6) if field != "T_e" else (0.75, 1.9))
            ax.set_yticklabels([str(t) for t in ax.get_yticks()])
            ax.minorticks_off()
            ns = "/".join(str(V["selection:" + s]["shell_ratio"][field][lab]["N"]) for s, _ in sel)
            ax.set_title(f"{ {'P': 'pressure', 'E': 'free electrons', 'T_e': 'electron temperature'}[field]}, "
                         f"{mass_label(lab)}" + (f"\nN = {ns} (clean / in 1R / 1-2R)" if i == 0 else ""))
            if j == 0:
                ax.set_ylabel(f"stacked {field}: subset / all haloes")
    note(fig, "Row subsets of the same bin with their inclusion weights; bands = paired-bootstrap 1-sigma (the full "
              "bin's design-stratified resamples restricted to the subset). Dotted: not strict-resolved.")
    bottom_legend(fig, axes[0, 0], ncol=3, y=0.02)
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    fig.savefig(FIG / "sample_overlap_classes.png", dpi=150)
    plt.close(fig)


# ============================================================================ 8. integrals vs mass
def fig_integrals():
    sel = [("A2ii", "A2-ii partition"), ("A2i", "A2-i"), ("rm_ov_mu0.1", "remove overlappers mu>=0.1"),
           ("exc_all_k1", "excision kappa=1"), ("clean", "CLEAN haloes only")]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    lm = np.array([b["logm"] for b in BINS.values()])
    for ax, field, ttl in zip(axes, ("P", "E"), (r"(a) $Y(<4R_{200c})$", r"(b) $N_e(<4R_{200c})$")):
        for k, (s, txt) in enumerate(sel):
            y = np.array([V["selection:" + s]["integral_ratio"][field][lab]["0-4"]["dev"] for lab in BINS]) * 100
            e = np.array([V["selection:" + s]["integral_ratio"][field][lab]["0-4"]["err"] for lab in BINS]) * 100
            ax.errorbar(lm + (k - 2) * 0.012, y, yerr=e, color=CAT[k], marker=MARKERS[k], ls=STYLES[k], ms=5, lw=1.3,
                        label=txt)
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.axvspan(12.45, 13.5, color="#f0efec", zorder=0)
        ax.set_ylim(-40, 10)
        ax.set_xlabel(r"inclusion-weighted mean log$_{10}$ M$_{200c}$ [M$_\odot$]")
        ax.set_title(ttl + ": selection / all gas")
    axes[0].set_ylabel("ratio of the stacked volume integral $-$ 1 [%]")
    axes[1].legend(loc="lower right", fontsize=8)
    note(fig, "Volume integral of the stacked (inclusion-weighted mean) normalised profile over 0-4 R200c; error bars: "
              "paired bootstrap. Shaded: exploratory bins below 10^13.5.")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(FIG / "integrated_ratio_vs_mass.png", dpi=150)
    plt.close(fig)


# ============================================================================ 9. observable impact
IMPACT_ROWS = ["selection:agn_kept", "selection:xray", "selection:e_fullion", "selection:e_gas_muxgp",
               "selection:A2ii", "selection:A2i", "selection:A2ii_A2i", "selection:no_other",
               "selection:rm_ov_mu0.1_keepsat", "selection:exc_nbr_k1", "selection:rm_ov_mu0.03",
               "selection:rm_ov_mu0.1", "selection:rm_ov_mu0.3", "selection:rm_ov_mu0.1_reach1R",
               "selection:exc_all_k0.5", "selection:exc_all_k1", "selection:exc_all_k2",
               "selection:no_satellites_or_other", "selection:host_bound", "selection:central_bound",
               "selection:cb_xray", "selection:clean", "selection:clean_mu0.03", "selection:clean_mu0.3",
               "selection:overlapped", "selection:overlap_in1R", "selection:overlap_annulus"]
IMPACT_COLS = [("P", "tSZ_yy_total_ell100", r"tSZ $C_\ell$, $\ell$=100"),
               ("P", "tSZ_yy_total_ell1000", r"tSZ $C_\ell$, $\ell$=1000"),
               ("P", "tSZ_yy_total_ell3000", r"tSZ $C_\ell$, $\ell$=3000"),
               ("E", "kSZ_1h_ell1000", r"kSZ 1h, $\ell$=1000"), ("E", "DM_mean", "mean DM"),
               ("E", "DM_var", "DM variance")]


def fig_impact():
    rows = IMPACT_ROWS
    fig, axes = plt.subplots(1, len(IMPACT_COLS), figsize=(19, 9.5), sharey=True)
    ypos = np.arange(len(rows))[::-1]
    for ax, (field, obs, ttl) in zip(axes, IMPACT_COLS):
        for yy, vid in zip(ypos, rows):
            d = V[vid]["impact"].get(field, {}).get(obs)
            if d is None:
                ax.text(0, yy, "n/a", ha="center", va="center", fontsize=7, color=MUTED)
                continue
            v, e = d["dO_over_O_pct"], d["err_pct"] or 0
            ax.errorbar([v], [yy], xerr=[e], color=CAT[0], marker="o", ms=4.5, lw=1.4, capsize=0)
            ax.text(1.0, yy, f"{v:+.2f}", transform=ax.get_yaxis_transform(), ha="right", va="center",
                    fontsize=6.8, color=INK2)
        ax.axvline(0, color=MUTED, lw=0.8)
        ax.set_xscale("symlog", linthresh=0.1, linscale=0.6)
        ax.set_xlim(-120, 900)
        ax.set_xticks([-100, -10, -1, -0.1, 0, 0.1, 1, 10])
        ax.set_xticklabels(["-100", "-10", "-1", "-0.1", "0", "0.1", "1", "10"], fontsize=7.5)
        ax.set_title(ttl, fontsize=9.5)
        ax.set_xlabel("change [%]")
        for k in range(len(rows)):
            if k % 2 == 0:
                ax.axhspan(k - 0.5, k + 0.5, color="#f3f2ee", zorder=0, lw=0)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(ypos)
    axes[0].set_yticklabels([V[v]["label"] for v in rows], fontsize=8.5)
    axes[0].set_ylim(-0.6, len(rows) - 0.4)
    fig.suptitle("First-order observable change from calibrating the 10 bins 10^13.5-15.5 on each selection "
                 "(sc.impact_from_stacks; error = paired bootstrap SD)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(FIG / "observable_impact.png", dpi=150)
    plt.close(fig)


# ============================================================================ 10. example profiles from the website records
def fig_profiles():
    recs = {(r["variant"], r["field"], r["bin"]): r for r in W["records"]}
    sel = [("ref", "all gas (reference)"), ("selection:A2ii", "A2-ii partition"),
           ("selection:no_other", "no gas bound to other haloes"), ("selection:host_bound", "host-bound"),
           ("selection:central_bound", "central-bound")]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    lab = "14.1-14.3"
    for ax, field, yl in zip(axes, ("P", "E"), (r"$P_e/P_{200}$", r"$n_e\mu_{\rm XGP}/(\rho_c f_b)$")):
        for k, (vid, txt) in enumerate(sel):
            r = recs[(vid, field, lab)]
            x = np.array(r["x"])
            y = arr(r["value"])
            y = np.where(y > 0, y, np.nan)
            split_plot(ax, x, y, r["resolved"], CAT[k], txt, ls=STYLES[k])
        xaxis(ax, 0.03, 4.0)
        ax.set_yscale("log")
        ax.set_ylabel(yl + " (stacked mean)")
        ax.set_title(f"{'pressure' if field == 'P' else 'free electrons'}, {mass_label(lab)}")
    axes[0].legend(loc="lower left", fontsize=8)
    note(fig, "Inclusion-weighted mean profiles from web_variants.json; dotted: not strict-resolved.")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(FIG / "profiles_ownership_example.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    fig_agn()
    fig_phase()
    fig_electrons()
    fig_owner_split()
    fig_ownership_ratios("Bound-membership selections", "ownership_membership_ratio.png",
                         [("central_bound", "central-bound only"), ("host_bound", "host-bound (central + satellites)"),
                          ("no_satellites_or_other", "central-bound + unbound"),
                          ("no_other", "no gas bound to other haloes")], True, (0.003, 1.3))
    fig_ownership_ratios("Neighbour / overlapper handling", "ownership_neighbour_ratio.png",
                         [("A2ii", "A2-ii partition"), ("A2i", "A2-i (painted-bound removed)"),
                          ("rm_ov_mu0.1", "remove overlappers mu>=0.1"),
                          ("rm_ov_mu0.1_keepsat", "remove neighbour overlappers"),
                          ("exc_all_k1", "excision kappa=1")], False, (0.3, 1.12))
    fig_heatmap("P")
    fig_heatmap("E")
    fig_sample()
    fig_integrals()
    fig_impact()
    fig_profiles()
    print(sorted(p.name for p in FIG.glob("*.png")))
