#!/usr/bin/env python
"""Figures of the 'estimators' group (reads stacks.npz and results.json written by compute_estimators.py)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1]
sys.path.insert(0, str(OUT / "core"))
import stackcore as sc  # noqa: E402

FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)
Z = np.load(HERE / "stacks.npz")
R = json.loads((HERE / "results.json").read_text())
X = Z["x"]
EDGES = Z["edges"]
LABELS = list(Z["labels"])
CAL = list(Z["cal_labels"])
G = "estimators"

# palette (reference instance of the dataviz skill; categorical slots in fixed order, with line-style/marker
# secondary encoding; sequential blue ramp for mass)
C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100", "magenta": "#e87ba4",
     "green": "#008300", "violet": "#4a3aa7", "red": "#e34948"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
SEQ = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
REP = ["13.5-13.7", "14.1-14.3", "14.5-14.7", "14.9-15.1"]
REP_COL = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
# five bins spanning the calibrated range for multi-bin line plots: ordinal blue ramp (steps 250-700) plus a distinct
# line style and marker per bin (secondary encoding), with direct labels at the line ends
FIVE = ["13.5-13.7", "13.9-14.1", "14.3-14.5", "14.7-14.9", "15.1-15.3"]
FIVE_COL = ["#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"]
FIVE_LS = ["-", "--", "-.", ":", (0, (5, 1.5, 1, 1.5))]
FIVE_MK = ["o", "s", "^", "D", "v"]


def five_lines(ax, getter, label_end=True, fmt="{}"):
    """Plot getter(lab) for the FIVE bins (resolved shells only) with colour, style and marker per bin."""
    for lab, col, ls, mk in zip(FIVE, FIVE_COL, FIVE_LS, FIVE_MK):
        res = resolved(lab)
        y = np.where(res, getter(lab), np.nan)
        ax.plot(X, y, color=col, ls=ls, marker=mk, ms=4.5, markevery=(1, 4), label=fmt.format(lab))
        if label_end and np.isfinite(y).any():
            k = np.flatnonzero(np.isfinite(y))[-1]
            ax.annotate(lab, (X[k], y[k]), xytext=(4, 0), textcoords="offset points", fontsize=8, color=INK2,
                        va="center")
plt.rcParams.update({"font.size": 11, "axes.titlesize": 11.5, "axes.labelsize": 11, "legend.fontsize": 10,
                     "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 150,
                     "lines.linewidth": 2.0})


def g(kind, vid, fld, lab):
    return Z[f"{kind}|{vid}|{fld}|{lab}"]


def dg(name, lab):
    return Z[f"diag|{name}|{lab}"]


def resolved(lab):
    return Z[f"res|ref|P|{lab}"].astype(bool)


def shade_unresolved(ax, lab):
    res = resolved(lab)
    if res.any():
        first = EDGES[:-1][res][0]
        if first > EDGES[1]:
            ax.axvspan(0.02, first, color="#efefec", zorder=0, lw=0)
    else:
        ax.axvspan(0.02, 4.0, color="#efefec", zorder=0, lw=0)


def xaxis(ax, label=True):
    ax.set_xscale("log")
    ax.set_xlim(0.02, 4.2)
    ax.set_xticks([0.03, 0.1, 0.3, 1, 2, 4])
    ax.set_xticklabels(["0.03", "0.1", "0.3", "1", "2", "4"])
    ax.axvline(1.0, color=INK2, lw=0.8, ls=":")
    if label:
        ax.set_xlabel(r"$r/R_{200c}$")


def band_plot(ax, y, lo, hi, col, ls="-", marker=None, label=None, res=None, alpha=0.18):
    y = np.asarray(y, float)
    if res is not None:
        yr = np.where(res, y, np.nan)
        yu = np.where(~res, y, np.nan)
        # connect the unresolved part to the first resolved point
        if res.any() and (~res).any():
            k = np.flatnonzero(res)[0]
            yu[k] = y[k]
        ax.plot(X, yu, color=col, ls=":", lw=1.2, alpha=0.8)
        ax.plot(X, yr, color=col, ls=ls, marker=marker, ms=4, markevery=3, label=label)
        if lo is not None:
            ax.fill_between(X, np.where(res, lo, np.nan), np.where(res, hi, np.nan), color=col, alpha=alpha, lw=0)
    else:
        ax.plot(X, y, color=col, ls=ls, marker=marker, ms=4, markevery=3, label=label)
        if lo is not None:
            ax.fill_between(X, lo, hi, color=col, alpha=alpha, lw=0)


def note_unres(fig, bands=True):
    fig.text(0.01, 0.005, "Grey area: shells failing the strict resolution rule for at least one halo of the bin "
             "(not plotted or dotted)." + (" Bands: 16-84% of 200 paired halo-bootstrap replicates." if bands else
                                           " Inclusion-weighted mean stacks; values in results.json -> diagnostics."),
             fontsize=9, color=INK2)


# ============================================================================ fig 1: pressure estimators
EST = [("Q_V", r"$Q_V$ (SPH volume; $Q_{V,y}$ identical)", C["blue"], "-", "o"),
       ("Q_M", r"$Q_M$ (mass weighted)", C["orange"], "--", "s"),
       ("Q_X", r"$Q_X$ (X-ray weighted)", C["violet"], "-.", "^"),
       ("nT_X", r"$n_e\,kT_X$ (X-ray route)", C["green"], (0, (5, 1.5, 1, 1.5)), "D")]
fig, axes = plt.subplots(2, 4, figsize=(16, 8.2), sharex=True)
for j, lab in enumerate(REP):
    res = resolved(lab)
    for row, ax in enumerate(axes[:, j]):
        shade_unresolved(ax, lab)
        for vid, name, col, ls, mk in EST:
            if row == 1 and vid in ("Q_M", "Q_X"):
                continue
            k = f"{G}:{vid}"
            band_plot(ax, g("ratio", k, "P", lab), g("ratio_q16", k, "P", lab), g("ratio_q84", k, "P", lab),
                      col, ls, mk, name if (j == 0 and row == 0) else None, res)
        ax.axhline(1, color=INK, lw=0.8)
        xaxis(ax, row == 1)
    f = dg("fill", lab)
    axes[1, j].plot(X, np.where(res, 1 / f, np.nan), color=INK, ls="--", lw=1.4,
                    label=r"$1/\langle f\rangle$ (inverse SPH filling factor)" if j == 0 else None)
    axes[0, j].set_yscale("log")
    axes[0, j].set_ylim(0.8, 700)
    axes[1, j].set_ylim(0.9, 1.75)
    axes[0, j].set_title(rf"$\log_{{10}}M_{{200c}}$ = {lab}  (N = {R['bin_info'][lab]['N']})")
axes[0, 0].set_ylabel("pressure estimator / geometric\n(log scale)")
axes[1, 0].set_ylabel("estimator / geometric (zoom)")
h0, l0 = axes[0, 0].get_legend_handles_labels()
h1, l1 = axes[1, 0].get_legend_handles_labels()
fig.legend(h0 + h1[-1:], l0 + l1[-1:], loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 0.02))
fig.suptitle("Electron pressure: shell estimators relative to the geometric estimator (sum y / V_shell), "
             "inclusion-weighted mean stacks", y=0.995)
fig.tight_layout(rect=(0, 0.07, 1, 0.98))
note_unres(fig)
fig.savefig(FIG / "fig1_pressure_estimators.png")
plt.close(fig)

# ============================================================================ fig 2: filling factor and clumping
tab = sc.halo_table()
Eg = np.asarray(sc.per_halo(sc.make_key("E")))
EQV = np.asarray(sc.per_halo(sc.make_key("E", "Q_V")))
Crho = np.asarray(sc.per_halo(sc.make_key("C_rho", "weighted")))
with np.errstate(divide="ignore", invalid="ignore"):
    FILL = np.where(EQV > 0, Eg / EQV, np.nan)
fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
five_lines(axes[0], lambda lb: dg("fill", lb), label_end=False)
five_lines(axes[1], lambda lb: dg("crho_med", lb), label_end=False)
lab = "14.3-14.5"
res = resolved(lab)
axes[0].fill_between(X, np.where(res, dg("fill_q16", lab), np.nan), np.where(res, dg("fill_q84", lab), np.nan),
                     color=FIVE_COL[2], alpha=0.15, lw=0, label="16-84% of haloes, 14.3-14.5")
axes[0].axhline(1, color=INK, lw=0.8)
axes[0].set_ylabel(r"SPH filling factor $f=\sum_i m_i/\rho_i\,/\,V_{\rm shell}$")
axes[0].set_ylim(0.7, 1.03)
axes[0].set_title("(a) summed SPH volume / geometric shell volume")
axes[1].set_yscale("log")
axes[1].set_ylabel(r"median density clumping $C_\rho=\langle\rho^2\rangle_V/\langle\rho\rangle_V^2$")
axes[1].set_title("(b) density clumping of the shell gas (halo median)")
for ax in axes[:2]:
    xaxis(ax)
axes[0].legend(title=r"$\log_{10}M_{200c}$ (resolved shells; other bins in results.json)", ncol=2, fontsize=8.5,
               title_fontsize=9, loc="lower left", frameon=False)
# (c) per-halo f vs C_rho at 2-3 R200c, calibrated bins, per-halo resolved
rows = np.concatenate([sc.find_bin("M200c_0.2", lb)["rows"] for lb in CAL])
mask_ph = np.zeros((len(rows), len(X)), bool)
o = 0
for lb in CAL:
    b = sc.find_bin("M200c_0.2", lb)
    mask_ph[o:o + b["N"]] = sc.resolution_mask(b["rows"], "fit", "per_halo")
    o += b["N"]
sel = (X >= 2) & (X < 3)
fv = FILL[rows][:, sel][mask_ph[:, sel]]
cv = Crho[rows][:, sel][mask_ph[:, sel]]
ok = np.isfinite(fv) & np.isfinite(cv) & (cv > 0)
hb = axes[2].hexbin(np.log10(cv[ok]), fv[ok], gridsize=45, bins="log", cmap=LinearSegmentedColormap.from_list(
    "seqblue", ["#f4f8fd", "#86b6ef", "#2a78d6", "#0d366b"]), mincnt=1)
cb = fig.colorbar(hb, ax=axes[2])
cb.set_label("halo-shells per cell")
sp = [e for e in R["diagnostics"]["filling_factor"]["spearman_fill_vs_lnCrho_calibrated_resolved"] if e["band"] == "2-3"][0]
axes[2].set_xlabel(r"$\log_{10} C_\rho$ of the halo-shell")
axes[2].set_ylabel("filling factor f of the halo-shell")
axes[2].set_title(f"(c) per halo, 2-3 R200c, 13.5-15.5 (Spearman {sp['spearman_fill_vs_lnCrho']:+.2f}, "
                  f"n = {sp['n']})")
fig.suptitle("Why SPH-volume estimators read high: the particles' SPH volumes do not fill the shell, increasingly "
             "so beyond R200c (clumping is only a weak correlate, panel c)", y=0.995)
fig.tight_layout(rect=(0, 0, 1, 0.97))
fig.savefig(FIG / "fig2_filling_factor_clumping.png")
plt.close(fig)

# ============================================================================ fig 3: density estimators
fig, axes = plt.subplots(1, 2, figsize=(13, 5.0))
five_lines(axes[0], lambda lb: g("ratio", f"{G}:Q_V", "E", lb), label_end=False)
five_lines(axes[1], lambda lb: g("ratio", f"{G}:Q_M", "G", lb), label_end=False)
lab = "14.5-14.7"
axes[0].plot(X, np.where(resolved(lab), 1 / dg("fill", lab), np.nan), color=INK, ls="--", lw=1.3,
             label=r"$1/\langle f\rangle$, 14.5-14.7")
axes[0].set_ylabel(r"$n_e$: $Q_V$ / geometric (gas density $Q_V$ identical)")
axes[0].set_ylim(0.98, 1.32)
axes[0].axhline(1, color=INK, lw=0.8)
axes[0].set_title(r"(a) electrons: SPH-volume estimator $Q_V$")
axes[1].set_yscale("log")
axes[1].set_ylabel(r"gas density: $Q_M$ / geometric $=C_\rho/f$")
axes[1].set_title(r"(b) gas density: mass-weighted estimator $Q_M=\sum m\rho/\sum m$")
for ax in axes:
    xaxis(ax)
axes[0].legend(title=r"$\log_{10}M_{200c}$", ncol=2, fontsize=8.5, title_fontsize=9, frameon=False, loc="upper left")
fig.suptitle("Electron and gas density estimators relative to the geometric estimator (resolved shells, "
             "calibrated bins)", y=0.995)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(FIG / "fig3_density_estimators.png")
plt.close(fig)

# ============================================================================ fig 4: temperatures
TV = [("ref", r"$T_e$ (electron weighted; reference)", INK, "-", None),
      (f"{G}:T_M", r"$T_M$ (mass weighted)", C["blue"], "--", "o"),
      (f"{G}:T_V", r"$T_V$ (SPH-volume weighted)", C["orange"], "-.", "s"),
      (f"{G}:T_X", r"$T_X$ (ROSAT-luminosity weighted, X-ray phase)", C["aqua"], (0, (5, 1.5, 1, 1.5)), "^"),
      (f"{G}:T_PE", r"$\langle P_e\rangle/(k\langle n_e\rangle)$ from the two stacks", C["violet"], ":", "D")]
fig, axes = plt.subplots(2, 4, figsize=(16, 8.2), sharex=True)
for j, lab in enumerate(REP):
    res = resolved(lab)
    for vid, name, col, ls, mk in TV:
        v = g("val", vid, "T_e", lab)
        e = g("err", vid, "T_e", lab)
        band_plot(axes[0, j], v, v - e, v + e, col, ls, mk, name if j == 0 else None, res)
        if vid != "ref":
            band_plot(axes[1, j], g("ratio", vid, "T_e", lab), g("ratio_q16", vid, "T_e", lab),
                      g("ratio_q84", vid, "T_e", lab), col, ls, mk, None, res)
    for ax in axes[:, j]:
        shade_unresolved(ax, lab)
    axes[1, j].axhline(1, color=INK, lw=0.8)
    xaxis(axes[0, j], False)
    xaxis(axes[1, j])
    axes[0, j].set_ylim(0, 2.0)
    axes[1, j].set_ylim(0.75, 1.8)
    axes[0, j].set_title(rf"$\log_{{10}}M_{{200c}}$ = {lab}")
axes[0, 0].set_ylabel(r"$T/T_{200}$  ($kT_{200}=0.59\,m_p GM_{200c}/2R_{200c}$)")
axes[1, 0].set_ylabel(r"$T/T_e$")
h, l_ = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l_, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.015))
fig.suptitle("Temperature definitions, inclusion-weighted mean stacks. Bands: bootstrap SD in the top row, "
             "paired-bootstrap 16-84% range in the bottom row", y=0.995)
fig.tight_layout(rect=(0, 0.08, 1, 0.98))
note_unres(fig)
fig.savefig(FIG / "fig4_temperature_definitions.png")
plt.close(fig)

# ============================================================================ fig 5: statistics
ST = [("unweighted_mean", "unweighted mean", C["blue"], "-", "o"),
      ("median", "median", C["orange"], "--", "s"),
      ("geomean", "geometric mean (log-mean)", C["aqua"], "-.", "^"),
      ("y_weighted_mean", r"$Y_{200}$-weighted mean (B12)", C["yellow"], (0, (5, 1.5, 1, 1.5)), "D"),
      ("trimmed_mean", "5% trimmed mean", C["magenta"], ":", "v")]
FL = [("P", r"$P_e$"), ("E", r"$n_e$"), ("G", r"$\rho_{\rm gas}$"), ("T_e", r"$T_e$")]
YL = {"P": (0.35, 1.15), "E": (0.75, 1.1), "G": (0.75, 1.1), "T_e": (0.75, 1.06)}
fig, axes = plt.subplots(4, 4, figsize=(16, 15), sharex=True)
for i, (fld, fname) in enumerate(FL):
    for j, lab in enumerate(REP):
        ax = axes[i, j]
        res = resolved(lab)
        shade_unresolved(ax, lab)
        for st, name, col, ls, mk in ST:
            k = f"{G}:{st}"
            band_plot(ax, g("ratio", k, fld, lab), g("ratio_q16", k, fld, lab), g("ratio_q84", k, fld, lab), col, ls,
                      mk, name if (i == 0 and j == 0) else None, res, alpha=0.13)
        ax.axhline(1, color=INK, lw=0.8)
        ax.set_ylim(*YL[fld])
        xaxis(ax, i == 3)
        if i == 0:
            ax.set_title(rf"$\log_{{10}}M_{{200c}}$ = {lab}  (N = {R['bin_info'][lab]['N']})")
        if j == 0:
            ax.set_ylabel(f"{fname}: statistic / inclusion-weighted mean")
h, l_ = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l_, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 0.012))
fig.suptitle("Stacking statistics relative to the inclusion-weighted mean (geometric estimator)", y=0.997)
fig.tight_layout(rect=(0, 0.04, 1, 0.99))
note_unres(fig)
fig.savefig(FIG / "fig5_statistics_vs_mean.png")
plt.close(fig)

# ============================================================================ fig 6: lognormal test
fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True)
for i, fld in enumerate(("P", "E")):
    for j, lab in enumerate(REP):
        res = resolved(lab)
        col = REP_COL[j]
        meas = dg(f"mean_over_geomean|{fld}", lab)
        pred = dg(f"pred_lognormal|{fld}", lab)
        mmd = dg(f"mean_over_median|{fld}", lab)
        sk = dg(f"skew_ln|{fld}", lab)
        axes[i, 0].plot(X, np.where(res, meas, np.nan), color=col, label=f"{lab} measured")
        axes[i, 0].plot(X, np.where(res, pred, np.nan), color=col, ls="--", lw=1.4)
        axes[i, 0].plot(X, np.where(res, dg(f"pred_cumulant4|{fld}", lab), np.nan), color=col, ls=":", lw=1.4)
        axes[i, 1].plot(X, np.where(res, mmd, np.nan), color=col, label=lab)
        axes[i, 1].plot(X, np.where(res, pred, np.nan), color=col, ls="--", lw=1.4)
        axes[i, 2].plot(X, np.where(res, sk, np.nan), color=col, label=lab)
    nm = r"$P_e$" if fld == "P" else r"$n_e$"
    axes[i, 0].set_ylabel(f"{nm}: mean / geometric mean")
    axes[i, 1].set_ylabel(f"{nm}: mean / median")
    axes[i, 2].set_ylabel(rf"{nm}: skewness of $\ln F$ over haloes")
    axes[i, 2].axhline(0, color=INK, lw=0.8)
    for ax in axes[i]:
        xaxis(ax, i == 1)
    for ax in axes[i, :2]:
        ax.axhline(1, color=INK, lw=0.8)
        ax.set_yscale("log")
        ax.set_ylim(0.95, 2.1 if fld == "P" else 1.3)
        ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_yticks([1.0, 1.1, 1.2, 1.5, 2.0] if fld == "P" else [1.0, 1.05, 1.1, 1.2, 1.3])
axes[0, 0].set_title(r"(a) mean/geomean (solid) vs lognormal $e^{\sigma_{\ln}^2/2}$ (dashed)")
axes[0, 1].set_title(r"(b) mean/median (solid) vs lognormal $e^{\sigma_{\ln}^2/2}$ (dashed)")
axes[0, 2].set_title(r"(c) skewness of $\ln F$ (0 for a lognormal)")
handles = [Line2D([], [], color=c, lw=2, label=lb) for c, lb in zip(REP_COL, REP)] + \
          [Line2D([], [], color=INK2, lw=1.4, ls="--", label=r"lognormal expectation $e^{\sigma_{\ln}^2/2}$"),
           Line2D([], [], color=INK2, lw=1.4, ls=":", label=r"(a) with skewness and kurtosis: $e^{\kappa_2/2+\kappa_3/6+\kappa_4/24}$")]
fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.005),
           title=r"$\log_{10}M_{200c}$ bin (resolved shells only)")
fig.suptitle("Halo-to-halo distribution: is the mean-median / mean-log-mean gap the lognormal $e^{\\sigma^2/2}$?",
             y=0.995)
fig.tight_layout(rect=(0, 0.12, 1, 0.98))
fig.savefig(FIG / "fig6_lognormal_test.png")
plt.close(fig)

# ============================================================================ fig 7: heatmaps vs mass and radius
cmap = LinearSegmentedColormap.from_list("div", [C["red"], "#f0efec", C["blue"]])
bands = R["bands_R200c"]
fig, axes = plt.subplots(2, 2, figsize=(15, 10.5))
for i, fld in enumerate(("P", "E")):
    for j, st in enumerate(("median", "geomean")):
        ax = axes[i, j]
        rows_ = LABELS[3:]
        M = np.full((len(rows_), len(bands)), np.nan)
        for a, lab in enumerate(rows_):
            for bnd, e in enumerate(R["band_ratios"][f"{G}:{st}"][fld][lab]):
                if e["ratio"] is not None:
                    M[a, bnd] = e["ratio"]
        norm = TwoSlopeNorm(vmin=0.5, vcenter=1.0, vmax=1.1)
        ax.imshow(M, cmap=cmap, norm=norm, aspect="auto", origin="lower")
        for a in range(M.shape[0]):
            for bnd in range(M.shape[1]):
                if np.isfinite(M[a, bnd]):
                    ax.text(bnd, a, f"{M[a, bnd]:.2f}", ha="center", va="center", fontsize=8.5,
                            color="white" if M[a, bnd] < 0.7 else INK)
                else:
                    ax.text(bnd, a, "unres.", ha="center", va="center", fontsize=7, color=INK2)
        ax.set_xticks(range(len(bands)))
        ax.set_xticklabels(bands)
        ax.set_yticks(range(len(rows_)))
        ax.set_yticklabels([lb + (" *" if float(lb.split("-")[0]) < 13.5 else "") for lb in rows_])
        ax.set_xlabel(r"radial band [$R_{200c}$]")
        ax.set_ylabel(r"$\log_{10}M_{200c}$ bin")
        ax.grid(False)
        nm = r"$P_e$" if fld == "P" else r"$n_e$"
        ax.set_title(f"{nm}: {'median' if st == 'median' else 'geometric mean'} / inclusion-weighted mean")
fig.text(0.01, 0.005, "Median over resolved shells in the band; 'unres.' = no shell passes the strict rule; "
         "* exploratory bins below 10^13.5. Paired-bootstrap errors in results.json (band_ratios).",
         fontsize=9, color=INK2)
fig.tight_layout(rect=(0, 0.02, 1, 1))
fig.savefig(FIG / "fig7_median_geomean_heatmap.png")
plt.close(fig)

# ============================================================================ fig 8: bootstrap errors
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
ax = axes[0, 0]
five_lines(ax, lambda lb: dg("boot|boot_over_analytic|P", lb), label_end=False)
ax.axhline(1, color=INK, lw=0.8)
ax.set_ylabel(r"$P_e$: bootstrap SD / ($s/\sqrt{N_{\rm eff}}$)")
ax.set_title("(a) bootstrap error of the mean vs the analytic error")
ax.set_ylim(0.3, 1.4)
ax.legend(title=r"$\log_{10}M_{200c}$", ncol=2, fontsize=8.5, title_fontsize=9, frameon=False, loc="lower left")
xaxis(ax)
ax = axes[0, 1]
mk = {"0.5-1": "o", "2-3": "s"}
for fld, ls, col in (("P", "-", C["blue"]), ("E", "--", C["orange"])):
    for bnd, m in mk.items():
        neff, rel = [], []
        for lab in CAL:
            e = [x for x in R["bootstrap"]["per_field"][fld][lab] if x["band"] == bnd][0]
            if e.get("resolved"):
                neff.append(R["bin_info"][lab]["N_eff"])
                rel.append(e["rel_err_mean"])
        neff, rel = np.array(neff), np.array(rel)
        ax.scatter(neff, 100 * rel, color=col, marker=m, s=46, edgecolor="white", linewidth=0.8, zorder=3,
                   label=f"{'P_e' if fld == 'P' else 'n_e'}, {bnd} R200c")
        nn = np.geomspace(5, 1200, 50)
        amp = np.median(rel * np.sqrt(neff))
        ax.plot(nn, 100 * amp / np.sqrt(nn), color=col, ls=ls if bnd == "0.5-1" else ":", lw=1.2)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$N_{\rm eff}$ of the bin")
ax.set_ylabel("relative bootstrap error of the mean [%]")
ax.set_title(r"(b) error of the mean vs $N_{\rm eff}$ (lines: $\propto N_{\rm eff}^{-1/2}$)")
ax.legend(fontsize=9, frameon=False, loc="upper right")
ax = axes[1, 0]
lab = "14.5-14.7"
res = resolved(lab)
for st, name, col, ls, m in ST:
    ax.plot(X, np.where(res, dg(f"boot|err_{st}_over_err_mean|P", lab), np.nan), color=col, ls=ls, marker=m, ms=4,
            markevery=3, label=name)
ax.axhline(1, color=INK, lw=0.8)
ax.axhline(np.sqrt(np.pi / 2), color=INK2, lw=0.8, ls="--")
ax.text(0.022, np.sqrt(np.pi / 2) + 0.02, r"Gaussian median: $\sqrt{\pi/2}$", fontsize=9, color=INK2)
ax.set_ylabel(r"$P_e$: bootstrap SD of statistic / SD of mean")
ax.set_title(f"(c) noise of each statistic, {lab} (N = 895)")
ax.set_ylim(0, 1.8)
ax.legend(fontsize=9, frameon=False, loc="lower left", ncol=2)
xaxis(ax)
ax = axes[1, 1]
for lab, col in (("14.5-14.7", C["blue"]), ("15.3-15.5", C["orange"])):
    res = resolved(lab)
    ax.plot(X, np.where(res, dg("boot|jack_over_boot|P", lab), np.nan), color=col, label=f"{lab}: 4$^3$-block jackknife / bootstrap")
    ax.plot(X, np.where(res, dg("boot|unstrat_over_strat|P", lab), np.nan), color=col, ls="--",
            label=f"{lab}: unstratified / stratified bootstrap")
ax.axhline(1, color=INK, lw=0.8)
ax.set_ylabel(r"$P_e$: ratio of error estimates")
ax.set_title("(d) error method: spatial jackknife and stratification")
top = np.nanmax([np.nanmax(np.where(resolved(lb), dg(f"boot|{nm}|P", lb), np.nan)) for lb in ("14.5-14.7", "15.3-15.5")
                 for nm in ("jack_over_boot", "unstrat_over_strat")])
ax.set_ylim(0.6, top + 0.9)
ax.legend(fontsize=9, frameon=False, loc="upper left")
xaxis(ax)
fig.tight_layout()
fig.savefig(FIG / "fig8_bootstrap_errors.png")
plt.close(fig)

# ============================================================================ fig 9: observable impact
VLIST = [("Q_V", r"$Q_V$ estimator"), ("nT_X", r"$n_e kT_X$ estimator"), ("unweighted_mean", "unweighted mean"),
         ("median", "median"), ("geomean", "geometric mean"), ("y_weighted_mean", r"$Y_{200}$-weighted mean"),
         ("trimmed_mean", "5% trimmed mean")]
OBS_P = [("tSZ_yy_total_ell100", r"tSZ $C_\ell^{yy}$, $\ell$=100", C["blue"], "o"),
         ("tSZ_yy_total_ell1000", r"tSZ $C_\ell^{yy}$, $\ell$=1000", C["orange"], "s"),
         ("tSZ_yy_total_ell3000", r"tSZ $C_\ell^{yy}$, $\ell$=3000", C["aqua"], "^"),
         ("y_mean", r"mean $y$", C["violet"], "D")]
OBS_E = [("kSZ_1h_ell100", r"kSZ 1h, $\ell$=100", C["blue"], "o"),
         ("kSZ_1h_ell1000", r"kSZ 1h, $\ell$=1000", C["orange"], "s"),
         ("kSZ_1h_ell3000", r"kSZ 1h, $\ell$=3000", C["aqua"], "^"),
         ("DM_mean", "mean DM", C["violet"], "D"), ("DM_var", "DM variance", C["red"], "v")]
fig, axes = plt.subplots(1, 2, figsize=(15, 6.2), sharey=True)
for ax, fld, obs in ((axes[0], "P", OBS_P), (axes[1], "E", OBS_E)):
    names = []
    for yi, (vid, vname) in enumerate(VLIST):
        k = f"{G}:{vid}"
        names.append(vname)
        if fld not in R["impact"].get(k, {}):
            ax.text(0, yi, "n/a (pressure only)", ha="center", va="center", fontsize=9, color=INK2)
            continue
        for oi, (o, oname, col, m) in enumerate(obs):
            e = R["impact"][k][fld][o]
            off = (oi - (len(obs) - 1) / 2) * 0.13
            ax.errorbar(100 * e["dO_over_O"], yi + off, xerr=100 * e["err"], fmt=m, color=col, ms=7, mec="white",
                        mew=0.8, elinewidth=1.4, capsize=2, label=oname if yi == 2 else None)
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_yticks(range(len(VLIST)))
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("first-order change of the observable vs reference [%]")
    ax.legend(fontsize=9, frameon=False, loc="upper right")
    ax.grid(axis="y", visible=False)
axes[0].set_title("pressure-side observables (P variants)")
axes[1].set_title("electron-side observables (n_e variants)")
qm = R["impact"][f"{G}:Q_M"]["P"]
qx = R["impact"][f"{G}:Q_X"]["P"]
fig.text(0.01, 0.01, f"Off scale: Q_M pressure: tSZ ell=100/1000/3000 {100 * qm['tSZ_yy_total_ell100']['dO_over_O']:+.0f}/"
         f"{100 * qm['tSZ_yy_total_ell1000']['dO_over_O']:+.0f}/{100 * qm['tSZ_yy_total_ell3000']['dO_over_O']:+.0f}%, "
         f"mean y {100 * qm['y_mean']['dO_over_O']:+.0f}%;  Q_X: {100 * qx['tSZ_yy_total_ell100']['dO_over_O']:+.0f}/"
         f"{100 * qx['tSZ_yy_total_ell1000']['dO_over_O']:+.0f}/{100 * qx['tSZ_yy_total_ell3000']['dO_over_O']:+.0f}%, "
         f"mean y {100 * qx['y_mean']['dO_over_O']:+.0f}%\n(far outside the first-order regime; mean y exact). "
         "Error bars: SD over 200 paired bootstrap replicates. Calibrated bins 13.5-15.5 only.",
         fontsize=8.5, color=INK2)
fig.suptitle("Observable impact of painting the variant stack instead of the reference "
             "(sc.impact_from_stacks, Test F importance Jacobians)", y=0.995)
fig.tight_layout(rect=(0, 0.07, 1, 0.97))
fig.savefig(FIG / "fig9_observable_impact.png")
plt.close(fig)
# ============================================================================ fig 10: why T_X differs from T_e
fig = plt.figure(figsize=(16, 9.6))
gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.05])
top = [fig.add_subplot(gs[0, j]) for j in range(4)]
TXP = [("TX_over_Te", r"$T_X/T_e$ (total)", INK, "-", None),
       ("phase_cut_Te_xray_over_Te", r"phase cut: $T_{e,\rm X\,phase}/T_e$", C["orange"], "--", "s"),
       ("LX_weighting_TX_over_Te_xray", r"$L_X$ weighting: $T_X/T_{e,\rm X\,phase}$", C["blue"], "-.", "o"),
       ("smooth_gas_TX_over_Te_xray", r"$L_X$ weighting, central-bound + unbound gas only", C["aqua"], ":", "^")]
for j, lab in enumerate(REP):
    ax = top[j]
    res = resolved(lab)
    shade_unresolved(ax, lab)
    for nm, name, col, ls, mk in TXP:
        y = np.where(res, dg(f"txd|{nm}", lab), np.nan)
        ax.plot(X, y, color=col, ls=ls, marker=mk, ms=4.5, markevery=(1, 4), label=name if j == 0 else None)
    ax.axhline(1, color=INK2, lw=0.8)
    xaxis(ax)
    ax.set_ylim(0.8, 1.75)
    ax.set_title(rf"({'abcd'[j]}) $\log_{{10}}M_{{200c}}$ = {lab}")
top[0].set_ylabel("temperature ratio")
ax = fig.add_subplot(gs[1, :2])
band_mk = {"0.25-0.5": ("o", "#86b6ef"), "0.5-1": ("s", "#5598e7"), "1-1.5": ("^", "#2a78d6"),
           "1.5-2": ("D", "#1c5cab"), "2-3": ("v", "#104281"), "3-4": ("P", "#0d366b")}
TXD = R["diagnostics"]["T_X_decomposition"]["per_bin_band"]
for bl, (mk, col) in band_mk.items():
    kt = [TXD[lb]["kTe_keV"][bl] for lb in CAL]
    fx = [TXD[lb]["LX_weighting_TX_over_Te_xray"][bl] for lb in CAL]
    ok = [(a, b) for a, b in zip(kt, fx) if a is not None and b is not None]
    if ok:
        ax.scatter(*zip(*ok), marker=mk, s=52, color=col, edgecolor="white", linewidth=0.8, zorder=3,
                   label=f"{bl} R200c")
ax.axhline(1, color=INK2, lw=0.8)
ax.set_xscale("log")
ax.set_xticks([0.2, 0.5, 1, 2, 5])
ax.set_xticklabels(["0.2", "0.5", "1", "2", "5"])
ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.set_xlabel(r"stacked electron-weighted temperature $kT_e$ [keV]")
ax.set_ylabel(r"$L_X$-weighting factor $T_X/T_{e,\rm X\,phase}$")
ax.set_title(r"(e) $L_X$-weighting factor vs gas temperature (10 calibrated bins, resolved bands)")
ax.legend(title="radial band", fontsize=9, title_fontsize=9, frameon=False, loc="upper right", ncol=2)
ax = fig.add_subplot(gs[1, 2:])
five_lines(ax, lambda lb: dg("txd|LX_share_substructure_bound", lb), label_end=False)
xaxis(ax)
ax.set_ylim(0, 1)
ax.set_ylabel("share of the X-ray-phase ROSAT luminosity\nfrom gas bound to other subhaloes")
ax.set_title(r"(f) share of $L_X$ from substructure-bound gas")
ax.legend(title=r"$\log_{10}M_{200c}$", fontsize=8.5, title_fontsize=9, frameon=False, loc="upper left")
h, l_ = top[0].get_legend_handles_labels()
fig.legend(h, l_, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.955), fontsize=9.5)
fig.suptitle(r"Why $T_X \neq T_e$: $T_X/T_e$ = (X-ray phase cut) $\times$ ($L_X$ weighting inside the X-ray phase); "
             "inclusion-weighted mean stacks", y=0.995)
fig.tight_layout(rect=(0, 0.02, 1, 0.93))
note_unres(fig, bands=False)
fig.savefig(FIG / "fig10_tx_decomposition.png")
plt.close(fig)

# ============================================================================ fig 11: what drives the mass weighting
fig, axes = plt.subplots(2, 3, figsize=(16, 9.4), sharex=True)
for i, lab in enumerate(("13.5-13.7", "14.5-14.7")):
    res = resolved(lab)
    ax = axes[i, 0]
    for nm, name, col, ls, mk in (
            ("mass_share_cold", "cold/SF gas: share of gas mass", C["blue"], "-", "o"),
            ("sum_m_rho_share_cold", r"cold/SF gas: share of $\sum m\rho$ (gas $Q_M$)", C["blue"], "--", "s"),
            ("y_share_cold", r"cold/SF gas: share of $\sum y$ (geometric $P_e$)", C["blue"], ":", "^"),
            ("mass_share_substructure_bound", "substructure-bound gas: share of gas mass", C["orange"], "-", "D"),
            ("sum_m_pe_share_substructure_bound", r"substructure-bound gas: share of $\sum m p_e$ ($P_e$ $Q_M$)",
             C["orange"], "--", "v")):
        ax.plot(X, np.where(res, dg(f"mwd|{nm}", lab), np.nan), color=col, ls=ls, marker=mk, ms=4.5,
                markevery=(1, 4), label=name if i == 0 else None)
    ax.set_yscale("log")
    ax.set_ylim(1e-7, 1.5)
    ax.set_ylabel(f"{lab}: share of the stacked sum")
    ax = axes[i, 1]
    for nm, name, col, ls, mk in (("G_QM_over_geom_all", "all gas", INK, "-", None),
                                  ("G_QM_over_geom_without_cold", "without cold/SF gas", C["blue"], "--", "o"),
                                  ("G_QM_over_geom_without_substructure", "without substructure-bound gas",
                                   C["orange"], "-.", "s"),
                                  ("G_QM_over_geom_smooth_hot_only", "smooth hot gas only", C["aqua"], ":", "^")):
        ax.plot(X, np.where(res, dg(f"mwd|{nm}", lab), np.nan), color=col, ls=ls, marker=mk, ms=4.5,
                markevery=(1, 4), label=name if i == 0 else None)
    ax.set_yscale("log")
    ax.set_ylabel(r"$\rho_{\rm gas}$: $Q_M$ / geometric")
    ax = axes[i, 2]
    for nm, name, col, ls, mk in (("P_QM_over_geom_all", "all gas", INK, "-", None),
                                  ("P_QM_over_geom_without_cold", "without cold/SF gas", C["blue"], "--", "o"),
                                  ("P_QM_over_geom_without_substructure", "without substructure-bound gas",
                                   C["orange"], "-.", "s"),
                                  ("P_QM_over_geom_smooth_hot_only", "smooth hot gas only", C["aqua"], ":", "^")):
        ax.plot(X, np.where(res, dg(f"mwd|{nm}", lab), np.nan), color=col, ls=ls, marker=mk, ms=4.5,
                markevery=(1, 4))
    ax.set_yscale("log")
    ax.set_ylabel(r"$P_e$: $Q_M$ / geometric")
    for ax in axes[i]:
        shade_unresolved(ax, lab)
        xaxis(ax, i == 1)
    for ax in axes[i, 1:]:
        ax.axhline(1, color=INK2, lw=0.8)
axes[0, 0].set_title("(a) mass vs mass-weighted sums")
axes[0, 1].set_title(r"(b) gas density $Q_M=\sum m\rho/\sum m$ (both stacks over the same gas)")
axes[0, 2].set_title(r"(c) pressure $Q_M=\sum m p_e/\sum m$ (same gas)")
h0, l0 = axes[0, 0].get_legend_handles_labels()
h1, l1 = axes[0, 1].get_legend_handles_labels()
fig.legend(h0, l0, loc="lower left", ncol=2, frameon=False, bbox_to_anchor=(0.01, 0.025), fontsize=9)
fig.legend(h1, l1, loc="lower right", ncol=2, frameon=False, bbox_to_anchor=(0.99, 0.025), fontsize=9,
           title="(b), (c)", title_fontsize=9)
fig.suptitle("Why mass-weighted estimators read high: dense cold/star-forming and substructure-bound gas (a minority of "
             r"the mass)" "\n" r"carries most of $\sum m\rho$ at every radius and of $\sum m p_e$ beyond 2 $R_{200c}$; "
             "the smooth hot gas alone (dotted) still reads high\nthrough its own clumping and density-pressure "
             "correlation, and gives most of the pressure excess inside 1 $R_{200c}$", y=0.997)
fig.tight_layout(rect=(0, 0.1, 1, 0.93), w_pad=2.5)
note_unres(fig, bands=False)
fig.savefig(FIG / "fig11_mass_weighting_decomposition.png")
plt.close(fig)
print("figures written:", sorted(p.name for p in FIG.glob("*.png")))
