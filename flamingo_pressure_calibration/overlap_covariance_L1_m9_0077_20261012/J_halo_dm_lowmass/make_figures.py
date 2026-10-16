"""Figures of Test J (local, from the fetched results/ and post/j_post.json; --smoke: results_smoke/, figures_smoke/).

Colour conventions (dataviz reference palette, validated orders): the four halo-component definitions are categorical
slots in fixed order D0 blue, D1 orange, D2 aqua, D3 yellow, always with a legend, direct labels and distinct line
styles (secondary encoding); mass bins are ordered, so they use one sequential blue ramp (light = low mass, dark = high
mass, ordinal steps 250-700); the five mass groups of J1_importance_vs_radius use the evenly spaced ordinal steps
250 / 400 / 500 / 600 / 700 (OKLab L 0.76 / 0.62 / 0.53 / 0.43 / 0.34) plus a distinct dash pattern per group
(secondary encoding); reference curves (published fallback, cosmic mean) are dark grey / near-black.
x = r / R200c; DM in pc cm^-3 (observer frame, z_s = 1); importance densities per unit ln x.
Usage: python -B make_figures.py [--smoke]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SEQ_STEPS = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281",
             "#0d366b"]
INK, INK2, GRID, REF = "#0b0b0b", "#52514e", "#e4e3df", "#3a3936"
DEFS = ("D0", "D1", "D2", "D3")
DCOL = dict(zip(DEFS, CAT[:4]))
DLS = {"D0": "-", "D1": "--", "D2": "-.", "D3": ":"}
DLAB = {"D0": "D0: A2-ii to 4 R$_{200c}$", "D1": "D1: A2-ii, sphere 1 R$_{200c}$",
        "D2": "D2: A2-ii, sphere 2 R$_{200c}$", "D3": "D3: host-bound to 4 R$_{200c}$"}
CUT = {"D0": 4.0, "D1": 1.0, "D2": 2.0, "D3": 4.0}
LABELS = ["12.5-12.7", "12.7-12.9", "12.9-13.1", "13.1-13.3", "13.3-13.5", "13.5-13.7", "13.7-13.9", "13.9-14.1",
          "14.1-14.3", "14.3-14.5", "14.5-14.7", "14.7-14.9", "14.9-15.1", "15.1-15.3", "15.3-15.5", ">=15.5"]
Q_NAMES = ("q16", "q50", "q84", "q975", "q999")
T_CMB_UK = 2.7255e6
MECKE_TM_TOL = 0.01    # G4 tolerance of the truth-mean Mecke maps
MECKE_SUB_TOL = 0.05   # G4 tolerance of the per-halo subsample Mecke maps (subsample noise)
# mass groups of the importance figure: ordinal blue steps 250 / 400 / 500 / 600 / 700 and one dash pattern each
GROUP_STYLE = [("#86b6ef", "-", 2.0), ("#3987e5", (0, (5, 1.6)), 1.6), ("#256abf", (0, (6, 1.6, 1.4, 1.6)), 1.6),
               ("#184f95", (0, (1.2, 1.2)), 1.9), ("#0d366b", (0, (9, 2.2)), 1.4)]
LEG_BOX = dict(frameon=True, facecolor="white", edgecolor="none", framealpha=0.85)
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                     "lines.linewidth": 1.6, "savefig.dpi": 150, "figure.dpi": 100})


def seq(n):
    """n ordered colours from the blue ramp (light -> dark)."""
    idx = np.linspace(0, len(SEQ_STEPS) - 1, n)
    return [SEQ_STEPS[int(round(i))] for i in idx]


def save(fig, out, name):
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.png", bbox_inches="tight")
    fig.savefig(out / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def mids(e):
    return 0.75 * (e[1:]**4 - e[:-1]**4) / (e[1:]**3 - e[:-1]**3)


def smooth_density(imp_shell, edges, dln, width=0.12):
    """Importance per unit ln x on the fine shells averaged over +-width/2 in ln x (the X200 grid has sliver shells)."""
    lm = np.log(mids(edges))
    out = np.empty(len(imp_shell))
    for i in range(len(imp_shell)):
        m = np.abs(lm - lm[i]) <= width / 2
        out[i] = np.sum(imp_shell[m]) / np.sum(dln[m])
    return out


def load(res, post):
    def js(p):
        return json.loads(p.read_text()) if p.exists() else None

    def npz(p):
        return dict(np.load(p, allow_pickle=False)) if p.exists() else None
    return {"truth": npz(res / "truth.npz"), "ts": js(res / "truth_summary.json"), "sens": npz(res / "sens.npz"),
            "ss": js(res / "sens_summary.json"), "stacks": npz(res / "stacks.npz"), "prep": js(res / "prepare.json"),
            "post": js(post / "j_post.json"), "mecke": js(post / "mecke_update.json"),
            "noise": js(HERE / "results_mecke_checks" / "mecke_noise.json") if res.name == "results" else None}


# ============================================================================ figure 1: DM PDF and statistics
def fig_pdf(D, out):
    tr, post = D["truth"], D["post"]
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.4), gridspec_kw={"width_ratios": [1.25, 1.0, 1.15]})
    ax = axs[0]
    dx = 0.125
    for d in DEFS:
        dens = tr.get(f"pdf|C14|{d}|T1|truth|density")
        if dens is None:
            continue
        x = np.arange(len(dens)) * dx
        cdf = np.cumsum(np.asarray(dens, float)) * dx                  # P(< x + dx/2) (CIC pmf)
        lb = np.geomspace(1.0, 2000.0, 61)                               # display: probability per ln DM in log bins
        cb = np.interp(lb, x + 0.5 * dx, cdf)
        ax.plot(np.sqrt(lb[1:] * lb[:-1]), np.diff(cb) / np.diff(np.log(lb)), color=DCOL[d], ls=DLS[d],
                label=DLAB[d])
        p0 = post["definitions"][f"C14|{d}"]["p_atom"]
        p0s = f"{p0:.3f}" if p0 >= 1e-3 else f"{p0:.1e}"
        ax.plot([], [], " ", label=f"   P(DM < 0.0625) = {p0s}  (no-crossing atom)")
    ax.set_xscale("log")
    ax.set_xlim(1, 2000)
    ax.set_xlabel("halo DM [pc cm$^{-3}$] (observer frame, z$_s$ = 1)")
    ax.set_ylabel("dP / d ln DM  (60 log bins, 1-2000)")
    ax.set_title("(a) halo-DM PDF, 14 calibrated bins (C14)", loc="left", fontsize=9.5)
    ax.legend(fontsize=7.5, loc="upper left")
    # (b) cumulative
    ax = axs[1]
    for d in DEFS:
        dens = tr.get(f"pdf|C14|{d}|T1|truth|density")
        if dens is None:
            continue
        x = np.arange(len(dens)) * dx
        cdf = np.cumsum(np.asarray(dens, float)) * dx
        ax.plot(x + 0.5 * dx, cdf, color=DCOL[d], ls=DLS[d], label=d)
    for lev in (0.16, 0.5, 0.84):
        ax.axhline(lev, color=INK2, lw=0.7, ls=(0, (2, 2)))
    ax.set_xscale("log")
    ax.set_xlim(1, 2000)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("halo DM [pc cm$^{-3}$]")
    ax.set_ylabel("P(< DM)")
    ax.set_title("(b) cumulative distribution (atoms at DM = 0 included)", loc="left", fontsize=9.5)
    ax.legend(fontsize=8, loc="lower right")
    # (c) statistics relative to D0
    ax = axs[2]
    stats = [("dm_mean", "mean"), ("dm_var", "variance"), ("q16", "Q16"), ("q50", "Q50"), ("q84", "Q84"),
             ("q975", "Q97.5"), ("q999", "Q99.9"), ("dmcltot_1000", "DM-field C$_\\ell$, $\\ell$=1000"),
             ("ydmtot_1000", "y$\\times$DM, $\\ell$=1000")]
    y = np.arange(len(stats))[::-1]
    for j, d in enumerate(("D1", "D2", "D3")):
        r = post["definitions_ratio_to_D0_C14"][d]
        vals = np.array([r[s] if r[s] is not None else np.nan for s, _ in stats], float)
        atom = np.array([s in post["quantiles_in_zero_atom"][f"C14|{d}"] for s, _ in stats])
        vals = np.where(atom, 1.2e-3, vals)
        ax.scatter(vals[~atom], (y + (j - 1) * 0.22)[~atom], s=36, color=DCOL[d], marker="osD"[j], label=DLAB[d],
                   zorder=3, edgecolor="white", linewidth=0.8)
        for k in np.flatnonzero(atom):
            ax.scatter([vals[k]], [y[k] + (j - 1) * 0.22], s=36, facecolor="none", edgecolor=DCOL[d], marker="osD"[j],
                       zorder=3)
            ax.text(1.6e-3, y[k] + (j - 1) * 0.22, f"{d} {stats[k][1]} = 0 (inside the DM = 0 atom)", fontsize=7,
                    va="center", color=INK2)
    ax.axvline(1.0, color=INK2, lw=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels([s[1] for s in stats])
    ax.set_xscale("log")
    ax.set_xlim(8e-4, 1.5)
    ax.set_xlabel("ratio to D0 (open marker: quantile inside the DM = 0 atom)")
    ax.set_title("(c) DM statistics relative to D0 (C14)", loc="left", fontsize=9.5)
    ax.legend(fontsize=7.5, loc="lower left")
    fig.tight_layout()
    save(fig, out, "J1_dm_pdf_definitions")


# ============================================================================ figure 2: importance vs radius
def fig_importance(D, out):
    S, ss = D["sens"], D["ss"]
    e, dln = S["x_edges"], S["dlnx"]
    xm = mids(e)
    jb = list(S["j_bins"])
    groups = [("12.7-13.1", [1, 2]), ("13.1-13.5", [3, 4]), ("13.5-14.1", [5, 6, 7]), ("14.1-14.7", [8, 9, 10]),
              ("14.7-15.5 (+ fallback > 10$^{15.39}$)", [11, 12, 13, 14, 15])]
    probes = [("dm_mean", "mean DM"), ("dm_var", "DM variance"), ("dm_q50", "DM median (Q50)"),
              ("dm_q999", "DM Q99.9")]
    fig, axs = plt.subplots(len(probes), 4, figsize=(15, 11), sharex=True)
    for col, d in enumerate(DEFS):
        for row, (name, lab) in enumerate(probes):
            ax = axs[row, col]
            key = f"imp__{d}__{name}"
            if key not in S:
                ax.set_visible(False)
                continue
            M = np.asarray(S[key], float)
            for (glab, bins), (c, ls, lw) in zip(groups, GROUP_STYLE):
                bins = [b for b in bins if b < M.shape[0]]
                ax.plot(xm, smooth_density(M[bins].sum(axis=0), e, dln), color=c, ls=ls, lw=lw, label=glab)
            ax.plot(xm, smooth_density(M.sum(axis=0), e, dln), color=INK, lw=2.2, label="all bins", zorder=1)
            ax.axvline(CUT[d], color=DCOL[d], lw=1.2, ls=DLS[d])
            ax.set_xscale("log")
            ax.set_xlim(0.02, 4.2)
            if row == 0:
                ax.set_title(DLAB[d], fontsize=9.5, loc="left", color=INK)
            if col == 0:
                ax.set_ylabel(f"{lab}\ndI/d ln x (per unit ln x)")
            if row == len(probes) - 1:
                ax.set_xlabel("x = r / R$_{200c}$")
            if name.startswith("dm_q"):
                # signed sum of the plotted (per-halo subsample) Mecke map with the whole DM = 0 atom (J1 revision
                # 2026-10-16, post/mecke_update.json); sens_summary.sum_rules holds |sum - 1|
                total = float(M.sum())
                qi = Q_NAMES.index(name[3:])
                mu = (D.get("mecke") or {}).get("mecke_sum_rule_minus_1", {}).get(d, {})
                tm = mu.get(f"whole|C14|{d}|T1|truthmean")
                tm_sum = None if tm is None else 1.0 + float(tm[qi])
                p_atom = D["post"]["definitions"][f"C14|{d}"]["p_atom"]
                atom = f"\n(atom P(DM = 0) = {p_atom:.1%} counted)" if p_atom > 1e-3 else ""
                nz = D.get("noise")
                sig = None if nz is None else float(nz["per_definition"][d]["sigma_boot"][qi])
                sub_tol = MECKE_SUB_TOL if sig is None else max(MECKE_SUB_TOL, 3 * sig)
                pm = "" if sig is None else f" +- {sig:.2f}"
                if (tm_sum is not None and abs(tm_sum - 1) > MECKE_TM_TOL) or abs(total - 1) > sub_tol:
                    note = (f"Mecke map sums to {total:.2f}{pm}: outside the G4 tolerance"
                            + ("" if tm_sum is None else f"\n(truth-mean map {tm_sum:.3f})"))
                else:
                    note = (f"Mecke map sums to {total:.2f}{pm}\n(per-halo subsample, bootstrap SD;\n"
                            + ("" if tm_sum is None else f"truth-mean map {tm_sum:.3f})") + atom)
                right = name == "dm_q999"
                lo, hi = ax.get_ylim()
                ax.set_ylim(lo, lo + 1.5 * (hi - lo))          # headroom so the note never covers a curve
                ax.text(0.97 if right else 0.03, 0.95, note, transform=ax.transAxes, ha="right" if right else "left",
                        va="top", fontsize=7, color=INK2,
                        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=0.85))
            if name.startswith("dm_q"):
                Q = D["post"]["definitions"][f"C14|{d}"][name[3:]]
                if Q is not None and Q <= 0.125:
                    ax.text(0.5, 0.5, "quantile inside the\nDM = 0 atom:\nimportance undefined", transform=ax.transAxes,
                            ha="center", va="center", fontsize=8, color=INK2)
            if row == 0 and col == 0:
                ax.legend(fontsize=7, loc="upper left")
    fig.suptitle("Where each DM statistic comes from: importance density per unit ln x, by mass group (C14; every map "
                 "sums to 1 over bins and shells: quantile (Mecke) maps with the whole DM = 0 atom, sums stated in each "
                 "panel;\nmass groups light to dark blue, each with its own dash pattern; the coloured vertical line is "
                 "the definition's boundary)", fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    save(fig, out, "J1_importance_vs_radius")


# ============================================================================ figure 3: low-mass stacks vs fallback
def fig_lowmass(D, out):
    st, prep = D["stacks"], D["prep"]
    e = st["x_edges"]
    xm = mids(e)
    low = [b for b in prep["low_bins"]]
    ed = np.r_[0.716 * e[1], e[1:]]                      # innermost shell drawn from its effective centre
    fig, axs = plt.subplots(3, len(low), figsize=(3.8 * len(low), 9.5), sharex=True, squeeze=False)
    for j, b in enumerate(low):
        xh = float(st["x_h"][b])
        # row 0: x^2 E (electrons per ln x up to a constant)
        ax = axs[0, j]
        ax.stairs(xm**2 * st["E_a2ii_T1"][b], ed, baseline=None, color=CAT[0], lw=1.6, label="A2-ii stack T1 (merged)")
        ax.plot(xm, xm**2 * st["E_a2ii_T3"][b], color=INK2, lw=0.6, alpha=0.8, label="A2-ii raw fine shells (T3)")
        ax.plot(xm, xm**2 * st["pubE_mean"][b], color=REF, lw=1.4, ls="--", label="published B16 (fallback)")
        ax.stairs(xm**2 * st["E_hb_T1"][b], ed, baseline=None, color=CAT[1], lw=1.2, label="host-bound stack T1")
        ax.set_yscale("log")
        ax.set_title(f"log M$_{{200c}}$ {LABELS[b]}  (x$_h$ = {xh:.2f})", fontsize=9.5, loc="left")
        if j == 0:
            ax.set_ylabel("x$^2$ E   (E = n$_e\\mu_{XGP}$/($\\rho_c f_b$))")
        # row 1: E / B16
        ax = axs[1, j]
        with np.errstate(all="ignore"):
            ax.stairs(st["E_a2ii_T1"][b] / st["pubE_mean"][b], ed, baseline=None, color=CAT[0], lw=1.6,
                    label="A2-ii / B16")
            ax.stairs(st["E_hb_T1"][b] / st["pubE_mean"][b], ed, baseline=None, color=CAT[1], lw=1.2,
                    label="host-bound / B16")
            ax.plot(xm, st["E_a2ii_T3"][b] / st["pubE_mean"][b], color=INK2, lw=0.6, alpha=0.8,
                    label="A2-ii raw fine shells (T3) / B16")
        ax.axhline(1, color=INK2, lw=0.8)
        ax.set_yscale("log")
        ax.set_ylim(0.01, 30)
        if j == 0:
            ax.set_ylabel("electrons: measured / published B16")
            ax.legend(fontsize=7, loc="upper center", **LEG_BOX)
        # row 2: P / B12
        ax = axs[2, j]
        with np.errstate(all="ignore"):
            ax.stairs(st["P_a2ii_T1"][b] / st["pubP_mean"][b], ed, baseline=None, color=CAT[0], lw=1.6,
                    label="A2-ii pressure stack T1 (merged) / B12")
            ax.plot(xm, st["P_a2ii_T3"][b] / st["pubP_mean"][b], color=INK2, lw=0.6, alpha=0.8,
                    label="A2-ii raw fine shells (T3) / B12")
        ax.axhline(1, color=INK2, lw=0.8)
        ax.set_yscale("log")
        ax.set_ylim(0.01, 30)
        ax.set_xlabel("x = r / R$_{200c}$")
        if j == 0:
            ax.set_ylabel("pressure: measured / published B12")
            ax.legend(fontsize=7, loc="upper center", **LEG_BOX)
        xs = float(st["softening_limit"][b])
        for row in range(3):
            axs[row, j].axvspan(0.02, xh, color=GRID, alpha=0.6, lw=0)
            axs[row, j].axvline(xs, color=INK, lw=1.0, ls=":", zorder=5,
                                label=f"softening limit 4$\\epsilon$/R$_{{200c}}$" if (row == 0 and j == 0) else None)
            axs[row, j].set_xscale("log")
            axs[row, j].set_xlim(0.02, 4.2)
        axs[0, j].text(xs * 1.06, 0.97, f"4$\\epsilon$/R = {xs:.3f}", transform=axs[0, j].get_xaxis_transform(),
                       ha="left", va="top", fontsize=7, color=INK, rotation=90)
        rec = prep["J1a_by_bin"][LABELS[b]]
        # bottom centre (x ~ 0.1-0.8 R200c), below every curve: the host-bound / B16 curves fall to <= 0.1 only
        # beyond ~2 R200c (lower right) and inside the cores (lower left)
        axs[1, j].text(0.5, 0.03, f"N$_e$(<1R): {rec['Ne_lt1R_ratio_A2ii_over_B16']:.2f} x B16\n"
                                  f"N$_e$(<2R): {rec['Ne_lt2R_ratio_A2ii_over_B16']:.2f} x B16\n"
                                  f"N$_e$(<4R): {rec['Ne_lt4R_ratio_A2ii_over_B16']:.2f} x B16",
                       transform=axs[1, j].transAxes, ha="center", va="bottom", fontsize=7.5, color=INK, zorder=6,
                       bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.85))
        axs[2, j].text(0.98, 0.04, f"Y(<1R): {rec['Y_lt1R_ratio_A2ii_over_B12']:.2f} x B12",
                       transform=axs[2, j].transAxes, ha="right", va="bottom", fontsize=7.5, color=INK)
    axs[0, 0].legend(fontsize=7, loc="lower center", **LEG_BOX)
    fig.suptitle("Low-mass FLAMINGO stacks (inclusion-weighted means, fine shells merged to >= 400 particles per stacked "
                 "shell) vs the published fallback the engine paints now;\ngrey band: inside the hydrodynamic-resolution "
                 "radius x$_h$ (shape unresolved, integral measured); dotted line: gravitational-softening limit "
                 "4$\\epsilon$/R$_{200c}$", fontsize=9.5, x=0.01, ha="left")
    fig.tight_layout()
    save(fig, out, "J1_lowmass_stacks_vs_fallback")


# ============================================================================ figure 4: bound vs A2-ii per mass bin
def fig_bound(D, out):
    st, prep = D["stacks"], D["prep"]
    e = st["x_edges"]
    xm = mids(e)
    bins = list(prep["j_bins"])
    n = len(bins)
    ncol = 5 if n > 4 else n
    nrow = int(np.ceil(n / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(3.3 * ncol, 2.8 * nrow), sharex=True, sharey=True, squeeze=False)
    Ec = float(st["E_cosmic"])
    ed = np.r_[0.716 * e[1], e[1:]]
    for k, b in enumerate(bins):
        ax = axs.flat[k]
        ax.stairs(st["E_a2ii_T1"][b], ed, baseline=None, color=CAT[0], lw=1.5, label="A2-ii")
        ax.stairs(st["E_hb_T1"][b], ed, baseline=None, color=CAT[1], lw=1.3, label="host-bound (D3)")
        ax.stairs(st["E_all_T1"][b], ed, baseline=None, color=INK2, lw=0.8, label="all gas")
        ax.axhline(Ec, color=INK, lw=1.0, ls=(0, (3, 2)), label=f"cosmic mean ({Ec:.3f})")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(0.03, 4.2)
        ax.set_ylim(1e-2, 3e3)
        rec = prep["J1a_by_bin"][LABELS[b]]
        ax.set_title(f"{LABELS[b]}: bound/A2-ii at 1-2R {rec['bands']['1-2R']['hb_over_A2ii']:.2f}, "
                     f"3-4R {rec['bands']['3-4R']['hb_over_A2ii']:.2f}", fontsize=7.5, loc="left")
        if k % ncol == 0:
            ax.set_ylabel("E = n$_e\\mu_{XGP}$/($\\rho_c f_b$)")
        if k >= n - ncol:
            ax.set_xlabel("x = r / R$_{200c}$")
        if k == 0:
            ax.legend(fontsize=6.5, loc="lower left")
    for k in range(n, nrow * ncol):
        axs.flat[k].set_visible(False)
    fig.suptitle("Electron profiles per mass bin (T1 stacks): A2-ii partition vs gas bound (HBT) to the halo or its own "
                 "satellites, with the cosmic mean electron density in the same units", fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    save(fig, out, "J1_bound_vs_a2ii_profiles")


# ============================================================================ figure 5: kSZ bound-only vs A2-ii
def fig_ksz(D, out):
    tr, post = D["truth"], D["post"]
    ell = None
    for k in tr:
        if k.endswith("|truth__ksz1h"):
            ell = np.arange(len(tr[k]))
            break
    import importlib.util  # noqa: F401
    ellv = D["ts"].get("ell")
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.4))
    ax = axs[0]
    L = np.asarray(D["sens"]["ell"], float)
    pref = L * (L + 1) / (2 * np.pi) * T_CMB_UK**2
    for d, col, ls in (("D0", CAT[0], "-"), ("D3", CAT[3], ":")):
        for term, lw, a_ in (("ksz1h", 1.8, 1.0), ("ksz2h", 1.0, 0.9), ("ksztot", 2.4, 1.0)):
            y = tr[f"C14|{d}|T1|truth__{term}"]
            lab = {"ksz1h": "1-halo", "ksz2h": "2-halo (approx.)", "ksztot": "total"}[term]
            ax.plot(L, pref * y, color=col, lw=lw, ls={"ksz1h": "-", "ksz2h": "--", "ksztot": "-"}[term],
                    alpha=a_ if term != "ksztot" else 0.45, label=f"{d} {lab}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(50, 2e4)
    ax.set_xlabel("multipole $\\ell$")
    ax.set_ylabel("$\\ell(\\ell+1)C_\\ell/2\\pi$  [$\\mu$K$^2$]  (T$_{CMB}$ = 2.7255 K)")
    ax.set_title("(a) halo kSZ: D0 (A2-ii to 4R, kept) and D3 (host-bound only), C14", loc="left", fontsize=9.5)
    ax.legend(fontsize=7, ncol=2, loc="lower left")
    ax = axs[1]
    for cfg, mk, alpha in (("C14", "o", 1.0), ("C10", "s", 0.6)):
        for term, col in (("ksz1h", CAT[0]), ("ksz2h", CAT[1]), ("ksztot", CAT[2])):
            r = tr[f"{cfg}|D3|T1|truth__{term}"] / tr[f"{cfg}|D0|T1|truth__{term}"]
            ax.plot(L, r, color=col, lw=1.6 if cfg == "C14" else 1.0, ls="-" if cfg == "C14" else "--",
                    alpha=alpha, label=f"{term.replace('ksz', '')} {cfg}")
    fo = post["first_order_vs_exact_C10_hostbound_over_allgas"]
    for k, ellk in (("kSZ_1h_ell100", 100), ("kSZ_1h_ell1000", 1000), ("kSZ_1h_ell3000", 3000)):
        ax.scatter([ellk], [1 + fo[k]["first_order_pct"] / 100], marker="x", s=50, color=CAT[7], zorder=4,
                   label="first-order estimate, 1h (host-bound / all gas, C10)" if ellk == 100 else None)
        ax.scatter([ellk], [1 + fo[k]["exact_pct"] / 100], marker="+", s=70, color=INK, zorder=4,
                   label="exact, 1h (host-bound / all gas, C10)" if ellk == 100 else None)
    ax.axhline(1, color=INK2, lw=0.8)
    ax.set_xscale("log")
    ax.set_xlim(50, 2e4)
    ax.set_ylim(0, 1.6)
    ax.set_xlabel("multipole $\\ell$")
    ax.set_ylabel("C$_\\ell$(host-bound) / C$_\\ell$(A2-ii to 4R)")
    ax.set_title("(b) ratio D3 / D0 (solid C14; dashed C10 = fallback below 10$^{13.5}$)", loc="left", fontsize=9.5)
    # legend in the empty upper-left corner (all ratios are < 0.9 below ell = 2000), clear of the curves and markers
    ax.legend(fontsize=7, loc="upper left", ncol=2, borderaxespad=0.3)
    fig.tight_layout()
    save(fig, out, "J1_ksz_bound_vs_a2ii")


# ============================================================================ supplementary figures
def fig_cumulative(D, out):
    S = D["sens"]
    e = S["x_edges"]
    probes = [("dm_mean", None, "mean DM"), ("dm_var", None, "DM variance"), ("dmcltot", 1000, "DM-field C$_\\ell$, "
              "$\\ell$=1000"), ("ksz1h", 1000, "kSZ 1h, $\\ell$=1000")]
    fig, axs = plt.subplots(1, len(probes) + 2, figsize=(19, 3.8))
    li = {int(v): i for i, v in enumerate(S["ell"])}
    for ax, (name, ell, lab) in zip(axs, probes):
        for d in DEFS:
            c = S.get(f"trunc_all__{d}__{name}")
            if c is None:
                continue
            kB = int(np.flatnonzero(np.isclose(e, CUT[d]))[0])
            c = c[:kB + 1] if c.ndim == 1 else c[:kB + 1, li[ell]]
            C = (c - c[0]) / (c[-1] - c[0])
            ax.plot(e[1:kB + 1], C[1:], color=DCOL[d], ls=DLS[d], label=d)
        ax.set_xscale("log")
        ax.set_xlim(0.02, 4.2)
        ax.set_ylim(-0.05, 1.08)
        ax.axhline(1, color=INK2, lw=0.7)
        ax.set_xlabel("truncation radius R$_t$ / R$_{200c}$")
        ax.set_title(lab, fontsize=9.5, loc="left")
    axs[0].set_ylabel("cumulative calibrated signal C$_O$(R$_t$)")
    axs[0].legend(fontsize=8)
    for ax, qn, lab in ((axs[-2], 1, "DM Q50 (bin-mean profiles)"), (axs[-1], 4, "DM Q99.9 (bin-mean profiles)")):
        for d in DEFS:
            q = S.get(f"trunc_all__{d}__dmq_mean")
            if q is None:
                continue
            kB = q.shape[0] - 1
            denom = q[-1, qn] - q[0, qn]
            if abs(denom) < 1e-9:
                continue
            C = (q[:, qn] - q[0, qn]) / denom
            ax.plot(e[1:kB + 1], C[1:], color=DCOL[d], ls=DLS[d], label=d)
        ax.set_xscale("log")
        ax.set_xlim(0.02, 4.2)
        ax.set_ylim(-0.05, 1.08)
        ax.axhline(1, color=INK2, lw=0.7)
        ax.set_xlabel("truncation radius R$_t$ / R$_{200c}$")
        ax.set_title(lab, fontsize=9.5, loc="left")
    fig.suptitle("Cumulative signal inside each definition's boundary (calibrated haloes cut at R$_t$; reaches 1 at the "
                 "boundary)", fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    save(fig, out, "J1_cumulative_vs_radius")


def fig_shares(D, out):
    post = D["post"]
    pb = post["per_bin_C14"]
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.2), sharey=True)
    for ax, key, lab in ((axs[0], "dm_mean_share", "share of mean DM"), (axs[1], "dm_var_share", "share of DM variance")):
        bins = pb["D0"]["bins"]
        x = np.arange(len(bins))
        w = 0.2
        for j, d in enumerate(DEFS):
            ax.bar(x + (j - 1.5) * w, pb[d][key], width=w * 0.9, color=DCOL[d], label=DLAB[d],
                   hatch=["", "//", "..", "xx"][j], edgecolor="white", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(bins, rotation=60, fontsize=7.5)
        ax.set_xlabel("engine mass bin, log$_{10}$ M$_{200c}$/M$_\\odot$")
        ax.set_title(f"{lab} carried by each calibrated bin (exact, own term)", fontsize=9.5, loc="left")
        ax.grid(axis="x", visible=False)
    axs[0].set_ylabel("fraction of the definition's total")
    axs[0].legend(fontsize=7.5)
    fig.tight_layout()
    save(fig, out, "J1_mass_shares")


def fig_treatments(D, out):
    post = D["post"]
    T = post["treatment_sensitivity_rel_change"]
    stats = ["dm_mean", "dm_var", "q16", "q50", "q84", "q975", "q999", "dmcltot_1000", "dmcltot_3000"]
    fig, ax = plt.subplots(figsize=(10, 4.4))
    y = np.arange(len(stats))[::-1]
    variants = [(t, d) for d in DEFS for t in ("T2", "T3", "S82") if f"{d}|{t}" in T]
    off = np.linspace(-0.35, 0.35, len(variants))
    for k, (t, d) in enumerate(variants):
        r = T[f"{d}|{t}"]
        vals = np.array([100 * r[s] if r.get(s) is not None else np.nan for s in stats], float)
        ax.scatter(vals, y + off[k], color=DCOL[d], marker={"T2": "o", "T3": "s", "S82": "^"}[t], s=24,
                   label=f"{d} {t}", edgecolor="white", linewidth=0.6, zorder=3)
    ax.axvspan(-1, 1, color=GRID, alpha=0.7, lw=0, label="$\\pm$1%")
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(stats)
    ax.set_xlabel("change relative to T1 [%]  (T2: B16 shape inside x$_h$; T3: raw fine shells; S82: bin 12.7-12.9 "
                  "from M$_{200c}$ $\\geq$ 6.6e12 only)")
    ax.legend(fontsize=7, ncol=4, loc="lower right")
    ax.set_title("Sensitivity of the DM statistics to the treatment of the unresolved low-mass cores (C14)",
                 fontsize=9.5, loc="left")
    fig.tight_layout()
    save(fig, out, "J1_treatment_sensitivity")


def fig_measured_vs_fallback(D, out):
    post = D["post"]
    stats = [("dm_mean", "mean"), ("dm_var", "variance"), ("q16", "Q16"), ("q50", "Q50"), ("q84", "Q84"),
             ("q975", "Q97.5"), ("q999", "Q99.9"), ("dmcltot_1000", "DM C$_\\ell$ 1000"), ("ydmtot_1000", "y$\\times$DM 1000"),
             ("ksz1h_1000", "kSZ 1h 1000")]
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.2), sharey=True)
    y = np.arange(len(stats))[::-1]
    for ax, key, title in ((axs[0], "measured_over_fallback_C14_over_C10",
                            "(a) calibrating 12.7-13.5: C14 / C10 (C10 = published fallback below 10$^{13.5}$)"),
                           (axs[1], "measured_over_published_C14_over_C0",
                            "(b) FLAMINGO-calibrated / published B16 everywhere (same cut)")):
        for j, d in enumerate(DEFS):
            r = post[key][d]
            vals = np.array([r.get(s) if r.get(s) is not None else np.nan for s, _ in stats], float)
            ax.scatter(vals, y + (j - 1.5) * 0.18, color=DCOL[d], marker="osD^"[j], s=30, label=DLAB[d],
                       edgecolor="white", linewidth=0.6, zorder=3)
        ax.axvline(1, color=INK2, lw=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels([s[1] for s in stats])
        ax.set_xscale("log")
        ax.set_xticks([0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.3])
        ax.set_xticklabels(["0.1", "0.2", "0.3", "0.5", "0.7", "1", "1.3"])
        ax.minorticks_off()
        ax.set_xlim(0.1, 1.4)
        ax.set_xlabel("ratio")
        ax.set_title(title, fontsize=9, loc="left")
    axs[0].legend(fontsize=7.5, loc="lower left")
    fig.text(0.01, -0.04, "D1 Q16 lies inside the DM = 0 atom in every configuration (ratio 1 by construction). In C10 and C0 "
             "no bound-only fallback exists: D3 there paints the published B16 to 4 R$_{200c}$ in the fallback bins, so the "
             "D3 ratios also measure bound-only vs all gas in those bins.", fontsize=7.5, color=INK2, ha="left")
    fig.tight_layout()
    save(fig, out, "J1_measured_vs_fallback")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    res = HERE / ("results_smoke" if a.smoke else "results")
    post = HERE / ("post_smoke" if a.smoke else "post")
    out = HERE / ("figures_smoke" if a.smoke else "figures")
    D = load(res, post)
    for f in (fig_pdf, fig_importance, fig_lowmass, fig_bound, fig_ksz, fig_cumulative, fig_shares, fig_treatments,
              fig_measured_vs_fallback):
        f(D, out)
        print("figure", f.__name__)


if __name__ == "__main__":
    main()
