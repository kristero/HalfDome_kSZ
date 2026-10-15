"""Figures of the 'resolution' group (reads work/results_main.json, work/arrays.npz, work/redshift.json)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import res_common as R  # noqa: E402

sc = R.sc
FIG = R.GROUP / "figures"
FIG.mkdir(exist_ok=True)
W = R.GROUP / "work"
RES = json.loads((W / "results_main.json").read_text())
ARR = np.load(W / "arrays.npz")
RZ = json.loads((W / "redshift.json").read_text())
EDGES = sc.edges(R.GRID)
XV = R.x_volume(EDGES)
BINS = sc.mass_bins(R.SCHEME)
LABELS = [b["label"] for b in BINS]
LOGM = {b["label"]: b["mean_logm200c"] for b in BINS}

# categorical palette (reference instance of the dataviz skill), fixed order; sequential blue ramp for ordinals
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
BLUE = ["#b7d3f6", "#86b6ef", "#5598e7", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
ORANGE = ["#f5b896", "#ef8b5c", "#eb6834", "#c4521f", "#8f3a14"]
INK, INK2, GRID_C = "#0b0b0b", "#52514e", "#e4e3df"
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]

plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.grid": True, "grid.color": GRID_C, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                     "figure.facecolor": "white", "savefig.dpi": 150, "lines.linewidth": 1.8})


def seq(n, ramp=BLUE):
    idx = np.linspace(1, len(ramp) - 1, n).round().astype(int)
    return [ramp[i] for i in idx]


def first_edges(sid):
    m = RES["masks"][sid]
    return np.array([np.nan if m[l]["first_edge"] is None else m[l]["first_edge"] for l in LABELS])


def shade_exploratory(ax):
    ax.axvspan(12.45, 13.5, color="#f0efec", zorder=0, lw=0)
    ax.text(12.98, 0.03, "exploratory\n(< 10$^{13.5}$)", transform=ax.get_xaxis_transform(), ha="center", va="bottom",
            fontsize=8, color=INK2)


def save(fig, name):
    fig.savefig(FIG / name, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


# ------------------------------------------------------------------------------------------- 1 innermost radius
def fig_innermost():
    lm = np.array([LOGM[l] for l in LABELS])
    fig, axs = plt.subplots(2, 2, figsize=(18, 9.4), sharex=True, sharey=True,
                            gridspec_kw={"wspace": 0.62, "hspace": 0.22})
    notes = ["0, 5, 10 differ from 20 (ref) only at 12.5-12.7;\n32 only at 12.7-12.9 and 13.1-13.3:\nthe smoothing-length test binds first",
             "first edge 0 (every shell kept) is drawn on the\ndotted centre line: 0 in every bin, 5 from 14.7,\n10 from 14.9, 20 and 50 from 15.1",
             "eps factors 0, 2, 3, 6 coincide with the reference;\nh factor 0 coincides with h factor 1",
             "'per-halo (any halo)': first shell where\nat least one halo is resolved"]
    panels = [
        ("(a) minimum particles per halo-shell (other criteria at reference)",
         [("mc0", "0"), ("mc5", "5"), ("mc10", "10"), ("ref", "20 (ref)"), ("mc32", "32 (L2p8)"), ("mc50", "50"),
          ("mc100", "100"), ("mc200", "200"), ("mc500", "500")], "seq"),
        ("(b) particle count alone (softening, h, min edge off)",
         [("mconly0", "0"), ("mconly5", "5"), ("mconly10", "10"), ("mconly20", "20"), ("mconly50", "50"),
          ("mconly100", "100"), ("mconly200", "200"), ("mconly500", "500")], "seq"),
        ("(c) smoothing-length factor, softening factor, min edge",
         [("hf0", "h factor 0"), ("hf1", "h factor 1"), ("ref", "h 2, eps 4, x>=0.04 (ref)"), ("hf3", "h factor 3"),
          ("sf0", "eps factor 0"), ("sf6", "eps factor 6"), ("me0", "min edge 0"), ("me0.1", "min edge 0.1")], "cat"),
        ("(d) masking rule (reference criteria)",
         [("ref", "strict: every halo (ref)"), ("frac95", ">= 95% of haloes"), ("frac84", ">= 84% of haloes"),
          ("per_halo_f50", "per-halo, >= 50% kept"), ("per_halo", "per-halo (any halo)"),
          ("diagnostic", "diagnostic (pooled)")], "cat"),
    ]
    for ax, (title, items, mode) in zip(axs.ravel(), panels):
        cols = seq(len(items)) if mode == "seq" else CAT
        for k, (short, lab) in enumerate(items):
            sid = R.SETTING_BY_SHORT[short]["id"]
            fe = first_edges(sid)
            fe = np.where(fe <= 0, 0.012, fe)            # x = 0 (centre) drawn at the bottom
            ax.plot(lm, fe, marker=(MARKERS + ["h"])[k % (len(MARKERS) + 1)], ms=6, color=cols[k], label=lab,
                    lw=2.4 if short == "ref" else 1.5, zorder=3 if short == "ref" else 2)
        shade_exploratory(ax)
        ax.set_yscale("log")
        ax.set_ylim(0.01, 5)
        ax.set_title(title, fontsize=10, loc="left")
        ax.legend(fontsize=7.5, loc="upper left", bbox_to_anchor=(1.01, 1.0))
        ax.axhline(0.0125, color=INK2, lw=0.6, ls=":")
        ax.text(1.02, 0.02, notes[list(axs.ravel()).index(ax)], transform=ax.transAxes, fontsize=7, color=INK2,
                va="bottom")
    for ax in axs[1]:
        ax.set_xlabel(r"bin mean $\log_{10} M_{200c}$ [M$_\odot$]")
    for ax in axs[:, 0]:
        ax.set_ylabel(r"innermost kept shell edge $r/R_{200c}$")
    fig.suptitle("Innermost kept radius per 0.2-dex bin (fit grid, z = 0); dotted line = centre (all shells kept); "
                 "missing point = no shell kept", fontsize=10)
    save(fig, "fig1_innermost_radius_vs_mass.png")


# ------------------------------------------------------------------------------------------- 2 counts per shell
def fig_counts():
    cp = RES["counts_profiles"]
    fig, axs = plt.subplots(3, 5, figsize=(15, 8.6), sharex=True, sharey=True)
    for ax, lab in zip(axs.ravel(), LABELS):
        d = cp[lab]
        x = XV
        q50 = np.maximum(np.array(d["count_q50"], float), 0.3)
        q16 = np.maximum(np.array(d["count_q16"], float), 0.3)
        q84 = np.maximum(np.array(d["count_q84"], float), 0.3)
        mn = np.maximum(np.array(d["count_min"], float), 0.3)
        ax.fill_between(x, q16, q84, color=CAT[0], alpha=0.22, lw=0, label="16-84% of haloes")
        ax.plot(x, q50, color=CAT[0], label="median")
        ax.plot(x, mn, color=CAT[1], lw=1.2, ls="--", label="minimum (sets 'strict')")
        for t, ls in ((20, "-"), (100, "--"), (500, ":")):
            ax.axhline(t, color=INK2, lw=0.8, ls=ls)
        fe = RES["masks"]["ref"][lab]["first_edge"]
        if fe is not None:
            ax.axvline(fe, color=CAT[6], lw=1.2)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(0.3, 3e5)
        ax.set_title(f"{lab}  (N = {len(sc.find_bin(R.SCHEME, lab)['rows'])})", fontsize=9)
    axs[0, 0].legend(fontsize=7, loc="upper left")
    axs[0, 1].text(0.03, 0.97, "horizontal lines: 20 / 100 / 500 particles\nviolet line: reference first kept edge",
                   transform=axs[0, 1].transAxes, fontsize=7, va="top", color=INK2)
    for ax in axs[-1]:
        ax.set_xlabel(r"shell radius $r/R_{200c}$")
    for ax in axs[:, 0]:
        ax.set_ylabel("gas particles per shell")
    fig.suptitle("Gas particles per fit-grid shell (0.069 dex inside 1 R200c), all owners and phases, z = 0; "
                 "zero counts drawn at 0.3", fontsize=10)
    save(fig, "fig2_particle_counts_per_shell.png")

    fig, axs = plt.subplots(3, 5, figsize=(15, 8.6), sharex=True, sharey=True)
    for ax, lab in zip(axs.ravel(), LABELS):
        d = cp[lab]
        x = EDGES[:-1]
        ok = x > 0
        q50, q16, q84 = (np.array(d[k], float) for k in ("xh_q50", "xh_q16", "xh_q84"))
        mn = np.array(d["xh_min"], float)
        ax.fill_between(x[ok], np.maximum(q16[ok], 0.05), q84[ok], color=CAT[2], alpha=0.25, lw=0,
                        label="16-84% of haloes")
        ax.plot(x[ok], np.maximum(q50[ok], 0.05), color=CAT[2], label="median")
        ax.plot(x[ok], np.maximum(mn[ok], 0.05), color=CAT[1], lw=1.2, ls="--", label="minimum (empty = 0)")
        for t, ls in ((1, ":"), (2, "-"), (3, "--")):
            ax.axhline(t, color=INK2, lw=0.8, ls=ls)
        fe = RES["masks"]["ref"][lab]["first_edge"]
        if fe is not None:
            ax.axvline(fe, color=CAT[6], lw=1.2)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(0.05, 200)
        ax.set_title(lab, fontsize=9)
    axs[0, 0].legend(fontsize=7, loc="upper left")
    for ax in axs[-1]:
        ax.set_xlabel(r"shell inner edge $r_{\rm in}/R_{200c}$")
    for ax in axs[:, 0]:
        ax.set_ylabel(r"$r_{\rm in}$ / upper-median $h$")
    fig.suptitle("Inner edge in units of the shell's upper-median SPH smoothing length (lines at 1, 2 (reference), 3)",
                 fontsize=10)
    save(fig, "fig2b_inner_edge_over_smoothing_length.png")


# ------------------------------------------------------------------------------------------- 3 per-halo bias
def fig_per_halo():
    pb = RES["per_halo_bias"]
    fields = [("P", r"$P_e$"), ("E", r"$n_e$"), ("G", r"$\rho_{\rm gas}$"), ("T_e", r"$T_e$ (electron-weighted)"),
              ("K", r"$K$ (entropy)")]
    cal = [l for l in LABELS if l in R.CAL_LABELS]
    cols = seq(len(cal))
    fig, axs = plt.subplots(1, 5, figsize=(17, 4.4), sharex=True)
    for ax, (f, name) in zip(axs, fields):
        for k, lab in enumerate(cal):
            d = pb[f][lab]
            fr = np.array(d["fraction_kept"], float)
            r = np.array([np.nan if v is None else v for v in d["ratio"]], float)
            e = np.array([np.nan if v is None else v for v in d["err"]], float)
            sel = (fr > 0) & (fr < 0.999) & np.isfinite(r)
            if sel.any():
                ax.errorbar(fr[sel], r[sel], yerr=e[sel], fmt=MARKERS[k % 8], ms=5, color=cols[k], lw=0.8,
                            capsize=0, label=lab)
        ax.axhline(1, color=INK2, lw=0.8)
        ax.set_xscale("log")
        ax.set_title(name, fontsize=10)
        if f in ("P", "E", "G"):
            ax.set_yscale("log")
    axs[0].set_ylabel("per-halo-masked mean / mean over all haloes")
    fig.supxlabel("fraction of the bin's haloes resolved in the shell (inclusion-weighted)", fontsize=10)
    axs[-1].legend(fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5), title="log M200c bin", title_fontsize=7)
    fig.suptitle("Per-halo masking keeps only the resolved (denser, cooler-entropy) haloes: bias of partially resolved "
                 "shells vs the fraction kept (z = 0, calibrated bins; 1-sigma paired bootstrap)", fontsize=10)
    save(fig, "fig3_per_halo_bias_vs_fraction.png")


# ------------------------------------------------------------------------------------------- 4 'none' inner shells
def fig_none_inner():
    sel = ["13.5-13.7", "13.9-14.1", "14.3-14.5", "14.9-15.1"]
    fig, axs = plt.subplots(2, 4, figsize=(16, 7.6), sharex=True)
    for j, lab in enumerate(sel):
        for i, (f, name, fam) in enumerate((("P", r"$P_e/P_{200}$", "B12"), ("E", r"$n_e\,\mu/(\rho_c f_b)$", "B16"))):
            ax = axs[i, j]
            v = ARR[f"stack|ref|{f}|{lab}|value"]
            er = ARR[f"stack|ref|{f}|{lab}|err"]
            kept = ARR[f"stack|ref|{f}|{lab}|resolved"].astype(bool)
            ok = v > 0
            ax.errorbar(XV[kept & ok], v[kept & ok], yerr=er[kept & ok], fmt="o", ms=4, color=CAT[0], lw=0.8,
                        label="stack, kept (reference)")
            ax.errorbar(XV[~kept & ok], v[~kept & ok], yerr=er[~kept & ok], fmt="o", ms=5, mfc="white",
                        color=CAT[1], lw=0.8, label="stack, rejected ('none' keeps)")
            for outer, ls, c in ((2, "-", INK), (4, "--", CAT[2])):
                key = f"fit|ref|{f}|{outer}|{lab}|model"
                if key in ARR:
                    ax.plot(XV, ARR[key], ls=ls, color=c, lw=1.4,
                            label=f"B12 (P) / B16 (n_e) fit to {outer} R200c, reference mask")
            key = f"fit|none|{f}|2|{lab}|model"
            if key in ARR:
                ax.plot(XV, ARR[key], ls=":", color=CAT[7], lw=1.6, label="B12 / B16 fit to 2 R200c, no mask")
            # what the rejected shells contain: hot (X-ray) phase alone, and P with recently AGN-heated gas kept
            alt_key, alt_lab, alt_c = ((f"stack|ref_P_agn_kept|{lab}|value", "stack with recent-AGN gas kept, rejected",
                                        CAT[3]) if f == "P" else
                                       (f"stack|ref_E_xray|{lab}|value", "hot (X-ray) phase n_e alone, rejected",
                                        CAT[6]))
            va = ARR[alt_key]
            sel = ~kept & (va > 0)
            ax.plot(XV[sel], va[sel], ls="none", marker="D" if f == "P" else "s", ms=4.5, mfc="none", mew=1.3,
                    color=alt_c, label=alt_lab)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_title(f"{lab}: {name}", fontsize=9)
            if i == 0:
                nz = np.array(RES["counts_profiles"][lab]["count_zero_fraction"], float)
                ax.text(0.02, 0.03, "haloes with zero gas particles\nat r/R200c = 0.027, 0.052, 0.11: " + ", ".join(
                    f"{nz[s]:.0%}" for s in (1, 5, 10)), transform=ax.transAxes, fontsize=7, color=INK2)
            if i == 1:
                ax.set_xlabel(r"$r/R_{200c}$")
    axs[0, 0].set_ylabel(r"$P_e/P_{200}$")
    axs[1, 0].set_ylabel(r"$n_e\,\mu_{\rm XGP}/(\rho_c f_b)$")
    hs, ls_ = [], []
    for ax in (axs[0, -1], axs[1, -1]):
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in ls_:
                hs.append(h)
                ls_.append(l)
    fig.legend(hs, ls_, fontsize=7.5, loc="upper left", bbox_to_anchor=(0.905, 0.88))
    fig.suptitle("Unmasked inner shells (below the reference resolution limit; z = 0, mean stacks, 1-sigma bootstrap "
                 "errors). Against the inward extrapolation of the reference 2R fit: P lies below it at 0.025-0.056 "
                 "R200c in every bin shown (to 0.105 R200c in 13.5-13.7),\n2-12% above it at 0.105-0.32 R200c in "
                 "13.5-13.7 and 0.056-0.17 R200c in 13.9-14.1, and at 1.40x the extrapolation in the central 0-0.025 "
                 "sphere of 13.5-13.7 (below it there in the other bins shown); n_e lies at or above it", fontsize=10)
    save(fig, "fig4_unmasked_inner_shells.png")


# ------------------------------------------------------------------------------------------- 5 fit parameters
FIT_SETS = [("mc0", "0 particles"), ("mc100", ">=100 particles"), ("mc500", ">=500 particles"),
            ("hf1", "h factor 1"), ("hf3", "h factor 3"), ("frac95", ">=95% haloes"), ("per_halo", "per-halo"),
            ("none", "no mask")]


def fig_fit_params(outer):
    fits = RES["fits"]
    cal = [l for l in LABELS if l in R.CAL_LABELS]
    lm = np.array([LOGM[l] for l in cal])
    names = {"P": ("ln P0", "ln x_c", "ln beta"), "E": ("ln rho0", "ln alpha", "ln beta")}
    fig, axs = plt.subplots(2, 3, figsize=(14, 7.6), sharex=True)
    for i, f in enumerate(("P", "E")):
        for p in range(3):
            ax = axs[i, p]
            for k, (short, lab) in enumerate(FIT_SETS):
                sid = R.SETTING_BY_SHORT[short]["id"]
                y, e = [], []
                for l in cal:
                    d = fits[sid][f][f"{outer}R"][l]
                    if d.get("dln_params_vs_ref") is None:
                        y.append(np.nan)
                        e.append(np.nan)
                    else:
                        y.append(d["dln_params_vs_ref"][p] if d["dln_params_vs_ref"][p] is not None else np.nan)
                        ee = d["dln_params_vs_ref_err"][p]
                        e.append(np.nan if ee is None else ee)
                ax.errorbar(lm + (k - 3.5) * 0.008, y, yerr=e, fmt=MARKERS[k], ms=5, color=CAT[k], lw=1.0,
                            capsize=0, label=lab)
            ax.axhline(0, color=INK2, lw=0.8)
            ax.set_title(f"{'B12 pressure' if f == 'P' else 'B16 electrons'}: $\\Delta$ {names[f][p]}", fontsize=9)
            if i == 1:
                ax.set_xlabel(r"bin mean $\log_{10} M_{200c}$ [M$_\odot$]")
    axs[0, 0].set_ylabel("ln(parameter / reference-fit parameter)")
    axs[1, 0].set_ylabel("ln(parameter / reference-fit parameter)")
    axs[0, -1].legend(fontsize=7.5, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.suptitle(f"Fitted-parameter shifts when the inner fit limit follows each resolution setting (outer limit "
                 f"{outer} R200c; 1-sigma paired bootstrap)", fontsize=10)
    save(fig, f"fig5_fit_parameter_shifts_{outer}R.png")


# ------------------------------------------------------------------------------------------- 6 integrals vs truth
def fig_integrals():
    fits = RES["fits"]
    cal = [l for l in LABELS if l in R.CAL_LABELS]
    lm = np.array([LOGM[l] for l in cal])
    sets = [("ref", "reference")] + FIT_SETS
    panels = [("P", "2", "int_lt1_vs_truth", r"$Y(<R_{200c})$, B12 fit to 2 R200c"),
              ("P", "4", "int_lt1_vs_truth", r"$Y(<R_{200c})$, B12 fit to 4 R200c"),
              ("P", "4", "int_lt4_vs_truth", r"$Y(<4R_{200c})$, B12 fit to 4 R200c"),
              ("E", "4", "int_lt4_vs_truth", r"$N_e(<4R_{200c})$, B16 fit to 4 R200c")]
    fig, axs = plt.subplots(1, 4, figsize=(17, 4.6), sharex=True)
    cols = [INK] + CAT
    mk = ["o"] + MARKERS
    for ax, (f, outer, key, title) in zip(axs, panels):
        for k, (short, lab) in enumerate(sets):
            sid = R.SETTING_BY_SHORT[short]["id"]
            y, e = [], []
            for l in cal:
                d = fits[sid][f][f"{outer}R"][l]
                r = d.get(key) or {}
                y.append(np.nan if r.get("ratio") is None else r["ratio"] - 1)
                e.append(np.nan if r.get("err") is None else r["err"])
            ax.errorbar(lm + (k - 4) * 0.008, 100 * np.array(y), yerr=100 * np.array(e), fmt=mk[k], ms=5,
                        color=cols[k], lw=1.0, capsize=0, label=lab)
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel(r"bin mean $\log_{10} M_{200c}$ [M$_\odot$]")
    axs[0].set_ylabel("fit integral / particle-sum integral - 1 [%]")
    axs[-1].legend(fontsize=7.5, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.suptitle("Integrated signal of the fitted profile vs the resolution-independent particle sums of the stack "
                 "(all shells, z = 0; 1-sigma paired bootstrap)", fontsize=10)
    save(fig, "fig6_fit_integrals_vs_particle_sums.png")


# ------------------------------------------------------------------------------------------- 7 observables
def fig_observables():
    imp = RES["impacts"]
    obs = [("P", "tSZ_yy_total_ell1000", r"tSZ $C_\ell$, $\ell$=1000"),
           ("P", "tSZ_yy_total_ell3000", r"tSZ $C_\ell$, $\ell$=3000"),
           ("E", "kSZ_1h_ell3000", r"kSZ 1h, $\ell$=3000"),
           ("E", "DM_mean", "mean DM"), ("E", "DM_var", "DM variance")]
    sets = [("ref", "reference")] + [s for s in FIT_SETS] + [("mc50", ">=50 particles"), ("mc200", ">=200 particles"),
                                                            ("frac84", ">=84% haloes"), ("diagnostic", "diagnostic"),
                                                            ("per_halo_f50", "per-halo >=50%"), ("sf6", "eps factor 6"),
                                                            ("me0.1", "min edge 0.1")]
    order = ["ref", "mc0", "mc50", "mc100", "mc200", "mc500", "hf1", "hf3", "sf6", "me0.1", "frac95", "frac84",
             "diagnostic", "per_halo", "per_halo_f50", "none"]
    labels = dict(sets)
    fig, axs = plt.subplots(1, 2, figsize=(14, 7.4), sharey=True)
    for ax, (tag, title) in zip(axs, (("fit4R_vs_ref_fit", "change vs the reference-mask fit (fit to 4 R200c)"),
                                      ("fit4R_vs_truth", "painted fit vs measured stack truth (fit to 4 R200c)"))):
        y0 = np.arange(len(order))
        for j, (f, o, lab) in enumerate(obs):
            vals, errs = [], []
            for short in order:
                d = imp[R.SETTING_BY_SHORT[short]["id"]][tag][f][o]
                vals.append(100 * d["dO_over_O"] if d["dO_over_O"] is not None else np.nan)
                errs.append(100 * d["err"] if d.get("err") is not None else np.nan)
            ax.errorbar(vals, y0 + (j - 2) * 0.14, xerr=errs, fmt=MARKERS[j], ms=6, color=CAT[j], lw=1.0, capsize=0,
                        label=lab)
        ax.axvline(0, color=INK2, lw=0.8)
        ax.set_yticks(y0)
        miss = RES["impacts"]
        ax.set_yticklabels([labels.get(s, s) + (" (*)" if any(
            miss[R.SETTING_BY_SHORT[s]["id"]]["missing_fit_bins"]["fit4R"][f]["variant"] for f in ("P", "E")) else "")
            for s in order])
        ax.invert_yaxis()
        ax.set_xlabel(r"$\Delta O / O$ [%] (first order, calibrated bins 13.5-15.5)")
        ax.set_title(title, fontsize=10)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, fontsize=9, bbox_to_anchor=(0.5, -0.04))
    m5 = RES["impacts"]["resolution:mc500"]
    mb = ", ".join(sorted(set(m5["missing_fit_bins"]["fit4R"]["P"]["variant"] + m5["missing_fit_bins"]["fit4R"]["E"]["variant"])))
    ms_t = m5["fit4R_vs_ref_fit"]["P"]["tSZ_yy_total_ell3000"]["missing_share"]
    ms_d = m5["fit4R_vs_ref_fit"]["E"]["DM_mean"]["missing_share"]
    fig.text(0.5, -0.075, "(*) a calibrated bin keeps fewer than 6 shells, so it has no fit and enters as 'no change' "
             f"(>= 500 particles: {mb}; missing importance share {100 * ms_t:.0f}% of tSZ(3000), {100 * ms_d:.0f}% "
             "of mean DM: that DM value is not a measurement)", ha="center", fontsize=8.5, color=INK2)
    fig.suptitle("Observable impact of the resolution setting through the B12/B16 fit (inner limit from the setting); "
                 "1-sigma paired bootstrap", fontsize=10)
    save(fig, "fig7_observable_impact.png")


# ------------------------------------------------------------------------------------------- 8 shells per setting
def fig_shell_count():
    order = ["mc0", "mc5", "mc10", "ref", "mc32", "mc50", "mc100", "mc200", "mc500", "mconly0", "mconly5", "mconly10",
             "mconly20", "mconly50", "mconly100", "mconly200", "mconly500", "sf0", "sf2", "sf3", "sf6", "hf0", "hf1",
             "hf3", "me0", "me0.1", "frac95", "frac84", "diagnostic", "per_halo_f50", "per_halo", "none"]
    M = np.array([[RES["masks"][R.SETTING_BY_SHORT[s]["id"]][l]["n_kept"] for l in LABELS] for s in order], float)
    fig, ax = plt.subplots(figsize=(12, 9))
    im = ax.imshow(M, cmap=matplotlib.colors.LinearSegmentedColormap.from_list("b", ["#f0efec"] + BLUE),
                   aspect="auto", vmin=0, vmax=35)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, int(M[i, j]), ha="center", va="center", fontsize=7,
                    color="white" if M[i, j] > 22 else INK)
    ax.set_xticks(range(len(LABELS)))
    ax.set_xticklabels(LABELS, rotation=45, ha="right")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([R.SETTING_BY_SHORT[s]["label"] for s in order], fontsize=7.5)
    ax.set_xlabel(r"0.2-dex $\log_{10} M_{200c}$ bin [M$_\odot$]")
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.03)
    cb.set_label("kept shells out of 35 (to 4 R200c)")
    ax.set_title("Number of kept fit-grid shells per setting and bin (z = 0)", fontsize=10)
    save(fig, "fig8_kept_shells_per_setting.png")


# ------------------------------------------------------------------------------------------- 9 redshift
def fig_redshift():
    emp = RZ["L2p8_empirical"]
    zs = sorted(emp, key=float)
    bins = ["13.0-13.2", "13.2-13.4", "13.4-13.6", "13.6-13.8", "13.8-14.0", "14.0-14.2", "14.2-14.4", "14.4-14.6"]
    cols = seq(len(bins))
    fig, axs = plt.subplots(1, 2, figsize=(14.5, 4.8), gridspec_kw={"wspace": 0.7})
    for k, b in enumerate(bins):
        z = [float(zz) for zz in zs if b in emp[zz]["bins"]]
        fe = [emp[zz]["bins"][b]["frac95_first_edge"] for zz in zs if b in emp[zz]["bins"]]
        axs[0].plot(z, [np.nan if v is None else v for v in fe], marker=MARKERS[k % 8], color=cols[k], label=b)
        xh = [emp[zz]["bins"][b]["median_x_over_h_at_x0.1"] for zz in zs if b in emp[zz]["bins"]]
        axs[1].plot(z, xh, marker=MARKERS[k % 8], color=cols[k], label=b)
    ana = RZ["analytic_L1m9_bins"]
    za = sorted(ana, key=float)
    for lab, ls in (("13.5-13.7", "--"), ("13.1-13.3", ":")):
        axs[0].plot([float(z) for z in za], [ana[z]["bins"][lab]["softening_limit_x"] for z in za], ls=ls, color=INK2,
                    lw=1.2, label=f"max(0.04, 4 eps/R200c)\nL1 {lab}, self-similar")
    axs[0].set_yscale("log")
    axs[0].set_xlabel("redshift z")
    axs[0].set_ylabel(r"first kept shell edge $r/R_{200c}$ (>= 95% of haloes resolved)")
    axs[0].set_title("(a) L2p8_m9 stacks: first kept shell (reference criteria)", fontsize=10)
    axs[0].legend(fontsize=7, ncol=1, loc="upper left", bbox_to_anchor=(1.01, 1.0), title="log M200c",
                  title_fontsize=7)
    axs[1].axhline(2, color=INK2, lw=0.8)
    axs[1].text(0.0, 2.05, "2 h (reference criterion)", fontsize=8, color=INK2, va="bottom")
    axs[1].set_ylim(0, None)
    axs[1].set_xlabel("redshift z")
    axs[1].set_ylabel(r"median $r_{\rm in}/h$ in the shell at $r \approx 0.1 R_{200c}$")
    axs[1].set_title("(b) inner edge / upper-median smoothing length at 0.1 R200c", fontsize=10)
    fig.suptitle("Redshift dependence at fixed M200c (FLAMINGO L2p8_m9, same particle mass as L1_m9; "
                 "softening 5.7 physical kpc below z = 2.91)", fontsize=10)
    save(fig, "fig9_redshift_dependence.png")


# ------------------------------------------------------------------------------------------- 10 per-halo band heatmap
def fig_band_heatmap():
    bands = RES["stack_band_ratios"]
    fig, axs = plt.subplots(1, 2, figsize=(15, 5.2), gridspec_kw={"wspace": 0.35})
    for ax, f in zip(axs, ("P", "E")):
        M = np.full((len(LABELS), len(R.BAND_LABELS)), np.nan)
        for i, l in enumerate(LABELS):
            for j, b in enumerate(R.BAND_LABELS):
                r = bands["resolution:per_halo"][f][l][b]["ratio"]
                M[i, j] = np.nan if r is None else 100 * (r - 1)
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
            "d", ["#1c5cab", "#86b6ef", "#f0efec", "#ef8b5c", "#c4521f"])
        im = ax.imshow(M, cmap=cmap, vmin=-100, vmax=100, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if np.isfinite(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:+.0f}", ha="center", va="center", fontsize=7, color=INK)
        ax.set_xticks(range(len(R.BAND_LABELS)))
        ax.set_xticklabels(R.BAND_LABELS, rotation=30)
        ax.set_yticks(range(len(LABELS)))
        ax.set_yticklabels(LABELS, fontsize=8)
        ax.set_xlabel(r"radial band [$R_{200c}$]")
        ax.set_ylabel(r"$\log_{10} M_{200c}$ bin")
        ax.grid(False)
        ax.set_title(f"{'pressure' if f == 'P' else 'electron density'}: per-halo / reference - 1 [%]", fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.04).set_label("%")
    fig.suptitle("Per-halo masking vs the all-halo mean, median over the shells each band keeps (blank = no kept shell)",
                 fontsize=10)
    save(fig, "fig10_per_halo_band_ratios.png")


if __name__ == "__main__":
    fig_innermost()
    fig_counts()
    fig_per_halo()
    fig_none_inner()
    for outer in (2, 4):
        fig_fit_params(outer)
    fig_integrals()
    fig_observables()
    fig_shell_count()
    fig_redshift()
    fig_band_heatmap()
