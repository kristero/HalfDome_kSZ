"""Test G, local post-processing: figures (PNG + SVG), report tables (Markdown + CSV), verification.json and
SHA256SUMS, from the fetched maas results (or results_smoke).

  python -B make_figures.py [--results results] [--smoke]

Colour conventions (reference palette of the dataviz skill): models keep one colour everywhere -
E0/P0 blue, E1/P1 orange, E2gf/P2gf aqua, E3/P2m yellow, E2g6/P2g6 magenta, P2i green, P2s violet, E3pb red;
mass bins use the sequential blue ramp (light = low mass).  Figures are static (report), light surface.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.lines  # noqa: E402,F401
import matplotlib.patches  # noqa: E402,F401
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
INK, INK2, MUTED, GRID, BASE, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
MCOL = {"E0": SLOTS[0], "P0": SLOTS[0], "E1": SLOTS[1], "P1": SLOTS[1], "E2gf": SLOTS[2], "P2gf": SLOTS[2],
        "E3": SLOTS[3], "P2m": SLOTS[3], "E2g6": SLOTS[4], "P2g6": SLOTS[4], "P2i": SLOTS[5], "P2s": SLOTS[6],
        "E3pb": SLOTS[7]}
BLUE_RAMP = ["#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab",
             "#184f95", "#104281", "#0d366b"]
DIVERGE = ("#2a78d6", "#f0efec", "#e34948")
plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "axes.edgecolor": BASE, "axes.labelcolor": INK2,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "axes.facecolor": SURFACE, "figure.facecolor": SURFACE,
                     "axes.titlesize": 9.5, "axes.titlecolor": INK, "legend.frameon": False, "lines.linewidth": 2.0,
                     "savefig.facecolor": SURFACE})


def bin_colours(n):
    idx = np.linspace(1, len(BLUE_RAMP) - 1, n).round().astype(int)
    return [BLUE_RAMP[i] for i in idx]


def save(fig, name, figdir):
    figdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figdir / f"{name}.png", dpi=150, bbox_inches="tight")
    fig.savefig(figdir / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def md_table(df, floatfmt=".3g"):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                cells.append("" if not np.isfinite(v) else format(v, floatfmt))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


class Ctx:
    def __init__(self, res):
        self.res = res
        self.tab = res / "tables"
        self.prep = dict(np.load(res / "prepare.npz"))
        self.prepj = json.loads((res / "prepare.json").read_text())
        self.labels = self.prepj["bins"]["labels"]
        self.logm = np.asarray(self.prepj["bins"]["mean_logm"], float)
        self.edges = None
        prof = res / "f_base" / "profiles.npz"
        if prof.exists():
            z = np.load(prof)
            self.edges = z["f_edges"]

    def csv(self, name):
        p = self.tab / name
        return pd.read_csv(p) if p.exists() and p.stat().st_size > 0 else None

    def json(self, name):
        p = self.res / name
        return json.loads(p.read_text()) if p.exists() else None


# ============================================================================ figures
def fig_normalisation(C, figdir):
    E_G, P_ET = C.prep["norm_E_over_G"], C.prep["norm_P_over_ET"]
    bands = C.prepj["normalisation"]["E_over_G_bands"]
    cols = bin_colours(len(C.labels))
    fig, axs = plt.subplots(1, 2, figsize=(9.5, 3.4))
    for ax, arr, ref, lab, title in ((axs[0], E_G, C.prepj["normalisation"]["KAPPA_E_theory_X0.752"], "E / G",
                                      "Electrons per gas density: E/G = n_e mu_XGP / rho_gas"),
                                     (axs[1], P_ET, C.prepj["normalisation"]["KAPPA_P_code"], "P/P200 / [(E/200)(T_M/T200)]",
                                      "Pressure vs (E/200)(T_M/T200)")):
        for c in range(arr.shape[0]):
            ax.plot(range(len(bands)), arr[c], color=cols[c], lw=1.5, marker="o", ms=3)
        ax.axhline(ref, color=INK2, lw=1, ls="--")
        ax.set_xticks(range(len(bands)), [b + " R" for b in bands])
        ax.set_ylabel(lab)
        ax.set_title(title, loc="left")
    axs[0].text(0.02, 0.05, f"dashed: (1+X)/2 mu_XGP/m_p = {C.prepj['normalisation']['KAPPA_E_theory_X0.752']:.4f} (X = 0.752)",
                transform=axs[0].transAxes, color=INK2, fontsize=8)
    axs[1].text(0.02, 0.05, f"dashed: 0.59 m_p / mu_XGP = {C.prepj['normalisation']['KAPPA_P_code']:.4f}",
                transform=axs[1].transAxes, color=INK2, fontsize=8)
    sm = plt.cm.ScalarMappable(cmap=matplotlib.colors.ListedColormap(cols),
                               norm=matplotlib.colors.BoundaryNorm(np.arange(len(cols) + 1) - 0.5, len(cols)))
    cb = fig.colorbar(sm, ax=axs, ticks=range(len(cols)), fraction=0.025)
    cb.ax.set_yticklabels(C.labels, fontsize=7)
    cb.set_label("log10 M200c bin", color=INK2)
    save(fig, "G1_normalisation", figdir)


def fig_two_halo(C, figdir):
    bands = C.prepj["normalisation"]["E_over_G_bands"]
    sel = [k for k, b in enumerate(bands) if float(b.split("-")[0]) >= 1.5]
    fig, axs = plt.subplots(1, 2, figsize=(9.5, 3.4), sharey=True)
    for ax, s, title in ((axs[0], "main", "All gas"), (axs[1], "a2ii", "A2-ii (painting partition)")):
        r = C.prep[f"ratio2h_{s}"]
        rb = C.prep[f"ratio2h_brief_{s}"]
        for j, k in enumerate(sel):
            ax.plot(C.logm, r[:, k], color=SLOTS[j], marker="o", ms=4, label=f"{bands[k]} R200c")
            ax.plot(C.logm, rb[:, k], color=SLOTS[j], marker="o", ms=3, lw=0.8, alpha=0.45, ls=":")
        ax.axhline(1, color=INK2, lw=1)
        ax.set_xlabel("log10 M200c [Msun] (bin mean)")
        ax.set_title(title, loc="left")
    axs[0].set_ylabel("stacked E / Ebar [1 + b xi_lin]")
    axs[0].legend(loc="upper right", fontsize=8)
    fig.text(0.01, -0.04, "Solid: Ebar = KAPPA_E Omega_m = %.4f (used: measured E/G of the 2-4 R200c gas x Omega_m). "
             "Dotted: Ebar = Omega_m = %.3f (the brief's convention)." % (C.prepj["normalisation"]["Ebar_used"],
                                                                          C.prepj["normalisation"]["Ebar_brief_Omega_m"]),
             color=INK2, fontsize=8)
    save(fig, "G2_two_halo_ratio", figdir)


def fig_temperature(C, fz, figdir):
    if C.edges is None:
        return
    x = np.sqrt(np.maximum(C.edges[:-1], 1e-3) * C.edges[1:])
    Te = C.prep["temp_shell_Te_main"]
    params = json.loads(str(fz["params_json"]))
    nb = Te.shape[0]
    pick = sorted(set(np.linspace(0, nb - 1, min(nb, 4)).round().astype(int)))
    fig, axs = plt.subplots(1, len(pick), figsize=(2.6 * len(pick), 3.0), sharey=True)
    axs = np.atleast_1d(axs)
    xf = np.geomspace(1.0, 4.0, 60)
    for ax, c in zip(axs, pick):
        ok = (x > 0.3) & (x < 4)
        ax.plot(x[ok], Te[c, ok], color=INK, lw=2, label="measured T_e/T200")
        for name in ("P2gf", "P2m"):
            key = f"P|{name}|main|a|4"
            if key not in params:
                continue
            pg = dict(zip(params[key]["names"][1], params[key]["pg"]))
            tau = pg["tau_0"] * (xf / 2.0)**(-pg["q_tau"])
            if "p_M" in pg:
                tau = tau * 10**(-pg["p_M"] * (C.logm[c] - 14.5))
            ax.plot(xf, tau, color=MCOL[name], lw=2, label=f"tau(x), {name} (a, 4R)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks([0.5, 1, 2, 4], ["0.5", "1", "2", "4"])
        ax.set_yticks([0.05, 0.1, 0.2, 0.5, 1, 2], ["0.05", "0.1", "0.2", "0.5", "1", "2"])
        ax.minorticks_off()
        ax.set_title(C.labels[c], loc="left")
        ax.set_xlabel("x = r / R200c")
    axs[0].set_ylabel("T / T200")
    axs[0].legend(fontsize=7, loc="lower left")
    fig.suptitle("Outer gas temperature: stacked electron-weighted T_e/T200 (from T_M, Test B) vs the fitted tau(x) of the "
                 "2-halo gas in P2", x=0.01, ha="left", color=INK, fontsize=9)
    save(fig, "G3_outer_temperature", figdir)


def fig_residuals(C, fz, field, models, figdir, target="main", obj="a", rf=4.0):
    if C.edges is None:
        return
    keys = [str(k) for k in fz["keys"]]
    z = np.load(C.res / "f_base" / "profiles.npz")
    mean = z[f"mean{field}_{'main' if target == 'main' else 'a2ii'}"][[int(b) for b in range(5, 5 + len(C.labels))]] \
        if len(C.labels) == 10 else None
    if mean is None:
        cal = np.flatnonzero(z["supported"]) if "supported" in z.files else None
        return
    base = z["base_mask"][5:15]
    x = np.sqrt(np.maximum(C.edges[:-1], 1e-3) * C.edges[1:])
    fig, axs = plt.subplots(2, 5, figsize=(13, 5.0), sharex=True, sharey=True)
    for c, ax in enumerate(axs.ravel()):
        for name in models:
            key = f"{field}|{name}|{target}|{obj}|{rf:g}"
            if key not in keys:
                continue
            S = fz["S35"][keys.index(key)][c]
            ok = base[c] & (mean[c] > 0)
            ax.plot(x[ok], 100 * (S[ok] / mean[c, ok] - 1), color=MCOL[name], lw=1.6, label=name)
        ax.axhline(0, color=INK2, lw=0.8)
        ax.axvspan(rf, 4.0, color=GRID, alpha=0.4) if rf < 4 else None
        ax.set_xscale("log")
        ax.set_ylim(-30, 30)
        ax.set_title(C.labels[c], loc="left", fontsize=8.5)
    for ax in axs[1]:
        ax.set_xlabel("x = r / R200c")
    for ax in axs[:, 0]:
        ax.set_ylabel("model / stack - 1 [%]")
    axs[0, 0].legend(fontsize=7, loc="lower left")
    fig.suptitle(f"{'Pressure' if field == 'P' else 'Electrons'}: fitted to {'all gas' if target == 'main' else 'A2-ii'}, "
                 f"objective ({obj}), R_fit = {rf:g} R200c", x=0.01, ha="left", color=INK)
    save(fig, f"G4_residuals_{field}_{target}_{obj}_Rf{rf:g}", figdir)


def fig_xt(C, rec, figdir):
    if rec is None:
        return
    rm = C.prep["r200m_over_r200c"]
    fig, axs = plt.subplots(1, 2, figsize=(9.5, 3.4), sharey=True)
    for ax, field, models, band, blabel in (
            (axs[0], "E", ("E2g6", "E2gf", "E3"), (1.0 * rm, 1.5 * rm), "splashback 1.0-1.5 R200m"),
            (axs[1], "P", ("P2g6", "P2gf", "P2m", "P2s"), (1.5 * rm, 3.0 * rm), "accretion shock 1.5-2 R_sp")):
        ax.fill_between(C.logm, band[0], band[1], color=GRID, alpha=0.9, lw=0, label=blabel)
        for name in models:
            sub = rec[(rec.field == field) & (rec.model == name) & (rec.fit_target == "main") & (rec.objective == "a")
                      & (rec.R_fit == 4.0)]
            if not len(sub) or "p_x_t" not in sub:
                continue
            ax.plot(C.logm[:len(sub)], sub["p_x_t"].values, color=MCOL[name], marker="o", ms=4, label=name)
            if "x_steepest" in sub:
                ax.plot(C.logm[:len(sub)], sub["x_steepest"].values, color=MCOL[name], marker="x", ms=4, lw=0.8, ls=":")
        ax.set_yscale("log")
        ax.set_yticks([1, 1.5, 2, 3, 5, 10, 20], ["1", "1.5", "2", "3", "5", "10", "20"])
        ax.minorticks_off()
        ax.set_xlabel("log10 M200c [Msun]")
        ax.set_title(("Electrons" if field == "E" else "Pressure") + ": x_t (o) and steepest log slope (x)", loc="left")
        ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3)
    axs[0].set_ylabel("x = r / R200c (fits: all gas, objective a, R_fit = 4)")
    save(fig, "G5_transition_vs_mass", figdir)


def fig_family_grid(C, scores, figdir):
    if scores is None:
        return
    for field, fam in (("P", "rms_tSZ"), ("E", "rms_kSZ"), ("E", "rms_DM")):
        sub = scores[(scores.field == field) & (scores["eval"] == "paint") & (scores.objective == "a")
                     & (scores.prediction == "det")]
        models = [m for m in ("P0", "P1", "P2g6", "P2gf", "P2i", "P2s", "P2m", "E0", "E1", "E2g6", "E2gf", "E3")
                  if m in set(sub.model)]
        if not models or fam not in sub:
            continue
        rfs = sorted(sub.R_fit.unique())
        rcs = sorted(sub.R_cut.unique())
        fig, axs = plt.subplots(1, len(models), figsize=(2.2 * len(models), 2.6), sharey=True)
        axs = np.atleast_1d(axs)
        vmax = np.nanpercentile(100 * sub[fam].values, 90) if np.isfinite(sub[fam]).any() else 10
        for ax, m in zip(axs, models):
            M = np.full((len(rfs), len(rcs)), np.nan)
            for _, r in sub[sub.model == m].iterrows():
                M[rfs.index(r.R_fit), rcs.index(r.R_cut)] = 100 * r[fam]
            im = ax.imshow(M, origin="lower", cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
                "seq", ["#cde2fb", "#3987e5", "#0d366b"]), vmin=0, vmax=vmax, aspect="auto")
            for i in range(len(rfs)):
                for j in range(len(rcs)):
                    if np.isfinite(M[i, j]):
                        ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center", fontsize=6.5,
                                color="white" if M[i, j] > 0.55 * vmax else INK)
            ax.set_xticks(range(len(rcs)), [f"{r:g}" for r in rcs])
            ax.set_yticks(range(len(rfs)), [f"{r:g}" for r in rfs])
            ax.set_title(m, loc="left", color=INK)
            ax.set_xlabel("R_cut")
            ax.grid(False)
        axs[0].set_ylabel("R_fit")
        fig.suptitle(f"{fam.replace('rms_', '')} family RMS |dO/O| [%], painting target (A2-ii truth), "
                     f"objective (a), deterministic", x=0.01, y=1.06, ha="left", color=INK)
        if fam == "rms_DM":
            fig.text(0.01, -0.06, "Empty cells: R_cut < 2 R200c, no exact DM-PDF pass (these cuts lose 20-55% of the DM "
                     "signal; mean and variance are in G_scores.csv).", color=INK2, fontsize=8)
        save(fig, f"G6_grid_{field}_{fam.replace('rms_', '')}", figdir)


def fig_ladder(C, sel, figdir):
    if sel is None:
        return
    for field in ("P", "E"):
        for dset in ("mass", "massc"):
            sub = sel[(sel.field == field) & (sel.dset == dset)]
            if not len(sub):
                continue
            rfs = sorted(sub.R_fit.unique())
            fig, axs = plt.subplots(2, len(rfs), figsize=(3.1 * len(rfs), 5.2), sharex=False)
            axs = np.atleast_2d(axs)
            if axs.shape[0] != 2:
                axs = axs.T
            kt = [3, 4, 6, 10, 15, 30, 60, 100]
            for j, rf in enumerate(rfs):
                s = sub[sub.R_fit == rf]
                for o, col in zip(("O0", "O1", "O2"), SLOTS[:3]):
                    so = s[s.outer == o]
                    if not len(so):
                        continue
                    axs[0, j].scatter(so.k, so.chi2nu_floor, s=16, color=col, label=o, zorder=3)
                    axs[1, j].scatter(so.k, so.dBIC_floor, s=16, color=col, zorder=3)
                rung = s.loc[s.groupby("k")["chi2_floor"].idxmin()].sort_values("k")
                axs[0, j].plot(rung.k, rung.chi2nu_floor, color=MUTED, lw=0.8, zorder=2)
                axs[1, j].plot(rung.k, rung.dBIC_floor, color=MUTED, lw=0.8, zorder=2)
                chosen = s[s.selected.astype(bool)]
                for _, r in chosen.iterrows():
                    axs[0, j].scatter([r.k], [r.chi2nu_floor], s=70, facecolors="none", edgecolors=INK, lw=1.3, zorder=4)
                    axs[1, j].scatter([r.k], [r.dBIC_floor], s=70, facecolors="none", edgecolors=INK, lw=1.3, zorder=4)
                    left = r.k > 0.5 * s.k.max()
                    axs[0, j].annotate(f"{r.inner}+{r.outer}", (r.k, r.chi2nu_floor), xytext=(-6 if left else 4, 6),
                                       textcoords="offset points", fontsize=7, color=INK, ha="right" if left else "left")
                axs[0, j].axhline(1, color=INK2, lw=0.8)
                axs[1, j].axhline(6, color=INK2, lw=0.8, ls="--")
                for ax in axs[:, j]:
                    ax.set_xscale("log")
                    ax.set_xticks([k for k in kt if k <= s.k.max() * 1.2], [str(k) for k in kt if k <= s.k.max() * 1.2])
                    ax.minorticks_off()
                axs[0, j].set_yscale("log")
                axs[1, j].set_yscale("symlog", linthresh=1.0)
                axs[1, j].set_ylim(-0.3, max(10.0, float(np.nanmax(s.dBIC_floor)) * 2))
                axs[0, j].set_title(f"R_fit = {rf:g} R200c", loc="left")
                axs[1, j].set_xlabel("free parameters k")
            axs[0, 0].set_ylabel("chi^2/nu (sigma_d floor)")
            axs[1, 0].set_ylabel("dBIC (floor) to the best")
            handles = [matplotlib.lines.Line2D([], [], color=c, marker="o", lw=0, label=lab) for c, lab in
                       zip(SLOTS[:3], ("O0: no outer term", "O1: physical outer term", "O2: O1 + transition x_t(M)"))]
            handles.append(matplotlib.lines.Line2D([], [], color=MUTED, lw=0.8, label="best model at each k"))
            fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=7.5, bbox_to_anchor=(0.5, -0.04))
            fig.suptitle(f"Parsimony ladder, {'pressure' if field == 'P' else 'electrons'}, data set '{dset}' "
                         f"(circled: selected = fewest parameters with dBIC <= 6 and CV within the best's fold scatter)",
                         x=0.01, ha="left", color=INK, fontsize=9)
            save(fig, f"G9_ladder_{field}_{dset}", figdir)


def fig_scatter(C, figdir, rf=4.0, obj="a", ev="paint"):
    """Relative error vs ell of the 1-halo-dominated families for det / S1 / S2 (small multiples per model)."""
    p = C.res / "evaluate.npz"
    if not p.exists():
        return
    z = np.load(p)
    labels = [str(x) for x in z["labels"]]
    obs = [str(x) for x in z["observables"]]
    rel = z["rel_error"]
    ells = [100, 300, 1000, 3000, 8000]
    rows = (("P", ("P0", "P1", "P2gf", "P2m"), "yytot", "tSZ yy (total)"),
            ("E", ("E0", "E1", "E2gf", "E3"), "ksz1h", "kSZ 1-halo"))
    fig, axs = plt.subplots(2, 4, figsize=(12, 5.2), sharex=True)
    styles = {"det": (INK, "-"), "S1": (INK2, "--"), "S2": (MUTED, ":")}
    for i, (f, models, key, lab) in enumerate(rows):
        for j, m in enumerate(models):
            ax = axs[i, j]
            for v, (col, ls) in styles.items():
                name = f"{f}|{m}|main|{obj}|{rf:g}|Rc{rf:g}|{ev}|{v}"
                if name not in labels:
                    continue
                r = rel[labels.index(name)]
                y = [100 * r[obs.index(f"{key}_{e}")] for e in ells]
                ax.plot(ells, y, color=col, ls=ls, lw=1.8, marker="o", ms=3, label=v)
            ax.axhline(0, color=BASE, lw=0.8)
            ax.set_xscale("log")
            ax.set_xticks(ells, [str(e) for e in ells])
            ax.minorticks_off()
            ax.set_title(f"{m}: {lab}", loc="left", color=MCOL.get(m, INK))
            if j == 0:
                ax.set_ylabel("model / truth - 1 [%]")
            if i == 1:
                ax.set_xlabel("multipole ell")
    axs[0, 0].legend(fontsize=7, loc="lower left", title="prediction", title_fontsize=7)
    fig.suptitle(f"Scatter models vs the scatter-inclusive truth ({'A2-ii painting target' if ev == 'paint' else ev}), "
                 f"R_fit = R_cut = {rf:g}, objective ({obj}); det = one profile per bin, S1 = one lognormal amplitude, "
                 f"S2 = inner + outer amplitudes", x=0.01, ha="left", color=INK, fontsize=9)
    save(fig, f"G7_scatter_{ev}_{obj}_Rf{rf:g}", figdir)


def fig_cv(C, figdir):
    cv = C.csv("G_cv_scores.csv")
    sc = C.csv("G_scores.csv")
    if cv is None or sc is None:
        return
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, (f, fam) in zip(axs, (("P", "rms_tSZ"), ("E", "rms_kSZ"), ("E", "rms_DM"))):
        sub = cv[(cv.field == f) & (cv["eval"] == "paint") & (cv.prediction == "det")]
        names, ins, oof = [], [], []
        for _, r in sub.iterrows():
            m = sc[(sc.field == f) & (sc.model == r.model) & (sc.objective == r.objective) & (sc.R_fit == r.R_fit)
                   & (sc.R_cut == 4.0) & (sc["eval"] == "paint") & (sc.prediction == "det")]
            if not len(m) or not np.isfinite(r[fam]):
                continue
            names.append(f"{r.model} ({r.objective}) {r.R_fit:g}")
            ins.append(100 * float(m[fam].values[0]))
            oof.append(100 * float(r[fam]))
        y = np.arange(len(names))
        ax.barh(y - 0.18, ins, height=0.34, color=SLOTS[0], label="in-sample")
        ax.barh(y + 0.18, oof, height=0.34, color=SLOTS[1], label="out-of-fold (8 spatial folds)")
        ax.set_yticks(y, names, fontsize=7.5)
        ax.invert_yaxis()
        ax.set_xlabel(f"{fam.replace('rms_', '')} family RMS [%]")
        ax.set_title(f"{'Pressure' if f == 'P' else 'Electrons'}: {fam.replace('rms_', '')}, painting target, det",
                     loc="left")
        ax.grid(axis="y", visible=False)
    handles = [matplotlib.patches.Patch(color=SLOTS[0], label="in-sample"),
               matplotlib.patches.Patch(color=SLOTS[1], label="out-of-fold (8 spatial folds, refitted on 7/8)")]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=8, bbox_to_anchor=(0.5, -0.06))
    save(fig, "G8_cv", figdir)


def fig_pkeep(C, figdir):
    if C.edges is None:
        return
    x = 0.75 * (C.edges[1:]**4 - C.edges[:-1]**4) / (C.edges[1:]**3 - C.edges[:-1]**3)
    cols = bin_colours(len(C.labels))
    fig, axs = plt.subplots(1, 2, figsize=(9.5, 3.3), sharey=True)
    for ax, f in zip(axs, ("P", "E")):
        meas, sm = C.prep[f"pkeep_measured_{f}"], C.prep[f"pkeep_smoothed_{f}"]
        ok = x > 1.0
        for c in range(meas.shape[0]):
            ax.plot(x[ok], meas[c, ok], color=cols[c], lw=0, marker="o", ms=3)
            ax.plot(x[ok], sm[c, ok], color=cols[c], lw=1.4)
        ax.set_xscale("log")
        ax.set_title(f"p_keep, {'pressure' if f == 'P' else 'electrons'} (dots: A2-ii/all stacks; lines: smoothed)",
                     loc="left")
        ax.set_xlabel("x = r / R200c")
    axs[0].set_ylabel("A2-ii / all gas")
    save(fig, "G10_pkeep", figdir)


# ============================================================================ tables and verification
def report_tables(C, outdir):
    out = {}
    outdir.mkdir(parents=True, exist_ok=True)
    scores = C.csv("G_scores.csv")
    if scores is not None:
        for field, cols in (("P", ["rms_tSZ", "rms_tSZ_vs_mean"]), ("E", ["rms_kSZ", "rms_DM", "rms_kSZ_vs_mean",
                                                                           "rms_DM_vs_mean"])):
            sub = scores[(scores.field == field) & (scores.R_cut == 4.0) & (scores.R_fit.isin([2.0, 3.0, 4.0]))
                         & (scores["eval"].isin(["all", "paint", "a2ii"]))].copy()
            keep = ["model", "objective", "R_fit", "eval", "prediction"] + [c for c in cols if c in sub] + \
                   [c for c in sub.columns if c.startswith("profile_rms_")]
            sub = sub[keep]
            for c in cols + [c for c in sub.columns if c.startswith("profile_rms_")]:
                if c in sub and not c.startswith("profile_"):
                    sub[c] = 100 * sub[c]
            sub = sub.sort_values(["eval", "prediction", "objective", "R_fit", "model"])
            sub.to_csv(outdir / f"G_report_scores_{field}.csv", index=False)
            out[f"scores_{field}"] = len(sub)
    return out


def verification(C, args):
    v = {}
    par = C.json("parity.json")
    if par:
        v["parity_with_F"] = {"passed": par["passed"], "deposit_identity_max_abs": par["deposit_identity_max_abs"],
                              "basis_vs_F_truth_max_rel": par["basis_vs_F_truth_max_rel"],
                              "pdf_truthmean_quantiles_vs_F_max_rel": par["pdf_vs_F_max_rel"]}
    pj = C.prepj
    n = pj["normalisation"]
    v["normalisation"] = {
        "passed": bool(1.08 < n["KAPPA_E_measured_2_4R"] < 1.11 and abs(np.mean(n["P_over_ET_range_inside_1R"]) / n["KAPPA_P_code"] - 1) < 0.02),
        "E_over_G_inside_1R": n["E_over_G_range_inside_1R"], "E_over_G_2_4R": n["E_over_G_range_2_4R"],
        "KAPPA_E_measured": n["KAPPA_E_measured_2_4R"], "KAPPA_E_theory": n["KAPPA_E_theory_X0.752"],
        "P_over_ET_inside_1R": n["P_over_ET_range_inside_1R"], "KAPPA_P": n["KAPPA_P_code"],
        "Ebar_used": n["Ebar_used"], "Ebar_brief": n["Ebar_brief_Omega_m"]}
    v["B_vs_F_profiles"] = {"passed": bool(pj["B_vs_F"]["orig_row_equal"] and max(
        pj["B_vs_F"][k] for k in pj["B_vs_F"] if k.endswith("maxabs")) < 1e-9 * 1e4), **pj["B_vs_F"]}
    sv = pj["xi_bias"]["spot_values"]
    v["xi_bias_spot_values"] = {"passed": bool(abs(sv["bias"]["1e+13"] - 1.14) < 0.006 and abs(sv["bias"]["1e+14"] - 2.02) < 0.006
                                               and abs(sv["bias"]["1e+15"] - 5.45) < 0.006 and abs(sv["xi_lin"]["1"] - 7.3) < 0.06
                                               and abs(sv["xi_lin"]["2"] - 4.1) < 0.06 and abs(sv["xi_lin"]["5"] - 1.6) < 0.06
                                               and abs(sv["xi_lin"]["10"] - 0.62) < 0.015), **sv}
    if not args.smoke:
        rb = C.prep["ratio2h_brief_main"]
        brief = {"13.5-13.7": [4.49, 2.29, 1.37], "14.1-14.3": [3.47, 1.70, 1.11], "14.5-14.7": [2.83, 1.52, 1.07]}
        got = {lab: [float(rb[C.labels.index(lab), k]) for k in (3, 4, 5)] for lab in brief}
        dev = max(abs(got[l][i] / brief[l][i] - 1) for l in brief for i in range(3))
        v["brief_two_halo_ratios_reproduced"] = {"passed": bool(dev < 0.02), "max_rel_dev": dev, "G": got, "brief": brief}
        v["S1_equals_F_lognormal_4R"] = {"passed": bool(pj["S1_vs_F_lognormal_main_4R_maxabs"] < 1e-12),
                                         "max_abs": pj["S1_vs_F_lognormal_main_4R_maxabs"]}
        tb = C.prep["temp_brief_implied"]
        rng = {band: [float(tb[:, k].min()), float(tb[:, k].max())] for band, k in (("1.5-2", 3), ("2-3", 4), ("3-4", 5))}
        brief_t = {"1.5-2": [0.3, 0.9], "2-3": [0.12, 0.37], "3-4": [0.05, 0.22]}
        ok = all(abs(rng[b][1] - brief_t[b][1]) < 0.05 and abs(rng[b][0] - brief_t[b][0]) < 0.05 for b in brief_t)
        v["brief_implied_temperatures_reproduced"] = {"passed": bool(ok), "G_per_bin_range": rng, "brief": brief_t,
                                                      "note": "brief pools >= 14.9; G is per 0.2-dex bin"}
    rec = C.csv("G_fit_records.csv")
    if rec is not None and "accepted" in rec:
        f01 = rec[rec.model.isin(["E0", "P0", "E1", "P1"])]
        v["fits_test_F_machinery"] = {"bins": int(len(f01)), "accepted": int(f01.accepted.astype(bool).sum()),
                                      "accepted_main_a_R4": int(f01[(f01.fit_target == "main") & (f01.objective == "a")
                                                                    & (f01.R_fit == 4.0)].accepted.astype(bool).sum()),
                                      "note": "acceptance = Test F's flags (success, bound distance > 1e-5, Jacobian "
                                              "condition < 1e6); R_fit <= 1.5 power-law fits are degenerate by design"}
    glob = C.csv("G_fit_globals.csv")
    if glob is not None:
        phys = glob[glob.success.notna()]
        v["fits"] = {"physical_fits": int(len(phys)), "converged": int(phys.success.astype(bool).sum()),
                     "not_converged": phys[~phys.success.astype(bool)].key.tolist(),
                     "globals_at_bound": {k: b for k, b in zip(phys.key, phys.globals_at_bound) if isinstance(b, str) and b}}
        v["fits"]["passed"] = bool(len(v["fits"]["not_converged"]) <= 0.05 * max(len(phys), 1))
    # engine parity with F: E0/P0 (a) shells vs F's grid_a models, E1/P1 (a, 4/4) vs flex_env_a
    fm = ROOT / "F_observables" / "results" / "models_main.npz"
    fz = np.load(C.res / "fits.npz") if (C.res / "fits.npz").exists() else None
    if fm.exists() and fz is not None and not args.smoke:
        F = np.load(fm)
        lab = [str(s) for s in F["labels"]]
        keys = [str(k) for k in fz["keys"]]
        rcs = [float(r) for r in fz["r_cut"]]
        worst = 0.0
        n = 0
        for f, arr in (("P", F["P"]), ("E", F["E"])):
            for rf in (1.0, 1.5, 2.0, 3.0, 4.0):
                for rc in rcs:
                    fl = f"grid_a_Rf{rf:g}_Rc{rc:g}"
                    key = f"{f}|{f}0|main|a|{rf:g}"
                    if fl in lab and key in keys:
                        G = fz["B"][keys.index(key), rcs.index(rc)] + fz["O"][keys.index(key), rcs.index(rc)]
                        Fv = arr[lab.index(fl)]
                        with np.errstate(divide="ignore", invalid="ignore"):
                            rel = np.where(Fv > 0, np.abs(G / Fv - 1), np.abs(G - Fv))
                        worst = max(worst, float(np.nanmax(rel)))
                        n += 1
            key = f"{f}|{f}1|main|a|4"
            if "flex_env_a" in lab and key in keys:
                G = fz["B"][keys.index(key), rcs.index(4.0)] + fz["O"][keys.index(key), rcs.index(4.0)]
                Fv = arr[lab.index("flex_env_a")]
                with np.errstate(divide="ignore", invalid="ignore"):
                    rel = np.where(Fv > 0, np.abs(G / Fv - 1), np.abs(G - Fv))
                v.setdefault("engine_parity", {})[f"{f}1_vs_flex_env_a_max_rel"] = float(np.nanmax(rel))
        v.setdefault("engine_parity", {}).update({"grid_a_configs_compared": n, "grid_a_shells_max_rel": worst})
        fs = ROOT / "F_observables" / "results" / "tables" / "F_model_scores.csv"
        sc = C.csv("G_scores.csv")
        if fs.exists() and sc is not None:
            Fsc = pd.read_csv(fs)
            diffs = []
            for f, fam in (("P", "rms_tSZ"), ("E", "rms_kSZ"), ("E", "rms_DM")):
                for rf in (2.0, 3.0, 4.0):
                    fr = Fsc[(Fsc.model == f"grid_a_Rf{rf:g}_Rc4") & (Fsc.prediction == "deterministic")]
                    gr = sc[(sc.field == f) & (sc.model == f"{f}0") & (sc.objective == "a") & (sc.R_fit == rf)
                            & (sc.R_cut == 4.0) & (sc["eval"] == "all") & (sc.prediction == "det")]
                    if len(fr) and len(gr) and np.isfinite(gr[fam].values[0]):
                        diffs.append(abs(float(fr[fam].values[0]) - float(gr[fam].values[0])))
            v["engine_parity"]["family_rms_vs_F_scores_max_abs_diff"] = max(diffs) if diffs else None
            v["engine_parity"]["family_rms_compared"] = len(diffs)
        ep = v["engine_parity"]
        ep["passed"] = bool(ep["grid_a_shells_max_rel"] < 1e-6 and (ep.get("family_rms_vs_F_scores_max_abs_diff") or 0) < 1e-6)
    cv = C.json("cv_summary.json")
    if cv:
        cvs = C.csv("G_cv_scores.csv")
        v["cv"] = {"passed": bool(cvs is not None and len(cvs) > 0), "candidates": cv["candidates"],
                   "fold_seconds": cv.get("fold_seconds"), "rows": 0 if cvs is None else int(len(cvs))}
    ss = C.json("selection_summary.json")
    if ss:
        v["selection"] = {"passed": bool(ss.get("chosen")), "chosen": ss["chosen"], "sigma_floor": {
            k: v2["sigma_ref"] for k, v2 in ss["sigma_floor"].items()} if isinstance(ss["sigma_floor"], dict) else None}
        sm = C.csv("G_selection_models.csv")
        if sm is not None:
            v["selection"]["models"] = int(len(sm))
            v["selection"]["floor_fits_reaching_max_nfev"] = int(sm.get("max_nfev_floor", pd.Series(dtype=bool)).astype(bool).sum())
            l22 = sm[np.isfinite(sm.get("chi2nu_L22", pd.Series(dtype=float)))]
            if len(l22):
                v["selection"]["lee22_convention"] = {
                    "chi2nu_L22_mean_cov_median": float(np.median(l22.chi2nu_L22)),
                    "chi2nu_L22_per_halo_scatter_median": float(np.median(l22.chi2nu_L22_scatter)),
                    "note": "Lee22 report chi2_r = 3.3 (P) / 5.1 (E) at nu ~ 4200: of the order of the covariance-of-the-"
                            "mean values here, orders of magnitude above the per-halo-scatter values"}
    for name in ("test_G_done.json", "test_G_failed.json", "test_G_receipt.json"):
        p = C.res / name
        if p.exists():
            v.setdefault("cluster_receipts", {})[name] = json.loads(p.read_text())
    man = C.json("run_manifest.json")
    if man:
        v["run_manifest"] = [{"host": inv.get("host"), "workers": inv.get("workers"), "exit": inv.get("exit"),
                              "wall_seconds": inv.get("wall_seconds"),
                              "stages": {k: {kk: vv for kk, vv in s.items() if kk != "traceback"}
                                         for k, s in inv.get("stages", {}).items()}} for inv in man["invocations"]]
    v["all_passed"] = bool(all(x.get("passed", True) for x in v.values() if isinstance(x, dict)))
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=None, help="write figures/tables/verification here (default: G_physical_outer)")
    args = ap.parse_args()
    res = Path(args.results) if args.results else HERE / ("results_smoke" if args.smoke else "results")
    out = Path(args.out) if args.out else HERE
    figdir = out / ("figures_smoke" if args.smoke else "figures")
    tabdir = out / ("tables_smoke" if args.smoke else "tables")
    C = Ctx(res)
    fz = np.load(res / "fits.npz") if (res / "fits.npz").exists() else None
    fig_normalisation(C, figdir)
    fig_two_halo(C, figdir)
    fig_pkeep(C, figdir)
    rec = C.csv("G_fit_records.csv")
    if fz is not None:
        fig_temperature(C, fz, figdir)
        if len(C.labels) == 10:
            for rf in (2.0, 4.0):
                fig_residuals(C, fz, "E", ("E0", "E1", "E2gf", "E3"), figdir, rf=rf)
                fig_residuals(C, fz, "P", ("P0", "P1", "P2gf", "P2m"), figdir, rf=rf)
            fig_residuals(C, fz, "E", ("E0", "E1", "E2gf", "E3"), figdir, target="a2ii", rf=4.0)
            fig_residuals(C, fz, "P", ("P0", "P1", "P2gf", "P2m"), figdir, target="a2ii", rf=4.0)
    fig_xt(C, rec, figdir)
    fig_family_grid(C, C.csv("G_scores.csv"), figdir)
    if len(C.labels) == 10:
        fig_scatter(C, figdir)
        fig_scatter(C, figdir, ev="all")
        fig_cv(C, figdir)
    fig_ladder(C, C.csv("G_selection_models.csv"), figdir)
    report_tables(C, tabdir)
    v = verification(C, args)
    (out / ("verification_smoke.json" if args.smoke else "verification.json")).write_text(json.dumps(v, indent=1, default=str) + "\n")
    if not args.smoke and out == HERE:
        sums = []
        for p in sorted(HERE.rglob("*")):
            rel = p.relative_to(HERE)
            if p.is_file() and not ({"work", "__pycache__", "results_smoke", "figures_smoke", "tables_smoke"} & set(rel.parts)) \
                    and p.name != "SHA256SUMS" and not (rel.parts[0] == "results" and "f_base" in rel.parts and "work" in rel.parts):
                sums.append(f"{sha(p)}  {rel}")
        (HERE / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    print(json.dumps({"all_passed": v["all_passed"], "figures": len(list(figdir.glob('*.png')))}))


if __name__ == "__main__":
    main()
