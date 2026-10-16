"""Figures of Test J2 (local, from results_j2/ and post_j2/; J_MODE=smoke: results_j2_smoke/, figures_j2_smoke/).

Colour conventions (dataviz reference palette, the validated light-mode order; same rcParams as Test J1's figures):
model classes are categorical slots in fixed order - B16 per mass bin (S400 / strict masks) blue, B16 with the RC fits in
12.7-13.5 orange, Lee22-ladder laws aqua - always with a legend and secondary encoding (marker shape = objective, line
style = model); the truth / reference curves are near-black or grey.  Units: x = r/R200c; DM in pc cm^-3 (observer
frame, sources at z_s = 1); errors in per cent; importance densities per unit ln x (dimensionless).
Usage: python -B j2_figures.py
"""
from __future__ import annotations

import csv

import j2_common as C
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

R = C.J2_RESULTS
POST = C.HERE / ("post_j2_smoke" if C.SMOKE else "post_j2")
OUT = C.HERE / ("figures_j2_smoke" if C.SMOKE else "figures_j2")
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, GRID, REF = "#0b0b0b", "#52514e", "#e4e3df", "#3a3936"
DLAB = {"D0": "D0: A2-ii gas to 4 R$_{200c}$", "D1": "D1: A2-ii gas, sphere r $\\leq$ R$_{200c}$",
        "D2": "D2: A2-ii gas, sphere r $\\leq$ 2 R$_{200c}$", "D3": "D3: host-bound gas to 4 R$_{200c}$"}
OBJ_MARK = {"a": "o", "d": "s", "L": "^", "dI": "D", "dIT": "v"}
OBJ_LAB = {"a": "objective (a)", "d": "objective (d)", "L": "objective (L*)",
           "dI": "dI: Test I's weight formula on J1's C14 maps, refitted in 14 bins",
           "dIT": "dIT: Test I's published fits 13.5-15.5 + dI fits below"}
LEG_BOX = dict(frameon=True, facecolor="white", edgecolor="none", framealpha=0.9)
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                     "lines.linewidth": 1.6, "savefig.dpi": 150, "figure.dpi": 100})


def plain_log(ax, x=True, y=True):
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
    fmt = FuncFormatter(lambda v, _: f"{v:g}")
    for on, axis in ((x, ax.xaxis), (y, ax.yaxis)):
        if on:
            axis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
            axis.set_major_formatter(fmt)
            axis.set_minor_formatter(NullFormatter())


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def f(v):
    try:
        x = float(v)
        return x if np.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def pretty(label):
    """Human-readable painting name from an item label."""
    if not label:
        return ""
    t = label.split("|")
    if t[0] in ("a", "rc"):
        obj = {"L": "L*"}.get(t[3], t[3])
        return f"B16 per bin ({obj}), R$_{{fit}}$ = {t[4][2:]}" + (", RC fits 12.7-13.5" if t[0] == "rc" else "")
    if t[0] == "l":
        return f"{t[2]} ({'mass' if t[3] == 'mass' else 'conc.'} cells), R$_{{fit}}$ = {t[4][2:]}"
    return label


def scores():
    with open(POST / "J2_scores.csv") as fh:
        return list(csv.DictReader(fh))


def recommended(summ, D):
    rec = summ.get("recommendations", {}).get(D, {})
    return rec.get("primary", {}).get("label"), rec.get("compact", {}).get("label")


# ============================================================================ 1. DM error vs number of parameters
def fig_error_vs_k(rows, summ):
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.2), sharex=True)
    for j, D in enumerate(C.DEFS):
        prim, comp = recommended(summ, D)
        floor = f(summ["per_definition"][D]["S1_floor_DM_rms"])
        for i, (col, ylab) in enumerate((("rms_DM", "S1 painting vs scatter-inclusive truth"),
                                         ("prof_DM", "profile only: det painting vs truth mean"))):
            ax = axes[i, j]
            for r in rows:
                if r["D"] != D or r["block"] not in ("J2a", "J2b") or f(r.get(col)) is None:
                    continue
                k, y = float(r["k"]), 100 * f(r[col])
                if r["block"] == "J2b":
                    if r.get("dset") == "massc":       # concentration laws at the bin-mean c (not candidates)
                        ax.plot(k, y, "s", ms=5, mfc="white", mec=INK2, mew=0.9, zorder=3)
                        continue
                    ax.plot(k, y, "o", ms=6, mfc=CAT[2], mec="white", mew=0.8, zorder=3)
                    if r.get("selected") in ("True", True):
                        ax.plot(k, y, "o", ms=11, mfc="none", mec=INK, mew=1.2, zorder=4)
                    continue
                c = CAT[0] if r["rule"] == "S400" else CAT[1]
                filled = abs(float(r["R_fit"]) - C.DEFS[D]["R"]) < 1e-9
                ax.plot(k * (1.0 if r["rule"] == "S400" else 1.08), y, OBJ_MARK.get(r["objective"], "o"), ms=6,
                        mfc=c if filled else "white", mec=c, mew=1.1, zorder=3)
            for lab, mk in ((prim, "*"), (comp, "P")):
                rr = [r for r in rows if r["label"] == lab and f(r.get(col)) is not None]
                if rr:
                    ax.plot(float(rr[0]["k"]) * (1.08 if rr[0]["rule"] == "RC" else 1.0), 100 * f(rr[0][col]), mk,
                            ms=15 if mk == "*" else 12, mfc="none", mec=INK, mew=1.0, zorder=5)
            if i == 0:
                txt = f"recommended: {pretty(prim)}" + ("" if comp == prim else f"\ncompact: {pretty(comp)}")
                ax.text(0.03, 0.03, txt, transform=ax.transAxes, fontsize=7, va="bottom", ha="left", color=INK,
                        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=GRID, alpha=0.95), zorder=6)
            if i == 0 and floor is not None and floor > 0:
                ax.axhline(100 * floor, color=REF, ls="--", lw=1.0)

            ax.set_xscale("log")
            ax.set_yscale("log")
            plain_log(ax)
            if i == 0:
                ax.set_title(DLAB[D], fontsize=9.5)
            if j == 0:
                ax.set_ylabel(f"DM family RMS error [%]\n{ylab}", fontsize=8.5)
            if i == 1:
                ax.set_xlabel("number of free parameters k (14 mass bins)")
    h = [plt.Line2D([], [], ls="none", marker=OBJ_MARK[o], mfc=CAT[0], mec=CAT[0], label=f"B16 per bin, {OBJ_LAB[o]}")
         for o in ("a", "d", "L", "dI", "dIT")]
    h += [plt.Line2D([], [], ls="none", marker="o", mfc=CAT[1], mec=CAT[1],
                     label="B16 per bin, RC fit in 12.7-13.5 (resolved shape + core integral)"),
          plt.Line2D([], [], ls="none", marker="o", mfc="white", mec=CAT[0],
                     label="open marker: R$_{fit}$ different from the boundary"),
          plt.Line2D([], [], ls="none", marker="o", mfc=CAT[2], mec="white",
                     label="Lee22 ladder law, mass cells (every fitted model)"),
          plt.Line2D([], [], ls="none", marker="s", mfc="white", mec=INK2,
                     label="Lee22 concentration law at the bin-mean c (not a candidate)"),
          plt.Line2D([], [], ls="none", marker="o", ms=10, mfc="none", mec=INK, label="model selected by G's rule"),
          plt.Line2D([], [], color=REF, ls="--", label="S1 floor: the definition's stack painted with S1"),
          plt.Line2D([], [], ls="none", marker="*", ms=12, mfc="none", mec=INK, label="recommended painting"),
          plt.Line2D([], [], ls="none", marker="P", ms=10, mfc="none", mec=INK, label="compact law (k <= 14)")]
    fig.legend(handles=h, loc="lower center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, -0.06), **LEG_BOX)
    fig.suptitle("Test J2: DM error of every fitted painting against the number of parameters (R$_{cut}$ = boundary; "
                 "DM family = mean, variance, Q16-Q99.9 to z$_s$ = 1; D1 without Q16)", fontsize=10)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    save(fig, "J2_dm_error_vs_k")


# ============================================================================ 2. residual profiles with the importance weight
def fig_residuals(summ):
    ev = C.load_npz(R / "evaluate.npz")
    inp = C.load_npz(C.J2_INPUT)
    stacks = C.load_npz(R / "j1base" / "stacks.npz")
    X = np.asarray(ev["x_edges"], float)
    xm = C.mids(X)
    dln = np.log(X[1:] / np.where(X[:-1] > 0, X[:-1], 0.716 * X[1]))
    jb = C.J_BINS
    groups = [("12.7-13.5", [c for c, b in enumerate(jb) if b < 5]), ("13.5-14.5", [c for c, b in enumerate(jb)
                                                                                  if 5 <= b < 10]),
              ("14.5-15.5", [c for c, b in enumerate(jb) if b >= 10])]
    groups = [g for g in groups if g[1]]
    fig, axes = plt.subplots(len(C.DEFS), len(groups), figsize=(4.0 * len(groups) + 3.2, 2.6 * len(C.DEFS)),
                             sharex=True, squeeze=False)
    for i, D in enumerate(C.DEFS):
        spec = C.DEFS[D]
        key = "E_a2ii" if spec["sel"] == "a2ii" else "E_hb"
        m_fine = np.asarray(stacks[f"{key}_T1"], float)[jb]
        W = np.asarray(inp[f"wfine_DM_{D}"], float)
        pd = summ["per_definition"][D]
        prim, comp = recommended(summ, D)
        cand = [("B16 (a), R$_{fit}$ = boundary", f"a|{D}|B16|a|Rf{spec['R']:g}", CAT[0], "-"),
                ("B16 (d), R$_{fit}$ = boundary", f"a|{D}|B16|d|Rf{spec['R']:g}", CAT[1], "--"),
                ("B16 (L*), R$_{fit}$ = boundary", f"a|{D}|B16|L|Rf{spec['R']:g}", CAT[6], ":")]
        sel = [lab for lab, v in (pd.get("ladder_selected") or {}).items() if "|mass|" in lab]
        if sel:
            cand.append((f"ladder, selected: {sel[0].split('|')[2]}", sel[0], CAT[2], "-."))
        if prim not in [c[1] for c in cand]:
            cand.append((f"recommended: {pretty(prim)}", prim, INK, (0, (1, 1))))
        mk = np.asarray(inp["mask35_" + spec["sel"]], bool)
        F35e = np.asarray(inp["f35_edges"], float)
        for j, (gname, cs) in enumerate(groups):
            ax = axes[i, j]
            ax2 = ax.twinx()
            first = [F35e[np.flatnonzero(mk[jb[c]])[0]] for c in cs if mk[jb[c]].any()]
            dens = W[cs].sum(axis=0) / dln
            ok = xm <= spec["R"] + 1e-9
            ax2.fill_between(xm[ok], 0, dens[ok], step="mid", color="#bdbcb6", alpha=0.45, lw=0)
            ax2.set_ylim(0, max(dens[ok].max() * 1.15, 1e-12))
            ax2.set_yticks([])
            ax2.grid(False)
            ymax = 0.0
            for lab, key_it, col, ls in cand:
                if f"M4__{key_it}" not in ev:
                    continue
                M = np.asarray(ev[f"M4__{key_it}"], float)
                with np.errstate(divide="ignore", invalid="ignore"):
                    d = np.log(M[cs] / m_fine[cs])
                w = W[cs]
                good = np.isfinite(d) & (m_fine[cs] > 0)
                num = np.sum(np.where(good, w * d, 0), axis=0)
                den = np.sum(np.where(good, w, 0), axis=0)
                with np.errstate(invalid="ignore"):
                    dm = np.where(den > 0, num / den, np.nan)
                ax.step(xm[ok], 100 * dm[ok], where="mid", color=col, ls=ls, lw=1.4,
                        label=lab + (" (recommended)" if key_it == prim and not lab.startswith("recommended") else ""))
                strong = ok & (dens > 0.05 * dens[ok].max()) & np.isfinite(dm)
                if strong.any():
                    ymax = max(ymax, float(np.nanmax(np.abs(100 * dm[strong]))))
            ax.axhline(0, color=INK2, lw=0.8)
            if first:
                # the innermost first fitted edge among the panel's bins: in every panel that of its most massive bin
                # (resolution improves with mass; in 14.5-15.5 the two top bins tie)
                ax.axvline(min(first), color=INK2, ls=":", lw=0.9)
                ax.text(min(first), 0.03, " innermost fit-mask edge", transform=ax.get_xaxis_transform(), fontsize=6.5,
                        color=INK2, rotation=90, va="bottom")
            ax.set_xscale("log")
            ax.set_xlim(0.02, 4.2)
            lim = float(np.clip(1.25 * ymax, 15, 100))
            ax.set_ylim(-lim, lim)
            ax.set_zorder(ax2.get_zorder() + 1)
            ax.patch.set_visible(False)
            if i == 0:
                ax.set_title(f"log$_{{10}}$ M$_{{200c}}$ [M$_\\odot$] {gname}", fontsize=9.5)
            if j == 0:
                ax.set_ylabel(f"{D}: 100 ln(model/stack)\n[%, DM-weight averaged]", fontsize=8.5)
            if i == len(C.DEFS) - 1:
                ax.set_xlabel("x = r / R$_{200c}$")
            if j == len(groups) - 1:
                ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1.0), **LEG_BOX)
    fig.suptitle("Test J2: residuals of key fitted profiles per definition (lines; weighted by the DM importance over "
                 "the bins of the panel)\nand where the DM signal comes from (grey: summed DM-family importance per "
                 "unit ln x, arbitrary scale); dotted: innermost first fitted edge of the 35-shell fit masks among the "
                 "panel's bins (that of its most massive bin)",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, "J2_residual_profiles")


# ============================================================================ 3. DM PDFs of the selected models vs truth
def fig_pdfs(summ):
    ev = C.load_npz(R / "evaluate.npz")
    grid = np.asarray(ev["dm_grid"], float)
    tr = C.load_npz(C.J1_TRUTH.parent / "truth.npz", keys=[f"pdf|C14|{D}|T1|truth|density" for D in C.DEFS])
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.0), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
    for j, D in enumerate(C.DEFS):
        d = np.asarray(tr[f"pdf|C14|{D}|T1|truth|density"], float)
        cdf_full = np.cumsum(d) * 0.125
        xg = np.arange(len(d)) * 0.125 + 0.0625
        cdf_t = np.interp(grid, xg, cdf_full)
        lg = np.log(grid)
        prim, comp = recommended(summ, D)
        curves = [("truth (J1, per-halo profiles)", cdf_t, INK, "-")]
        for lab, key, col, ls in (("recommended", prim, CAT[0], "--"), ("compact", comp, CAT[2], "-.")):
            if lab == "compact" and key == prim:
                curves[-1] = (curves[-1][0].replace("recommended:", "recommended = compact:"),) + curves[-1][1:]
                continue
            if key and f"cdf__{key}|S1" in ev:
                curves.append((f"{lab}: {pretty(key)} (S1)", np.asarray(ev[f"cdf__{key}|S1"], float), col, ls))
        ax, axr = axes[0, j], axes[1, j]
        mid = np.sqrt(grid[1:] * grid[:-1])
        ymax = 0.0
        for lab, cdf, col, ls in curves:
            p = np.diff(cdf) / np.diff(lg)
            ymax = max(ymax, float(np.nanmax(p[(mid >= 1) & (mid <= 2000)])))
            ax.plot(mid, p, color=col, ls=ls, lw=1.5, label=lab)
            if col != INK:
                axr.plot(grid, 100 * (cdf - cdf_t), color=col, ls=ls, lw=1.3)
        p0 = cdf_t[0]
        # top-left corner: the PDFs are low there in every panel (no overlap with the peak)
        ax.text(0.03, 0.97, f"truth: P(DM < 0.1) = {p0:.3f}\n(sightlines crossing no halo)", transform=ax.transAxes,
                fontsize=7.5, va="top", ha="left", color=INK2,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.85))
        ax.set_xscale("log")
        ax.set_xlim(1, 2000)
        ax.set_ylim(0, 1.3 * ymax)          # headroom so that the annotation never overlaps the PDF peak
        ax.set_title(DLAB[D], fontsize=9.5, pad=62)
        axr.axhline(0, color=INK2, lw=0.8)
        axr.set_xlabel("halo DM to z$_s$ = 1 [pc cm$^{-3}$]")
        if j == 0:
            ax.set_ylabel("dP / d ln DM")
            axr.set_ylabel("CDF(model) - CDF(truth)\n[percentage points]", fontsize=8.5)
        ax.legend(fontsize=6.8, loc="lower center", bbox_to_anchor=(0.5, 1.08), **LEG_BOX)
    fig.suptitle("Test J2: halo-DM PDFs of the recommended painting and the compact law (S1 painting) against "
                 "J1's truth, per definition", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, "J2_dm_pdf_selected")


# ============================================================================ 4. low-mass: fallback vs calibrated fits
def fig_lowmass():
    rows = []
    with open(POST / "J2_lowmass.csv") as fh:
        for r in csv.DictReader(fh):
            if r["objective"] == "dIT" and r["prediction"] == "S1":
                rows.append(r)
    by = {r["observable"]: r for r in rows}
    # the observables of the report's J2.4 table (same set and order)
    obs = ([(f"yytot_{e}", f"tSZ C$_\\ell^{{yy}}$, $\\ell$={e}") for e in (100, 300, 1000, 3000)] + [("ymean", "mean y")]
           + [(f"ksz1h_{e}", f"kSZ 1-halo, $\\ell$={e}") for e in (100, 300, 1000, 3000)]
           + [(f"ksz2h_{e}", f"kSZ 2-halo, $\\ell$={e}") for e in (100, 1000, 3000)]
           + [("dm_mean", "DM mean"), ("dm_var", "DM variance")]
           + [(q, f"DM {q.upper().replace('Q975', 'Q97.5').replace('Q999', 'Q99.9')}") for q in C.Q_NAMES]
           + [(f"dmcltot_{e}", f"DM-field C$_\\ell$, $\\ell$={e}") for e in (100, 1000, 3000)]
           + [(f"ydmtot_{e}", f"y x DM, $\\ell$={e}") for e in (100, 1000, 3000)])
    obs = [(k, lab) for k, lab in obs if k in by]
    if not obs:
        return
    y = np.arange(len(obs))[::-1]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 0.33 * len(obs) + 2.0), sharey=True,
                                   gridspec_kw={"width_ratios": [1, 1.2]})
    vals = np.array([100 * (f(by[k].get("change_C14_over_C10")) or np.nan) for k, _ in obs])
    ax1.barh(y, vals, height=0.6, color=CAT[0], edgecolor="white", lw=0.5)
    for yy, v in zip(y, vals):
        if np.isfinite(v):
            ax1.text(v, yy, f" {v:+.1f}", va="center", ha="left" if v >= 0 else "right", fontsize=7, color=INK)
    ax1.axvline(0, color=INK2, lw=0.8)
    ax1.margins(x=0.14)
    ax1.set_yticks(y)
    ax1.set_yticklabels([lab for _, lab in obs], fontsize=8)
    ax1.set_xlabel("C14 fit / C10 fit - 1 [%]")
    ax1.set_title("change when calibrated fits replace\nthe published fallback in 12.7-13.5", fontsize=9)
    hgt = 0.36
    for n, (key, lab, col) in enumerate((("err_C10_fit", "Test I as published: fallback below 10$^{13.5}$ (C10)",
                                          CAT[1]),
                                         ("err_C14_fit", "new fits in 12.7-13.5 added (C14)", CAT[2]))):
        v = np.array([100 * (f(by[k].get(key)) or np.nan) for k, _ in obs])
        ax2.barh(y + (0.5 - n) * hgt, v, height=hgt * 0.92, color=col, label=lab, edgecolor="white", lw=0.5)
    ax2.axvline(0, color=INK2, lw=0.8)
    ax2.set_xlabel("painting / truth - 1 [%]")
    ax2.set_title("error against the truth (J1 A2-ii stacks to 4 R$_{200c}$,\nall 14 bins, scatter-inclusive)",
                  fontsize=9)
    ax2.legend(fontsize=7.5, loc="lower right", **LEG_BOX)
    fig.suptitle("Test J2: Test I's published fits (13.5-15.5, unchanged) with new fits in 12.7-13.5 (B12 (a) "
                 "pressure, B16 (d) electrons with Test I's kSZ+DM weight formula; R$_{fit}$ = R$_{cut}$ = 4 R$_{200c}$; "
                 "S1 painting)", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "J2_lowmass_fallback_vs_calibrated")


# ============================================================================ 5. ladder chi2/nu vs k (supplementary)
def fig_ladder():
    rows = []
    with open(POST / "J2_ladder_compare.csv") as fh:
        rows = list(csv.DictReader(fh))
    defs = [D for D in C.DEFS if any(r["source"] == f"J2 {D}" for r in rows)]
    fig, axes = plt.subplots(1, len(defs), figsize=(4 * len(defs), 3.6), squeeze=False)
    for j, D in enumerate(defs):
        ax = axes[0, j]
        for n, rf in enumerate(C.DEFS[D]["lad_rfit"]):
            for ds, mk in (("mass", "o"), ("massc", "s")):
                rr = [r for r in rows if r["source"] == f"J2 {D}" and r["dset"] == ds and abs(float(r["R_fit"]) - rf)
                      < 1e-9]
                if not rr:
                    continue
                k = np.array([float(r["k"]) for r in rr])
                c2 = np.array([float(r["chi2nu_floor"]) for r in rr])
                o = np.argsort(k)
                ax.plot(k[o], c2[o], marker=mk, color=CAT[n], ms=4, lw=1.0, ls="-" if ds == "mass" else ":",
                        label=f"J2 {D}, R$_{{fit}}$ = {rf:g}, {ds}")
                s = [r for r in rr if r["selected"] == "True"]
                for r in s:
                    ax.plot(float(r["k"]), float(r["chi2nu_floor"]), "o", ms=10, mfc="none", mec=INK, mew=1.1)
                    ax.annotate(r["model"], (float(r["k"]), float(r["chi2nu_floor"])), xytext=(5, 6),
                                textcoords="offset points", fontsize=7)
        if D == "D0":
            rr = [r for r in rows if r["source"].startswith("Test I") and r["dset"] == "mass"
                  and abs(float(r["R_fit"]) - 4) < 1e-9]
            if rr:
                k = np.array([float(r["k"]) for r in rr])
                c2 = np.array([float(r["chi2nu_floor"]) for r in rr])
                o = np.argsort(k)
                ax.plot(k[o], c2[o], marker="x", color=REF, lw=0.9, ls="--", ms=5,
                        label="Test I A2-ii, 10 bins 13.5-15.5, R$_{fit}$ = 4, mass")
        ax.set_xscale("log")
        ax.set_yscale("log")
        plain_log(ax)
        ax.set_xlabel("number of free parameters k")
        ax.set_ylabel("$\\chi^2/\\nu$ (floor likelihood)")
        ax.set_title(DLAB[D], fontsize=9)
        ax.legend(fontsize=6.5, **LEG_BOX)
    fig.suptitle("Test J2: Lee22 parsimony ladders (best $\\chi^2/\\nu$ at each k; circles: selected by G's rule)",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "J2_ladder_chi2_vs_k")


# ============================================================================ 6. score vs error (supplementary)
def fig_score(rows):
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
    for j, D in enumerate(C.DEFS):
        ax = axes[j]
        for r in rows:
            if r["D"] != D or r["block"] not in ("J2a", "J2b") or f(r.get("rms_DM")) is None:
                continue
            c = CAT[2] if r["block"] == "J2b" else (CAT[0] if r["rule"] == "S400" else CAT[1])
            ax.plot(100 * f(r["S_DM"]), 100 * f(r["rms_DM"]), OBJ_MARK.get(r.get("objective"), "o"), color=c, ms=5,
                    mec="white", mew=0.5)
        ax.set_xscale("log")
        ax.set_yscale("log")
        plain_log(ax)
        ax.set_xlabel("sensitive-part score S$_{DM}$ [%]")
        ax.set_ylabel("DM family RMS error, S1 [%]")
        ax.set_title(DLAB[D], fontsize=9)
    h = [plt.Line2D([], [], ls="none", marker="o", color=CAT[0], label="B16 per bin"),
         plt.Line2D([], [], ls="none", marker="o", color=CAT[1], label="B16, RC fits in 12.7-13.5"),
         plt.Line2D([], [], ls="none", marker="o", color=CAT[2], label="Lee22 ladder law")]
    fig.legend(handles=h, loc="lower center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, -0.04), **LEG_BOX)
    fig.suptitle("Test J2: the sensitive-part score is not an observable error (signed residuals cancel)", fontsize=10)
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    save(fig, "J2_score_vs_error")


# ============================================================================ 7. R_fit dependence (supplementary)
def fig_rfit(rows):
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
    for j, D in enumerate(C.DEFS):
        ax = axes[j]
        for n, obj in enumerate(("a", "d", "L", "dI", "dIT")):
            rr = sorted([r for r in rows if r["D"] == D and r["block"] == "J2a" and r["rule"] == "S400"
                         and r["objective"] == obj and f(r.get("rms_DM")) is not None], key=lambda r: float(r["R_fit"]))
            if not rr:
                continue
            ax.plot([float(r["R_fit"]) for r in rr], [100 * f(r["rms_DM"]) for r in rr], marker=OBJ_MARK[obj],
                    color=CAT[[0, 1, 6, 2, 3][n]], label=f"B16, {OBJ_LAB[obj]}", ms=5)
        ax.axvline(C.DEFS[D]["R"], color=REF, ls=":", lw=0.9)
        ax.set_yscale("log")
        plain_log(ax, x=False)
        ax.set_xlabel("R$_{fit}$ [R$_{200c}$] (R$_{cut}$ = boundary, dotted)")
        ax.set_ylabel("DM family RMS error, S1 [%]")
        ax.set_title(DLAB[D], fontsize=9)
        ax.legend(fontsize=7, loc="best", **LEG_BOX)
    fig.suptitle("Test J2: DM error of the per-bin B16 fits against the fit radius", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "J2_rfit_dependence")


def main():
    rows = scores()
    summ = C.read_json(POST / "J2_summary.json")
    fig_error_vs_k(rows, summ)
    fig_residuals(summ)
    fig_pdfs(summ)
    fig_lowmass()
    fig_ladder()
    fig_score(rows)
    fig_rfit(rows)
    print("figures written to", OUT)


if __name__ == "__main__":
    main()
