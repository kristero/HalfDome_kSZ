"""Figures of Test H (local, from the fetched results/; --smoke reads results_smoke/ and writes figures_smoke/).

Colour conventions (dataviz reference palette): mass bins are ordered, so they use one sequential blue ramp (light =
low mass, dark = high mass; ordinal steps 250-700); the sum over all mass bins is near-black; probes / models are
categorical slots in fixed order (blue, orange, aqua, yellow, magenta, green, violet, red), always with a legend.
x = r / R200c throughout; importance densities are per unit ln x.
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
SEQ = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
ELLEQ = "$\\ell$="
CAL_LABELS = ["13.5-13.7", "13.7-13.9", "13.9-14.1", "14.1-14.3", "14.3-14.5", "14.5-14.7", "14.7-14.9", "14.9-15.1",
              "15.1-15.3", "15.3-15.5"]
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                     "lines.linewidth": 1.6, "savefig.dpi": 150, "figure.dpi": 100})


def load(res):
    def npz(name):
        p = res / name
        return dict(np.load(p, allow_pickle=False)) if p.exists() else None

    def js(name):
        p = res / name
        return json.loads(p.read_text()) if p.exists() else None
    post = HERE / "post" / "h_post.json"
    return {k: npz(k + ".npz") for k in ("sens", "truth", "beyond", "binning", "params")} | \
        {k + "_s": js(k + "_summary.json") for k in ("sens", "truth", "beyond", "binning", "params")} | \
        {"post": json.loads(post.read_text()) if post.exists() else None}


def save(fig, out, name):
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.png", bbox_inches="tight")
    fig.savefig(out / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def mids(e):
    return 0.75 * (e[1:]**4 - e[:-1]**4) / (e[1:]**3 - e[:-1]**3)


def smooth_density(imp_shell, edges, dln, width=0.12):
    """Importance per unit ln x on the fine shells, and a display version averaged over +-width in ln x (the
    X200 grid contains sliver shells, ~0.002-0.03 in ln x, whose per-shell densities are noisy)."""
    dens = imp_shell / dln
    lm = np.log(mids(edges))
    out = np.empty_like(dens)
    for i in range(len(dens)):
        m = np.abs(lm - lm[i]) <= width / 2
        out[i] = np.sum(imp_shell[m]) / np.sum(dln[m])
    return dens, out


PROBES = [("yytot", 1000, "tSZ C$_\\ell^{yy}$, $\\ell$=1000"), ("ksz1h", 1000, "kSZ 1-halo, $\\ell$=1000"),
          ("dm_mean", None, "mean DM"), ("dm_var", None, "DM variance"), ("dm_q50", None, "DM median (Q50)"),
          ("dm_q16", None, "DM Q16"), ("dm_q999", None, "DM Q99.9"), ("ydmtot_E", 1000, "tSZ$\\times$DM, $\\ell$=1000, e$^-$ side")]


def imp_at(S, sel, name, ell, tag="imp"):
    M = S[f"{tag}__{sel}__{name}"]
    if M.ndim == 3:
        i = int(np.flatnonzero(S["ell"] == ell)[0])
        M = M[..., i]
    return M


def fig_importance_mass(D, out, sel="a2ii"):
    S = D["sens"]
    e, dln = S["x_edges"], S["dlnx"]
    cal = S["cal_bins"]
    x = mids(e)
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.2), sharex=True)
    for ax, (name, ell, title) in zip(axes.ravel(), PROBES):
        M = imp_at(S, sel, name, ell)
        for j, b in enumerate(cal):
            _, sm = smooth_density(M[b], e, dln)
            ax.plot(x, sm, color=SEQ[j], lw=1.2, label=str(S["engine_bin_labels"][b]) if ax is axes[0, 0] else None)
        fb = M[[b for b in range(16) if b not in cal]].sum(axis=0)
        _, smf = smooth_density(fb, e, dln)
        ax.plot(x, smf, color=CAT[1], lw=1.2, ls="--", label="fallback bins (published B12/B16)" if ax is axes[0, 0] else None)
        _, smt = smooth_density(M.sum(axis=0), e, dln)
        ax.plot(x, smt, color=INK, lw=2.0, label="all mass bins" if ax is axes[0, 0] else None)
        ax.set_xscale("log")
        ax.set_xlim(0.02, 4)
        ax.axvline(1, color=INK2, lw=0.6, ls=":")
        ax.axvline(2, color=INK2, lw=0.6, ls=":")
        ax.set_title(title, fontsize=9)
        if ax in axes[:, 0]:
            ax.set_ylabel("importance per unit ln x")
        if ax in axes[1]:
            ax.set_xlabel("x = r / R$_{200c}$")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=7, fontsize=8, title="log$_{10}$ M$_{200c}$ [M$_\\odot$] (engine bins)",
               bbox_to_anchor=(0.5, -0.07))
    fig.suptitle(f"Where each probe's signal comes from: importance density by mass bin ({sel.replace('a2ii', 'A2-ii').replace('main', 'all gas')}, "
                 "67 X200 shells, display-smoothed over 0.12 in ln x)", fontsize=10)
    fig.tight_layout()
    save(fig, out, f"H3_importance_by_mass_{sel}")


def fig_importance_probes(D, out):
    S = D["sens"]
    e, dln = S["x_edges"], S["dlnx"]
    x = mids(e)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=False)
    sets = [[("yytot", 100, "tSZ yy, $\\ell$=100"), ("yytot", 1000, "tSZ yy, $\\ell$=1000"), ("yytot", 3000, "tSZ yy, $\\ell$=3000"),
             ("ksz1h", 100, "kSZ 1h, $\\ell$=100"), ("ksz1h", 1000, "kSZ 1h, $\\ell$=1000"), ("ksz1h", 3000, "kSZ 1h, $\\ell$=3000")],
            [("dm_mean", None, "mean DM"), ("dm_var", None, "DM variance"), ("dm_q16", None, "DM Q16"),
             ("dm_q50", None, "DM Q50"), ("dm_q84", None, "DM Q84"), ("dm_q999", None, "DM Q99.9")]]
    cols = [CAT[0], CAT[1], CAT[6], CAT[3], CAT[4], CAT[5]]          # blue, orange, violet, yellow, magenta, green
    marks = ["o", "s", "^", "D", "v", "P"]
    for ax, lst in zip(axes, sets):
        for j, (name, ell, lab) in enumerate(lst):
            for sel, ls in (("a2ii", "-"), ("main", ":")):
                M = imp_at(S, sel, name, ell).sum(axis=0)
                _, sm = smooth_density(M, e, dln)
                ax.plot(x, sm, color=cols[j], ls=ls, lw=1.6 if sel == "a2ii" else 1.1,
                        marker=marks[j] if sel == "a2ii" else None, markevery=6, ms=4,
                        label=lab if sel == "a2ii" else None)
        ax.set_xscale("log")
        ax.set_xlim(0.02, 4)
        ax.set_xlabel("x = r / R$_{200c}$")
        ax.set_ylabel("importance per unit ln x (all mass bins)")
        ax.axvline(1, color=INK2, lw=0.6, ls=":")
        ax.legend(fontsize=8, loc="upper left")
    axes[0].set_title("tSZ and kSZ (solid A2-ii, dotted all gas)", fontsize=9)
    axes[1].set_title("FRB DM (solid A2-ii, dotted all gas)", fontsize=9)
    fig.tight_layout()
    save(fig, out, "H3_importance_probes")


def fig_cumulative(D, out, sel="a2ii"):
    S = D["sens"]
    e = S["x_edges"]
    ell = S["ell"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    lst = [("yytot", 100, "tSZ yy $\\ell$=100"), ("yytot", 1000, "tSZ yy $\\ell$=1000"), ("yytot", 3000, "tSZ yy $\\ell$=3000"),
           ("ksz1h", 100, "kSZ 1h $\\ell$=100"), ("ksz1h", 1000, "kSZ 1h $\\ell$=1000"), ("ksz1h", 3000, "kSZ 1h $\\ell$=3000"),
           ("ydmtot", 1000, "tSZ$\\times$DM $\\ell$=1000")]
    ax = axes[0]
    cols1 = [CAT[0], CAT[1], CAT[6], CAT[0], CAT[1], CAT[6], CAT[4]]
    ls1 = ["-", "-", "-", "--", "--", "--", "-."]
    for j, (name, at, lab) in enumerate(lst):
        c = S[f"trunc_all__{sel}__{name}"]
        i = int(np.flatnonzero(ell == at)[0])
        c = c[:, i]
        C = (c - c[0]) / (c[-1] - c[0])
        ax.plot(e, C, color=cols1[j], ls=ls1[j], label=lab)
        for X, mk in ((0.5, "o"), (0.9, "s"), (0.99, "^")):
            k = np.flatnonzero(C >= X)
            if len(k):
                ax.plot(e[k[0]], C[k[0]], mk, color=cols1[j], ms=5, mec="white", mew=0.6)
    ax2 = axes[1]
    dm = [("dm_mean", "mean DM"), ("dm_var", "DM variance")]
    for j, (name, lab) in enumerate(dm):
        c = S[f"trunc_all__{sel}__{name}"]
        C = (c - c[0]) / (c[-1] - c[0])
        ax2.plot(e, C, color=CAT[j], label=lab)
        for X, mk in ((0.5, "o"), (0.9, "s"), (0.99, "^")):
            k = np.flatnonzero(C >= X)
            if len(k):
                ax2.plot(e[k[0]], C[k[0]], mk, color=CAT[j], ms=5, mec="white", mew=0.6)
    q = S[f"trunc_all__{sel}__dmq_mean"]
    colq = [CAT[6], CAT[3], CAT[4], CAT[5], CAT[7]]                  # violet, yellow, magenta, green, red (dashed)
    for p, (qn, lab) in enumerate((("q16", "DM Q16"), ("q50", "DM Q50"), ("q84", "DM Q84"), ("q975", "DM Q97.5"), ("q999", "DM Q99.9"))):
        C = (q[:, p] - q[0, p]) / (q[-1, p] - q[0, p]) if q[-1, p] != q[0, p] else np.full(len(e), np.nan)
        ax2.plot(e, C, color=colq[p], ls="--", label=lab + " (truth mean)")
        for X, mk in ((0.5, "o"), (0.9, "s"), (0.99, "^")):
            k = np.flatnonzero(C >= X)
            if len(k):
                ax2.plot(e[k[0]], C[k[0]], mk, color=colq[p], ms=5, mec="white", mew=0.6)
    for a, t in ((ax, "tSZ / kSZ / cross: calibrated-signal fraction vs truncation radius"),
                 (ax2, "FRB DM: calibrated-signal fraction vs truncation radius")):
        a.set_xscale("log")
        a.set_xlim(0.05, 4)
        a.set_ylim(-0.05, 1.08)
        a.axhline(1, color=INK2, lw=0.6)
        a.set_xlabel("truncation radius R$_t$ / R$_{200c}$")
        a.set_ylabel("C(R$_t$) = [O(R$_t$) - O(0)] / [O(4R) - O(0)]")
        a.set_title(t, fontsize=9)
        a.legend(fontsize=7.5, loc="upper left")
    fig.text(0.5, -0.02, "markers: FIRST fine edge reaching 50% (circle), 90% (square), 99% (triangle) of the calibrated "
             "haloes' signal (first crossings; where C overshoots 1 the converged radii R$_X^{conv}$ in the tables are "
             "larger)\nsolid: tSZ / DM moments, dashed: kSZ / DM quantiles (truth-mean DM-PDF passes), dash-dot: "
             f"tSZ$\\times$DM; {'A2-ii' if sel == 'a2ii' else 'all gas'}", ha="center", fontsize=8, color=INK2)
    fig.tight_layout()
    save(fig, out, f"H3_cumulative_truncation_{sel}")


def fig_cumulative_by_mass(D, out, sel="a2ii"):
    """Per calibrated bin: fraction of the bin's own 1-halo (quadratic) or linear signal kept by a cut at R_t."""
    S = D["sens"]
    e = S["x_edges"]
    ell = S["ell"]
    lst = [("yy1h", 1000, "tSZ 1-halo, $\\ell$=1000"), ("yy1h", 100, "tSZ 1-halo, $\\ell$=100"),
           ("ksz1h", 1000, "kSZ 1-halo, $\\ell$=1000"), ("ksz1h", 100, "kSZ 1-halo, $\\ell$=100"),
           ("dm_mean", None, "mean DM"), ("dm_var", None, "DM variance")]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True, sharey=True)
    for ax, (name, at, title) in zip(axes.ravel(), lst):
        own = S[f"trunc_own__{sel}__{name}"]
        for c in range(own.shape[0]):
            v = own[c] if at is None else own[c, :, int(np.flatnonzero(ell == at)[0])]
            C = (v - v[0]) / (v[-1] - v[0])
            lab = str(S["engine_bin_labels"][S["cal_bins"][c]])
            ax.plot(e, C, color=SEQ[c], lw=1.3, label=lab if ax is axes[0, 0] else None)
        ax.axhline(1, color=INK2, lw=0.6)
        for lev in (0.9, 0.99):
            ax.axhline(lev, color=INK2, lw=0.6, ls=":")
        ax.set_xscale("log")
        ax.set_xlim(0.05, 4)
        ax.set_ylim(-0.02, 1.1)
        ax.set_title(title, fontsize=9)
        if ax in axes[1]:
            ax.set_xlabel("truncation radius R$_t$ / R$_{200c}$")
        if ax in axes[:, 0]:
            ax.set_ylabel("fraction of the bin's own signal kept")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=10, fontsize=8, title="calibrated bin, log$_{10}$ M$_{200c}$ [M$_\\odot$]",
               bbox_to_anchor=(0.5, -0.06))
    fig.suptitle(f"Cumulative signal of each mass bin vs truncation radius ({'A2-ii' if sel == 'a2ii' else 'all gas'}; "
                 "dotted lines 90% and 99%; light = low mass, dark = high mass)", fontsize=10)
    fig.tight_layout()
    save(fig, out, f"H3_cumulative_by_mass_{sel}")


def fig_radii_vs_mass(D, out, sel="a2ii"):
    """R50 / R90 / R99 of each bin's own 1-halo / linear contribution vs mass."""
    import csv
    rows = list(csv.DictReader(open(D["res"] / "tables" / "H3_radii.csv")))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    sets = [("yy1h", ("100", "1000", "3000"), "tSZ 1-halo"), ("ksz1h", ("100", "1000", "3000"), "kSZ 1-halo"),
            ("dm_mean", ("scalar",), "mean DM / DM variance")]
    for ax, (name, ells, title) in zip(axes, sets):
        names = [name] if name != "dm_mean" else ["dm_mean", "dm_var"]
        j = 0
        for nm in names:
            for at in ells:
                rr = [r for r in rows if r["selection"] == sel and r["observable"] == nm and r["ell"] == at
                      and r["scope"] == "own bin"]
                if not rr:
                    continue
                m = [float(r["mass_bin"].split("-")[0]) + 0.1 for r in rr]
                for key, ls, mk in (("R50", ":", "o"), ("R90", "-", "s"), ("R99", "--", "^")):
                    v = [float(r[key]) if r[key] else np.nan for r in rr]
                    ax.plot(m, v, ls=ls, marker=mk, ms=4, color=CAT[j],
                            label=f"{nm} {'' if at == 'scalar' else ELLEQ + at} {key}")
                j += 1
        ax.set_yscale("log")
        ax.set_xlabel("log$_{10}$ M$_{200c}$ [M$_\\odot$] (bin centre)")
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=6.5, ncol=1, loc="lower left")
    axes[0].set_ylabel("radius containing X% of the bin's own signal [R$_{200c}$]")
    fig.suptitle("Radii containing 50 / 90 / 99% of each mass bin's own 1-halo (or linear) signal, A2-ii", fontsize=10)
    fig.tight_layout()
    save(fig, out, f"H3_radii_vs_mass_{sel}")


def fig_shares(D, out, sel="a2ii"):
    """Per-mass-bin shares (importance summed over radius).  Engine bin 15.3-15.5 is split into its lightcone haloes
    10^15.3-10^15.388 painted with the FLAMINGO stack (built from 6 FLAMINGO haloes) and the lightcone haloes above
    10^15.388 painted with the published fallback, using the exact decomposition of post/h_post.json (h_post.py)."""
    S, s, post = D["sens"], D["sens_s"], D["post"]
    labs = list(S["engine_bin_labels"])
    obs = [("yytot@100", "tSZ C$_\\ell^{yy}$, $\\ell$=100"), ("yytot@1000", "tSZ C$_\\ell^{yy}$, $\\ell$=1000"),
           ("yytot@3000", "tSZ C$_\\ell^{yy}$, $\\ell$=3000"), ("ksz1h@100", "kSZ 1-halo, $\\ell$=100"),
           ("ksz1h@1000", "kSZ 1-halo, $\\ell$=1000"), ("ksz1h@3000", "kSZ 1-halo, $\\ell$=3000"),
           ("dm_mean@scalar", "mean DM"), ("dm_var@scalar", "DM variance"), ("dm_q16@scalar", "DM Q16"),
           ("dm_q50@scalar", "DM Q50"), ("dm_q999@scalar", "DM Q99.9"),
           ("ydmtot_E@1000", "tSZ$\\times$DM, $\\ell$=1000, e$^-$ side")]
    top = labs.index("15.3-15.5")
    cols = labs[:top] + ["15.3-15.388\ncalibrated", ">15.388\nfallback"] + labs[top + 1:]
    M = np.zeros((len(obs), len(cols)))
    for i, (k, _) in enumerate(obs):
        row = [s["shares"][f"{sel}__{k}"][b] for b in labs]
        fb = post["calibrated_shares"][k]["fallback_inside_calibrated_bins"] if (post and sel == "a2ii") else 0.0
        M[i] = row[:top] + [row[top] - fb, fb] + row[top + 1:]
    fig, ax = plt.subplots(figsize=(12, 5.2))
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("blue", ["#f7f9fc", "#b7d3f6", "#5598e7", "#1c5cab", "#0d366b"])
    im = ax.imshow(np.clip(M, 0, None), cmap=cmap, vmin=0, vmax=max(0.3, float(np.nanmax(M))), aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if abs(M[i, j]) >= 0.005:
                ax.text(j, i, f"{100 * M[i, j]:.0f}", ha="center", va="center", fontsize=7,
                        color="white" if M[i, j] > 0.18 else INK)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=60, fontsize=7.5)
    for j in (len(cols) - 2, len(cols) - 1, *range(5)):
        ax.get_xticklabels()[j].set_color(CAT[1])
    ax.set_yticks(range(len(obs)))
    ax.set_yticklabels([o[1] for o in obs], fontsize=8)
    ax.grid(False)
    for v in (4.5, len(cols) - 2.5):
        ax.axvline(v, color=CAT[1], lw=1.4)
    cb = fig.colorbar(im, ax=ax, fraction=0.025)
    cb.set_label("share of the observable (fraction; cell numbers in %)")
    ax.set_title(f"Per-mass-bin shares of each observable (importance summed over radius), "
                 f"{'A2-ii' if sel == 'a2ii' else 'all gas'}.  Orange labels / outside the orange lines: haloes painted "
                 "with the published\nB12/B16 fallback (below 10$^{13.5}$, above 10$^{15.388}$ inside engine bin "
                 "15.3-15.5, and $\\geq$10$^{15.5}$); between the lines: lightcone haloes painted with the FLAMINGO stacks", fontsize=9)
    ax.set_xlabel("engine mass bin, log$_{10}$ M$_{200c}$ [M$_\\odot$]")
    fig.tight_layout()
    save(fig, out, f"H3_mass_shares_{sel}")


def fig_tolerance(D, out, sel="a2ii"):
    """Profile tolerance per 0.1 dex from the calibrated engine bins 13.5-15.5 (h_sens definition: i(x) sums the
    importance over the calibrated engine bins; bin 15.3-15.5 also contains the fallback haloes above 10^15.388,
    which change these curves by <= 1% at their minima, post/h_post.json:tolerance)."""
    S = D["sens"]
    e, dln = S["x_edges"], S["dlnx"]
    x = mids(e)
    cal = S["cal_bins"]
    lst = [("yytot", 100), ("yytot", 1000), ("yytot", 3000), ("ksz1h", 1000), ("dm_mean", None), ("dm_var", None),
           ("dm_q50", None), ("dm_q999", None)]
    labels = ["tSZ C$_\\ell^{yy}$, $\\ell$=100", "tSZ C$_\\ell^{yy}$, $\\ell$=1000",
              "tSZ C$_\\ell^{yy}$, $\\ell$=3000", "kSZ 1-halo, $\\ell$=1000", "mean DM", "DM variance",
              "DM median (Q50)", "DM Q99.9"]
    style = [(CAT[0], ":"), (CAT[0], "-"), (CAT[0], "--"), (CAT[1], "-"), (CAT[5], "-"), (CAT[4], "-"),
             (CAT[5], "--"), (CAT[6], "-.")]
    k = {"yytot": 2, "ksz1h": 2, "dm_var": 2}
    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    for j, ((name, ell), lab) in enumerate(zip(lst, labels)):
        M = imp_at(S, sel, name, ell)[cal].sum(axis=0)
        _, sm = smooth_density(M, e, dln)
        with np.errstate(divide="ignore"):
            tol = 0.01 / np.abs(k.get(name, 1) * sm * 0.1 * np.log(10))
        col, ls = style[j]
        ax.plot(x, 100 * tol, color=col, ls=ls, lw=1.8 if j == 1 else 1.5, label=lab)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.02, 4)
    ax.set_ylim(1, 1e4)
    ax.set_xlabel("x = r / R$_{200c}$")
    ax.set_ylabel("coherent stack error over 0.1 dex that\nchanges the observable by 1% [%]")
    ax.legend(fontsize=8, ncol=1, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.set_title(f"Profile tolerance, all calibrated engine bins (13.5-15.5) changed together, "
                 f"{'A2-ii' if sel == 'a2ii' else 'all gas'}\n(display-smoothed over 0.12 in ln x; lower = more "
                 "sensitive)", fontsize=9)
    fig.tight_layout()
    save(fig, out, f"H3_tolerance_{sel}")


def fig_beyond(D, out):
    B = D["beyond"]
    if B is None:
        return
    r = np.r_[4.0, B["r_ext"]]
    fig, axes = plt.subplots(1, 4, figsize=(17, 4), gridspec_kw={"width_ratios": [1, 1, 1, 1.05]})
    groups = [("tSZ", [("yytot", 100), ("yytot", 1000), ("ymean", None)]),
              ("kSZ", [("ksz1h", 100), ("ksz1h", 1000), ("ksz2h", 100)]),
              ("DM", [("dm_mean", None), ("dm_var", None), ("dmcltot", 100)])]
    models = [("ii_B12B16_fit4R", "-", "B12/B16 fit to 4R, extrapolated"), ("iiia_loglin_3to4R", "--", "power law fitted at 3-4R"),
              ("iiib_loglin_2.5to4R", ":", "power law fitted at 2.5-4R")]
    ell = D["sens"]["ell"]
    nice = {"yytot": "tSZ C$_\\ell^{yy}$", "ymean": "mean y", "ksz1h": "kSZ 1-halo", "ksz2h": "kSZ 2-halo",
            "dm_mean": "mean DM", "dm_var": "DM variance", "dmcltot": "DM-field C$_\\ell$"}
    for ax, (title, lst) in zip(axes[:3], groups):
        for j, (key, at) in enumerate(lst):
            for lab, ls, mlab in models:
                v = [0.0]
                for rr in B["r_ext"]:
                    a = B[f"chg__a2ii__{lab}__{key}__R{rr:g}"]
                    v.append(float(a if np.ndim(a) == 0 else a[int(np.flatnonzero(ell == at)[0])]))
                ax.plot(r, 100 * np.array(v), color=CAT[j], ls=ls,
                        label=f"{nice[key]}{'' if at is None else ', ' + ELLEQ + str(at)}" if ls == "-" else None)
        ax.set_xlabel("painting extended to R$_{ext}$ / R$_{200c}$")
        ax.set_title(f"{title}: change vs the 4R-truncated truth (A2-ii)", fontsize=9)
        ax.legend(fontsize=7.5, loc="upper left")
    axes[0].set_ylabel("$\\Delta$O / O [%]  (model-dependent)")
    dc = (D["beyond_s"] or {}).get("double_counting", {})
    ax = axes[3]
    if dc:
        labs = CAL_LABELS
        m = np.arange(len(labs))
        for j, (fld, nm) in enumerate((("P", "pressure"), ("E", "electrons"))):
            for k, (band, ls) in enumerate((("2-3R", "-"), ("3-4R", "--"), ("3.6342-4R", ":"))):
                v = [dc[f"{fld}_{band}"][b] for b in labs]
                ax.plot(m, v, color=CAT[j], ls=ls, marker="o", ms=3,
                        label=f"{nm}, {band.replace('3.6342', '3.63')}")
        ax.set_xticks(m)
        ax.set_xticklabels(labs, rotation=60, fontsize=7)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("calibrated bin, log$_{10}$ M$_{200c}$ [M$_\\odot$]")
        ax.set_ylabel("A2-ii / all gas in the shell")
        ax.set_title("Double counting: A2-ii's share of all gas, volume-weighted over\nthe shell range (the rest is "
                     "nearer another painted halo; measured to 4R only)", fontsize=9)
        ax.legend(fontsize=7, loc="lower left", ncol=2)
    from matplotlib.lines import Line2D
    fig.legend([Line2D([], [], color=INK, ls=ls) for _, ls, _ in models], [m for _, _, m in models],
               loc="lower center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    save(fig, out, "H4_beyond_4R")


def fig_binning(D, out):
    """H5: (a) binning alone at R_cut = 4 (det painting vs the truth mean; the reference scheme is exact);
    (b) the same painted with S1 scatter vs the scatter-inclusive truth (what a painting would actually give);
    (c-f) binning-only error at every truncation radius (tables/H_report_binning_only.csv: scheme minus the
    reference at the same cut, so the truncation loss common to all schemes cancels)."""
    s = D["binning_s"]
    if s is None:
        return
    import csv
    schemes = ["M200c_0.1", "M200c_0.2", "M200c_0.3", "M200c_0.5", "M500c_0.2", "M500c_0.5"]
    fams = ["tSZ", "kSZ", "DM", "tSZxDM"]
    fig = plt.figure(figsize=(13, 7.6))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 0.95], hspace=0.55, wspace=0.32)
    w = 0.18
    for col, (scat, ref, title) in enumerate((
            ("det", "family_rms_vs_truth_mean", "(a) binning alone: no scatter, R$_{cut}$ = 4,\nvs the truth mean "
                                                "(M200c_0.2 piecewise = engine bins: exact)"),
            ("S1", "family_rms_vs_truth", "(b) TOTAL error with S1 scatter (binning + S1 model), R$_{cut}$ = 4,\nvs "
                                          "the scatter-inclusive truth (binning-only S1 error: panels c-f)"))):
        ax = fig.add_subplot(gs[0, 2 * col:2 * col + 2])
        for j, fam in enumerate(fams):
            for m, (mode, mk) in enumerate((("piecewise", "o"), ("interp", "D"))):
                v = [100 * s["errors"][f"{sc}|{mode}|4|{scat}"][ref][fam] for sc in schemes]
                ax.plot(np.arange(len(schemes)) + (j - 1.5) * w + (0.04 if m else -0.04), v, mk, color=CAT[j],
                        ms=6 if m == 0 else 5, mfc=CAT[j] if m == 0 else "white", label=f"{fam} ({mode})")
        ax.set_xticks(range(len(schemes)))
        ax.set_xticklabels(schemes, fontsize=8)
        ax.set_ylim(0, 1.4 * max(ln.get_ydata().max() for ln in ax.get_lines()))
        ax.set_ylabel("family RMS error [%]")
        ax.set_title(title, fontsize=9)
        if col == 0:
            ax.legend(fontsize=7, ncol=4, loc="upper left")
    p = D["res"] / "tables" / "H_report_binning_only.csv"
    rows = list(csv.DictReader(open(p))) if p.exists() else []
    labels = {"tSZ": "tSZ (C$_\\ell^{yy}$)", "kSZ": "kSZ 1h", "DM_moments": "DM mean + variance",
              "tSZxDM": "tSZ$\\times$DM"}
    for k, fam in enumerate(("tSZ", "kSZ", "DM_moments", "tSZxDM")):
        ax = fig.add_subplot(gs[1, k])
        for j, sc in enumerate(schemes):
            rr = sorted((float(r["R_cut"]), float(r[f"{fam}_rms_pct"])) for r in rows
                        if r["scheme"] == sc and r["mode"] == "piecewise" and r["scatter"] == "S1")
            if rr:
                ax.plot([a for a, _ in rr], [b for _, b in rr], marker="o", ms=3.5, color=CAT[j], label=sc)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("R$_{cut}$ / R$_{200c}$")
        ax.set_title(f"({'cdef'[k]}) {labels[fam]}, S1 painting", fontsize=9)
        if k == 0:
            ax.set_ylabel("binning-only RMS error [%]\n(S1 scheme - S1 M200c_0.2, same cut)")
    h, l = fig.axes[-1].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=6, fontsize=8, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Mass-binning propagation (A2-ii stacks painted per halo; families: tSZ / kSZ / tSZxDM at "
                 "$\\ell$ = 100-3000; DM in (a, b) = mean, variance and Q16-Q99.9)", fontsize=10)
    fig.text(0.5, 0.005, "(c-f): S1 painting minus the S1 M200c_0.2 painting at the same R$_{cut}$ (DM: mean + variance "
             "only), so the S1 model error and the common truncation loss cancel; at R$_{cut}$ = 4 they therefore differ "
             "from (a), which is the det painting vs the truth mean.", ha="center", fontsize=8, color=INK2)
    save(fig, out, "H5_binning")


def fig_params(D, out):
    P, s = D["params"], D["params_s"]
    if P is None:
        return
    e = P["x_edges"]
    x = mids(e)
    dln = np.diff(np.log(np.where(e > 0, e, 0.716 * e[1])))
    cases = [("P", "yytot@1000", "tSZ C$_\\ell^{yy}$, $\\ell$=1000 (pressure)"), ("E", "ksz1h@1000", "kSZ 1-halo, $\\ell$=1000 (electrons)"),
             ("E", "dm_mean", "mean DM (electrons)"), ("E", "dm_q50", "DM Q50 (electrons)")]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8))
    xe = e[1:]                                                   # outer edge of each fine shell
    for ax, (field, lab, title) in zip(axes, cases):
        Dk = P[f"direct__{field}__{lab}"].sum(axis=0)
        ax.plot(xe, np.cumsum(Dk), color=INK, lw=2.0, label="direct kernel (truth)")
        for j, r in enumerate(("4", "2")):
            T = P[f"transfer__Rf{r}__{field}__{lab}"].sum(axis=0)
            ax.plot(xe, np.cumsum(T), color=CAT[j], label=f"through the fit, R$_{{fit}}$ = {r}")
        ax.set_xscale("log")
        ax.set_xlim(0.02, 4)
        ax.axhline(0, color=INK2, lw=0.6)
        for v in (1.0, 2.0):
            ax.axvline(v, color=INK2, lw=0.6, ls=":")
        ax.set_xlabel("x = r / R$_{200c}$")
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=7.5, loc="upper left")
    axes[0].set_ylabel("cumulative kernel inside x\n(d ln O / k$_O$ for a coherent d ln F inside x)")
    fig.suptitle("How a change of the stacked profile reaches the observable: directly vs through a B12/B16 fit "
                 "(A2-ii, objective a, cut 4R; calibrated bins summed; per-shell kernels in results/params.npz)",
                 fontsize=10)
    fig.tight_layout()
    save(fig, out, "H6_fit_transfer")
    # parameter sensitivities: rows = field, columns = parameter, lines = observables, x = calibrated mass bin
    rowsdef = [("P", ("P0", "x$_c$", "$\\beta$"), [("yytot@100", "tSZ C$_\\ell^{yy}$, $\\ell$=100"), ("yytot@1000", "tSZ C$_\\ell^{yy}$, $\\ell$=1000"),
                                                  ("yytot@3000", "tSZ C$_\\ell^{yy}$, $\\ell$=3000"), ("ydmtot_P@1000", "tSZ$\\times$DM, $\\ell$=1000 (P side)")]),
               ("E", ("$\\rho_0$", "$\\alpha$", "$\\beta$"), [("ksz1h@1000", "kSZ 1-halo, $\\ell$=1000"), ("dm_mean", "mean DM"),
                                                          ("dm_var", "DM variance"), ("dm_q50", "DM Q50"),
                                                          ("dm_q999", "DM Q99.9"), ("ydmtot_E@1000", "tSZ$\\times$DM, $\\ell$=1000 (e side)")])]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.2), sharex=True)
    for row, (field, names, lst) in enumerate(rowsdef):
        for k in range(3):
            ax = axes[row, k]
            for j, (lab, t) in enumerate(lst):
                key = f"g__Rf4__{field}__{lab}"
                if key not in P:
                    continue
                g = P[key]
                m = np.arange(g.shape[0])
                ax.plot(m, g[:, k], marker="o", ms=4, color=CAT[j], label=t)
                key2 = f"g__Rf2__{field}__{lab}"
                if key2 in P:
                    ax.plot(m, P[key2][:, k], ls=":", lw=1.1, color=CAT[j])
            ax.axhline(0, color=INK2, lw=0.6)
            ax.set_title(f"{'B12 pressure' if field == 'P' else 'B16 electrons'}: d ln O / d ln {names[k]}", fontsize=9)
            if k == 0:
                ax.set_ylabel("d ln O / d ln $\\theta$ (one bin's parameter)")
            if row == 1:
                ax.set_xticks(range(len(CAL_LABELS)))
                ax.set_xticklabels(CAL_LABELS, rotation=60, fontsize=7.5)
                ax.set_xlabel("calibrated bin, log$_{10}$ M$_{200c}$ [M$_\\odot$]")
        h, l = axes[row, 0].get_legend_handles_labels()
        axes[row, 2].legend(h, l, fontsize=7.5, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.suptitle("Sensitivity of each observable to one mass bin's B12 / B16 parameter (A2-ii fit, objective a; "
                 "solid R$_{fit}$ = 4, dotted R$_{fit}$ = 2; painted to 4R with S1 scatter)", fontsize=10)
    fig.tight_layout()
    save(fig, out, "H6_parameter_sensitivity")


def fig_contribution(D, out, sel="a2ii"):
    """1-halo contribution split: the shell's own (diagonal) term K_ss <F_s^2> / O vs the full contribution
    sum_t K_st <F_s F_t> / O (= the importance of the calibrated bins, Euler); the difference is the cross-shell part."""
    S = D["sens"]
    e, dln = S["x_edges"], S["dlnx"]
    x = mids(e)
    cal = S["cal_bins"]
    lst = [("yy1h", 1000, "tSZ 1-halo, $\\ell$=1000"), ("ksz1h", 1000, "kSZ 1-halo, $\\ell$=1000"),
           ("dmcl1h", 1000, "DM-field 1-halo, $\\ell$=1000"), ("dm_var", None, "DM variance")]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
    for ax, (name, ell, title) in zip(axes, lst):
        M = imp_at(S, sel, name, ell)[cal].sum(axis=0)
        Sd = S[f"self__{sel}__{name}"]
        if Sd.ndim == 3:
            Sd = Sd[..., int(np.flatnonzero(S["ell"] == ell)[0])]
        Sd = Sd.sum(axis=0)
        _, sm = smooth_density(M, e, dln)
        _, sd = smooth_density(Sd, e, dln)
        ax.plot(x, sm, color=INK, lw=2.0, label="importance of the calibrated engine bins (= full contribution of "
                                                 "the calibrated haloes, + the fallback haloes > 10$^{15.388}$ in "
                                                 "15.3-15.5)")
        ax.plot(x, sd, color=CAT[0], lw=1.4, label="own-shell (diagonal) part")
        ax.plot(x, sm - sd, color=CAT[1], lw=1.2, ls="--", label="cross-shell part")
        ax.set_xscale("log")
        ax.set_xlim(0.02, 4)
        ax.axhline(0, color=INK2, lw=0.6)
        ax.set_xlabel("x = r / R$_{200c}$")
        ax.set_title(title, fontsize=9)
    axes[0].set_ylabel("per unit ln x (calibrated bins, normalised by O)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle(f"Signal decomposition of the quadratic 1-halo terms ({'A2-ii' if sel == 'a2ii' else 'all gas'}): the "
                 "own-shell part scales with the shell width (it vanishes as the grid is refined),\nso a shell's "
                 "contribution is its full row sum over partner shells, which equals its importance", fontsize=10)
    fig.tight_layout()
    save(fig, out, f"H3_contribution_split_{sel}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    res = HERE / ("results_smoke" if a.smoke else "results")
    out = HERE / ("figures_smoke" if a.smoke else "figures")
    D = load(res)
    D["res"] = res
    for sel in ("a2ii", "main"):
        fig_importance_mass(D, out, sel)
    fig_importance_probes(D, out)
    fig_cumulative(D, out, "a2ii")
    fig_cumulative(D, out, "main")
    fig_cumulative_by_mass(D, out, "a2ii")
    fig_radii_vs_mass(D, out)
    fig_shares(D, out, "a2ii")
    fig_tolerance(D, out)
    fig_contribution(D, out, "a2ii")
    fig_beyond(D, out)
    fig_binning(D, out)
    fig_params(D, out)
    print("figures written to", out)


if __name__ == "__main__":
    main()
