"""L1 figures (local, from results/ only): PNG and SVG in figures/ (figures_smoke/ in smoke mode).

Colour: categorical series use the validated reference palette in its fixed slot order (blue, orange, aqua, yellow,
magenta, green; adjacent-pair CVD dE >= 9.1, normal-vision dE >= 19.6 on the light surface), always with a distinct
marker and line style as secondary encoding; mass is a one-hue sequential blue ramp (light = low mass).  Every value
plotted is also in results/tables/*.csv.
"""
from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")
import l_common as LC  # noqa: E402
import matplotlib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize  # noqa: E402

SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
MARK = ["o", "s", "^", "D", "v", "P"]
LS_ = ["-", "--", "-.", ":", (0, (5, 1.5, 1, 1.5)), (0, (3, 1))]
BLUE = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
MASS_CMAP = LinearSegmentedColormap.from_list("mass_blue", BLUE)
matplotlib.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "font.size": 9.5,
    "axes.titlesize": 10, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7, "grid.linestyle": "-",
    "legend.frameon": False, "legend.fontsize": 8.2, "lines.linewidth": 1.5, "lines.solid_capstyle": "round",
    "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none"})
FIELD_NAME = {"P": "electron pressure (tSZ)", "E": "free electrons (kSZ, DM)"}


def save(fig, name):
    for ext in ("png", "svg"):
        fig.savefig(LC.FIGURES / f"{name}.{ext}", dpi=170 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)


def mass_colour(logm):
    return MASS_CMAP(Normalize(12.7, 15.4)(logm))


def mass_colourbar(fig, axes, label="mean log10(M200c / Msun) of the bin"):
    sm = plt.cm.ScalarMappable(cmap=MASS_CMAP, norm=Normalize(12.7, 15.4))
    cb = fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.02)
    cb.set_label(label, color=INK2)
    cb.outline.set_visible(False)


def series(ax, x, y, i, label, lo=None, hi=None, lw=1.5, ms=5.5):
    ax.plot(x, y, color=CAT[i], ls=LS_[i], marker=MARK[i], ms=ms, mec=SURF, mew=1.0, lw=lw, label=label, zorder=3)
    if lo is not None:
        ax.fill_between(x, lo, hi, color=CAT[i], alpha=0.12, lw=0, zorder=1)


def main():
    rad = LC.jload(LC.RESULTS / "radii.json")
    st = np.load(LC.RESULTS / "stacks_L.npz")
    cur = np.load(LC.RESULTS / "curves.npz")
    glob = LC.jload(LC.RESULTS / "global.json")
    bins = [int(b) for b in st["bins"]]
    labs = [LC.LABELS[b] for b in bins]
    logm = np.array([rad["bins"][lab]["mean_logm200c"] for lab in labs])

    def R(rule, field, unit="value"):
        out = np.full((3, len(labs)), np.nan)
        for i, lab in enumerate(labs):
            s = rad["bins"][lab]["rules"][rule][field]
            if unit == "value":
                v, lo, hi = s["value"], s["q16"], s["q84"]
            else:
                v, lo, hi = s[unit]["value"], s[unit]["q16"], s[unit]["q84"]
            out[:, i] = [np.nan if v is None else v, lo, hi]
        return out

    # ------------------------------------------------------------------ 1. bound-fraction profiles
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), sharex=True, sharey=True)
    for c, f in enumerate(("P", "E")):
        for r, (fname, title) in enumerate((("hb/all", "host-bound / all gas"), ("cb/all", "central-bound / all gas"))):
            ax = axes[r, c]
            for b, lm in zip(bins, logm):
                x = st[f"fracx_{f}_{fname}_{b}"]
                y = st[f"frac_{f}_{fname}_{b}"]
                col = mass_colour(lm)
                ax.plot(x, y, color=col, lw=1.3, zorder=2)
                if b in (3, 7, 10, 13):
                    ax.fill_between(x, st[f"fracq16_{f}_{fname}_{b}"], st[f"fracq84_{f}_{fname}_{b}"], color=col,
                                    alpha=0.18, lw=0, zorder=1)
            for thr, ls, txt in ((0.32, "--", "0.32 (B32)"), (0.5, ":", "0.50 (B50)")):
                ax.axhline(thr, color=INK2, ls=ls, lw=1.0, zorder=1)
                ax.text(3.9, thr + 0.015, txt, ha="right", va="bottom", color=INK2, fontsize=8)
            ax.set_xscale("log")
            ax.set_xlim(0.3, 4)
            ax.set_ylim(0, 1.02)
            ax.set_title(f"{FIELD_NAME[f]}: {title}")
            if r == 1:
                ax.set_xlabel("r / R200c (stacked shells, merged shells in 12.7-13.5)")
            if c == 0:
                ax.set_ylabel("bound fraction of the stacked shell")
            ax.set_xticks([0.3, 0.5, 1, 1.5, 2, 3, 4])
            ax.set_xticklabels(["0.3", "0.5", "1", "1.5", "2", "3", "4"])
    mass_colourbar(fig, axes)
    fig.suptitle("FLAMINGO L1_m9, z = 0: bound share of the gas per spherical shell (bands: bootstrap 16-84% for four bins)",
                 fontsize=10.5)
    save(fig, "L1_bound_fraction_profiles")

    # ------------------------------------------------------------------ 2./3. radii vs mass
    d3 = glob["d3_convergence"]["own_bin"].get("dm_mean@scalar", {})
    sets = {"L1_radii_bound_rules": [("B32", "B32: host-bound/all = 0.32 (primary)"),
                                     ("B32cb", "B32cb: central-bound/all = 0.32"),
                                     ("B50", "B50: host-bound/all = 0.50"),
                                     ("C68", "C68: 68% of the bound gas inside R")],
            "L1_radii_alternatives": [("B32", "B32 (primary, reference)"),
                                      ("Req", "Req: A2-ii gas inside R = bound gas inside 4R"),
                                      ("SB", "SB: steepest log slope of the all-gas profile (where well defined)"),
                                      ("2H2", "2H2: all gas / linear field = 2 (P: ESTIMATE, no halo exclusion)"),
                                      ("MDK15", "MDK15: dark-matter splashback, More+15 Eq. 7 (reference)")]}
    for name, rules in sets.items():
        fig, axes = plt.subplots(3, 2, figsize=(10.5, 9.6), sharex=True)
        for c, f in enumerate(("P", "E")):
            for r, (unit, ulab) in enumerate((("value", "R200c"), ("R200m", "R200m"), ("R500c", "R500c"))):
                ax = axes[r, c]
                for i, (rule, lab_) in enumerate(rules):
                    a = R(rule, f, unit)
                    if rule == "SB":
                        shallow = np.array([rad["bins"][lab]["rules"]["SB"][f].get("shallow", False) for lab in labs])
                        a[:, shallow] = np.nan
                    series(ax, logm, a[0], i, lab_, a[1], a[2])
                if name == "L1_radii_alternatives" and f == "E" and unit == "value" and d3:
                    y = [d3.get(lab, {}).get("R90", np.nan) for lab in labs]
                    ax.plot(logm, y, color=CAT[5], ls=LS_[5], marker=MARK[5], ms=5.5, mec=SURF, mew=1.0,
                            label="D3 R90: 90% of the bound component's mean DM (Test J truth)", zorder=3)
                ax.set_ylabel(f"truncation radius [{ulab}]")
                if r == 0:
                    ax.set_title(FIELD_NAME[f])
                if r == 2:
                    ax.set_xlabel("mean log10(M200c / Msun) of the 0.2-dex bin")
        h, l_ = axes[0, 1].get_legend_handles_labels()
        fig.legend(h, l_, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.045))
        fig.suptitle("Truncation radius per mass bin (bands: 16-84% of 200 paired halo bootstraps)", fontsize=10.5)
        fig.tight_layout(rect=(0, 0.04, 1, 0.97))
        save(fig, name)

    # ------------------------------------------------------------------ 4. alternative diagnostics
    grid = cur["slope_grid"]
    fig, axes = plt.subplots(3, 2, figsize=(10.5, 10.2), sharex=True)
    panels = [(axes[0, 0], "slope__SB__E", "d ln E_all / d ln x", "SB", "E"),
              (axes[0, 1], "slope__SB__P", "d ln P_all / d ln x", "SB", "P"),
              (axes[2, 0], "slope__SH_TM__E", "d ln T_M / d ln x (mass-weighted T)", "SH_TM", "E"),
              (axes[2, 1], "slope__SH_K__E", "d ln K / d ln x (entropy)", "SH_K", "E")]
    for ax, k, ylab, rule, f in panels:
        for b, lm, lab in zip(bins, logm, labs):
            y = cur[f"{k}__{b}"]
            ax.plot(grid, y, color=mass_colour(lm), lw=1.2)
            s = rad["bins"][lab]["rules"][rule][f]
            if s["value"] is not None and s["status"] == "ok":
                ax.plot([s["value"]], [np.interp(s["value"], grid, y)], "o", color=mass_colour(lm), ms=5, mec=SURF, mew=1)
        ax.set_ylabel(ylab)
        ax.axvspan(*(LC.SLOPE_RANGE if rule == "SB" else LC.SHOCK_RANGE), color=GRID, alpha=0.35, lw=0, zorder=0)
    for ax, f, ylab in ((axes[1, 0], "E", "E_all / E_2h"), (axes[1, 1], "P", "P_all / P_2h")):
        for b, lm in zip(bins, logm):
            ax.plot(cur[f"ratio2hx__2H2__{f}__{b}"], cur[f"ratio2h__2H2__{f}__{b}"], color=mass_colour(lm), lw=1.2)
        ax.axhline(2, color=INK2, ls="--", lw=1)
        ax.axhline(1, color=INK2, ls=":", lw=1)
        ax.set_yscale("log")
        ax.set_ylim(0.3, 300)
        ax.set_ylabel(ylab + "  (linear field expectation)")
        ax.text(0.31, 2.2, "halo excess = field (2H2)", color=INK2, fontsize=8)
        ax.text(0.31, 1.07, "profile = field (2H1)", color=INK2, fontsize=8)
    for ax in axes.ravel():
        ax.set_xscale("log")
        ax.set_xlim(0.3, 4)
        ax.set_xticks([0.3, 0.5, 1, 1.5, 2, 3, 4])
        ax.set_xticklabels(["0.3", "0.5", "1", "1.5", "2", "3", "4"])
    for ax in axes[2]:
        ax.set_xlabel("r / R200c")
    axes[0, 0].set_title("electrons: logarithmic slope (dots: steepest point, SB)")
    axes[0, 1].set_title("pressure: logarithmic slope (dots: steepest point, SB)")
    axes[1, 0].set_title("electrons over the 2-halo expectation")
    axes[1, 1].set_title("pressure over the 2-halo expectation (ESTIMATE)")
    axes[2, 0].set_title("mass-weighted temperature slope\n(grey: search range; dots: interior minima, all shallow)")
    axes[2, 1].set_title("entropy slope; dots: interior minima\n(all shallow, only below 10^13.3)")
    fig.subplots_adjust(hspace=0.36)
    mass_colourbar(fig, axes)
    fig.suptitle("Alternative edges: profile steepening, the 2-halo floor, and shock signatures (all gas, z = 0)",
                 fontsize=10.5)
    save(fig, "L1_alternative_diagnostics")

    # ------------------------------------------------------------------ 5. inside fractions
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), sharey=True)
    rules = [("B32", "B32"), ("B50", "B50"), ("C68", "C68"), ("Req", "Req"), ("SB", "SB (where well defined)"),
             ("2H2", "2H2 (P: ESTIMATE)")]
    for c, f in enumerate(("P", "E")):
        ax = axes[c]
        for i, (rule, lab_) in enumerate(rules):
            a = R(rule, f, "inside_a2ii")
            if rule == "SB":
                shallow = np.array([rad["bins"][lab]["rules"]["SB"][f].get("shallow", False) for lab in labs])
                a[:, shallow] = np.nan
            series(ax, logm, a[0], i, lab_, a[1], a[2])
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("mean log10(M200c / Msun)")
        ax.set_title(("Y(<R_t) / Y(<4R) of the painted A2-ii pressure" if f == "P"
                      else "N_e(<R_t) / N_e(<4R) of the painted A2-ii electrons"))
    axes[0].set_ylabel("share of the 4R200c content inside R_t")
    axes[1].legend(loc="lower right", ncol=2)
    save(fig, "L1_inside_fractions")

    # ------------------------------------------------------------------ 6. observable retention heatmap
    terms = [("ymean", "mean y"), ("yy1h@100", "tSZ 1h, l=100"), ("yy1h@1000", "tSZ 1h, l=1000"),
             ("yy1h@3000", "tSZ 1h, l=3000"), ("dm_mean", "mean DM"), ("dm_var", "DM variance"),
             ("ksz1h@100", "kSZ 1h, l=100"), ("ksz1h@1000", "kSZ 1h, l=1000"), ("ksz1h@3000", "kSZ 1h, l=3000"),
             ("dmcl1h@1000", "DM C_l 1h, l=1000")]
    hrules = ["B32", "B32cb", "B50", "C68", "Req", "SB", "2H2"]
    M = np.full((len(terms), len(hrules)), np.nan)
    for i, (t, _) in enumerate(terms):
        for j, r in enumerate(hrules):
            v = glob["additive_terms"].get(t, {}).get(r)
            if v:
                M[i, j] = v["T_perbin"]
    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    im = ax.imshow(M, cmap=LinearSegmentedColormap.from_list("b", ["#f2f7fd"] + BLUE[2:]), vmin=0, vmax=1, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8.5,
                        color=SURF if M[i, j] > 0.55 else INK)
    ax.set_xticks(range(len(hrules)))
    ax.set_xticklabels(hrules)
    ax.set_yticks(range(len(terms)))
    ax.set_yticklabels([t[1] for t in terms])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("O(inside R_t) / O(4R) of the painted A2-ii target", color=INK2)
    cb.outline.set_visible(False)
    ax.set_title("Share of each 4R-painted 1-halo / linear term fixed by data inside each rule's R_t\n"
                 "(the fitted law is still painted to 4 R200c: 1 - value depends on the extrapolated region;\n"
                 "exact at fine edges, Test J's per-bin truncation curves; HalfDome lightcone, C14)", fontsize=9.5)
    save(fig, "L1_observable_retention")

    # ------------------------------------------------------------------ 7. pilot comparison
    cx = LC.jload(LC.RESULTS / "codex.json") if (LC.RESULTS / "codex.json").exists() else None
    if cx:
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))
        ax = axes[0]
        i = 0
        for b in (12, 13, 14):
            if b not in bins:
                continue
            x = st[f"fracx_P_hb/all_{b}"]
            series(ax, x, st[f"frac_P_hb/all_{b}"], i, f"ours: stack {LC.LABELS[b]} (z = 0)",
                   st[f"fracq16_P_hb/all_{b}"], st[f"fracq84_P_hb/all_{b}"], ms=0)
            i += 1
        for sim in ("L1_m9", "L2p8_m9"):
            p = cx["pilot"]["codex"][sim]
            series(ax, np.array(p["x"]), np.array(p["f_hb"]), i,
                   f"Codex pilot {sim}: log M = {p['log10_m200c']:.2f}, z = {p['redshift']:.2f}", ms=4)
            i += 1
        ax.axhline(0.32, color=INK2, ls="--", lw=1)
        ax.set_xscale("log")
        ax.set_xlim(0.3, 4)
        ax.set_ylim(0, 1.02)
        ax.set_xticks([0.3, 0.5, 1, 1.5, 2, 3, 4])
        ax.set_xticklabels(["0.3", "0.5", "1", "1.5", "2", "3", "4"])
        ax.set_xlabel("r / R200c")
        ax.set_ylabel("bound (central + own satellites) / all, pressure")
        ax.set_title("per shell")
        ax.legend(loc="lower left", fontsize=7.4)
        ax = axes[1]
        Rs = np.array([1, 2, 3, 4])
        i = 0
        for lab, o in cx["pilot"]["ours_pressure"].items():
            y = [o["stack_cumulative"][str(r)]["hb"] for r in Rs]
            lo = [o["per_halo_cumulative_hb"][str(r)]["q16"] for r in Rs]
            hi = [o["per_halo_cumulative_hb"][str(r)]["q84"] for r in Rs]
            series(ax, Rs, y, i, f"ours {lab} (band: per-halo 16-84%)", lo, hi)
            i += 1
        for sim in ("L1_m9", "L2p8_m9"):
            p = cx["pilot"]["codex"][sim]
            series(ax, Rs, [p["cumulative_hb_over_all"][str(r)] for r in Rs], i, f"Codex pilot {sim}")
            i += 1
        ax.set_ylim(0.4, 1.0)
        ax.set_xticks(Rs)
        ax.set_xlabel("sphere radius R / R200c")
        ax.set_ylabel("bound share of Y(<R)")
        ax.set_title("cumulative (pressure-volume inside the sphere)")
        ax.legend(loc="lower left", fontsize=7.4)
        fig.suptitle("Bound pressure fractions: our z = 0 stacks of the most massive bins vs the Codex particle pilot",
                     fontsize=10.5)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        save(fig, "L1_pilot_comparison")

        # -------------------------------------------------------------- 8. HalfDome map retention
        fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.2), sharey=True)
        edges = [0.2, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]
        for c, fld in enumerate(("tsz", "ksz", "dm")):
            ax = axes[c]
            for i, (band, eng) in enumerate((("77", "100"), ("325", "300"), ("1025", "1000"), ("3025", "3000"))):
                v = cx["halfdome_maps"][fld][band]
                ax.plot(edges, v["T_at_edges"][1:], color=CAT[i], ls="-", marker=MARK[i], ms=5, mec=SURF, mew=1,
                        label=f"HalfDome B12/B16 maps, l ~ {v['ell_centre']:.0f}")
                e = cx["engine"][fld][eng]["T_at"]
                ax.plot([1, 1.5, 2, 3, 4], [e["1"], e["1.5"], e["2"], e["3"], e["4"]], color=CAT[i], ls="--",
                        marker=MARK[i], ms=4, mfc=SURF, mew=1.2, label=f"engine, FLAMINGO A2-ii, l = {eng}")
            g = glob["global"]["B32"]["common"]["R"]
            ax.axvspan(g, 4.1, color=GRID, alpha=0.45, lw=0, zorder=0)
            ax.axvline(g, color=INK2, ls=":", lw=1.2)
            ax.text(g * 1.03, 0.04, f"fit radius x_t = {g:.2f}\n(beyond: extrapolated,\nstill painted)", color=INK2,
                    fontsize=7.6)
            ax.set_title({"tsz": "tSZ C_l^yy", "ksz": "kSZ C_l", "dm": "FRB DM C_l (z_s = 1)"}[fld])
            ax.set_xlabel("every halo cut at R / R200c")
            ax.set_xlim(0.15, 4.1)
            ax.set_ylim(0, 1.05)
        axes[0].set_ylabel("C_l(every halo cut at R) / C_l(4R)")
        h, l_ = axes[1].get_legend_handles_labels()
        fig.legend(h, l_, loc="lower center", ncol=4, fontsize=7.6, bbox_to_anchor=(0.5, -0.12), handlelength=4.2)
        fig.suptitle("Share of the 4R-painted map power that comes from inside R: HalfDome B12/B16 maps (Codex shell "
                     "cross-spectra, solid) and our engine (dashed).\nA hard cut is not the plan: the fitted law is "
                     "painted to 4 R200c, so 1 - share is what its extrapolation beyond x_t supplies", fontsize=10)
        fig.tight_layout(rect=(0, 0.03, 1, 0.94))
        save(fig, "L1_halfdome_map_retention")

    # ------------------------------------------------------------------ 10. extrapolation region (added 2026-10-16)
    ep = LC.RESULTS / "extrap.npz"
    if ep.exists():
        ex = np.load(ep)
        exj = LC.jload(LC.RESULTS / "extrap.json")
        x_t = float(ex["x_t"])
        xr_sh = LC.shell_radius(LC.X[:-1], LC.X[1:])

        def cum_at(S, R):
            return LC.cumulative_at(S, R)
        fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.6))
        pb = [b for b in (3, 10) if b in bins]
        for r, f in enumerate(("P", "E")):
            for c, b in enumerate(pb):
                ax = axes[r, c]
                Sa, Shb, Sall, Sp = (ex[f"Sa2ii__{f}__{b}"], ex[f"Shb__{f}__{b}"], ex[f"Sall__{f}__{b}"],
                                     ex[f"Spub__{f}__{b}"])
                Sp_r = Sp * cum_at(Sa, x_t) / cum_at(Sp, x_t)            # published shape, same content inside x_t
                m = xr_sh > 0.25
                for i, (S, lab_) in enumerate(((Sa, "A2-ii gas (painted target; upper)"),
                                               (Shb, "host-bound gas (lower)"),
                                               (Sp_r, "published " + ("B12" if f == "P" else "B16") +
                                                " shape, matched inside x_t"),
                                               (Sall, "all gas"))):
                    ax.plot(xr_sh[m], 4 * np.pi * xr_sh[m] ** 3 * S[m], color=CAT[i], ls=LS_[i], lw=1.6 if i < 3 else 1.1,
                            label=lab_, zorder=3 - 0.1 * i)
                ax.axvspan(x_t, 4.0, color=GRID, alpha=0.45, lw=0, zorder=0)
                ax.axvline(x_t, color=INK2, ls=":", lw=1.1)
                ax.text(x_t * 1.04, 0.96, "extrapolated region\n(painted, not fitted)", transform=ax.get_xaxis_transform(),
                        va="top", color=INK2, fontsize=7.8)
                ax.set_xscale("log")
                ax.set_xlim(0.25, 4)
                ax.set_ylim(bottom=0)
                ax.set_xticks([0.3, 0.5, 1, 1.5, 2, 3, 4])
                ax.set_xticklabels(["0.3", "0.5", "1", "1.5", "2", "3", "4"])
                ax.set_title(f"{FIELD_NAME[f]}, {LC.LABELS[b]}")
                ax.set_ylabel("4 pi x^3 P  [P200 R200c^3 per ln x]" if f == "P"
                              else "4 pi x^3 E  [rho_c f_b R200c^3 / mu_XGP per ln x]")
                if r == 1:
                    ax.set_xlabel("r / R200c")
                if c == 0:                       # one legend per row: the published law is B12 (P) / B16 (E)
                    ax.legend(loc="upper left", fontsize=7.4)
            ax = axes[r, 2]
            labs_e = [lab for lab in labs if lab in exj["bins"]]
            lm = np.array([exj["bins"][lab]["mean_logm200c"] for lab in labs_e])
            for i, (k, lab_) in enumerate((("s_a2ii", "A2-ii gas (upper)"), ("s_hb", "host-bound gas (lower)"),
                                           ("s_pub", "published " + ("B12" if f == "P" else "B16") + " shape"),
                                           ("s_all", "all gas"))):
                v = np.array([[exj["bins"][lab][f]["global"][k][q] for lab in labs_e] for q in ("value", "q16", "q84")])
                series(ax, lm, v[0], i, lab_, v[1], v[2], ms=4.5)
            ax.set_ylim(0, 0.85)
            ax.set_ylabel(f"share of {'Y' if f == 'P' else 'N_e'}(<4R) from x_t - 4 R200c")
            ax.set_title(f"{FIELD_NAME[f]}: extrapolated share")
            if r == 1:
                ax.set_xlabel("mean log10(M200c / Msun) of the 0.2-dex bin")
            ax.legend(loc="upper right", fontsize=7.4)
        fig.suptitle(f"The region the painting extrapolates into (beyond the fit radius x_t = {x_t:.2f} R200c, to 4 R200c): "
                     "measured all-gas and bound-gas stacks and the published law's shape (z = 0; bands: 16-84% of 200 "
                     "halo bootstraps)", fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        save(fig, "L1_extrapolation_bracket")

    # ------------------------------------------------------------------ 9. signal weights
    fig, axes = plt.subplots(2, 1, figsize=(8.4, 6.8), sharex=True)
    jl = [LC.LABELS[b] for b in LC.J_BINS]
    jm = np.array([rad["bins"][lab]["mean_logm200c"] if lab in rad["bins"] else np.nan for lab in jl])
    ax = axes[0]
    for i, (key, lab_) in enumerate((("P:family", "P family weights (D0 importance shares)"),
                                     ("E:family", "E family weights (D0 importance shares)"))):
        w = np.array([glob["weights"][key][lab] for lab in jl])
        series(ax, jm, w, i, lab_)
    for i, (f, lab_) in enumerate((("P", "P: truncated-target 1-halo/linear weights (B32)"),
                                   ("E", "E: truncated-target 1-halo/linear weights (B32)"))):
        tw = glob["truncated_target_weights"]["B32"].get(f)
        if tw:
            series(ax, jm, [tw["weights"][lab] for lab in jl], i + 2, lab_)
    ax.set_ylabel("share of the signal per 0.2-dex bin")
    ax.legend(fontsize=7.6)
    ax = axes[1]
    gtxt = []
    for i, f in enumerate(("P", "E")):
        a = R("B32", f)
        series(ax, logm, a[0], i, f"B32 per bin, {f}", a[1], a[2])
        g = glob["global"]["B32"][f]["R"]["family"]
        ax.axhline(g["value"], color=CAT[i], ls=":", lw=1.2)
        gtxt.append(f"{f}: {g['value']:.3f} [{g['q16']:.3f}, {g['q84']:.3f}]")
    ax.text(0.98, 0.95, "signal-weighted mean (dotted)\n" + "\n".join(gtxt), transform=ax.transAxes, ha="right",
            va="top", color=INK2, fontsize=8)
    ax.set_xlabel("log10(M200c / Msun)")
    ax.set_ylabel("R_t [R200c]")
    ax.set_title("B32 radius per bin and its signal-weighted mean (dotted)")
    ax.legend(fontsize=7.6, loc="upper left")
    save(fig, "L1_signal_weights")
    LC.log("figures written to", LC.FIGURES)


if __name__ == "__main__":
    main()
