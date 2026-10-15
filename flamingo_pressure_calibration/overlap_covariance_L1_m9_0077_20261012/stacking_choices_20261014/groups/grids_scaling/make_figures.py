"""Figures of the 'grids_scaling' group (reads work/*.json, work/*.pkl; writes figures/*.png)."""
from __future__ import annotations

import json
import pickle

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import gs_common as C  # noqa: E402

sc = C.sc
WORK = C.HERE / "work"
FIG = C.HERE / "figures"
FIG.mkdir(exist_ok=True)
M = json.loads((WORK / "main_results.json").read_text())
FR = json.loads((WORK / "fits_results.json").read_text())
with open(WORK / "figdata_main.pkl", "rb") as fh:
    D = pickle.load(fh)
with open(WORK / "figdata_fits.pkl", "rb") as fh:
    DF = pickle.load(fh)

# reference palette (dataviz skill): categorical slots in fixed order; one-hue blue ramp for mass (ordinal)
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
MARK = ["o", "s", "^", "D", "v", "P", "X", "*"]
LS = ["-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1)), "-", "--"]
RAMP = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                     "figure.dpi": 130, "savefig.dpi": 150, "lines.linewidth": 1.6})
FIT_E = sc.edges("fit")
XF = C.xcentre(FIT_E)
CAL = D["cal"]
BINS = {b["label"]: b for b in D["bins"]}
QMED = M["R500c_over_R200c_median"]
G = C.GROUP


def mass_color(i, n=10):
    return RAMP[int(round(i * (len(RAMP) - 1) / max(n - 1, 1)))]


def save(fig, name):
    fig.savefig(FIG / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", name)


def stk(vid, field, lab):
    return D["stk"][(vid, field, lab)]


# ---------------------------------------------------------------------------- 1. grid layout
def fig_grid_layout():
    fig, ax = plt.subplots(figsize=(9.5, 3.6))
    rows = [("fit", "fit (35, reference)", 1.0), ("x200", "x200 (67, storage)", 1.0), ("lee22", "lee22 (27)", 1.0),
            ("old", "radii234 old (36)", 1.0), ("x500", "x500 (40) x median R500c/R200c", QMED),
            ("b24", "b24 (30) x median R500c/R200c", QMED)]
    lab = "14.1-14.3"
    b = BINS[lab]
    for k, (g, name, s) in enumerate(rows):
        y = len(rows) - k
        e = sc.edges(g) * s
        e_plot = np.where(e > 0, e, 0.012)
        resm = M["resolution_masks"][g][lab]
        first = resm["first_resolved_edge"]
        if first is not None:
            ax.fill_between([first * s, e[-1]], y - 0.32, y + 0.32, color="#cde2fb", lw=0, zorder=0)
        ax.vlines(e_plot, y - 0.3, y + 0.3, color=INK, lw=0.9)
        for h in resm["holes"]:
            ax.fill_between([h[0] * s, h[1] * s], y - 0.32, y + 0.32, color="#e34948", lw=0, zorder=1)
        ax.text(0.0105, y, name, ha="right", va="center", fontsize=9, color=INK)
    for xv in (1.0, 1.34, 2.0, 4.0):
        ax.axvline(xv, color=INK2, lw=0.8, ls=":")
    ax.set_xscale("log")
    ax.set_xlim(0.0105, 6.5)
    ax.set_ylim(0.4, len(rows) + 0.6)
    ax.set_yticks([])
    ax.grid(False)
    ax.set_xlabel(r"$r/R_{200c}$ (R500c grids converted with the median $R_{500c}/R_{200c}$ = %.3f)" % QMED)
    ax.set_title(f"Shell edges of the six cached grids; blue = strict-resolved range, red = unresolved holes "
                 f"(bin {lab})", fontsize=10, loc="left")
    save(fig, "grid_layout.png")


# ---------------------------------------------------------------------------- 2. stacks on the grids
def fig_grid_profiles():
    lab = "14.1-14.3"
    grids = [("ref", "fit", "fit (35)"), (f"{G}:x200", "x200", "x200 (67)"), (f"{G}:lee22", "lee22", "lee22 (27)"),
             (f"{G}:old", "old", "old (36)")]
    fig, axs = plt.subplots(2, 2, figsize=(10, 6.4), sharex=True, gridspec_kw={"height_ratios": [2, 1.2]})
    for j, (field, ylab) in enumerate((("P", r"$x^3\,P_e/P_{200}$"), ("E", r"$x^3\,n_e\mu_{\rm XGP}/(\rho_c f_b)$"))):
        for k, (vid, g, name) in enumerate(grids):
            r = stk(vid, field, lab)
            e = sc.edges(g)
            x = C.xcentre(e)
            ok = np.asarray(r["resolved"], bool)
            axs[0, j].plot(x[ok], (x**3 * r["value"])[ok], ls="none", marker=MARK[k], ms=4.5 if k else 6,
                           mfc="none" if k else CAT[k], color=CAT[k], label=name)
            fe = np.asarray(r["err"]) / np.asarray(r["value"])
            axs[1, j].plot(x[ok], 100 * fe[ok], ls="none", marker=MARK[k], ms=4.5 if k else 6,
                           mfc="none" if k else CAT[k], color=CAT[k])
            bad = ~ok & (x > 0.03)
            if bad.any() and k == 1:
                axs[1, j].plot(x[bad], 100 * fe[bad], ls="none", marker="x", ms=5, color="#e34948",
                               label="x200 shells failing the strict rule\n(inner core and thin slivers)")
        axs[0, j].set_yscale("log")
        axs[0, j].set_ylabel(ylab)
        axs[1, j].set_ylabel("bootstrap error of the\nstacked mean [%]")
        axs[1, j].set_xscale("log")
        axs[1, j].set_xlabel(r"$r/R_{200c}$")
        axs[1, j].set_yscale("log")
        axs[0, j].set_title(("Electron pressure" if field == "P" else "Free-electron density") + f", M200c bin {lab}",
                            fontsize=10, loc="left")
    axs[0, 0].legend(loc="lower right", ncol=2, fontsize=8.5)
    axs[1, 0].legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    save(fig, "grid_profiles.png")


# ---------------------------------------------------------------------------- 3. fit sensitivity to the grid
def fig_grid_fits():
    fig, axs = plt.subplots(2, 3, figsize=(12, 6.4), sharex=True)
    for i, field in enumerate(("P", "E")):
        for j, rfit in enumerate((1.34, 2.0, 4.0)):
            ax = axs[i, j]
            case = f"{field}_Rfit{rfit:g}"
            for k, g in enumerate(("x200", "old", "lee22")):
                if (case, g) not in DF["figd"]:
                    continue
                dd = DF["figd"][(case, g)]
                dI = dd["dI"][:, :, 2]          # ln Y_model(<4) or N_e,model(<4)
                v = 100 * (np.exp(dI[:, 0]) - 1)
                er = 100 * np.nanstd(dI[:, 1:], axis=1, ddof=1)
                ax.errorbar(dd["logm"] + 0.012 * (k - 1), v, er, ls=LS[k], marker=MARK[k + 1], ms=5,
                            color=CAT[k + 1], capsize=2,
                            label=("old (to 1.3508) vs fit grid" if g == "old" and rfit == 1.34   # no 1.34 edge
                                   else f"{g} vs fit grid"))
            ax.axhline(0, color=INK2, lw=0.8)
            ax.set_title(f"{'B12 pressure' if field == 'P' else 'B16 electrons'}, $R_{{\\rm fit}}$ = {rfit:g} $R_{{200c}}$",
                         fontsize=10, loc="left")
            if j == 0:
                ax.set_ylabel(("$Y$" if field == "P" else "$N_e$") + r"$_{\rm model}(<4R_{200c})$ change [%]")
            if i == 1:
                ax.set_xlabel(r"bin mean $\log_{10} M_{200c}\,[M_\odot]$")
    axs[0, 0].legend(fontsize=8, loc="upper left")   # the empty corner (the 13.6 point sits lower left)
    fig.suptitle("Fitted-model integral to 4 R200c: grid variant minus fit grid (objective (a), strict shells; "
                 "error bars = 50 paired bootstrap replicates)", fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    save(fig, "grid_fit_sensitivity.png")


# ---------------------------------------------------------------------------- 4. self-similarity: profiles
def fig_selfsim_profiles():
    sets = [("ref", "P", r"$r/R_{200c}$", r"$x^3 P_e/P_{200}$", "R200c, P200 (reference)", 1.0),
            (f"{G}:fit_P500", "P", r"$r/R_{200c}$", r"$x^3 P_e/P_{e,500}$", "R200c, P_e,500 (B24)", 1.0),
            (f"{G}:x500_P200", "P", r"$r/R_{500c}$", r"$x^3 P_e/P_{200}$", "R500c, P200", QMED),
            (f"{G}:x500_B24", "P", r"$r/R_{500c}$", r"$x^3 P_e/P_{e,500}$", "R500c, P_e,500 (B24)", QMED),
            (f"{G}:x500_A10M", "P", r"$r/R_{500c}$", r"$x^3 P_e/[P_{500}(M_{500}/3\times10^{14})^{0.12}]$",
             "R500c, A10 P500 x M^0.12", QMED),
            ("ref", "E", r"$r/R_{200c}$", r"$x^3 n_e\mu_{\rm XGP}/(\rho_c f_b)$", "Electrons: R200c (reference)", 1.0),
            (f"{G}:x500_B24", "E", r"$r/R_{500c}$", r"$x^3 n_e/n_{e,500}$", "Electrons: R500c, n_e,500 (B24)", QMED)]
    fig, axs = plt.subplots(2, 4, figsize=(15, 7.2))
    axs = axs.ravel()
    for a, (vid, field, xl, yl, title, q) in enumerate(sets):
        ax = axs[a]
        g = D["stk"][(vid, field, CAL[0])].get("grid", "fit")
        e = sc.edges(g)
        x = C.xcentre(e)
        for i, lab in enumerate(CAL):
            r = stk(vid, field, lab)
            ok = np.asarray(r["resolved"], bool)
            ax.plot(x[ok], (x**3 * r["value"])[ok], color=mass_color(i), lw=1.4,
                    label=lab if a == 0 and i in (0, 3, 6, 9) else None)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_title(title, fontsize=10, loc="left")
        lo, hi = (0.04, 4.0) if q == 1.0 else (0.04 / q, 4.0 / q)
        ax.set_xlim(lo, hi)
    axs[0].legend(title=r"$\log_{10}M_{200c}$ bin", fontsize=8, title_fontsize=8, loc="lower right")
    axs[-1].axis("off")
    axs[-1].text(0.0, 0.9, "Calibrated M200c bins 13.5-15.5 (light = low mass,\ndark = high mass); strict-resolved "
                 "shells only.\nR500c panels span the same physical range\n(4 R200c = %.2f R500c at the median "
                 "ratio)." % (4 / QMED), fontsize=9, va="top", color=INK2, transform=axs[-1].transAxes)
    fig.tight_layout()
    save(fig, "selfsim_profiles.png")


# ---------------------------------------------------------------------------- 5. self-similarity metrics
def fig_selfsim_metrics():
    """Same bin set as the report tables: at every radius, the 13.5-15.1 bins resolved in EVERY scaling of the field
    (>= 4 bins); markers = the tabulated values (results.json self_similarity.*.common_bins_*)."""
    fig, axs = plt.subplots(2, 2, figsize=(11, 7.6), sharex=True)
    for row, field in enumerate(("P", "E")):
        ss = D["selfsim"][field]
        names = list(ss)
        well = np.asarray(ss[names[0]]["well"], bool)
        lnF = {n: np.asarray(ss[n]["lnF_dense"]) for n in names}
        sdl = {n: np.asarray(ss[n]["sdln_dense"]) for n in names}
        common = well[:, None] & np.all([np.isfinite(lnF[n]) & np.isfinite(sdl[n]) for n in names], axis=0)
        ncom = common.sum(axis=0)
        xd = np.asarray(ss[names[0]]["x_dense"])
        for k, name in enumerate(names):
            spread = np.array([np.std(lnF[name][common[:, j], j], ddof=1) if ncom[j] >= 4 else np.nan
                               for j in range(len(xd))])
            med = np.array([np.median(sdl[name][common[:, j], j]) if ncom[j] >= 4 else np.nan
                            for j in range(len(xd))])
            axs[row, 0].plot(xd, spread, color=CAT[k], ls=LS[k], label=name)
            axs[row, 1].plot(xd, med, color=CAT[k], ls=LS[k], label=name)
            tab = M["self_similarity"][field][name]
            xt = np.asarray(tab["x_R200c_equivalent"], float)
            for col, keyt in ((0, "common_bins_spread_sd_lnF"), (1, "common_bins_within_bin_sd_ln_median")):
                yt = np.array([np.nan if v is None else v for v in tab[keyt]], float)
                axs[row, col].plot(xt, yt, ls="none", marker=MARK[k], ms=4.5, color=CAT[k], mfc="white")
            if k == 0:
                noise = np.asarray(ss[name]["noise_dense"])
                axs[row, 0].plot(xd, noise, color=INK2, lw=1.0, ls=":", label="median bootstrap error of a stack")
        axs[row, 0].set_ylabel(r"cross-bin spread: SD over bins of $\ln F$")
        axs[row, 1].set_ylabel(r"within-bin halo scatter: median SD($\ln F$)")
        fl = "Pressure" if field == "P" else "Electrons"
        axs[row, 0].set_title(f"{fl}: mass dependence (common resolved bins of 13.5-15.1)", fontsize=10, loc="left")
        axs[row, 1].set_title(f"{fl}: halo-to-halo scatter (median over the same bins)", fontsize=10, loc="left")
        # electrons: the curves fill the upper left (0.19-0.33 R200c-equiv.); raise the y-limit and put the legend in
        # the empty band above all curves so it covers none of them
        if field == "E":
            axs[row, 0].set_ylim(0, 1.32 * axs[row, 0].get_ylim()[1])
        axs[row, 0].legend(fontsize=7.5, loc="upper left" if field == "P" else "upper right")
        ax2 = axs[row, 1].twinx()
        ax2.step(xd, ncom, where="mid", color="#b8b6b0", lw=0.9)
        ax2.set_ylim(0, 64)
        ax2.set_yticks([4, 8])
        ax2.set_ylabel("common bins", color="#8a8984", fontsize=8, loc="bottom")
        ax2.tick_params(axis="y", colors="#8a8984", labelsize=8)
        ax2.grid(False)
        ax2.spines["right"].set_visible(True)
        ax2.spines["top"].set_visible(False)
    for ax in axs[1]:
        ax.set_xlabel(r"radius in $R_{200c}$-equivalents ($r/R_{500c}\times$ %.3f for R500c scalings)" % QMED)
    for ax in axs.ravel():
        ax.set_xscale("log")
        ax.set_xlim(0.05, 3.8)
        ax.set_ylim(bottom=0)
    fig.text(0.01, -0.01, "Lines: dense radii; open markers: the values tabulated in the report (same bins). "
             "Grey step (right axis): number of common bins (>= 4 required).", fontsize=8, color=INK2)
    fig.tight_layout()
    save(fig, "selfsim_metrics.png")


# ---------------------------------------------------------------------------- 6. cumulative budgets
def fig_cumulative():
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for j, (field, name) in enumerate((("P", r"$Y(<x)/Y(<4R_{200c})$"), ("E", r"$N_e(<x)/N_e(<4R_{200c})$"))):
        ax = axs[j]
        for i, lab in enumerate(CAL):
            c = D["cum"][field][lab]
            ax.plot(FIT_E[1:], c["cum"], color=mass_color(i), lw=1.4, label=lab if i in (0, 3, 6, 9) else None)
            ax.fill_between(FIT_E[1:], c["cum_lo"], c["cum_hi"], color=mass_color(i), alpha=0.15, lw=0)
        for xv, t in ((1.0, "1"), (1.34, "1.34"), (2.0, "2"), (3.0, "3")):
            ax.axvline(xv, color=INK2, lw=0.8, ls=":")
        ax.set_xscale("log")
        ax.set_xlim(0.05, 4.0)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel(r"$x = r/R_{200c}$")
        ax.set_ylabel(name)
        ax.set_title(("Compton Y (tSZ)" if field == "P" else "Free electrons (kSZ, DM)") +
                     ": stacked cumulative fraction", fontsize=10, loc="left")
    axs[0].legend(title=r"$\log_{10}M_{200c}$ bin", fontsize=8, title_fontsize=8, loc="upper left")
    fig.tight_layout()
    save(fig, "cumulative_budgets.png")


def fig_outside_fraction():
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    labs = [b["label"] for b in D["bins"]]
    logm = np.array([b["mean_logm200c"] for b in D["bins"]])
    for j, field in enumerate(("P", "E")):
        ax = axs[j]
        cu = M["cumulative_fractions_fit_grid"][field]
        for k, xv in enumerate((1.0, 1.34, 2.0, 3.0)):
            v = np.array([cu[lab][f"beyond_{xv:g}"]["value"] for lab in labs])
            e = np.array([cu[lab][f"beyond_{xv:g}"]["err"] for lab in labs])
            ax.errorbar(logm, 100 * v, 100 * e, color=CAT[k], marker=MARK[k], ms=4.5, ls=LS[k], capsize=2,
                        label=f"beyond {xv:g} $R_{{200c}}$")
        ax.axvspan(12.45, 13.5, color="#f0efec", zorder=0)
        ax.text(12.55, 2, "exploratory bins\n(cores unresolved)", fontsize=8, color=INK2, va="bottom")
        ax.set_xlabel(r"bin mean $\log_{10} M_{200c}\,[M_\odot]$")
        ax.set_ylabel(("Y" if field == "P" else r"$N_e$") + r" outside $x$, % of the value inside 4 $R_{200c}$")
        ax.set_title(("Compton Y" if field == "P" else "Free electrons") + " beyond x (stacked mean, all gas)",
                     fontsize=10, loc="left")
        ax.set_ylim(0, 100)
    axs[1].legend(fontsize=8.5, loc="upper right")
    fig.tight_layout()
    save(fig, "outside_fraction_vs_mass.png")


# ---------------------------------------------------------------------------- 7. truncation impacts
def fig_truncation():
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.3))
    tr = M["truncation_first_order"]
    for j, (field, obs) in enumerate((("P", C.OBS_P), ("E", C.OBS_E))):
        ax = axs[j]
        cuts = sorted(float(c) for c in tr[field]) + [4.0]
        for k, o in enumerate(obs):
            v = [100 * tr[field][f"{c:g}"][o]["dO_over_O"] if c < 4 else 0.0 for c in cuts]
            ax.plot(cuts, v, color=CAT[k], marker=MARK[k], ms=5, ls=LS[k], label=o.replace("_", " "))
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_xlabel(r"painting cut $R_{\rm cut}/R_{200c}$ (reference stack, calibrated bins only)")
        ax.set_ylabel("first-order change of the observable [%]")
        ax.set_title(("Pressure" if field == "P" else "Electrons") + ": signal lost by cutting below 4 $R_{200c}$",
                     fontsize=10, loc="left")
        ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    save(fig, "truncation_first_order.png")


# ---------------------------------------------------------------------------- 8. R500c stack painted at M200c
def fig_r500_paint():
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True)
    show = ["13.5-13.7", "14.1-14.3", "14.5-14.7", "14.9-15.1"]
    lo, hi = 1.0, 1.0
    for j, field in enumerate(("P", "E")):
        ax = axs[j]
        for i, lab in enumerate(show):
            ci = D["conv"][(f"{G}:x500_paint", field, lab)]
            ref = D["conv"][("ref", field, lab)]
            ok = np.asarray(ci["ok"], bool)
            r = ci["value"] / ref["value"]
            er = ci["err"] / ref["value"]
            col = CAT[i]
            ax.plot(XF[ok], r[ok], color=col, marker=MARK[i], ms=4, ls=LS[i], label=f"{lab}")
            ax.fill_between(XF[ok], (r - er)[ok], (r + er)[ok], color=col, alpha=0.15, lw=0)
            lo, hi = min(lo, np.nanmin((r - er)[ok])), max(hi, np.nanmax((r + er)[ok]))
            hv = D["conv"][(f"{G}:x500_halo", field, lab)]
            okh = np.asarray(hv["ok"], bool)
            rh = (hv["value"] / ref["value"])[okh]
            ax.plot(XF[okh], rh, color=col, ls=":", lw=1.1,
                    label="per-halo conversion (check)" if i == 0 else None)
            lo, hi = min(lo, np.nanmin(rh)), max(hi, np.nanmax(rh))
        ax.axhline(1, color=INK2, lw=0.8)
        ax.set_xscale("log")
        ax.set_xlabel(r"$r/R_{200c}$")
        ax.set_title(("Pressure" if field == "P" else "Electrons") +
                     ": r/R500c, B24-normalised stack painted at M200c", fontsize=10, loc="left")
    axs[0].set_ylabel("painted mean / reference stack (dimensionless)")
    pad = 0.15 * (hi - lo)
    axs[0].set_ylim(lo - 0.02, hi + pad)
    axs[0].legend(fontsize=7.5, loc="upper left", ncol=2,
                  title="M200c bin (solid: bin-mean conversion, band = bootstrap SD)", title_fontsize=7.5)
    fig.tight_layout()
    save(fig, "r500_stack_painted_at_m200c.png")


# ---------------------------------------------------------------------------- 9. clumping
def fig_clumping():
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.2))
    for i, lab in enumerate(CAL):
        c = D["clump"][lab]
        ok = np.asarray(c["resolved"], bool)
        col = mass_color(i)
        kw = dict(color=col, lw=1.3, label=lab if i in (0, 3, 6, 9) else None)
        axs[0].plot(XF[ok], np.asarray(c["C_P"])[ok], **kw)
        axs[1].plot(XF[ok], np.asarray(c["C_rho"])[ok], **kw)
        axs[2].plot(XF[ok], np.asarray(c["f_V"])[ok], **kw)
    axs[0].set_ylabel(r"$C_P=\langle P_e^2\rangle_V/\langle P_e\rangle_V^2$")
    axs[1].set_ylabel(r"$C_\rho=\langle\rho^2\rangle_V/\langle\rho\rangle_V^2$")
    axs[2].set_ylabel(r"$f_V=\sum_i m_i/\rho_i\;/\;V_{\rm shell}$")
    for ax, t in zip(axs, ("Pressure clumping", "Gas-density clumping (all phases)",
                           "SPH-volume filling of the shell")):
        ax.set_xscale("log")
        ax.set_xlabel(r"$r/R_{200c}$")
        ax.set_title(t, fontsize=10, loc="left")
    axs[0].set_yscale("log")
    axs[1].set_yscale("log")
    axs[0].legend(title=r"$\log_{10}M_{200c}$ bin", fontsize=8, title_fontsize=8, loc="upper left")
    fig.tight_layout()
    save(fig, "clumping_and_volume_filling.png")


# ---------------------------------------------------------------------------- 10. conversion factors
def fig_conversion_factors():
    t = sc.halo_table()
    q = t["r500c"] / t["r200c"]
    f = t["p500_B24_pa"] / t["p200_pa"]
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.0))
    for ax, v, yl in ((axs[0], q, r"$R_{500c}/R_{200c}$"), (axs[1], f, r"$P_{e,500}^{\rm B24}/P_{200}$")):
        ax.scatter(t["logm200c"], v, s=3, color=CAT[0], alpha=0.35, lw=0, rasterized=True)
        for b in D["bins"]:
            rows = sc.find_bin(C.SCHEME, b["label"])
            w = rows["weights"]
            vv = v[rows["rows"]]
            m = np.average(vv, weights=w)
            sdv = np.sqrt(np.average((vv - m)**2, weights=w))
            ax.errorbar(b["mean_logm200c"], m, sdv, color=CAT[1], marker="s", ms=4, capsize=2)
        ax.set_xlabel(r"$\log_{10} M_{200c}\,[M_\odot]$")
        ax.set_ylabel(yl)
    axs[0].set_title("Per-halo radius ratio (dots) and bin mean +- SD (squares)", fontsize=10, loc="left")
    axs[1].set_title("Per-halo pressure-scale ratio, P_e,500 / P200", fontsize=10, loc="left")
    fig.tight_layout()
    save(fig, "conversion_factors.png")


if __name__ == "__main__":
    fig_grid_layout()
    fig_grid_profiles()
    fig_grid_fits()
    fig_selfsim_profiles()
    fig_selfsim_metrics()
    fig_cumulative()
    fig_outside_fraction()
    fig_truncation()
    fig_r500_paint()
    fig_clumping()
    fig_conversion_factors()
