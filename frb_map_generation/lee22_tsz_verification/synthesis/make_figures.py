#!/usr/bin/env python
"""Synthesis figures for the tSZ / Lee22 profile verification (2026-09-29).

Inputs: OUT/synthesis_inputs.npz (extract_inputs.py) plus the workstream tables read directly:
  C_flamingo/pressure/pressure_metrics_raw.csv, C_flamingo/density/density_metrics.csv,
  verify_B_class_sz/v4_compare.json (exact per-halo catalogue ratios),
  lee22_normalization_forensics_20260928/D_gas_fractions/konietzka_implied_fe_shells.json.
Outputs (OUT): fig_tsz1_pressure_profiles, fig_tsz2_redshift_trend, fig_tsz3_yy_power, fig_tsz4_density (.png, .pdf),
  fig_numbers.json (every number printed on the figures).
Run: LD_LIBRARY_PATH=/home/cbllover/miniconda3/envs/halfdome/lib /home/cbllover/miniconda3/envs/halfdome/bin/python make_figures.py [1 2 3 4]
"""
import csv
import json
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator  # noqa: E402

ROOT = "/home/cbllover/HalfDome/frb_map_generation"
VER = f"{ROOT}/outputs/lee22_tsz_verification_20260929"
OUT = f"{VER}/synthesis"
S = np.load(f"{OUT}/synthesis_inputs.npz")
NUMBERS = {}

# colours (task convention)
DATA, B12C, PAINT, CORR, LIT, OTHER = "black", "#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7"
GREY = "0.55"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 14, "axes.labelsize": 15, "axes.titlesize": 15,
    "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 13, "figure.facecolor": "white",
    "axes.facecolor": "white", "savefig.facecolor": "white", "axes.linewidth": 1.0,
    "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    "lines.linewidth": 2.2, "axes.grid": False,
})


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}/{name}.{ext}", dpi=150 if ext == "png" else None)
    plt.close(fig)
    print("wrote", f"{OUT}/{name}.png")


def log_ticks(ax, axis, ticks, labels=None):
    a = ax.yaxis if axis == "y" else ax.xaxis
    a.set_major_locator(FixedLocator(ticks))
    a.set_minor_locator(NullLocator())
    a.set_major_formatter(NullFormatter())
    (ax.set_yticklabels if axis == "y" else ax.set_xticklabels)(labels or [f"{t:g}" for t in ticks])


def read_csv(path):
    return list(csv.DictReader(open(path)))


FLP = read_csv(f"{VER}/C_flamingo/pressure/pressure_metrics_raw.csv")
FLD = read_csv(f"{VER}/C_flamingo/density/density_metrics.csv")


def fl_rows(rows, lo, zmax=2.01):
    sel = [r for r in rows if abs(float(r["log10_m_lo"]) - lo) < 1e-6 and float(r["z"]) <= zmax]
    return sorted(sel, key=lambda r: float(r["z"]))


# =============================================================================================== fig 1
Z_COLS = [0.0, 0.5, 1.0, 2.0]
ROWS1 = [  # (label, CAMELS bin lo, FLAMINGO bin lo)
    (r"$M_{200c}\approx10^{13.2\text{-}13.3}\,M_\odot$", 13.0, 13.2),
    (r"$M_{200c}\approx10^{13.7}\,M_\odot$", 13.5, 13.6),
    (r"$M_{200c}\approx10^{14.5}\,M_\odot$", None, 14.4),
]
PMODELS = [("P_b12", "B12", B12C, "-"), ("P_lee_printed", "Lee22", PAINT, "-"), ("P_lee_a2", "Lee22_a2", CORR, "--")]


def fig1():
    xg = S["xg"]
    xs = (xg >= 0.03) & (xg <= 2.0)
    fig = plt.figure(figsize=(18.5, 21.5))
    gs = fig.add_gridspec(9, 4, height_ratios=[2.3, 1.35, 0.42, 2.3, 1.35, 0.42, 2.3, 1.35, 0.01],
                          hspace=0.0, wspace=0.07, left=0.085, right=0.985, top=0.893, bottom=0.045)
    ylims = [(0.012, 30), (0.012, 30), (0.012, 60)]
    nums = {}
    for i, (rlabel, clo, flo) in enumerate(ROWS1):
        for j, zt in enumerate(Z_COLS):
            ax = fig.add_subplot(gs[3 * i, j])
            axr = fig.add_subplot(gs[3 * i + 1, j], sharex=ax)
            ctag = f"cam_z{zt:g}_m{clo:g}" if clo is not None else None
            has_c = ctag is not None and (ctag + "_P") in S.files
            ftag = f"fl_z{zt:g}_m{flo:g}"
            has_f = (ftag + "_P") in S.files
            for a in (ax, axr):
                a.set_xscale("log")
                a.set_yscale("log")
                a.axvspan(1.34, 2.0, color="0.93", zorder=0, lw=0)
            txt = []
            if has_c:
                P, E = S[ctag + "_P"], S[ctag + "_Perr"]
                ax.fill_between(xg[xs], (P - E)[xs], (P + E)[xs], color="0.75", lw=0, zorder=2)
                ax.plot(xg[xs], P[xs], color=DATA, lw=2.6, zorder=4)
                txt.append(f"TNG: {int(S[ctag + '_nh'])} haloes (z={float(S[ctag + '_z']):.2f})")
                for key, name, col, ls in PMODELS:
                    ax.plot(xg[xs], S[f"{ctag}_{key}"][xs], color=col, ls=ls, lw=2.2, zorder=3)
                    r, re = S[f"{ctag}_{key}_over_data"], S[f"{ctag}_{key}_over_data_err"]
                    axr.plot(xg[xs], r[xs], color=col, ls=ls, lw=2.2, zorder=3)
                axr.fill_between(xg[xs], 1 - (E / P)[xs], 1 + (E / P)[xs], color="0.75", lw=0, zorder=1)
                k = int(np.argmin(np.abs(xg - 0.5)))
                nums[f"row{i}_z{zt:g}_TNG_x0.5"] = {n: float(S[f"{ctag}_{key}_over_data"][k]) for key, n, _c, _l in PMODELS}
            if has_f:
                x, P, E = S[ftag + "_x"], S[ftag + "_P"], S[ftag + "_Perr"]
                ax.errorbar(x, P, E, fmt="s", ms=6.5, mfc="white", mec=DATA, mew=1.4, ecolor=DATA, elinewidth=1.2, zorder=5)
                txt.append(f"FLAMINGO: {int(S[ftag + '_nh'])} haloes")
                for key, name, col, ls in PMODELS:
                    m = S[f"{ftag}_{name}"]
                    if not has_c:
                        ax.plot(x, m, color=col, ls=ls, lw=2.2, zorder=3)
                    axr.plot(x, m / P, color=col, ls=":", lw=1.3, zorder=3)
                    axr.plot(x, m / P, "s", ms=6, mfc="white", mec=col, mew=1.5, zorder=4)
                k = int(np.argmin(np.abs(np.log(x / 0.5))))
                nums[f"row{i}_z{zt:g}_FL_x{x[k]:.2f}"] = {n: float(S[f"{ftag}_{n}"][k] / P[k]) for _k, n, _c, _l in PMODELS}
            axr.axhline(1, color=DATA, lw=1.0, zorder=2)
            if not has_c:
                ax.text(0.04, 0.05, "no TNG (CAMELS) haloes" if clo is not None else "no TNG (CAMELS) haloes\nin this mass range",
                        transform=ax.transAxes, fontsize=12, color="0.35", va="bottom")
            ax.text(0.97, 0.96, "\n".join(txt), transform=ax.transAxes, ha="right", va="top", fontsize=11.5, color="0.25")
            ax.set_xlim(0.03, 2.0)
            ax.set_ylim(*ylims[i])
            axr.set_ylim(0.035, 7)
            log_ticks(axr, "y", [0.1, 0.3, 1, 3], ["0.1", "0.3", "1", "3"])
            log_ticks(ax, "y", [0.1, 1, 10], ["0.1", "1", "10"])
            log_ticks(axr, "x", [0.03, 0.1, 0.3, 1], ["0.03", "0.1", "0.3", "1"])
            plt.setp(ax.get_xticklabels(), visible=False)
            if j > 0:
                plt.setp(ax.get_yticklabels(), visible=False)
                plt.setp(axr.get_yticklabels(), visible=False)
            else:
                ax.set_ylabel(r"$P_e/P_{200}$")
                axr.set_ylabel("model / sim.")
                ax.text(-0.25, 0.5, rlabel, transform=ax.transAxes, rotation=90, ha="center", va="center", fontsize=16,
                        fontweight="bold")
            if i == 0:
                ax.set_title(f"z = {zt:g}", fontsize=17, pad=8)
            axr.set_xlabel(r"$r/R_{200c}$", labelpad=2)
    # legend (outside the axes)
    h = [Line2D([], [], color=DATA, lw=2.6), Patch(color="0.75"),
         Line2D([], [], color=DATA, marker="s", mfc="white", ls="none", ms=7, mew=1.4),
         Line2D([], [], color=B12C), Line2D([], [], color=PAINT), Line2D([], [], color=CORR, ls="--"),
         Line2D([], [], color="0.3", lw=2.2), Line2D([], [], color="0.3", marker="s", mfc="white", ls=":", lw=1.3, ms=6),
         Patch(color="0.93")]
    lab = ["TNG model (CAMELS CV, 27 boxes; stack mean)", "TNG box-jackknife 1$\\sigma$",
           "FLAMINGO L2p8_m9 (resolved shells, bootstrap 1$\\sigma$)",
           "Battaglia+12, $P_e=0.5176\\,P_{th}$ (= XGPaint = CLASS-SZ)", "Lee22 no-c Table 7, as printed = as painted",
           r"Lee22 $\times(1+z)^2$ (M1 bookkeeping correction)",
           "ratio panels: model / TNG", "ratio panels: model / FLAMINGO", r"outside Lee22 fit range ($x>1.34$)"]
    fig.legend(h, lab, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.53, 0.995), fontsize=13.5,
               handlelength=2.6, columnspacing=1.6)
    fig.text(0.53, 0.927, "Electron pressure profiles; $P_{200}=200\\,G M_{200c}\\,\\rho_c(z) f_b/(2R_{200c})$ with each "
             "simulation's own $f_b$, physical $\\rho_c(z)$ and physical $R_{200c}$; models averaged over the same haloes",
             ha="center", fontsize=13, color="0.2")
    NUMBERS["fig1"] = nums
    save(fig, "fig_tsz1_pressure_profiles")


# =============================================================================================== fig 2
COLS2 = [  # (title, CAMELS lo, FLAMINGO lo)
    (r"$M_{200c}\approx10^{13.2\text{-}13.3}\,M_\odot$", 13.0, 13.2),
    (r"$M_{200c}\approx10^{13.7}\,M_\odot$", 13.5, 13.6),
    (r"$M_{200c}\approx10^{14.5}\,M_\odot$", None, 14.4),
]


def zaxis(ax):
    ax.set_xscale("log")
    ax.set_xlim(0.97, 3.1)
    log_ticks(ax, "x", [1, 1.5, 2, 2.5, 3], ["0", "0.5", "1", "1.5", "2"])


def fit_const(z, r, p):
    """geometric-mean F with r ~ (1+z)^p / F  over finite points"""
    ok = np.isfinite(r) & (r > 0)
    return float(np.exp(np.mean(np.log((1 + z[ok]) ** p / r[ok]))))


def fig2():
    fig, axes = plt.subplots(2, 3, figsize=(18.5, 12.2), sharex=True)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.83, bottom=0.075, wspace=0.08, hspace=0.08)
    zc = S["cam_trend_z"]
    zl = np.linspace(0, 2, 60)
    nums = {}
    for j, (title, clo, flo) in enumerate(COLS2):
        axp, axd = axes[0, j], axes[1, j]
        for a in (axp, axd):
            zaxis(a)
            a.set_yscale("log")
        # ---- pressure
        axp.axhline(1, color=PAINT, lw=2.4, zorder=2)
        axp.plot(1 + zl, (1 + zl) ** 2, color=CORR, ls="--", lw=2.4, zorder=2)
        fr = fl_rows(FLP, flo)
        zf = np.array([float(r["z"]) for r in fr])
        yf = np.array([float(r["Yhat200_flamingo"]) / float(r["Yhat200_Lee22"]) for r in fr])
        ef = np.array([float(r["Yhat200_flamingo_err"]) / float(r["Yhat200_flamingo"]) for r in fr]) * yf
        yb_f = np.array([float(r["Yhat200_B12"]) / float(r["Yhat200_Lee22"]) for r in fr])
        if clo is not None:
            r = S[f"cam_trend_m{clo:g}_Y_data_over_printed"]
            e = S[f"cam_trend_m{clo:g}_Y_data_over_printed_err"] * r
            yb = S[f"cam_trend_m{clo:g}_Y_B12_over_printed"]
            ok = np.isfinite(r) & (S[f"cam_trend_m{clo:g}_nh"] >= 3)
            axp.plot(1 + zc[ok], yb[ok], color=B12C, lw=2.4, zorder=3)
            axp.errorbar(1 + zc[ok], r[ok], e[ok], fmt="o", ms=7, color=DATA, elinewidth=1.2, zorder=5)
            ez, ep = S[f"emul_m{clo:g}_z"], S[f"emul_m{clo:g}_P_over_printed"]
            axp.plot(1 + ez, ep, "D", ms=9, mfc="white", mec=DATA, mew=1.6, zorder=6)
            kfit = np.polyfit(np.log(1 + zc[ok]), np.log(r[ok]), 1)[0]
            nums[f"col{j}_TNG_Y_slope"] = float(kfit)
            nums[f"col{j}_TNG_Y_over_printed_(1+z)^2"] = {f"{z:.2f}": float(v / (1 + z) ** 2) for z, v in zip(zc[ok], r[ok])
                                                          if min(abs(z - t) for t in (0, 0.465, 1.048, 1.479, 2.002)) < 0.01}
            nums[f"col{j}_emul_P_over_printed"] = dict(zip([f"{z:.2f}" for z in ez], map(float, ep)))
        else:
            axp.plot(1 + zf, yb_f, color=B12C, lw=2.4, zorder=3)
        axp.errorbar(1 + zf, yf, ef, fmt="s", ms=8, mfc="white", mec=DATA, mew=1.6, ecolor=DATA, zorder=5)
        sf = np.polyfit(np.log(1 + zf), np.log(yf), 1)[0]
        nums[f"col{j}_FL_Y_slope_rel_printed"] = float(sf)
        nums[f"col{j}_FL_Y_over_printed_(1+z)^2"] = dict(zip([f"{z:g}" for z in zf], map(float, yf / (1 + zf) ** 2)))
        axp.set_ylim(0.3, 20)
        log_ticks(axp, "y", [0.5, 1, 2, 5, 10], ["0.5", "1", "2", "5", "10"])
        axp.set_title(title, fontsize=16, pad=8)
        axp.text(0.04, 0.95, f"FLAMINGO / printed $\\propto(1+z)^{{{sf:.2f}}}$" +
                 (f"\nTNG / printed $\\propto(1+z)^{{{kfit:.2f}}}$" if clo is not None else ""),
                 transform=axp.transAxes, va="top", fontsize=13, color="0.15")
        # ---- density (electron fraction inside R200c)
        axd.axhline(1, color=LIT, lw=2.4, ls=":", zorder=2)
        axd.axhline(0.6016, color=PAINT, lw=2.4, zorder=2)
        axd.plot(1 + zl, 0.2841790493880425 * (1 + zl) ** 2.483190184608004, color=OTHER, lw=2.4, ls="-.", zorder=2)
        dr = fl_rows(FLD, flo)
        zfd = np.array([float(r["z"]) for r in dr])
        yfd = np.array([float(r["f_e_FL"]) / float(r["f_e_Lee22_literal"]) for r in dr])
        efd = np.array([float(r["f_e_FL_err"]) / float(r["f_e_FL"]) for r in dr]) * yfd
        yb16_f = np.array([float(r["f_gas_B16"]) / float(r["f_e_Lee22_literal"]) for r in dr])
        Fn_fl = fit_const(zfd, yfd, 3)
        if clo is not None:
            r = S[f"cam_trend_m{clo:g}_fgas_over_printed_fe"]
            e = S[f"cam_trend_m{clo:g}_fgas_over_printed_fe_err"] * r
            yb = S[f"cam_trend_m{clo:g}_fB16_over_printed_fe"]
            ok = np.isfinite(r) & (S[f"cam_trend_m{clo:g}_nh"] >= 3)
            axd.plot(1 + zc[ok], yb[ok], color=B12C, lw=2.4, zorder=3)
            axd.errorbar(1 + zc[ok], r[ok], e[ok], fmt="o", ms=7, color=DATA, elinewidth=1.2, zorder=5)
            ez, en = S[f"emul_m{clo:g}_z"], S[f"emul_m{clo:g}_n_over_printed"]
            axd.plot(1 + ez, en, "D", ms=9, mfc="white", mec=DATA, mew=1.6, zorder=6)
            Fn = fit_const(zc[ok], r[ok], 3)
            kd = np.polyfit(np.log(1 + zc[ok]), np.log(r[ok]), 1)[0]
            nums[f"col{j}_TNG_fgas_slope_rel_printed"] = float(kd)
            nums[f"col{j}_TNG_Fn_geomean"] = Fn
            axd.plot(1 + zl, (1 + zl) ** 3 / Fn, color=CORR, ls="--", lw=2.4, zorder=2)
            lab_fn = f"$(1+z)^3/F_n$, $F_n$ = {Fn:.1f} (TNG)"
        else:
            axd.plot(1 + zfd, yb16_f, color=B12C, lw=2.4, zorder=3)
            axd.plot(1 + zl, (1 + zl) ** 3 / Fn_fl, color=CORR, ls="--", lw=2.4, zorder=2)
            lab_fn = f"$(1+z)^3/F_n$, $F_n$ = {Fn_fl:.1f} (FLAMINGO)"
        axd.errorbar(1 + zfd, yfd, efd, fmt="s", ms=8, mfc="white", mec=DATA, mew=1.6, ecolor=DATA, zorder=5)
        sfd = np.polyfit(np.log(1 + zfd), np.log(yfd), 1)[0]
        nums[f"col{j}_FL_fe_slope_rel_printed"] = float(sfd)
        nums[f"col{j}_FL_Fn_geomean"] = Fn_fl
        axd.set_ylim(0.08, 8)
        log_ticks(axd, "y", [0.1, 0.2, 0.5, 1, 2, 5], ["0.1", "0.2", "0.5", "1", "2", "5"])
        axd.text(0.04, 0.95, lab_fn, transform=axd.transAxes, va="top", fontsize=13, color=CORR)
        axd.set_xlabel("z")
        if j == 0:
            axp.set_ylabel("$Y_{sph}(<R_{200c})$ / Lee22 printed\n(pressure normalisation)")
            axd.set_ylabel("$f_e(<R_{200c})$ / Lee22 printed\n(electron-density normalisation)")
        else:
            plt.setp(axp.get_yticklabels(), visible=False)
            plt.setp(axd.get_yticklabels(), visible=False)
        if clo is None:
            axp.text(0.96, 0.05, "no TNG (CAMELS) haloes", transform=axp.transAxes, ha="right", fontsize=12, color="0.35")
            axd.text(0.96, 0.05, "no TNG (CAMELS) haloes", transform=axd.transAxes, ha="right", fontsize=12, color="0.35")
    h = [Line2D([], [], color=DATA, marker="o", ls="none", ms=7),
         Line2D([], [], color=DATA, marker="s", mfc="white", ls="none", ms=8, mew=1.6),
         Line2D([], [], color=DATA, marker="D", mfc="white", ls="none", ms=9, mew=1.6),
         Line2D([], [], color=B12C), Line2D([], [], color=PAINT), Line2D([], [], color=CORR, ls="--"),
         Line2D([], [], color=LIT, ls=":"), Line2D([], [], color=OTHER, ls="-.")]
    lab = ["TNG model (CAMELS CV, 27 boxes; box-jackknife 1$\\sigma$); density: $f_{gas}$ (all gas)",
           "FLAMINGO L2p8_m9 (bootstrap 1$\\sigma$); density: electrons ($n_e=0$ in star-forming gas)",
           "Lee22 bookkeeping (group_particles) applied to CAMELS cells, 4 boxes",
           "Battaglia+12 $P_e$ (top) / Battaglia+16 gas, full ionisation (bottom)",
           "Lee22 as printed = as painted (top); production DM $\\times$0.6016 (bottom)",
           r"Lee22 corrected: $\times(1+z)^2$ (top); $\times(1+z)^3/F_n$ (bottom)",
           "Lee22 density as printed (literal Table 8)", r"Lee22 density $\times C(z)=0.284(1+z)^{2.48}$ (TNG-DM calibrated)"]
    fig.legend(h, lab, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.53, 0.998), fontsize=13,
               handlelength=2.4, columnspacing=1.8)
    NUMBERS["fig2"] = nums
    save(fig, "fig_tsz2_redshift_trend")


# =============================================================================================== fig 3
EDGES = np.unique(np.round(np.geomspace(30, 8000, 31)).astype(int))
CEN = np.sqrt(EDGES[:-1] * EDGES[1:])


def band(cl):
    ell = S["ell"].astype(float)
    dl = ell * (ell + 1) * cl / (2 * np.pi)
    return np.array([dl[lo:hi].mean() for lo, hi in zip(EDGES[:-1], EDGES[1:])])


def fig3():
    v4 = json.load(open(f"{VER}/verify_B_class_sz/v4_compare.json"))
    ev = np.array(v4["ells"])
    D = {k: band(S["cl_" + k]) for k in ("map_B12", "map_L22", "th_CS_b12_paint_all_hiprec", "th_CSX_lee22", "th_CSX_lee22m1",
                                         "th_CAT_lee22", "th_CAT_lee22m1", "flamingo_lc0", "flamingo_lc1")}
    # Lee22 x (1+z)^2 map prediction: painted Lee22 map x (catalogue-weighted halo model M1 / as painted)
    tr = D["th_CAT_lee22m1"] / D["th_CAT_lee22"]
    D["pred_L22m1"] = D["map_L22"] * tr
    tr_ver = np.array(v4["ratios"]["M1/as painted my cat (1h + CS 2h)"])
    tr_at = np.exp(np.interp(np.log(ev), np.log(CEN), np.log(tr)))
    fig = plt.figure(figsize=(13.5, 15.5))
    gs = fig.add_gridspec(3, 1, height_ratios=[3.0, 1.25, 1.6], hspace=0.0, left=0.12, right=0.975, top=0.86, bottom=0.06)
    ax = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1], sharex=ax)
    ax2 = fig.add_subplot(gs[2], sharex=ax)
    fl = D["flamingo_lc0"]
    ax.plot(CEN, 1e12 * fl, color=DATA, lw=3.2, zorder=6)
    ax.plot(CEN, 1e12 * D["flamingo_lc1"], color=DATA, lw=1.6, ls=(0, (1.5, 1.5)), zorder=6)
    ax.plot(CEN, 1e12 * D["map_B12"], color=B12C, lw=2.6, marker="o", ms=5, zorder=5)
    ax.plot(CEN, 1e12 * D["map_L22"], color=PAINT, lw=2.6, marker="o", ms=5, zorder=5)
    ax.plot(CEN, 1e12 * D["th_CS_b12_paint_all_hiprec"], color=B12C, lw=2.2, ls="--", zorder=4)
    ax.plot(CEN, 1e12 * D["th_CSX_lee22"], color=PAINT, lw=2.2, ls="--", zorder=4)
    ax.plot(CEN, 1e12 * D["th_CSX_lee22m1"], color=CORR, lw=2.2, ls="--", zorder=4)
    ax.plot(CEN, 1e12 * D["pred_L22m1"], color=CORR, lw=2.6, marker="o", ms=5, mfc="white", zorder=5)
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_ylim(0.03, 12)
    ax.set_ylabel(r"$10^{12}\,D_\ell^{yy}=10^{12}\,\ell(\ell+1)C_\ell^{yy}/2\pi$")
    log_ticks(ax, "y", [0.03, 0.1, 0.3, 1, 3, 10], ["0.03", "0.1", "0.3", "1", "3", "10"])
    # direct labels at the right end / where lines separate
    lbl = [(2200, D["map_L22"], PAINT, "Lee22 as painted", -0.33), (6000, D["map_B12"], B12C, "B12", -0.1),
           (4500, D["pred_L22m1"], CORR, r"Lee22$\times(1+z)^2$", 0.12), (2300, fl, DATA, "FLAMINGO", 0.1)]
    for xl, y, c, t, dy in lbl:
        yv = np.exp(np.interp(np.log(xl), np.log(CEN), np.log(1e12 * y)))
        ax.text(xl, yv * 10 ** dy, t, color=c, fontsize=13, ha="center", va="center", fontweight="bold")
    # ratio 1: map / model of the same profile
    ax1.axhline(1, color=DATA, lw=1.0)
    ax1.plot(CEN, D["map_B12"] / D["th_CS_b12_paint_all_hiprec"], color=B12C, lw=2.4, ls="--")
    ax1.plot(CEN, D["map_L22"] / D["th_CSX_lee22"], color=PAINT, lw=2.4, ls="--")
    ax1.plot(ev, v4["ratios"]["B12 map / (my cat 1h + CS 2h)"], "o", ms=11, color=B12C, mec="white", mew=1.0, zorder=5)
    ax1.plot(ev, v4["ratios"]["Lee22 map / (my cat 1h + CS 2h)"], "D", ms=7, color=PAINT, mec="white", mew=0.8, zorder=6)
    ax1.set_ylim(0.72, 1.24)
    ax1.set_yscale("linear")
    ax1.set_yticks([0.8, 0.9, 1.0, 1.1, 1.2])
    ax1.set_ylabel("map / model\n(same profile)")
    ax1.text(0.985, 0.94, "dashed: map / CLASS-SZ (Tinker08 $M_{200c}$, $x_{out}=4$)\npoints: map / exact per-halo sum over "
             "the catalogue (1h) + CLASS-SZ 2h", transform=ax1.transAxes, fontsize=11.5, color="0.25", ha="right", va="top")
    # ratio 2: / FLAMINGO lc0
    ax2.fill_between(CEN, np.minimum(1, D["flamingo_lc1"] / fl), np.maximum(1, D["flamingo_lc1"] / fl), color="0.78", lw=0)
    ax2.axhline(1, color=DATA, lw=1.0)
    ax2.plot(CEN, D["map_B12"] / fl, color=B12C, lw=2.6, marker="o", ms=5)
    ax2.plot(CEN, D["map_L22"] / fl, color=PAINT, lw=2.6, marker="o", ms=5)
    ax2.plot(CEN, D["pred_L22m1"] / fl, color=CORR, lw=2.6, marker="o", ms=5, mfc="white")
    ax2.plot(CEN, D["th_CSX_lee22m1"] / fl, color=CORR, lw=2.0, ls="--")
    ax2.set_yscale("log")
    ax2.set_ylim(0.2, 8)
    log_ticks(ax2, "y", [0.25, 0.5, 1, 2, 4], ["0.25", "0.5", "1", "2", "4"])
    ax2.set_ylabel("HalfDome / FLAMINGO lc0")
    ax2.text(0.015, 0.06, "grey band: FLAMINGO lightcone 1 / lightcone 0", transform=ax2.transAxes, fontsize=11.5, color="0.3")
    for a in (ax, ax1, ax2):
        a.axvspan(30, 100, color="0.94", lw=0, zorder=0)
        a.set_xlim(27, 8500)
    log_ticks(ax2, "x", [30, 100, 300, 1000, 3000, 8000], ["30", "100", "300", "1000", "3000", "8000"])
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax1.get_xticklabels(), visible=False)
    ax2.set_xlabel(r"$\ell$")
    ax.text(0.012, 0.975, "shaded: $\\ell<100$, cosmic-variance\ndominated (35 haloes above\n$10^{14.8}h^{-1}M_\\odot$ at z < 0.1)",
            transform=ax.transAxes, fontsize=11.5, color="0.3", va="top")
    h = [Line2D([], [], color=DATA, lw=3.2), Line2D([], [], color=DATA, lw=1.6, ls=(0, (1.5, 1.5))),
         Line2D([], [], color=B12C, marker="o", ms=5), Line2D([], [], color=PAINT, marker="o", ms=5),
         Line2D([], [], color=CORR, marker="o", ms=5, mfc="white"),
         Line2D([], [], color=B12C, ls="--"), Line2D([], [], color=PAINT, ls="--"), Line2D([], [], color=CORR, ls="--")]
    lab = ["FLAMINGO L2p8_m9 lightcone 0 ($/w_\\ell^2$)", "FLAMINGO lightcone 1",
           "HalfDome B12 map (XGPaint, 4$R_{200c}$ aperture)", "HalfDome Lee22 map (Table 7 as printed)",
           r"Lee22$\times(1+z)^2$: Lee22 map $\times$ catalogue-model ratio (estimate)",
           "CLASS-SZ B12 (high precision)", "CLASS-SZ Lee22 as painted", r"CLASS-SZ Lee22$\times(1+z)^2$"]
    fig.legend(h, lab, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.54, 0.997), fontsize=12.5,
               handlelength=2.6, columnspacing=1.4)
    fig.text(0.54, 0.868, "All HalfDome/CLASS-SZ curves in the paint cosmology ($h$=0.68, $\\Omega_b$=0.049, $\\Omega_c$=0.261); "
             "maps are point samples (no pixel window)", ha="center", fontsize=11.5, color="0.25")
    nums = {}
    for l in (100, 300, 1000, 2000, 3000, 5000):
        k = int(np.argmin(np.abs(CEN - l)))
        nums[str(l)] = {"band": f"{EDGES[k]}-{EDGES[k + 1]}", "1e12_Dl_FL_lc0": float(1e12 * fl[k]),
                        "B12map/FL": float(D["map_B12"][k] / fl[k]), "L22map/FL": float(D["map_L22"][k] / fl[k]),
                        "L22m1pred/FL": float(D["pred_L22m1"][k] / fl[k]), "CS_L22m1/FL": float(D["th_CSX_lee22m1"][k] / fl[k]),
                        "B12map/CS": float(D["map_B12"][k] / D["th_CS_b12_paint_all_hiprec"][k]),
                        "L22map/CSX": float(D["map_L22"][k] / D["th_CSX_lee22"][k]),
                        "L22m1pred/B12map": float(D["pred_L22m1"][k] / D["map_B12"][k]),
                        "lc1/lc0": float(D["flamingo_lc1"][k] / fl[k])}
    nums["M1_transfer_B_catalogue_vs_verifier_exact"] = dict(zip([f"{e:g}" for e in ev], map(float, tr_at / tr_ver)))
    NUMBERS["fig3"] = nums
    save(fig, "fig_tsz3_yy_power")


# =============================================================================================== fig 4
def fig4():
    K = json.load(open(f"{ROOT}/outputs/lee22_normalization_forensics_20260928/D_gas_fractions/konietzka_implied_fe_shells.json"))
    ksh = [s for s in K["shells"] if s["z_hi"] <= 2.0001]
    # rebin the 0.1-wide Konietzka shells into 0.2-wide bins (inverse-variance mean) for legibility
    kz, kf, ke = [], [], []
    for a in range(0, len(ksh), 2):
        g = ksh[a:a + 2]
        w = np.array([1 / s["fe_tng_implied_err"] ** 2 for s in g])
        f = np.array([s["fe_tng_implied"] for s in g])
        kz.append(np.mean([0.5 * (s["z_lo"] + s["z_hi"]) for s in g]))
        kf.append(np.sum(w * f) / np.sum(w))
        ke.append(np.std(f) / np.sqrt(len(g)) if len(g) > 1 else g[0]["fe_tng_implied_err"])
    kz, kf, ke = map(np.array, (kz, kf, ke))
    fig, axes = plt.subplots(1, 3, figsize=(18.5, 7.6), sharey=True)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.74, bottom=0.12, wspace=0.07)
    zc = S["cam_trend_z"]
    nums = {}
    for j, (title, clo, flo) in enumerate(COLS2):
        ax = axes[j]
        zaxis(ax)
        ax.set_yscale("log")
        ax.axhline(1, color="0.6", lw=1.0, ls="-", zorder=1)
        dr = fl_rows(FLD, flo)
        zf = np.array([float(r["z"]) for r in dr])
        fe_f = np.array([float(r["f_e_FL"]) for r in dr])
        ee_f = np.array([float(r["f_e_FL_err"]) for r in dr])
        if clo is not None:
            pre = f"cam_trend_m{clo:g}_"
            ok = np.isfinite(S[pre + "fgas"]) & (S[pre + "nh"] >= 3)
            zz = zc[ok]
            ax.plot(1 + zz, S[pre + "fe_printed"][ok], color=LIT, ls=":", lw=2.6)
            ax.plot(1 + zz, S[pre + "fe_prod"][ok], color=PAINT, lw=2.4)
            ax.plot(1 + zz, S[pre + "fe_tngcal"][ok], color=OTHER, lw=2.4, ls="-.")
            Fn = fit_const(zz, S[pre + "fgas"][ok] / S[pre + "fe_printed"][ok], 3)
            ax.plot(1 + zz, S[pre + "fe_printed"][ok] * (1 + zz) ** 3 / Fn, color=CORR, ls="--", lw=2.4)
            ax.plot(1 + zz, S[pre + "fgas_B16"][ok], color=B12C, lw=2.4)
            ax.errorbar(1 + zz, S[pre + "fgas"][ok], S[pre + "fgas_err"][ok], fmt="o", ms=7, color=DATA, elinewidth=1.2, zorder=5)
            nums[f"col{j}_TNG_fgas"] = dict(zip([f"{z:.2f}" for z in zz[::6]], map(float, S[pre + "fgas"][ok][::6])))
            nums[f"col{j}_Lee22_printed_fe"] = dict(zip([f"{z:.2f}" for z in zz[::6]], map(float, S[pre + "fe_printed"][ok][::6])))
            nums[f"col{j}_Fn_TNG"] = Fn
            ax.text(0.04, 0.05, f"green: $F_n$ = {Fn:.1f} (fit to TNG)", transform=ax.transAxes,
                    fontsize=12.5, color=CORR)
        else:
            ax.plot(1 + zf, [float(r["f_e_Lee22_literal"]) for r in dr], color=LIT, ls=":", lw=2.6)
            ax.plot(1 + zf, [float(r["f_e_Lee22_prod"]) for r in dr], color=PAINT, lw=2.4)
            ax.plot(1 + zf, [float(r["f_e_Lee22_tngcal"]) for r in dr], color=OTHER, lw=2.4, ls="-.")
            lit = np.array([float(r["f_e_Lee22_literal"]) for r in dr])
            Fn = fit_const(zf, fe_f / lit, 3)
            ax.plot(1 + zf, lit * (1 + zf) ** 3 / Fn, color=CORR, ls="--", lw=2.4)
            ax.plot(1 + zf, [float(r["f_gas_B16"]) for r in dr], color=B12C, lw=2.4)
            nums[f"col{j}_Fn_FL"] = Fn
            ax.text(0.04, 0.05, f"green: $F_n$ = {Fn:.1f} (fit to FLAMINGO)", transform=ax.transAxes,
                    fontsize=12.5, color=CORR)
            ax.text(0.96, 0.95, "no TNG (CAMELS) haloes", transform=ax.transAxes, ha="right", va="top", fontsize=12, color="0.35")
        ax.errorbar(1 + zf, fe_f, ee_f, fmt="s", ms=8, mfc="white", mec=DATA, mew=1.6, ecolor=DATA, zorder=5)
        nums[f"col{j}_FL_fe"] = dict(zip([f"{z:g}" for z in zf], map(float, fe_f)))
        if j == 0:
            ax.errorbar(1 + kz, kf, ke, fmt="^", ms=9, mfc="white", mec=DATA, mew=1.6, ecolor=DATA, zorder=6)
            nums["Konietzka_fe_rebinned"] = dict(zip([f"{z:.2f}" for z in kz], map(float, kf)))
            ax.set_ylabel("$f_e(<R_{200c})$  [units of $f_b M_{200c}$, fully ionised]")
        ax.set_title(title, fontsize=16, pad=8)
        ax.set_xlabel("z")
        ax.set_ylim(0.1, 6)
        log_ticks(ax, "y", [0.1, 0.2, 0.5, 1, 2, 5], ["0.1", "0.2", "0.5", "1", "2", "5"])
    h = [Line2D([], [], color=DATA, marker="o", ls="none", ms=7),
         Line2D([], [], color=DATA, marker="s", mfc="white", ls="none", ms=8, mew=1.6),
         Line2D([], [], color=DATA, marker="^", mfc="white", ls="none", ms=9, mew=1.6),
         Line2D([], [], color=B12C), Line2D([], [], color=LIT, ls=":"), Line2D([], [], color=PAINT),
         Line2D([], [], color=OTHER, ls="-."), Line2D([], [], color=CORR, ls="--")]
    lab = ["TNG model (CAMELS CV): $f_{gas}$, all gas", "FLAMINGO L2p8_m9: electrons ($n_e$=0 in star-forming gas)",
           "TNG300, implied by Konietzka halo DM ($10^{13}$-$10^{14}M_\\odot$, HMF-weighted)",
           "Battaglia+16 gas (full ionisation)", "Lee22 density as printed (Table 8)", "Lee22 $\\times$0.6016 (production DM)",
           r"Lee22 $\times C(z)$ (TNG-DM calibrated)", r"Lee22 $\times(1+z)^3/F_n$ (M1 + constant M2)"]
    fig.legend(h, lab, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.53, 0.995), fontsize=12.5,
               handlelength=2.4, columnspacing=1.5)
    NUMBERS["fig4"] = nums
    save(fig, "fig_tsz4_density")


if __name__ == "__main__":
    which = sys.argv[1:] or ["1", "2", "3", "4"]
    for w in which:
        {"1": fig1, "2": fig2, "3": fig3, "4": fig4}[w]()
    fn = f"{OUT}/fig_numbers.json"
    try:
        old = json.load(open(fn))
    except (OSError, ValueError):
        old = {}
    old.update(NUMBERS)
    json.dump(old, open(fn, "w"), indent=1)
