#!/usr/bin/env python3
"""Replicas of Lee et al. 2022 (arXiv:2205.01710v1) Fig. 2 (electron pressure) and Fig. 5 (electron density) with the
HalfDome implementations (2026-10-01). Same layout as the paper: two mass bins, rows x^3 A/A200, Delta A/A and
Delta A/sigma, with Delta = (A_sim - A_model)/A_sim and sigma the paper's error of the mean (red bars); both definitions
are reproduced exactly by extract_lee22_fig2_fig5_full.py.

z = 0 (fig_lee22_fig{2,5}_replica_z0): the simulation is the paper's own TNG profile, digitized from the vector figure
(lowest-concentration quartile, 52 radial bins with +-sigma and +-scatter bars). Units are physical, n200 = 200 rho_cr f_b
/(X_H m_p) and P200 = 200 G M200 rho_cr f_b/(2 R200) evaluated physically; the density figure's TNG curve is converted from
Lee22's code units with the derived h^3 X_H^2 (= 0.1795 at z = 0, the (1+z)^3 factor is 1), the pressure needs no
conversion (F_P = 1, the (1+z)^2 factor is 1). Models at the bin-centre mass, like the paper's fit curves:
  Battaglia (XGPaint, as painted in HalfDome): B12 P_e = 0.5176 P_th, B16 n_e = 0.9 rho_gas/m_per_e;
  Battaglia as drawn in Lee22 (digitized): B12 thermal pressure, B16 rho_fit/(200 f_b) with exponent -(beta-gamma)/alpha;
  Lee22 through unedited XGPaint: density x 0.6016 (ne2d), pressure as printed (previous production);
  Lee22 corrected (final): density x [h_TNG (1+z)]^3 X_H^2, pressure x (1+z)^2;
  CAMELS-TNG (27 CV boxes, all halos in the bin, independent of Lee22's code): open circles.
z > 0 (fig_lee22_fig{2,5}_replica_camels_zgt0): the paper shows no z > 0 profiles, so the simulation is CAMELS-TNG at
z = 0.47 and 1.05 in the paper's lower mass bin (the upper bin holds <= 4 CAMELS halos above z = 0.4); models evaluated at
each halo's M200c and z and averaged over the same halos; sigma = delete-one-box jackknife, scatter = halo std.
Run from the repository root with the halfdome env (LD_LIBRARY_PATH=<env>/lib)."""
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

FMG = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(FMG / "lee22_normalization_forensics/C_pressure_consistency"))
import lee22_models as lm  # noqa: E402

DOC = FMG / "outputs/lee22_corrected_20260930/doc"
DIG = DOC / "lee22_fig2_fig5_full_digitization.npz"
CURVES = FMG / "outputs/lee22_normalization_forensics_20260928/C_pressure_consistency/lee22_fig2_fig5_vector_curves.csv"
CAM = FMG / "outputs/lee22_tsz_verification_20260929/D_tng_camels/halo_tables"
H_TNG, X_H, XGP = 0.6774, 0.76, 0.6016316602183194
BINS = ((13.2, 13.4), (13.8, 14.0))     # log10 M200c / (h^-1 Msun), the paper's panels
C = dict(sim="#d62728", bar="#0000ff", bat="#17becf", xgp="#ff7f0e", cor="#9467bd", noz="#7f7f7f", cam="#000000")
RC = {"font.family": "DejaVu Sans", "font.size": 15, "axes.labelsize": 17, "xtick.labelsize": 14, "ytick.labelsize": 14,
      "legend.fontsize": 12.2, "xtick.direction": "out", "ytick.direction": "out", "savefig.facecolor": "white",
      "pdf.fonttype": 42}
XLIM = (0.0296, 1.79)                   # the paper's axis range


def digitized(q, col):
    D = np.load(str(DIG))
    k = "{}_{}".format(q, col)
    return D[k + "_x"], D[k + "_sim"], D[k + "_sigma"], D[k + "_scatter"]


def paper_curve(q, col, curve):
    with open(str(CURVES)) as fh:
        rows = [r for r in csv.DictReader(fh) if r["quantity"] == ("x3_Pe_over_P200" if q == "P" else "x3_ne_over_n200")
                and r["panel"] == str(col) and r["curve"] == curve]
    x = np.array([float(r["x_r_over_R200"]) for r in rows]); v = np.array([float(r["value"]) for r in rows])
    o = np.argsort(x)
    return x[o], v[o]


def models(q, x, m_msun, z):
    """Model profiles A/A200 (physical) at mass m_msun [Msun] and redshift z; Lee22 masses in h^-1 Msun (h_TNG)."""
    mh = m_msun * H_TNG
    if q == "P":
        fit = lm.lee_Pbar(x, mh, z)
        return dict(bat=lm.b12_Pe_over_P200(x, m_msun, z), xgp=fit, cor=fit * (1 + z) ** 2)
    fit = lm.lee_nbar(x, mh, z)
    return dict(bat=lm.b16_nbar(x, m_msun, z), xgp=fit * XGP, cor=fit * (H_TNG * (1 + z)) ** 3 * X_H ** 2,
                noz=fit * H_TNG ** 3 * X_H ** 2)


def camels(q, z, lo, hi, per_halo_models=False):
    """CAMELS-TNG stack in a bin of log10 M200c/(h^-1 Msun): mean, jackknife sigma, halo scatter (+ halo-averaged models)."""
    P = dict(np.load(str(CAM / "precomputed_halos.npz"))); D = dict(np.load(str(CAM / "direct_halos.npz")))
    xg = P["xg"]
    zd = np.unique(np.round(D["z"], 3)); zd = zd[np.argmin(np.abs(zd - z))]
    d = (np.abs(D["z"] - zd) < 0.01) & (D["M200"] >= 1e13)       # group-mass electron fraction (see fig_paper_profiles.py)
    fxe = np.nansum(D["nbar_xe"][d], 0) / np.nansum(D["nbar_fullion"][d], 0)
    s = (np.abs(P["z"] - z) < 0.01) & (P["M200"] >= 10 ** lo / H_TNG) & (P["M200"] < 10 ** hi / H_TNG)
    prof = P["nbar"][s] * fxe[None, :] if q == "n" else P["Pbar"][s]
    boxes = P["box"][s]; ub = np.unique(boxes)
    mean = np.nanmean(prof, 0)
    jk = np.array([np.nanmean(prof[boxes != b], 0) for b in ub])
    sigma = np.sqrt((len(ub) - 1) / len(ub) * np.nansum((jk - jk.mean(0)) ** 2, 0))
    out = dict(x=xg, mean=mean, sigma=sigma, scatter=np.nanstd(prof, 0), n=int(s.sum()), nbox=len(ub),
               z=float(np.mean(P["z"][s])), fxe_z=float(zd), nfxe=int(d.sum()))
    if per_halo_models:
        ms = [models(q, xg, m, zz) for m, zz in zip(P["M200"][s], P["z"][s])]
        out["models"] = {k: np.mean([mm[k] for mm in ms], 0) for k in ms[0]}
    return out


def style(ax, row, ylab=None):
    ax.set_xscale("log"); ax.set_xlim(*XLIM)
    ax.tick_params(which="both", top=False, right=False)
    if ylab:
        ax.set_ylabel(ylab)
    if row < 2:
        ax.tick_params(labelbottom=False)


def residual_rows(axes, x_ref, ref, sig, curves, lims, ticks, cam=None):
    """Delta A/A and Delta A/sigma, Delta = (A_sim - A_model)/A_sim; curves: [(x, A, kwargs, short label)]. Values outside
    the row's range are not drawn; each off-scale curve is listed inside the row with its extreme value."""
    for row, ax in ((1, axes[1]), (2, axes[2])):
        ax.axhline(0, color="k", lw=1.2, zorder=1)
        lo, hi = lims[row - 1]
        notes = {"lo": [], "hi": []}
        for x, a, kw, short in curves:
            r = np.exp(np.interp(np.log(x), np.log(x_ref), np.log(ref)))
            s = np.exp(np.interp(np.log(x), np.log(x_ref), np.log(sig)))
            d = (r - a) / (r if row == 1 else s)
            ok = (x >= x_ref.min() * 0.999) & (x <= x_ref.max() * 1.001)
            ax.plot(x[ok], np.where((d[ok] >= lo) & (d[ok] <= hi), d[ok], np.nan), **kw)
            if (ok & (d < lo)).any():
                notes["lo"].append((short, d[ok].min(), kw["color"]))
            if (ok & (d > hi)).any():
                notes["hi"].append((short, d[ok].max(), kw["color"]))
        if cam is not None:
            k = np.isfinite(cam["mean"]) & (cam["x"] >= x_ref.min()) & (cam["x"] <= x_ref.max())
            r = np.exp(np.interp(np.log(cam["x"][k]), np.log(x_ref), np.log(ref)))
            s = np.exp(np.interp(np.log(cam["x"][k]), np.log(x_ref), np.log(sig)))
            d = (r - cam["mean"][k]) / (r if row == 1 else s)
            inside = (d >= lo) & (d <= hi)
            ax.plot(cam["x"][k][inside], d[inside], "o", ms=5.5, mfc="white", mec=C["cam"], mew=1.3, zorder=6)
            if (d < lo).any():
                notes["lo"].append(("CAMELS", d.min(), C["cam"]))
            if (d > hi).any():
                notes["hi"].append(("CAMELS", d.max(), C["cam"]))
        for side, items in notes.items():
            for i, (short, v, col) in enumerate(items):
                fmt = "{} {} {:+.1f}" if row == 1 else "{} {} {:+.0f}"
                ax.text(.012 + .2 * i, .1 if side == "lo" else .9, fmt.format("\u25bc" if side == "lo" else "\u25b2", short, v),
                        transform=ax.transAxes, ha="left", va="center", fontsize=11.5, color=col,
                        bbox=dict(facecolor="white", edgecolor="none", pad=1.0, alpha=.85))
        ax.set_ylim(lo, hi); ax.set_yticks(ticks[row - 1])


def figure_z0(q):
    lab = {"P": (r"P$_e$/P$_{200}$ (r/R$_{200}$)$^3$", r"$\Delta$ P$_e$/P$_e$", r"$\Delta$ P$_e$/$\sigma$"),
           "n": (r"n$_e$/n$_{200}$ (r/R$_{200}$)$^3$", r"$\Delta$ n$_e$/n$_e$", r"$\Delta$ n$_e$/$\sigma$")}[q]
    conv = 1.0 if q == "P" else H_TNG ** 3 * X_H ** 2
    lims = {"P": ((-1.0, 1.0), (-30, 30)), "n": ((-1.0, 1.0), (-30, 30))}[q]
    ticks = ((-0.5, 0, 0.5), (-20, 0, 20))
    short = dict(bat="B12" if q == "P" else "B16", xgp="unedited" if q == "P" else "\u00d70.6016", cor="corrected")
    report = []
    with plt.rc_context(RC):
        fig = plt.figure(figsize=(17.5, 9.8))
        gs = fig.add_gridspec(3, 2, height_ratios=(3, 1, 1), hspace=0, wspace=0, left=.075, right=.995, top=.865, bottom=.075)
        for col, (lo, hi) in enumerate(BINS):
            axes = [fig.add_subplot(gs[r, col]) for r in range(3)]
            for a in axes[1:]:
                a.sharex(axes[0])
            xs, vs, sg, sc = digitized(q, col)
            xt, vt = paper_curve(q, col, "simulation")
            ax = axes[0]
            ax.errorbar(xs, xs ** 0 * vs * conv, yerr=sc * conv, fmt="none", ecolor=C["bar"], alpha=.5, lw=1.5, zorder=2)
            ax.errorbar(xs, vs * conv, yerr=sg * conv, fmt="none", ecolor=C["sim"], alpha=.5, lw=1.5, zorder=3)
            ax.plot(xt, vt * conv, color=C["sim"], lw=5, zorder=4, solid_capstyle="butt")
            if q == "n":
                ax.plot(xt, vt, color=C["sim"], lw=1.6, ls=(0, (1.2, 1.6)), alpha=.8, zorder=4)
            xb, vb = paper_curve(q, col, "battaglia")
            ax.plot(xb, vb, color=C["bat"], lw=1.6, ls=(0, (1.2, 1.6)), alpha=.9, zorder=3)
            x = np.geomspace(xt.min(), xt.max(), 200)
            mh = 10 ** (0.5 * (lo + hi)); m = mh / H_TNG
            M = models(q, x, m, 0.0)
            styles = dict(bat=dict(color=C["bat"], lw=2.2, ls="-."), xgp=dict(color=C["xgp"], lw=2.4, ls=(0, (5, 3))),
                          cor=dict(color=C["cor"], lw=2.6, ls="-"))
            for k in ("bat", "cor", "xgp"):
                ax.plot(x, x ** 3 * M[k], zorder=5, **styles[k])
            cam = camels(q, 0.0, lo, hi)
            k = np.isfinite(cam["mean"]) & (cam["x"] >= XLIM[0]) & (cam["x"] <= XLIM[1])
            ax.errorbar(cam["x"][k], cam["x"][k] ** 3 * cam["mean"][k], yerr=cam["x"][k] ** 3 * cam["sigma"][k], fmt="o", ms=5.5,
                        mfc="white", mec=C["cam"], ecolor=C["cam"], mew=1.3, lw=1.1, capsize=2, zorder=6)
            ax.set_yscale("log")
            ax.text(.04, .93, r"$10^{%.1f}-10^{%.1f}\ h^{-1}M_\odot$" % (lo, hi), transform=ax.transAxes, fontsize=19, va="top")
            ax.text(.04, .80, "z = 0;  CAMELS: %d halos, %d boxes" % (cam["n"], cam["nbox"]), transform=ax.transAxes, fontsize=12.5,
                    va="top", color=".3")
            residual_rows(axes, xt, vt * conv, np.exp(np.interp(np.log(xt), np.log(xs), np.log(sg))) * conv,
                          [(x, x ** 3 * M[kk], styles[kk], short[kk]) for kk in ("bat", "cor", "xgp")], lims, ticks,
                          cam=dict(cam, mean=cam["x"] ** 3 * cam["mean"]))
            for r, a in enumerate(axes):
                style(a, r, lab[r] if col == 0 else None)
                if col == 1:
                    a.tick_params(labelleft=False)
            axes[2].set_xlabel(r"r/R$_{200}$")
            # numbers for the text: ratio model / converted simulation at 0.3 < x < 1
            sel = (xt > .3) & (xt < 1)
            for kk in ("bat", "xgp", "cor"):
                mm = models(q, xt[sel], m, 0.0)[kk] * xt[sel] ** 3
                report.append((q, col, kk, float(np.exp(np.mean(np.log(mm / (vt[sel] * conv)))))))
            cs = k & (cam["x"] > .3) & (cam["x"] < 1)
            r_cam = np.exp(np.mean(np.log(cam["x"][cs] ** 3 * cam["mean"][cs] / np.exp(np.interp(np.log(cam["x"][cs]), np.log(xt), np.log(vt * conv))))))
            report.append((q, col, "camels", float(r_cam)))
            if col == 1:
                ylo, yhi = axes[0].get_ylim()
            axes_top = axes[0]
            if col == 0:
                left_top = axes_top
        # common y range of the two top panels
        lims_top = (min(left_top.get_ylim()[0], axes_top.get_ylim()[0]), max(left_top.get_ylim()[1], axes_top.get_ylim()[1]))
        for a in (left_top, axes_top):
            a.set_ylim(lims_top[0], lims_top[1] * (2.0 if q == "P" else 2.6))
        if q == "P":
            h = [Line2D([], [], color=C["sim"], lw=5, label="Simulation (TNG300, Lee22 Fig. 2, digitized)"),
                 Line2D([], [], color=C["bat"], lw=2.2, ls="-.", label="Battaglia et al. (2012), XGPaint $P_e$ (HalfDome)"),
                 Line2D([], [], color=C["bat"], lw=1.6, ls=(0, (1.2, 1.6)), label="Battaglia (2012) as drawn in Lee22 ($P_{th}$)"),
                 Line2D([], [], color=C["xgp"], lw=2.4, ls=(0, (5, 3)), label="Lee22 via unedited XGPaint (as printed)"),
                 Line2D([], [], color=C["cor"], lw=2.6, label=r"Lee22 corrected, $\times(1+z)^2$ (= 1 at z = 0)"),
                 Line2D([], [], color=C["cam"], marker="o", mfc="white", ls="none", label="CAMELS-TNG (all halos, independent)")]
        else:
            h = [Line2D([], [], color=C["sim"], lw=5, label=r"Simulation (Lee22 Fig. 5) $\times h^3X_H^2$ (code units $\to$ physical)"),
                 Line2D([], [], color=C["sim"], lw=1.6, ls=(0, (1.2, 1.6)), label="Simulation as drawn in Lee22 (code units)"),
                 Line2D([], [], color=C["bat"], lw=2.2, ls="-.", label="Battaglia (2016), XGPaint $n_e$ (HalfDome)"),
                 Line2D([], [], color=C["bat"], lw=1.6, ls=(0, (1.2, 1.6)), label="Battaglia (2016) as drawn in Lee22"),
                 Line2D([], [], color=C["xgp"], lw=2.4, ls=(0, (5, 3)), label=r"Lee22 via unedited XGPaint ($\times$0.6016)"),
                 Line2D([], [], color=C["cor"], lw=2.6, label=r"Lee22 corrected, $\times[h(1+z)]^3X_H^2$ (= 0.1795 at z = 0)"),
                 Line2D([], [], color=C["cam"], marker="o", mfc="white", ls="none", label="CAMELS-TNG (all halos, independent)")]
        fig.legend(handles=h, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(.5, 1.0),
                   handlelength=3.2, columnspacing=1.6, fontsize=12.8)
        DOC.mkdir(parents=True, exist_ok=True)
        name = "fig_lee22_fig%d_replica_z0" % (2 if q == "P" else 5)
        for ext in ("png", "pdf"):
            fig.savefig(str(DOC / (name + "." + ext)), dpi=150 if ext == "png" else None)
        plt.close(fig)
    return report


def figure_camels(q, zs=(0.465, 1.048), mbin=BINS[0]):
    lab = {"P": (r"P$_e$/P$_{200}$ (r/R$_{200}$)$^3$", r"$\Delta$ P$_e$/P$_e$", r"$\Delta$ P$_e$/$\sigma$"),
           "n": (r"n$_e$/n$_{200}$ (r/R$_{200}$)$^3$", r"$\Delta$ n$_e$/n$_e$", r"$\Delta$ n$_e$/$\sigma$")}[q]
    lims = ((-1.0, 1.0), (-15, 15))
    ticks = ((-0.5, 0, 0.5), (-10, 0, 10))
    short = dict(bat="B12" if q == "P" else "B16", xgp="unedited" if q == "P" else "\u00d70.6016", cor="corrected",
                 noz="no (1+z)\u00b3")
    lo, hi = mbin
    report = []
    with plt.rc_context(RC):
        fig = plt.figure(figsize=(17.5, 9.4))
        gs = fig.add_gridspec(3, 2, height_ratios=(3, 1, 1), hspace=0, wspace=0, left=.075, right=.995, top=.9, bottom=.08)
        tops = []
        for col, z in enumerate(zs):
            axes = [fig.add_subplot(gs[r, col]) for r in range(3)]
            for a in axes[1:]:
                a.sharex(axes[0])
            cam = camels(q, z, lo, hi, per_halo_models=True)
            k = np.isfinite(cam["mean"]) & (cam["mean"] > 0) & (cam["x"] >= .03) & (cam["x"] <= 1.6)
            xc, A = cam["x"][k], cam["x"][k] ** 3 * cam["mean"][k]
            sg, sc = cam["x"][k] ** 3 * cam["sigma"][k], cam["x"][k] ** 3 * cam["scatter"][k]
            ax = axes[0]
            ax.errorbar(xc, A, yerr=sc, fmt="none", ecolor=C["bar"], alpha=.5, lw=1.5, zorder=2)
            ax.errorbar(xc, A, yerr=sg, fmt="none", ecolor=C["sim"], alpha=.5, lw=1.5, zorder=3)
            ax.plot(xc, A, color=C["sim"], lw=5, zorder=4)
            styles = dict(bat=dict(color=C["bat"], lw=2.2, ls="-."), xgp=dict(color=C["xgp"], lw=2.4, ls=(0, (5, 3))),
                          cor=dict(color=C["cor"], lw=2.6, ls="-"), noz=dict(color=C["noz"], lw=1.8, ls=(0, (1.2, 1.6))))
            keys = ("bat", "xgp", "cor") + (("noz",) if q == "n" else ())
            Mx = {kk: xc ** 3 * cam["models"][kk][k] for kk in keys}
            for kk in keys:
                ax.plot(xc, Mx[kk], zorder=5, **styles[kk])
            ax.set_yscale("log")
            ax.text(.04, .93, r"$10^{%.1f}-10^{%.1f}\ h^{-1}M_\odot$,  z = %.2f" % (lo, hi, cam["z"]), transform=ax.transAxes,
                    fontsize=19, va="top")
            ax.text(.04, .80, "simulation: CAMELS-TNG, %d halos in %d boxes" % (cam["n"], cam["nbox"]), transform=ax.transAxes,
                    fontsize=12.5, va="top", color=".3")
            residual_rows(axes, xc, A, sg, [(xc, Mx[kk], styles[kk], short[kk]) for kk in keys], lims, ticks)
            for r, a in enumerate(axes):
                style(a, r, lab[r] if col == 0 else None)
                if col == 1:
                    a.tick_params(labelleft=False)
            axes[2].set_xlabel(r"r/R$_{200}$")
            tops.append(ax)
            sel = (xc > .3) & (xc < 1)
            for kk in keys:
                report.append((q, round(cam["z"], 3), kk, float(np.exp(np.mean(np.log(Mx[kk][sel] / A[sel]))))))
        yl = (min(a.get_ylim()[0] for a in tops), max(a.get_ylim()[1] for a in tops))
        for a in tops:
            a.set_ylim(yl[0], yl[1] * 2.2)
        if q == "P":
            h = [Line2D([], [], color=C["sim"], lw=5, label="Simulation: CAMELS-TNG (27 CV boxes)"),
                 Line2D([], [], color=C["bat"], lw=2.2, ls="-.", label="Battaglia et al. (2012), XGPaint $P_e$ (HalfDome)"),
                 Line2D([], [], color=C["xgp"], lw=2.4, ls=(0, (5, 3)), label="Lee22 via unedited XGPaint (as printed)"),
                 Line2D([], [], color=C["cor"], lw=2.6, label=r"Lee22 corrected, $\times(1+z)^2$")]
        else:
            h = [Line2D([], [], color=C["sim"], lw=5, label="Simulation: CAMELS-TNG (27 CV boxes)"),
                 Line2D([], [], color=C["bat"], lw=2.2, ls="-.", label="Battaglia (2016), XGPaint $n_e$ (HalfDome)"),
                 Line2D([], [], color=C["xgp"], lw=2.4, ls=(0, (5, 3)), label=r"Lee22 via unedited XGPaint ($\times$0.6016)"),
                 Line2D([], [], color=C["cor"], lw=2.6, label=r"Lee22 corrected, $\times[h(1+z)]^3X_H^2$"),
                 Line2D([], [], color=C["noz"], lw=1.8, ls=(0, (1.2, 1.6)), label=r"$\times h^3X_H^2$ only (without $(1+z)^3$)")]
        fig.legend(handles=h, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(.5, 1.0), handlelength=3.2,
                   columnspacing=1.6, fontsize=12.8)
        name = "fig_lee22_fig%d_replica_camels_zgt0" % (2 if q == "P" else 5)
        for ext in ("png", "pdf"):
            fig.savefig(str(DOC / (name + "." + ext)), dpi=150 if ext == "png" else None)
        plt.close(fig)
    return report


def main():
    # the thick simulation line (103 points) and the errorbar line (52 bins) are the same profile
    for q in ("P", "n"):
        for col in (0, 1):
            xs, vs, _, _ = digitized(q, col); xt, vt = paper_curve(q, col, "simulation")
            k = (xs >= xt.min()) & (xs <= xt.max())
            print("%s panel %d: 52-bin errorbar line / thick line: max |ratio-1| = %.4f" % (
                q, col, np.max(np.abs(vs[k] / np.exp(np.interp(np.log(xs[k]), np.log(xt), np.log(vt))) - 1))))
    rows = figure_z0("P") + figure_z0("n") + figure_camels("P") + figure_camels("n")
    names = dict(bat="Battaglia (XGPaint)", xgp="Lee22 unedited XGPaint", cor="Lee22 corrected", noz="x h^3 X_H^2 only",
                 camels="CAMELS-TNG")
    print("model / simulation, geometric mean over 0.3 < x < 1 (z = 0: paper TNG in physical units; z > 0: CAMELS):")
    for q, c, k, r in rows:
        print("  {} {:>6} {:24s} {:.3f}".format(q, c, names[k], r))
    with open(str(DOC / "fig_lee22_fig2_fig5_replica_ratios.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["quantity", "panel_or_z", "model", "model_over_simulation_x0p3_1"]); w.writerows(rows)


if __name__ == "__main__":
    main()
