#!/usr/bin/env python3
"""Battaglia, the Lee22 XGPaint pipeline and Lee22 corrected, electron pressure and electron density at three masses and three
redshifts (2026-10-01), drawn like the Lee22 Fig. 2 / Fig. 5 replicas (fig_lee22_fig2_fig5_replica.py, whose model function is reused).
Rows: M200c = 2e13, 1e14, 5e14 Msun (10^13.13, 10^13.83, 10^14.53 h^-1 Msun with h_TNG = 0.6774). Lee22 fitted bins from 1e13 to about
10^14 h^-1 Msun at z <= 0.2 and lower at higher z, so the 5e14 row (and 1e14 at z >= 1) is extrapolated; columns: z = 0, 1, 2 (Lee22 fitted
z <= 2). Models (physical units, n200 = 200 rho_cr f_b/(X_H m_p), P200 = 200 G M200 rho_cr f_b/(2 R200)):
  Battaglia (XGPaint, as painted in HalfDome): B12 P_e = 0.5176 P_th, B16 n_e = 0.9 rho_gas/m_per_e;
  Lee22 XGPaint pipeline: density x 0.6016 (ne2d), pressure as printed (previous production);
  Lee22 corrected (final): density x [h_TNG (1+z)]^3 X_H^2, pressure x (1+z)^2.
Lower panels: model / Lee22 corrected. No simulation is shown: TNG profiles exist only for the paper's z = 0 bins (Fig. 1a/1b) and
CAMELS-TNG has few halos above 1e14 Msun or z = 1 (Fig. 2b/2c).
Writes outputs/lee22_corrected_20260930/doc/fig_lee22_models_grid_{pressure,density}.{png,pdf}. Run from the repository root with
the halfdome env (LD_LIBRARY_PATH=<env>/lib)."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, NullFormatter
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fig_lee22_fig2_fig5_replica as R  # noqa: E402

MASSES = (2e13, 1e14, 5e14)            # M200c [Msun]
ZS = (0.0, 1.0, 2.0)
X = np.geomspace(R.XLIM[0], R.XLIM[1], 300)
FIT_X = (0.04, 1.34)                   # radial range of the Lee22 fits
STY = dict(bat=dict(color=R.C["bat"], lw=2.2, ls="-."), xgp=dict(color=R.C["xgp"], lw=2.4, ls=(0, (5, 3))),
           cor=dict(color=R.C["cor"], lw=2.6, ls="-"))


def mass_label(m):
    e = int(np.floor(np.log10(m))); a = m / 10 ** e
    lead = "" if abs(a - 1) < 1e-9 else r"%g\times" % a
    return r"$M_{200c} = %s10^{%d}\,M_\odot$  ($10^{%.2f}\,h^{-1}M_\odot$)" % (lead, e, np.log10(m * R.H_TNG))


def figure(q):
    ylab = {"P": r"P$_e$/P$_{200}$ (r/R$_{200}$)$^3$", "n": r"n$_e$/n$_{200}$ (r/R$_{200}$)$^3$"}[q]
    rows = []
    with plt.rc_context(dict(R.RC, **{"font.size": 13, "axes.labelsize": 14, "xtick.labelsize": 12, "ytick.labelsize": 12})):
        fig = plt.figure(figsize=(17.5, 15.5))
        outer = fig.add_gridspec(3, 3, hspace=.14, wspace=.04, left=.07, right=.995, top=.905, bottom=.05)
        tops, bots = [], []
        for i, m in enumerate(MASSES):
            for j, z in enumerate(ZS):
                g = outer[i, j].subgridspec(2, 1, height_ratios=(2.6, 1), hspace=0)
                top = fig.add_subplot(g[0]); bot = fig.add_subplot(g[1], sharex=top)
                M = R.models(q, X, m, z)
                for k in ("bat", "xgp", "cor"):
                    zo = {"bat": 4, "cor": 5, "xgp": 6}[k]   # dashes on top: at z = 0 the printed pressure equals the corrected one
                    top.plot(X, X ** 3 * M[k], zorder=zo, **STY[k])
                    bot.plot(X, M[k] / M["cor"], zorder=zo, **STY[k])
                sel = (X > .3) & (X < 1)
                gm = {k: float(np.exp(np.mean(np.log(M[k][sel] / M["cor"][sel])))) for k in ("bat", "xgp")}
                rows.append((q, m, z, gm["bat"], gm["xgp"]))
                for ax in (top, bot):
                    ax.set_xscale("log"); ax.set_xlim(*R.XLIM)
                    ax.axvspan(R.XLIM[0], FIT_X[0], color=".5", alpha=.08, lw=0)
                    ax.axvspan(FIT_X[1], R.XLIM[1], color=".5", alpha=.08, lw=0)
                top.set_yscale("log"); top.tick_params(labelbottom=False)
                bot.set_yscale("log"); bot.set_ylim(.06, 16); bot.axhline(1, color="k", lw=1.1, zorder=1)
                bot.yaxis.set_major_locator(FixedLocator([.1, 1, 10])); bot.yaxis.set_minor_formatter(NullFormatter())
                bot.set_yticklabels(["0.1", "1", "10"])
                top.text(.04, .94, mass_label(m), transform=top.transAxes, fontsize=12.2, va="top")
                note = "at 0.3 < x < 1, / corrected:\nB%s %.2f,  %s %.2f" % ("12" if q == "P" else "16", gm["bat"],
                                                                     "pipeline", gm["xgp"])
                top.text(.97, .05, note, transform=top.transAxes, fontsize=11, ha="right", va="bottom", color=".25",
                         linespacing=1.3, bbox=dict(facecolor="white", edgecolor="none", pad=1.0, alpha=.85))
                if i == 0:
                    top.set_title("z = %g" % z, fontsize=16, pad=8)
                if j == 0:
                    top.set_ylabel(ylab); bot.set_ylabel("model /\ncorrected")
                else:
                    top.tick_params(labelleft=False); bot.tick_params(labelleft=False)
                if i == 2:
                    bot.set_xlabel(r"r/R$_{200}$")
                else:
                    bot.tick_params(labelbottom=False)
                tops.append(top); bots.append(bot)
        lo = min(a.get_ylim()[0] for a in tops); hi = max(a.get_ylim()[1] for a in tops)
        for a in tops:
            a.set_ylim(max(lo, 3e-6), hi * 3)
        if q == "P":
            h = [Line2D([], [], label="Battaglia et al. (2012), XGPaint $P_e$ (HalfDome)", **STY["bat"]),
                 Line2D([], [], label="Lee22 XGPaint pipeline", **STY["xgp"]),
                 Line2D([], [], label=r"Lee22 corrected, $\times(1+z)^2$", **STY["cor"])]
            sub = (r"Electron pressure.  Pipeline / corrected = $(1+z)^{-2}$ = 1, 0.25, 0.11 at z = 0, 1, 2.  "
                   r"Shaded: outside the radial range of the Lee22 fits (0.04–1.34 $R_{200c}$).")
        else:
            h = [Line2D([], [], label="Battaglia (2016), XGPaint $n_e$ (HalfDome)", **STY["bat"]),
                 Line2D([], [], label="Lee22 XGPaint pipeline", **STY["xgp"]),
                 Line2D([], [], label=r"Lee22 corrected, $\times[h(1+z)]^3X_H^2$", **STY["cor"])]
            sub = (r"Electron density.  Pipeline / corrected = 0.6016 / $[0.6774(1+z)]^3X_H^2$ = 3.35, 0.42, 0.12 at z = 0, 1, 2.  "
                   r"Shaded: outside the radial range of the Lee22 fits (0.04–1.34 $R_{200c}$).")
        fig.legend(handles=h, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(.5, 1.0), handlelength=3.2,
                   columnspacing=2.0, fontsize=13.5)
        fig.text(.5, .945, sub, ha="center", va="center", fontsize=12.5, color=".3")
        name = "fig_lee22_models_grid_" + ("pressure" if q == "P" else "density")
        for ext in ("png", "pdf"):
            fig.savefig(str(R.DOC / (name + "." + ext)), dpi=140 if ext == "png" else None)
        plt.close(fig)
    return rows


def main():
    rows = figure("P") + figure("n")
    print("geometric mean over 0.3 < x < 1 of model / Lee22 corrected:")
    for q, m, z, b, x in rows:
        print("  %s  M200c=%.0e  z=%g   Battaglia %.3f   Lee22 XGPaint pipeline %.3f" % (q, m, z, b, x))


if __name__ == "__main__":
    main()
