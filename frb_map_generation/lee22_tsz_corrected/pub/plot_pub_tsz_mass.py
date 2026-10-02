#!/usr/bin/env python3
"""Publication figure: Compton-y power of HalfDome split by halo mass, Battaglia12 vs Lee22 corrected (2026-10-02).

Linear y axis (stacked areas), shared by both panels. One-halo power from the exact catalogue-weighted halo model of the 2026-09-29 verification (workstream B, CatGrid): the (z, M200c)
histogram of all 85.2M painted HalfDome halos times the painted y_ell profiles (projected aperture as in the maps), which reproduces
the maps to 1-3 % at ell = 30-3000. One-halo power is additive in mass, so each band is that mass range's exact share; the two-halo
term (not additive) is <= 6 % (B12) and <= 2 % (Lee22) of the total at ell >= 300 and is not shown. Lee22 corrected = Table 7 pressure
x (1+z)^2. FLAMINGO L2p8_m9 lightcone 0 (divided by pixwin^2) for scale. Catalogue masses are Msun/h; bins are in physical Msun
(h = 0.68). Writes outputs/lee22_pub_figures_20261001/pub_tsz_mass_split.{pdf,png} and the band fractions to captions_tsz_mass.txt.
Run from the repository root with the halfdome env (LD_LIBRARY_PATH=<env>/lib)."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter
import numpy as np

HERE = Path(__file__).resolve().parent
FMG = HERE.parents[1]
sys.path.insert(0, str(FMG / "lee22_tsz_verification/B_class_sz")); sys.path.insert(0, str(HERE))
import common as C  # noqa: E402
import halo_model as HM  # noqa: E402
from analysis import CatGrid  # noqa: E402
from plot_pub_figures import RC, OUT, save  # noqa: E402

H = 0.68
EDGES = [C.CAT_M_MIN / H, 1e13, 10 ** 13.5, 1e14, 10 ** 14.5, 1e15, C.CAT_M_MAX / H * 1.01]   # physical Msun
LABELS = [r"$<10^{13}$", r"$10^{13}$–$10^{13.5}$", r"$10^{13.5}$–$10^{14}$", r"$10^{14}$–$10^{14.5}$", r"$10^{14.5}$–$10^{15}$",
          r"$>10^{15}$"]
COLORS = plt.get_cmap("Blues")(np.linspace(0.18, 0.95, len(LABELS)))   # one hue, light (low mass) to dark (high mass)
LMIN, LMAX = 100, 8000


def split(g, y=None):
    parts = np.array([g.cl(m_lo=lo * H, m_hi=hi * H, y=y)["cl_1h"] for lo, hi in zip(EDGES[:-1], EDGES[1:])])
    tot = g.cl(y=y)
    assert np.allclose(parts.sum(0), tot["cl_1h"], rtol=1e-10)
    return parts, tot


def main():
    ing = HM.Ingredients()
    gb, gl = CatGrid(ing, "b12", "aperture"), CatGrid(ing, "lee22", "aperture")
    ell = gl.ell
    k = (ell >= LMIN) & (ell <= LMAX)
    pre = ell * (ell + 1) / (2 * np.pi)
    models = (("Battaglia12", *split(gb)), ("Lee22 corrected", *split(gl, gl.y * ((1 + gl.z) ** 2)[:, None, None])))
    S = np.load(str(FMG / "outputs/lee22_tsz_verification_20260929/synthesis/synthesis_inputs.npz"))
    es = S["ell"].astype(float); ke = (es >= LMIN) & (es <= LMAX)
    fl = es * (es + 1) * S["cl_flamingo_lc0"] / (2 * np.pi)
    fls = np.convolve(fl, np.ones(25) / 25, mode="same")
    lines = []
    with plt.rc_context(RC):
        fig = plt.figure(figsize=(7.1, 4.4))
        gs = fig.add_gridspec(2, 2, height_ratios=(2.1, 1), hspace=0.0, wspace=0.08, left=0.085, right=0.995, bottom=0.1, top=0.84)
        for j, (name, parts, tot) in enumerate(models):
            ax = fig.add_subplot(gs[0, j]); fx = fig.add_subplot(gs[1, j], sharex=ax)
            dl = parts[:, k] * pre[k] * 1e12
            ax.stackplot(ell[k], dl, colors=COLORS, edgecolor="white", linewidth=0.4, zorder=2)
            ax.plot(es[ke], 1e12 * fls[ke], color="k", lw=1.2, ls=(0, (4, 2)), zorder=4)
            frac = parts[:, k] / tot["cl_1h"][k]
            fx.stackplot(ell[k], frac, colors=COLORS, edgecolor="white", linewidth=0.4)
            ax.set_xscale("log"); ax.set_xlim(LMIN, LMAX); ax.set_ylim(0, 4.2)        # linear: stacked band heights = shares
            ax.set_yticks([1, 2, 3, 4])
            ax.tick_params(labelbottom=False)
            ax.text(0.04, 0.93, name, transform=ax.transAxes, va="top")
            fx.set_ylim(0, 1); fx.set_yticks([0, 0.5, 1]); fx.set_xlabel(r"$\ell$")
            fx.xaxis.set_minor_formatter(NullFormatter())
            if j == 0:
                ax.set_ylabel(r"$10^{12}\,\ell(\ell+1)\,C_\ell^{yy}/2\pi$"); fx.set_ylabel("fraction")
            else:
                ax.tick_params(labelleft=False); fx.tick_params(labelleft=False)
            for L in (300, 1000, 3000):
                f = [np.interp(L, ell, p) / np.interp(L, ell, tot["cl_1h"]) for p in parts]
                lines.append("%-16s ell = %4d: " % (name, L) + "  ".join("%s %.2f" % (l.replace("$", "").replace("^", ""), x)
                                                                       for l, x in zip(LABELS, f)))
            lines.append("%-16s 2h/total at ell = 300 / 1000 / 3000: %s" % (name, " / ".join(
                "%.3f" % (np.interp(L, ell, tot["cl_2h"]) / np.interp(L, ell, tot["cl_1h"] + tot["cl_2h"])) for L in (300, 1000, 3000))))
        h = [Patch(facecolor=c, edgecolor="none", label=l) for c, l in zip(COLORS, LABELS)]
        h.append(Line2D([], [], color="k", lw=1.2, ls=(0, (4, 2)), label="FLAMINGO"))
        fig.legend(handles=h, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.54, 1.0), handlelength=1.8,
                   columnspacing=1.3, title=r"$M_{200c}\ [M_\odot]$", title_fontsize=8.5)
        save(fig, "pub_tsz_mass_split")
    (OUT / "captions_tsz_mass.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
