#!/usr/bin/env python3
"""Publication figures for the halo kSZ (2026-10-02), from the paint_ksz_websky.jl maps (HalfDome cosmology Omega_m 0.3089,
h 0.6774; exact profiles for z < 0.1; HalfDome N-body velocities) painted on idark:

  pub_ksz_websky             like-for-like comparison with the WebSky halo kSZ (Stein et al. 2020, Fig. 6): HalfDome painted with
                             WebSky's recipe (Battaglia16, Lee22 corrected), with the ratio to WebSky and two closure tests from
                             the same one-halo engine (WebSky's own halos; HalfDome halos with WebSky's velocities)
  pub_ksz_power              physical halo kSZ (gas inside 4 R200c, catalogue M200c, all halos): Battaglia16, Lee22 XGPaint
                             pipeline, Lee22 corrected, against the WebSky halo curve
  pub_ksz_mass_split         catalogue one-halo of the physical maps split by halo mass (Battaglia16, Lee22 corrected)
  pub_ksz_websky_mass_split  the same for the WebSky-recipe maps (mass M200c_ws, compensated)

Spectra: healpy anafast of the NSIDE 4096 maps (lmax 8192, point-sampled, no pixel window), ksz_websky_spectra.py on idark.
One-halo terms: poisson_mass_split_websky.jl on idark. D_ell in uK^2 with T_CMB = 2.7255 K. Run from the repository root with the
halfdome env (LD_LIBRARY_PATH=<env>/lib)."""
import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import NullFormatter
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from plot_pub_figures import MOD, RC, OUT, save  # noqa: E402

FMG = HERE.parents[1] / "outputs"
KSZ = FMG / "lee22_ksz_websky_20261002"
WEBSKY_CSV = FMG / "lee22_ksz_20261002/websky_halo_stein2020_fig6_vector_binned.csv"   # exact PDF vectors, ell 49-5764
T2 = 2.7255e6 ** 2
B16C, B16LS = MOD[0][3], MOD[0][4]
PIPEC, PIPELS = MOD[1][3], MOD[1][4]
CORRC, CORRLS = MOD[2][3], MOD[2][4]
PHYS = (("b16_phys", "Battaglia16", B16C, B16LS), ("lee22pipe_phys", "Lee22 XGPaint pipeline", PIPEC, PIPELS),
        ("lee22corr_phys", "Lee22 corrected", CORRC, CORRLS))
WSR = (("b16_websky", "Battaglia16", B16C, B16LS), ("lee22corr_websky", "Lee22 corrected", CORRC, CORRLS))
LMIN, LMAX = 100, 8000
LCLOSE = 300            # lowest ell of the one-halo closure curves in pub_ksz_websky
MASS_LABELS = [r"$<10^{13}$", r"$10^{13}$–$10^{13.5}$", r"$10^{13.5}$–$10^{14}$", r"$10^{14}$–$10^{14.5}$", r"$10^{14.5}$–$10^{15}$",
               r"$>10^{15}$"]


def websky():
    d = np.loadtxt(WEBSKY_CSV, delimiter=",")
    return d[:, 0], d[:, 1]


def spectra():
    return dict(np.load(str(KSZ / "spectra/ksz_websky_spectra.npz")))


def split(name="ksz_websky_mass_split.h5"):
    """{key: (mass bin, ell) C_ell} plus ell and mass edges (HDF5.jl writes column-major: h5py sees (bin, ell))."""
    with h5py.File(KSZ / name, "r") as f:
        return {k: f[k][:] for k in f}


def band(ell, dl, edges):
    return np.array([dl[(ell >= lo) & (ell < hi)].mean() for lo, hi in zip(edges[:-1], edges[1:])])


def dl_of(ell, cl):
    return ell * (ell + 1) * cl / (2 * np.pi) * T2


def at(x, y, L):
    return float(np.exp(np.interp(np.log(L), np.log(x), np.log(y))))


def setup_power_axes(fig):
    gs = fig.add_gridspec(2, 1, height_ratios=(2.4, 1.15), hspace=0.0, left=0.17, right=0.97, bottom=0.1, top=0.98)
    ax = fig.add_subplot(gs[0]); rx = fig.add_subplot(gs[1], sharex=ax)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(LMIN, LMAX)
    ax.set_ylabel(r"$\ell(\ell+1)\,C_\ell^{\mathrm{kSZ}}/2\pi$  [$\mu$K$^2$]"); ax.tick_params(labelbottom=False)
    ax.yaxis.set_minor_formatter(NullFormatter())
    rx.set_ylabel("/ WebSky"); rx.set_xlabel(r"$\ell$"); rx.axhline(1, color="k", lw=0.8)
    return ax, rx


def fig_websky(S, M, W, wx, wy, captions):
    edges = np.unique(np.round(np.geomspace(LMIN, LMAX, 25)).astype(int)); cen = np.sqrt(edges[:-1] * edges[1:])
    ell = S["ell"].astype(float); e1 = M["ell"]; inw = (cen >= wx.min()) & (cen <= wx.max())
    wsi = lambda x: np.exp(np.interp(np.log(x), np.log(wx), np.log(wy)))
    Ls = (300, 1000, 3000, 5000)
    captions.append("pub_ksz_websky: D_ell [uK^2] at ell = %s; / WebSky halo (Stein+20 Fig. 6) at the same ell" % " / ".join(map(str, Ls)))
    captions.append("  %-40s %s" % ("WebSky halo", " / ".join("%.4f" % wsi(L) for L in Ls)))
    with plt.rc_context(RC):
        fig = plt.figure(figsize=(3.45, 4.3))
        ax, rx = setup_power_axes(fig)
        ax.plot(wx, wy, color="k", lw=1.6, zorder=5, label="WebSky halo")
        for key, lab, col, ls in WSR:
            d = band(ell, dl_of(ell, S["cl_" + key]), edges)
            ax.plot(cen, d, color=col, ls=ls, lw=1.2, zorder=4, label=lab)
            rx.plot(cen[inw], d[inw] / wsi(cen[inw]), color=col, ls=ls, lw=1.2)
            oh = dl_of(e1, M["cl_1h_" + key].sum(0))
            captions.append("  %-40s %s | %s | map/one-halo %s" % (
                "HalfDome, WebSky recipe, " + lab, " / ".join("%.4f" % at(cen, d, L) for L in Ls),
                " / ".join("%.3f" % (at(cen, d, L) / wsi(L)) for L in Ls), " / ".join("%.3f" % (at(cen, d, L) / at(e1, oh, L)) for L in Ls)))
        # closure tests (one-halo, same engine): HalfDome halos with WebSky's per-cell velocities; WebSky's own halos
        # from ell = 300: WebSky's catalogue histogram has dz = 0.025 bins without centroids, which biases the nearest halos
        kin = (e1 >= wx.min()) & (e1 <= wx.max()) & (e1 >= LCLOSE)
        for key, lab, col in (("b16_websky", "Battaglia16", B16C),):
            vel = dl_of(e1, M["cl_1h_%s_wscellvel" % key].sum(0)); wsc = dl_of(e1, W["cl_1h_" + key].sum(0))
            rx.plot(e1[kin], vel[kin] / wsi(e1[kin]), color=col, ls=":", lw=1.3)
            rx.plot(e1[kin], wsc[kin] / wsi(e1[kin]), color="0.45", ls=(0, (6, 2)), lw=1.0)
            smooth = dl_of(e1, M["cl_1h_%s_wsvel" % key].sum(0))
            for nm, y in (("one-halo, HalfDome halos + WebSky velocities", vel), ("  (smooth velocity-ratio fit instead)", smooth),
                          ("one-halo, WebSky halos (same engine)", wsc)):
                captions.append("  %-40s %s | %s" % (nm + ", " + lab, " / ".join("%.4f" % at(e1, y, L) for L in Ls),
                                                    " / ".join("%.3f" % (at(e1, y, L) / wsi(L)) for L in Ls)))
        key = "lee22corr_websky"; wsc = dl_of(e1, W["cl_1h_" + key].sum(0)); vel = dl_of(e1, M["cl_1h_%s_wscellvel" % key].sum(0))
        for nm, y in (("one-halo, HalfDome halos + WebSky velocities", vel), ("one-halo, WebSky halos (same engine)", wsc)):
            captions.append("  %-40s %s | %s" % (nm + ", Lee22 corrected", " / ".join("%.4f" % at(e1, y, L) for L in Ls),
                                                " / ".join("%.3f" % (at(e1, y, L) / wsi(L)) for L in Ls)))
        ax.set_ylim(0.009, 2.0); ax.set_yticks([0.01, 0.03, 0.1, 0.3, 1]); ax.set_yticklabels(["0.01", "0.03", "0.1", "0.3", "1"])
        ax.legend(loc="lower right", frameon=False, handlelength=2.6, borderaxespad=0.6)
        rx.set_ylim(0.7, 2.05); rx.set_yticks([0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]); rx.set_yticklabels(["0.8", "1", "1.2", "1.4", "1.6", "1.8", "2"])
        h = [Line2D([], [], color=B16C, ls=":", lw=1.3, label="WebSky velocities"),
             Line2D([], [], color="0.45", ls=(0, (6, 2)), lw=1.0, label="WebSky halos")]
        rx.legend(handles=h, loc="upper left", frameon=False, handlelength=2.4, fontsize=7.5, ncol=2, columnspacing=1.0, borderaxespad=0.3)
        save(fig, "pub_ksz_websky")


def fig_power(S, wx, wy, captions):
    edges = np.unique(np.round(np.geomspace(LMIN, LMAX, 25)).astype(int)); cen = np.sqrt(edges[:-1] * edges[1:])
    ell = S["ell"].astype(float); inw = (cen >= wx.min()) & (cen <= wx.max())
    wsi = lambda x: np.exp(np.interp(np.log(x), np.log(wx), np.log(wy)))
    captions.append("pub_ksz_power: D_ell [uK^2] at ell = 300 / 1000 / 3000 / 5000, and / WebSky halo at ell = 1000 / 3000 / 5000")
    with plt.rc_context(RC):
        fig = plt.figure(figsize=(3.45, 4.1))
        ax, rx = setup_power_axes(fig)
        ax.plot(wx, wy, color="k", lw=1.6, zorder=5, label="WebSky halo")
        for key, lab, col, ls in PHYS:
            d = band(ell, dl_of(ell, S["cl_" + key]), edges)
            ax.plot(cen, d, color=col, ls=ls, lw=1.2, zorder=4, label=lab)
            rx.plot(cen[inw], d[inw] / wsi(cen[inw]), color=col, ls=ls, lw=1.2)
            captions.append("  %-24s %s   | %s" % (lab, " / ".join("%.3f" % at(cen, d, L) for L in (300, 1000, 3000, 5000)),
                                                    " / ".join("%.2f" % (at(cen, d, L) / wsi(L)) for L in (1000, 3000, 5000))))
        ax.set_ylim(0.04, 2.2)
        ax.set_yticks([0.05, 0.1, 0.2, 0.5, 1, 2]); ax.set_yticklabels(["0.05", "0.1", "0.2", "0.5", "1", "2"])
        ax.legend(loc="lower right", frameon=False, handlelength=2.6, borderaxespad=0.6)
        rx.set_yscale("log"); rx.set_ylim(0.3, 12); rx.set_yticks([0.5, 1, 2, 5, 10]); rx.set_yticklabels(["0.5", "1", "2", "5", "10"])
        rx.yaxis.set_minor_formatter(NullFormatter())
        save(fig, "pub_ksz_power")


def fig_mass(S, M, wx, wy, captions, variants, name):
    ell = M["ell"]; mb = M["log10_mass_bin_edges_msun"]
    assert len(MASS_LABELS) == len(mb) - 1
    parts = {k: M["cl_1h_" + k] for k, _ in variants}
    colors = plt.get_cmap("Blues")(np.linspace(0.18, 0.95, len(MASS_LABELS)))
    k = (ell >= LMIN) & (ell <= LMAX); pre = ell * (ell + 1) / (2 * np.pi) * T2
    em = S["ell"].astype(float)
    with plt.rc_context(RC):
        fig = plt.figure(figsize=(7.1, 4.4))
        gs = fig.add_gridspec(2, 2, height_ratios=(2.1, 1), hspace=0.0, wspace=0.08, left=0.085, right=0.995, bottom=0.1, top=0.84)
        ymax = 1.1 * max((parts[n].sum(0) * pre)[k].max() for n in parts)
        for j, (key, title) in enumerate(variants):
            ax = fig.add_subplot(gs[0, j]); fx = fig.add_subplot(gs[1, j], sharex=ax)
            p = parts[key]
            ax.stackplot(ell[k], p[:, k] * pre[k], colors=colors, edgecolor="white", linewidth=0.4, zorder=2)
            ax.plot(wx, wy, color="k", lw=1.2, ls=(0, (4, 2)), zorder=4)
            fx.stackplot(ell[k], p[:, k] / p[:, k].sum(0), colors=colors, edgecolor="white", linewidth=0.4)
            ax.set_xscale("log"); ax.set_xlim(LMIN, LMAX); ax.set_ylim(0, ymax); ax.tick_params(labelbottom=False)
            ax.set_yticks(np.arange(0.2, ymax, 0.2))
            ax.text(0.04, 0.93, title, transform=ax.transAxes, va="top")
            fx.set_ylim(0, 1); fx.set_yticks([0, 0.5, 1]); fx.set_xlabel(r"$\ell$"); fx.xaxis.set_minor_formatter(NullFormatter())
            if j == 0:
                ax.set_ylabel(r"$\ell(\ell+1)\,C_\ell^{\mathrm{kSZ}}/2\pi$  [$\mu$K$^2$]"); fx.set_ylabel("fraction")
            else:
                ax.tick_params(labelleft=False); fx.tick_params(labelleft=False)
            dmap = dl_of(em, S["cl_" + key])
            for L in (300, 1000, 3000):
                tot = at(ell, p.sum(0), L)
                fr = [at(ell, np.maximum(p[b], 1e-300), L) / tot for b in range(len(MASS_LABELS))]
                mapv = dmap[int(0.9 * L):int(1.1 * L) + 1].mean(); poi = L * (L + 1) / (2 * np.pi) * T2 * tot
                captions.append("%-26s ell = %4d: one-halo %.4f uK^2, map / one-halo %.3f; fractions %s" % (
                    name + " " + title, L, poi, mapv / poi,
                    "  ".join("%s %.2f" % (lb.replace("$", "").replace("^", ""), x) for lb, x in zip(MASS_LABELS, fr))))
        h = [Patch(facecolor=c, edgecolor="none", label=l) for c, l in zip(colors, MASS_LABELS)]
        h.append(Line2D([], [], color="k", lw=1.2, ls=(0, (4, 2)), label="WebSky halo"))
        mlab = r"$M_{200c}\ [M_\odot]$"
        fig.legend(handles=h, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.54, 1.0), handlelength=1.8,
                   columnspacing=1.3, title=mlab, title_fontsize=8.5)
        save(fig, name)


def main():
    wx, wy = websky()
    S = spectra(); M = split(); W = split("ksz_websky_mass_split_wscat.h5")
    captions = []
    fig_websky(S, M, W, wx, wy, captions)
    fig_power(S, wx, wy, captions)
    fig_mass(S, M, wx, wy, captions, (("b16_phys", "Battaglia16"), ("lee22corr_phys", "Lee22 corrected")), "pub_ksz_mass_split")
    fig_mass(S, M, wx, wy, captions, (("b16_websky", "Battaglia16"), ("lee22corr_websky", "Lee22 corrected")), "pub_ksz_websky_mass_split")
    (OUT / "captions_ksz.txt").write_text("\n".join(captions) + "\n")
    print("\n".join(captions))


if __name__ == "__main__":
    main()
