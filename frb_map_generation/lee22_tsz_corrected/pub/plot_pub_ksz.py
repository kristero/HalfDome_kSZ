#!/usr/bin/env python3
"""Publication figures for the halo kSZ (2026-10-02).

  pub_ksz_power        D_ell of the HalfDome halo kSZ maps (Battaglia16, Lee22 XGPaint pipeline, Lee22 corrected;
                       paint_ksz_variants.jl, 4 R200c sphere, catalogue velocities) against the WebSky halo-only kSZ
                       (Stein et al. 2020, Fig. 6, digitized), with the ratio to WebSky
  pub_ksz_mass_split   catalogue one-halo kSZ power split by halo mass (poisson_mass_split.jl), Battaglia16 vs Lee22 corrected

Spectra: healpy anafast of the NSIDE 4096 maps (lmax 8192, point-sampled maps, no pixel-window correction), cached in
outputs/lee22_ksz_20261002/spectra/ksz_spectra.npz. D_ell in uK^2 with T_CMB = 2.7255 K. Run from the repository root with the
halfdome env (LD_LIBRARY_PATH=<env>/lib)."""
import pickle
import sys
from pathlib import Path

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

KSZ = HERE.parents[1] / "outputs/lee22_ksz_20261002"
DIGITIZED = Path("/tmp/claude-0/-home-cbllover-HalfDome/015746ae-94b7-42cb-8dc5-0e8c33dc8490/scratchpad/stein2020_fig6_digitized.pkl")
WEBSKY_CSV = KSZ / "websky_halo_stein2020_fig6_vector_binned.csv"   # exact PDF vectors (ksz_debug/websky_def), ell 49-5764
T2 = 2.7255e6 ** 2
NAMES = (("b16", "Battaglia16", MOD[0][3], MOD[0][4]), ("lee22pipe", "Lee22 XGPaint pipeline", MOD[1][3], MOD[1][4]),
         ("lee22corr", "Lee22 corrected", MOD[2][3], MOD[2][4]))
LMIN, LMAX = 100, 8000


def websky():
    """WebSky halo-only kSZ D_ell [uK^2], Stein+20 Fig. 6 (pixel-colour digitization), log-binned to a clean curve."""
    if not WEBSKY_CSV.exists():
        x, y = (np.asarray(v, float) for v in pickle.load(open(DIGITIZED, "rb"))["websky_halo"])
        edges = np.geomspace(x.min(), x.max() * 1.0001, 40)
        idx = np.digitize(x, edges) - 1
        rows = [(np.exp(np.mean(np.log(x[idx == i]))), np.median(y[idx == i])) for i in range(len(edges) - 1) if np.any(idx == i)]
        WEBSKY_CSV.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(WEBSKY_CSV, np.array(rows), delimiter=",", header="ell,D_ell_uK2  (Stein et al. 2020 Fig. 6, WebSky halo, digitized)")
    d = np.loadtxt(WEBSKY_CSV, delimiter=",")
    return d[:, 0], d[:, 1]


def spectra():
    path = KSZ / "spectra/ksz_spectra.npz"
    if not path.exists():
        import healpy as hp
        out = {}
        for key, *_ in NAMES:
            m = hp.read_map(str(KSZ / "maps" / ("halfdome_ksz_%s_sphere4p0_nside4096.fits" % key)), dtype=np.float64)
            out["cl_" + key] = hp.anafast(m, lmax=8192, iter=1)
            del m
        out["ell"] = np.arange(8193)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **out)
    return dict(np.load(str(path)))


def band(ell, dl, edges):
    return np.array([dl[(ell >= lo) & (ell < hi)].mean() for lo, hi in zip(edges[:-1], edges[1:])])


def fig_power(S, wx, wy, captions):
    edges = np.unique(np.round(np.geomspace(LMIN, LMAX, 25)).astype(int)); cen = np.sqrt(edges[:-1] * edges[1:])
    ell = S["ell"].astype(float)
    with plt.rc_context(RC):
        fig = plt.figure(figsize=(3.45, 4.1))
        gs = fig.add_gridspec(2, 1, height_ratios=(2.4, 1), hspace=0.0, left=0.17, right=0.97, bottom=0.11, top=0.98)
        ax = fig.add_subplot(gs[0]); rx = fig.add_subplot(gs[1], sharex=ax)
        ax.plot(wx, wy, color="k", lw=1.6, zorder=5, label="WebSky halo")
        rx.axhline(1, color="k", lw=0.8)
        inw = (cen >= wx.min()) & (cen <= wx.max())
        captions.append("pub_ksz_power: D_ell [uK^2] at ell = 300 / 1000 / 3000 / 5000, and / WebSky halo at ell = 1000 / 3000 / 5000")
        for key, lab, col, ls in NAMES:
            d = band(ell, ell * (ell + 1) * S["cl_" + key] / (2 * np.pi) * T2, edges)
            ax.plot(cen, d, color=col, ls=ls, lw=1.2, zorder=4, label=lab)
            rx.plot(cen[inw], d[inw] / np.interp(np.log(cen[inw]), np.log(wx), wy), color=col, ls=ls, lw=1.2)
            at = lambda L: float(np.interp(np.log(L), np.log(cen), d))
            captions.append("  %-24s %s   | %s" % (lab, " / ".join("%.3f" % at(L) for L in (300, 1000, 3000, 5000)),
                                                    " / ".join("%.2f" % (at(L) / np.interp(np.log(L), np.log(wx), wy)) for L in (1000, 3000, 5000))))
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(LMIN, LMAX)
        ax.set_ylabel(r"$\ell(\ell+1)\,C_\ell^{\mathrm{kSZ}}/2\pi$  [$\mu$K$^2$]")
        ax.tick_params(labelbottom=False); ax.set_ylim(0.04, 2.2)
        ax.set_yticks([0.05, 0.1, 0.2, 0.5, 1, 2]); ax.set_yticklabels(["0.05", "0.1", "0.2", "0.5", "1", "2"])
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.legend(loc="lower right", frameon=False, handlelength=2.6, borderaxespad=0.6)
        rx.set_yscale("log"); rx.set_ylim(0.3, 12); rx.set_yticks([0.5, 1, 2, 5, 10]); rx.set_yticklabels(["0.5", "1", "2", "5", "10"])
        rx.yaxis.set_minor_formatter(NullFormatter())
        rx.set_ylabel("/ WebSky"); rx.set_xlabel(r"$\ell$")
        save(fig, "pub_ksz_power")


def fig_mass(S, wx, wy, captions):
    import h5py
    with h5py.File(KSZ / "ksz_poisson_mass_split.h5", "r") as f:
        ell = f["ell"][:]; mb = f["log10_mass_bin_edges_msun"][:]
        parts = {k: f["cl_1h_" + k][:] for k in ("b16", "lee22corr")}         # (mass bin, ell): HDF5.jl writes column-major
    labels = [r"$<10^{13}$", r"$10^{13}$–$10^{13.5}$", r"$10^{13.5}$–$10^{14}$", r"$10^{14}$–$10^{14.5}$", r"$10^{14.5}$–$10^{15}$", r"$>10^{15}$"]
    assert len(labels) == len(mb) - 1
    colors = plt.get_cmap("Blues")(np.linspace(0.18, 0.95, len(labels)))
    k = (ell >= LMIN) & (ell <= LMAX); pre = ell * (ell + 1) / (2 * np.pi) * T2
    em = S["ell"].astype(float)
    with plt.rc_context(RC):
        fig = plt.figure(figsize=(7.1, 4.4))
        gs = fig.add_gridspec(2, 2, height_ratios=(2.1, 1), hspace=0.0, wspace=0.08, left=0.085, right=0.995, bottom=0.1, top=0.84)
        ymax = 1.1 * max((parts[n].sum(0) * pre)[k].max() for n in parts)
        for j, (key, name) in enumerate((("b16", "Battaglia16"), ("lee22corr", "Lee22 corrected"))):
            ax = fig.add_subplot(gs[0, j]); fx = fig.add_subplot(gs[1, j], sharex=ax)
            p = parts[key]
            ax.stackplot(ell[k], p[:, k] * pre[k], colors=colors, edgecolor="white", linewidth=0.4, zorder=2)
            ax.plot(wx, wy, color="k", lw=1.2, ls=(0, (4, 2)), zorder=4)
            fx.stackplot(ell[k], p[:, k] / p[:, k].sum(0), colors=colors, edgecolor="white", linewidth=0.4)
            ax.set_xscale("log"); ax.set_xlim(LMIN, LMAX); ax.set_ylim(0, ymax); ax.tick_params(labelbottom=False)
            ax.set_yticks(np.arange(0.2, ymax, 0.2))
            ax.text(0.04, 0.93, name, transform=ax.transAxes, va="top")
            fx.set_ylim(0, 1); fx.set_yticks([0, 0.5, 1]); fx.set_xlabel(r"$\ell$"); fx.xaxis.set_minor_formatter(NullFormatter())
            if j == 0:
                ax.set_ylabel(r"$\ell(\ell+1)\,C_\ell^{\mathrm{kSZ}}/2\pi$  [$\mu$K$^2$]"); fx.set_ylabel("fraction")
            else:
                ax.tick_params(labelleft=False); fx.tick_params(labelleft=False)
            dmap = em * (em + 1) * S["cl_" + key] / (2 * np.pi) * T2
            for L in (300, 1000, 3000):
                tot = np.interp(np.log(L), np.log(ell), np.log(p.sum(0))); tot = np.exp(tot)
                fr = [np.exp(np.interp(np.log(L), np.log(ell), np.log(np.maximum(p[b], 1e-300)))) / tot for b in range(len(labels))]
                mapv = dmap[int(0.9 * L):int(1.1 * L) + 1].mean(); poi = L * (L + 1) / (2 * np.pi) * T2 * tot
                captions.append("%-16s ell = %4d: map / one-halo %.2f; fractions %s" % (
                    name, L, mapv / poi, "  ".join("%s %.2f" % (lb.replace("$", "").replace("^", ""), x) for lb, x in zip(labels, fr))))
        h = [Patch(facecolor=c, edgecolor="none", label=l) for c, l in zip(colors, labels)]
        h.append(Line2D([], [], color="k", lw=1.2, ls=(0, (4, 2)), label="WebSky halo"))
        fig.legend(handles=h, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.54, 1.0), handlelength=1.8,
                   columnspacing=1.3, title=r"$M_{200c}\ [M_\odot]$", title_fontsize=8.5)
        save(fig, "pub_ksz_mass_split")


def main():
    wx, wy = websky()
    S = spectra()
    captions = []
    fig_power(S, wx, wy, captions)
    fig_mass(S, wx, wy, captions)
    (OUT / "captions_ksz.txt").write_text("\n".join(captions) + "\n")
    print("\n".join(captions))


if __name__ == "__main__":
    main()
