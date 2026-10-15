#!/usr/bin/env python3
"""Truncation-radius series figures: HalfDome Compton-y maps with the spherical truncation at N = 1..5 R200c
(painted on idark), against CLASS-SZ Battaglia12 with x_outSZ = N and the WebSky public y map."""
import argparse, csv, json
from pathlib import Path
import numpy as np, healpy as hp
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

RADII = (1, 2, 3, 4, 5)
COL = {1: "#cde2fb", 2: "#86b6ef", 3: "#3987e5", 4: "#1c5cab", 5: "#0d366b"}   # one hue, light -> dark with N
COL = {1: "#9ec5f4", 2: "#5598e7", 3: "#256abf", 4: "#184f95", 5: "#0d366b"}
PROJ_COL = "#eb6834"; WEBSKY_COL = "#1baf7a"; CS_COL = "#52514e"

def log_bins(lmin, lmax, per_decade):
    return np.unique(np.round(np.logspace(np.log10(lmin), np.log10(lmax), int(per_decade * np.log10(lmax / lmin)) + 1)).astype(int))

def bin_dl(ell, dl, edges):
    out, centres = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (ell >= lo) & (ell < hi); w = 2 * ell[sel] + 1.0
        out.append(np.sum(w * dl[sel]) / np.sum(w)); centres.append(np.exp(np.sum(w * np.log(ell[sel])) / np.sum(w)))
    return np.array(centres), np.array(out)

def interp_loglog(x_th, y_th, x):
    good = y_th > 0
    return np.exp(np.interp(np.log(x), np.log(x_th[good]), np.log(y_th[good])))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-cl", required=True); ap.add_argument("--provenance", default=None)
    ap.add_argument("--class-sz-dir", default="class_sz"); ap.add_argument("--websky", default="websky_tsz_cl.npz")
    ap.add_argument("--out-dir", default="figures"); ap.add_argument("--nside", type=int, default=4096); ap.add_argument("--lmin", type=int, default=30)
    a = ap.parse_args(); out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    rows = list(csv.reader(open(a.raw_cl))); names = rows[0][1:]; data = np.array(rows[1:], float)
    ell = data[:, 0]; lmax = int(ell[-1]); pw = hp.pixwin(a.nside, lmax=lmax)
    cl = {n: data[:, i + 1] / pw ** 2 for i, n in enumerate(names)}
    dlf = ell * (ell + 1) / (2 * np.pi); edges = log_bins(a.lmin, lmax, 20); b = {}
    for n in names: lc, b[n] = bin_dl(ell[2:], (dlf * cl[n])[2:], edges)
    cs = {}
    for N in RADII:
        d = np.load(Path(a.class_sz_dir) / f"class_sz_tsz_xout{N}.npz")
        cs[N] = {k: bin_dl(ell[2:], interp_loglog(d["ell"], d[k], ell[2:]), edges)[1] for k in ("dl_total", "dl_1h", "dl_2h")}
    w = np.load(a.websky); wl = w["ell"].astype(float); wcl = w["cl_pixwin_corrected"]; wl_dl = wl * (wl + 1) * wcl / (2 * np.pi)
    _, b["websky"] = bin_dl(wl[2:lmax + 1], wl_dl[2:lmax + 1], edges)
    S = 1e12
    plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.6, "axes.edgecolor": "#c3c2b7",
                         "axes.linewidth": 0.8, "xtick.direction": "in", "ytick.direction": "in", "legend.frameon": False})
    # ---------------- figure 1: spectra + ratios
    fig, ax = plt.subplots(4, 1, figsize=(8.6, 14), sharex=True, gridspec_kw={"height_ratios": [3.2, 1.4, 1.4, 1.4], "hspace": 0.06})
    for N in RADII:
        ax[0].plot(lc, S * b[f"y_sphere{N}"], color=COL[N], lw=2, label=f"HalfDome, gas inside {N} $R_{{200c}}$ (sphere)")
    ax[0].plot(lc, S * b["y_projected4"], color=PROJ_COL, lw=1.6, ls="--", label="HalfDome, projected 4 $R_{200c}$ (XGPaint/WebSky convention)")
    for N in RADII:
        ax[0].plot(lc, S * cs[N]["dl_total"], color=COL[N], lw=1.0, ls=":", alpha=0.9, label="CLASS-SZ 1h+2h, $x_{\\rm out}$ = N" if N == 5 else None)
    ax[0].plot(lc, S * b["websky"], color=WEBSKY_COL, lw=2, ls="-.", label="WebSky public $y$ map (Stein+2020, 4 $R_{200c}$)")
    ax[0].set_yscale("log"); ax[0].set_ylabel(r"$10^{12}\,\ell(\ell+1)C_\ell^{yy}/2\pi$")
    ax[0].set_title("tSZ auto-spectrum vs truncation radius: HalfDome Battaglia12, NSIDE 4096, spherical truncation", fontsize=11, loc="left")
    ax[0].legend(fontsize=8.6, loc="lower right", ncol=1)
    for N in RADII: ax[1].plot(lc, b[f"y_sphere{N}"] / b["y_sphere5"], color=COL[N], lw=2, label=f"N = {N}")
    ax[1].plot(lc, b["y_projected4"] / b["y_sphere5"], color=PROJ_COL, lw=1.6, ls="--", label="projected 4")
    ax[1].axhline(1, color="#9a9891", lw=0.8); ax[1].set_ylabel("map(N) / map(N = 5)"); ax[1].legend(fontsize=8.6, ncol=3, loc="lower right")
    for N in RADII: ax[2].plot(lc, b[f"y_sphere{N}"] / cs[N]["dl_total"], color=COL[N], lw=2, label=f"N = {N}")
    ax[2].plot(lc, b["y_projected4"] / cs[4]["dl_total"], color=PROJ_COL, lw=1.6, ls="--", label="projected 4 / CLASS-SZ($x_{\\rm out}$=4)")
    ax[2].axhline(1, color="#9a9891", lw=0.8); ax[2].set_ylabel("map(N) / CLASS-SZ($x_{\\rm out}$ = N)"); ax[2].legend(fontsize=8.6, ncol=3, loc="best")
    for N in RADII: ax[3].plot(lc, b[f"y_sphere{N}"] / b["websky"], color=COL[N], lw=2, label=f"N = {N}")
    ax[3].plot(lc, b["y_projected4"] / b["websky"], color=PROJ_COL, lw=1.6, ls="--", label="projected 4")
    ax[3].axhline(1, color="#9a9891", lw=0.8); ax[3].set_ylabel("map(N) / WebSky"); ax[3].legend(fontsize=8.6, ncol=3, loc="best")
    ax[3].set_xscale("log"); ax[3].set_xlim(a.lmin, lmax); ax[3].set_xlabel(r"multipole $\ell$")
    for x in ax: x.tick_params(which="both", top=True, right=True)
    fig.savefig(out / "tsz_truncation_radius_series.png", dpi=190, bbox_inches="tight"); fig.savefig(out / "tsz_truncation_radius_series.pdf", bbox_inches="tight"); plt.close(fig)
    # ---------------- figure 2: CLASS-SZ's own radius dependence and 1h/2h split (context)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    for N in RADII:
        ax[0].plot(lc, cs[N]["dl_total"] / cs[5]["dl_total"], color=COL[N], lw=2, label=f"CLASS-SZ $x_{{\\rm out}}$ = {N} / 5")
        ax[0].plot(lc, b[f"y_sphere{N}"] / b["y_sphere5"], color=COL[N], lw=1.1, ls=":")
    ax[0].axhline(1, color="#9a9891", lw=0.8); ax[0].set_ylabel("ratio to N = 5"); ax[0].set_title("Radius dependence: CLASS-SZ (solid) vs HalfDome maps (dotted)", fontsize=10.5, loc="left"); ax[0].legend(fontsize=8.6)
    ax[1].plot(lc, S * cs[4]["dl_1h"], color=CS_COL, lw=1.8, label="CLASS-SZ 1h, HalfDome setup ($x_{\\rm out}$=4)"); ax[1].plot(lc, S * cs[4]["dl_2h"], color=CS_COL, lw=1.8, ls="--", label="CLASS-SZ 2h, HalfDome setup")
    ax[1].plot(lc, S * b["y_sphere4"], color=COL[4], lw=2, label="HalfDome map, sphere 4"); ax[1].plot(lc, S * b["websky"], color=WEBSKY_COL, lw=2, ls="-.", label="WebSky map")
    for fname, lab, ls_ in (("class_sz_tsz_websky_setup_xout4_mmin1e12.npz", "CLASS-SZ 1h+2h, WebSky cosmology, $M>10^{12}$ Msun/h", ":"),
                            ("class_sz_tsz_websky_setup_xout4_halfdome_mmin.npz", "CLASS-SZ 1h+2h, WebSky cosmology, HalfDome mass floor", (0, (3, 1, 1, 1)))):
        f = Path(a.class_sz_dir) / fname
        if f.exists():
            d = np.load(f); _, v = bin_dl(ell[2:], interp_loglog(d["ell"], d["dl_total"], ell[2:]), edges)
            ax[1].plot(lc, S * v, color=WEBSKY_COL, lw=1.3, ls=ls_, label=lab)
    ax[1].set_yscale("log"); ax[1].set_ylabel(r"$10^{12}\,D_\ell^{yy}$"); ax[1].legend(fontsize=7.8); ax[1].set_title("One- and two-halo terms at 4 $R_{200c}$; WebSky-setup references", fontsize=10.5, loc="left")
    for x in ax: x.set_xscale("log"); x.set_xlim(a.lmin, lmax); x.set_xlabel(r"multipole $\ell$"); x.tick_params(which="both", top=True, right=True)
    fig.tight_layout(); fig.savefig(out / "tsz_truncation_radius_context.png", dpi=170, bbox_inches="tight"); plt.close(fig)
    # ---------------- table
    with open(out / "radius_series_table.csv", "w", newline="") as fh:
        wri = csv.writer(fh); head = ["ell_eff"] + [f"1e12_Dl_sphere{N}" for N in RADII] + ["1e12_Dl_projected4", "1e12_Dl_websky"] + [f"1e12_Dl_classsz_xout{N}" for N in RADII] \
            + [f"sphere{N}_over_sphere5" for N in RADII] + [f"sphere{N}_over_classsz{N}" for N in RADII] + [f"sphere{N}_over_websky" for N in RADII] + ["projected4_over_sphere4", "projected4_over_websky"]
        wri.writerow(head)
        for i in range(len(lc)):
            wri.writerow([f"{v:.6g}" for v in [lc[i]] + [S * b[f"y_sphere{N}"][i] for N in RADII] + [S * b["y_projected4"][i], S * b["websky"][i]] + [S * cs[N]["dl_total"][i] for N in RADII]
                          + [b[f"y_sphere{N}"][i] / b["y_sphere5"][i] for N in RADII] + [b[f"y_sphere{N}"][i] / cs[N]["dl_total"][i] for N in RADII] + [b[f"y_sphere{N}"][i] / b["websky"][i] for N in RADII]
                          + [b["y_projected4"][i] / b["y_sphere4"][i], b["y_projected4"][i] / b["websky"][i]]])
    summary = {}
    for l0 in (100, 300, 1000, 3000, 6000):
        if l0 > lmax: continue
        i = int(np.argmin(np.abs(lc - l0)))
        summary[f"ell~{lc[i]:.0f}"] = {**{f"sphere{N}_over_sphere5": round(float(b[f"y_sphere{N}"][i] / b["y_sphere5"][i]), 4) for N in RADII},
                                       **{f"sphere{N}_over_classsz{N}": round(float(b[f"y_sphere{N}"][i] / cs[N]["dl_total"][i]), 4) for N in RADII},
                                       **{f"sphere{N}_over_websky": round(float(b[f"y_sphere{N}"][i] / b["websky"][i]), 4) for N in RADII},
                                       "projected4_over_sphere4": round(float(b["y_projected4"][i] / b["y_sphere4"][i]), 4),
                                       "classsz4_over_websky": round(float(cs[4]["dl_total"][i] / b["websky"][i]), 4)}
    (out / "radius_series_summary.json").write_text(json.dumps(summary, indent=1)); print(json.dumps(summary, indent=1))

if __name__ == "__main__":
    main()
