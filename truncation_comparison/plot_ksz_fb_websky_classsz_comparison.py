#!/usr/bin/env python3
"""Halo-only kSZ power spectrum: our XGPaint painting (with f_b, the production convention,
and without f_b, the reading that only matches Lee+2022's own Fig. 5 rendering of Battaglia
2016) against CLASS-SZ's independent halo-model prediction and against Stein et al. 2020
(WebSky, arXiv:2001.08787) Figure 6, digitized directly from the published PDF: the "WebSky
Halo" orange curve and the "Battaglia et al. (2010)" red-triangle point, both of which the
WebSky paper computed from the same Battaglia (2016)/Battaglia (2010) hydrodynamical
simulations used to fit the free-electron profile applied here (see paper text, Sec. 4.4.2).

Units: our XGPaint map and the CLASS-SZ reference are dimensionless ell(ell+1)Cl/2pi in
(Delta T/T)^2 (see TSZ_KSZ_TRUNCATION_COMPARISON_20260917.md Sec. 3.2, confirmed against the
catalogue Poisson term); the digitized WebSky data is in physical uK^2. Converted using
T_CMB = 2.7255e6 uK.
"""
import argparse
import csv
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

T_CMB_UK = 2.7255e6


def load_theory(path):
    import json
    d = np.load(path)
    md = json.loads(str(d["metadata_json"]))
    return d["ell"], d["dl_1h"], d["dl_2h"], md


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--binned-fb", type=Path, default=Path("truncation_comparison/spectra/binned_spectra_fb.csv"))
    ap.add_argument("--class-sz-ksz", type=Path, default=Path("truncation_comparison/spectra/class_sz_ksz_x4_reference.npz"))
    ap.add_argument("--digitized", type=Path,
                     default=Path("/tmp/claude-0/-home-cbllover-HalfDome/015746ae-94b7-42cb-8dc5-0e8c33dc8490/scratchpad/stein2020_fig6_digitized.pkl"))
    ap.add_argument("--output-dir", type=Path, default=Path("truncation_comparison/spectra"))
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.binned_fb)))
    ell = np.array([float(r["ell_eff"]) for r in rows])
    dl_with = np.array([float(r["Dl_ksz_with_fb_dimensionless"]) for r in rows]) * T_CMB_UK**2
    dl_without = np.array([float(r["Dl_ksz_no_fb_dimensionless"]) for r in rows]) * T_CMB_UK**2

    e_th, h1, h2, md = load_theory(a.class_sz_ksz)
    dl_th_1h = h1 * T_CMB_UK**2
    dl_th_tot = (h1 + h2) * T_CMB_UK**2

    dig = pickle.load(open(a.digitized, "rb"))
    xs_halo, ys_halo = dig["websky_halo"]
    ell_b2010, dl_b2010 = dig["battaglia2010_point"]

    plt.rcParams.update({"font.size": 13, "axes.labelsize": 14, "legend.fontsize": 10.5})
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.plot(ell, dl_with, color="#0b6b78", lw=2.8, label="XGPaint halo kSZ, with f_b (production)")
    ax.plot(ell, dl_without, color="#d55e00", lw=2.4, ls="--", label="XGPaint halo kSZ, without f_b (Lee+22 Fig.5-only reading)")
    ax.plot(e_th, dl_th_1h, color="#555", lw=1.8, ls=":", label="CLASS-SZ B16 halo kSZ, 1h")
    ax.plot(e_th, dl_th_tot, color="#555", lw=1.8, label="CLASS-SZ B16 halo kSZ, 1h+2h")
    o = np.argsort(xs_halo)
    ax.scatter(xs_halo[o][::6], ys_halo[o][::6], s=5, color="#ff4500", alpha=0.45,
               label="Stein+20 Fig.6, digitized: WebSky Halo")
    ax.scatter([ell_b2010], [dl_b2010], marker="v", s=140, facecolors="none", edgecolors="#d62728", linewidths=2,
               label="Stein+20 Fig.6, digitized: Battaglia et al. (2010)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(60, 8000); ax.set_ylim(3e-3, 45)
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(r"$\ell(\ell+1)C_\ell/2\pi$  [$\mu K^2$]")
    ax.grid(alpha=0.15, which="both")
    ax.tick_params(direction="in", which="both", top=True, right=True)
    ax.legend(loc="upper left", fontsize=9.5, framealpha=0.95)
    fig.suptitle("Halo-only kSZ power spectrum: with/without f_b vs CLASS-SZ and WebSky (Stein+2020)",
                 fontsize=12.5)
    fig.tight_layout()
    for ext in ("png", "pdf", "svg"):
        fig.savefig(str(a.output_dir / ("ksz_fb_websky_classsz_comparison." + ext)), dpi=170 if ext == "png" else None,
                    bbox_inches="tight")
    print("Saved", a.output_dir / "ksz_fb_websky_classsz_comparison.png")

    print("\nell   with_fb   without_fb   classsz_1h+2h   websky_halo(nearest)")
    for target in (300, 1000, 2000, 3000, 5000):
        w = np.interp(target, ell, dl_with)
        wo = np.interp(target, ell, dl_without)
        cs = np.interp(target, e_th, dl_th_tot)
        wb = np.interp(target, xs_halo[o], ys_halo[o])
        print(f"{target:5d} {w:10.4f} {wo:10.4f} {cs:14.4f} {wb:10.4f}")


if __name__ == "__main__":
    main()
