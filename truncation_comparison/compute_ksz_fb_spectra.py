#!/usr/bin/env python3
"""Angular power spectra of the with-f_b / without-f_b halo kSZ maps
(paint_ksz_fb_comparison_maps.jl), pixel-window corrected. Saves raw Cl and a binned CSV;
no plotting (done locally against the digitized WebSky/CLASS-SZ references)."""
import argparse
from pathlib import Path
import numpy as np
import healpy as hp


def log_bins(lmin, lmax, per_decade):
    edges = np.unique(np.round(np.logspace(np.log10(lmin), np.log10(lmax), int(per_decade * np.log10(lmax / lmin)) + 1)).astype(int))
    return edges


def bin_dl(ell, dl, edges):
    out, centres = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (ell >= lo) & (ell < hi)
        if not np.any(sel):
            continue
        w = 2 * ell[sel] + 1.0
        out.append(np.sum(w * dl[sel]) / np.sum(w))
        centres.append(np.exp(np.sum(w * np.log(ell[sel])) / np.sum(w)))
    return np.array(centres), np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps-dir", default="maps")
    ap.add_argument("--out-dir", default="spectra")
    ap.add_argument("--lmax", type=int, default=8192)
    a = ap.parse_args()
    maps_dir, out_dir = Path(a.maps_dir), Path(a.out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    names = ["ksz_sphere_with_fb", "ksz_sphere_no_fb"]
    raw = {}
    for n in names:
        path = next(maps_dir.glob(f"halfdome_{n}_nside*_r200cx4.fits"))
        m = hp.read_map(path, dtype=np.float64)
        nside = hp.get_nside(m)
        lmax = min(a.lmax, 3 * nside - 1)
        print(f"anafast {path.name}: nside={nside} lmax={lmax} mean={m.mean():.3e} rms={m.std():.3e}", flush=True)
        cl = hp.anafast(m, lmax=lmax, iter=0)
        raw[n] = cl / hp.pixwin(nside, lmax=lmax) ** 2
        raw[n + "_nopixwin"] = cl
        del m
    raw["nside"] = nside
    raw["lmax"] = lmax
    np.savez(out_dir / "raw_cl_fb.npz", **raw)
    ell = np.arange(lmax + 1, dtype=float)
    dl = {n: ell * (ell + 1) * raw[n] / (2 * np.pi) for n in names}
    edges = log_bins(30, lmax, 20)
    import csv
    with open(out_dir / "binned_spectra_fb.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["ell_eff", "Dl_ksz_with_fb_dimensionless", "Dl_ksz_no_fb_dimensionless", "ratio_no_fb_over_with_fb"])
        lc, d_with = bin_dl(ell[2:], dl["ksz_sphere_with_fb"][2:], edges)
        _, d_without = bin_dl(ell[2:], dl["ksz_sphere_no_fb"][2:], edges)
        for row in zip(lc, d_with, d_without, d_without / d_with):
            w.writerow(row)
    print("saved", out_dir / "raw_cl_fb.npz", "and", out_dir / "binned_spectra_fb.csv")


if __name__ == "__main__":
    main()
