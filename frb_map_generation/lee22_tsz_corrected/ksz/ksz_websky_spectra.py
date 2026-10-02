#!/usr/bin/env python3
"""Angular power spectra of the paint_ksz_websky.jl maps (2026-10-02): healpy anafast, lmax 8192, iter 1, one map at a time.
Maps are point-sampled DeltaT/T, so no pixel-window correction. Writes <out>/spectra/ksz_websky_spectra.npz (cl_<variant>, ell)."""
import argparse
from pathlib import Path

import healpy as hp
import numpy as np

VARIANTS = ("b16_phys", "lee22corr_phys", "lee22pipe_phys", "b16_websky", "lee22corr_websky")
ap = argparse.ArgumentParser()
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--nside", type=int, default=4096)
a = ap.parse_args()
res = {"ell": np.arange(8193)}
for v in VARIANTS:
    m = hp.read_map(str(a.out / "maps" / ("halfdome_ksz_%s_nside%d.fits" % (v, a.nside))), dtype=np.float64)
    res["cl_" + v] = hp.anafast(m, lmax=8192, iter=1)
    dl = res["ell"] * (res["ell"] + 1) * res["cl_" + v] / (2 * np.pi) * 2.7255e6 ** 2
    print(v, "D_ell [uK^2] at ell 1000/3000/5000:", np.round([dl[900:1101].mean(), dl[2700:3301].mean(), dl[4500:5501].mean()], 4), flush=True)
    del m
(a.out / "spectra").mkdir(parents=True, exist_ok=True)
np.savez(a.out / "spectra" / "ksz_websky_spectra.npz", **res)
print("saved", a.out / "spectra" / "ksz_websky_spectra.npz")
