#!/usr/bin/env python3
"""Smooth fit of the WebSky / HalfDome halo line-of-sight velocity dispersion at matched M200m and z (2026-10-02), for the
velocity-matched one-halo term of poisson_mass_split_websky.jl. Same fit as ksz_debug/velocities/velocity_ratio_grid.py:
  ln(sigma_WS / sigma_HD) = c0 + c1 x + c2 x^2 + c3 z + c4 z^2 + c5 x z,   x = log10 M200m [Msun] - 13.5,
with cells weighted by sqrt(min(N_WS, N_HD)), from the WebSky catalogue (halos.pksc, 2LPT velocities) and HalfDome lightcone
(N-body velocities) histograms of ksz_debug/velocities. Writes <out>/velocity_ratio_fit.h5 (coef, 6 values)."""
import argparse
from pathlib import Path

import h5py
import numpy as np

FMG = Path(__file__).resolve().parents[2]
ap = argparse.ArgumentParser()
ap.add_argument("--vel-dir", type=Path, default=FMG / "outputs/lee22_ksz_debug_20261002/velocities")
ap.add_argument("--out", type=Path, default=FMG / "outputs/lee22_ksz_websky_20261002")
a = ap.parse_args()
w = np.load(a.vel_dir / "websky_velocity_hist.npz"); d = np.load(a.vel_dir / "halfdome_velocity_hist.npz")
LE, ZE = w["logm200m_edges"], w["z_edges"]
lm, zc = 0.5 * (LE[1:] + LE[:-1]), 0.5 * (ZE[1:] + ZE[:-1])
nw, sw, nh, sh = w["count"], w["sum_vlos2"], d["m200m_count"], d["m200m_sum_vlos2"]
ok = (nw > 20) & (nh > 20) & (zc[None, :] < 4.0)
X = np.broadcast_to(lm[:, None] - 13.5, nw.shape)[ok]; Z = np.broadcast_to(zc[None, :], nw.shape)[ok]
y = 0.5 * np.log((sw[ok] / nw[ok]) / (sh[ok] / nh[ok]))
wt = np.sqrt(np.minimum(nw[ok], nh[ok]))
A = np.c_[np.ones_like(X), X, X**2, Z, Z**2, X * Z]
coef, *_ = np.linalg.lstsq(A * wt[:, None], y * wt, rcond=None)
res = y - A @ coef
print("coef ln(sigma_WS/sigma_HD):", coef.round(5), " weighted rms residual %.4f" % np.sqrt(np.average(res**2, weights=wt**2)))
model = lambda x, z: np.exp(np.c_[np.ones_like(x), x, x**2, z, z**2, x * z] @ coef)
for m in (13.0, 13.5, 14.0, 14.5, 15.0):
    print("log10 M200m %.1f: <v^2>_WS/<v^2>_HD at z = 0.1/0.5/1/2/3 =" % m, (model(np.full(5, m - 13.5), np.array([0.1, 0.5, 1, 2, 3])) ** 2).round(3))
a.out.mkdir(parents=True, exist_ok=True)
with h5py.File(a.out / "velocity_ratio_fit.h5", "w") as f:
    f["coef"] = coef
    f.attrs["model"] = "ln(sigma_WS/sigma_HD) = c0 + c1 x + c2 x^2 + c3 z + c4 z^2 + c5 x z, x = log10 M200m[Msun] - 13.5"
    f.attrs["source"] = str(a.vel_dir)
print("wrote", a.out / "velocity_ratio_fit.h5")
