#!/usr/bin/env python3
"""WebSky's own <(v_los/c)^2> per (log10 M200c_ws [Msun], z) cell of the painter grid (2026-10-02), for the per-cell velocity
variant of poisson_mass_split_websky.jl, and the WebSky catalogue's one-halo histogram in the painter's format. Source: the
WebSky catalogue histograms of ksz_debug/websky_catalogue_test/websky_histogram.py (halos.pksc, 2LPT velocities, M200c_ws =
M200m x NFW(c200m = 7) ratio in WebSky's cosmology, WebSky cut M200c_ws > 1e13 Msun and theta200c > 0.5 arcmin).
Writes <out>/websky_cell_v2.h5 and <out>/maps/websky_catalogue_histogram.h5 (bin-centre centroids; no centroids in the source)."""
import argparse
from pathlib import Path

import h5py
import numpy as np

FMG = Path(__file__).resolve().parents[2]
ap = argparse.ArgumentParser()
ap.add_argument("--src", type=Path, default=FMG / "outputs/lee22_ksz_debug_20261002/websky_catalogue_test/websky_halo_mass_redshift_histograms.npz")
ap.add_argument("--out", type=Path, default=FMG / "outputs/lee22_ksz_websky_20261002")
a = ap.parse_args()
d = np.load(a.src)
le, ze = d["logm_edges"], d["z_edges"]
assert np.allclose(le, np.arange(12.0, 15.7 + 1e-9, 0.025)) and np.allclose(ze, np.arange(0.0, 4.0 + 1e-9, 0.025)), "grid differs from the painter's"
n, v2 = d["c7m_count_sel"], d["c7m_v2_sel"]                  # (log10 M200c_ws, z)
lmc, zc = 0.5 * (le[1:] + le[:-1]), 0.5 * (ze[1:] + ze[:-1])
(a.out / "maps").mkdir(parents=True, exist_ok=True)
with h5py.File(a.out / "websky_cell_v2.h5", "w") as f:              # h5py (z, M) -> Julia (M, z)
    f["logm_edges"] = le; f["z_edges"] = ze; f["count"] = n.T.astype(np.int64); f["sum_vlos_over_c_squared"] = v2.T
    f.attrs["source"] = str(a.src); f.attrs["note"] = "WebSky-selected halos (c7m), per-cell count and sum (v_los/c)^2"
with h5py.File(a.out / "maps" / "websky_catalogue_histogram.h5", "w") as f:
    z0 = np.zeros_like(v2).T
    f["logm_edges"] = le; f["z_edges"] = ze
    f["count"] = z0.astype(np.int64); f["sum_vlos_over_c_squared"] = z0; f["sum_v2_log10m"] = z0; f["sum_v2_z"] = z0
    f["websky_count"] = n.T.astype(np.int64); f["websky_sum_vlos_over_c_squared"] = v2.T
    f["websky_sum_v2_log10m"] = (v2 * lmc[:, None]).T; f["websky_sum_v2_z"] = (v2 * zc[None, :]).T
    f.attrs["note"] = "WebSky catalogue, WebSky-selected halos only (physical fields zero); centroids = bin centres"
print("WebSky selected halos %d; wrote %s and %s" % (n.sum(), a.out / "websky_cell_v2.h5", a.out / "maps/websky_catalogue_histogram.h5"))
