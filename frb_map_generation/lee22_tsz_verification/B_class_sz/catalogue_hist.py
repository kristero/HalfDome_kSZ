#!/usr/bin/env python3
"""Histogram the full HalfDome lightcone catalogue (all 85,224,251 halos painted in the y maps) in (z, ln M200c)
for the catalogue-weighted halo-model terms, and check its sky coverage.

Masses: halo_mass_m200c is in Msun/h; the painter divides by h = 0.68 for physical Msun. The histogram is in
M200c [Msun/h] with h = 0.68, i.e. the catalogue numbers unchanged (CLASS-SZ convention with h = 0.68).
Output: OUT/catalogue/catalogue_hist.npz (N[z, lnM], bin edges, sky occupancy of a 1-in-20 subsample)."""
import sys
from pathlib import Path
import h5py
import numpy as np

sys.path.insert(0, "/home/cbllover/HalfDome/frb_map_generation/lee22_tsz_verification/B_class_sz")
import common as C  # noqa: E402

OUTD = Path(C.OUT) / "catalogue"; OUTD.mkdir(parents=True, exist_ok=True)
def _pieces(breaks, step, log=False):
    e = []
    for a, b in zip(breaks[:-1], breaks[1:]):
        k = max(1, int(np.ceil(((np.log(b / a)) if log else (b - a)) / step)))
        seg = np.geomspace(a, b, k + 1) if log else np.linspace(a, b, k + 1)
        e += list(seg[:-1])
    return np.array(e + [breaks[-1]])


# exact edges at the decomposition / selection boundaries (z = 0.1, 0.25, 0.5, 1, 2; M = 1e13, 10^14.8 Msun/h)
Z_EDGES = np.concatenate([np.geomspace(0.002, 0.1, 121)[:-1], _pieces([0.1, 0.25, 0.5, 1.0, 2.0, 3.86], 0.0125)])
LNM_EDGES = np.log(_pieces([4.45e12, 1e13, 3e13, 1e14, 3e14, C.LEE_CLIP_HINV, 2.58e15], 0.027, log=True))

if __name__ == "__main__":
    N = np.zeros((Z_EDGES.size - 1, LNM_EDGES.size - 1))
    occ = None
    try:
        import healpy as hp
        nside = 16
        occ = np.zeros(hp.nside2npix(nside))
    except Exception:
        hp = None
    with h5py.File(C.CATALOGUE, "r") as f:
        n = f["redshift"].shape[0]
        zmin, zmax, mmin, mmax = np.inf, -np.inf, np.inf, -np.inf
        n_above_clip = 0
        step = 5_000_000
        for s0 in range(0, n, step):
            sl = slice(s0, min(n, s0 + step))
            z = f["redshift"][sl].astype(float)
            m = f["halo_mass_m200c"][sl].astype(float)
            zmin, zmax = min(zmin, z.min()), max(zmax, z.max())
            mmin, mmax = min(mmin, m.min()), max(mmax, m.max())
            H, _, _ = np.histogram2d(z, np.log(m), bins=(Z_EDGES, LNM_EDGES))
            N += H
            n_above_clip += int((m >= C.LEE_CLIP_HINV).sum())
            if hp is not None:
                pos = f["Position"][s0:min(n, s0 + step):20].astype(float)
                occ += np.bincount(hp.vec2pix(nside, pos[:, 0], pos[:, 1], pos[:, 2]), minlength=occ.size)
            print(s0, flush=True)
    assert int(N.sum()) == n, (N.sum(), n)
    out = dict(N=N, z_edges=Z_EDGES, lnm_edges=LNM_EDGES, n_halos=n, z_range=[zmin, zmax], m_range_msunh=[mmin, mmax],
               n_above_clip=n_above_clip)
    if occ is not None:
        out["sky_occupancy_nside16"] = occ
        print("sky: empty nside-16 pixels", int((occ == 0).sum()), "of", occ.size, "; rms/mean", occ.std() / occ.mean())
    np.savez_compressed(OUTD / "catalogue_hist.npz", **out)
    print("halos", n, "z", zmin, zmax, "M200c [Msun/h]", mmin, mmax, "above 10^14.8/h:", n_above_clip,
          "(histogram:", int(N[:, LNM_EDGES[:-1] >= np.log(C.LEE_CLIP_HINV) - 1e-12].sum()), ")")
