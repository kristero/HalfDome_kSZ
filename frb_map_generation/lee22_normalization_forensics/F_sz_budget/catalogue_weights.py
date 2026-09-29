#!/usr/bin/env python3
"""Histogram the full HalfDome lightcone catalogue (all 85,224,251 halos painted in the SZ maps) onto the (z, lnM)
nodes of tables_validation.npz: counts N_ij and sum of (v_los/c)^2. Used for the 'catalogue-weighted' 1-halo term

    C_l^1h(painted expectation) = (1/4pi) sum_halos (v_los,i/c)^2 T^2 |tau_l(M_i, z_i)|^2      (kSZ)
    C_l^1h(painted expectation) = (1/4pi) sum_halos |y_l(M_i, z_i)|^2                          (tSZ)

which removes the Tinker08 abundance and linear-velocity assumptions from the validation."""
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import szhm as S  # noqa: E402

CAT = "/home/cbllover/HalfDome/lightcone_100.hdf5"
OUT = Path("/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_normalization_forensics_20260928/F_sz_budget")
h = S.PAINT_COSMO.h
t = np.load(OUT / "tables_validation.npz")
z, lnM = t["z"], np.log(t["M"])
ze = np.concatenate([[0.0], np.sqrt(z[1:] * z[:-1]), [np.inf]])
me = np.concatenate([[-np.inf], 0.5 * (lnM[1:] + lnM[:-1]), [np.inf]])
N = np.zeros((len(z), len(lnM)))
V = np.zeros_like(N)
with h5py.File(CAT, "r") as f:
    n = f["redshift"].shape[0]
    step = 5_000_000
    for s0 in range(0, n, step):
        sl = slice(s0, min(n, s0 + step))
        zz = f["redshift"][sl].astype(float)
        mm = np.log(f["halo_mass_m200c"][sl] / h)
        pos = f["Position"][sl].astype(float)
        vel = f["Velocity"][sl].astype(float)
        vl = (vel * pos).sum(1) / np.linalg.norm(pos, axis=1) / S.C_KMS
        H1, _, _ = np.histogram2d(zz, mm, bins=(ze, me))
        H2, _, _ = np.histogram2d(zz, mm, bins=(ze, me), weights=vl ** 2)
        N += H1; V += H2
        print(s0, flush=True)
assert int(N.sum()) == n, (N.sum(), n)
np.savez_compressed(OUT / "catalogue_weights_validation.npz", N=N, V=V, z=z, lnM=lnM, n_halos=n)
print("halos", int(N.sum()), "rms v_los/c", np.sqrt(V.sum() / N.sum()))
