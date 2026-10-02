"""Cumulative WebSky-selected halo counts in catalogue row order, to compare with the painter's progress lines (review 2026-10-02)."""
import sys, os
import numpy as np, h5py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
cos = Cosmo(0.6774, 0.0486, 0.2603); zt, rt = ratio_c7_table(cos.Om)
with h5py.File("/home/cbllover/HalfDome/lightcone_100.hdf5", "r") as f:
    z = f["redshift"][:].astype(np.float64); mm = f["halo_mass_m200m"][:] / 0.6774
mws = mm * np.interp(z, zt, rt)
sel = (mws > 1e13) & (cos.theta200c(mws, z) > 0.5 * ARCMIN)
cs = np.cumsum(sel); lz = np.cumsum(z < 0.1)
for n in list(range(1_000_000, len(z), 1_000_000)) + [len(z)]:
    print(f"rows 1:{n}  websky-selected={cs[n - 1]}  exact-lowz={lz[n - 1]}")
# boundary sensitivity: halos within 1e-6 (relative) of either cut
near = (np.abs(mws / 1e13 - 1) < 1e-6) | (np.abs(cos.theta200c(mws, z) / (0.5 * ARCMIN) - 1) < 1e-6)
print("halos within 1e-6 of a cut boundary:", int(near.sum()))
