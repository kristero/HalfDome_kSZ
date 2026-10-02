"""Chunked sample of the HalfDome lightcone catalogue for velocity-convention checks (2026-10-02, ksz_debug/velocities).
Reads NCHUNK contiguous blocks of LEN rows evenly spaced over the 85.2M rows (catalogue sorted by decreasing z) and saves
Position, Velocity, masses, redshift, Aemit, Length, Vdisp to an npz in the debug output dir."""
import numpy as np, h5py, os
OUT = "/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_ksz_debug_20261002/velocities"
NCHUNK, LEN = 400, 5000
with h5py.File("/home/cbllover/HalfDome/lightcone_100.hdf5", "r") as f:
    N = f["redshift"].shape[0]
    starts = np.linspace(0, N - LEN, NCHUNK).astype(np.int64)
    cols = {k: [] for k in ("Position", "Velocity", "halo_mass_m200c", "halo_mass_m200m", "log10_m_rfof", "redshift", "Aemit", "Length", "Vdisp", "ID")}
    for s in starts:
        for k in cols:
            cols[k].append(f[k][s:s + LEN])
    data = {k: np.concatenate(v) for k, v in cols.items()}
    data["row"] = np.concatenate([np.arange(s, s + LEN) for s in starts])
np.savez(os.path.join(OUT, "catalogue_sample.npz"), **data)
print({k: v.shape for k, v in data.items()})
