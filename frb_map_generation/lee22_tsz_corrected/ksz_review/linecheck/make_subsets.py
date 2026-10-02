#!/usr/bin/env python3
"""Review subsets of the HalfDome lightcone for paint_ksz_websky.jl (2026-10-02, ksz_review/linecheck).
subset_random.h5: 190k uniformly random rows + 10k random rows with z < 0.1 (exact low-z table path), row order kept.
subset_first200k.h5 is not written: the literal --max-halos=200000 run reads the catalogue directly (rows 1..200000, z ~ 3.4-3.9)."""
import argparse
import h5py
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--catalog", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--seed", type=int, default=20261002)
a = ap.parse_args()
rng = np.random.default_rng(a.seed)
with h5py.File(a.catalog, "r") as f:
    N = f["redshift"].shape[0]
    z = f["redshift"][:]
    low = np.flatnonzero(z < 0.1)
    print("N", N, "z<0.1 halos", low.size, "z range", z.min(), z.max(), flush=True)
    pick = np.union1d(rng.choice(N, 190_000, replace=False), rng.choice(low, 10_000, replace=False))
    print("subset size", pick.size, "z<0.1 in subset", np.sum(z[pick] < 0.1), flush=True)
    with h5py.File(a.out, "w") as g:
        g["source_row"] = pick
        step = 10_000_000
        for k in ("Position", "Velocity", "halo_mass_m200c", "halo_mass_m200m", "redshift"):
            d = f[k]; parts = []
            for s in range(0, N, step):                      # chunked contiguous reads (h5py point selection is slow)
                sel = pick[(pick >= s) & (pick < s + step)] - s
                if sel.size:
                    parts.append(d[s:min(s + step, N)][sel])
            g.create_dataset(k, data=np.concatenate(parts))
        g.attrs["source"] = a.catalog
        g.attrs["seed"] = a.seed
print("wrote", a.out)
