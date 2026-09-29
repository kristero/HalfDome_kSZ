#!/usr/bin/env python3
"""Ray-bootstrap uncertainty of the z-resolved TNG/Lee22-literal ratio (coarse slices, 1e13-1e14 Msun).
Only the statistical noise of the 120000 TNG rays is captured (not TNG box variance, HMF or
mass-definition systematics)."""
import json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.argv = [sys.argv[0], "x"]
import zresolved_mean_dm_test as zt  # noqa: E402
zs = np.load(zt.KON / "redshifts_DMhalo.npy")
arr = np.load(zt.KON / "DMhalo_r200_mass13to14.npy")
edges = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.3, 1.6, 2.0])
zg = np.linspace(0, 2.0, 81)
dd = np.array([zt.dmean_dz(z) for z in zg])
lee = np.array([np.trapz(dd[(zg >= a - 1e-9) & (zg <= b + 1e-9)], zg[(zg >= a - 1e-9) & (zg <= b + 1e-9)]) for a, b in zip(edges[:-1], edges[1:])])
zm = 0.5 * (edges[:-1] + edges[1:])
idx = [np.argmin(np.abs(zs - b)) for b in edges[1:]]
cols = np.r_[np.zeros((1, arr.shape[1])), arr[idx]]
inc = np.diff(cols, axis=0)                     # (8 slices, rays)
rng = np.random.default_rng(1)
out = []
for it in range(400):
    s = rng.integers(0, arr.shape[1], arr.shape[1]) if it else np.arange(arr.shape[1])
    t = inc[:, s].mean(axis=1)
    R = t / lee
    k, lnA = np.polyfit(np.log(1 + zm), np.log(R), 1, w=np.sqrt(t))
    lnA3 = np.average(np.log(R) - 3 * np.log(1 + zm), weights=t)
    out.append((k, np.exp(-lnA), np.exp(-lnA3), 1 / R[0]))
out = np.array(out)
names = ["free_slope", "literal_over_TNG_at_z0_free", "F_n_fixed_slope3", "literal_over_TNG_slice_0_0.2"]
res = {n: dict(value=float(out[0, i]), boot_sd=float(out[1:, i].std())) for i, n in enumerate(names)}
print(json.dumps(res, indent=1))
json.dump(res, open(zt.OUT / "zresolved_bootstrap.json", "w"), indent=1)
