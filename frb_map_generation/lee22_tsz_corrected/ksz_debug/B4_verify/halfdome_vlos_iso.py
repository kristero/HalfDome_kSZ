#!/usr/bin/env python3
"""Block-sampled HalfDome lightcone: v_los rms and |v|^2/v_los^2 per z bin (isotropy check)."""
import h5py, numpy as np, json, sys
fn = "/home/cbllover/HalfDome/lightcone_100.hdf5"
with h5py.File(fn, "r") as f:
    keys = list(f.keys()); print(keys)
    N = f["redshift"].shape[0]
    nb, L = 60, 20000
    starts = np.linspace(0, N - L, nb).astype(np.int64)
    P, V, Z = [], [], []
    for s in starts:
        P.append(f["Position"][s:s+L]); V.append(f["Velocity"][s:s+L]); Z.append(f["redshift"][s:s+L])
P = np.concatenate(P).astype(np.float64); V = np.concatenate(V).astype(np.float64); Z = np.concatenate(Z)
r = np.linalg.norm(P, axis=1); vlos = np.einsum("ij,ij->i", P, V) / r; v2 = np.einsum("ij,ij->i", V, V)
edges = [0, 0.1, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
rows = []
for lo, hi in zip(edges[:-1], edges[1:]):
    m = (Z >= lo) & (Z < hi)
    if m.sum() < 100: continue
    rows.append(dict(zlo=lo, zhi=hi, n=int(m.sum()), vlos_rms=float(np.sqrt(np.mean(vlos[m]**2))),
                     v3d_rms=float(np.sqrt(np.mean(v2[m]))), ratio_v2_over_vlos2=float(np.mean(v2[m]) / np.mean(vlos[m]**2))))
    print(rows[-1])
print("all: ratio", np.mean(v2) / np.mean(vlos**2), "n", len(Z))
json.dump(rows, open(sys.argv[1], "w"), indent=1)
