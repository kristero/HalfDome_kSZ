"""Strided sample of the WebSky halo catalogue (halos.pksc) on idark (2026-10-02, ksz_debug/velocities).
Format (WebSky docs): int32 Nhalo, float32 RTHmax, float32 redshiftbox, then Nhalo x 10 float32
(x, y, z [comoving Mpc], vx, vy, vz [km/s], R_TH [Mpc], x_lag, y_lag, z_lag).
Reads NCH contiguous blocks of LEN halos evenly spaced through the file and saves them to an npz."""
import numpy as np, sys, os
fn = "/home/kristero10/sims/halos.pksc"
out = sys.argv[1] if len(sys.argv) > 1 else "websky_sample.npz"
NCH, LEN = 2000, 2000
with open(fn, "rb") as f:
    N = int(np.fromfile(f, dtype=np.int32, count=1)[0])
    RTHmax, zbox = np.fromfile(f, dtype=np.float32, count=2)
    size = os.path.getsize(fn)
    print("Nhalo", N, "RTHmax", RTHmax, "redshiftbox", zbox, "file size", size, "expected", 12 + 40 * N, flush=True)
    starts = np.linspace(0, N - LEN, NCH).astype(np.int64)
    blocks = []
    for s in starts:
        f.seek(12 + 40 * int(s))
        blocks.append(np.fromfile(f, dtype=np.float32, count=10 * LEN).reshape(LEN, 10))
    data = np.concatenate(blocks)
np.savez(out, data=data, starts=starts, LEN=LEN, N=N, RTHmax=RTHmax, zbox=zbox)
print("saved", out, data.shape)
r = np.sqrt((data[:, :3].astype(float) ** 2).sum(1))
print("r range", r.min(), r.max(), "R_TH range", data[:, 6].min(), data[:, 6].max())
