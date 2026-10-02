# HalfDome lightcone: (log10 M [Msun], z) histograms in M200m (catalogue halo_mass_m200m / h) and M200c (as painted), with count,
# sum (v_los/c)^2 and sum |v|^2/c^2, plus the M200c/M200m ratio per z, for the WebSky catalogue comparison (2026-10-02, ksz_debug).
# Chunked read (one process); the painter's own M200c histogram is reproduced as a check.
import sys, os, time
import numpy as np, h5py

cat = "/home/cbllover/HalfDome/lightcone_100.hdf5"
outdir = sys.argv[1]
H = 0.68; C_KMS = 299792.458
LOGM_EDGES = np.round(np.arange(12.0, 15.7 + 1e-9, 0.025), 6); Z_EDGES = np.round(np.arange(0.0, 4.0 + 1e-9, 0.025), 6)
NM, NZ = len(LOGM_EDGES) - 1, len(Z_EDGES) - 1
acc = {k: np.zeros(NM * NZ) for k in ("m200m_count", "m200m_v2", "m200m_vtot2", "m200c_count", "m200c_v2", "m200c_vtot2")}
ratio_sum = np.zeros(NZ); ratio_n = np.zeros(NZ); lr_sum = np.zeros(NZ)
mins = {"m200m": np.inf, "m200c": np.inf}
t0 = time.time()
with h5py.File(cat, "r") as f:
    N = f["redshift"].shape[0]; ch = 4_000_000
    for s in range(0, N, ch):
        e = min(N, s + ch)
        z = f["redshift"][s:e].astype(np.float64)
        mm = f["halo_mass_m200m"][s:e] / H; mc = f["halo_mass_m200c"][s:e] / H
        pos = f["Position"][s:e].astype(np.float64); vel = f["Velocity"][s:e].astype(np.float64)
        ok = np.isfinite(mm) & np.isfinite(mc) & (mm > 0) & (mc > 0) & np.isfinite(z) & (z > 0) & np.all(np.isfinite(pos), 1) & np.all(np.isfinite(vel), 1)
        z, mm, mc, pos, vel = z[ok], mm[ok], mc[ok], pos[ok], vel[ok]
        r = np.sqrt((pos**2).sum(1)); vlos = (pos * vel).sum(1) / r
        v2 = (vlos / C_KMS)**2; vt2 = (vel**2).sum(1) / C_KMS**2
        iz = np.floor(z / 0.025 + 1e-12).astype(np.int64)
        mins["m200m"] = min(mins["m200m"], mm.min()); mins["m200c"] = min(mins["m200c"], mc.min())
        for tag, m in (("m200m", mm), ("m200c", mc)):
            im = np.floor((np.log10(m) - 12.0) / 0.025 + 1e-12).astype(np.int64)
            g = (im >= 0) & (im < NM) & (iz >= 0) & (iz < NZ)
            idx = im[g] * NZ + iz[g]
            acc[f"{tag}_count"] += np.bincount(idx, minlength=NM * NZ)
            acc[f"{tag}_v2"] += np.bincount(idx, weights=v2[g], minlength=NM * NZ)
            acc[f"{tag}_vtot2"] += np.bincount(idx, weights=vt2[g], minlength=NM * NZ)
        g = (iz >= 0) & (iz < NZ)
        ratio_sum += np.bincount(iz[g], weights=(mc / mm)[g], minlength=NZ); ratio_n += np.bincount(iz[g], minlength=NZ)
        lr_sum += np.bincount(iz[g], weights=np.log(mc / mm)[g], minlength=NZ)
        print(f"rows {s}-{e} elapsed {time.time() - t0:.0f} s", flush=True)
save = {k: v.reshape(NM, NZ) for k, v in acc.items()}
save.update(logm_edges=LOGM_EDGES, z_edges=Z_EDGES, mean_m200c_over_m200m=ratio_sum / np.maximum(ratio_n, 1),
            geomean_m200c_over_m200m=np.exp(lr_sum / np.maximum(ratio_n, 1)), n_per_z=ratio_n)
np.savez(os.path.join(outdir, "halfdome_halo_histograms_m200m_m200c.npz"), **save)
print("min masses [Msun]", mins, "total", time.time() - t0)
