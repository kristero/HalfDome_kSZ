"""Full chunked pass over the HalfDome lightcone catalogue: per (log10 M [Msun], z) cell (M200m and M200c binnings, same
edges as websky_velocity_hist_idark.py) the count, sum v_los^2, sum |v|^2, sum v_los, and in M200c cells the sum of
log10 M200m (to map M200c -> M200m). 2026-10-02, ksz_debug/velocities."""
import numpy as np, h5py, time
O = "/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_ksz_debug_20261002/velocities/"
h = 0.68
LE = np.arange(12.0, 15.8001, 0.05); ZE = np.arange(0.0, 4.7001, 0.05)
nM, nZ = len(LE) - 1, len(ZE) - 1
keys = ("count", "sum_vlos2", "sum_v2", "sum_vlos")
Hm = {k: np.zeros((nM, nZ)) for k in keys}; Hc = {k: np.zeros((nM, nZ)) for k in keys}
Hc["sum_log10m200m"] = np.zeros((nM, nZ))
CH = 2_000_000; t0 = time.time()
with h5py.File("/home/cbllover/HalfDome/lightcone_100.hdf5", "r") as f:
    N = f["redshift"].shape[0]
    for s in range(0, N, CH):
        e = min(N, s + CH)
        P = f["Position"][s:e].astype(np.float64); V = f["Velocity"][s:e].astype(np.float64)
        z = f["redshift"][s:e].astype(np.float64)
        lmm = np.log10(f["halo_mass_m200m"][s:e] / h); lmc = np.log10(f["halo_mass_m200c"][s:e] / h)
        vlos = (P * V).sum(1) / np.sqrt((P * P).sum(1)); v2 = (V * V).sum(1)
        iz = np.floor(z / 0.05).astype(np.int64)
        for H, lm in ((Hm, lmm), (Hc, lmc)):
            im = np.floor((lm - LE[0]) / 0.05).astype(np.int64)
            ok = (im >= 0) & (im < nM) & (iz >= 0) & (iz < nZ)
            idx = im[ok] * nZ + iz[ok]
            for k, w in (("count", None), ("sum_vlos2", vlos[ok] ** 2), ("sum_v2", v2[ok]), ("sum_vlos", vlos[ok])):
                H[k] += np.bincount(idx, weights=w, minlength=nM * nZ).reshape(nM, nZ)
            if H is Hc:
                H["sum_log10m200m"] += np.bincount(idx, weights=lmm[ok], minlength=nM * nZ).reshape(nM, nZ)
        if (s // CH) % 5 == 0:
            print("%d/%d %.0f s" % (e, N, time.time() - t0), flush=True)
np.savez(O + "halfdome_velocity_hist.npz", logm_edges=LE, z_edges=ZE, **{"m200m_" + k: v for k, v in Hm.items()},
         **{"m200c_" + k: v for k, v in Hc.items()})
print("done", time.time() - t0)
