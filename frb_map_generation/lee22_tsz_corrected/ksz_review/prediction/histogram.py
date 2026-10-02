"""HalfDome lightcone histograms for the one-halo prediction of the paint_ksz_websky.jl maps (review 2026-10-02).
h = 0.6774 masses. Two histograms in (log10 M, z) with count, sum (v_los/c)^2, sum (v/c)^2 log10 M, sum (v/c)^2 z:
  phys:   catalogue M200c (all halos, as painted for b16_phys / lee22corr_phys)
  websky: M200c_ws = catalogue M200m x NFW(c200m = 7) ratio, only halos passing the WebSky cut (M200c_ws > 1e13 Msun and
          theta200c(M200c_ws) > 0.5 arcmin), evaluated per halo with the XGPaint-like cosmology (radiation included).
Fine z bins at low z (dz = 0.0025 below z = 0.1, 0.0125 above), dlog10 M = 0.02."""
import sys, time, os
import numpy as np, h5py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

cat = sys.argv[1] if len(sys.argv) > 1 else "/home/cbllover/HalfDome/lightcone_100.hdf5"
out = sys.argv[2] if len(sys.argv) > 2 else "/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_ksz_review_20261002/prediction/halfdome_hist_h06774.npz"
H = float(os.environ.get("HALO_H", "0.6774"))          # 0.68 with HALO_OB=0.049 HALO_OC=0.261 reproduces the earlier map setup
cos = Cosmo(H, float(os.environ.get("HALO_OB", "0.0486")), float(os.environ.get("HALO_OC", "0.2603")))
zt, rt = ratio_c7_table(cos.Om)
LE = np.round(np.arange(12.0, 15.7 + 1e-9, 0.02), 6)
ZE = np.unique(np.round(np.concatenate([np.arange(0, 0.1, 0.0025), np.arange(0.1, 4.0 + 1e-9, 0.0125)]), 6))
NM, NZ = len(LE) - 1, len(ZE) - 1
keys = ["n", "v2", "v2lm", "v2z"]
acc = {f"{t}_{k}": np.zeros(NM * NZ) for t in ("phys", "ws") for k in keys}
vz_n = np.zeros(NZ); vz_v2 = np.zeros(NZ)
stats = dict(n_read=0, n_kept=0, n_ws=0, v2_sum=0.0, n_lowz=0)
t0 = time.time()
with h5py.File(cat, "r") as f:
    N = f["redshift"].shape[0]
    ch = 4_000_000
    for s in range(0, N, ch):
        e = min(N, s + ch)
        z = f["redshift"][s:e].astype(np.float64)
        mm = f["halo_mass_m200m"][s:e] / H
        mc = f["halo_mass_m200c"][s:e] / H
        pos = f["Position"][s:e].astype(np.float64)
        vel = f["Velocity"][s:e].astype(np.float64)
        ok = (np.isfinite(mm) & np.isfinite(mc) & (mm > 0) & (mc > 0) & np.isfinite(z) & (z > 0)
              & np.all(np.isfinite(pos), 1) & np.all(np.isfinite(vel), 1))
        stats["n_read"] += e - s
        z, mm, mc, pos, vel = z[ok], mm[ok], mc[ok], pos[ok], vel[ok]
        stats["n_kept"] += len(z)
        r = np.sqrt((pos ** 2).sum(1))
        v2 = ((pos * vel).sum(1) / r / C_KMS) ** 2
        stats["v2_sum"] += v2.sum()
        stats["n_lowz"] += int((z < 0.1).sum())
        iz = np.searchsorted(ZE, z, side="right") - 1
        mws = mm * np.interp(z, zt, rt)
        th = cos.theta200c(mws, z)
        sel = (mws > 1e13) & (th > 0.5 * ARCMIN)
        stats["n_ws"] += int(sel.sum())
        for tag, m, g in (("phys", mc, np.ones(len(z), bool)), ("ws", mws, sel)):
            lm = np.log10(m)
            im = np.searchsorted(LE, lm, side="right") - 1
            gg = g & (im >= 0) & (im < NM) & (iz >= 0) & (iz < NZ)
            if (g & ~gg).any():
                raise RuntimeError(f"{tag}: {int((g & ~gg).sum())} halos outside histogram range")
            idx = im[gg] * NZ + iz[gg]
            acc[f"{tag}_n"] += np.bincount(idx, minlength=NM * NZ)
            acc[f"{tag}_v2"] += np.bincount(idx, weights=v2[gg], minlength=NM * NZ)
            acc[f"{tag}_v2lm"] += np.bincount(idx, weights=(v2 * lm)[gg], minlength=NM * NZ)
            acc[f"{tag}_v2z"] += np.bincount(idx, weights=(v2 * z)[gg], minlength=NM * NZ)
        vz_n += np.bincount(iz, minlength=NZ); vz_v2 += np.bincount(iz, weights=v2, minlength=NZ)
        print(f"rows {s}-{e}  ws-selected so far {stats['n_ws']}  elapsed {time.time() - t0:.0f} s", flush=True)
save = {k: v.reshape(NM, NZ) for k, v in acc.items()}
save.update(logm_edges=LE, z_edges=ZE, vz_n=vz_n, vz_v2=vz_v2, ratio_z=zt, ratio_c7=rt)
np.savez(out, **save)
print("stats", stats, "vlos_rms_kms", np.sqrt(stats["v2_sum"] / stats["n_kept"]) * C_KMS)
print("saved", out, "elapsed", time.time() - t0)
