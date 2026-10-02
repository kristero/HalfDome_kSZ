#!/usr/bin/env python3
"""Per-thread pixel load of the production run's last chunks under Threads.@threads :static (16 threads, 1M-row chunks)."""
import sys, numpy as np, h5py
src = open(sys.argv[1]).read().split('# ------------------------------------------------------------------ catalogue')[0]
sys.argv = ['x', '--catalog', 'none', '--tag', 't', '--out', '.']
exec(src)
om = 4 * np.pi / (12 * 4096**2); NT = 16
with h5py.File('/home/cbllover/HalfDome/lightcone_100.hdf5', 'r') as f:
    N = f['redshift'].shape[0]
    tot_max = 0; tot_sum = 0
    for s in range(75_000_000, N, 1_000_000):
        e = min(s + 1_000_000, N)
        mc = f['halo_mass_m200c'][s:e] / H; mm = f['halo_mass_m200m'][s:e] / H; z = f['redshift'][s:e].astype(float)
        dA = d_A(z); th = np.arctan(r200c(mc, z) / dA)
        mws = mm * np.interp(z, zr, ratio_r); thw = np.arctan(r200c(mws, z) / dA)
        cut = (mws > 1e13) & (thw > 0.5 * ARCMIN)
        Rc = (mm * MSUN / (4 * np.pi * OM * RHO_C0 * (1 + z)**3))**(1 / 3); thc = np.arctan(Rc / dA)
        thmax = np.maximum(4 * th, np.where(cut, np.maximum(1.5 * thw, thc), 0))
        npx = 2 * np.pi * (1 - np.cos(np.minimum(thmax, np.pi))) / om
        loads = np.array([b.sum() for b in np.array_split(npx, NT)])
        nlow = np.array([np.sum(b < 0.1) for b in np.array_split(z, NT)])
        tot_max += loads.max(); tot_sum += loads.sum()
        print('rows %d-%d: total %.3e  max-thread %.3e  imbalance %.1f  z<0.1 per thread max %d  zmin %.4f' % (s + 1, e, loads.sum(), loads.max(), loads.max() / loads.mean(), nlow.max(), z.min()))
print('rows 75M-end: sum %.3e, sum of per-chunk max-thread loads %.3e (balanced would be %.3e)' % (tot_sum, tot_max, tot_sum / NT))
