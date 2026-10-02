#!/usr/bin/env python3
"""Expected NSIDE-4096 pixel updates per 1M-row chunk of the production run (disc areas / pixel area), to extrapolate job 600479's walltime."""
import sys, numpy as np, h5py
src = open(sys.argv[1]).read().split('# ------------------------------------------------------------------ catalogue')[0]
sys.argv = ['x', '--catalog', 'none', '--tag', 't', '--out', '.']
exec(src)
om = 4 * np.pi / (12 * 4096**2)
out = []
with h5py.File('/home/cbllover/HalfDome/lightcone_100.hdf5', 'r') as f:
    N = f['redshift'].shape[0]
    for s in range(0, N, 1_000_000):
        e = min(s + 1_000_000, N)
        mc = f['halo_mass_m200c'][s:e] / H; mm = f['halo_mass_m200m'][s:e] / H; z = f['redshift'][s:e].astype(float)
        dA = d_A(z); th = np.arctan(r200c(mc, z) / dA)
        mws = mm * np.interp(z, zr, ratio_r); thw = np.arctan(r200c(mws, z) / dA)
        cut = (mws > 1e13) & (thw > 0.5 * ARCMIN)
        Rc = (mm * MSUN / (4 * np.pi * OM * RHO_C0 * (1 + z)**3))**(1 / 3); thc = np.arctan(Rc / dA)
        thmax = np.maximum(4 * th, np.where(cut, np.maximum(1.5 * thw, thc), 0))
        area = 2 * np.pi * (1 - np.cos(np.minimum(thmax, np.pi)))
        out.append((e, (area / om).sum(), cut.sum(), np.sum(z < 0.1), z.min()))
out = np.array(out)
np.savetxt(sys.argv[2] if len(sys.argv) > 2 else 'pixel_load.txt', out, header='last_row expected_pixel_updates websky_selected n_z<0.1 zmin')
c = np.cumsum(out[:, 1])
print('total expected pixel updates %.4e; websky-selected %d; z<0.1 halos %d' % (c[-1], out[:, 2].sum(), out[:, 3].sum()))
for k in (9, 19, 29, 39, 49, 59, 69, 79, 84):
    if k < len(out): print('after row %d: cumulative expected %.4e (%.1f%%)' % (out[k, 0], c[k], 100 * c[k] / c[-1]))
