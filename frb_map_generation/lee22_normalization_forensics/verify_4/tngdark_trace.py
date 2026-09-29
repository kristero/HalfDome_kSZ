"""Independent ray tracing of the TNG300-1-Dark FoF halo lightcone (1e13<=M200c/Msun<1e14, z<=1)
with the Lee22 no-c density (literal and xgpnative), recording each ray-halo crossing, so the
cumulative halo DM can be evaluated for any source redshift z_s <= 1.  No Tinker08 involved.

Geometry follows the production painter: theta_max = atan(R200c/D_A(z)), x = theta/theta_max,
DM = chord through the R200c sphere / (1+z).  Code is independent (scipy cKDTree + scipy quad).
"""
import math, time, json, numpy as np, h5py
from multiprocessing import Pool
from scipy.spatial import cKDTree
from common import TNG, lee22_chord_dm_literal, XGP_OVER_LITERAL

ROOT = '/home/cbllover/HalfDome/frb_map_generation/outputs/tng300_lee22_mass13to14_20260921/'
OUT = '/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_normalization_forensics_20260928/verify_4/'
c = TNG

def chord_worker(args):
    x, m, z = args
    return lee22_chord_dm_literal(c, x, m, z)

def main():
    t0 = time.time()
    with h5py.File(ROOT + 'ray_configuration.h5', 'r') as f:
        ra = f['frb_ra_rad'][:]; dec = f['frb_dec_rad'][:]
    ray = np.stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)], axis=1)
    with h5py.File(ROOT + 'tng_lightcone.h5', 'r') as f:
        pos = f['position_comoving_mpc'][:]; mass = f['mass_msun'][:]; zh = f['redshift'][:]
        hid = f['halo_id'][:]; snap = f['snapshot'][:]
    sel = (mass >= 1e13) & (mass < 1e14) & (zh > 0) & (zh <= 1)
    print('halos', len(mass), 'selected', sel.sum(), 'load %.1fs' % (time.time() - t0), flush=True)
    pos = pos[sel]; mass = mass[sel]; zh = zh[sel]
    chi = np.linalg.norm(pos, axis=1)
    # consistency of lightcone redshift with my chi(z)
    chk = np.random.default_rng(0).choice(len(chi), 20000, replace=False)
    rel = chi[chk] / c.chi(zh[chk]) - 1
    print('lightcone |pos| / chi_mine(z) - 1: median %.2e, max abs %.2e' % (np.median(rel), np.abs(rel).max()), flush=True)
    DA = c.chi(zh) / (1 + zh)
    r200 = c.r200c(mass, zh)
    thmax = np.arctan2(r200, DA)
    u = pos / chi[:, None]
    tree = cKDTree(ray)
    pairs_h, pairs_r = [], []
    chunk = 1_000_000
    for s in range(0, len(mass), chunk):
        e = min(len(mass), s + chunk)
        rr = 2 * np.sin(0.5 * thmax[s:e]) * (1 + 1e-9)
        lists = tree.query_ball_point(u[s:e], rr, workers=20)
        for i, l in enumerate(lists):
            if l:
                pairs_h.extend([s + i] * len(l)); pairs_r.extend(l)
    pairs_h = np.array(pairs_h, dtype=np.int64); pairs_r = np.array(pairs_r, dtype=np.int64)
    print('candidate pairs', len(pairs_h), '%.1fs' % (time.time() - t0), flush=True)
    uh = u[pairs_h]; ur = ray[pairs_r]
    theta = np.arctan2(np.linalg.norm(np.cross(uh, ur), axis=1), np.sum(uh * ur, axis=1))
    keep = theta <= thmax[pairs_h]
    pairs_h, pairs_r, theta = pairs_h[keep], pairs_r[keep], theta[keep]
    x = theta / thmax[pairs_h]
    print('crossings', len(x), flush=True)
    args = list(zip(x.tolist(), mass[pairs_h].tolist(), zh[pairs_h].tolist()))
    with Pool(20) as p:
        dm_lit = np.array(p.map(chord_worker, args, chunksize=500))
    print('chords done %.1fs' % (time.time() - t0), flush=True)
    np.savez(OUT + 'tngdark_crossings.npz', ray=pairs_r, halo=pairs_h, x=x, mass=mass[pairs_h], z=zh[pairs_h],
             dm_literal=dm_lit, n_rays=len(ray))
    # validation against stored production per-ray DM at z_s = 1
    with h5py.File(ROOT + 'tng/halo_dm.h5', 'r') as f:
        ref = f['dm_pc_cm3'][0, :]
        ra2 = f['frb_ra_rad'][:]
    assert np.array_equal(ra2, ra)
    mine = np.bincount(pairs_r, weights=XGP_OVER_LITERAL * dm_lit, minlength=len(ray))
    both = (ref > 0) | (mine > 0)
    print('validation z_s=1 vs production painter: mean mine %.4f ref %.4f; hit mine %.5f ref %.5f' %
          (mine.mean(), ref.mean(), (mine > 0).mean(), (ref > 0).mean()))
    r = mine[both] / np.where(ref[both] > 0, ref[both], np.nan)
    print('per-ray ratio mine/ref: median %.6f, 1st-99th pct %.5f %.5f, rays hit only in one: %d' %
          (np.nanmedian(r), np.nanpercentile(r, 1), np.nanpercentile(r, 99), int(np.sum((ref > 0) != (mine > 0)))))

if __name__ == '__main__':
    main()
