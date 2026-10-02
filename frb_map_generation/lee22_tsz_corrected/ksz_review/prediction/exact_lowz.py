"""Exact per-halo one-halo term for the HalfDome halos at z < 0.1 (31-39 % of the one-halo power at ell = 300, where (M, z) bin
centres are biased), for the same variants as onehalo_predict.py. Usage: python exact_lowz.py new|old"""
import sys, os, time
import numpy as np, h5py
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
import onehalo_predict as op          # setup from sys.argv[1]

ZMAX = 0.1
cat = "/home/cbllover/HalfDome/lightcone_100.hdf5"
with h5py.File(cat, "r") as f:
    zall = f["redshift"][:]
    idx = np.nonzero(zall < ZMAX)[0]
    z = zall[idx].astype(np.float64)
    mc = f["halo_mass_m200c"][idx] / op.H
    mm = f["halo_mass_m200m"][idx] / op.H
    pos = f["Position"][idx, :].astype(np.float64)
    vel = f["Velocity"][idx, :].astype(np.float64)
r = np.sqrt((pos ** 2).sum(1))
v2 = ((pos * vel).sum(1) / r / C_KMS) ** 2
mws = mm * np.interp(z, op.zt, op.rt)
sel_ws = (mws > 1e13) & (op.cos.theta200c(mws, z) > 0.5 * ARCMIN)


def work(a):
    name, M, zz, m200m = a
    tag, model, X, comp = op.VARIANTS[name]
    return tau_ell_cell(op.cos, M, zz, op.ELLS, model, X, comp_m200m=(m200m if comp else None), shape_clip=op.SHAPE_CLIP)


if __name__ == "__main__":
    t0 = time.time()
    print(f"halos z < {ZMAX}: {len(z)}; WebSky-selected {sel_ws.sum()}", flush=True)
    res = {"ell": op.ELLS}
    with Pool(4) as pool:
        for name, (tag, model, X, comp) in op.VARIANTS.items():
            s = sel_ws if tag == "ws" else np.ones(len(z), bool)
            M = mws[s] if tag == "ws" else mc[s]
            taus = np.array(pool.map(work, [(name, a, b, c) for a, b, c in zip(M, z[s], mm[s])], chunksize=32))
            cl = (v2[s, None] * taus ** 2).sum(0) / (4 * np.pi)
            # same z range from the fine centroid histogram of onehalo_predict.py
            h = np.load(op.OUT + f"onehalo_{op.setup}.npz")
            lm, zc, w, n = h["cells_" + name]; t = h["tau_" + name]; lo = zc < ZMAX
            clh = (w[lo, None] * t[lo] ** 2).sum(0) / (4 * np.pi)
            res["cl_" + name] = cl; res["clhist_" + name] = clh
            dl, dh = dl_from_cl(op.ELLS, cl), dl_from_cl(op.ELLS, clh)
            print(f"{name:26s} n={s.sum():6d} exact z<0.1 D_ell: " + " ".join(f"{int(L)}:{dl[op.ELLS == L][0]:.4f}" for L in op.ELL_T)
                  + "  | hist/exact: " + " ".join(f"{(dh / dl)[op.ELLS == L][0]:.3f}" for L in op.ELL_T) + f"  ({time.time() - t0:.0f} s)", flush=True)
    np.savez(op.OUT + f"exact_lowz_{op.setup}.npz", **res)
