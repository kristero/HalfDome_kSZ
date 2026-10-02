"""Engine check: the earlier painter's coarse histogram (0.025 x 0.025 bins, bin centres; h = 0.68) through this engine, to compare
with poisson_mass_split.jl (logs/poisson.log) and to measure the bin-centre bias against the fine centroid histogram."""
import sys, os
import numpy as np, h5py
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

OUT = "/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_ksz_review_20261002/prediction/"
H = 0.68
cos = Cosmo(H, 0.049, 0.261)
ELLS = np.array([291., 300., 973., 1000., 3031., 5000.])
f = h5py.File("/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_ksz_20261002/maps/halo_mass_redshift_histogram.h5", "r")
le, ze = f["logm_edges"][:], f["z_edges"][:]
cnt, v2 = f["count"][:].T, f["sum_vlos_over_c_squared"][:].T      # (im, iz)
im, iz = np.nonzero(v2 > 0)
lm = 0.5 * (le[im] + le[im + 1]); zc = 0.5 * (ze[iz] + ze[iz + 1]); w = v2[im, iz]


def work(a):
    model, l, z = a
    return tau_ell_cell(cos, 10 ** l, z, ELLS, model, 4.0, shape_clip=10 ** 14.8 / H)


if __name__ == "__main__":
    ref = {"b16_xgp": (0.1105, 0.3547, 0.6469), "lee22corr": (0.1442, 0.5407, 1.2426), "lee22pipe": (0.7355, 1.2596, 0.8822)}
    old = np.load(OUT + "onehalo_old.npz")
    with Pool(4) as p:
        for model, oname in (("b16_xgp", "b16"), ("lee22corr", "lee22corr"), ("lee22pipe", "lee22pipe")):
            t = np.array(p.map(work, [(model, a, b) for a, b in zip(lm, zc)], chunksize=64))
            dl = dl_from_cl(ELLS, (w[:, None] * t ** 2).sum(0) / (4 * np.pi))
            fine = np.array([dl_from_cl(L, old["cl_" + oname][old["ell"] == L][0]) for L in ELLS])
            print(f"{model:10s} coarse-centre D_ell at {list(ELLS.astype(int))}: {np.round(dl, 4)}")
            print(f"{'':10s} / poisson_mass_split.jl (291, 973, 3031): {np.round(dl[[0, 2, 4]] / np.array(ref[model]), 4)}")
            print(f"{'':10s} fine-centroid / coarse-centre: {np.round(fine / dl, 4)}")
