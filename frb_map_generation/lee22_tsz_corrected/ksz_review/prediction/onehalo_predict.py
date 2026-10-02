"""One-halo (catalogue Poisson) prediction of the paint_ksz_websky.jl maps, independent of XGPaint (review 2026-10-02):
  C_ell = (1/4pi) sum_cells sum_i (v_los,i/c)^2 tau_ell(M, z)^2,  cells at their (v/c)^2-weighted mean log10 M and z.
setup 'new': HalfDome cosmology (h 0.6774, Omega_b 0.0486, Omega_c 0.2603), the five idark variants.
setup 'old': the earlier maps' setup (h 0.68, Omega_b 0.049, Omega_c 0.261; 4 R200c; b16 / lee22corr / lee22pipe), used to
             validate this engine against poisson_mass_split.jl and to measure map / one-halo on the earlier maps.
Usage: python onehalo_predict.py new|old  (4 worker processes)."""
import sys, os, time
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

OUT = "/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_ksz_review_20261002/prediction/"
setup = sys.argv[1]
ELL_T = np.array([300., 500., 1000., 2000., 3000., 5000.])
ELL_G = np.unique(np.round(np.geomspace(30, 8192, 70)))
ELLS = np.unique(np.concatenate([ELL_T, ELL_G, [291., 973., 3031.]]))
MASS_BINS = np.array([12.0, 13.0, 13.5, 14.0, 14.5, 15.0, 15.7])
if setup == "new":
    H, OB, OC = 0.6774, 0.0486, 0.2603
    HIST = OUT + "halfdome_hist_h06774.npz"
    VARIANTS = {  # name: (histogram, model, X, websky)
        "b16_phys": ("phys", "b16_xgp", 4.0, False), "lee22corr_phys": ("phys", "lee22corr", 4.0, False),
        "lee22pipe_phys": ("phys", "lee22pipe", 4.0, False),
        "b16_websky": ("ws", "b16_ws", 1.5, True), "lee22corr_websky": ("ws", "lee22corr", 1.5, True),
        # diagnostics: WebSky selection and truncation without the compensating sphere
        "b16_websky_nocomp": ("ws", "b16_ws", 1.5, False), "lee22corr_websky_nocomp": ("ws", "lee22corr", 1.5, False),
    }
else:
    H, OB, OC = 0.68, 0.049, 0.261
    HIST = OUT + "halfdome_hist_h068_oldsetup.npz"
    VARIANTS = {"b16": ("phys", "b16_xgp", 4.0, False), "lee22corr": ("phys", "lee22corr", 4.0, False),
                "lee22pipe": ("phys", "lee22pipe", 4.0, False)}
SHAPE_CLIP = 10 ** 14.8 / H
cos = Cosmo(H, OB, OC)
d = np.load(HIST)
zt, rt = d["ratio_z"], d["ratio_c7"]


def cells(tag):
    n, v2 = d[f"{tag}_n"], d[f"{tag}_v2"]
    im, iz = np.nonzero(v2 > 0)
    w = v2[im, iz]
    return d[f"{tag}_v2lm"][im, iz] / w, d[f"{tag}_v2z"][im, iz] / w, w, n[im, iz]


def work(args):
    name, lm, z = args
    tag, model, X, comp = VARIANTS[name]
    M = 10 ** lm
    cm = M / np.interp(z, zt, rt) if comp else None
    return tau_ell_cell(cos, M, z, ELLS, model, X, comp_m200m=cm, shape_clip=SHAPE_CLIP)


if __name__ == "__main__":
    t0 = time.time()
    res = {"ell": ELLS, "mass_bins": MASS_BINS}
    with Pool(4) as pool:
        for name, (tag, model, X, comp) in VARIANTS.items():
            lm, z, w, n = cells(tag)
            taus = np.array(pool.map(work, [(name, a, b) for a, b in zip(lm, z)], chunksize=64))
            cl = (w[:, None] * taus ** 2).sum(0) / (4 * np.pi)
            b = np.clip(np.searchsorted(MASS_BINS, lm, side="right") - 1, 0, len(MASS_BINS) - 2)
            split = np.array([(w[b == k, None] * taus[b == k] ** 2).sum(0) / (4 * np.pi) for k in range(len(MASS_BINS) - 1)])
            res["cl_" + name] = cl; res["split_" + name] = split
            res["tau_" + name] = taus; res["cells_" + name] = np.stack([lm, z, w, n])
            dl = dl_from_cl(ELLS, cl)
            print(f"{name:26s} cells {len(lm):6d}  D_ell[uK^2] at " +
                  "  ".join(f"{int(L)}:{dl[ELLS == L][0]:.4f}" for L in list(ELL_T) + [291., 973., 3031.]) +
                  f"   ({time.time() - t0:.0f} s)", flush=True)
    np.savez(OUT + f"onehalo_{setup}.npz", **res)
    print("saved", OUT + f"onehalo_{setup}.npz")
