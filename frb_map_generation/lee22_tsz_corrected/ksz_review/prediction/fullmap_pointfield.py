"""Full-map (one-halo + all halo-halo cross terms) prediction at ell = 300, 500, 1000, 2000, without painting profiles (review 2026-10-02).
For spherically symmetric profiles a_lm = sum_i (v_i/c) tau_ell(M_i, z_i) Y*_lm(n_i) exactly, so at a fixed ell the painted map's power
equals that of a point field with weights w_i = (v_i/c) tau_ell(M_i, z_i). Bin the weights into an NSIDE-2048 density map:
  C_ell(anafast) = sum w^2 / 4pi  (shot noise = one-halo, exact)  +  pixwin_ell^2 x C_ell^{cross}  (+ negligible aliasing).
tau_ell(M, z) on a (log10 M, log z) grid with the engine of common.py (linear interpolation; compensated tau can be negative).
Usage: python fullmap_pointfield.py new|old"""
import sys, os, time
import numpy as np, h5py, healpy as hp
from multiprocessing import Pool
from scipy.interpolate import RegularGridInterpolator
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

OUT = "/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_ksz_review_20261002/prediction/"
setup = sys.argv[1]
ELL_T = np.array([300., 500., 1000., 2000.])
NSIDE = 2048
FRAC = 0.1
if setup == "new":
    H, OB, OC = 0.6774, 0.0486, 0.2603
    VARIANTS = {"b16_phys": ("phys", "b16_xgp", 4.0, False), "lee22corr_phys": ("phys", "lee22corr", 4.0, False),
                "lee22pipe_phys": ("phys", "lee22pipe", 4.0, False),
                "b16_websky": ("ws", "b16_ws", 1.5, True), "lee22corr_websky": ("ws", "lee22corr", 1.5, True)}
else:
    H, OB, OC = 0.68, 0.049, 0.261
    VARIANTS = {"b16": ("phys", "b16_xgp", 4.0, False), "lee22corr": ("phys", "lee22corr", 4.0, False),
                "lee22pipe": ("phys", "lee22pipe", 4.0, False)}
SHAPE_CLIP = 10 ** 14.8 / H
cos = Cosmo(H, OB, OC)
zt, rt = ratio_c7_table(cos.Om)
LMG = np.arange(12.55, 15.76, 0.01)
LZG = np.linspace(np.log(0.0019), np.log(3.9), 500)


def grid_work(a):
    name, lm, lz = a
    tag, model, X, comp = VARIANTS[name]
    M, z = 10 ** lm, np.exp(lz)
    return tau_ell_cell(cos, M, z, ELL_T, model, X, comp_m200m=(M / np.interp(z, zt, rt) if comp else None), shape_clip=SHAPE_CLIP)


if __name__ == "__main__":
    t0 = time.time()
    tabs = {}
    tabfile = OUT + f"fullmap_tau_grid_{setup}.npz"
    if os.path.exists(tabfile):
        tabs = dict(np.load(tabfile))
    with Pool(4) as pool:
        for name in [n for n in VARIANTS if n not in tabs]:
            g = pool.map(grid_work, [(name, a, b) for a in LMG for b in LZG], chunksize=256)
            tabs[name] = np.array(g).reshape(len(LMG), len(LZG), len(ELL_T))
            print(f"tau grid {name} done ({time.time() - t0:.0f} s)", flush=True)
    np.savez(tabfile, **tabs)
    with h5py.File("/home/cbllover/HalfDome/lightcone_100.hdf5", "r") as f:
        z = f["redshift"][:].astype(np.float64)
        mc = f["halo_mass_m200c"][:] / H
        mm = f["halo_mass_m200m"][:] / H
        pos = f["Position"][:]
        vel = f["Velocity"][:]
    pos64 = pos.astype(np.float64)
    r = np.sqrt((pos64 ** 2).sum(1))
    v = (pos64 * vel.astype(np.float64)).sum(1) / r / C_KMS
    pix = hp.vec2pix(NSIDE, pos64[:, 0], pos64[:, 1], pos64[:, 2])
    del pos, vel, pos64, r
    mws = mm * np.interp(z, zt, rt)
    sel_ws = (mws > 1e13) & (cos.theta200c(mws, z) > 0.5 * ARCMIN)
    lz = np.log(z)
    print(f"catalogue loaded: {len(z)} halos, ws-selected {sel_ws.sum()} ({time.time() - t0:.0f} s)", flush=True)
    npix = hp.nside2npix(NSIDE); opix = 4 * np.pi / npix
    pw = hp.pixwin(NSIDE, lmax=int(ELL_T.max() * (1 + FRAC)) + 2)
    res = {"ell": ELL_T}
    for name, (tag, model, X, comp) in VARIANTS.items():
        s = sel_ws if tag == "ws" else np.ones(len(z), bool)
        lmass = np.log10(mws[s] if tag == "ws" else mc[s])
        pts = np.column_stack([lmass, lz[s]])
        F = []; SN = []; TOT = []
        for j, L in enumerate(ELL_T):
            itp = RegularGridInterpolator((LMG, LZG), tabs[name][:, :, j], method="linear", bounds_error=True)
            w = v[s] * itp(pts)
            sn = (w ** 2).sum() / (4 * np.pi)
            m = np.bincount(pix[s], weights=w, minlength=npix) / opix
            lmax = int(L * (1 + FRAC)) + 1
            cl = hp.anafast(m, lmax=lmax, iter=1)
            e = np.arange(lmax + 1); win = (e >= L * (1 - FRAC)) & (e <= L * (1 + FRAC))
            cross = ((cl[win] - sn) / pw[:lmax + 1][win] ** 2).mean()
            F.append(1 + cross / sn); SN.append(sn); TOT.append(sn + cross)
            print(f"{name:18s} ell {int(L):5d}: one-halo D = {dl_from_cl(L, sn):.4f}  full D = {dl_from_cl(L, sn + cross):.4f}  "
                  f"map/1h = {1 + cross / sn:.4f}  ({time.time() - t0:.0f} s)", flush=True)
            del m, w
        res["F_" + name] = np.array(F); res["cl1h_" + name] = np.array(SN); res["clfull_" + name] = np.array(TOT)
    np.savez(OUT + f"fullmap_pointfield_{setup}.npz", **res)
    print("saved", OUT + f"fullmap_pointfield_{setup}.npz")
