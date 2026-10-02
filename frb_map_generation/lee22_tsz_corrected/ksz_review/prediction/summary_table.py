"""Prediction table (review 2026-10-02): one-halo D_ell [uK^2] of the five idark maps at the target ells (+-10 % window means, like
map_factor.py), expected map/one-halo factors, expected map D_ell, and ratios to the exact Stein+20 Fig. 6 'Halo' vector."""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

OUT = "/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_ksz_review_20261002/prediction/"
ELL_T = np.array([300., 500., 1000., 2000., 3000., 5000.])
new = np.load(OUT + "onehalo_new.npz"); old = np.load(OUT + "onehalo_old.npz")
mf = np.load(OUT + "map_factor.npz"); me = np.load(OUT + "map_over_onehalo_earlier.npz")
ws = np.loadtxt("/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_ksz_debug_20261002/websky_def/stein2020_fig6_vector_halo.csv", delimiter=",")
ws = ws[ws[:, 0] > 10]


def w1h(ell, cl, L, frac=0.1):
    e = np.arange(int(np.ceil(L * (1 - frac))), int(np.floor(L * (1 + frac))) + 1).astype(float)
    return dl_from_cl(e, np.exp(np.interp(np.log(e), np.log(ell), np.log(cl)))).mean()


def wvec(L, frac=0.1):
    s = np.abs(ws[:, 0] / L - 1) <= frac
    return ws[s, 1].mean()


vec = np.array([wvec(L) for L in ELL_T]); vec3 = np.array([wvec(L, 0.03) for L in ELL_T])
rows = {}
print("ell                          " + "  ".join(f"{int(L):>7d}" for L in ELL_T))
print("Stein+20 Fig.6 Halo (+-10%)  " + "  ".join(f"{x:7.4f}" for x in vec))
print("Stein+20 Fig.6 Halo (+-3%)   " + "  ".join(f"{x:7.4f}" for x in vec3))
# factor range for the WebSky-recipe maps: low = model-transferred two-halo; high = phys factor of the same profile family
FAM = {"b16_websky": "b16", "lee22corr_websky": "lee22corr"}
for n in ("b16_phys", "lee22corr_phys", "lee22pipe_phys", "b16_websky", "lee22corr_websky", "b16_websky_nocomp", "lee22corr_websky_nocomp"):
    d1 = np.array([w1h(new["ell"], new["cl_" + n], L) for L in ELL_T])
    r = dict(onehalo=d1, onehalo_over_vec=d1 / vec)
    if "F_" + n in mf:
        F = mf["F_" + n]
        r["F"] = F; r["map"] = d1 * F; r["map_over_vec"] = d1 * F / vec
    rows[n] = r
    print(f"\n{n:28s} 1h  " + "  ".join(f"{x:7.4f}" for x in d1))
    print(f"{'':28s} 1h/vec " + " ".join(f"{x:7.3f}" for x in d1 / vec))
    if "F" in r:
        print(f"{'':28s} F   " + "  ".join(f"{x:7.3f}" for x in r["F"]))
        print(f"{'':28s} map " + "  ".join(f"{x:7.4f}" for x in r["map"]))
        print(f"{'':28s} map/vec " + "".join(f"{x:8.3f}" for x in r["map_over_vec"]))
    if n in FAM:
        Fp = me["F_" + FAM[n]]
        print(f"{'':28s} map if F=phys factor  " + "  ".join(f"{x:7.4f}" for x in d1 * Fp) + "   (upper bracket)")
print("\nnew/old one-halo (cosmology change, phys 4 R200c): " + "; ".join(
    f"{a}: " + " ".join(f"{w1h(new['ell'], new['cl_' + a + '_phys'], L) / w1h(old['ell'], old['cl_' + a], L):.3f}" for L in ELL_T)
    for a in ("b16", "lee22corr", "lee22pipe")))
json.dump({n: {k: [float(x) for x in v] for k, v in r.items()} for n, r in rows.items()} | {"ell": list(ELL_T), "stein_vec_pm10": list(vec)},
          open(OUT + "prediction_table.json", "w"), indent=1)
