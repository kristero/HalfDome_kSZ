"""Final prediction table for the idark paint_ksz_websky.jl maps (review 2026-10-02). All D_ell [uK^2] are +-10 % ell-window means.
 one-halo: onehalo_new.npz (fine centroid histogram, converged: exact per-halo z < 0.1 agrees to < 0.1 %)
 map, ell <= 2000: point-field full-map calculation (fullmap_pointfield_new.npz; one-halo + every halo-halo cross term; the same
                   method reproduces the earlier painted maps to 0.1-0.7 % at ell 300-2000)
 map, ell 3000/5000: one-halo x map/one-halo of the earlier painted maps for the phys variants (includes the pixel-sampling excess);
                   for the WebSky-recipe maps cross-term trend (~0.993 / 0.995) x the phys pixel-sampling excess of the same profile family
                   (B16 1.008 / 1.044, Lee22 1.000 / 1.013), quoted with a range."""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

OUT = "/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_ksz_review_20261002/prediction/"
ELL = np.array([300., 500., 1000., 2000., 3000., 5000.])
new = np.load(OUT + "onehalo_new.npz"); pf = np.load(OUT + "fullmap_pointfield_new.npz"); me = np.load(OUT + "map_over_onehalo_earlier.npz")
ws = np.loadtxt("/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_ksz_debug_20261002/websky_def/stein2020_fig6_vector_halo.csv", delimiter=",")
ws = ws[ws[:, 0] > 10]
vec = np.array([ws[np.abs(ws[:, 0] / L - 1) <= 0.1, 1].mean() for L in ELL])


def w1h(name, L, frac=0.1):
    e = np.arange(int(np.ceil(L * (1 - frac))), int(np.floor(L * (1 + frac))) + 1).astype(float)
    return dl_from_cl(e, np.exp(np.interp(np.log(e), np.log(new["ell"]), np.log(new["cl_" + name])))).mean()


FAM = {"b16_phys": "b16", "lee22corr_phys": "lee22corr", "lee22pipe_phys": "lee22pipe", "b16_websky": "b16", "lee22corr_websky": "lee22corr"}
HI = {"b16_websky": (0.99, 1.01, 1.00, 1.05), "lee22corr_websky": (0.99, 1.005, 1.00, 1.02)}   # F ranges at 3000, 5000
table = {"ell": ELL.tolist(), "stein_fig6_halo_vector": vec.tolist()}
lines = ["ell                    " + "".join(f"{int(L):>9d}" for L in ELL), "Stein+20 Fig.6 Halo    " + "".join(f"{x:9.4f}" for x in vec)]
for n, fam in FAM.items():
    d1 = np.array([w1h(n, L) for L in ELL])
    F = np.empty(6); lo = np.empty(6); hi = np.empty(6)
    F[:4] = pf["F_" + n]; lo[:4] = hi[:4] = F[:4]
    if n.endswith("_phys"):
        F[4:] = me["F_" + fam][4:]; lo[4:] = hi[4:] = F[4:]
    else:
        F[4] = 0.993 * me["F_" + fam][4]; F[5] = 0.995 * me["F_" + fam][5]
        lo[4], hi[4], lo[5], hi[5] = HI[n]
    mp = d1 * F
    table[n] = dict(onehalo=d1.tolist(), map_over_onehalo=F.tolist(), F_lo=lo.tolist(), F_hi=hi.tolist(), map=mp.tolist(),
                    map_lo=(d1 * lo).tolist(), map_hi=(d1 * hi).tolist(), onehalo_over_vec=(d1 / vec).tolist(), map_over_vec=(mp / vec).tolist())
    lines += ["", f"{n}", "  one-halo             " + "".join(f"{x:9.4f}" for x in d1),
              "  map/one-halo         " + "".join(f"{x:9.3f}" for x in F),
              "  expected map         " + "".join(f"{x:9.4f}" for x in mp),
              "  one-halo / Fig.6     " + "".join(f"{x:9.3f}" for x in d1 / vec),
              "  map / Fig.6          " + "".join(f"{x:9.3f}" for x in mp / vec)]
    if not n.endswith("_phys"):
        lines.append("  map range (3000,5000)" + " " * 36 + "".join(f"  {a:.3f}-{b:.3f}" for a, b in zip((d1 * lo)[4:], (d1 * hi)[4:])))
txt = "\n".join(lines)
print(txt)
open(OUT + "final_prediction_table.txt", "w").write(__doc__ + "\n" + txt + "\n")
json.dump(table, open(OUT + "final_prediction_table.json", "w"), indent=1)
