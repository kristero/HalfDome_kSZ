"""Independent check of the Fig. 2d/2e caption ratios with lee22_models.py (not the replica/grid scripts)."""
import sys, json
import numpy as np
sys.path.insert(0, "/home/cbllover/HalfDome/frb_map_generation/lee22_normalization_forensics/C_pressure_consistency")
import lee22_models as L
h = 0.6774; XH = 0.76
x = np.geomspace(0.04, 1.34, 400); mid = (x > 0.3) & (x < 1); core = x < 0.1
out = {}
for M in (2e13, 1e14, 5e14):
    for z in (0.0, 1.0, 2.0):
        Pc = L.lee_Pbar(x, M * h, z) * (1 + z) ** 2
        Pb = L.b12_Pe_over_P200(x, M, z)
        nc = L.lee_nbar(x, M * h, z) * (h * (1 + z)) ** 3 * XH ** 2
        nb = L.b16_nbar(x, M, z)
        rp = Pb / Pc; rn = nb / nc
        gm = lambda r: float(np.exp(np.mean(np.log(r[mid]))))
        out[f"{M:.0e}_z{z:g}"] = dict(P_mid=[round(float(rp[mid].min()), 3), round(float(rp[mid].max()), 3), round(gm(rp), 3)],
                                       P_core=[round(float(rp[core].min()), 2), round(float(rp[core].max()), 2)],
                                       P_out134=round(float(rp[-1]), 2),
                                       n_mid=[round(float(rn[mid].min()), 3), round(float(rn[mid].max()), 3), round(gm(rn), 3)],
                                       n_core=[round(float(rn[core].min()), 2), round(float(rn[core].max()), 2)])
for k, v in out.items(): print(k, v)
allP = [v["P_mid"][i] for v in out.values() for i in (0, 1)]; allN = [v["n_mid"][i] for v in out.values() for i in (0, 1)]
print("P mid range", min(allP), max(allP), " n mid range", min(allN), max(allN))
print("pipeline/corrected density:", [round(0.6016317 / ((h * (1 + z)) ** 3 * XH ** 2), 3) for z in (0, 1, 2)])
json.dump(out, open("/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_verify_v4_20261001/revision_audit/grid_ratios.json", "w"), indent=1)
