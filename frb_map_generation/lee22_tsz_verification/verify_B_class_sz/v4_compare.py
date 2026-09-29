"""Compare the independent catalogue Poisson 1-halo terms (v2) with the maps and with the workstream's numbers.
Total = my catalogue 1h + CLASS-SZ Tinker08/Tinker10 2h (B12: b12_paint_all_hiprec; Lee22: medprec low+high 2h, lower
bracket without the cross term; calib: lee22_paint_calib_medprec). The 2h is 12 % of B12 at ell 100, 2 % at 1000."""
import json
import numpy as np

OUT = "/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_tsz_verification_20260929/verify_B_class_sz/"
CS = "/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_tsz_verification_20260929/B_class_sz/class_sz/"
v2 = json.load(open(OUT + "v2_catalogue_poisson.json"))
v3 = json.load(open(OUT + "v3_map_vs_class_sz.json"))
ELLS = v2["ells"]; C = {k: np.array(v) for k, v in v2["cl_1h_catalogue"].items()}
sp = np.load("/home/cbllover/HalfDome/frb_map_generation/outputs/tsz_dm_fullsky_20260918/spectra/fullsky_spectra.npz")
ell = sp["ell"].astype(float)

def band(cl, l):
    s = (ell >= 0.9 * l) & (ell <= 1.1 * l)
    return float(np.mean(cl[s]))

def cs2h(name):
    d = np.load(CS + name + ".npz", allow_pickle=True)
    c = np.exp(np.interp(np.log(ell[2:]), np.log(d["ell"]), np.log(d["cl_2h"])))
    c = np.concatenate([[0, 0], c])
    return np.array([band(c, l) for l in ELLS])
def cs1h(name):
    d = np.load(CS + name + ".npz", allow_pickle=True)
    c = np.exp(np.interp(np.log(ell[2:]), np.log(d["ell"]), np.log(d["cl_1h"])))
    c = np.concatenate([[0, 0], c])
    return np.array([band(c, l) for l in ELLS])
b12_2h = cs2h("b12_paint_all_hiprec")
lee_2h = cs2h("lee22_paint_low_medprec") + cs2h("lee22_paint_high_medprec")
calib_2h = cs2h("lee22_paint_calib_medprec")
m1_2h = cs2h("lee22m1_paint_low_medprec") + cs2h("lee22m1_paint_high_medprec")
mapB = np.array(v3["bands"]["cl_y_b12"]); mapL = np.array(v3["bands"]["cl_y_lee22p"]); mapC = np.array(v3["bands"]["cl_y_lee22p_calib"])
mapBpw = np.array(v3["bands"]["cl_y_b12_pixwin_deconv"])
R = {}
R["B12 map / my cat 1h"] = mapB / C["b12_all"]
R["B12 map / (my cat 1h + CS 2h)"] = mapB / (C["b12_all"] + b12_2h)
R["B12 map pixwin-deconvolved / (my cat 1h + CS 2h)"] = mapBpw / (C["b12_all"] + b12_2h)
R["Lee22 map / (my cat 1h + CS 2h)"] = mapL / (C["lee22_all"] + lee_2h)
R["Lee22 calib map / (my cat 1h + CS 2h)"] = mapC / (C["lee22_calib"] + calib_2h)
R["Lee22/B12 my cat (1h+CS 2h)"] = (C["lee22_all"] + lee_2h) / (C["b12_all"] + b12_2h)
R["Lee22/B12 my cat 1h only"] = C["lee22_all"] / C["b12_all"]
R["my cat 1h / CLASS-SZ hiprec 1h (B12)"] = C["b12_all"] / cs1h("b12_paint_all_hiprec")
R["my cat 1h / CLASS-SZ medprec 1h (Lee22 low+high)"] = C["lee22_all"] / (cs1h("lee22_paint_low_medprec") + cs1h("lee22_paint_high_medprec"))
R["M1/as painted my cat (1h + CS 2h)"] = (C["lee22m1_all"] + m1_2h) / (C["lee22_all"] + lee_2h)
R["M1/B12 my cat (1h + CS 2h)"] = (C["lee22m1_all"] + m1_2h) / (C["b12_all"] + b12_2h)
R["M1 calib/as painted calib, my cat 1h"] = C["lee22m1_calib"] / C["lee22_calib"]
tot_b = C["b12_all"] + b12_2h
R["share of Lee22/B12-1 from M>10^14.8/h (1h, over total B12)"] = (C["lee22_highmass"] - C["b12_highmass"]) / tot_b
R["share of Lee22/B12-1 from M>10^14.8/h and z<0.1 (1h)"] = (C["lee22_high_zlt0.1"] - C["b12_high_zlt0.1"]) / tot_b
for i in range(6):
    R[f"mass bin {i}: (L22-B12)/B12tot (1h)"] = (C[f"lee22_mbin{i}"] - C[f"b12_mbin{i}"]) / tot_b
for i in range(6):
    R[f"z bin {i}: (L22-B12)/B12tot (1h)"] = (C[f"lee22_zbin{i}"] - C[f"b12_zbin{i}"]) / tot_b
R["Lee22/B12 per mass bin, 1h, bin 5 (above clip)"] = C["lee22_mbin5"] / C["b12_mbin5"]
out = {"ells": ELLS, "ratios": {k: [float(x) for x in v] for k, v in R.items()}, "counts": v2["counts"],
       "per_halo_Y_ratio": v2["per_halo_Y_ratio"], "note": __doc__}
json.dump(out, open(OUT + "v4_compare.json", "w"), indent=1)
print("ell" + " " * 58 + "".join(f"{int(l):>8d}" for l in ELLS))
for k, v in R.items():
    print(f"{k:61s}" + "".join(f"{x:8.3f}" for x in v))
print(v2["counts"])
for r in v2["per_halo_Y_ratio"]:
    print(f"z={r['z']:.3f} M={r['M200c_msunh']:.3g} Y_L22/Y_B12={r['Y_lee22_over_b12']:.3f}")
