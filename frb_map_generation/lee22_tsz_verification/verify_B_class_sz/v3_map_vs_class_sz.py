"""Recompute the map-level band ratios from the stored map spectra and the workstream's CLASS-SZ npz files.

Band = arithmetic mean over integer ell in [0.9 l, 1.1 l] (the workstream's definition). CLASS-SZ spectra (log-spaced
ell, dlogell 0.05) are interpolated in log C - log ell onto integer ell. Maps: fullsky_spectra.npz (map2alm lmax 8192,
iter 3, pixel window NOT deconvolved - read from its metadata). Pixel window NSIDE 4096 from healpy.
Lee22 'as painted' in CLASS-SZ has no single run (two mass pieces); bracket its 2-halo cross term:
  lower = 1h_low + 1h_high + 2h_low + 2h_high;  upper = lower + 2 sqrt(2h_low 2h_high)  (Cauchy-Schwarz).
"""
import json
import numpy as np
import healpy as hp

CS = "/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_tsz_verification_20260929/B_class_sz/class_sz/"
OUT = "/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_tsz_verification_20260929/verify_B_class_sz/"
SP = "/home/cbllover/HalfDome/frb_map_generation/outputs/tsz_dm_fullsky_20260918/spectra/fullsky_spectra.npz"
ELLS = [30, 100, 300, 1000, 2000, 3000, 5000]

sp = np.load(SP, allow_pickle=True)
meta = json.loads(str(sp["metadata_json"]))
ell = sp["ell"].astype(float)

def band(cl, l):
    s = (ell >= 0.9 * l) & (ell <= 1.1 * l)
    return float(np.mean(cl[s]))

def cs(name):
    d = np.load(CS + name + ".npz", allow_pickle=True)
    out = {}
    for t in ["cl_1h", "cl_2h"]:
        c = np.interp(np.log(ell[2:]), np.log(d["ell"]), np.log(np.clip(d[t], 1e-300, None)))
        out[t] = np.concatenate([[0, 0], np.exp(c)])
    return out

pw2 = hp.pixwin(4096, lmax=8192) ** 2
maps = {k: sp[k] for k in ["cl_y_b12", "cl_y_lee22p", "cl_y_lee22p_calib"]}
res = {"pixel_window_in_spectra_metadata": meta.get("pixel_window"), "ells": ELLS, "bands": {}, "ratios": {}}
B = {}
for k, v in maps.items():
    B[k] = [band(v, l) for l in ELLS]
    B[k + "_pixwin_deconv"] = [band(v / pw2, l) for l in ELLS]
runs = {}
for n in ["b12_paint_all", "b12_paint_all_hiprec", "b12_paint_all_medprec", "lee22_paint_low_medprec", "lee22_paint_high_medprec",
          "lee22_paint_calib_medprec", "lee22m1_paint_low_medprec", "lee22m1_paint_high_medprec", "lee22m1_paint_calib_medprec",
          "lee22_paint_low", "lee22_paint_high", "b12_paint_cosmomodel_nodamp", "lee22_paint_low_hiprec"]:
    runs[n] = cs(n)
def tot(n):
    return runs[n]["cl_1h"] + runs[n]["cl_2h"]
B["cs_b12_hiprec"] = [band(tot("b12_paint_all_hiprec"), l) for l in ELLS]
B["cs_b12_medprec"] = [band(tot("b12_paint_all_medprec"), l) for l in ELLS]
B["cs_b12_default"] = [band(tot("b12_paint_all"), l) for l in ELLS]
B["cs_b12_cosmomodel_default"] = [band(tot("b12_paint_cosmomodel_nodamp"), l) for l in ELLS]
for pre in ["lee22", "lee22m1"]:
    lo, hi = runs[pre + "_paint_low_medprec"], runs[pre + "_paint_high_medprec"]
    lower = lo["cl_1h"] + hi["cl_1h"] + lo["cl_2h"] + hi["cl_2h"]
    upper = lower + 2 * np.sqrt(lo["cl_2h"] * hi["cl_2h"])
    B[f"cs_{pre}_lower"] = [band(lower, l) for l in ELLS]
    B[f"cs_{pre}_upper"] = [band(upper, l) for l in ELLS]
    B[f"cs_{pre}_calib"] = [band(tot(pre + "_paint_calib_medprec"), l) for l in ELLS]
B["pixwin2"] = [band(pw2, l) for l in ELLS]
res["bands"] = B
r = lambda a, b: [x / y for x, y in zip(B[a], B[b])]
R = res["ratios"]
R["B12 map / CLASS-SZ hiprec (raw)"] = r("cl_y_b12", "cs_b12_hiprec")
R["B12 map / CLASS-SZ medprec (raw)"] = r("cl_y_b12", "cs_b12_medprec")
R["B12 map / CLASS-SZ default (raw)"] = r("cl_y_b12", "cs_b12_default")
R["B12 map pixwin-deconvolved / CLASS-SZ hiprec"] = r("cl_y_b12_pixwin_deconv", "cs_b12_hiprec")
R["CLASS-SZ cosmo_model / massless (default)"] = r("cs_b12_cosmomodel_default", "cs_b12_default")
R["CLASS-SZ default / hiprec"] = r("cs_b12_default", "cs_b12_hiprec")
R["Lee22/B12 maps"] = r("cl_y_lee22p", "cl_y_b12")
R["Lee22 calib / B12 maps"] = r("cl_y_lee22p_calib", "cl_y_b12")
R["Lee22/B12 CLASS-SZ (lower 2h cross)"] = r("cs_lee22_lower", "cs_b12_hiprec")
R["Lee22/B12 CLASS-SZ (upper 2h cross)"] = r("cs_lee22_upper", "cs_b12_hiprec")
R["Lee22 map / CLASS-SZ (lower)"] = r("cl_y_lee22p", "cs_lee22_lower")
R["Lee22 map / CLASS-SZ (upper)"] = r("cl_y_lee22p", "cs_lee22_upper")
R["Lee22 calib map / CLASS-SZ calib medprec"] = r("cl_y_lee22p_calib", "cs_lee22_calib")
R["M1/as painted CLASS-SZ (lower)"] = r("cs_lee22m1_lower", "cs_lee22_lower")
R["M1/as painted CLASS-SZ (upper)"] = r("cs_lee22m1_upper", "cs_lee22_upper")
R["M1/B12 CLASS-SZ (lower)"] = r("cs_lee22m1_lower", "cs_b12_hiprec")
R["M1 calib / as painted calib CLASS-SZ"] = r("cs_lee22m1_calib", "cs_lee22_calib")
json.dump(res, open(OUT + "v3_map_vs_class_sz.json", "w"), indent=1)
print("pixel window metadata:", meta.get("pixel_window"))
print("ell      " + "".join(f"{l:>9d}" for l in ELLS))
for k, v in R.items():
    print(f"{k:50s}" + "".join(f"{x:9.3f}" for x in v))
print("pixwin^2 " + "".join(f"{x:9.3f}" for x in B["pixwin2"]))
