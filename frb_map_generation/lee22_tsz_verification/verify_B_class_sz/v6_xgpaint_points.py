"""Check the workstream's XGPaint y(theta) points (B_class_sz/profiles/xgpaint_profile_points.csv) against my own LOS
integral: y = sigma_T/(m_e c^2) * P200 * R200 * P0 * int ds p(sqrt(X^2+s^2)), with (a) XGPaint's own R200, rho_crit, f_b
from the CSV and (b) my own geometry (flat LCDM Om 0.31, h 0.68, no radiation). B12: P0 x 0.5176 (thermal -> electron);
Lee22: fit is P_e/P200 (h^-1 Msun masses = M_phys x 0.68)."""
import csv, json
import numpy as np
from scipy.integrate import quad
import sys
sys.path.insert(0, "/home/cbllover/HalfDome/frb_map_generation/lee22_tsz_verification/verify_B_class_sz")
G = 6.67430e-8; MSUN = 1.98847e33; MPC = 3.0856775814913673e24; SIGMA_T = 6.6524587321e-25; MEC2 = 8.1871057769e-7
H = 0.68; OM = 0.31; FB = 0.049 / 0.31

def pars(model, M, z):
    zp = 1 + z
    if model == "b12":
        m = M / 1e14
        return 0.5176 * 18.1 * m ** 0.154 * zp ** -0.758, 0.497 * m ** -0.00865 * zp ** 0.731, 4.35 * m ** 0.0393 * zp ** 0.415
    mh = M * H; mcut = 10 ** 13.6; r = mh / mcut; rs = min(mh, 10 ** 14.8) / mcut
    return 2.8 * (r ** 1.08 if r < 1 else r ** 0.80) * zp ** -1.89, 2.10 * rs ** -0.35 * zp ** 0.53, 9.4 * rs ** -0.06 * zp ** 0.60

def los(X, xc, beta):
    f = lambda s: (np.sqrt(X * X + s * s) / xc) ** -0.3 * (1 + np.sqrt(X * X + s * s) / xc) ** -beta
    return 2 * (quad(f, 0, 1, limit=400, epsrel=1e-12)[0] + quad(f, 1, np.inf, limit=400, epsrel=1e-12)[0])

rows = list(csv.DictReader(open("/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_tsz_verification_20260929/B_class_sz/profiles/xgpaint_profile_points.csv")))
out = []
for r in rows:
    M, z, X = float(r["M200c_msun"]), float(r["z"]), float(r["x_over_theta200"])
    P0, xc, beta = pars(r["model"], M, z)
    # (a) XGPaint geometry
    R = float(r["R200_mpc"]) * MPC; rc = float(r["rho_crit_g_cm3"]); fb = float(r["f_b"])
    P200 = 200 * G * M * MSUN * rc * fb / (2 * R)
    ya = SIGMA_T / MEC2 * P200 * R * P0 * los(X, xc, beta)
    # (b) my geometry
    H0 = 100 * H * 1e5 / MPC; rcm = 3 * H0 ** 2 / (8 * np.pi * G) * (OM * (1 + z) ** 3 + 1 - OM)
    Rm = (3 * M * MSUN / (4 * np.pi * 200 * rcm)) ** (1 / 3); P200m = 200 * G * M * MSUN * rcm * FB / (2 * Rm)
    yb = SIGMA_T / MEC2 * P200m * Rm * P0 * los(X, xc, beta)
    out.append({"model": r["model"], "M": M, "z": z, "X": X, "xgp/mine_xgpgeom": float(r["y"]) / ya, "xgp/mine_owngeom": float(r["y"]) / yb})
res = {}
for mdl in ["b12", "lee22"]:
    a = np.array([o["xgp/mine_xgpgeom"] for o in out if o["model"] == mdl]); b = np.array([o["xgp/mine_owngeom"] for o in out if o["model"] == mdl])
    res[mdl] = {"xgp/mine (XGPaint geometry) min,max": [a.min(), a.max()], "xgp/mine (own geometry) min,max": [b.min(), b.max()], "n": int(a.size)}
print(json.dumps(res, indent=1))
json.dump({"summary": res, "points": out}, open("/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_tsz_verification_20260929/verify_B_class_sz/v6_xgpaint_points.json", "w"), indent=1)
