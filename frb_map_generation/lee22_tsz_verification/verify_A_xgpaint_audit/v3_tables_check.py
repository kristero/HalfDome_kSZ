"""Verifier A, check 3: recompute profile_tables.npz entries from scratch (own constants/formulas, no lee22_models.py)."""
import json
import numpy as np
A = "/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_tsz_verification_20260929/A_xgpaint_audit"
O = "/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_tsz_verification_20260929/verify_A_xgpaint_audit"
d = np.load(f"{A}/profile_tables.npz")
# CGS constants (CODATA 2018 / IAU)
G = 6.67430e-8; mp = 1.67262192369e-24; me = 9.1093837015e-28; Msun = 1.98847e33; Mpc = 3.0856775814913673e24
keV = 1.602176634e-9; h = 0.68; Om = 0.31; Ob = 0.049; fb = Ob / Om; XH = 0.76
lm = np.arange(12.5, 15.5001, 0.25); zz = np.array([0, 0.25, 0.5, 0.75, 1, 1.5, 2, 3.0])
x = d["x_r200c"]
M = 10 ** lm[:, None, None]; Z = zz[None, :, None]; X = x[None, None, :]
rhoc = 3 * (100e5 * h / Mpc) ** 2 / (8 * np.pi * G) * (Om * (1 + Z) ** 3 + 1 - Om)          # g/cm^3 physical, no radiation
R200 = (3 * M * Msun / (4 * np.pi * 200 * rhoc)) ** (1 / 3)
P200 = 200 * G * M * Msun * rhoc * fb / (2 * R200) / keV                                     # keV/cm^3
n200 = 200 * rhoc * fb / (XH * mp)
g = lambda s, xc, b, gam=-0.3: (s / xc) ** gam * (1 + s / xc) ** (-b)
# B12 (P_e = 0.5176 P_th)
m14 = M / 1e14
b12 = 0.5176 * 18.1 * m14 ** 0.154 * (1 + Z) ** -0.758 * g(X, 0.497 * m14 ** -0.00865 * (1 + Z) ** 0.731, 4.35 * m14 ** 0.0393 * (1 + Z) ** 0.415)
# Lee22 Table 7 (pressure) and Table 8 (density), masses in h^-1 Msun
Mh = M * h
rP = Mh / 10 ** 13.60; rPs = np.minimum(Mh, 10 ** 14.8) / 10 ** 13.60
leeP = 2.8 * np.where(rP < 1, rP ** 1.08, rP ** 0.80) * (1 + Z) ** -1.89 * g(X, 2.10 * rPs ** -0.35 * (1 + Z) ** 0.53, 9.4 * rPs ** -0.06 * (1 + Z) ** 0.60)
rN = Mh / 10 ** 13.61; rNs = np.minimum(Mh, 10 ** 14.8) / 10 ** 13.61
bpl = lambda r, lo, hi: np.where(r < 1, r ** lo, r ** hi)
leeN = 6.8 * rN ** 0.68 * (1 + Z) ** -2.11 * g(X, 7.9 * (1 + Z) ** -0.67 * bpl(rNs, 0.47, -0.45), 19.5 * (1 + Z) ** -0.31 * bpl(rNs, 0.70, -0.18))
# B16 XGPaint: rho_gas/(f_b rho_c) = P0 (x/xc)^gam [1+(x/xc)^al]^(-(be+gam)/al); n_e = 0.9 rho_gas/m_per_e
al = 0.88 * m14 ** -0.03 * (1 + Z) ** 0.19; be = 3.83 * m14 ** 0.04 * (1 + Z) ** -0.025
s = X / 0.5
b16rho = 4.0e3 * m14 ** 0.29 * (1 + Z) ** -0.66 * s ** -0.2 * (1 + s ** al) ** (-(be - 0.2) / al)
m_per_e = me + 2 * XH / (XH + 1) * mp + (1 - XH) / (2 * (1 + XH)) * 4 * mp
ne_b16 = 0.9 * b16rho * fb * rhoc / m_per_e
fac = 0.9 * XH * mp / m_per_e
C = 0.2841790493880425 * (1 + np.minimum(Z, 2)) ** 2.483190184608004
mine = {"Pbar_b12": b12, "Pe_keV_cm3_b12": b12 * P200, "Pbar_lee22_painted": leeP, "Pe_keV_cm3_lee22_painted": leeP * P200,
        "Pbar_lee22_x1pz2": leeP * (1 + Z) ** 2, "Pe_keV_cm3_lee22_x1pz2": leeP * (1 + Z) ** 2 * P200,
        "Pth_keV_cm3_b12": b12 / 0.5176 * P200,
        "nbar_lee22_literal": leeN, "ne_cm3_lee22_literal": leeN * n200, "nbar_lee22_xgpaint_ne2d": leeN * fac,
        "ne_cm3_lee22_xgpaint_ne2d": leeN * fac * n200, "nbar_lee22_tngcal": leeN * C, "ne_cm3_lee22_tngcal": leeN * C * n200,
        "ne_cm3_b16_xgpaint": ne_b16, "nbar_b16_xgpaint": ne_b16 / n200,
        "R200c_kpc": R200[:, :, 0] / (Mpc / 1e3), "P200_keV_cm3": P200[:, :, 0], "n200_cm3": n200[0, :, 0]}
out = {}
for k, v in mine.items():
    t = d[k]
    rel = np.max(np.abs(t / np.broadcast_to(v, t.shape) - 1))
    out[k] = float(rel)
    print(f"{k:28s} max|table/mine-1| = {rel:.2e}")
out["factor_xgpaint_ne2d_mine"] = fac; out["factor_table"] = float(d["factor_xgpaint_ne2d"])
# fit-range mask check: 1e13 <= M h <= 10^14.8, z <= 2, 0.04 <= x <= 1.34
mask = (Mh >= 1e13) & (Mh <= 10 ** 14.8) & (Z <= 2) & (X >= 0.04) & (X <= 1.34)
out["mask_mismatch_count"] = int(np.sum(np.broadcast_to(mask, d["in_lee22_fit_range"].shape) != d["in_lee22_fit_range"]))
print("factor", fac, d["factor_xgpaint_ne2d"], "mask mismatches", out["mask_mismatch_count"])
# spot values for the README
i14 = int(np.argmin(np.abs(lm - 14))); ix = int(np.argmin(np.abs(x - 0.1)))
print("1e14 Msun z=0: R200c", R200[i14, 0, 0] / (Mpc / 1e3), "kpc; P200", P200[i14, 0, 0], "keV/cm3; n200", n200[0, 0, 0], "cm^-3; x=", x[ix],
      "Pe_b12", (b12 * P200)[i14, 0, ix], "Pe_lee", (leeP * P200)[i14, 0, ix])
json.dump(out, open(f"{O}/v3_tables_check.json", "w"), indent=1)
