"""Verifier A: production Julia density (v5 CSV) vs my own formulas (as in v3), with and without XGPaint's radiation rho_crit."""
import csv, json
import numpy as np
O = "/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_tsz_verification_20260929/verify_A_xgpaint_audit"
G = 6.67430e-8; mp = 1.67262192369e-24; me = 9.1093837015e-28; Mpc = 3.0856775814913673e24
h = 0.68; Om = 0.31; fb = 0.049 / Om; XH = 0.76
g = lambda s, xc, b, gam=-0.3: (s / xc) ** gam * (1 + s / xc) ** (-b)
bpl = lambda r, lo, hi: np.where(r < 1, r ** lo, r ** hi)
m_per_e = me + 2 * XH / (XH + 1) * mp + (1 - XH) / (2 * (1 + XH)) * 4 * mp
fac = 0.9 * XH * mp / m_per_e
def mine(var, lm, z, x, rho):  # rho in g/cm^3 -> n_e in m^-3
    M = 10 ** lm; Mh = M * h; n200 = 200 * rho * fb / (XH * mp)
    if var == "b16_xgpaint":
        m14 = M / 1e14; al = 0.88 * m14 ** -0.03 * (1 + z) ** 0.19; be = 3.83 * m14 ** 0.04 * (1 + z) ** -0.025; s = x / 0.5
        return 0.9 * 4.0e3 * m14 ** 0.29 * (1 + z) ** -0.66 * s ** -0.2 * (1 + s ** al) ** (-(be - 0.2) / al) * fb * rho / m_per_e * 1e6
    rN = Mh / 10 ** 13.61; rNs = min(Mh, 10 ** 14.8) / 10 ** 13.61
    nb = 6.8 * rN ** 0.68 * (1 + z) ** -2.11 * g(x, 7.9 * (1 + z) ** -0.67 * bpl(rNs, 0.47, -0.45), 19.5 * (1 + z) ** -0.31 * bpl(rNs, 0.70, -0.18))
    f = {"lee22_literal": 1.0, "lee22_xgpaint_ne2d": fac, "lee22_tngcal": 0.2841790493880425 * (1 + min(z, 2)) ** 2.483190184608004}[var]
    return float(nb * f * n200 * 1e6)
out = {}
for r in csv.DictReader(open(f"{O}/v5_julia_density_points.csv")):
    lm, z, x = float(r["log10M"]), float(r["z"]), float(r["x"])
    rho_lit = 3 * (100e5 * h / Mpc) ** 2 / (8 * np.pi * G) * (Om * (1 + z) ** 3 + 1 - Om)
    rho_x = float(r["rho_crit_kg_m3"]) * 1e-3
    a = float(r["ne_m3"]); b_lit = mine(r["variant"], lm, z, x, rho_lit); b_x = mine(r["variant"], lm, z, x, rho_x)
    o = out.setdefault(r["variant"], dict(max_rel_vs_mine_literal_rho=0.0, max_rel_vs_mine_xgpaint_rho=0.0))
    o["max_rel_vs_mine_literal_rho"] = max(o["max_rel_vs_mine_literal_rho"], abs(a / b_lit - 1))
    o["max_rel_vs_mine_xgpaint_rho"] = max(o["max_rel_vs_mine_xgpaint_rho"], abs(a / b_x - 1))
print(json.dumps(out, indent=1)); json.dump(out, open(f"{O}/v5_compare.json", "w"), indent=1)
