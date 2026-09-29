#!/usr/bin/env python3
"""verify_0 / step 3: evaluate Lee22 eq. 9 + eq. 10 directly (independent Python, scipy quadrature) and compare with
(a) XGPaint's unmodified ne2d painted in step 2 for three amplitude mappings, and (b) the repository implementation.

Two versions of rho_cr(z):
  printed : 3 H0^2/(8 pi G) [Om (1+z)^3 + OL], OL = 1 - Om      (Lee22 eq. before eq. 7, as printed)
  xgprad  : the same plus radiation, OL = 1 - Om - Or, with Or from Tcmb=2.7255, Neff=3.046, exactly as
            XGPaint.get_cosmology builds it (the cosmology XGPaint and the repository code actually use).
"""
import csv
import json
from pathlib import Path

import numpy as np
from scipy.integrate import quad

ROOT = Path("/home/cbllover/HalfDome/frb_map_generation")
OUT = ROOT / "outputs/lee22_normalization_forensics_20260928/verify_0"
K = dict(l.strip().split("=") for l in open(OUT / "step2_constants.txt"))
K = {k: float(v) for k, v in K.items()}
MSUN, G, MP, ME, MPC = K["M_sun_kg"], K["G_SI"], K["m_p_kg"], K["m_e_kg"], K["Mpc_m"]
XH = 0.76


def rho_crit(z, h, Om, radiation):
    H0 = 100e3 * h / MPC  # s^-1
    if radiation:
        Tcmb, Neff = 2.7255, 3.046
        Og = 4.48131e-7 * Tcmb ** 4 / h ** 2
        Or = Og * (1 + Neff * (7 / 8) * (4 / 11) ** (4 / 3))
    else:
        Or = 0.0
    OL = 1 - Om - Or
    E2 = Om * (1 + z) ** 3 + Or * (1 + z) ** 4 + OL
    return 3 * H0 ** 2 / (8 * np.pi * G) * E2, E2


def g(x, xc, bp):
    s = x / xc
    return s ** -0.3 * (1 + s) ** (-bp)


def los(X, xc, bp, lmax=1e5):
    # 2 int_0^lmax g(sqrt(X^2+l^2)) dl, substitution l = X*sinh(u) for accuracy at small and large l
    umax = np.arcsinh(lmax / X)
    f = lambda u: g(X * np.cosh(u), xc, bp) * X * np.cosh(u)
    val, err = quad(f, 0, umax, epsabs=0, epsrel=1e-13, limit=500)
    return 2 * val


rows = list(csv.DictReader(open(OUT / "step2_xgpaint_ne2d.csv")))
m_per_e = ME + 2 * MP / (1 + XH)
ratio_expected = 0.9 * XH * MP / m_per_e
print(f"0.9 X_H m_p / m_per_e from XGPaint's CODATA2018 constants = {ratio_expected:.13f}")
print(f"claim: 0.6016316602  ; 1/ratio = {1/ratio_expected:.10f} (claim 1.6621)")

res = []
for r in rows:
    Oc, Ob, h = {"halfdome": (0.261, 0.049, 0.68), "tng": (0.3089 - 0.0486, 0.0486, 0.6774)}[r["cosmo"]]
    Om = Oc + Ob
    fb = Ob / Om
    M = float(r["M_msun"]); z = float(r["z"]); x = float(r["x"]); F = float(r["F"])
    # parameters from the TeX (step-1 grid) -- look them up again from step1
    res.append((r, Om, fb, h, M, z, x, F))

grid = {(row["cosmo"], float(row["M_msun"]), round(float(row["z"]), 3)): row for row in csv.DictReader(open(OUT / "step1_param_grid.csv"))}

out_rows = []
maxdev = {}
cache = {}
for (r, Om, fb, h, M, z, x, F) in res:
    p = grid[(r["cosmo"], M, round(z, 3))]
    n0, xc, bp = float(p["n0"]), float(p["xc"]), float(p["beta_prime"])
    key = (r["cosmo"], M, z, x)
    if key not in cache:
        vals = {}
        for rad in (False, True):
            rc, E2 = rho_crit(z, h, Om, rad)
            R200 = (3 * M * MSUN / (4 * np.pi * 200 * rc)) ** (1 / 3)
            n200 = 200 * rc * fb / (XH * MP)
            col = n0 * n200 * R200 * los(x, xc, bp)
            n3d = n0 * n200 * g(x, xc, bp)
            vals[rad] = (col, n3d, R200, rc, E2)
        cache[key] = vals
    v = cache[key]
    xgp = float(r["xgp_ne2d_m2"])
    rec = dict(cosmo=r["cosmo"], M=M, z=z, x=x, F=F,
               xgp_over_eq9_printed=xgp / v[False][0],
               xgp_over_eq9_xgprad=xgp / v[True][0],
               repo_lit_over_eq9_xgprad=float(r["repo_literal_col_m2"]) / v[True][0],
               repo_lit_over_eq9_printed=float(r["repo_literal_col_m2"]) / v[False][0],
               repo_nat_over_repo_lit=float(r["repo_xgpnative_col_m2"]) / float(r["repo_literal_col_m2"]),
               repo_nat3d_over_lit3d=float(r["repo_xgpnative_n3d_m3"]) / float(r["repo_literal_n3d_m3"]),
               repo_lit3d_over_eq9_xgprad=float(r["repo_literal_n3d_m3"]) / v[True][1],
               repo_lit3d_over_eq9_printed=float(r["repo_literal_n3d_m3"]) / v[False][1],
               r200_xgp_over_mine=float(r["r200c_m"]) / v[True][2],
               rhoc_xgp_over_mine_rad=float(r["rho_crit_kgm3"]) / v[True][3],
               E2_rad_over_printed=v[True][4] / v[False][4],
               repo_n0_over_tex=float(r["repo_n0"]) / n0, repo_xc_over_tex=float(r["repo_xc"]) / xc,
               repo_bp_over_tex=float(r["repo_beta_prime"]) / bp,
               repo_P0_over_200n0=float(r["repo_xgp_P0"]) / (200 * n0),
               repo_beta_minus_bp=float(r["repo_xgp_beta"]) - bp)
    out_rows.append(rec)

with open(OUT / "step3_comparison.csv", "w") as f:
    w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
    w.writeheader(); w.writerows(out_rows)

summary = {"ratio_expected_0.9XHmp_over_mpere": ratio_expected, "inverse": 1 / ratio_expected}
def stats(key, sel):
    a = np.array([o[key] for o in out_rows if sel(o)])
    return dict(min=float(a.min()), max=float(a.max()), n=int(a.size))

for Fv, lab in ((1.0, "F=1 (production P0=200n0)"), (1.6621, "F=1.6621"), (1 / 0.6016316602183194, "F=exact")):
    sel = lambda o, Fv=Fv: abs(o["F"] - Fv) < 1e-9
    for key in ("xgp_over_eq9_xgprad", "xgp_over_eq9_printed"):
        summary[f"{lab} | {key}"] = stats(key, sel)
    # redshift dependence of the printed-rho_cr comparison
    for zz in (0.0, 0.5, 1.0, 2.0, 4.0):
        summary[f"{lab} | xgp_over_eq9_printed at z={zz}"] = stats("xgp_over_eq9_printed", lambda o, Fv=Fv, zz=zz: abs(o["F"] - Fv) < 1e-9 and abs(o["z"] - zz) < 1e-9)
all_ = lambda o: abs(o["F"] - 1) < 1e-12
for key in ("repo_lit_over_eq9_xgprad", "repo_lit_over_eq9_printed", "repo_nat_over_repo_lit", "repo_nat3d_over_lit3d",
            "repo_lit3d_over_eq9_xgprad", "repo_lit3d_over_eq9_printed", "r200_xgp_over_mine", "rhoc_xgp_over_mine_rad",
            "E2_rad_over_printed", "repo_n0_over_tex", "repo_xc_over_tex", "repo_bp_over_tex", "repo_P0_over_200n0",
            "repo_beta_minus_bp"):
    summary[key] = stats(key, all_)
json.dump(summary, open(OUT / "step3_summary.json", "w"), indent=1)
for k, v in summary.items():
    print(k, v)
