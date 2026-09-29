"""Verifier A, check 2: production interpolator values (workstream CSV, produced by Julia from the production caches)
vs MY OWN LOS integral (vlib.py, written from the printed fits, not lee22_models.py); my own rho_crit/dA with radiation
as Cosmology.jl; plus Ysph(<R200c) Lee22/B12 ratios and 4R200c aperture capture fractions."""
import csv, json, sys
import numpy as np
from scipy.integrate import quad
sys.path.insert(0, "/home/cbllover/HalfDome/frb_map_generation/lee22_tsz_verification/verify_A_xgpaint_audit")
import vlib

A = "/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_tsz_verification_20260929/A_xgpaint_audit"
O = "/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_tsz_verification_20260929/verify_A_xgpaint_audit"
rows = list(csv.DictReader(open(f"{A}/production_interpolator_eval.csv")))
cos = {"b12": vlib.Cos(neff=3.04), "lee22": vlib.Cos(neff=3.046)}
cos0 = vlib.Cos(rad=False)
res = {}
for mod in ("b12", "lee22"):
    R = [r for r in rows if r["model"] == mod]
    m = 10 ** np.array([float(r["log10M200c_msun"]) for r in R]); z = np.array([float(r["z"]) for r in R])
    x = np.array([float(r["x_r200c"]) for r in R]); th = np.array([float(r["theta_rad"]) for r in R])
    yi = np.array([float(r["y_interpolator"]) for r in R]); yd = np.array([float(r["y_xgpaint_direct"]) for r in R])
    c = cos[mod]
    r200 = c.r200_m(m, z) / vlib.MPC; dA = c.dA_mpc(z); th200 = np.arctan(r200 / dA)
    X = th / th200            # the painter evaluates theta = x*theta200(XGPaint); I recompute X from my own theta200
    ymine = np.array([vlib.y_of_X(mod, mi, zi, np.array([Xi]), c)[0] for mi, zi, Xi in zip(m, z, X)])
    ylit = np.array([vlib.y_of_X(mod, mi, zi, np.array([xi]), cos0)[0] for mi, zi, xi in zip(m, z, x)])
    rel_i = yi / ymine - 1; rel_d = yd / ymine - 1; rel_lit = yi / ylit - 1
    worst = int(np.argmax(np.abs(rel_i)))
    res[mod] = dict(
        n=len(R),
        max_abs_interp_vs_mine=float(np.max(np.abs(rel_i))), rms_interp_vs_mine=float(np.sqrt(np.mean(rel_i ** 2))),
        worst_point=dict(log10M=float(np.log10(m[worst])), z=float(z[worst]), x=float(x[worst]), rel=float(rel_i[worst])),
        max_abs_interp_vs_mine_x_le_1p34=float(np.max(np.abs(rel_i[x <= 1.34]))),
        max_abs_interp_vs_mine_excl_logM15=float(np.max(np.abs(rel_i[np.log10(m) < 14.99]))),
        max_abs_xgpdirect_vs_mine=float(np.max(np.abs(rel_d))),
        max_abs_interp_vs_mine_norad_literal=float(np.max(np.abs(rel_lit))),
        rhocrit_csv_over_mine_minus1_maxabs=float(np.max(np.abs(np.array([float(r["rho_crit_xgpaint_kg_m3"]) for r in R]) / c.rho_crit(z) - 1))),
        dA_csv_over_mine_minus1_maxabs=float(np.max(np.abs(np.array([float(r["dA_xgpaint_mpc"]) for r in R]) / dA - 1))),
        r200_csv_over_mine_minus1_maxabs=float(np.max(np.abs(np.array([float(r["r200c_mpc"]) for r in R]) / r200 - 1))),
        theta200_csv_over_mine_minus1_maxabs=float(np.max(np.abs(np.array([float(r["theta200_rad"]) for r in R]) / th200 - 1))),
        rhocrit_rad_over_norad_minus1=[float(c.rho_crit(zz) / cos0.rho_crit(zz) - 1) for zz in (0.1, 0.5, 1, 2, 3)],
    )
    print(mod, json.dumps(res[mod], indent=1))

# Ysph(<R200c) Lee22/B12 at fixed (M, z): P200, R200 common, ratio of p0 int_0^1 g x^2 dx
def ysph(mod, mm, zz, xmax=1.0):
    p0, xc, beta = (vlib.b12_params if mod == "b12" else vlib.lee_params)(mm, zz)
    return p0 * quad(lambda s: vlib.g3d(s, xc, beta) * s * s, 0, xmax, epsabs=0, epsrel=1e-11, limit=200)[0]
ratios = {}
for lm in (13.0, 13.5, 14.0, 14.5, 15.0):
    for zz in (0.0, 0.5, 1.0, 2.0):
        r = ysph("lee22", 10 ** lm, zz) / ysph("b12", 10 ** lm, zz)
        ratios[f"logM{lm}_z{zz}"] = dict(as_painted=r, x1pz2=r * (1 + zz) ** 2)
print(json.dumps({k: v for k, v in ratios.items() if k.startswith(("logM14.0", "logM14.5", "logM13.5"))}, indent=0))
res["Ysph_R200c_ratio_lee22_over_b12"] = ratios

# aperture capture: Ycyl(<4)/Ytot, Ytot = int 4 pi x^2 g (to 1e5); Ycyl(<4) = Ytot - int_{r>...} via cylinder integral
def capture(mod, mm, zz):
    p0, xc, beta = (vlib.b12_params if mod == "b12" else vlib.lee_params)(mm, zz)
    f = lambda s: vlib.g3d(s, xc, beta) * s * s
    edges = [0] + list(np.logspace(-6, 5, 45))
    tot = sum(quad(f, a, b, epsabs=0, epsrel=1e-12, limit=200)[0] for a, b in zip(edges[:-1], edges[1:])) * 4 * np.pi
    Xg = np.logspace(-6, np.log10(4.0), 3000)
    I = vlib.los(Xg, xc, beta)
    cyl = 2 * np.pi * np.trapz(I * Xg, Xg)   # int_0^4 2 pi X I(X) dX
    return cyl / tot
cap = {mod: {f"logM{lm}_z{zz}": capture(mod, 10 ** lm, zz) for lm in (13.0, 14.0, 15.0) for zz in (0.1, 2.0)} for mod in ("b12", "lee22")}
print(json.dumps(cap, indent=1))
res["aperture_capture_Ycyl4_over_Ytot"] = cap
json.dump(res, open(f"{O}/v2_interpolator_and_ratios.json", "w"), indent=1)
