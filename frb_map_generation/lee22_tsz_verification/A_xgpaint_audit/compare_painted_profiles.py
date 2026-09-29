"""Workstream A, goal 2 (Python half): independent line-of-sight Compton-y of the 3-D electron pressure from
lee22_models.py, compared with the production interpolators evaluated by eval_production_interpolators.jl.

  y(x) = sigma_T/(m_e c^2) * P200 * R200c * 2 int_0^Lmax p(sqrt(x^2 + l^2)) dl ,   x = r_perp / R200c
  p = P_e/P200:  B12  -> b12_Pe_over_P200(r, M200c[Msun], z)  (= 0.5176 x Battaglia12 thermal fit, P0 pivot 1e14 Msun)
                 Lee22-> lee_Pbar(r, M200c*h [h^-1 Msun], z)  (Table 7 no-c, P_e fit, Mcut 10^13.60 h^-1 Msun,
                         shape (xc, beta) frozen above 10^14.8 h^-1 Msun as in production)
  P200 = 200 G M200c rho_cr(z) f_b / (2 R200c), all physical; f_b = 0.049/0.31, h = 0.68.

Two rho_cr(z) readings: 'literal' = lee22_models (H^2 = H0^2 [Om(1+z)^3 + 1 - Om], no radiation), 'matched' =
XGPaint's rho_crit (Cosmology.jl FlatLCDM with photons + massless neutrinos, taken from the Julia CSV).
y ∝ P200 R200c = 100 G M rho_cr f_b, so the two differ by rho_xgp/rho_literal only.

Also computes the fraction of the total (untruncated) Y captured by the painter's 4 R200c projected aperture.
Run:  LD_LIBRARY_PATH=/home/cbllover/miniconda3/envs/halfdome/lib /home/cbllover/miniconda3/envs/halfdome/bin/python compare_painted_profiles.py
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import quad

sys.path.insert(0, "/home/cbllover/HalfDome/frb_map_generation/lee22_normalization_forensics/C_pressure_consistency")
import lee22_models as LM  # noqa: E402  (validated formulas; not modified)

OUT = Path("/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_tsz_verification_20260929/A_xgpaint_audit")
C = LM.HALFDOME
H = C.h
SIGMA_T = 6.6524587321e-25       # cm^2 (CODATA 2018, as XGPaint)
ME_C2 = 8.1871057769e-7          # erg (CODATA 2018)
LMAX = 1.0e5                     # R200c units, XGPaint _nfw_profile_los_quadrature zmax
BREAKS = [0.0, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 100.0, 1.0e3, 1.0e4, LMAX]


def pbar(model, r, m_msun, z):
    if model == "b12":
        return LM.b12_Pe_over_P200(r, m_msun, z)
    return LM.lee_Pbar(r, m_msun * H, z, which="noc", shape_clip_hinv=10 ** 14.8)


def los_integral(model, x, m, z):
    f = lambda l: float(pbar(model, np.sqrt(x * x + l * l), m, z))
    tot = 0.0
    for a, b in zip(BREAKS[:-1], BREAKS[1:]):
        v, _ = quad(f, a, b, epsabs=0.0, epsrel=1e-12, limit=400)
        tot += v
    return 2.0 * tot


def y_independent(model, x, m, z):
    """literal rho_cr (no radiation)"""
    return SIGMA_T / ME_C2 * C.P200(m, z) * C.r_delta(m, z) * los_integral(model, x, m, z)


def aperture_capture(model, m, z, X=4.0):
    """Y_cyl(< X R200c) / Y_total with the untruncated 3-D profile (sphere to 1e5 R200c)."""
    g = lambda r: float(pbar(model, r, m, z)) * r * r
    br = [1e-8, 0.01, 0.1, 0.5, 1.0, 2.0, X, 10.0, 100.0, 1e3, 1e5]
    tot = sum(quad(g, a, b, epsabs=0, epsrel=1e-11, limit=400)[0] for a, b in zip(br[:-1], br[1:]))
    inner = sum(quad(g, a, b, epsabs=0, epsrel=1e-11, limit=400)[0] for a, b in zip(br[:-1], br[1:]) if b <= X)
    h = lambda r: g(r) * (1.0 - np.sqrt(1.0 - (X / r) ** 2))
    outer = sum(quad(h, a, b, epsabs=0, epsrel=1e-11, limit=400)[0] for a, b in zip(br[:-1], br[1:]) if a >= X)
    y_sph1 = sum(quad(g, a, b, epsabs=0, epsrel=1e-11, limit=400)[0] for a, b in zip(br[:-1], br[1:]) if b <= 1.0)
    return (inner + outer) / tot, y_sph1 / tot


def main():
    rows = list(csv.DictReader(open(OUT / "production_interpolator_eval.csv")))
    out_rows = []
    for r in rows:
        model, lm, z, x = r["model"], float(r["log10M200c_msun"]), float(r["z"]), float(r["x_r200c"])
        m = 10 ** lm
        y_lit = y_independent(model, x, m, z)
        rho_lit = C.rho_crit(z) * 1e3  # g/cm^3 -> kg/m^3
        rho_xgp = float(r["rho_crit_xgpaint_kg_m3"])
        y_match = y_lit * rho_xgp / rho_lit
        yi, yd = float(r["y_interpolator"]), float(r["y_xgpaint_direct"])
        out_rows.append(dict(model=model, logM=lm, z=z, x=x, y_itp=yi, y_direct=yd, y_indep_literal=y_lit,
                             y_indep_matched=y_match, rho_ratio_xgp_over_literal=rho_xgp / rho_lit,
                             r200c_mpc_xgp=float(r["r200c_mpc"]), r200c_mpc_literal=C.r_delta(m, z) / LM.MPC,
                             inside=int(r["inside_aperture"])))
    keys = list(out_rows[0].keys())
    with open(OUT / "painted_profile_vs_independent_los.csv", "w") as fh:
        fh.write(",".join(keys) + "\n")
        for d in out_rows:
            fh.write(",".join(f"{d[k]:.12e}" if isinstance(d[k], float) else str(d[k]) for k in keys) + "\n")

    summary = {}
    for model in ("b12", "lee22"):
        sel = [d for d in out_rows if d["model"] == model]
        f = lambda a, b: np.array([d[a] / d[b] - 1.0 for d in sel])
        itp_match, dir_match, itp_dir, itp_lit = f("y_itp", "y_indep_matched"), f("y_direct", "y_indep_matched"), \
            f("y_itp", "y_direct"), f("y_itp", "y_indep_literal")
        rho = np.array([d["rho_ratio_xgp_over_literal"] - 1 for d in sel])
        worst = sel[int(np.argmax(np.abs(itp_match)))]
        summary[model] = {
            "n_points": len(sel),
            "max_abs_rel_interpolator_vs_independent_matched_rho": float(np.max(np.abs(itp_match))),
            "rms_rel_interpolator_vs_independent_matched_rho": float(np.sqrt(np.mean(itp_match ** 2))),
            "worst_point_interpolator": {k: worst[k] for k in ("logM", "z", "x")},
            "max_abs_rel_xgpaint_direct_vs_independent_matched_rho": float(np.max(np.abs(dir_match))),
            "max_abs_rel_interpolator_vs_xgpaint_direct": float(np.max(np.abs(itp_dir))),
            "max_abs_rel_interpolator_vs_independent_literal_rho": float(np.max(np.abs(itp_lit))),
            "mean_rel_interpolator_vs_independent_literal_rho": float(np.mean(itp_lit)),
            "rho_crit_xgpaint_over_literal_minus1_range": [float(rho.min()), float(rho.max())],
            "max_abs_rel_interpolator_vs_indep_matched_x_le_1p34": float(np.max(np.abs(itp_match[np.array([d['x'] <= 1.34 for d in sel])]))),
        }
    cap = {}
    for model in ("b12", "lee22"):
        cap[model] = {}
        for lm in (13.0, 13.5, 14.0, 14.5, 15.0):
            for z in (0.1, 0.5, 1.0, 2.0):
                c4, s1 = aperture_capture(model, 10 ** lm, z)
                cap[model][f"logM{lm}_z{z}"] = {"Ycyl_lt_4R200c_over_Ytot": c4, "Ysph_lt_R200c_over_Ytot": s1}
    summary["aperture_capture"] = cap
    summary["conventions"] = ("x = r_perp/R200c (physical); M200c physical Msun (catalogue Msun/h divided by h=0.68); "
                              "P_e (electron) pressure; B12 = 0.5176 x thermal fit; Lee22 Table 7 fit is P_e/P200, painted "
                              "via P0_XG = P0/0.5176 and beta_XG = beta + 0.3; y = sigma_T/(m_e c^2) int P_e dl (physical dl); "
                              "LOS to 1e5 R200c (no 3-D truncation); the 4 R200c aperture is a projected cut, "
                              "y = 0 for theta >= 4 theta200.")
    json.dump(summary, open(OUT / "painted_profile_check_summary.json", "w"), indent=1)
    for model in ("b12", "lee22"):
        s = summary[model]
        print(model, {k: v for k, v in s.items() if k != "worst_point_interpolator"}, s["worst_point_interpolator"])
    for model in ("b12", "lee22"):
        vals = [(k, v["Ycyl_lt_4R200c_over_Ytot"], v["Ysph_lt_R200c_over_Ytot"]) for k, v in cap[model].items()]
        print(model, "aperture capture Ycyl(<4R200c)/Ytot range", min(v[1] for v in vals), max(v[1] for v in vals),
              "; Ysph(<R200c)/Ytot range", min(v[2] for v in vals), max(v[2] for v in vals))


if __name__ == "__main__":
    main()
