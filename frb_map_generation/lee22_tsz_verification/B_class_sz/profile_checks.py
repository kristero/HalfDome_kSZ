#!/usr/bin/env python3
"""Profile-level verification of the Battaglia12 and Lee22 tSZ profiles: XGPaint vs lee22_models vs CLASS-SZ.

A. XGPaint Compton-y y(theta) (xgpaint_profile_points.jl, production model constructors) against an independent
   line-of-sight integral of lee22_models.b12_Pe_over_P200 / lee_Pbar (P_e/P200) x P200, full LOS (as XGPaint).
B. CLASS-SZ P_200 (get_P_delta_at_m_and_z_b12, eV cm^-3) against 0.5176 x P200 (Battaglia thermal P200 x P_e/P_th).
C. CLASS-SZ's own tabulated y_ell(M, z) (x_outSZ = 4 sphere) against the Python transform (halo_model.py) for B12
   and for the Lee22 CLASS-SZ parameterization (common.lee22_class_sz_pars), at 1e13/1e14/1e15 Msun, z 0.1-2.
D. The Lee22 -> CLASS-SZ parameter mapping, evaluated with CLASS-SZ's formula (class_sz_tools.c) on a dense
   (x, M, z) grid spanning M_cut and the shape clip, against lee22_models.lee_Pbar/0.5176 (what CLASS-SZ must hold).
E. Physical profile table: P_e(r) [keV cm^-3] for B12 and Lee22 at the test points (for the figure) and
   integrated Y_sph(<R200c, <4R200c) ratios Lee22/B12.
Outputs: OUT/profiles/profile_checks.json, profile_curves.npz.
"""
import json, sys
from pathlib import Path
import numpy as np
from scipy import integrate

sys.path.insert(0, "/home/cbllover/HalfDome/frb_map_generation/lee22_tsz_verification/B_class_sz")
import common as C  # noqa: E402
import halo_model as HM  # noqa: E402
LM = HM.LM
OUTD = Path(C.OUT) / "profiles"; OUTD.mkdir(parents=True, exist_ok=True)
CS = Path(C.OUT) / "class_sz"
h = C.PAINT["h"]
COS = LM.Cosmo("paint", C.PAINT["Omega_b"] + C.PAINT["Omega_c"], C.PAINT["Omega_b"], h)
EV = 1.602176634e-12
KEV = 1e3 * EV
SIGT_ME = HM.SIGMA_T / HM.ME_C2


def pe_over_p200(model, x, mphys, z):
    if model == "b12":
        return LM.b12_Pe_over_P200(x, mphys, z)
    if model == "lee22":
        return LM.lee_Pbar(x, mphys * h, z, which="noc", shape_clip_hinv=C.LEE_CLIP_HINV)
    raise ValueError(model)


def los(model, x, mphys, z, lmax=1e5):
    f = lambda l: pe_over_p200(model, np.sqrt(x * x + l * l), mphys, z)
    # split the infinite LOS at scales where the integrand changes character
    pts = [0, x, 1, 4, 30, 300, lmax]
    pts = sorted(set([p for p in pts if p <= lmax]))
    return 2 * sum(integrate.quad(f, a, b, limit=400, epsrel=1e-11, epsabs=0)[0] for a, b in zip(pts[:-1], pts[1:]) if b > a)


def check_A():
    """XGPaint y vs independent LOS integral. XGPaint's own R200/rho_crit (radiation included) are used for P200,
    and lee22_models' matter+Lambda values are compared separately."""
    import csv
    rows = list(csv.DictReader(open(OUTD / "xgpaint_profile_points.csv")))
    out = []
    for r in rows:
        m, z, x = float(r["M200c_msun"]), float(r["z"]), float(r["x_over_theta200"])
        rc_x = float(r["rho_crit_g_cm3"]); r200_x = float(r["R200_mpc"]) * LM.MPC
        p200_x = 200 * LM.G * m * LM.MSUN * rc_x * float(r["f_b"]) / (2 * r200_x)
        y_py = SIGT_ME * p200_x * r200_x * los(r["model"], x, m, z)
        out.append(dict(model=r["model"], M=m, z=z, x=x, y_xgpaint=float(r["y"]), y_python=y_py,
                        ratio=float(r["y"]) / y_py,
                        rho_crit_xgp_over_matter_lambda=rc_x / COS.rho_crit(z),
                        r200_xgp_over_python=r200_x / COS.r_delta(m, z), f_b=float(r["f_b"])))
    return out


def check_B():
    d = np.load(CS / "b12_paint_all.npz")
    M, Z, P = d["test_M_msun"], d["test_z"], d["P200_b12"]
    out = []
    for i, m in enumerate(M):
        for j, z in enumerate(Z):
            p_py = C.PE_OVER_PTH * COS.P200(m, z) / EV          # eV cm^-3
            out.append(dict(M=float(m), z=float(z), P200e_class_sz_eV_cm3=float(P[i, j]), P200e_python=p_py,
                            ratio=float(P[i, j] / p_py)))
    # the CLASS-SZ constant 2.61051e-18: (300/8pi) (1e5/Mpc_cm)^2 Msun_g/Mpc_cm / eV  x 0.5176
    K = 300 / (8 * np.pi) * (1e5 / LM.MPC) ** 2 * LM.MSUN / LM.MPC / EV
    return dict(points=out, P200_constant_thermal_eV=K, P200_constant_x0p5176=K * C.PE_OVER_PTH,
                class_sz_constant=2.61051e-18, class_sz_over_0p5176K=2.61051e-18 / (K * C.PE_OVER_PTH),
                note="CLASS-SZ P_200 = M/R f_b 2.61051e-18 (100 h E)^2 eV/cm^3 already contains 0.5176, i.e. is the "
                     "ELECTRON-pressure normalization; its sigma_T/m_e c^2 factor 283.298/0.5176 undoes the 0.5176 "
                     "hidden in the szfast constant 283.298, so y = sigma_T/(m_e c^2) 0.5176 P200 x fit (= XGPaint)")


def check_C(ing):
    out = {}
    for run, tag, model in (("b12_paint_all", "b12", "b12"), ("lee22_paint_low", "lee22_low", "lee22"),
                            ("lee22_paint_high", "lee22_high", "lee22")):
        f = CS / f"{run}.npz"
        if not f.exists():
            continue
        d = np.load(f)
        M, Z, ELL, Y = d["test_M_msun"], d["test_z"], d["test_ell"], d[f"yell_{tag}"]
        rows = []
        for i, m in enumerate(M):
            mh = m * h
            if tag == "lee22_low" and mh >= C.LEE_CLIP_HINV:
                continue
            if tag == "lee22_high" and mh < C.LEE_CLIP_HINV:
                continue
            for j, z in enumerate(Z):
                y = HM.y_ell_grid(ing, model, np.array([z]), np.log(np.array([mh])), ELL)[0, 0]
                rows.append(dict(M=float(m), z=float(z), ell=ELL.tolist(), y_class_sz=Y[i, j].tolist(),
                                 y_python=y.tolist(), ratio_class_sz_over_python=(Y[i, j] / y).tolist()))
        out[tag] = rows
    return out


def class_sz_formula(x, m_hinv, z, p):
    """CLASS-SZ tabulate_gas_pressure_profile_B12_fft integrand (P/P200 fed to 0.5176 P200)."""
    mphys = m_hinv / h
    P0 = p["P0_B12"] * (mphys / 1e14) ** p["alpha_m_P0_B12"] * (1 + z) ** p["alpha_z_P0_B12"]
    xc = p["xc_B12"] * (mphys / 1e14) ** p["alpha_m_xc_B12"] * (1 + z) ** p["alpha_z_xc_B12"]
    beta = p["beta_B12"] * (mphys / 1e14) ** p["alpha_m_beta_B12"] * (1 + z) ** p["alpha_z_beta_B12"]
    val = P0 * (x / xc) ** p["gamma_B12"] * (1 + (x / xc) ** p["alpha_B12"]) ** (-beta)
    if p.get("use_broken_pressure", 0) == 1:
        mb = p["M_break_pressure"] / h
        val = np.where(mphys < mb, val * (mphys / mb) ** p["alpha_break_pressure"], val)
    return val


def check_D():
    x = np.geomspace(0.01, 4, 40)[:, None, None]
    mh = np.geomspace(C.CAT_M_MIN, C.CAT_M_MAX, 301)[None, :, None]
    z = np.linspace(0.0, 3.86, 40)[None, None, :]
    X, MH, Z = np.broadcast_arrays(x, mh, z)
    lee = LM.lee_Pbar(X, MH, Z, which="noc", shape_clip_hinv=C.LEE_CLIP_HINV) / C.PE_OVER_PTH
    cs = np.where(MH < C.LEE_CLIP_HINV, class_sz_formula(X, MH, Z, C.lee22_class_sz_pars(h, "low")),
                  class_sz_formula(X, MH, Z, C.lee22_class_sz_pars(h, "high")))
    m1 = np.where(MH < C.LEE_CLIP_HINV, class_sz_formula(X, MH, Z, C.lee22_class_sz_pars(h, "low", 2.0)),
                  class_sz_formula(X, MH, Z, C.lee22_class_sz_pars(h, "high", 2.0)))
    b12cs = class_sz_formula(X, MH, Z, C.b12_class_sz_pars())
    b12lm = LM.b12_Pth_over_P200(X, MH / h, Z)
    return dict(lee22_max_rel_dev=float(np.max(np.abs(cs / lee - 1))),
                lee22_m1_max_rel_dev=float(np.max(np.abs(m1 / (lee * (1 + Z) ** 2) - 1))),
                b12_max_rel_dev=float(np.max(np.abs(b12cs / b12lm - 1))),
                grid="x 0.01-4 (40), M200c 4.46e12-2.57e15 Msun/h (301, spans Mcut 10^13.6 and clip 10^14.8), z 0-3.86 (40)")


def table_E():
    x = np.geomspace(0.02, 4.0, 200)
    out = dict(x=x)
    ms, zs = np.array([1e13, 1e14, 1e15]), np.array([0.1, 0.5, 1.0, 2.0])
    Yr = np.zeros((3, 4, 2))
    for i, m in enumerate(ms):
        for j, z in enumerate(zs):
            p200 = COS.P200(m, z) / KEV
            for mod in ("b12", "lee22"):
                out[f"Pe_keV_{mod}_{i}_{j}"] = pe_over_p200(mod, x, m, z) * p200
            for k, X in enumerate((1.0, 4.0)):
                yy = [integrate.quad(lambda s: 4 * np.pi * s * s * pe_over_p200(mod, s, m, z), 0, X, limit=200)[0]
                      for mod in ("lee22", "b12")]
                Yr[i, j, k] = yy[0] / yy[1]
    out["M"], out["z"], out["Ysph_lee22_over_b12_R200_4R200"] = ms, zs, Yr
    return out


if __name__ == "__main__":
    ing = HM.Ingredients()
    res = {"A_xgpaint_vs_python": check_A(), "B_P200": check_B(), "C_yell": check_C(ing), "D_mapping": check_D()}
    E = table_E()
    res["E_Ysph_ratio_lee22_over_b12"] = dict(M_msun=E["M"].tolist(), z=E["z"].tolist(),
                                              ratio=E["Ysph_lee22_over_b12_R200_4R200"].tolist(),
                                              axes="[M][z][R200c, 4R200c]")
    np.savez(OUTD / "profile_curves.npz", **E)
    json.dump(res, open(OUTD / "profile_checks.json", "w"), indent=1)
    A = res["A_xgpaint_vs_python"]
    for mod in ("b12", "lee22"):
        r = np.array([a["ratio"] for a in A if a["model"] == mod])
        print(f"A {mod}: XGPaint/python y ratio min {r.min():.8f} max {r.max():.8f}")
    print("A rho_crit XGPaint/matter+Lambda range", min(a["rho_crit_xgp_over_matter_lambda"] for a in A),
          max(a["rho_crit_xgp_over_matter_lambda"] for a in A))
    B = res["B_P200"]
    print("B P200 ratio", min(p["ratio"] for p in B["points"]), max(p["ratio"] for p in B["points"]),
          "const ratio", B["class_sz_over_0p5176K"])
    for tag, rows in res["C_yell"].items():
        rr = np.array([r["ratio_class_sz_over_python"] for r in rows])
        print(f"C {tag}: CLASS-SZ/python y_ell ratio range {rr.min():.4f}-{rr.max():.4f} (by ell: {np.round(rr.mean(0), 4)})")
    print("D", res["D_mapping"])
    print("E Y ratio", np.round(E["Ysph_lee22_over_b12_R200_4R200"], 3))
