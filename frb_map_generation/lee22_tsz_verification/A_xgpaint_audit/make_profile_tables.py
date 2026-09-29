"""Workstream A, goal 3: shared 3-D pressure and density tables for the other workstreams and the synthesis.

Grid: log10 M200c/Msun = 12.5..15.5 step 0.25 (physical Msun), z in {0,0.25,0.5,0.75,1,1.5,2,3},
      60 log-spaced radii x = r/R200c in [0.01, 5] (R200c physical, 200 x rho_cr(z)).
Arrays are (n_mass, n_z, n_x) unless stated.  Formulas: lee22_models.py (validated, not modified); HalfDome
cosmology Om = 0.31 (Ob 0.049 + Oc 0.261), h = 0.68, f_b = Ob/Om = 0.158065; rho_cr(z) = 3H0^2/(8 pi G)
[Om (1+z)^3 + 1 - Om] (no radiation; XGPaint's rho_crit is higher by 4e-5 (z=0.1) .. 8e-4 (z=2) .. 1.1e-3 (z=3)).

Pressure (ELECTRON pressure P_e, physical keV cm^-3; P200 = 200 G M rho_cr f_b/(2 R200c)):
  b12            0.5176 x Battaglia+12 AGN thermal fit (XGPaint Battaglia16ThermalSZProfile, as painted)
  lee22_painted  Lee22 Table 7 no-c P_e/P200 (Mcut 10^13.60 h^-1 Msun, M h in h^-1 Msun, shape frozen above
                 10^14.8 h^-1 Msun, P0 extrapolated with slope 0.80) = the painted Lee22 y map
  lee22_x1pz2    lee22_painted x (1+z)^2  (M1 code-level bookkeeping hypothesis for y_prof.cpp)
Density (FREE-ELECTRON density n_e, physical cm^-3; n200 = 200 rho_cr f_b/(X_H m_p), X_H = 0.76):
  b16_xgpaint    XGPaint BattagliaTauProfile gas density x 0.9/m_per_e (ne2d), exponent -(beta+gamma)/alpha
  lee22_literal  Lee22 Table 8 no-c n_e/n200, eq. 9 read literally (Mcut 10^13.61 h^-1, shape clip 10^14.8 h^-1)
  lee22_xgpaint_ne2d   lee22_literal x 0.6016316602183194 (production FRB/kSZ normalization)
  lee22_tngcal   lee22_literal x C(z), C = 0.2841790493880425 (1+min(z,2))^2.483190184608004 (tng_calibrated)
Run: LD_LIBRARY_PATH=/home/cbllover/miniconda3/envs/halfdome/lib /home/cbllover/miniconda3/envs/halfdome/bin/python make_profile_tables.py
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/cbllover/HalfDome/frb_map_generation/lee22_normalization_forensics/C_pressure_consistency")
import lee22_models as LM  # noqa: E402

OUT = Path("/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_tsz_verification_20260929/A_xgpaint_audit")
C = LM.HALFDOME
H = C.h
XGP_NE2D = 0.9 * LM.X_H * LM.M_P / LM.M_PER_E_XGP          # 0.6016316602...
C0, KTNG, ZCLIP = 0.2841790493880425, 2.483190184608004, 2.0
SHAPE_CLIP_HINV = 10 ** 14.8

LOGM = np.round(np.arange(12.5, 15.5 + 1e-9, 0.25), 4)
ZS = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0])
X = np.exp(np.linspace(np.log(0.01), np.log(5.0), 60))


def c_tng(z):
    return C0 * (1 + np.minimum(z, ZCLIP)) ** KTNG


def build():
    nm, nz, nx = len(LOGM), len(ZS), len(X)
    M = 10 ** LOGM
    shape = (nm, nz, nx)
    t = {k: np.zeros(shape) for k in ("Pbar_b12", "Pbar_lee22_painted", "Pbar_lee22_x1pz2", "Pbar_b12_thermal",
                                      "nbar_b16_xgpaint", "nbar_lee22_literal", "nbar_lee22_xgpaint_ne2d",
                                      "nbar_lee22_tngcal")}
    per = {k: np.zeros((nm, nz)) for k in ("R200c_kpc", "P200_keV_cm3", "kT200_keV")}
    n200 = np.array([C.n200(z) for z in ZS])
    rho = np.array([C.rho_crit(z) for z in ZS])
    for i, m in enumerate(M):
        for j, z in enumerate(ZS):
            t["Pbar_b12_thermal"][i, j] = LM.b12_Pth_over_P200(X, m, z)
            t["Pbar_b12"][i, j] = LM.b12_Pe_over_P200(X, m, z)
            t["Pbar_lee22_painted"][i, j] = LM.lee_Pbar(X, m * H, z, which="noc", shape_clip_hinv=SHAPE_CLIP_HINV)
            t["Pbar_lee22_x1pz2"][i, j] = t["Pbar_lee22_painted"][i, j] * (1 + z) ** 2
            t["nbar_b16_xgpaint"][i, j] = LM.b16_nbar(X, m, z)
            t["nbar_lee22_literal"][i, j] = LM.lee_nbar(X, m * H, z, which="noc", shape_clip_hinv=SHAPE_CLIP_HINV)
            t["nbar_lee22_xgpaint_ne2d"][i, j] = XGP_NE2D * t["nbar_lee22_literal"][i, j]
            t["nbar_lee22_tngcal"][i, j] = c_tng(z) * t["nbar_lee22_literal"][i, j]
            per["R200c_kpc"][i, j] = C.r_delta(m, z) / LM.KPC
            per["P200_keV_cm3"][i, j] = C.P200(m, z) / LM.KEV
            per["kT200_keV"][i, j] = C.kT200_keV(m, z)
    arrays = {}
    for k, v in t.items():
        arrays[k] = v
        if k.startswith("Pbar_"):
            arrays["Pe_keV_cm3_" + k[5:] if "thermal" not in k else "Pth_keV_cm3_b12"] = v * per["P200_keV_cm3"][:, :, None]
        else:
            arrays["ne_cm3_" + k[5:]] = v * n200[None, :, None]
    arrays.update(per)
    arrays.update(dict(log10_M200c_msun=LOGM, M200c_msun=M, M200c_hinv_msun=M * H, z=ZS, x_r200c=X,
                       n200_cm3=n200, rho_crit_g_cm3=rho, E2=np.array([C.E2(z) for z in ZS]),
                       C_tngcal=c_tng(ZS), factor_xgpaint_ne2d=np.array(XGP_NE2D)))
    fit = np.zeros((len(LOGM), len(ZS), len(X)), bool)
    mh = (M * H)[:, None, None]
    fit[:] = (mh >= 1e13) & (mh <= 10 ** 14.8) & (ZS[None, :, None] <= 2.0) & (X[None, None, :] >= 0.04) & (X[None, None, :] <= 1.34)
    arrays["in_lee22_fit_range"] = fit
    return arrays


def describe(arrays):
    d = {
        "file": "profile_tables.npz", "created": "2026-09-29", "producer": str(Path(__file__).resolve()),
        "axes": {"log10_M200c_msun": "physical M200c [Msun], 12.5..15.5 step 0.25 (axis 0)",
                 "z": "halo redshift (axis 1)", "x_r200c": "r/R200c, 60 log-spaced points 0.01..5 (axis 2); R200c physical"},
        "cosmology": {"Omega_m": C.om, "Omega_b": C.ob, "h": C.h, "f_b": C.fb, "X_H": LM.X_H,
                      "rho_crit": "3H0^2/(8piG)[Om(1+z)^3+1-Om], physical, no radiation (XGPaint adds radiation: +4e-5 at z=0.1 to +8e-4 at z=2, +1.1e-3 at z=3)",
                      "M200c": "200 x rho_crit(z), physical Msun; Lee22 masses in h^-1 Msun are M200c_hinv_msun = M200c_msun x h",
                      "P200": "200 G M200c rho_crit(z) f_b / (2 R200c) (Battaglia12/Lee22 eq. 9), physical",
                      "n200": "200 rho_crit(z) f_b / (X_H m_p) (Lee22 eq. 9 as printed), physical cm^-3",
                      "P_e_vs_P_th": "all Pe_* arrays are ELECTRON pressure; B12 is 0.5176 x its thermal fit (0.5176 = mu/mu_e for X_H=0.76); Lee22 Table 7 fits P_e directly (y_prof.cpp electron pressure); Pth_keV_cm3_b12 is the B12 thermal pressure",
                      "mu_e": LM.MU_E, "mu": LM.MU, "m_per_e_xgpaint_g": LM.M_PER_E_XGP},
        "arrays": {}}
    units = {"Pbar": "P_e/P200 (dimensionless)", "Pe_keV_cm3": "P_e [keV cm^-3], physical", "Pth_keV": "P_th [keV cm^-3], physical",
             "nbar": "n_e/n200 (dimensionless, n200 as above)", "ne_cm3": "n_e [cm^-3], physical free-electron density"}
    what = {
        "b12": "Battaglia+12 AGN fit x 0.5176 (XGPaint Battaglia16ThermalSZProfile params 18.1/0.154/-0.758, 0.497/-0.00865/0.731, 4.35/0.0393/0.415, gamma -0.3, alpha 1; mass pivot 1e14 physical Msun) = painted B12 y map",
        "b12_thermal": "Battaglia+12 thermal P_th/P200",
        "lee22_painted": "Lee22 Table 7 no-c P_e/P200 exactly as painted (P0 BPL 1.08/0.80 about Mcut=10^13.60 h^-1 Msun, (1+z)^-1.89; xc, beta frozen above 10^14.8 h^-1 Msun; extrapolated outside 1e13-10^14.8 h^-1 Msun, z>2, x outside 0.04-1.34)",
        "lee22_x1pz2": "lee22_painted x (1+z)^2 : M1 hypothesis (comoving (ckpc/h)^3 shell volume + physical rho_crit(Time) + comoving R200 in P200 => printed fit = truth x a^2)",
        "b16_xgpaint": "XGPaint BattagliaTauProfile (B16 AGN density params 4e3/0.29/-0.66, xc 0.5, alpha 0.88/-0.03/0.19, beta 3.83/0.04/-0.025, gamma -0.2; exponent -(beta+gamma)/alpha as coded) x f_b rho_crit, n_e = 0.9 rho_gas / m_per_e (ne2d)",
        "lee22_literal": "Lee22 Table 8 no-c n_e/n200, eq. 9 as printed (n0 6.8 (M/Mcut)^0.68 (1+z)^-2.11, Mcut 10^13.61 h^-1 Msun pivot as production :mcut; shape clip 10^14.8 h^-1 Msun)",
        "lee22_xgpaint_ne2d": "lee22_literal x 0.6016316602183194 (production normalization=:xgpaint_ne2d, FRB DM and kSZ maps)",
        "lee22_tngcal": "lee22_literal x C(z), C = 0.2841790493880425 (1+min(z,2))^2.483190184608004 (redshift_scaling=:tng_calibrated)",
    }
    for k, v in arrays.items():
        a = np.asarray(v)
        entry = {"shape": list(a.shape), "dtype": str(a.dtype)}
        for pre, u in units.items():
            if k.startswith(pre):
                entry["units"] = u
                key = k[len(pre) + 1:] if not k.startswith("Pth") else "b12_thermal"
                key = key.replace("keV_cm3_", "")
                entry["model"] = what.get(key, what.get(key.replace("cm3_", ""), ""))
        d["arrays"][k] = entry
    extra = {"R200c_kpc": "physical R200c [kpc] (n_mass, n_z)", "P200_keV_cm3": "P200 [keV cm^-3] (n_mass, n_z)",
             "kT200_keV": "G M mu m_p/(2 R200c) [keV], mu = 0.588", "n200_cm3": "n200(z) [cm^-3]",
             "rho_crit_g_cm3": "rho_crit(z) [g cm^-3]", "E2": "H^2/H0^2", "C_tngcal": "C(z) on the z axis",
             "factor_xgpaint_ne2d": "0.9 X_H m_p / m_per_e", "in_lee22_fit_range": "True where 1e13 <= M h <= 10^14.8, z <= 2, 0.04 <= x <= 1.34"}
    for k, v in extra.items():
        d["arrays"][k]["description"] = v
    d["arrays"]["Pbar_b12_thermal"]["units"] = "P_th/P200 (dimensionless, THERMAL pressure)"
    for k in ("log10_M200c_msun", "M200c_msun", "M200c_hinv_msun", "z", "x_r200c"):
        d["arrays"][k]["description"] = d["axes"].get(k, {"M200c_msun": "physical M200c [Msun]",
                                                          "M200c_hinv_msun": "M200c x h [h^-1 Msun] (Lee22 mass unit)"}.get(k, ""))
    return d


def julia_crosscheck(arrays):
    path = OUT / "julia_production_density_crosscheck.csv"
    if not path.exists():
        return None
    rows = list(csv.DictReader(open(path)))
    res = {}
    for r in rows:
        i = int(np.argmin(np.abs(LOGM - float(r["log10M"]))))
        j = int(np.argmin(np.abs(ZS - float(r["z"]))))
        k = int(np.argmin(np.abs(X - float(r["x"]))))
        assert abs(X[k] / float(r["x"]) - 1) < 1e-9
        key = {"lee22_literal": "ne_cm3_lee22_literal", "lee22_xgpaint_ne2d": "ne_cm3_lee22_xgpaint_ne2d",
               "lee22_xgpaint_pipeline": "ne_cm3_lee22_xgpaint_ne2d", "lee22_tngcal": "ne_cm3_lee22_tngcal",
               "b16_xgpaint": "ne_cm3_b16_xgpaint"}[r["variant"]]
        tab = arrays[key][i, j, k]
        jul = float(r["ne_m3"]) * 1e-6
        rho_ratio = float(r["rho_crit_xgpaint_kg_m3"]) / (arrays["rho_crit_g_cm3"][j] * 1e3)
        res.setdefault(r["variant"], []).append((float(r["log10M"]), float(r["z"]), float(r["x"]), jul, tab,
                                                 jul / tab - 1, jul / (tab * rho_ratio) - 1))
    summ = {v: {"n": len(a), "max_abs_rel_julia_vs_table": float(max(abs(t[5]) for t in a)),
                "max_abs_rel_after_rho_crit_radiation_correction": float(max(abs(t[6]) for t in a))} for v, a in res.items()}
    with open(OUT / "julia_crosscheck_comparison.csv", "w") as fh:
        fh.write("variant,log10M,z,x,ne_cm3_julia,ne_cm3_table,rel_diff,rel_diff_rho_corrected\n")
        for v, a in res.items():
            for t in a:
                fh.write(f"{v},{t[0]},{t[1]},{t[2]:.8e},{t[3]:.10e},{t[4]:.10e},{t[5]:.3e},{t[6]:.3e}\n")
    return summ


def main():
    arrays = build()
    np.savez_compressed(OUT / "profile_tables.npz", **arrays)
    desc = describe(arrays)
    cc = julia_crosscheck(arrays)
    if cc is not None:
        desc["julia_production_crosscheck"] = cc
    json.dump(desc, open(OUT / "profile_tables.json", "w"), indent=1)
    # quick sanity numbers
    i14 = int(np.argmin(np.abs(LOGM - 14.0)))
    k1 = int(np.argmin(np.abs(X - 1.0)))
    print("factor_xgpaint_ne2d", XGP_NE2D)
    print("M=1e14, x~%.3f: z, P200[keV cm^-3], Pe_b12, Pe_lee22_painted, ne_b16, ne_lee22_literal" % X[k1])
    for j, z in enumerate(ZS):
        print(z, arrays["P200_keV_cm3"][i14, j], arrays["Pe_keV_cm3_b12"][i14, j, k1],
              arrays["Pe_keV_cm3_lee22_painted"][i14, j, k1], arrays["ne_cm3_b16_xgpaint"][i14, j, k1],
              arrays["ne_cm3_lee22_literal"][i14, j, k1])
    print("julia crosscheck:", json.dumps(cc, indent=1))


if __name__ == "__main__":
    main()
