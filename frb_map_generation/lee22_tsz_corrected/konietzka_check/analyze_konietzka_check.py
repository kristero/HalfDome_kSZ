#!/usr/bin/env python3
"""Combine (1) the CAMELS cell-level definition tests (camels_konietzka_defs.py outputs), (2) the Tinker08 halo model
of the Konietzka shells and of the window-extension hypothesis, into the test table, the F_n numbers and one figure.

Everything is expressed as the free-electron fraction f_e = N_e / [f_b M200c (1+X_H)/(2 m_p)] of the 1e13-1e14 Msun
(M200c) window, HMF (Tinker08, Planck15/TNG) and mass weighted per 0.1-wide redshift shell, exactly as the
Konietzka-implied values of lee22_normalization_forensics_20260928/D_gas_fractions/konietzka_implied_fe_shells.json.
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/cbllover/HalfDome")
sys.path.insert(0, str(ROOT / "frb_map_generation" / "analytic_pdf_torkamani2026"))
sys.path.insert(0, str(ROOT / "frb_map_generation" / "lee22_normalization_forensics" / "D_gas_fractions"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import halo_model_dm_pdf as H  # noqa: E402
import gasfrac_lib as GL  # noqa: E402
import konietzka_window_hmf as KW  # noqa: E402

OUT = ROOT / "frb_map_generation" / "outputs" / "lee22_konietzka_check_20260930"
DEFS = OUT / "camels_defs"
COSMO = H.PLANCK15_TNG
FB = COSMO.omega_b / COSMO.omega_m
C_EXT = 9.0          # effective upper-edge factor of the Konietzka windows (window algebra, konietzka_window_hmf.py)
# TNG300-1 hydro halo abundance / Tinker08 (Pillepich+18b counts: >=1e13: 3733 at z=0, 2333 at z=1; >=1e14: 280 at z=0)
TINKER_BOX = dict(z0_13_14=4648.4, z0_14_16=296.5, z1_13_14=2801.1, z1_14_16=52.1)
A_LO = {0.0: (3733 - 280) / TINKER_BOX["z0_13_14"], 1.0: (2333 - 0.945 * 52.1) / TINKER_BOX["z1_13_14"]}
A_HI = 280 / TINKER_BOX["z0_14_16"]


def a_lo(z):
    return np.interp(z, [0.0, 1.0], [A_LO[0.0], A_LO[1.0]])


# ------------------------------------------------------------------ CAMELS cell-level definitions
def load_defs():
    files = sorted(glob.glob(str(DEFS / "konietzka_defs_CV_*_*.npz")))
    if not files:
        return None
    keys = None
    rows = {}
    for f in files:
        d = np.load(f)
        if keys is None:
            keys = [k for k in d.files if d[k].ndim >= 1 and k not in ("XS", "thresholds_code")]
        for k in keys:
            rows.setdefault(k, []).append(d[k])
        XS = d["XS"]
    T = {k: np.concatenate(v) for k, v in rows.items()}
    T["XS"] = XS
    T["files"] = files
    return T


def def_ratios(T, zval, lo=13.0, hi=14.0, nboot=500, seed=2):
    """electron ratio sum(Me_def)/sum(Me_base) and area ratio sum(A_def R^2)/sum(pi R200c^2) for halos in the window"""
    lm = np.log10(T["M200c"])
    s = (np.abs(T["z"] - zval) < 0.02) & (lm >= lo) & (lm < hi)
    n = int(s.sum())
    if n == 0:
        return None, 0
    XS = list(T["XS"])
    i1, i2, i15 = XS.index(1.0), XS.index(2.0), XS.index(1.5)
    base = T["Me_sph_xe"][s, i1]
    R2 = T["R200c_ckpch"][s] ** 2
    Anorm = T["A_sph1_2d_cells"][s]          # pixel-method area of the R200c cell set (method calibration)
    E = {
        "R200c, SF n_e = 0 (Konietzka+25)": (T["Me_sph_sf0"][s, i1], R2),
        "R200c, SF fully ionised": (T["Me_sph_sfion"][s, i1], R2),
        "R500c sphere": (T["Me_r500c_xe"][s], (T["R500c_ckpch"][s]) ** 2),
        "R200m sphere": (T["Me_r200m_xe"][s], (T["R200m_ckpch"][s]) ** 2),
        "1.5 R200c sphere": (T["Me_sph_xe"][s, i15], 2.25 * R2),
        "2 R200c sphere": (T["Me_sph_xe"][s, i2], 4.0 * R2),
        "FoF members, all": (T["Me_fof_xe"][s], T["A_fof"][s] / Anorm * R2),
        "FoF members inside R200c": (T["Me_fof_in1_xe"][s], T["A_fof_in1"][s] / Anorm * R2),
        "rho_DM > 57 rho_crit,0 (comoving), FoF": (T["Me_thrc057_fof_xe"][s], T["A_thrc057_fof"][s] / Anorm * R2),
        "rho_DM > 57 rho_crit,0 (comoving), FoF, SF n_e=0": (T["Me_thrc057_fof_sf0"][s], T["A_thrc057_fof"][s] / Anorm * R2),
        "rho_DM > 57 rho_crit,0 (comoving), any cell < 3 R200c": (T["Me_thrc057_3R_xe"][s], T["A_thrc057_3R"][s] / Anorm * R2),
        "rho_DM > 28.5 rho_crit,0 (comoving), FoF": (T["Me_thrc028_fof_xe"][s], T["A_thrc028_fof"][s] / Anorm * R2),
        "rho_DM > 114 rho_crit,0 (comoving), FoF": (T["Me_thrc114_fof_xe"][s], T["A_thrc114_fof"][s] / Anorm * R2),
        "rho_DM > 57 rho_crit(z) (physical), FoF": (T["Me_thrp057_fof_xe"][s], T["A_thrp057_fof"][s] / Anorm * R2),
    }
    rng = np.random.default_rng(seed)
    out = {}
    for name, (me, ar) in E.items():
        r_e = me.sum() / base.sum()
        r_a = ar.sum() / R2.sum()
        be, ba = [], []
        for _ in range(nboot):
            k = rng.integers(0, n, n)
            be.append(me[k].sum() / base[k].sum())
            ba.append(ar[k].sum() / R2[k].sum())
        out[name] = dict(Ne=float(r_e), Ne_err=float(np.std(be)), area=float(r_a), area_err=float(np.std(ba)))
    # f_e(<R200c) of these halos: N_e/[f_b M (1+X_H)/(2 m_p)], with N_e m_p = Me = sum x_e X_H m
    fe = float(base.sum() / (0.049 / 0.3 * T["M200c"][s].sum() * 0.5 * (1 + 0.76)))
    out["_meta"] = dict(n_halos=n, z=float(np.mean(T["z"][s])), fe_R200c=fe,
                        mean_logM=float(np.mean(lm[s])),
                        med_M200c_over_M200m=float(np.median(T["M200c"][s] / T["M200m"][s])),
                        med_M200c_over_Mfof=float(np.median(T["M200c"][s] / T["Mfof"][s])),
                        med_M200c_over_M500c=float(np.median(T["M200c"][s] / T["M500c"][s])),
                        pix_area_sph1=float(np.mean(Anorm)),
                        fof_frac_cells_beyond_3R=float(np.mean(T["fof_frac_cells_beyond_3R"][s])))
    return out, n


# ------------------------------------------------------------------ halo-model shells
def lee22_a3_fe(lm, z):
    """Lee22 no-c literal x (1+z)^3 free-electron fraction inside R200c (shape clip at 10^14.8/h as painted)"""
    lmh = lm + np.log10(COSMO.h)
    n0, xc, beta = GL.lee22_noc_params(lmh, z, shape_clip_log10m_h=14.8)
    n0, xc, beta = (np.broadcast_to(np.asarray(a, float), lmh.shape) for a in (n0, xc, beta))
    u = np.linspace(np.log(1e-8), 0.0, 4000)
    x = np.exp(u)[None, :]
    sx = x / np.asarray(xc)[:, None]
    f = sx ** -0.3 * (1 + sx) ** (-np.asarray(beta)[:, None]) * x ** 3
    mom = np.trapz(f, u, axis=1)
    return GL.F_E_PER_NBAR * 3 * n0 * mom * (1 + z) ** 3


def shell_model(z0, z1, lm_lo, lm_hi, fe_camels, mass_shift=1.0, nsub=8, dlm=0.01):
    """sums over a shell of the f_e = 1 DM weight w (Tinker08), of w f_e(CAMELS), of w f_e(Lee22 (1+z)^3) and Nbar.
    mass_shift r: the window [lm_lo, lm_hi] refers to a mass M_x = M200c / r, i.e. M200c in [r 10^lo, r 10^hi]."""
    E_PER_MSUN = H.MSUN_G * (1 + 0.76) / (2 * H.M_P_G)
    lm = np.arange(lm_lo + np.log10(mass_shift), lm_hi + np.log10(mass_shift), dlm) + dlm / 2
    M = 10 ** lm
    zz = np.linspace(z0, z1, nsub + 1)
    acc = np.zeros(4)
    for zm, dz in zip(0.5 * (zz[1:] + zz[:-1]), np.diff(zz)):
        dn = H.tinker08_dndlnm(COSMO, M, zm) * np.log(10) * dlm
        path = (1 / COSMO.H_over_c(zm)) * (1 + zm) * dz
        w = dn * FB * M * E_PER_MSUN / H.MPC_CM ** 3 * path * H.MPC_CM / H.PC_CM
        r = H.r200c_mpc(COSMO, M, zm)
        chi = COSMO.chi(zm)[0]
        th = np.arctan(r / (chi / (1 + zm)))
        acc += [w.sum(), (w * fe_camels(lm, zm)).sum(), (w * lee22_a3_fe(lm, zm)).sum(),
                (chi ** 2 / COSMO.H_over_c(zm) * dz * dn * np.pi * th ** 2).sum()]
    return acc


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    zs, lgrid, tab, tab25, cnt, xe_corr = KW.camels_fe_table()
    fe_c, _ = KW.make_fe_func(zs, lgrid, tab)
    shells = json.loads(KW.FSH.read_text())["shells"][:15]
    alg = KW.window_algebra()
    # ---- CAMELS definitions
    T = load_defs()
    deftab = {}
    if T is not None:
        for zv in sorted(set(np.round(T["z"], 2))):
            r, n = def_ratios(T, zv)
            if r is not None:
                deftab[float(zv)] = r
                print(f"z={zv}: {n} halos, f_e(R200c) = {r['_meta']['fe_R200c']:.3f}")
                for k, v in r.items():
                    if not k.startswith("_"):
                        print(f"   {k:58s} Ne x{v['Ne']:.3f}+-{v['Ne_err']:.3f}  area x{v['area']:.3f}+-{v['area_err']:.3f}")
                print("   meta", {k: round(v, 3) for k, v in r["_meta"].items()})
    zdef = np.array(sorted(deftab))

    def ratio_at(name, z, kind="Ne"):
        if len(zdef) == 0:
            return np.nan
        vals = np.array([deftab[zz][name][kind] for zz in zdef])
        return float(np.interp(z, zdef, vals))

    # ---- shells: data vs scenarios
    rows = []
    for s in shells:
        z0, z1 = s["z_lo"], s["z_hi"]
        zc = 0.5 * (z0 + z1)
        U, Cn, Ln, Nn = shell_model(z0, z1, 13.0, 14.0, fe_c)
        _, Ch, Lh, Nh = shell_model(z0, z1, 14.0, 14.0 + np.log10(C_EXT), fe_c)
        K = s["fe_tng_implied"]
        row = dict(z_lo=z0, z_hi=z1, z=zc, fe_K=K, fe_K_err=s["fe_tng_implied_err"], tng_dDM=s["tng_dDM"],
                   fe_camels_nominal=Cn / U, fe_camels_ext=(Cn + Ch) / U,
                   fe_camels_nominal_hydroN=a_lo(zc) * Cn / U,
                   fe_camels_ext_hydroN=(a_lo(zc) * Cn + A_HI * Ch) / U,
                   hit_ext_over_nom=(Nn + Nh) / Nn, hit_ext_over_nom_hydroN=(a_lo(zc) * Nn + A_HI * Nh) / Nn,
                   Fn_K_nominal=Ln / (K * U), Fn_camels_nominal=Ln / Cn,
                   Fn_K_ext_hydroN=(a_lo(zc) * Ln + A_HI * Lh) / (K * U), Fn_camels_ext=(Ln + Lh) / (Cn + Ch))
        Us, Cs, Ls, Ns = shell_model(z0, z1, 13.0, 14.0, fe_c, mass_shift=1.0 / COSMO.h)
        row["window_in_Msun_over_h_Ne"] = Cs / Cn
        row["window_in_Msun_over_h_hit"] = Ns / Nn
        for key, r in (("M200m", "med_M200c_over_M200m"), ("Mfof", "med_M200c_over_Mfof"), ("M500c", "med_M200c_over_M500c")):
            if len(zdef):
                rr = float(np.interp(zc, zdef, [deftab[zz]["_meta"][r] for zz in zdef]))
                Us, Cs, Ls, Ns = shell_model(z0, z1, 13.0, 14.0, fe_c, mass_shift=rr)
                row[f"window_in_{key}_Ne"] = Cs / Cn
                row[f"window_in_{key}_hit"] = Ns / Nn
        for name in (deftab[zdef[0]].keys() if len(zdef) else []):
            if name.startswith("_"):
                continue
            row["def:" + name + ":Ne"] = ratio_at(name, zc, "Ne")
            row["def:" + name + ":area"] = ratio_at(name, zc, "area")
        rows.append(row)
    # observed hit ratio TNG / nominal model per shell (from konietzka_window_hmf output)
    hitcsv = OUT / "window_extension_shells.csv"
    hit = {}
    if hitcsv.exists():
        import csv
        for r in csv.DictReader(open(hitcsv)):
            hit[round(float(r["z_lo"]), 2)] = float(r["hit_ratio_tng_over_model"])
    for r in rows:
        r["hit_tng_over_nominal"] = hit.get(round(r["z_lo"], 2), np.nan)
    # constant F_n fits (cumulative DM to z_s = 0.1 ... 1.5)
    zc = np.array([r["z"] for r in rows])
    Kc = np.array([r["fe_K"] for r in rows])
    summary = {}
    for key in ("Fn_K_nominal", "Fn_camels_nominal", "Fn_K_ext_hydroN", "Fn_camels_ext"):
        v = np.array([r[key] for r in rows])
        w = np.array([r["tng_dDM"] for r in rows])
        summary[key] = dict(values={f"{r['z']:.2f}": round(r[key], 3) for r in rows},
                            DM_weighted_harmonic_mean=float(np.sum(w) / np.sum(w / v)),
                            powerlaw_fit_1pz=list(np.polyfit(np.log(1 + zc), np.log(v), 1)))
    # z-trend of K / scenario
    trend = {}
    for key in ("fe_camels_nominal", "fe_camels_ext", "fe_camels_nominal_hydroN", "fe_camels_ext_hydroN"):
        v = np.array([r[key] for r in rows])
        rat = Kc / v
        p = np.polyfit(np.log(1 + zc), np.log(rat), 1)
        trend[key] = dict(ratio_K_over={f"{r['z']:.2f}": round(float(q), 3) for r, q in zip(rows, rat)},
                          powerlaw_amp=float(np.exp(p[1])), powerlaw_slope=float(p[0]),
                          mean_z_lt_0p5=float(np.mean(rat[zc < 0.5])), mean_z_0p5_1p5=float(np.mean(rat[zc >= 0.5])))
    # ---- test table: electrons and hit area relative to CAMELS R200c spheres, nominal M200c window, Tinker08
    ztab = [0.05, 0.25, 0.45, 1.05, 1.45]
    sel = [min(range(len(rows)), key=lambda i: abs(rows[i]["z"] - zt)) for zt in ztab]
    table = []

    def add(name, ne, ar):
        table.append(dict(candidate=name, **{f"Ne_z{rows[i]['z']:.2f}": float(ne(rows[i])) for i in sel},
                          **{f"area_z{rows[i]['z']:.2f}": float(ar(rows[i])) for i in sel}))
    add("TARGET: Konietzka / CAMELS-R200c model", lambda r: r["fe_K"] / r["fe_camels_nominal"], lambda r: r["hit_tng_over_nominal"])
    for name in [k for k in rows[0] if k.startswith("def:") and k.endswith(":Ne")]:
        nm = name[4:-3]
        add(nm, lambda r, nm=nm: r[f"def:{nm}:Ne"], lambda r, nm=nm: r[f"def:{nm}:area"])
    if "window_in_M200m_Ne" in rows[0]:
        for key in ("M200m", "Mfof", "M500c"):
            add(f"window read in {key} (R200c spheres)", lambda r, k=key: r[f"window_in_{k}_Ne"], lambda r, k=key: r[f"window_in_{k}_hit"])
        add("R200m sphere + window in M200m", lambda r: r["def:R200m sphere:Ne"] * r["window_in_M200m_Ne"],
            lambda r: r["def:R200m sphere:area"] * r["window_in_M200m_hit"])
    add("window read in Msun/h", lambda r: r["window_in_Msun_over_h_Ne"], lambda r: r["window_in_Msun_over_h_hit"])
    add("missing 1/(1+z) in DM", lambda r: 1 + r["z"], lambda r: 1.0)
    add("TNG300 hydro abundance only (Pillepich+18b counts)", lambda r: a_lo(r["z"]), lambda r: a_lo(r["z"]))
    add(f"window upper edge x{C_EXT:g} (to 9e14), Tinker08", lambda r: r["fe_camels_ext"] / r["fe_camels_nominal"], lambda r: r["hit_ext_over_nom"])
    add(f"window upper edge x{C_EXT:g}, TNG300 hydro abundance", lambda r: r["fe_camels_ext_hydroN"] / r["fe_camels_nominal"],
        lambda r: r["hit_ext_over_nom_hydroN"])
    import csv
    with open(OUT / "test_table.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(table[0].keys()))
        wr.writeheader()
        for t in table:
            wr.writerow({k: (f"{v:.3f}" if isinstance(v, float) else v) for k, v in t.items()})
    for t in table:
        print(f"{t['candidate'][:62]:62s}", " ".join(f"{v:5.2f}" for k, v in t.items() if k.startswith("Ne")), " | ",
              " ".join(f"{v:5.2f}" for k, v in t.items() if k.startswith("area")))
    res = dict(c_ext=C_EXT, test_table=table, A_lo=A_LO, A_hi=A_HI, window_algebra=alg, camels_definitions=deftab, shells=rows,
               Fn=summary, K_over_scenario=trend,
               camels_files=[Path(f).name for f in (T["files"] if T is not None else [])])
    (OUT / "konietzka_check_results.json").write_text(json.dumps(res, indent=1, default=float))
    for k, v in trend.items():
        print("K /", k, {kk: round(vv, 3) for kk, vv in v.items() if not isinstance(vv, dict)})
    for k, v in summary.items():
        print(k, "DM-weighted", round(v["DM_weighted_harmonic_mean"], 3), "slope", round(v["powerlaw_fit_1pz"][0], 3),
              "amp", round(float(np.exp(v["powerlaw_fit_1pz"][1])), 3))
    return res


if __name__ == "__main__":
    main()
