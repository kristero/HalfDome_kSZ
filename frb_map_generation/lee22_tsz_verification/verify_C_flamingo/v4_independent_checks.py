"""Second verifier (2026-09-29): independent re-derivations for workstream C that the first (stopped) verifier did not do.

Self-contained: my own Lee22 Table-7 pressure (typed from the painted-map provenance strings, NOT lee22_models), my own
B12, my own rho_cr(z) from the FLAMINGO cosmology (not from SOAP R200 or H(z)), my own integrals.

  A  own Lee22 P_e/P200 vs lee22_models (did the workstream evaluate the model that was painted?)
  B  a-factor audit of the FLAMINGO pressure: particle-sum Y(<R200c) (stack products) vs SOAP SO/200_crit/ComptonY, and
     idark sum n_e kT V vs sum ComptonY, per snapshot; electron-weighted T_e(<R200c)/T200 per snapshot
     (a (1+z)^2 unit error in y would show up as a (1+z)^2 trend in T_e/T200 and a z-dependent y/pe ratio)
  C  pressure redshift slopes per 0.2-dex bin from particle Y(<R200c) of the stack haloes with MY P200
     (rho_cr from E(z), R200 from M200c) and my Lee22/B12 integrals; medians of Lee22 x (1+z)^2 / FL per z
  D  outskirts: SOAP apertures R100c->R50c and R50c->5R500c, all flagged centrals, Lee22 x (1+z)^2 / FL
  E  density: F_n per z (two bin sets), n_e/n200 slope at the shells containing x = 0.5 and x = 1, f_e(<R200c) per z
  F  B12 / FL Y(<R200c) in 10^13-13.2 groups at z = 0
Output: OUTV/v4_independent_checks.json
"""
import json
from pathlib import Path

import numpy as np

ST = Path("/home/cbllover/HalfDome/flamingo_pressure_calibration/notebook_cache/stacks")
CACHE = Path("/home/cbllover/HalfDome/flamingo_pressure_calibration/notebook_cache")
DENS = Path("/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_tsz_verification_20260929/C_flamingo/idark_density/products")
OUTV = Path("/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_tsz_verification_20260929/verify_C_flamingo")
SNAPS = {78: 0.0, 68: 0.5, 58: 1.0, 48: 1.5, 38: 2.0, 18: 3.0}
# SI, CODATA 2018
G, MSUN, MPC = 6.67430e-11, 1.98841e30, 3.0856775814913673e22
SIGMA_T, ME_C2, KB, MP = 6.6524587321e-29, 9.1093837015e-31 * 299792458.0**2, 1.380649e-23, 1.67262192369e-27
# FLAMINGO D3A
H_FL, OB, OM = 0.681, 0.0486, 0.306
FB = OB / OM
X_H = 0.76
MU = 4 / (3 + 5 * X_H)
MU_E = 2 / (1 + X_H)

LNX = np.linspace(np.log(1e-6), 0.0, 6001)
XG = np.exp(LNX)


def rho_crit(z):
    H0 = 100 * H_FL * 1e3 / MPC
    return 3 * H0**2 * (OM * (1 + z) ** 3 + 1 - OM) / (8 * np.pi * G)


def gnfw(x, xc, beta, gamma=-0.3, alpha=1.0):
    s = x / xc
    return s**gamma * (1 + s**alpha) ** (-beta)


def lee_P_own(x, m_msun, z):
    """Lee22 no-c Table 7 as in the painted map's provenance: P0 = 2.8 (M/Mcut)^1.08|0.80 (1+z)^-1.89,
    xc = 2.10 (M/Mcut)^-0.35 (1+z)^0.53, beta = 9.4 (M/Mcut)^-0.06 (1+z)^0.60, Mcut = 10^13.6 h^-1 Msun,
    shape (xc, beta) clipped at 10^14.8 h^-1 Msun; M in h^-1 Msun = M[Msun] * h."""
    mh = m_msun * H_FL
    r = mh / 10**13.6
    rs = min(mh, 10**14.8) / 10**13.6
    P0 = 2.8 * (r**1.08 if r < 1 else r**0.80) * (1 + z) ** -1.89
    return P0 * gnfw(x, 2.10 * rs**-0.35 * (1 + z) ** 0.53, 9.4 * rs**-0.06 * (1 + z) ** 0.60)


def b12_Pe_own(x, m_msun, z):
    m = m_msun / 1e14
    P0 = 18.1 * m**0.154 * (1 + z) ** -0.758
    return 0.5176 * P0 * gnfw(x, 0.497 * m**-0.00865 * (1 + z) ** 0.731, 4.35 * m**0.0393 * (1 + z) ** 0.415)


def yint(fun, m, z, x1=0.0, x2=1.0):
    """4 pi int_x1^x2 Pbar x^2 dx on a log grid (x1 = 0 -> from 1e-6)."""
    lnx = np.linspace(np.log(max(x1, 1e-6)), np.log(x2), 3001)
    x = np.exp(lnx)
    return 4 * np.pi * np.trapz(fun(x, m, z) * x**3, lnx)


def slope(z, v):
    return float(np.polyfit(np.log1p(np.asarray(z)), np.log(np.asarray(v)), 1)[0])


def main():
    out = {}
    # ---------------------------------------------------------------- A
    import sys
    sys.path.insert(0, "/home/cbllover/HalfDome/frb_map_generation/lee22_normalization_forensics/C_pressure_consistency")
    import lee22_models as L
    dev = 0.0
    for lm in np.arange(12.8, 15.6, 0.1):
        for z in (0.0, 0.5, 1.0, 2.0, 3.0):
            a = lee_P_own(XG[::50], 10**lm, z)
            b = L.lee_Pbar(XG[::50], 10**lm * H_FL, z)
            dev = max(dev, float(np.max(np.abs(a / b - 1))))
    out["A_own_lee22_vs_lee22_models_max_rel"] = dev
    # ---------------------------------------------------------------- B, C, F
    rowsC, B = [], {}
    for s, zn in SNAPS.items():
        d = np.load(ST / f"L2p8_m9_{s:04d}_stack_products.npz")
        z = float(d["redshift"])
        m, r_soap = d["cat_m200c_msun"], d["cat_r_200_crit_mpc"]
        r_own = (3 * m * MSUN / (4 * np.pi * 200 * rho_crit(z))) ** (1 / 3) / MPC
        ypart, ysoap = d["y_r200_raw"], d["cat_y_200_crit_mpc2"]
        ok = ysoap > 0
        B[str(s)] = {"z": z, "R200_own_over_SOAP_median": float(np.median(r_own / r_soap)),
                     "Y200_particles_over_SOAP_median": float(np.median(ypart[ok] / ysoap[ok])),
                     "Y200_particles_over_SOAP_p5_p95": [float(np.percentile(ypart[ok] / ysoap[ok], 5)),
                                                          float(np.percentile(ypart[ok] / ysoap[ok], 95))]}
        # idark products: sum n_e kT V vs sum y, T_e(<R200c)/T200
        if (DENS / f"L2p8_m9_{s:04d}_density_products.npz").exists():
            p = np.load(DENS / f"L2p8_m9_{s:04d}_density_products.npz")
            e = p["edges"]
            y_from_pe = p["pe3"].sum(1) * SIGMA_T / ME_C2 / MPC**2          # Pa m^3 -> Mpc^2
            B[str(s)]["sum_nekTV_over_sum_y_median"] = float(np.median(y_from_pe / p["y3"].sum(1)))
            inner = e[1:] <= 1.0 + 1e-9                                       # shells fully inside R200c
            Te = p["pe3"][:, inner].sum(1) / p["ne3"][:, inner].sum(1)       # J, electron-weighted kT
            r200 = p["r200_mpc"] * MPC
            kT200 = G * p["m200c_msun"] * MSUN * MU * MP / (2 * r200)
            big = p["m200c_msun"] >= 10**13.6
            B[str(s)]["Te_ew_R200_over_T200_median_M>=13.6"] = float(np.median(Te[big] / kT200[big]))
            B[str(s)]["Ne_mp_over_Mgas_R200_median"] = float(np.median(p["ne_r200"] * MP / (p["m_r200"] * MSUN)))
        # pressure Y200 in P200 units
        P200 = G * m * MSUN * 200 * rho_crit(z) * FB / (2 * r_own * MPC)
        yhat = ypart * MPC**2 / (SIGMA_T / ME_C2 * P200 * (r_own * MPC) ** 3)
        for b in np.unique(d["bin"]):
            sel = d["bin"] == b
            if sel.sum() < 8:
                continue
            lo = round(float(d["bin_edges"][b]), 1)
            yl = np.array([yint(lee_P_own, mi, z) for mi in m[sel]])
            yb = np.array([yint(b12_Pe_own, mi, z) for mi in m[sel]])
            rowsC.append({"snap": s, "z": zn, "lo": lo, "n": int(sel.sum()), "FL": float(yhat[sel].mean()),
                          "Lee": float(yl.mean()), "B12": float(yb.mean())})
    out["B_units"] = B
    slopes = []
    for lo in sorted({r["lo"] for r in rowsC}):
        sel = [r for r in rowsC if r["lo"] == lo and r["z"] <= 2]
        if len(sel) < 3:
            continue
        zz = [r["z"] for r in sel]
        sF = slope(zz, [r["FL"] for r in sel])
        slopes.append({"lo": lo, "n_z": len(sel), "s_FL": sF, "s_Lee_minus_FL": slope(zz, [r["Lee"] for r in sel]) - sF,
                       "s_B12_minus_FL": slope(zz, [r["B12"] for r in sel]) - sF})
    out["C_Y200_slopes"] = slopes
    med = {}
    for zn in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0):
        sel = [r for r in rowsC if r["z"] == zn and 13.6 <= r["lo"] <= 14.6 + 1e-9]
        med[str(zn)] = {"Lee22_over_FL": float(np.median([r["Lee"] / r["FL"] for r in sel])),
                        "Lee22_a2_over_FL": float(np.median([r["Lee"] * (1 + zn) ** 2 / r["FL"] for r in sel])),
                        "B12_over_FL": float(np.median([r["B12"] / r["FL"] for r in sel])), "n_bins": len(sel)}
    out["C_Y200_median_ratios_13.6-14.6"] = med
    g = [r for r in rowsC if r["z"] == 0.0 and r["lo"] in (13.0, 13.2)]
    out["F_B12_over_FL_Y200_groups_z0"] = {str(r["lo"]): r["B12"] / r["FL"] for r in g}
    hi = [r for r in rowsC if r["z"] == 0.0 and r["lo"] >= 15.0]
    out["C_Lee22_over_FL_Y200_z0_above_range"] = {str(r["lo"]): r["Lee"] / r["FL"] for r in hi}
    # ---------------------------------------------------------------- D outskirts (all flagged centrals)
    D = []
    for s, zn in SNAPS.items():
        if zn > 2:
            continue
        c = np.load(CACHE / f"L2p8_m9_{s:04d}_reduced_haloes.npz")
        z = zn
        cen = c["is_central"].astype(bool) & (c["m200c_msun"] > 0)
        m = c["m200c_msun"][cen]
        r200 = c["r_200_crit_mpc"][cen]
        P200 = G * m * MSUN * 200 * rho_crit(z) * FB / (2 * r200 * MPC)
        norm = SIGMA_T / ME_C2 * P200 * (r200 * MPC) ** 3 / MPC**2
        shells = {"R100c-R50c": ("r_100_crit_mpc", "y_100_crit_mpc2", "r_50_crit_mpc", "y_50_crit_mpc2"),
                  "R50c-5R500c": ("r_50_crit_mpc", "y_50_crit_mpc2", "r_5xR_500_crit_mpc", "y_5xR_500_crit_mpc2")}
        lm = np.log10(m)
        for lo in np.round(np.arange(13.6, 14.8, 0.2), 1):
            sel = np.where((lm >= lo) & (lm < lo + 0.2))[0]
            if len(sel) < 20:
                continue
            sel = sel[:: max(1, len(sel) // 150)]                          # <= ~150 haloes per bin for speed
            rec = {"z": zn, "lo": lo, "n": int(len(sel))}
            for name, (r1k, y1k, r2k, y2k) in shells.items():
                x1 = c[r1k][cen][sel] / r200[sel]
                x2 = c[r2k][cen][sel] / r200[sel]
                yfl = (c[y2k][cen][sel] - c[y1k][cen][sel]) / norm[sel]
                ylee = np.array([yint(lee_P_own, m[i], z, a, b) for i, a, b in zip(sel, x1, x2)]) * (1 + z) ** 2
                rec[name] = float(ylee.mean() / yfl.mean())
                rec[name + "_x"] = [float(np.median(x1)), float(np.median(x2))]
            D.append(rec)
    out["D_outskirts_Lee22a2_over_FL"] = D
    out["D_medians"] = {k: float(np.median([r[k] for r in D])) for k in ("R100c-R50c", "R50c-5R500c")}
    # ---------------------------------------------------------------- E density
    E = []
    for s, zn in SNAPS.items():
        if zn > 2:
            continue
        p = np.load(DENS / f"L2p8_m9_{s:04d}_density_products.npz")
        st = np.load(ST / f"L2p8_m9_{s:04d}_stack_products.npz")
        assert np.array_equal(p["halo_index"], st["cat_halo_index"])
        z = zn
        m, e = p["m200c_msun"], p["edges"]
        r200 = p["r200_mpc"] * MPC
        n200 = 200 * rho_crit(z) * FB / (X_H * MP)
        fe = p["ne_r200"] * MU_E * MP / (FB * m * MSUN)
        for b in np.unique(st["bin"]):
            sel = st["bin"] == b
            if sel.sum() < 8:
                continue
            lo = round(float(st["bin_edges"][b]), 1)
            fl = np.array([3 * MU_E / X_H * np.trapz(L.lee_nbar(XG, mi * H_FL, z) * XG**3, LNX) for mi in m[sel]])
            rec = {"z": zn, "lo": lo, "fe_FL": float(fe[sel].mean()), "fe_Lee": float(fl.mean()),
                   "Fn": float(fl.mean() * (1 + z) ** 3 / fe[sel].mean())}
            for xq in (0.5, 1.0):
                k = int(np.searchsorted(e, xq) - 1)
                if xq == 1.0:
                    k = int(np.searchsorted(e, 1.0, side="left") - 1)      # shell containing x = 1
                full = (p["r100_mpc"] / p["r200_mpc"] >= e[k + 1]) & sel
                vol = 4 / 3 * np.pi * (e[k + 1] ** 3 - e[k] ** 3) * r200**3
                ne = p["ne3"][:, k] / vol / n200
                rr = np.linspace(e[k], e[k + 1], 201)
                lee = np.array([np.trapz(L.lee_nbar(rr, mi * H_FL, z) * rr**2, rr) / np.trapz(rr**2, rr) for mi in m[full]])
                rec[f"ne_FL_x{xq:g}"] = float(ne[full].mean()) if full.sum() >= 3 else np.nan
                rec[f"ne_Lee_x{xq:g}"] = float(lee.mean()) if full.sum() >= 3 else np.nan
                rec[f"shell_x{xq:g}"] = [float(e[k]), float(e[k + 1])]
            E.append(rec)
    es = []
    for lo in sorted({r["lo"] for r in E}):
        sel = [r for r in E if r["lo"] == lo]
        if len(sel) < 3:
            continue
        zz = [r["z"] for r in sel]
        rec = {"lo": lo, "s_fe_Lee_minus_FL": slope(zz, [r["fe_Lee"] for r in sel]) - slope(zz, [r["fe_FL"] for r in sel])}
        for xq in ("0.5", "1"):
            gsel = [r for r in sel if np.isfinite(r[f"ne_FL_x{xq}"])]
            if len(gsel) >= 3:
                zg = [r["z"] for r in gsel]
                rec[f"s_ne_Lee_minus_FL_x{xq}"] = slope(zg, [r[f"ne_Lee_x{xq}"] for r in gsel]) - slope(zg, [r[f"ne_FL_x{xq}"] for r in gsel])
        es.append(rec)
    out["E_density_slopes"] = es
    out["E_Fn_by_z"] = {}
    for zn in (0.0, 0.5, 1.0, 1.5, 2.0):
        a = [r["Fn"] for r in E if r["z"] == zn and 13.6 <= r["lo"] <= 14.6 + 1e-9]
        b = [r["Fn"] for r in E if r["z"] == zn]
        c = [r["fe_FL"] for r in E if r["z"] == zn and r["lo"] >= 13.6]
        out["E_Fn_by_z"][str(zn)] = {"median_13.6-14.6": float(np.median(a)), "median_all_bins": float(np.median(b)),
                                     "fe_FL_median_13.6+": float(np.median(c)), "fe_FL_range_13.6+": [float(min(c)), float(max(c))]}
    out["E_Fn_median_13.6-14.6_all_z"] = float(np.median([r["Fn"] for r in E if 13.6 <= r["lo"] <= 14.6 + 1e-9]))
    out["E_Fn_median_all"] = float(np.median([r["Fn"] for r in E]))
    out["E_Fn_range_all"] = [float(min(r["Fn"] for r in E)), float(max(r["Fn"] for r in E))]
    out["C_rows"] = rowsC
    out["E_rows"] = E
    OUTV.mkdir(parents=True, exist_ok=True)
    (OUTV / "v4_independent_checks.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    print("A own Lee22 vs lee22_models max rel", dev)
    print(json.dumps(B, indent=0))
    print("C slopes: lo s_FL Lee-FL B12-FL")
    for r in slopes:
        print(f"  {r['lo']:.1f} {r['s_FL']:+.3f} {r['s_Lee_minus_FL']:+.3f} {r['s_B12_minus_FL']:+.3f}")
    print(json.dumps(med, indent=0))
    print("F groups", out["F_B12_over_FL_Y200_groups_z0"], "Lee above range z0", out["C_Lee22_over_FL_Y200_z0_above_range"])
    print("D", out["D_medians"])
    for r in D:
        print("  ", r)
    print("E slopes")
    for r in es:
        print("  ", {k: round(v, 3) for k, v in r.items()})
    print(json.dumps(out["E_Fn_by_z"], indent=0), out["E_Fn_median_13.6-14.6_all_z"], out["E_Fn_median_all"], out["E_Fn_range_all"])


if __name__ == "__main__":
    main()
