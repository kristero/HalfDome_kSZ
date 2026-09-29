"""Independent re-derivation of workstream C's density claims.

(a) From the idark density products (per-halo N_e(<R200c) = ne_r200, M_gas(<R200c) = m_r200) of the stack haloes:
    f_e = N_e m_p mu_e / (f_b M200c), mu_e = 2/(1+0.76) (Lee22 convention), bin means per (snapshot, 0.2-dex bin),
    and the Lee22 Table-8 printed f_e = 3 (mu_e/X_H) int_0^1 nbar x^2 dx (my own log-trapezoid), slopes over z <= 2,
    F_n = f_e,Lee22 (1+z)^3 / f_e,FL.
    Also check the products' halo list against the stack products (same halo_index order).
(b) With ALL flagged centrals (SOAP GasMass(<R200c), catalogue caches): slope of f_gas = M_gas/(f_b M200c) vs z per bin,
    an independent sample (no idark products), to be compared with s_FL for f_e (which also includes the N_e/M_gas trend).
(c) Shell n_e/n200 at the shell containing x = 0.5 (no interpolation), fully covered haloes only.
Output: OUTV/v2_density_fe.json
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/cbllover/HalfDome/frb_map_generation/lee22_normalization_forensics/C_pressure_consistency")
import lee22_models as L  # noqa: E402

FPC = Path("/home/cbllover/HalfDome/flamingo_pressure_calibration/notebook_cache")
DENS = Path("/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_tsz_verification_20260929/C_flamingo/idark_density/products")
OUTV = Path("/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_tsz_verification_20260929/verify_C_flamingo")
SNAPS = [78, 68, 58, 48, 38]
H = 0.681
FB = 0.0486 / 0.30600007943162144
X_H = 0.76
MU_E = 2 / (1 + X_H)
MP = 1.67262192369e-27
MSUN = 1.98841e30
MPC = 3.0856775814913673e22
lnx = np.linspace(np.log(1e-5), 0.0, 4001)
X = np.exp(lnx)


def fe_lee(m, z):
    return np.array([3 * MU_E / X_H * np.trapz(L.lee_nbar(X, mi * H, z) * X**3, lnx) for mi in m])


def fgas_lee(m, z):
    # printed Lee22 electrons converted to gas mass with mu_e (fully ionised): f_gas = f_e
    return fe_lee(m, z)


def slope(lz, v):
    return float(np.polyfit(lz, np.log(v), 1)[0])


def main():
    rows, soap_rows, shell_rows, idx_check = [], [], [], {}
    for s in SNAPS:
        d = np.load(DENS / f"L2p8_m9_{s:04d}_density_products.npz")
        st = np.load(FPC / "stacks" / f"L2p8_m9_{s:04d}_stack_products.npz")
        z = float(d["redshift"])
        idx_check[str(s)] = {"same_halo_index": bool(np.array_equal(d["halo_index"], st["cat_halo_index"])),
                             "same_m200": bool(np.allclose(d["m200c_msun"], st["cat_m200c_msun"])),
                             "n": int(len(d["halo_index"]))}
        m = d["m200c_msun"]
        fe = d["ne_r200"] * MU_E * MP / (FB * m * MSUN)
        fg = d["m_r200"] / (FB * m)
        # shell containing x = 0.5
        edges = d["edges"]
        k = int(np.searchsorted(edges, 0.5) - 1)
        r200 = d["r200_mpc"] * MPC
        rho200 = m * MSUN / (4 / 3 * np.pi * r200**3)
        n200 = rho200 * FB / (X_H * MP)
        vol = 4 / 3 * np.pi * (edges[k + 1] ** 3 - edges[k] ** 3) * r200**3
        full = d["r100_mpc"] / d["r200_mpc"] >= edges[k + 1]
        ne_shell = d["ne3"][:, k] / vol / n200
        lm = np.log10(m)
        for b in np.unique(st["bin"]):
            sel = st["bin"] == b
            if sel.sum() < 8:
                continue
            lo = float(st["bin_edges"][b])
            fl = fe_lee(m[sel], z)
            # Lee22 shell mean over the same shell
            rr = np.linspace(edges[k], edges[k + 1], 201)
            wv = rr**2
            ne_lee_shell = np.array([np.trapz(L.lee_nbar(rr, mi * H, z) * wv, rr) / np.trapz(wv, rr) for mi in m[sel]])
            use = sel & full
            rows.append({"snap": s, "z": round(z, 3), "lo": round(lo, 1), "n": int(sel.sum()), "fe_FL": float(fe[sel].mean()),
                         "fgas_FL": float(fg[sel].mean()), "fe_Lee": float(fl.mean()),
                         "Fn": float(fl.mean() * (1 + z) ** 3 / fe[sel].mean()),
                         "ne_shell_FL": float(ne_shell[use].mean()) if use.sum() >= 3 else np.nan,
                         "ne_shell_Lee": float(ne_lee_shell[full[sel]].mean()) if use.sum() >= 3 else np.nan,
                         "shell": f"{edges[k]:.3f}-{edges[k + 1]:.3f}"})
        # all flagged centrals, SOAP gas mass
        cat = np.load(FPC / f"L2p8_m9_{s:04d}_reduced_haloes.npz")
        c = cat["is_central"].astype(bool) & (cat["m200c_msun"] > 0)
        mc = cat["m200c_msun"][c]
        fgc = cat["gas_mass_200c_msun"][c] / (FB * mc)
        lmc = np.log10(mc)
        for lo in np.round(np.arange(13.0, 15.0, 0.2), 1):
            sel = (lmc >= lo) & (lmc < lo + 0.2)
            if sel.sum() >= 20:
                soap_rows.append({"z": round(z, 3), "lo": lo, "n": int(sel.sum()), "fgas": float(fgc[sel].mean())})
    out = {"halo_index_check": idx_check, "per_stack": rows, "slopes": [], "soap_fgas_slopes": []}
    for lo in sorted({r["lo"] for r in rows}):
        sel = [r for r in rows if r["lo"] == lo]
        if len(sel) < 3:
            continue
        lz = np.log(1 + np.array([r["z"] for r in sel]))
        rec = {"lo": lo, "n_z": len(sel), "s_fe_FL": slope(lz, [r["fe_FL"] for r in sel]),
               "s_fgas_FL": slope(lz, [r["fgas_FL"] for r in sel]), "s_fe_Lee": slope(lz, [r["fe_Lee"] for r in sel]),
               "Fn_by_z": [round(r["Fn"], 2) for r in sel]}
        rec["s_Lee_minus_FL_fe"] = rec["s_fe_Lee"] - rec["s_fe_FL"]
        rec["s_tngcal_minus_FL_fe"] = rec["s_Lee_minus_FL_fe"] + 2.483190184608004
        good = [r for r in sel if np.isfinite(r["ne_shell_FL"])]
        if len(good) >= 3:
            lzg = np.log(1 + np.array([r["z"] for r in good]))
            rec["s_Lee_minus_FL_ne_x0.5shell"] = slope(lzg, [r["ne_shell_Lee"] for r in good]) - slope(lzg, [r["ne_shell_FL"] for r in good])
        out["slopes"].append(rec)
    for lo in sorted({r["lo"] for r in soap_rows}):
        sel = [r for r in soap_rows if r["lo"] == lo]
        if len(sel) >= 3:
            lz = np.log(1 + np.array([r["z"] for r in sel]))
            out["soap_fgas_slopes"].append({"lo": lo, "s_fgas_allcentrals": slope(lz, [r["fgas"] for r in sel]),
                                            "fgas_by_z": [round(r["fgas"], 3) for r in sel], "n_by_z": [r["n"] for r in sel]})
    fn = [r["Fn"] for r in rows if r["lo"] >= 13.6 - 1e-9 and r["lo"] <= 14.6 + 1e-9]
    out["Fn_median_13.6-14.6"] = float(np.median(fn))
    out["Fn_range_13.6-14.6"] = [float(np.min(fn)), float(np.max(fn))]
    out["Fn_median_by_z_13.6+"] = {str(zz): float(np.median([r["Fn"] for r in rows if abs(r["z"] - zz) < 1e-3 and r["lo"] >= 13.6 - 1e-9]))
                                   for zz in (0.0, 0.5, 1.0, 1.5, 2.0)}
    out["Fn_median_all"] = float(np.median([r["Fn"] for r in rows]))
    OUTV.mkdir(parents=True, exist_ok=True)
    (OUTV / "v2_density_fe.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    print(json.dumps(idx_check))
    print("lo   s_fe_FL s_fgas_FL s_fe_Lee Lee-FL tngcal-FL ne0.5shell(Lee-FL)  Fn_by_z")
    for r in out["slopes"]:
        print(f"{r['lo']:.1f} {r['s_fe_FL']:+.3f} {r['s_fgas_FL']:+.3f} {r['s_fe_Lee']:+.3f} {r['s_Lee_minus_FL_fe']:+.3f} "
              f"{r['s_tngcal_minus_FL_fe']:+.3f} {r.get('s_Lee_minus_FL_ne_x0.5shell', np.nan):+.3f} {r['Fn_by_z']}")
    for r in out["soap_fgas_slopes"]:
        print("SOAP all centrals", r)
    print("Fn median 13.6-14.6:", out["Fn_median_13.6-14.6"], out["Fn_range_13.6-14.6"], out["Fn_median_by_z_13.6+"], "all", out["Fn_median_all"])


if __name__ == "__main__":
    main()
