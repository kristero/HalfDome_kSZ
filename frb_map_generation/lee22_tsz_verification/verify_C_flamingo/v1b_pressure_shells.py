"""Shell-level check of the P_e/P200 redshift slopes at x ~ 0.3, 0.5, 1 (no interpolation, no resolution-cut code reuse).

From the raw stack products: per halo, shell pressure = y3_raw * m_e c^2/sigma_T / V_shell (only shells fully inside
the halo's R100c), / P200 recomputed from M200c, R200c and 200 rho_cr = M/(4/3 pi R^3); equal-weight halo mean per
(snapshot, 0.2-dex bin); Lee22 printed (no-c Table 7, M x 0.681) and B12 P_e shell means on the same shells;
slopes dln/dln(1+z) over z <= 2 per bin (OLS on bin means).
Also P_e/P_total for hot gas (T > 1e6 K) from the 109 z = 0 particle cutouts (mass- and volume-weighted).
Output: OUTV/v1b_pressure_shells.json
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/cbllover/HalfDome/frb_map_generation/lee22_normalization_forensics/C_pressure_consistency")
import lee22_models as L  # noqa: E402

ST = Path("/home/cbllover/HalfDome/flamingo_pressure_calibration/notebook_cache/stacks")
PART = Path("/home/cbllover/HalfDome/flamingo_pressure_calibration/notebook_cache/particles")
OUTV = Path("/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_tsz_verification_20260929/verify_C_flamingo")
SNAPS = [78, 68, 58, 48, 38]
H, FB = 0.681, 0.0486 / 0.30600007943162144
G, MSUN, MPC = 6.67430e-11, 1.98841e30, 3.0856775814913673e22
SIGMA_T, ME_C2 = 6.6524587321e-29, 9.1093837015e-31 * 299792458.0**2
KB = 1.380649e-23


def shell_avg(fun, lo, hi, m, z):
    r = np.linspace(lo, hi, 257)
    w = r**2
    return np.array([np.trapz(fun(r, mi, z) * w, r) / np.trapz(w, r) for mi in m])


def main():
    rows = []
    for s in SNAPS:
        d = np.load(ST / f"L2p8_m9_{s:04d}_stack_products.npz")
        z = float(d["redshift"])
        e = d["edges"]
        m, r = d["cat_m200c_msun"], d["cat_r_200_crit_mpc"]
        x100 = d["cat_r_100_crit_mpc"] / r
        mkg, rm = m * MSUN, r * MPC
        rho200 = mkg / (4 / 3 * np.pi * rm**3)
        p200 = G * mkg * rho200 * FB / (2 * rm)
        for xq in (0.3, 0.5, 1.0):
            k = int(np.searchsorted(e, xq) - 1)
            vol = 4 / 3 * np.pi * (e[k + 1] ** 3 - e[k] ** 3) * rm**3
            pe = d["y3_raw"][:, k] * MPC**2 * ME_C2 / SIGMA_T / vol
            ratio = pe / p200
            full = x100 >= e[k + 1]
            for b in np.unique(d["bin"]):
                sel = (d["bin"] == b) & full
                if (d["bin"] == b).sum() < 8 or sel.sum() < 5:
                    continue
                lo = round(float(d["bin_edges"][b]), 1)
                lee = shell_avg(lambda xx, mm, zz: L.lee_Pbar(xx, mm * H, zz), e[k], e[k + 1], m[sel], z).mean()
                b12 = shell_avg(L.b12_Pe_over_P200, e[k], e[k + 1], m[sel], z).mean()
                rows.append({"z": round(z, 3), "lo": lo, "x": xq, "shell": f"{e[k]:.3f}-{e[k + 1]:.3f}", "n": int(sel.sum()),
                             "FL": float(ratio[sel].mean()), "Lee": float(lee), "B12": float(b12),
                             "min_particles": int(d["n3"][sel, k].min())})
    slopes = []
    for xq in (0.3, 0.5, 1.0):
        for lo in sorted({r["lo"] for r in rows}):
            sel = [r for r in rows if r["lo"] == lo and r["x"] == xq]
            if len(sel) < 3:
                continue
            lz = np.log(1 + np.array([r["z"] for r in sel]))
            f = lambda k: float(np.polyfit(lz, np.log([r[k] for r in sel]), 1)[0])  # noqa: E731
            slopes.append({"x": xq, "lo": lo, "n_z": len(sel), "s_FL": f("FL"), "s_Lee_minus_FL": f("Lee") - f("FL"),
                           "s_B12_minus_FL": f("B12") - f("FL"), "min_particles": min(r["min_particles"] for r in sel)})
    # P_e / P_total
    pm, pv = [], []
    for fpath in sorted(glob.glob(str(PART / "*.npz"))):
        p = np.load(fpath)
        hot = p["temperature_k"] > 1e6
        pe = p["ne_m3"][hot] * KB * p["temperature_k"][hot].astype(float)
        pt = p["p_total_pa"][hot].astype(float)
        mass = p["mass_msun"][hot].astype(float)
        vol = mass / p["density_msun_mpc3"][hot].astype(float)
        pm.append(np.sum(pe * mass) / np.sum(pt * mass))
        pv.append(np.sum(pe * vol) / np.sum(pt * vol))
    out = {"rows": rows, "slopes": slopes, "Pe_over_Ptot_hot": {"n": len(pm), "mass_weighted_median": float(np.median(pm)),
                                                                   "volume_weighted_median": float(np.median(pv)),
                                                                   "volume_weighted_range": [float(min(pv)), float(max(pv))]}}
    (OUTV / "v1b_pressure_shells.json").write_text(json.dumps(out, indent=1) + "\n")
    print(" x    lo   n_z  s_FL   Lee-FL  B12-FL  minpart")
    for r in slopes:
        print(f"{r['x']:.1f} {r['lo']:.1f} {r['n_z']} {r['s_FL']:+.3f} {r['s_Lee_minus_FL']:+.3f} {r['s_B12_minus_FL']:+.3f} {r['min_particles']}")
    print(out["Pe_over_Ptot_hot"])


if __name__ == "__main__":
    main()
