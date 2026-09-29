"""Independent re-derivation of workstream C's pressure claims from SOAP Y(<R200c).

Uses ALL flagged centrals in the catalogue caches (notebook_cache/L2p8_m9_NNNN_reduced_haloes.npz, ~7-9k per snapshot),
not only the 10-200 haloes per bin of the preliminary stacks, so it is statistically independent of the stack selection.

Checks
  (a) R200c physical: M200c / (4/3 pi R200c^3) / (200 rho_cr(z)) with rho_cr from SOAP's own H(z) (stored as
      'H0_km_s_mpc' in the cache json - it is H(z) in km/s/Mpc).
  (b) Yhat200 = Y_SOAP(<R200c) / [sigma_T/(m_e c^2) P200 R200c^3], P200 = G M200c (200 rho_cr) f_b / (2 R200c),
      recomputed here from M200c, R200c and my own constants (not cat_p200_pa).
  (c) Lee22 no-c (Table 7, via lee22_models, masses x h = 0.681) and B12 (P_e = 0.5176 P_th) Y200/(P200 R200^3)
      = 4 pi int_0^1 Pbar x^2 dx, per halo, with a dense trapezoid on log x (independent of the Gauss rules used there).
  (d) slopes s = dln Yhat/dln(1+z) over z = 0-2 per 0.2-dex M200c bin (unweighted OLS on the bin mean, and on the
      bin median), and the ratios Lee22 x (1+z)^2 / FLAMINGO.
Output: OUTV/v1_pressure_y200.json and .csv
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/cbllover/HalfDome/frb_map_generation/lee22_normalization_forensics/C_pressure_consistency")
import lee22_models as L  # noqa: E402

CACHE = Path("/home/cbllover/HalfDome/flamingo_pressure_calibration/notebook_cache")
OUTV = Path("/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_tsz_verification_20260929/verify_C_flamingo")
SNAPS = [78, 68, 58, 48, 38, 18]
H = 0.681
# SI constants (CODATA 2018)
G = 6.67430e-11
MSUN = 1.98841e30
MPC = 3.0856775814913673e22
SIGMA_T = 6.6524587321e-29
ME_C2 = 9.1093837015e-31 * 299792458.0**2

lnx = np.linspace(np.log(1e-5), 0.0, 4001)
X = np.exp(lnx)


def y_model(pbar_fun, masses, z):
    """4 pi int_0^1 Pbar x^2 dx = 4 pi int Pbar x^3 dlnx, per halo (trapezoid on 4001 log nodes)."""
    out = np.empty(len(masses))
    for i, m in enumerate(masses):
        f = pbar_fun(X, m, z) * X**3
        out[i] = 4 * np.pi * np.trapz(f, lnx)
    return out


def main():
    rows = []
    checks = {}
    for s in SNAPS:
        d = np.load(CACHE / f"L2p8_m9_{s:04d}_reduced_haloes.npz")
        meta = json.loads((CACHE / f"L2p8_m9_{s:04d}_reduced_haloes.json").read_text())
        z = meta["cosmology"]["redshift"]
        fb = meta["cosmology"]["Omega_b"] / meta["cosmology"]["Omega_m_total"]
        cos = meta["cosmology"]
        Hz_lcdm = 100 * cos["h"] * np.sqrt(cos["Omega_m_total"] * (1 + z) ** 3 + 1 - cos["Omega_m_total"]) * 1e3 / MPC
        Hz = cos["H0_km_s_mpc"] * 1e3 / MPC if "H0_km_s_mpc" in cos else Hz_lcdm   # stored value is H(z)
        rho_cr = 3 * Hz**2 / (8 * np.pi * G)
        c = d["is_central"].astype(bool) & (d["m200c_msun"] > 0) & (d["y_200_crit_mpc2"] > 0)
        m = d["m200c_msun"][c]
        r = d["r_200_crit_mpc"][c]
        y = d["y_200_crit_mpc2"][c]
        mkg, rm = m * MSUN, r * MPC
        rho200 = mkg / (4 / 3 * np.pi * rm**3)
        checks[str(s)] = {"z": z, "n_centrals": int(c.sum()), "Hz_soap_over_lcdm": float(Hz / Hz_lcdm),
                          "rho200_over_200rhocr_median": float(np.median(rho200 / (200 * rho_cr))),
                          "rho200_over_200rhocr_range": [float(np.min(rho200 / (200 * rho_cr))), float(np.max(rho200 / (200 * rho_cr)))]}
        P200 = G * mkg * (200 * rho_cr) * fb / (2 * rm)          # Pa, using the exact 200 rho_cr
        yhat = y * MPC**2 / (SIGMA_T / ME_C2 * P200 * rm**3)
        lm = np.log10(m)
        for lo in np.round(np.arange(13.0, 15.4, 0.2), 1):
            sel = (lm >= lo) & (lm < lo + 0.2)
            if sel.sum() < 8:
                continue
            mm = m[sel]
            ylee = y_model(lambda x, mq, zq: L.lee_Pbar(x, mq * H, zq), mm, z)
            yb12 = y_model(lambda x, mq, zq: L.b12_Pe_over_P200(x, mq, zq), mm, z)
            rows.append({"snap": s, "z": round(z, 3), "lo": lo, "n": int(sel.sum()),
                         "Y_FL_mean": float(yhat[sel].mean()), "Y_FL_median": float(np.median(yhat[sel])),
                         "Y_FL_sem": float(yhat[sel].std() / np.sqrt(sel.sum())),
                         "Y_Lee_mean": float(ylee.mean()), "Y_B12_mean": float(yb12.mean()),
                         "Lee_over_FL": float(ylee.mean() / yhat[sel].mean()),
                         "Lee_a2_over_FL": float(ylee.mean() * (1 + z) ** 2 / yhat[sel].mean()),
                         "B12_over_FL": float(yb12.mean() / yhat[sel].mean())})
    # slopes over z <= 2
    slopes = []
    for lo in sorted({r["lo"] for r in rows}):
        sel = [r for r in rows if r["lo"] == lo and r["z"] <= 2.01]
        if len(sel) < 3:
            continue
        lz = np.log(1 + np.array([r["z"] for r in sel]))
        fit = lambda v: float(np.polyfit(lz, np.log(v), 1)[0])  # noqa: E731
        sFL = fit([r["Y_FL_mean"] for r in sel])
        sFLmed = fit([r["Y_FL_median"] for r in sel])
        sLee = fit([r["Y_Lee_mean"] for r in sel])
        sB12 = fit([r["Y_B12_mean"] for r in sel])
        slopes.append({"lo": lo, "n_z": len(sel), "s_FL_mean": sFL, "s_FL_median": sFLmed, "s_Lee": sLee, "s_B12": sB12,
                       "s_Lee_minus_FL": sLee - sFL, "s_Lee_minus_FLmedian": sLee - sFLmed, "s_B12_minus_FL": sB12 - sFL,
                       "s_Lee_a2_minus_FL": sLee + 2 - sFL})
    # medians of ratios per z over 13.6 <= lo < 14.8 (inside Lee22's range: M h <= 10^14.8 -> M <= 10^14.97)
    med = {}
    for zz in (0.0, 0.5, 1.0, 1.5, 2.0):
        sel = [r for r in rows if abs(r["z"] - zz) < 1e-3 and 13.6 <= r["lo"] <= 14.6 + 1e-9]
        med[f"z{zz:g}"] = {k: float(np.median([r[k] for r in sel])) for k in ("Lee_over_FL", "Lee_a2_over_FL", "B12_over_FL")}
        med[f"z{zz:g}"]["n_bins"] = len(sel)
    OUTV.mkdir(parents=True, exist_ok=True)
    (OUTV / "v1_pressure_y200.json").write_text(json.dumps({"checks": checks, "slopes_zle2": slopes, "median_ratios_13.6-14.8": med},
                                                          indent=1) + "\n")
    with open(OUTV / "v1_pressure_y200.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(json.dumps(checks, indent=0))
    print("lo    s_FL   s_FLmed s_Lee  Lee-FL  Lee-FLmed  B12-FL  Lee_a2-FL")
    for r in slopes:
        print(f"{r['lo']:.1f} {r['s_FL_mean']:+.3f} {r['s_FL_median']:+.3f} {r['s_Lee']:+.3f} {r['s_Lee_minus_FL']:+.3f} "
              f"{r['s_Lee_minus_FLmedian']:+.3f} {r['s_B12_minus_FL']:+.3f} {r['s_Lee_a2_minus_FL']:+.3f}")
    print(json.dumps(med, indent=0))


if __name__ == "__main__":
    main()
