#!/usr/bin/env python3
"""Part (1): the gas content implied by Lee22's density fits under each normalization reading.

Outputs (frb_map_generation/outputs/lee22_normalization_forensics_20260928/D_gas_fractions/):
  validation_vs_julia_table.json      Python formula vs the Julia-tabulated production profile (xgpnative)
  lee22_gas_fractions_grid.csv        f_e(<R200c), f_e(<R500c) for every variant, fit, mass, redshift
  lee22_gas_fractions_summary.json    same, nested, plus the (1+z) power of the literal enclosed content
"""
import csv
import json
from pathlib import Path

import h5py
import numpy as np
from scipy import integrate

import gasfrac_lib as L

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "lee22_normalization_forensics_20260928" / "D_gas_fractions"
TABLE = ROOT / "outputs" / "analytic_halo_pdf_20260925" / "tables" / "halfdome_sphere_profiles.h5"
OUT.mkdir(parents=True, exist_ok=True)

LOG10M_H = [13.0, 13.4, 13.8, 14.2, 14.6]
ZS = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0]
h_tng = L.TNG["h"]


def variants(z):
    """Multiplicative factor on the literal eq.-9 electron density."""
    zp3 = (1 + z) ** 3
    return {
        "literal": 1.0,
        "xgpnative_0.6016": L.XGP_NATIVE_FACTOR,
        "XH2_0.5776": L.X_H ** 2,
        "electron_count_0.6688": L.X_H * (1 + L.X_H) / 2,
        "comoving_hyp_(1+z)^3/E^2": zp3 / L.E2(L.TNG, z),
        "(1+z)^3": zp3,
        "(1+z)^3*h^2": zp3 * h_tng ** 2,
        "(1+z)^3*h^3": zp3 * h_tng ** 3,
        "(1+z)^3*h^3*XH^2": zp3 * h_tng ** 3 * L.X_H ** 2,
        "(1+z)^3*XH^2": zp3 * L.X_H ** 2,
        "(1+z)^3*XH^2*fb(=Ob/Om)": zp3 * L.X_H ** 2 * L.TNG["ob"] / L.TNG["om"],
    }


# ------------------------------------------------------------------ validation against the Julia table
def validate():
    cosmo = L.HALFDOME
    h = cosmo["h"]
    G = 6.6743e-11
    MPC_M = 3.085677581491367e22
    PC_M = 3.0856775814913674e16
    MSUN = 1.98847e30
    fb = cosmo["ob"] / cosmo["om"]
    rho0 = 3 * (100 * h * 1e3 / MPC_M) ** 2 / (8 * np.pi * G)      # kg m^-3
    with h5py.File(TABLE, "r") as f:
        lm = f["log10_m200c_msun"][:]
        zz = f["redshift"][:]
        xx = f["x_r200c"][:]
        tab = f["lee22_noconc_xgpnative/chord_mean_dm_pc_cm3"]
        r200tab = f["lee22_noconc_xgpnative/r200c_mpc"]
        rng = np.random.default_rng(20260928)
        # nodes: random plus a grid inside the fit range
        nodes = [(int(rng.integers(0, len(lm))), int(rng.integers(0, len(zz))), int(rng.integers(0, len(xx))))
                 for _ in range(300)]
        for j in np.searchsorted(lm, [12.83, 13.2, 13.6, 14.0, 14.5, 15.0]):
            for i in (0, 25, 49, 74, 99):
                for k in (0, 60, 100, 150, 199):
                    nodes.append((int(j), i, k))
        worst, worst_shape, worst_r = 0.0, 0.0, 0.0
        rows = []
        for (j, i, k) in nodes:
            m = 10 ** lm[j]
            z = zz[i]
            x = xx[k]
            lmh = lm[j] + np.log10(h)
            n0, xc, beta = L.lee22_noc_params(lmh, z, shape_clip_log10m_h=14.8)
            rhoc = rho0 * L.E2(cosmo, z, radiation=True)
            r200_m = (3 * m * MSUN / (4 * np.pi * 200 * rhoc)) ** (1 / 3)
            n200_cm3 = 200 * rhoc * fb / (L.X_H * L._MP) * 1e-6
            Lc = np.sqrt(1 - x ** 2)
            integ, _ = integrate.quad(lambda l: L.gshape(np.sqrt(x ** 2 + l ** 2), xc, beta), 0, Lc,
                                      epsabs=0, epsrel=1e-11, limit=400)
            g = L.XGP_NATIVE_FACTOR * n0 * n200_cm3 * (r200_m / PC_M) / (1 + z) * integ / Lc
            ref = tab[j, i, k]
            d = abs(g / ref - 1)
            worst = max(worst, d)
            worst_r = max(worst_r, abs(r200_m / MPC_M / r200tab[j, i] - 1))
            rows.append(d)
    res = dict(n_nodes=len(nodes), max_rel_diff_chord_mean_dm=worst, median_rel_diff=float(np.median(rows)),
               max_rel_diff_r200c=worst_r,
               note="Python Lee22 no-c formula (Mcut pivot, shape clip 10^14.8 h^-1, factor 0.9 X_H m_p/m_per_e, "
                    "XGPaint rho_crit incl. radiation) vs Julia table lee22_noconc_xgpnative")
    print("validation:", res)
    (OUT / "validation_vs_julia_table.json").write_text(json.dumps(res, indent=2))
    return res


# ------------------------------------------------------------------ grid
def main():
    val = validate()
    rows = []
    summary = {"validation": val, "grid": {}}
    for fit in ("noc_table8", "c_table3_tngmean"):
        for lmh in LOG10M_H:
            for z in ZS:
                c = float(L.tng_mean_concentration(lmh, z))
                if fit == "noc_table8":
                    n0, xc, beta = L.lee22_noc_params(lmh, z)
                else:
                    n0, xc, beta = L.lee22_c_params(lmh, z, c=c)
                n0, xc, beta = float(n0), float(xc), float(beta)
                x500, m500_m200 = L.nfw_r_over_r200(c, 500.0)
                mom1 = L.shape_moment(xc, beta, 1.0)
                mom500 = L.shape_moment(xc, beta, x500)
                mom_in = L.shape_moment(xc, beta, 0.04)          # below the fitted radial range
                nbar_vol = 3 * n0 * mom1                         # <n_e/n200> inside R200c (literal)
                fe200_lit = L.F_E_PER_NBAR * nbar_vol            # f_e(<R200c) literal
                fe500_lit = L.F_E_PER_NBAR * 3 * n0 * mom500 / m500_m200
                for vname, K in variants(z).items():
                    row = dict(fit=fit, log10M_h=lmh, M200c_msun_tng=10 ** lmh / h_tng, z=z, c200_proxy=c,
                               n0=n0, xc=xc, beta_prime=beta, nbar_vol_mean_R200c_literal=nbar_vol,
                               frac_inside_0p04=mom_in / mom1, R500_over_R200=x500, M500_over_M200=m500_m200,
                               variant=vname, factor=K,
                               fe_R200c=K * fe200_lit, fe_R500c=K * fe500_lit)
                    rows.append(row)
                    summary["grid"].setdefault(fit, {}).setdefault(vname, {}).setdefault(str(lmh), {})[str(z)] = \
                        dict(fe_R200c=K * fe200_lit, fe_R500c=K * fe500_lit)
    with open(OUT / "lee22_gas_fractions_grid.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    # effective redshift power of the literal enclosed content, per mass (least squares in log over z=0..2)
    pw = {}
    for fit in ("noc_table8", "c_table3_tngmean"):
        for lmh in LOG10M_H:
            zz = np.array(ZS)
            fe = np.array([summary["grid"][fit]["literal"][str(lmh)][str(z)]["fe_R200c"] for z in ZS])
            p = np.polyfit(np.log(1 + zz), np.log(fe), 1)[0]
            pw.setdefault(fit, {})[str(lmh)] = p
    summary["literal_fe_R200c_power_of_1pz"] = pw
    (OUT / "lee22_gas_fractions_summary.json").write_text(json.dumps(summary, indent=1))
    # console table
    for fit in ("noc_table8", "c_table3_tngmean"):
        print("\n==", fit, " f_e(<R200c) = M_gas,ion(<R200c)/(f_b M200c)")
        for vname in variants(0.0):
            print("  --", vname)
            print("     log10Mh " + " ".join("z={:<5}".format(z) for z in ZS))
            for lmh in LOG10M_H:
                vals = [summary["grid"][fit][vname][str(lmh)][str(z)]["fe_R200c"] for z in ZS]
                print("     {:5.1f}   ".format(lmh) + " ".join("{:7.3f}".format(v) for v in vals))
        print("  power of (1+z) of literal f_e(<R200c):", {k: round(v, 3) for k, v in pw[fit].items()})


if __name__ == "__main__":
    main()
