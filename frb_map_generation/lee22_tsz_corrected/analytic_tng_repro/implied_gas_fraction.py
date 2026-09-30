#!/usr/bin/env python3
"""Electron fraction inside R200c implied by Konietzka's TNG300 halo DM, per mass decade:
  f_e(implied) = <DM>_TNG(decade) / (A_eff x <DM>(f_e = 1)),
where <DM>(f_e = 1) is the mean DM over all rays if every halo held f_b M200c of fully ionized gas (0.88 electrons per
m_p) inside R200c (Tinker08, Planck15; profile-independent), and A_eff the TNG300/Tinker08 abundance of that decade
(hit fractions, mass_decomposition.py; Pillepich+18b-consistent 0.81 for the reinterpreted [1e13, 9e14] window).
Compared with the same ratio for each analytic profile (their own f_e inside R200c) and with CAMELS-TNG gas fractions.
Output: analysis/implied_gas_fraction.json."""
import json

import numpy as np

import repro_lib as R
import halo_model_dm_pdf as hm

COSMO = R.COSMO
FB = COSMO.omega_b / COSMO.omega_m
E_PER_MSUN = hm.MSUN_G * hm.ELECTRONS_PER_MP / hm.M_P_G


def unit_mean(lo, hi, z_s, nz=200, dlm=0.01):
    lm = np.arange(np.log10(lo), np.log10(hi), dlm) + dlm / 2
    M = 10 ** lm
    zz = (np.arange(nz) + 0.5) * z_s / nz
    tot = 0.0
    for z in zz:
        dn = hm.tinker08_dndlnm(COSMO, M, z) * np.log(10) * dlm                   # comoving Mpc^-3
        ne_com = (dn * FB * M).sum() * E_PER_MSUN / hm.MPC_CM ** 3                  # comoving electrons cm^-3
        tot += ne_com * (1 + z) * (1 / COSMO.H_over_c(z)) * (z_s / nz) * hm.MPC_CM / hm.PC_CM
    return tot


def main():
    md = json.loads((R.OUT / "analysis" / "mass_decomposition.json").read_text())["decades"]
    out = {}
    for key in ("1e9-1e10", "1e10-1e11", "1e11-1e12", "1e12-1e13", "1e13-9e14 reinterpreted", "all (>=1e9)"):
        d = md[key]
        lo, hi = d["window_msun"]
        out[key] = {}
        for z_s in (0.1, 0.5, 1.0):
            u = unit_mean(lo, hi, z_s)
            a = d["A_eff_from_hits"][str(z_s)] or d["A_eff_from_hits"]["0.7"]
            row = dict(unit_fe1_tinker=u, A_eff=a, fe_implied_tng=d["tng"][str(z_s)]["mean_all"] / (a * u))
            for m in R.MODELS:
                row[f"fe_{m}"] = d["model"][f"{m}|tinker08"][str(z_s)]["mean_all"] / u
            out[key][str(z_s)] = row
            print(f"{key:26s} z_s {z_s}: f_e implied TNG {row['fe_implied_tng']:.3f} (A_eff {a:.3f}) | "
                  + "  ".join(f"{m[:14]} {row['fe_' + m]:.3f}" for m in R.MODELS))
    (R.OUT / "analysis" / "implied_gas_fraction.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
