#!/usr/bin/env python3
"""Integrated content of Lee22's no-c fits (arXiv v1 Table 7 pressure, Table 8 density) vs M and z.

(1) Check the Table 7/8 transcription and the pivot convention against the orange
    'without c, with BPL' vector curves of Figs. 2 and 5 (z=0).
(2) For x_max = 1 (R200) compute, as functions of z at fixed M (h^-1 Msun, as in the paper):
      nbar = <n_e/n200>_V,   M_gas/(f_b M200) = (mu_e/X_H) nbar,
      pbar = <P_e/P200>_V,   (M_gas/f_b M200)(T_mw/T200) = pbar/0.5176,
      T_mw/T200 = (X_H/mu) pbar/nbar          [volume-independent]
    and fit d ln Q / d ln(1+z) over z = 0..2 (the fitted snapshot range).
(3) Tabulate the predicted alpha_z shifts of simple bookkeeping errors (see README in the results json).
"""
from pathlib import Path
import json
import numpy as np
from scipy.integrate import quad

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "outputs/lee22_normalization_forensics_20260928/B_paper_code_forensics"
XH = 0.76
MU_E = 2 / (1 + XH)
MU = 4 / (3 + 5 * XH)
PE_PTH = 0.5176


def bpl(r, a, ap):
    return r ** a if r < 1 else r ** ap


def dens_pars(lm, z, n0_pivot="mcut"):
    r = 10 ** (lm - 13.61)
    rn0 = r if n0_pivot == "mcut" else 10 ** (lm - 14.0)
    n0 = 6.8 * rn0 ** 0.68 * (1 + z) ** -2.11
    xc = 7.9 * bpl(r, 0.47, -0.45) * (1 + z) ** -0.67
    be = 19.5 * bpl(r, 0.70, -0.18) * (1 + z) ** -0.31
    return n0, xc, be


def pres_pars(lm, z, shape_pivot="mcut"):
    r = 10 ** (lm - 13.60)
    rs = r if shape_pivot == "mcut" else 10 ** (lm - 14.0)
    p0 = 2.8 * bpl(r, 1.08, 0.80) * (1 + z) ** -1.89
    xc = 2.10 * rs ** -0.35 * (1 + z) ** 0.53
    be = 9.4 * rs ** -0.06 * (1 + z) ** 0.60
    return p0, xc, be


def gnfw(x, a0, xc, be):
    return a0 * (x / xc) ** -0.3 * (1 + x / xc) ** -be


def vol_mean(f, xmax=1.0):
    return 3 * quad(lambda x: x * x * f(x), 0, xmax, limit=200)[0] / xmax ** 3


def main():
    res = {}
    d = np.genfromtxt(OUT / "lee22_fig2_fig5_vector_curves.csv", delimiter=",", names=True, dtype=None, encoding="utf-8")
    print("=== (1) transcription/pivot check vs the orange (no c, BPL) figure curves, z=0 ===")
    chk = {}
    for fig, fun, opts in (("fig5_density", dens_pars, ("mcut", "1e14")), ("fig2_pressure", pres_pars, ("mcut", "1e14"))):
        for panel, lc in ((0, 13.3), (1, 13.9)):
            m = (d["figure"] == fig) & (d["panel"] == panel) & (d["curve"] == "without_c_bpl")
            x, q = d["x_r_over_R200"][m], d["x3_times_ratio"][m]
            for o in opts:
                a0, xc, be = fun(lc, 0.0, o)
                model = x ** 3 * gnfw(x, a0, xc, be)
                r = q / model - 1
                chk[f"{fig}_panel{panel}_pivot{o}"] = dict(mean=float(r.mean()), min=float(r.min()), max=float(r.max()))
                print(f"{fig} panel {panel} (logM h={lc}) pivot={o:5s}: figure/model-1 mean {100*r.mean():+.2f}%  range [{100*r.min():+.2f},{100*r.max():+.2f}]%")
    res["transcription_check"] = chk

    print("\n=== (2) integrated content of the fits within R200 ===")
    zs = np.array([0.0, 0.2, 0.5, 0.7, 1.0, 1.5, 2.0])
    tab = {}
    for lm in (13.0, 13.3, 13.61, 13.9, 14.2, 14.5):
        rows = []
        for z in zs:
            n0, xcn, ben = dens_pars(lm, z)
            p0, xcp, bep = pres_pars(lm, z)
            nbar = vol_mean(lambda x: gnfw(x, n0, xcn, ben))
            pbar = vol_mean(lambda x: gnfw(x, p0, xcp, bep))
            # mass-weighted temperature = int P_e dV / int n_e dV in T200 units
            rows.append(dict(z=float(z), nbar=nbar, fgas=MU_E / XH * nbar, pbar=pbar, fgasT=pbar / PE_PTH,
                             Tmw=XH / MU * pbar / nbar, T_at_R200=XH / MU * gnfw(1.0, p0, xcp, bep) / gnfw(1.0, n0, xcn, ben)))
        lz = np.log(1 + zs)
        slopes = {k: float(np.polyfit(lz, np.log([r[k] for r in rows]), 1)[0]) for k in ("nbar", "pbar", "Tmw", "T_at_R200")}
        tab[str(lm)] = dict(rows=rows, dlnQ_dln1pz_z0to2=slopes)
        print(f"log10(M h/Msun)={lm}:  slopes d ln/d ln(1+z): nbar {slopes['nbar']:+.2f}  pbar {slopes['pbar']:+.2f}  Tmw {slopes['Tmw']:+.2f}")
        for r in rows:
            print(f"   z={r['z']:.1f}  M_gas/(f_b M200)={r['fgas']:.3f}  (M_gas T_mw)/(f_b M200 T200)={r['fgasT']:.3f}  T_mw/T200={r['Tmw']:.3f}  T(R200)/T200={r['T_at_R200']:.3f}")
    res["integrated_fits_R200"] = tab

    # (3) predicted alpha_z shift and z=0 amplitude factor for bookkeeping errors, (measured/true) = F0 (1+z)^k
    h = 0.6774
    hyps = [
        ("H0 none (paper as printed)", 1, 0, 1, 0),
        ("H1 shell volume from comoving (ckpc/h) coordinates, mass physical, n200/P200 physical", h ** -3, -3, h ** -3, -3),
        ("H1b comoving shell volume but h handled (V in ckpc^3)", 1, -3, 1, -3),
        ("H2 volume + masses in code units (1e10 Msun/h, (ckpc/h)^3) vs physical n200", h ** -2, -3, h ** -2, -3),
        ("H3 all quantities consistently in comoving h-units, n200/P200 with rho_cr(z) physical", 1, -3, 1, -3),
        ("H4 H3 + R200 comoving in P200 (P200 ~ M/R: P200 too low by (1+z)^-1 at fixed M)", 1, -3, 1, -2),
        ("H5 eq. 9 printed 1/(X_H m_p), code used X_H/m_p (n only)", XH ** -2, 0, 1, 0),
        ("H6 f_b applied twice in n200 (n only)", 1 / 0.157334, 0, 1, 0),
        ("H7 Omega_m omitted in n200 (200 rho_cr Omega_b/(X_H m_p))", 1 / 0.3089, 0, 1, 0),
        ("H8 n200 from rho_cr(z=0) (no E^2), physical volumes", "E^2(z)", 0, 1, 0),
        ("H9 comoving densities over rho_cr(z=0): repo ':comoving_hypothesis' inverse", "E^2/(1+z)^3", 0, 1, 0),
        ("H10 Rockstar M200c [Msun/h] used as Msun in P200 (R200 recomputed): P200 ~ M^(2/3)", 1, 0, h ** (-2 / 3), 0),
    ]
    res["hypotheses_measured_over_true"] = [dict(name=a, n_F0=str(b), n_k=c, P_F0=str(d_), P_k=e) for a, b, c, d_, e in hyps]
    print("\n=== (3) hypotheses: measured/true = F0 (1+z)^k  (true alpha_z = fitted alpha_z - k) ===")
    for a, b, c, d_, e in hyps:
        print(f"{a}\n     n: F0={b if isinstance(b,str) else round(b,4)}  k={c}  -> true az(n0) = {-2.11 - c:+.2f};   "
              f"P: F0={d_ if isinstance(d_,str) else round(d_,4)}  k={e} -> true az(P0) = {-1.89 - e:+.2f}")
    with open(OUT / "fit_table_scalings.json", "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
