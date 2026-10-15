#!/usr/bin/env python3
"""Independent check of Fig. 2d/2e (fig_lee22_models_grid.py) of lee22_calibration.html v4.

Written from scratch from:
  Lee22 arXiv source (sample631.tex) Tables 7 (pressure, no c, BPL in P0, logMcut 13.60) and 8 (density, no c, BPL in x'c, beta',
  logMcut 13.61), eq. 10/11 with the (M/Mcut) pivot for every parameter, masses in h^-1 Msun;
  Battaglia+12 Table (AGN, Delta=200): P0 18.1/0.154/-0.758, xc 0.497/-0.00865/0.731, beta 4.35/0.0393/0.415, alpha 1, gamma -0.3;
  Battaglia+16 Table (AGN): rho0 4e3/0.29/-0.66, alpha 0.88/-0.03/0.19, beta 3.83/0.04/-0.025, xc 0.5, gamma -0.2;
  XGPaint conventions (vendor/XGPaint/src/profiles_y.jl, profiles_tau.jl): P_e = 0.5176 P_th; rho_gas = f_b rho_cr rho_fit with
  exponent -(beta+gamma)/alpha; ne = 0.9 rho_gas / m_per_e, m_per_e = m_e + 2X/(X+1) m_p + (1-X)/(2(1+X)) 4 m_p (CODATA 2018).
No repository module is imported.
"""
import json
from pathlib import Path

import numpy as np

OUT = Path("/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_verify_v4_20261001/models_grid")
MP, ME = 1.67262192369e-27, 9.1093837015e-31
XH, H = 0.76, 0.6774
MPERE = ME + 2 * XH / (XH + 1) * MP + (1 - XH) / (2 * (1 + XH)) * 4 * MP
NE2D = 0.9 * XH * MP / MPERE                      # XGPaint ne2d relative to printed n200 = 200 rho f_b/(X_H m_p)


def gnfw_b(x, xc, beta, gamma=-0.3, alpha=1.0):   # Battaglia12 / Lee22 printed form
    s = x / xc
    return s ** gamma * (1 + s ** alpha) ** (-beta)


def bpl(r, a, b):
    return np.where(r < 1, r ** a, r ** b)


def lee_P(x, mh, z):
    r = mh / 10 ** 13.60; zp = 1 + z
    P0 = 2.8 * bpl(r, 1.08, 0.80) * zp ** -1.89
    xc = 2.10 * r ** -0.35 * zp ** 0.53
    be = 9.4 * r ** -0.06 * zp ** 0.60
    return P0 * gnfw_b(x, xc, be)


def lee_n(x, mh, z):
    r = mh / 10 ** 13.61; zp = 1 + z
    n0 = 6.8 * r ** 0.68 * zp ** -2.11
    xc = 7.9 * bpl(r, 0.47, -0.45) * zp ** -0.67
    be = 19.5 * bpl(r, 0.70, -0.18) * zp ** -0.31
    return n0 * gnfw_b(x, xc, be)


def b12_Pe(x, m, z):
    mm = m / 1e14; zp = 1 + z
    return 0.5176 * 18.1 * mm ** 0.154 * zp ** -0.758 * gnfw_b(x, 0.497 * mm ** -0.00865 * zp ** 0.731,
                                                           4.35 * mm ** 0.0393 * zp ** 0.415)


def b16_ne_over_n200(x, m, z):
    mm = m / 1e14; zp = 1 + z
    rho0 = 4e3 * mm ** 0.29 * zp ** -0.66
    al = 0.88 * mm ** -0.03 * zp ** 0.19
    be = 3.83 * mm ** 0.04 * zp ** -0.025
    g = -0.2
    s = x / 0.5
    rho_fit = rho0 * s ** g * (1 + s ** al) ** (-(be + g) / al)   # rho_gas/(f_b rho_cr)
    # n_e/n200 = 0.9 f_b rho_cr rho_fit / m_per_e * X_H m_p / (200 rho_cr f_b)
    return NE2D / 200 * rho_fit


def main():
    res = {"NE2D": NE2D}
    zs = (0.0, 1.0, 2.0); ms = (2e13, 1e14, 5e14)
    res["pipe_over_cor_P"] = [(1 + z) ** -2 for z in zs]
    res["pipe_over_cor_n"] = [NE2D / ((H * (1 + z)) ** 3 * XH ** 2) for z in zs]
    res["z_equal"] = (NE2D / (H ** 3 * XH ** 2)) ** (1 / 3) - 1
    res["log10_mh"] = [float(np.log10(m * H)) for m in ms]
    res["log10_mh_h068"] = [float(np.log10(m * 0.68)) for m in ms]
    xs = np.geomspace(0.3, 1.0, 4001)
    xs300 = np.geomspace(0.0296, 1.79, 300); xs300 = xs300[(xs300 > .3) & (xs300 < 1)]
    rows = []
    for m in ms:
        for z in zs:
            mh = m * H
            d = {"M": m, "z": z}
            for q, bat, lee, conv in (("P", b12_Pe, lee_P, (1 + z) ** 2), ("n", b16_ne_over_n200, lee_n,
                                                                          (H * (1 + z)) ** 3 * XH ** 2)):
                f = lambda x: bat(x, m, z) / (lee(x, mh, z) * conv)
                r = f(xs); r300 = f(xs300)
                d[q + "_gm"] = float(np.exp(np.mean(np.log(r))))
                d[q + "_gm300"] = float(np.exp(np.mean(np.log(r300))))
                d[q + "_am"] = float(np.mean(r))
                d[q + "_min"], d[q + "_max"] = float(r.min()), float(r.max())
                for xx in (0.0296, 0.04, 0.06, 0.1, 1.0, 1.34, 1.79):
                    d["%s_at_%g" % (q, xx)] = float(f(np.array([xx]))[0])
                xc = np.geomspace(0.04, 0.1, 200)
                d[q + "_core_gm_0.04_0.1"] = float(np.exp(np.mean(np.log(f(xc)))))
                xo = np.geomspace(1.0, 1.34, 200)
                d[q + "_out_gm_1_1.34"] = float(np.exp(np.mean(np.log(f(xo)))))
            rows.append(d)
    res["rows"] = rows
    (OUT / "grid_check.json").write_text(json.dumps(res, indent=1))
    print("NE2D = %.5f   pipeline/corrected density: %s   pressure: %s   z_equal = %.4f" % (
        NE2D, ["%.4f" % v for v in res["pipe_over_cor_n"]], ["%.4f" % v for v in res["pipe_over_cor_P"]], res["z_equal"]))
    print("log10 M [h^-1 Msun] (h_TNG):", ["%.4f" % v for v in res["log10_mh"]], "  (h=0.68):",
          ["%.4f" % v for v in res["log10_mh_h068"]])
    for q in ("P", "n"):
        print("\n%s: model/corrected   gm(0.3-1) gm300  am  [min,max]  | x=0.0296 0.04 0.06 0.1 | core gm 0.04-0.1 | x=1 1.34 1.79" % q)
        for d in rows:
            print("  M=%.0e z=%g  %.3f %.3f %.3f [%.2f,%.2f] | %.2f %.2f %.2f %.2f | %.2f | %.2f %.2f %.2f" % (
                d["M"], d["z"], d[q + "_gm"], d[q + "_gm300"], d[q + "_am"], d[q + "_min"], d[q + "_max"],
                d[q + "_at_0.0296"], d[q + "_at_0.04"], d[q + "_at_0.06"], d[q + "_at_0.1"], d[q + "_core_gm_0.04_0.1"],
                d[q + "_at_1"], d[q + "_at_1.34"], d[q + "_at_1.79"]))


if __name__ == "__main__":
    main()
