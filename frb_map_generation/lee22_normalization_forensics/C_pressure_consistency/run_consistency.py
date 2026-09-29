#!/usr/bin/env python3
"""Lee22 internal-consistency forensics (task C): implied electron temperature, comparison with
Battaglia12 (pressure) / Battaglia16 (density), observational X-ray anchors, and the paper's own
Figure 2 / Figure 5 curves (including how the paper drew the Battaglia reference curves).

Inputs: lee22_models.py, outputs/.../lee22_fig2_fig5_vector_curves.csv (extract_lee22_fig2_fig5.py),
X-COP stacked density table (Ghirardini+2019, A&A 621, A41, Table 'Stacked thermodynamic
profiles', transcribed below from the arXiv:1805.00042 TeX source).
Outputs: outputs/lee22_normalization_forensics_20260928/C_pressure_consistency/*.json|*.csv
"""
import csv
import json
from pathlib import Path

import numpy as np
from scipy import integrate

import lee22_models as L

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
OUT = REPO / "frb_map_generation/outputs/lee22_normalization_forensics_20260928/C_pressure_consistency"
CURVES = OUT / "lee22_fig2_fig5_vector_curves.csv"
C = L.TNG
h = C.h
FB = C.fb
ZSNAP = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.5, 2.0])   # TNG full snapshots <= 2
RESULTS = {}


def q(f, a, b):
    return integrate.quad(f, a, b, epsabs=0, epsrel=1e-10, limit=400)[0]


def powerlaw(z, y):
    """least-squares y = A (1+z)^k in log space; returns A, k, max |log residual| (dex)."""
    lz, ly = np.log10(1 + np.asarray(z)), np.log10(np.asarray(y))
    k, a = np.polyfit(lz, ly, 1)
    return 10 ** a, k, float(np.max(np.abs(ly - (a + k * lz))))


def mean_abs_frac(a, b):
    return float(np.mean(np.abs(a / b - 1)))


# ================================================================== A. the paper's own figures
def load_curves():
    cur = {}
    with CURVES.open() as f:
        for r in csv.DictReader(f):
            key = (r["figure_asset"], int(r["panel"]), r["curve"])
            cur.setdefault(key, [[], []])
            cur[key][0].append(float(r["x_r_over_R200"]))
            cur[key][1].append(float(r["value"]))
    return {k: (np.array(v[0]), np.array(v[1])) for k, v in cur.items()}


def figure_forensics():
    cur = load_curves()
    res = {}
    bins = {0: (13.2, 13.4), 1: (13.8, 14.0)}
    fb = FB
    for panel, (lo, hi) in bins.items():
        lm = 0.5 * (lo + hi)
        lms = np.linspace(lo, hi, 101)
        mh = 10 ** lm
        d = {}
        # --- pressure figure (fig_2.pdf): orange = Table 7 (no c, BPL)
        xP, orangeP = cur[("fig_2.pdf", panel, "without_c_bpl")]
        _, simP = cur[("fig_2.pdf", panel, "simulation")]
        _, cyanP = cur[("fig_2.pdf", panel, "battaglia")]
        modP = xP ** 3 * L.lee_Pbar(xP, mh, 0.0)
        modP_bin = xP ** 3 * np.mean([L.lee_Pbar(xP, 10 ** l, 0.0) for l in lms], axis=0)
        d["table7_vs_fig2_orange_mean_abs_frac_centre"] = mean_abs_frac(modP, orangeP)
        d["table7_vs_fig2_orange_mean_abs_frac_binavg"] = mean_abs_frac(modP_bin, orangeP)
        d["table7_vs_fig2_orange_median_ratio_centre"] = float(np.median(modP / orangeP))
        d["fig2_orange_vs_red_simulation_median_ratio"] = float(np.median(orangeP / simP))
        d["fig2_red_simulation_over_cyan_B12Pth_median_x>0.5"] = float(np.median((simP / cyanP)[xP > 0.5]))
        d["fig2_red_simulation_over_cyan_B12Pth_median_all"] = float(np.median(simP / cyanP))
        d["fig2_red_simulation_over_B12_electron_pressure_median_x>0.5"] = float(
            np.median((simP / (xP ** 3 * L.b12_Pe_over_P200(xP, 10 ** lm / h, 0.0)))[xP > 0.5]))
        # --- how did Lee22 draw Battaglia12? cyan / B12 thermal-pressure model
        for lab, mass in (("M=bin centre/h [Msun]", 10 ** lm / h), ("M=bin centre taken as Msun", 10 ** lm)):
            b12th = xP ** 3 * L.b12_Pth_over_P200(xP, mass, 0.0)
            r = cyanP / b12th
            d[f"fig2_cyan_over_B12_Pth ({lab})"] = {"median": float(np.median(r)),
                                                    "min": float(r.min()), "max": float(r.max())}
        # --- density figure (fig_4.pdf = paper Figure 5): orange = Table 8
        xn, orangen = cur[("fig_4.pdf", panel, "without_c_bpl")]
        _, simn = cur[("fig_4.pdf", panel, "simulation")]
        _, cyann = cur[("fig_4.pdf", panel, "battaglia")]
        modn = xn ** 3 * L.lee_nbar(xn, mh, 0.0)
        d["table8_vs_fig5_orange_mean_abs_frac_centre"] = mean_abs_frac(modn, orangen)
        d["table8_vs_fig5_orange_median_ratio_centre"] = float(np.median(modn / orangen))
        d["fig5_orange_vs_red_simulation_median_ratio"] = float(np.median(orangen / simn))
        # how did Lee22 draw Battaglia16? cyan / [x^3 rho0 g(x) / 200]
        for lab, mass in (("M=bin centre/h [Msun]", 10 ** lm / h), ("M=bin centre taken as Msun", 10 ** lm)):
            base = xn ** 3 * L.b16_rho_over_fb_rhocrit(xn, mass, 0.0) / 200.0
            r = cyann / base
            d[f"fig5_cyan_over_[x3*rho0*g/200] ({lab})"] = {"median": float(np.median(r)),
                                                           "min": float(r.min()), "max": float(r.max())}
            # the physically normalised XGPaint B16 in the same units
            phys = xn ** 3 * L.b16_nbar(xn, mass, 0.0)
            d[f"fig5_cyan_over_B16_XGPaint_physical ({lab})"] = float(np.median(cyann / phys))
            d[f"fig5_red_simulation_over_B16_XGPaint_physical ({lab}) at x>0.5"] = float(
                np.median((simn / phys)[xn > 0.5]))
        d["fig5_red_simulation_over_cyan_B16_at_x>0.5_median"] = float(np.median((simn / cyann)[xn > 0.5]))
        d["fig5_red_simulation_over_cyan_B16_at_x<0.2_median"] = float(np.median((simn / cyann)[xn < 0.2]))
        # --- measured TNG T_e/T200 straight from the red simulation curves (same bins, z=0, low-c quartile)
        xs = np.geomspace(max(xP[0], xn[0]), min(xP[-1], xn[-1]), 60)
        Ps = np.exp(np.interp(np.log(xs), np.log(xP), np.log(simP)))
        ns = np.exp(np.interp(np.log(xs), np.log(xn), np.log(simn)))
        tt = Ps / ns * L.X_H / L.MU
        kT200 = C.kT200_keV(10 ** lm / h, 0.0)
        d["measured_sim_Te_over_T200"] = {f"x={xv:.2f}": float(np.interp(np.log(xv), np.log(xs), tt))
                                          for xv in (0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.3)}
        # number-weighted mean inside 0.05..1 R200 (log-spaced trapezoid in x^3 * dlnx)
        sel = xs <= 1.0
        d["measured_sim_Tnw_over_T200_x0.05_1"] = float(np.trapz(Ps[sel], np.log(xs[sel])) /
                                                        np.trapz(ns[sel], np.log(xs[sel])) * L.X_H / L.MU)
        d["kT200_keV_bin_centre"] = kT200
        d["measured_sim_kT_keV_at_x=0.5"] = float(np.interp(np.log(0.5), np.log(xs), tt)) * kT200
        # Battaglia benchmark T from the cyan curves the paper itself drew
        Pc = np.exp(np.interp(np.log(xs), np.log(xP), np.log(cyanP)))
        nc = np.exp(np.interp(np.log(xs), np.log(xn), np.log(cyann)))
        d["paper_cyan_B12_over_cyan_B16_Te_over_T200_at_x=0.5"] = float(
            np.interp(np.log(0.5), np.log(xs), Pc / nc * L.X_H / L.MU))
        res[f"panel{panel}_log10Mh_{lo}-{hi}"] = d
    # candidate conversion constants for the B16 curve (see report)
    muE = L.MU_E
    res["candidate_constants_for_fig5_cyan_over_[x3*rho0*g/200]"] = {
        "physical, with f_b, n_e = rho/(mu_e m_p)": L.X_H / muE,
        "physical, with f_b, XGPaint 0.9 rho/m_per_e": 0.9 * L.X_H * L.M_P / L.M_PER_E_XGP,
        "physical, with f_b, n_e = X_H rho/m_p": L.X_H ** 2,
        "B16 as printed (no f_b), n_e = rho/(mu_e m_p)": L.X_H / (muE * fb),
        "B16 as printed (no f_b), XGPaint 0.9 rho/m_per_e": 0.9 * L.X_H * L.M_P / L.M_PER_E_XGP / fb,
        "B16 as printed (no f_b), n_e = X_H rho/m_p": L.X_H ** 2 / fb,
        "B16 as printed (no f_b), n_e = rho/(X_H m_p) [i.e. nbar = rho/(200 f_b rho_c)]": 1 / fb,
    }
    res["candidate_constants_for_fig2_cyan_over_B12_Pth"] = {"electron pressure 0.5176 P_th": 0.5176,
                                                             "thermal pressure plotted as P_e": 1.0}
    return res


# ================================================================== B. implied temperature from the fits
def temperature_analysis():
    logmh = np.round(np.arange(13.0, 14.81, 0.1), 2)
    xs = np.array([0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.3])
    rows = []
    fits = {}
    for fam in ("noc", "c"):
        for lm in logmh:
            mh = 10 ** lm
            m = mh / h
            tt_x = {x: [] for x in xs}
            tnw = []
            for z in ZSNAP:
                P = lambda x: L.lee_Pbar(x, mh, z, fam)
                n = lambda x: L.lee_nbar(x, mh, z, fam)
                for x in xs:
                    t = P(x) / n(x) * L.X_H / L.MU
                    tt_x[x].append(t)
                t_nw = q(lambda x: P(x) * x * x, 0, 1) / q(lambda x: n(x) * x * x, 0, 1) * L.X_H / L.MU
                tnw.append(t_nw)
                kT = C.kT200_keV(m, z)
                rows.append([fam, lm, z, kT, t_nw, t_nw * kT] + [tt_x[x][-1] for x in xs])
            fits[f"{fam}|{lm}"] = {"Tnw<R200": dict(zip(("A", "k_T", "maxres_dex"), powerlaw(ZSNAP, tnw))),
                                   **{f"x={x}": dict(zip(("A", "k_T", "maxres_dex"), powerlaw(ZSNAP, tt_x[x])))
                                      for x in xs}}
    # Battaglia benchmark: B12 electron pressure / B16 fully ionised electron density (no 0.9)
    bench = {}
    brows = []
    for lm in logmh:
        m = 10 ** lm / h
        tnw = []
        tx = {x: [] for x in xs}
        for z in ZSNAP:
            P = lambda x: L.b12_Pe_over_P200(x, m, z)
            n = lambda x: L.b16_rho_over_fb_rhocrit(x, m, z) * L.X_H / (200 * L.MU_E)  # rho/(mu_e m_p) / n200
            tnw.append(q(lambda x: P(x) * x * x, 0, 1) / q(lambda x: n(x) * x * x, 0, 1) * L.X_H / L.MU)
            for x in xs:
                tx[x].append(P(x) / n(x) * L.X_H / L.MU)
            brows.append([lm, z, tnw[-1]] + [tx[x][-1] for x in xs])
        bench[str(lm)] = {"Tnw<R200": dict(zip(("A", "k_T", "maxres_dex"), powerlaw(ZSNAP, tnw))),
                          **{f"x={x}": dict(zip(("A", "k_T", "maxres_dex"), powerlaw(ZSNAP, tx[x]))) for x in xs}}
    with (OUT / "temperature_grid.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fit_family(noc=Tables7+8,c=Tables1+3 with TNG-mean c)", "log10_M200c_hinv", "z",
                    "kT200_keV", "Tnw_over_T200_within_R200", "kTnw_keV"] + [f"Te_over_T200_x={x}" for x in xs])
        w.writerows(rows)
    with (OUT / "temperature_grid_battaglia_benchmark.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["log10_M200c_hinv", "z", "Tnw_over_T200_within_R200"] + [f"Te_over_T200_x={x}" for x in xs])
        w.writerows(brows)
    # compact summary
    summ = {}
    for fam in ("noc", "c"):
        for lm in (13.0, 13.3, 13.6, 13.9, 14.2, 14.5, 14.8):
            key = f"{fam}|{lm}"
            fz = fits[key]
            z0 = [r for r in rows if r[0] == fam and r[1] == lm and r[2] == 0.0][0]
            z1 = [r for r in rows if r[0] == fam and r[1] == lm and r[2] == 1.0][0]
            z2 = [r for r in rows if r[0] == fam and r[1] == lm and r[2] == 2.0][0]
            summ[key] = {"kT200_keV_z0": z0[3], "Tnw/T200_z0": z0[4], "Tnw/T200_z1": z1[4], "Tnw/T200_z2": z2[4],
                         "kTnw_keV_z0": z0[5], "Te/T200_x0.5_z0": z0[6 + 4], "k_T_Tnw": fz["Tnw<R200"]["k_T"],
                         "k_T_x0.1": fz["x=0.1"]["k_T"], "k_T_x0.3": fz["x=0.3"]["k_T"],
                         "k_T_x0.5": fz["x=0.5"]["k_T"], "k_T_x1.0": fz["x=1.0"]["k_T"],
                         "maxres_dex_Tnw": fz["Tnw<R200"]["maxres_dex"]}
    for lm in (13.0, 13.3, 13.6, 13.9, 14.2, 14.5, 14.8):
        b = bench[str(lm)]
        z0 = [r for r in brows if r[0] == lm and r[1] == 0.0][0]
        summ[f"battaglia|{lm}"] = {"Tnw/T200_z0": z0[2], "Te/T200_x0.5_z0": z0[3 + 4],
                                   "k_T_Tnw": b["Tnw<R200"]["k_T"], "k_T_x0.5": b["x=0.5"]["k_T"],
                                   "k_T_x1.0": b["x=1.0"]["k_T"]}
    kts = [fits[f"noc|{lm}"]["Tnw<R200"]["k_T"] for lm in logmh]
    kts5 = [fits[f"noc|{lm}"]["x=0.5"]["k_T"] for lm in logmh]
    ktc = [fits[f"c|{lm}"]["Tnw<R200"]["k_T"] for lm in logmh]
    ktb = [bench[str(lm)]["Tnw<R200"]["k_T"] for lm in logmh]
    t0 = [r[4] for r in rows if r[0] == "noc" and r[2] == 0.0]
    t0c = [r[4] for r in rows if r[0] == "c" and r[2] == 0.0]
    t0b = [r[2] for r in brows if r[1] == 0.0]
    summ["range_over_mass_13.0-14.8"] = {
        "noc k_T(Tnw)": [min(kts), max(kts)], "noc k_T(x=0.5)": [min(kts5), max(kts5)],
        "c k_T(Tnw)": [min(ktc), max(ktc)], "battaglia k_T(Tnw)": [min(ktb), max(ktb)],
        "noc Tnw/T200 z=0": [min(t0), max(t0)], "c Tnw/T200 z=0": [min(t0c), max(t0c)],
        "battaglia Tnw/T200 z=0": [min(t0b), max(t0b)],
    }
    return summ


# ================================================================== C. integrated Y and N_e vs B12/B16
def integrated_ratios():
    masses = (10 ** 13.5, 1e14, 10 ** 14.5)
    zs = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0)
    rows, fits = [], {}
    for m in masses:
        mh = m * h
        rY, rN, rYc, rNc, fgl, fgb, yl, yb = [], [], [], [], [], [], [], []
        for z in zs:
            Y_lee = q(lambda x: L.lee_Pbar(x, mh, z) * x * x, 0, 1)
            Y_leec = q(lambda x: L.lee_Pbar(x, mh, z, "c") * x * x, 0, 1)
            Y_b12 = q(lambda x: L.b12_Pe_over_P200(x, m, z) * x * x, 0, 1)
            N_lee = q(lambda x: L.lee_nbar(x, mh, z) * x * x, 0, 1)
            N_leec = q(lambda x: L.lee_nbar(x, mh, z, "c") * x * x, 0, 1)
            N_b16 = q(lambda x: L.b16_nbar(x, m, z) * x * x, 0, 1)
            # gas mass inside R200c in units of f_b M200 (Lee: fully ionised mu_e; B16: rho_gas directly)
            fg_lee = 3 * N_lee * L.MU_E / L.X_H
            fg_b16 = 3 * q(lambda x: L.b16_rho_over_fb_rhocrit(x, m, z) * x * x, 0, 1) / 200
            rY.append(Y_lee / Y_b12); rN.append(N_lee / N_b16)
            rYc.append(Y_leec / Y_b12); rNc.append(N_leec / N_b16)
            fgl.append(fg_lee); fgb.append(fg_b16); yl.append(3 * Y_lee); yb.append(3 * Y_b12)
            rows.append([np.log10(m), z, Y_lee / Y_b12, Y_leec / Y_b12, N_lee / N_b16, N_leec / N_b16,
                         fg_lee, fg_b16, 3 * Y_lee, 3 * Y_b12])
        key = f"log10M={np.log10(m):.1f}"
        fits[key] = {
            "Y_lee_noc/Y_B12": dict(zip(("A", "k", "maxres_dex"), powerlaw(zs, rY))),
            "Y_lee_c/Y_B12": dict(zip(("A", "k", "maxres_dex"), powerlaw(zs, rYc))),
            "Ne_lee_noc/Ne_B16": dict(zip(("A", "k", "maxres_dex"), powerlaw(zs, rN))),
            "Ne_lee_c/Ne_B16": dict(zip(("A", "k", "maxres_dex"), powerlaw(zs, rNc))),
            "fgas_lee_noc(<R200c)/f_b": dict(zip(("A", "k", "maxres_dex"), powerlaw(zs, fgl))),
            "fgas_B16(<R200c)/f_b": dict(zip(("A", "k", "maxres_dex"), powerlaw(zs, fgb))),
            "Y/(P200 V200)_lee_noc": dict(zip(("A", "k", "maxres_dex"), powerlaw(zs, yl))),
            "Y/(P200 V200)_B12e": dict(zip(("A", "k", "maxres_dex"), powerlaw(zs, yb))),
            "values": {f"z={z}": {"Y_ratio_noc": a, "Y_ratio_c": b, "Ne_ratio_noc": c_, "Ne_ratio_c": d_,
                                  "fgas_lee_noc/f_b": e, "fgas_B16/f_b": g}
                       for z, a, b, c_, d_, e, g in zip(zs, rY, rYc, rN, rNc, fgl, fgb)},
        }
    with (OUT / "integrated_ratios_vs_battaglia.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["log10_M200c_msun", "z", "Y(<R200c)_Lee_noc/B12e", "Y_Lee_c/B12e", "Ne(<R200c)_Lee_noc/B16",
                    "Ne_Lee_c/B16", "Mgas(<R200c)/(f_b M200)_Lee_noc", "Mgas/(f_b M200)_B16",
                    "Y/(P200 V200)_Lee_noc", "Y/(P200 V200)_B12e"])
        w.writerows(rows)
    # reference exponents of pure bookkeeping factors over the same z grid
    zz = np.array(zs)
    fits["reference_exponents"] = {
        "a^3=(1+z)^-3": -3.0, "a^2": -2.0,
        "E^2(z) a^3 (comoving volume + z=0 critical density)": powerlaw(zz, C.E2(zz) / (1 + zz) ** 3)[1],
        "E^2(z) a^2": powerlaw(zz, C.E2(zz) / (1 + zz) ** 2)[1],
    }
    return fits


# ================================================================== D. observational anchors
# X-COP stacked median density, Ghirardini+2019 Table (arXiv:1805.00042 TeX, 'Stacked thermodynamic profiles')
XCOP_X = np.array([1.058e-02, 1.179e-02, 1.313e-02, 1.463e-02, 1.630e-02, 1.816e-02, 2.024e-02, 2.255e-02, 2.513e-02,
                   2.799e-02, 3.119e-02, 3.475e-02, 3.872e-02, 4.314e-02, 4.807e-02, 5.356e-02, 5.967e-02, 6.649e-02,
                   7.408e-02, 8.254e-02, 9.197e-02, 1.025e-01, 1.142e-01, 1.272e-01, 1.417e-01, 1.579e-01, 1.759e-01,
                   1.960e-01, 2.184e-01, 2.434e-01, 2.712e-01, 3.021e-01, 3.366e-01, 3.751e-01, 4.179e-01, 4.656e-01,
                   5.188e-01, 5.780e-01, 6.440e-01, 7.176e-01, 7.995e-01, 8.908e-01, 9.925e-01, 1.106e+00, 1.232e+00,
                   1.373e+00, 1.530e+00, 1.704e+00, 1.899e+00])
XCOP_NE = np.array([8.814e-03, 8.751e-03, 8.684e-03, 8.712e-03, 8.409e-03, 8.110e-03, 8.007e-03, 7.720e-03, 7.627e-03,
                    7.593e-03, 7.513e-03, 7.334e-03, 7.086e-03, 6.889e-03, 6.675e-03, 6.408e-03, 6.161e-03, 5.628e-03,
                    5.255e-03, 4.875e-03, 4.473e-03, 4.141e-03, 3.813e-03, 3.429e-03, 3.011e-03, 2.678e-03, 2.385e-03,
                    2.142e-03, 1.892e-03, 1.675e-03, 1.444e-03, 1.193e-03, 1.013e-03, 8.583e-04, 7.145e-04, 5.997e-04,
                    4.926e-04, 4.043e-04, 3.391e-04, 2.742e-04, 2.199e-04, 1.783e-04, 1.423e-04, 1.142e-04, 8.635e-05,
                    6.274e-05, 4.767e-05, 3.720e-05, 2.769e-05])   # n_e E(z)^-2 [cm^-3], median of 12


def observational():
    res = {}
    rows = []
    cases = (("M200c=1e14 Msun", 1e14), ("M200c=1e15 h^-1 Msun", 1e15 / h))
    for label, m200 in cases:
        mh = m200 * h
        for cname, cfun in (("TNG-mean proxy", lambda m, z: L.tng_mean_c(m * h, z)),
                            ("Duffy08", lambda m, z: L.duffy08_c200c(m, z, h))):
            for z in (0.0, 0.1, 1.0):
                c200 = cfun(m200, z)
                m5, s5 = L.nfw_m_ratio(c200, 500.0)
                m500 = m5 * m200
                r200 = C.r_delta(m200, z)
                r500 = s5 * r200
                ez2 = C.E2(z)
                n200 = C.n200(z)
                P200 = C.P200(m200, z) / L.KEV   # keV cm^-3
                d = {"c200": c200, "M500/M200": m5, "R500/R200": s5, "M500_msun": m500,
                     "R500_kpc": r500 / L.KPC, "n200_cm-3": n200, "P200_keV_cm-3": P200,
                     "kT200_keV": C.kT200_keV(m200, z)}
                for xr in (0.15, 0.3, 0.5, 1.0):
                    x200 = xr * s5
                    ne_lee = L.lee_nbar(x200, mh, z) * n200
                    Pe_lee = L.lee_Pbar(x200, mh, z) * P200
                    ne_leec = L.lee_nbar(x200, mh, z, "c") * n200
                    Pe_leec = L.lee_Pbar(x200, mh, z, "c") * P200
                    Pe_upp = L.arnaud10_upp_P(xr, m500, z, C)
                    ne_xcop = np.exp(np.interp(np.log(xr), np.log(XCOP_X), np.log(XCOP_NE))) * ez2
                    ne_b16 = L.b16_nbar(x200, m200, z) * n200
                    Pe_b12 = L.b12_Pe_over_P200(x200, m200, z) * P200
                    d[f"r={xr}R500"] = {"ne_Lee_literal_noc": ne_lee, "ne_Lee_literal_c": ne_leec,
                                        "ne_XCOP_median_Ez2": ne_xcop, "ne_B16_XGPaint": ne_b16,
                                        "Pe_Lee_noc": Pe_lee, "Pe_Lee_c": Pe_leec, "Pe_Arnaud10_UPP": Pe_upp,
                                        "Pe_B12": Pe_b12,
                                        "ne_Lee/XCOP": ne_lee / ne_xcop, "ne_Lee/B16": ne_lee / ne_b16,
                                        "Pe_Lee/UPP": Pe_lee / Pe_upp, "Pe_Lee/B12": Pe_lee / Pe_b12,
                                        "Pe_B12/UPP": Pe_b12 / Pe_upp, "ne_B16/XCOP": ne_b16 / ne_xcop,
                                        "kTe_Lee_keV": Pe_lee / ne_lee,
                                        "kTe_UPP/XCOP_keV": Pe_upp / ne_xcop}
                    rows.append([label, cname, z, xr, ne_lee, ne_leec, ne_xcop, ne_b16, Pe_lee, Pe_leec, Pe_upp,
                                 Pe_b12, Pe_lee / ne_lee, Pe_upp / ne_xcop])
                # integrated quantities inside R500
                vol = lambda f: 4 * np.pi * r200 ** 3 * q(lambda x: f(x) * x * x, 0, s5)
                Ne_lee = vol(lambda x: L.lee_nbar(x, mh, z)) * n200
                Ne_b16 = vol(lambda x: L.b16_nbar(x, m200, z)) * n200
                Yint_lee = vol(lambda x: L.lee_Pbar(x, mh, z)) * P200
                Yint_b12 = vol(lambda x: L.b12_Pe_over_P200(x, m200, z)) * P200
                Yint_upp = 4 * np.pi * r500 ** 3 * q(lambda s: L.arnaud10_upp_P(s, m500, z, C) * s * s, 1e-6, 1)
                Ne_xcop = 4 * np.pi * r500 ** 3 * ez2 * q(
                    lambda s: np.exp(np.interp(np.log(s), np.log(XCOP_X), np.log(XCOP_NE))) * s * s, 1e-6, 1)
                mgas = lambda ne: ne * L.MU_E * L.M_P / L.MSUN
                d["fgas500_Lee_literal_noc"] = mgas(Ne_lee) / m500
                d["fgas500_B16_fully_ionised"] = mgas(Ne_b16 / 0.9 * L.M_PER_E_XGP / (L.MU_E * L.M_P)) / m500
                d["fgas500_XCOP_median_profile"] = mgas(Ne_xcop) / m500
                h73 = h / 0.73
                d["fgas500_Sun09_tier1+2+clusters"] = 0.0616 * h73 ** -1.5 * (m500 * h73 / 1e13) ** 0.135
                d["fgas500_Sun09_tier1+clusters"] = 0.0724 * h73 ** -1.5 * (m500 * h73 / 1e13) ** 0.093
                d["Ysph500_Lee/UPP"] = Yint_lee / Yint_upp
                d["Ysph500_B12/UPP"] = Yint_b12 / Yint_upp
                d["Ysph500_Lee/B12"] = Yint_lee / Yint_b12
                d["Ne500_Lee/XCOP"] = Ne_lee / Ne_xcop
                d["Ne500_B16/XCOP"] = Ne_b16 / Ne_xcop
                d["Ne500_Lee/B16"] = Ne_lee / Ne_b16
                # temperature anchor: Sun+2009 E(z) M500 = 1.26e14 h73^-1 (T500/3 keV)^1.65 (Tier1+2+clusters)
                d["kT500_spec_Sun09_keV"] = 3.0 * (np.sqrt(ez2) * m500 * h73 / 1.26e14) ** (1 / 1.65)
                # Lee22 number-weighted kT in 0.15-1 R500 (Sun09 aperture)
                d["kT_nw_0.15-1R500_Lee_keV"] = (q(lambda x: L.lee_Pbar(x, mh, z) * x * x, 0.15 * s5, s5) * P200 /
                                                 (q(lambda x: L.lee_nbar(x, mh, z) * x * x, 0.15 * s5, s5) * n200))
                d["kT_nw_0.15-1R500_Lee_c_keV"] = (
                    q(lambda x: L.lee_Pbar(x, mh, z, "c") * x * x, 0.15 * s5, s5) * P200 /
                    (q(lambda x: L.lee_nbar(x, mh, z, "c") * x * x, 0.15 * s5, s5) * n200))
                d["kT_nw_0.15-1R500_UPP/XCOP_keV"] = (
                    q(lambda s: L.arnaud10_upp_P(s, m500, z, C) * s * s, 0.15, 1) /
                    (ez2 * q(lambda s: np.exp(np.interp(np.log(s), np.log(XCOP_X), np.log(XCOP_NE))) * s * s, 0.15, 1)))
                res[f"{label}|c={cname}|z={z}"] = d
    return res


def main():
    RESULTS["A_figure_forensics"] = figure_forensics()
    RESULTS["B_temperature"] = temperature_analysis()
    RESULTS["C_integrated_vs_battaglia"] = integrated_ratios()
    RESULTS["D_observations"] = observational()
    RESULTS["meta"] = {"cosmology": "TNG300 Planck15 Om=0.3089 Ob=0.0486 h=0.6774", "X_H": L.X_H, "mu": L.MU,
                       "mu_e": L.MU_E, "Lee22 masses": "M200c h^-1 Msun, Mcut pivot, shape clip 10^14.8 h^-1",
                       "B12/B16 masses": "physical Msun, pivot 1e14 Msun"}
    (OUT / "consistency_results.json").write_text(json.dumps(RESULTS, indent=1, default=float))
    print(json.dumps(RESULTS["A_figure_forensics"], indent=1, default=float))


if __name__ == "__main__":
    main()
