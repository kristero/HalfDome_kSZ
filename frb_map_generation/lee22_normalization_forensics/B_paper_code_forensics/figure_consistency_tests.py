#!/usr/bin/env python3
"""Consistency tests of Lee22's own z=0 figure data (arXiv v1 Figs. 2 and 5, vector curves).

T1  What normalisation did Lee22 use for the cyan Battaglia curves?  Fit a constant K such that
    figure_curve = K * reference, for B12 (P_th/P200) and B16 (rho_gas/(200 f_b rho_cr)).
T2  Enclosed gas mass implied by the red simulation curve of Fig. 5 with the printed n200 (eq. 9):
    M_gas(<R)/(f_b M200) = (mu_e / X_H) * 3 int_0^R (x^3 n_e/n200) dln x, mu_e = 2/(1+X_H) (full ionisation).
    With x_e<=1.158 this is also a lower bound on the gas mass for any ionisation state.
T3  Thermal content implied by the red curve of Fig. 2:
    (M_gas/(f_b M200)) * (T_mw/T200) = (1/0.5176) * 3 int (x^3 P_e/P200) dln x,  kT200 = mu m_p G M200/(2 R200).
T4  Temperature from the ratio of the two red curves (volume cancels):
    T_e(x)/T200 = (X_H/mu) * (P_e/P200)/(n_e/n200), mu = 4/(3+5 X_H).
Everything is z = 0, where comoving and physical coordinates coincide (a=1).
"""
from pathlib import Path
import json
import numpy as np
from scipy.integrate import quad

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "outputs/lee22_normalization_forensics_20260928/B_paper_code_forensics"
XH = 0.76
MU_E = 2.0 / (1.0 + XH)
MU = 4.0 / (3.0 + 5.0 * XH)
PE_OVER_PTH = 0.5176          # (1+X)/2 / [(1+X)/2 + (X+Y/4)] for full ionisation; XGPaint's constant
H_TNG = 0.6774
OMB, OMM = 0.0486, 0.3089
FB = OMB / OMM

data = np.genfromtxt(OUT / "lee22_fig2_fig5_vector_curves.csv", delimiter=",", names=True,
                     dtype=None, encoding="utf-8")


def curve(fig, panel, name):
    m = (data["figure"] == fig) & (data["panel"] == panel) & (data["curve"] == name)
    return data["x_r_over_R200"][m], data["x3_times_ratio"][m]


def b12_pth_over_p200(x, m_msun, z=0.0):
    m = m_msun / 1e14
    p0 = 18.1 * m ** 0.154 * (1 + z) ** -0.758
    xc = 0.497 * m ** -0.00865 * (1 + z) ** 0.731
    beta = 4.35 * m ** 0.0393 * (1 + z) ** 0.415
    return p0 * (x / xc) ** -0.3 * (1 + x / xc) ** -beta


def b16_rho_over_fb_rhoc(x, m_msun, z=0.0):
    m = m_msun / 1e14
    rho0 = 4e3 * m ** 0.29 * (1 + z) ** -0.66
    al = 0.88 * m ** -0.03 * (1 + z) ** 0.19
    be = 3.83 * m ** 0.04 * (1 + z) ** -0.025
    ga, xc = -0.2, 0.5
    return rho0 * (x / xc) ** ga * (1 + (x / xc) ** al) ** (-(be + ga) / al)


def fit_constant(q_fig, q_ref):
    lr = np.log(q_fig / q_ref)
    return float(np.exp(lr.mean())), float(lr.std()), float(np.exp(lr.min())), float(np.exp(lr.max()))


def enclosed(x, q, xmax=1.0):
    """3 int q dln x from the first vertex to xmax (log-linear interpolation), plus a power-law
    extrapolation to x=0 using the innermost two vertices."""
    lx, lq = np.log(x), np.log(q)
    f = lambda t: np.exp(np.interp(t, lx, lq))
    main = 3 * quad(f, lx[0], np.log(xmax), limit=400)[0]
    s = (lq[1] - lq[0]) / (lx[1] - lx[0])          # local slope of x^3 n
    inner = 3 * q[0] / s if s > 0 else np.nan        # int_{-inf}^{lx0} q0 e^{s(t-lx0)} dt
    return main, inner


def main():
    res = {"constants": dict(X_H=XH, mu_e=MU_E, mu=MU, Pe_over_Pth=PE_OVER_PTH, h_TNG=H_TNG, f_b=FB)}
    bins = {0: (13.2, 13.4), 1: (13.8, 14.0)}
    print("=== T1: normalisation of the cyan reference curves (figure / formula) ===")
    t1 = {}
    for panel, (lo, hi) in bins.items():
        lc = 0.5 * (lo + hi)
        for mass_label, m in (("M=10^lc/h Msun (h=0.6774)", 10 ** lc / H_TNG), ("M=10^lc Msun (no h)", 10 ** lc)):
            x, q = curve("fig2_pressure", panel, "battaglia")
            k = fit_constant(q, x ** 3 * b12_pth_over_p200(x, m))
            x5, q5 = curve("fig5_density", panel, "battaglia")
            k5 = fit_constant(q5, x5 ** 3 * b16_rho_over_fb_rhoc(x5, m) / 200.0)
            print(f"panel {panel} [{lo}-{hi}] {mass_label}:")
            print(f"   B12: Fig2 cyan / (P_th/P200)_B12       K={k[0]:.4f}  sd(lnK)={k[1]:.2e}  range {k[2]:.4f}-{k[3]:.4f}")
            print(f"   B16: Fig5 cyan / [rho/(200 f_b rho_c)]  K={k5[0]:.4f}  sd(lnK)={k5[1]:.2e}  range {k5[2]:.4f}-{k5[3]:.4f}")
            t1[f"panel{panel}_{'h' if 'h=' in mass_label else 'noh'}"] = dict(B12_K=k, B16_K=k5)
    res["T1_reference_curve_normalisation"] = t1
    print("   (1 = plotted as is; 0.669 = X_H(1+X_H)/2 electron conversion to printed n200; 6.36 = 1/f_b; 0.5176 = P_e/P_th)")

    print("\n=== T2/T3/T4: enclosed content implied by the red simulation curves (z=0, low-c quartile) ===")
    t2 = {}
    for panel, (lo, hi) in bins.items():
        xn, qn = curve("fig5_density", panel, "simulation")
        xp, qp = curve("fig2_pressure", panel, "simulation")
        xb, qb = curve("fig5_density", panel, "battaglia")
        rows = {}
        for R in (0.5, 1.0, 1.34):
            n_main, n_in = enclosed(xn, qn, R)
            p_main, p_in = enclosed(xp, qp, R)
            b_main, b_in = enclosed(xb, qb, R)
            nbar = (n_main + n_in) / R ** 3
            pbar = (p_main + p_in) / R ** 3
            fgas = MU_E / XH * (n_main + n_in)                  # M_gas(<R)/(f_b M200)
            thermal = (p_main + p_in) / PE_OVER_PTH             # (M_gas/(f_b M200)) (T_mw/T200)
            fgas_b16 = MU_E / XH * (b_main + b_in)
            rows[str(R)] = dict(mean_ne_over_n200=nbar, mean_Pe_over_P200=pbar,
                                Mgas_over_fbM200=fgas, Mgas_over_fbM200_lower_bound_any_ionisation=fgas,
                                thermal_Mgas_T_over_fbM200_T200=thermal, Tmw_over_T200=thermal / fgas,
                                Mgas_over_fbM200_from_cyan_B16_curve=fgas_b16,
                                inner_extrapolation_fraction_n=n_in / (n_main + n_in))
            print(f"panel {panel} [{lo}-{hi}] R={R}: <n_e/n200>={nbar:.3f}  M_gas/(f_b M200)={fgas:.3f}  "
                  f"<P_e/P200>={pbar:.4f}  f_gas*T/T200={thermal:.3f}  => T_mw/T200={thermal/fgas:.3f}  "
                  f"[cyan B16 curve would give M_gas/(f_b M200)={fgas_b16:.3f}]")
        # local temperature profile (volume-independent)
        xg = np.geomspace(0.05, 1.3, 12)
        tn = np.exp(np.interp(np.log(xg), np.log(xn), np.log(qn)))
        tp = np.exp(np.interp(np.log(xg), np.log(xp), np.log(qp)))
        TT = XH / MU * tp / tn
        rows["T_over_T200_profile"] = dict(x=xg.tolist(), T_over_T200=TT.tolist())
        print("   local T_e/T200 (from P/n ratio of red curves):", " ".join(f"{a:.2f}:{b:.2f}" for a, b in zip(xg, TT)))
        t2[f"panel{panel}"] = rows
    res["T2_T4_simulation_curve_integrals"] = t2
    with open(OUT / "figure_consistency_tests.json", "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
