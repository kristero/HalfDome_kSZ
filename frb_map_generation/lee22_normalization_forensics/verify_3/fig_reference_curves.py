"""verify_3: what did Lee22 draw as the Battaglia reference curves (cyan) in Fig. 2 (B12 pressure)
and Fig. 5 (B16 density)? Uses the vector-extracted curves (two independent extractions by
investigators B and C agree to ~1e-5) and B16/B12 formulas taken from the papers' TeX:
  B16 (arXiv:1607.02442, eq. A.1, Table 2 AGN): rho_gas = rho_crit(z) rho_fit,
      rho_fit = rho0 (x/xc)^gamma [1+(x/xc)^alpha]^(-(beta-gamma)/alpha), xc=0.5, gamma=-0.2
      rho0 = 4e3 M14^0.29 (1+z)^-0.66, alpha = 0.88 M14^-0.03 (1+z)^0.19, beta = 3.83 M14^0.04 (1+z)^-0.025
  B12 thermal pressure: P0=18.1 M14^0.154, xc=0.497 M14^-0.00865, beta=4.35 M14^0.0393 (z=0), alpha=1, gamma=-0.3
Fig. 2/5 panels: 10^13.2-13.4 and 10^13.8-14.0 h^-1 Msun, z = 0, lowest concentration quartile.
"""
import numpy as np, json, os, csv
import lee22_c4_lib as L

CSV = "/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_normalization_forensics_20260928/B_paper_code_forensics/lee22_fig2_fig5_vector_curves.csv"
OUT = "/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_normalization_forensics_20260928/verify_3"
data = {}
with open(CSV) as f:
    for r in csv.DictReader(f):
        data.setdefault((r["figure"], int(r["panel"]), r["curve"]), []).append((float(r["x_r_over_R200"]), float(r["x3_times_ratio"])))
data = {k: np.array(sorted(v)) for k, v in data.items()}


def b16_fit(x, m, z=0.0, exponent="printed"):
    m14 = m / 1e14
    rho0 = 4e3 * m14 ** 0.29 * (1 + z) ** -0.66
    al = 0.88 * m14 ** -0.03 * (1 + z) ** 0.19
    be = 3.83 * m14 ** 0.04 * (1 + z) ** -0.025
    ga, xc = -0.2, 0.5
    ex = (be - ga) / al if exponent == "printed" else (be + ga) / al
    u = x / xc
    return rho0 * u ** ga * (1 + u ** al) ** (-ex)


def b12_pth(x, m, z=0.0):
    m14 = m / 1e14
    return L.gnfw(x, 18.1 * m14 ** 0.154, 0.497 * m14 ** -0.00865, 4.35 * m14 ** 0.0393)


res = {}
panels = {0: 10 ** 13.3, 1: 10 ** 13.9}
for p, mbin in panels.items():
    d = data[("fig5_density", p, "battaglia")]
    x, y = d[:, 0], d[:, 1]
    for mlab, m in [("M=bin centre as Msun", mbin), ("M=bin centre/h", mbin / L.H)]:
        for ex in ["printed", "xgpaint"]:
            ref = x ** 3 * b16_fit(x, m, exponent=ex) / 200.0     # = x^3 rho_gas/(200 rho_c) (no f_b)
            K = y / ref
            res[f"fig5_p{p}|{mlab}|B16 exponent {ex}"] = dict(
                median_K=float(np.median(K)), min_K=float(K.min()), max_K=float(K.max()),
                rms_log_scatter=float(np.std(np.log(K))))
    d = data[("fig2_pressure", p, "battaglia")]
    x, y = d[:, 0], d[:, 1]
    for mlab, m in [("M=bin centre as Msun", mbin), ("M=bin centre/h", mbin / L.H)]:
        K = y / (x ** 3 * b12_pth(x, m))
        res[f"fig2_p{p}|{mlab}|B12 P_th"] = dict(median_K=float(np.median(K)), min_K=float(K.min()),
                                                 max_K=float(K.max()), rms_log_scatter=float(np.std(np.log(K))))
    # simulation curves: T/T200 from Lee22's own TNG stacks (Fig.2 red / Fig.5 red)
    dp = data[("fig2_pressure", p, "simulation")]
    dn = data[("fig5_density", p, "simulation")]
    xs = np.exp(np.linspace(np.log(max(dp[0, 0], dn[0, 0])), np.log(min(dp[-1, 0], 1.0)), 400))
    pp = np.interp(np.log(xs), np.log(dp[:, 0]), dp[:, 1]) / xs ** 3
    nn = np.interp(np.log(xs), np.log(dn[:, 0]), dn[:, 1]) / xs ** 3
    lx = np.log(xs)
    from scipy.integrate import simpson
    Tnw = (L.X_H / L.MU) * simpson(pp * xs ** 3, x=lx) / simpson(nn * xs ** 3, x=lx)
    res[f"sim_p{p}_Tnw_over_T200_{xs[0]:.3f}-{xs[-1]:.2f}"] = float(Tnw)
    res[f"sim_p{p}_Mgas_over_fbM200_{xs[0]:.3f}-1"] = float(simpson(nn * 3 * xs ** 3, x=lx) * L.MU_E / L.X_H)

cands = {"(iii) n_e=rho/(X_H m_p), no f_b: 1/f_b": 1 / L.FB,
         "(i) n_e=rho/(mu_e m_p), no f_b": L.X_H / (L.MU_E * L.FB),
         "(ii) n_e=X_H rho/m_p, no f_b": L.X_H ** 2 / L.FB,
         "with f_b, n_e=rho/(X_H m_p)": 1.0,
         "with f_b, n_e = rho/(mu_e m_p)": L.X_H / L.MU_E}
res["candidate_K_values"] = cands
json.dump(res, open(os.path.join(OUT, "fig_reference_curves.json"), "w"), indent=1)
for k, v in res.items():
    print(k, v)
