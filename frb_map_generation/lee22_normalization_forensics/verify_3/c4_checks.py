"""verify_3: adversarial checks of claim C4 (extra z-independent density factor F_n ~ 3-5.5).

1. z=0: implied electron-weighted T/T200, gas mass inside R200c, local gas/DM ratio at R200c.
2. Hydrostatic test: electron density that the Lee22 PRESSURE fit can support in an NFW
   potential; F_n^HSE = n_Lee22 / n_HSE (assumes the pressure fit is correct, as C4 does).
3. Redshift dependence of F_n^HSE after undoing a^3 (density) and a^2 (pressure).
4. Pressure amplitude at z=0: Y_sph(R500) vs Arnaud+10 UPP; P_e vs B12 electron pressure.
"""
import json, os
import numpy as np
from scipy import integrate, optimize
import lee22_c4_lib as L

OUT = "/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_normalization_forensics_20260928/verify_3"
res = {}

# ------------------------------------------------------------------ 1 + 2 at z = 0
rows = []
for lmh in [13.0, 13.2, 13.4, 13.61, 13.8, 14.0, 14.2, 14.5]:
    mh = 10 ** lmh
    z = 0.0
    n_in = L.vol_int(lambda x: L.nhat(x, mh, z), 1e-4, 1.0)
    p_in = L.vol_int(lambda x: L.phat(x, mh, z), 1e-4, 1.0)
    n_in2 = L.vol_int(lambda x: L.nhat(x, mh, z), 0.05, 1.0)
    p_in2 = L.vol_int(lambda x: L.phat(x, mh, z), 0.05, 1.0)
    t_nw = (L.X_H / L.MU) * p_in / n_in
    t_nw2 = (L.X_H / L.MU) * p_in2 / n_in2
    mgas = n_in * L.MU_E / L.X_H
    row = dict(log10Mh=lmh, nhat_mean_lt_R200=n_in, Mgas_over_fbM200=mgas,
               Tnw_over_T200_0_1=t_nw, Tnw_over_T200_0p05_1=t_nw2,
               kT200_keV=L.kT200_keV(mh / L.H, 0.0),
               kTnw_keV=t_nw * L.kT200_keV(mh / L.H, 0.0))
    for x in [0.1, 0.3, 0.5, 1.0]:
        row[f"T_over_T200_x{x}"] = float((L.X_H / L.MU) * L.phat(x, mh, z) / L.nhat(x, mh, z))
    # local gas/DM ratio at R200 in units of f_b (literal Lee22)
    for cname, c in [("lee22c", L.lee22_klypin_c_z0(mh)), ("duffy", L.duffy08_c200c(mh, 0.0))]:
        rho_nfw_R200_over_200rhoc = c ** 2 / 3.0 / ((1 + c) ** 2 * (np.log1p(c) - c / (1 + c)))
        rho_gas_R200_over_200fbrhoc = L.nhat(1.0, mh, z) * L.MU_E / L.X_H
        row[f"c_{cname}"] = c
        row[f"local_gas_over_fb_total_at_R200_{cname}"] = rho_gas_R200_over_200fbrhoc / rho_nfw_R200_over_200rhoc
        for fnt in [0.0, 0.15]:
            tag = f"{cname}_fnt{fnt}"
            nh = lambda x, c=c, fnt=fnt: L.nhat_hse(x, mh, z, c, fnt=fnt)
            for x in [0.2, 0.5, 1.0]:
                row[f"FnHSE_x{x}_{tag}"] = float(L.nhat(x, mh, z) / nh(np.array([x]))[0])
                # HSE temperature
                if fnt == 0.0:
                    xc, beta = L.pressure_params(mh, z)[1:]
                    row[f"T_HSE_over_T200_x{x}_{cname}"] = float(
                        2 * L.m_nfw(x, c) / (x * (-L.dlng_dlnx(x, xc, beta))) * (1 - fnt))
            row[f"FnHSE_vol_0p2_1_{tag}"] = L.vol_int(lambda x: L.nhat(x, mh, z), 0.2, 1.0) / L.vol_int(nh, 0.2, 1.0)
            row[f"FnHSE_vol_0p1_1_{tag}"] = L.vol_int(lambda x: L.nhat(x, mh, z), 0.1, 1.0) / L.vol_int(nh, 0.1, 1.0)
            row[f"FnHSE_vol_0p5_1_{tag}"] = L.vol_int(lambda x: L.nhat(x, mh, z), 0.5, 1.0) / L.vol_int(nh, 0.5, 1.0)
    # concentration sensitivity (+-30 %)
    for fac in [0.7, 1.3]:
        c = L.lee22_klypin_c_z0(mh) * fac
        nh = lambda x, c=c: L.nhat_hse(x, mh, z, c)
        row[f"FnHSE_vol_0p2_1_lee22c_x{fac}"] = L.vol_int(lambda x: L.nhat(x, mh, z), 0.2, 1.0) / L.vol_int(nh, 0.2, 1.0)
    rows.append(row)
res["z0_table"] = rows

# ------------------------------------------------------------------ 3 redshift dependence
zrows = []
for lmh in [13.0, 13.61, 14.0]:
    mh = 10 ** lmh
    c0 = L.lee22_klypin_c_z0(mh)
    for z in [0.0, 0.5, 1.0, 1.5, 2.0]:
        c = c0 * (1 + z) ** (-0.47)          # Duffy08 redshift scaling applied to the Lee22 z=0 c
        nlee = lambda x: L.nhat(x, mh, z) * (1 + z) ** 3          # a^3 removed
        for pscale_name, ps in [("P_x(1+z)^2", (1 + z) ** 2), ("P_uncorrected", 1.0)]:
            nh = lambda x: L.nhat_hse(x, mh, z, c, pressure_scale=ps)
            F = L.vol_int(nlee, 0.2, 1.0) / L.vol_int(nh, 0.2, 1.0)
            zrows.append(dict(log10Mh=lmh, z=z, c=c, pressure=pscale_name,
                              FnHSE_vol_0p2_1_after_a3=F,
                              FnHSE_x0p5=float(nlee(0.5) / nh(np.array([0.5]))[0]),
                              FnHSE_x1=float(nlee(1.0) / nh(np.array([1.0]))[0])))
res["z_dependence"] = zrows

# ------------------------------------------------------------------ 4 pressure amplitude at z=0
def m500_from_m200(m200, c):
    """NFW: find R500/R200 and M500/M200"""
    f = lambda y: np.log1p(y) - y / (1 + y)
    # mean density inside x R200 = 200 rho_c m(x)/x^3 ; solve = 500 rho_c
    g = lambda x: 200 * L.m_nfw(x, c) / x ** 3 - 500
    x500 = optimize.brentq(g, 0.2, 1.0)
    return x500, L.m_nfw(x500, c)


def arnaud_upp_pe(r_mpc, m500_msun, r500_mpc, z):
    h70 = L.H / 0.7
    P0, c500, gam, alp, bet = 8.403 * h70 ** -1.5, 1.177, 0.3081, 1.0510, 5.4905
    xx = r_mpc / r500_mpc
    px = P0 / ((c500 * xx) ** gam * (1 + (c500 * xx) ** alp) ** ((bet - gam) / alp))
    ap = 0.22 / (1 + 8 * xx ** 3) * 0  # alpha'_P term omitted (<~5 % effect); see Arnaud+10 eq. 13
    return 1.65e-3 * L.E(z) ** (8 / 3) * (m500_msun / (3e14 / h70)) ** (2 / 3 + 0.12 + ap) * px * h70 ** 2


def b12_pth_over_P200(x, m200_msun, z):
    m14 = m200_msun / 1e14
    P0 = 18.1 * m14 ** 0.154 * (1 + z) ** -0.758
    xc = 0.497 * m14 ** -0.00865 * (1 + z) ** 0.731
    beta = 4.35 * m14 ** 0.0393 * (1 + z) ** 0.415
    return L.gnfw(x, P0, xc, beta)


prow = []
for label, m200 in [("1e14 Msun", 1e14), ("1e14/h Msun", 1e14 / L.H), ("1e13.5/h Msun", 10 ** 13.5 / L.H)]:
    z = 0.0
    mh = m200 * L.H
    for cname, c in [("lee22c", L.lee22_klypin_c_z0(mh)), ("duffy", L.duffy08_c200c(mh, 0))]:
        x500, f500 = m500_from_m200(m200, c)
        r200 = L.r200c_mpc(m200, z)
        r500 = x500 * r200
        m500 = f500 * m200
        P200 = L.P200_keV_cm3(m200, z)
        # Y_sph(R500) in Mpc^2:  sigma_T/(m_e c^2) int P_e dV
        mec2_kev = L.M_E_G * L.C_CMS ** 2 / L.KEV_ERG
        fac = L.SIGMA_T / mec2_kev  # cm^2 / keV
        lx = np.linspace(np.log(1e-4), np.log(x500), 3000)
        xx = np.exp(lx)
        Ylee = fac * integrate.simpson(L.phat(xx, mh, z) * P200 * 4 * np.pi * (xx * r200 * L.MPC_CM) ** 3, x=lx)
        Yupp = fac * integrate.simpson(arnaud_upp_pe(xx * r200, m500, r500, z) * 4 * np.pi * (xx * r200 * L.MPC_CM) ** 3, x=lx)
        Yb12 = fac * integrate.simpson(L.PE_OVER_PTH * b12_pth_over_P200(xx, m200, z) * P200 * 4 * np.pi * (xx * r200 * L.MPC_CM) ** 3, x=lx)
        rr = dict(mass=label, c=cname, c_val=c, x500=x500, M500_over_M200=f500,
                  Ysph500_Lee_over_UPP=Ylee / Yupp, Ysph500_B12e_over_UPP=Yb12 / Yupp,
                  Ysph500_Lee_over_B12e=Ylee / Yb12)
        for xr in [0.3, 0.5, 1.0]:  # in units of R500
            xg = xr * x500
            rr[f"Pe_Lee_over_UPP_at_{xr}R500"] = float(L.phat(xg, mh, z) * P200 / arnaud_upp_pe(xg * r200, m500, r500, z))
            rr[f"Pe_Lee_over_B12e_at_{xr}R500"] = float(L.phat(xg, mh, z) / (L.PE_OVER_PTH * b12_pth_over_P200(xg, m200, z)))
        prow.append(rr)
res["pressure_z0"] = prow

json.dump(res, open(os.path.join(OUT, "c4_checks.json"), "w"), indent=1)

# ---- print a compact summary
print("# z=0: log10Mh  <n>  Mgas/fbM  Tnw/T200(0-1)  Tnw/T200(.05-1)  T/T200 x=.1 .3 .5 1   kTnw keV  local gas/(fb tot) R200")
for r in rows:
    print(f"{r['log10Mh']:5.2f} {r['nhat_mean_lt_R200']:6.3f} {r['Mgas_over_fbM200']:5.2f} {r['Tnw_over_T200_0_1']:.3f} {r['Tnw_over_T200_0p05_1']:.3f}  "
          f"{r['T_over_T200_x0.1']:.3f} {r['T_over_T200_x0.3']:.3f} {r['T_over_T200_x0.5']:.3f} {r['T_over_T200_x1.0']:.3f}  "
          f"{r['kTnw_keV']:.3f}  {r['local_gas_over_fb_total_at_R200_lee22c']:.2f}")
print("# HSE: log10Mh c  T_HSE/T200 x=.2 .5 1 | F_n^HSE x=.2 .5 1 | vol[.2,1] vol[.1,1] vol[.5,1] (fnt=0) | vol[.2,1] fnt=0.15 | Duffy vol[.2,1] | c x0.7, x1.3")
for r in rows:
    print(f"{r['log10Mh']:5.2f} {r['c_lee22c']:.2f}  {r['T_HSE_over_T200_x0.2_lee22c']:.2f} {r['T_HSE_over_T200_x0.5_lee22c']:.2f} {r['T_HSE_over_T200_x1.0_lee22c']:.2f} | "
          f"{r['FnHSE_x0.2_lee22c_fnt0.0']:.2f} {r['FnHSE_x0.5_lee22c_fnt0.0']:.2f} {r['FnHSE_x1.0_lee22c_fnt0.0']:.2f} | "
          f"{r['FnHSE_vol_0p2_1_lee22c_fnt0.0']:.2f} {r['FnHSE_vol_0p1_1_lee22c_fnt0.0']:.2f} {r['FnHSE_vol_0p5_1_lee22c_fnt0.0']:.2f} | "
          f"{r['FnHSE_vol_0p2_1_lee22c_fnt0.15']:.2f} | {r['FnHSE_vol_0p2_1_duffy_fnt0.0']:.2f} | "
          f"{r['FnHSE_vol_0p2_1_lee22c_x0.7']:.2f} {r['FnHSE_vol_0p2_1_lee22c_x1.3']:.2f}")
print("# z-dependence of F_n^HSE (vol 0.2-1) after removing a^3 from density")
for r in zrows:
    print(f"{r['log10Mh']:5.2f} z={r['z']:.1f} c={r['c']:.2f} {r['pressure']:14s} F={r['FnHSE_vol_0p2_1_after_a3']:.2f}  x0.5 {r['FnHSE_x0p5']:.2f}  x1 {r['FnHSE_x1']:.2f}")
print("# pressure at z=0")
for r in prow:
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
