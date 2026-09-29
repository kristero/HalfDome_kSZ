"""verify_3: TNG300 z=0 anchors from Pop et al. 2022 (arXiv:2205.11528, TeX source fetched
2026-09-28; Tables/table_Tx_M500.tex, table_Txce_M500.tex, table_Mgas_M500.tex, SBPL.tex eq. SBPL):
  T_X(<R500,SIM)      SPL: A=0.248, alpha=0.537  (log10 keV at M500=1e14 Msun)
  T_X,ce(0.15-1 R500) SPL: A=0.228, alpha=0.551
  M_gas,500 (hot, non-SF, subhalo gas excluded) SBPL: A=13.093, a1=1.671, a2=1.086, delta=0.19, Xp=0.57e14
Compared with Lee22 no-c fits (literal) at z=0:
  F_T  = T_TNG / T_Lee22,ew  (same 0.15-1 R500 aperture for T_X,ce)   -> F_n/F_p
  F_g  = f_gas500,Lee22-literal / f_gas500,TNG                         -> F_n (upper-ish: subhalo gas excluded in Pop22)
Also the HMF-weighted TNG gas fraction inside R200c in the 1e13-1e14 Msun window from the
Ayromlou+23 - Pillepich+18b table digitised by investigator D (not re-digitised here)."""
import numpy as np, json, os
from scipy import optimize
import lee22_c4_lib as L

OUT = "/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_normalization_forensics_20260928/verify_3"


def spl(A, a, x):
    return 10 ** A * (x / 1e14) ** a


def sbpl(A, a1, a2, delta, xp, x, xnorm=1e14):
    ratio = (x / xnorm) ** ((a2 + a1) / 2) * (np.cosh(np.log10(x / xp) / delta) / np.cosh(np.log10(xnorm / xp) / delta)) ** ((a2 - a1) / 2 * delta * np.log(10))
    return 10 ** A * ratio


def m200_from_m500(m500):
    """iterate: M200c (Msun) whose NFW (Lee22 c(M)) M500c equals m500"""
    def f(lm200):
        m200 = 10 ** lm200
        c = L.lee22_klypin_c_z0(m200 * L.H)
        x500 = optimize.brentq(lambda x: 200 * L.m_nfw(x, c) / x ** 3 - 500, 0.2, 1.0)
        return np.log10(L.m_nfw(x500, c) * m200) - np.log10(m500)
    lm = optimize.brentq(f, np.log10(m500), np.log10(m500) + 0.5)
    m200 = 10 ** lm
    c = L.lee22_klypin_c_z0(m200 * L.H)
    x500 = optimize.brentq(lambda x: 200 * L.m_nfw(x, c) / x ** 3 - 500, 0.2, 1.0)
    return m200, c, x500


rows = []
for m500 in [1e13, 2e13, 3e13, 5e13, 1e14, 2e14]:
    m200, c, x500 = m200_from_m500(m500)
    mh = m200 * L.H
    kT200 = L.kT200_keV(m200, 0.0)
    ew = lambda a, b: (L.X_H / L.MU) * L.vol_int(lambda x: L.phat(x, mh, 0.0), a, b) / L.vol_int(lambda x: L.nhat(x, mh, 0.0), a, b) * kT200
    T_lee_ce = ew(0.15 * x500, x500)
    T_lee_500 = ew(1e-4, x500)
    T_lee_200 = ew(1e-4, 1.0)
    Tx = spl(0.248, 0.537, m500)
    Txce = spl(0.228, 0.551, m500)
    fg_tng = sbpl(13.093, 1.671, 1.086, 0.19, 0.57e14, m500) / m500 / L.FB
    fg_lee = L.vol_int(lambda x: L.nhat(x, mh, 0.0), 1e-4, x500) / 3 * 3 * (L.MU_E / L.X_H) * 200 * x500 ** 3 / (500 * x500 ** 3)
    # careful: M_gas(<x500)/(f_b M500) = [int_0^x500 nhat 3x^2 dx] * mu_e/X_H * (200/500) / x500^3 ... see note
    mg_over_fbM200 = L.vol_int(lambda x: L.nhat(x, mh, 0.0), 1e-4, x500) * L.MU_E / L.X_H   # M_gas(<R500)/(f_b M200)
    m500_over_m200 = m500 / m200
    fg_lee = mg_over_fbM200 / m500_over_m200
    rows.append(dict(M500_msun=m500, M200_msun=m200, log10M200h=np.log10(mh), c=c, x500=x500,
                     kT200_keV=kT200, kT_Lee_ew_0p15_1R500=T_lee_ce, kT_Lee_ew_lt_R500=T_lee_500,
                     kT_Lee_ew_lt_R200=T_lee_200, kTX_Pop22=Tx, kTXce_Pop22=Txce,
                     F_T_ce=Txce / T_lee_ce, F_T_R500=Tx / T_lee_500,
                     TX_over_T200_Pop22=Tx / kT200,
                     fgas500_over_fb_Pop22=fg_tng, fgas500_over_fb_Lee_literal=fg_lee,
                     F_gas500=fg_lee / fg_tng))
res = {"pop22_rows": rows}

# HMF-weighted TNG gas fraction inside R200c, 1e13-1e14 Msun (D's digitisation of Ayromlou+23 - Pillepich+18b)
D = json.load(open("/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_normalization_forensics_20260928/D_gas_fractions/tng_literature_gas_fractions.json"))
t = D["tng_gas_fraction_R200c_z0"]
lm_t, fg_t = np.array(t["log10_m200c_msun"]), np.array(t["fgas_over_fbcos"])
kz = json.load(open(os.path.join(OUT, "konietzka_lowz_anchor.json")))
# weight dn/dlnM * M ~ approximate with a power law dn/dlnM ~ M^-0.9 (Tinker08 slope near 1e13.5 at z~0.05)
lm = np.linspace(13, 14, 101)
for slope in [-0.8, -1.0, -1.2]:
    w = 10 ** (lm * (1 + slope))
    res[f"hmf_weighted_fgas_R200c_TNG_lit_13to14_slope{slope}"] = float(np.sum(w * np.interp(lm, lm_t, fg_t)) / np.sum(w))
res["konietzka_implied_fe_13to14_zs0.1"] = kz["mass13to14|zs=0.1"]["implied_fe_TNG"]
json.dump(res, open(os.path.join(OUT, "pop22_anchor.json"), "w"), indent=1)
for r in rows:
    print({k: round(v, 3) if isinstance(v, float) else v for k, v in r.items()})
print({k: v for k, v in res.items() if k != "pop22_rows"})
