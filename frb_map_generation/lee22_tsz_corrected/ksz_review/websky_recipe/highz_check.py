"""Review of paint_ksz_websky.jl (2026-10-02, ksz_review/websky_recipe): does WebSky's halo range z < 4.6 (vs HalfDome z < 3.86)
matter at ell 500-5000?  Independent one-halo engine (not ksz_debug/onehalo.py) on the WebSky catalogue histogram
(ksz_debug websky_halo_hist.npz: M200c from c200m = 7, Omega_m 0.31, h 0.68, 0 < z < 4.75):
  C_ell = (1/4pi) sum_cells sum_v (v/c)^2 |tau_ell|^2,  tau_ell = sigma_T/dA^2 [int_0^{X R200c} 4pi r^2 n_e j0(k r) dr - N_comp W(k R_comp)]
Recipes: 'websky' = cut (M200c > 1e13 Msun, theta200c > 0.5') + n_e (1-Y/2) rho/m_p + 1.5 R200c sphere + uniform sphere f_b M200m at 3 rho_m;
         'phys'   = all halos, 0.9 x XGPaint electrons, 4 R200c sphere, no compensation (the *_phys maps).
Usage: python3 highz_check.py HIST_NPZ FIG6_CSV OUTDIR"""
import sys, json
import numpy as np

HIST, FIG6, OUT = sys.argv[1:4]
MSUN = 1.98847e30; MPC = 3.0856775814913673e22; MP = 1.67262192369e-27; ME = 9.1093837015e-31; SIGT = 6.6524587321e-29
C_KMS = 299792.458; TCMB = 2.7255e6
XH = 0.76; M_PER_E = ME + 2 * XH / (XH + 1) * MP + (1 - XH) / (2 * (1 + XH)) * 4 * MP
h, Om, Ob = 0.68, 0.31, 0.049; fb = Ob / Om
E = lambda z: np.sqrt(Om * (1 + z)**3 + 1 - Om)
zz = np.linspace(0, 6, 60001); inv = 1 / E(zz)
chi = np.concatenate([[0], np.cumsum(0.5 * (inv[1:] + inv[:-1]) * np.diff(zz))]) * C_KMS / (100 * h)
DA = lambda z: np.interp(z, zz, chi) / (1 + z)
rhoc0 = 2.77536627e11 * h**2
rhoc = lambda z: rhoc0 * E(z)**2
mnfw = lambda x: np.log1p(x) - x / (1 + x)


def ratio_c7(z):
    omz = Om * (1 + z)**3 / E(z)**2; tg = mnfw(7.0) / (7.0**3 * omz); lo, hi = 1e-3, 7.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if mnfw(mid) / mid**3 > tg: lo = mid
        else: hi = mid
    return mnfw(0.5 * (lo + hi)) / mnfw(7.0)


def tau_ell(M, z, ells, X, ne, comp):
    m = M / 1e14; z1 = 1 + z
    P0 = 4e3 * m**0.29 * z1**-0.66; a = 0.88 * m**-0.03 * z1**0.19; b = 3.83 * m**0.04 * z1**-0.025; g = -0.2; xc = 0.5
    R200 = (3 * M / (4 * np.pi * 200 * rhoc(z)))**(1 / 3); dA = DA(z)
    epm = 0.9 * MSUN / M_PER_E if ne == "xgp" else 0.88 * MSUN / MP
    lnr = np.linspace(np.log(1e-5 * R200), np.log(X * R200), 1500); r = np.exp(lnr)
    xb = r / R200 / xc
    n_e = fb * rhoc(z) * P0 * xb**g * (1 + xb**a)**(-(b + g) / a) * epm          # electrons / Mpc^3
    k = ells / dA
    w = 4 * np.pi * r**3 * n_e
    Nk = np.trapz(w[None, :] * np.sinc(np.outer(k, r) / np.pi), lnr, axis=1) + 4 * np.pi / 3 * n_e[0] * r[0]**3
    if comp:
        Mm = M / ratio_c7(z)
        R3 = (3 * Mm / (4 * np.pi * 3.0 * Om * rhoc0 * (1 + z)**3))**(1 / 3)
        q = k * R3
        W = np.where(q < 1e-3, 1 - q**2 / 10, 3 * (np.sin(q) - q * np.cos(q)) / np.maximum(q, 1e-30)**3)
        Nk = Nk - fb * Mm * 0.88 * MSUN / MP * W
    return SIGT * Nk / (dA * MPC)**2


d = np.load(HIST)
le, ze, n, v2 = d["logm_edges"], d["z_edges"], d["count_m200c"], d["sum_vr_over_c_sq_m200c"]
lm = 0.5 * (le[1:] + le[:-1]); zc = 0.5 * (ze[1:] + ze[:-1])
ells = np.array([300., 500., 1000., 2000., 3000., 5000.])
pre = ells * (ells + 1) / (2 * np.pi) * TCMB**2
ws = np.loadtxt(FIG6, delimiter=","); ws = ws[ws[:, 0] > 10]
wsv = np.array([ws[np.abs(np.log(ws[:, 0] / L)) < 0.1, 1].mean() for L in ells])
ARC = np.pi / 180 / 60
res = {}
for recipe in ("websky", "phys"):
    cl_all = np.zeros(len(ells)); cl_lo = np.zeros(len(ells)); cl_hi = np.zeros(len(ells)); n_hi_sel = 0; n_sel = 0
    for i, j in zip(*np.nonzero(n > 0)):
        M, z = 10**lm[i], zc[j]
        if recipe == "websky":
            R200 = (3 * M / (4 * np.pi * 200 * rhoc(z)))**(1 / 3)
            if not (M > 1e13 and R200 / DA(z) > 0.5 * ARC):
                continue
            t = tau_ell(M, z, ells, 1.5, "websky", True)
        else:
            t = tau_ell(M, z, ells, 4.0, "xgp", False)
        c = v2[i, j] * t**2 / (4 * np.pi)
        cl_all += c; n_sel += n[i, j]
        if ze[j + 1] <= 3.86 + 1e-9: cl_lo += c
        else: cl_hi += c; n_hi_sel += n[i, j]
    D = pre * cl_all
    res[recipe] = dict(ell=ells.tolist(), D_all=D.tolist(), frac_z_gt_3p86=(cl_hi / cl_all).tolist(),
                       ratio_lo_over_all=(cl_lo / cl_all).tolist(), D_over_fig6=(D / wsv).tolist(), n_selected=float(n_sel),
                       n_selected_z_gt_3p86=float(n_hi_sel))
    print(recipe, "N selected %.4g, of which z > 3.86 (cells entirely above): %.4g" % (n_sel, n_hi_sel))
    print("  ell            ", " ".join("%9.0f" % L for L in ells))
    print("  D_ell [uK^2]   ", " ".join("%9.4f" % x for x in D))
    print("  / Fig.6 halo   ", " ".join("%9.4f" % x for x in D / wsv))
    print("  z>3.86 share   ", " ".join("%9.2e" % x for x in cl_hi / cl_all))
# halos above z = 3.86 in the catalogue at all, and with M200c > 1e13
hi = ze[:-1] >= 3.86 - 1e-9
print("WebSky halos in z bins starting >= 3.86: %d (M200c > 1e13: %d)" % (n[:, hi].sum(), n[lm > 13, :][:, hi].sum()))
# largest theta200c of any z > 3.86 cell with halos
best = 0
for i, j in zip(*np.nonzero(n > 0)):
    if ze[j] >= 3.86:
        M, z = 10**lm[i], zc[j]; th = (3 * M / (4 * np.pi * 200 * rhoc(z)))**(1 / 3) / DA(z) / ARC
        best = max(best, th)
print("largest theta200c [arcmin] among occupied z >= 3.86 cells: %.3f (cut 0.5)" % best)
res["max_theta200c_arcmin_z_gt_3p86"] = best
json.dump(res, open(OUT + "/highz_check.json", "w"), indent=1)
