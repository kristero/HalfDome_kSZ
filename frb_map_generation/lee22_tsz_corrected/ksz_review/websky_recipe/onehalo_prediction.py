"""Review of paint_ksz_websky.jl (2026-10-02, ksz_review/websky_recipe): one-halo prediction for the b16_websky map from the HalfDome
catalogue with EXACTLY the painter's recipe (HalfDome cosmology, M200c = M200m x NFW c200m=7 ratio, cut M200c > 1e13 Msun and
theta200c > 0.5', 1.5 R200c sphere, n_e = (1-Y/2) rho/m_p, minus a uniform sphere f_b M200m at 3 rho_m(z)), plus the sensitivity of
that prediction to the recipe's ambiguous choices (catalogue M200c instead of c7, Delta = 4, 0.25' cut, 0.9 on the compensation,
the 0.9 kept on the profile, X = 2).  Independent engine (not ksz_debug/onehalo.py).
Usage: python3 onehalo_prediction.py CATALOG FIG6_CSV OUTDIR"""
import sys, json, time
import numpy as np, h5py

CAT, FIG6, OUT = sys.argv[1:4]
H = 0.6774; OB = 0.0486; OC = 0.2603; OM = OB + OC; FB = OB / OM
MSUN = 1.98847e30; MPC = 3.0856775814913673e22; MP = 1.67262192369e-27; ME = 9.1093837015e-31; SIGT = 6.6524587321e-29
C_KMS = 299792.458; TCMB = 2.7255e6; G = 6.67430e-11
XH = 0.76; M_PER_E = ME + 2 * XH / (XH + 1) * MP + (1 - XH) / (2 * (1 + XH)) * 4 * MP
OG = 4.48131e-7 * 2.7255**4 / H**2; OR = OG * (1 + 3.046 * 7 / 8 * (4 / 11)**(4 / 3)); OL = 1 - OM - OR
E2 = lambda z: OM * (1 + z)**3 + OR * (1 + z)**4 + OL
ZG = np.linspace(0, 6, 600001); inv = 1 / np.sqrt(E2(ZG))
CHI = np.concatenate([[0], np.cumsum(0.5 * (inv[1:] + inv[:-1]) * np.diff(ZG))]) * C_KMS / (100 * H)
DA = lambda z: np.interp(z, ZG, CHI) / (1 + z)                        # Mpc
RHOC0 = 3 * (100 * H * 1e3 / MPC)**2 / (8 * np.pi * G) / MSUN * MPC**3  # Msun/Mpc^3
rhoc = lambda z: RHOC0 * E2(z)
mnfw = lambda x: np.log1p(x) - x / (1 + x)


def ratio_c7_exact(z):
    omz = OM * (1 + z)**3 / (OM * (1 + z)**3 + 1 - OM); tg = mnfw(7.0) / (7.0**3 * omz); lo, hi = 1e-3, 7.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if mnfw(mid) / mid**3 > tg: lo = mid
        else: hi = mid
    return mnfw(0.5 * (lo + hi)) / mnfw(7.0)


RZ = np.linspace(0, 5, 2001); RV = np.array([ratio_c7_exact(z) for z in RZ]); ratio_c7 = lambda z: np.interp(z, RZ, RV)

# ---------------------------------------------------------------- catalogue histograms: (log10 M200c_c7, z) and (log10 M200c_cat, z)
LE = np.arange(12.0, 15.7001, 0.025); ZE = np.arange(0.0, 4.0001, 0.025)
nb = (len(LE) - 1) * (len(ZE) - 1)
hist = {k: np.zeros(nb) for k in ("n_c7", "v2_c7", "n_cat", "v2_cat")}
t0 = time.time()
with h5py.File(CAT, "r") as f:
    N = f["redshift"].shape[0]; CH = 4_000_000
    for s in range(0, N, CH):
        e = min(s + CH, N)
        z = f["redshift"][s:e].astype(np.float64); mc = f["halo_mass_m200c"][s:e] / H; mm = f["halo_mass_m200m"][s:e] / H
        pos = f["Position"][s:e].astype(np.float64); vel = f["Velocity"][s:e].astype(np.float64)
        ok = np.isfinite(mc) & (mc > 0) & np.isfinite(mm) & (mm > 0) & np.isfinite(z) & (z > 0) & \
            np.all(np.isfinite(pos), 1) & np.all(np.isfinite(vel), 1)
        z, mc, mm, pos, vel = z[ok], mc[ok], mm[ok], pos[ok], vel[ok]
        v2 = (np.sum(pos * vel, 1) / np.sqrt(np.sum(pos**2, 1)) / C_KMS)**2
        iz = np.clip(np.searchsorted(ZE, z, side="right") - 1, 0, len(ZE) - 2)
        for tag, m in (("c7", mm * ratio_c7(z)), ("cat", mc)):
            im = np.clip(np.searchsorted(LE, np.log10(m), side="right") - 1, 0, len(LE) - 2)
            idx = im * (len(ZE) - 1) + iz
            hist["n_" + tag] += np.bincount(idx, minlength=nb); hist["v2_" + tag] += np.bincount(idx, v2, nb)
        print("rows %d-%d %.0f s" % (s, e, time.time() - t0), flush=True)
for k in hist: hist[k] = hist[k].reshape(len(LE) - 1, len(ZE) - 1)
np.savez(OUT + "/halfdome_hist_c7_and_catalogue_m200c.npz", logm_edges=LE, z_edges=ZE, **hist)

# ---------------------------------------------------------------- one-halo engine
ells = np.array([300., 500., 1000., 2000., 3000., 5000.]); pre = ells * (ells + 1) / (2 * np.pi) * TCMB**2
ws = np.loadtxt(FIG6, delimiter=","); ws = ws[ws[:, 0] > 10]
wsv = np.array([ws[np.abs(np.log(ws[:, 0] / L)) < 0.1, 1].mean() for L in ells])
ARC = np.pi / 180 / 60


def tau_ell(M, z, X, ne_fac, comp_D, comp_fac):
    m = M / 1e14; z1 = 1 + z
    P0 = 4e3 * m**0.29 * z1**-0.66; a = 0.88 * m**-0.03 * z1**0.19; b = 3.83 * m**0.04 * z1**-0.025; g = -0.2; xc = 0.5
    R200 = (3 * M / (4 * np.pi * 200 * rhoc(z)))**(1 / 3); dA = DA(z)
    lnr = np.linspace(np.log(1e-5 * R200), np.log(X * R200), 1500); r = np.exp(lnr); xb = r / R200 / xc
    n_e = FB * rhoc(z) * P0 * xb**g * (1 + xb**a)**(-(b + g) / a) * ne_fac * MSUN / MP
    k = ells / dA
    Nk = np.trapz((4 * np.pi * r**3 * n_e)[None, :] * np.sinc(np.outer(k, r) / np.pi), lnr, axis=1) + 4 * np.pi / 3 * n_e[0] * r[0]**3
    if comp_D:
        Mm = M / ratio_c7(z)
        R3 = (3 * Mm / (4 * np.pi * comp_D * OM * RHOC0 * (1 + z)**3))**(1 / 3); q = k * R3
        W = np.where(q < 1e-3, 1 - q**2 / 10, 3 * (np.sin(q) - q * np.cos(q)) / np.maximum(q, 1e-30)**3)
        Nk = Nk - FB * Mm * comp_fac * MSUN / MP * W
    return SIGT * Nk / (dA * MPC)**2


lmc = 0.5 * (LE[1:] + LE[:-1]); zc = 0.5 * (ZE[1:] + ZE[:-1])
WS_NE = 0.88; XGP_NE = 0.9 * MP / M_PER_E
RECIPES = {
    "painter (c7, 0.5' radius, D=3, X=1.5, no 0.9)": dict(mass="c7", th=0.5, X=1.5, ne=WS_NE, D=3.0, cf=WS_NE),
    "catalogue M200c instead of c7":                  dict(mass="cat", th=0.5, X=1.5, ne=WS_NE, D=3.0, cf=WS_NE),
    "Delta = 4 (delta = 3)":                          dict(mass="c7", th=0.5, X=1.5, ne=WS_NE, D=4.0, cf=WS_NE),
    "theta cut 0.25' (0.5' diameter)":                dict(mass="c7", th=0.25, X=1.5, ne=WS_NE, D=3.0, cf=WS_NE),
    "0.9 on the compensation only":                   dict(mass="c7", th=0.5, X=1.5, ne=WS_NE, D=3.0, cf=0.9 * WS_NE),
    "keep XGPaint 0.9 on the profile":                dict(mass="c7", th=0.5, X=1.5, ne=XGP_NE, D=3.0, cf=WS_NE),
    "X = 2.0 R200c":                                  dict(mass="c7", th=0.5, X=2.0, ne=WS_NE, D=3.0, cf=WS_NE),
}
res = {"ell": ells.tolist(), "fig6": wsv.tolist()}
base = None
for name, r in RECIPES.items():
    n, v2 = hist["n_" + r["mass"]], hist["v2_" + r["mass"]]
    cl = np.zeros(len(ells))
    for i, j in zip(*np.nonzero(n > 0)):
        M, z = 10**lmc[i], zc[j]
        R200 = (3 * M / (4 * np.pi * 200 * rhoc(z)))**(1 / 3)
        if not (M > 1e13 and np.arctan(R200 / DA(z)) > r["th"] * ARC):
            continue
        cl += v2[i, j] * tau_ell(M, z, r["X"], r["ne"], r["D"], r["cf"])**2 / (4 * np.pi)
    D = pre * cl
    if base is None: base = D
    res[name] = dict(D=D.tolist(), over_fig6=(D / wsv).tolist(), over_painter=(D / base).tolist())
    print("%-48s D/Fig6 %s | /painter %s" % (name, " ".join("%.3f" % x for x in D / wsv), " ".join("%.3f" % x for x in D / base)), flush=True)
print("ells", ells.tolist())
json.dump(res, open(OUT + "/onehalo_prediction.json", "w"), indent=1)
print("done %.0f s" % (time.time() - t0))
