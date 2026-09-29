"""Independent catalogue-weighted one-halo (Poisson) C_ell^yy for the painted B12, Lee22 (as painted) and Lee22 x (1+z)^2 maps.

Written from scratch (does not import the workstream's halo_model.py / common.py / lee22_models.py):
  * halos: lightcone_100.hdf5, halo_mass_m200c [Msun/h] -> M = m/h physical Msun (h = 0.68), 'redshift'
  * cosmology: flat LCDM Omega_m = 0.049 + 0.261 = 0.31, h = 0.68, no radiation; rho_c(z) physical; R200c physical
  * P200 = 200 G M rho_c(z) f_b / (2 R200), f_b = Omega_b/Omega_m
  * B12 (AGN, thermal) x 0.5176 -> P_e;  Lee22 Table 7 no-c fit is P_e/P200 directly (masses h^-1 Msun = catalogue value,
    Mcut = 10^13.60, shape (xc, beta) frozen above 10^14.8 h^-1 Msun, P0 keeps slope 0.80)
  * y(R) = sigma_T/(m_e c^2) int P_e dl, infinite LOS, disc truncated at 4 R200c (the painter's aperture)
  * flat-sky y_ell = 2 pi theta200^2 int_0^4 X y(X) J0((ell+1/2) theta200 X) dX (band mean over [0.9, 1.1] ell, Simpson);  C_ell^1h = sum_i y_ell,i^2 / (4 pi)
Halos are binned in (log10 m: 0.02 dex, log10 z: 0.01 dex) x (6 mass x 6 z decomposition classes); the per-bin kernel is
evaluated at the mean log m, log z of the halos in the bin.
"""
import os, json, time
import numpy as np
import h5py
from scipy.special import j0
from scipy.interpolate import RegularGridInterpolator
from scipy.integrate import quad

OUT = "/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_tsz_verification_20260929/verify_B_class_sz"
CAT = "/home/cbllover/HalfDome/lightcone_100.hdf5"

# ---------------------------------------------------------------- constants, cosmology
G = 6.67430e-8; MSUN = 1.98847e33; MPC = 3.0856775814913673e24; C_KMS = 299792.458
SIGMA_T = 6.6524587321e-25; MEC2 = 8.1871057769e-7   # cm^2, erg
H = 0.68; OB = 0.049; OC = 0.261; OM = OB + OC; FB = OB / OM
H0 = 100 * H * 1e5 / MPC                                  # s^-1

def E(z):
    return np.sqrt(OM * (1 + z) ** 3 + 1 - OM)

zt = np.linspace(0, 4.0, 40001)
chi_t = np.concatenate([[0], np.cumsum(0.5 * (1 / E(zt[1:]) + 1 / E(zt[:-1])) * np.diff(zt))]) * C_KMS / (100 * H)  # Mpc

def D_A_cm(z):
    return np.interp(z, zt, chi_t) / (1 + z) * MPC

def rho_c(z):
    return 3 * H0 ** 2 / (8 * np.pi * G) * E(z) ** 2

def R200(Mphys, z):
    return (3 * Mphys * MSUN / (4 * np.pi * 200 * rho_c(z))) ** (1 / 3)

def P200(Mphys, z):
    return 200 * G * Mphys * MSUN * rho_c(z) * FB / (2 * R200(Mphys, z))

# ---------------------------------------------------------------- profile parameters (gamma = -0.3, alpha = 1 for both)
def b12_params(Mphys, z):
    m = Mphys / 1e14; zp = 1 + z
    return (0.5176 * 18.1 * m ** 0.154 * zp ** -0.758, 0.497 * m ** -0.00865 * zp ** 0.731, 4.35 * m ** 0.0393 * zp ** 0.415)

def lee_params(mh, z, zextra=0.0):
    mcut = 10 ** 13.60; clip = 10 ** 14.8
    r = mh / mcut; rs = np.minimum(mh, clip) / mcut; zp = 1 + z
    P0 = 2.8 * np.where(r < 1, r ** 1.08, r ** 0.80) * zp ** (-1.89 + zextra)
    return P0, 2.10 * rs ** -0.35 * zp ** 0.53, 9.4 * rs ** -0.06 * zp ** 0.60

# ---------------------------------------------------------------- Pi1(Y; beta) = int_-inf^inf s^-0.3 (1+s)^-beta dv, s = sqrt(Y^2+v^2)
lnY = np.linspace(np.log(1e-6), np.log(20.0), 500)
betas = np.linspace(3.0, 32.0, 291)
v = np.concatenate([[0.0], np.logspace(-9, 5, 3000)])
def build_pi1():
    tab = np.empty((lnY.size, betas.size))
    for i, ly in enumerate(lnY):
        Y = np.exp(ly)
        s = np.sqrt(Y ** 2 + v ** 2)
        f = s[None, :] ** -0.3 * (1 + s[None, :]) ** -betas[:, None]
        tab[i] = 2 * np.trapz(f, v, axis=1)
    return tab
t0 = time.time()
PI1 = build_pi1()
# spot check against quad
for Y, b in [(1e-3, 4.35), (0.5, 9.0), (3.0, 20.0)]:
    ref = 2 * quad(lambda vv: np.sqrt(Y * Y + vv * vv) ** -0.3 * (1 + np.sqrt(Y * Y + vv * vv)) ** -b, 0, np.inf, limit=500)[0]
    ib = np.argmin(abs(betas - b)); il = np.argmin(abs(lnY - np.log(Y)))
    print(f"Pi1 table check Y={Y} beta={b}: table/quad at nearest node = {PI1[il, ib] / (2 * quad(lambda vv: np.sqrt(np.exp(lnY[il])**2 + vv * vv) ** -0.3 * (1 + np.sqrt(np.exp(lnY[il])**2 + vv * vv)) ** -betas[ib], 0, np.inf, limit=500)[0]):.6f}")
lnPI = RegularGridInterpolator((lnY, betas), np.log(PI1), method="linear")
print("Pi1 table built in %.1f s" % (time.time() - t0))

X = np.concatenate([np.logspace(-5, np.log10(0.004), 60)[:-1], np.linspace(0.004, 4.0, 2400)])
ELLS = np.array([30, 100, 300, 1000, 2000, 3000, 5000], float)
FR = np.array([0.9, 1.0, 1.1]); FW = np.array([1, 4, 1]) / 6.0   # Simpson band mean
ELL_EVAL = (ELLS[:, None] * FR[None, :]).ravel()                # 35 multipoles, band mean = mean over FR

def yell(P0, xc, beta, theta200, amp):
    """amp = sigma_T/mec2 * P200 * R200 (dimensionless y scale); returns y_ell at ELL_EVAL (+ ell->0 value first)."""
    n = P0.size
    Yg = (X[None, :] / xc[:, None])
    pts = np.stack([np.log(np.clip(Yg, 1e-6, 20.0)).ravel(), np.repeat(beta, X.size)], axis=1)
    Pi = xc[:, None] * np.exp(lnPI(pts)).reshape(n, X.size)
    w = X[None, :] * Pi
    q = (ELL_EVAL[None, :, None] + 0.5) * theta200[:, None, None] * X[None, None, :]
    integ = np.trapz(w[:, None, :] * j0(q), X, axis=2)             # (n, nell)
    Y0 = np.trapz(w, X, axis=1)
    pref = 2 * np.pi * theta200 ** 2 * amp * P0
    return pref[:, None] * integ, pref * Y0

# ---------------------------------------------------------------- catalogue binning
MEDGES = np.array([4.4e12, 1e13, 3e13, 1e14, 3e14, 10 ** 14.8, 3e15])        # Msun/h
ZEDGES = np.array([0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0])
LM0, DLM, NLM = 12.64, 0.02, 140
LZ0, DLZ, NLZ = -2.70, 0.01, 330
def main():
    t0 = time.time()
    NK = 36 * 2 * NLM * NLZ
    cnt = np.zeros(NK); slm = np.zeros(NK); slz = np.zeros(NK)
    counts = {"N_total": 0, "N_above_1e14.8_msunh": 0, "N_above_1e14.8_z_lt_0.25": 0, "N_above_6.3e14_z_lt_0.1": 0, "N_calib": 0,
              "m_min": np.inf, "m_max": 0.0, "z_min": np.inf, "z_max": 0.0}
    with h5py.File(CAT, "r") as f:
        N = f["redshift"].shape[0]
        for s0 in range(0, N, 10_000_000):
            m = f["halo_mass_m200c"][s0:s0 + 10_000_000]
            z = f["redshift"][s0:s0 + 10_000_000].astype(np.float64)
            counts["N_total"] += int(m.size)
            counts["N_above_1e14.8_msunh"] += int(np.sum(m >= 10 ** 14.8))
            counts["N_above_1e14.8_z_lt_0.25"] += int(np.sum((m >= 10 ** 14.8) & (z < 0.25)))
            counts["N_above_6.3e14_z_lt_0.1"] += int(np.sum((m >= 6.3e14) & (z < 0.1)))
            counts["N_calib"] += int(np.sum((m >= 1e13) & (m < 10 ** 14.8) & (z <= 2.0)))
            counts["m_min"] = float(min(counts["m_min"], m.min())); counts["m_max"] = float(max(counts["m_max"], m.max()))
            counts["z_min"] = float(min(counts["z_min"], z.min())); counts["z_max"] = float(max(counts["z_max"], z.max()))
            lm = np.log10(m); lz = np.log10(z)
            im = np.clip(((lm - LM0) / DLM).astype(np.int64), 0, NLM - 1)
            iz = np.clip(((lz - LZ0) / DLZ).astype(np.int64), 0, NLZ - 1)
            mb = np.clip(np.searchsorted(MEDGES, m, side="right") - 1, 0, 5)
            zb = np.clip(np.searchsorted(ZEDGES, z, side="right") - 1, 0, 5)
            calib = ((m >= 1e13) & (m < 10 ** 14.8) & (z <= 2.0)).astype(np.int64)
            key = (((mb * 6 + zb) * 2 + calib) * NLM + im) * NLZ + iz
            cnt += np.bincount(key, minlength=NK); slm += np.bincount(key, weights=lm, minlength=NK)
            slz += np.bincount(key, weights=lz, minlength=NK)
    print("catalogue binned %.1f s" % (time.time() - t0)); print(counts)
    uk = np.nonzero(cnt)[0]; cnt = cnt[uk]
    lmb = slm[uk] / cnt; lzb = slz[uk] / cnt
    rest = uk // (NLM * NLZ); calib_b = rest % 2; cls = rest // 2; mbin = cls // 6; zbin = cls % 6
    print("bins: %d (%.1f s)" % (uk.size, time.time() - t0))
    mh = 10 ** lmb; zz = 10 ** lzb; Mphys = mh / H
    r200 = R200(Mphys, zz); th = r200 / D_A_cm(zz)
    amp = SIGMA_T / MEC2 * P200(Mphys, zz) * r200
    res = {}
    for name, pars in [("b12", b12_params(Mphys, zz)), ("lee22", lee_params(mh, zz)), ("lee22m1", lee_params(mh, zz, 2.0))]:
        yl = np.empty((mh.size, ELL_EVAL.size)); Y0 = np.empty(mh.size)
        for s in range(0, mh.size, 200):
            sl = slice(s, s + 200)
            yl[sl], Y0[sl] = yell(pars[0][sl], pars[1][sl], pars[2][sl], th[sl], amp[sl])
        res[name] = (yl, Y0)
        print(name, "done %.1f s" % (time.time() - t0))
    # C_ell per (class) = sum cnt * y_ell^2 / 4pi, band mean over FR
    out = {"counts": counts, "ells": ELLS.tolist(), "convention": __doc__}
    sel_all = np.ones(mh.size, bool)
    def cl(name, sel):
        yl = res[name][0][sel]
        c = (cnt[sel][:, None] * yl ** 2).sum(0) / (4 * np.pi)
        return (c.reshape(ELLS.size, FR.size) * FW[None, :]).sum(1)
    tab = {}
    tab["b12_all"] = cl("b12", sel_all); tab["lee22_all"] = cl("lee22", sel_all); tab["lee22m1_all"] = cl("lee22m1", sel_all)
    tab["lee22_calib"] = cl("lee22", calib_b == 1); tab["lee22m1_calib"] = cl("lee22m1", calib_b == 1)
    tab["b12_lowmass"] = cl("b12", mbin < 5); tab["b12_highmass"] = cl("b12", mbin == 5)
    tab["lee22_lowmass"] = cl("lee22", mbin < 5); tab["lee22_highmass"] = cl("lee22", mbin == 5)
    for i in range(6):
        tab[f"b12_mbin{i}"] = cl("b12", mbin == i); tab[f"lee22_mbin{i}"] = cl("lee22", mbin == i)
        tab[f"b12_zbin{i}"] = cl("b12", zbin == i); tab[f"lee22_zbin{i}"] = cl("lee22", zbin == i)
    tab["b12_high_zlt0.1"] = cl("b12", (mbin == 5) & (zbin == 0)); tab["lee22_high_zlt0.1"] = cl("lee22", (mbin == 5) & (zbin == 0))
    out["cl_1h_catalogue"] = {k: v.tolist() for k, v in tab.items()}
    np.savez(OUT + "/v2_catalogue_poisson.npz", ells=ELLS, **tab)
    # per-halo Y ratios at the workstream's grid
    ph = []
    for zq in [0.030750465455969357, 0.09967383160646465, 0.29934766321292927, 1.0, 2.0]:
        for mq in [1e13, 3e13, 1e14, 3e14, 6.29e14, 1e15, 2e15]:
            Mp = np.array([mq / H]); zq_ = np.array([zq]); r = R200(Mp, zq_); t = r / D_A_cm(zq_); a = SIGMA_T / MEC2 * P200(Mp, zq_) * r
            Yb = yell(*[np.atleast_1d(p) for p in b12_params(Mp, zq_)], t, a)[1][0]
            Yl = yell(*[np.atleast_1d(p) for p in lee_params(np.array([mq]), zq_)], t, a)[1][0]
            ph.append({"z": zq, "M200c_msunh": mq, "Y_lee22_over_b12": float(Yl / Yb)})
    out["per_halo_Y_ratio"] = ph
    json.dump(out, open(OUT + "/v2_catalogue_poisson.json", "w"), indent=1)
    print("total %.1f s" % (time.time() - t0))

if __name__ == "__main__":
    main()
