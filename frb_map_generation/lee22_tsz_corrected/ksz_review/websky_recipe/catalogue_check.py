"""Review of paint_ksz_websky.jl (2026-10-02, ksz_review/websky_recipe): catalogue-level checks run on idark.
  (1) the c200m = 7 M200m -> M200c ratio: painter algorithm (bisection on m(x)/x^3, 2001-point z grid, linear interpolation)
      vs an independent NFW solution (numerical mass integral of rho_NFW + brentq on the mean density) vs Hu & Kravtsov (2003);
  (2) HalfDome catalogue ranges: z, M200c, M200m, the WebSky-recipe Mws = M200m x ratio(z) and how many halos fall outside the
      painter's cache mass range [1e12, 10^15.7] Msun (they are silently dropped from the websky maps, line 159);
  (3) WebSky cut statistics (M200c > 1e13 Msun and theta200c > 0.5' radius vs 0.25' i.e. a 0.5' diameter), per redshift.
Usage: python3 catalogue_check.py CATALOG OUTDIR"""
import sys, json, time
import numpy as np, h5py
from scipy.integrate import quad
from scipy.optimize import brentq

CAT, OUT = sys.argv[1], sys.argv[2]
H = 0.6774; OB = 0.0486; OC = 0.2603; OM = OB + OC
C200M = 7.0
MPC = 3.0856775814913673e22; G = 6.67430e-11; MSUN = 1.98847e30; C_KMS = 299792.458
# XGPaint cosmology (Cosmology.jl with radiation from Tcmb = 2.7255, Neff = 3.046), as used by theta_r200c and d_A in the painter
OG = 4.48131e-7 * 2.7255**4 / H**2; OR = OG * (1 + 3.046 * 7 / 8 * (4 / 11)**(4 / 3)); OL = 1 - OM - OR
E2 = lambda z: OM * (1 + z)**3 + OR * (1 + z)**4 + OL
RHOC0 = 3 * (100 * H * 1e3 / MPC)**2 / (8 * np.pi * G)                 # kg/m^3
ZG = np.linspace(0, 6, 600001)
_inv = 1 / np.sqrt(E2(ZG))
CHI = np.concatenate([[0], np.cumsum(0.5 * (_inv[1:] + _inv[:-1]) * np.diff(ZG))]) * C_KMS / (100 * H)   # Mpc
dA = lambda z: np.interp(z, ZG, CHI) / (1 + z)                          # Mpc


# ---------------------------------------------------------------- (1) c200m = 7 conversion
mnfw = lambda x: np.log1p(x) - x / (1 + x)
omz_painter = lambda z: OM * (1 + z)**3 / (OM * (1 + z)**3 + 1 - OM)   # painter: no radiation


def ratio_painter_exact(z):
    target = mnfw(C200M) / (C200M**3 * omz_painter(z)); lo, hi = 1e-3, C200M
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if mnfw(mid) / mid**3 > target: lo = mid
        else: hi = mid
    return mnfw(0.5 * (lo + hi)) / mnfw(C200M)


RZ = np.linspace(0, 5, 2001); RV = np.array([ratio_painter_exact(z) for z in RZ])
ratio_painter = lambda z: np.interp(z, RZ, RV)


def ratio_independent(z, omz=omz_painter):
    """M200c/M200m from rho_NFW(r) = rho_s / ((r/rs)(1+r/rs)^2), rho_s fixed by M(<R200m) = M200m (numerical integral), R200m = 1."""
    rs = 1.0 / C200M
    shape = lambda r: 4 * np.pi * r**2 / ((r / rs) * (1 + r / rs)**2)
    norm = quad(shape, 0, 1.0, epsabs=0, epsrel=1e-13, limit=200)[0]
    mass = lambda r: quad(shape, 0, r, epsabs=0, epsrel=1e-13, limit=200)[0] / norm   # M(<r)/M200m
    # mean density inside r in units of 200 rho_m = mass(r)/r^3; R200c where it equals rho_c/rho_m = 1/Omega_m(z)
    f = lambda r: mass(r) / r**3 - 1.0 / omz(z)
    r200c = brentq(f, 1e-3, 1.0, xtol=1e-15, rtol=1e-15)
    return mass(r200c)


def ratio_hu_kravtsov(z):
    """Hu & Kravtsov (2003) appendix fit: M_h/M_v for f = Delta_h/Delta_v ... used as x(f) -> conversion between overdensities."""
    a1, a2, a3, a4 = 0.5116, -0.4283, -3.13e-3, -3.52e-5
    # f(x) = x^3 [ln(1+1/x) - 1/(1+x)], x = rs/r_v; find r_h/rs for Delta_h = 200 rho_c, Delta_v = 200 rho_m
    # f(rs/r) is proportional to the mean density inside r, so f(rs/r_h) = (Delta_h/Delta_v) f(1/c_v), Delta_h/Delta_v = 1/Omega_m(z)
    fh = (1 / C200M)**3 * (np.log(1 + C200M) - C200M / (1 + C200M)) / omz_painter(z)
    p = a2 + a3 * np.log(fh) + a4 * np.log(fh)**2
    x_of_f = (a1 * fh**(2 * p) + 0.5625)**-0.5 + 2 * fh
    # r_h/rs = 1/x_of_f ; M_h/M_v = (Delta_h/Delta_v) (r_h c / r_v ... ) -> (Delta_h/Delta_v) (c x_of_f)^-3
    return (1.0 / omz_painter(z)) * (C200M * x_of_f)**-3


zt = np.array([0.0, 0.01, 0.1, 0.3, 0.5, 1.0, 1.5, 2.0, 3.0, 3.86])
conv = []
for z in zt:
    conv.append(dict(z=float(z), painter_grid=float(ratio_painter(z)), painter_exact=float(ratio_painter_exact(z)),
                     independent=float(ratio_independent(z)), hu_kravtsov=float(ratio_hu_kravtsov(z)),
                     independent_with_radiation=float(ratio_independent(z, omz=lambda zz: OM * (1 + zz)**3 / E2(zz)))))
zdense = np.linspace(0, 3.9, 400)
dev_grid = max(abs(ratio_painter(z) / ratio_independent(z) - 1) for z in zdense[::10])
print("c200m=7 conversion M200c/M200m:")
for c in conv:
    print("  z=%5.2f painter_grid %.6f painter_exact %.6f independent %.6f (rad. %.6f) HK03 %.6f" % (
        c['z'], c['painter_grid'], c['painter_exact'], c['independent'], c['independent_with_radiation'], c['hu_kravtsov']))
print("  max |painter grid / independent - 1| over 0<z<3.9: %.2e" % dev_grid, flush=True)

# ---------------------------------------------------------------- (2), (3) catalogue pass
LOGM_LO, LOGM_HI = 12.0, 15.7
ARC = np.pi / 180 / 60
zedges = np.arange(0, 4.0001, 0.1)
stats = dict(n=0, zmin=np.inf, zmax=-np.inf, lmc_min=np.inf, lmc_max=-np.inf, lmm_min=np.inf, lmm_max=-np.inf, lws_max=-np.inf,
             n_mc_out=0, n_ws_above=0, n_ws_below_cache=0, n_lowz=0, n_cut=0, n_cut_diam=0, n_mcut_only=0, n_dropped_cut=0)
per_z = {k: np.zeros(len(zedges) - 1) for k in ("n_all", "n_cut", "v2_cut", "n_cut_diam", "v2_cut_diam", "n_m13", "v2_all")}
dropped = []
t0 = time.time()
with h5py.File(CAT, "r") as f:
    N = f["redshift"].shape[0]; CH = 4_000_000
    for s in range(0, N, CH):
        e = min(s + CH, N)
        z = f["redshift"][s:e].astype(np.float64)
        mc = f["halo_mass_m200c"][s:e] / H; mm = f["halo_mass_m200m"][s:e] / H
        pos = f["Position"][s:e].astype(np.float64); vel = f["Velocity"][s:e].astype(np.float64)
        ok = np.isfinite(mc) & (mc > 0) & np.isfinite(mm) & (mm > 0) & np.isfinite(z) & (z > 0) & \
            np.all(np.isfinite(pos), 1) & np.all(np.isfinite(vel), 1)
        z, mc, mm, pos, vel = z[ok], mc[ok], mm[ok], pos[ok], vel[ok]
        vlos = np.sum(pos * vel, 1) / np.sqrt(np.sum(pos**2, 1)) / C_KMS
        mws = mm * ratio_painter(z)
        lmc, lmm, lws = np.log10(mc), np.log10(mm), np.log10(mws)
        rhoc = RHOC0 * E2(z)
        r200w = (3 * mws * MSUN / (4 * np.pi * 200 * rhoc))**(1 / 3) / MPC              # Mpc physical
        th200w = np.arctan(r200w / dA(z))
        mcut = mws > 1e13
        cut = mcut & (th200w > 0.5 * ARC); cutd = mcut & (th200w > 0.25 * ARC)
        incache = (lws >= LOGM_LO) & (lws <= LOGM_HI)
        stats["n"] += len(z); stats["zmin"] = min(stats["zmin"], z.min()); stats["zmax"] = max(stats["zmax"], z.max())
        stats["lmc_min"] = min(stats["lmc_min"], lmc.min()); stats["lmc_max"] = max(stats["lmc_max"], lmc.max())
        stats["lmm_min"] = min(stats["lmm_min"], lmm.min()); stats["lmm_max"] = max(stats["lmm_max"], lmm.max())
        stats["lws_max"] = max(stats["lws_max"], lws.max())
        stats["n_mc_out"] += int(np.sum((lmc < LOGM_LO) | (lmc > LOGM_HI)))
        stats["n_ws_above"] += int(np.sum(lws > LOGM_HI)); stats["n_ws_below_cache"] += int(np.sum(lws < LOGM_LO))
        stats["n_lowz"] += int(np.sum(z < 0.1)); stats["n_cut"] += int(np.sum(cut)); stats["n_cut_diam"] += int(np.sum(cutd))
        stats["n_mcut_only"] += int(np.sum(mcut)); stats["n_dropped_cut"] += int(np.sum(cut & ~incache))
        for i in np.nonzero(cut & ~incache)[0]:
            dropped.append(dict(z=float(z[i]), log10_m200c=float(lmc[i]), log10_m200m=float(lmm[i]), log10_mws=float(lws[i]),
                                vlos_over_c=float(vlos[i])))
        iz = np.clip(np.searchsorted(zedges, z, side="right") - 1, 0, len(zedges) - 2)
        per_z["n_all"] += np.bincount(iz, minlength=len(zedges) - 1); per_z["v2_all"] += np.bincount(iz, vlos**2, len(zedges) - 1)
        per_z["n_cut"] += np.bincount(iz[cut], minlength=len(zedges) - 1); per_z["v2_cut"] += np.bincount(iz[cut], vlos[cut]**2, len(zedges) - 1)
        per_z["n_cut_diam"] += np.bincount(iz[cutd], minlength=len(zedges) - 1)
        per_z["v2_cut_diam"] += np.bincount(iz[cutd], vlos[cutd]**2, len(zedges) - 1)
        per_z["n_m13"] += np.bincount(iz[mcut], minlength=len(zedges) - 1)
        print("rows %d-%d elapsed %.0f s" % (s, e, time.time() - t0), flush=True)
stats = {k: (float(v) if isinstance(v, (np.floating, float)) else int(v)) for k, v in stats.items()}
print(json.dumps(stats, indent=1))
print("halos passing the WebSky cut but outside the cache mass range (dropped from the websky maps):", len(dropped))
for d in sorted(dropped, key=lambda d: -d["log10_mws"])[:20]:
    print("  ", d)
print("per-z (z_lo, N_all, N_M200c>1e13, N_cut(0.5' radius), N_cut(0.25'), sum v^2 cut):")
for j in range(len(zedges) - 1):
    if per_z["n_all"][j] > 0:
        print("  %.1f %10d %9d %9d %9d %.4e" % (zedges[j], per_z["n_all"][j], per_z["n_m13"][j], per_z["n_cut"][j],
                                               per_z["n_cut_diam"][j], per_z["v2_cut"][j]))
np.savez(OUT + "/catalogue_check.npz", zedges=zedges, **per_z)
json.dump(dict(conversion=conv, max_dev_grid_vs_independent=float(dev_grid), stats=stats, dropped=dropped),
          open(OUT + "/catalogue_check.json", "w"), indent=1)
print("done %.0f s" % (time.time() - t0))
