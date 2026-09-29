"""Per-halo profile processing shared by the precomputed (illstack, coarse ckpc/h bins) and direct (fine x bins)
CAMELS profiles: bin-average -> point-value correction, interpolation onto a common x = r/R200c grid, enclosed
integrals inside R200c, and model evaluation stacked exactly like the data."""
import numpy as np

import camels_common as cc
LM = cc.LM


def binavg_to_point_factor(s, q):
    """<y>_shell / y(r_c) for y ~ r^s in a shell [r1, q r1], r_c = sqrt(r1 r2) (volume average)."""
    s = np.asarray(s, float)
    out = np.empty_like(s)
    near = np.abs(s + 3) < 1e-6
    ss = s[~near]
    out[~near] = 3 * (q ** (ss + 3) - 1) / ((ss + 3) * (q ** 3 - 1)) * q ** (-ss / 2)
    out[near] = 3 * np.log(q) / (q ** 3 - 1) * q ** 1.5
    return out


def local_slopes(r_cen, yavg, ok):
    """local logarithmic slope of a binned profile from neighbouring populated bins (clipped to [-4.5, 1])"""
    y = np.where(ok, yavg, np.nan)
    ly = np.log(y)
    lr = np.log(r_cen)
    s = np.full(len(y), np.nan)
    for i in range(len(y)):
        if not ok[i]:
            continue
        lo = i - 1 if i - 1 >= 0 and ok[i - 1] else i
        hi = i + 1 if i + 1 < len(y) and ok[i + 1] else i
        if hi > lo:
            s[i] = (ly[hi] - ly[lo]) / (lr[hi] - lr[lo])
    return np.clip(np.nan_to_num(s, nan=-2.0), -4.5, 1.0)


def point_values(r_cen, yavg, q, ok):
    """Convert shell averages (log-spaced bins, ratio q) to point values at the geometric centres using the
    local logarithmic slope from neighbouring populated bins."""
    y = np.where(ok, yavg, np.nan)
    ly = np.log(y)
    lr = np.log(r_cen)
    s = np.full(len(y), np.nan)
    for i in range(len(y)):
        if not ok[i]:
            continue
        lo = i - 1 if i - 1 >= 0 and ok[i - 1] else i
        hi = i + 1 if i + 1 < len(y) and ok[i + 1] else i
        if hi > lo:
            s[i] = (ly[hi] - ly[lo]) / (lr[hi] - lr[lo])
    s = np.clip(np.nan_to_num(s, nan=-2.0), -4.5, 1.0)
    return y / binavg_to_point_factor(s, q)


def to_grid(x_cen, y_point, xg):
    ok = np.isfinite(y_point) & (y_point > 0)
    if ok.sum() < 2:
        return np.full(len(xg), np.nan)
    return np.exp(np.interp(np.log(xg), np.log(x_cen[ok]), np.log(y_point[ok]), left=np.nan, right=np.nan))


def enclosed(edges_x, yavg, xmax=1.0, ok=None):
    """3 * integral_0^xmax y x^2 dx from shell averages: full shells inside xmax plus the straddling shell's
    share assuming y ~ x^s inside that shell, s = local slope from the neighbouring bins
    (a uniform-in-volume share biases f_gas and Y low by ~7 % for the 0.24 dex illstack bins)."""
    x1, x2 = edges_x[:-1], edges_x[1:]
    y = np.where(np.isfinite(yavg), yavg, 0.0)
    full = x2 <= xmax
    tot = np.sum(y[full] * (x2[full] ** 3 - x1[full] ** 3))
    k = np.where((x1 < xmax) & (x2 > xmax))[0]
    if len(k):
        k = k[0]
        if ok is None:
            ok = y > 0
        s = local_slopes(np.sqrt(x1 * x2), y, ok)[k]
        e = s + 3
        frac = (xmax ** e - x1[k] ** e) / (x2[k] ** e - x1[k] ** e) if abs(e) > 1e-6 else \
            np.log(xmax / x1[k]) / np.log(x2[k] / x1[k])
        tot += y[k] * (x2[k] ** 3 - x1[k] ** 3) * frac
    return tot


def model_curves(xg, M200_msun, z, cosmo=cc.CAMELS):
    """Model profiles on the grid for one halo (M200c in physical Msun). Returns dict of arrays."""
    mh = M200_msun * cc.H_TNG
    d = {}
    dn = cc.lee_density_variants(xg, mh, z)
    dp = cc.lee_pressure_variants(xg, mh, z)
    for k, v in dn.items():
        d["n_lee_" + k] = v
    for k, v in dp.items():
        d["P_lee_" + k] = v
    d["P_b12"] = LM.b12_Pe_over_P200(xg, M200_msun, z)
    d["rho_b16"] = LM.b16_rho_over_fb_rhocrit(xg, M200_msun, z)      # rho_gas / (f_b rho_crit)
    d["n_b16_xgp"] = LM.b16_nbar(xg, M200_msun, z)                     # XGPaint ne2d (0.9 x rho/m_per_e) / n200
    return d


def model_integrals(M200_msun, z, nx=2000):
    """3 int_0^1 (model) x^2 dx for the Lee22 variants, B12 and B16 (dimensionless)."""
    x = np.logspace(-5, 0, nx)
    d = model_curves(x, M200_msun, z)
    return {k: 3 * np.trapz(v * x ** 2, x) for k, v in d.items()}
