"""Shared helpers for workstream C (FLAMINGO L2p8_m9 pressure / density against B12, B16 and Lee22).

Conventions (every comparison in this directory):
  * x = r / R200c with R200c PHYSICAL (SOAP SO/200_crit/SORadius, physical Mpc); M200c = SOAP SO/200_crit/TotalMass in
    physical Msun. 200 rho_cr(z) = M200c / (4/3 pi R200c^3) exactly (physical rho_cr of FLAMINGO's cosmology).
  * FLAMINGO cosmology: h = 0.681, Omega_b = 0.0486, Omega_m = 0.306 (incl. massive neutrinos), f_b = Omega_b/Omega_m = 0.15882.
  * P200 = G M200c (200 rho_cr) f_b / (2 R200c)   (Battaglia12 / XGPaint / Lee22 eq. 9; thermal-pressure normalisation).
  * FLAMINGO P_e = n_e k_B T from SWIFT's particle ComptonYParameters (y_i = sigma_T/(m_e c^2) n_e k_B T m_i/rho_i, checked
    to 5e-5 per halo), n_e from SWIFT's cooling tables (ionisation + metals), X_H = 0.752 primordial. All gas within the
    halo's R100c ('raw'); the 'noagn' variant drops SOAP's recently-AGN-heated particles.
  * n200 = 200 rho_cr f_b / (X_H m_p) with X_H = 0.76 (Lee22's definition and value; the FLAMINGO sums are divided by the
    same n200 so n_e/n200 is a pure number in Lee22's units).
  * Lee22 masses: h^-1 Msun, converted as M_h = M200c[Msun] * h_sim (h_sim = 0.681; production paints with h = 0.68).
  * Models: B12 P_e = 0.5176 P_th (XGPaint); Lee22 no-c Table 7 (pressure) / Table 8 (density) with Mcut pivots and the
    shape clip at 10^14.8 h^-1 Msun as in production (P0/n0 mass slopes extrapolated above it).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

FPC = Path("/home/cbllover/HalfDome/flamingo_pressure_calibration")
LEE = Path("/home/cbllover/HalfDome/frb_map_generation/lee22_normalization_forensics/C_pressure_consistency")
for p in (FPC, LEE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import flamingo_reduced_io as fio  # noqa: E402
import flamingo_stacks as fs  # noqa: E402
import lee22_models as L  # noqa: E402

OUT = Path("/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_tsz_verification_20260929/C_flamingo")
STACKS = FPC / "notebook_cache" / "stacks"
SNAPSHOTS = [78, 68, 58, 48, 38, 18, 14]          # z = 0, 0.5, 1, 1.5, 2, 3, 4
H_SIM = 0.681
FB_SIM = 0.0486 / 0.30600007943162144

# colours (task convention)
BLACK, BLUE, ORANGE, GREEN, VERMILLION, PINK = "black", "#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7"

# FLAMINGO joint nine-coefficient fit painted on HalfDome (halfdome_tsz_comparison/flamingo_joint_fit.toml)
FLAMINGO_FIT = np.array([4.255329379619701, 1.3377910707221534, 6.029207007048246,
                         0.6398160485222005, -0.16438043008778933, 0.05281626430871426,
                         0.5684515143587939, 0.07623000158582573, 0.30633510789892787])

# TNG-calibrated density correction (lee2022_frb_dm_profile.jl redshift_scaling = :tng_calibrated)
C_A, C_K = 0.2841790493880425, 2.483190184608004
PRODUCTION_NE2D = 0.6016   # production normalization = :xgpaint_ne2d is exactly 0.6016 x the printed Lee22 density


def c_tng(z):
    return C_A * (1 + np.minimum(np.asarray(z, float), 2.0)) ** C_K


# ------------------------------------------------------------------ pressure models, Pe/P200 (M in physical Msun)
def pe_b12(x, m, z):
    return L.b12_Pe_over_P200(x, m, z)


def pe_lee(x, m, z):
    return L.lee_Pbar(x, m * H_SIM, z)


def pe_lee_a2(x, m, z):
    return L.lee_Pbar(x, m * H_SIM, z) * (1 + np.asarray(z, float)) ** 2


def pe_flfit(x, m, z):
    p0, xc, beta = fs.coefficients(FLAMINGO_FIT, m, z)
    return fs.gnfw(x, p0, xc, beta)


PRESSURE_MODELS = {
    "B12": (pe_b12, BLUE, "-", "Battaglia+12 (P$_e$ = 0.5176 P$_{th}$)"),
    "Lee22": (pe_lee, ORANGE, "-", "Lee22 no-c, as painted"),
    "Lee22_a2": (pe_lee_a2, GREEN, "--", r"Lee22 $\times(1+z)^2$ (M1)"),
    "FLfit": (pe_flfit, PINK, ":", "FLAMINGO 9-coeff. fit (painted on HalfDome)"),
}


# ------------------------------------------------------------------ density models, n_e/n200 (Lee22 units, X_H = 0.76)
def ne_lee_literal(x, m, z):
    return L.lee_nbar(x, m * H_SIM, z)


def ne_lee_prod(x, m, z):
    return PRODUCTION_NE2D * L.lee_nbar(x, m * H_SIM, z)


def ne_lee_tng(x, m, z):
    return c_tng(z) * L.lee_nbar(x, m * H_SIM, z)


def ne_lee_a3(x, m, z):
    return L.lee_nbar(x, m * H_SIM, z) * (1 + np.asarray(z, float)) ** 3


def ne_b16(x, m, z):
    return L.b16_nbar(x, m, z)


DENSITY_MODELS = {
    "B16": (ne_b16, BLUE, "-", "Battaglia+16 (XGPaint electrons)"),
    "Lee22_literal": (ne_lee_literal, VERMILLION, ":", "Lee22 no-c, printed (literal)"),
    "Lee22_prod": (ne_lee_prod, ORANGE, "-", r"Lee22 $\times$0.6016 (production DM)"),
    "Lee22_tngcal": (ne_lee_tng, GREEN, "--", r"Lee22 $\times$C(z) (TNG-calibrated)"),
}


def shell_model_mean(func, masses, z, inner, outer, weight, order=24):
    """Equal-weight halo mean of the exactly shell-averaged model (same haloes and truncated shells as the data).

    inner (n_shells,), outer (n_haloes, n_shells) effective outer edges, weight (n_haloes, n_shells) 0/1 completeness.
    """
    inner = np.broadcast_to(np.asarray(inner, float)[None, :], outer.shape)
    safe_outer = np.where(np.isfinite(outer), outer, inner * 1.1 + 1e-6)
    radius, vw = fs.shell_nodes(inner, safe_outer, order)
    vals = np.sum(vw * func(radius, masses[:, None, None], z), axis=-1)
    w = np.asarray(weight, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(w.sum(0) > 0, (w * np.nan_to_num(vals)).sum(0) / w.sum(0), np.nan)


def interp_log(xq, x, y, ok):
    """log-log interpolation of y(x) over the entries flagged ok; NaN outside their span."""
    x, y, ok = np.asarray(x), np.asarray(y), np.asarray(ok) & np.isfinite(y) & (np.asarray(y) > 0)
    if ok.sum() < 2:
        return np.full(np.shape(xq), np.nan)
    lx, ly = np.log(x[ok]), np.log(y[ok])
    lq = np.log(np.atleast_1d(xq))
    out = np.interp(lq, lx, ly)
    out[(lq < lx.min() - 1e-9) | (lq > lx.max() + 1e-9)] = np.nan
    return np.exp(out)


def slope_fit(lz, ly, sigma=None):
    """Weighted least squares ly = a + s lz; returns s, sigma_s (from sigma if given, else from scatter)."""
    lz, ly = np.asarray(lz, float), np.asarray(ly, float)
    ok = np.isfinite(ly)
    if sigma is not None:
        ok &= np.isfinite(sigma) & (np.asarray(sigma) > 0)
    if ok.sum() < 3:
        return np.nan, np.nan
    w = np.ones(ok.sum()) if sigma is None else 1 / np.asarray(sigma)[ok] ** 2
    A = np.vstack([np.ones(ok.sum()), lz[ok]]).T
    cov = np.linalg.inv(A.T @ (A * w[:, None]))
    coef = cov @ (A.T @ (w * ly[ok]))
    if sigma is None:
        resid = ly[ok] - A @ coef
        dof = max(ok.sum() - 2, 1)
        s_err = np.sqrt(cov[1, 1] * np.sum(resid**2) / dof)
    else:
        resid = ly[ok] - A @ coef
        chi2 = np.sum(w * resid**2) / max(ok.sum() - 2, 1)
        s_err = np.sqrt(cov[1, 1] * max(chi2, 1.0))       # inflate by reduced chi2 when > 1
    return float(coef[1]), float(s_err)
