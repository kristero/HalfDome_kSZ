"""Shared settings for workstream B (CLASS-SZ verification of the Battaglia12 and Lee22 tSZ).

Conventions (explicit, because these are what went wrong in Lee22):
  * Masses: M200c. The HalfDome catalogue stores halo_mass_m200c in Msun/h; the painter divides by h = 0.68
    (paint cosmology) to get physical Msun. CLASS-SZ integrates over M200c in Msun/h, so with h = 0.68 the
    catalogue numbers are used unchanged as CLASS-SZ M_min/M_max.
  * Lee22 "h^-1 Msun" masses (M_cut = 10^13.60, shape clip 10^14.8) are converted with the PAINT h = 0.68,
    exactly as lee2022_tsz_pressure_profile.jl does (M_cut = 5.8545e13 Msun, clip = 9.2788e14 Msun).
  * R200c, rho_crit(z) physical; P200 = 200 G M200 rho_cr(z) f_b / (2 R200), f_b = Omega_b/Omega_m.
  * Battaglia12 fits the THERMAL pressure: P_e = 0.5176 P_th (XGPaint and CLASS-SZ both apply 0.5176;
    CLASS-SZ hides it in the constant 2.61051e-18 of P_200, see profile_checks.py).
    Lee22 fits the ELECTRON pressure P_e/P200 directly, so it is fed to both codes as P0 = P0_Lee/0.5176.
  * Compton-y: y = sigma_T/(m_e c^2) int P_e dl (physical length). Spectra are C_ell of dimensionless y.
"""
import numpy as np

# ------------------------------------------------------------------ paths
REPO = "/home/cbllover/HalfDome"
FG = REPO + "/frb_map_generation"
CODE = FG + "/lee22_tsz_verification/B_class_sz"
OUT = FG + "/outputs/lee22_tsz_verification_20260929/B_class_sz"
CATALOGUE = REPO + "/lightcone_100.hdf5"
FULLSKY_SPECTRA = FG + "/outputs/tsz_dm_fullsky_20260918/spectra/fullsky_spectra.npz"
SPLIT_SPECTRA = FG + "/outputs/lee22_normalization_forensics_20260928/verify_5/tsz_split_cl.npz"
TRUNC_CLASS_SZ = REPO + "/truncation_comparison/radius_series/class_sz"
LEE22_MODELS_DIR = FG + "/lee22_normalization_forensics/C_pressure_consistency"

# ------------------------------------------------------------------ cosmologies
# paint cosmology of the production y maps (provenance: Omega_b=0.049, Omega_c=0.261, h=0.68);
# sigma8 and n_s: published HalfDome values (arXiv:2407.17462), as in szhm.PAINT_COSMO
PAINT = dict(name="paint", h=0.68, Omega_b=0.049, Omega_c=0.261, sigma8=0.8159, n_s=0.9667)
# published HalfDome cosmology (= TNG/Planck15), used by the 2026-09-25 truncation-series CLASS-SZ runs
PUBLISHED = dict(name="published", h=0.6774, Omega_b=0.0486, Omega_c=0.2603, sigma8=0.8159, n_s=0.9667)

# ------------------------------------------------------------------ catalogue bounds (halo_mass_m200c, Msun/h)
CAT_M_MIN = 4458834966504.787
CAT_M_MAX = 2573960654045765.0
CAT_Z_MIN = 0.002141595
CAT_Z_MAX = 3.8554277

# ------------------------------------------------------------------ Lee22 constants (h^-1 Msun, converted with h_paint)
LEE_P_MCUT_HINV = 10 ** 13.60
LEE_CLIP_HINV = 10 ** 14.8
LEE_CALIB_MMIN_HINV = 1e13
LEE_CALIB_ZMAX = 2.0
PE_OVER_PTH = 0.5176


def class_sz_base(cosmo, x_out=4.0, z_min=CAT_Z_MIN, z_max=CAT_Z_MAX, m_min=CAT_M_MIN, m_max=CAT_M_MAX,
                  output="tSZ_1h,tSZ_2h", massless_nu=True, damping_1h=0):
    """Settings of the 2026-09-25 truncation series (compute_class_sz_radius_series.py), cosmology selectable.

    massless_nu=True OMITS "cosmo_model": classy_sz's Class.set() (classy.pyx, v0.1.70.post40) overwrites
    N_ur=2.0328, N_ncdm=1, m_ncdm=0.06 eV whenever cosmo_model is passed, so the earlier runs that passed
    cosmo_model=0 together with N_ncdm=0 actually had a 0.06 eV neutrino (Omega_m 0.3114 instead of 0.31).
    damping_1h=0 switches off CLASS-SZ's default 1-halo large-scale damping 1-exp(-(k/0.01 Mpc^-1)^2), which the
    painted maps do not have (it matters only at ell < ~50)."""
    h = cosmo["h"]
    base = {"omega_b": cosmo["Omega_b"] * h ** 2, "omega_cdm": cosmo["Omega_c"] * h ** 2, "h": h,
            "n_s": cosmo["n_s"], "sigma8": cosmo["sigma8"], "output": output, "damping_1h_term": int(damping_1h)}
    if massless_nu:
        base.update({"N_ncdm": 0, "N_ur": 3.046})
    else:
        base.update({"N_ncdm": 0, "N_ur": 3.046, "cosmo_model": 0})   # exactly the 2026-09-25 dictionary
    return dict(base, **{
            "ell_min": 10.0, "ell_max": 8192.0, "dell": 0.0, "dlogell": 0.05,
            "z_min": float(z_min), "z_max": float(z_max), "M_min": float(m_min), "M_max": float(m_max),
            "mass_function": "T08M200c", "pressure_profile": "B12", "concentration_parameter": "D08",
            "x_outSZ": float(x_out), "ndim_masses": 200, "ndim_redshifts": 100,
            "n_m_pressure_profile": 100, "n_z_pressure_profile": 100, "N_samp_fftw": 4096,
            "use_fft_for_profiles_transform": 1,
            "redshift_epsabs": 1e-40, "redshift_epsrel": 1e-3, "mass_epsabs": 1e-40, "mass_epsrel": 1e-3,
            "hm_consistency": 0})


def b12_class_sz_pars():
    """XGPaint / Battaglia12 AGN defaults (these are also CLASS-SZ's defaults; passed explicitly)."""
    return {"P0_B12": 18.1, "alpha_m_P0_B12": 0.154, "alpha_z_P0_B12": -0.758,
            "xc_B12": 0.497, "alpha_m_xc_B12": -0.00865, "alpha_z_xc_B12": 0.731,
            "beta_B12": 4.35, "alpha_m_beta_B12": 0.0393, "alpha_z_beta_B12": 0.415,
            "alpha_B12": 1.0, "gamma_B12": -0.3, "use_broken_pressure": 0}


def lee22_class_sz_pars(h, piece, pressure_z_extra=0.0):
    """Lee22 Table 7 no-c electron-pressure fit mapped exactly onto CLASS-SZ's generalized B12 parameters.

    CLASS-SZ (class_sz_tools.c tabulate_gas_pressure_profile_B12_fft, v0.1.70.post40):
        P/P200 = P0 (x/xc)^gamma [1+(x/xc)^alpha]^(-beta),  A = A_B12 (M/1e14 Msun)^alpha_m (1+z)^alpha_z
        (M physical Msun = m[Msun/h]/h), times (M/M_break)^alpha_break for M < M_break if use_broken_pressure
        (M_break_pressure given in Msun/h). y uses 0.5176 P200 (electron) times this fit.
    Lee22 (as painted, lee2022_tsz_pressure_profile.jl):
        P_e/P200 = P0 (x/xc)^-0.3 (1+x/xc)^-beta,  P0 = 2.8 (M/Mcut)^{1.08 | 0.80} (1+z)^-1.89,
        xc = 2.10 (Ms/Mcut)^-0.35 (1+z)^0.53,  beta = 9.4 (Ms/Mcut)^-0.06 (1+z)^0.60,  Ms = min(M, clip)
    piece = "low"  (M < clip):  shape parameters keep their mass slopes
    piece = "high" (M >= clip): shape parameters frozen at the clip mass (alpha_m = 0)
    P0 uses the break at Mcut through use_broken_pressure (slope 0.80 above, 0.80 + 0.28 = 1.08 below), and is
    divided by 0.5176 so that CLASS-SZ's 0.5176 P200 x fit = Lee22's P_e.
    pressure_z_extra adds to the (1+z) exponent of P0 (M1 bookkeeping hypothesis: +2).
    """
    mcut = LEE_P_MCUT_HINV / h          # physical Msun
    clip = LEE_CLIP_HINV / h
    r14 = 1e14 / mcut                   # (1e14 Msun / Mcut)
    pars = {"P0_B12": 2.8 * r14 ** 0.80 / PE_OVER_PTH, "alpha_m_P0_B12": 0.80, "alpha_z_P0_B12": -1.89 + pressure_z_extra,
            "alpha_B12": 1.0, "gamma_B12": -0.3,
            "use_broken_pressure": 1, "M_break_pressure": LEE_P_MCUT_HINV, "alpha_break_pressure": 1.08 - 0.80}
    if piece == "low":
        pars.update({"xc_B12": 2.10 * r14 ** -0.35, "alpha_m_xc_B12": -0.35, "alpha_z_xc_B12": 0.53,
                     "beta_B12": 9.4 * r14 ** -0.06, "alpha_m_beta_B12": -0.06, "alpha_z_beta_B12": 0.60})
    elif piece == "high":
        rc = clip / mcut
        pars.update({"xc_B12": 2.10 * rc ** -0.35, "alpha_m_xc_B12": 0.0, "alpha_z_xc_B12": 0.53,
                     "beta_B12": 9.4 * rc ** -0.06, "alpha_m_beta_B12": 0.0, "alpha_z_beta_B12": 0.60})
    else:
        raise ValueError(piece)
    return pars


def band(ell, cl, centre, frac=0.05):
    """Mean of cl over |ell - centre| <= frac*centre (integer ell arrays)."""
    s = (ell >= centre * (1 - frac)) & (ell <= centre * (1 + frac))
    return float(np.mean(cl[s]))
