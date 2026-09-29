"""Shared helpers for workstream D (CAMELS IllustrisTNG CV profiles vs Lee22 / B12 / B16).

Conventions used everywhere in this workstream (all PHYSICAL unless a name says _code / comoving):
  * CAMELS CV cosmology: Omega_m = 0.3, Omega_b = 0.049, h = 0.6711, Omega_L = 0.7 (sigma8 = 0.8);  f_b = 0.16333.
  * M200c, R200c from SUBFIND (Group_M_Crit200 [1e10 Msun/h], Group_R_Crit200 [ckpc/h]); converted to
    M200c [Msun] = 1e10 M / h and R200c [physical kpc] = R a / h.  Mass bins are in log10 M200c/Msun (no h).
  * rho_crit(z) = 3 H0^2 E^2(z) / (8 pi G) physical,  E^2 = Om (1+z)^3 + OL.
  * Lee22 normalisations: n200 = 200 rho_crit(z) f_b / (X_H m_p), P200 = 200 G M200 rho_crit f_b / (2 R200) (physical R200).
  * X_H = 0.76.  Electron pressure P_e = n_e k_B T (from ElectronAbundance in the direct profiles);
    thermal P_th = (gamma-1) rho u.  For a fully ionised H+He plasma n_e = rho/(mu_e m_p), mu_e = 2/(1+X_H) = 1.1364,
    and P_e/P_th = mu/mu_e = 0.5176 (the XGPaint / Battaglia12 factor).
  * Lee22 fit functions take M in h^-1 Msun with the TNG300 h = 0.6774: M_h = M200c[Msun] * 0.6774.
    Battaglia12/16 take M200c in physical Msun.
"""
import os
import sys

import numpy as np

sys.path.insert(0, "/home/cbllover/HalfDome/frb_map_generation/lee22_normalization_forensics/C_pressure_consistency")
import lee22_models as LM  # noqa: E402  (validated formulas; do not modify)

G = LM.G
M_P = LM.M_P
MSUN = LM.MSUN
KPC = LM.KPC
KEV = LM.KEV
X_H = 0.76
MU_E = 2.0 / (1.0 + X_H)
MU = 4.0 / (3.0 + 5.0 * X_H)
PE_OVER_PTH = MU / MU_E   # 0.5176 for full ionisation
H_TNG = 0.6774

CAMELS = LM.Cosmo("CAMELS CV (TNG model)", 0.3, 0.049, 0.6711)

OUTDIR = "/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_tsz_verification_20260929/D_tng_camels"
PROFDIR = os.path.join(OUTDIR, "camels_profiles")
DIRECTDIR = os.path.join(OUTDIR, "direct")

# old 34-snapshot numbering (precomputed profiles) -> redshift (illstack_CAMELS generate.py red_dict, 5 d.p.)
OLD_SNAP_Z = {0: 6.0, 1: 5.0, 2: 4.0, 3: 3.5, 4: 3.0, 5: 2.81329, 6: 2.63529, 7: 2.46560, 8: 2.30383, 9: 2.14961,
              10: 2.00259, 11: 1.86243, 12: 1.72882, 13: 1.60144, 14: 1.48001, 15: 1.36424, 16: 1.25388, 17: 1.14868,
              18: 1.04838, 19: 0.95276, 20: 0.86161, 21: 0.77471, 22: 0.69187, 23: 0.61290, 24: 0.53761, 25: 0.46584,
              26: 0.39741, 27: 0.33218, 28: 0.27, 29: 0.21072, 30: 0.15420, 31: 0.10033, 32: 0.04896, 33: 0.0}
NEW_SNAPS = [14, 18, 24, 28] + [32 + 2 * k for k in range(30)]   # old i -> NEW_SNAPS[i]; old 33 -> 90


def old_to_new(i):
    return NEW_SNAPS[i]


_ZHDR = None


def snap_z(box, old):
    """exact group-catalogue header redshift (snapshot_redshifts.csv from fetch_snapshot_redshifts.py);
    falls back to the rounded illstack red_dict value"""
    global _ZHDR
    if _ZHDR is None:
        _ZHDR = {}
        fn = os.path.join(OUTDIR, "snapshot_redshifts.csv")
        if os.path.exists(fn):
            for line in open(fn).read().split("\n")[1:]:
                if line.strip():
                    b, o, n, z = line.split(",")[:4]
                    _ZHDR[(int(b), int(o))] = float(z)
    return max(_ZHDR.get((box, old), OLD_SNAP_Z[old]), 0.0)


# target redshifts of the analysis and the old snapshot used for each
TARGETS = {0.0: 33, 0.5: 25, 1.0: 18, 1.5: 14, 2.0: 10}   # z = 0, 0.466, 1.048, 1.480, 2.003
MASS_BINS = [(12.5, 13.0), (13.0, 13.5), (13.5, 14.0), (14.0, 15.0)]   # log10 M200c/Msun
X_QUOTE = [0.1, 0.3, 0.5, 1.0]


def x_grid(n=31, lo=0.03, hi=3.0):
    """log grid in x = r/R200c plus the quote points 0.1, 0.3, 0.5, 1 (and Lee22's 0.04, 1.34 fit limits)"""
    g = np.concatenate([np.logspace(np.log10(lo), np.log10(hi), n), X_QUOTE, [0.04, 1.34]])
    return np.unique(np.round(g, 10))


def load_precomputed(box, old_snap, profdir=PROFDIR, cosmo=CAMELS):
    """Precomputed illstack profiles converted to physical CGS units.

    Returns dict: z, r_kpc (bin centres, physical kpc), edges_kpc, rho [g/cm^3] gas density (all gas),
    Pth [erg/cm^3] thermal pressure (all gas), Tmw [K], ncell, M200 [Msun], R200 [phys kpc], ID.
    Units in the file (illstack_CAMELS profiles_expand.py + CompHaloProperties.py):
      r: bin centres in ckpc/h of 25 log bins [0.1, 1e5] ckpc/h; Profiles[0]: sum Masses / V_shell(ckpc/h)^3;
      Profiles[1]: sum (gamma-1) Masses u / V_shell;  Profiles[3]: mass-weighted T/1e10 K.
    """
    import h5py
    fn = os.path.join(profdir, f"CV_{box}", f"IllustrisTNG_CV_{box}_{old_snap:03d}.hdf5")
    z = snap_z(box, old_snap)
    a = 1.0 / (1.0 + z)
    h = cosmo.h
    with h5py.File(fn, "r") as f:
        P = f["Profiles"][:]
        n = f["n"][:]
        r = f["r"][:]
        M = f["Group_M_Crit200"][:]
        R = f["Group_R_Crit200"][:]
        ID = f["ID"][:]
    dens_fac = 1e10 * MSUN / KPC ** 3 * h ** 2 / a ** 3            # -> g/cm^3 physical
    bs = np.log(1e6) / 25
    edges = 0.1 * np.exp(bs * np.arange(26))
    return dict(z=z, a=a, r_kpc=r * a / h, edges_kpc=edges * a / h,
                rho=P[0] * dens_fac, Pth=P[1] * dens_fac * 1e10, Tmw=P[3] * 1e10,
                ncell=n[0], M200=M * 1e10 / h, R200=R * a / h, ID=ID.astype(int))


def interp_profile(r_cen, prof, R200, xg, ncell=None, min_cells=1):
    """Log-log interpolation of a binned profile (bin centres r_cen) of one halo onto x = r/R200 grid.
    Bins with no cells are dropped; points outside the covered range are NaN."""
    ok = prof > 0
    if ncell is not None:
        ok &= ncell >= min_cells
    if ok.sum() < 2:
        return np.full(len(xg), np.nan)
    lx = np.log(r_cen[ok] / R200)
    ly = np.log(prof[ok])
    out = np.exp(np.interp(np.log(xg), lx, ly, left=np.nan, right=np.nan))
    return out


def ne_n200_full_ion(rho, z, cosmo=CAMELS):
    """n_e/n200 for a fully ionised gas: (rho/(mu_e m_p)) / (200 rho_c f_b/(X_H m_p)) = X_H rho / (mu_e 200 rho_c f_b)."""
    return rho / (MU_E * M_P) / cosmo.n200(z)


def lee_density_variants(x, mh_hinv, z):
    """Dict of Lee22 no-c density variants (n_e/n200): printed, production (x0.6016), x C(z), x (1+z)^3."""
    pr = LM.lee_nbar(x, mh_hinv, z, "noc")
    zz = np.minimum(z, 2.0)
    Cz = 0.2841790493880425 * (1 + zz) ** 2.483190184608004
    return {"printed": pr, "production": 0.6016 * pr, "tngcal": Cz * pr, "a3": (1 + z) ** 3 * pr}


def lee_pressure_variants(x, mh_hinv, z):
    """Dict of Lee22 no-c pressure variants (P_e/P200): printed (= production painted P_e), x (1+z)^2 (M1)."""
    pr = LM.lee_Pbar(x, mh_hinv, z, "noc")
    return {"printed": pr, "a2": (1 + z) ** 2 * pr}


COLORS = dict(data="black", b12="#0072B2", lee_prod="#E69F00", lee_corr="#009E73", lee_print="#D55E00", other="#CC79A7")
