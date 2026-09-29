#!/usr/bin/env python3
"""Part (3): the free-electron content of TNG300 halos implied by Konietzka's within-R200 halo DM.

Key identity (no profile shape needed): for gas counted only inside each halo's R200c sphere, the mean
halo DM over ALL rays for sources at z_s is

    <DM>(z_s) = int_0^{z_s} dz c/H(z) (1+z) int dlnM dn/dlnM(M, z) N_e(M, z)          [comoving dn/dlnM]

(the chord/disc average of a halo's DM is N_e / (pi R200c^2 (1+z)); the solid angle pi theta200^2 and the
chi^2 of the volume element cancel against D_A^2). With N_e = f_e(M, z) f_b M (1+X_H)/(2 m_p) this gives
the HMF- and mass-weighted mean free-electron fraction f_e in every redshift shell directly from the
increments of Konietzka's cumulative per-ray DM (the rays are the same for every z_s; DM is monotone).

Also computed on the same (M, z) quadrature: the hit statistic Nbar = sum dN/dOmega pi theta200^2
(checks halo abundance x cross-section against TNG's hit fraction), and the per-shell DM predicted by
Battaglia16 (XGPaint) and by every Lee22 reading, so the redshift dependence of each model can be tested
against TNG's own electrons shell by shell.

Mass window 1e13-1e14 Msun (Konietzka's labels, taken as physical M200c; the hit fractions support that).
"""
import json
import sys
from pathlib import Path

import numpy as np

import gasfrac_lib as L

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analytic_pdf_torkamani2026"))
import halo_model_dm_pdf as H  # noqa: E402

OUT = ROOT / "outputs" / "lee22_normalization_forensics_20260928" / "D_gas_fractions"
KON = ROOT.parent / "frb_catalog_comparison_outputs" / "haloDM-20260819T104507Z-1-001" / "haloDM"

COSMO = H.PLANCK15_TNG
FB = COSMO.omega_b / COSMO.omega_m
MPC_CM, PC_CM, MSUN_G, MP_G = H.MPC_CM, H.PC_CM, H.MSUN_G, H.M_P_G
E_PER_MSUN = MSUN_G * (1 + L.X_H) / (2 * MP_G)     # free electrons per Msun of fully ionized gas


def shape_moment_vec(xc, beta, X=1.0, n=8000):
    """int_0^X g x^2 dx for arrays xc, beta (trapezoid in ln x from 1e-8 X; validated against quad)."""
    u = np.linspace(np.log(1e-8 * X), np.log(X), n)
    x = np.exp(u)[None, :]
    s = x / xc[:, None]
    f = s ** -0.3 * (1 + s) ** (-beta[:, None]) * x ** 3
    return H._trapz(f, u, axis=1)


def lee22_fe(log10m_msun, z, fit):
    lmh = log10m_msun + np.log10(COSMO.h)
    if fit == "noc":
        n0, xc, beta = L.lee22_noc_params(lmh, z)
    else:
        n0, xc, beta = L.lee22_c_params(lmh, z)
    n0, xc, beta = (np.broadcast_to(np.asarray(a, float), lmh.shape) for a in (n0, xc, beta))
    return L.F_E_PER_NBAR * 3 * n0 * shape_moment_vec(np.array(xc), np.array(beta))


def check_moment():
    xc = np.array([0.3, 2.0, 7.9, 12.0]); beta = np.array([3.0, 10.0, 19.5, 30.0])
    a = shape_moment_vec(xc, beta)
    b = np.array([L.shape_moment(c, bb, 1.0) for c, bb in zip(xc, beta)])
    err = float(np.max(np.abs(a / b - 1)))
    print('shape_moment_vec vs quad: max rel diff', err)
    assert err < 1e-5
    return err


def main():
    moment_err = check_moment()
    zs_grid = np.load(KON / "redshifts_DMhalo.npy")
    arr = np.load(KON / "DMhalo_r200_mass13to14.npy")          # (55, 120000), cumulative per ray
    assert np.all(np.diff(arr, axis=0) >= 0)
    tng_mean = arr.mean(axis=1)
    tng_hit = (arr > 0).mean(axis=1)
    nrays = arr.shape[1]
    # bootstrap errors of the shell increments (rays are independent sightlines)
    rng = np.random.default_rng(1)
    boots = []
    for _ in range(200):
        idx = rng.integers(0, nrays, nrays)
        boots.append(arr[:, idx].mean(axis=1))
    boots = np.array(boots)
    kmax = int(np.searchsorted(zs_grid, 3.0 + 1e-9))           # shells up to z = 3
    edges = np.concatenate([[0.0], zs_grid[:kmax]])

    # quadrature
    dlm = 0.01
    lm = np.arange(13.0, 14.0, dlm) + dlm / 2
    M = 10 ** lm
    b16_fe_grid = {}
    rows = []
    cum = {k: 0.0 for k in ("unit", "b16", "noc_literal", "c_literal", "noc_cmv", "nbar", "mass_density")}
    variants = {
        "literal": lambda z: 1.0,
        "xgpnative": lambda z: L.XGP_NATIVE_FACTOR,
        "XH2": lambda z: L.X_H ** 2,
        "comoving_hyp": lambda z: (1 + z) ** 3 / (COSMO.E(z) ** 2),
        "(1+z)^3": lambda z: (1 + z) ** 3,
        "(1+z)^3*h^3*XH^2": lambda z: (1 + z) ** 3 * COSMO.h ** 3 * L.X_H ** 2,
        "(1+z)^3*XH^2*fb": lambda z: (1 + z) ** 3 * L.X_H ** 2 * FB,
    }
    shell_out = []
    for k in range(kmax):
        z0, z1 = edges[k], edges[k + 1]
        nsub = 20
        zz = np.linspace(z0, z1, nsub + 1)
        zm = 0.5 * (zz[1:] + zz[:-1])
        dz = np.diff(zz)
        acc = dict(unit=0.0, b16=0.0, nbar=0.0, rho=0.0)
        accL = {(fit, v): 0.0 for fit in ("noc", "c") for v in variants}
        for z, d in zip(zm, dz):
            dn = H.tinker08_dndlnm(COSMO, M, z) * np.log(10) * dlm            # comoving Mpc^-3 per cell
            path = (1 / COSMO.H_over_c(z)) * (1 + z) * d                       # Mpc (comoving-DM kernel)
            # DM [pc cm^-3] per unit f_e: electrons per comoving Mpc^3 x Mpc path -> cm^-3 x cm / pc
            w_unit = dn * FB * M * E_PER_MSUN / MPC_CM ** 3 * path * MPC_CM / PC_CM
            acc["unit"] += w_unit.sum()
            acc["rho"] += (dn * M).sum() * d
            fe_b16 = np.array([L.b16_free_electron_fraction(m, z) for m in M[::10]])
            fe_b16 = np.interp(lm, lm[::10], fe_b16)
            acc["b16"] += (w_unit * fe_b16).sum()
            for fit in ("noc", "c"):
                fe = lee22_fe(lm, np.full_like(lm, z), fit)
                for v, fac in variants.items():
                    accL[(fit, v)] += (w_unit * fe * fac(z)).sum()
            r = H.r200c_mpc(COSMO, M, z)
            chi = COSMO.chi(z)[0]
            th = np.arctan(r / (chi / (1 + z)))
            acc["nbar"] += (chi ** 2 / COSMO.H_over_c(z) * d * dn * np.pi * th ** 2).sum()
        tng_prev = 0.0 if k == 0 else tng_mean[k - 1]
        dtng = tng_mean[k] - tng_prev
        bprev = 0.0 if k == 0 else boots[:, k - 1]
        derr = float(np.std(boots[:, k] - bprev))
        cum["unit"] += acc["unit"]
        cum["b16"] += acc["b16"]
        cum["nbar"] += acc["nbar"]
        sh = dict(z_lo=float(z0), z_hi=float(z1), tng_dDM=float(dtng), tng_dDM_err=derr,
                  model_dDM_fe1=acc["unit"], fe_tng_implied=float(dtng / acc["unit"]),
                  fe_tng_implied_err=derr / acc["unit"],
                  b16_dDM=acc["b16"], fe_b16=acc["b16"] / acc["unit"],
                  tng_over_b16=float(dtng / acc["b16"]),
                  nbar_cum_analytic=cum["nbar"], nbar_cum_tng=float(-np.log(1 - tng_hit[k])),
                  tng_mean_cum=float(tng_mean[k]), model_cum_fe1=cum["unit"], b16_cum=cum["b16"])
        for (fit, v), val in accL.items():
            sh[f"lee22_{fit}_{v}_dDM"] = val
            sh[f"lee22_{fit}_{v}_fe"] = val / acc["unit"]
            sh[f"lee22_{fit}_{v}_over_tng"] = val / dtng
            cum_key = f"cum_{fit}_{v}"
            cum[cum_key] = cum.get(cum_key, 0.0) + val
            sh[f"lee22_{fit}_{v}_cum"] = cum[cum_key]
        shell_out.append(sh)
        print("z {:.1f}-{:.1f}: TNG dDM {:6.2f}+-{:.2f}  f_e(TNG) {:.3f}  f_e(B16) {:.3f}  f_e Lee22 noc lit {:.3f} "
              "xgp {:.3f} (1+z)^3 {:.3f} | Nbar ana {:.4f} tng {:.4f}".format(
                  z0, z1, dtng, derr, dtng / acc["unit"], acc["b16"] / acc["unit"],
                  accL[("noc", "literal")] / acc["unit"], accL[("noc", "xgpnative")] / acc["unit"],
                  accL[("noc", "(1+z)^3")] / acc["unit"], cum["nbar"], -np.log(1 - tng_hit[k])), flush=True)
    # halo counts in the TNG300 box (for the Pillepich+18b count check)
    vol = (205.0 / COSMO.h) ** 3
    counts = {}
    for z in (0.0, 1.0):
        for lo, hi in ((13, 14), (14, 16), (13, 16)):
            lmq = np.arange(lo, hi, 0.001) + 0.0005
            counts[f"z{z}_{lo}_{hi}"] = float((H.tinker08_dndlnm(COSMO, 10 ** lmq, z) * np.log(10) * 0.001).sum() * vol)
    res = dict(shells=shell_out, tinker08_counts_tng300_box=counts,
               tng300_counts_pillepich18b=dict(z0_ge1e13=3733, z0_ge1e14=280, z1_ge1e13=2333,
                                               source="Pillepich+18b arXiv:1707.03406 Sec. 2 (M200c, Msun)"),
               moment_check_rel_err=moment_err, notes="f_e = N_e(<R200c)/(f_b M200c (1+X_H)/(2 m_p)); Tinker08 M200c HMF, Planck15; mass window 1e13-1e14 Msun")
    (OUT / "konietzka_implied_fe_shells.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(counts, indent=1))


if __name__ == "__main__":
    main()
