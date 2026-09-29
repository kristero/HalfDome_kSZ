#!/usr/bin/env python3
"""Lee22 normalization forensics, test A: how does TNG / model depend on HALO REDSHIFT?

TNG side: Konietzka's legacy TNG300 within-R200 halo-DM arrays (55 source redshifts x 120,000 rays, cumulative
along the same rays). The increment between consecutive source redshifts on the same ray is the DM contributed
by the halos in that redshift slice.

Model side (no ray tracing needed for means): for a halo population with comoving abundance dn/dlnM and N_e(M, z)
free electrons inside the R200c sphere, the mean halo DM per ray contributed by a comoving path dchi is

    d<DM> = dchi  int dlnM dn/dlnM  N_e(M, z) (1+z)          [N_e(1+z) = int DM dA_comoving]

and the mean number of halos crossed is dchi int dlnM dn/dlnM pi R200c_comoving^2. Both hold for a light cone
from one observer and for parallel rays through a replicated box. validate_models.py checks the enclosed-electron
shortcut against the chord tables of the Julia pipeline (5e-5).

Abundance: Tinker08 M200c (halo_model_dm_pdf.py), Planck15/TNG cosmology. Two redshift assignments:
  continuous : halos, profiles and (1+z) at the redshift of the path element;
  snapshot   : Konietzka et al. 2025 (arXiv:2507.07090) trace each ray segment through ONE of the 13 full TNG
               snapshots z_i = 0, 0.1, ..., 0.5, 0.7, 1, 1.5, 2, 3, 4, 5, segment boundaries at the redshift midpoints;
               halos, masses, profiles and (1+z_i) are those of the snapshot, the path length is the real one.
Models: Lee22 no-c literal (eq. 9 as printed), Lee22 XGPaint-native (0.60163 x literal), Battaglia16 (XGPaint,
physically normalized), fb_full (all cosmic baryons f_b M200c inside R200c, fully ionized; a pure bookkeeping
control whose C(z) is TNG's effective free-electron fraction inside R200c).

C(z_slice) = <Delta DM>_TNG / <Delta DM>_model per source-redshift slice, bootstrap over rays (same resampled rays for
all slices, so the covariance between slices is kept). Fit C = C0 (1+z)^k over slices inside [0.1, 2].
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]          # repository root
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "frb_map_generation" / "analytic_pdf_torkamani2026"))
import models as M                     # noqa: E402
import halo_model_dm_pdf as hm         # noqa: E402

TNG_DIR = ROOT / "frb_catalog_comparison_outputs" / "haloDM-20260819T104507Z-1-001" / "haloDM"
OUT = ROOT / "frb_map_generation" / "outputs" / "lee22_normalization_forensics_20260928" / "A_tng_redshift_test"
OUT.mkdir(parents=True, exist_ok=True)

COSMO = M.PLANCK15_TNG            # own implementation (no radiation), used for R200c, chi, electrons
HMC = hm.PLANCK15_TNG             # same parameters, used for Tinker08 (CAMB P(k), sigma8 = 0.8159)
assert abs(COSMO.omega_m - HMC.omega_m) < 1e-12 and abs(COSMO.h - HMC.h) < 1e-12

WINDOWS = {"all": (1e9, 1e16), "mass13to14": (1e13, 1e14), "mass12to14": (1e12, 1e14),
           "mass10to13": (1e10, 1e13), "mass10to12": (1e10, 1e12)}
MODELS = ("lee22_literal", "lee22_xgpnative", "b16", "fb_full")
SNAPS = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0])
SNAP_EDGES = np.concatenate([[0.0], 0.5 * (SNAPS[1:] + SNAPS[:-1]), [np.inf]])
DZ = 0.005
N_BOOT = 1000
FIT_RANGE = (0.1, 2.0)
CANDIDATES = {"h^2": 0.6774 ** 2, "h^3": 0.6774 ** 3, "X_H^2": 0.76 ** 2, "h^3 X_H^2": 0.6774 ** 3 * 0.76 ** 2,
              "h^2 X_H^2": 0.6774 ** 2 * 0.76 ** 2, "X_H(1+X_H)/2": 0.76 * 1.76 / 2, "xgpnative 0.6016": M.XGP_NATIVE_FACTOR}


def snapshot_of(z):
    return SNAPS[np.searchsorted(SNAP_EDGES, z, side="right") - 1]


# ------------------------------------------------------------------------------------------------ TNG
def load_tng():
    zs = np.load(TNG_DIR / "redshifts_DMhalo.npy")
    arr = {w: np.load(TNG_DIR / "DMhalo_r200_{}.npy".format(w)) for w in WINDOWS}
    return np.round(zs, 6), arr


def tng_bootstrap(arr, rng):
    """Per window: increments (55, N), bootstrap replicates of slice means and slice hit fractions."""
    out = {}
    n = next(iter(arr.values())).shape[1]
    idx_sets = [rng.integers(0, n, n) for _ in range(N_BOOT)]
    counts = np.stack([np.bincount(ix, minlength=n) for ix in idx_sets]).astype(np.float64)   # (B, N)
    for w, a in arr.items():
        inc = np.diff(np.vstack([np.zeros((1, n)), a]), axis=0)        # (55, N), >= 0 (checked)
        hit = (inc > 0).astype(np.float64)
        cum_hit = (a > 0).astype(np.float64)
        out[w] = dict(
            inc_mean=inc.mean(1), hit_mean=hit.mean(1), cum_mean=a.mean(1), cum_hit=cum_hit.mean(1),
            inc_boot=counts @ inc.T / n, hit_boot=counts @ hit.T / n,
            cum_boot=counts @ a.T / n, cum_hit_boot=counts @ cum_hit.T / n,
            mean_per_hit=np.array([a[i][a[i] > 0].mean() if (a[i] > 0).any() else np.nan for i in range(a.shape[0])]))
    return out


# ------------------------------------------------------------------------------------------------ model
def model_grids(zmax):
    """Per fine z (midpoints, step DZ) and per mass cell: mean-DM and Nbar weights for every model, in both
    redshift-assignment modes. Returns dict[mode][model] -> (nz, nm) arrays of d<DM>, plus dict[mode]['nbar']."""
    lm_edges = np.round(np.arange(9.0, 16.0 + 1e-9, 0.01), 10)
    lm = 0.5 * (lm_edges[1:] + lm_edges[:-1])
    mass = 10 ** lm
    dlnm = 0.01 * np.log(10)
    zf = np.arange(DZ / 2, zmax, DZ)
    dchi = C_over_H = M.C_KMS / (100 * COSMO.h * COSMO.E(zf)) * DZ          # comoving Mpc per fine step
    out = {}
    hmf_cache = {}

    def hmf(z):
        key = round(float(z), 6)
        if key not in hmf_cache:
            hmf_cache[key] = hm.tinker08_dndlnm(HMC, mass, key)             # comoving Mpc^-3 per ln M
        return hmf_cache[key]

    elec_cache = {}

    def electrons(model, z):
        key = (model, round(float(z), 6))
        if key not in elec_cache:
            elec_cache[key] = M.electrons_in_r200(model, COSMO, mass, np.full_like(mass, key[1]))
        return elec_cache[key]

    for mode in ("continuous", "snapshot"):
        res = {m: np.zeros((len(zf), len(lm))) for m in MODELS}
        nbar = np.zeros((len(zf), len(lm)))
        zeff = np.zeros(len(zf))
        for i, z in enumerate(zf):
            ze = z if mode == "continuous" else snapshot_of(z)
            zeff[i] = ze
            dn = hmf(ze) * dlnm                                                # comoving Mpc^-3
            r200_com = COSMO.r200c_m(mass, ze) / M.MPC_M * (1 + ze)           # comoving Mpc
            nbar[i] = dchi[i] * dn * np.pi * r200_com ** 2
            for m in MODELS:
                ne = electrons(m, ze)                                          # count
                # d<DM> [pc cm^-3] = dchi dn N_e (1+z) / area ; dchi in Mpc, dn in Mpc^-3 -> per Mpc^2 -> per m^2
                res[m][i] = dchi[i] * dn * ne * (1 + ze) / M.MPC_M ** 2 * M.M2_TO_PC_CM3
        res["nbar"] = nbar
        res["zeff"] = zeff
        out[mode] = res
    return zf, lm_edges, lm, out


def window_fraction(lm_edges, lo, hi):
    return np.clip((np.minimum(lm_edges[1:], np.log10(hi)) - np.maximum(lm_edges[:-1], np.log10(lo))) / 0.01, 0, 1)


# ------------------------------------------------------------------------------------------------ fits
def fit_powerlaw(zmid, C, cov_logC):
    """GLS fit of log C = log C0 + k log(1+z) with the bootstrap covariance of log C."""
    X = np.vstack([np.ones_like(zmid), np.log1p(zmid)]).T
    ci = np.linalg.pinv(cov_logC)
    A = X.T @ ci @ X
    beta = np.linalg.solve(A, X.T @ ci @ np.log(C))
    r = np.log(C) - X @ beta
    chi2 = float(r @ ci @ r)
    cov = np.linalg.inv(A)
    return beta, cov, chi2


def fit_fixed_k(zmid, C, cov_logC, k):
    ci = np.linalg.pinv(cov_logC)
    y = np.log(C) - k * np.log1p(zmid)
    one = np.ones_like(zmid)
    a = float(one @ ci @ y / (one @ ci @ one))
    r = y - a
    return a, float(np.sqrt(1 / (one @ ci @ one))), float(r @ ci @ r)


def main():
    rng = np.random.default_rng(20260928)
    zs, arr = load_tng()
    # sanity: cumulative along the same rays
    mono = {w: bool((np.diff(a, axis=0) >= 0).all()) for w, a in arr.items()}
    tb = tng_bootstrap(arr, rng)
    zf, lm_edges, lm, grids = model_grids(zmax=zs[-1] + 1e-9)
    slice_lo = np.concatenate([[0.0], zs[:-1]])
    slice_hi = zs
    nsl = len(zs)
    # fine-z membership of slices
    member = [(zf >= slice_lo[j] - 1e-9) & (zf < slice_hi[j] - 1e-9) for j in range(nsl)]

    rows_cum, rows_slice, fits, boot_store = [], [], [], {}
    summary = dict(monotonic_cumulative_per_ray=mono, n_rays=int(next(iter(arr.values())).shape[1]),
                   source_redshifts=zs.tolist(), n_boot=N_BOOT, fit_range=FIT_RANGE, candidates=CANDIDATES)
    for w, (lo, hi) in WINDOWS.items():
        frac = window_fraction(lm_edges, lo, hi)
        t = tb[w]
        for mode in ("continuous", "snapshot"):
            g = grids[mode]
            nb_slice = np.array([(g["nbar"][member[j]] * frac).sum() for j in range(nsl)])
            dm_slice = {m: np.array([(g[m][member[j]] * frac).sum() for j in range(nsl)]) for m in MODELS}
            zw = {m: np.array([np.average(zf[member[j]], weights=(g[m][member[j]] * frac).sum(1) + 1e-300)
                               for j in range(nsl)]) for m in MODELS}
            nb_cum = np.cumsum(nb_slice)
            for j in range(nsl):
                row = dict(window=w, mode=mode, z_lo=slice_lo[j], z_hi=slice_hi[j],
                           tng_dDM=t["inc_mean"][j], tng_dDM_err=t["inc_boot"][:, j].std(ddof=1),
                           tng_slice_hit=t["hit_mean"][j], tng_slice_hit_err=t["hit_boot"][:, j].std(ddof=1),
                           tng_slice_nbar=-np.log1p(-t["hit_mean"][j]),
                           model_slice_nbar=nb_slice[j], model_slice_hit=-np.expm1(-nb_slice[j]))
                row["nbar_ratio_tng_over_model"] = row["tng_slice_nbar"] / nb_slice[j] if nb_slice[j] > 0 else np.nan
                for m in MODELS:
                    row["model_dDM_" + m] = dm_slice[m][j]
                    row["zmean_dm_weighted_" + m] = zw[m][j]
                    cb = t["inc_boot"][:, j] / dm_slice[m][j]
                    row["C_" + m] = t["inc_mean"][j] / dm_slice[m][j]
                    row["C_err_" + m] = cb.std(ddof=1)
                rows_slice.append(row)
            for j in range(nsl):
                row = dict(window=w, mode=mode, z_s=zs[j], tng_mean_all=t["cum_mean"][j],
                           tng_mean_all_err=t["cum_boot"][:, j].std(ddof=1), tng_hit=t["cum_hit"][j],
                           tng_hit_err=t["cum_hit_boot"][:, j].std(ddof=1), tng_mean_per_hit=t["mean_per_hit"][j],
                           model_nbar=nb_cum[j], model_hit_unclustered=-np.expm1(-nb_cum[j]))
                for m in MODELS:
                    cm = dm_slice[m][:j + 1].sum()
                    row["model_mean_all_" + m] = cm
                    row["model_mean_per_hit_" + m] = cm / row["model_hit_unclustered"]
                    row["cumratio_" + m] = t["cum_mean"][j] / cm
                    row["cumratio_err_" + m] = (t["cum_boot"][:, j] / cm).std(ddof=1)
                rows_cum.append(row)
            # fits over slices with z_lo >= 0.1 and z_hi <= 2.0
            for zr in (FIT_RANGE, (0.0, 2.0), (0.1, 1.0), (1.0, 2.0), (0.1, 3.0)):
                sel = np.where((slice_lo >= zr[0] - 1e-9) & (slice_hi <= zr[1] + 1e-9))[0]
                for m in MODELS:
                    zmid = zw[m][sel]
                    C = t["inc_mean"][sel] / dm_slice[m][sel]
                    Cb = t["inc_boot"][:, sel] / dm_slice[m][sel][None]
                    if np.any(C <= 0) or np.any(Cb <= 0):
                        continue
                    covl = np.cov(np.log(Cb).T)
                    beta, cov, chi2 = fit_powerlaw(zmid, C, covl)
                    # bootstrap refits (same covariance) for robust errors
                    bb = np.array([fit_powerlaw(zmid, Cb[b], covl)[0] for b in range(N_BOOT)])
                    fk3 = fit_fixed_k(zmid, C, covl, 3.0)
                    fk0 = fit_fixed_k(zmid, C, covl, 0.0)
                    fits.append(dict(window=w, mode=mode, model=m, z_range="{}-{}".format(*zr), n_slices=len(sel),
                                     C0=float(np.exp(beta[0])), C0_err=float(np.exp(beta[0]) * np.sqrt(cov[0, 0])),
                                     k=float(beta[1]), k_err=float(np.sqrt(cov[1, 1])),
                                     corr_logC0_k=float(cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])),
                                     C0_boot_std=float(np.exp(bb[:, 0]).std(ddof=1)), k_boot_std=float(bb[:, 1].std(ddof=1)),
                                     chi2=chi2, dof=len(sel) - 2,
                                     C_at_z0p5=float(np.exp(beta[0]) * 1.5 ** beta[1]),
                                     C0_fixed_k3=float(np.exp(fk3[0])), C0_fixed_k3_err=float(np.exp(fk3[0]) * fk3[1]),
                                     chi2_fixed_k3=fk3[2],
                                     C0_fixed_k0=float(np.exp(fk0[0])), C0_fixed_k0_err=float(np.exp(fk0[0]) * fk0[1]),
                                     chi2_fixed_k0=fk0[2]))
                    if zr == FIT_RANGE:
                        boot_store["{}|{}|{}".format(w, mode, m)] = dict(beta=beta.tolist(), cov=cov.tolist())
    import csv
    for name, rows in (("tng_vs_model_slices.csv", rows_slice), ("tng_vs_model_cumulative.csv", rows_cum),
                       ("powerlaw_fits.csv", fits)):
        with open(OUT / name, "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            wr.writeheader()
            for r in rows:
                wr.writerow({k: (float(v) if isinstance(v, (np.floating,)) else v) for k, v in r.items()})
    # TNG-only per-z_s statistics for every window file
    all_w = ["all", "mass13to14", "mass12to14", "mass12to15", "mass10to12", "mass10to13", "mass10to14", "mass10to15",
             "mass11to14", "mass9to13", "mass9to14"]
    with open(OUT / "tng_window_stats_by_zs.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["window", "z_s", "mean_dm_all_rays", "hit_fraction", "mean_dm_per_hit", "median_dm_per_hit"])
        for w in all_w:
            a = np.load(TNG_DIR / "DMhalo_r200_{}.npy".format(w))
            for j in range(len(zs)):
                v = a[j]; h = v > 0
                wr.writerow([w, zs[j], v.mean(), h.mean(), v[h].mean() if h.any() else np.nan,
                             np.median(v[h]) if h.any() else np.nan])
    np.savez(OUT / "model_grids.npz", zf=zf, lm=lm, lm_edges=lm_edges,
             **{"{}__{}".format(mode, k): v for mode, g in grids.items() for k, v in g.items()})
    summary["fit_gls"] = boot_store
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    print("monotonic:", mono)
    for f in fits:
        if f["z_range"] == "0.1-2.0" and f["window"] in ("mass13to14", "mass12to14"):
            print("{window:11s} {mode:10s} {model:16s} C0={C0:.4f}+-{C0_err:.4f} k={k:.3f}+-{k_err:.3f} "
                  "chi2/dof={chi2:.1f}/{dof}  C0(k=3)={C0_fixed_k3:.4f} chi2={chi2_fixed_k3:.1f}  "
                  "C0(k=0)={C0_fixed_k0:.4f} chi2={chi2_fixed_k0:.1f}".format(**f))


if __name__ == "__main__":
    main()
