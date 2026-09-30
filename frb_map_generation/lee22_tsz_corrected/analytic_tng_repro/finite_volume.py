#!/usr/bin/env python3
"""Can TNG300's finite volume (205 Mpc/h box, versus HalfDome's 3.75 Gpc/h lightcone) explain the near-empty nominal
1e14-1e15 Msun window of Konietzka's catalogue?

1. Expected number of 1e14-1e15 Msun (M200c) halos in one TNG300 snapshot volume (Tinker08, Planck15), at the snapshot
   redshifts used for rays to z_s <= 1 (0, 0.1, ..., 0.5, 0.7, 1.0), times TNG300's hydro abundance (Pillepich+18b:
   280 halos >= 1e14 at z = 0 against 296.5 from Tinker08; 3 above 1e15). Poisson and cosmic-variance scatter
   (sigma_cv = b_eff D(z) sigma_lin(R_box), top-hat sphere of the box volume).
2. Expected fraction of rays hitting a 1e14-1e15 R200c sphere in a lightcone built from TNG300 snapshots as in
   Konietzka+25 (continuous method: each snapshot fills the comoving interval between the mid-points to its neighbours,
   rays traverse the periodic box, 120,000 rays), with Monte-Carlo box realizations (counts Poisson + cosmic variance,
   masses drawn from the HMF, the same box at every snapshot so the fluctuation is taken fully correlated).
3. The same for the reinterpreted window [9e14, 9e15] and for HalfDome's own lightcone counts.
Output: analysis/finite_volume.json (+ printed table)."""
import json

import numpy as np
from scipy import stats

import repro_lib as R
import halo_model_dm_pdf as hm

COSMO = R.COSMO
L_BOX = 205.0 / COSMO.h                       # Mpc
V_BOX = L_BOX ** 3
SNAPS = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.5])
N_RAYS = 120000
HD_COUNTS = R.FRB / "outputs" / "lee22_corrected_20260930" / "analytic" / "analysis" / "halfdome_catalog_counts_m200c_z.npz"


def hmf_counts(z, lo, hi, dlm=0.005):
    lm = np.arange(np.log10(lo), np.log10(hi), dlm) + dlm / 2
    M = 10 ** lm
    dn = hm.tinker08_dndlnm(COSMO, M, z) * np.log(10) * dlm          # Mpc^-3 per cell
    return lm, M, dn


def snapshot_intervals(z_s):
    """comoving distance interval [chi_lo, chi_hi] filled by each snapshot for rays to z_s (mid-points in chi)."""
    chi = COSMO.chi(SNAPS)
    mids = np.concatenate([[0.0], 0.5 * (chi[1:] + chi[:-1]), [np.inf]])
    chis = COSMO.chi(z_s)[0]
    out = []
    for i, z in enumerate(SNAPS):
        a, b = mids[i], min(mids[i + 1], chis)
        if b > a:
            out.append((z, a, b))
    return out


def per_snapshot(z, lo, hi):
    lm, M, dn = hmf_counts(z, lo, hi)
    n_box = float(dn.sum() * V_BOX)
    r_com = hm.r200c_mpc(COSMO, M, z) * (1 + z)                     # comoving R200c
    area_density = float((dn * np.pi * r_com ** 2).sum())          # Mpc^-1: crossings per comoving Mpc of ray
    b = hm.tinker10_bias(COSMO, M, z)
    b_eff = float((dn * b).sum() / dn.sum())
    r_eq = (3 * V_BOX / (4 * np.pi)) ** (1 / 3)
    sig_lin = float(np.sqrt(COSMO._sigma2_R(*COSMO.power(), r_eq))[0] * COSMO.growth(z)[0])
    return dict(z=z, n_box_tinker=n_box, area_density=area_density, b_eff=b_eff, sigma_lin_box=sig_lin,
                sigma_cv_frac=b_eff * sig_lin, sigma_poisson_frac=1 / np.sqrt(max(n_box, 1e-9)), lm=lm, M=M, dn=dn, r_com=r_com)


def expected_hits(lo, hi, z_sources, abundance=1.0, n_mc=20000, seed=7):
    rng = np.random.default_rng(seed)
    snaps = {z: per_snapshot(z, lo, hi) for z in SNAPS}
    res = {}
    for z_s in z_sources:
        ints = snapshot_intervals(z_s)
        nbar = sum(snaps[z]["area_density"] * abundance * (b - a) for z, a, b in ints)
        # continuous Tinker08 halo model (actual z along the ray, not snapshot z) for comparison
        zz = np.linspace(0, z_s, 201)[1:] - z_s / 400
        nb_cont = 0.0
        for zm in zz:
            lm, M, dn = hmf_counts(zm, lo, hi, 0.01)
            r_com = hm.r200c_mpc(COSMO, M, zm) * (1 + zm)
            nb_cont += (dn * np.pi * r_com ** 2).sum() * (1 / COSMO.H_over_c(zm)) * (z_s / 200)
        # Monte Carlo over box realizations: one Gaussian delta (the same box at all snapshots), Poisson counts with
        # a common uniform (comonotonic) draw, masses from the HMF shape
        delta = rng.standard_normal(n_mc)
        u = rng.random(n_mc)
        nbar_mc = np.zeros(n_mc)
        for z, a, b in ints:
            s = snaps[z]
            # sigma_lin_box already carries D(z): the same Gaussian mode of the box grows linearly
            mean_n = s["n_box_tinker"] * abundance * np.clip(1 + s["b_eff"] * s["sigma_lin_box"] * delta, 0, None)
            n_draw = stats.poisson.ppf(u, mean_n).astype(int)
            p = s["dn"] / s["dn"].sum()
            area = np.pi * s["r_com"] ** 2
            tot_area = np.array([area[rng.choice(len(p), size=k, p=p)].sum() if k > 0 else 0.0 for k in n_draw])
            nbar_mc += tot_area / V_BOX * (b - a)
        hit_mc = 1 - np.exp(-nbar_mc)
        res[z_s] = dict(nbar_snapshots=nbar, hit_snapshots=1 - np.exp(-nbar), nbar_continuous=nb_cont,
                        hit_continuous=1 - np.exp(-nb_cont), hit_mc_median=float(np.median(hit_mc)),
                        hit_mc_p0005=float(np.quantile(hit_mc, 0.0005)), hit_mc_p9995=float(np.quantile(hit_mc, 0.9995)),
                        hit_mc_min=float(hit_mc.min()), hit_mc_frac_rel_std=float(hit_mc.std() / hit_mc.mean()),
                        expected_crossings_in_120k_rays=float(nbar * N_RAYS))
    return snaps, res


def main():
    out = {"box_mpc": L_BOX, "v_box_mpc3": V_BOX}
    z_sources = (0.1, 0.5, 1.0)
    # observed
    obs = {}
    for z_s in z_sources:
        d = R.tng("m1e10_to_1e15", z_s) - R.tng("m1e10_to_1e14", z_s)
        obs[z_s] = dict(hit=float(np.mean(d > 0.01)), n_rays=int(np.sum(d > 0.01)), mean_all=float(d.mean()))
    out["observed_10to15_minus_10to14"] = {str(k): v for k, v in obs.items()}
    for key, (lo, hi), ab in (("nominal_1e14_1e15", (1e14, 1e15), R.A_HI), ("reinterp_9e14_9e15", (R.C_EXT * 1e14, R.C_EXT * 1e15), R.A_HI)):
        snaps, res = expected_hits(lo, hi, z_sources, abundance=ab)
        out[key] = dict(abundance_factor=ab,
                        snapshots={f"{z:g}": {k: v for k, v in s.items() if k not in ("lm", "M", "dn", "r_com")} for z, s in snaps.items()},
                        hits={str(z): v for z, v in res.items()})
        print(key)
        for z, s in snaps.items():
            print(f"  snapshot z={z:.1f}: Tinker08 N(box) = {s['n_box_tinker']:8.2f}  x hydro {ab:.3f} = {s['n_box_tinker'] * ab:8.2f}   "
                  f"sigma_Poisson {100 * s['sigma_poisson_frac']:5.1f}%  sigma_cv {100 * s['sigma_cv_frac']:5.1f}% (b_eff {s['b_eff']:.2f})")
        for z_s, r in res.items():
            print(f"  z_s={z_s}: expected hit {100 * r['hit_snapshots']:.3f}% (continuous {100 * r['hit_continuous']:.3f}%), "
                  f"MC 99.9% range {100 * r['hit_mc_p0005']:.3f}-{100 * r['hit_mc_p9995']:.3f}% (min {100 * r['hit_mc_min']:.3f}%), "
                  f"rel. scatter {100 * r['hit_mc_frac_rel_std']:.1f}%; observed {100 * obs[z_s]['hit']:.3f}% ({obs[z_s]['n_rays']} rays) "
                  f"-> observed/expected {obs[z_s]['hit'] / r['hit_snapshots']:.3f}; expected crossings in 120k rays {r['expected_crossings_in_120k_rays']:.0f}")
    # Tinker >= 1e14 and >= 1e13 at z = 0, 1 against Pillepich+18b
    chk = {}
    for z in (0.0, 1.0):
        for lo in (1e13, 1e14, 1e15):
            lm, M, dn = hmf_counts(z, lo, 1e16)
            chk[f"z{z:g}_ge{np.log10(lo):.0f}"] = float(dn.sum() * V_BOX)
    out["tinker08_box_counts"] = chk
    out["pillepich18b_counts"] = dict(z0_ge13=3733, z0_ge14=280, z0_ge15=3, z1_ge13=2333, max_m200c_z0=1.54e15)
    print("Tinker08 box counts", {k: round(v, 1) for k, v in chk.items()}, "; Pillepich+18b: >=1e13 3733 (z=0) 2333 (z=1), >=1e14 280, >=1e15 3")
    # HalfDome lightcone (M200c from lightcone_100.hdf5 counts on the analytic cells, z < 1)
    if HD_COUNTS.exists():
        d = np.load(HD_COUNTS)
        c, ze, le = d["counts"], d["z_edges"], d["log10m_edges"]
        lc = 0.5 * (le[1:] + le[:-1])
        s = (lc >= 14) & (lc < 15)
        hd = {}
        for z_s in z_sources:
            zsel = 0.5 * (ze[1:] + ze[:-1]) < z_s
            hd[str(z_s)] = float(c[np.ix_(zsel, s)].sum())
        out["halfdome_lightcone_1e14_1e15_counts_below_zs"] = hd
        # TNG300-snapshot lightcone equivalent: number of distinct box volumes along a ray to z_s
        out["tng300_box_lengths_to_zs"] = {str(z): float(COSMO.chi(z)[0] / L_BOX) for z in z_sources}
        print("HalfDome full-sky lightcone 1e14-1e15 halos below z_s:", hd, "; TNG300 box lengths along a ray:",
              out["tng300_box_lengths_to_zs"])
    (R.OUT / "analysis" / "finite_volume.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
