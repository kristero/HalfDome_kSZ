#!/usr/bin/env python3
"""Structure of the Konietzka TNG300 halo-DM mass windows, and a Tinker08 halo model of the hypothesis that each
window [lo, hi] effectively extends to [lo, c x hi] in M200c.

1. Window algebra on the 120,000 rays (read-only): overlap of 10to13 and 13to14, of 10to12 and 12to14, and the
   rays with DM(10to15) > DM(10to14) (i.e. the nominal 1e14-1e15 content).
2. Halo model (Tinker08 M200c HMF, Planck15/TNG cosmology, R200c spheres, same quadrature as
   lee22_normalization_forensics/D_gas_fractions/konietzka_implied_gas_fraction.py):
   - hit fraction of [c 1e14, c 1e15] vs the observed 10to15-minus-10to14 hit fraction -> c
   - DM- and ray-weighted overlap of [1e13, c 1e13] inside [1e13, c 1e14] vs the observed overlap
   - apparent f_e(z) = sum_{[1e13, c 1e14]} f_e(M,z) M dn / sum_{[1e13,1e14]} M dn with the CAMELS R200c f_e(M,z)
     (27-box precomputed halo table, x_e-corrected) vs the Konietzka-implied f_e.
Outputs JSON + CSV in outputs/lee22_konietzka_check_20260930/.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/cbllover/HalfDome")
sys.path.insert(0, str(ROOT / "frb_map_generation" / "analytic_pdf_torkamani2026"))
import halo_model_dm_pdf as H  # noqa: E402

KON = ROOT / "frb_catalog_comparison_outputs" / "haloDM-20260819T104507Z-1-001" / "haloDM"
DOUT = ROOT / "frb_map_generation" / "outputs" / "lee22_tsz_verification_20260929" / "D_tng_camels"
FSH = ROOT / "frb_map_generation" / "outputs" / "lee22_normalization_forensics_20260928" / "D_gas_fractions" / "konietzka_implied_fe_shells.json"
OUT = ROOT / "frb_map_generation" / "outputs" / "lee22_konietzka_check_20260930"
COSMO = H.PLANCK15_TNG
FB = COSMO.omega_b / COSMO.omega_m


def window_algebra():
    z = np.load(KON / "redshifts_DMhalo.npy")
    names = ["mass10to12", "mass10to13", "mass10to14", "mass10to15", "mass12to14", "mass13to14", "mass12to15", "all"]
    A = {n: np.load(KON / f"DMhalo_r200_{n}.npy", mmap_mode="r") for n in names}
    rows = []
    for zs in (0.1, 0.2, 0.3, 0.5, 1.0, 1.5):
        i = int(np.argmin(abs(z - zs)))
        d = {n: np.asarray(A[n][i], float) for n in names}
        c, lo, u = d["mass13to14"], d["mass10to13"], d["mass10to14"]
        ov = c + lo - u
        s = c > 0
        f = ov[s] / c[s]
        c2, lo2, u2 = d["mass12to14"], d["mass10to12"], d["mass10to14"]
        ov2 = c2 + lo2 - u2
        s2 = c2 > 0
        diff = d["mass10to15"] - d["mass10to14"]
        rows.append(dict(z_s=float(z[i]), hit_13to14=float(s.mean()), mean_13to14=float(c.mean()),
                         overlap13_DMfrac=float(ov[s].sum() / c[s].sum()), overlap13_rayfrac_full=float(np.mean(f > 0.99)),
                         overlap13_rayfrac_none=float(np.mean(f < 0.01)),
                         overlap12_DMfrac=float(ov2[s2].sum() / c2[s2].sum()),
                         hit_10to15_minus_10to14=float(np.mean(diff > 0.01)), mean_10to15_minus_10to14=float(diff.mean()),
                         mean_10to14=float(u.mean())))
    return rows


def camels_fe_table():
    """<f_e(<R200c)> of CAMELS halos (27 boxes, precomputed gas x f_xe) on a (logM, z) grid; mean per 0.2-dex bin."""
    d = np.load(DOUT / "halo_tables" / "precomputed_halos.npz")
    dd = np.load(DOUT / "halo_tables" / "direct_halos.npz")
    # electron correction fe/fgas from the direct boxes (M >= 1e13): ~0.97-0.99
    m = dd["M200"] >= 1e13
    xe_corr = float(np.sum(dd["fe"][m] * dd["M200"][m]) / np.sum(dd["fgas"][m] * dd["M200"][m]))
    zs = np.unique(np.round(d["z"], 2))
    lgrid = np.arange(12.5, 14.61, 0.2)
    tab = np.full((len(zs), len(lgrid)), np.nan)
    tab25 = np.full((len(zs), len(lgrid)), np.nan)
    cnt = np.zeros((len(zs), len(lgrid)), int)
    lm = np.log10(d["M200"])
    for i, zz in enumerate(zs):
        sz = np.abs(d["z"] - zz) < 0.006
        for j, l0 in enumerate(lgrid):
            s = sz & (np.abs(lm - l0) < 0.1)
            cnt[i, j] = s.sum()
            if s.sum() >= 3:
                tab[i, j] = np.mean(d["fgas"][s])
                tab25[i, j] = np.mean(d["fgas_2p5"][s])
    return zs, lgrid, tab * xe_corr, tab25 * xe_corr, cnt, xe_corr


def make_fe_func(zs, lgrid, tab, cap=0.88):
    """smooth f_e(logM, z): per-z linear fit in logM over bins with data (weights = counts), capped; linear in z."""
    coef = []
    for i in range(len(zs)):
        ok = np.isfinite(tab[i])
        p = np.polyfit(lgrid[ok] - 13.5, tab[i, ok], 1)
        coef.append(p)
    coef = np.array(coef)

    def fe(logm, z):
        p0 = np.interp(z, zs, coef[:, 1])
        p1 = np.interp(z, zs, coef[:, 0])
        return np.clip(p0 + p1 * (logm - 13.5), 0.05, cap)
    return fe, coef


def shell_sums(z0, z1, lm_lo, lm_hi, fe_func=None, nsub=10, dlm=0.01):
    """per shell: DM per unit f_e (pc cm^-3, f_e = 1), f_e-weighted DM, and Nbar = sum pi theta200^2 dN/dOmega."""
    E_PER_MSUN = H.MSUN_G * (1 + 0.76) / (2 * H.M_P_G)
    lm = np.arange(lm_lo, lm_hi, dlm) + dlm / 2
    M = 10 ** lm
    zz = np.linspace(z0, z1, nsub + 1)
    unit = fe_w = nbar = 0.0
    for zm, dz in zip(0.5 * (zz[1:] + zz[:-1]), np.diff(zz)):
        dn = H.tinker08_dndlnm(COSMO, M, zm) * np.log(10) * dlm
        path = (1 / COSMO.H_over_c(zm)) * (1 + zm) * dz
        w = dn * FB * M * E_PER_MSUN / H.MPC_CM ** 3 * path * H.MPC_CM / H.PC_CM
        unit += w.sum()
        if fe_func is not None:
            fe_w += (w * fe_func(lm, zm)).sum()
        r = H.r200c_mpc(COSMO, M, zm)
        chi = COSMO.chi(zm)[0]
        th = np.arctan(r / (chi / (1 + zm)))
        nbar += (chi ** 2 / COSMO.H_over_c(zm) * dz * dn * np.pi * th ** 2).sum()
    return unit, fe_w, nbar


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    alg = window_algebra()
    for r in alg:
        print({k: round(v, 4) for k, v in r.items()})
    zs, lgrid, tab, tab25, cnt, xe_corr = camels_fe_table()
    fe_func, coef = make_fe_func(zs, lgrid, tab)
    print("xe correction", xe_corr)
    shells = json.loads(FSH.read_text())["shells"]
    edges = [0.0] + [s["z_hi"] for s in shells]
    # c from the 1e14-1e15 emptiness: cumulative Nbar of [c 1e14, c 1e15] to z_s
    cs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 9.0, 10.0, 12.0]
    res_c = {}
    for c in cs:
        lc = np.log10(c)
        nb = {}
        cum = 0.0
        for k in range(15):
            _, _, n = shell_sums(edges[k], edges[k + 1], 14 + lc, 15 + lc)
            cum += n
            nb[round(edges[k + 1], 2)] = 1 - np.exp(-cum)
        # overlap predictions at z_s = 0.1 and 0.2 (shells 0-0.1, 0.1-0.2)
        o = {}
        for zsx, kk in ((0.1, 1), (0.2, 2)):
            u_in = f_in = n_in = u_all = f_all = n_all = 0.0
            for k in range(kk):
                _, fa, na = shell_sums(edges[k], edges[k + 1], 13.0, 14 + lc, fe_func)
                _, fi, ni = shell_sums(edges[k], edges[k + 1], 13.0, min(13 + lc, 14 + lc), fe_func)
                f_all += fa; n_all += na; f_in += fi; n_in += ni
            o[zsx] = dict(DMfrac=f_in / f_all, rayfrac=n_in / n_all)
        res_c[c] = dict(hit_c1e14_c1e15={str(k): v for k, v in nb.items() if k in (0.1, 0.5, 1.0, 1.5)}, overlap=o)
        print(f"c={c}: hit[c1e14,c1e15] z_s=0.1 {nb[0.1]:.5f} 0.5 {nb[0.5]:.5f} 1.0 {nb[1.0]:.5f}; "
              f"overlap(z_s=0.1) DM {o[0.1]['DMfrac']:.3f} rays {o[0.1]['rayfrac']:.3f}", flush=True)
    # apparent f_e per shell for the extended windows, and the hit ratio
    rows = []
    zgrid = np.load(KON / "redshifts_DMhalo.npy")
    arr = np.load(KON / "DMhalo_r200_mass13to14.npy")
    for k, s in enumerate(shells[:16]):
        z0, z1 = s["z_lo"], s["z_hi"]
        u0, f0, n0 = shell_sums(z0, z1, 13.0, 14.0, fe_func)
        inc = arr[k] - (arr[k - 1] if k > 0 else 0.0)
        assert abs(zgrid[k] - z1) < 1e-6
        row = dict(z_lo=z0, z_hi=z1, fe_konietzka=s["fe_tng_implied"], fe_konietzka_err=s["fe_tng_implied_err"],
                   fe_camels_R200c=f0 / u0, hit_shell_tng=float(np.mean(inc > 0)), hit_shell_model=float(1 - np.exp(-n0)),
                   hit_ratio_tng_over_model=float(np.mean(inc > 0) / (1 - np.exp(-n0))))
        for c in (3.0, 5.0, 8.0, 9.0, 10.0):
            u1, f1, n1 = shell_sums(z0, z1, 13.0, 14.0 + np.log10(c), fe_func)
            row[f"fe_apparent_c{c:g}"] = f1 / u0
            row[f"hit_ratio_c{c:g}"] = n1 / n0
            row[f"fe_true_window_c{c:g}"] = f1 / u1
        rows.append(row)
        print({kk: round(v, 3) for kk, v in row.items()}, flush=True)
    with open(OUT / "window_extension_shells.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    json.dump(dict(window_algebra=alg, c_scan={str(k): v for k, v in res_c.items()}, xe_correction=xe_corr,
                   camels_fe_fit_coef=dict(z=zs.tolist(), slope=coef[:, 0].tolist(), at13p5=coef[:, 1].tolist()),
                   camels_fe_table=dict(z=zs.tolist(), logM=lgrid.tolist(), fe=np.where(np.isfinite(tab), tab, -1).tolist(),
                                        fe_2p5R200c=np.where(np.isfinite(tab25), tab25, -1).tolist(), counts=cnt.tolist())),
              open(OUT / "window_structure_and_hmf.json", "w"), indent=1)


if __name__ == "__main__":
    main()
