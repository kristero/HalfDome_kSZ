#!/usr/bin/env python3
"""Redshift-resolved test of the bookkeeping hypotheses with the all-ray MEAN halo DM.

TNG300 (Konietzka within-R200c catalogue, 1e13-1e14 Msun window, 120000 rays, cumulative in z_s)
versus the analytic mean of the Lee22 no-c density fit (Table 8, n0 pivot M_cut) truncated at the
R200c sphere, Tinker08 M200c HMF, Planck15/TNG cosmology:

    <DM>(z_s) = int_0^zs dz c (1+z)/H(z) int dlnM dn/dlnM [comoving] N_e(<R200c; M, z)

(the mean over all rays needs neither clustering nor projection geometry). The ratio of the
increments  R(z) = d<DM>_TNG/dz / d<DM>_Lee22/dz  is the mean TNG/Lee22 electron-content ratio of
halos at redshift z. Hypotheses: R = const (z-independent normalisation error) versus
R = (1+z)^3 / F_n (comoving shell volume normalised by the physical n200(z)).
The Tinker08 (gravity-only) HMF versus the hydro TNG halo abundance is a z-independent-ish
systematic of order 10%, not a (1+z)^3 effect.
"""
from pathlib import Path
import json
import sys
import numpy as np
from scipy.integrate import quad

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "analytic_pdf_torkamani2026"))
import halo_model_dm_pdf as hm  # noqa: E402

OUT = HERE.parents[1] / "outputs/lee22_normalization_forensics_20260928/B_paper_code_forensics"
KON = HERE.parents[2] / "frb_catalog_comparison_outputs/haloDM-20260819T104507Z-1-001/haloDM"
C = hm.PLANCK15_TNG
XH = 0.76


def lee_pars(m_msun, z):
    r = m_msun * C.h / 10 ** 13.61
    bpl = lambda a, ap: r ** a if r < 1 else r ** ap
    return (6.8 * r ** 0.68 * (1 + z) ** -2.11, 7.9 * bpl(0.47, -0.45) * (1 + z) ** -0.67,
            19.5 * bpl(0.70, -0.18) * (1 + z) ** -0.31)


def n_electrons_literal(m_msun, z):
    """N_e inside the R200c sphere for the literal Lee22 model (eq. 9 n200, physical)."""
    n0, xc, be = lee_pars(m_msun, z)
    shape = quad(lambda x: x * x * (x / xc) ** -0.3 * (1 + x / xc) ** -be, 0, 1)[0]
    r = hm.r200c_mpc(C, m_msun, z) * hm.MPC_CM                       # cm, physical
    n200 = 200 * C.rho_crit(z) * hm.MSUN_G / hm.MPC_CM ** 3 * (C.omega_b / C.omega_m) / (XH * hm.M_P_G)
    return n0 * n200 * 4 * np.pi * r ** 3 * shape


def dmean_dz(z, lm_lo=13.0, lm_hi=14.0, nm=41):
    lm = np.linspace(lm_lo, lm_hi, nm)
    m = 10 ** lm
    dn = hm.tinker08_dndlnm(C, m, z)                                   # Mpc^-3 comoving
    ne = np.array([n_electrons_literal(mm, z) for mm in m])
    ncom = np.trapz(dn * ne, np.log(m))                                # electrons per comoving Mpc^3
    col = hm.C_KMS / (100 * C.h * C.E(z)) * (1 + z) * ncom             # per Mpc^2 per unit z
    return col / hm.MPC_CM ** 2 / hm.PC_CM                             # pc cm^-3 per unit z


def main():
    zs = np.load(KON / "redshifts_DMhalo.npy")
    tng = np.load(KON / "DMhalo_r200_mass13to14.npy")
    tng_mean = tng.mean(axis=1)
    zgrid = np.linspace(0.0, 3.0, 121)
    dd = np.array([dmean_dz(z) for z in zgrid])
    lee_cum = np.array([np.trapz(dd[zgrid <= s], zgrid[zgrid <= s]) if s <= 3 else np.nan for s in zs])
    sel = zs <= 3.0 + 1e-9
    res = dict(z_s=zs[sel].tolist(), tng_mean_all_rays=tng_mean[sel].tolist(),
               lee22_literal_mean_all_rays=lee_cum[sel].tolist())
    print("z_s  <DM>_TNG  <DM>_Lee22lit  ratio_cum   | slice z_mid  R_slice(TNG/Lee)  R*(1+z)^-3")
    edges = [0.0] + list(zs[sel])
    rows = []
    prev_t, prev_l = 0.0, 0.0
    for s, t, l in zip(zs[sel], tng_mean[sel], lee_cum[sel]):
        zlo = edges[len(rows)]
        zmid = 0.5 * (zlo + s)
        rs = (t - prev_t) / (l - prev_l)
        rows.append(dict(z_lo=float(zlo), z_hi=float(s), z_mid=float(zmid), R=float(rs),
                         R_times_1pz_m3=float(rs / (1 + zmid) ** 3)))
        print(f"{s:4.1f} {t:9.3f} {l:12.3f} {t/l:9.3f}   | {zmid:5.2f} {rs:9.3f} {rs/(1+zmid)**3:9.3f}")
        prev_t, prev_l = t, l
    res["slices"] = rows
    # fits over slices with z_mid <= 2 (Lee22's fitted range), weighted by the TNG increment
    z_mid = np.array([r["z_mid"] for r in rows])
    R = np.array([r["R"] for r in rows])
    w = np.diff(np.r_[0.0, tng_mean[sel]])
    ok = (z_mid <= 2.0) & (R > 0)
    k, lnA = np.polyfit(np.log(1 + z_mid[ok]), np.log(R[ok]), 1, w=np.sqrt(w[ok]))
    res["powerlaw_fit_R_vs_1pz_zmid_le2"] = dict(slope_k=float(k), amplitude_at_z0=float(np.exp(lnA)))
    # fixed slope 3: F_n = 1/amplitude
    lnA3 = np.average(np.log(R[ok]) - 3 * np.log(1 + z_mid[ok]), weights=w[ok])
    res["fixed_slope3_Fn"] = float(np.exp(-lnA3))
    lnA0 = np.average(np.log(R[ok]), weights=w[ok])
    res["fixed_slope0_R"] = float(np.exp(lnA0))
    for lab, pred in (("slope-3 model", np.exp(lnA3) * (1 + z_mid[ok]) ** 3), ("constant", np.exp(lnA0) + 0 * z_mid[ok]),
                      ("free power law", np.exp(lnA) * (1 + z_mid[ok]) ** k)):
        rr = np.log(R[ok] / pred)
        res[f"rms_log_resid_{lab.replace(' ', '_')}"] = float(np.sqrt(np.average(rr ** 2, weights=w[ok])))
    print(f"\nfree power law R = {np.exp(lnA):.3f} (1+z)^{k:.2f}  (z_mid<=2, weights = TNG increment)")
    print(f"fixed (1+z)^3: F_n = {res['fixed_slope3_Fn']:.2f};  constant: R = {res['fixed_slope0_R']:.3f}")
    print("weighted rms log residuals:", {k_: round(v, 3) for k_, v in res.items() if k_.startswith("rms")})
    with open(OUT / "zresolved_mean_dm_test.json", "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__" and len(sys.argv) == 1:
    main()


def coarse_and_form_tests():
    """Coarse slices (robust to the snapshot stepping of the TNG light cones) and a comparison of
    the functional forms (1+z)^3, E^2(z), (1+z)^3/E^2(z), const, for two mass windows."""
    zs = np.load(KON / "redshifts_DMhalo.npy")
    windows = {"1e13-1e14 Msun": (np.load(KON / "DMhalo_r200_mass13to14.npy"), 13.0, 14.0),
               "1e14-1e15 Msun": (np.load(KON / "DMhalo_r200_mass10to15.npy") - np.load(KON / "DMhalo_r200_mass10to14.npy"), 14.0, 15.0)}
    edges = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.3, 1.6, 2.0])
    out = {}
    for name, (arr, lo, hi) in windows.items():
        tm = np.r_[0.0, arr.mean(axis=1)]
        zz = np.r_[0.0, zs]
        zg = np.linspace(0, 2.0, 81)
        dd = np.array([dmean_dz(z, lo, hi) for z in zg])
        rows = []
        for a, b in zip(edges[:-1], edges[1:]):
            t = np.interp(b, zz, tm) - np.interp(a, zz, tm)
            m = (zg >= a - 1e-9) & (zg <= b + 1e-9)
            l = np.trapz(dd[m], zg[m])
            zm = 0.5 * (a + b)
            rows.append(dict(z_lo=a, z_hi=b, tng=float(t), lee_literal=float(l), R=float(t / l),
                             R_over_1pz3=float(t / l / (1 + zm) ** 3), R_over_E2=float(t / l / C.E(zm) ** 2)))
        zm = np.array([0.5 * (r["z_lo"] + r["z_hi"]) for r in rows])
        R = np.array([r["R"] for r in rows])
        w = np.array([r["tng"] for r in rows])
        forms = {"const": np.ones_like(zm), "(1+z)^3": (1 + zm) ** 3, "E^2(z)": C.E(zm) ** 2,
                 "(1+z)^3/E^2 [repo comoving_hypothesis]": (1 + zm) ** 3 / C.E(zm) ** 2}
        ft = {}
        for fn, f in forms.items():
            lnA = np.average(np.log(R / f), weights=w)
            ft[fn] = dict(A=float(np.exp(lnA)), inverse_A=float(np.exp(-lnA)),
                          rms_log=float(np.sqrt(np.average((np.log(R / f) - lnA) ** 2, weights=w))))
        k, lnA = np.polyfit(np.log(1 + zm), np.log(R), 1, w=np.sqrt(w))
        ft["free power law"] = dict(A=float(np.exp(lnA)), slope=float(k))
        out[name] = dict(slices=rows, fits=ft)
        print(f"\n=== {name}: coarse slices ===")
        print(" z_lo-z_hi   TNG_inc   Lee_lit_inc     R=TNG/Lee   R/(1+z)^3   R/E^2")
        for r in rows:
            print(f" {r['z_lo']:.1f}-{r['z_hi']:.1f}  {r['tng']:9.3f} {r['lee_literal']:12.3f} {r['R']:12.3f} {r['R_over_1pz3']:10.3f} {r['R_over_E2']:8.3f}")
        for fn, v in ft.items():
            print("  fit", fn, {kk: round(vv, 3) for kk, vv in v.items()})
    with open(OUT / "zresolved_mean_dm_coarse_forms.json", "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "coarse":
    coarse_and_form_tests()
