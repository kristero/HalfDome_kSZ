"""verify_2 / test T2: redshift-resolved TNG300 (Konietzka) halo DM vs Lee22 literal, independent code.

Model (all independent of the repo engine):
  * cosmology: flat LCDM TNG/Planck15; HMF: `hmf` package, Tinker08, M200c (SOCritical 200), CAMB T(k)
  * literal Lee22 electrons inside R200c: N_e(M,z) = M200c f_b I_n(M,z) / (X_H m_p)   [eq. 9 printed]
  * mean all-ray halo DM from slice [z1,z2]: int dz (c/H) (1+z) int dlnM dn/dlnM N_e   (exact, no projection)
  * mean number of R200c discs pierced: Nbar = int dz (c/H) int dlnM dn/dlnM pi R200c_com^2
TNG side: per-ray increments of the cumulative within-R200 halo DM between successive z_s.
Outputs: per-slice TNG/literal ratios (all-ray and per-hit), bootstrap errors, fits of
C(z) = A (1+z)^k and of fixed shapes: (1+z)^3 [claim C3], (1+z)^3/E^2 [repo :comoving_hypothesis],
constant, (1+z)^2.
"""
import json, sys, time
import numpy as np
from scipy import integrate
from hmf import MassFunction
from astropy.cosmology import FlatLambdaCDM
import v2_lee22_lib as L

OUT = "/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_normalization_forensics_20260928/verify_2/"
KD = "/home/cbllover/HalfDome/frb_catalog_comparison_outputs/haloDM-20260819T104507Z-1-001/haloDM/"

astro = FlatLambdaCDM(H0=100 * L.H, Om0=L.OM, Ob0=L.OB, Tcmb0=2.7255)
H0 = 100 * L.H


def Hz(z):
    return H0 * np.sqrt(L.E2(z))  # km/s/Mpc


def rho_crit_msun_mpc3(z):
    # 3 H^2 / (8 pi G), physical
    Hs = Hz(z) * 1e5 / L.MPC_CM  # 1/s
    rho = 3 * Hs ** 2 / (8 * np.pi * L.G_CGS)  # g/cm^3
    return rho * L.MPC_CM ** 3 / L.MSUN_G


def r200c_phys_mpc(m_msun, z):
    return (3 * m_msun / (4 * np.pi * 200 * rho_crit_msun_mpc3(z))) ** (1 / 3)


# vectorised I_n on a log-x grid
LX = np.linspace(np.log(1e-7), 0.0, 6001)
XX = np.exp(LX)


def In_grid(table, mh, z):
    amp, xc, beta = L.shape(table, mh, z)
    y = XX ** 3 * L.gnfw(XX, amp, xc, beta)
    return 3 * integrate.simpson(y, x=LX)


def main(windows=((13.0, 14.0), (12.0, 14.0)), mass_unit_h=False, nboot=300):
    t0 = time.time()
    zs = np.load(KD + "redshifts_DMhalo.npy")
    nsl = int(np.searchsorted(zs, 2.0 + 1e-9)) + 1  # slices up to z_s = 2.0
    edges = np.concatenate([[0.0], zs[:nsl]])
    mf = MassFunction(cosmo_model=astro, sigma_8=0.8159, n=0.9667, hmf_model="Tinker08",
                      mdef_model="SOCritical", mdef_params={"overdensity": 200},
                      Mmin=11.0, Mmax=15.5, dlog10m=0.005, z=0.0, transfer_model="CAMB")
    # validate I_n against quad
    for (lm, z) in ((13.0, 0.0), (13.61, 1.0), (14.0, 2.0)):
        a = In_grid(L.TABLE8_N, 10 ** lm, z); b = L.vol_mean(L.TABLE8_N, 10 ** lm, z)
        assert abs(a / b - 1) < 1e-6, (a, b)
    results = {}
    for (lo, hi) in windows:
        key = f"mass{int(lo)}to{int(hi)}" + ("_hunits" if mass_unit_h else "")
        arr = np.load(KD + f"DMhalo_r200_mass{int(lo)}to{int(hi)}.npy")[:nsl]
        inc = np.diff(np.vstack([np.zeros((1, arr.shape[1])), arr]), axis=0)  # (nsl, nray)
        hit = inc > 0
        nray = inc.shape[1]
        # model per slice
        mod_all, mod_N, zmid = [], [], []
        for i in range(nsl):
            zz = np.linspace(edges[i], edges[i + 1], 17)
            fa, fn, fz = [], [], []
            for z in zz:
                mf.update(z=z)
                mh_all = mf.m  # Msun/h
                sel = (np.log10(mh_all / (L.H if not mass_unit_h else 1.0)) >= lo) & \
                      (np.log10(mh_all / (L.H if not mass_unit_h else 1.0)) <= hi)
                mh = mh_all[sel]
                m_msun = mh / L.H
                dndlnm = mf.dndlnm[sel] * L.H ** 3  # Mpc^-3 comoving
                In = np.array([In_grid(L.TABLE8_N, x, z) for x in mh])
                Ne = m_msun * L.MSUN_G * L.FB * In / (L.XH * L.MP_G)
                lnm = np.log(m_msun)
                cH = L.C_KMS / Hz(z)  # Mpc
                dm = cH * (1 + z) * integrate.simpson(dndlnm * Ne, x=lnm)  # Mpc^-2
                dm *= 1.0 / L.MPC_CM ** 2 / L.PC_CM  # -> pc cm^-3 per unit z
                Rcom = r200c_phys_mpc(m_msun, z) * (1 + z)
                nb = cH * integrate.simpson(dndlnm * np.pi * Rcom ** 2, x=lnm)
                fa.append(dm); fn.append(nb); fz.append(dm * z)
            A = integrate.simpson(fa, x=zz); N = integrate.simpson(fn, x=zz)
            mod_all.append(A); mod_N.append(N); zmid.append(integrate.simpson(fz, x=zz) / A)
        mod_all = np.array(mod_all); mod_N = np.array(mod_N); zmid = np.array(zmid)
        mod_hit = 1 - np.exp(-mod_N)
        tng_all = inc.mean(axis=1); tng_hit = hit.mean(axis=1)
        tng_perhit = np.array([inc[i][hit[i]].mean() for i in range(nsl)])
        tng_med = np.array([np.median(inc[i][hit[i]]) for i in range(nsl)])
        mod_perhit = mod_all / mod_hit
        C_all = tng_all / mod_all
        C_hit = tng_perhit / mod_perhit
        # bootstrap over rays
        rng = np.random.default_rng(12345)
        bs_all = np.empty((nboot, nsl)); bs_hit = np.empty((nboot, nsl))
        for b in range(nboot):
            idx = rng.integers(0, nray, nray)
            s = inc[:, idx]; h = s > 0
            ta = s.mean(axis=1); th = h.mean(axis=1)
            bs_all[b] = ta / mod_all
            bs_hit[b] = (ta / th) / mod_perhit
        e_all = bs_all.std(axis=0); e_hit = bs_hit.std(axis=0)
        fits = {}
        for name, C, e, bs in (("all_ray", C_all, e_all, bs_all), ("per_hit", C_hit, e_hit, bs_hit)):
            x = np.log1p(zmid); y = np.log(C); w = 1 / (e / C) ** 2
            # free power law
            p, cov = np.polyfit(x, y, 1, w=np.sqrt(w), cov="unscaled")
            chi_free = np.sum(w * (y - np.polyval(p, x)) ** 2)
            # bootstrap slope
            kb = [np.polyfit(x, np.log(bb), 1, w=np.sqrt(w))[0] for bb in bs]
            shapes = {"(1+z)^3": 3 * x, "(1+z)^3/E^2": 3 * x - np.log(L.E2(zmid)), "const": 0 * x,
                      "(1+z)^2": 2 * x, "(1+z)^2.5": 2.5 * x}
            fs = {}
            for sn, sh in shapes.items():
                lnA = np.sum(w * (y - sh)) / np.sum(w)
                chi = np.sum(w * (y - sh - lnA) ** 2)
                resid_slope = np.polyfit(x, y - sh, 1, w=np.sqrt(w))[0]
                rms = np.sqrt(np.mean((y - sh - lnA) ** 2))
                fs[sn] = dict(A=float(np.exp(lnA)), chi2=float(chi), dof=len(x) - 1,
                              residual_slope=float(resid_slope), rms_log=float(rms))
            fits[name] = dict(k=float(p[0]), k_err_formal=float(np.sqrt(cov[0, 0])), k_boot_std=float(np.std(kb)),
                              A=float(np.exp(p[1])), chi2_free=float(chi_free), dof_free=len(x) - 2, shapes=fs)
        results[key] = dict(z_edges=edges.tolist(), zmid=zmid.tolist(), tng_all=tng_all.tolist(),
                            tng_hit=tng_hit.tolist(), tng_perhit=tng_perhit.tolist(), tng_median_perhit=tng_med.tolist(),
                            mod_all=mod_all.tolist(), mod_hit=mod_hit.tolist(), mod_perhit=mod_perhit.tolist(),
                            C_all=C_all.tolist(), C_all_err=e_all.tolist(), C_hit=C_hit.tolist(), C_hit_err=e_hit.tolist(),
                            hit_ratio_tng_over_model=(tng_hit / mod_hit).tolist(), fits=fits)
        print(f"== {key}  ({time.time()-t0:.0f}s)")
        print(" z1-z2   zmid  TNGall  MODall  C_all(+-)      TNGhit MODhit hitratio  TNGperhit MODperhit C_hit(+-)   TNGmedian")
        for i in range(nsl):
            print(f" {edges[i]:.1f}-{edges[i+1]:.1f} {zmid[i]:.3f} {tng_all[i]:7.3f} {mod_all[i]:7.3f} {C_all[i]:.3f}({e_all[i]:.3f}) "
                  f"{tng_hit[i]:.4f} {mod_hit[i]:.4f} {tng_hit[i]/mod_hit[i]:.3f}  {tng_perhit[i]:7.2f} {mod_perhit[i]:7.2f} "
                  f"{C_hit[i]:.3f}({e_hit[i]:.3f}) {tng_med[i]:7.2f}")
        for name in fits:
            f = fits[name]
            print(f" [{name}] free: C = {f['A']:.3f}(1+z)^{f['k']:.3f} +- {f['k_boot_std']:.3f} (boot), chi2 {f['chi2_free']:.1f}/{f['dof_free']}")
            for sn, s in f["shapes"].items():
                print(f"    {sn:12s} A={s['A']:.3f} chi2={s['chi2']:9.1f}/{s['dof']}  residual slope={s['residual_slope']:+.3f}  rms(ln)={s['rms_log']:.3f}")
    return results


if __name__ == "__main__":
    res = main()
    res_h = main(windows=((13.0, 14.0),), mass_unit_h=True, nboot=100)
    res.update(res_h)
    json.dump(res, open(OUT + "T2_konietzka_slices.json", "w"), indent=1)
