#!/usr/bin/env python3
"""All CLASS-SZ runs of workstream B, sequentially in ONE process (never run two classy_sz processes at once).

Each run -> OUT/class_sz/<name>.npz with ell, cl_1h, cl_2h (C_ell of dimensionless Compton-y), and metadata.
Runs already on disk are skipped (restartable). Selected runs also store:
  * yell_*   : CLASS-SZ's own tabulated y_ell(M, z) at test points, recovered from get_dydzdlnm_at_z_and_m
               divided by dn/dlnM x chi^2/H (profile-level check in profile_checks.py)
  * P3d_*    : CLASS-SZ 3-D P/P200 and P_200 at test points
  * tables   : dn/dlnM (Tinker08 M200c), Tinker10 b(M), P_lin(k=(l+1/2)/chi, z), D_A, chi, H on a (z, M) grid, the
               ingredients of the independent Python halo model (halo_model.py)

Usage: run_class_sz.py [run-name-substring ...]   (no argument: all runs)
"""
import faulthandler, json, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

faulthandler.enable()
sys.path.insert(0, "/home/cbllover/HalfDome/frb_map_generation/lee22_tsz_verification/B_class_sz")
import common as C  # noqa: E402

OUTD = Path(C.OUT) / "class_sz"
OUTD.mkdir(parents=True, exist_ok=True)
H = C.PAINT["h"]
CLIP = C.LEE_CLIP_HINV                 # Msun/h with h = 0.68 (= 10^14.8 h^-1 Msun of Lee22, paint convention)
MMIN, MMAX, ZMIN, ZMAX = C.CAT_M_MIN, C.CAT_M_MAX, C.CAT_Z_MIN, C.CAT_Z_MAX

# test points for the profile-level check: physical M200c [Msun] x z
TEST_M_MSUN = np.array([1e13, 1e14, 1e15])
TEST_Z = np.array([0.1, 0.5, 1.0, 2.0])
TEST_ELL = np.array([100.0, 300.0, 1000.0, 3000.0])
TEST_X = np.geomspace(0.01, 4.0, 60)

# (M, z) grid of the halo-model tables
TAB_Z = np.concatenate([np.geomspace(ZMIN, 0.05, 25, endpoint=False), np.linspace(0.05, ZMAX, 176)])
TAB_LNM = np.linspace(np.log(MMIN), np.log(MMAX), 161)


def runs():
    R = []
    b12, lee = C.b12_class_sz_pars(), C.lee22_class_sz_pars
    P = C.PAINT
    # --- map-level B12, paint cosmology (primary) + tables and profile extraction
    R.append(dict(name="b12_paint_all", cosmo=P, prof=b12, extract="b12", tables=True))
    # --- conventions / reuse: the 2026-09-25 setup reproduced, then one change at a time
    R.append(dict(name="b12_published_cosmomodel_damp", cosmo=C.PUBLISHED, prof=b12, massless_nu=False, damping_1h=1))
    R.append(dict(name="b12_published_massless_nodamp", cosmo=C.PUBLISHED, prof=b12))
    R.append(dict(name="b12_paint_cosmomodel_nodamp", cosmo=P, prof=b12, massless_nu=False))
    R.append(dict(name="b12_paint_massless_damp", cosmo=P, prof=b12, damping_1h=1))
    # --- B12 split at the Lee22 clip mass (compare with the verify_5 split maps Bl, Bh)
    R.append(dict(name="b12_paint_low", cosmo=P, prof=b12, m_max=CLIP))
    R.append(dict(name="b12_paint_high", cosmo=P, prof=b12, m_min=CLIP))
    # --- Lee22 as painted: exact in each mass piece
    R.append(dict(name="lee22_paint_low", cosmo=P, prof=lee(H, "low"), m_max=CLIP, extract="lee22_low"))
    R.append(dict(name="lee22_paint_high", cosmo=P, prof=lee(H, "high"), m_min=CLIP, extract="lee22_high"))
    R.append(dict(name="lee22_paint_calib", cosmo=P, prof=lee(H, "low"), m_min=C.LEE_CALIB_MMIN_HINV, m_max=CLIP,
                  z_max=C.LEE_CALIB_ZMAX))
    # --- Lee22 extrapolated beyond the clip WITHOUT the shape clip (the 'low' parameters to M_max): diagnostic only
    R.append(dict(name="lee22_paint_high_noclip", cosmo=P, prof=lee(H, "low"), m_min=CLIP))
    # --- M1 bookkeeping hypothesis: pressure x (1+z)^2
    for piece, kw in (("low", dict(m_max=CLIP)), ("high", dict(m_min=CLIP)),
                      ("calib", dict(m_min=C.LEE_CALIB_MMIN_HINV, m_max=CLIP, z_max=C.LEE_CALIB_ZMAX))):
        R.append(dict(name="lee22m1_paint_" + piece, cosmo=P, prof=lee(H, "high" if piece == "high" else "low", 2.0), **kw))
    # --- B12 above the clip as the 'hybrid' (Lee22 below, B12 above): = lee22_paint_low + b12_paint_high (1h)
    # --- mass decomposition (1-halo terms add over mass ranges); edges in Msun/h (h = 0.68)
    medges = [MMIN, 1e13, 3e13, 1e14, 3e14, CLIP, MMAX]
    for i in range(len(medges) - 1):
        lo, hi = medges[i], medges[i + 1]
        R.append(dict(name=f"b12_paint_mbin{i}", cosmo=P, prof=b12, m_min=lo, m_max=hi))
        R.append(dict(name=f"lee22_paint_mbin{i}", cosmo=P, prof=lee(H, "high" if lo >= CLIP else "low"), m_min=lo, m_max=hi))
    # --- redshift decomposition (1-halo and 2-halo both add over redshift ranges); Lee22 needs its two mass pieces
    zedges = [ZMIN, 0.1, 0.25, 0.5, 1.0, 2.0, ZMAX]
    for i in range(len(zedges) - 1):
        lo, hi = zedges[i], zedges[i + 1]
        R.append(dict(name=f"b12_paint_zbin{i}", cosmo=P, prof=b12, z_min=lo, z_max=hi))
        R.append(dict(name=f"lee22_paint_low_zbin{i}", cosmo=P, prof=lee(H, "low"), m_max=CLIP, z_min=lo, z_max=hi))
        R.append(dict(name=f"lee22_paint_high_zbin{i}", cosmo=P, prof=lee(H, "high"), m_min=CLIP, z_min=lo, z_max=hi))
    return R


def extract_profiles(c, h, tag):
    """CLASS-SZ's y_ell(M,z) at the test points and its 3-D P/P200, P_200."""
    out = {}
    yl = np.full((len(TEST_M_MSUN), len(TEST_Z), len(TEST_ELL)), np.nan)
    p200 = np.full((len(TEST_M_MSUN), len(TEST_Z)), np.nan)
    for i, mphys in enumerate(TEST_M_MSUN):
        m = mphys * h
        for j, z in enumerate(TEST_Z):
            hmf = c.get_dndlnM_at_z_and_M(z, m)
            chi_h = c.angular_distance(z) * (1 + z) * h                 # Mpc/h
            H_h = c.Hubble(z) / h                                       # h/Mpc
            vol = chi_h ** 2 / H_h
            for k, l in enumerate(TEST_ELL):
                d = c.get_dydzdlnm_at_z_and_m(z, m, l)
                yl[i, j, k] = d / (hmf * vol) if hmf > 1e-90 else np.nan
            p200[i, j] = c.get_P_delta_at_m_and_z_b12(m, z)
    out[f"yell_{tag}"] = yl
    out[f"P200_{tag}"] = p200
    out["test_M_msun"], out["test_z"], out["test_ell"], out["test_x"] = TEST_M_MSUN, TEST_Z, TEST_ELL, TEST_X
    return out


def extract_tables(c, h):
    """Halo-model ingredients on (TAB_Z, TAB_LNM) and the CLASS-SZ ell grid."""
    nz, nm = len(TAB_Z), len(TAB_LNM)
    dndlnm = np.zeros((nz, nm)); bias = np.zeros((nz, nm))
    for j, z in enumerate(TAB_Z):
        for i, lnm in enumerate(TAB_LNM):
            dndlnm[j, i] = c.get_dndlnM_at_z_and_M(z, np.exp(lnm))
        b = np.asarray(c.get_first_order_bias_at_z_and_m(np.array([z]), np.exp(TAB_LNM)), float).reshape(-1)
        bias[j] = b
    da = np.array([c.angular_distance(z) for z in TAB_Z])          # Mpc (physical angular-diameter distance)
    hub = np.array([c.Hubble(z) for z in TAB_Z])                   # 1/Mpc (H/c)
    return dict(tab_z=TAB_Z, tab_lnm_msunh=TAB_LNM, tab_dndlnm_h3mpc3=dndlnm, tab_bias=bias, tab_da_mpc=da,
                tab_H_over_c_invmpc=hub, tab_rho_crit_h2msun_mpc3=np.array([c.get_rho_crit_at_z(z) for z in TAB_Z]))


def pk_table(c, h, ell):
    chi = np.array([c.angular_distance(z) * (1 + z) for z in TAB_Z])   # Mpc
    pk = np.zeros((len(TAB_Z), len(ell)))
    for j, z in enumerate(TAB_Z):
        for i, l in enumerate(ell):
            kh = (l + 0.5) / (chi[j] * h)                                # h/Mpc
            pk[j, i] = c.get_pk_lin_at_k_and_z(kh, z)                    # (Mpc/h)^3
    return dict(tab_pklin_mpch3=pk)


def run_one(spec):
    from classy_sz import Class
    name = spec["name"]
    out = OUTD / f"{name}.npz"
    if out.exists():
        print(f"skip {name} (exists)", flush=True)
        return
    cosmo = spec["cosmo"]
    kw = {k: spec[k] for k in ("m_min", "m_max", "z_min", "z_max", "massless_nu", "damping_1h") if k in spec}
    output = "tSZ_1h,tSZ_2h,dndlnM" if (spec.get("tables") or spec.get("extract")) else "tSZ_1h,tSZ_2h"
    pars = C.class_sz_base(cosmo, output=output, **kw)
    pars.update(spec["prof"])
    c = Class()
    for k in ("T_ncdm", "deg_ncdm", "m_ncdm", "omega_ncdm", "Omega_ncdm"):
        c.pars.pop(k, None)
    c.set(pars)
    t0 = time.time(); c.compute(); dt = time.time() - t0
    h = c.h()
    r = c.cl_sz()
    ell = np.asarray(r["ell"], float)
    fac = 2 * np.pi / (ell * (ell + 1)) * 1e-12
    cl1, cl2 = np.asarray(r["1h"], float) * fac, np.asarray(r["2h"], float) * fac
    md = dict(created_utc=datetime.now(timezone.utc).isoformat(), name=name, parameters=pars, compute_seconds=dt,
              Omega_m=c.Omega_m(), f_b=c.get_f_b(), sigma8=c.sigma8(), Neff=c.Neff(),
              classy_sz_version="0.1.70.post40",
              convention="cl_1h, cl_2h: C_ell of dimensionless Compton-y (class_sz D_ell*1e12 converted); "
                         "3-D pressure truncated at x_outSZ R200c; M200c bounds in Msun/h")
    extra = {}
    if spec.get("extract"):
        extra.update(extract_profiles(c, h, spec["extract"]))
    if spec.get("tables"):
        extra.update(extract_tables(c, h)); extra.update(pk_table(c, h, ell))
    np.savez(out, ell=ell, cl_1h=cl1, cl_2h=cl2, metadata_json=json.dumps(md, sort_keys=True, default=float), **extra)
    i = int(np.argmin(np.abs(ell - 3000)))
    print(f"{name}: {dt:.1f} s, Omega_m={c.Omega_m():.5f} f_b={c.get_f_b():.5f}; "
          f"1e12 D3000 1h={1e12 * cl1[i] * ell[i] * (ell[i] + 1) / 2 / np.pi:.4f} 2h={1e12 * cl2[i] * ell[i] * (ell[i] + 1) / 2 / np.pi:.5f}",
          flush=True)
    c.struct_cleanup(); c.empty()


if __name__ == "__main__":
    sel = sys.argv[1:]
    for spec in runs():
        if sel and not any(s in spec["name"] for s in sel):
            continue
        run_one(spec)
    print("ALL DONE", flush=True)
