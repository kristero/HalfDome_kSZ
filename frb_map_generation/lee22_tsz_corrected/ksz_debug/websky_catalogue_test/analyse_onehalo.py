# One-halo kSZ of the WebSky and HalfDome halo catalogues through OUR B16 machinery (tau_ell_table_b16.h5 from tau_ell_table.jl),
# compared with the digitized WebSky halo curve (Stein+2020 Fig. 6) and decomposed into abundance / velocity / profile parts.
# (2026-10-02, ksz_debug/websky_catalogue_test)
import os, json, sys
import numpy as np, h5py

OUT = "/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_ksz_debug_20261002/websky_catalogue_test"
PROD = "/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_ksz_20261002"
T2 = 2.7255e6**2
OM, OB, H = 0.31, 0.049, 0.68; FB = OB / OM
SIGMA_T = 6.6524587e-29; M_P = 1.67262192e-27; MSUN = 1.98847e30; MPC = 3.0856775814913673e22
RHOM_COM = 2.775e11 * OM * H**2          # Msun / Mpc^3 comoving
ARCMIN = np.pi / 180 / 60
LREP = np.array([300.0, 500.0, 1000.0, 2000.0, 3000.0, 5000.0])

tab = h5py.File(os.path.join(OUT, "tau_ell_table_b16.h5"), "r")
ell = tab["ell"][()]; le = tab["logm_edges"][()]; ze = tab["z_edges"][()]
lmc = 0.5 * (le[1:] + le[:-1]); zc = 0.5 * (ze[1:] + ze[:-1]); Mc = 10**lmc
A = {X: tab[f"a_b16_x{X}p0"][()].transpose(2, 1, 0) for X in (4, 2, 1)}          # (ell, M, z)
A0 = {X: tab[f"a0_b16_x{X}p0"][()].T for X in (4, 2, 1)}
TH200 = tab["theta200c_rad"][()].T; dA = tab["dA_mpc"][()]
NL, NM, NZ = A[4].shape

def cl1h(v2, a):
    return np.einsum("mz,lmz->l", v2, a * a) / (4 * np.pi)
def dl(c):
    return ell * (ell + 1) * c / (2 * np.pi) * T2
def at(c, L=LREP):
    return np.exp(np.interp(np.log(L), np.log(ell), np.log(np.maximum(c, 1e-300))))

# ------------------------------------------------------------- inputs
with h5py.File(os.path.join(PROD, "maps", "halo_mass_redshift_histogram.h5"), "r") as f:
    hd_n = f["count"][()].T.astype(float); hd_v2 = f["sum_vlos_over_c_squared"][()].T
assert hd_n.shape == (NM, NZ)
hd2 = np.load(os.path.join(OUT, "halfdome_halo_histograms_m200m_m200c.npz"))
ws = np.load(os.path.join(OUT, "websky_halo_mass_redshift_histograms.npz"))
dig = np.loadtxt(os.path.join(PROD, "websky_halo_stein2020_fig6_digitized_binned.csv"), delimiter=",")
def websky_dig(L):
    return np.exp(np.interp(np.log(L), np.log(dig[:, 0]), np.log(dig[:, 1]), left=np.nan, right=np.nan))
res = {"ell_rep": LREP.tolist()}

# ------------------------------------------------------------- validation against the production one-halo numbers
C_hd = cl1h(hd_v2, A[4])
il = [np.argmin(abs(ell - L)) for L in (300, 1000, 3000)]
print("validation HalfDome B16 4R200c one-halo D_ell at ell", [round(ell[i]) for i in il], [round(dl(C_hd)[i], 4) for i in il],
      "(poisson.log: 0.1105 0.3547 0.6469)")
assert np.allclose([dl(C_hd)[i] for i in il], [0.1105, 0.3547, 0.6469], rtol=2e-3)
assert np.allclose(hd2["m200c_count"], hd_n) and np.allclose(hd2["m200c_v2"], hd_v2, rtol=1e-6), "HalfDome M200c histogram mismatch"

# ------------------------------------------------------------- helper: z-only M200m -> M200c remap of a (logM200m, z) histogram
def remap(h, shift):
    """Shift every z column of a (logM, z) histogram by shift[iz] dex, redistributing uniformly within the 0.025-dex bins."""
    out = np.zeros_like(h); w = 0.025
    for iz in range(NZ):
        s = shift[iz] / w; k = int(np.floor(s)); fr = s - k
        col = h[:, iz]
        for j in range(NM):
            if col[j] == 0: continue
            for jj, ww in ((j + k, 1 - fr), (j + k + 1, fr)):
                if 0 <= jj < NM and ww > 0: out[jj, iz] += ww * col[j]
    return out
def omz(z): return OM * (1 + z)**3 / (OM * (1 + z)**3 + 1 - OM)
def delta_bn(z):
    x = omz(z) - 1; return 18 * np.pi**2 + 82 * x - 39 * x**2
ratio_sis = np.sqrt(delta_bn(zc) / 200.0)                                    # pks2map: mh = sqrt(Delta_c^BN / 200) M_RTH
ratio_c7m_zc = np.interp(zc, ws["ratio_ztab"], ws["ratio_c7m"])
WS = {}
for k in ("c7m", "c7c", "duffy", "poly", "om035", "m200m"):
    WS[k] = {s: ws[f"{k}_{s}"].astype(float) for s in ("count", "v2", "vtot2", "count_sel", "v2_sel")}
WS["sis_remap"] = {"count": remap(WS["m200m"]["count"], np.log10(ratio_sis)), "v2": remap(WS["m200m"]["v2"], np.log10(ratio_sis))}
WS["c7m_remap"] = {"count": remap(WS["m200m"]["count"], np.log10(ratio_c7m_zc)), "v2": remap(WS["m200m"]["v2"], np.log10(ratio_c7m_zc))}

sel_cell = (Mc[:, None] > 1e13) & (TH200 > 0.5 * ARCMIN)                   # WebSky halo-map cut at cell centres
def m13(h): return h * (Mc[:, None] > 1e13)

# ------------------------------------------------------------- compensation (Stein+2020 Sec 3.2.1): minus a uniform sphere of the
# same position and mass, radius at overdensity 3 (vs the comoving mean matter density); a_sph = sigma_T N_e / d_A^2 * 3 j1(x)/x
def w_sphere(x):
    x = np.maximum(x, 1e-6); return 3 * (np.sin(x) - x * np.cos(x)) / x**3
chi = dA * (1 + zc)
def a_sphere(mass_comp_msun, ne_per_msun, X=None, a0=None):
    """mass_comp: (M,) or (M,z) Msun; returns (ell, M, z). If a0 is given, the sphere carries the painted tau integral a0."""
    mcomp = np.broadcast_to(mass_comp_msun if np.ndim(mass_comp_msun) == 2 else mass_comp_msun[:, None], (NM, NZ))
    R3 = (3 * mcomp / (4 * np.pi * 3.0 * RHOM_COM))**(1 / 3)                  # comoving Mpc
    th3 = R3 / chi[None, :]
    amp = SIGMA_T * FB * mcomp * MSUN * ne_per_msun / (dA[None, :] * MPC)**2 if a0 is None else a0
    return amp[None] * w_sphere(ell[:, None, None] * th3[None])
NE_WEBSKY = 1.0 / (1.136 * M_P)                                            # electrons per kg of gas (1 - Yp/2 = 0.88), no 0.9
gas_ledger = A0[4] / (SIGMA_T * FB * Mc[:, None] * MSUN * NE_WEBSKY / (dA[None, :] * MPC)**2)
res["gas_ledger_painted_tau_over_fb_M200c_websky_ne"] = {f"logM{lm}_z{z}": float(gas_ledger[np.argmin(abs(lmc - lm)), np.argmin(abs(zc - z))])
                                                      for lm in (13.0, 14.0, 15.0) for z in (0.1, 0.5, 1.0, 2.0)}
print("painted tau integral (B16 4R200c, incl. XGPaint 0.9) / [sigma_T f_b M200c/(1.136 m_p) / d_A^2]:", res["gas_ledger_painted_tau_over_fb_M200c_websky_ne"])
A_comp_we = A[4] - a_sphere(Mc, NE_WEBSKY)                                  # Websky-literal: sphere gas = f_b M200c
A_comp_zero = A[4] - a_sphere(Mc, None, a0=A0[4])                           # zero net electrons per halo

# ------------------------------------------------------------- spectra
spec = {}
spec["halfdome_b16_x4"] = dl(C_hd)
for X in (4, 2, 1):
    spec[f"halfdome_b16_x{X}"] = dl(cl1h(hd_v2, A[X]))
    for k in ("c7m", "c7c", "duffy", "poly", "om035", "m200m", "sis_remap", "c7m_remap"):
        spec[f"websky_{k}_b16_x{X}"] = dl(cl1h(WS[k]["v2"], A[X]))
for k in ("c7m", "c7c", "duffy", "poly", "om035"):
    spec[f"websky_{k}_b16_x4_sel_halo"] = dl(cl1h(WS[k]["v2_sel"], A[4]))          # per-halo WebSky cut
    spec[f"websky_{k}_b16_x4_sel_cell"] = dl(cl1h(WS[k]["v2"] * sel_cell, A[4]))
spec["websky_sis_remap_b16_x4_sel_cell"] = dl(cl1h(WS["sis_remap"]["v2"] * sel_cell, A[4]))
spec["halfdome_b16_x4_sel_cell"] = dl(cl1h(hd_v2 * sel_cell, A[4]))
spec["halfdome_b16_x4_m13"] = dl(cl1h(m13(hd_v2), A[4]))
spec["websky_c7m_b16_x4_m13"] = dl(cl1h(m13(WS["c7m"]["v2"]), A[4]))
for tag, Acomp in (("comp_websky", A_comp_we), ("comp_zero", A_comp_zero)):
    spec[f"halfdome_b16_x4_{tag}"] = dl(cl1h(hd_v2, Acomp))
    spec[f"websky_c7m_b16_x4_{tag}"] = dl(cl1h(WS["c7m"]["v2"], Acomp))
    spec[f"websky_c7m_b16_x4_sel_{tag}"] = dl(cl1h(WS["c7m"]["v2_sel"], Acomp))
    spec[f"websky_sis_b16_x4_sel_{tag}"] = dl(cl1h(WS["sis_remap"]["v2"] * sel_cell, Acomp))
# velocity-index-bug emulation (pks2map, external report): v^2 per cell replaced by the z-average at fixed mass (upper bound)
vbar_m = WS["c7m"]["v2"].sum(1) / np.maximum(WS["c7m"]["count"].sum(1), 1)
spec["websky_c7m_b16_x4_vshuffle"] = dl(cl1h(WS["c7m"]["count"] * vbar_m[:, None], A[4]))

# ------------------------------------------------------------- decomposition HalfDome / WebSky (c7m, B16 4 R200c, no cut)
def decomp(nA, vA, nB, vB, a, minc=10):
    VB = np.where(nB >= minc, vB / np.maximum(nB, 1), np.nan)
    vz = vB.sum(0) / np.maximum(nB.sum(0), 1)                                 # per-z fallback
    VB = np.where(np.isfinite(VB), VB, vz[None, :])
    CA = cl1h(vA, a); CB = cl1h(vB, a); Cx = cl1h(nA * VB, a)
    return CA / CB, CA / Cx, Cx / CB                                        # total, velocity part, abundance part
dec = {}
for tag, (nB, vB) in (("c7m", (WS["c7m"]["count"], WS["c7m"]["v2"])), ("duffy", (WS["duffy"]["count"], WS["duffy"]["v2"])),
                      ("sis", (WS["sis_remap"]["count"], WS["sis_remap"]["v2"]))):
    tot, vel, ab = decomp(hd_n, hd_v2, nB, vB, A[4])
    dec[tag] = {"total": at(tot).tolist(), "velocity": at(vel).tolist(), "abundance": at(ab).tolist()}
# same in M200m (both catalogues' own M200m), painted with the c7m conversion's M200c? -> abundance in native mass definition
res["decomposition_halfdome_over_websky"] = dec

# ------------------------------------------------------------- abundance diagnostics
def cum_counts(n, lo):
    return float(n[lmc >= lo].sum())
ab = {}
for zlo, zhi in ((0.0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 4.0)):
    zs = (zc > zlo) & (zc < zhi)
    row = {}
    for lo in (13.0, 13.5, 14.0, 14.5):
        row[f"N(M200c>{lo}) HD/WS_c7m"] = cum_counts(hd_n[:, zs], lo) / max(cum_counts(WS["c7m"]["count"][:, zs], lo), 1)
        row[f"N(M200m>{lo}) HD/WS"] = cum_counts(hd2["m200m_count"][:, zs], lo) / max(cum_counts(WS["m200m"]["count"][:, zs], lo), 1)
    row["vlos_rms_kms_HD_M>1e13"] = float(np.sqrt(m13(hd_v2)[:, zs].sum() / m13(hd_n)[:, zs].sum()) * 299792.458)
    row["vlos_rms_kms_WS_M>1e13"] = float(np.sqrt(m13(WS["c7m"]["v2"])[:, zs].sum() / m13(WS["c7m"]["count"])[:, zs].sum()) * 299792.458)
    row["v3d_over_sqrt3_kms_WS_M>1e13"] = float(np.sqrt(m13(WS["c7m"]["vtot2"])[:, zs].sum() / 3 / m13(WS["c7m"]["count"])[:, zs].sum()) * 299792.458)
    row["v3d_over_sqrt3_kms_HD_M>1e13"] = float(np.sqrt(m13(hd2["m200c_vtot2"])[:, zs].sum() / 3 / m13(hd_n)[:, zs].sum()) * 299792.458)
    row["M200c/M200m HD (geomean)"] = float(np.exp(np.mean(np.log(hd2["geomean_m200c_over_m200m"][zs]))))
    row["M200c/M200m WS c7m"] = float(np.mean(ratio_c7m_zc[zs]))
    row["M200c/M200m WS sis"] = float(np.mean(ratio_sis[zs]))
    ab[f"z{zlo}-{zhi}"] = row
res["abundance_velocity_by_z"] = ab

# ------------------------------------------------------------- report
res["websky_digitized"] = websky_dig(LREP).tolist()
res["spectra_at_ell_rep"] = {k: at(v).tolist() for k, v in spec.items()}
res["ratio_to_websky_digitized"] = {k: (at(v) / websky_dig(LREP)).tolist() for k, v in spec.items()}
res["ratio_halfdome_over_websky_catalogue_same_profile"] = {f"x{X}": (at(spec[f'halfdome_b16_x{X}']) / at(spec[f'websky_c7m_b16_x{X}'])).tolist() for X in (4, 2, 1)}
np.savez(os.path.join(OUT, "onehalo_spectra.npz"), ell=ell, **spec)
with open(os.path.join(OUT, "onehalo_summary.json"), "w") as g:
    json.dump(res, g, indent=1)
np.set_printoptions(precision=3, suppress=True, linewidth=200)
print("ell", LREP, "\nWebSky digitized", np.array(res["websky_digitized"]))
for k in spec:
    print(f"{k:45s}", np.array(res["spectra_at_ell_rep"][k]), " /WS-dig", np.array(res["ratio_to_websky_digitized"][k]))
print("HalfDome / WebSky-catalogue (same B16 profile, c7m):", res["ratio_halfdome_over_websky_catalogue_same_profile"])
print(json.dumps(dec, indent=1)); print(json.dumps(ab, indent=1))
