"""map / one-halo factors (review 2026-10-02).
1. Measured on the earlier maps (h 0.68 setup, 4 R200c, b16 / lee22corr / lee22pipe): F = anafast D_ell / one-halo D_ell, both averaged
   over the same +-10 % ell window (onehalo_old.npz from this engine with the fine centroid histogram).
2. Crude Limber two-halo model, used only as a ratio to transfer F to the WebSky-recipe variants (compensated, 1.5 R200c, cut):
     C2h(ell) ~ sum_shells [sum_cells N b tau_ell]^2 P_lin(ell/chi, z) <v_los^2>(z) / (16 pi^2 chi^2 dchi)
   (tau-weighted halo field times a coherent bulk velocity; Tinker+10 bias at M200m, CAMB linear P(k) in HalfDome's cosmology).
   F_ws - 1 = (F_phys,measured - 1) x [C2h/C1h]_ws / [C2h/C1h]_phys."""
import sys, os
import numpy as np, camb
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

OUT = "/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_ksz_review_20261002/prediction/"
ELL_T = np.array([300., 500., 1000., 2000., 3000., 5000.])
old = np.load(OUT + "onehalo_old.npz"); new = np.load(OUT + "onehalo_new.npz")
spec = np.load("/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_ksz_20261002/spectra/ksz_spectra.npz")
lm_ = spec["ell"].astype(float)


def window_mean_map(cl, L, frac=0.1):
    s = (lm_ >= L * (1 - frac)) & (lm_ <= L * (1 + frac))
    return dl_from_cl(lm_[s], cl[s]).mean()


def window_mean_1h(ell, cl, L, frac=0.1):
    e = np.arange(int(np.ceil(L * (1 - frac))), int(np.floor(L * (1 + frac))) + 1).astype(float)
    return dl_from_cl(e, np.exp(np.interp(np.log(e), np.log(ell), np.log(cl)))).mean()


F = {}
print("1. earlier maps (h 0.68 setup): window-averaged (+-10 %) D_ell [uK^2]")
for n in ("b16", "lee22corr", "lee22pipe"):
    m = np.array([window_mean_map(spec["cl_" + n], L) for L in ELL_T])
    o = np.array([window_mean_1h(old["ell"], old["cl_" + n], L) for L in ELL_T])
    F[n] = m / o
    print(f"  {n:10s} map      " + " ".join(f"{x:.4f}" for x in m))
    print(f"  {n:10s} 1h       " + " ".join(f"{x:.4f}" for x in o))
    print(f"  {n:10s} map/1h   " + " ".join(f"{x:.3f}" for x in F[n]))
np.savez(OUT + "map_over_onehalo_earlier.npz", ell=ELL_T, **{"F_" + k: v for k, v in F.items()})

# ---------------------------------------------------------------- two-halo ratio model
h, Ob, Oc, s8, ns = 0.6774, 0.0486, 0.2603, 0.8159, 0.9667
cos = Cosmo(h, Ob, Oc)
pars = camb.CAMBparams()
pars.set_cosmology(H0=100 * h, ombh2=Ob * h * h, omch2=Oc * h * h, mnu=0.0, num_massive_neutrinos=0, omk=0)
pars.InitPower.set_params(ns=ns, As=2.1e-9)
zg = np.linspace(0, 4, 41)
pars.set_matter_power(redshifts=list(zg[::-1]), kmax=50.0)
pars.NonLinear = camb.model.NonLinear_none
res = camb.get_results(pars)
kh, zz, pk = res.get_matter_power_spectrum(minkh=1e-4, maxkh=50, npoints=1500)
pk = pk * (s8 / res.get_sigma8()[-1]) ** 2           # (Mpc/h)^3, rows ordered by zz increasing
zz = np.array(zz)
k = kh * h; P = pk / h ** 3                              # 1/Mpc, Mpc^3
rho_m = cos.Om * cos.rhoc0 * MPC ** 3 / MSUN             # Msun / Mpc^3 comoving


def sigma_M(M, iz):
    R = (3 * M / (4 * np.pi * rho_m)) ** (1 / 3)
    x = np.outer(R, k)
    W = 3 * (np.sin(x) - x * np.cos(x)) / x ** 3
    return np.sqrt(np.trapz(P[iz] * W ** 2 * k ** 3 / (2 * np.pi ** 2), np.log(k), axis=1))


def tinker10_bias(nu, Delta=200.0):
    y = np.log10(Delta)
    A = 1 + 0.24 * y * np.exp(-(4 / y) ** 4); a = 0.44 * y - 0.88
    B = 0.183; b = 1.5; C = 0.019 + 0.107 * y + 0.19 * np.exp(-(4 / y) ** 4); c = 2.4
    dc = 1.686
    return 1 - A * nu ** a / (nu ** a + dc ** a) + B * nu ** b + C * nu ** c


def two_halo_ratio(name, ells, zt, rt, m200c_is_ws):
    lm, z, w, n = new["cells_" + name]; t = new["tau_" + name]; e = new["ell"]
    M = 10 ** lm
    m200m = M / np.interp(z, zt, rt)
    edges = np.arange(0, 4.0001, 0.05)
    c2 = np.zeros(len(e)); c1 = (w[:, None] * t ** 2).sum(0) / (4 * np.pi)
    vz = (w / np.maximum(n, 1))
    for a, b in zip(edges[:-1], edges[1:]):
        s = (z >= a) & (z < b)
        if not s.any():
            continue
        zc = np.average(z[s], weights=n[s]); iz = np.argmin(np.abs(zz - zc))
        nu = 1.686 / sigma_M(m200m[s], iz)
        bias = tinker10_bias(nu)
        S = (n[s, None] * bias[:, None] * t[s]).sum(0)
        chi = cos.chi(zc); dchi = cos.chi(b) - cos.chi(a)
        v2 = w[s].sum() / n[s].sum()
        Pk = np.interp(np.log(e / chi), np.log(k), P[iz])
        c2 += S ** 2 * Pk * v2 / (16 * np.pi ** 2 * chi ** 2 * dchi)
    return np.array([np.interp(L, e, c2 / c1) for L in ells])


d = np.load(OUT + "halfdome_hist_h06774.npz"); zt, rt = d["ratio_z"], d["ratio_c7"]
print("\n2. crude two-halo model C2h/C1h (new setup) and transferred factors")
R = {n: two_halo_ratio(n, ELL_T, zt, rt, n.endswith("websky")) for n in
     ("b16_phys", "lee22corr_phys", "lee22pipe_phys", "b16_websky", "lee22corr_websky")}
for n, v in R.items():
    print(f"  model C2h/C1h {n:18s} " + " ".join(f"{x:.3f}" for x in v))
out = {}
for n, ref in (("b16_phys", "b16"), ("lee22corr_phys", "lee22corr"), ("lee22pipe_phys", "lee22pipe"),
               ("b16_websky", "b16"), ("lee22corr_websky", "lee22corr")):
    phys = ref + "_phys" if not n.endswith("_phys") else n
    Fx = 1 + (F[ref] - 1) * R[n] / R[phys]
    out[n] = Fx
    print(f"  expected map/1h {n:18s} " + " ".join(f"{x:.3f}" for x in Fx) + f"   (measured {ref} phys - 1 scaled by model ratio)")
print("  model C2h/C1h vs measured (F-1) on phys: " + "; ".join(
    f"{r}: " + " ".join(f"{a:.2f}" for a in R[r + '_phys'] / np.maximum(F[r] - 1, 1e-3)) for r in ("b16", "lee22corr", "lee22pipe")))
np.savez(OUT + "map_factor.npz", ell=ELL_T, **{"F_" + k: v for k, v in out.items()}, **{"R2h_" + k: v for k, v in R.items()})
