"""How much of the one-halo kSZ power comes from HalfDome halo velocities exceeding linear theory smoothed on the
halo Lagrangian radius (what Peak-Patch/2LPT halo velocities follow)? Per (M, z) cell: ratio v_lin,RL^2 / <v_los^2>_cat,
power-weighted with the sphere-4 (and compensated) one-halo contributions."""
import numpy as np, camb
OUT = "/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_ksz_debug_20261002/painting_geometry"
h = 0.68; Ob = 0.049; Oc = 0.261; Om = Ob + Oc
pars = camb.set_params(H0=100 * h, ombh2=Ob * h**2, omch2=Oc * h**2, ns=0.965, As=2.1e-9, WantTransfer=True)
zgrid = np.linspace(0, 4, 41)
pars.set_matter_power(redshifts=list(zgrid[::-1]), kmax=50.0)
res = camb.get_results(pars)
s8 = res.get_sigma8_0(); scale = (0.81 / s8) ** 2
kh, zz, pk = res.get_matter_power_spectrum(minkh=1e-4, maxkh=50, npoints=3000); zz = np.array(zz)
k = kh * h
P0 = pk[np.argmin(zz)] * scale / h**3
Dz = np.sqrt(np.array([np.trapz(pk[i] , kh) for i in range(len(zz))]) / np.trapz(pk[np.argmin(zz)], kh))
order = np.argsort(zz); zs_sorted = zz[order]; Dz = Dz[order]
rho_m0 = Om * 2.775e11 * h**2
d = np.load(OUT + "/onehalo_scan_amplitudes.npz")
lm, z, cnt, sv2 = d["logm"], d["z"], d["count"], d["sum_v2"]
C = 299792.458
# linear 1-D sigma_v(z, R_L(M))
lmg = np.arange(12.8, 15.8, 0.05)
S0 = []
for l in lmg:
    R = (3 * 10**l / (4 * np.pi * rho_m0)) ** (1 / 3); x = k * R
    W = 3 * (np.sin(x) - x * np.cos(x)) / x**3
    S0.append(np.trapz(P0 * W**2, k) / (2 * np.pi**2) / 3)
S0 = np.array(S0)
def vlin2(l, zc):
    E = np.sqrt(Om * (1 + zc)**3 + 1 - Om); f = (Om * (1 + zc)**3 / E**2) ** 0.55
    D = np.interp(zc, zs_sorted, Dz)
    return (f * 100 * h * E / (1 + zc) * D) ** 2 * np.interp(l, lmg, S0)
ratio = np.array([vlin2(l, zc) for l, zc in zip(lm, z)]) / (sv2 / cnt * C**2)
ells = d["ells"]; XS = d["XS"]; iX4 = int(np.argmin(abs(XS - 4)))
w = sv2 / (4 * np.pi)
for lab, amp in (("sphere4", d["a_halo"][:, iX4]), ("sphere4-comp(M200m,3)", d["a_halo"][:, iX4] - d["a_sph"][:, 0]),
                 ("sphere1.75-comp(M200m,3)", d["a_halo"][:, int(np.argmin(abs(XS - 1.75)))] - d["a_sph"][:, 0])):
    p = w[:, None] * amp**2
    fac = (p * ratio[:, None]).sum(0) / p.sum(0)
    print("%-26s power factor if halo v_los^2 = linear(R_L): " % lab + " ".join("l%d=%.3f" % (L, fac[np.argmin(abs(ells - L))]) for L in (300, 1000, 3000, 5000)))
vl2 = np.array([vlin2(l, zc) for l, zc in zip(lm, z)])
print("sum(N v_lin,RL^2)/sum(v_cat^2): z<1: %.3f, z 1-2: %.3f, z>2: %.3f" % tuple((cnt[m] * vl2[m]).sum() / (sv2[m] * C**2).sum() for m in (z < 1, (z >= 1) & (z < 2), z >= 2)))
