"""Linear-theory halo velocity rms vs redshift for the HalfDome cosmology (2026-10-02, ksz_debug/velocities).

sigma_v,1D(M, z) = a H(z) f(z) sigma_{-1}(R_L, z) sqrt(1 - sigma_0^4 / (sigma_1^2 sigma_{-1}^2)) / sqrt(3)
with sigma_j^2 = (1/2pi^2) int k^{2+2j} P_lin(k, z) W^2(k R_L) dk, R_L the Lagrangian radius of mass M (top-hat or Gaussian).
Also the unsmoothed rms (W = 1, no peak factor). Units: k [h/Mpc], P [(Mpc/h)^3], H = 100 E(z) km/s/(Mpc/h) -> km/s.
Writes linear_sigma_v.npz to the debug output dir.
"""
import numpy as np, camb, os, sys
from scipy.integrate import simpson

OUT = "/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_ksz_debug_20261002/velocities"
h = 0.68; Ob = 0.049; Om = 0.31; ns = 0.965; s8_target = 0.81
pars = camb.CAMBparams()
pars.set_cosmology(H0=100 * h, ombh2=Ob * h**2, omch2=(Om - Ob) * h**2, mnu=0.0, omk=0)
pars.InitPower.set_params(As=2.1e-9, ns=ns)
zs = np.array(sorted(np.round(np.arange(0.0, 4.01, 0.1), 3).tolist(), reverse=True))
pars.set_matter_power(redshifts=zs.tolist(), kmax=200.0)
pars.NonLinear = camb.model.NonLinear_none
res = camb.get_results(pars)
s8_0 = res.get_sigma8_0()
pars.InitPower.set_params(As=2.1e-9 * (s8_target / s8_0) ** 2, ns=ns)
res = camb.get_results(pars)
print("sigma8(z=0) =", res.get_sigma8_0())
kh, zz, pk = res.get_matter_power_spectrum(minkh=1e-5, maxkh=150.0, npoints=4000)   # zz ascending
fs8 = res.get_fsigma8()[::-1]; s8z = res.get_sigma8()[::-1]                            # reorder to ascending z
f = fs8 / s8z
lnk = np.log(kh)

def W_th(x):
    x = np.maximum(x, 1e-8)
    return 3 * (np.sin(x) - x * np.cos(x)) / x**3

def W_g(x):
    return np.exp(-0.5 * x**2)

def moments(P, R, W):
    w2 = W(kh * R) ** 2
    s = {}
    for j in (-1, 0, 1):
        s[j] = simpson(kh ** (3 + 2 * j) * P * w2, x=lnk) / (2 * np.pi**2)
    return s

rho_m = 2.775e11 * Om          # (Msun/h) / (Mpc/h)^3
logM_h = np.arange(12.5, 15.51, 0.25)  # log10 M [Msun/h]
out = {"z": zz, "f": f, "logM_msun_h": logM_h}
sig_unsm = np.zeros(len(zz)); sig_th = np.zeros((len(zz), len(logM_h))); sig_th_pk = np.zeros_like(sig_th)
sig_g_pk = np.zeros_like(sig_th)
for iz, z in enumerate(zz):
    a = 1 / (1 + z)
    Hz = 100 * res.hubble_parameter(z) / (100 * h)   # km/s/(Mpc/h)
    pref = a * Hz * f[iz]
    P = pk[iz]
    sig_unsm[iz] = pref * np.sqrt(simpson(kh * P, x=lnk) / (2 * np.pi**2)) / np.sqrt(3)
    for im, lm in enumerate(logM_h):
        R = (3 * 10**lm / (4 * np.pi * rho_m)) ** (1 / 3)
        s = moments(P, R, W_th)
        sig_th[iz, im] = pref * np.sqrt(s[-1]) / np.sqrt(3)
        sig_th_pk[iz, im] = sig_th[iz, im] * np.sqrt(1 - s[0] ** 2 / (s[1] * s[-1]))
        sg = moments(P, R / np.sqrt(5), W_g)   # Gaussian with the same 'mass' to second order (R_G = R_TH/sqrt5)
        sig_g_pk[iz, im] = pref * np.sqrt(sg[-1]) * np.sqrt(1 - sg[0] ** 2 / (sg[1] * sg[-1])) / np.sqrt(3)
out.update(sigma_v1d_unsmoothed=sig_unsm, sigma_v1d_tophat=sig_th, sigma_v1d_tophat_peak=sig_th_pk, sigma_v1d_gauss_peak=sig_g_pk,
           aHf=np.array([1 / (1 + z) * 100 * res.hubble_parameter(z) / (100 * h) * f[i] for i, z in enumerate(zz)]))
os.makedirs(OUT, exist_ok=True)
np.savez(os.path.join(OUT, "linear_sigma_v.npz"), **out)
print(" z     f     1D unsm  | 1D top-hat+peak for log10 M[Msun/h] =", logM_h[[2, 4, 6, 8]])
for iz, z in enumerate(zz):
    if abs(z * 10 - round(z * 10)) < 1e-6 and round(z * 10) % 2 == 0:
        print("%4.1f %5.3f %7.1f   | %s" % (z, f[iz], sig_unsm[iz], " ".join("%6.1f" % sig_th_pk[iz, k] for k in (2, 4, 6, 8))))
