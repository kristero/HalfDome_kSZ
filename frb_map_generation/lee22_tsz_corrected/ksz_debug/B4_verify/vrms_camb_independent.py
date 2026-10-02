#!/usr/bin/env python3
"""Independent check of CLASS-SZ get_vrms2_at_z (1-D or 3-D?).

Cosmology = the one used in truncation_comparison/compute_class_sz_references.py
(h 0.6774, Ob 0.0486, Oc 0.2603, sigma8 0.8159, ns 0.9667, N_ur 3.046, massless nu).

Method A: sigma_v,3D^2 = (f a H)^2 int P_lin(k) dk / (2 pi^2), f from CAMB fsigma8/sigma8.
Method B: CAMB Newtonian-gauge CDM / baryon velocity transfers (-k v_N / calH), i.e. no f approximation:
          sigma_v,3D^2 = int (calH/k)^2 P_vv(k) k^2 dk/(2pi^2) ... = calH^2 int P_vv(k) dk / (2 pi^2)
Velocities in km/s (peculiar, physical). 1-D = 3-D / 3.
"""
import json, sys
import numpy as np
import camb

H = 0.6774; OB = 0.0486; OC = 0.2603; S8 = 0.8159; NS = 0.9667
C_KMS = 299792.458
ZS = np.array([0.0, 0.25, 0.5, 1.0, 2.0, 3.0])

pars = camb.CAMBparams()
pars.set_cosmology(H0=100 * H, ombh2=OB * H**2, omch2=OC * H**2, mnu=0.0, num_massive_neutrinos=0, nnu=3.046)
pars.InitPower.set_params(As=2.1e-9, ns=NS)
pars.set_matter_power(redshifts=list(ZS[::-1]), kmax=50.0)
pars.NonLinear = camb.model.NonLinear_none
res = camb.get_results(pars)
s8_raw = res.get_sigma8_0()
As = 2.1e-9 * (S8 / s8_raw) ** 2
pars.InitPower.set_params(As=As, ns=NS)
res = camb.get_results(pars)
print("sigma8(z=0) after rescale:", res.get_sigma8_0())

kmax = 50.0
k = np.geomspace(1e-5, kmax, 20000)  # 1/Mpc
out = {}
fs8 = res.get_fsigma8()[::-1]   # camb returns in order of decreasing z? check below
s8z = res.get_sigma8()[::-1]
zs_camb = np.array(res.transfer_redshifts)
print("camb transfer redshifts:", zs_camb)
# get_fsigma8 / get_sigma8 return arrays ordered like transfer_redshifts (increasing-z ordering in the
# Python API corresponds to reversed internal order); map explicitly by sorting
order = np.argsort(zs_camb)
fs8_z = np.asarray(res.get_fsigma8())
s8_z = np.asarray(res.get_sigma8())
print("raw fsigma8:", fs8_z, " raw sigma8:", s8_z)

PK_dd = camb.get_matter_power_interpolator(pars, nonlinear=False, hubble_units=False, k_hunit=False,
                                           kmax=kmax, zmax=3.5, var1="delta_tot", var2="delta_tot")
PK_vc = camb.get_matter_power_interpolator(pars, nonlinear=False, hubble_units=False, k_hunit=False,
                                           kmax=kmax, zmax=3.5, var1="v_newtonian_cdm", var2="v_newtonian_cdm")
PK_vb = camb.get_matter_power_interpolator(pars, nonlinear=False, hubble_units=False, k_hunit=False,
                                           kmax=kmax, zmax=3.5, var1="v_newtonian_baryon", var2="v_newtonian_baryon")

rows = []
for z in ZS:
    a = 1.0 / (1.0 + z)
    Hz = res.hubble_parameter(z)          # km/s/Mpc
    aH = a * Hz                            # km/s/Mpc (conformal Hubble times c)
    Pdd = PK_dd.P(z, k)
    Pvc = PK_vc.P(z, k)
    Pvb = PK_vb.P(z, k)
    # f from the small-scale ratio P_vv/P_dd (sub-horizon, k ~ 0.05-0.5/Mpc)
    sel = (k > 0.05) & (k < 0.5)
    f_ratio = np.sqrt(np.median(Pvc[sel] / Pdd[sel]))
    # f from camb fsigma8/sigma8 at this z
    iz = int(np.argmin(np.abs(zs_camb - z)))
    f_fs8 = fs8_z[iz] / s8_z[iz]
    # f from Omega_m(z)^0.55
    Om_z = (OB + OC) * (1 + z)**3 * (100 * H)**2 / Hz**2
    f_lind = Om_z**0.55
    I_dd = np.trapz(Pdd, k) / (2 * np.pi**2)       # Mpc^2
    sig3_A = f_fs8 * aH * np.sqrt(I_dd)                # km/s
    # Method B: total-matter mass-weighted velocity: use CDM and baryon velocities weighted by Omega
    fc, fb = OC / (OB + OC), OB / (OB + OC)
    I_vc = np.trapz(Pvc, k) / (2 * np.pi**2)
    I_vb = np.trapz(Pvb, k) / (2 * np.pi**2)
    sig3_Bc = aH * np.sqrt(I_vc)
    sig3_Bb = aH * np.sqrt(I_vb)
    row = dict(z=float(z), f_fsigma8=float(f_fs8), f_Pvv_over_Pdd=float(f_ratio), f_Om055=float(f_lind),
               sigma3D_methodA_kms=float(sig3_A), sigma1D_methodA_kms=float(sig3_A / np.sqrt(3)),
               sigma3D_methodB_cdm_kms=float(sig3_Bc), sigma1D_methodB_cdm_kms=float(sig3_Bc / np.sqrt(3)),
               sigma1D_methodB_baryon_kms=float(sig3_Bb / np.sqrt(3)))
    rows.append(row)

# class_sz values from the 2026-09-17 reference log (vrms2/c^2)
csz = {0.0: 1.057991571659584e-06, 0.25: 1.0678686628536172e-06, 0.5: 1.0182491552234921e-06,
       1.0: 8.707031855590474e-07, 2.0: 6.291892141485e-07, 3.0: 4.825407687747477e-07}
print(f"{'z':>5} {'f_fs8':>6} {'f_Pvv':>6} {'f_Om':>6} {'3D_A':>7} {'1D_A':>7} {'1D_Bcdm':>8} {'1D_Bb':>7} {'sqrt(csz)':>9} {'csz/1D_A':>8} {'csz/3D_A':>8}")
for r in rows:
    vc = np.sqrt(csz[r['z']]) * C_KMS
    r['class_sz_sqrt_vrms2_kms'] = float(vc)
    r['ratio_csz_over_1D_A'] = float(vc / r['sigma1D_methodA_kms'])
    r['ratio_csz_over_3D_A'] = float(vc / r['sigma3D_methodA_kms'])
    print(f"{r['z']:5.2f} {r['f_fsigma8']:6.3f} {r['f_Pvv_over_Pdd']:6.3f} {r['f_Om055']:6.3f} {r['sigma3D_methodA_kms']:7.1f} "
          f"{r['sigma1D_methodA_kms']:7.1f} {r['sigma1D_methodB_cdm_kms']:8.1f} {r['sigma1D_methodB_baryon_kms']:7.1f} "
          f"{vc:9.1f} {r['ratio_csz_over_1D_A']:8.4f} {r['ratio_csz_over_3D_A']:8.4f}")

out_path = sys.argv[1] if len(sys.argv) > 1 else "vrms_camb_independent.json"
with open(out_path, "w") as fh:
    json.dump(dict(cosmology=dict(h=H, Ob=OB, Oc=OC, sigma8=S8, ns=NS), rows=rows), fh, indent=1)
print("saved", out_path)
