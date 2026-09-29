"""Independent analytic mean halo DM (1e13<=M200c/Msun<1e14, R200c sphere) vs Konietzka TNG300.

<DM>(z_s) = int_0^{z_s} dz c/H(z) (1+z) int dlnM dn/dlnM(comoving) N_e(M,z)   [exact, no projection]
Nbar(z_s) = int dz chi^2 c/H int dlnM dn/dlnM pi theta200^2,  theta200 = R200c/D_A
Per-crossing DM distribution from the chord integral with x^2 uniform on [0,1].
"""
import json, math, numpy as np
from common import TNG, Tinker08, lee22_volume_mean_nbar, lee22_chord_dm_literal, n200_cm3, XGP_OVER_LITERAL, MSUN_KG, MP_KG, MPC_M, PC_M, X_H

OUT = '/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_normalization_forensics_20260928/verify_4/'
KDIR = '/home/cbllover/HalfDome/frb_catalog_comparison_outputs/haloDM-20260819T104507Z-1-001/haloDM/'
c = TNG
t08 = Tinker08(c)

# conversion: DM contribution of N_e electrons (in units Msun/m_p) per comoving Mpc^3 over c/H dz (Mpc)
# n_e,com [cm^-3] = N_e[Msun/m_p] * MSUN_KG/MP_KG / (MPC_M*100)^3 ; path in pc: Mpc*1e6
NE_TO_CM3 = MSUN_KG / MP_KG / (MPC_M * 100.0) ** 3
MPC_TO_PC = 1e6

dz = 0.005
zmid = np.arange(dz / 2, 2.0, dz)
lgm = np.arange(13.0 + 0.0125, 14.0, 0.025)  # dlog10M = 0.025
dlnm = 0.025 * math.log(10)
mass = 10 ** lgm

dmdz_lit = np.zeros_like(zmid)   # d<DM>/dz literal (all rays)
dNdz = np.zeros_like(zmid)       # expected crossings per ray per dz
for i, z in enumerate(zmid):
    hmf = t08.dndlnm(mass, z)                          # comoving Mpc^-3
    I = np.array([lee22_volume_mean_nbar(m, z, c.h) for m in mass])
    Ne = c.fb * mass / X_H * I                          # Msun/m_p units
    dmdz_lit[i] = c.c_over_H(z) * (1 + z) * np.sum(hmf * Ne * dlnm) * NE_TO_CM3 * MPC_TO_PC
    DA = c.chi(z) / (1 + z)
    th = c.r200c(mass, z) / DA
    dNdz[i] = c.chi(z) ** 2 * c.c_over_H(z) * np.sum(hmf * math.pi * th ** 2 * dlnm)

cum_lit = np.cumsum(dmdz_lit * dz)
cum_N = np.cumsum(dNdz * dz)
zedge = zmid + dz / 2

zs_grid = np.load(KDIR + 'redshifts_DMhalo.npy')
tng = np.load(KDIR + 'DMhalo_r200_mass13to14.npy')
res = {'zs': [], 'tng_mean_all': [], 'tng_hit': [], 'tng_mean_hit': [], 'lit_mean_all': [], 'xgp_mean_all': [],
       'poisson_hit': [], 'xgp_mean_hit_poisson': [], 'xgp_mean_hit_tnghit': [], 'ratio_tng_over_xgp_all': [],
       'ratio_tng_over_xgp_hit': []}
for j, zs in enumerate(zs_grid):
    if zs > 2.0 + 1e-9:
        break
    k = np.argmin(abs(zedge - zs)); assert abs(zedge[k] - zs) < 1e-9
    lit = cum_lit[k]; xgp = XGP_OVER_LITERAL * lit
    ph = 1 - math.exp(-cum_N[k])
    t = tng[j]; th = (t > 0).mean()
    res['zs'].append(float(zs)); res['tng_mean_all'].append(float(t.mean())); res['tng_hit'].append(float(th))
    res['tng_mean_hit'].append(float(t[t > 0].mean())); res['lit_mean_all'].append(float(lit)); res['xgp_mean_all'].append(float(xgp))
    res['poisson_hit'].append(ph); res['xgp_mean_hit_poisson'].append(xgp / ph); res['xgp_mean_hit_tnghit'].append(xgp / th)
    res['ratio_tng_over_xgp_all'].append(float(t.mean() / xgp)); res['ratio_tng_over_xgp_hit'].append(float(t[t > 0].mean() / (xgp / ph)))

print(f"{'zs':>4} {'TNGall':>7} {'XGPall':>7} {'T/X':>5} | {'TNGhit':>6} {'Pois':>6} | {'TNG/hit':>7} {'XGP/hit':>7} {'T/X':>5} | {'XGP/TNGhit':>9}")
for q in range(len(res['zs'])):
    print(f"{res['zs'][q]:4.1f} {res['tng_mean_all'][q]:7.2f} {res['xgp_mean_all'][q]:7.2f} {res['ratio_tng_over_xgp_all'][q]:5.3f} | "
          f"{res['tng_hit'][q]:6.4f} {res['poisson_hit'][q]:6.4f} | {res['tng_mean_hit'][q]:7.2f} {res['xgp_mean_hit_poisson'][q]:7.2f} "
          f"{res['ratio_tng_over_xgp_hit'][q]:5.3f} | {res['xgp_mean_hit_tnghit'][q]:9.2f}")

# per-shell C(z) = Delta TNG / Delta literal (all-ray means)
zs = np.array(res['zs']); T = np.array(res['tng_mean_all']); L = np.array(res['lit_mean_all'])
dT = np.diff(np.concatenate([[0], T])); dL = np.diff(np.concatenate([[0], L]))
C = dT / dL; zc = zs - 0.05
print('\nshell  C=dTNG/dlit   dTNG   dlit')
for a, b, d1, d2 in zip(zc, C, dT, dL):
    print(f'{a:5.2f} {b:7.3f} {d1:7.3f} {d2:7.3f}')
out = {'per_shell': {'zmid': zc.tolist(), 'C': C.tolist(), 'dTNG': dT.tolist(), 'dlit': dL.tolist()}}
for zmax in (1.0, 2.0):
    s = zc < zmax
    # weighted least squares in log with Poisson-ish weights ~ dlit (just unweighted here + weighted)
    p = np.polyfit(np.log(1 + zc[s]), np.log(C[s]), 1)
    pw = np.polyfit(np.log(1 + zc[s]), np.log(C[s]), 1, w=np.sqrt(dT[s]))
    zcross = math.exp((math.log(XGP_OVER_LITERAL) - p[1]) / p[0]) - 1
    zcrossw = math.exp((math.log(XGP_OVER_LITERAL) - pw[1]) / pw[0]) - 1
    print(f'fit shells z<{zmax}: C = {math.exp(p[1]):.3f} (1+z)^{p[0]:.2f} (unweighted), zcross(0.6016)={zcross:.3f}; '
          f'weighted: {math.exp(pw[1]):.3f}(1+z)^{pw[0]:.2f}, zcross={zcrossw:.3f}')
    out[f'fit_z<{zmax}'] = {'A': math.exp(p[1]), 'k': p[0], 'zcross': zcross, 'A_w': math.exp(pw[1]), 'k_w': pw[0], 'zcross_w': zcrossw}
    # constant fit chi^2-like comparison of residual RMS
    rms_pl = np.sqrt(np.mean((np.log(C[s]) - np.polyval(p, np.log(1 + zc[s]))) ** 2))
    rms_c = np.sqrt(np.mean((np.log(C[s]) - np.mean(np.log(C[s]))) ** 2))
    print(f'   rms log-residual power law {rms_pl:.3f}  constant {rms_c:.3f}')
    out[f'fit_z<{zmax}'].update({'rms_log_pl': rms_pl, 'rms_log_const': rms_c})
# z_s = 1 literal-DM-weighted mean of C = cumulative TNG / literal
k1 = np.argmin(abs(zs - 1.0))
wC = float(np.sum(C[:k1 + 1] * dL[:k1 + 1]) / np.sum(dL[:k1 + 1]))
print('z_s=1 literal-weighted <C> =', wC, ' = cumulative TNG/lit =', T[k1] / L[k1])
out['zs1_weighted_C'] = wC
out['cumulative'] = res
np.savez(OUT + 'analytic_mean_dm_grid.npz', zmid=zmid, dmdz_lit=dmdz_lit, dNdz=dNdz)
json.dump(out, open(OUT + 'analytic_mean_dm.json', 'w'), indent=1)
