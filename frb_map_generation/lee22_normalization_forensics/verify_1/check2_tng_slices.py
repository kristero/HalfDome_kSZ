"""verify_1 check 2: Konietzka TNG300 within-R200 halo DM per redshift slice vs an independent
Lee22-literal halo model (hmf 3.5.2 Tinker08, SOCritical(200), CAMB transfer, TNG cosmology).

For each slice (z_{i-1}, z_i] (z_s grid of the catalogue) we compare:
  * all-ray mean DM increment      (TNG: mean of DM(z_i)-DM(z_{i-1}) over 120k rays)
  * hit fraction                    (TNG: fraction of rays with increment > 0; model 1-exp(-Nbar))
  * mean DM per crossing            (TNG: all-ray mean / hit fraction x [Nbar/(1-exp(-Nbar))] correction
                                     for overlap is applied to the MODEL side instead)
Model: dN/dz/dlnM per ray = dn/dlnM_com * dchi/dz * pi R200_com^2;  <DM>_disc = <n_e>_vol 4/3 R200/(1+z).
Mass windows interpreted as physical Msun (default) or Msun/h (sensitivity).
"""
import sys, json, warnings
import numpy as np
from scipy import integrate, interpolate, optimize
warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/cbllover/HalfDome/frb_map_generation/lee22_normalization_forensics/verify_1')
import lee22_verify_lib as L
from hmf import MassFunction
from astropy.cosmology import FlatLambdaCDM

ROOT = '/home/cbllover/HalfDome/'
OUT = ROOT + 'frb_map_generation/outputs/lee22_normalization_forensics_20260928/verify_1/'
K = ROOT + 'frb_catalog_comparison_outputs/haloDM-20260819T104507Z-1-001/haloDM/'

# ------------------------------------------------------------------ TNG side
zs = np.load(K + 'redshifts_DMhalo.npy')
NS = 30  # use slices up to z=3.0
rng = np.random.default_rng(12345)
NB = 400
tng = {}
for win in ['mass13to14', 'mass12to14', 'mass12to13_derived']:
    if win == 'mass12to13_derived':
        a = np.load(K + 'DMhalo_r200_mass12to14.npy')[:NS] - np.load(K + 'DMhalo_r200_mass13to14.npy')[:NS]
    else:
        a = np.load(K + f'DMhalo_r200_{win}.npy')[:NS]
    inc = np.diff(np.vstack([np.zeros((1, a.shape[1])), a]), axis=0)  # (NS, nray)
    hit = inc > 0
    mean_all = inc.mean(1)
    hf = hit.mean(1)
    med_hit = np.array([np.median(inc[i][hit[i]]) if hit[i].any() else np.nan for i in range(NS)])
    mean_hit = np.array([inc[i][hit[i]].mean() for i in range(NS)])
    # bootstrap over rays (same resample for all slices -> covariance)
    bs = np.empty((NB, NS))
    idx_all = rng.integers(0, a.shape[1], size=(NB, a.shape[1]))
    for b in range(NB):
        bs[b] = inc[:, idx_all[b]].mean(1)
    tng[win] = dict(mean_all=mean_all, hf=hf, med_hit=med_hit, mean_hit=mean_hit, boot=bs,
                    neg=float((a < 0).mean()))
zlo = np.concatenate([[0.0], zs[:NS - 1]]); zhi = zs[:NS]; zmid = 0.5 * (zlo + zhi)

# ------------------------------------------------------------------ model side
cosmo = FlatLambdaCDM(H0=100 * L.H, Om0=L.OM, Ob0=L.OB, Tcmb0=2.7255)
mf = MassFunction(z=0.0, Mmin=11.8, Mmax=14.4, dlog10m=0.01, hmf_model='Tinker08',
                  mdef_model='SOCritical', mdef_params={'overdensity': 200}, cosmo_model=cosmo,
                  sigma_8=0.8159, n=0.9667, transfer_model='CAMB',
                  transfer_params={'extrapolate_with_eh': True})
lmh_grid = np.log10(mf.m)  # log10 M [h^-1 Msun]

# precompute literal <nbar>(<R200c) on a (lmh, z) grid and interpolate (smooth in both)
lg = np.arange(11.8, 14.41, 0.05); zg = np.linspace(0, 3.0, 61)
NBAR = np.array([[L.mean_nbar_inside(10 ** m, z) for z in zg] for m in lg])
nbar_interp = interpolate.RectBivariateSpline(lg, zg, np.log(NBAR), kx=3, ky=3)
# check interpolation at an off-grid point
chk = np.exp(nbar_interp(13.337, 0.777)[0, 0]) / L.mean_nbar_inside(10 ** 13.337, 0.777) - 1


def chi_mpc(z):
    return cosmo.comoving_distance(z).value


def dchidz(z):
    return L.C_KMS / (100 * L.H) / np.sqrt(L.E2(z))  # Mpc (no radiation, consistent with Lee22 eq.5)


def slice_model(z1, z2, lm_lo, lm_hi, units='phys', nz=24, model='literal'):
    """returns Nbar (expected crossings per ray), mean DM per ray, mean DM per crossing"""
    zz = np.linspace(z1, z2, nz + 1); zz = 0.5 * (zz[1:] + zz[:-1]); dz = (z2 - z1) / nz
    N = 0.0; D = 0.0
    for z in zz:
        mf.update(z=z)
        Mh = mf.m; dndlnm = mf.dndlnm * L.H ** 3  # comoving Mpc^-3
        Mphys = Mh / L.H
        lm_win = np.log10(Mphys) if units == 'phys' else np.log10(Mh)
        sel = (lm_win >= lm_lo) & (lm_win < lm_hi)
        Rc = L.r200c_mpc(Mphys[sel], z) * (1 + z)
        w = dndlnm[sel] * dchidz(z) * np.pi * Rc ** 2 * dz  # crossings per ray per dlnM
        if model == 'literal':
            nb = np.exp(nbar_interp.ev(np.log10(Mh[sel]), np.full(sel.sum(), z)))
        elif model == 'fb':  # physically normalized control: M_gas = f_b M200, fully ionized
            nb = np.full(sel.sum(), 1 / L.MGAS_PER_NBAR)
        dm = L.disc_mean_dm(Mphys[sel], z, nb)
        dlnm = np.log(10) * 0.01
        N += np.sum(w) * dlnm; D += np.sum(w * dm) * dlnm
    return N, D, D / N


windows = {'mass13to14': (13.0, 14.0), 'mass12to14': (12.0, 14.0), 'mass12to13_derived': (12.0, 13.0)}
res = {}
for units in ['phys', 'hinv']:
    for win, (lo, hi) in windows.items():
        Nm = np.zeros(NS); Dm = np.zeros(NS); Dc = np.zeros(NS); Df = np.zeros(NS)
        for i in range(NS):
            Nm[i], Dm[i], Dc[i] = slice_model(zlo[i], zhi[i], lo, hi, units)
            _, Df[i], _ = slice_model(zlo[i], zhi[i], lo, hi, units, nz=8, model='fb')
        res[(units, win)] = dict(N=Nm, D=Dm, Dc=Dc, Dfb=Df)
        print('done', units, win, flush=True)


def fit_powerlaw(y, cov, x):
    """GLS fit of log y = logA + k log(1+z) using bootstrap covariance of log y"""
    X = np.vstack([np.ones_like(x), np.log(1 + x)]).T
    Ci = np.linalg.inv(cov)
    beta = np.linalg.solve(X.T @ Ci @ X, X.T @ Ci @ np.log(y))
    r = np.log(y) - X @ beta
    chi2 = r @ Ci @ r
    # constant
    X0 = X[:, :1]
    b0 = np.linalg.solve(X0.T @ Ci @ X0, X0.T @ Ci @ np.log(y)); r0 = np.log(y) - X0 @ b0
    chi20 = r0 @ Ci @ r0
    # also unweighted (diagonal-free) least squares, to not rely on bootstrap errors
    bu = np.polyfit(np.log(1 + x), np.log(y), 1)
    rms_pl = np.sqrt(np.mean((np.log(y) - np.polyval(bu, np.log(1 + x))) ** 2))
    rms_c = np.sqrt(np.mean((np.log(y) - np.log(y).mean()) ** 2))
    return dict(A=float(np.exp(beta[0])), k=float(beta[1]), chi2=float(chi2), dof=len(x) - 2,
                A_const=float(np.exp(b0[0])), chi2_const=float(chi20), dof_const=len(x) - 1,
                A_uw=float(np.exp(bu[1])), k_uw=float(bu[0]), rms_log_pl=float(rms_pl), rms_log_const=float(rms_c))


lines = []
P = lambda s: (print(s), lines.append(s))
P(f'# interpolation check of <nbar>: rel err {chk:.2e}; TNG negative values frac {tng["mass13to14"]["neg"]}')
summary = {}
for units in ['phys', 'hinv']:
    for win in windows:
        t = tng[win]; m = res[(units, win)]
        ratio = t['mean_all'] / m['D']
        ratio_fb = t['mean_all'] / m['Dfb']
        lb = np.log(t['boot'] / m['D'][None, :])
        P(f'\n## window {win}  (mass units {units})')
        P(' zlo-zhi | TNG hf  model hf | TNG mean/ray  literal mean/ray | C=TNG/lit | TNG mean/hit  TNG med/hit  lit mean/crossing | TNG/fb-control')
        for i in range(NS):
            hfm = 1 - np.exp(-m['N'][i])
            if i < 22 or i % 3 == 2:
                P(f' {zlo[i]:.1f}-{zhi[i]:.1f} | {t["hf"][i]:.4f} {hfm:.4f} | {t["mean_all"][i]:9.4f} {m["D"][i]:9.4f} | {ratio[i]:.4f} | '
                  f'{t["mean_hit"][i]:7.2f} {t["med_hit"][i]:7.2f} {m["Dc"][i] * m["N"][i] / hfm:8.2f} | {ratio_fb[i]:.3f}')
        for zr in [(0.0, 2.0), (0.0, 1.0), (0.1, 2.0), (1.0, 2.0), (0.0, 3.0)]:
            s = (zlo >= zr[0] - 1e-9) & (zhi <= zr[1] + 1e-9)
            cov = np.cov(lb[:, s].T)
            f = fit_powerlaw(ratio[s], cov, zmid[s])
            cov_fb = np.cov(np.log(t['boot'][:, s] / m['Dfb'][None, s]).T)
            ffb = fit_powerlaw(ratio_fb[s], cov_fb, zmid[s])
            key = f'{units}|{win}|{zr[0]}-{zr[1]}'
            summary[key] = dict(literal=f, fb_control=ffb)
            P(f' fit z {zr}: C = {f["A"]:.3f}(1+z)^{f["k"]:.3f}  chi2={f["chi2"]:.0f}/{f["dof"]};  const {f["A_const"]:.3f} chi2={f["chi2_const"]:.0f}/{f["dof_const"]}; '
              f'unweighted {f["A_uw"]:.3f}(1+z)^{f["k_uw"]:.3f}, rms(ln) pl {f["rms_log_pl"]:.3f} vs const {f["rms_log_const"]:.3f}')
            P(f'            fb-control: TNG/fb = {ffb["A"]:.3f}(1+z)^{ffb["k"]:.3f} chi2={ffb["chi2"]:.0f}/{ffb["dof"]} ; unweighted k {ffb["k_uw"]:.3f}')
json.dump(summary, open(OUT + 'check2_fits.json', 'w'), indent=1)
np.savez(OUT + 'check2_slices.npz', zlo=zlo, zhi=zhi,
         **{f'tng_{w}_{k}': v for w in tng for k, v in tng[w].items() if k != 'boot' and k != 'neg'},
         **{f'model_{u}_{w}_{k}': v for (u, w) in res for k, v in res[(u, w)].items()})
open(OUT + 'check2_tng_slices.txt', 'w').write('\n'.join(lines) + '\n')
