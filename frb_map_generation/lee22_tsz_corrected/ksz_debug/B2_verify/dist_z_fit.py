# Independent distance-redshift fit of the HalfDome lightcone catalogue (B2 verification, 2026-10-02).
import h5py, numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
f = h5py.File('/home/cbllover/HalfDome/lightcone_100.hdf5', 'r')
N = f['redshift'].shape[0]
rng = np.random.default_rng(1)
# chunked strided reads: 40 chunks of 5000 rows spread over catalogue
pos = []; z = []; a = []; L = []; lm = []; m200c=[]; m200m=[]
for s in np.linspace(0, N - 5000, 40).astype(int):
    sl = slice(s, s + 5000)
    pos.append(f['Position'][sl]); z.append(f['redshift'][sl]); a.append(f['Aemit'][sl])
    L.append(f['Length'][sl]); lm.append(f['log10_m_rfof'][sl]); m200c.append(f['halo_mass_m200c'][sl]); m200m.append(f['halo_mass_m200m'][sl])
pos = np.concatenate(pos).astype(np.float64); z = np.concatenate(z).astype(np.float64); a = np.concatenate(a).astype(np.float64)
L = np.concatenate(L); lm = np.concatenate(lm).astype(np.float64)
r = np.linalg.norm(pos, axis=1)
print('rows', len(z), 'z range', z.min(), z.max(), 'r range', r.min(), r.max())
print('mean position (origin check)', pos.mean(0))
zA = 1 / a - 1
print('redshift - (1/Aemit - 1): median %.3e, max abs %.3e' % (np.median(z - zA), np.abs(z - zA).max()))
mp = 10**lm / L
print('M_rfof/Length: median %.5e, 1-99%%: %.5e %.5e' % (np.median(mp), *np.percentile(mp, [1, 99])))
rho_c = 2.775366e11  # h^2 Msun / Mpc^3
for Om in (0.3089, 0.31, 0.3111, 0.3):
    for Lbox, Np in ((3750, 6144),):
        print('  Om=%.4f L=%d N=%d^3: m_p = %.5e Msun/h' % (Om, Lbox, Np, rho_c * Om * (Lbox / Np)**3))
print('Om implied by m_p (L=3750, N=6144):', np.median(mp) / (rho_c * (3750 / 6144)**3))
# distance-redshift fit, flat LCDM, chi in Mpc/h depends on Om only (plus radiation)
zg = np.linspace(0, 4.5, 4501)
def chi_grid(Om, Or=0.0):
    E = np.sqrt(Om * (1 + zg)**3 + Or * (1 + zg)**4 + (1 - Om - Or))
    integ = 1 / E
    c = np.concatenate([[0], np.cumsum(0.5 * (integ[1:] + integ[:-1]) * np.diff(zg))])
    return 2997.92458 * c
for zcol, lab in ((z, 'redshift'), (zA, '1/Aemit-1')):
    for Or in (0.0, 9.1e-5):
        def cost(Om):
            return np.mean((r - np.interp(zcol, zg, chi_grid(Om, Or)))**2)
        res = minimize_scalar(cost, bounds=(0.25, 0.36), method='bounded', options={'xatol': 1e-6})
        resid = r - np.interp(zcol, zg, chi_grid(res.x, Or))
        print('fit using %-10s Omega_r=%.1e: Om=%.5f  rms resid %.4f Mpc/h  max %.4f' % (lab, Or, res.x, resid.std(), np.abs(resid).max()))
        for Om in (0.3089, 0.31):
            rr = r - np.interp(zcol, zg, chi_grid(Om, Or))
            print('    Om=%.4f: mean resid %.4f Mpc/h, rms %.4f' % (Om, rr.mean(), np.sqrt(np.mean(rr**2))))
