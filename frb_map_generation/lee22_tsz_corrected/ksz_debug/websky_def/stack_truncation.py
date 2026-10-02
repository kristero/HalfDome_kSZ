"""Fine-binned, pixel-averaged velocity-weighted stacks of the public WebSky kSZ map to pin down the truncation radius and the
compensation of WebSky's halo profiles.  Models (n_e = (1-Y/2) rho_gas/m_p, XGPaint B16 exponent, M200c from c200m=7) are
averaged over 16 NSIDE-16384 sub-pixels of each NSIDE-4096 pixel; fit data = A*model + B (constant pedestal) per model,
jackknife (10 groups) errors."""
import numpy as np, healpy as hp, sys, time
sys.path.insert(0, '/home/cbllover/HalfDome/frb_map_generation/lee22_tsz_corrected/ksz_debug/websky_def')
from onehalo import *
OUT = '/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_ksz_debug_20261002/websky_def/'
TCMB = 2.7255e6; cos = Cosmo(); rng = np.random.default_rng(2)
cat = np.fromfile(OUT + 'halos_top3M.pksc', dtype=np.float32, offset=12).reshape(-1, 10).astype(np.float64)
x, y, zz, vx, vy, vz, R = (cat[:, k] for k in range(7))
chi = np.sqrt(x*x + y*y + zz*zz); vr = (x*vx + y*vy + zz*vz)/chi
za = np.linspace(0, 6, 100000); dz = za[1]-za[0]
chia = np.cumsum(C_KMS/(68*np.sqrt(0.31*(1+za)**3 + 0.69)))*dz
z = np.interp(chi, chia, za)
M200m = 4*np.pi/3*(2.775e11*0.31*0.68**2)*R**3
zt, rat = m200c_over_m200m_table(cos); M200c = M200m*np.interp(z, zt, rat)
m = hp.read_map(OUT + 'websky_ksz_total_v0.0.fits', dtype=np.float32)
NS = 4096; XMAX = 7.0
xe = np.concatenate([np.arange(0, 3.0001, 0.125), [3.5, 4, 4.5, 5, 6, 7]]); xc = 0.5*(xe[1:]+xe[:-1]); nb = len(xc)
xg = np.concatenate([[0], np.geomspace(1e-3, XMAX, 400)])
def proj(Mc, zh, X, ne_conv='websky', comp=None, Mcomp=None):
    rhoc = cos.rhoc(zh); R200 = (3*Mc/(4*np.pi*200*rhoc))**(1/3)
    epm = 0.9*MSUN/M_PER_E if ne_conv == 'xgp' else 0.88*MSUN/MP
    fb = cos.Ob/cos.Om
    s = np.geomspace(1e-4, X, 400); xx = xg[:, None]; L = np.sqrt(np.maximum(X**2 - xx**2, 0))
    r = np.sqrt(xx**2 + (s[None, :]*(L/X))**2)
    ne = fb*rhoc*b16_shape(np.maximum(r, 1e-5), Mc, zh)*epm/MPC**3
    tau = SIGT*2*np.trapz(ne*s[None, :], np.log(s), axis=1)*(L[:, 0]/X)*R200*MPC
    if comp is not None:
        rhom = cos.Om*cos.rhoc0*(1+zh)**3; R3 = (3*Mcomp/(4*np.pi*comp*rhom))**(1/3)
        n3 = fb*Mcomp*epm/(4*np.pi/3*R3**3)/MPC**3
        tau = tau - SIGT*n3*2*np.sqrt(np.maximum(R3**2 - (xg*R200)**2, 0))*MPC
    return tau
MODELS = {'X1.5_comp3': (1.5, 3.0), 'X2.0_comp3': (2.0, 3.0), 'X4.0_comp3': (4.0, 3.0), 'X1.5_nocomp': (1.5, None),
          'X1.25_comp3': (1.25, 3.0), 'X1.75_comp3': (1.75, 3.0)}
BINS = [('z0.03-0.12_M>2e14', 2e14, 1e16, 0.03, 0.12), ('z0.12-0.25_M2-6e14', 2e14, 6e14, 0.12, 0.25),
        ('z0.25-0.5_M1-3e14', 1e14, 3e14, 0.25, 0.5), ('z0.5-1.0_M1-3e14', 1e14, 3e14, 0.5, 1.0)]
NMAX = 4000; NJK = 10
res = {}
for name, m1, m2, z1, z2 in BINS:
    sel = np.nonzero((M200c >= m1) & (M200c < m2) & (z >= z1) & (z < z2))[0]
    if len(sel) > NMAX: sel = rng.choice(sel, NMAX, replace=False)
    t0 = time.time()
    num = np.zeros((NJK, nb)); cnt = np.zeros((NJK, nb)); mod = {k: np.zeros(nb) for k in MODELS}; th = []
    for jj, i in enumerate(sel):
        g = jj % NJK
        R200 = (3*M200c[i]/(4*np.pi*200*cos.rhoc(z[i])))**(1/3); t200 = R200/cos.DA(z[i]); th.append(t200)
        vec = np.array([x[i], y[i], zz[i]])/chi[i]
        pix = hp.query_disc(NS, vec, XMAX*t200)
        pv = np.array(hp.pix2vec(NS, pix)); xp = np.arccos(np.clip(vec @ pv, -1, 1))/t200
        b = np.digitize(xp, xe) - 1; ok = (b >= 0) & (b < nb)
        w = -vr[i]/C_KMS
        num[g] += np.bincount(b[ok], weights=w*m[pix[ok]]/TCMB, minlength=nb)
        cnt[g] += np.bincount(b[ok], minlength=nb)*w*w
        ch = (hp.ring2nest(NS, pix)[:, None]*16 + np.arange(16)[None, :]).ravel()
        cv = np.array(hp.pix2vec(NS*4, ch, nest=True)); xs = np.arccos(np.clip(vec @ cv, -1, 1))/t200
        for k, (X, comp) in MODELS.items():
            prof = proj(M200c[i], z[i], X, comp=comp, Mcomp=M200m[i])
            pm = np.interp(xs, xg, prof, right=prof[-1]).reshape(-1, 16).mean(1)
            mod[k] += np.bincount(b[ok], weights=w*w*pm[ok], minlength=nb)
    C = cnt.sum(0); data = num.sum(0)/C
    jk = np.array([(num.sum(0)-num[g])/(C-cnt[g]) for g in range(NJK)])
    err = np.sqrt((NJK-1)/NJK*((jk - jk.mean(0))**2).sum(0))
    models = {k: v/C for k, v in mod.items()}
    print(f'\n== {name}: N={len(sel)} median theta200c={np.median(th)*180*60/np.pi:.2f} arcmin ({time.time()-t0:.0f} s)')
    print('    x      data      err  ' + ' '.join('%11s' % k for k in MODELS))
    for j in range(nb):
        print('%6.3f %9.3e %8.1e ' % (xc[j], data[j], err[j]) + ' '.join('%11.3e' % models[k][j] for k in MODELS))
    fit = {}
    for k in MODELS:
        Am = np.vstack([models[k], np.ones(nb)]).T/err[:, None]
        coef, *_ = np.linalg.lstsq(Am, data/err, rcond=None)
        chi2 = (((data - Am[:, 0]*err*coef[0] - coef[1])/err)**2).sum()
        fit[k] = (coef[0], coef[1], chi2)
        print('   fit %-12s A=%.3f  pedestal=%.2e  chi2=%.1f (dof %d)' % (k, coef[0], coef[1], chi2, nb-2))
    res[name] = dict(xc=xc, data=data, err=err, models=models, fit=fit, n=len(sel))
np.save(OUT + 'websky_truncation_stacks.npy', res, allow_pickle=True)
