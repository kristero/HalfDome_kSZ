"""Velocity-weighted stack of the public WebSky total kSZ map (ksz.fits, uK) on WebSky halos, to measure the tau profile
WebSky actually painted, and compare with B16 conventions evaluated at the same pixels.
   tau_hat(theta) = sum_i (-v_i/c) dT_i(theta)/T_CMB / sum_i (v_i/c)^2      (v_i = catalogue radial velocity)
The same weighted average is formed for each model (exact per-pixel evaluation)."""
import numpy as np, healpy as hp, sys, time
sys.path.insert(0, '/home/cbllover/HalfDome/frb_map_generation/lee22_tsz_corrected/ksz_debug/websky_def')
from onehalo import *
OUT = '/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_ksz_debug_20261002/websky_def/'
TCMB = 2.7255e6
cos = Cosmo()
rng = np.random.default_rng(1)
cat = np.fromfile(OUT + 'halos_top3M.pksc', dtype=np.float32, offset=12).reshape(-1, 10).astype(np.float64)
x, y, zz, vx, vy, vz, R = (cat[:, k] for k in range(7))
chi = np.sqrt(x*x + y*y + zz*zz); vr = (x*vx + y*vy + zz*vz)/chi
za = np.linspace(0, 6, 100000); dz = za[1]-za[0]
chia = np.cumsum(C_KMS/(68*np.sqrt(0.31*(1+za)**3 + 0.69)))*dz
z = np.interp(chi, chia, za)
M200m = 4*np.pi/3*(2.775e11*0.31*0.68**2)*R**3
zt, rat = m200c_over_m200m_table(cos)
M200c = M200m*np.interp(z, zt, rat)
print('catalogue rows', len(M200m), 'min M200m %.3e' % M200m.min())
m = hp.read_map(OUT + 'websky_ksz_total_v0.0.fits', dtype=np.float64)
nside = 4096
XMAX = 8.0
xe = np.concatenate([[0], np.geomspace(0.05, XMAX, 26)]); xc = np.sqrt(np.maximum(xe[:-1], 0.02)*xe[1:])
xg = np.concatenate([[0], np.geomspace(1e-3, XMAX, 300)])
def proj_profile(Mc, zh, X=4.0, exp_form='xgp', ne_conv='xgp', comp=None, Mcomp=None):
    """projected tau(theta = x * theta200c) on grid xg for one halo, spherical truncation at X R200c."""
    rhoc = cos.rhoc(zh); R200 = (3*Mc/(4*np.pi*200*rhoc))**(1/3)
    epm = 0.9*MSUN/M_PER_E if ne_conv == 'xgp' else 0.88*MSUN/MP
    fb = cos.Ob/cos.Om
    s = np.geomspace(1e-4, X, 400)                       # LOS coordinate in R200 units
    xx = xg[:, None]; L = np.sqrt(np.maximum(X**2 - xx**2, 0))
    r = np.sqrt(xx**2 + (s[None, :]*np.minimum(1, L/X))**2)   # scale LOS grid to chord half-length
    ds_scale = np.minimum(1, L/X)[:, 0]
    shp = b16_shape(np.maximum(r, 1e-5), Mc, zh, exp_form)
    if ne_conv == 'meanb': rho = cos.Ob*cos.rhoc0*(1+zh)**3*shp
    else: rho = fb*rhoc*shp
    ne = rho*epm/MPC**3                                  # electrons / m^3 (Msun/Mpc^3 * e/Msun -> per Mpc^3 -> per m^3)
    integ = 2*np.trapz(ne*s[None, :], np.log(s), axis=1)*ds_scale*R200*MPC     # 2 * int_0^L n_e dl (m^-2)
    tau = SIGT*integ
    if comp is not None:
        rhom = cos.Om*cos.rhoc0*(1+zh)**3; R3 = (3*Mcomp/(4*np.pi*comp*rhom))**(1/3)
        n3 = fb*Mcomp*epm/(4*np.pi/3*R3**3)/MPC**3
        b = xg*R200
        tau = tau - SIGT*n3*2*np.sqrt(np.maximum(R3**2 - b**2, 0))*MPC
    return tau
BINS = [('M2e14-5e14_z0.1-0.3', 2e14, 5e14, 0.1, 0.3), ('M1e14-2e14_z0.2-0.5', 1e14, 2e14, 0.2, 0.5),
        ('M5e13-1e14_z0.3-0.7', 5e13, 1e14, 0.3, 0.7), ('M1e14-3e14_z0.8-1.2', 1e14, 3e14, 0.8, 1.2),
        ('M1e14-3e14_z1.5-2.5', 1e14, 3e14, 1.5, 2.5)]
MODELS = {'xgp': dict(), 'xgp_no0.9': dict(ne_conv='websky'), 'printed_exp': dict(exp_form='printed'),
          'printed_exp_no0.9': dict(exp_form='printed', ne_conv='websky'), 'meanbaryon': dict(ne_conv='meanb'),
          'xgp_X2': dict(X=2.0), 'xgp_comp3': dict(comp=3.0)}
NMAX = 6000
res = {}
for name, m1, m2, z1, z2 in BINS:
    sel = np.nonzero((M200c >= m1) & (M200c < m2) & (z >= z1) & (z < z2))[0]
    if len(sel) > NMAX: sel = rng.choice(sel, NMAX, replace=False)
    t0 = time.time()
    num = np.zeros(len(xc)); cnt = np.zeros(len(xc)); mod = {k: np.zeros(len(xc)) for k in MODELS}; den = 0.0
    th200 = []
    for i in sel:
        rhoc = cos.rhoc(z[i]); R200 = (3*M200c[i]/(4*np.pi*200*rhoc))**(1/3); t200 = R200/cos.DA(z[i]); th200.append(t200)
        vec = np.array([x[i], y[i], zz[i]])/chi[i]
        pix = hp.query_disc(nside, vec, XMAX*t200)
        pv = np.array(hp.pix2vec(nside, pix))
        th = np.arccos(np.clip(vec @ pv, -1, 1)); xp = th/t200
        b = np.digitize(xp, xe) - 1; ok = (b >= 0) & (b < len(xc))
        w = -vr[i]/C_KMS
        num += np.bincount(b[ok], weights=w*m[pix[ok]]/TCMB, minlength=len(xc))
        c_i = np.bincount(b[ok], minlength=len(xc)); cnt += c_i*w*w
        den += w*w
        for k, kw in MODELS.items():
            kw2 = dict(kw)
            if 'comp' in kw2: kw2['Mcomp'] = M200m[i]
            prof = proj_profile(M200c[i], z[i], **kw2)
            mod[k] += np.bincount(b[ok], weights=w*w*np.interp(xp[ok], xg, prof), minlength=len(xc))
    data = num/np.maximum(cnt, 1e-300)            # weighted mean tau per annulus
    models = {k: v/np.maximum(cnt, 1e-300) for k, v in mod.items()}
    res[name] = dict(x=xc, data=data, n=len(sel), theta200_arcmin=np.median(th200)*180*60/np.pi, **models)
    print(f'\n== {name}: N={len(sel)}  median theta200c={np.median(th200)*180*60/np.pi:.2f} arcmin  ({time.time()-t0:.0f} s)')
    bg = data[(xc > 6) & (xc < 8)].mean()
    print('   x     data       ' + '  '.join('%11s' % k for k in MODELS) + '   | data/model ratios')
    for j in range(len(xc)):
        print('%6.3f %10.3e ' % (xc[j], data[j]) + '  '.join('%11.3e' % models[k][j] for k in MODELS) + '   | ' +
              ' '.join('%5.2f' % (data[j]/models[k][j]) if models[k][j] > 0 else '  nan' for k in MODELS))
    print('   background (6<x<8) %.3e' % bg)
np.save(OUT + 'websky_tau_stacks.npy', res, allow_pickle=True)
