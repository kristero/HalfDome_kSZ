"""B1 verification, step 2: effect of the low-z cache error on the catalogue one-halo kSZ power.
Per-halo (actual z, actual v_los) for z < 0.03, fine (dz = 0.0025, dlogM = 0.025) bins for 0.03 <= z < 0.1, and the
production (log10 M, z) histogram (bin centres) for z >= 0.1.  Profiles: (a) production cache (natural-spline replica,
checked against the Julia interpolator), (b) direct numpy chord integral.  Full-sky Legendre transform:
tau_ell = 2 pi int tau(theta) P_ell(cos theta) sin theta dtheta;  C_ell = (1/4pi) sum (v/c)^2 tau_ell^2."""
import sys, time
sys.path.insert(0, '.')
import numpy as np, h5py
from scipy.special import eval_legendre
from scipy.interpolate import CubicSpline
import b16_direct_np as b

b._GL_U, b._GL_W = np.polynomial.legendre.leggauss(100)
OUT = '/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_ksz_debug_20261002/verify_B1/'
CACHE = '/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_ksz_20261002/caches/tau_b16_sphere4p0_chordmean_interpolator.jld2'
HIST = '/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_ksz_20261002/maps/halo_mass_redshift_histogram.h5'
CAT = '/home/cbllover/HalfDome/lightcone_100.hdf5'
ELLS = np.array([30, 61, 124, 253, 500, 1000])
NU = 800
U, WU = np.polynomial.legendre.leggauss(NU); U = 0.5 * (U + 1); WU = 0.5 * WU   # u in [0,1]
X = b.XSPH
XG = X * (1 - U**2); DXG = 2 * X * U * WU                                          # x = X(1-u^2), dx = 2Xu du
LG = np.sqrt((X - XG) * (X + XG))
cr = b.CacheReplica(CACHE)
t0 = time.time()

# z -> d_A table (fine), interpolated (cubic) for speed
ztab = np.concatenate([np.linspace(1e-4, 0.2, 4001), np.linspace(0.2005, 4.1, 3000)])
dAtab = b.dA_mpc(ztab)
dA_of = CubicSpline(ztab, dAtab)


def tau_ell(M, z, which, slab=None, row=None):
    """Returns tau_ell (len(ELLS)) for the painted tau inside 4 R200c; which = 'direct' or 'cache'."""
    dA = float(dA_of(z))
    th200 = float(b.theta200(M, z, dA))
    th = XG * th200
    if which == 'direct':
        _, xc, al, be, ga = b.params(M, z)
        tau = b.amplitude(M, z) * b.chord_integral(XG, al, be, xc, ga)
    else:
        if row is None:
            row = CubicSpline(cr.logM, slab, axis=0, bc_type='natural')(np.log10(M))
        g = np.exp(CubicSpline(cr.logth, row, bc_type='natural')(np.log(th)))
        tau = g * 2 * LG
    w = 2 * np.pi * tau * np.sin(th) * th200 * DXG
    ct = np.cos(th)
    return np.array([np.sum(w * eval_legendre(l, ct)) for l in ELLS])



with h5py.File(HIST, 'r') as f:
    cnt = f['count'][()]; sv2 = f['sum_vlos_over_c_squared'][()]; le = f['logm_edges'][()]; ze = f['z_edges'][()]
prev = np.load(OUT + 'step2_onehalo_effect.npz')
cl_hi = prev['cl_hi']
lo = {'direct': np.zeros(len(ELLS)), 'cache': np.zeros(len(ELLS))}
for iz in range(4):
    z = 0.5 * (ze[iz] + ze[iz + 1]); slab = cr.eval_z(z)
    ims = np.where(cnt[iz] > 0)[0]
    rows = CubicSpline(cr.logM, slab, axis=0, bc_type='natural')(0.5 * (le[ims] + le[ims + 1]))
    for j, im in enumerate(ims):
        M = 10**(0.5 * (le[im] + le[im + 1]))
        for which in ('direct', 'cache'):
            t = tau_ell(M, z, which, row=rows[j])
            lo[which] += sv2[iz, im] * t**2 / (4 * np.pi)
T2 = 2.7255e6**2
D = lambda cl: ELLS * (ELLS + 1) * cl / (2 * np.pi) * T2
tc = cl_hi + lo['cache']; td = cl_hi + lo['direct']
lines = ['BIN-CENTRE variant (z<0.1 from histogram bins of dz=0.025 at their centres):',
         'ell: ' + ' '.join(f'{l:>9d}' for l in ELLS),
         'D_ell cache  [uK^2]: ' + ' '.join(f'{v:9.4g}' for v in D(tc)),
         'D_ell direct [uK^2]: ' + ' '.join(f'{v:9.4g}' for v in D(td)),
         'cache/direct:        ' + ' '.join(f'{v:9.4f}' for v in tc / td),
         'direct/cache:        ' + ' '.join(f'{v:9.4f}' for v in td / tc)]
with h5py.File('/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_ksz_20261002/ksz_poisson_mass_split.h5', 'r') as f:
    ell_p = f['ell'][()]; clp = f['cl_1h_b16'][()]
clp_tot = clp.sum(axis=0) if clp.shape[0] != len(ell_p) else clp.sum(axis=1)
Dp = ell_p * (ell_p + 1) * clp_tot / (2 * np.pi) * T2
lines.append('production poisson_mass_split (flat-sky, bin centres, cache) D_ell at ELLS (log-interp): ' +
             ' '.join(f'{v:9.4g}' for v in np.exp(np.interp(np.log(ELLS), np.log(ell_p), np.log(Dp)))))
txt = '\n'.join(lines); print(txt); open(OUT + 'step3_bincentre_variant.txt', 'w').write(txt + '\n')
