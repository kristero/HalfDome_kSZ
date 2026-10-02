"""HalfDome vs WebSky halo catalogues in (M200c, z): abundance N(>M) and radial-velocity rms (the two catalogue inputs of
the one-halo kSZ term)."""
import numpy as np, h5py
OUT = '/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_ksz_debug_20261002/websky_def/'
D = '/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_ksz_20261002/'
h = h5py.File(D + 'maps/halo_mass_redshift_histogram.h5', 'r')
leH, zeH = h['logm_edges'][:], h['z_edges'][:]; nH = h['count'][:].T; vH = h['sum_vlos_over_c_squared'][:].T
d = np.load(OUT + 'websky_halo_hist.npz')
leW, zeW, nW, vW = d['logm_edges'], d['z_edges'], d['count_m200c'], d['sum_vr_over_c_sq_m200c']
C = 299792.458
print('HalfDome halos', nH.sum(), ' WebSky halos', nW.sum(), ' WebSky N header', d['N'])
def agg(le, ze, n, v, m1, m2, z1, z2):
    lm = 0.5*(le[1:]+le[:-1]); zc = 0.5*(ze[1:]+ze[:-1])
    s = ((lm >= m1) & (lm < m2))[:, None] & ((zc >= z1) & (zc < z2))[None, :]
    return n[s].sum(), v[s].sum()
print('\nCumulative abundance N(>M200c) ratio HalfDome/WebSky (same full sky):')
zb = [(0, 0.5), (0.5, 1), (1, 1.5), (1.5, 2), (2, 3), (3, 3.9)]
print('log10M>   ' + ' '.join('%11s' % ('z%.1f-%.1f' % b) for b in zb))
for m in [13.0, 13.25, 13.5, 13.75, 14.0, 14.25, 14.5, 14.75, 15.0]:
    row = []
    for z1, z2 in zb:
        a, _ = agg(leH, zeH, nH, vH, m, 16, z1, z2); b, _ = agg(leW, zeW, nW, vW, m, 16, z1, z2)
        row.append('%5.2f(%6.0e)' % (a/b if b > 0 else np.nan, b))
    print('%5.2f    ' % m + ' '.join(row))
print('\nRadial velocity rms [km/s] HalfDome / WebSky, by mass and z:')
for m1, m2 in [(13, 13.5), (13.5, 14), (14, 14.5), (14.5, 15.5)]:
    row = []
    for z1, z2 in zb:
        a, va = agg(leH, zeH, nH, vH, m1, m2, z1, z2); b, vb = agg(leW, zeW, nW, vW, m1, m2, z1, z2)
        row.append('%4.0f/%4.0f' % (np.sqrt(va/max(a, 1))*C, np.sqrt(vb/max(b, 1))*C))
    print('M %4.1f-%4.1f ' % (m1, m2) + ' '.join('%11s' % r for r in row))
zc = 0.5*(zeW[1:]+zeW[:-1])
print('\nWebSky all-halo v_r rms and 3D/sqrt3 rms by z:')
for z1, z2 in [(0, 0.5), (0.5, 1), (1, 2), (2, 3), (3, 4.6)]:
    s = (zc >= z1) & (zc < z2)
    print('z %.1f-%.1f  v_r rms %.0f  sqrt(<v3d^2>/3) %.0f  N %.3e' % (z1, z2, np.sqrt(d['vz_sum_vr2'][s].sum()/d['vz_count'][s].sum()),
          np.sqrt(d['vz_sum_v3d2'][s].sum()/d['vz_count'][s].sum()/3), d['vz_count'][s].sum()))
# kSZ-weighted catalogue factor: sum over cells of (v/c)^2 * M^p for p ~ 2.6 (tau ~ M^1.3)
for p in [2.0, 2.6]:
    for z1, z2 in [(0, 1), (1, 2), (2, 4.6)]:
        out = []
        for le, ze, n, v in ((leH, zeH, nH, vH), (leW, zeW, nW, vW)):
            lm = 0.5*(le[1:]+le[:-1]); zcc = 0.5*(ze[1:]+ze[:-1])
            s = ((lm >= 13))[:, None] & ((zcc >= z1) & (zcc < z2))[None, :]
            out.append((v*(10**lm[:, None])**p)[s].sum())
        print('sum (v/c)^2 M^%.1f (M>1e13) z%.0f-%.1f: HalfDome/WebSky = %.3f' % (p, z1, z2, out[0]/out[1]))
