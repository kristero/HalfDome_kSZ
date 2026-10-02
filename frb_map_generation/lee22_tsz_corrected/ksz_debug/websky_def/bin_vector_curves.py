import numpy as np, pickle
OUT = '/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_ksz_debug_20261002/websky_def/'
C = {k: np.loadtxt(OUT + f'stein2020_fig6_vector_{k}.csv', delimiter=',') for k in ['halo', 'field', 'total']}
for k in C: C[k] = C[k][C[k][:, 0] > 10]
ell = np.round(C['halo'][:, 0]).astype(int)
edges = np.unique(np.round(np.geomspace(49, 5765, 41)).astype(int))
rows = []
for a, b in zip(edges[:-1], edges[1:]):
    m = (ell >= a) & (ell < b)
    rows.append([np.sqrt(a*(b-1)), a, b-1] + [C[k][m, 1].mean() for k in ['halo', 'field', 'total']])
rows = np.array(rows)
np.savetxt(OUT + 'stein2020_fig6_vector_binned.csv', rows, delimiter=',', fmt='%.6g',
           header='ell_center,ell_min,ell_max,D_halo_uK2,D_field_uK2,D_total_uK2  (Stein+2020 JCAP Fig.6, exact PDF marker positions, '
                  'per-ell C_ell averaged in bins; total = anafast of public ksz.fits, NSIDE 4096, no pixel-window correction)')
print('binned (selected):')
for L in [100, 300, 500, 1000, 2000, 3000, 4000, 5000]:
    m = np.abs(np.log(ell/L)) < 0.1
    print('ell %5d  halo %.4f  field %.4f  total %.4f  total-halo-field %.4f' % (L, *[C[k][m, 1].mean() for k in ['halo', 'field', 'total']],
          C['total'][m, 1].mean() - C['halo'][m, 1].mean() - C['field'][m, 1].mean()))
# old digitization vs exact
old = np.loadtxt('/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_ksz_20261002/websky_halo_stein2020_fig6_digitized_binned.csv', delimiter=',')
ex = np.array([C['halo'][np.abs(np.log(ell/L)) < 0.03, 1].mean() for L in old[:, 0]])
r = old[:, 1]/ex
print('\nold CSV / exact (+-3%% ell avg): min %.3f max %.3f median %.3f' % (r.min(), r.max(), np.median(r)))
for i in range(0, len(old), 6): print('  ell %.0f old %.4f exact %.4f ratio %.3f' % (old[i, 0], old[i, 1], ex[i], r[i]))
p = pickle.load(open('/tmp/claude-0/-home-cbllover-HalfDome/015746ae-94b7-42cb-8dc5-0e8c33dc8490/scratchpad/stein2020_fig6_digitized.pkl', 'rb'))
print('old pickle Battaglia point', p['battaglia2010_point'])
print('halo D crosses the axis floor 0.1 at ell ~', ell[np.argmax(np.convolve(C['halo'][:, 1], np.ones(21)/21, 'same') > 0.1)])
