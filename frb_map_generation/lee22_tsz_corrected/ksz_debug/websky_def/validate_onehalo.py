import numpy as np, h5py, sys, time
sys.path.insert(0, '/home/cbllover/HalfDome/frb_map_generation/lee22_tsz_corrected/ksz_debug/websky_def')
from onehalo import *
D = '/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_ksz_20261002/'
h = h5py.File(D + 'maps/halo_mass_redshift_histogram.h5', 'r')
le, ze = h['logm_edges'][:], h['z_edges'][:]; n = h['count'][:].T; v2 = h['sum_vlos_over_c_squared'][:].T
f = h5py.File(D + 'ksz_poisson_mass_split.h5', 'r'); ell = f['ell'][:]; c = f['cl_1h_b16'][:].T.sum(1)
cos = Cosmo()
t0 = time.time()
cl, _ = onehalo_cl(cos, le, ze, n, v2, ell)
print('time', time.time()-t0)
T2 = 2.7255e6**2
for L in [100, 300, 1000, 3000, 5000, 8000]:
    i = np.argmin(abs(ell-L)); pre = ell[i]*(ell[i]+1)/2/np.pi*T2
    print('ell %5.0f  julia %.4f  python %.4f  ratio %.4f' % (ell[i], pre*c[i], pre*cl[i], cl[i]/c[i]))
