"""Search for near-duplicate halos (split / double-counted FoF groups at lightcone slab boundaries): close pairs in 3-D
comoving position among HalfDome halos with 0.2 < z < 0.6, compared with their R200c (comoving)."""
import h5py, numpy as np, time
from scipy.spatial import cKDTree
f = h5py.File('/home/cbllover/HalfDome/lightcone_100.hdf5', 'r'); N = f['redshift'].shape[0]
z = f['redshift'][:]
sel = np.nonzero((z > 0.2) & (z < 0.6))[0]; lo, hi = sel[0], sel[-1] + 1
print('rows', lo, hi, hi - lo, 'contiguous:', hi - lo == len(sel), flush=True)
P = f['Position'][lo:hi].astype(np.float64); M = f['halo_mass_m200c'][lo:hi]; Z = z[lo:hi]; V = f['Velocity'][lo:hi]; mid = f['MinID'][lo:hi]
om = 0.3089
rhoc = 2.775e11 * (om * (1 + Z) ** 3 + 1 - om)            # (Msun/h)/(Mpc/h)^3 physical
r200c_com = (3 * M / (4 * np.pi * 200 * rhoc)) ** (1 / 3) * (1 + Z)   # comoving Mpc/h
t0 = time.time(); tree = cKDTree(P); pairs = tree.query_pairs(1.0, output_type='ndarray'); print('pairs <1 Mpc/h', len(pairs), time.time() - t0, flush=True)
i, j = pairs[:, 0], pairs[:, 1]
sep = np.sqrt(((P[i] - P[j]) ** 2).sum(1)); rmax = np.maximum(r200c_com[i], r200c_com[j]); rsum = r200c_com[i] + r200c_com[j]
print('halos', len(P), ' median R200c comoving', np.median(r200c_com))
for lab, m in (('sep < 0.25 Rmax', sep < 0.25 * rmax), ('sep < 0.5 Rmax', sep < 0.5 * rmax), ('sep < Rmax', sep < rmax), ('sep < Rsum', sep < rsum)):
    print(f'{lab}: {m.sum()} pairs ({m.sum() / len(P):.2e} per halo)')
q = sep < 0.5 * rmax
print('examples (sep, R200c_i, R200c_j, M_i, M_j, dz, |dv| km/s, same MinID):')
for k in np.nonzero(q)[0][:10]:
    print(f'  {sep[k]:.3f} {r200c_com[i[k]]:.3f} {r200c_com[j[k]]:.3f} {M[i[k]]:.3g} {M[j[k]]:.3g} {abs(Z[i[k]] - Z[j[k]]):.4f} {np.linalg.norm(V[i[k]] - V[j[k]]):.0f} {mid[i[k]] == mid[j[k]]}')
# fraction of one-halo-relevant mass in such pairs: weight by M^(5/3) as a crude tau^2 proxy
w = M ** (5 / 3); inpair = np.zeros(len(P), bool); inpair[i[q]] = True; inpair[j[q]] = True
print('fraction of halos in sep<0.5Rmax pairs:', inpair.mean(), ' M^(5/3)-weighted:', w[inpair].sum() / w.sum())
