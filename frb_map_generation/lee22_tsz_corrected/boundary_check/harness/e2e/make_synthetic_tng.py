"""Synthetic TNG300-geometry snapshot + FoF catalogue in the Illustris HDF5 layout read by
group_particles' examples/y_prof.cpp (CallbackUtils::meta::Illustris, name::Illustris,
meta_init::IllustrisCosmology).

Box 205000 ckpc/h; groups with M200c 1e13-1e15 Msun/h and comoving R200c for z = 0-2, placed
uniformly, near low faces (x/y/z = 0), near high faces, on low edges/corners, with an exact 0
coordinate, or with a coordinate equal to the largest float below the box size.  Each group gets
an isotropic cloud of gas particles out to 2.6 R200c (float64 coordinates, as in TNG); a few
cloud particles are put exactly on the face plane (0.0, or 204999.999 which is 205000.0f after
the library's double -> float conversion).  Uniform background particles fill the box.

Writes groups_<snap>/fof_subhalo_tab_<snap>.<i>.hdf5, snapdir_<snap>/snap_<snap>.<i>.hdf5,
truth.npz (all arrays in file order), n_grp_chunks, n_prt_chunks.
"""
import argparse, os
import numpy as np
import h5py

ap = argparse.ArgumentParser()
ap.add_argument('--outdir', required=True)
ap.add_argument('--snap', default='099')
ap.add_argument('--seed', type=int, default=20261008)
ap.add_argument('--n-per-class', type=int, default=30)
ap.add_argument('--n-cloud', type=int, default=4000)
ap.add_argument('--n-background', type=int, default=300000)
ap.add_argument('--n-grp-chunks', type=int, default=3)
ap.add_argument('--n-prt-chunks', type=int, default=6)
ap.add_argument('--redshift', type=float, default=0.0)
a = ap.parse_args()

B = 205000.0
Om, OL, Ob, h = 0.3089, 0.6911, 0.0486, 0.6774
RHOC0 = 2.775e-8                    # 1e10 Msun/h / (kpc/h)^3, as in y_prof.cpp
rng = np.random.default_rng(a.seed)
f32 = np.float32
B_1ulp = float(np.nextafter(f32(B), f32(0)))

def r200c_comoving(M10, z):
    E2 = Om * (1 + z) ** 3 + OL
    return np.cbrt(3 * M10 / (4 * np.pi * 200 * RHOC0 * E2)) * (1 + z)

classes = ['uniform', 'low_face', 'high_face', 'low_edge', 'low_corner', 'high_corner', 'mixed_low_high',
           'exact_zero_coord', 'coord_Bminus1ulp']
pos, M10s, R200s, cls = [], [], [], []
for ci, c in enumerate(classes):
    for _ in range(a.n_per_class):
        M10 = 10 ** rng.uniform(np.log10(1.02e3), 5.0)
        R200 = float(f32(r200c_comoving(M10, rng.uniform(0, 2))))
        R = 2.5 * R200
        low = lambda: rng.uniform(0, R)
        high = lambda: B - rng.uniform(0, R)
        inner = lambda: rng.uniform(R + 7000.0, B - R - 7000.0)
        ax = rng.permutation(3)
        p = np.array([inner(), inner(), inner()])
        if c == 'uniform':
            p = rng.uniform(0, B, 3)
        elif c == 'low_face':
            p[ax[0]] = low()
        elif c == 'high_face':
            p[ax[0]] = high()
        elif c == 'low_edge':
            p[ax[0]] = low(); p[ax[1]] = low()
        elif c == 'low_corner':
            p[:] = [low(), low(), low()]
        elif c == 'high_corner':
            p[:] = [high(), high(), high()]
        elif c == 'mixed_low_high':
            p[ax[0]] = low(); p[ax[1]] = high()
        elif c == 'exact_zero_coord':
            p[ax[0]] = 0.0
        elif c == 'coord_Bminus1ulp':
            p[ax[0]] = B_1ulp; p[ax[1]] = low()
        pos.append(p); M10s.append(M10); R200s.append(R200); cls.append(ci)
gpos = np.array(pos, dtype=f32)
gM = np.array(M10s, dtype=f32)
gR = np.array(R200s, dtype=f32)
gcls = np.array(cls, dtype=np.int32)
Ng = len(gpos)

# particles
chunks = []
owner = []
for gi in range(Ng):
    n = a.n_cloud
    v = rng.normal(size=(n, 3)); v /= np.linalg.norm(v, axis=1)[:, None]
    r = 2.6 * float(gR[gi]) * np.cbrt(rng.uniform(size=n))
    x = gpos[gi].astype(np.float64)[None, :] + r[:, None] * v
    # every 40th particle of a face-near group: one coordinate exactly on the face plane
    for k in range(3):
        if gpos[gi, k] < 2.6 * gR[gi] or B - gpos[gi, k] < 2.6 * gR[gi]:
            sel = np.arange(n) % 40 == k
            x[sel, k] = np.where(np.arange(sel.sum()) % 2 == 0, 0.0, 204999.999)
    x = np.mod(x, B)
    chunks.append(x); owner.append(np.full(n, gi, dtype=np.int32))
chunks.append(rng.uniform(0, B, size=(a.n_background, 3))); owner.append(np.full(a.n_background, -1, dtype=np.int32))
X = np.concatenate(chunks)
own = np.concatenate(owner)
perm = rng.permutation(len(X))
X, own = X[perm], own[perm]
Np = len(X)
mass = (7.6e-4 * rng.uniform(0.8, 1.2, Np)).astype(f32)          # 1e10 Msun/h
u = rng.uniform(1e3, 1e5, Np).astype(f32)                         # (km/s)^2
xe = rng.uniform(1.0, 1.18, Np).astype(f32)

os.makedirs(os.path.join(a.outdir, f'groups_{a.snap}'), exist_ok=True)
os.makedirs(os.path.join(a.outdir, f'snapdir_{a.snap}'), exist_ok=True)
gsplit = np.array_split(np.arange(Ng), a.n_grp_chunks)
for i, s in enumerate(gsplit):
    with h5py.File(os.path.join(a.outdir, f'groups_{a.snap}', f'fof_subhalo_tab_{a.snap}.{i}.hdf5'), 'w') as f:
        hd = f.create_group('Header')
        hd.attrs['Ngroups_ThisFile'] = np.int32(len(s))
        g = f.create_group('Group')
        g['GroupPos'] = gpos[s]
        g['Group_M_Crit200'] = gM[s]
        g['Group_R_Crit200'] = gR[s]
psplit = np.array_split(np.arange(Np), a.n_prt_chunks)
z = a.redshift
for i, s in enumerate(psplit):
    with h5py.File(os.path.join(a.outdir, f'snapdir_{a.snap}', f'snap_{a.snap}.{i}.hdf5'), 'w') as f:
        hd = f.create_group('Header')
        hd.attrs['BoxSize'] = np.float64(B)
        npt = np.zeros(6, dtype=np.int32); npt[0] = len(s)
        hd.attrs['NumPart_ThisFile'] = npt
        hd.attrs['MassTable'] = np.zeros(6, dtype=np.float64)
        for k, val in [('HubbleParam', h), ('Omega0', Om), ('OmegaLambda', OL), ('OmegaBaryon', Ob),
                       ('Redshift', z), ('Time', 1 / (1 + z))]:
            hd.attrs[k] = np.float64(val)
        p0 = f.create_group('PartType0')
        p0['Coordinates'] = X[s]
        p0['Masses'] = mass[s]
        p0['InternalEnergy'] = u[s]
        p0['ElectronAbundance'] = xe[s]
np.savez(os.path.join(a.outdir, 'truth.npz'), gpos=gpos, gM=gM, gR=gR, gcls=gcls, classes=np.array(classes),
         X=X, own=own, mass=mass, u=u, xe=xe, B=B, redshift=z)
open(os.path.join(a.outdir, 'n_grp_chunks'), 'w').write(str(a.n_grp_chunks))
open(os.path.join(a.outdir, 'n_prt_chunks'), 'w').write(str(a.n_prt_chunks))
print(f'groups {Ng}, particles {Np}; float32 coordinates equal to B: {(X.astype(f32) == f32(B)).sum()}, '
      f'equal to 0: {(X == 0).sum()}')
