#!/usr/bin/env python
"""Python replica of the Lee22-era group_particles profile extraction (y_prof.cpp / ne_prof.cpp, c3fe9b8) on one
raw CAMELS IllustrisTNG snapshot, with the 'fixed' (plain minimum-image) and the 'buggy' (low-face pairs dropped)
pair sets, for several periodic translations of the box (shift 0 = the actual CAMELS box faces).

Per group (Group_M_Crit200 > 100 = 1e12 Msun/h, Group_R_Crit200 > 0, as the edited C++ builds) and Lee22 shell
(128 shells, 0.03-2.5 R200c, float32 bin arithmetic as in prt_insert):
  fix_{ne,pe,n}[k]      : sums over all pairs with r <= 2.5 R200c (plain path; = C++ plain/perfix/naive)
  bug_{ne,pe,n}[s, k]   : the same without pairs whose minimum image is across a low face, for translation s
  expL_{ne,pe}[k]       : sum w prod_i(1 - max(0,-d_i)/L): expectation of the buggy sum over uniformly random
                          translations of THIS box (exact; d = minimum-image displacement particle - group)
  expT_{ne,pe}[k]       : the same expectation for a box of side L_TNG300 = 205 Mpc/h with the same gas
                          (the 'CAMELS measurement rescaled to TNG300')
Weights: ne = x_e X_H m / mprot (ne_prof.cpp, mprot with h = 0.6774), pe = 4 x_e X_H/(1+3X_H+4X_H x_e)(gamma-1) m u
(y_prof.cpp).  Profiles = sums / shell volume (comoving (ckpc/h)^3), as the C++ save().

Usage: python camels_bug_replica.py SNAP.hdf5 GROUPS.hdf5 SHIFTS.txt OUT.npz
"""
import os
import sys
import time

import h5py
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bugmodel as bm  # noqa: E402

snapf, grpf, shiftf, outf = sys.argv[1:5]
shifts = np.atleast_2d(np.loadtxt(shiftf))
assert not np.any(shifts[0]), "first shift must be 0 0 0"
NS = len(shifts)
t0 = time.time()

with h5py.File(grpf, "r") as g:
    gpos = g["Group/GroupPos"][:].astype(np.float32)
    m200 = g["Group/Group_M_Crit200"][:].astype(np.float32)
    r200 = g["Group/Group_R_Crit200"][:].astype(np.float32)
with h5py.File(snapf, "r") as f:
    H = dict(f["Header"].attrs)
    pos = f["PartType0/Coordinates"][:].astype(np.float32)
    mass = f["PartType0/Masses"][:].astype(np.float64)
    u = f["PartType0/InternalEnergy"][:].astype(np.float64)
    xe = f["PartType0/ElectronAbundance"][:].astype(np.float64)
L = float(H["BoxSize"])
L32 = np.float32(L)
z = float(H["Redshift"])
a = float(H["Time"])
print(f"read {len(mass)} cells z={z:.4f} L={L} in {time.time() - t0:.0f} s; coords [{pos.min()}, {pos.max()}], "
      f"== L: {int(np.sum(pos == L32))}, > L or < 0: {int(np.sum((pos > L32) | (pos < 0)))}", flush=True)

w_ne = xe * bm.XH * mass / bm.MPROT_NE_PROF
w_pe = 4.0 * xe * bm.XH / (1.0 + 3.0 * bm.XH + 4.0 * bm.XH * xe) * (bm.GAMMA - 1.0) * mass * u
tree = cKDTree(np.mod(pos.astype(np.float64), L), boxsize=L)
print(f"tree built {time.time() - t0:.0f} s", flush=True)

sel = np.where((m200 > np.float32(1e2)) & (r200 > 0))[0]
NG = len(sel)
NB = bm.NB
out = {k: np.zeros((NG, NB)) for k in ("fix_ne", "fix_pe", "fix_n", "expL_ne", "expL_pe", "expT_ne", "expT_pe")}
for k in ("bug_ne", "bug_pe", "bug_n"):
    out[k] = np.zeros((NS, NG, NB))
vol = np.zeros((NG, NB))
rlo = np.zeros((NG, NB))
rhi = np.zeros((NG, NB))
dlow = np.zeros((NS, NG))           # distance of the group centre to the nearest low face, per translation
gpos_s = np.zeros((NS, NG, 3))

for j, ig in enumerate(sel):
    c32 = gpos[ig]
    R32 = np.float32(2.5) * r200[ig]
    Rsq_max = R32 * R32
    logRmin, logRmax, dlogR, vol[j], rlo[j], rhi[j] = bm.lee22_bins(r200[ig])
    idx = np.asarray(tree.query_ball_point(np.mod(c32.astype(np.float64), L), float(R32) * 1.001 + 0.01), dtype=np.int64)
    xp = pos[idx]
    # plain path (abs_periodic_dist + hypotsq in float32)
    ad = np.abs(xp - c32)
    ad = np.where(ad > np.float32(0.5) * L32, L32 - ad, ad).astype(np.float32)
    Rsq = (ad[:, 0] * ad[:, 0] + ad[:, 1] * ad[:, 1] + ad[:, 2] * ad[:, 2]).astype(np.float32)
    ok = Rsq <= Rsq_max
    idx, xp, Rsq = idx[ok], xp[ok], Rsq[ok]
    kb = bm.bin_index(Rsq, logRmin, logRmax, dlogR)
    ok = kb >= 0
    idx, xp, kb = idx[ok], xp[ok], kb[ok]
    wn, wp = w_ne[idx], w_pe[idx]
    out["fix_ne"][j] = np.bincount(kb, weights=wn, minlength=NB)
    out["fix_pe"][j] = np.bincount(kb, weights=wp, minlength=NB)
    out["fix_n"][j] = np.bincount(kb, minlength=NB)
    d = xp.astype(np.float64) - c32.astype(np.float64)
    d -= L * np.round(d / L)
    for key, LL in (("expL", L), ("expT", bm.L_TNG300)):
        pk = np.prod(1.0 - np.maximum(0.0, -d) / LL, axis=1)
        out[key + "_ne"][j] = np.bincount(kb, weights=wn * pk, minlength=NB)
        out[key + "_pe"][j] = np.bincount(kb, weights=wp * pk, minlength=NB)
    for s in range(NS):
        xs = bm.shift_positions(xp, shifts[s], L).astype(np.float64)
        cs = bm.shift_positions(c32[None, :], shifts[s], L)[0].astype(np.float64)
        keep = ~np.any(xs - cs > 0.5 * L, axis=1)
        out["bug_ne"][s, j] = np.bincount(kb[keep], weights=wn[keep], minlength=NB)
        out["bug_pe"][s, j] = np.bincount(kb[keep], weights=wp[keep], minlength=NB)
        out["bug_n"][s, j] = np.bincount(kb[keep], minlength=NB)
        dlow[s, j] = cs.min()
        gpos_s[s, j] = cs
    if j % 20 == 0:
        print(f"group {j}/{NG} logM={np.log10(m200[ig] * 1e10):.2f} npair={len(kb)} t={time.time() - t0:.0f} s", flush=True)

# P200 as y_prof.cpp grp_store_P (code units), n200 is common to all groups of a snapshot
Om, OL, Ob = float(H["Omega0"]), float(H["OmegaLambda"]), float(H["OmegaBaryon"])
rho_crit = 2.775e-8 * (Om / a ** 3 + OL)
P200 = 100.0 * 4.30091e4 * m200[sel].astype(np.float64) * rho_crit * Ob / Om / r200[sel].astype(np.float64)
np.savez_compressed(outf, z=z, a=a, L=L, h=float(H["HubbleParam"]), Om=Om, OL=OL, Ob=Ob, shifts=shifts, ID=sel,
                    M200_code=m200[sel].astype(np.float64), R200_code=r200[sel].astype(np.float64),
                    GroupPos=gpos[sel].astype(np.float64), gpos_shifted=gpos_s, dlow=dlow, P200_code=P200,
                    vol=vol, r_lo=rlo, r_hi=rhi, snapfile=os.path.abspath(snapf), groupsfile=os.path.abspath(grpf), **out)
print(f"wrote {outf}: {NG} groups, {NS} translations, {time.time() - t0:.0f} s", flush=True)
