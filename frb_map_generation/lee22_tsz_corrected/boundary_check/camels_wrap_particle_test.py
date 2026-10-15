#!/usr/bin/env python
"""Particle-level periodic-boundary test of OUR CAMELS profile code (workstream D, camels_direct_profiles.py).

For one raw CAMELS IllustrisTNG CV snapshot + group catalogue, every halo with M200c >= 10^MINLOGM Msun is profiled
twice with exactly the per-cell quantities and bins of camels_direct_profiles.py:
  'wrap'  : the production method (positions mod L, cKDTree(boxsize=L), minimum-image separations);
  'nowrap': the failure mode a boundary bug would produce (non-periodic tree, raw separations: gas across a face is
            simply missing).
Bins: (B) ground truth in x = r/R200c (40 log bins 0.01-5 + centre bin), (C) Lee22 group_particles emulation
(128 log shells 0.03-2.5 R200c, innermost shell from the origin).  The 'wrap' run is compared with the stored
production file direct_CV_<box>_<snap>.npz (must agree to float round-off); 'nowrap'/'wrap' is compared with the
geometric surrogate f_in (geom.py).

Usage: python camels_wrap_particle_test.py SNAP.hdf5 GROUPS.hdf5 STORED_DIRECT.npz OUT.npz [--minlogm 12.0]
"""
import argparse
import os
import sys
import time

import h5py
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import geom  # noqa: E402

M_P = 1.67262192369e-24
MSUN = 1.98847e33
X_H = 0.76
GAMMA = 5.0 / 3.0

p = argparse.ArgumentParser()
p.add_argument("snap")
p.add_argument("groups")
p.add_argument("stored")
p.add_argument("out")
p.add_argument("--minlogm", type=float, default=12.0)
p.add_argument("--xmax", type=float, default=5.0)
args = p.parse_args()

t0 = time.time()
with h5py.File(args.groups, "r") as g:
    gpos = g["Group/GroupPos"][:].astype(np.float64)
    m200 = g["Group/Group_M_Crit200"][:].astype(np.float64)
    r200 = g["Group/Group_R_Crit200"][:].astype(np.float64)
with h5py.File(args.snap, "r") as f:
    shdr = dict(f["Header"].attrs)
    pos = f["PartType0/Coordinates"][:].astype(np.float64)
    mass = f["PartType0/Masses"][:].astype(np.float64)
    u = f["PartType0/InternalEnergy"][:].astype(np.float64)
    xe = f["PartType0/ElectronAbundance"][:].astype(np.float64)
L = float(shdr["BoxSize"])
h = float(shdr["HubbleParam"])
z = float(shdr["Redshift"])
print(f"read {len(mass)} cells z={z:.4f} L={L} in {time.time() - t0:.0f} s; raw coordinate range "
      f"[{pos.min():.6f}, {pos.max():.6f}]", flush=True)
n_outside = int(np.sum((pos < 0) | (pos >= L)))
pos = np.mod(pos, L)
tree_p = cKDTree(pos, boxsize=L)
tree_n = cKDTree(pos)
print(f"trees built {time.time() - t0:.0f} s; cells with a raw coordinate outside [0, L): {n_outside}", flush=True)

mu_cell = 4.0 / (1.0 + 3.0 * X_H + 4.0 * X_H * xe)
m_g = mass * 1e10 * MSUN / h
Ne_cell = xe * X_H * m_g / M_P
PthV_cell = (GAMMA - 1.0) * m_g * u * 1e10
PeV_cell = (xe * X_H * mu_cell) * PthV_cell
mprot_code = h * 1.6726219 / 1.98847 * 1e-67
lee_Ne = xe * X_H * mass / mprot_code
lee_PeV = 4.0 * xe * X_H / (1.0 + 3.0 * X_H + 4.0 * X_H * xe) * (GAMMA - 1.0) * mass * u

NB_T = 40
x_edges = np.concatenate([[0.0], np.logspace(-2, np.log10(args.xmax), NB_T + 1)])
NB_LEE = 128
logm = np.log10(np.where(m200 > 0, m200, 1e-30) * 1e10 / h)
sel = np.where((logm >= args.minlogm) & (r200 > 0))[0]
nh = len(sel)
print(f"{nh} halos with log10 M200c/Msun >= {args.minlogm}", flush=True)

modes = ("wrap", "nowrap")
res = {}
for md in modes:
    for k in ("Ne", "PeV", "Mgas", "PthV", "n"):
        res[f"true_{k}_{md}"] = np.zeros((nh, NB_T + 1))
    for k in ("Ne", "PeV"):
        res[f"in200_{k}_{md}"] = np.zeros(nh)
    for k in ("ne", "pe", "npart"):
        res[f"lee_{k}_{md}"] = np.zeros((nh, NB_LEE))
res["finx_true"] = np.zeros((nh, NB_T + 1))
res["finx_lee"] = np.zeros((nh, NB_LEE))
res["dmin_face_over_R200"] = np.zeros(nh)

for j, ig in enumerate(sel):
    c = gpos[ig]
    Rc = r200[ig]
    Rq = max(args.xmax, 2.5) * Rc
    logRmin = np.log(0.03 * Rc)
    logRmax = np.log(2.5 * Rc)
    dlogR = (logRmax - logRmin) / (NB_LEE - 1)
    kk = np.arange(NB_LEE)
    r1 = np.where(kk == 0, 0.0, np.exp(logRmin + dlogR * (kk - 1)))
    r2 = np.where(kk == 0, np.exp(logRmin), np.exp(logRmin + dlogR * kk))
    shell = 4 * np.pi / 3 * (r2 ** 3 - r1 ** 3)
    for md in modes:
        if md == "wrap":
            idx = np.asarray(tree_p.query_ball_point(c, Rq), dtype=np.int64)
            d = pos[idx] - c
            d -= L * np.round(d / L)
        else:
            idx = np.asarray(tree_n.query_ball_point(c, Rq), dtype=np.int64)
            d = pos[idx] - c
        r = np.sqrt((d ** 2).sum(1))
        x = r / Rc
        s = x < args.xmax
        xx, ii = x[s], idx[s]
        for key, arr in (("Ne", Ne_cell), ("PeV", PeV_cell), ("Mgas", m_g / MSUN), ("PthV", PthV_cell)):
            res[f"true_{key}_{md}"][j] = np.histogram(xx, x_edges, weights=arr[ii])[0]
        res[f"true_n_{md}"][j] = np.histogram(xx, x_edges)[0]
        inn = xx < 1.0
        res[f"in200_Ne_{md}"][j] = Ne_cell[ii][inn].sum()
        res[f"in200_PeV_{md}"][j] = PeV_cell[ii][inn].sum()
        s = r < 2.5 * Rc
        rr, ii = r[s], idx[s]
        lr = np.log(np.maximum(rr, 1e-30))
        kb = np.where(lr < logRmin, 0, 1 + np.floor((lr - logRmin) / dlogR)).astype(np.int64)
        ok = kb < NB_LEE
        kb, ii = kb[ok], ii[ok]
        res[f"lee_ne_{md}"][j] = np.bincount(kb, weights=lee_Ne[ii], minlength=NB_LEE) / shell
        res[f"lee_pe_{md}"][j] = np.bincount(kb, weights=lee_PeV[ii], minlength=NB_LEE) / shell
        res[f"lee_npart_{md}"][j] = np.bincount(kb, minlength=NB_LEE)
    res["finx_true"][j] = geom.f_in_shells(c, x_edges * Rc, L)
    res["finx_lee"][j] = geom.f_in_shells(c, np.concatenate([[0.0], r2]), L)
    res["dmin_face_over_R200"][j] = np.min(np.minimum(c, L - c)) / Rc
    if j % 20 == 0:
        print(f"halo {j}/{nh} logM={logm[ig]:.2f} dmin/R200={res['dmin_face_over_R200'][j]:.2f} t={time.time() - t0:.0f} s",
              flush=True)

# reproduction of the stored production file (wrap mode)
st = np.load(args.stored)
pos_in_stored = {int(i): k for k, i in enumerate(st["ID"])}
ks = np.array([pos_in_stored[int(i)] for i in sel])
repro = {}
for key in ("Ne", "PeV", "Mgas", "PthV", "n"):
    a_, b_ = res[f"true_{key}_wrap"], st["true_" + key][ks]
    m = b_ != 0
    repro["true_" + key] = float(np.max(np.abs(a_[m] / b_[m] - 1))) if m.any() else 0.0
    repro["true_" + key + "_zero_mismatch"] = int(np.sum((a_ == 0) != (b_ == 0)))
for key in ("ne", "pe", "npart"):
    a_, b_ = res[f"lee_{key}_wrap"], st["lee_" + key][ks]
    m = b_ != 0
    repro["lee_" + key] = float(np.max(np.abs(a_[m] / b_[m] - 1))) if m.any() else 0.0
print("max |recomputed(wrap)/stored - 1|:", repro, flush=True)

np.savez_compressed(args.out, z=z, h=h, L=L, ID=sel, M200_code=m200[sel], R200_code=r200[sel], GroupPos=gpos[sel],
                    x_edges=x_edges, n_raw_outside=n_outside, repro_keys=np.array(list(repro.keys())),
                    repro_vals=np.array(list(repro.values()), dtype=float), **res)
print(f"wrote {args.out} in {time.time() - t0:.0f} s", flush=True)
