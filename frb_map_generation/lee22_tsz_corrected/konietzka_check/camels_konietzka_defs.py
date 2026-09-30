#!/usr/bin/env python
"""Electron content and projected (ray-hit) area of CAMELS IllustrisTNG halos under candidate definitions of the
Konietzka et al. TNG300 "halo DM within R200" catalogue.  Runs on idark (python3.8, numpy, scipy, h5py).

Gas fields used (read from the full raw snapshot, which the PBS job downloads and deletes afterwards; the datasets
are chunked + gzip-compressed so range requests do not help): Coordinates, Masses, Density, ElectronAbundance,
StarFormationRate and SubfindDMDensity (local comoving DM density, SPH kernel over 64+-4 DM particles; the field
Artale+22 use).

Definitions evaluated per halo (M200c >= 10^12.5 Msun), all from the SAME cells:
  sph[x]      : cells with r < x R200c, x in XS (x = 1 is the baseline of the CAMELS/Lee22 workstream D)
  r200m, r500c: cells inside Group_R_Mean200 / Group_R_Crit500
  fof         : all gas cells of the FoF group (snapshot ordering + GroupLenType offsets), any radius
  fof_in1     : FoF cells with r < R200c
  thr<k>_fof  : FoF cells with SubfindDMDensity > threshold k
  thr<k>_3R   : all cells within 3 R200c with SubfindDMDensity > threshold k (spatial upper bound)
  thr<k>_in1  : threshold cells with r < R200c
  thresholds  : 'c057' = 57 rho_crit,0 in COMOVING density (Artale+22 / Haider+16, as cited by Hussaini+25),
                'c028', 'c114' = x0.5, x2 of that, 'p057' = 57 rho_crit(z) PHYSICAL.
Electron bookkeeping for every set: Me = sum x_e X_H m (electron-equivalent mass, N_e = Me/m_p) with
  (i) ElectronAbundance as stored ('xe'), (ii) star-forming cells set to n_e = 0 ('sf0', Konietzka+25 Sec. III.7),
  (iii) star-forming cells fully ionised ('sfion', (1+X_H)/2 m).
Projected area: cells of a set are projected along x, y, z onto a grid of pixel R200c/40 over +-3 R200c, each cell a
disk of its volume-equivalent radius (3 m/(4 pi rho))^(1/3); area in units of pi R200c^2, mean of the 3 axes.

Usage: python camels_konietzka_defs.py BOX SNAP OUTDIR SNAPFILE [--groups-dir DIR] [--rawdir DIR]
"""
import argparse
import os
import subprocess
import time

import h5py
import numpy as np
from scipy.spatial import cKDTree

BASE = "https://users.flatironinstitute.org/~camels/Sims/IllustrisTNG/CV"
FIELDS = ["Coordinates", "Masses", "Density", "ElectronAbundance", "StarFormationRate", "SubfindDMDensity"]
X_H = 0.76
RHO_CRIT0_CODE = 2.775e-8          # 1e10 Msun/h / (kpc/h)^3  (rho_crit,0, h-free in these units)
XS = np.array([0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0])
NPIX_R = 40
EXT = 3.0


def read_snapshot_fields(snapfile):
    """the gas fields are chunked + gzip-compressed in the CAMELS files, so the whole snapshot is downloaded
    (by the PBS job) and read here"""
    with h5py.File(snapfile, "r") as h:
        hdr = dict(h["Header"].attrs)
        data = {k: h["PartType0/" + k][:] for k in FIELDS}
    return hdr, data


def disk_offsets(k):
    r = np.arange(-k, k + 1)
    di, dj = np.meshgrid(r, r, indexing="ij")
    m = di ** 2 + dj ** 2 <= (k + 0.5) ** 2
    return di[m], dj[m]


_OFF = {k: disk_offsets(k) for k in range(0, 61)}


def proj_area(d, rc, R):
    """mean over the 3 axes of the projected area (units of pi R^2) of disks (radius rc) at 3D offsets d"""
    if len(d) == 0:
        return 0.0
    pix = R / NPIX_R
    N = int(2 * EXT * NPIX_R)
    kpix = np.minimum(np.round(rc / pix).astype(np.int64), 60)
    areas = []
    for ax in range(3):
        a1, a2 = [i for i in range(3) if i != ax]
        i0 = np.floor((d[:, a1] + EXT * R) / pix).astype(np.int64)
        j0 = np.floor((d[:, a2] + EXT * R) / pix).astype(np.int64)
        grid = np.zeros((N, N), dtype=bool)
        for k in np.unique(kpix):
            s = kpix == k
            di, dj = _OFF[int(k)]
            ii = (i0[s][:, None] + di[None, :]).ravel()
            jj = (j0[s][:, None] + dj[None, :]).ravel()
            ok = (ii >= 0) & (ii < N) & (jj >= 0) & (jj < N)
            grid[ii[ok], jj[ok]] = True
        areas.append(grid.sum() * pix ** 2 / (np.pi * R ** 2))
    return float(np.mean(areas))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("box", type=int)
    p.add_argument("snap", type=int)
    p.add_argument("outdir")
    p.add_argument("snapfile")
    p.add_argument("--groups-dir", default="/lustre/work/kristero10/lee22_tsz_verification_20260929_D/raw")
    p.add_argument("--rawdir", default="/lustre/work/kristero10/lee22_konietzka_check_20260930/raw")
    p.add_argument("--logm-min", type=float, default=12.5)
    a = p.parse_args()
    t0 = time.time()
    out = os.path.join(a.outdir, "konietzka_defs_CV_%d_%03d.npz" % (a.box, a.snap))
    if os.path.exists(out):
        print("exists", out)
        return
    gfile = os.path.join(a.groups_dir, "CV_%d" % a.box, "groups_%03d.hdf5" % a.snap)
    if not os.path.exists(gfile):
        gfile = os.path.join(a.rawdir, "groups_CV_%d_%03d.hdf5" % (a.box, a.snap))
        if not os.path.exists(gfile):
            os.makedirs(a.rawdir, exist_ok=True)
            subprocess.run(["curl", "-s", "-S", "--retry", "5", "-o", gfile + ".part",
                            "%s/CV_%d/groups_%03d.hdf5" % (BASE, a.box, a.snap)], check=True)
            os.rename(gfile + ".part", gfile)
    hdr, D = read_snapshot_fields(a.snapfile)
    print("read fields in %.0f s" % (time.time() - t0), flush=True)
    with h5py.File(gfile, "r") as g:
        G = {k: g["Group/" + k][:] for k in ["GroupPos", "Group_M_Crit200", "Group_R_Crit200", "Group_M_Mean200",
                                               "Group_R_Mean200", "Group_M_Crit500", "Group_R_Crit500", "GroupMass",
                                               "GroupLenType", "GroupFirstSub"]}
        submass = g["Subhalo/SubhaloMass"][:]
        ghdr = dict(g["Header"].attrs)
    L = float(hdr["BoxSize"])
    h = float(hdr["HubbleParam"])
    z = max(float(hdr["Redshift"]), 0.0)
    aa = 1.0 / (1.0 + z)
    Om, OL = float(hdr["Omega0"]), float(hdr["OmegaLambda"])
    assert abs(float(ghdr["Redshift"]) - float(hdr["Redshift"])) < 1e-6
    E2 = Om * (1 + z) ** 3 + OL
    ncell = D["Masses"].shape[0]
    print("CV_%d snap %03d z=%.4f cells %d groups %d" % (a.box, a.snap, z, ncell, len(G["GroupMass"])), flush=True)

    pos = np.mod(D["Coordinates"].astype(np.float64), L)
    mass = D["Masses"].astype(np.float64)
    rho = D["Density"].astype(np.float64)
    xe = D["ElectronAbundance"].astype(np.float64)
    sf = D["StarFormationRate"] > 0
    rdm = D["SubfindDMDensity"].astype(np.float64)
    m_msun = mass * 1e10 / h
    Me = {"xe": xe * X_H * m_msun}
    Me["sf0"] = np.where(sf, 0.0, Me["xe"])
    Me["sfion"] = np.where(sf, 0.5 * (1 + X_H) * m_msun, Me["xe"])
    rc = (3.0 * mass / rho / (4 * np.pi)) ** (1.0 / 3.0)          # ckpc/h, volume-equivalent radius
    thr = {"c057": 57.0 * RHO_CRIT0_CODE, "c028": 28.5 * RHO_CRIT0_CODE, "c114": 114.0 * RHO_CRIT0_CODE,
           "p057": 57.0 * RHO_CRIT0_CODE * E2 * aa ** 3}
    above = {k: rdm > v for k, v in thr.items()}
    fof_len = G["GroupLenType"][:, 0].astype(np.int64)
    fof_off = np.concatenate([[0], np.cumsum(fof_len)[:-1]])
    print("thresholds (code comoving):", {k: "%.3e" % v for k, v in thr.items()},
          "frac cells above:", {k: round(float(v.mean()), 4) for k, v in above.items()}, flush=True)
    tree = cKDTree(pos, boxsize=L)
    print("tree %.0f s" % (time.time() - t0), flush=True)

    m200 = G["Group_M_Crit200"] * 1e10 / h
    sel = np.where((m200 >= 10 ** a.logm_min) & (G["Group_R_Crit200"] > 0))[0]
    nh = len(sel)
    res = {"box": np.full(nh, a.box), "snap": np.full(nh, a.snap), "z": np.full(nh, z), "gid": sel,
           "M200c": m200[sel], "R200c_ckpch": G["Group_R_Crit200"][sel].astype(float),
           "M200m": G["Group_M_Mean200"][sel] * 1e10 / h, "R200m_ckpch": G["Group_R_Mean200"][sel].astype(float),
           "M500c": G["Group_M_Crit500"][sel] * 1e10 / h, "R500c_ckpch": G["Group_R_Crit500"][sel].astype(float),
           "Mfof": G["GroupMass"][sel] * 1e10 / h,
           "Msub_central": np.where(G["GroupFirstSub"][sel] >= 0, submass[np.maximum(G["GroupFirstSub"][sel], 0)],
                                    0.0) * 1e10 / h,
           "XS": XS}
    ekeys = ("xe", "sf0", "sfion")
    for e in ekeys:
        res["Me_sph_" + e] = np.zeros((nh, len(XS)))
        for s in ["r200m", "r500c", "fof", "fof_in1", "fof_in2"] + ["thr%s_%s" % (k, w) for k in thr for w in ("fof", "3R", "in1")]:
            res["Me_%s_%s" % (s, e)] = np.zeros(nh)
    res["Mgas_sph1"] = np.zeros(nh)
    res["Mgas_sf_sph1"] = np.zeros(nh)
    akeys = ["sph1", "sph1_2d_cells", "fof", "fof_in1", "thrc057_fof", "thrc057_3R", "thrp057_fof", "thrp057_3R",
             "thrc028_fof", "thrc114_fof"]
    for k in akeys:
        res["A_" + k] = np.zeros(nh)
    res["fof_frac_cells_beyond_3R"] = np.zeros(nh)
    res["fof_rmax_over_R200c"] = np.zeros(nh)
    for j, ig in enumerate(sel):
        c = G["GroupPos"][ig].astype(np.float64)
        R = float(G["Group_R_Crit200"][ig])
        Rm = float(G["Group_R_Mean200"][ig])
        R5 = float(G["Group_R_Crit500"][ig])
        Rq = max(EXT * R, 1.01 * Rm)
        idx = np.asarray(tree.query_ball_point(c, Rq), dtype=np.int64)
        d = pos[idx] - c
        d -= L * np.round(d / L)
        r = np.sqrt((d ** 2).sum(1))
        x = r / R
        infof = (idx >= fof_off[ig]) & (idx < fof_off[ig] + fof_len[ig])
        # whole FoF group (any radius)
        fsl = slice(fof_off[ig], fof_off[ig] + fof_len[ig])
        dfof = pos[fsl] - c
        dfof -= L * np.round(dfof / L)
        rfof = np.sqrt((dfof ** 2).sum(1))
        res["fof_frac_cells_beyond_3R"][j] = float(np.mean(rfof > EXT * R)) if len(rfof) else 0.0
        res["fof_rmax_over_R200c"][j] = float(rfof.max() / R) if len(rfof) else 0.0
        in1 = x < 1.0
        in2 = x < 2.0
        res["Mgas_sph1"][j] = m_msun[idx[in1]].sum()
        res["Mgas_sf_sph1"][j] = m_msun[idx[in1 & sf[idx]]].sum()
        masks = {"r200m": r < Rm, "r500c": r < R5, "fof_in1": infof & in1, "fof_in2": infof & in2}
        for k in thr:
            ab = above[k][idx]
            masks["thr%s_fof" % k] = ab & infof
            masks["thr%s_3R" % k] = ab & (x < EXT)
            masks["thr%s_in1" % k] = ab & in1
        for e in ekeys:
            me = Me[e][idx]
            cs = np.array([me[x < xx].sum() for xx in XS])
            res["Me_sph_" + e][j] = cs
            for s, m in masks.items():
                res["Me_%s_%s" % (s, e)][j] = me[m].sum()
            res["Me_fof_" + e][j] = Me[e][fsl].sum()
        # projected areas
        rci = rc[idx]
        res["A_sph1"][j] = 1.0
        res["A_sph1_2d_cells"][j] = proj_area(d[in1], rci[in1], R)
        res["A_fof"][j] = proj_area(d[infof & (x < EXT)], rci[infof & (x < EXT)], R)
        res["A_fof_in1"][j] = proj_area(d[masks["fof_in1"]], rci[masks["fof_in1"]], R)
        for k in ("thrc057_fof", "thrc057_3R", "thrp057_fof", "thrp057_3R", "thrc028_fof", "thrc114_fof"):
            m = masks[k] & (x < EXT)
            res["A_" + k][j] = proj_area(d[m], rci[m], R)
        if j % 20 == 0:
            print("halo %d/%d logM=%.2f ncell=%d t=%.0f s" % (j, nh, np.log10(m200[ig]), len(idx), time.time() - t0),
                  flush=True)
    res["thresholds_code"] = np.array([thr[k] for k in ("c057", "c028", "c114", "p057")])
    np.savez(out, **res)
    print("wrote %s (%d halos) in %.0f s" % (out, nh, time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
