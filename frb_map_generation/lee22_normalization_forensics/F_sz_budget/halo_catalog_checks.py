#!/usr/bin/env python3
"""Checks of the halo-model ingredients against the HalfDome lightcone catalogue painted in the SZ maps:
(1) halo counts per (z, M200c) bin versus Tinker08 x full-sky comoving volume;
(2) rms line-of-sight peculiar velocity of halos per z bin versus the linear-theory <v_los^2>^(1/2) used in szhm.
Reads all masses/redshifts (about 1 GB) and velocities/positions from 40 contiguous row blocks."""
import json
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import szhm as S  # noqa: E402

CAT = "/home/cbllover/HalfDome/lightcone_100.hdf5"
OUT = Path("/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_normalization_forensics_20260928/F_sz_budget")
cosmo = S.PAINT_COSMO
h = cosmo.h
res = {}
with h5py.File(CAT, "r") as f:
    n = f["redshift"].shape[0]
    z = f["redshift"][:].astype(float)
    M = f["halo_mass_m200c"][:] / h                 # Msun/h -> Msun (as the painters)
    zb = np.array([0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 3.86])
    mb = 10 ** np.array([12.8167, 13.0, 13.5, 14.0, 14.5, 15.6])
    H, _, _ = np.histogram2d(z, M, bins=(zb, mb))
    pred = np.zeros_like(H)
    for i in range(len(zb) - 1):
        zz = np.linspace(zb[i], zb[i + 1], 41)[1:] - 0.5 * (zb[i + 1] - zb[i]) / 40
        dV = 4 * np.pi * cosmo.chi(zz) ** 2 / cosmo.H_over_c(zz) * (zb[i + 1] - zb[i]) / 40
        for j in range(len(mb) - 1):
            mm = np.geomspace(mb[j], mb[j + 1], 41)
            pred[i, j] = sum(dV[k] * np.trapz(S.tinker08_dndlnm(cosmo, mm, zz[k]), np.log(mm)) for k in range(len(zz)))
    res["counts"] = dict(z_edges=zb.tolist(), log10M_edges=np.log10(mb).tolist(), catalogue=H.tolist(),
                         tinker08=pred.tolist(), ratio_catalogue_over_tinker08=(H / pred).tolist())
    del z, M
    # velocities from 40 blocks
    starts = np.linspace(0, n - 250000, 40).astype(int)
    zs, vl, ms = [], [], []
    for s0 in starts:
        sl = slice(s0, s0 + 250000)
        pos = f["Position"][sl].astype(float)
        vel = f["Velocity"][sl].astype(float)
        rhat = pos / np.linalg.norm(pos, axis=1)[:, None]
        vl.append((vel * rhat).sum(1)); zs.append(f["redshift"][sl].astype(float)); ms.append(f["halo_mass_m200c"][sl] / h)
    zs, vl, ms = map(np.concatenate, (zs, vl, ms))
    rows = []
    for i in range(len(zb) - 1):
        sel = (zs >= zb[i]) & (zs < zb[i + 1])
        if sel.sum() < 100:
            continue
        zc = float(np.median(zs[sel]))
        lin = float(np.sqrt(S.vlos2_kms2(cosmo, zc)[0][0]))
        row = dict(z_lo=float(zb[i]), z_hi=float(zb[i + 1]), n=int(sel.sum()), z_median=zc,
                   vlos_rms_catalogue=float(np.sqrt(np.mean(vl[sel] ** 2))), vlos_rms_linear=lin)
        for lo, hi in ((6.5e12, 3e13), (3e13, 3e14), (3e14, 1e16)):
            s2 = sel & (ms >= lo) & (ms < hi)
            if s2.sum() > 50:
                row["vlos_rms_M_{:.0e}_{:.0e}".format(lo, hi)] = float(np.sqrt(np.mean(vl[s2] ** 2)))
        rows.append(row)
    res["velocities"] = rows
(OUT / "halo_catalog_checks.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
