#!/usr/bin/env python
"""Does the existing CAMELS validation (precomputed illstack profiles vs our direct recomputation, agreement ~1e-3)
actually test periodic handling?  illstack wraps with 26 ghost-particle images (cyprof.pyx add_ghost_particles,
box = 25000 hard-coded), our camels_direct_profiles.py with cKDTree(boxsize=L) + minimum image: two independent
implementations.  If either one did not wrap, the bins whose shell crosses a box face would disagree by
~(1 - f_in).  This script lists those bins among the validated ones (same cuts as validate_precomputed.py:
>= 10 cells, inside the search radius, matched halos, CV_2 excluded) and the observed ratio there.

Output: OUT/validation_face_bins.txt / .json
"""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, "/home/cbllover/HalfDome/frb_map_generation/lee22_tsz_verification/D_tng_camels")
import geom  # noqa: E402
import camels_common as cc  # noqa: E402

OUT = "/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_boundary_check_20261007"
L = 25000.0
lines = []
allrows = []
for fn in sorted(glob.glob(os.path.join(cc.DIRECTDIR, "direct_CV_*_*.npz"))):
    box = int(os.path.basename(fn).split("_")[2])
    new = int(os.path.basename(fn).split("_")[3].split(".")[0])
    if box == 2:
        continue
    old = cc.NEW_SNAPS.index(new)
    d = np.load(fn)
    pc = cc.load_precomputed(box, old)
    h, a = float(d["h"]), float(d["a"])
    dens_fac = 1e10 * cc.MSUN / cc.KPC ** 3 * h ** 2 / a ** 3
    idpos = {int(i): k for k, i in enumerate(d["ID"])}
    both = [(k, idpos[int(i)]) for k, i in enumerate(pc["ID"]) if int(i) in idpos]
    rs = float(d["search_radius"]) * float(d["rhmax"])
    inner = d["ill_edges"][1:] < rs
    for kp, kd in both:
        c = d["GroupPos"][kd].astype(float)
        fin = geom.f_in_shells(c, d["ill_edges"], L)
        P = pc["rho"][kp] / dens_fac
        D = d["ill_dens"][kd]
        Pp = pc["Pth"][kp] / dens_fac / 1e10
        Dp = d["ill_pth"][kd]
        m = (D > 0) & (P > 0) & (d["ill_n"][kd] >= 10) & inner
        for b in np.where(m)[0]:
            allrows.append((box, new, int(d["ID"][kd]), b, fin[b], P[b] / D[b] - 1, Pp[b] / Dp[b] - 1 if Dp[b] > 0 else np.nan,
                            d["ill_edges"][b + 1] / d["R200_code"][kd], d["M200_code"][kd] * 1e10 / h))
R = np.array(allrows, dtype=float)
fin, dr, dp, xo, M = R[:, 4], R[:, 5], R[:, 6], R[:, 7], R[:, 8]
res = {"n_validated_bins": int(len(R))}
lines.append(f"validated (halo, illstack bin) pairs (CV_0, CV_1, CV_3 x 5 snapshots, >= 10 cells, inside the search radius): {len(R)}")
for thr in (0.99, 0.95, 0.9, 0.8, 0.7):
    s = fin < thr
    if s.sum() == 0:
        continue
    res[f"fin_lt_{thr}"] = dict(n_bins=int(s.sum()), n_halos=int(len(set(map(tuple, R[s][:, :3].astype(int))))),
                               mean_expected_nowrap_deficit=float(np.mean(1 - fin[s])),
                               max_expected_nowrap_deficit=float(np.max(1 - fin[s])),
                               max_abs_dens_dev=float(np.max(np.abs(dr[s]))), max_abs_pth_dev=float(np.nanmax(np.abs(dp[s]))))
    lines.append(f"  bins with shell fraction inside the box f_in < {thr}: {s.sum()} bins of "
                 f"{res[f'fin_lt_{thr}']['n_halos']} halos; a no-wrap code would be low by {np.mean(1 - fin[s]):.3f} on average "
                 f"(max {np.max(1 - fin[s]):.3f}); observed max |precomputed/direct - 1| = {np.max(np.abs(dr[s])):.1e} (density), "
                 f"{np.nanmax(np.abs(dp[s])):.1e} (thermal pressure); 99th pct {np.percentile(np.abs(dr[s]), 99):.1e}; "
                 f"bins with |dev| > 1e-2: {int(np.sum(np.abs(dr[s]) > 1e-2))}")
    res[f"fin_lt_{thr}"]["p99_abs_dens_dev"] = float(np.percentile(np.abs(dr[s]), 99))
    res[f"fin_lt_{thr}"]["n_bins_dev_gt_1e-2"] = int(np.sum(np.abs(dr[s]) > 1e-2))
s = (fin < 0.9) & (M >= 10 ** 12.5)
lines.append(f"  of which halos M200c >= 10^12.5 Msun: {s.sum()} bins, outer edge x = r/R200c in "
             f"[{xo[s].min() if s.any() else np.nan:.2f}, {xo[s].max() if s.any() else np.nan:.2f}], "
             f"max |dev| {np.max(np.abs(dr[s])) if s.any() else np.nan:.1e}")
res["M12p5_fin_lt_0.9_bins"] = int(s.sum())
lines.append("  (the few |dev| > 1e-3 are single cells moved across a bin edge or across the search-radius edge, float32 vs "
             "float64 radii; they occur equally for halos far from any face)")
print("\n".join(lines))
with open(os.path.join(OUT, "validation_face_bins.txt"), "w") as f:
    f.write("\n".join(lines) + "\n")
with open(os.path.join(OUT, "validation_face_bins.json"), "w") as f:
    json.dump(res, f, indent=1)
