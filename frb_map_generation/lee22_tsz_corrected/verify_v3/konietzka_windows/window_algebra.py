#!/usr/bin/env python3
"""Independent window algebra of the Konietzka TNG300 halo-DM arrays (verify_v3, 2026-10-01).
Reads the pre-release .npy arrays read-only (never copied). Writes only a small JSON of summary numbers."""
import json
from pathlib import Path

import numpy as np

KON = Path("/home/cbllover/HalfDome/frb_catalog_comparison_outputs/haloDM-20260819T104507Z-1-001/haloDM")
OUT = Path("/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_verify_v3_20261001/konietzka_windows")
Z = np.load(KON / "redshifts_DMhalo.npy")
NAMES = ["all", "mass9to13", "mass9to14", "mass10to12", "mass10to13", "mass10to14", "mass10to15", "mass11to14",
         "mass12to14", "mass12to15", "mass13to14"]
A = {n: np.load(KON / f"DMhalo_r200_{n}.npy", mmap_mode="r") for n in NAMES}
A["lss"] = np.load(KON / "DMall_IGM_halo.npy", mmap_mode="r")


def get(n, zs):
    i = int(np.flatnonzero(np.isclose(Z, zs))[0])
    return np.asarray(A[n][i], float)


res = {}
for zs in (0.1, 0.5, 1.0, 1.5, 2.0):
    d = {n: get(n, zs) for n in A}
    r = {}
    for n in NAMES:
        v = d[n]
        r[f"hit_{n}"] = float(np.mean(v > 0))
        r[f"mean_{n}"] = float(v.mean())
        r[f"min_{n}"] = float(v.min())
    # nesting differences
    def diff(a, b, thr=0.0):
        x = d[a] - d[b]
        return dict(frac_pos=float(np.mean(x > thr)), frac_pos_0p01=float(np.mean(x > 0.01)),
                    frac_neg=float(np.mean(x < -1e-9)), frac_neg_0p01=float(np.mean(x < -0.01)),
                    n_pos_0p01=int(np.sum(x > 0.01)), mean=float(x.mean()),
                    median_pos=float(np.median(x[x > 0.01])) if np.any(x > 0.01) else None,
                    max=float(x.max()), min=float(x.min()))
    r["10to15-10to14"] = diff("mass10to15", "mass10to14")
    r["12to15-12to14"] = diff("mass12to15", "mass12to14")
    r["10to14-10to13"] = diff("mass10to14", "mass10to13")
    r["9to14-9to13"] = diff("mass9to14", "mass9to13")
    r["9to14-10to14"] = diff("mass9to14", "mass10to14")
    r["9to13-10to13"] = diff("mass9to13", "mass10to13")
    r["all-10to15"] = diff("all", "mass10to15")
    r["all-9to14"] = diff("all", "mass9to14")
    r["10to14-11to14"] = diff("mass10to14", "mass11to14")
    r["11to14-12to14"] = diff("mass11to14", "mass12to14")
    r["12to14-13to14"] = diff("mass12to14", "mass13to14")
    r["10to13-10to12"] = diff("mass10to13", "mass10to12")
    # nominal identities (exact if windows are disjoint & nested as labelled)
    nom = {
        "10to14 = 10to13 + 13to14": d["mass10to14"] - d["mass10to13"] - d["mass13to14"],
        "10to14 = 10to12 + 12to14": d["mass10to14"] - d["mass10to12"] - d["mass12to14"],
        "10to12 = 10to14 - 12to14": d["mass10to12"] - (d["mass10to14"] - d["mass12to14"]),
        "10to13 = 10to14 - 13to14": d["mass10to13"] - (d["mass10to14"] - d["mass13to14"]),
        "12to15 - 12to14 = 10to15 - 10to14": (d["mass12to15"] - d["mass12to14"]) - (d["mass10to15"] - d["mass10to14"]),
        # x10 upper edge hypothesis H10: [lo, 10 hi): 10to14 = 10to12 + 13to14 exactly (both [1e10,1e13) U [1e13,1e15))
        "H10: 10to14 = 10to12 + 13to14": d["mass10to14"] - d["mass10to12"] - d["mass13to14"],
        # H10: 10to13 = [1e10,1e14) = 10to12 + [1e13,1e14); 13to14 = [1e13,1e15) => 10to13-10to12 <= 13to14
        "H10: 13to14 - (10to13 - 10to12) >= 0": d["mass13to14"] - (d["mass10to13"] - d["mass10to12"]),
        # H10: 12to14 = [1e12,1e15) = (10to12 - [1e10,1e12)) + 13to14 ; [1e10,1e12) = 10to14 - 12to14 (H10: [1e10,1e15) - [1e12,1e15))
        "H10: 12to14 = 10to12 - (10to14 - 12to14) + 13to14": d["mass12to14"] - (d["mass10to12"] - (d["mass10to14"] - d["mass12to14"]) + d["mass13to14"]),
        "9to14 = 9to13 + 13to14": d["mass9to14"] - d["mass9to13"] - d["mass13to14"],
    }
    scale = np.maximum(d["mass13to14"].mean(), 1e-30)
    r["identities"] = {}
    for k, x in nom.items():
        r["identities"][k] = dict(frac_abs_gt_0p01=float(np.mean(np.abs(x) > 0.01)),
                                  frac_abs_gt_1=float(np.mean(np.abs(x) > 1.0)),
                                  mean=float(x.mean()), mean_abs=float(np.abs(x).mean()),
                                  max_abs=float(np.abs(x).max()), mean_over_mean13to14=float(x.mean() / scale))
    # overlap per ray of 13to14 with 10to13
    hit13 = d["mass13to14"] > 0.01
    ov = d["mass10to13"] + d["mass13to14"] - d["mass10to14"]
    fr = np.where(hit13, ov / np.where(hit13, d["mass13to14"], 1), np.nan)
    r["overlap_13to14_in_10to13"] = dict(dm_fraction=float(ov.sum() / d["mass13to14"].sum()),
                                         rays_full=float(np.mean(np.abs(fr[hit13] - 1) < 1e-3)),
                                         rays_none=float(np.mean(np.abs(fr[hit13]) < 1e-3)),
                                         rays_partial=float(np.mean((fr[hit13] > 1e-3) & (fr[hit13] < 1 - 1e-3))),
                                         mean_dm_per_hit_none=float(d["mass13to14"][hit13][np.abs(fr[hit13]) < 1e-3].mean()),
                                         mean_dm_per_hit_full=float(d["mass13to14"][hit13][np.abs(fr[hit13] - 1) < 1e-3].mean()))
    # 10to12 contains any 13to14 DM?
    ov12 = d["mass10to12"] + d["mass13to14"] - d["mass10to14"]
    r["10to12+13to14-10to14"] = dict(frac_abs_gt_0p01=float(np.mean(np.abs(ov12) > 0.01)), mean=float(ov12.mean()))
    res[str(zs)] = r
    print(f"\n=== z_s = {zs}")
    for k in ("10to15-10to14", "12to15-12to14", "10to14-10to13", "9to14-9to13", "all-10to15", "all-9to14", "9to14-10to14"):
        print(f"  {k:16s} pos>0.01: {100 * r[k]['frac_pos_0p01']:.4f}% (n={r[k]['n_pos_0p01']}) neg: {100 * r[k]['frac_neg_0p01']:.4f}%  mean {r[k]['mean']:.4f} median_pos {r[k]['median_pos']}")
    for k, v in r["identities"].items():
        print(f"  {k:55s} |res|>0.01 on {100 * v['frac_abs_gt_0p01']:.3f}%  >1 on {100 * v['frac_abs_gt_1']:.3f}%  mean/mean13to14 {v['mean_over_mean13to14']:+.4f} max {v['max_abs']:.3g}")
    print("  overlap", r["overlap_13to14_in_10to13"])
    print("  hits", {n: round(100 * r[f'hit_{n}'], 3) for n in NAMES})
(OUT / "window_algebra.json").write_text(json.dumps(res, indent=1))
