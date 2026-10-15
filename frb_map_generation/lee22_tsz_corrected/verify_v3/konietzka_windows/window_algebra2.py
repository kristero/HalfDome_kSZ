#!/usr/bin/env python3
"""Second pass: H10 reading ([10^A, 10^(B+1)) for 'AtoB') decade hit fractions, 'all' consistency, c=9 test."""
import json
from pathlib import Path

import h5py
import numpy as np

KON = Path("/home/cbllover/HalfDome/frb_catalog_comparison_outputs/haloDM-20260819T104507Z-1-001/haloDM")
OUT = Path("/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_verify_v3_20261001/konietzka_windows")
HD = Path("/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_corrected_20260930/rays")
Z = np.load(KON / "redshifts_DMhalo.npy")
NAMES = ["all", "mass9to13", "mass9to14", "mass10to12", "mass10to13", "mass10to14", "mass10to15", "mass11to14",
         "mass12to14", "mass12to15", "mass13to14"]
A = {n: np.load(KON / f"DMhalo_r200_{n}.npy", mmap_mode="r") for n in NAMES}


def get(n, zs):
    return np.asarray(A[n][int(np.flatnonzero(np.isclose(Z, zs))[0])], float)


def hd_hits(zs):
    tag = f"{zs:.1f}".replace(".", "p")
    f = HD / f"zsrc{tag}_nside4096_nrays120000_allhalos_sphere1p0_m200c_lee22_phys_seed42" / "halfdome_uniform_fixedz_foreground_mass_histograms.h5"
    with h5py.File(f, "r") as h:
        lab = [l.decode() if isinstance(l, bytes) else l for l in h["window_label"][:]]
        c = h["pdf_count"][:]
        return {l: float(c[i].sum() / 120000) for i, l in enumerate(lab)}


res = {}
for zs in (0.1, 0.5, 1.0):
    d = {n: get(n, zs) for n in NAMES}
    hd = hd_hits(zs)
    thr = 0.01
    # H10 decades
    dec = {
        "[1e9,1e10)": d["mass9to14"] - d["mass10to14"],
        "[1e10,1e11)": d["mass10to14"] - d["mass11to14"],
        "[1e11,1e12)": d["mass11to14"] - d["mass12to14"],
        "[1e12,1e13)": d["mass12to14"] - d["mass13to14"],
        "[1e13,1e14)": d["mass10to13"] - d["mass10to12"],
        "[1e14,1e15)": d["mass10to14"] - d["mass10to13"],
        "[1e15,1e16)": d["mass10to15"] - d["mass10to14"],
        "[1e13,1e15) = 13to14": d["mass13to14"],
    }
    r = {"H10_decade_hits": {k: float(np.mean(v > thr)) for k, v in dec.items()},
         "H10_decade_mean": {k: float(v.mean()) for k, v in dec.items()},
         "H10_decade_mean_per_hit": {k: float(v[v > thr].mean()) for k, v in dec.items()},
         "H10_decade_min": {k: float(v.min()) for k, v in dec.items()},
         "halfdome_hits": hd}
    # two independent routes to [1e12,1e13): 12to14-13to14 vs 10to12-(10to14-12to14)
    alt = d["mass10to12"] - (d["mass10to14"] - d["mass12to14"])
    r["[1e12,1e13) two routes max|diff|"] = float(np.abs(alt - dec["[1e12,1e13)"]).max())
    # c=9 (lo nominal) predicts 10to14 - 10to12 - 13to14 = DM[9e12,1e13) > 0 on a few % of rays
    r["max|10to14-10to12-13to14|"] = float(np.abs(d["mass10to14"] - d["mass10to12"] - d["mass13to14"]).max())
    # 'all' vs H10 union [1e9,1e16) = 9to14 + (10to15-10to14)
    u = d["mass9to14"] + dec["[1e15,1e16)"]
    x = d["all"] - u
    r["all_minus_union_1e9_1e16"] = dict(frac_pos=float(np.mean(x > thr)), frac_neg=float(np.mean(x < -thr)),
                                         mean=float(x.mean()), mean_neg=float(x[x < -thr].mean()) if np.any(x < -thr) else 0,
                                         min=float(x.min()), max=float(x.max()), mean_all=float(d["all"].mean()),
                                         neg_sum_over_all_sum=float(-x[x < 0].sum() / d["all"].sum()))
    res[str(zs)] = r
    print(f"\n=== z_s = {zs}")
    for k in dec:
        print(f"  {k:24s} hit {100 * r['H10_decade_hits'][k]:7.3f}%  mean {r['H10_decade_mean'][k]:8.3f}  per-hit {r['H10_decade_mean_per_hit'][k]:8.2f}  min {r['H10_decade_min'][k]:.3g}")
    print("  HalfDome hits:", {k: round(100 * v, 3) for k, v in hd.items()})
    print("  [1e12,1e13) routes max diff", r["[1e12,1e13) two routes max|diff|"], " max|10to14-10to12-13to14|", r["max|10to14-10to12-13to14|"])
    print("  all - union:", r["all_minus_union_1e9_1e16"])
(OUT / "window_algebra2.json").write_text(json.dumps(res, indent=1))
