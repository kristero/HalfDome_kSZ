#!/usr/bin/env python
"""Round-4 re-verification, part C (read-only, own numpy from the raw cache): section 3's lognormal check for P.

Per main bin and band: band median of mean/geomean against the band median of exp(sd_ln^2/2), and their relative
difference (unrounded); also the unrounded n_e kT_X band ratio at <0.1 R200c.
Run: python -B verify/v4_lognormal.py
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
GRP = HERE.parent
CACHE = GRP.parents[1] / "cache"
IDX = json.loads((CACHE / "index.json").read_text())["variants"]
TAB = dict(np.load(CACHE / "halo_table.npz"))
EDG = np.load(CACHE / "edges_fit.npy")
A, B = EDG[:-1], EDG[1:]
XC = 0.75 * (B**4 - A**4) / (B**3 - A**3)
CNT = np.load(CACHE / "counts_fit.npy")
HS = np.load(CACHE / "hsml_fit.npy")
BANDS = [(0.0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 3.0), (3.0, 4.0)]
BL = ["<0.1", "0.1-0.25", "0.25-0.5", "0.5-1", "1-1.5", "1.5-2", "2-3", "3-4"]
P = np.load(CACHE / IDX["P|geometric|removed|all|all|fit|P200"]["file"]).astype(float)
PnTX = np.load(CACHE / IDX["P|nT_X|removed|all|all|fit|P200"]["file"]).astype(float)
edges = TAB["mass_edges"]
worst = {}
for k in range(len(edges) - 1):
    lo_m, hi_m = edges[k], edges[k + 1]
    if lo_m < 13.5 - 1e-9 or hi_m > 15.1 + 1e-9:
        continue
    rows = np.flatnonzero((TAB["logm200c"] >= lo_m) & (TAB["logm200c"] < hi_m))
    w = TAB["weight"][rows]
    v = P[rows]
    x = EDG[:-1]
    eps = TAB["eps_over_r200c"][rows]
    with np.errstate(invalid="ignore"):
        res = (x[None, :] >= np.maximum(0.04, 4 * eps[:, None])) & (CNT[rows] >= 20) & (x[None, :] >= 2 * HS[rows])
    strict = res.all(0)
    fin = np.isfinite(v)
    pos = fin & (v > 0)
    Wf = np.where(fin, w[:, None], 0.0)
    Wl = np.where(pos, w[:, None], 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = (Wf * np.where(fin, v, 0)).sum(0) / Wf.sum(0)
        lv = np.where(pos, np.log(np.where(pos, v, 1.0)), 0.0)
        tl = Wl.sum(0)
        ml = (Wl * lv).sum(0) / tl
        varl = (Wl * (lv - ml) ** 2).sum(0) / tl
        corr = 1 - (Wl**2).sum(0) / tl**2
        sd = np.sqrt(varl / corr)
        mgm = mean / np.exp(ml)
        pred = np.exp(sd**2 / 2)
        nt = (Wf * np.where(fin, PnTX[rows], 0)).sum(0) / Wf.sum(0) / mean
    lab = f"{lo_m:.1f}-{hi_m:.1f}"
    out = []
    for (lo, hi), bl in zip(BANDS, BL):
        m = (XC >= lo) & (XC < hi) & strict
        if not m.any():
            continue
        a, p = np.median(mgm[m]), np.median(pred[m])
        d = a / p - 1
        worst[bl] = max(worst.get(bl, 0), abs(d))
        out.append(f"{bl}: {a:.5f}/{p:.5f} ({d:+.4%})")
        if bl == "<0.1":
            out.append(f"[nT_X <0.1 = {np.median(nt[m]):.5f}]")
    print(lab + "  " + "  ".join(out))
print("largest |mean/geomean / lognormal - 1| per band over the 8 main bins:",
      {k: f"{v:.3%}" for k, v in worst.items()})
