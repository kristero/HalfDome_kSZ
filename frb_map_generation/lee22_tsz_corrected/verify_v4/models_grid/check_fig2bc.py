#!/usr/bin/env python3
"""Fig. 2b/2c caption numbers: CAMELS-TNG (27 CV boxes, precomputed stacks, electron fraction from the direct boxes) in
10^13.2-10^13.4 h^-1 Msun at z = 0.47 and 1.05, models (check_grid.py, own code) averaged over the same halos; geometric mean
of model / CAMELS over 0.3 < x < 1. Data read with numpy only."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_grid as G  # noqa: E402

T = Path("/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_tsz_verification_20260929/D_tng_camels/halo_tables")
P = np.load(T / "precomputed_halos.npz"); D = np.load(T / "direct_halos.npz")
xg = P["xg"]
for zt in (0.0, 0.47, 1.05):
    zd = np.unique(np.round(D["z"], 3)); zd = zd[np.argmin(np.abs(zd - zt))]
    d = (np.abs(D["z"] - zd) < 0.01) & (D["M200"] >= 1e13)
    fxe = np.nansum(D["nbar_xe"][d], 0) / np.nansum(D["nbar_fullion"][d], 0)
    for lo, hi in ((13.2, 13.4), (13.8, 14.0)):
        s = (np.abs(P["z"] - zt) < 0.01) & (P["M200"] * G.H >= 10 ** lo) & (P["M200"] * G.H < 10 ** hi)
        if s.sum() == 0:
            continue
        sel = (xg > .3) & (xg < 1)
        out = []
        for q in ("P", "n"):
            cam = np.nanmean(P["nbar"][s] * fxe if q == "n" else P["Pbar"][s], 0)
            mods = {"cor": [], "pipe": [], "bat": [], "noz": []}
            for m, z in zip(P["M200"][s], P["z"][s]):
                mh = m * G.H
                if q == "P":
                    f = G.lee_P(xg, mh, z); mods["cor"].append(f * (1 + z) ** 2); mods["pipe"].append(f)
                    mods["bat"].append(G.b12_Pe(xg, m, z)); mods["noz"].append(f)
                else:
                    f = G.lee_n(xg, mh, z); mods["cor"].append(f * (G.H * (1 + z)) ** 3 * G.XH ** 2)
                    mods["pipe"].append(f * G.NE2D); mods["bat"].append(G.b16_ne_over_n200(xg, m, z))
                    mods["noz"].append(f * G.H ** 3 * G.XH ** 2)
            r = {k: float(np.exp(np.nanmean(np.log(np.mean(v, 0)[sel] / cam[sel])))) for k, v in mods.items()}
            out.append("%s: cor %.4f pipe %.4f bat %.4f noz %.4f" % (q, r["cor"], r["pipe"], r["bat"], r["noz"]))
        print("z=%.2f (fxe z=%.2f, %d direct)  bin %.1f-%.1f  N=%d boxes=%d  mean z=%.3f |  %s" % (
            zt, zd, d.sum(), lo, hi, s.sum(), len(np.unique(P["box"][s])), P["z"][s].mean(), "  |  ".join(out)))
