#!/usr/bin/env python3
"""Fig. 1a/1b caption numbers from the digitized Lee22 TNG profiles (the doc's npz, data only) with the models of check_grid.py."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_grid as G  # noqa: E402  (own independent models)

D = np.load("/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_corrected_20260930/doc/lee22_fig2_fig5_full_digitization.npz")
BINS = ((13.2, 13.4), (13.8, 14.0))
for q in ("P", "n"):
    for b, (lo, hi) in enumerate(BINS):
        x = D["%s_%d_x" % (q, b)]; sim = D["%s_%d_sim" % (q, b)] / x ** 3
        sig, sc = D["%s_%d_sigma" % (q, b)], D["%s_%d_scatter" % (q, b)]
        N = np.median((sc / sig) ** 2)
        mh = 10 ** (0.5 * (lo + hi)); m = mh / G.H
        conv = 1.0 if q == "P" else G.H ** 3 * G.XH ** 2      # z = 0
        simp = sim * conv
        lee = G.lee_P(x, mh, 0.0) if q == "P" else G.lee_n(x, mh, 0.0) * conv
        pipe = G.lee_P(x, mh, 0.0) if q == "P" else G.lee_n(x, mh, 0.0) * G.NE2D
        bat = G.b12_Pe(x, m, 0.0) if q == "P" else G.b16_ne_over_n200(x, m, 0.0)
        s = (x > .3) & (x < 1)
        gm = lambda a: float(np.exp(np.mean(np.log(a[s] / simp[s]))))
        print("%s bin %.1f-%.1f: N~(scatter/sigma)^2 median %.0f;  model/TNG gm 0.3-1: corrected %.3f  pipeline %.3f  Battaglia %.3f"
              "  [corrected min/max %.3f/%.3f]" % (q, lo, hi, N, gm(lee), gm(pipe), gm(bat), (lee[s] / simp[s]).min(),
                                                  (lee[s] / simp[s]).max()))
