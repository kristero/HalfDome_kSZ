"""Read-only check (2026-10-14): the n_e kT_X pressure stack over the geometric pressure stack equals the per-halo
T_X/T_e averaged with weights w P_e (w = inclusion weight), shell by shell, for the 10 calibrated bins.
Prints per bin the largest |difference| over the shells that pass the strict resolution rule (the shells the band
ratios use), the same over all shells (unresolved inner shells included, for reference), and the 3-4 R200c band median.
Run: python -B check_ntx_weighting.py   (under 1 s; reads the cache memory-mapped)
"""
import sys
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(OUT / "core"))
import stackcore as sc  # noqa: E402


def K(f, e=None):
    return sc.make_key(f, e or ("weighted" if f in ("T_X", "T_e") else "geometric"), grid="fit")


Pg = np.asarray(sc.per_halo(K("P")), float)
PnTX = np.asarray(sc.per_halo(K("P", "nT_X")), float)
Te = np.asarray(sc.per_halo(K("T_e")), float)
TX = np.asarray(sc.per_halo(K("T_X")), float)
E = sc.edges("fit")
A, B = E[:-1], E[1:]
XC = 0.75 * (B**4 - A**4) / (B**3 - A**3)
RES_OPTS = dict(min_count=20, softening_factor=4.0, hsml_factor=2.0, min_edge=0.04)
gmax = gmax_all = 0.0
for b in [b for b in sc.mass_bins("M200c_0.2") if b["lo"] >= 13.5 - 1e-9]:
    r, w = b["rows"], b["weights"]
    direct, pw = np.full(len(XC), np.nan), np.full(len(XC), np.nan)
    for s in range(len(XC)):
        p, n, te, tx = Pg[r, s], PnTX[r, s], Te[r, s], TX[r, s]
        fp, fn = np.isfinite(p), np.isfinite(n)
        if fp.sum() == 0 or fn.sum() == 0:
            continue
        direct[s] = (np.sum(w[fn] * n[fn]) / np.sum(w[fn])) / (np.sum(w[fp] * p[fp]) / np.sum(w[fp]))
        ok = fp & np.isfinite(tx / te)
        if ok.sum() and np.sum(w[ok] * p[ok]) > 0:
            pw[s] = np.sum(w[ok] * p[ok] * tx[ok] / te[ok]) / np.sum(w[ok] * p[ok])
    res = sc.resolution_mask(r, "fit", "strict", **RES_OPTS)
    d_all = np.abs(direct - pw)
    d = d_all[res & np.isfinite(d_all)]
    gmax, gmax_all = max(gmax, d.max()), max(gmax_all, np.nanmax(d_all))
    s34 = (XC >= 3.0) & (XC < 4.0) & res
    print(f"{b['label']}: max |stack ratio - P-weighted T_X/T_e| = {d.max():.2e} over {d.size} resolved shells "
          f"({np.nanmax(d_all):.2e} over all shells); "
          f"3-4 R200c median {np.nanmedian(direct[s34]):.4f} vs {np.nanmedian(pw[s34]):.4f}")
print(f"all calibrated bins: max |difference| = {gmax:.2e} (resolved shells), {gmax_all:.2e} (all shells)")
