"""Mass-binning group: accuracy test of rebin.rescale (the radial rescaling used for bin-mean normalisation).

Analytic profiles (a gNFW pressure profile with Arnaud-like parameters in R200c units and a beta-model density) are
averaged exactly over the source shells (x200 grid, own R200c units), moved with rebin.rescale to the fit grid in
units of Rbar = R200c / s, and compared with the exact shell averages of the scaled profile f(x'/s) on the fit grid.
s covers the range met in the bins (0.2 dex: 0.93-1.09; 0.5 dex: 0.85-1.30).
Output: work/res_rebin_test.json.
"""
import json

import numpy as np

from mbcommon import EDGES_FIT, WORK, sc, sig
from rebin import rescale

EX = np.asarray(sc.edges("x200"), float)
S_VALUES = (0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.3)


def gnfw(x, c=2.5, g=0.31, a=1.05, b=5.49):
    y = c * x
    return 1.0 / (y**g * (1.0 + y**a)**((b - g) / a))


def beta_model(x, xc=0.1, bb=2.0 / 3.0):
    return (1.0 + (x / xc)**2)**(-1.5 * bb)


def shell_avg(f, e, s=1.0):
    out = []
    for lo, hi in zip(e[:-1], e[1:]):
        xx = np.geomspace(max(lo, 1e-6), hi, 8001)
        yy = f(xx / s) * xx**2
        out.append(np.sum(0.5 * (yy[1:] + yy[:-1]) * np.diff(xx)) / ((hi**3 - lo**3) / 3.0))
    return np.array(out)


res = {"s_values": list(S_VALUES), "profiles": {}, "notes": (
    "fractional |rescaled / exact - 1| per fit shell, for shells whose inner edge is >= 0.04 (the strict rule's "
    "min_edge) and whose outer edge lies inside the source coverage (4 R200c_h, i.e. <= 4 s); 'x>=0.1' restricts to "
    "inner edges >= 0.1.  s = 1 must reproduce the source exactly because every fit edge is an x200 edge.")}
for name, f in (("gNFW", gnfw), ("beta", beta_model)):
    src = shell_avg(f, EX)[None]
    per_s, allr, allr01 = {}, [], []
    for s in S_VALUES:
        got = rescale(src, EX, np.array([s]), EDGES_FIT)[0]
        true = shell_avg(f, EDGES_FIT, s)
        ok = np.isfinite(got) & (EDGES_FIT[:-1] >= 0.04) & (EDGES_FIT[1:] <= 4.0 * s + 1e-9)
        rel = np.abs(got[ok] / true[ok] - 1)
        ok01 = ok & (EDGES_FIT[:-1] >= 0.1)
        rel01 = np.abs(got[ok01] / true[ok01] - 1)
        per_s[str(s)] = {"median": sig(np.median(rel)), "max": sig(rel.max()), "n_shells": int(ok.sum()),
                         "median_x>=0.1": sig(np.median(rel01)), "max_x>=0.1": sig(rel01.max())}
        if s != 1.0:
            allr.append(rel)
            allr01.append(rel01)
    allr, allr01 = np.concatenate(allr), np.concatenate(allr01)
    narrow = np.concatenate([np.abs(rescale(src, EX, np.array([s]), EDGES_FIT)[0] / shell_avg(f, EDGES_FIT, s) - 1)[
        np.isfinite(rescale(src, EX, np.array([s]), EDGES_FIT)[0]) & (EDGES_FIT[:-1] >= 0.04)
        & (EDGES_FIT[1:] <= 4.0 * s + 1e-9)] for s in (0.95, 1.05)])
    res["profiles"][name] = {"per_s": per_s,
                             "all_s_ne_1": {"median": sig(np.median(allr)), "max": sig(allr.max()),
                                            "p95": sig(np.percentile(allr, 95))},
                             "all_s_ne_1_x>=0.1": {"median": sig(np.median(allr01)), "max": sig(allr01.max())},
                             "s_0.95_1.05": {"median": sig(np.median(narrow)), "max": sig(narrow.max())}}
(WORK / "res_rebin_test.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
