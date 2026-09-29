#!/usr/bin/env python3
"""verify_0 / step 4: independent re-extraction of Lee22 Fig. 5 (arXiv source figures/fig_4.pdf) curves.

pdftocairo -svg converts the vector figure; this script parses the SVG paths itself (no reuse of the earlier
audit's extraction), finds the axis tick marks, maps page coordinates to (r/R200, x^3 n_e/n200) and writes the
orange ('without c, with BPL in mass') and red ('Simulation') curves of the two top panels.
"""
import re
import sys
from pathlib import Path

import numpy as np

OUT = Path("/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_normalization_forensics_20260928/verify_0")
svg = (OUT / "fig5_from_arxiv.svg").read_text()

path_re = re.compile(r"<path ([^>]*)/>")
num = r"-?[0-9.]+(?:e-?[0-9]+)?"


def parse_d(d):
    toks = re.findall(r"[MLCZ]|" + num, d)
    pts, i, cmd = [], 0, None
    while i < len(toks):
        t = toks[i]
        if t in "MLCZ":
            cmd = t; i += 1
            if cmd == "Z":
                continue
        if cmd in ("M", "L"):
            pts.append((float(toks[i]), float(toks[i + 1]))); i += 2
        elif cmd == "C":
            pts.append((float(toks[i + 4]), float(toks[i + 5]))); i += 6  # keep end points only
        else:
            i += 1
    return np.array(pts)


paths = []
for m in path_re.finditer(svg):
    attrs = m.group(1)
    st = re.search(r'style="([^"]*)"', attrs)
    d = re.search(r' d="([^"]*)"', attrs)
    tr = re.search(r'transform="matrix\(([^)]*)\)"', attrs)
    if not d:
        continue
    style = st.group(1) if st else ""
    sc = re.search(r"stroke:(rgb\([^)]*\)|none)", style)
    lw = re.search(r"stroke-width:([0-9.]+)", style)
    dash = "stroke-dasharray" in style
    P = parse_d(d.group(1))
    if len(P) == 0:
        continue
    if tr:
        a, b, c, dd, e, f = [float(v) for v in tr.group(1).split(",")]
        P = np.column_stack([a * P[:, 0] + c * P[:, 1] + e, b * P[:, 0] + dd * P[:, 1] + f])
    paths.append(dict(stroke=sc.group(1) if sc else "none", lw=float(lw.group(1)) if lw else None, dash=dash, P=P))

BLACK = "rgb(0%,0%,0%)"
ORANGE = "rgb(100%,49.803162%,5.490112%)"
RED = "rgb(83.920288%,15.293884%,15.686035%)"
CYAN = "rgb(9.01947%,74.508667%,81.175232%)"
# tick marks: black 2-point paths that are purely horizontal or vertical and short
ticks = []
for p in paths:
    if p["stroke"] == BLACK and len(p["P"]) == 2:
        (x0, y0), (x1, y1) = p["P"]
        L = np.hypot(x1 - x0, y1 - y0)
        if 0 < L < 15:
            ticks.append((x0, y0, x1, y1, L))
ticks = np.array(ticks)
if "--dump" in sys.argv:
    for t in ticks:
        print(np.round(t, 3))
    for p in paths:
        if p["stroke"] in (ORANGE, RED, CYAN):
            print(p["stroke"], p["lw"], p["dash"], len(p["P"]), p["P"][:, 0].min(), p["P"][:, 0].max(), p["P"][:, 1].min(), p["P"][:, 1].max())

# ---- axis calibration from tick marks (top panels) ----
XT = {0: (404.68, 905.664), 1: (1296.512, 1797.496)}  # major x ticks at r/R200 = 0.1 and 1 (left, right panel)
YT = (396.992, 284.891, 172.785, 60.68)                 # major y ticks at 1e-3, 1e-2, 1e-1, 1e0
ydec = np.polyfit([-3, -2, -1, 0], YT, 1)                # page_y = a*log10(y) + b
print("y px/decade", ydec[0], "residual", np.array(YT) - np.polyval(ydec, [-3, -2, -1, 0]))


def to_data(P, panel):
    x0, x1 = XT[panel]
    lx = -1 + (P[:, 0] - x0) / (x1 - x0)
    ly = (P[:, 1] - ydec[1]) / ydec[0]
    return 10 ** lx, 10 ** ly


curves = {}
for p in paths:
    if len(p["P"]) != 103 or p["P"][:, 1].max() > 430:
        continue
    panel = 0 if p["P"][:, 0].max() < 1000 else 1
    name = {ORANGE: "orange_noc_bpl", RED: "red_simulation", CYAN: "cyan_B16"}.get(p["stroke"])
    if name:
        curves[(panel, name)] = to_data(p["P"], panel)
with open(OUT / "step4_fig5_curves_reextracted.csv", "w") as f:
    f.write("panel,curve,x_r_over_R200,x3_ne_over_n200\n")
    for (panel, name), (x, y) in sorted(curves.items()):
        for a, b in zip(x, y):
            f.write(f"{panel},{name},{a:.8e},{b:.8e}\n")

# ---- model comparison (z = 0), Table 8 from the TeX parse of step 1 ----
sys.path.insert(0, str(Path(__file__).parent))
from step1_params_from_tex import noc_params, t8  # noqa: E402


def gnfw(x, xc, be):
    s = x / xc
    return s ** -0.3 * (1 + s) ** (-be)


results = {}
for panel, lmh in ((0, 13.3), (1, 13.9)):
    x, y = curves[(panel, "orange_noc_bpl")]
    for h in (0.68, 0.6774):
        M = 10 ** lmh / h
        for pivot in ("mcut", "1e14", "1e14h"):
            n0, xc, be = noc_params(M, 0.0, h, pivot)
            model = x ** 3 * n0 * gnfw(x, xc, be)
            r = y / model - 1
            key = f"panel{panel}_logMh{lmh}_h{h}_pivot{pivot}"
            results[key] = dict(fig_over_model_minus1_mean=float(r.mean()), min=float(r.min()), max=float(r.max()),
                                model_over_fig_mean=float((model / y).mean()))
            print(key, {k: round(v, 4) for k, v in results[key].items()})
import json  # noqa: E402
json.dump(results, open(OUT / "step4_fig5_pivot_test.json", "w"), indent=1)
