#!/usr/bin/env python3
"""Extract the vector curves of Lee22 (arXiv:2205.01710v1) Figures 2 and 5.

Figure 2 (source file figures/fig_2.pdf): x^3 P_e/P200 at z=0, lowest-concentration quartile,
    mass bins 10^13.2-13.4 and 10^13.8-14.0 h^-1 Msun, with the cyan Battaglia et al. (2012) line.
Figure 5 (source file figures/fig_4.pdf): x^3 n_e/n200, same bins, with the cyan Battaglia (2016) line.

Axis calibration uses the labelled major ticks read from the vector PDF (no rasterisation).
Needs PyMuPDF (run with the scratch venv: pdfenv/bin/python).
Output: <OUT>/lee22_fig2_fig5_vector_curves.csv
"""
from pathlib import Path
import csv
import numpy as np
import pymupdf


def to_rgb(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "outputs/lee22_normalization_forensics_20260928/B_paper_code_forensics"
SRC = OUT / "arxiv_src/x/figures"
COLORS = {"simulation": "#d62728", "with_c_bpl": "#2ca02c", "with_c_no_bpl": "#7f7f7f",
          "without_c_bpl": "#ff7f0e", "without_c_no_bpl": "#8c564b", "battaglia": "#17becf"}

# (x at 10^-1, x at 10^0, panel offset, y of major ticks list (value exponent, y))
CAL = {
    "fig2_pressure": dict(file="fig_2.pdf", x01=388.093, x1=889.621, offset=892.8,
                          yt=[(-1, 44.764), (-2, 169.548), (-3, 294.331), (-4, 419.115)], split=1016.0),
    "fig5_density": dict(file="fig_4.pdf", x01=404.0775146484375, x1=905.6053466796875, offset=892.8,
                         yt=[(0, 60.74560546875), (-1, 172.971923828125), (-2, 285.1982116699219),
                             (-3, 397.4245300292969)], split=1031.984),
}


def extract(name, cal):
    doc = pymupdf.open(SRC / cal["file"])
    page = doc[0]
    drawings = page.get_drawings()
    yt = np.array(cal["yt"], dtype=float)
    # linear fit exponent vs y (checks consistency of the major ticks)
    A = np.vstack([yt[:, 1], np.ones(len(yt))]).T
    slope, icpt = np.linalg.lstsq(A, yt[:, 0], rcond=None)[0]
    resid = np.max(np.abs(A @ [slope, icpt] - yt[:, 0]))
    assert resid < 1e-4, resid
    out = {}
    for col in (0, 1):
        for cname, hexc in COLORS.items():
            rgb = np.array(to_rgb(hexc))
            cands = [d for d in drawings if len(d["items"]) == 102 and d["color"] is not None
                     and np.max(np.abs(np.array(d["color"]) - rgb)) < 2e-3
                     and d["rect"].y1 < 432 and int(d["rect"].x0 > cal["split"]) == col]
            if len(cands) != 1:
                print(name, col, cname, "candidates:", len(cands))
                continue
            items = cands[0]["items"]
            pts = [items[0][1]] + [it[2] for it in items]
            px = np.array([p.x for p in pts]) - col * cal["offset"]
            py = np.array([p.y for p in pts])
            x = 10 ** (-1 + (px - cal["x01"]) / (cal["x1"] - cal["x01"]))
            q = 10 ** (slope * py + icpt)
            out[(col, cname)] = (x, q)
    return out, resid


def main():
    rows = []
    for name, cal in CAL.items():
        curves, resid = extract(name, cal)
        print(name, "tick-fit residual (dex):", resid, "curves:", len(curves))
        for (col, cname), (x, q) in curves.items():
            print(f"  panel {col} {cname:18s} x=[{x.min():.4f},{x.max():.4f}] q=[{q.min():.3e},{q.max():.3e}]")
            rows += [(name, col, cname, xi, qi) for xi, qi in zip(x, q)]
    with open(OUT / "lee22_fig2_fig5_vector_curves.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["figure", "panel", "curve", "x_r_over_R200", "x3_times_ratio"])
        w.writerows(rows)


if __name__ == "__main__":
    main()
