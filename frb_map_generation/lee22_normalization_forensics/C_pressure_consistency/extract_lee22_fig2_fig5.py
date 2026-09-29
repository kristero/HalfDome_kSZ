#!/usr/bin/env python3
"""Extract the vector curves of Lee et al. 2022 (arXiv:2205.01710v1) Figure 2 (electron pressure,
source asset figures/fig_2.pdf) and Figure 5 (electron density, source asset figures/fig_4.pdf).

Both figures: z = 0, lowest Klypin-concentration quartile, mass bins 10^13.2-13.4 (panel 0) and
10^13.8-14.0 (panel 1) h^-1 Msun, ordinate x^3 A/A200 with x = r/R200.
Curve colours (matplotlib tab10): red = TNG simulation mean profile, green = with c + BPL,
grey = with c, no BPL, orange = no c + BPL (arXiv Table 7 / Table 8), brown = no c, no BPL,
cyan = Battaglia 2012 (pressure figure) / Battaglia 2016 (density figure).

Axis calibration uses the drawn major tick marks (asserted below); for fig_4 the calibration is the
one validated in frb_map_generation/lee22_fig5_audit/compare_fig5.py.

Requires PyMuPDF (not in the halfdome env); run with
  PYTHONPATH=<dir with pymupdf> /home/cbllover/miniconda3/envs/halfdome/bin/python extract_lee22_fig2_fig5.py
The source PDFs are copies of the arXiv source (https://arxiv.org/e-print/2205.01710v1) kept in
outputs/lee22_normalization_forensics_20260928/C_pressure_consistency/reference/.
"""
import csv
from pathlib import Path

import numpy as np
import pymupdf

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "outputs/lee22_normalization_forensics_20260928/C_pressure_consistency"
REF = OUT / "reference"

COLORS = {"simulation": (0.8392, 0.1529, 0.1569), "with_c_bpl": (0.1725, 0.6275, 0.1725),
          "with_c_no_bpl": (0.498, 0.498, 0.498), "without_c_bpl": (1.0, 0.498, 0.0549),
          "without_c_no_bpl": (0.549, 0.3373, 0.2941), "battaglia": (0.0902, 0.7451, 0.8118)}
PANEL_OFFSET = 892.8

# (x pixel of 10^-1, x pixel of 10^0, y pixel of 10^0 on the ordinate, pixels per dex in y, top-panel y limit)
CALIB = {
    "fig_2.pdf": dict(x01=388.093, x1=889.621, y_ref=44.764, y_ref_val=-1.0, y_dex=124.784, ymax=432.0,
                      text="Battaglia et al. (2012)"),
    "fig_4.pdf": dict(x01=404.0775146484375, x1=905.6053466796875, y_ref=60.74560546875, y_ref_val=0.0,
                      y_dex=(397.4245300292969 - 60.74560546875) / 3, ymax=432.0,
                      text="Without c and with BPL in mass"),
}


def check_ticks(page, cal, name):
    """Verify that the calibration reproduces the drawn major ticks."""
    ticks_y, ticks_x = [], []
    for d in page.get_drawings():
        if d["color"] is None or len(d["items"]) != 1 or d["items"][0][0] != "l":
            continue
        if max(abs(v) for v in d["color"]) > 1e-6:
            continue
        p, q = d["items"][0][1], d["items"][0][2]
        if abs(p.y - q.y) < 1e-6 and abs(p.x - 123.2) < 1e-3 and q.x < p.x and 3.0 < abs(q.x - p.x) < 5.0 and p.y < cal["ymax"]:
            ticks_y.append(p.y)
        if abs(p.x - q.x) < 1e-6 and 3.0 < abs(q.y - p.y) < 5.0 and p.x < 1016 and abs(p.y - 431.208) < 1e-2:
            ticks_x.append(p.x)
    return sorted(set(round(t, 3) for t in ticks_y)), sorted(set(round(t, 3) for t in ticks_x))


def extract(name):
    cal = CALIB[name]
    with pymupdf.open(REF / name) as doc:
        page = doc[0]
        assert cal["text"] in page.get_text(), name
        ty, tx = check_ticks(page, cal, name)
        drawings = page.get_drawings()
    if name == "fig_2.pdf":
        # major y ticks at 10^-1..10^-4 and major x ticks at 10^-1, 10^0
        assert np.allclose(ty, [44.764, 169.548, 294.331, 419.115], atol=2e-3), ty
        assert np.allclose(np.diff(ty), cal["y_dex"], atol=2e-3)
        assert np.allclose(tx, [388.093, 889.621], atol=2e-3), tx
    curves = {}
    for col in (0, 1):
        for cname, rgb in COLORS.items():
            cands = [d for d in drawings if d["color"] is not None and len(d["items"]) == 102
                     and max(abs(a - b) for a, b in zip(d["color"], rgb)) < 2e-3
                     and d["rect"].y1 < cal["ymax"] and int(d["rect"].x0 > 1032) == col]
            assert len(cands) == 1, (name, col, cname, len(cands))
            items = cands[0]["items"]
            assert all(it[0] == "l" for it in items)
            pts = [items[0][1]] + [it[2] for it in items]
            px = np.array([p.x for p in pts]) - col * PANEL_OFFSET
            py = np.array([p.y for p in pts])
            x = 10 ** ((px - cal["x1"]) / (cal["x1"] - cal["x01"]))
            q = 10 ** ((cal["y_ref"] - py) / cal["y_dex"] + cal["y_ref_val"])
            assert np.all(np.diff(x) > 0)
            curves[(col, cname)] = (x, q)
    return curves


def main():
    rows = []
    for name, quantity in (("fig_2.pdf", "x3_Pe_over_P200"), ("fig_4.pdf", "x3_ne_over_n200")):
        curves = extract(name)
        for (col, cname), (x, q) in curves.items():
            rows += [(name, quantity, col, cname, xi, qi) for xi, qi in zip(x, q)]
        # cross-check fig_4 against the earlier audit extraction
        if name == "fig_4.pdf":
            prev = OUT.parents[1] / "lee22_fig5_audit_20260922/figure5_vector_curves.csv"
            if prev.exists():
                ref = {}
                with prev.open() as f:
                    for r in csv.DictReader(f):
                        ref.setdefault((int(r["panel"]), r["curve"]), []).append(float(r["x3_ne_over_n200"]))
                for (col, cname), (x, q) in curves.items():
                    key = (col, "battaglia16" if cname == "battaglia" else cname)
                    if key in ref:
                        assert np.allclose(q, ref[key], rtol=1e-9), key
                print("fig_4 extraction identical to lee22_fig5_audit_20260922/figure5_vector_curves.csv")
    with (OUT / "lee22_fig2_fig5_vector_curves.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["figure_asset", "quantity", "panel", "curve", "x_r_over_R200", "value"])
        w.writerows(rows)
    print("wrote", OUT / "lee22_fig2_fig5_vector_curves.csv", len(rows), "rows")


if __name__ == "__main__":
    main()
