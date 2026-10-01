#!/usr/bin/env python3
"""Everything drawn in Lee et al. 2022 (arXiv:2205.01710v1) Fig. 2 (electron pressure, source asset figures/fig_2.pdf) and
Fig. 5 (electron density, figures/fig_4.pdf), read from the vector paths (2026-10-01). Extends
lee22_normalization_forensics/C_pressure_consistency/extract_lee22_fig2_fig5.py (top-panel model curves, same calibration)
with what the replicas of the two figures need:
  * the TNG measurement: thick line at 103 log-spaced radii (x = 0.0425-1.48); errorbar data line (red, width 1.5), red bars (+-sigma, error of the
    mean) and the translucent blue bars (+-halo-to-halo scatter) at every second radius (52 points);
  * the two residual rows of each panel, Delta A/A and Delta A/sigma, for the four fits (linear axes calibrated on the
    drawn ticks 0.5/0/-0.5 (Fig. 2), 0.25/0/-0.25 (Fig. 5) and 10/0/-10).
Writes outputs/lee22_corrected_20260930/doc/lee22_fig2_fig5_full_digitization.npz and checks the residual definition:
Delta A/A = (A_sim - A_fit)/A_sim and Delta A/sigma = (A_sim - A_fit)/sigma_red on the 52 bins.
Needs PyMuPDF: PYTHONPATH=<dir with pymupdf> /home/cbllover/miniconda3/envs/halfdome/bin/python extract_lee22_fig2_fig5_full.py
"""
from pathlib import Path

import numpy as np
import pymupdf

HERE = Path(__file__).resolve().parent
FMG = HERE.parents[1]
REF = FMG / "outputs/lee22_normalization_forensics_20260928/C_pressure_consistency/reference"
OUT = FMG / "outputs/lee22_corrected_20260930/doc"
RED, BLUE = (0.8392, 0.1529, 0.1569), (0.0, 0.0, 1.0)
FITS = {"with_c_bpl": (0.1725, 0.6275, 0.1725), "with_c_no_bpl": (0.498, 0.498, 0.498),
        "without_c_bpl": (1.0, 0.498, 0.0549), "without_c_no_bpl": (0.549, 0.3373, 0.2941),
        "battaglia": (0.0902, 0.7451, 0.8118)}
PANEL_OFFSET = 892.8
CAL = {  # log-x and log-y top panel (as in extract_lee22_fig2_fig5.py); residual rows: (pixel, value) of two drawn ticks
    "fig_2.pdf": dict(x01=388.093, x1=889.621, y_ref=44.764, y_ref_val=-1.0, y_dex=124.784,
                      mid=((451.399, 0.5), (552.353, -0.5)), bot=((606.018, 10.0), (680.406, -10.0)), key="P"),
    "fig_4.pdf": dict(x01=404.0775146484375, x1=905.6053466796875, y_ref=60.74560546875, y_ref_val=0.0,
                      y_dex=(397.4245300292969 - 60.74560546875) / 3,
                      mid=((457.709, 0.25), (546.044, -0.25)), bot=((601.643, 10.0), (684.781, -10.0)), key="n"),
}
TOP_MAX, MID_MAX = 432.0, 560.0   # pixel y separating the top, middle and bottom rows


def same(c, rgb):
    return c is not None and max(abs(a - b) for a, b in zip(c, rgb)) < 2e-3


def path_points(d):
    items = d["items"]
    assert all(it[0] == "l" for it in items)
    return [items[0][1]] + [it[2] for it in items]


def main():
    out = {}
    for name, cal in CAL.items():
        with pymupdf.open(REF / name) as doc:
            drawings = doc[0].get_drawings()
        X = lambda px: 10 ** ((px - cal["x1"]) / (cal["x1"] - cal["x01"]))
        Y = lambda py: 10 ** ((cal["y_ref"] - py) / cal["y_dex"] + cal["y_ref_val"])

        def lin(row):
            (p0, v0), (p1, v1) = cal[row]
            return lambda py: v0 + (py - p0) * (v1 - v0) / (p1 - p0)
        q = cal["key"]
        for col in (0, 1):
            right = lambda d: int(d["rect"].x0 > 1032) == col
            # TNG measurement: errorbar data line (52 points, width 1.5) and the two bar sets
            line = [d for d in drawings if same(d["color"], RED) and len(d["items"]) == 51 and right(d)]
            assert len(line) == 1, (name, col, len(line))
            pts = path_points(line[0])
            xs = X(np.array([p.x for p in pts]) - col * PANEL_OFFSET); vs = Y(np.array([p.y for p in pts]))
            bars = {}
            for cname, rgb in (("sigma", RED), ("scatter", BLUE)):
                segs = [d for d in drawings if same(d["color"], rgb) and len(d["items"]) == 1 and right(d)
                        and abs(d["items"][0][1].x - d["items"][0][2].x) < 1e-6 and (d.get("width") or 0) == 1.5]
                segs.sort(key=lambda d: d["items"][0][1].x)
                assert len(segs) == 52, (name, col, cname, len(segs))
                bx = X(np.array([d["items"][0][1].x for d in segs]) - col * PANEL_OFFSET)
                assert np.allclose(bx, xs, rtol=1e-6)
                lo = Y(np.array([max(d["items"][0][1].y, d["items"][0][2].y) for d in segs]))
                hi = Y(np.array([min(d["items"][0][1].y, d["items"][0][2].y) for d in segs]))
                # the bottom panel edge clips bars that reach below the axis: keep the (unclipped) upper half
                bars[cname] = (hi - lo) / 2; bars[cname + "_upper"] = hi - vs
                if cname == "sigma":
                    assert np.allclose((hi + lo) / 2, vs, rtol=2e-3), np.max(np.abs((hi + lo) / 2 / vs - 1))
            key = "{}_{}".format(q, col)
            out[key + "_x"] = xs; out[key + "_sim"] = vs
            out[key + "_sigma"] = bars["sigma_upper"]; out[key + "_scatter"] = bars["scatter_upper"]
            # residual rows of the four fits
            for row in ("mid", "bot"):
                f = lin(row)
                for cname, rgb in FITS.items():
                    if cname == "battaglia":
                        continue
                    cands = [d for d in drawings if same(d["color"], rgb) and len(d["items"]) > 40 and right(d)
                             and d["rect"].y0 > TOP_MAX and ((d["rect"].y1 < MID_MAX) if row == "mid" else (d["rect"].y0 > MID_MAX - 20))]
                    if len(cands) != 1:   # a curve that crosses the row boundary is still a single path; pick by mean y
                        cands = [d for d in drawings if same(d["color"], rgb) and len(d["items"]) > 40 and right(d) and d["rect"].y0 > TOP_MAX - 5]
                        cands = [d for d in cands if (np.mean([p.y for p in path_points(d)]) < MID_MAX) == (row == "mid")]
                    assert len(cands) == 1, (name, col, row, cname, len(cands))
                    pp = path_points(cands[0])
                    out["{}_{}_{}_x".format(key, row, cname)] = X(np.array([p.x for p in pp]) - col * PANEL_OFFSET)
                    out["{}_{}_{}".format(key, row, cname)] = f(np.array([p.y for p in pp]))
            # check the residual definition with the top-panel no-c BPL fit (orange) at the 52 bins
            top = [d for d in drawings if same(d["color"], FITS["without_c_bpl"]) and len(d["items"]) == 102 and right(d)
                   and d["rect"].y1 < TOP_MAX]
            assert len(top) == 1
            tp = path_points(top[0])
            tx = X(np.array([p.x for p in tp]) - col * PANEL_OFFSET); tv = Y(np.array([p.y for p in tp]))
            fit = np.exp(np.interp(np.log(xs), np.log(tx), np.log(tv)))
            mx = out["{}_mid_without_c_bpl_x".format(key)]; mv = out["{}_mid_without_c_bpl".format(key)]
            bv = out["{}_bot_without_c_bpl".format(key)]
            ok = (xs >= mx.min()) & (xs <= mx.max())
            d_mid = np.interp(xs[ok], mx, mv); d_bot = np.interp(xs[ok], out["{}_bot_without_c_bpl_x".format(key)], bv)
            cands = {"(sim-fit)/sim": (vs - fit) / vs, "(sim-fit)/fit": (vs - fit) / fit, "(fit-sim)/sim": (fit - vs) / vs}
            for lab, val in cands.items():
                print("{} panel {} Delta/A  {:14s} rms diff {:.4f}".format(name, col, lab, np.sqrt(np.mean((val[ok] - d_mid) ** 2))))
            for lab, sig in (("sigma (red)", out[key + "_sigma"]), ("scatter (blue)", out[key + "_scatter"])):
                print("{} panel {} Delta/sigma {:14s} rms diff {:.3f}".format(name, col, lab, np.sqrt(np.mean(((vs - fit)[ok] / sig[ok] - d_bot) ** 2))))
            print("   sigma/sim: median {:.3f}, scatter/sim median {:.3f}, ratio median {:.1f} (N_halo ~ ratio^2)".format(
                np.median(out[key + "_sigma"] / vs), np.median(out[key + "_scatter"] / vs), np.median(out[key + "_scatter"] / out[key + "_sigma"])))
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "lee22_fig2_fig5_full_digitization.npz", **out)
    print("wrote", OUT / "lee22_fig2_fig5_full_digitization.npz", len(out), "arrays")


if __name__ == "__main__":
    main()
