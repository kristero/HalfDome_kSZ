#!/usr/bin/env python3
"""Trace black BP best-fit curves, NOT data points, from Medlock & Nagai Fig. 5.

Uses the original arXiv:2608.06455v1 raster asset. The plot has logarithmic x,
linear y. One-and-a-half image pixels is a conservative readout uncertainty,
not a statistical model confidence band. Near-zero unresolved tails are flagged.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from compare_halfdome_takahashi import sha256, write_rows
from digitize_frb_observational_figures import calibration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", default="frb_map_generation/outputs/observational_comparison_inputs_20260913")
    parser.add_argument("--output", default="frb_map_generation/outputs/takahashi_100k_20260914/references")
    args = parser.parse_args()
    source = Path(args.inputs)/"source_figures/BP_Model_Fit.png"
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rgb = np.asarray(Image.open(source).convert("RGB"))
    if rgb.shape != (676, 1704, 3):
        raise ValueError("Axis calibration requires the original 1704 x 676 asset")
    panels = (
        ("act", (510, 826, 225, 392), calibration([594,718,843], [1,2,3],
             [100,171,241,312,382], [8,6,4,2,0])),
        ("planck_milca", (922,1244,340,418), calibration([909,1084,1260], [1,2,3],
             [77,133,189,244,300,355,411], [6,5,4,3,2,1,0])),
        ("planck_nilc", (1335,1651,250,391), calibration([1320,1496,1672], [1,2,3],
             [118,184,250,316,382], [4,3,2,1,0])),
    )
    rows, calibrations = [], {}
    fig, ax = plt.subplots(figsize=(17, 6.76))
    ax.imshow(rgb)
    for name, (x0,x1,y0,y1), cal in panels:
        calibrations[name] = cal
        xs, ys = [], []
        for x in range(x0, x1+1, 3):
            column = rgb[y0:y1+1, x].astype(float)
            dark = (column.max(1) < 110) & (np.ptp(column, axis=1) < 12)
            selected = np.flatnonzero(dark)+y0
            if len(selected) == 0:
                continue  # No invented curve through obscured points.
            if np.ptp(selected) > 12 or np.any(np.diff(selected) > 1):
                raise ValueError("Ambiguous black features at {} x={}".format(name, x))
            y = float(np.median(selected))
            w = float(np.polyval(cal["y"], y))
            error = 1.5*abs(cal["y"][0])
            xs.append(x)
            ys.append(y)
            rows.append(dict(survey=name, theta_arcmin=10**np.polyval(cal["x"], x),
                w_yDM_pc_cm3=w, digitization_absolute_uncertainty=error,
                tail_unresolved=abs(w) < 3*error, pixel_x=x, pixel_y=y,
                paper="arXiv:2608.06455v1", figure=5,
                curve="BP maximum-likelihood curve for this panel; not a measured datum",
                source_redshift=2.0, beam="none (as published)",
                extraction="neutral dark pixels; calibrated log-x and linear-y; no extrapolation"))
        if len(xs) < 80:
            raise ValueError("Too few curve pixels for " + name)
        ax.plot(xs, ys, color="#e83e8c", lw=0, marker=".", ms=2)
    ax.axis("off")
    fig.tight_layout(pad=0)
    fig.savefig(str(output/"medlock_fig5_best_fit_trace_QA.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    write_rows(output/"medlock_fig5_best_fit_digitized.csv", rows)
    (output/"digitization_provenance.json").write_text(json.dumps(dict(
        source=str(source), sha256=sha256(source), axis_calibrations=calibrations,
        reference="https://arxiv.org/abs/2608.06455",
        caveat="Published BP curves use all sources at z=2 and no beam. They are not matched to the observed-redshift HalfDome kernel.",
        uncertainty="1.5 pixels per coordinate, not posterior uncertainty; unresolved low-amplitude tails flagged",
        count=len(rows)), indent=2))
    print("Digitized {} BP curve points into {}".format(len(rows), output))


if __name__ == "__main__":
    main()
