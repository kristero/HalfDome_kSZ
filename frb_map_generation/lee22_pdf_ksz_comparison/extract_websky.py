#!/usr/bin/env python3
"""Recover the orange halo-only samples from Stein+2020's vector kSZ figure.

This is a figure extraction, not a replacement Websky map calculation. The
reference keeps the paper's halo selection, compensation, and redshift range.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pymupdf

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "frb_map_generation/outputs/lee22_pdf_ksz_comparison_20260922/reference"


def main():
    source = OUT / "websky_source/ksz.pdf"
    raw = source.read_bytes()
    document = pymupdf.open(stream=raw, filetype="pdf")
    page = document[0]
    orange = np.array([1., 0.2705882489681244, 0.])
    xy = []
    for drawing in page.get_drawings():
        color = drawing.get("fill")
        rect = drawing["rect"]
        if color is not None and np.allclose(color, orange, atol=1e-7, rtol=0):
            # Each scientific point is a circular marker, not the legend line.
            if len(drawing["items"]) == 8 and abs(rect.width-2) < .01 and abs(rect.height-2) < .01:
                xy.append(((rect.x0+rect.x1)/2, (rect.y0+rect.y1)/2))
    xy = np.asarray(xy)
    assert len(xy) > 5000, len(xy)
    # Tick positions measured from the vector axes. x is logarithmic, y linear.
    x100, x1000 = 109.64906311035156, 462.9506530761719
    y05, y10 = 281.1744079589844, 215.94239807128906
    ell = 10**(2 + (xy[:, 0]-x100)/(x1000-x100))
    dl = .5 + .5*(y05-xy[:, 1])/(y05-y10)
    # Samples are integer multipoles. Rounding removes PDF-coordinate precision.
    rounding_error = np.abs(ell-np.rint(ell))
    # The PDF stores finite-precision coordinates: the largest displacement is
    # about 0.045 in ell near 5750, far below half an integer-multipole spacing.
    assert rounding_error.max() < .1, rounding_error.max()
    ell = np.rint(ell).astype(int)
    keep = (ell >= 70) & (ell <= 5500) & (dl > 0)
    order = np.argsort(ell[keep])
    ell, dl = ell[keep][order], dl[keep][order]
    assert len(np.unique(ell)) == len(ell)
    np.savetxt(OUT / "websky_halo_digitized.csv", np.column_stack((ell, dl)),
               delimiter=",", header="ell,D_ell_uK2", comments="", fmt=["%d", "%.12g"])
    report = dict(
        source="https://arxiv.org/src/2001.08787v1", paper="https://arxiv.org/abs/2001.08787",
        figure="ksz.pdf; orange WebSky Halo markers", source_sha256=hashlib.sha256(raw).hexdigest(),
        method="Vector marker centres; log x and linear y calibrated to vector axis ticks.",
        input_markers=len(xy), retained_samples=len(ell), ell_range=[int(ell.min()), int(ell.max())],
        maximum_integer_ell_rounding_error=float(rounding_error.max()), units="D_ell in microkelvin squared",
        reference_selection="M200c > 1e13 Msun AND theta200c > 0.5 arcmin; z < 4.6.",
        reference_gas="Battaglia16 AGN with uniform-density halo counterterm (overdensity 3).",
        comparison_limit="Broader, compensated reference; not a matched 3R200c prediction.")
    (OUT / "websky_extraction.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
