#!/usr/bin/env python3
"""Check the pixel-space estimator against previous harmonic-space predictions.

This uses complete old DM maps ONLY as an independent numerical check. The
new individual source DMs are never replaced by these averaged maps.
"""
import argparse
import json
from pathlib import Path
import healpy as hp
import numpy as np
from compare_halfdome_takahashi import MODELS, gaussian_beam, annular_correlation, write_rows, sha256
from compare_takahashi_sightlines import annulus_window


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    previous, output = Path(args.previous), Path(args.output)
    tsz = previous/"inputs/battaglia12_full_lightcone_repaint.fits"
    with np.load(str(previous/"spectra/battaglia16.npz")) as f:
        expected = json.loads(str(f["metadata_json"].item()))["tsz_map_sha256"]
    if sha256(tsz) != expected:
        raise ValueError("tSZ checksum mismatch")
    y = hp.read_map(str(tsz), dtype=np.float64, verbose=False)
    y -= y.mean()
    alm = hp.map2alm(y, lmax=8192, iter=3, pol=False)
    del y
    rows = []
    # Small ACT scale, first valid Planck scale, broad one-degree bin.
    for survey, beam, lo in (("act", 1.6, 10**.25), ("planck", 10., 10.), ("planck", 10., 10**1.75)):
        hi = lo*10**.25
        window = annulus_window(8192, lo, hi)*gaussian_beam(np.arange(8193), beam)
        window[0] = 0
        field = hp.alm2map(hp.almxfl(alm, window), 4096, pol=False, verbose=False)
        for model, _, _, _ in MODELS:
            dm = hp.read_map(str(previous/"maps"/(model+"_"+survey+".fits")), dtype=np.float64, verbose=False)
            dm -= dm.mean()
            pixel_mean = np.dot(dm, field)/len(dm)
            del dm
            with np.load(str(previous/"spectra"/(model+".npz"))) as ref:
                harmonic_mean = annular_correlation(ref["cl_y_dm_"+survey], [lo], [hi], beam)[0]
            fractional = pixel_mean/harmonic_mean-1
            if abs(fractional) > .01:
                raise ValueError("Pixel/harmonic mean differ by >1%: " + str(fractional))
            row = dict(survey=survey, model=model, theta_lower_arcmin=lo,
                       theta_upper_arcmin=hi, pixel_mean=pixel_mean, harmonic_mean=harmonic_mean,
                       fractional_difference=fractional)
            rows.append(row)
            print(row, flush=True)
        del field
    write_rows(output/"analysis/pixel_vs_harmonic_mean_check.csv", rows)


if __name__ == "__main__":
    main()
