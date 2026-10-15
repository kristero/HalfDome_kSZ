#!/usr/bin/env python3
"""Compute fresh B16 CLASS-SZ predictions and spectra of the finite-sphere maps."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import tomllib
from pathlib import Path

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "frb_map_generation/outputs/lee22_pdf_ksz_comparison_20260922/ksz"
sys.path.insert(0, str(ROOT / "tau_ksz_comparison"))
from reproduce_tau_ksz import theory, save_npz, write_json


def class_predictions(cfg, bounds):
    # Reuse the established full-CLASS calculation, changing the physical
    # selection, cosmology and *three-dimensional* electron-profile cutoff.
    baseline = tomllib.loads((ROOT / "tau_ksz_comparison/settings.toml").read_text())
    cfg["class_sz"] = baseline["class_sz"]
    cfg["class_sz"]["x_out_truncated_density_profile (electrons)"] = cfg["sphere_r200c"]
    for selection in ("restricted", "full"):
        selected = dict(bounds)
        if selection == "restricted":
            selected.update(mass_min_msun=1e13, mass_max_msun=1e14, z_max=1.)
        folder = OUT / ("class_sz_" + selection)
        folder.mkdir(exist_ok=True)
        write_json(folder / "selection.json", selected)
        theory(cfg, selected, folder)


def map_predictions(cfg):
    import h5py
    import healpy as hp
    import numpy as np

    results = {"ell": np.arange(cfg["lmax"] + 1)}
    runtime = {}
    with h5py.File(OUT / "ksz_maps.h5", "r") as maps:
        assert maps.attrs["complete"] and maps.attrs["sphere_r200c"] == 3
        scale = float(maps.attrs["lee22_paper_over_previous"])
        results["lee22_paper_over_previous"] = scale
        for name in sorted(maps):
            checkpoint = OUT / ("spectrum_" + name + ".npz")
            if checkpoint.exists():
                with np.load(checkpoint) as saved:
                    data = dict(saved)
                assert int(data["lmax"]) == cfg["lmax"]
            else:
                started = time.monotonic()
                sky = maps[name][:]
                assert sky.size == hp.nside2npix(cfg["nside"])
                assert np.isfinite(sky).all()
                mean = float(sky.mean())
                sky -= mean
                variance = float(np.dot(sky, sky) / sky.size)
                print(f"Transforming {name}, lmax={cfg['lmax']}", flush=True)
                cl = hp.anafast(sky, lmax=cfg["lmax"], iter=cfg["anafast_iterations"], pol=False)
                del sky
                assert np.isfinite(cl).all() and (cl >= 0).all()
                data = dict(cl=cl, map_mean=mean, map_variance=variance,
                            seconds=time.monotonic()-started, lmax=cfg["lmax"])
                save_npz(checkpoint, **data)
            results["cl_" + name] = data["cl"]
            runtime[name] = {key: float(data[key]) for key in ("map_mean", "map_variance", "seconds")}
            if "lee22_previous" in name:
                results["cl_" + name.replace("previous", "paper")] = data["cl"] * scale**2
            print(f"Completed {name}: {float(data['seconds']):.1f} s", flush=True)
    save_npz(OUT / "map_spectra.npz", **results)
    write_json(OUT / "map_spectra_runtime.json", dict(
        maps=runtime, healpy_version=hp.__version__, lmax=cfg["lmax"],
        nside=cfg["nside"], iterations=cfg["anafast_iterations"],
        monopole_removed=True, pixel_window_deconvolved=False,
        units="C_ell of dimensionless deltaT/T", finite_sphere_r200c=3,
        paper_normalization="Exact map amplitude rescaling; C_ell multiplied by amplitude ratio squared."))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("theory", "spectra"))
    args = parser.parse_args()
    cfg = tomllib.loads((OUT / "settings.toml").read_text())
    bounds = tomllib.loads((OUT / "bounds.toml").read_text())
    if args.stage == "theory":
        class_predictions(cfg, bounds)
    else:
        map_predictions(cfg)


if __name__ == "__main__":
    main()
