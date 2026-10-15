#!/usr/bin/env python3
"""Additional halo-kSZ panel in microkelvin squared, from completed spectra.

The reference image's red Websky Halo curve identifies the observable and plot
style. The red curve here is the measured HalfDome/XGPaint halo spectrum; no
Websky points are digitized and no amplitude is fitted to the reference image.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, ScalarFormatter
import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_RUN = HERE / "outputs" / "nside4096"
ELL_LIMITS = (70.0, 5500.0)
POWER_LIMITS_UK2 = (0.1, 2.6)
CLASS_DEFAULT_TCMB_K = 2.7255  # CLASS input.c default used by the saved run.


def make_figure(run_dir: Path):
    map_path = run_dir / "map_spectra.npz"
    theory_path = run_dir / "class_sz_spectra.npz"
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    parameters = json.loads((run_dir / "class_sz_parameters.json").read_text())
    temperature_k = float(parameters.get("T_cmb", CLASS_DEFAULT_TCMB_K))
    if not np.isfinite(temperature_k) or temperature_k <= 0:
        raise ValueError("CMB temperature must be positive and finite")
    if manifest["max_halos"] != 0:
        raise ValueError("This figure requires the completed all-halo run, not smoke data")

    with np.load(map_path, allow_pickle=False) as maps:
        ell = np.array(maps["ell"], dtype=float)
        cl_ksz = np.array(maps["cl_ksz"], dtype=float)
    with np.load(theory_path, allow_pickle=False) as theory:
        theory_ell = np.array(theory["ell"], dtype=float)
        dl_1h = np.array(theory["dl_ksz_1h"], dtype=float)
        dl_2h = np.array(theory["dl_ksz_2h"], dtype=float)

    for x, curves in ((ell, [cl_ksz]), (theory_ell, [dl_1h, dl_2h])):
        if (x.ndim != 1 or not np.isfinite(x).all() or np.any(np.diff(x) <= 0)
                or x[0] > ELL_LIMITS[0] or x[-1] < ELL_LIMITS[1]):
            raise ValueError("Saved spectra must cover the requested multipole interval")
        for curve in curves:
            if curve.shape != x.shape or not np.isfinite(curve).all() or (curve < 0).any():
                raise ValueError("Invalid saved kSZ spectrum")

    # Map C_l is for dimensionless DeltaT/T. CLASS-SZ arrays are already D_l.
    # Multiplying each by (T_CMB in microkelvin)^2 converts temperature power.
    temperature_squared_uk2 = (temperature_k * 1e6)**2
    dl_map_uk2 = ell * (ell + 1) * cl_ksz / (2*np.pi) * temperature_squared_uk2
    dl_theory_uk2 = (dl_1h + dl_2h) * temperature_squared_uk2

    with plt.rc_context({"font.family": "serif", "mathtext.fontset": "cm",
                         "font.size": 12, "axes.linewidth": 0.8}):
        fig, ax = plt.subplots(figsize=(7.4, 3.68))
        ax.semilogx(ell[ell > 0], dl_map_uk2[ell > 0], color="orangered",
                    linewidth=1.2, label="XGPaint")
        ax.semilogx(theory_ell, dl_theory_uk2, color="black", linestyle="--",
                    linewidth=1.7, label="CLASS-SZ")
        ax.set_xlim(*ELL_LIMITS)
        ax.set_ylim(*POWER_LIMITS_UK2)
        ax.set_xlabel(r"$\ell$")
        ax.set_ylabel(r"$\ell(\ell+1)C_\ell/(2\pi)\;[\mu\mathrm{K}^2]$")
        ax.xaxis.set_major_locator(FixedLocator([100, 300, 1000, 3000]))
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.set_yticks([0.5, 1.0, 1.5, 2.0, 2.5])
        ax.tick_params(which="both", direction="in")
        ax.legend(loc="upper center", ncol=2, frameon=False)
        fig.tight_layout(pad=0.8)
        for suffix in ("png", "pdf"):
            path = run_dir / f"ksz_halo_reference_comparison.{suffix}"
            fig.savefig(path, dpi=200)
            print(path)
        plt.close(fig)

    np.savez(run_dir / "ksz_halo_reference_comparison_arrays.npz",
             ell_map=ell, dl_xgpaint_uk2=dl_map_uk2,
             ell_class_sz=theory_ell, dl_class_sz_uk2=dl_theory_uk2,
             dl_class_sz_1h_uk2=dl_1h*temperature_squared_uk2,
             dl_class_sz_2h_uk2=dl_2h*temperature_squared_uk2)
    metadata = {
        "observable": "halo kSZ temperature angular power D_ell",
        "temperature_k": temperature_k,
        "temperature_source": "saved CLASS-SZ parameter or CLASS default 2.7255 K",
        "units": "microkelvin squared",
        "ell_limits": ELL_LIMITS,
        "y_limits_uk2": POWER_LIMITS_UK2,
        "theory": "CLASS-SZ B16 1h + 2h",
        "reference_usage": "observable, axis limits, and red halo-curve style",
        "map_source": "existing HalfDome/XGPaint spectrum; no amplitude adjustment",
        "input_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in [map_path, theory_path, manifest_path]},
    }
    (run_dir / "ksz_halo_reference_comparison_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN,
                        help="Directory containing the completed spectra")
    args = parser.parse_args()
    make_figure(args.run_dir.resolve())


if __name__ == "__main__":
    main()
