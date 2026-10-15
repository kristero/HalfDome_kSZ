#!/usr/bin/env python3
"""Run the unchanged validated painter locally, with strictly supported mass cuts."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

from preflight import GENERATOR, OUTPUT, ROOT, WINDOWS, input_audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Without this flag, print the command only.")
    parser.add_argument("--threads", type=int, default=20)
    args = parser.parse_args()
    if not 1 <= args.threads <= 26:
        parser.error("Use 1-26 threads for this memory-audited configuration.")
    audit = input_audit()
    destination = OUTPUT / "halfdome"
    julia = "/home/kn18001/.juliaup/bin/julia"
    command = [
        julia, f"--threads={args.threads}", str(GENERATOR),
        f"--catalog={ROOT / 'lightcone_100.hdf5'}", "--source-redshift=1.0",
        "--nfrb=120000", "--seed=42", "--nside=4096", "--unique-pixels=true",
        "--chunk-size=250000", "--max-catalog-halos=0", "--progress-every-batches=20",
        "--mass-windows=" + ",".join(f"{k}:{lo:.17g}:{hi:.17g}" for k, lo, hi in WINDOWS),
        "--dm-profile=lee2022_xgpaint", "--lee2022-concentration-mode=none",
        "--lee2022-normalization=xgpaint_ne2d", "--lee2022-n0-pivot=mcut",
        "--lee2022-shape-mass-clip=fit", "--lee2022-redshift-scaling=physical",
        "--halo-boundary=spherical", "--halo-extension-r200-multiplier=1.0",
        "--xgpaint-profile-mass-definition=m200c",
        f"--dm-cache={audit['cache_path']}", "--dm-cache-overwrite=false",
        "--pdf-edge-count=300", "--pdf-spacing=log", "--pdf-dm-min=0.1", "--pdf-dm-max=30000",
        "--save-ray-dm=true", "--overwrite=false",
        f"--output={destination / 'halo_dm.h5'}",
        f"--summary={destination / 'summary.csv'}",
        f"--provenance={destination / 'provenance.txt'}",
    ]
    print("\n".join(command), flush=True)
    if not args.run:
        return
    destination.mkdir(parents=True, exist_ok=True)
    if any((destination / name).exists() for name in ("halo_dm.h5", "summary.csv", "provenance.txt", "run.log")):
        raise FileExistsError("Preserving previous run; choose a new output directory before rerunning.")
    environment = os.environ.copy()
    environment.update(JULIA_DEPOT_PATH="/home/kn18001/.julia", HDF5_USE_FILE_LOCKING="FALSE",
                       JULIA_NUM_THREADS=str(args.threads), OPENBLAS_NUM_THREADS="1",
                       OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    # A local 12 GiB address-space ceiling makes an unexpectedly large run fail
    # before it can approach the user's 40 GB threshold. No PBS job is submitted.
    import resource

    def set_memory_ceiling():
        resource.setrlimit(resource.RLIMIT_AS, (12 * 2**30, 12 * 2**30))

    timed = ["/usr/bin/time", "-v", "-o", str(destination / "resources.txt"), *command]
    with (destination / "run.log").open("x") as log:
        subprocess.run(timed, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT,
                       preexec_fn=set_memory_ceiling, check=True)
    print(f"Completed local HalfDome run: {destination}", flush=True)


if __name__ == "__main__":
    main()
