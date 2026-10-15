#!/usr/bin/env python3
"""Execute the local TNG painter with recorded runtime and bounded address space."""

import argparse
import os
import resource
import shutil
import subprocess
from pathlib import Path

from preflight import HERE, OUTPUT, ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--halfdome", action="store_true")
    parser.add_argument("--ray-configuration", type=Path,
                        help="HDF5 with reference rays and requested physical mass windows.")
    args = parser.parse_args()
    directory = OUTPUT / ("halfdome_direct" if args.halfdome else "tng")
    directory.mkdir(parents=True, exist_ok=True)
    stage = "validation" if args.validate_only else "paint"
    environment = os.environ.copy()
    environment.update(JULIA_DEPOT_PATH="/home/kn18001/.julia", HDF5_USE_FILE_LOCKING="FALSE",
                       JULIA_NUM_THREADS="20", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
                       TNG_COMPARISON_OUTPUT=str(OUTPUT))
    if args.ray_configuration:
        environment["TNG_REFERENCE_RAYS"] = str(args.ray_configuration.resolve(strict=True))
    command = ["/usr/bin/time", "-v", "-o", str(directory / f"{stage}_resources.txt"),
               "/home/kn18001/.juliaup/bin/julia", "--threads=20", str(HERE / "paint_tng.jl")]
    if args.validate_only:
        command.append("--validate-only=true")
    if args.halfdome:
        command.append("--simulation=halfdome")

    def limit_memory():
        resource.setrlimit(resource.RLIMIT_AS, (12 * 2**30, 12 * 2**30))

    with (directory / f"{stage}.log").open("x") as log:
        shutil.copy2(HERE / "paint_tng.jl", directory / f"{stage}_used.jl")
        subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT,
                       preexec_fn=limit_memory, check=True)
    print(f"Completed {stage}: {directory}", flush=True)


if __name__ == "__main__":
    main()
