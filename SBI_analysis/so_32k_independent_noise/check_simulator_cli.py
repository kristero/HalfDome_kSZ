#!/usr/bin/env python3
"""Validate real Julia worker commands without generating full-resolution maps."""
import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import generate as g


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--julia", default="julia")
    args = parser.parse_args()
    base = json.loads((g.BUNDLE / "config.json").read_text())
    designs = g.load_designs(base)
    report = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for mode in g.MODES:
            for row, beam, deprojections in ((1, 2.0, [0]), (base["n_rows"], 3.0, [2, 0])):
                config = dict(base, noise_cases=["baseline", "goal"], deprojections=deprojections,
                              beam_fwhm_arcmin=beam)
                g.validate_config(config)
                cmd = g.command(config, mode, row, designs[mode][0], tmp/"not_read.hdf5",
                                tmp/"raw", tmp/"cache", args.julia, 2) + ["--validate-only"]
                result = subprocess.run(cmd, check=True, capture_output=True, text=True,
                                        timeout=300, env=g.simulator_environment(2))
                expected = {"Validated beam": beam, "Validated deprojections": ",".join(map(str, deprojections)),
                            "Validated mask seed": config["mask_seed"],
                            "Validated noise seed": g.seeds(config, mode, row)[0]}
                for key, value in expected.items():
                    if f"{key}: {value}\n" not in result.stdout:
                        raise ValueError(f"Runtime config mismatch: {key}\n{result.stdout}")
                report.append(dict(mode=mode, row=row, checked=expected, passed=True))
                print(mode, row, "runtime CLI passed", flush=True)
    g.write_json(g.BUNDLE / "validation_plots/runtime_cli_report.json", report)


if __name__ == "__main__":
    main()
