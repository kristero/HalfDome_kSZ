#!/usr/bin/env python3
"""Make new Sobol designs or extract exact blocks from existing full CSVs."""
import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np
from generate import BUNDLE, MODES, NAMES, sha256, validate_config, write_json

def make_designs(config, output, seed=42, sources=None):
    validate_config(config)
    output.mkdir(parents=True, exist_ok=False)
    for mode in MODES:
        offset, n = config["sequence_offset"], config["n_rows"]
        path = output / (mode + ".csv")
        metadata = dict(mode=mode, n_rows=n, sequence_offset=offset, config=config)
        if sources:
            # Preserve the original sequence, not a new Sobol seed/version.
            source = sources[mode]
            with source.open(newline="") as src, path.open("w", newline="") as dst:
                reader, writer = csv.reader(src), csv.writer(dst)
                header = next(reader)
                if tuple(header) != NAMES:
                    raise ValueError(f"Unexpected parameter order in {source}")
                writer.writerow(header)
                count = 0
                for row in itertools.islice(reader, offset, offset + n):
                    writer.writerow(row)
                    count += 1
                if count != n:
                    raise ValueError(f"{source} does not contain the requested block")
            metadata.update(source=str(source.resolve()), source_sha256=sha256(source))
        else:
            import scipy
            from scipy.stats import qmc
            indices = [0, 2] if mode == "two_param" else list(range(9))
            engine = qmc.Sobol(d=len(indices), scramble=True, seed=seed)
            if offset:
                engine.fast_forward(offset)
                unit = engine.random(n)
            else:
                unit = engine.random_base2(int(np.log2(n)))
            low, high = np.array(config["prior_low"]), np.array(config["prior_high"])
            theta = np.tile(config["fiducial"], (n, 1)).astype(float)
            theta[:, indices] = low[indices] + unit * (high[indices] - low[indices])
            with path.open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(NAMES)
                writer.writerows(theta)
            metadata.update(seed=seed, scipy=scipy.__version__, scramble=True)
        metadata["csv_sha256"] = sha256(path)
        write_json(output / (mode + ".json"), metadata)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--source-two-param", type=Path)
    parser.add_argument("--source-nine-param", type=Path)
    args = parser.parse_args()
    if bool(args.source_two_param) != bool(args.source_nine_param):
        parser.error("Supply BOTH original CSV paths or neither")
    sources = dict(zip(MODES, (args.source_two_param, args.source_nine_param))) if args.source_two_param else None
    config = json.loads((BUNDLE / "config.json").read_text())
    make_designs(config, args.output, args.seed, sources)
    print("New designs:", args.output, "; set design_dir in config.json and use a NEW OUTPUT_ROOT.")


if __name__ == "__main__":
    main()
