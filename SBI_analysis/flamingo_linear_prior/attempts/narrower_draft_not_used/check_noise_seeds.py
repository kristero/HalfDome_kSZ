"""Check collision-free, reproducible seeds through the supported 524k size."""
import argparse
from pathlib import Path

import numpy as np

from noise_seeds import ALGORITHM, PREFLIGHT_OFFSET, split_seeds
from prior import digest, write_json


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    import json
    run = json.loads((root/"run_config.json").read_text())
    preflight_count = len(json.loads((root/"preflight/cases.json").read_text()))
    all_seeds = np.array([split_seeds(i,run["noise_master_seed"])
                          for i in range(PREFLIGHT_OFFSET+preflight_count)],dtype=np.int64)
    assert np.all(all_seeds > 0)
    assert len(np.unique(all_seeds)) == all_seeds.size
    assert not np.isin(all_seeds,[22446,22447]).any()
    design = np.load(root/"design/noise_split_seeds.npy")
    np.testing.assert_array_equal(design, all_seeds[:run["count"]])
    for i in (0, 1, 8191, 524287):
        np.testing.assert_array_equal(all_seeds[i], split_seeds(i,run["noise_master_seed"]))
    result = dict(passed=True, algorithm=ALGORITHM, maximum_production_rows=PREFLIGHT_OFFSET,
                  preflight_rows=preflight_count, unique_split_seeds=int(all_seeds.size),
                  prefix_stable=True, observation_seed_overlap=False,
                  first_three_rows=design[:3].tolist(),
                  seed_code_sha256=digest(root/"code/noise_seeds.py"),
                  design_sha256=digest(root/"design/noise_split_seeds.npy"))
    write_json(root/"noise_seed_check.json",result)
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
