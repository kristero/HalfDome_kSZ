"""Check collision-free, reproducible seeds through the supported 524k size."""
import argparse
from pathlib import Path

import numpy as np

from noise_seeds import ALGORITHM, PREFLIGHT_OFFSET, split_seeds
from prior import JointPrior, digest, write_json
from design import old_pool


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    import json
    run = json.loads((root/"run_config.json").read_text())
    prior = JointPrior(json.loads((root/"manifest.json").read_text())["prior"])
    _, selected, _, _ = old_pool(root,prior)
    case_count = len(json.loads((root/"preflight/cases.json").read_text()))
    ids = np.concatenate([selected, run["fresh_noise_row_offset"]+np.arange(524288),
                          PREFLIGHT_OFFSET+np.arange(case_count)])
    assert len(np.unique(ids)) == len(ids)
    all_seeds = np.array([split_seeds(int(i),run["noise_master_seed"]) for i in ids],dtype=np.int64)
    assert np.all(all_seeds > 0)
    assert len(np.unique(all_seeds)) == all_seeds.size
    assert not np.isin(all_seeds,[22446,22447]).any()
    design = np.load(root/"design/noise_split_seeds.npy")
    design_ids = np.load(root/"design/noise_seed_row_ids.npy")
    expected = np.array([split_seeds(int(i),run["noise_master_seed"]) for i in design_ids])
    np.testing.assert_array_equal(design, expected)
    for i in (0, 1, 8191, 524287):
        np.testing.assert_array_equal(all_seeds[i], split_seeds(int(ids[i]),run["noise_master_seed"]))
    result = dict(passed=True, algorithm=ALGORITHM, maximum_production_rows=PREFLIGHT_OFFSET,
                  preflight_rows=case_count, unique_split_seeds=int(all_seeds.size), reused_design_points=len(selected),
                  fresh_noise_row_offset=run["fresh_noise_row_offset"],
                  prefix_stable=True, observation_seed_overlap=False,
                  first_three_rows=design[:3].tolist(),
                  seed_code_sha256=digest(root/"code/noise_seeds.py"),
                  design_sha256=digest(root/"design/noise_split_seeds.npy"))
    write_json(root/"noise_seed_check.json",result)
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
