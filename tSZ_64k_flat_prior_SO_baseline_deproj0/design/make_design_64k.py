"""Write (or --check) the 65,536-row flat-prior Sobol design and its SO noise seeds.

The generator is the one that produced the validated 256-row test design
(design_tests.py, a byte-identical copy): a scrambled nine-dimensional Sobol
sequence with seed 20260920, mapped to independent uniforms in the physical
parameter values, with no rejection. The seed and dimension are fixed, so
rows 0..255 of this design are exactly the 256-row test design.

Every row gets two SO split seeds, noise_seed(row, split, 'train') for split 1
and 2. The 256-row test used the separate 'engineering' stream, so no noise
realization is shared with it.

The saved .npy files are authoritative. --check regenerates the design with the
installed scipy and reports whether it still matches the saved arrays.
"""
import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import scipy

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from design_tests import HIGH, LOW, NAMES, design, noise_seed  # noqa: E402

COUNT = 65536
THETA = HERE / 'theta_design_65536.npy'
SEEDS = HERE / 'noise_seeds_65536.npy'
CSV = HERE / 'battaglia_flat_sobol_65536.csv'
MANIFEST = HERE / 'design_manifest.json'
TEST256 = HERE.parent / 'reference' / 'test256'
# Fixed seeds used elsewhere in the validated pipeline (anchors, legacy noise).
OTHER_SEEDS = {39001001, 39001002, 12345, 22446, 22447}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build():
    theta = design(COUNT)
    seeds = np.array([[noise_seed(row, split, 'train') for split in (1, 2)]
                      for row in range(COUNT)], dtype=np.int64)
    return theta, seeds


def verify(theta, seeds):
    """Return the recorded checks; raise AssertionError on any failure."""
    assert theta.shape == (COUNT, 9) and theta.dtype == np.float64
    assert np.all(np.isfinite(theta)) and np.all((theta >= LOW) & (theta <= HIGH))
    unit = (theta - LOW) / (HIGH - LOW)
    strata = {}
    for bins in (64, 4096):
        counts = np.stack([np.histogram(unit[:, j], bins=bins, range=(0, 1))[0] for j in range(9)])
        assert np.all(counts == COUNT // bins), f'1-D stratification failed for {bins} bins'
        strata[str(bins)] = int(COUNT // bins)
    # The prefix property is what lets rows 0..N-1 (N a power of two) serve as a
    # balanced smaller dataset, and ties rows 0..255 to the 256-row test.
    for size in (256, 8192):
        assert np.array_equal(design(size), theta[:size])
    test_theta = np.load(TEST256 / 'theta_design.npy')
    assert np.array_equal(test_theta, theta[:256]), 'rows 0..255 differ from the 256-row test design'

    assert seeds.shape == (COUNT, 2) and seeds.dtype == np.int64 and np.all(seeds > 0)
    flat = seeds.ravel().tolist()
    assert len(set(flat)) == 2 * COUNT, 'repeated noise seed'
    engineering = {noise_seed(row, split, 'engineering') for row in range(COUNT) for split in (1, 2)}
    test_seeds = set(np.load(TEST256 / 'noise_seeds.npy').ravel().tolist())
    assert test_seeds <= engineering
    assert not set(flat) & engineering, 'overlap with the engineering (256-row test) stream'
    assert not set(flat) & OTHER_SEEDS
    return dict(finite_and_inside_bounds=True, one_dimensional_rows_per_bin=strata,
                prefix_rows_equal_design_256_and_8192=True,
                rows_0_255_equal_test256_theta_design=True,
                unique_split_seeds=2 * COUNT,
                overlap_with_engineering_stream_rows_0_65535=0,
                overlap_with_fixed_pipeline_seeds=0)


def write_csv(theta):
    lines = [','.join(NAMES)]
    lines += [','.join(repr(value) for value in row) for row in theta.tolist()]
    CSV.write_text('\n'.join(lines) + '\n')
    reread = np.loadtxt(CSV, delimiter=',', skiprows=1)
    assert np.array_equal(reread, theta), 'CSV does not round-trip exactly'


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--check', action='store_true', help='verify the saved files; write nothing')
    args = parser.parse_args()
    theta, seeds = build()
    if args.check:
        saved_theta, saved_seeds = np.load(THETA), np.load(SEEDS)
        verify(saved_theta, saved_seeds)
        manifest = json.loads(MANIFEST.read_text())
        for name, digest in manifest['sha256'].items():
            assert sha256(HERE / name) == digest, f'{name} changed after it was written'
        assert np.array_equal(np.loadtxt(CSV, delimiter=',', skiprows=1), saved_theta)
        same = np.array_equal(theta, saved_theta) and np.array_equal(seeds, saved_seeds)
        print('Saved design and seeds verified (sha256, bounds, strata, prefix, seed uniqueness).')
        print(f'Regeneration with scipy {scipy.__version__}: '
              + ('identical to the saved design.' if same else
                 'DIFFERS from the saved design; use the saved .npy files, they are authoritative.'))
        return
    checks = verify(theta, seeds)
    np.save(THETA, theta)
    np.save(SEEDS, seeds)
    write_csv(theta)
    manifest = dict(
        count=COUNT, parameter_order=NAMES, lower=LOW.tolist(), upper=HIGH.tolist(),
        density='independent uniform in the physical parameter values; no rejection, no weighting',
        sampler=('scipy.stats.qmc.Sobol(d=9, scramble=True, seed=20260920).random_base2(16); '
                 'theta = lower + (upper - lower) * u'),
        prefix='rows 0..N-1 equal design(N) for every power of two N <= 65536; rows 0..255 are the 256-row test design',
        noise=dict(survey='Simons Observatory LAT baseline, ILC tSZ, Deproj-0 (noise table column 2)',
                   seeds_per_row=2, stream='train', master_seed=20260920,
                   rule="int.from_bytes(sha256(f'sphere4-flat-v1|20260920|train|{row}|{split}').digest()[:8], 'big') & (2**63 - 1), split in (1, 2)",
                   noise_seeds_columns=['split1', 'split2']),
        files=dict(theta_design=THETA.name, noise_seeds=SEEDS.name, csv=CSV.name),
        generated_with=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__),
        checks=checks,
        sha256={path.name: sha256(path) for path in (THETA, SEEDS, CSV)})
    MANIFEST.write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(dict(checks=checks, sha256=manifest['sha256']), indent=2))


if __name__ == '__main__':
    main()
