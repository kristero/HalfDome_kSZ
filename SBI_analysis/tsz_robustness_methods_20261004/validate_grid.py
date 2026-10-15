#!/usr/bin/env python3
"""Validate new painting against the old same-seed baseline and all ten masks."""
import argparse
import json
from pathlib import Path
import tomllib

import numpy as np

from common import ROOT, OLD, MASKS, NOISE_SEEDS, THETA, sha, write_json


def validate(root, seed):
    folder=root/f'results/seed_{seed}'
    meta=tomllib.loads((folder/'complete.toml').read_text())
    assert meta['noise_seeds']==NOISE_SEEDS
    np.testing.assert_array_equal(meta['theta'],THETA)
    assert meta['mask_seeds']==MASKS
    reference=tomllib.loads((OLD/'results/seed_100/complete.toml').read_text())
    assert meta['noise_sha256']==reference['noise_sha256']
    checks={}
    for mask in MASKS:
        case=folder/f'mask_{mask}'
        m=tomllib.loads((case/'complete.toml').read_text())
        assert m['mask_seed']==mask and m['simulation_seed']==seed
        for filename,expected in m['spectrum_sha256'].items():
            assert sha(case/filename)==expected,case/filename
    for name in ['clean_cl','production_cl','noise_cl','beam_corrected_cl']:
        a=np.load(folder/f'mask_12345/{name}.npy')[80:]
        b=np.load(OLD/f'results/seed_{seed}/mask_12345/{name}.npy')[80:]
        error=float(np.max(np.abs(a-b))/np.max(np.abs(b)))
        checks[name]=error
        assert error<1e-6,checks
    write_json(folder/'VALIDATED.json',dict(seed=seed,masks=MASKS,relative_errors=checks))
    print('VALIDATED',seed,checks,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--seed',type=int,required=True)
    a=p.parse_args();validate(a.root,a.seed)
