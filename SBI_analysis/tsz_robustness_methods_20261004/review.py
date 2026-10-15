#!/usr/bin/env python3
"""Inspect progress or verify a completed report without loading large posterior files."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from common import ROOT, OLD, SEEDS, sha, write_json


def review(root, verify=False):
    plan=json.loads((root/'plan.json').read_text())
    trained=list((root/'candidates').glob('*/*_seed[123].json'))
    screened=list((root/'candidates').glob('*/evaluation.json'))
    final=list((root/'candidates').glob('*/evaluation_final.json'))
    bounded=list((root/'candidates').glob('*/bounded/evaluation_final.json'))
    complete=root/'COMPLETE.json'
    result=dict(utc=datetime.now(timezone.utc).isoformat(),complete=complete.exists(),
        validated_simulations=[s for s in SEEDS if (root/f'results/seed_{s}/VALIDATED.json').exists()],
        trained_primary_networks=len(trained),expected_primary_networks=3*len(plan['candidates']),
        screened_methods=len(screened),final_primary_methods=len(final),final_bounded_methods=len(bounded))
    if verify:
        assert complete.exists(), 'The full inference pipeline has not completed'
        completion=json.loads(complete.read_text())
        assert completion['observations']==100
        assert completion['methods']==len(plan['candidates'])+1
        assert sha(root/'plan.json')==completion['plan_sha256']
        assert sha(root/'input_sha256.json')==completion['inputs_sha256']
        for path,expected in completion['report_sha256'].items():
            assert sha(root/path)==expected, root/path
        model_hashes={}
        paths=list((OLD/'models').glob('*/*_seed[123].pt'))
        for recipe in plan['candidates']:
            folder=root/'candidates'/recipe['name']
            for seed in plan['training_seeds']:
                path=folder/f"{recipe['network']}_seed{seed}.pt"
                assert path.exists(),path
                paths.append(path)
                bounded_path=folder/'bounded'/path.name
                if bounded_path.exists():paths.append(bounded_path)
        for path in paths:
            model_hashes[str(path)]=sha(path)
        evaluation_hashes={}
        for path in screened+final+bounded:
            record=json.loads(path.read_text())
            assert np.array(record['median']).shape==(100,9)
            assert np.isfinite(record['median']).all()
            assert np.all(np.array(record['width68'])>0)
            evaluation_hashes[str(path)]=sha(path)
        result.update(report_artifacts=len(completion['report_sha256']),
                      trained_model_sha256=model_hashes,evaluation_sha256=evaluation_hashes,
                      completion_sha256=sha(complete),status='verified')
        write_json(root/'VERIFIED.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--verify',action='store_true')
    args=p.parse_args();review(args.root,args.verify)
