"""Independently rerun anchor analysis locally from retrieved spectra.

The original anchor noise spectra were already downloaded in the first audit.
Use those identical retained controls to avoid another large network transfer.
"""
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
RUN = ROOT.parent / 'tsz_8192_validation_recovery_20260921'
OLD = ROOT.parent / 'tsz_8192_validation_20260920'
sys.path.insert(0,str(RUN))
import analyze

original_load = analyze.load_dl


def load_existing(path):
    path = Path(path)
    if not path.exists() and 'noise' in path.parts:
        path = OLD / path.relative_to(RUN)
    return original_load(path)


def main():
    analyze.ROOT=RUN
    analyze.load_dl=load_existing
    plan=json.loads((RUN/'plan.json').read_text())
    plan['reference_root']=str(ROOT.parent/'tsz_spherical_preflight_20260920')
    saved=json.loads((RUN/'results/report.json').read_text())
    results=[]
    for anchor in saved['anchors']:
        task=next(t for t in plan['tasks'] if t.get('anchor')==anchor['anchor'] and 'derivative_index' not in t)
        current=analyze.analyze_anchor(anchor['anchor'],task,plan['tasks'],plan)
        errors=[]
        for a,b in zip(current['cuts'],anchor['cuts']):
            for key in ['spherical_moped_distance','frozen_moped_distance',
                        'reference16384_spherical_moped_distance','information_trace_fraction']:
                np.testing.assert_allclose(a[key],b[key],rtol=1e-6,atol=1e-10)
                errors.append(abs(a[key]-b[key])/max(abs(b[key]),1e-30))
        results.append(dict(anchor=anchor['anchor'],max_relative_difference=max(errors),passed=True))
    (ROOT/'results/analysis_reproduction.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results,indent=2))


if __name__=='__main__':
    main()
