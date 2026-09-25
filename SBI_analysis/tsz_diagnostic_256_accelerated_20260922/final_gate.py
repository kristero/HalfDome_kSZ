"""Observable-level acceptance: interpolation, scheduling and output synthesis.

One percent refers to fractional CLEAN pseudo-D_ell error at every retained ell,
not to a fractional noisy cross spectrum, which can cross zero. The raw-pixel
8192/16384 residual is a separate measured systematic, not covered by this gate.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import toml
from launch import ROOT,atomic_json
from manage import audits

ELL=np.arange(80,7980)
FACTOR=ELL*(ELL+1)/(2*np.pi)
CONTEXT=np.load(ROOT/'reference/compression_context.npz')
WEIGHTS=[k for k in CONTEXT.files if '__' not in k]
INPUTS={}


def spectrum(path):
    data=np.load(path)
    assert data.shape==(7980,) and np.isfinite(data).all() and np.all(data>=0),str(path)
    INPUTS[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    return data[ELL]*FACTOR


def compare(candidate,reference,label=None):
    delta=candidate-reference
    assert np.all(reference>0)
    result=dict(relative_Dell_l2=float(np.linalg.norm(delta)/np.linalg.norm(reference)),
        max_fractional_Dell=float(np.max(abs(delta/reference))))
    if label is not None:
        distances={}
        for key in WEIGHTS:
            covariance=np.cov(CONTEXT[label+'__'+key],rowvar=False,ddof=1)
            values,vectors=np.linalg.eigh(covariance)
            assert np.all(values>0)
            distances[key]=float(np.linalg.norm(((delta@CONTEXT[key])@vectors)/np.sqrt(values)))
        result['conditional_moped']=distances
        result['distance9']=max(v for k,v in distances.items() if '1e-06' in k)
    return result


def main():
    assert toml.load(ROOT/'tests/gate/gate.toml')['passed']
    assert json.loads((ROOT/'results/training_software_test.json').read_text())['passed']
    independent=json.loads((ROOT/'results/independent_los.json').read_text())
    assert independent['passed'] and independent['rows']==256
    assert json.loads((ROOT/'results/unbinned_moped_tests.json').read_text())['unbinned_input_features']==7900
    records=audits()
    metrics=[];curves={'ell':ELL};passed=True
    for case in json.loads((ROOT/'anchors.json').read_text()):
        label=case['label'];folder=ROOT/'tests/anchors'/label
        coarse=spectrum(folder/'output4096/masked_clean_cl.npy')
        fine=spectrum(folder/'output8192/masked_clean_cl.npy')
        same=spectrum(ROOT/'reference'/label/'smooth_half_cl.npy')
        reference=spectrum(ROOT/'reference'/label/'smooth_double_cl.npy')
        for kind,other in [('scheduling',same),('cache',reference),('output_resolution',fine)]:
            metric=dict(label=label,kind=kind,**compare(coarse,other,label))
            metric['passed']=metric['max_fractional_Dell']<(.000000001 if kind=='scheduling' else .01)
            if kind!='scheduling':
                # 0.1 conditional-noise unit is a declared engineering budget.
                # The user permits residual bright-extreme sensitivity; report
                # it separately, without cutting the parameter distribution.
                metric['passed'] &= metric['distance9']<.1 or label=='extended_shallow'
            passed &= metric['passed'];metrics.append(metric)
        curves[label+'_candidate']=coarse;curves[label+'_output8192']=fine
        curves[label+'_cache_reference']=reference
    for group in range(2):
        task=json.loads((ROOT/f'tasks/spot{group}.json').read_text())
        assert json.loads((ROOT/f'tests/spot{group}/status.json').read_text())['returncode']==0
        for case in task['cases'][::2]:
            row=case['row'];folder=ROOT/f'tests/spot{group}'
            candidate=spectrum(folder/f'{row:05d}_candidate/masked_clean_cl.npy')
            reference=spectrum(folder/f'{row:05d}_reference/masked_clean_cl.npy')
            metric=dict(label=f'row{row}',kind='prior_spot_cache',nodes=case['nodes'],**compare(candidate,reference))
            metric['passed']=metric['max_fractional_Dell']<.01
            passed &= metric['passed'];metrics.append(metric)
            curves[f'row{row}_candidate']=candidate;curves[f'row{row}_cache_reference']=reference
    counts={}
    for record in records:
        key='x'.join(map(str,record['attempts'][-1]['nodes']))
        counts[key]=counts.get(key,0)+1
    np.savez_compressed(ROOT/'results/final_gate_spectra.npz',**curves)
    result=dict(passed=bool(passed),output_nside=4096,raw_nside=8192,ell_min=80,ell_max=7979,
        metrics=metrics,rows_directly_probed=len(records),cache_counts=counts,
        independent_projection={k:v for k,v in independent.items() if k!='records'},
        max_profile_visible_relative_error=max(r['attempts'][-1]['max_relative_visible'] for r in records),
        max_sampled_radial_L1_relative_error=max(r['max_sampled_radial_L1_relative_error'] for r in records),
        target='Less than 1 percent clean per-ell accelerated-versus-reference error',
        caveats=['Finite sampled tests, not a proof across continuous halo coordinates',
                 'Pixel-sampling residual at NSIDE8192 remains; separate 16384 reference results apply',
                 'Bright/shallow noise-unit budget explicitly advisory; fractional target still required'],
        input_sha256=INPUTS)
    atomic_json(ROOT/'results/final_gate.json',result)
    print(json.dumps(result,indent=2),flush=True)
    if not passed:raise RuntimeError('Accuracy gate failed: do not start the dataset; investigate saved discrepancies')


if __name__=='__main__':main()
