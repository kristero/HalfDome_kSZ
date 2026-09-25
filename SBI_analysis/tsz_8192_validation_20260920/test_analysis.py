"""Synthetic end-to-end MOPED/cutoff analysis with all 7900 multipoles."""
import json
from pathlib import Path
import tempfile
import numpy as np
import analyze

ROOT=Path(__file__).resolve().parent
rng=np.random.default_rng(87)
plan=json.loads((ROOT/'plan.json').read_text())
tasks=[t for t in plan['tasks'] if t.get('anchor')=='Battaglia12']
base=tasks[0]
with tempfile.TemporaryDirectory(prefix='tsz_analysis_test_') as directory:
    fake=Path(directory);(fake/'inputs').mkdir()
    import shutil
    shutil.copy2(ROOT/'inputs/frozen_unbinned_moped_transform.npz',fake/'inputs/frozen_unbinned_moped_transform.npz')
    x=analyze.ELL/1000;mean=(1+x)**-1*1e-12
    shape=rng.normal(size=(7900,9))*1e-14
    width=np.array(plan['upper'])-np.array(plan['lower'])
    def write(path,dl):
        path.parent.mkdir(parents=True,exist_ok=True)
        cl=np.zeros(7980);cl[analyze.ELL]=dl/analyze.FACTOR;np.save(path,cl)
    for task in tasks:
        out=fake/'controls'/f"{task['id']:03d}"
        delta=(np.array(task['theta'])-np.array(base['theta']))/width
        dl=mean+shape@delta
        write(out/'masked_clean_cl.npy',dl)
    out=fake/'controls'/f"{base['id']:03d}"
    shift=.001*shape[:,1]
    write(out/'paired4096_clean_cl.npy',mean+shift)
    for i in range(128):
        noise=rng.normal(size=7900)*1e-14
        write(out/'noise'/f'{i:03d}.npy',mean+noise)
        write(out/'noise'/f'{i:03d}.npy.paired4096.npy',mean+noise+shift)
    previous=analyze.ROOT;analyze.ROOT=fake
    result=analyze.analyze_anchor('Battaglia12',base,tasks,plan)
    analyze.ROOT=previous
    assert len(result['cuts'])==8
    assert all(np.isfinite(r['spherical_moped_distance']) for r in result['cuts'])
    assert max(result['derivative_step_relative_change'])<1e-8
    assert abs(result['cuts'][-1]['information_trace_fraction']-1)<1e-12
(ROOT/'results/analysis_software_test.json').write_text(json.dumps(dict(
    passed=True,features=7900,cutoffs=plan['ell_cuts'],
    scope='Synthetic analysis execution; no scientific resolution conclusion'),indent=2)+'\n')
print('PASS synthetic 7900-feature analysis at eight cutoffs')
