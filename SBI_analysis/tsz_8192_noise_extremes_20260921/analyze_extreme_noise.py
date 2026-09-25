"""Frozen unbinned-MOPED resolution shifts with each extreme's own noise."""
import json,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent/'tsz_8192_noise_extremes_20260921'
sys.path.insert(0,str(HERE.parent/'tsz_8192_validation_20260920'))
from analyze import load_dl,frozen_coordinates,covariance_distance,ELL

transform=np.load(ROOT/'inputs/frozen_unbinned_moped_transform.npz')
records=[]
for task in json.loads((ROOT/'plan.json').read_text())['tasks']:
    path=ROOT/'controls'/f"{task['id']:03d}"
    assert json.loads((path/'status.json').read_text())['returncode']==0
    current=np.array([load_dl(path/'noise'/f'{i:03d}.npy') for i in range(128)])
    paired=np.array([load_dl(path/'noise'/f'{i:03d}.npy.paired4096.npy') for i in range(128)])
    rows=[]
    for cutoff in [1000,1500,2000,3000,4000,5000,6000,7979]:
        keep=ELL<=cutoff
        a=frozen_coordinates(current,transform,keep);b=frozen_coordinates(paired,transform,keep)
        shift=(b[64:]-a[64:]).mean(0)
        distance,meta=covariance_distance(shift,a[:64])
        alternative,_=covariance_distance(shift,a[64:])
        rows.append(dict(ell_max=cutoff,distance=distance,alternate_covariance_distance=alternative,metadata=meta))
    records.append(dict(label=task['label'],cuts=rows))
(ROOT/'results').mkdir(exist_ok=True)
(ROOT/'results/extreme_noise.json').write_text(json.dumps(dict(records=records,
    scope='Own-sky conditional split-noise covariance; frozen old unbinned transform; no posterior calibration'),indent=2)+'\n')
print(json.dumps(records,indent=2))
