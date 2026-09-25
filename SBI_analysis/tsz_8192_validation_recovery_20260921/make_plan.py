"""Explicit finite preflight task list: candidate 8192, comparison 4096."""
import hashlib
import json
from pathlib import Path
import numpy as np
from design_tests import LOW,HIGH,NAMES

ROOT=Path(__file__).resolve().parent
cases=json.loads((ROOT/'inputs/cases.json').read_text())
tasks=[]

def add(label,theta,nside=8192,**options):
    tasks.append(dict(id=len(tasks),label=label,theta=list(theta),nside=nside,
        grid_factor=options.pop('grid_factor',1),noise_draws=options.pop('noise_draws',0),
        lean_maps=options.pop('lean_maps',False),threads=options.pop('threads',26),**options))

for anchor in ['Battaglia12','FL_L1_m9']:
    add(anchor,cases[anchor],anchor=anchor,noise_draws=128,paired_4096=True)

extremes={
    'compact': [18.1,.025,16,.154,-.00865,.0393,-.758,.731,.415],
    'extended_shallow': [18.1,4,2.8,.154,-.00865,0,-.758,.731,0],
    'combined_tails': cases['combined_tails'],
    'high_amplitude_tails': [60,.025,16,-.6,-1,.4,-6,3,2],
}
for label,theta in extremes.items():
    for nside in [8192,4096]:add(label,theta,nside,noise_draws=2 if nside==8192 else 0)
for anchor in ['Battaglia12','FL_L1_m9','compact','extended_shallow']:
    add(anchor,cases.get(anchor,extremes.get(anchor)),grid_factor=2)
for anchor in ['Battaglia12','FL_L1_m9']:
    theta=np.array(cases[anchor])
    for j in range(1,9):
        for fraction in [1.,.5]:
            step=.005*(HIGH[j]-LOW[j])*fraction
            for sign in [-1,1]:
                point=theta.copy();point[j]+=sign*step
                add(f'{anchor}_d{j}_{fraction}_{sign}',point,anchor=anchor,
                    derivative_index=j,step=step,step_fraction=fraction,sign=sign)
add('Battaglia12_fast26',cases['Battaglia12'],lean_maps=True,benchmark=True)
add('Battaglia12_fast13',cases['Battaglia12'],lean_maps=True,threads=13,benchmark=True)
for task in tasks:
    assert np.all(np.asarray(task['theta'])>=LOW) and np.all(np.asarray(task['theta'])<=HIGH)
    task['identity_sha256']=hashlib.sha256(json.dumps(task,sort_keys=True).encode()).hexdigest()
plan=dict(tasks=tasks,parameter_order=NAMES,lower=LOW.tolist(),upper=HIGH.tolist(),
    candidate_raw_nside=8192,comparison_raw_nside=4096,reference_only_nside=16384,
    output_nside=4096,beam_arcmin=2.,fsky=.4,mask_seed=12345,
    ell_min=80,ell_max=7979,ell_cuts=[1000,1500,2000,3000,4000,5000,6000,7979],
    covariance_draws=64,validation_draws=64,
    derivative_step_fraction_of_prior_width=.005,derivative_refinement=.5,
    derivative_P0='Exact 2*D/P0; all other parameters use two central-difference steps',
    noise_seed_rule='62000000+task_id*10000+2*draw+split; independent of prepared design',
    reference_root='/lustre/work/kristero10/tsz_spherical_preflight_20260920',
    scope='Preflight controls only; no 256-row dataset generation',production_certified=False)
(ROOT/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
print(f'Prepared {len(tasks)} control tasks; no scheduler submission.')
