"""Synthetic software check only; never generates a HalfDome diagnostic row."""
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
from design_tests import LOW,HIGH,NAMES,design
from diagnostic_analysis import rebin_unbinned
ROOT=Path(__file__).resolve().parent
out=ROOT/'results/software_smoke_unbinned';out.mkdir(exist_ok=True)
theta=design(64);rng=np.random.default_rng(71)
ell=np.arange(80,7980);factor=ell*(ell+1)/(2*np.pi)
features=rng.normal(0,.2,(9,len(ell)));u=(theta-LOW)/(HIGH-LOW)
raw_clean=np.exp((u-.5)@features)*1e-13
raw_noisy=raw_clean+rng.normal(0,1e-14,raw_clean.shape)
clean=rebin_unbinned(raw_clean,ell);noisy=rebin_unbinned(raw_noisy,ell)
for i in range(len(theta)):
    folder=out/'rows'/f'{i:05d}';folder.mkdir(parents=True,exist_ok=True)
    for name,data in [('masked_clean_cl.npy',raw_clean),('masked_noisy_cross_cl.npy',raw_noisy)]:
        spectrum=np.zeros(7980);spectrum[ell]=data[i]/factor
        np.save(folder/name,spectrum)
np.save(out/'theta_design.npy',theta);np.save(out/'heldout_design.npy',np.arange(64)>=48)
np.savez(out/'dataset.npz',row_id=np.arange(64),theta=theta,clean_dl=clean,noisy_dl=noisy,
         lower=LOW,upper=HIGH,parameter_order=NAMES)
(out/'manifest.json').write_text(json.dumps(dict(count=64,scope='synthetic smoke, no physical validation')))
(out/'observations').mkdir(exist_ok=True)
np.savez(out/'observations/synthetic_external.npz',noisy_dl=noisy[0],
    ell_unbinned=ell,noisy_dl_unbinned=raw_noisy[0])
cmd=[sys.executable,str(ROOT/'diagnostic_analysis.py'),'--run-root',str(out),'--train',
     '--threads','2','--epochs','1','--posterior-samples','16','--max-proposals','2048','--seeds','71']
with (out/'run.log').open('w') as log:code=subprocess.call(cmd,stdout=log,stderr=subprocess.STDOUT)
(out/'status.json').write_text(json.dumps(dict(returncode=code,scope='Synthetic software execution only'),indent=2))
print('Synthetic analysis smoke return code:',code,flush=True)
if code:print((out/'run.log').read_text()[-7000:]);raise SystemExit(code)
