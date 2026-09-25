"""Automatic report after the resolution PBS test; no production authorization."""
import json
from pathlib import Path
import re
import numpy as np
from plot_results import ROOT,STUDY,bins,spectra,COV

records=[];pairwise=[]
cov=COV['covariance'];chol=np.linalg.cholesky(cov)
for case in ['Battaglia12','FL_L1_m9']:
    variants={}
    for n in [4096,8192,16384]:
        path=ROOT/'fullsky'/case/f'nside{n}_grid1'
        if not (path/'status.json').exists():continue
        status=json.loads((path/'status.json').read_text())
        if status['returncode']!=0:continue
        cl=bins(np.load(path/'masked_clean_cl.npy'))
        rss=re.search(r'Maximum resident set size \(kbytes\):\s*(\d+)',(path/'time.txt').read_text())
        variants[n]=dict(path=path,cl=cl,seconds=status['seconds'],rss_gib=int(rss[1])/2**20)
    for n,reference in [(4096,8192),(4096,16384),(8192,16384)]:
        if n not in variants or reference not in variants:continue
        delta=variants[n]['cl']-variants[reference]['cl']
        pairwise.append(dict(case=case,nside=n,reference_nside=reference,
            whitened_discrepancy=float(np.linalg.norm(np.linalg.solve(chol,delta))),
            max_fractional_bandpower_error=float(np.max(abs(delta/variants[reference]['cl']))),
            scope='40-bin conditional-noise spectrum check, not an unbinned MOPED posterior bias'))
    for n,r in variants.items():
        row=dict(case=case,nside=n,seconds=r['seconds'],rss_gib=r['rss_gib'])
        if 16384 in variants:
            delta=r['cl']-variants[16384]['cl']
            row.update(whitened_discrepancy=float(np.linalg.norm(np.linalg.solve(chol,delta))),
                max_bin_error_over_sigma=float(np.max(abs(delta)/np.sqrt(np.diag(cov)))))
        records.append(row)
result=dict(measurements=records,pairwise=pairwise,scope='Spherical point painter; matched coarse log-z cache and same 2 arcmin beam/mask',
    covariance='Saved 64-draw conditional B12 split-noise covariance, excludes cosmic variance; approximate at FL fit',
    production_certified=False,
    remaining='Cache refinement, flat-prior joint extremes, full-sky pre-beam/pixel integration, actual noisy-row timing, nine-parameter inference checks')
(ROOT/'results').mkdir(exist_ok=True)
(ROOT/'results/fullsky.json').write_text(json.dumps(result,indent=2)+'\n')
(ROOT/'plots').mkdir(exist_ok=True);spectra()
print(json.dumps(result,indent=2),flush=True)
