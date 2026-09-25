"""Reanalyse completed spherical controls; never interpret bins as MOPED."""
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
PREVIOUS=ROOT.parent/'tsz_spherical_preflight_20260920'
sys.path.insert(0,str(PREVIOUS))
from plot_results import bins,COV,EDGES,ELLB

low=PREVIOUS/'fullsky/Battaglia12/nside4096_grid1'
high=PREVIOUS/'fullsky/Battaglia12/nside8192_grid1'
for path in [low,high]:assert json.loads((path/'status.json').read_text())['returncode']==0
a,b=[bins(np.load(path/'masked_clean_cl.npy')) for path in [low,high]]
rows=[]
for cutoff in [1000,1500,2000,3000,4000,5000,6000,7979]:
    selected=EDGES[1:]-1<=cutoff
    delta=(a-b)[selected];covariance=COV['covariance'][np.ix_(selected,selected)]
    distance=np.linalg.norm(np.linalg.solve(np.linalg.cholesky(covariance),delta))
    rows.append(dict(requested_ell_max=cutoff,last_complete_bin_ell_max=int((EDGES[1:]-1)[selected][-1]),
        bins=int(selected.sum()),conditional_binned_noise_distance=float(distance),
        max_fractional_difference=float(np.max(abs(delta/b[selected])))))
ell=np.arange(80,7980);sigma=(2*np.pi/10800)/np.sqrt(8*np.log(2))
beam2=np.exp(-ell*(ell+1)*sigma**2)
plt.rcParams.update({'font.size':15,'axes.labelsize':17,'legend.fontsize':12})
fig,axes=plt.subplots(3,1,figsize=(9,11),sharex=True)
axes[0].semilogy(ELLB,a,label='4096');axes[0].semilogy(ELLB,b,label='8192')
axes[0].set(ylabel=r'$D_b^{yy}$',title='Spherical Battaglia12, 2 arcmin beam');axes[0].legend(title='Raw NSIDE')
axes[1].plot(ELLB,100*(a/b-1),color='#D55E00');axes[1].axhline(0,color='k',lw=.7)
axes[1].set_ylabel('4096 / 8192 - 1 [%]')
axes[2].plot([r['last_complete_bin_ell_max'] for r in rows],
    [r['conditional_binned_noise_distance'] for r in rows],'-o',label='40-bin noise metric')
axes[2].set(xlabel=r'$\ell_{\max}$',ylabel='Cumulative discrepancy');axes[2].legend()
for axis in axes:axis.grid(alpha=.2)
fig.tight_layout()
for suffix in ['png','pdf']:fig.savefig(ROOT/'plots'/f'initial_resolution_cutoffs.{suffix}',dpi=180)
result=dict(rows=rows,beam_power={str(e):float(beam2[e-80]) for e in [1000,2000,4000,6000,7979]},
    scope='Completed spherical B12 controls. Saved 64-draw conditional 40-bin SO covariance; excludes cosmic variance. '
          'Not unbinned MOPED, posterior bias, or validation of all prior combinations.')
(ROOT/'results/initial_evidence.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
