"""Report a dated snapshot without waiting for all derivative controls."""
import json,sys,re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
try:import tomllib
except ImportError:import toml as tomllib

ROOT=Path(__file__).resolve().parent
DATA=ROOT.parent/'tsz_8192_validation_20260920'
REFERENCE=ROOT.parent/'tsz_spherical_preflight_20260920'
sys.path.insert(0,str(DATA))
from analyze import load_dl,frozen_coordinates,covariance_distance,ELL

def folder(i):return DATA/'controls'/f'{i:03d}'

def main():
    plan=json.loads((DATA/'plan.json').read_text());tasks=plan['tasks']
    complete=[t for t in tasks if (folder(t['id'])/'status.json').exists() and
        json.loads((folder(t['id'])/'status.json').read_text())['returncode']==0]
    result=dict(completed=len(complete),planned=80,remaining=80-len(complete),
        scheduling_issue='Cross-node flock ineffective on localflock Lustre; old jobs stopped and disjoint shards submitted.',
        result_scope='Preserved data: source, shape and finiteness checked; repeat clean controls queued for provenance validation.',
        anchors=[],extremes=[],cache=[],timings=[])
    transform=np.load(DATA/'inputs/frozen_unbinned_moped_transform.npz')
    plt.rcParams.update({'font.size':14,'axes.labelsize':16,'legend.fontsize':12})
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for i,label in enumerate(['Battaglia12','FL_L1_m9']):
        path=folder(i);clean=load_dl(path/'masked_clean_cl.npy')
        low=load_dl(path/'paired4096_clean_cl.npy')
        noisy=np.array([load_dl(path/'noise'/f'{j:03d}.npy') for j in range(128)])
        lower=np.array([load_dl(path/'noise'/f'{j:03d}.npy.paired4096.npy') for j in range(128)])
        records=[]
        for cutoff in plan['ell_cuts']:
            keep=ELL<=cutoff
            high_t=frozen_coordinates(noisy,transform,keep);low_t=frozen_coordinates(lower,transform,keep)
            differences=low_t[64:]-high_t[64:]
            distance,meta=covariance_distance(differences.mean(0),high_t[:64])
            # Covariance estimated from 64 mocks has sampling uncertainty.
            # Compare an independent covariance half and an OAS compressed fit.
            alternate,_=covariance_distance(differences.mean(0),high_t[64:])
            records.append(dict(ell_max=cutoff,distance=distance,validation_cov_distance=alternate,
                covariance_rank=meta['covariance_rank'],mean_shift=differences.mean(0).tolist(),
                paired_shift_standard_error=(differences.std(0,ddof=1)/8).tolist()))
        independent=load_dl(REFERENCE/'fullsky'/label/'nside8192_grid1/masked_clean_cl.npy')
        parity=float(np.linalg.norm(independent-clean)/np.linalg.norm(independent))
        noise_info=tomllib.loads((path/'control.toml').read_text())
        first=load_dl(path/'first_draw_map_route.npy')
        full_parity=float(np.linalg.norm(first-noisy[0])/np.linalg.norm(first))
        result['anchors'].append(dict(label=label,cuts=records,
            independent_four_thread_clean_parity=parity,full_size_noise_route_parity=full_parity,
            unique_noise_hashes=len(set(v[k] for v in noise_info['noise_draws'] for k in ['noise1_sha256','noise2_sha256']))))
        axes[0].plot([r['ell_max'] for r in records],[r['distance'] for r in records],'-o',label=label)
        axes[1].plot([r['ell_max'] for r in records],[r['validation_cov_distance'] for r in records],'-o',label=label)
    axes[0].set_title('Frozen unbinned MOPED');axes[1].set_title('Independent covariance half')
    for ax in axes:
        ax.set(xlabel=r'$\ell_{\max}$',ylabel='4096-8192 shift / noise scale');ax.legend();ax.grid(alpha=.2)
    fig.tight_layout()
    for suffix in ['pdf','png']:fig.savefig(ROOT/'plots'/f'frozen_moped_cutoffs.{suffix}',dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,9),sharex=True)
    for ax,(label,high,low) in zip(axes.ravel(),[('Compact',2,3),('Extended / shallow',4,5),('Combined tails',6,7),('High-amplitude tails',8,9)]):
        a,b=load_dl(folder(low)/'masked_clean_cl.npy'),load_dl(folder(high)/'masked_clean_cl.npy')
        groups=np.array_split(np.arange(len(ELL)),79);ell=np.array([ELL[g].mean() for g in groups])
        ratio=np.array([a[g].mean()/b[g].mean()-1 for g in groups])
        ax.plot(ell,ratio*100);ax.set(title=label,ylabel='4096 / 8192 - 1 [%]');ax.grid(alpha=.2)
        result['extremes'].append(dict(label=label,fractional_dl_l2_error=float(np.linalg.norm(a-b)/np.linalg.norm(b)),
            max_plot_band_fractional_error=float(np.max(abs(ratio))),
            power_ratio_at_ell4000=float(np.mean(a[3900:4000])/np.mean(b[3900:4000]))))
    for ax in axes[1]:ax.set_xlabel(r'$\ell$')
    fig.tight_layout()
    for suffix in ['pdf','png']:fig.savefig(ROOT/'plots'/f'extreme_resolution.{suffix}',dpi=180)
    plt.close(fig)
    for refined,coarse in [(10,0),(11,1),(12,2),(13,4)]:
        a,b=load_dl(folder(coarse)/'masked_clean_cl.npy'),load_dl(folder(refined)/'masked_clean_cl.npy')
        result['cache'].append(dict(label=tasks[coarse]['label'],
            fractional_dl_l2_error=float(np.linalg.norm(a-b)/np.linalg.norm(b)),
            max_fractional_ell_error=float(np.max(abs(a/b-1)))))
    for task in complete:
        path=folder(task['id']);meta=tomllib.loads((path/'control.toml').read_text())
        status=json.loads((path/'status.json').read_text())
        text=(path/f"time_{status['attempt']:02d}.txt").read_text()
        match=re.search(r'Maximum resident set size \(kbytes\):\s*(\d+)',text)
        result['timings'].append(dict(id=task['id'],label=task['label'],nside=task['nside'],
            wall_seconds=status['seconds'],phases=meta['phases'],
            rss_gib=int(match[1])/2**20 if match else None,
            noise_seconds=[r['seconds'] for r in meta['noise_draws']],
            underflow_cells=meta['cache']['underflow_count']))
    result['reference']=json.loads((REFERENCE/'results/fullsky.json').read_text())
    (ROOT/'results/snapshot.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['timings','reference']},indent=2))

if __name__=='__main__':main()
