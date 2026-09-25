"""Resolution effects before and after fixed unbinned compression, by ell cutoff.

All uncertainty scales are conditional SO split-noise scales on fixed skies.
No covariance here includes cosmic variance, and no automatic run is authorized.
"""
import json
from pathlib import Path
import re
try:
    import tomllib
except ImportError:
    import toml as tomllib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from unbinned_moped import oas_inverse_action

ROOT=Path(__file__).resolve().parent
ELL=np.arange(80,7980);FACTOR=ELL*(ELL+1)/(2*np.pi)

def load_dl(path):
    value=np.load(path)
    if value.ndim!=1 or len(value)<7980:raise ValueError('Missing multipoles: '+str(path))
    value=value[ELL]*FACTOR
    if not np.isfinite(value).all():raise ValueError('Nonfinite spectrum: '+str(path))
    return value

def covariance_distance(delta,samples):
    """Empirical compressed covariance; expose rather than invent weak modes."""
    covariance=np.atleast_2d(np.cov(samples,rowvar=False,ddof=1))
    eigenvectors=None
    eigenvalues,eigenvectors=np.linalg.eigh(covariance)
    keep=eigenvalues>max(eigenvalues[-1]*1e-10,1e-30)
    if not np.any(keep):raise ValueError('Compressed noise covariance has zero measurable rank')
    whitened=(np.asarray(delta)@eigenvectors[:,keep])/np.sqrt(eigenvalues[keep])
    return float(np.linalg.norm(whitened)),dict(covariance_rank=int(keep.sum()),
        covariance_eigenvalues=eigenvalues.tolist(),covariance_samples=len(samples))

def moped_matrix(derivatives,residuals,rcond=1e-3):
    inv,action,metadata=oas_inverse_action(residuals,derivatives)
    fisher=derivatives.T@inv
    values,vectors=np.linalg.eigh((fisher+fisher.T)/2)
    values,vectors=values[::-1],vectors[:,::-1]
    keep=values>max(values[0]*rcond**2,1e-28)
    matrix=(inv@vectors[:,keep])/np.sqrt(values[keep])
    error=np.linalg.norm(matrix.T@action(matrix)-np.eye(keep.sum()))
    if error>1e-5:raise ValueError('MOPED covariance identity failed')
    metadata.update(rank=int(keep.sum()),rcond=rcond,singular_values=np.sqrt(np.maximum(values,0)).tolist(),
        covariance_identity_error=float(error))
    return matrix,fisher,metadata

def frozen_coordinates(dl,transform,keep):
    z=(np.arcsinh(dl/transform['scale'])-transform['mean'])/transform['std']
    # Constants cancel in shifts and covariance. For a reduced ell cutoff this
    # restricts existing weights; it is not a newly trained lower-ell network.
    return z[...,keep]@transform['matrix'][keep]/transform['output_std']

def folder(task):return ROOT/'controls'/f"{task['id']:03d}"

def analyze_anchor(anchor,task,tasks,plan):
    path=folder(task);current=load_dl(path/'masked_clean_cl.npy')
    lower=load_dl(path/'paired4096_clean_cl.npy')
    noisy=np.array([load_dl(path/'noise'/f'{i:03d}.npy') for i in range(128)])
    paired=np.array([load_dl(path/'noise'/f'{i:03d}.npy.paired4096.npy') for i in range(128)])
    residual=noisy-current
    sigma=np.maximum(np.std(residual[:64],axis=0,ddof=1),1e-30)
    transform=np.load(ROOT/'inputs/frozen_unbinned_moped_transform.npz')
    np.testing.assert_array_equal(transform['ell'],ELL)
    derivative_tasks=[t for t in tasks if t.get('anchor')==anchor and 'derivative_index' in t]
    matrices={fraction:np.empty((7900,9)) for fraction in [1.,.5]}
    width=np.array(plan['upper'])-np.array(plan['lower'])
    for fraction,derivatives in matrices.items():
        derivatives[:,0]=2*current/task['theta'][0]*width[0]
        for j in range(1,9):
            selected=[t for t in derivative_tasks if t['derivative_index']==j and t['step_fraction']==fraction]
            plus=next(t for t in selected if t['sign']==1);minus=next(t for t in selected if t['sign']==-1)
            derivatives[:,j]=(load_dl(folder(plus)/'masked_clean_cl.npy')-
                              load_dl(folder(minus)/'masked_clean_cl.npy'))/(2*plus['step'])*width[j]
    change=np.linalg.norm((matrices[1.]-matrices[.5])/sigma[:,None],axis=0)
    change/=np.maximum(np.linalg.norm(matrices[.5]/sigma[:,None],axis=0),1e-30)
    rows=[];fishers=[]
    for cutoff in plan['ell_cuts']:
        keep=ELL<=cutoff;delta=(lower-current)[keep]/sigma[keep]
        noise=residual[:,keep]/sigma[keep]
        derivatives=matrices[.5][keep]/sigma[keep,None]
        weights,fisher,meta=moped_matrix(derivatives,noise[:64])
        compressed=noise[64:]@weights
        shift=delta@weights
        distance,validation=covariance_distance(shift,compressed)
        original=frozen_coordinates(noisy,transform,keep)
        alternative=frozen_coordinates(paired,transform,keep)
        difference=alternative[64:]-original[64:]
        frozen_distance,frozen_meta=covariance_distance(difference.mean(0),original[:64])
        clean_frozen=frozen_coordinates(lower,transform,keep)-frozen_coordinates(current,transform,keep)
        raw_inverse,_,raw_meta=oas_inverse_action(noise[:64],delta[:,None])
        raw_distance=float(np.sqrt(max(delta@raw_inverse[:,0],0)))
        refinements=[]
        for fraction in [1.,.5]:
            for rcond in [1e-3,1e-4,1e-6]:
                try:
                    w,_,m=moped_matrix(matrices[fraction][keep]/sigma[keep,None],noise[:64],rcond)
                    d,_=covariance_distance(delta@w,noise[64:]@w)
                    refinements.append(dict(step_fraction=fraction,rcond=rcond,rank=m['rank'],distance=d))
                except ValueError as error:refinements.append(dict(step_fraction=fraction,rcond=rcond,error=str(error)))
        rows.append(dict(ell_max=cutoff,raw_shrunk_noise_distance=raw_distance,
            spherical_moped_distance=distance,spherical_moped=meta,heldout_noise_covariance=validation,
            frozen_moped_distance=frozen_distance,frozen_moped_covariance=frozen_meta,
            frozen_paired_mean_shift=difference.mean(0).tolist(),
            frozen_paired_mean_shift_standard_error=(difference.std(0,ddof=1)/8).tolist(),
            frozen_clean_context_shift=clean_frozen.tolist(),sensitivity=refinements))
        reference=Path(plan['reference_root'])/'fullsky'/anchor/'nside16384_grid1'
        marker=reference/'status.json'
        if marker.exists() and json.loads(marker.read_text())['returncode']==0:
            delta_reference=(current-load_dl(reference/'masked_clean_cl.npy'))[keep]/sigma[keep]
            rows[-1]['reference16384_spherical_moped_distance']=covariance_distance(
                delta_reference@weights,compressed)[0]
        fishers.append(fisher)
    full=fishers[-1];eigenvalues,eigenvectors=np.linalg.eigh(full)
    mask=eigenvalues>eigenvalues[-1]*1e-6
    normalizer=eigenvectors[:,mask]/np.sqrt(eigenvalues[mask])
    for row,fisher in zip(rows,fishers):
        row['information_trace_fraction']=float(np.trace(fisher)/np.trace(full))
        row['identified_mode_information_fractions']=np.linalg.eigvalsh(normalizer.T@fisher@normalizer).tolist()
    return dict(anchor=anchor,derivative_step_relative_change=change.tolist(),cuts=rows,
        noise_scope='Fixed sky, same split seeds at both resolutions; independent draws 0:64 covariance fit, 64:128 validation',
        frozen_scope='Historical fixed-noise-trained asinh MOPED; lower cuts restrict weights, do not retrain its SBI',
        spherical_scope='Local linear-Dell fixed-covariance Fisher/MOPED, not full nonlinear nine-parameter posterior bias')

def main():
    plan=json.loads((ROOT/'plan.json').read_text());tasks=plan['tasks']
    out=ROOT/'results';out.mkdir(exist_ok=True);(ROOT/'plots').mkdir(exist_ok=True)
    status=[];complete={}
    for task in tasks:
        marker=folder(task)/'status.json'
        record=json.loads(marker.read_text()) if marker.exists() else dict(returncode=None)
        status.append({**record,'task_id':task['id'],'label':task['label']})
        if record['returncode']==0:complete[task['id']]=task
    result=dict(completed=len(complete),requested=len(tasks),statuses=status,
        production_certified=False,diagnostic_256_submitted=False,anchors=[],rendering=[],benchmarks=[],cache_refinement=[],analysis_errors=[],
        interpretation='8192 is a candidate, 4096 is tested, 16384 is reference only; no automatic launch')
    for anchor in ['Battaglia12','FL_L1_m9']:
        required=[t for t in tasks if t.get('anchor')==anchor]
        if all(t['id'] in complete for t in required):
            try:result['anchors'].append(analyze_anchor(anchor,required[0],tasks,plan))
            except (ValueError,np.linalg.LinAlgError) as error:
                result['analysis_errors'].append(dict(anchor=anchor,error=str(error)))
    for label in ['compact','extended_shallow','combined_tails','high_amplitude_tails']:
        selected=[t for t in tasks if t['label']==label and t['grid_factor']==1]
        if not all(t['id'] in complete for t in selected):continue
        low=next(t for t in selected if t['nside']==4096);high=next(t for t in selected if t['nside']==8192)
        a,b=load_dl(folder(low)/'masked_clean_cl.npy'),load_dl(folder(high)/'masked_clean_cl.npy')
        result['rendering'].append(dict(label=label,fractional_l2_error=float(np.linalg.norm(a-b)/np.linalg.norm(b))))
    for refined in [t for t in tasks if t['grid_factor']==2]:
        coarse=next(t for t in tasks if t['label']==refined['label'] and t['nside']==8192 and t['grid_factor']==1)
        if coarse['id'] not in complete or refined['id'] not in complete:continue
        a=load_dl(folder(coarse)/'masked_clean_cl.npy');b=load_dl(folder(refined)/'masked_clean_cl.npy')
        result['cache_refinement'].append(dict(label=refined['label'],
            fractional_l2_error=float(np.linalg.norm(a-b)/np.linalg.norm(b)),
            max_absolute_error=float(np.max(abs(a-b)))))
    for task in tasks:
        if task['id'] not in complete:continue
        status_record=json.loads((folder(task)/'status.json').read_text())
        attempt=status_record['attempt']
        text=(folder(task)/f'time_{attempt:02d}.txt').read_text()
        match=re.search(r'Maximum resident set size \(kbytes\):\s*(\d+)',text)
        row=dict(task_id=task['id'],label=task['label'],threads=task['threads'],lean_maps=task['lean_maps'],
            wall_seconds=status_record['seconds'],peak_rss_gib=float(match[1])/2**20 if match else None)
        measurement=folder(task)/'control.toml'
        if measurement.exists():
            timing=tomllib.loads(measurement.read_text())
            row['phases_seconds']=timing['phases']
            row['candidate_signal_seconds']=sum(value for key,value in timing['phases'].items()
                if key.startswith('raw8192_'))
            noise=timing['noise_draws']
            if noise:row['noise_draw_seconds_median']=float(np.median([r['seconds'] for r in noise]))
        if task.get('benchmark'):
            reference=folder(tasks[0])/'masked_clean_cl.npy'
            if reference.exists():
                a=load_dl(folder(task)/'masked_clean_cl.npy');b=load_dl(reference)
                row['spectrum_relative_l2_error']=float(np.linalg.norm(a-b)/np.linalg.norm(b))
                row['spectrum_equivalence_pass']=bool(row['spectrum_relative_l2_error']<1e-8)
        result['benchmarks'].append(row)
    baseline=next((row for row in result['benchmarks'] if row['task_id']==0),None)
    if baseline and baseline.get('candidate_signal_seconds',0)>0:
        for row in result['benchmarks']:
            if tasks[row['task_id']].get('benchmark') and row.get('candidate_signal_seconds',0)>0:
                row['signal_speedup_vs_standard26']=baseline['candidate_signal_seconds']/row['candidate_signal_seconds']
                row['signal_core_seconds']=row['candidate_signal_seconds']*row['threads']
    result['reference_controls']=reference_comparison(plan)
    if result['anchors']:
        plt.rcParams.update({'font.size':15,'axes.labelsize':17,'legend.fontsize':12})
        fig,axes=plt.subplots(1,3,figsize=(17,5))
        for record in result['anchors']:
            cuts=record['cuts'];x=[r['ell_max'] for r in cuts]
            axes[0].semilogy(x,[max(r['frozen_moped_distance'],1e-8) for r in cuts],'-o',label=record['anchor'])
            axes[1].semilogy(x,[max(r['spherical_moped_distance'],1e-8) for r in cuts],'-o',label=record['anchor'])
            axes[2].plot(x,[r['information_trace_fraction'] for r in cuts],'-o',label=record['anchor'])
        for axis,title in zip(axes,['Existing unbinned MOPED','Spherical local MOPED','Local information retained']):
            axis.set(xlabel=r'$\ell_{\max}$',title=title);axis.grid(alpha=.2);axis.legend()
        axes[0].set_ylabel('Resolution shift / noise scale');axes[1].set_ylabel('Resolution shift / noise scale')
        axes[2].set_ylabel('Prior-scaled Fisher trace fraction')
        fig.tight_layout()
        for suffix in ['png','pdf']:fig.savefig(ROOT/'plots'/('moped_resolution_cutoffs.'+suffix),dpi=180)
    (out/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    summary=[f"# Resolution preflight\n\nCompleted {len(complete)}/{len(tasks)} controls.",
        'Raw NSIDE 8192 is the candidate; 4096 is an alternative and 16384 is reference only. '
        'All output maps are 4096, with the same 2 arcmin beam and sky mask.',
        'The primary comparison is the resolution shift in a fixed unbinned compressed space '
        'relative to the scatter of a single observation. It is not the uncertainty of the '
        'mean of 64 mocks, and it is not a measured nine-parameter posterior bias.',
        'Noise covariance is conditional on each fixed sky. It excludes cosmic variance, '
        'foreground residuals and model discrepancy. Split noise uses the historical multiplier 1 '
        'per split; half-depth survey splits would require a separate noise convention.',
        'The existing frozen compressor and spherical local MOPED answer different questions. '
        'The former measures what the old transform notices; the latter tests sensitivity '
        'of the new model. Derivative-step and rank-threshold sensitivity are retained.',
        'There is no universal NSIDE requirement. Compare compressed shift, retained information, '
        'cache convergence and extreme profiles at the intended multipole cut before selecting '
        '4096 or 8192. Neither is automatically certified by this report.']
    for record in result['anchors']:
        summary.append('## '+record['anchor']+'\n\n| ell max | Frozen MOPED shift | Spherical MOPED shift | Information trace fraction |\n|---:|---:|---:|---:|')
        summary.extend(f"| {r['ell_max']} | {r['frozen_moped_distance']:.3g} | {r['spherical_moped_distance']:.3g} | {r['information_trace_fraction']:.3g} |" for r in record['cuts'])
    summary.append('See `results/report.json` for failed or pending controls, timings, covariance ranks '
        'and sensitivity tests. The 256-row dataset is not submitted by these scripts.')
    (ROOT/'REPORT.md').write_text('\n\n'.join(summary)+'\n')
    print(json.dumps(dict(completed=len(complete),requested=len(tasks),anchors=len(result['anchors'])),indent=2))

def reference_comparison(plan):
    """Keep existing 16384 controls as references, never as production choices."""
    records=[]
    plt.rcParams.update({'font.size':15,'axes.labelsize':17,'legend.fontsize':12})
    for anchor in ['Battaglia12','FL_L1_m9']:
        variants={}
        for nside in [4096,8192,16384]:
            path=Path(plan['reference_root'])/'fullsky'/anchor/f'nside{nside}_grid1'
            marker=path/'status.json'
            if marker.exists() and json.loads(marker.read_text())['returncode']==0:
                variants[nside]=load_dl(path/'masked_clean_cl.npy')
        if not variants:continue
        fig,axes=plt.subplots(2,1,figsize=(9,8),sharex=True,gridspec_kw={'height_ratios':[2,1]})
        reference=variants.get(16384,variants.get(8192,next(iter(variants.values()))))
        # Plot-only averaging makes the signed unbinned analysis more legible.
        groups=np.array_split(np.arange(len(ELL)),79)
        centres=np.array([ELL[g].mean() for g in groups])
        for nside,dl in variants.items():
            label=str(nside)+(' (reference)' if nside==16384 else '')
            axes[0].loglog(centres,[dl[g].mean() for g in groups],label=label)
            ratio=np.array([(dl[g].mean()/reference[g].mean()-1)*100 for g in groups])
            axes[1].semilogx(centres,ratio,label=label)
        axes[0].set(ylabel=r'$D_\ell^{yy}$',title=anchor);axes[0].legend()
        axes[1].set(xlabel=r'$\ell$',ylabel='Difference [%]');axes[1].axhline(0,color='k',lw=.7)
        for axis in axes:axis.grid(alpha=.2)
        fig.tight_layout()
        for suffix in ['png','pdf']:fig.savefig(ROOT/'plots'/f'resolution_{anchor}.{suffix}',dpi=180)
        plt.close(fig)
        records.append(dict(anchor=anchor,available_nsides=list(variants),
            scope='Existing spherical controls; 100-ell plot averaging only; no production 16384 choice'))
    return records

if __name__=='__main__':main()
