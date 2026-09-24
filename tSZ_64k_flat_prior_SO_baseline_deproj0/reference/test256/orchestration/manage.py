"""Prepare validated batches, run disjoint workers, and collect every saved draw.

Commands never redraw failed parameters. The count is inferred from the saved
design; --workers and --batch-size control scheduling, not the physical prior.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import numpy as np
import toml
from launch import ROOT,atomic_json,launch,verify_sources
from design_tests import LOW,HIGH,NAMES

RUN=ROOT/'diagnostic_256'


def atomic_json(path,value):
    temporary=path.with_name(path.name+'.'+str(os.getpid())+'.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n')
    temporary.replace(path)


def audits():
    theta=np.load(RUN/'theta_design.npy')
    records=[]
    for i in range(len(theta)):
        path=ROOT/f'tests/audit{i//64}/{i:05d}/audit.toml'
        record=toml.load(path)
        if not record['accepted']:
            raise RuntimeError('Direct LOS accuracy target unresolved for row '+str(i))
        nodes=record['attempts'][-1]['nodes']
        probes=np.loadtxt(path.parent/f'probes_{nodes[0]}.csv',delimiter=',').reshape(64,39,6)
        flux_errors=[]
        for halo in probes:
            x=halo[:,2];direct=halo[:,3];estimate=halo[:,4]
            # The radial area measure is x dx. Include the faint outskirts:
            # a tail can matter for integrated Y despite low central contrast.
            denominator=np.trapz(direct*x,x)
            error=np.trapz(abs(estimate-direct)*x,x)/denominator
            if not np.isfinite(error) or error>.004:
                raise RuntimeError('Area-weighted cache accuracy unresolved for row '+str(i))
            flux_errors.append(float(error))
        record['max_sampled_radial_L1_relative_error']=max(flux_errors)
        records.append(record)
    return records


def prepare_spots():
    records=audits()
    theta=np.load(RUN/'theta_design.npy')
    # Test the largest remaining cache discrepancies and extreme integrated
    # amplitude proxies. Selection is numerical; no draw is removed.
    chosen=list(np.argsort([a['attempts'][-1]['max_relative_visible'] for a in records])[-2:])
    sums=[]
    for i,record in enumerate(records):
        nodes=record['attempts'][-1]['nodes']
        path=ROOT/f'tests/audit{i//64}/{i:05d}/probes_{nodes[0]}.csv'
        values=np.loadtxt(path,delimiter=',')
        sums.append(float(np.linalg.norm(values[:,3])))
    for i in [int(np.argmin(sums)),int(np.argmax(sums))]+list(np.argsort(sums)[::-1]):
        if i not in chosen:chosen.append(i)
        if len(chosen)==4:break
    for group in range(2):
        cases=[]
        for i in chosen[group*2:group*2+2]:
            nodes=records[i]['attempts'][-1]['nodes']
            for suffix,grid in [('candidate',nodes),('reference',[2*n for n in nodes])]:
                cases.append(dict(label=f'{i:05d}_{suffix}',theta=theta[i].tolist(),nodes=grid,row=int(i)))
        atomic_json(ROOT/f'tasks/spot{group}.json',dict(mode='maps',cases=cases,output_nsides=[4096]))
    atomic_json(ROOT/'results/spot_selection.json',dict(rows=[int(i) for i in chosen],
        rule='Two largest post-refinement probe errors plus low/high direct-column norm'))


def prepare_batches(workers,batch_size):
    verify_sources()
    gate=json.loads((ROOT/'results/final_gate.json').read_text())
    if not gate['passed'] or gate['output_nside']!=4096:
        raise RuntimeError('The complete accuracy gate must pass before batching')
    if not json.loads((ROOT/'results/training_software_test.json').read_text())['passed']:
        raise RuntimeError('SBI API smoke test did not pass')
    records=audits()
    theta=np.load(RUN/'theta_design.npy');seeds=np.load(RUN/'noise_seeds.npy')
    old=json.loads((RUN/'manifest.json').read_text())
    for name,expected in old['design_sha256'].items():
        assert hashlib.sha256((RUN/name).read_bytes()).hexdigest()==expected,name
    if (RUN/'batches').exists():
        raise RuntimeError('Batch plan already exists; inspect rather than replace it')
    (RUN/'batches').mkdir()
    files=['manage.py','dispatch.py','diagnostic_analysis.py','unbinned_moped.py',
        'forward_checks.py','report.py','design_tests.py','final_gate.py',
        'production.pbs','analysis.pbs','forward.pbs','reference/compression_context.npz',
        'reference/raw_resolution_spectra.npz']
    atomic_json(ROOT/'production_source_manifest.json',dict(sha256={
        name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in files}))
    batches=[]
    for start in range(0,len(theta),batch_size):
        cases=[]
        for i in range(start,min(start+batch_size,len(theta))):
            cases.append(dict(label=f'{i:05d}',row=i,theta=theta[i].tolist(),
                nodes=records[i]['attempts'][-1]['nodes'],seeds=seeds[i].tolist()))
        batches.append(dict(id=len(batches),mode='maps',cases=cases,output_nsides=[4096]))
    atomic_json(RUN/'batch_plan.json',dict(workers=workers,batch_size=batch_size,batches=batches))
    shutil.copy2(RUN/'manifest.json',RUN/'manifest.prepared.json')
    old.update(submission_authorized=True,cache_grid_nodes='Per-row directly validated grid',
        cache_nodes_per_row=[a['attempts'][-1]['nodes'] for a in records],
        painting='NSIDE8192 point samples; smooth chord-mean cache; fused geometry; greedy256',
        final_gate_sha256=hashlib.sha256((ROOT/'results/final_gate.json').read_bytes()).hexdigest(),
        production_certified=False,workers=workers,batch_size=batch_size,
        known_limitations=['Finite point-sampling resolution; see measured 8192/16384 errors',
                          '256 rows do not establish precise nine-parameter posterior coverage'])
    atomic_json(RUN/'manifest.json',old)


def worker(number):
    verify_production_sources()
    plan=json.loads((RUN/'batch_plan.json').read_text())
    assert 0<=number<plan['workers']
    # Prefer disjoint work first, then claim any unstarted batch. Available
    # workers can finish the dataset even if other jobs remain in the queue.
    order=plan['batches'][number::plan['workers']]+[
        task for task in plan['batches'] if task['id']%plan['workers']!=number]
    for task in order:
        relative=f"diagnostic_256/batches/{task['id']:03d}"
        folder=ROOT/relative
        if (folder/'claim').exists():continue
        try:launch(task,relative,26)
        except FileExistsError:continue  # Another worker won the atomic mkdir.
        timing=toml.load(folder/'batch.toml')
        for case in task['cases']:
            source=folder/case['label'];target=RUN/'rows'/case['label']
            target.parent.mkdir(exist_ok=True)
            try:target.symlink_to(source, target_is_directory=True)
            except FileExistsError:
                if target.resolve()!=source.resolve():raise RuntimeError('Row path collision')
            status=dict(row=case['row'],returncode=0,batch=task['id'],theta=case['theta'],
                nodes=case['nodes'],split_seeds=case['seeds'],selected_halos=timing['selected_halos'],
                batch_status=json.loads((folder/'status.json').read_text()))
            status['sha256']={f:hashlib.sha256((source/f).read_bytes()).hexdigest() for f in
                ['masked_clean_cl.npy','masked_noisy_cross_cl.npy','unmasked_clean_cl.npy','observation.toml']}
            atomic_json(source/'status.json',status)
    count=len(np.load(RUN/'theta_design.npy'))
    if all((RUN/'rows'/f'{i:05d}'/'status.json').exists() for i in range(count)):
        claim=RUN/'analysis_submission_claim'
        try:claim.mkdir()
        except FileExistsError:return
        # Dataset completeness, rather than completion of idle queued workers,
        # starts analysis. Only this campaign's unused QUEUED workers are removed.
        from dispatch import submit
        job=submit('tsz256_analysis','analysis.pbs')
        atomic_json(ROOT/'analysis_submission.json',dict(job=job,trigger='All saved row statuses present'))
        jobs=json.loads((ROOT/'production_submission.json').read_text())
        cancelled=[]
        for name,identifier in jobs.items():
            if not name.startswith('worker') or identifier==os.environ.get('PBS_JOBID'):continue
            q=subprocess.run(['qstat','-f',identifier],text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            if q.returncode==0 and 'job_state = Q' in q.stdout and 'Job_Name = tsz256_w' in q.stdout:
                subprocess.run(['qdel',identifier],check=True);cancelled.append(identifier)
        atomic_json(ROOT/'unused_worker_cleanup.json',dict(cancelled_queued_jobs=cancelled))


def collect():
    verify_production_sources()
    from diagnostic_analysis import rebin_unbinned
    theta=np.load(RUN/'theta_design.npy');seeds=np.load(RUN/'noise_seeds.npy')
    ell=np.arange(80,7980);factor=ell*(ell+1)/(2*np.pi)
    rows=[];clean=[];noisy=[];noise_hashes=[];masks=[]
    for i in range(len(theta)):
        folder=RUN/'rows'/f'{i:05d}'
        status=json.loads((folder/'status.json').read_text())
        assert status['returncode']==0 and status['row']==i
        np.testing.assert_array_equal(status['theta'],theta[i])
        np.testing.assert_array_equal(status['split_seeds'],seeds[i])
        assert status['selected_halos']==85224251
        for name,expected in status['sha256'].items():
            assert hashlib.sha256((folder/name).read_bytes()).hexdigest()==expected
        metadata=toml.load(folder/'observation.toml')
        assert metadata['output_nside']==4096 and metadata['ell_max']==7979
        assert metadata['noise_beam_applied'] is False and metadata['split_Nell_multiplier']==1.
        np.testing.assert_array_equal(metadata['split_seeds'],seeds[i])
        noise_hashes.extend(metadata['noise_sha256']);masks.append(metadata['mask_sha256'])
        for filename,target in [('masked_clean_cl.npy',clean),('masked_noisy_cross_cl.npy',noisy)]:
            cl=np.load(folder/filename)
            assert cl.shape==(7980,) and np.isfinite(cl).all()
            if filename=='masked_clean_cl.npy':assert np.all(cl>=0)
            target.append(cl[ell]*factor)
        rows.append(status)
    assert len(set(noise_hashes))==2*len(theta),'Repeated SO noise realization'
    assert len(set(masks))==1 and masks[0]=='a6c3d64d5ab83e79b6d9b76cccbaf66a708c3e68adce85b9a0d3f610d4100689'
    clean=np.asarray(clean);noisy=np.asarray(noisy)
    with (RUN/'dataset.tmp').open('wb') as stream:
        np.savez_compressed(stream,row_id=np.arange(len(theta)),theta=theta,ell_unbinned=ell,
            clean_dl_unbinned=clean,noisy_dl_unbinned=noisy,
            clean_dl=rebin_unbinned(clean,ell),noisy_dl=rebin_unbinned(noisy,ell),
            lower=LOW,upper=HIGH,parameter_order=NAMES,noise_seeds=seeds)
    (RUN/'dataset.tmp').replace(RUN/'dataset.npz')
    # Independent B12 control is not added to the training design.
    anchor=ROOT/'tests/anchors/Battaglia12/output4096'
    clean_b12=np.load(anchor/'masked_clean_cl.npy')[ell]*factor
    noisy_b12=np.load(anchor/'masked_noisy_cross_cl.npy')[ell]*factor
    np.savez(RUN/'observations/Battaglia12.npz',ell_unbinned=ell,
        clean_dl_unbinned=clean_b12,noisy_dl_unbinned=noisy_b12,
        clean_dl=rebin_unbinned(clean_b12[None,:],ell)[0],noisy_dl=rebin_unbinned(noisy_b12[None,:],ell)[0])
    # Probe the TRAINED network's response to a measured clean resolution error.
    # The stochastic residual is deliberately held fixed: these are sensitivity
    # experiments, not exact maps with matched signal-noise cross terms.
    resolution=np.load(ROOT/'reference/raw_resolution_spectra.npz')
    for nside in [4096,16384]:
        delta=resolution['raw'+str(nside)]-resolution['raw8192']
        clean_shift=clean_b12+delta;noisy_shift=noisy_b12+delta
        np.savez(RUN/f'observations/Battaglia12_raw{nside}_sensitivity.npz',ell_unbinned=ell,
            clean_dl_unbinned=clean_shift,noisy_dl_unbinned=noisy_shift,
            clean_dl=rebin_unbinned(clean_shift[None,:],ell)[0],
            noisy_dl=rebin_unbinned(noisy_shift[None,:],ell)[0])
    atomic_json(RUN/'observations/resolution_sensitivity.json',dict(
        construction='Add the measured clean raw-resolution difference to the same B12 noisy observation',
        qualification='Noise residual held fixed; not an exact matched-noise map comparison',
        source_sha256=hashlib.sha256((ROOT/'reference/raw_resolution_spectra.npz').read_bytes()).hexdigest()))
    atomic_json(RUN/'summary.json',dict(requested=len(theta),completed=len(rows),rows=rows,
        independent_noise_hashes=len(set(noise_hashes)),mask_sha256=masks[0],
        dataset_sha256=hashlib.sha256((RUN/'dataset.npz').read_bytes()).hexdigest()))


def verify_production_sources():
    verify_sources()
    manifest=json.loads((ROOT/'production_source_manifest.json').read_text())
    for name,expected in manifest['sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=expected:
            raise RuntimeError('Production source changed after preparation: '+name)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['spots','prepare','worker','collect'])
    parser.add_argument('--workers',type=int,default=6);parser.add_argument('--batch-size',type=int,default=4)
    parser.add_argument('--worker',type=int,default=0);args=parser.parse_args()
    if args.action=='spots':prepare_spots()
    elif args.action=='prepare':prepare_batches(args.workers,args.batch_size)
    elif args.action=='worker':worker(args.worker)
    else:collect()
