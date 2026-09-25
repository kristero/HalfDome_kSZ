"""Actual full-catalogue forward checks of saved posterior draws.

Two draws per observation are a diagnostic, not a posterior-predictive credible
band. They are never added to the 256 training/held-out parameter rows.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import toml
from launch import ROOT,atomic_json,launch

RUN=ROOT/'diagnostic_256'


def prepare():
    rng=np.random.default_rng(20260922)
    cases=[];skipped=[]
    for label in ['Battaglia12','L1_m9','fgas-8sigma','Mstar-1sigma']:
        for number,seed in enumerate([20260920,20260921]):
            path=RUN/f'analysis/flamingo_{label}_moped_fixed_{seed}.npy'
            samples=np.load(path)
            if len(samples)!=512:
                skipped.append(dict(observation=label,seed=seed,accepted=len(samples)))
                continue
            index=int(rng.integers(len(samples)))
            cases.append(dict(label=label+'_draw'+str(number),observation=label,
                theta=samples[index].astype(float).tolist(),posterior_seed=seed,posterior_index=index,
                seeds=[int(v) for v in rng.integers(1,2**31-1,size=2)]))
    atomic_json(ROOT/'forward_plan.json',dict(cases=cases,skipped=skipped,
        selection='One reproducible random draw from each of two independently trained MOPED posteriors'))
    if cases:
        from dispatch import submit
        job=submit('tsz256_forward','forward.pbs',queue='mini2')
        atomic_json(ROOT/'forward_submission.json',dict(job=job,cases=len(cases)))
    else:
        atomic_json(RUN/'analysis/forward_summary.json',dict(completed=0,skipped=skipped,
            limitation='No complete observation posterior; actual forward checks could not run'))
        from report import main
        main()


def run():
    from diagnostic_analysis import project
    from manage import verify_production_sources
    verify_production_sources()
    plan=json.loads((ROOT/'forward_plan.json').read_text());cases=plan['cases']
    launch(dict(mode='audit',cases=cases),'forward/audit',26)
    for case in cases:
        folder=ROOT/'forward/audit'/case['label'];record=toml.load(folder/'audit.toml')
        if not record['accepted']:raise RuntimeError('Posterior draw needs numerical refinement: '+case['label'])
        case['nodes']=record['attempts'][-1]['nodes']
        probes=np.loadtxt(folder/f"probes_{case['nodes'][0]}.csv",delimiter=',').reshape(64,39,6)
        for halo in probes:
            x=halo[:,2]
            error=np.trapz(abs(halo[:,4]-halo[:,3])*x,x)/np.trapz(halo[:,3]*x,x)
            assert np.isfinite(error) and error<=.004
    ell=np.arange(80,7980);factor=ell*(ell+1)/(2*np.pi)
    output={'ell':ell};results=[]
    for start in range(0,len(cases),4):
        group=cases[start:start+4];relative=f'forward/batches/{start//4:03d}'
        launch(dict(mode='maps',cases=group,output_nsides=[4096]),relative,26)
        for case in group:
            folder=ROOT/relative/case['label']
            clean=np.load(folder/'masked_clean_cl.npy')[ell]*factor
            noisy=np.load(folder/'masked_noisy_cross_cl.npy')[ell]*factor
            observation=np.load(RUN/'observations'/(case['observation']+'.npz'))
            with np.load(RUN/f"analysis/transform_moped_fixed_192.npz") as f:
                state={k:f[k] for k in f.files}
            discrepancy=project(noisy[None,:],state)-project(observation['noisy_dl_unbinned'][None,:],state)
            results.append(dict(label=case['label'],observation=case['observation'],theta=case['theta'],
                posterior_seed=case['posterior_seed'],posterior_index=case['posterior_index'],
                clean_spectrum_relative_l2=float(np.linalg.norm(clean-observation['clean_dl_unbinned'])/
                    np.linalg.norm(observation['clean_dl_unbinned'])),
                trained_feature_distance=float(np.linalg.norm(discrepancy)),nodes=case['nodes']))
            output[case['label']+'_clean']=clean;output[case['label']+'_noisy']=noisy
    np.savez_compressed(RUN/'analysis/forward_spectra.npz',**output)
    atomic_json(RUN/'analysis/forward_summary.json',dict(completed=len(results),cases=results,
        skipped=plan['skipped'],limitation='Two actual posterior-draw forward checks per observation; not a credible band or a calibrated PPC p-value'))
    from report import main
    main()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','run'])
    args=parser.parse_args()
    prepare() if args.action=='prepare' else run()
