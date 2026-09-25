"""Prepare (never submit) the 256-row pilot with the validated 8192 operator."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'tsz_spherical_preflight_20260920'
VALIDATED=ROOT.parent/'tsz_8192_validation_recovery_20260921'
NEW=ROOT.parent/'tsz_diagnostic_256_8192_prepared_20260921'
REMOTE='/lustre/work/kristero10/'+NEW.name


def replace_once(text,old,new):
    assert text.count(old)==1,old
    return text.replace(old,new)


def main():
    NEW.mkdir(exist_ok=True)
    if (NEW/'diagnostic_256/rows').exists():
        raise RuntimeError('Refusing to regenerate a started preparation')
    for name in ['design_tests.py','run_fullsky.py','diagnostic_analysis.py',
                 'unbinned_moped.py','stage_observations.py','test_unbinned_moped.py']:
        shutil.copy2(OLD/name,NEW/name)
    for name in ['fullsky_test.jl','spherical_truncation_profiles.jl']:
        shutil.copy2(VALIDATED/name,NEW/name)
    text=(OLD/'diagnostic.py').read_text()
    text=text.replace("nside='4096'","nside='8192'")
    text=text.replace('raw_nside=4096,output_nside=4096','raw_nside=8192,output_nside=4096')
    text=text.replace('raw/output4096','raw8192/output4096')
    text=replace_once(text,"        if previous['returncode']==0:return", "        if previous['returncode']==0:return\n        raise RuntimeError('Previously failed row requires inspection before retry')")
    text=text.replace("PREFLIGHT_OUTPUT=str(out),PREFLIGHT_GRID_FACTOR='1',", "PREFLIGHT_OUTPUT=str(out),PREFLIGHT_GRID_FACTOR='1',VALIDATION_LEAN_MAPS='1',")
    start=text.index('def run_row(');end=text.index('\ndef _summarize(',start)
    text=text[:start]+'''def run_row(row,theta,threads):
    # mkdir is exclusive across Lustre clients, unlike node-local flock.
    claims=RUN/'claims';claims.mkdir(parents=True,exist_ok=True)
    claim=claims/f'{row:05d}'
    claim.mkdir()
    try:
        _run_row(row,theta,threads)
        status=json.loads((RUN/'rows'/f'{row:05d}'/'status.json').read_text())
        if status['returncode']!=0:
            raise RuntimeError('Assigned row failed; retained original parameters: '+str(row))
    finally:
        claim.rmdir()

'''+text[end:]
    start=text.index('def summarize(');end=text.index('\ndef prepare(',start)
    text=text[:start]+'''def summarize(count):
    # Called once after all disjoint workers finish, never from each worker.
    RUN.mkdir(parents=True,exist_ok=True)
    claim=RUN/'summary_claim'
    claim.mkdir()
    try:
        _summarize(count)
        result=json.loads((RUN/'summary.json').read_text())
        if result['completed']!=count:
            raise RuntimeError('Dataset incomplete; do not train on the successful subset')
    finally:
        claim.rmdir()

'''+text[end:]
    text=replace_once(text,'            summarize(args.count)\n    summarize(args.count)',
                           '    if args.summarize_only:\n        summarize(args.count)')
    text=replace_once(text,'    deadline=datetime.fromisoformat',
        "    for name,expected in manifest.get('design_sha256',{}).items():\n        if hashlib.sha256((RUN/name).read_bytes()).hexdigest()!=expected:\n            raise RuntimeError('Prepared design changed: '+name)\n    deadline=datetime.fromisoformat")
    text=replace_once(text,"['diagnostic.py','diagnostic_row.jl','fullsky_test.jl','spherical_truncation_profiles.jl','design_tests.py']",
        "['diagnostic.py','diagnostic_row.jl','fullsky_test.jl','spherical_truncation_profiles.jl','design_tests.py','run_fullsky.py','diagnostic_analysis.py','unbinned_moped.py']")
    (NEW/'diagnostic.py').write_text(text)
    row=(OLD/'diagnostic_row.jl').read_text().replace('cfg.base_cfg.nside==4096','cfg.base_cfg.nside==8192')
    control=(VALIDATED/'control.jl').read_text()
    allocation=control[control.index('if get(ENV,"VALIDATION_LEAN_MAPS"'):control.index('\nfunction measured_signal')]
    row=replace_once(row,'include(joinpath(@__DIR__,"fullsky_test.jl"))',
                         'include(joinpath(@__DIR__,"fullsky_test.jl"))\n\n'+allocation)
    row=replace_once(row,'\nnoisy_diagnostic()\n', '\nget(ENV,"DIAGNOSTIC_LOAD_ONLY","0")=="1" || noisy_diagnostic()\n')
    (NEW/'diagnostic_row.jl').write_text(row)
    run=NEW/'diagnostic_256';run.mkdir(exist_ok=True)
    for name in ['theta_design.npy','noise_seeds.npy','heldout_design.npy']:
        shutil.copy2(OLD/'diagnostic_256'/name,run/name)
    shutil.copytree(OLD/'diagnostic_256/observations',run/'observations',dirs_exist_ok=True)
    manifest=json.loads((OLD/'diagnostic_256/manifest.json').read_text())
    manifest.update(raw_nside=8192,lean_maps=True,submission_authorized=False,
                    planned_workers=4,application_threads=26,
                    preparation_note='Same saved design/seeds; validated spherical sources; atomic claims; one final collector')
    sources=['diagnostic.py','diagnostic_row.jl','fullsky_test.jl','spherical_truncation_profiles.jl',
             'design_tests.py','run_fullsky.py','diagnostic_analysis.py','unbinned_moped.py']
    manifest['source_sha256']={name:hashlib.sha256((NEW/name).read_bytes()).hexdigest() for name in sources}
    manifest['design_sha256']={name:hashlib.sha256((run/name).read_bytes()).hexdigest()
                                for name in ['theta_design.npy','noise_seeds.npy','heldout_design.npy']}
    manifest['design_sha256'].update({p.relative_to(run).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                                      for p in (run/'observations').iterdir() if p.is_file()})
    (run/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    header='''#!/bin/bash
#PBS -q mini
#PBS -l select=1:ncpus=26:mem=64gb
#PBS -l walltime=23:59:00
#PBS -j oe
set -euo pipefail
'''
    worker=header+f'''cd {REMOTE}
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=26 MKL_NUM_THREADS=1
: "${{DIAGNOSTIC_WORKER:?Set one of 0,1,2,3}}"
/home/anaconda3/bin/python3 diagnostic.py --count 256 --threads 26 --worker "$DIAGNOSTIC_WORKER" --workers 4
'''
    analysis=header+f'''cd {REMOTE}
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=26 MKL_NUM_THREADS=26 MPLBACKEND=Agg
/home/anaconda3/bin/python3 diagnostic.py --count 256 --summarize-only
/home/anaconda3/bin/python3 diagnostic_analysis.py --run-root {REMOTE}/diagnostic_256 --train --threads 26 --seeds 20260920 20260921
'''
    for name,value in [('diagnostic.pbs',worker),('diagnostic_analysis.pbs',analysis)]:
        (NEW/name).write_bytes(value.encode())
    (NEW/'README.md').write_text('''# Prepared 8192 diagnostic - not submitted

256 saved independent-uniform Sobol rows, unchanged parameter values and split
noise seeds; 192 training-pool and 64 held-out rows. Raw NSIDE8192 is smoothed
with the 2 arcmin beam and synthesized to output NSIDE4096, with the original
fsky=0.4 mask. The pressure support is a 4 R200 sphere.

MOPED reads all 7900 individual multipoles at ell=80..7979. Separate 40-bin
and PCA runs are comparison baselines. No lower-ell cutoff has been adopted.

Numerical profile files are byte-identical to the completed 8192 validation.
The lean allocation block is copied from the validated control.jl. Four
workers own disjoint row IDs modulo four; atomic mkdir claims guard against
duplicate writers across Lustre clients. Only the final analysis job collects
the full dataset. Failed rows retain their identities and stop that worker;
the collector refuses incomplete data. No failed draw is replaced.

The scripts remain unsubmitted. Generation requires four separately assigned
DIAGNOSTIC_WORKER values and analysis after all four jobs succeed. The existing
256 design has not been relabelled as scientifically certified. Numerical
accuracy across the entire prior and full-catalogue pixel integration remain
open questions for a diagnostic/validation sample.
''')
    print(NEW)


if __name__=='__main__':
    main()
