#!/usr/bin/env python3
"""Metadata preflight and task preparation; heavy work runs only in PBS jobs."""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from common import (ROOT, OLD, BEAMFIX, FISHER, SEEDS, MASKS, THETA, PARAMETERS,
                    DISCOVERY_SEEDS, DISCOVERY_MASKS, VALIDATION_SEEDS,
                    VALIDATION_MASKS, NOISE_SEEDS, recipes, sha, write_json)


def prepare(root):
    if (root / 'plan.json').exists():
        raise RuntimeError('Run already prepared; inspect the existing plan instead of overwriting it')
    for name in ['code', 'tasks', 'logs', 'results', 'data', 'candidates', 'report', 'snapshot']:
        (root / name).mkdir(parents=True, exist_ok=True)
    old_plan = json.loads((OLD / 'plan.json').read_text())
    assert (OLD / 'analysis/COMPLETE.json').exists()
    np.testing.assert_array_equal(old_plan['truth'], THETA)
    assert old_plan['noise_seeds'] == NOISE_SEEDS
    assert FISHER.exists()
    # Freeze the exact inference and physical code used in the previous completed experiment.
    shutil.copytree(OLD / 'snapshot', root / 'snapshot', dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy2(OLD / 'code/paint_seed.jl', root / 'code/paint_seed.jl')
    shutil.copytree(OLD / 'results/seed_100', root / 'results/seed_100', dirs_exist_ok=True)
    write_json(root / 'results/seed_100/VALIDATED.json',
               dict(source=str(OLD / 'results/seed_100'), copied_completed_masks=10,
                    baseline_check=json.loads((OLD / 'analysis/baseline_check.json').read_text())))
    for seed in SEEDS[1:]:
        catalogue = next(c for c in old_plan['catalogues'] if c['seed']==seed)
        p = Path(catalogue['path'])
        assert p.stat().st_size == catalogue['bytes'], p
        task = dict(out=str(root / f'results/seed_{seed}'), catalogue=str(p),
                    simulation_seed=seed, catalogue_rows=catalogue['rows'],
                    theta=THETA.tolist(), mask_seeds=MASKS, noise_seeds=NOISE_SEEDS)
        (root / f'tasks/seed_{seed}.toml').write_text('\n'.join(
            f'{key} = {json.dumps(value)}' for key,value in task.items())+'\n')
    cases = [dict(label=f's{seed}_m{mask}', simulation_seed=seed, mask_seed=mask,
                  path=f'results/seed_{seed}/mask_{mask}') for seed in SEEDS for mask in MASKS]
    # Verify metadata coverage without loading any multipole arrays.
    row_ids = np.load(BEAMFIX / 'inputs/row_id.npy')
    upload_rows, upload_bytes = [], 0
    batches = []
    cached_rows = set()
    for stage in sorted((BEAMFIX / 'stages').glob('stage*')):
        if (stage / 'beam_dl_unbinned.npy').exists():
            cached_rows.update(np.load(stage / 'rows.npy').tolist())
    for stage in sorted((BEAMFIX / 'upload').glob('stage*')):
        for batch in sorted(stage.glob('b*')):
            if not (batch / 'rows.npy').exists():
                continue
            assert (batch / 'UPLOADED').exists(), batch
            assert (batch / 'clean_dl_unbinned.npy').exists(), batch
            ids = np.load(batch / 'rows.npy')
            upload_rows.extend(ids.tolist())
            upload_bytes += (batch / 'clean_dl_unbinned.npy').stat().st_size
            batches.append(str(batch))
    assert set(row_ids).issubset(set(upload_rows))
    # Missing staged beam rows are rebuilt from their OWN production and noise-only spectra.
    nn_names = {p.name for p in (BEAMFIX / 'nn').iterdir() if p.suffix=='.npy'}
    missing_beam = sorted(set(row_ids)-cached_rows)
    assert all(f'r{int(row):05d}.npy' in nn_names for row in missing_beam)
    references = [OLD / 'plan.json', OLD / 'analysis/COMPLETE.json', FISHER,
                  OLD / 'models/linear/features_merlin.npz',
                  OLD / 'reference/whitening_pca.npz', BEAMFIX / 'inputs/row_id.npy',
                  BEAMFIX / 'inputs/theta.npy']
    hashes = {str(p): sha(p) for p in references}
    code_hashes = {str(p.relative_to(root)): sha(p) for p in (root / 'code').iterdir() if p.is_file()}
    write_json(root / 'input_sha256.json', dict(references=hashes, code=code_hashes))
    plan = dict(root=str(root), previous_run=str(OLD), cases=cases, simulation_seeds=SEEDS,
                mask_seeds=MASKS, parameters=PARAMETERS, truth=THETA.tolist(),
                noise_seeds=NOISE_SEEDS, primary_param_space='linear',
                discovery_seeds=DISCOVERY_SEEDS, discovery_masks=DISCOVERY_MASKS,
                validation_seeds=VALIDATION_SEEDS, validation_masks=VALIDATION_MASKS,
                candidates=recipes(), posterior_samples_screen=12000,
                posterior_samples_final=90000, prior_test_samples=600, prior_test_rows=1500,
                training_seeds=[1,2,3], training_rows=61440, training_patience=30,
                learning_rate_patience=10, row_ids_count=len(row_ids),
                cached_beam_rows=len(set(row_ids)&cached_rows), rebuild_beam_rows=len(missing_beam),
                clean_batches=batches, source_clean_bytes=upload_bytes,
                band_width=50, ell_min=80, ell_max=7979,
                constraints_reference='uncut original unbounded Merlin, same observation',
                ranking='Discovery Pareto front in truth bias, agreement and width cost; validate chosen methods on unused seeds AND masks',
                approximation='Sky augmentation uses B12 signal ratios across the prior and keeps SO residuals; exact new-sky noise and general parameter dependence are not simulated',
                resources=dict(paint_jobs=9, pipeline_jobs=1, maximum_submissions=20,
                               cpus=26, paint_mem_gb=64, pipeline_mem_gb=48, walltime='23:59:00',
                               queues=['mini','mini2','mini_B'], estimated_new_disk_GiB=6,
                               estimated_pipeline_peak_GiB=10,
                               saved_maps=False))
    write_json(root / 'plan.json', plan)
    print(json.dumps({k:plan[k] for k in ['root','row_ids_count','cached_beam_rows','rebuild_beam_rows','resources']},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT)
    prepare(p.parse_args().root)
