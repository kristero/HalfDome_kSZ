#!/usr/bin/env python3
"""Freeze existing idark inputs and write ten tasks; never submit jobs implicitly.

Run with idark's /home/kristero10/envs/sbi27/bin/python. Only small code, model,
whitening and reference spectra are copied. Catalogues and Julia runtime stay in place.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path('/lustre/work/kristero10/tsz_seed_mask_variation_20261002')
TRAIN = Path('/home/kristero10/sbi_reruns_20261001/SBI_analysis/tsz64k_sbi_local')
STAGE = Path('/home/kristero10/tsz64k_fixbeta_stage')
BUNDLE = 'tSZ_64k_flat_prior_SO_baseline_deproj0'
MODEL = 'results_merlin_unbinned_beam_nnwhite_asinh_p30'
THETA = [18.1, .497, 4.35, .154, -.00865, .0393, -.758, .731, .415]
PARAMS = ['P0_amp', 'x_c_amp', 'beta_amp', 'P0_alpha_m', 'x_c_alpha_m',
          'beta_alpha_m', 'P0_alpha_z', 'x_c_alpha_z', 'beta_alpha_z']


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def prepare(root):
    if (root / 'plan.json').exists():
        raise RuntimeError('This run is already frozen; use its existing tasks, or choose a new root.')
    root.mkdir(parents=True, exist_ok=True)
    for d in ('logs', 'tasks', 'results', 'analysis', 'reference', 'models', 'snapshot'):
        (root / d).mkdir(exist_ok=True)
    bundle_src = STAGE / 'HalfDome_kSZ' / BUNDLE
    bundle_dst = root / 'snapshot' / BUNDLE
    for name in ('engine', 'halfdome_sources'):
        shutil.copytree(bundle_src / name, bundle_dst / name, dirs_exist_ok=True)
    ref = bundle_dst / 'reference/test256'
    ref.mkdir(parents=True, exist_ok=True)
    shutil.copy2(bundle_src / 'reference/test256/base_command.json', ref)
    inference = root / 'snapshot/inference'
    inference.mkdir(exist_ok=True)
    for f in TRAIN.glob('*.py'):
        shutil.copy2(f, inference / f.name)
    for space, suffix in [('logit', ''), ('linear', '_lin')]:
        target = root / 'models' / space
        target.mkdir(exist_ok=True)
        source = TRAIN / (MODEL + suffix)
        for name in ['features_merlin.npz', 'prepare.json', 'metrics_merlin.json'] + [
                f'merlin_seed{s}.{ext}' for s in (1, 2, 3) for ext in ('pt', 'json')]:
            shutil.copy2(source / name, target / name)
    inputs = TRAIN / 'results_merlin_unbinned_beam_nnwhite/inputs'
    for name in ('whitening_pca.npz', 'b12.npz'):
        shutil.copy2(inputs / name, root / 'reference' / name)
    mock = TRAIN / 'mocks/pass1_B12/B12'
    for name in ('masked_clean_cl.npy', 'noisy_cross_39001001_39001002.npy'):
        shutil.copy2(mock / name, root / 'reference' / name)
    beamfix = Path('/home/kristero10/tsz64k_beamfix_20260930')
    shutil.copy2(beamfix / 'b12draws/nn/B12_39001001_39001002.npy', root / 'reference/noise_cl.npy')
    shutil.copy2(beamfix / 'beam_bl.npy', root / 'reference/beam_bl.npy')
    catalogue_dir = Path('/lustre/work/ruizm/halos')
    manifests = {x['filename']: x for x in json.loads((catalogue_dir / 'manifest.json').read_text())}
    # HDF5 metadata only; do not load the 85-million-row arrays during preparation.
    script = """import json,h5py
out={}
for s in range(100,120,2):
 with h5py.File('/lustre/work/ruizm/halos/lightcone_%d.hdf5'%s) as f:
  out[s]=int(f['halo_mass_m200c'].shape[0])
print(json.dumps(out))
"""
    counts = json.loads(subprocess.check_output(
        ['/home/kristero10/.conda/envs/SZ/bin/python', '-c', script], text=True))
    masks = list(range(12345, 12355))
    cases, catalogues = [], []
    for seed in range(100, 120, 2):
        path = catalogue_dir / f'lightcone_{seed}.hdf5'
        record = manifests[path.name]
        assert path.stat().st_size == record['length'], path
        selected_masks = masks if seed == 100 else masks[:1]
        task = dict(out=str(root / f'results/seed_{seed}'), catalogue=str(path),
                    simulation_seed=seed, catalogue_rows=counts[str(seed)],
                    theta=THETA, mask_seeds=selected_masks, noise_seeds=[39001001, 39001002])
        (root / f'tasks/seed_{seed}.toml').write_text('\n'.join(
            f'{key} = {json.dumps(value)}' for key, value in task.items()) + '\n')
        catalogues.append(dict(seed=seed, path=str(path), bytes=path.stat().st_size,
                               rows=counts[str(seed)], manifest_sha512=record['sha512'],
                               sha512_recomputed=False))
        for mask in selected_masks:
            cases.append(dict(label=f's{seed}_m{mask}', simulation_seed=seed, mask_seed=mask,
                              path=f'results/seed_{seed}/mask_{mask}'))
    frozen = {}
    for folder in ('snapshot', 'models', 'reference', 'tasks', 'code'):
        for p in sorted((root / folder).rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts:
                frozen[str(p.relative_to(root))] = sha(p)
    (root / 'input_sha256.json').write_text(json.dumps(frozen, indent=2) + '\n')
    plan = dict(root=str(root), model_source=str(TRAIN / MODEL), primary_space='linear',
                spaces=['linear', 'logit'], baseline='s100_m12345', cases=cases,
                parameters=PARAMS, truth=THETA, noise_seeds=[39001001, 39001002],
                noise_prescription='SO baseline deproj0, full-depth N_ell per split (training convention)',
                noise_fixed=True, simulation_seeds=list(range(100, 120, 2)), mask_seeds=masks,
                catalogues=catalogues, ell_min=80, ell_max=7979, beam_fwhm_arcmin=2.,
                fsky=.4, apodization_arcmin=60., samples_per_posterior=90000,
                posterior_rng_seed=2, raw_nside=8192, output_nside=4096,
                mass_definition='M200c', sphere_R200c=4., mass_min_Msun=1e12,
                resources=dict(paint_jobs=10, analysis_jobs=1, queue='mini', cpus_per_job=26,
                               paint_memory_gb=64, analysis_memory_gb=24, walltime='23:59:00',
                               estimated_map_peak_GiB='25-40', new_disk_GiB_upper_estimate=2,
                               catalogue_bytes_read_once=sum(x['bytes'] for x in catalogues)),
                interpretation='Noise-conditional robustness test. Rotated masks overlap; not independent mocks.')
    (root / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    print(json.dumps(dict(root=str(root), cases=len(cases), frozen_files=len(frozen),
                         resources=plan['resources']), indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=ROOT)
    prepare(p.parse_args().root)
