#!/usr/bin/env python3
"""Evaluate frozen Merlin models on the same 1,500 reserved parameter cases.

Run on an idark compute node. Linear and logit models use fresh interpreters
because the frozen loader patches parameter transforms when it is imported.
This script never trains a model or changes the earlier method selection.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from diagnostic_metrics import PARAMETERS, summarize

PARENT = Path('/lustre/work/kristero10/tsz_robustness_methods_20261004')
OLD = Path('/lustre/work/kristero10/tsz_seed_mask_variation_20261002')
ROOT = Path('/lustre/work/kristero10/tsz_truth_mean_validation_20261006')
METHODS = ['original', 'band50', 'lmin580']


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    def encode(item):
        if isinstance(item, np.ndarray): return item.tolist()
        if isinstance(item, np.generic): return item.item()
        if isinstance(item, Path): return str(item)
        raise TypeError(type(item))
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=encode, allow_nan=False)+'\n')
    temporary.replace(path)


def folder_for(method, space):
    if method == 'original':
        return OLD/'models'/space
    name = 'merlin_band50_full' if method == 'band50' else 'merlin_lmin580'
    folder = PARENT/'comparison/candidates'/name
    return folder/'bounded' if space == 'logit' else folder


def reference():
    """Odd held-row positions were reserved from method screening in the old study."""
    with np.load(PARENT/'data/design.npz') as design:
        positions = np.arange(len(design['held']))[1::2]
        assert len(positions) == 1500
        ids = design['held'][positions]
        return positions, ids, design['theta'][ids], design['low'], design['high']


def check_features(features, positions, ids, truth, low, high):
    np.testing.assert_array_equal(features['held'][positions], ids)
    np.testing.assert_array_equal(features['theta_held'][positions], truth)
    np.testing.assert_array_equal(features['low'], low)
    np.testing.assert_array_equal(features['high'], high)
    x = features['x_held'][positions]
    assert x.shape[0] == 1500 and np.isfinite(x).all()
    return x


def verify_frozen_inputs(root):
    """Compare known model/loader inputs with the earlier verified study hashes."""
    expected = json.loads((root/'evidence/frozen_input_hashes.json').read_text())['inputs']
    for filename, checksum in expected.items():
        if sha(filename) != checksum:
            raise RuntimeError(f'Frozen comparison input changed: {filename}')
    print('Verified', len(expected), 'previously frozen inputs', flush=True)
    return expected


def evaluate_method(args, method, space, reference_data):
    import torch
    import merlin_compression as mc
    import train_ensemble as te
    positions, ids, truth, low, high = reference_data
    folder = folder_for(method, space)
    ens = mc.ensemble(folder, 'merlin', args.threads)
    assert all(not net.training for net in ens.nets)
    x = check_features(ens.f, positions, ids, truth, low, high)
    feature_error = None
    if method != 'original':
        # Reconstruct every tested banded feature from its saved noisy spectrum,
        # not just its theta label. Use the frozen training-only transform.
        noisy = np.load(PARENT/'data/noisy_bands.npy', mmap_mode='r')
        with np.load(folder_for(method, 'linear')/'compression.npz') as c:
            projected = (noisy[ids]/c['fid'])@c['weights']
            reproduced = ((np.arcsinh((projected-c['s_center'])/c['s_scale'])
                           - c['z_mean'])/c['z_sd']).astype(np.float32)
        feature_error = float(np.max(np.abs(reproduced-x)))
        np.testing.assert_allclose(reproduced, x, rtol=1e-6, atol=1e-5)
        del noisy
    assert args.samples % len(ens.nets) == 0
    assert args.corner_samples % len(ens.nets) == 0
    if space == 'logit':
        # The bounded companion must receive exactly the same observation.
        with np.load(folder_for(method, 'linear')/'features_merlin.npz') as f:
            np.testing.assert_array_equal(x, f['x_held'][positions])
    mean = np.empty_like(truth, dtype=np.float64)
    std = np.empty_like(mean)
    mean_mcse = np.empty_like(mean)
    half_mean = np.empty((2, *truth.shape))
    quantiles = np.empty((5, *truth.shape))
    outside = np.empty(len(truth))
    truth_log_prob = np.empty(len(truth))
    model_number = METHODS.index(method) + (3 if space == 'logit' else 0)
    torch.manual_seed(202610060+model_number)
    rng = np.random.default_rng(202610060+model_number)
    started = time.time()
    for begin in range(0, len(x), args.chunk):
        end = min(begin+args.chunk, len(x))
        block_x = x[begin:end]
        u = ens.sample_u(block_x, args.samples, rng=rng)
        draws = te.from_u(u, low, high)
        assert draws.shape == (args.samples, end-begin, 9)
        assert np.isfinite(draws).all()
        mean[begin:end] = draws.mean(axis=0, dtype=np.float64)
        std[begin:end] = draws.std(axis=0, ddof=1, dtype=np.float64)
        # Pooled standard error is conservative for a stratified equal-weight
        # ensemble. These are sampling errors, not model/calibration errors.
        mean_mcse[begin:end] = std[begin:end]/np.sqrt(args.samples)
        half = args.samples//2
        half_mean[0, begin:end] = draws[:half].mean(0, dtype=np.float64)
        half_mean[1, begin:end] = draws[half:].mean(0, dtype=np.float64)
        quantiles[:, begin:end] = np.quantile(draws, [.025, .16, .5, .84, .975], axis=0)
        inside = np.all((draws >= low) & (draws <= high), axis=-1)
        outside[begin:end] = 1-inside.mean(0)
        true_u = te.to_u(truth[begin:end], low, high)
        truth_log_prob[begin:end] = (
            ens.log_prob_u(true_u[None], block_x)[0]
            - te.log_jacobian_theta_per_u(true_u, low, high))
        print(space, method, end, '/', len(x), 'seconds', round(time.time()-started), flush=True)
        del u, draws
    if space == 'logit':
        assert np.max(outside) == 0
    output = args.root/'data'/f'{method}_{space}.npz'
    np.savez_compressed(output, truth=truth, mean=mean, std=std, mean_mcse=mean_mcse,
                        half_mean=half_mean, quantiles=quantiles, outside_prior=outside,
                        truth_log_prob=truth_log_prob, low=low, high=high,
                        held_positions=positions, design_ids=ids)
    record = summarize(truth, mean, quantiles, low, high, mean_mcse, half_mean)
    record.update(method=method, space=space, cases=len(truth), posterior_draws=args.samples,
                  all_test_banded_features_max_abs_error=feature_error,
                  mean_log_prob_truth=truth_log_prob.mean(),
                  mean_outside_any_prior_fraction=outside.mean(), seconds=time.time()-started,
                  source_sha256={str(p): sha(p) for p in
                      [folder/'features_merlin.npz'] + sorted(folder.glob('merlin_seed*.pt'))
                      + sorted(folder.glob('merlin_seed*.json'))})
    if method != 'original':
        compression = folder_for(method, 'linear')/'compression.npz'
        record['source_sha256'][str(compression)] = sha(compression)
    write_json(args.root/f'metrics_{method}_{space}.json', record)
    return ens, x


def select_examples(root):
    """Two seeded-random cases plus the worst original linear mean-error case.

    Random cases are chosen without looking at the new model. The stress case
    is explicitly selected for the original's error, not presented as typical.
    Neither selection changes the aggregate 1,500-row evaluation.
    """
    with np.load(root/'data/original_linear.npz') as d:
        score = np.sqrt(np.mean(((d['mean']-d['truth'])/(d['high']-d['low']))**2, axis=1))
        worst = int(np.argmax(score))
        available = np.setdiff1d(np.arange(len(score)), [worst])
        random_cases = np.random.default_rng(20261006).choice(available, 2, replace=False)
        indices = [*map(int, random_cases), worst]
        records = [dict(index=i, held_position=int(d['held_positions'][i]),
                        design_id=int(d['design_ids'][i]), truth=d['truth'][i],
                        original_normalized_rmse=float(score[i]),
                        selection='seeded random' if k < 2 else 'largest original linear mean error')
                   for k, i in enumerate(indices)]
    write_json(root/'corner_examples.json', dict(seed=20261006, cases=records))
    return indices


def run_space(args):
    os.environ['MERLIN_PARAM_SPACE'] = args.space
    sys.path.insert(0, str(PARENT/'snapshot/inference'))
    from threadpoolctl import threadpool_limits
    threadpool_limits(args.threads)
    reference_data = reference()
    models = ['original', 'band50', 'lmin580'] if args.space == 'linear' else ['original', 'band50']
    for method in models:
        evaluate_method(args, method, args.space, reference_data)
    if args.space == 'linear':
        indices = select_examples(args.root)
    else:
        indices = [row['index'] for row in json.loads((args.root/'corner_examples.json').read_text())['cases']]
    import torch
    import merlin_compression as mc
    import train_ensemble as te
    for k, method in enumerate(['original', 'band50']):
        ens = mc.ensemble(folder_for(method, args.space), 'merlin', args.threads)
        positions, _, truth, low, high = reference_data
        x = ens.f['x_held'][positions[indices]]
        torch.manual_seed(6300+k+(2 if args.space == 'logit' else 0))
        u = ens.sample_u(x, args.corner_samples, rng=np.random.default_rng(6300+k))
        draws = te.from_u(u, low, high)
        assert np.isfinite(draws).all()
        np.savez_compressed(args.root/'data'/f'corners_{method}_{args.space}.npz',
                            draws=draws.astype(np.float32), truth=truth[indices], indices=indices)


def main(args):
    if args.root.resolve() != ROOT:
        raise ValueError(f'This frozen experiment writes only to its new root: {ROOT}')
    (args.root/'data').mkdir(exist_ok=True)
    if args.worker:
        run_space(args)
        return
    start = time.time()
    expected = verify_frozen_inputs(args.root)
    # The earlier preparation reproduced original Merlin inputs from the same
    # beam-corrected noisy spectra on an audited subset. Preserve that chain.
    training_audit_path = PARENT/'data/training_complete.json'
    training_audit = json.loads(training_audit_path.read_text())
    assert training_audit['saved_merlin_feature_audit_rows'] >= 32
    assert training_audit['saved_merlin_feature_max_error'] < 1e-4
    for name, checksum in training_audit['sha256'].items():
        assert sha(PARENT/'data'/name) == checksum, f'Training spectrum input changed: {name}'
    for space in ['linear', 'logit']:
        env = os.environ.copy()
        env['MERLIN_PARAM_SPACE'] = space
        env.pop('MERLIN_TRAIN_ROWS', None)
        subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker', '--space', space,
                        '--root', str(args.root), '--threads', str(args.threads),
                        '--samples', str(args.samples), '--corner-samples', str(args.corner_samples),
                        '--chunk', str(args.chunk)], env=env, check=True)
    assert verify_frozen_inputs(args.root) == expected
    inputs = [PARENT/'data/design.npz', PARENT/'data/noisy_bands.npy',
              training_audit_path, PARENT/'comparison/COMPLETE.json']
    inputs += sorted((PARENT/'snapshot/inference').glob('*.py'))
    write_json(args.root/'INFERENCE_COMPLETE.json', dict(
        finished_utc=datetime.now(timezone.utc).isoformat(), seconds=time.time()-start,
        cases=1500, parameter_order=PARAMETERS,
        posterior_draws_per_case=args.samples, corner_draws_per_case=args.corner_samples,
        previous_study_inputs_verified=expected,
        original_feature_parity_from_prior_preparation=training_audit,
        test_scope='Odd reserved held rows, fixed simulation100/mask12345; varying theta and SO noise',
        frozen_inputs={str(p): sha(p) for p in inputs},
        source={p.name: sha(p) for p in args.root.glob('*.py')},
        outputs={str(p.relative_to(args.root)): sha(p) for p in
                 sorted((args.root/'data').glob('*.npz'))+sorted(args.root.glob('metrics_*.json'))
                 +[args.root/'corner_examples.json']}))
    subprocess.run([sys.executable, str(args.root/'plot_truth_mean.py'), '--root', str(args.root)], check=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--threads', type=int, default=26)
    p.add_argument('--samples', type=int, default=9000)
    p.add_argument('--corner-samples', type=int, default=45000)
    p.add_argument('--chunk', type=int, default=20)
    p.add_argument('--worker', action='store_true')
    p.add_argument('--space', choices=['linear', 'logit'], default='linear')
    main(p.parse_args())
