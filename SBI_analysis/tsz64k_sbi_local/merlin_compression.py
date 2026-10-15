"""The Merlin data compression on the 64k flat-prior SO spectra, compared with MOPED and plain PCA.

Merlin: Wernersson, Franco-Abellan & Canas-Herrera 2026, arXiv:2609.24946, Sect. 3.3, and their code
(merlin/preprocessing.py, merlin/network.py). The steps:
1. Whitening: every spectrum x~ = L^-1 x, with C = L L^T the covariance of the Battaglia12 SO noise draws.
   --source unbinned (default): all 7,900 multipoles, the inputs of merlin_unbinned_prepare.py (the 40-band
   estimator applied to the 20 B12 draws at full resolution); --source bins40: the 40 bands and the Fisher
   covariance (results_logit/fisher_b12_samples.npz).
2. PCA on the whitened noise-free training spectra (merlin: torch.pca_lowrank(Cells_chol, q=500, center=True)),
   keeping the components whose singular value exceeds 0.001 % of the sum (merlin: (S / S.sum()) * 100 > 0.001).
   The projections of the whitened noisy spectra, s = x~ V (merlin: (C_ells + noise) @ V_proj), are the inputs.
3. Input transform (prepare --input-transform):
   asinh (default): per component j, z = arcsinh((s - m_j) / c_j), with m_j the training-pool median of s and c_j
   the noise scale 1.4826 x MAD over the pool of s_noisy - s_clean, then z-scored with its pool mean and sd. The
   spectra span many decades over the flat prior, so the pool sd of s is set by the brightest rows and a linear
   z-score leaves most rows, B12 included, within ~1e-3 sd of each other; the asinh is linear within a noise unit
   of the centre and logarithmic beyond, so that the rows spread over O(1) sd and relative differences count at
   every signal level.
   linear: the raw s of the first runs; the 'pca' variant and the Merlin head z-score it with the pool mean and sd.
4. Learned compression head (merlin Network): standardization -> Linear 2048 -> ReLU -> Linear 512 -> ReLU ->
   Linear 9 x N_feat -> BatchNorm1d, N_feat = 2 features per parameter (18), trained jointly with the posterior
   network. Merlin trains it with marginal ratio estimators (swyft); here the posterior network is the logit-space
   MAF of the 64k study, so that only the compression differs between the variants. The standardization layer
   stands for swyft's online one, which converges to the pool mean and sd; for asinh inputs it is the identity.
Variants, one training setup (merlin's: AdamW lr 5e-4, batch 256, ReduceLROnPlateau x0.1 after 3 epochs, early
stop after 5, 20 % validation; MAF 128x5; 3 seeds; the 61,440-row training pool and 3,000 test rows of
sbi_final.py):
  merlin   inputs -> learned head (18) -> flow
  pca      standardized inputs -> flow
  moped    the fixed-B12 MOPED features of sbi_final.py (9, from all multipoles) -> flow
Merlin also redraws the Gaussian noise every epoch; here each simulation has its one map-noise realization, as for
the other variants. The B12 test is the fitted SO mock (x_obs), whose noise did not enter the whitening covariance;
the 21 B12 noise draws (x_draws) include the 20 that did, so their mean is kept only for comparison.

Reads: data/rows_64440 (row_id, theta, meta.json; clean_dl, noisy_dl for bins40), results_final/split.npz,
results_final/moped_noisy/N61440/features.npz, mocks/pass1_B12/B12/noisy_cross_*.npy (the 21 B12 noise draws), the
B12 observation and MOPED matrix of the 256-row test (analyze_64k.TEST256); for unbinned
results_merlin_unbinned/inputs/{s_noisy.npy, s_clean.npy, b12.npz, clean.json, noisy.json}, for bins40
results_logit/fisher_b12_samples.npz.
Writes to results_merlin_<source><MERLIN_TAG>/: prepare: features_merlin.npz, features_moped.npz, prepare.json;
train: <variant>_seed<k>.pt and .json; evaluate: metrics_<variant>.json, b12_samples_<variant>.npy;
plot: fig_corner_merlin_moped_pca.png, fig_corner_<variant>.png, comparison.md.

  merlin_compression.py [--source unbinned|bins40] prepare [--input-transform asinh|linear]
  merlin_compression.py [--source ..] train [--variants ..] [--seeds ..] [--workers 9] [--threads 2]
  merlin_compression.py [--source ..] evaluate [--variants ..] [--threads 8] [--samples 1000]
  merlin_compression.py [--source ..] plot
prepare runs here (the spectra arrive here); train and evaluate run on idark (idark_merlin/), plot anywhere.
MERLIN_PATIENCE and MERLIN_LR_PATIENCE (epochs) lengthen the training for a longer-training check, and MERLIN_TAG
names its output folder. MERLIN_TRAIN_ROWS=N trains on the first N pool rows (the nested training sets of
sbi_final.py) for a training-set-size series; use it with its own MERLIN_TAG and the prepared features linked in. prepare refuses to overwrite the inputs of networks trained with the other input
transform: use MERLIN_TAG (e.g. MERLIN_TAG=_asinh) for a new run.
MERLIN_PARAM_SPACE=linear trains, loads and evaluates the flows on u = (theta - low) / (high - low) instead of its
logit, so a posterior is no longer confined to the prior box: mass beyond a bound is the flow's continuation of the
posterior past the edge (there are no training simulations there). It replaces train_ensemble's to_u / from_u /
log_jacobian_theta_per_u for the whole process, so such a process must not use logit-space networks; every seed
records its space (settings.param_space) and ensemble() refuses a mismatch. Use it with its own MERLIN_TAG.
"""
import argparse
import concurrent.futures as cf
import json
import multiprocessing as mp
import os
from pathlib import Path
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROWS = HERE / 'data' / 'rows_64440'              # the 64,440 analysed rows: row ids, theta, 40-band spectra
SPLIT = HERE / 'results_final' / 'split.npz'     # training pool ('order') and test rows ('test') of sbi_final.py
B12_DRAWS = HERE / 'mocks' / 'pass1_B12' / 'B12'  # the 21 B12 SO map-noise draws, noisy_cross_*.npy
VARIANTS = ('merlin', 'pca', 'moped')            # the compressions compared (plot shows these three)
# 'raw' (train / evaluate --variants raw): no binning and no compression, all 7,900 multipoles with an asinh at each
# multipole's noise level (features_raw.npz, raw_features.py) into the same learned head as Merlin
SEEDS = (1, 2, 3)                                # training seeds of each variant's ensemble
N_FEAT = 2                                       # learned features per parameter (merlin N_feat): 9 x 2 = 18
INPUT_TRANSFORMS = ('asinh', 'linear')           # prepare --input-transform choices, the first is the default
MAD_TO_SD = 1.4826                               # sd / median absolute deviation of a Gaussian
MIN_GAIN = 1e-6                                  # validation-loss decrease that counts as an improvement
VAL_CHUNK = 4096                                 # validation rows per loss evaluation (bounds the memory)
B12_SAMPLES = 30000                              # posterior samples for the B12 mock
COLORS = dict(merlin='#2a78d6', moped='#eb6834', pca='#1baf7a')  # plot colour per variant
# merlin's training setup; MERLIN_PATIENCE / MERLIN_LR_PATIENCE override the two patiences for a longer-training
# check of the same comparison, written to results_merlin_<source><MERLIN_TAG> (out_dir)
SETTINGS = dict(lr=5e-4,                                                # AdamW learning rate
                batch=256,                                              # mini-batch size
                patience=int(os.environ.get('MERLIN_PATIENCE', 5)),     # early stop: epochs without improvement
                lr_patience=int(os.environ.get('MERLIN_LR_PATIENCE', 3)),  # ReduceLROnPlateau patience, epochs
                lr_factor=0.1,                                          # ReduceLROnPlateau learning-rate factor
                val_fraction=0.2,                                       # validation share of the training pool
                max_epochs=1000,                                        # bound; early stopping ends training first
                hidden=128,                                             # MAF hidden units
                transforms=5,                                           # MAF transforms
                head=(2048, 512),                                       # hidden widths of the compression head
                eps_v_percent=0.001)                                    # PCA: keep S / S.sum() * 100 above this
PARAM_SPACE = os.environ.get('MERLIN_PARAM_SPACE', 'logit')            # 'logit' (prior box) or 'linear' (unbounded)
SETTINGS['param_space'] = PARAM_SPACE
if PARAM_SPACE == 'linear':
    import train_ensemble as _train_ensemble

    def _to_u_linear(theta, low, high):
        """u = (theta - low) / (high - low), float64, unbounded (the linear counterpart of train_ensemble.to_u)."""
        return (np.asarray(theta, dtype=np.float64) - low) / (high - low)

    def _from_u_linear(u, low, high):
        """theta = low + (high - low) u, float64 of the shape of u (inverse of _to_u_linear)."""
        return low + (high - low) * np.asarray(u, dtype=np.float64)

    def _log_jacobian_linear(u, low, high):
        """log |d theta / d u| = sum log (high - low), constant, of shape u.shape[:-1]."""
        return np.full(np.shape(u)[:-1], float(np.sum(np.log(np.asarray(high) - np.asarray(low)))))

    # every consumer imports these names from train_ensemble at call time (Ensemble.posterior / hpd_level,
    # sbi_final.row_metrics, build, train_one, evaluate), so replacing the module attributes switches all of them
    _train_ensemble.to_u = _to_u_linear
    _train_ensemble.from_u = _from_u_linear
    _train_ensemble.log_jacobian_theta_per_u = _log_jacobian_linear
elif PARAM_SPACE != 'logit':
    raise ValueError(f'MERLIN_PARAM_SPACE must be logit or linear, not {PARAM_SPACE!r}')


def out_dir(source):
    """Output folder of one input source ('unbinned' or 'bins40'): results_merlin_<source><MERLIN_TAG>."""
    return HERE / f"results_merlin_{source}{os.environ.get('MERLIN_TAG', '')}"


def b12_noise_draws():
    """The 21 B12 SO map-noise draws at every multipole: D_ell of shape (21, 7900), in file-name order."""
    from train_ensemble import ELL, DL
    return np.stack([np.load(p)[ELL] * DL for p in sorted(B12_DRAWS.glob('noisy_cross_*.npy'))])


def moped_inputs():
    """The MOPED inputs of the 64k study.

    Returns (mf, moped_draws, draws): mf the features.npz of results_final/moped_noisy/N61440 (training pool, test
    rows and the B12 observation, 9 features each), moped_draws (21, 9) the same features of the 21 B12 noise
    draws, draws (21, 7900) the D_ell of those draws.
    """
    import sbi_final as S
    import analyze_64k as A
    mf = np.load(HERE / 'results_final' / 'moped_noisy' / 'N61440' / 'features.npz')
    maps, _ = S.variant_features(argparse.Namespace(out=HERE / 'results_final'))
    # the fixed B12 MOPED matrix (7900 x 9); the moped_noisy map adds the study's asinh standardization
    matrix = np.load(A.TEST256 / 'reference' / 'compression_context.npz')['Battaglia12_rcond1e-06']
    draws = b12_noise_draws()
    # the maps take (clean40, noisy40, moped_raw_noisy, z); only the raw MOPED projections enter this one
    return mf, maps['moped_noisy'](None, None, draws @ matrix, None), draws


def fit_asinh(s_noisy, s_clean):
    """Parameters of the asinh input transform, fitted on the training pool.

    s_noisy, s_clean: (rows, n_pca) projections of the whitened noisy and noise-free training spectra.
    Returns a dict of (n_pca,) float64 arrays: s_center (pool median of s_noisy), s_scale (noise scale), and
    z_mean, z_sd (pool mean and sd of z = arcsinh((s_noisy - s_center) / s_scale)).
    """
    # noise scale: the sd of the noise part r = s_noisy - s_clean, from its median absolute deviation, which is not
    # pulled up by the bright rows (their sample variance is large)
    residual = s_noisy - s_clean
    scale = MAD_TO_SD * np.median(np.abs(residual - np.median(residual, axis=0)), axis=0)
    if not np.all(scale > 0):
        raise ValueError('zero noise scale in a PCA component')
    center = np.median(s_noisy, axis=0)
    # arcsinh(y) ~ y for |y| < 1 and ~ sign(y) log(2|y|) beyond: linear within a noise unit of the centre,
    # logarithmic across the decades of the prior
    z = np.arcsinh((s_noisy - center) / scale)
    return dict(s_center=center, s_scale=scale, z_mean=z.mean(0), z_sd=z.std(0))


def asinh_standardize(s, params):
    """The asinh inputs of projections s (rows, n_pca): (arcsinh((s - s_center) / s_scale) - z_mean) / z_sd.

    params: the dict of fit_asinh. Returns float64 of the shape of s, of pool mean 0 and sd 1 per component.
    """
    z = np.arcsinh((s - params['s_center']) / params['s_scale'])
    return (z - params['z_mean']) / params['z_sd']


def refuse_mixed_transform(out, input_transform):
    """Raise before prepare overwrites the inputs of networks in out that were trained with another transform."""
    previous = out / 'prepare.json'
    if not previous.exists() or not any(out.glob('*_seed*.pt')):
        return
    # prepare.json files written before the option existed are linear
    trained_with = json.loads(previous.read_text()).get('input_transform', 'linear')
    if trained_with != input_transform:
        raise RuntimeError(f'{out} holds networks trained on {trained_with} inputs; to prepare {input_transform} '
                           f'inputs set MERLIN_TAG (e.g. MERLIN_TAG=_{input_transform}) for a separate folder')


def write_features(out, s, s_obs, s_draws, info, s_clean=None, input_transform=INPUT_TRANSFORMS[0]):
    """Write the inputs of the three variants and prepare.json to the folder out.

    s: (64440, n_pca) projections of the whitened noisy spectra, rows of data/rows_64440 in its order; s_obs:
    (1, n_pca) the fitted B12 SO mock; s_draws: (21, n_pca) the B12 noise draws; info: dict for prepare.json
    (updated in place); s_clean: (64440, n_pca) projections of the noise-free spectra, needed for 'asinh';
    input_transform: 'asinh' or 'linear' (module docstring). Pool and test rows: the split of sbi_final.py.
    features_merlin.npz: x_pool, x_held, x_obs, x_draws (float32 inputs), theta_pool, theta_held, pool, held, low,
    high, s_mean and s_sd (the standardization of load_features('pca') and of the head: pool mean and sd for linear
    inputs, 0 and 1 for asinh inputs, which are stored standardized), input_transform, and for asinh the transform
    parameters s_center, s_scale, z_mean, z_sd. features_moped.npz: the same layout for the MOPED features.
    """
    if input_transform not in INPUT_TRANSFORMS:
        raise ValueError(f'unknown input transform {input_transform!r}')
    if input_transform == 'asinh' and s_clean is None:
        raise ValueError('the asinh input transform needs the clean projections s_clean')
    refuse_mixed_transform(out, input_transform)
    theta = np.load(ROWS / 'theta.npy')
    meta = json.loads((ROWS / 'meta.json').read_text())
    split = np.load(SPLIT)
    pool = split['order']
    test = split['test']
    if input_transform == 'linear':
        # the raw projections; load_features('pca') and the head z-score them with the pool mean and sd
        x, x_obs, x_draws = s, s_obs, s_draws
        mean, sd = s[pool].mean(0), s[pool].std(0)
        transform_params = {}
    else:
        # a linear z-score is dominated by the brightest rows (the spectra span many decades over the flat prior)
        # and leaves most rows, B12 included, within ~1e-3 sd of each other: asinh with a noise-level scale first,
        # then the z-score, stored here so that the later standardizations are identities
        transform_params = fit_asinh(s[pool], s_clean[pool])
        x, x_obs, x_draws = (asinh_standardize(v, transform_params) for v in (s, s_obs, s_draws))
        mean, sd = np.zeros(s.shape[1]), np.ones(s.shape[1])
    mf, moped_draws, draws = moped_inputs()
    assert np.array_equal(mf['pool'], pool) and np.array_equal(mf['held'], test)
    out.mkdir(exist_ok=True)
    np.savez(out / 'features_merlin.npz', x_pool=x[pool].astype(np.float32), x_held=x[test].astype(np.float32),
             x_obs=np.asarray(x_obs, np.float32), x_draws=np.asarray(x_draws, np.float32),
             theta_pool=theta[pool], theta_held=theta[test], pool=pool, held=test,
             low=np.array(meta['lower']), high=np.array(meta['upper']), s_mean=mean, s_sd=sd,
             input_transform=np.array(input_transform), **transform_params)
    np.savez(out / 'features_moped.npz', x_pool=mf['x_pool'], x_held=mf['x_held'], x_obs=mf['x_obs'],
             x_draws=moped_draws.astype(np.float32), theta_pool=mf['theta_pool'], theta_held=mf['theta_held'],
             pool=pool, held=test, low=mf['low'], high=mf['high'])
    info.update(n_feat_per_parameter=N_FEAT, train_rows=int(len(pool)), test_rows=int(len(test)),
                b12_draws=int(len(draws)), settings=SETTINGS, input_transform=input_transform)
    if transform_params:
        scale = transform_params['s_scale']
        info['asinh'] = dict(center='training-pool median of s',
                             scale='1.4826 MAD over the pool of s_noisy - s_clean',
                             scale_range=[float(scale.min()), float(scale.max())])
    (out / 'prepare.json').write_text(json.dumps(info, indent=1) + '\n')
    print(f"N_PCA = {s.shape[1]}, {input_transform} inputs; features written to {out}", flush=True)


def prepare_unbinned(out, input_transform=INPUT_TRANSFORMS[0]):
    """prepare for --source unbinned: write_features on the all-multipole projections (merlin_unbinned_prepare.py).

    Reads results_merlin_unbinned/inputs: s_noisy.npy and s_clean.npy (65536, n_pca) in rows_65536 positions,
    b12.npz (s_obs (1, n_pca), s_draws (21, n_pca)) and the reports clean.json and noisy.json (for prepare.json).
    """
    inputs = HERE / 'results_merlin_unbinned' / 'inputs'
    row_ids = np.load(ROWS / 'row_id.npy')
    # rows_65536 positions are row ids: this picks the analysed rows in the order of data/rows_64440
    s_noisy = np.load(inputs / 's_noisy.npy')[row_ids]
    s_clean = np.load(inputs / 's_clean.npy')[row_ids] if input_transform == 'asinh' else None
    b12 = np.load(inputs / 'b12.npz')
    clean, noisy = (json.loads((inputs / f'{k}.json').read_text()) for k in ('clean', 'noisy'))
    regenerated = dict(clean['regenerated_rows_collaborator_vs_idark'],
                       **noisy['regenerated_rows_collaborator_vs_idark'])
    info = dict(source='all 7,900 multipoles (merlin_unbinned_prepare.py)', covariance=clean['covariance'],
                pca=clean['pca'], n_pca=int(s_noisy.shape[1]),
                data_checks=dict(clean=clean['checks'], noisy=noisy['checks']),
                regenerated_rows_collaborator_vs_idark=regenerated)
    write_features(out, s_noisy, b12['s_obs'], b12['s_draws'], info, s_clean, input_transform)


def prepare_bins40(out, input_transform=INPUT_TRANSFORMS[0]):
    """prepare for --source bins40: whitening, PCA and projections of the 40-band spectra, then write_features.

    Reads data/rows_64440 clean_dl.npy and noisy_dl.npy (64440, 40), the Fisher covariance C (40, 40) of
    results_logit/fisher_b12_samples.npz, the B12 observation of the 256-row test and the 21 B12 noise draws
    (rebinned to the 40 bands).
    """
    from scipy.linalg import solve_triangular
    import analyze_64k as A
    from train_ensemble import A_rebin
    clean = np.load(ROWS / 'clean_dl.npy')
    noisy = np.load(ROWS / 'noisy_dl.npy')
    pool = np.load(SPLIT)['order']
    # C = L L^T, the 40-band Fisher covariance of the B12 noise draws
    chol = np.linalg.cholesky(np.load(HERE / 'results_logit' / 'fisher_b12_samples.npz')['C'])

    def white(spectra):
        """Whitened spectra L^-1 x, (rows, 40) for x of shape (rows, 40) or (40,): unit noise covariance."""
        return solve_triangular(chol, np.atleast_2d(spectra).T, lower=True).T

    white_clean = white(clean)
    # PCA: SVD of the centred whitened clean training pool; keep singular values above eps_v_percent of their sum
    _, singular, vt = np.linalg.svd(white_clean[pool] - white_clean[pool].mean(0), full_matrices=False)
    keep = singular / singular.sum() * 100 > SETTINGS['eps_v_percent']
    V = vt[keep].T                                                   # (40, n_pca) projection
    obs = np.load(A.TEST256 / 'cluster_snapshot' / 'diagnostic_256' / 'observations' / 'Battaglia12.npz')
    draws = b12_noise_draws()
    info = dict(source='40 bands', covariance='results_logit/fisher_b12_samples.npz C', n_pca=int(keep.sum()),
                singular_value_percent=(100 * singular / singular.sum()).tolist())
    write_features(out, white(noisy) @ V, white(obs['noisy_dl'][None]) @ V, white(A_rebin(draws)) @ V, info,
                   white_clean @ V, input_transform)


def load_features(out, name):
    """The inputs of one variant, a dict of the arrays of out/features_<merlin|moped>.npz.

    'pca' reads features_merlin.npz and z-scores x_pool, x_held, x_obs, x_draws with s_mean and s_sd (the pool mean
    and sd for linear inputs; 0 and 1, so unchanged, for asinh inputs, which are stored standardized).
    """
    features = dict(np.load(out / f"features_{'merlin' if name == 'pca' else name}.npz"))   # raw: features_raw.npz
    rows = os.environ.get('MERLIN_TRAIN_ROWS')
    if rows:
        # a training-set-size series: the first N rows of the pool, which is in the order of sbi_final.py, so the
        # subsets are the nested training sets of its convergence study; test rows and observation are unchanged
        n = int(rows)
        assert 0 < n <= len(features['x_pool']), n
        for key in ('x_pool', 'theta_pool', 'pool'):
            features[key] = features[key][:n]
    if name == 'pca':
        for key in ('x_pool', 'x_held', 'x_obs', 'x_draws'):
            features[key] = ((features[key] - features['s_mean']) / features['s_sd']).astype(np.float32)
    return features


def make_head(n_in, mean, sd):
    """The Merlin compression head (merlin Network.norm + Network.sequential), a torch module in eval mode.

    n_in: number of inputs (PCA components); mean, sd: (n_in,) standardization of the inputs. The module maps
    inputs (batch, n_in) to features (batch, 9 x N_FEAT).
    """
    import torch
    from torch import nn

    class Head(nn.Module):
        """merlin Network: standardize -> 2048 -> ReLU -> 512 -> ReLU -> 9 x N_feat -> BatchNorm1d"""

        def __init__(self):
            super().__init__()
            # the pool mean and sd as fixed buffers, for swyft's online standardizing layer that converges to
            # them; kept for fidelity, they are 0 and 1 (an identity) for asinh inputs
            self.register_buffer('mean', torch.as_tensor(mean, dtype=torch.float32))
            self.register_buffer('sd', torch.as_tensor(sd, dtype=torch.float32))
            layers = []
            width = n_in
            for units in SETTINGS['head']:
                layers += [nn.Linear(width, units), nn.ReLU()]
                width = units
            # the final BatchNorm1d keeps the learned features at unit scale for the flow
            layers += [nn.Linear(width, 9 * N_FEAT), nn.BatchNorm1d(9 * N_FEAT)]
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            """(batch, n_in) inputs -> (batch, 9 x N_FEAT) features."""
            return self.net((x - self.mean) / self.sd)

    # eval mode, because sbi probes the embedding with one sample (BatchNorm1d needs two in train mode);
    # train_one switches to train mode for the epochs
    return Head().eval()


def build(name, f):
    """The untrained density estimator of one variant: an sbi MAF of theta in logit space u (as train_ensemble.py),
    or in the linear u of MERLIN_PARAM_SPACE=linear.

    name: 'merlin' (the compression head as embedding net), 'pca' or 'moped'; f: the dict of load_features. The
    estimator models u (rows, 9) given the inputs (rows, n_in); sbi z-scores u, the inputs arrive standardized.
    """
    import torch
    from sbi.neural_nets import posterior_nn
    from train_ensemble import to_u
    u = torch.as_tensor(to_u(f['theta_pool'], f['low'], f['high']), dtype=torch.float32)
    x = torch.as_tensor(f['x_pool'], dtype=torch.float32)
    extra = {}
    if name in ('merlin', 'raw'):
        # raw: the same head on the 7,900 asinh-standardized multipoles instead of the 75 whitened PCA components
        extra['embedding_net'] = make_head(x.shape[1], f['s_mean'], f['s_sd'])
    builder = posterior_nn(model='maf', hidden_features=SETTINGS['hidden'], num_transforms=SETTINGS['transforms'],
                           z_score_x='none', z_score_theta='independent', **extra)
    return builder(u, x)


def train_one(out, name, seed, threads):
    """Train one variant with one seed on the training pool; writes out/<name>_seed<seed>.pt and .json.

    A seed-dependent permutation puts 20 % of the pool in validation. AdamW on the mean negative log density of u,
    ReduceLROnPlateau on the validation loss, early stopping after SETTINGS['patience'] epochs without improvement;
    the best-validation state is saved. The .json holds the settings, history and timing. Returns a summary line.
    """
    import torch
    from train_ensemble import to_u
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    np.random.seed(seed)
    f = load_features(out, name)
    net = build(name, f)
    u = torch.as_tensor(to_u(f['theta_pool'], f['low'], f['high']), dtype=torch.float32)
    x = torch.as_tensor(f['x_pool'], dtype=torch.float32)
    perm = torch.from_numpy(np.random.default_rng(seed).permutation(len(u)))
    n_val = int(SETTINGS['val_fraction'] * len(u))
    val, fit = perm[:n_val], perm[n_val:]
    optimizer = torch.optim.AdamW(net.parameters(), lr=SETTINGS['lr'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=SETTINGS['lr_factor'],
                                                           patience=SETTINGS['lr_patience'])
    best_loss = np.inf
    best_state = None
    epochs_without_gain = 0
    history = []
    start = time.monotonic()
    for epoch in range(SETTINGS['max_epochs']):
        net.train()
        shuffled = fit[torch.randperm(len(fit))]
        for i in range(0, len(shuffled), SETTINGS['batch']):
            batch = shuffled[i:i + SETTINGS['batch']]
            if len(batch) < 2:                       # BatchNorm1d needs two rows in train mode
                continue
            optimizer.zero_grad()
            loss = net.loss(u[batch], condition=x[batch]).mean()
            loss.backward()
            optimizer.step()
        net.eval()
        with torch.no_grad():
            chunks = [net.loss(u[val[i:i + VAL_CHUNK]], condition=x[val[i:i + VAL_CHUNK]])
                      for i in range(0, len(val), VAL_CHUNK)]
            val_loss = float(torch.cat(chunks).mean())
        scheduler.step(val_loss)
        history.append(dict(epoch=epoch + 1, val_loss=val_loss, lr=optimizer.param_groups[0]['lr']))
        if val_loss < best_loss - MIN_GAIN:
            best_loss = val_loss
            epochs_without_gain = 0
            best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
        else:
            epochs_without_gain += 1
            if epochs_without_gain >= SETTINGS['patience']:
                break
    net.load_state_dict(best_state)
    torch.save(net.state_dict(), out / f'{name}_seed{seed}.pt')
    record = dict(variant=name, seed=seed, seconds=time.monotonic() - start, epochs=len(history),
                  best_validation_loss=best_loss, history=history, settings=SETTINGS, train_rows=int(len(u)),
                  parameters=int(sum(p.numel() for p in net.parameters())))
    (out / f'{name}_seed{seed}.json').write_text(json.dumps(record, indent=1) + '\n')
    return f"{name} seed {seed}: {record['epochs']} epochs, {record['seconds']:.0f} s, best val {best_loss:.3f}"


def cmd_train(args):
    """train: every (variant, seed) of args without a saved network, in parallel worker processes."""
    jobs = [(v, s) for v in args.variants for s in args.seeds if not (args.out / f'{v}_seed{s}.pt').exists()]
    context = mp.get_context('spawn')                # fresh interpreters for torch, no forked thread pools
    with cf.ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as pool:
        futures = [pool.submit(train_one, args.out, v, s, args.threads) for v, s in jobs]
        for fut in cf.as_completed(futures):
            print(fut.result(), flush=True)


def ensemble(out, name, threads):
    """The trained seeds of one variant as an object with the train_ensemble.Ensemble interface.

    Attributes: f (load_features dict), low, high (prior bounds, (9,)), seeds, nets, settings; methods log_prob_u,
    sample_u, posterior, hpd_level of train_ensemble.Ensemble (equal-weight mixture of the seeds, u space).
    """
    import torch
    from train_ensemble import Ensemble
    torch.set_num_threads(threads)
    f = load_features(out, name)
    ens = Ensemble.__new__(Ensemble)                 # Ensemble.__init__ reads the train_ensemble folder layout
    ens.f = f
    ens.low = f['low']
    ens.high = f['high']
    ens.seeds = list(SEEDS)
    ens.nets = []
    for seed in ens.seeds:
        trained = json.loads((out / f'{name}_seed{seed}.json').read_text())['settings'].get('param_space', 'logit')
        assert trained == PARAM_SPACE, (f'{out.name}/{name} seed {seed} was trained in {trained} space; '
                                        f'set MERLIN_PARAM_SPACE={trained}')
        net = build(name, f)
        net.load_state_dict(torch.load(out / f'{name}_seed{seed}.pt'))
        net.eval()
        ens.nets.append(net)
    ens.settings = dict(SETTINGS, variant=name)
    return ens


def seed_log_prob_theta(ens, theta, x):
    """Log posterior density of theta under each seed of the ensemble alone, in theta units.

    ens: an ensemble(); theta: (rows, 9) parameters; x: (rows, n_in) inputs. Returns (n_seeds, rows):
    log q_seed(u | x) - log |d theta / d u|, the per-seed counterpart of the ensemble log p of row_metrics.
    """
    import torch
    from train_ensemble import to_u, log_jacobian_theta_per_u
    u = to_u(theta, ens.low, ens.high)
    u_tensor = torch.as_tensor(u[None], dtype=torch.float32)          # (1, rows, 9): one sample per row
    x_tensor = torch.as_tensor(np.atleast_2d(x), dtype=torch.float32)
    with torch.no_grad():
        log_q_u = np.stack([net.log_prob(u_tensor, condition=x_tensor).numpy()[0] for net in ens.nets])
    return log_q_u - log_jacobian_theta_per_u(u, ens.low, ens.high)


def cmd_evaluate(args):
    """evaluate: test-row metrics, the B12 posterior and the log density of the true B12 parameters, per variant.

    Writes b12_samples_<variant>.npy (B12_SAMPLES x 9 posterior samples for the fitted B12 mock, theta units) and
    metrics_<variant>.json: sbi_final.row_metrics on the test rows plus
      test_log_prob_seeds / _err: per-seed mean test log p, and the sd over seeds / sqrt(n_seeds)
      b12, b12_width68_over_prior: summary of the B12 posterior, 68 % widths over the prior range
      b12_log_prob_truth / _seeds / _err: log p of the true B12 parameters for the fitted SO mock x_obs (ensemble,
        per seed, sd over seeds / sqrt(n_seeds))
      b12_log_prob_truth_draws / _draws_sem: the mean and SEM over the 21 B12 noise draws (caveat below)
      training: the training records without their histories
    All log densities are in theta units.
    """
    import torch
    import analyze_64k as A
    from sbi_final import row_metrics
    from train_ensemble import to_u, log_jacobian_theta_per_u, summarize
    for name in args.variants:
        ens = ensemble(args.out, name, args.threads)
        f = ens.f
        n_seeds = len(ens.seeds)
        m, _ = row_metrics(ens, f['x_held'], f['theta_held'], args.samples)
        torch.manual_seed(2)
        b12_samples, b12_lp_u, _ = ens.posterior(f['x_obs'][:1], B12_SAMPLES, rng=np.random.default_rng(2))
        np.save(args.out / f'b12_samples_{name}.npy', b12_samples[:, 0])
        # the row SEM of the test log p ignores the scatter between trainings; the per-seed means measure it
        # (sample sd, ddof=1, as there are only a few seeds)
        test_seeds = seed_log_prob_theta(ens, f['theta_held'], f['x_held']).mean(axis=1)
        u_b12 = to_u(A.B12[None], ens.low, ens.high)
        log_jac_b12 = log_jacobian_theta_per_u(u_b12, ens.low, ens.high)[0]
        # the fitted SO mock: its noise realization did not enter the whitening covariance
        b12_truth = float(ens.log_prob_u(u_b12[None], f['x_obs'][:1])[0, 0] - log_jac_b12)
        b12_seeds = seed_log_prob_theta(ens, A.B12[None], f['x_obs'][:1])[:, 0]
        # caveat: the 21 draws include the 20 that built the whitening covariance (merlin_unbinned_prepare
        # whitener; the Fisher covariance for bins40), so for the Merlin and PCA inputs they are not independent
        # test noise
        u_draws = np.repeat(u_b12[None], len(f['x_draws']), axis=1)   # (1, 21, 9): the truth for every draw
        lp_draws = ens.log_prob_u(u_draws, f['x_draws'])[0] - log_jac_b12
        b12 = summarize(b12_samples[:, 0], b12_lp_u[:, 0], A.B12)
        b12_width = (np.array(b12['q84']) - np.array(b12['q16'])) / (ens.high - ens.low)
        training = [dict(json.loads((args.out / f'{name}_seed{s}.json').read_text()), history=None)
                    for s in ens.seeds]
        m.update(variant=name,
                 test_log_prob_seeds=test_seeds.tolist(),
                 test_log_prob_err=float(test_seeds.std(ddof=1) / np.sqrt(n_seeds)),
                 b12=b12,
                 b12_width68_over_prior=b12_width.tolist(),
                 b12_log_prob_truth=b12_truth,
                 b12_log_prob_truth_seeds=b12_seeds.tolist(),
                 b12_log_prob_truth_err=float(b12_seeds.std(ddof=1) / np.sqrt(n_seeds)),
                 b12_log_prob_truth_draws=float(lp_draws.mean()),
                 b12_log_prob_truth_draws_sem=float(lp_draws.std() / np.sqrt(len(lp_draws))),
                 training=training)
        (args.out / f'metrics_{name}.json').write_text(json.dumps(m, indent=1) + '\n')
        print(f"{name}: test log p {m['mean_log_prob_theta']:.2f} ± {m['test_log_prob_err']:.2f} (seeds; rows "
              f"± {m['mean_log_prob_theta_sem']:.2f}), TARP {m['tarp68']:.2f}/{m['tarp95']:.2f}, "
              f"mean r {np.mean(m['pearson_r']):.3f}, B12 log p(truth) {b12_truth:.2f} ± "
              f"{m['b12_log_prob_truth_err']:.2f} (21-draw mean {lp_draws.mean():.2f}), "
              f"B12 mean width {np.mean(b12_width):.3f}", flush=True)


def value_and_error(metrics, key, err_key):
    """'value ± error' (2 decimals) from a metrics record, 'n/a' if it predates the key (evaluate again)."""
    if key not in metrics or err_key not in metrics:
        return 'n/a'
    return f"{metrics[key]:.2f} ± {metrics[err_key]:.2f}"


def mean_of(metrics, key):
    """Mean over parameters of a per-parameter metric (3 decimals), 'n/a' if the record predates the key."""
    return f"{np.mean(metrics[key]):.3f}" if key in metrics else 'n/a'


def cmd_plot(args):
    """plot: B12 corner plots of the three variants (together and alone) and the metrics table comparison.md."""
    import plot_posteriors as PP
    import analyze_64k as A
    info = json.loads((args.out / 'prepare.json').read_text())
    metrics = {v: json.loads((args.out / f'metrics_{v}.json').read_text()) for v in VARIANTS}
    names = dict(merlin=f"Merlin: whitened PCA ({info['n_pca']}) -> learned head (N_feat = {N_FEAT}) -> NPE",
                 moped='MOPED (fixed B12, 9) -> NPE', pca=f"whitened PCA ({info['n_pca']}) -> NPE")
    sets = [(np.load(args.out / f'b12_samples_{v}.npy'), names[v], COLORS[v]) for v in VARIANTS]
    # the patience the networks were trained with (their training records), not this process's MERLIN_PATIENCE
    patience = sorted({t['settings']['patience'] for m in metrics.values() for t in m['training']})
    spectra = '40 bands' if info['source'].startswith('40 bands') else 'all multipoles'
    # the training-set size from the training records (MERLIN_TRAIN_ROWS series), else the whole pool
    train_rows = sorted({t.get('train_rows', info['train_rows']) for m in metrics.values() for t in m['training']})
    train_rows = train_rows[0] if len(train_rows) == 1 else info['train_rows']
    title = (f"Battaglia+12 SO mock, {spectra}, {train_rows:,} training simulations, {len(SEEDS)} seeds "
             f"each (early stop {'/'.join(map(str, patience))} epochs): Merlin vs MOPED vs PCA")
    PP.corner([sets[0], sets[2], sets[1]], truth=A.B12, title=title,
              path=args.out / 'fig_corner_merlin_moped_pca.png')
    for (samples, label, color), v in zip(sets, VARIANTS):
        PP.corner([(samples, label, color)], truth=A.B12, title=f'Battaglia+12 SO mock: {label}',
                  path=args.out / f'fig_corner_{v}.png')
    rows = ['| input | test log p | TARP 68 / 95 | mean r | mean test width / prior | mean test width / flat '
            'prior | mean edge mass ratio | B12 log p(truth) | B12 mean width / prior |',
            '|---|---|---|---|---|---|---|---|---|']
    for v in VARIANTS:
        m = metrics[v]
        rows.append(f"| {names[v]} | {value_and_error(m, 'mean_log_prob_theta', 'test_log_prob_err')} | "
                    f"{100 * m['tarp68']:.0f} / {100 * m['tarp95']:.0f} | {np.mean(m['pearson_r']):.3f} | "
                    f"{np.mean(m['median_width68_over_prior']):.3f} | {mean_of(m, 'median_width68_over_flat')} | "
                    f"{mean_of(m, 'edge_mass_ratio')} | "
                    f"{value_and_error(m, 'b12_log_prob_truth', 'b12_log_prob_truth_err')} | "
                    f"{np.mean(m['b12_width68_over_prior']):.3f} |")
    notes = ['',
             f"Inputs: {info.get('input_transform', 'linear')}. Errors: sd over the {len(SEEDS)} seeds / "
             f"sqrt({len(SEEDS)}) (test: of the per-seed mean log p over the {info['test_rows']:,} test rows; "
             'B12: of the per-seed log p of the true parameters for the fitted SO mock). Widths: median 68 % '
             'interval over the prior range (flat prior 0.683) and over 0.683 x the range (flat prior 1). '
             "Edge mass ratio: posterior mass in the outer 1 % of each prior range over the flat prior's 2 % "
             '(1 on average for an exact posterior). n/a: metrics from an older evaluate.']
    (args.out / 'comparison.md').write_text('\n'.join(rows + notes) + '\n')
    print('\n'.join(rows + notes))


def main():
    """Command line: see the module docstring."""
    parser = argparse.ArgumentParser(description='Merlin compression vs MOPED vs PCA on the 64k SO spectra')
    parser.add_argument('--source', choices=('unbinned', 'bins40'), default='unbinned')
    sub = parser.add_subparsers(dest='cmd', required=True)
    prepare_parser = sub.add_parser('prepare')
    prepare_parser.add_argument('--input-transform', choices=INPUT_TRANSFORMS, default=INPUT_TRANSFORMS[0],
                                help='asinh: noise-scaled asinh, then z-score (default); linear: raw projections')
    train_parser = sub.add_parser('train')
    train_parser.add_argument('--variants', nargs='+', default=list(VARIANTS))
    train_parser.add_argument('--workers', type=int, default=9)
    train_parser.add_argument('--threads', type=int, default=2)
    train_parser.add_argument('--seeds', nargs='+', type=int, default=list(SEEDS))
    evaluate_parser = sub.add_parser('evaluate')
    evaluate_parser.add_argument('--variants', nargs='+', default=list(VARIANTS))
    evaluate_parser.add_argument('--threads', type=int, default=8)
    evaluate_parser.add_argument('--samples', type=int, default=1000)
    sub.add_parser('plot')
    args = parser.parse_args()
    args.out = out_dir(args.source)
    if args.cmd == 'prepare':
        prepare = prepare_unbinned if args.source == 'unbinned' else prepare_bins40
        prepare(args.out, args.input_transform)
    else:
        commands = dict(train=cmd_train, evaluate=cmd_evaluate, plot=cmd_plot)
        commands[args.cmd](args)


if __name__ == '__main__':
    main()
