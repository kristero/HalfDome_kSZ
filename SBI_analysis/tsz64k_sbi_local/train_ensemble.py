"""Logit-space NPE ensembles for the partial tSZ 64k dataset: noisy (SO mock) and noise-free (clean spectra).

Trains and evaluates neural posterior estimators (sbi NPE with MAF or NSF flows) of the 9 Battaglia+12 pressure
parameters (P0, xc, beta and their mass and redshift slopes) from compressed tSZ power spectra. With respect to
analyze_64k.py everything is identical (data, fixed B12 MOPED compression, random split with seed 0, 5 % held out
and the first 768 held-out rows evaluated, observations) except:
  * parameters are learned in logit space, u = logit((theta - low) / width), whose prior is an independent logistic
    distribution (the image of the flat prior), so every posterior sample lies inside the prior box: no truncation,
    no leakage;
  * a higher epoch cap (early stopping still decides) and a larger batch;
  * several seeds trained in parallel processes, evaluated as their equal-weight ensemble.
The module also holds the pieces other scripts import (sbi_final.py, merlin_compression.py, ...): the 40-bin
compressions (fit_binned, project_binned), the parameter transform (to_u, from_u, log_jacobian_theta_per_u), the
Ensemble class, summarize, and the band averaging of unbinned spectra (A_rebin, with ELL and DL).

Reads
  <dataset>/meta.json (prior bounds 'lower', 'upper'), theta.npy (rows, 9), clean_dl_unbinned.npy and
  noisy_dl_unbinned.npy (rows, 7900: D_ell at ell = 80..7979, dimensionless Compton-y), clean_dl.npy and
  noisy_dl.npy (rows, 40 band powers); the four observations analyze_64k.TEST256/cluster_snapshot/diagnostic_256/
  observations/<label>.npz; for 'realizations' <spectra>/noisy_cross_<seed1>_<seed2>.npy and masked_clean_cl.npy
  (C_ell indexed by ell).
Writes, in <out>/<kind>/ ('noisy' and 'clean' for MOPED, '<kind>_<method>' for the 40-bin compressions)
  prepare       features.npz, transform.npz, prepare.json
  train         estimator_seed<s>.pt (network state dict), train_seed<s>.json
  evaluate      metrics.json, recovered.json, heldout.npz, posterior_<label>.npy, fig_battaglia12.png,
                fig_flamingo.png, sbc_ranks_hist.png, sbc_ranks_cdf.png, tarp.png, truth_vs_mean.png
  realizations  b12_realizations.json, fig_b12_noise_realizations.png (in <out>/noisy/)

Run
  train_ensemble.py prepare      --dataset D --out O                    # features for both kinds
  train_ensemble.py train        --out O --kinds noisy clean --seeds ...  # parallel training
  train_ensemble.py evaluate     --out O --kind noisy                   # diagnostics, figures, recovered.json
  train_ensemble.py realizations --out O --spectra DIR                  # B12 SO mocks with many noise draws
"""
import argparse
import concurrent.futures as cf
import json
import multiprocessing as mp
from pathlib import Path
import re
import time

import numpy as np

HERE = Path(__file__).resolve().parent                # folder of this script
ELL = np.arange(80, 7980)                             # multipoles of the unbinned spectra (7900 values)
DL = ELL * (ELL + 1) / (2 * np.pi)                    # C_ell -> D_ell factor, ell (ell + 1) / (2 pi)
BAND_EDGES = np.r_[np.arange(80, 7881, 200), 7980]    # edges of the 40 bands: 200 multipoles wide, the last 100
N_BANDS = 40                                          # number of band powers of the binned spectra
OBS_LABELS = ['Battaglia12', 'L1_m9', 'fgas-8sigma', 'Mstar-1sigma']   # B12 control, then three FLAMINGO y-maps
SPLIT_SEED = 0                                        # seed of the permutation that splits pool and held-out rows
HELDOUT_FRACTION = 0.05                               # share of the rows held out from training
VALIDATION_FRACTION = 0.1                             # share of the pool sbi keeps aside for early stopping
UNIT_CLIP = 1e-9                                      # (theta - low) / width is clipped to [c, 1 - c] before logit
SCALE_FLOOR = 1e-30                                   # smallest asinh scale (a band whose median is zero)
SPREAD_FLOOR = 1e-8                                   # smallest standard deviation used in a z-score
FITTED_NOISE_SEEDS = (39001001, 39001002)             # noise seeds of the B12 observation that is fitted


def to_u(theta, low, high):
    """Map parameters from the prior box to logit space.

    theta: (..., dim) parameters; low, high: (dim,) prior bounds. Returns u = logit((theta - low) / (high - low)),
    float64 of the same shape. A flat prior on theta becomes an independent standard logistic prior on u.
    """
    unit = np.clip((np.asarray(theta, dtype=np.float64) - low) / (high - low), UNIT_CLIP, 1 - UNIT_CLIP)
    # logit(s) = log s - log(1 - s); the clip keeps points on the box edges finite
    return np.log(unit) - np.log1p(-unit)


def from_u(u, low, high):
    """Inverse of to_u: theta = low + (high - low) * sigmoid(u), float64 of the shape of u (..., dim)."""
    return low + (high - low) / (1 + np.exp(-np.asarray(u, dtype=np.float64)))


def log_jacobian_theta_per_u(u, low, high):
    """log |d theta / d u| summed over parameters (theta-space density = u-space density / this).

    u: (..., dim) logit-space points; low, high: (dim,) prior bounds. Returns (...,) float64.
    """
    u = np.asarray(u, dtype=np.float64)
    # d theta / du = width * sigmoid(u) * sigmoid(-u), and -log sigmoid(u) = log(1 + e^-u) = logaddexp(0, -u)
    return np.sum(np.log(high - low) - np.logaddexp(0, u) - np.logaddexp(0, -u), axis=-1)


def logit_prior(dim):
    """The u-space prior: dim independent standard logistic variables, the logit image of a flat unit box.

    Returns a torch distribution with event shape (dim,), used as the NPE prior.
    """
    import torch
    from torch.distributions import Independent, TransformedDistribution, Uniform
    from torch.distributions.transforms import SigmoidTransform
    unit_box = Uniform(torch.zeros(dim), torch.ones(dim))
    logistic = TransformedDistribution(unit_box, [SigmoidTransform().inv])   # logit of a uniform variable
    return Independent(logistic, 1)


def fit_binned(method, clean40, noisy40, scale=None):
    """Fit a 40-bin compression on training rows (diagnostic_analysis.transform_fit of the 256-row test).

    Every band power is mapped through a signed asinh, y = asinh(x / scale), and z-scored with the training mean
    and standard deviation; 'bins40' keeps the 40 z-scores, 'pca<k>' (e.g. 'pca9') projects them on their k
    leading principal components. The kept features are standardised once more on the training rows.

    asinh(x / scale) is linear for |x| << scale and logarithmic above it, so the scale decides which rows the
    network can tell apart: a difference dx maps to about dx / scale in the linear regime and to the relative
    difference dx / |x| above it. The default rule (the per-band median of |clean40|) leaves the lower half of the
    prior (B12 included) in the linear regime, where the differences are small compared with the log-regime spread
    that sets the z-score; a noise-level scale keeps noise-level differences at O(1).

    method: 'bins40' or 'pca<k>'. clean40: (rows, 40) noise-free band powers D_ell (dimensionless), used only by
    the default scale rule. noisy40: (rows, 40) band powers the compression is fitted to (noisy or clean).
    scale: None for the default rule (median |clean40| per band, floored at 1e-30) or an array of 40 positive
    floats, in the units of noisy40, used as the asinh scale instead.
    Returns (state, meta). state: scale (40,), center (40,), spread (40,), matrix (40, k), mean (k,), std (k,),
    the input of project_binned. meta: input_features, retained and, for PCA, explained_variance_fraction.
    """
    if scale is None:
        scale = np.maximum(np.median(np.abs(clean40), axis=0), SCALE_FLOOR)
    else:
        scale = np.asarray(scale, dtype=np.float64)
        if scale.shape != (N_BANDS,) or not np.all(np.isfinite(scale) & (scale > 0)):
            raise ValueError(f'scale must be {N_BANDS} positive finite floats, got shape {scale.shape}')
    y = np.arcsinh(noisy40 / scale)
    # z-score every band with the training statistics (floored so a constant band does not divide by zero)
    center = y.mean(0)
    spread = np.maximum(y.std(0), SPREAD_FLOOR)
    z = (y - center) / spread
    if method == 'bins40':
        matrix = np.eye(N_BANDS)
        meta = dict(input_features=N_BANDS, retained=N_BANDS)
    elif method.startswith('pca'):
        retained = int(method[3:])
        # principal axes of the z-scored training features: the rows of vt, by decreasing singular value
        _, singular_values, vt = np.linalg.svd(z, full_matrices=False)
        matrix = vt[:retained].T
        explained = float(np.sum(singular_values[:retained] ** 2) / np.sum(singular_values ** 2))
        meta = dict(input_features=N_BANDS, retained=retained, explained_variance_fraction=explained)
    else:
        raise ValueError(method)
    projected = z @ matrix
    state = dict(scale=scale, center=center, spread=spread, matrix=matrix, mean=projected.mean(0),
                 std=np.maximum(projected.std(0), SPREAD_FLOOR))
    return state, meta


def project_binned(state, noisy40):
    """Apply a fitted 40-bin compression.

    state: the dict of fit_binned; noisy40: (rows, 40) or (40,) band powers. Returns (rows, k) float32 features.
    """
    z = (np.arcsinh(np.atleast_2d(noisy40) / state['scale']) - state['center']) / state['spread']
    return ((z @ state['matrix'] - state['mean']) / state['std']).astype(np.float32)


def cmd_prepare(args):
    """Write the training and held-out features (the 'prepare' subcommand).

    Without --compressions: the fixed B12 MOPED compression (analyze_64k.fit_transform) of the unbinned spectra,
    for both kinds, into <out>/noisy and <out>/clean. With --compressions: the 40-bin compressions (fit_binned) of
    every kind in --kinds, into <out>/<kind>_<method>. Each folder gets features.npz (x_pool, x_held, theta_pool,
    theta_held, pool, held, low, high, obs_labels, x_obs), transform.npz (the fitted state) and prepare.json.
    """
    import analyze_64k as A
    meta = json.loads((args.dataset / 'meta.json').read_text())
    low, high = np.array(meta['lower']), np.array(meta['upper'])
    theta = np.load(args.dataset / 'theta.npy')
    n_rows = len(theta)
    clean_unbinned = np.load(args.dataset / 'clean_dl_unbinned.npy', mmap_mode='r')
    noisy_unbinned = np.load(args.dataset / 'noisy_dl_unbinned.npy', mmap_mode='r')
    # random split: the first 5 % of a seed-0 permutation is held out (of which the first --heldout-rows in row
    # order are evaluated), the rest is the training pool
    perm = np.random.default_rng(SPLIT_SEED).permutation(n_rows)
    n_heldout = int(round(HELDOUT_FRACTION * n_rows))
    held = np.sort(perm[:n_heldout])[:args.heldout_rows]
    pool = np.sort(perm[n_heldout:])
    obs_dir = A.TEST256 / 'cluster_snapshot' / 'diagnostic_256' / 'observations'
    obs = {label: np.load(obs_dir / f'{label}.npz') for label in OBS_LABELS}
    if args.compressions:
        clean40 = np.load(args.dataset / 'clean_dl.npy')
        noisy40 = np.load(args.dataset / 'noisy_dl.npy')
        for kind in args.kinds:
            source = noisy40 if kind == 'noisy' else clean40
            key = 'noisy_dl' if kind == 'noisy' else 'clean_dl'
            for method in args.compressions:
                # the compression is fitted on the training pool only
                state, cmeta = fit_binned(method, clean40[pool], source[pool])
                folder = args.out / f'{kind}_{method}'
                folder.mkdir(parents=True, exist_ok=True)
                x_obs = np.concatenate([project_binned(state, obs[label][key]) for label in OBS_LABELS])
                np.savez(folder / 'features.npz', x_pool=project_binned(state, source[pool]),
                         x_held=project_binned(state, source[held]), theta_pool=theta[pool],
                         theta_held=theta[held], pool=pool, held=held, low=low, high=high,
                         obs_labels=np.array(OBS_LABELS), x_obs=x_obs)
                np.savez(folder / 'transform.npz', **state)
                info = dict(dataset=str(args.dataset), kind=kind, compression=method, meta=cmeta,
                            observation_key=key)
                (folder / 'prepare.json').write_text(json.dumps(info, indent=1) + '\n')
                print(folder.name, 'features written', cmeta, flush=True)
        return
    for kind in ('noisy', 'clean'):
        source = noisy_unbinned if kind == 'noisy' else clean_unbinned
        state, cmeta = A.fit_transform('moped_fixed', clean_unbinned[pool], source[pool])
        key = 'noisy_dl_unbinned' if kind == 'noisy' else 'clean_dl_unbinned'
        folder = args.out / kind
        folder.mkdir(parents=True, exist_ok=True)
        x_obs = np.stack([A.project(state, obs[label][key][None])[0] for label in OBS_LABELS])
        np.savez(folder / 'features.npz', x_pool=A.project(state, source[pool]),
                 x_held=A.project(state, source[held]), theta_pool=theta[pool], theta_held=theta[held], pool=pool,
                 held=held, low=low, high=high, obs_labels=np.array(OBS_LABELS), x_obs=x_obs)
        np.savez(folder / 'transform.npz', **{name: value for name, value in state.items() if name != 'method'})
        info = dict(dataset=str(args.dataset), rows=int(n_rows), pool=int(len(pool)),
                    heldout_evaluated=int(len(held)), kind=kind, compression=cmeta, observation_key=key)
        (folder / 'prepare.json').write_text(json.dumps(info, indent=1) + '\n')
        print(kind, 'features written:', len(pool), 'training pool rows,', len(held), 'held-out rows', flush=True)


def make_builder(model, hidden, transforms, bins):
    """sbi density-estimator builder: a 'maf' or 'nsf' flow with the given width, depth and (NSF) spline bins.

    x enters as it is (the features are already standardised); u is z-scored per parameter inside the network.
    """
    from sbi.neural_nets import posterior_nn
    extra = dict(num_bins=bins) if model == 'nsf' else {}
    return posterior_nn(model=model, hidden_features=hidden, num_transforms=transforms, z_score_x='none',
                        z_score_theta='independent', **extra)


def build_estimator(u_pool, x_pool, hidden, transforms, model='maf', bins=10):
    """An untrained flow with the architecture of train_one, to load a saved state dict into.

    u_pool: (rows, dim) logit-space parameters and x_pool: (rows, xdim) features of the training pool; sbi uses
    them for the shapes and the u z-scoring (both overwritten by load_state_dict).
    """
    import torch
    builder = make_builder(model, hidden, transforms, bins)
    return builder(torch.tensor(u_pool, dtype=torch.float32), torch.tensor(x_pool, dtype=torch.float32))


def train_one(out, kind, seed, threads, epochs, stop_after, batch, lr, hidden, transforms, model='maf', bins=10):
    """Train one seed of one kind with sbi NPE in logit space (runs in a worker process).

    out: output root (str); kind: subfolder with features.npz; seed: torch and numpy seed; threads: torch threads;
    epochs: epoch cap; stop_after: early-stopping patience (epochs); batch, lr: training batch size and learning
    rate; hidden, transforms, model, bins: flow architecture (see make_builder). Writes estimator_seed<seed>.pt and
    train_seed<seed>.json in <out>/<kind> and returns a one-line report.
    """
    import torch
    from sbi.inference import NPE
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    np.random.seed(seed)
    folder = Path(out) / kind
    features = np.load(folder / 'features.npz')
    u = to_u(features['theta_pool'], features['low'], features['high'])
    builder = make_builder(model, hidden, transforms, bins)
    inference = NPE(prior=logit_prior(u.shape[1]), density_estimator=builder, device='cpu',
                    show_progress_bars=False)
    inference.append_simulations(torch.tensor(u, dtype=torch.float32),
                                 torch.tensor(features['x_pool'], dtype=torch.float32))
    t0 = time.monotonic()
    # early stopping: training ends after stop_after epochs without improvement of the validation loss
    estimator = inference.train(training_batch_size=batch, learning_rate=lr,
                                validation_fraction=VALIDATION_FRACTION, stop_after_epochs=stop_after,
                                max_num_epochs=epochs, show_train_summary=False)
    torch.save(estimator.state_dict(), folder / f'estimator_seed{seed}.pt')
    summary = {k: (np.asarray(v).tolist() if np.ndim(v) else float(v)) for k, v in inference.summary.items()}
    settings = dict(epochs=epochs, stop_after=stop_after, batch=batch, lr=lr, hidden=hidden, transforms=transforms,
                    threads=threads, model=model, bins=bins)
    record = dict(kind=kind, seed=seed, seconds=time.monotonic() - t0, epochs=summary.get('epochs_trained'),
                  best_validation_loss=summary.get('best_validation_loss'),
                  validation_loss=summary.get('validation_loss'),
                  hit_epoch_cap=bool(np.ravel(summary.get('epochs_trained', [0]))[-1] >= epochs),
                  settings=settings)
    (folder / f'train_seed{seed}.json').write_text(json.dumps(record, indent=1) + '\n')
    return (f"{kind} seed {seed}: {record['seconds']:.0f} s, epochs {record['epochs']}, "
            f"cap hit {record['hit_epoch_cap']}")


def cmd_train(args):
    """Train every (kind, seed) pair in parallel spawned processes (the 'train' subcommand).

    Noisy kinds use --seeds, kinds starting with 'clean' use --seeds-clean. With --features-from, each
    <out>/<kind> first gets symlinks to the features.npz, transform.npz and prepare.json of <out>/<features-from>
    (an architecture variant trained on the same features).
    """
    if args.features_from:
        for kind in args.kinds:
            folder = args.out / kind
            folder.mkdir(parents=True, exist_ok=True)
            for name in ('features.npz', 'transform.npz', 'prepare.json'):
                source = args.out / args.features_from / name
                if not (folder / name).exists() and source.exists():
                    (folder / name).symlink_to(source.resolve())
    jobs = []
    for kind in args.kinds:
        for seed in (args.seeds_clean if kind.startswith('clean') else args.seeds):
            jobs.append((kind, seed))
    with cf.ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('spawn')) as pool:
        futures = {pool.submit(train_one, str(args.out), kind, seed, args.threads, args.epochs, args.stop_after,
                               args.batch, args.lr, args.hidden, args.transforms, args.model, args.bins):
                   (kind, seed) for kind, seed in jobs}
        for fut in cf.as_completed(futures):
            print(fut.result(), flush=True)


class Ensemble:
    """Equal-weight mixture of the trained seeds, in logit space.

    Attributes: f (the features.npz NpzFile), low and high (dim,) prior bounds, seeds (sorted ints), settings (the
    training settings of the first seed) and nets (the sbi flows, eval mode). The methods use only nets, low and
    high, so other scripts (merlin_compression.ensemble) fill these attributes on an Ensemble.__new__ instance.
    """

    def __init__(self, folder, threads):
        """Load features.npz and every estimator_seed<s>.pt of folder (a Path) with torch using threads threads."""
        import torch
        torch.set_num_threads(threads)
        self.f = np.load(folder / 'features.npz')
        self.low, self.high = self.f['low'], self.f['high']
        weight_files = folder.glob('estimator_seed*.pt')
        self.seeds = sorted(int(re.search(r'seed(\d+)', p.name).group(1)) for p in weight_files)
        self.settings = json.loads((folder / f'train_seed{self.seeds[0]}.json').read_text())['settings']
        u_pool = to_u(self.f['theta_pool'], self.low, self.high)
        self.nets = []
        for seed in self.seeds:
            net = build_estimator(u_pool, self.f['x_pool'], self.settings['hidden'], self.settings['transforms'],
                                  self.settings.get('model', 'maf'), self.settings.get('bins', 10))
            net.load_state_dict(torch.load(folder / f'estimator_seed{seed}.pt'))
            net.eval()
            self.nets.append(net)

    def sample_u(self, x, n, max_per_call=100_000, rng=None):
        """(n, rows, dim) u-space samples, n // len(nets) from each seed, shuffled along the sample axis.

        x: (rows, xdim) or (xdim,) features; n: samples per row (n // len(nets) * len(nets) are returned);
        max_per_call: largest number of samples per network call (memory); rng: numpy Generator of the shuffle
        (default_rng(0) when None). Returns float32.
        """
        import torch
        x = torch.as_tensor(np.atleast_2d(x), dtype=torch.float32)
        per_seed = n // len(self.nets)
        rows = x.shape[0]
        rows_per_call = max(1, max_per_call // per_seed)
        parts = []
        with torch.no_grad():
            for net in self.nets:
                chunks = [net.sample((per_seed,), condition=x[i:i + rows_per_call]).numpy()
                          for i in range(0, rows, rows_per_call)]
                parts.append(np.concatenate(chunks, axis=1))
        u = np.concatenate(parts, axis=0)
        if rng is None:
            rng = np.random.default_rng(0)
        # the samples are stacked seed by seed: an independent random permutation per row mixes the seeds
        order = np.argsort(rng.random(u.shape[:2]), axis=0)
        return np.take_along_axis(u, order[:, :, None], axis=0)

    def log_prob_u(self, u, x, max_per_call=100_000):
        """log of the ensemble density in u space, for u of shape (n, rows, dim) and x (rows, xdim).

        Row j of u is conditioned on row j of x; u is evaluated in float32. max_per_call: largest number of points
        per network call. Returns (n, rows).
        """
        import torch
        x = torch.as_tensor(np.atleast_2d(x), dtype=torch.float32)
        u = np.asarray(u, dtype=np.float32)
        samples_per_call = max(1, max_per_call // u.shape[1])
        terms = []
        with torch.no_grad():
            for net in self.nets:
                chunks = [net.log_prob(torch.from_numpy(u[i:i + samples_per_call]), condition=x).numpy()
                          for i in range(0, u.shape[0], samples_per_call)]
                terms.append(np.concatenate(chunks, axis=0))
        # equal-weight mixture: log of the mean of the seed densities
        return np.logaddexp.reduce(np.stack(terms), axis=0) - np.log(len(self.nets))

    def posterior(self, x, n, rng=None):
        """theta samples (n, rows, dim) and the ensemble's log density of each sample in u (logit) space.

        x: (rows, xdim) features; n: samples per row; rng: Generator of the shuffle (see sample_u). Returns theta
        (n, rows, dim) float64 inside the prior box, the u-space log density (n, rows) and the u samples.

        The mode is taken in u space, where the network models the posterior: converting to theta space divides
        by d theta / d u, which vanishes at the prior edges, so a theta-space density estimated from samples spikes
        at the box edges whenever the learned tails are heavier than the logistic prior's.

        Flows in logit space can put excess posterior mass in the outer ~1 % of each prior range (a known artefact
        of the learned tails; see the edge_mass_ratio diagnostic in sbi_final.row_metrics).
        """
        u = self.sample_u(x, n, rng=rng)
        return from_u(u, self.low, self.high), self.log_prob_u(u, x), u

    def hpd_level(self, x, truth, n=4000, rng=None):
        """Fraction of posterior mass denser than the truth (u-space HPD regions; 0 = mode, 1 = tail).

        x: (rows, xdim) features; truth: (rows, dim) or (dim,) parameters; n: samples per row. Returns (rows,).
        Uniform on [0, 1] over simulations for a calibrated posterior.
        """
        u = self.sample_u(x, n, rng=rng)
        lp_samples = self.log_prob_u(u, x)
        u_truth = to_u(np.atleast_2d(truth), self.low, self.high)
        lp_truth = self.log_prob_u(u_truth[None], x)[0]
        return np.mean(lp_samples > lp_truth[None], axis=0)


def summarize(samples, lp_u, truth=None):
    """Marginal summary of one posterior; 'map' is the sample of highest ensemble density in u (logit) space.

    samples: (n, dim) theta samples; lp_u: (n,) their u-space log densities; truth: optional (dim,) parameters.
    Returns a dict of lists: median, q16, q84, q025, q975, mean, map and, with truth, truth and truth_in_68
    (truth inside the marginal 16-84 % interval).
    """
    q = np.quantile(samples, [.025, .16, .5, .84, .975], axis=0)
    out = dict(median=q[2].tolist(), q16=q[1].tolist(), q84=q[3].tolist(), q025=q[0].tolist(), q975=q[4].tolist(),
               mean=samples.mean(0).tolist(), map=samples[int(np.argmax(lp_u))].tolist())
    if truth is not None:
        out['truth'] = [float(value) for value in truth]
        out['truth_in_68'] = ((truth >= q[1]) & (truth <= q[3])).tolist()
    return out


def cmd_evaluate(args):
    """Held-out diagnostics and observation posteriors of one trained kind (the 'evaluate' subcommand).

    Held-out rows (--samples per row): RMSE of the posterior mean over that of the prior mean, Pearson r, marginal
    68/95 % coverage, SBC ranks and TARP (sbi), u-space HPD coverage and the mean log posterior density of the
    truth in theta space. Observations (--obs-samples each): summaries and the HPD level of the B12 truth.
    Writes metrics.json, recovered.json, heldout.npz, posterior_<label>.npy and six figures in <out>/<kind>.
    """
    import torch
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sbi.diagnostics import check_sbc, check_tarp
    from sbi.diagnostics.sbc import _run_sbc
    from sbi.diagnostics.tarp import _run_tarp, get_tarp_references
    from sbi.utils.metrics import l2
    from sbi.analysis.plot import sbc_rank_plot
    import analyze_64k as A
    import plot_posteriors as PP
    from diagnostics_64k import style, pearson_r, LABELS, PARAM_COLORS
    style()
    folder = args.out / args.kind
    ens = Ensemble(folder, args.threads)
    f, low, high = ens.f, ens.low, ens.high
    width = high - low
    truth = f['theta_held']
    rows = len(truth)
    n_train = len(f['x_pool'])
    rng = np.random.default_rng(1)
    t0 = time.monotonic()
    theta_s, lp_u_s, u_s = ens.posterior(f['x_held'], args.samples, rng=rng)       # (N, rows, 9)
    n_samples = theta_s.shape[0]                            # N // seeds per seed: the count actually drawn
    mean, std = theta_s.mean(0), theta_s.std(0)
    q16, q84 = np.quantile(theta_s, [.16, .84], axis=0)
    q025, q975 = np.quantile(theta_s, [.025, .975], axis=0)
    # RMSE of the posterior mean in prior widths, and that of the prior mean for reference (ratio 1: no gain)
    prior_rmse = np.sqrt(np.mean(((truth - (low + high) / 2) / width) ** 2, axis=0))
    rmse = np.sqrt(np.mean(((mean - truth) / width) ** 2, axis=0))
    # SBC: rank of each true parameter among its posterior samples (uniform for a calibrated posterior)
    thetas_t = torch.tensor(truth, dtype=torch.float32)
    samples_t = torch.tensor(theta_s, dtype=torch.float32)
    ranks = _run_sbc(thetas_t, torch.zeros(rows, 1), samples_t, 'marginals', False)
    check = check_sbc(ranks, thetas_t, samples_t[0], num_posterior_samples=n_samples)
    # TARP: expected coverage of balls around random reference points, a joint (not marginal) calibration test
    torch.manual_seed(1)
    references = get_tarp_references(thetas_t)
    ecp, alpha = _run_tarp(samples_t, thetas_t, references, l2, None, True)
    atc, _ = check_tarp(ecp, alpha)
    c68, c95 = np.interp([0.68, 0.95], np.ravel(alpha.numpy()), np.ravel(ecp.numpy()))
    u_truth = to_u(truth, low, high)
    lp_u_t = ens.log_prob_u(u_truth[None], f['x_held'])[0]
    # expected log posterior density of the true parameters on held-out simulations, in theta space (the NPE loss;
    # it rewards sharp and calibrated posteriors together, so it ranks configurations)
    lp_theta_t = lp_u_t - log_jacobian_theta_per_u(u_truth, low, high)
    hpd = np.mean(lp_u_s > lp_u_t[None], axis=0)
    r_vals = [pearson_r(truth[:, i], mean[:, i]) for i in range(9)]
    print(f'{args.kind}: held-out posteriors in {time.monotonic() - t0:.0f} s', flush=True)

    # observations
    obs_res, obs_samples = {}, {}
    for i, label in enumerate(f['obs_labels']):
        samples, lp_u, _ = ens.posterior(f['x_obs'][i][None], args.obs_samples, rng=np.random.default_rng(10 + i))
        samples, lp_u = samples[:, 0], lp_u[:, 0]
        obs_samples[str(label)] = samples
        np.save(folder / f'posterior_{label}.npy', samples.astype(np.float32))
        obs_res[str(label)] = summarize(samples, lp_u, A.B12 if label == 'Battaglia12' else None)
    b12_hpd = ens.hpd_level(f['x_obs'][0][None], A.B12, n=8000, rng=np.random.default_rng(3))
    obs_res['Battaglia12']['truth_hpd_level'] = float(b12_hpd[0])
    b12_res = obs_res['Battaglia12']
    train_info = [json.loads((folder / f'train_seed{s}.json').read_text()) for s in ens.seeds]
    training = [dict(seed=t['seed'], epochs=t['epochs'], seconds=t['seconds'], hit_epoch_cap=t['hit_epoch_cap'])
                for t in train_info]
    metrics = dict(kind=args.kind, seeds=ens.seeds, heldout_rows=rows, training_rows=int(n_train),
                   training=training, rmse_over_prior_mean_rmse=(rmse / prior_rmse).tolist(),
                   pearson_r_truth_vs_mean=r_vals,
                   marginal_coverage68=np.mean((truth >= q16) & (truth <= q84), axis=0).tolist(),
                   marginal_coverage95=np.mean((truth >= q025) & (truth <= q975), axis=0).tolist(),
                   sbc_ks_pvalues=check['ks_pvals'].numpy().tolist(),
                   sbc_c2st_ranks=check['c2st_ranks'].numpy().tolist(),
                   tarp_atc=float(atc), tarp_coverage_at_68=float(c68), tarp_coverage_at_95=float(c95),
                   hpd_joint_coverage={str(c): float(np.mean(hpd <= c)) for c in (0.5, 0.68, 0.9, 0.95)},
                   median_width68_over_prior=np.median((q84 - q16) / width, axis=0).tolist(), observations=obs_res,
                   heldout_mean_log_prob_theta=float(np.mean(lp_theta_t)),
                   heldout_mean_log_prob_theta_sem=float(np.std(lp_theta_t) / np.sqrt(rows)),
                   settings=ens.settings,
                   b12_width68_over_prior=((np.array(b12_res['q84']) - np.array(b12_res['q16'])) / width).tolist())
    (folder / 'metrics.json').write_text(json.dumps(metrics, indent=1) + '\n')
    recovered = {k: dict(map=v['map'], median=v['median'], mean=v['mean']) for k, v in obs_res.items()}
    (folder / 'recovered.json').write_text(json.dumps(recovered, indent=1) + '\n')
    np.savez_compressed(folder / 'heldout.npz', rows=f['held'], truth=truth, mean=mean, std=std, q16=q16, q84=q84,
                        ranks=ranks.numpy(), hpd_level=hpd, ecp=ecp.numpy(), alpha=alpha.numpy())

    kind_text = 'SO mock (signal + noise)' if args.kind.startswith('noisy') else 'noise-free (clean spectra)'
    compression_text = f", {args.kind.split('_', 1)[1]}" if '_' in args.kind else ', MOPED'
    model_name = ens.settings.get('model', 'maf').upper()
    network_text = f", {model_name} {ens.settings['hidden']}x{ens.settings['transforms']}"
    tag = kind_text + compression_text + network_text
    ensemble_text = f'ensemble of {len(ens.seeds)} seeds'
    # corner plots (with the flat prior), via plot_posteriors.corner
    PP.corner([(obs_samples['Battaglia12'], ensemble_text, '#1f77b4')], truth=A.B12,
              title=f'Battaglia+12 control, {tag}: logit-space NPE, {n_train:,} training rows; '
                    'diagonal: median and 68 % interval',
              path=folder / 'fig_battaglia12.png')
    PP.corner([(obs_samples[label], name, color) for label, name, color in PP.FLAMINGO], truth=A.B12,
              title=f'FLAMINGO y-maps, {tag}: effective Battaglia+12 parameters ({n_train:,} training rows); '
                    'diagonal: median and 68 % interval',
              path=folder / 'fig_flamingo.png')
    # SBC rank histograms and CDFs (sbi)
    fig, ax = sbc_rank_plot(ranks, n_samples, num_bins=20, plot_type='hist', parameter_labels=LABELS,
                            ranks_labels=[ensemble_text], num_cols=3, figsize=(12, 9.5))
    fig.subplots_adjust(hspace=0.42, wspace=0.12, top=0.90)
    fig.suptitle(f'SBC rank histograms (sbi.analysis.sbc_rank_plot), {tag}, logit-space NPE, N = {n_train:,}\n'
                 f'{rows} held-out simulations; grey band: 99 % range for a calibrated posterior', fontsize=10)
    fig.savefig(folder / 'sbc_ranks_hist.png')
    plt.close(fig)
    fig, ax = sbc_rank_plot([ranks], n_samples, plot_type='cdf', parameter_labels=LABELS,
                            ranks_labels=['ensemble'], colors=['#2ca02c'], num_cols=3, figsize=(12, 10),
                            params_in_subplots=True)
    fig.subplots_adjust(hspace=0.42, wspace=0.12, top=0.90)
    fig.suptitle(f'SBC rank CDFs (sbi), {tag}, logit-space NPE, N = {n_train:,}, {rows} held-out simulations',
                 fontsize=10)
    fig.savefig(folder / 'sbc_ranks_cdf.png')
    plt.close(fig)
    # TARP expected coverage against credibility level
    fig, ax = plt.subplots(figsize=(10 / 2.54, 9.5 / 2.54))
    ax.plot([0, 1], [0, 1], 'k:', lw=0.9, label='calibrated')
    ax.plot(np.ravel(alpha.numpy()), np.ravel(ecp.numpy()), color='#2ca02c', lw=1.5,
            label=f'ensemble: 68 % level covers {100 * c68:.0f} %, 95 % covers {100 * c95:.0f} %')
    ax.set_xlabel('credibility level')
    ax.set_ylabel('expected coverage')
    ax.set_title(f'TARP joint coverage (sbi TARP, ATC {float(atc):+.3f})\n{tag}, logit-space NPE', fontsize=9)
    ax.text(0.97, 0.03, 'below the diagonal: overconfident\nabove: underconfident', ha='right', va='bottom',
            fontsize=7, transform=ax.transAxes)
    ax.legend(frameon=False, fontsize=6.5, loc='upper left')
    ax.grid(True, alpha=0.25, lw=0.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    fig.tight_layout()
    fig.savefig(folder / 'tarp.png')
    plt.close(fig)
    # truth against posterior mean, one panel per parameter, with the B12 observation as a star
    b12 = obs_samples['Battaglia12']
    fig, axes = plt.subplots(3, 3, figsize=(18 / 2.54, 17 / 2.54))
    axes = axes.ravel()
    for i, ax in enumerate(axes):
        ax.errorbar(truth[:, i], mean[:, i], yerr=std[:, i], fmt='none', ecolor=PARAM_COLORS[0], alpha=0.10,
                    elinewidth=0.45, zorder=1)
        ax.scatter(truth[:, i], mean[:, i], s=9, color=PARAM_COLORS[0], alpha=0.7, edgecolor='none', zorder=2)
        ax.errorbar([A.B12[i]], [b12[:, i].mean()], yerr=[b12[:, i].std()], fmt='*', color='k', ms=9, elinewidth=1,
                    zorder=4, label='Battaglia+12 observation' if i == 0 else None)
        ax.plot([low[i], high[i]], [low[i], high[i]], color='k', lw=0.8, ls=':', zorder=0)
        ax.axhline((low[i] + high[i]) / 2, color='gray', lw=0.6, ls='--', zorder=0)
        ax.set_xlim(low[i], high[i])
        ax.set_ylim(low[i], high[i])
        ax.set_title(rf"{LABELS[i]}  $r={r_vals[i]:.2f}$", pad=2)
        ax.set_xlabel('True')
        ax.set_ylabel('Posterior mean')
        ax.grid(True, alpha=0.25, lw=0.5)
    axes[0].legend(loc='upper left', frameon=False, fontsize=7)
    fig.suptitle(f'Truth vs posterior mean on {rows} held-out simulations: {tag}, logit-space NPE, '
                 f'N = {n_train:,}, {ensemble_text}\n(error bars: posterior std; dashed: prior mean)', fontsize=9)
    fig.tight_layout()
    fig.savefig(folder / 'truth_vs_mean.png')
    plt.close(fig)
    shown = ('tarp_coverage_at_68', 'tarp_coverage_at_95', 'hpd_joint_coverage', 'pearson_r_truth_vs_mean',
             'rmse_over_prior_mean_rmse', 'heldout_mean_log_prob_theta', 'b12_width68_over_prior')
    print(json.dumps({k: metrics[k] for k in shown}, indent=1))
    print('B12 truth HPD level', b12_res['truth_hpd_level'], 'MAP', np.round(b12_res['map'], 3).tolist())


def cmd_realizations(args):
    """Posteriors of the B12 SO mock under many noise realizations of one sky (the 'realizations' subcommand).

    Uses the noisy MOPED ensemble of <out>/noisy and every <spectra>/noisy_cross_<s1>_<s2>.npy (C_ell of one noise
    draw, indexed by ell); the fitted observation (noise seeds 39001001, 39001002) goes first and is checked
    against the stored observation. Writes b12_realizations.json (per-draw summaries, truth HPD levels, scatter of
    the medians over the mean 68 % half-width, coverage fractions) and fig_b12_noise_realizations.png in
    <out>/noisy.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import analyze_64k as A
    from diagnostics_64k import style, LABELS
    style()
    folder = args.out / 'noisy'
    ens = Ensemble(folder, args.threads)
    low, high = ens.low, ens.high
    transform = dict(np.load(folder / 'transform.npz'))
    transform['method'] = 'moped_fixed'
    files = sorted(args.spectra.glob('noisy_cross_*.npy'))
    noise_seeds = [tuple(map(int, re.findall(r'(\d+)_(\d+)', p.stem)[0])) for p in files]
    # the fitted observation first, then the new draws in seed order
    order = sorted(range(len(files)), key=lambda k: (noise_seeds[k] != FITTED_NOISE_SEEDS, noise_seeds[k]))
    files = [files[k] for k in order]
    noise_seeds = [noise_seeds[k] for k in order]
    dl = np.stack([np.load(p)[ELL] * DL for p in files])
    obs = np.load(A.TEST256 / 'cluster_snapshot' / 'diagnostic_256' / 'observations' / 'Battaglia12.npz')
    check = None
    if noise_seeds[0] == FITTED_NOISE_SEEDS:
        # largest difference to the stored observation, relative to |noisy| + |clean| at each multipole
        reference = np.abs(obs['noisy_dl_unbinned']) + np.abs(obs['clean_dl_unbinned'])
        check = float(np.max(np.abs(dl[0] - obs['noisy_dl_unbinned']) / reference))
    x = A.project(transform, dl)
    rows = []
    for k in range(len(files)):
        samples, lp_u, _ = ens.posterior(x[k][None], args.samples, rng=np.random.default_rng(100 + k))
        samples, lp_u = samples[:, 0], lp_u[:, 0]
        res = summarize(samples, lp_u, A.B12)
        truth_hpd = ens.hpd_level(x[k][None], A.B12, n=4000, rng=np.random.default_rng(200 + k))
        res['truth_hpd_level'] = float(truth_hpd[0])
        res['seeds'] = list(noise_seeds[k])
        rows.append(res)
    med = np.array([r['median'] for r in rows])
    lo = np.array([r['q16'] for r in rows])
    hi = np.array([r['q84'] for r in rows])
    hpd = np.array([r['truth_hpd_level'] for r in rows])
    summary = dict(realizations=len(rows), observation_reproduced_max_rel_diff=check,
                   median_scatter_over_mean_halfwidth=(med.std(0) / np.mean((hi - lo) / 2, axis=0)).tolist(),
                   fraction_truth_in_marginal68=np.mean((A.B12 >= lo) & (A.B12 <= hi), axis=0).tolist(),
                   truth_hpd_levels=hpd.tolist(), fraction_truth_in_joint68=float(np.mean(hpd <= 0.68)),
                   fraction_truth_in_joint95=float(np.mean(hpd <= 0.95)), rows=rows)
    (folder / 'b12_realizations.json').write_text(json.dumps(summary, indent=1) + '\n')

    # medians and 68 % intervals per parameter: the fitted observation in red, new noise draws in blue
    fig = plt.figure(figsize=(18 / 2.54, 21 / 2.54))
    gs = fig.add_gridspec(4, 3, height_ratios=[1, 1, 1, 1.05], hspace=0.55, wspace=0.42)
    idx = np.arange(len(rows))
    for i in range(9):
        ax = fig.add_subplot(gs[i // 3, i % 3])
        ax.axhspan(low[i], high[i], color='0.93', zorder=0)
        ax.errorbar(idx[1:], med[1:, i], yerr=[med[1:, i] - lo[1:, i], hi[1:, i] - med[1:, i]], fmt='o', ms=2.5,
                    color='#1f77b4', elinewidth=0.8, label='new noise draws' if i == 0 else None)
        ax.errorbar(idx[:1], med[:1, i], yerr=[med[:1, i] - lo[:1, i], hi[:1, i] - med[:1, i]], fmt='s', ms=3.5,
                    color='#d62728', elinewidth=1.0, label='fitted observation' if i == 0 else None)
        ax.axhline(A.B12[i], color='k', ls='--', lw=0.9, label='Battaglia+12' if i == 0 else None)
        ax.set_ylim(low[i], high[i])
        ax.set_xlim(-1, len(rows))
        n_in68 = int(np.sum((A.B12[i] >= lo[:, i]) & (A.B12[i] <= hi[:, i])))
        ax.set_title(f"{LABELS[i]}: in 68 % for {n_in68}/{len(rows)}", fontsize=8, pad=2)
        ax.tick_params(labelsize=7)
        if i // 3 == 2:
            ax.set_xlabel('noise realization', fontsize=8)
        if i == 0:
            ax.legend(frameon=False, fontsize=6, loc='upper right')
    # joint HPD level of the truth for every draw
    ax = fig.add_subplot(gs[3, 0])
    ax.bar(idx, hpd, color=['#d62728'] + ['#1f77b4'] * (len(rows) - 1), width=0.8)
    ax.axhline(0.68, color='k', ls=':', lw=0.9)
    ax.axhline(0.95, color='k', ls='--', lw=0.9)
    ax.set_ylim(0, 1)
    ax.set_xlabel('noise realization', fontsize=8)
    ax.set_ylabel('HPD level of the truth', fontsize=8)
    ax.set_title(f'joint: inside 68 % for {int(np.sum(hpd <= 0.68))}/{len(rows)}, '
                 f'95 % for {int(np.sum(hpd <= 0.95))}/{len(rows)}', fontsize=8, pad=2)
    ax.tick_params(labelsize=7)
    # the 40-band spectra of all draws against the clean masked B12 spectrum
    ax = fig.add_subplot(gs[3, 1:])
    binned = A_rebin(dl)
    clean = A_rebin(np.load(args.spectra / 'masked_clean_cl.npy')[ELL][None] * DL)[0]
    centres = np.array([np.mean(b) for b in bins_edges()])
    for k in range(1, len(rows)):          # log axis: negative (noise-dominated) band powers drawn open at |D_ell|
        negative = binned[k] <= 0
        ax.plot(centres, np.where(binned[k] > 0, binned[k], np.nan), color='#1f77b4', lw=0.5, alpha=0.5)
        ax.plot(centres[negative], -binned[k][negative], 'o', ms=2.5, mfc='white', mec='#1f77b4', mew=0.6,
                alpha=0.7)
    negative = binned[0] <= 0
    ax.plot(centres, np.where(binned[0] > 0, binned[0], np.nan), color='#d62728', lw=1.1,
            label='fitted observation')
    ax.plot(centres[negative], -binned[0][negative], 'o', ms=3.5, mfc='white', mec='#d62728', mew=0.9)
    ax.plot(centres, clean, color='k', lw=1.4, label='clean Battaglia+12 (masked)')
    ax.plot([], [], 'o', ms=3, mfc='white', mec='0.4', mew=0.8,
            label=r'open: negative band power, $|D_\ell|$ shown')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$\ell$', fontsize=8)
    ax.set_ylabel(r'$D_\ell$ (40 bins)', fontsize=8)
    ax.set_xticks([200, 500, 1000, 2000, 5000])
    ax.set_xticklabels(['200', '500', '1000', '2000', '5000'])
    ax.minorticks_off()
    ax.set_title('SO mocks of Battaglia+12: one clean sky, different noise', fontsize=8, pad=2)
    ax.legend(frameon=False, fontsize=6.5)
    ax.tick_params(labelsize=7)
    fig.suptitle(f'Battaglia+12 recovery across {len(rows)} SO noise realizations of the same sky '
                 f'(logit-space NPE ensemble, noisy)\n'
                 f'points: posterior median with 68 % interval; grey band: prior range', fontsize=9)
    fig.savefig(folder / 'fig_b12_noise_realizations.png')
    plt.close(fig)
    print(json.dumps({k: v for k, v in summary.items() if k != 'rows'}, indent=1))


def bins_edges():
    """The multipoles of each of the 40 bands: a list of 40 integer arrays (ell = 80..7979 in total)."""
    return [np.arange(a, b) for a, b in zip(BAND_EDGES[:-1], BAND_EDGES[1:])]


def A_rebin(values):
    """Average unbinned spectra into the 40 bands with weights 2 ell + 1 (the number of modes per multipole).

    values: (rows, 7900) spectra on ELL, in any units (e.g. D_ell). Returns (rows, 40) band averages, same units.
    """
    bands = []
    for band in bins_edges():
        in_band = (ELL >= band[0]) & (ELL <= band[-1])
        bands.append(np.average(values[:, in_band], axis=1, weights=2 * band + 1))
    return np.stack(bands, axis=1)


def main():
    """Command-line entry point: prepare, train, evaluate or realizations (see the module docstring)."""
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)
    a = sub.add_parser('prepare')
    a.add_argument('--dataset', type=Path, required=True)
    a.add_argument('--out', type=Path, required=True)
    a.add_argument('--heldout-rows', type=int, default=768)
    a.add_argument('--compressions', nargs='*', default=[],
                   help='bins40, pca9, ...: written to <out>/<kind>_<method>')
    a.add_argument('--kinds', nargs='+', default=['noisy'])
    a = sub.add_parser('train')
    a.add_argument('--out', type=Path, required=True)
    a.add_argument('--kinds', nargs='+', default=['noisy', 'clean'])
    a.add_argument('--seeds', nargs='+', type=int, default=[1, 2, 3, 4, 5])
    a.add_argument('--seeds-clean', nargs='+', type=int, default=[1, 2, 3])
    a.add_argument('--workers', type=int, default=3)
    a.add_argument('--threads', type=int, default=3)
    a.add_argument('--epochs', type=int, default=2000)
    a.add_argument('--stop-after', type=int, default=30)
    a.add_argument('--batch', type=int, default=512)
    a.add_argument('--lr', type=float, default=7e-4)
    a.add_argument('--hidden', type=int, default=64)
    a.add_argument('--transforms', type=int, default=5)
    a.add_argument('--model', default='maf', choices=['maf', 'nsf'])
    a.add_argument('--bins', type=int, default=10)
    a.add_argument('--features-from', default=None, help='folder whose features an architecture variant reuses')
    a = sub.add_parser('evaluate')
    a.add_argument('--out', type=Path, required=True)
    a.add_argument('--kind', required=True)
    a.add_argument('--samples', type=int, default=1000)
    a.add_argument('--obs-samples', type=int, default=20000)
    a.add_argument('--threads', type=int, default=12)
    a = sub.add_parser('realizations')
    a.add_argument('--out', type=Path, required=True)
    a.add_argument('--spectra', type=Path, required=True)
    a.add_argument('--samples', type=int, default=10000)
    a.add_argument('--threads', type=int, default=12)
    args = p.parse_args()
    commands = dict(prepare=cmd_prepare, train=cmd_train, evaluate=cmd_evaluate, realizations=cmd_realizations)
    commands[args.cmd](args)


if __name__ == '__main__':
    main()
