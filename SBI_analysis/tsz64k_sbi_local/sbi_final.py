"""SBI on the finished tSZ_64k rows: convergence with training-set size, truth-mean correlation, SO baseline/goal.

The 64,440 flat-prior rows of data/rows_64440 (9 pressure-profile parameters theta, 40 band powers D_ell of the
clean spectrum and of the SO mock, and the fixed-B12 MOPED projections of the mock) are split once into 3,000 fixed
test rows (random, seed 0) and a training pool of 61,440 rows. A random permutation of the pool (seed 1) defines
nested training sets N = 2,000, 4,000, ..., 61,440: every set is a prefix of the next. For every variant and N a
3-seed NPE ensemble in logit space is trained with train_ensemble.py and evaluated on the same test rows.

Commands (every output goes below --out, default results_final/):
  sbi_final.py prepare  [--data data/rows_64440] [--asinh-scale median|noise]
      -> split.npz, noise_model.json, prepare.json, <variant>/N<size>/features.npz
  sbi_final.py train    --variants ... [--workers 6]     train_ensemble.py train in every N folder
  sbi_final.py evaluate --variants ...                   -> <variant>/N<size>/metrics.json
  sbi_final.py report                                    -> fig_convergence.png, fig_correlation_vs_N.png,
                                                            convergence.json, convergence_table.md
  sbi_final.py checks                                    training scatter, B12 against a linear Fisher forecast,
                                                         network capacity
                                                         -> checks.json, checks_rows.npz, fig_checks.png
  sbi_final.py validate [--validation data/rows_validation_1096]
                                                         independent rows regenerated on idark
                                                         -> validation.json, fig_validation.png
Other inputs: the B12 observation and the fixed MOPED matrix of the 256-row test (analyze_64k.TEST256), the SO
map-noise mocks of that observation (mocks/pass1_B12/B12), the SO noise curves (SO below) and, for checks, the B12
Fisher matrix of fisher_b12.py (results_logit/).

Features (as in train_ensemble.py): 40 bands with a per-band signed asinh and z-score, or the 9 fixed-B12 MOPED
numbers with the same transform, all fitted on the whole training pool. asinh(x / scale) is linear for |x| << scale
and logarithmic above. --asinh-scale median (default) takes the median |clean D_ell| of each band (median |raw| per
MOPED component); --asinh-scale noise takes the baseline noise sd at zero signal, sqrt(a_b) = N_b / sqrt(nu_b), for
the four 40-band variants and 1 for MOPED (its matrix is normalized to unit noise at B12).
Variants
  bins40_noisy      40 bands of the SO mock (baseline Deproj-0), MAF 128x5
  moped_noisy       fixed-B12 MOPED of the SO mock, MAF 64x5
  bins40_clean      40 bands without noise (the zero-noise limit), MAF 128x5
  bins40_gbase      clean 40 bands + Gaussian band noise with the baseline level (validates the noise model)
  bins40_ggoal      the same draws scaled to SO goal (Deproj-0) noise
Noise model per band b (fitted on the training pool, no maps needed): the split cross spectrum at fixed sky has
Var = (2 S N + N^2) / nu, so E[(noisy - clean)^2] = a + s S with N = 2a/s, nu = 4a/s^2. Goal: N scaled by the
band average of N_ell(goal)/N_ell(baseline) from the SO files; nu unchanged.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

import analyze_64k as A
from train_ensemble import fit_binned, project_binned, ELL, DL, A_rebin

HERE = Path(__file__).resolve().parent
# nested training-set sizes; the whole pool (61,440 rows for the finished data) is appended
BASE_SIZES = [2000, 4000, 8000, 16000, 32000]
# fixed test rows, the same for every variant and N
TEST_ROWS = 3000
# network size per variant: MAF with `hidden` units and `transforms` layers
VARIANTS = dict(bins40_noisy=dict(hidden=128, transforms=5), moped_noisy=dict(hidden=64, transforms=5),
                bins40_clean=dict(hidden=128, transforms=5), bins40_gbase=dict(hidden=128, transforms=5),
                bins40_ggoal=dict(hidden=128, transforms=5))
# SO LAT noise curves (Deproj-0 ILC tSZ, fsky 0.4), baseline and goal
SO = (HERE.parent.parent / 'tSZ_64k_flat_prior_SO_baseline_deproj0' / 'halfdome_sources' / 'code' / 'halfdome'
      / 'other_sims' / 'SO')
# edges in ell of the 40 bands, 200 wide from 80 to 7980
EDGES = np.r_[np.arange(80, 7881, 200), 7980]
# SO map-noise mocks of the B12 observation; the fitted one is the observation itself
B12_MOCKS = HERE / 'mocks' / 'pass1_B12' / 'B12'
B12_FITTED_MOCK = 'noisy_cross_39001001_39001002.npy'
# seeds of the standard-normal draws of the Gaussian-noise variants: all rows (prepare and variant_features must
# agree) and the B12 observation (prepare and checks must agree)
GAUSS_ROWS_SEED = 2
GAUSS_OBS_SEED = 3
# central 68.27 % interval of a flat prior as a fraction of its range (68 % width of an uninformative posterior)
FLAT_WIDTH68 = 0.6827
# width of the prior-edge zone of edge_mass_ratio, as a fraction of the range at each end (flat prior: 2 * EDGE)
EDGE = 0.01
# a saturating-fit parameter closer than this fraction of its allowed range to a bound counts as at the bound
FIT_BOUND_TOLERANCE = 1e-3
# plot labels of the 9 parameters, in the order of theta
PARAM_LABELS = [r'$P_0$', r'$x_{\rm c}$', r'$\beta$', r'$\alpha_{m,P_0}$', r'$\alpha_{m,x_{\rm c}}$',
                r'$\alpha_{m,\beta}$', r'$\alpha_{z,P_0}$', r'$\alpha_{z,x_{\rm c}}$', r'$\alpha_{z,\beta}$']
# colour, marker and legend name of each variant in the figures
COLORS = dict(bins40_noisy='#2a78d6', moped_noisy='#eb6834', bins40_clean='#1baf7a', bins40_gbase='#eda100',
              bins40_ggoal='#e87ba4')
MARKS = dict(bins40_noisy='o', moped_noisy='s', bins40_clean='D', bins40_gbase='^', bins40_ggoal='v')
NAMES = dict(bins40_noisy='40 bands, SO baseline (map noise)', moped_noisy='MOPED, SO baseline (map noise)',
             bins40_clean='40 bands, no noise', bins40_gbase='40 bands, Gaussian noise at baseline',
             bins40_ggoal='40 bands, Gaussian noise at goal')


def so_band_noise(level):
    """Band-averaged SO Deproj-0 per-split noise: D_ell = N_ell l (l + 1) / 2 pi averaged with weights 2l + 1 over
    each of the 40 bands, as in the 40-band binning. level: 'baseline' or 'goal'. Returns (40,), file units."""
    table = np.loadtxt(SO / f'SO_LAT_Nell_T_atmv1_{level}_fsky0p4_ILC_tSZ.txt')
    ell, nl = table[:, 0], table[:, 1]
    dl = nl * ell * (ell + 1) / (2 * np.pi)
    return np.array([np.average(dl[(ell >= a) & (ell < b)], weights=2 * ell[(ell >= a) & (ell < b)] + 1)
                     for a, b in zip(EDGES[:-1], EDGES[1:])])


def fit_noise_model(clean40, noisy40, bins=60):
    """Fit the per-band noise model of the SO mock from the rows themselves (no maps needed).

    Around the clean band power S the split cross spectrum has variance (2 S N + N^2) / nu, so the squared residual
    r^2 = (noisy - clean)^2 has E[r^2] = a + s S with a = N^2 / nu and s = 2 N / nu. a and s come from a weighted
    straight-line fit of the mean r^2 in `bins` equal-count bins of S.

    clean40, noisy40: (rows, 40) band D_ell. Returns N (40,) in D_ell units, nu (40,) (effective number of modes)
    and the reduced chi^2 of each band's straight-line fit (list of 40 floats).
    """
    N, nu, reduced_chi2 = np.empty(40), np.empty(40), []
    for b in range(40):
        signal = clean40[:, b]
        resid2 = (noisy40[:, b] - clean40[:, b]) ** 2
        # equal-count bins of the clean signal
        edges = np.quantile(signal, np.linspace(0, 1, bins + 1))
        in_bin = np.clip(np.searchsorted(edges, signal, side='right') - 1, 0, bins - 1)
        mean_signal = np.array([signal[in_bin == k].mean() for k in range(bins)])
        mean_resid2 = np.array([resid2[in_bin == k].mean() for k in range(bins)])
        err = np.array([resid2[in_bin == k].std() / np.sqrt(np.sum(in_bin == k)) for k in range(bins)])
        # weighted least squares for mean_resid2 = a + s * mean_signal: every bin divided by its standard error
        design = np.c_[np.ones(bins), mean_signal] / err[:, None]
        a, s = np.linalg.lstsq(design, mean_resid2 / err, rcond=None)[0]
        a, s = max(a, 1e-300), max(s, 1e-300)                  # keeps N and nu positive
        # invert a = N^2 / nu, s = 2 N / nu
        N[b], nu[b] = 2 * a / s, 4 * a / s ** 2
        reduced_chi2.append(float(np.sum(((mean_resid2 - a - s * mean_signal) / err) ** 2) / (bins - 2)))
    return N, nu, reduced_chi2


def band_var(S, N, nu):
    """Variance of the split cross spectrum around the clean band power S, (2 S N + N^2) / nu (S clipped at 0).
    S: (..., 40) band D_ell; N, nu: (40,) from fit_noise_model. Returns an array shaped like S (D_ell units^2)."""
    return (2 * np.clip(S, 0, None) * N + N ** 2) / nu


def bins40_scale(choice, N_band, nu_band):
    """asinh scale of the 40-band features for --asinh-scale `choice`.

    'median': None, i.e. fit_binned's default (the median |clean D_ell| of each band over the training rows);
    'noise': the baseline noise sd at zero signal, sqrt(a_b) = N_b / sqrt(nu_b), since a_b = N_b^2 / nu_b.
    N_band, nu_band: (40,) baseline noise model. Returns None or (40,) in D_ell units.
    """
    if choice == 'median':
        return None
    if choice == 'noise':
        return N_band / np.sqrt(nu_band)
    raise ValueError(f'unknown asinh scale {choice!r}')


def fit_bins40(clean40, noisy40, scale):
    """train_ensemble.fit_binned('bins40', ...) on training rows (rows, 40) with the asinh scale of bins40_scale
    (None: fit_binned's default). Returns the state dict that project_binned takes."""
    if scale is None:
        return fit_binned('bins40', clean40, noisy40)[0]
    return fit_binned('bins40', clean40, noisy40, scale=scale)[0]


def fit_moped(raw_train, choice):
    """The MOPED feature transform, fitted on the training rows raw_train (rows, 9) of raw MOPED projections.

    Signed asinh with a per-component scale ('median': median |raw|; 'noise': 1, the unit noise of the fixed B12
    MOPED matrix), then a z-score with the training mean and sd. Returns a function that maps raw projections
    (rows, 9) to float32 features (rows, 9).
    """
    if choice == 'median':
        scale = np.maximum(np.median(np.abs(raw_train), axis=0), 1e-30)
    elif choice == 'noise':
        scale = np.ones(raw_train.shape[1])
    else:
        raise ValueError(f'unknown asinh scale {choice!r}')
    y = np.arcsinh(raw_train / scale)
    center = y.mean(0)
    spread = np.maximum(y.std(0), 1e-8)

    def transform(raw):
        return ((np.arcsinh(raw / scale) - center) / spread).astype(np.float32)
    return transform


def b12_observation():
    """The B12 observation of the 256-row test (dict-like npz: clean_dl, noisy_dl (40,), noisy_dl_unbinned (7900,))
    and its fixed MOPED matrix (7900, 9), which maps unbinned D_ell to the 9 raw MOPED numbers."""
    obs = np.load(A.TEST256 / 'cluster_snapshot' / 'diagnostic_256' / 'observations' / 'Battaglia12.npz')
    matrix = np.load(A.TEST256 / 'reference' / 'compression_context.npz')['Battaglia12_rcond1e-06']
    return obs, matrix


def cmd_prepare(args):
    """Split the rows, fit the noise model and write the features of every variant and training-set size.

    Reads --data (meta.json, theta.npy, clean_dl.npy, noisy_dl.npy, moped_raw_noisy.npy, row_id.npy). Writes to
    --out: split.npz (test rows, pool order, row ids), noise_model.json, <variant>/N<size>/features.npz (x_pool,
    theta_pool, x_held, theta_held, pool, held, low, high, obs_labels, x_obs) and prepare.json.
    """
    data = args.data
    choice = getattr(args, 'asinh_scale', 'median')
    meta = json.loads((data / 'meta.json').read_text())
    theta = np.load(data / 'theta.npy')                         # (rows, 9)
    clean40 = np.load(data / 'clean_dl.npy')                    # (rows, 40) band D_ell without noise
    noisy40 = np.load(data / 'noisy_dl.npy')                    # (rows, 40) band D_ell of the SO mock
    raw_moped = np.load(data / 'moped_raw_noisy.npy')           # (rows, 9) fixed-B12 MOPED of the SO mock
    low, high = np.array(meta['lower']), np.array(meta['upper'])
    n_rows = len(theta)
    # fixed test rows; the prefixes of a random order of the pool are the nested training sets
    perm = np.random.default_rng(0).permutation(n_rows)
    test, pool = np.sort(perm[:TEST_ROWS]), perm[TEST_ROWS:]
    order = pool[np.random.default_rng(1).permutation(len(pool))]
    train_sizes = [s for s in BASE_SIZES if s < len(order)] + [len(order)]
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez(args.out / 'split.npz', test=test, order=order, row_id=np.load(data / 'row_id.npy'))
    # B12 observation: the fitted SO mock of the 256-row test and its clean spectrum
    obs, matrix = b12_observation()
    obs_c40, obs_n40 = obs['clean_dl'][None], obs['noisy_dl'][None]

    # noise model from the training pool; the goal level scales N by the SO goal / baseline band ratio
    N, nu, redchi = fit_noise_model(clean40[order], noisy40[order])
    ratio = so_band_noise('goal') / so_band_noise('baseline')
    # check the modelled band sd at B12 against the other 20 SO map-noise mocks of the observation
    fitted = B12_MOCKS / B12_FITTED_MOCK
    other_mocks = [p for p in sorted(B12_MOCKS.glob('noisy_cross_*.npy')) if p != fitted]
    draws = A_rebin(np.stack([np.load(p)[ELL] * DL for p in other_mocks]))
    sd_draws = draws.std(0, ddof=1)
    sd_model = np.sqrt(band_var(obs_c40[0], N, nu))
    # Gaussian-noise variants: clean bands + N(0, band_var) noise at the baseline and the goal level
    z = np.random.default_rng(GAUSS_ROWS_SEED).standard_normal(clean40.shape)
    z_obs = np.random.default_rng(GAUSS_OBS_SEED).standard_normal((1, 40))
    gauss = {}
    for name, N_level in (('gbase', N), ('ggoal', N * ratio)):
        gauss[name] = (clean40 + z * np.sqrt(band_var(clean40, N_level, nu)),
                       obs_c40 + z_obs * np.sqrt(band_var(obs_c40, N_level, nu)))
    sd_ratio = sd_model / sd_draws
    goal_sd_ratio = np.sqrt(band_var(obs_c40[0], N * ratio, nu)) / sd_model
    noise = dict(N_band=N.tolist(), nu_band=nu.tolist(), fit_reduced_chi2=redchi,
                 goal_over_baseline_band=ratio.tolist(), b12_sd_model_over_20_draws=sd_ratio.tolist(),
                 b12_goal_sd_over_baseline_sd=goal_sd_ratio.tolist())
    (args.out / 'noise_model.json').write_text(json.dumps(noise, indent=1) + '\n')
    print('noise model: B12 sd model / 20 draws, median %.3f [%.3f, %.3f]; '
          'goal / baseline sd median %.3f, min %.3f'
          % (np.median(sd_ratio), np.min(sd_ratio), np.max(sd_ratio), np.median(goal_sd_ratio),
             np.min(goal_sd_ratio)), flush=True)

    # features per variant: (all rows, B12 observation), transforms fitted on the whole pool (the same for every N)
    print(f'asinh scale: {choice}', flush=True)
    scale40 = bins40_scale(choice, N, nu)
    band_inputs = dict(bins40_noisy=(noisy40, obs_n40), bins40_clean=(clean40, obs_c40),
                       bins40_gbase=gauss['gbase'], bins40_ggoal=gauss['ggoal'])
    feats = {}
    for name, (rows_in, obs_in) in band_inputs.items():
        state = fit_bins40(clean40[order], rows_in[order], scale40)
        feats[name] = (project_binned(state, rows_in), project_binned(state, obs_in))
    moped = fit_moped(raw_moped[order], choice)
    feats['moped_noisy'] = (moped(raw_moped), moped(obs['noisy_dl_unbinned'][None] @ matrix))
    for name, (x, x_obs) in feats.items():
        for size in train_sizes:
            folder = args.out / name / f'N{size}'
            folder.mkdir(parents=True, exist_ok=True)
            train = order[:size]
            np.savez(folder / 'features.npz', x_pool=x[train], theta_pool=theta[train], x_held=x[test],
                     theta_held=theta[test], pool=train, held=test, low=low, high=high,
                     obs_labels=np.array(['Battaglia12']), x_obs=x_obs.astype(np.float32))
        print(name, 'features', x.shape, flush=True)
    summary = dict(data=str(data), rows=int(n_rows), test_rows=TEST_ROWS, sizes=train_sizes, variants=VARIANTS,
                   missing_rows=len(meta['missing_rows']), asinh_scale=choice)
    (args.out / 'prepare.json').write_text(json.dumps(summary, indent=1) + '\n')


def sizes(args):
    """The training-set sizes N written by prepare (list of int, ascending; the last is the whole pool)."""
    return json.loads((args.out / 'prepare.json').read_text())['sizes']


def run_folders(out, name, SIZES):
    """Folder names of variant `name` below `out`: N<size> for every training size in SIZES, then the architecture
    variants N<size>_<net> (run_capacity.sh), sorted by size."""
    extra = [p.name for p in (out / name).glob('N*_*') if p.is_dir()]
    return [f'N{s}' for s in SIZES] + sorted(extra, key=lambda k: (int(k[1:].split('_')[0]), k))


def cmd_train(args):
    """Train 3 seeds per N folder of every requested variant with train_ensemble.py train, largest N first."""
    train_sizes = sizes(args)
    for name in args.variants:
        net = VARIANTS[name]
        cmd = [sys.executable, str(HERE / 'train_ensemble.py'), 'train', '--out', str(args.out / name), '--kinds',
               *[f'N{s}' for s in sorted(train_sizes, reverse=True)], '--seeds', '1', '2', '3',
               '--workers', str(args.workers), '--threads', str(args.threads), '--hidden', str(net['hidden']),
               '--transforms', str(net['transforms']), '--epochs', str(args.epochs)]
        print(' '.join(cmd), flush=True)
        subprocess.run(cmd, check=True)


def edge_fraction(theta, low, high):
    """Per parameter, the fraction of the values theta (..., 9) whose position in the prior box,
    s = (theta - low) / (high - low), lies within EDGE of either end, pooled over all leading axes.
    low, high: (9,) prior bounds. Returns (9,)."""
    width = high - low
    fractions = []
    for i in range(theta.shape[-1]):
        s = (theta[..., i] - low[i]) / width[i]
        fractions.append(np.mean((s < EDGE) | (s > 1 - EDGE)))
    return np.array(fractions)


def row_metrics(ens, x, truth, samples):
    """Posterior metrics of the ensemble `ens` on rows x (rows, xdim) with true parameters truth (rows, 9).

    `samples` posterior draws per row. Returns a dict and the per-row log p of the truth (rows,) in theta units:
      mean_log_prob_theta (+ sem)   mean log posterior density of the truth [nats, theta units]
      tarp68, tarp95                TARP expected coverage of the 68 / 95 % regions
      marginal_coverage68           per parameter, fraction of truths inside the 16-84 % quantiles
      median_width68_over_prior     per parameter, median over rows of (q84 - q16) / (high - low)
      median_width68_over_flat      the same divided by FLAT_WIDTH68: 1 = no narrowing relative to a flat prior
      pearson_r                     per parameter, correlation of truth and posterior mean over the rows
      rmse_over_prior_rmse          RMSE of the posterior mean over the RMSE of the prior centre
      edge_mass_ratio               per parameter, the fraction of all samples within EDGE of a prior bound over
                                    its flat-prior value 2 * EDGE (an exact posterior gives 1 on average over
                                    prior draws)
      edge_mass_ratio_truth         the same for the true parameters (reference, ~1)
    """
    import torch
    from sbi.diagnostics.tarp import _run_tarp, get_tarp_references
    from sbi.utils.metrics import l2
    from train_ensemble import to_u, log_jacobian_theta_per_u
    low, high = ens.low, ens.high
    width = high - low
    torch.manual_seed(1)
    s, _, _ = ens.posterior(x, samples, rng=np.random.default_rng(1))          # (samples, rows, 9)
    mean = s.mean(0)
    q16, q84 = np.quantile(s, [.16, .84], axis=0)
    # density of the truth: the network's u-space density divided by |d theta / d u|
    u_t = to_u(truth, low, high)
    lp = ens.log_prob_u(u_t[None], x)[0] - log_jacobian_theta_per_u(u_t, low, high)
    torch.manual_seed(1)
    truth_t = torch.tensor(truth, dtype=torch.float32)
    references = get_tarp_references(truth_t)                  # random reference points (torch seed 1)
    ecp, alpha = _run_tarp(torch.tensor(s, dtype=torch.float32), truth_t, references, l2, None, True)
    c68, c95 = np.interp([.68, .95], np.ravel(alpha.numpy()), np.ravel(ecp.numpy()))
    # the prior's RMSE: the posterior mean replaced by the centre of the box
    prior_rmse = np.sqrt(np.mean(((truth - (low + high) / 2) / width) ** 2, axis=0))
    flat_edge = 2 * EDGE
    metrics = dict(mean_log_prob_theta=float(lp.mean()),
                   mean_log_prob_theta_sem=float(lp.std() / np.sqrt(len(lp))),
                   tarp68=float(c68), tarp95=float(c95),
                   marginal_coverage68=np.mean((truth >= q16) & (truth <= q84), axis=0).tolist(),
                   median_width68_over_prior=np.median((q84 - q16) / width, axis=0).tolist(),
                   pearson_r=[float(np.corrcoef(truth[:, i], mean[:, i])[0, 1]) for i in range(9)],
                   rmse_over_prior_rmse=(np.sqrt(np.mean(((mean - truth) / width) ** 2, axis=0))
                                         / prior_rmse).tolist(),
                   median_width68_over_flat=np.median((q84 - q16) / (FLAT_WIDTH68 * width), axis=0).tolist(),
                   edge_mass_ratio=(edge_fraction(s, low, high) / flat_edge).tolist(),
                   edge_mass_ratio_truth=(edge_fraction(truth, low, high) / flat_edge).tolist())
    return metrics, lp


def cmd_evaluate(args):
    """metrics.json per N folder: row_metrics on the test rows, the B12 posterior summary (20,000 samples), its
    68 % widths over the prior range and the HPD level of the B12 truth, and the training summary of each seed."""
    import torch
    from train_ensemble import Ensemble, summarize
    torch.set_num_threads(args.threads)
    train_sizes = sizes(args)
    for name in args.variants:
        for size in train_sizes:
            folder = args.out / name / f'N{size}'
            if (folder / 'metrics.json').exists() and not args.force:
                continue
            if len(list(folder.glob('estimator_seed*.pt'))) < 3:
                print(f'{name} N{size}: not trained yet', flush=True)
                continue
            ens = Ensemble(folder, args.threads)
            f, width = ens.f, ens.high - ens.low
            metrics, _ = row_metrics(ens, f['x_held'], f['theta_held'], args.samples)
            torch.manual_seed(2)
            sb, lpb, _ = ens.posterior(f['x_obs'][:1], 20000, rng=np.random.default_rng(2))
            b12 = summarize(sb[:, 0], lpb[:, 0], A.B12)
            hpd = ens.hpd_level(f['x_obs'][:1], A.B12, n=4000, rng=np.random.default_rng(3))
            b12['truth_hpd_level'] = float(hpd[0])
            train = [json.loads((folder / f'train_seed{k}.json').read_text()) for k in ens.seeds]
            m = dict(variant=name, N=size, seeds=ens.seeds, test_rows=int(len(f['theta_held'])), **metrics,
                     b12=b12,
                     b12_width68_over_prior=((np.array(b12['q84']) - np.array(b12['q16'])) / width).tolist(),
                     training=[dict(seed=t['seed'], seconds=t['seconds'], epochs=t['epochs'],
                                    best_validation_loss=t['best_validation_loss']) for t in train],
                     settings=ens.settings)
            (folder / 'metrics.json').write_text(json.dumps(m, indent=1) + '\n')
            print(f"{name} N{size}: log p {m['mean_log_prob_theta']:.2f}, "
                  f"TARP {m['tarp68']:.2f}/{m['tarp95']:.2f}, "
                  f"mean r {np.mean(m['pearson_r']):.3f}, B12 width {np.mean(m['b12_width68_over_prior']):.2f}",
                  flush=True)


def cmd_validate(args):
    """The trained ensembles on independent rows: the rows the collaborator run lost, regenerated on idark with the
    same engine (manage.py collect --allow-missing + small arrays, see validation_idark.sh).

    log p for every N and every capacity variant; the full row_metrics for the whole pool; the same numbers on
    the 3,000 test rows (from metrics.json, else checks.json) for comparison; KS p-values of the validation theta
    against the flat prior. -> validation.json, fig_validation.png
    """
    import torch
    from scipy.stats import kstest
    from train_ensemble import Ensemble, to_u, log_jacobian_theta_per_u
    torch.set_num_threads(args.threads)
    train_sizes = sizes(args)
    pool = f'N{train_sizes[-1]}'
    maps, info = variant_features(args)
    low, high = info['low'], info['high']
    width = high - low
    v = args.validation
    vmeta = json.loads((v / 'meta.json').read_text())
    theta, c40, n40, raw_n = (np.load(v / f'{k}.npy')
                              for k in ('theta', 'clean_dl', 'noisy_dl', 'moped_raw_noisy'))
    rows = np.load(v / 'row_id.npy')
    assert np.allclose(vmeta['lower'], low) and np.allclose(vmeta['upper'], high)
    analysed = set(np.load(args.out / 'split.npz')['row_id'].tolist())
    assert not analysed & set(rows.tolist()), 'validation rows overlap the analysed rows'
    checks_path = args.out / 'checks.json'
    checks = json.loads(checks_path.read_text())['results'] if checks_path.exists() else {}
    z = np.random.default_rng(6).standard_normal(c40.shape)           # Gaussian-noise draws of the validation rows
    ks = [float(kstest((theta[:, i] - low[i]) / width[i], 'uniform').pvalue) for i in range(9)]
    out = dict(rows=int(len(rows)), source=str(v), created_utc=vmeta.get('created_utc'), prior_ks_pvalues=ks,
               results={})
    for name in args.variants:
        x = maps[name](c40, n40, raw_n, z)
        for kind in run_folders(args.out, name, train_sizes):
            folder = args.out / name / kind
            key = f'{name}/{kind}'
            if len(list(folder.glob('estimator_seed*.pt'))) < 3:
                continue
            ens = Ensemble(folder, args.threads)
            if kind.startswith(pool):
                entry, _ = row_metrics(ens, x, theta, args.samples)
            else:
                u_t = to_u(theta, low, high)
                lp = ens.log_prob_u(u_t[None], x)[0] - log_jacobian_theta_per_u(u_t, low, high)
                entry = dict(mean_log_prob_theta=float(lp.mean()),
                             mean_log_prob_theta_sem=float(lp.std() / np.sqrt(len(lp))))
            # the same numbers on the test rows, where available
            test = folder / 'metrics.json'
            if test.exists():
                entry['test'] = {k: vv for k, vv in json.loads(test.read_text()).items() if k in entry}
            elif key in checks:
                entry['test'] = dict(mean_log_prob_theta=checks[key]['ensemble_log_prob'])
            else:
                entry['test'] = None
            out['results'][key] = entry
            t = entry['test']['mean_log_prob_theta'] if entry['test'] else float('nan')
            extra = ''
            if 'tarp68' in entry:
                extra = (f", TARP {entry['tarp68']:.2f}/{entry['tarp95']:.2f}, "
                         f"mean r {np.mean(entry['pearson_r']):.3f}")
            print(f"{name} {kind}: validation log p {entry['mean_log_prob_theta']:.2f} ± "
                  f"{entry['mean_log_prob_theta_sem']:.2f} (test {t:.2f})" + extra, flush=True)
    (args.out / 'validation.json').write_text(json.dumps(out, indent=1) + '\n')
    plot_validation(args.out, out)


def plot_validation(out, report):
    """fig_validation.png in `out` from the validation.json dict `report`: (a) validation against test log p for
    every ensemble, (b) the per-parameter 68 % coverage on the validation rows for the whole pool."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from ppc_plots import style, INK, INK2, GRID
    R = report['results']
    style()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(18 / 2.54, 8 / 2.54), gridspec_kw=dict(wspace=0.3))
    for name in VARIANTS:
        # (test log p, validation log p, its sem, larger network?) for every ensemble of this variant
        pts = [(e['test']['mean_log_prob_theta'], e['mean_log_prob_theta'], e['mean_log_prob_theta_sem'],
                '_' in k.split('/', 1)[1])
               for k, e in R.items() if k.split('/')[0] == name and e['test']]
        for x, y, err, net in pts:
            a1.errorbar(x, y, yerr=err, color=COLORS[name], marker='s' if net else MARKS[name],
                        mfc='white' if net else COLORS[name], ms=4, lw=0.9, capsize=1.5)
        a1.plot([], [], color=COLORS[name], marker=MARKS[name], ls='none', ms=4, label=NAMES[name])
    a1.plot([], [], color=INK2, marker='s', mfc='white', ls='none', ms=4,
            label='larger networks (40 bands + map noise)')
    lim = [min(e['mean_log_prob_theta'] for e in R.values()) - 0.4,
           max(e['mean_log_prob_theta'] for e in R.values()) + 0.4]
    a1.plot(lim, lim, color=INK2, lw=0.8, ls='--')
    a1.set_xlim(lim)
    a1.set_ylim(lim)
    a1.set_xlabel('log p on the 3,000 test rows (nats)')
    a1.set_ylabel(f"log p on the {report['rows']:,} idark rows (nats)")
    a1.set_title('(a) every ensemble, N = 2,000 ... 61,440', fontsize=8)
    a1.legend(fontsize=6, frameon=False, loc='upper left')
    k9 = np.arange(9)
    shown = (('bins40_noisy/N61440', COLORS['bins40_noisy'], 'o', 'MAF 128×5, 40 bands'),
             ('bins40_noisy/N61440_maf256x8', INK, 's', 'MAF 256×8, 40 bands'),
             ('moped_noisy/N61440', COLORS['moped_noisy'], 's', 'MAF 64×5, MOPED'))
    for i, (key, color, marker, label) in enumerate(shown):
        if key in R and 'marginal_coverage68' in R[key]:
            e = R[key]
            a2.plot(k9 + 0.15 * (i - 1), e['marginal_coverage68'], marker=marker, color=color, ls='none', ms=4.5,
                    label=f"{label}: TARP {100 * e['tarp68']:.0f} / {100 * e['tarp95']:.0f} %")
    a2.axhline(0.68, color=INK2, lw=0.8, ls='--')
    # 95 % binomial range of the coverage of a calibrated posterior on this many rows
    band = 1.96 * np.sqrt(0.68 * 0.32 / report['rows'])
    a2.axhspan(0.68 - band, 0.68 + band, color=GRID, alpha=0.6, lw=0)
    a2.set_xticks(k9)
    a2.set_xticklabels(PARAM_LABELS, fontsize=7)
    a2.set_ylim(0.55, 0.85)
    a2.set_ylabel('fraction of truths inside the 68 % interval')
    a2.set_title('(b) marginal coverage on the idark rows, N = 61,440\n'
                 f"(grey: 95 % range for {report['rows']:,} rows)", fontsize=8)
    a2.legend(fontsize=6, frameon=False, loc='upper left')
    for ax in (a1, a2):
        ax.grid(True, color=GRID, lw=0.5)
        ax.set_axisbelow(True)
    fig.suptitle(f"Independent validation: the {report['rows']:,} rows regenerated on idark", fontsize=9, y=1.08)
    fig.savefig(out / 'fig_validation.png', bbox_inches='tight')
    plt.close(fig)


def saturating(N, L, A, alpha):
    """Saturating model of log p against training-set size: L - A (N / 1e4)^-alpha (L = asymptote in nats)."""
    return L - A * (N / 1e4) ** (-alpha)


def cmd_report(args):
    """Convergence with training-set size from the metrics.json files of evaluate.

    fig_convergence.png: (a) mean log p of the truth with the saturating fit, (b) TARP 68 %, (c) mean Pearson r,
    (d) 68 % widths over the prior range, with the flat-prior value FLAT_WIDTH68. fig_correlation_vs_N.png: r per
    parameter. convergence.json: per variant N, log p, the gain of the last doubling of N, the fit and its
    extrapolation to 2, 4, 8 x the largest N; fit_at_bound flags a fit that ended at one of its bounds, whose
    asymptote is then set by the bound and not by the data. convergence_table.md: one row per variant and N.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit
    from ppc_plots import style, INK2, GRID
    train_sizes = sizes(args)
    M = {}
    for name in VARIANTS:
        for size in train_sizes:
            path = args.out / name / f'N{size}' / 'metrics.json'
            if path.exists():
                M[name, size] = json.loads(path.read_text())
    report = {}
    lines = ['68 % widths are the 16-84 % interval divided by the full prior range (high - low); '
             f'a flat prior (no information) gives {FLAT_WIDTH68:.3f}.', '',
             '| variant | N | log p(truth) [nats] | TARP 68/95 | mean r | mean 68 % width / prior range | '
             'B12 mean width / prior range | B12 truth HPD |',
             '|---|---|---|---|---|---|---|---|']
    style()
    fig, axes = plt.subplots(2, 2, figsize=(18 / 2.54, 15 / 2.54), gridspec_kw=dict(hspace=0.35, wspace=0.3))
    (a1, a2), (a3, a4) = axes
    for name in VARIANTS:
        pts = [(s, M[name, s]) for s in train_sizes if (name, s) in M]
        if not pts:
            continue
        N = np.array([s for s, _ in pts], dtype=float)
        lp = np.array([m['mean_log_prob_theta'] for _, m in pts])
        se = np.array([m['mean_log_prob_theta_sem'] for _, m in pts])
        kw = dict(color=COLORS[name], marker=MARKS[name], ms=4.5, lw=1.3, label=NAMES[name])
        a1.errorbar(N, lp, yerr=se, **kw)
        a2.plot(N, [m['tarp68'] for _, m in pts], **kw)
        a3.plot(N, [np.mean(m['pearson_r']) for _, m in pts], **kw)
        a4.plot(N, [np.mean(m['median_width68_over_prior']) for _, m in pts], **kw)
        a4.plot(N, [np.mean(m['b12_width68_over_prior']) for _, m in pts], color=COLORS[name], marker=MARKS[name],
                ms=3.5, lw=0.9, ls=(0, (3, 1.5)), mfc='white')
        # log p gained by the last doubling of N, interpolated linearly in log N
        gain = float(lp[-1] - np.interp(np.log(N[-1] / 2), np.log(N), lp)) if len(N) > 1 else None
        entry = dict(N=N.tolist(), log_prob=lp.tolist(), gain_last_doubling=gain)
        if len(N) >= 4:
            # weighted fit of the saturating model with L_inf within [-1, +20] nats of the last point
            lower, upper = [lp[-1] - 1, 0, 0.01], [lp[-1] + 20, 1e3, 3]
            try:
                (L, A_, al), _ = curve_fit(saturating, N, lp, p0=(lp[-1] + 1, 1, 0.5), sigma=np.maximum(se, 1e-3),
                                           maxfev=20000, bounds=(lower, upper))
                grid = np.logspace(np.log10(N[0]), np.log10(8 * N[-1]), 100)
                a1.plot(grid, saturating(grid, L, A_, al), color=COLORS[name], lw=0.8, ls=':')
                entry.update(fit=dict(L_inf=float(L), A=float(A_), alpha=float(al)),
                             predicted=dict((f'x{k}', float(saturating(k * N[-1], L, A_, al))) for k in (2, 4, 8)),
                             remaining_to_asymptote=float(L - saturating(N[-1], L, A_, al)))
                at_bound = [p for p, value, lo, hi in zip(('L_inf', 'A', 'alpha'), (L, A_, al), lower, upper)
                            if min(value - lo, hi - value) <= FIT_BOUND_TOLERANCE * (hi - lo)]
                entry['fit_at_bound'] = bool(at_bound)
                if at_bound:
                    entry['fit_parameters_at_bound'] = at_bound
                    print(f"WARNING {name}: the saturating fit ended at its bound for {', '.join(at_bound)}; "
                          'L_inf, the predictions and remaining_to_asymptote are set by the bound, '
                          'not by the data', flush=True)
            except (RuntimeError, ValueError) as error:
                entry['fit_error'] = str(error)
        report[name] = entry
        for s, m in pts:
            lines.append(f"| {NAMES[name]} | {s:,} | "
                         f"{m['mean_log_prob_theta']:.2f} ± {m['mean_log_prob_theta_sem']:.2f} | "
                         f"{100 * m['tarp68']:.0f} / {100 * m['tarp95']:.0f} | {np.mean(m['pearson_r']):.3f} | "
                         f"{np.mean(m['median_width68_over_prior']):.3f} | "
                         f"{np.mean(m['b12_width68_over_prior']):.3f} | {m['b12']['truth_hpd_level']:.3f} |")
    a2.axhline(0.68, color=INK2, ls='--', lw=0.8)
    flat = a4.axhline(FLAT_WIDTH68, color=INK2, ls='--', lw=0.8, label='flat prior (no information)')
    a4.legend(handles=[flat], fontsize=6.5, frameon=False, loc='best')
    panels = ((a1, '(a) mean log posterior density of the true parameters', 'nats (higher is better)'),
              (a2, '(b) calibration: TARP coverage of the 68 % region', 'coverage'),
              (a3, '(c) truth-mean correlation, averaged over the nine', 'mean Pearson r'),
              (a4, '(d) 68 % width / prior: test rows (solid), B12 (dashed)', 'mean over the nine'))
    for ax, title, ylabel in panels:
        ax.set_xscale('log')
        ax.set_xlabel('training simulations N')
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=8)
        ax.grid(True, color=GRID, lw=0.5)
        ax.set_axisbelow(True)
    handles, labs = a1.get_legend_handles_labels()
    handles.append(plt.Line2D([], [], color=INK2, ls=':', lw=0.8))
    labs.append('dotted: saturating fit $L_\\infty - A N^{-\\alpha}$')
    fig.legend(handles, labs, loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=2, frameon=False, fontsize=7)
    fig.suptitle('NPE convergence with training-set size (3,000 fixed test simulations, 3 seeds per point; '
                 f'{train_sizes[-1]:,} = all produced rows)', fontsize=9, y=0.99)
    fig.savefig(args.out / 'fig_convergence.png')
    plt.close(fig)

    # truth-mean correlation per parameter
    fig, axes = plt.subplots(3, 3, figsize=(18 / 2.54, 15 / 2.54), sharex=True,
                             gridspec_kw=dict(hspace=0.35, wspace=0.3))
    for i, ax in enumerate(axes.ravel()):
        for name in VARIANTS:
            pts = [(s, M[name, s]) for s in train_sizes if (name, s) in M]
            if pts:
                ax.plot([s for s, _ in pts], [m['pearson_r'][i] for _, m in pts], color=COLORS[name],
                        marker=MARKS[name], ms=3.5, lw=1.1, label=NAMES[name] if i == 0 else None)
        ax.set_xscale('log')
        ax.set_ylim(0, 1)
        ax.set_title(PARAM_LABELS[i], fontsize=9, pad=2)
        ax.grid(True, color=GRID, lw=0.5)
        ax.set_axisbelow(True)
    for ax in axes[-1]:
        ax.set_xlabel('training simulations N')
    for ax in axes[:, 0]:
        ax.set_ylabel('r(truth, posterior mean)')
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=2,
               frameon=False, fontsize=7)
    fig.suptitle('Truth-posterior mean correlation per parameter on the 3,000 test simulations', fontsize=9,
                 y=0.995)
    fig.savefig(args.out / 'fig_correlation_vs_N.png')
    plt.close(fig)
    (args.out / 'convergence.json').write_text(json.dumps(report, indent=1) + '\n')
    (args.out / 'convergence_table.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    # the extrapolation, except where the fit ended at a bound
    summary = {}
    for name, entry in report.items():
        if entry.get('fit_at_bound'):
            keys = ('gain_last_doubling', 'fit_at_bound', 'fit_parameters_at_bound')
        else:
            keys = ('gain_last_doubling', 'predicted', 'remaining_to_asymptote')
        summary[name] = {k: entry[k] for k in keys if k in entry}
    print(json.dumps(summary, indent=1))


def fisher_reference(low, width, Nb, nu, ratio, draws=4_000_000, centres=200, per_centre=100_000):
    """Linear Fisher forecast of the 40 bands at B12, times the 64k flat prior as a hard box.

    The Fisher matrix of fisher_b12.py (current-engine derivatives J, covariance C of 20 SO baseline noise draws)
    and its SO-goal version (band sd scaled with the noise model of prepare: Nb, nu, ratio, each (40,)).
    low, width: (9,) prior box. The forecast assumes spectra linear in theta over the box, which fails along the
    best-measured directions (curved degeneracy), so it is not an attainable ideal there.
    Per level ('baseline', 'goal'): the eigen-widths sigma_eig (prior-width units), the fraction of the Gaussian
    centred on B12 inside the box, the sd of the boxed posterior along the baseline eigenvectors, its entropy H
    (theta units, nats) and the expected log density of the truth over noise draws,
    E_x log p(theta_B12 | x) = -log det(2 pi Sigma)/2 - 9/2 - E_m log Z(m), with the linear-model MLE
    m ~ N(theta_B12, Sigma) and Z(m) the Gaussian mass inside the box. Monte Carlo: `draws` samples for the widths
    and H, `centres` values of m with `per_centre` samples each for Z. Also returns the baseline eigenvectors
    (9, 9, one per column).
    """
    chunk = 200_000                                         # Monte Carlo samples per batch
    fisher_input = np.load(HERE / 'results_logit' / 'fisher_b12_samples.npz')
    J, C, d0 = fisher_input['J'], fisher_input['C'], fisher_input['d0']
    # goal / baseline band sd at the B12 spectrum d0
    sd = np.sqrt(band_var(d0, Nb * ratio, nu) / band_var(d0, Nb, nu))
    mu = (A.B12 - low) / width                              # the truth in unit-box coordinates
    out, base = {}, None
    for level, cov in (('baseline', C), ('goal', C * np.outer(sd, sd))):
        F = J.T @ np.linalg.solve(cov, J) * np.outer(width, width)                   # prior-width units
        lam, vec = np.linalg.eigh((F + F.T) / 2)
        order = np.argsort(lam)[::-1]
        lam, vec = lam[order], vec[:, order]                # best-measured direction first
        base = vec if base is None else base
        sigma = 1 / np.sqrt(lam)
        rng = np.random.default_rng(4)
        # the Gaussian posterior centred on the truth, truncated to the unit box
        inside, quad, proj = 0, 0.0, []
        for _ in range(draws // chunk):
            g = rng.standard_normal((chunk, 9))
            x = mu + (g * sigma) @ vec.T
            ok = np.all((x > 0) & (x < 1), axis=1)
            inside += int(ok.sum())
            quad += float(np.sum(g[ok] ** 2))
            proj.append((x[ok] - mu) @ base)
        # entropy of the truncated Gaussian: log Z + log det(2 pi Sigma)/2 + E[chi^2 | inside]/2
        log_norm = 0.5 * np.sum(np.log(2 * np.pi * sigma ** 2))
        H = np.log(inside / draws) + log_norm + 0.5 * quad / inside
        # log Z(m) for MLEs m scattered around the truth (floored at half a sample)
        log_z = []
        for _ in range(centres):
            m = mu + (rng.standard_normal(9) * sigma) @ vec.T
            y = m + (rng.standard_normal((per_centre, 9)) * sigma) @ vec.T
            log_z.append(np.log(max(np.mean(np.all((y > 0) & (y < 1), axis=1)), 0.5 / per_centre)))
        # unit-box densities to theta units: subtract sum log width
        out[level] = dict(sigma_eig=sigma.tolist(), inside_fraction=inside / draws,
                          std_along_baseline_eigenvectors=np.concatenate(proj).std(0).tolist(),
                          entropy_theta=float(H + np.sum(np.log(width))),
                          expected_log_prob_truth_theta=float(-log_norm - 4.5 - np.mean(log_z)
                                                              - np.sum(np.log(width))),
                          expected_log_prob_truth_sem=float(np.std(log_z) / np.sqrt(centres)))
    out['baseline_eigenvectors'] = base.tolist()
    return out


def variant_features(args):
    """The input features of the five variants, rebuilt exactly as in prepare (transforms fitted on the training
    pool, with the asinh scale recorded in prepare.json, median if it is missing).

    args.out: the prepare folder. Returns ({variant: f(clean40, noisy40, moped_raw_noisy, z)}, info): f maps
    (rows, 40) clean and noisy band D_ell, (rows, 9) raw MOPED projections and (rows, 40) standard-normal draws z
    (used by the Gaussian-noise variants only; the unused inputs may be None) to float32 features.
    info: low, high (9,), the noise model Nb, nu, ratio (40,), the draws z_train of prepare (rows, 40) and the
    data folder.
    """
    prepared = json.loads((args.out / 'prepare.json').read_text())
    d = Path(prepared['data'])
    choice = prepared.get('asinh_scale', 'median')
    meta = json.loads((d / 'meta.json').read_text())
    clean40 = np.load(d / 'clean_dl.npy')
    noisy40 = np.load(d / 'noisy_dl.npy')
    raw_n = np.load(d / 'moped_raw_noisy.npy')
    order = np.load(args.out / 'split.npz')['order']
    noise = json.loads((args.out / 'noise_model.json').read_text())
    Nb, nu, ratio = (np.array(noise[k]) for k in ('N_band', 'nu_band', 'goal_over_baseline_band'))
    z_train = np.random.default_rng(GAUSS_ROWS_SEED).standard_normal(clean40.shape)
    scale40 = bins40_scale(choice, Nb, nu)
    st_noisy = fit_bins40(clean40[order], noisy40[order], scale40)
    st_clean = fit_bins40(clean40[order], clean40[order], scale40)

    def bins40_noisy(clean, noisy, moped_raw, z):
        return project_binned(st_noisy, noisy)

    def bins40_clean(clean, noisy, moped_raw, z):
        return project_binned(st_clean, clean)

    def gaussian(N_level):
        """features of clean bands + Gaussian noise with the model variance at noise level N_level (40,)"""
        noisy_train = clean40 + z_train * np.sqrt(band_var(clean40, N_level, nu))
        state = fit_bins40(clean40[order], noisy_train[order], scale40)

        def features(clean, noisy, moped_raw, z):
            return project_binned(state, clean + z * np.sqrt(band_var(clean, N_level, nu)))
        return features

    moped = fit_moped(raw_n[order], choice)

    def moped_noisy(clean, noisy, moped_raw, z):
        return moped(moped_raw)

    maps = dict(bins40_noisy=bins40_noisy, bins40_clean=bins40_clean, bins40_gbase=gaussian(Nb),
                bins40_ggoal=gaussian(Nb * ratio), moped_noisy=moped_noisy)
    info = dict(low=np.array(meta['lower']), high=np.array(meta['upper']), Nb=Nb, nu=nu, ratio=ratio,
                z_train=z_train, data=d)
    return {k: maps[k] for k in VARIANTS}, info


def cmd_checks(args):
    """Is the NPE limited by the training set, the network or the noise?

    Per N folder of every variant: the log p of each seed alone and of the ensemble on the test rows, the B12
    posterior sd along the Fisher eigenvectors, and the B12 truth density averaged over many noise draws of the
    observation, against the linear Fisher forecast (fisher_reference). Paired differences to bins40_noisy, the
    log-linear gain per doubling of N, and the capacity control (folders N<size>_<net> next to N<size>, e.g. from
    run_capacity.sh). Folders already in checks.json are reused unless --force.
    -> checks.json, checks_rows.npz (per-row ensemble log p), fig_checks.png
    """
    import torch
    from train_ensemble import Ensemble, to_u, log_jacobian_theta_per_u
    torch.set_num_threads(args.threads)
    train_sizes = sizes(args)
    maps, info = variant_features(args)
    low, high, Nb, nu, ratio = info['low'], info['high'], info['Nb'], info['nu'], info['ratio']
    width = high - low

    # the B12 observation of each variant (checked against features.npz) and many noise draws of it
    obs, matrix = b12_observation()
    obs_c40 = obs['clean_dl'][None]
    mocks = np.stack([np.load(p)[ELL] * DL for p in sorted(B12_MOCKS.glob('noisy_cross_*.npy'))])    # (21, 7900)
    z_obs = np.random.default_rng(GAUSS_OBS_SEED).standard_normal((1, 40))
    z_new = np.random.default_rng(5).standard_normal((100, 40))
    inputs = {}
    for name, fn in maps.items():
        x_obs = fn(obs_c40, obs['noisy_dl'][None], obs['noisy_dl_unbinned'][None] @ matrix, z_obs)
        if name in ('bins40_gbase', 'bins40_ggoal'):
            x_draws = fn(np.repeat(obs_c40, len(z_new), axis=0), None, None, z_new)
            inputs[name] = (x_obs, x_draws, f'{len(z_new)} Gaussian draws')
        elif name == 'bins40_clean':
            inputs[name] = (x_obs, x_obs, 'noise-free')
        else:
            x_draws = fn(None, A_rebin(mocks), mocks @ matrix, None)
            inputs[name] = (x_obs, x_draws, f'{len(mocks)} SO map-noise draws')
    fisher = fisher_reference(low, width, Nb, nu, ratio)
    vec = np.array(fisher['baseline_eigenvectors'])
    stored = json.loads((HERE / 'results_logit' / 'fisher_goal_vs_baseline.json').read_text())
    stored_goal = stored['all nine free']['goal']['sigma_eig']
    rebuilt_over_stored = np.array(fisher['goal']['sigma_eig']) / stored_goal
    print('goal Fisher widths, rebuilt / stored:', np.round(rebuilt_over_stored, 3).tolist(), flush=True)

    u_b12 = to_u(A.B12[None], low, high)
    jac_b12 = float(log_jacobian_theta_per_u(u_b12, low, high)[0])
    done = (args.out / 'checks.json').exists() and (args.out / 'checks_rows.npz').exists() and not args.force
    previous = json.loads((args.out / 'checks.json').read_text())['results'] if done else {}
    old_rows = dict(np.load(args.out / 'checks_rows.npz')) if done else {}
    R, rows = {}, {}
    for name in args.variants:
        x_obs, x_draws, draw_note = inputs[name]
        for kind in run_folders(args.out, name, train_sizes):
            key, folder = f'{name}/{kind}', args.out / name / kind
            if key in previous and key.replace('/', '__') in old_rows:
                R[name, kind] = previous[key]
                rows[key] = old_rows[key.replace('/', '__')]
                continue
            if len(list(folder.glob('estimator_seed*.pt'))) < 3:
                print(f'{name} {kind}: fewer than 3 seeds, skipped', flush=True)
                continue
            ens = Ensemble(folder, args.threads)
            f = ens.f
            assert np.allclose(f['x_obs'], x_obs, atol=1e-5), \
                f'{name}: rebuilt B12 input differs from features.npz'
            # log p of the test truths per seed (theta units), and of their equal-weight mixture
            u_t = to_u(f['theta_held'], low, high)
            jac = log_jacobian_theta_per_u(u_t, low, high)
            u_tensor = torch.as_tensor(u_t[None], dtype=torch.float32)
            x_tensor = torch.as_tensor(f['x_held'], dtype=torch.float32)
            with torch.no_grad():
                per_seed = np.stack([net.log_prob(u_tensor, condition=x_tensor).numpy()[0]
                                     for net in ens.nets]) - jac
            mix = np.logaddexp.reduce(per_seed, axis=0) - np.log(len(ens.nets))
            # B12 posterior, and the density of the B12 truth given each noise draw of the observation
            torch.manual_seed(7)
            s, _, _ = ens.posterior(x_obs, 30000, rng=np.random.default_rng(7))
            lq = ens.log_prob_u(np.repeat(u_b12[None], len(x_draws), axis=1), x_draws)[0] - jac_b12
            R[name, kind] = dict(seed_log_prob=per_seed.mean(1).tolist(), ensemble_log_prob=float(mix.mean()),
                                 ensemble_log_prob_sem=float(mix.std() / np.sqrt(len(mix))),
                                 b12_std_along_fisher_eigenvectors=(((s[:, 0] - A.B12) / width)
                                                                    @ vec).std(0).tolist(),
                                 b12_log_prob_truth=float(lq.mean()),
                                 b12_log_prob_truth_sem=float(lq.std() / np.sqrt(len(lq))),
                                 b12_draws=draw_note, settings=ens.settings)
            rows[key] = mix
            print(f"{name} {kind}: seeds {np.round(per_seed.mean(1), 2).tolist()} ensemble {mix.mean():.2f}; "
                  f"B12 log p(truth) {lq.mean():.2f} ± {lq.std() / np.sqrt(len(lq)):.2f}", flush=True)

    # paired differences to the SO-baseline map-noise run; ensemble training noise taken as seed scatter / sqrt(3)
    diff = {}
    for name in args.variants:
        for kind in sorted({k for n, k in R if n == name}):
            this, reference = (name, kind), ('bins40_noisy', kind.split('_')[0])
            if this == reference or reference not in R:
                continue
            dd = rows[f'{name}/{kind}'] - rows[f'{reference[0]}/{reference[1]}']
            seed_var = (np.var(R[this]['seed_log_prob'], ddof=1)
                        + np.var(R[reference]['seed_log_prob'], ddof=1)) / 3
            diff[f'{name}/{kind}'] = dict(mean=float(dd.mean()), sem_rows=float(dd.std() / np.sqrt(len(dd))),
                                          error=float(np.sqrt(dd.var() / len(dd) + seed_var)))
    # gain per doubling of N: slope of the ensemble log p against log2 N
    slope = {}
    for name in args.variants:
        pts = [(s, R[name, f'N{s}']['ensemble_log_prob']) for s in train_sizes if (name, f'N{s}') in R]
        if len(pts) >= 3:
            slope[name] = float(np.polyfit(np.log2([p[0] for p in pts]), [p[1] for p in pts], 1)[0])
    notes = ('log p in theta units (nats); test rows: 3,000 fixed rows; B12 log p(truth) averaged over noise '
             'draws of the B12 observation; the ideal analysis is the boxed Fisher posterior (approximate: linear '
             'in theta); difference errors: test rows (paired) + seed scatter / sqrt(3) for each ensemble')
    report = dict(prior_log_density_theta=float(-np.sum(np.log(width))), fisher=fisher,
                  gain_per_doubling_loglinear=slope, differences_to_bins40_noisy=diff,
                  results={f'{n}/{k}': v for (n, k), v in R.items()}, notes=notes)
    (args.out / 'checks.json').write_text(json.dumps(report, indent=1) + '\n')
    np.savez(args.out / 'checks_rows.npz', **{k.replace('/', '__'): v for k, v in rows.items()})
    plot_checks(args.out, report, train_sizes, args.variants)
    ideal = {k: fisher[k]['expected_log_prob_truth_theta'] for k in ('baseline', 'goal')}
    print(json.dumps(dict(prior=report['prior_log_density_theta'], slope=slope, diff=diff, ideal=ideal), indent=1))


def plot_checks(out, report, SIZES, variants):
    """fig_checks.png in `out` from the checks.json dict `report` (SIZES: training-set sizes, variants: names):
    (a) test log p relative to bins40_noisy, (b) B12 posterior sd along the Fisher eigen-directions, (c) B12 truth
    log p against the linear Fisher forecast and the prior, (d) network capacity."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.ticker
    import matplotlib.pyplot as plt
    from ppc_plots import style, INK, INK2, GRID
    R, diff, fisher = report['results'], report['differences_to_bins40_noisy'], report['fisher']
    slope = report['gain_per_doubling_loglinear']
    prior_lp = report['prior_log_density_theta']
    style()
    fig, axes = plt.subplots(2, 2, figsize=(18 / 2.54, 15.5 / 2.54), gridspec_kw=dict(hspace=0.5, wspace=0.32))
    (a1, a2), (a3, a4) = axes

    # (a) paired differences, slightly shifted in N per variant for legibility
    for k, name in enumerate(n for n in variants if n != 'bins40_noisy'):
        pts = [(s, diff[f'{name}/N{s}']) for s in SIZES if f'{name}/N{s}' in diff]
        if pts:
            a1.errorbar(np.array([s for s, _ in pts]) * (1 + 0.04 * (k - 1.5)), [p['mean'] for _, p in pts],
                        yerr=[p['error'] for _, p in pts], color=COLORS[name], marker=MARKS[name], ms=4, lw=1.1,
                        capsize=2)
    a1.axhline(0, color=COLORS['bins40_noisy'], lw=1.1)
    if 'bins40_noisy' in slope:
        a1.axhline(slope['bins40_noisy'], color=INK2, lw=0.8, ls='--')
        a1.text(0.02, slope['bins40_noisy'] + 0.04,
                f"doubling N (40 bands + map noise): +{slope['bins40_noisy']:.2f}", fontsize=6.5, color=INK2,
                va='bottom', transform=a1.get_yaxis_transform())
    forecast_gain = (fisher['goal']['expected_log_prob_truth_theta']
                     - fisher['baseline']['expected_log_prob_truth_theta'])
    a1.axhline(forecast_gain, color=COLORS['bins40_ggoal'], lw=0.9, ls=':')
    a1.text(0.02, forecast_gain + 0.04, f'linear Fisher forecast at B12, goal − baseline: +{forecast_gain:.2f}',
            fontsize=6.5, color=COLORS['bins40_ggoal'], va='bottom', transform=a1.get_yaxis_transform())
    a1.set_ylim(-1.5, 1.65)
    a1.set_title('(a) test log p relative to 40 bands + SO baseline map noise\n(bars: test rows + seed scatter)',
                 fontsize=8)
    a1.set_ylabel('Δ nats')
    a1.set_xscale('log')
    a1.set_xlabel('training simulations N')

    # (b) B12 posterior sd along the Fisher eigenvectors for four training sizes
    kk = np.arange(1, 10)
    for size, alpha_ in zip((SIZES[0], SIZES[2], SIZES[-2], SIZES[-1]), (0.3, 0.5, 0.75, 1.0)):
        if f'bins40_noisy/N{size}' in R:
            a2.semilogy(kk, R[f'bins40_noisy/N{size}']['b12_std_along_fisher_eigenvectors'], marker='o', ms=3.5,
                        lw=1.0, color=COLORS['bins40_noisy'], alpha=alpha_,
                        label=f'NPE, 40 bands + map noise, N = {size:,}')
    a2.semilogy(kk, fisher['baseline']['std_along_baseline_eigenvectors'], color=INK, marker='s', ms=3.5, lw=1.0,
                label='linear Fisher forecast at B12, baseline')
    a2.semilogy(kk, fisher['goal']['std_along_baseline_eigenvectors'], color=COLORS['bins40_ggoal'], marker='v',
                ms=3.5, lw=1.0, ls='--', label='linear Fisher forecast at B12, goal')
    a2.axhline(1 / np.sqrt(12), color=INK2, lw=0.8, ls=':', label='flat prior')     # sd of a unit flat prior
    a2.set_xticks(kk)
    a2.set_xlabel('Fisher eigen-direction (1 = best measured)')
    a2.set_ylabel('posterior sd / prior width')
    a2.set_title('(b) B12 posterior width along the Fisher eigen-directions', fontsize=8)
    a2.legend(fontsize=6, frameon=False, loc='lower right')

    # (c) B12 truth log p over noise draws
    for name in variants:
        pts = [(s, R[f'{name}/N{s}']) for s in SIZES if f'{name}/N{s}' in R]
        if pts:
            a3.errorbar([s for s, _ in pts], [r['b12_log_prob_truth'] for _, r in pts],
                        yerr=[r['b12_log_prob_truth_sem'] for _, r in pts], color=COLORS[name], marker=MARKS[name],
                        ms=4, lw=1.1, capsize=2, label=NAMES[name])
    for level, color, ls, dy, va in (('baseline', INK, '--', -0.35, 'top'),
                                     ('goal', COLORS['bins40_ggoal'], ':', 0.3, 'bottom')):
        v = fisher[level]['expected_log_prob_truth_theta']
        a3.axhline(v, color=color, lw=0.9, ls=ls)
        a3.text(0.02, v + dy, f'linear Fisher forecast, {level}: {v:+.1f}', fontsize=6.5, color=color, va=va,
                transform=a3.get_yaxis_transform())
    a3.axhline(prior_lp, color=INK2, lw=0.8, ls='-.')
    a3.text(0.02, prior_lp + 0.3, f'prior: {prior_lp:.1f}', fontsize=6.5, color=INK2, va='bottom',
            transform=a3.get_yaxis_transform())
    a3.set_ylim(prior_lp - 1.2, fisher['goal']['expected_log_prob_truth_theta'] + 2.2)
    a3.set_xscale('log')
    a3.set_xlabel('training simulations N')
    a3.set_ylabel('nats')
    a3.set_title('(c) B12 truth: log p, mean over noise draws', fontsize=8)

    # (d) capacity: the default MAF 128x5 against larger networks, N >= 8,000
    nets = {}
    for key, r in R.items():
        name, kind = key.split('/', 1)
        size, _, net = kind[1:].partition('_')
        if name == 'bins40_noisy' and int(size) >= 8000:
            nets.setdefault(net or 'maf128x5', []).append((int(size), r))
    look = dict(maf128x5=(COLORS['bins40_noisy'], 'o', 'MAF 128×5 (used above)', 1.0),
                maf256x8=(INK, 's', 'MAF 256×8', 1.06), nsf128x5=(COLORS['moped_noisy'], 'D', 'NSF 128×5', 0.94))
    for net, pts in sorted(nets.items(), key=lambda kv: list(look).index(kv[0]) if kv[0] in look else 9):
        pts.sort(key=lambda p: p[0])
        color, marker, label, shift = look.get(net, (INK2, 'x', net, 1.0))
        a4.plot([shift * n for n, _ in pts], [r['ensemble_log_prob'] for _, r in pts], color=color, marker=marker,
                ms=5, lw=1.2, label=label)
        for n, r in pts:
            a4.plot([shift * n] * len(r['seed_log_prob']), r['seed_log_prob'], marker=marker, ms=3.5, mfc='white',
                    color=color, ls='none', alpha=0.8)
    if len(nets) > 1:
        a4.plot([], [], 'o', ms=3.5, mfc='white', color=INK2, ls='none', label='single seeds (open)')
        a4.legend(fontsize=6.5, frameon=False, loc='lower right')
    else:
        a4.text(0.5, 0.5, 'capacity runs not finished', ha='center', va='center', transform=a4.transAxes,
                fontsize=8, color=INK2)
    a4.set_xscale('log')
    a4.set_xlabel('training simulations N')
    a4.set_ylabel('test log p (nats)')
    ticks = [n for n in SIZES if n >= 8000]
    a4.set_xticks(ticks)
    a4.set_xticklabels([f'{n / 1000:.0f}k' if n % 1000 == 0 else f'{n:,}' for n in ticks])
    a4.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    a4.set_title('(d) network size: 40 bands + map noise, 3 seeds each', fontsize=8)
    for ax in axes.ravel():
        ax.grid(True, color=GRID, lw=0.5)
        ax.set_axisbelow(True)
    a3.legend(fontsize=6.5, frameon=False, loc='upper center', bbox_to_anchor=(1.1, -0.22), ncol=3)
    fig.suptitle('Is the NPE limited by the training set, the network or the noise?', fontsize=9, y=0.99)
    fig.savefig(out / 'fig_checks.png', bbox_inches='tight')
    plt.close(fig)


def main():
    """Command line: see the module docstring."""
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, default=HERE / 'results_final')
    sub = p.add_subparsers(dest='cmd', required=True)
    a = sub.add_parser('prepare')
    a.add_argument('--data', type=Path, default=HERE / 'data' / 'rows_64440')
    a.add_argument('--asinh-scale', choices=('median', 'noise'), default='median',
                   help='asinh scale of the features: median |clean| per band (default) or the baseline '
                        'noise sd at zero signal (MOPED: 1)')
    a = sub.add_parser('train')
    a.add_argument('--variants', nargs='+', default=list(VARIANTS))
    a.add_argument('--workers', type=int, default=6)
    a.add_argument('--threads', type=int, default=3)
    a.add_argument('--epochs', type=int, default=2000, help='epoch cap (early stopping decides)')
    a = sub.add_parser('evaluate')
    a.add_argument('--variants', nargs='+', default=list(VARIANTS))
    a.add_argument('--samples', type=int, default=1000)
    a.add_argument('--threads', type=int, default=8)
    a.add_argument('--force', action='store_true')
    sub.add_parser('report')
    a = sub.add_parser('checks')
    a.add_argument('--variants', nargs='+', default=list(VARIANTS))
    a.add_argument('--threads', type=int, default=2)
    a.add_argument('--force', action='store_true', help='recompute folders already in checks.json')
    a = sub.add_parser('validate')
    a.add_argument('--variants', nargs='+', default=list(VARIANTS))
    a.add_argument('--validation', type=Path, default=HERE / 'data' / 'rows_validation_1096')
    a.add_argument('--samples', type=int, default=1000)
    a.add_argument('--threads', type=int, default=4)
    args = p.parse_args()
    commands = dict(prepare=cmd_prepare, train=cmd_train, evaluate=cmd_evaluate, report=cmd_report,
                    checks=cmd_checks, validate=cmd_validate)
    commands[args.cmd](args)


if __name__ == '__main__':
    main()
