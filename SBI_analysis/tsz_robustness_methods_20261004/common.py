"""Scientific contracts and numerical helpers for the seed/mask robustness experiment."""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.linalg import cholesky, solve_triangular
from scipy.sparse import csr_matrix, diags

OLD = Path('/lustre/work/kristero10/tsz_seed_mask_variation_20261002')
TRAIN = Path('/home/kristero10/sbi_reruns_20261001/SBI_analysis/tsz64k_sbi_local')
BEAMFIX = Path('/home/kristero10/tsz64k_beamfix_20260930')
FISHER = Path('/home/kristero10/tsz64k_fisher_b12_20261002/results_beam/fisher_forecast_b12.npz')
ROOT = Path('/lustre/work/kristero10/tsz_robustness_methods_20261004')
ELL = np.arange(80, 7980)
DL = ELL * (ELL + 1) / (2 * np.pi)
SEEDS = list(range(100, 120, 2))
MASKS = list(range(12345, 12355))
DISCOVERY_SEEDS = SEEDS[:6]
DISCOVERY_MASKS = MASKS[:5]
VALIDATION_SEEDS = SEEDS[6:]
VALIDATION_MASKS = MASKS[5:]
THETA = np.array([18.1, .497, 4.35, .154, -.00865, .0393, -.758, .731, .415])
PARAMETERS = ['P0', 'xc', 'beta', 'alpha_m_P0', 'alpha_m_xc', 'alpha_m_beta',
              'alpha_z_P0', 'alpha_z_xc', 'alpha_z_beta']
LABELS = [r'$P_0$', r'$x_c$', r'$\beta$', r'$\alpha_{P,M}$', r'$\alpha_{x,M}$',
          r'$\alpha_{\beta,M}$', r'$\alpha_{P,z}$', r'$\alpha_{x,z}$', r'$\alpha_{\beta,z}$']
NOISE_SEEDS = [39001001, 39001002]
EDGES = np.arange(80, 7981, 50)  # Exactly 158 bands; all original multipoles enter once.


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def band_operator():
    a = np.zeros((158, 7900))
    for i, (lo, hi) in enumerate(zip(EDGES[:-1], EDGES[1:])):
        m = (ELL >= lo) & (ELL < hi)
        a[i, m] = (2 * ELL[m] + 1) / np.sum(2 * ELL[m] + 1)
    assert np.allclose(a.sum(1), 1)
    return a


def noise_covariance(sd, rho, operator):
    """Band the saved beam-consistent SO covariance, preserving all fitted lag correlations."""
    diagonals, offsets = [sd**2], [0]
    for lag in range(1, len(rho)):
        values = rho[lag] * sd[lag:] * sd[:-lag]
        diagonals.extend([values, values]); offsets.extend([lag, -lag])
    cov = diags(diagonals, offsets, shape=(7900, 7900), format='csr')
    a = csr_matrix(operator)
    result = (a @ cov @ a.T).toarray()
    return (result + result.T) / 2


def safe_cholesky(covariance):
    """Only numerical jitter; record any jitter used, rather than silently changing the covariance."""
    c = (covariance + covariance.T) / 2
    scale = np.mean(np.diag(c))
    for jitter in [0., 1e-12, 1e-10, 1e-8]:
        try:
            return cholesky(c + jitter * scale * np.eye(len(c)), lower=True), jitter
        except np.linalg.LinAlgError:
            continue
    raise np.linalg.LinAlgError('Covariance is not positive definite')


def moped_weights(derivatives, covariance, prior_width):
    """Stable QR implementation of mean-score/MOPED compression in all nine parameter directions.

    For C=L L^T, whiten the prior-scaled derivatives, QR= L^-1 J diag(prior_width).
    W=L^-T Q maps a data row to nine summaries. W^T C W=I; no Fisher eigenmode is dropped.
    This preserves local mean-derivative Fisher information for the chosen input and covariance.
    Actual global inference and sky robustness still require the held-out tests.
    """
    l, jitter = safe_cholesky(covariance)
    j = solve_triangular(l, derivatives * prior_width, lower=True)
    singular = np.linalg.svd(j, compute_uv=False)
    if singular[-1] / singular[0] < 1e-14:
        raise ValueError('Nine derivative directions are numerically unresolved')
    q, _ = np.linalg.qr(j, mode='reduced')
    w = solve_triangular(l.T, q, lower=False)
    error = float(np.max(np.abs(w.T @ covariance @ w - np.eye(9))))
    if error > 1e-5:
        raise ValueError(f'MOPED covariance orthogonality failed: {error}')
    return w, dict(jitter=jitter, orthogonality_max_error=error,
                  derivative_condition=float(singular[0]/singular[-1]), directions=9)


def recipes():
    """Cuts bracket both large and small scales; soft weights retain the entire multipole range."""
    candidates = [dict(name='moped_noiseonly_full', covariance='noiseonly'),
                  dict(name='moped_full', covariance='total_noise')]
    for high in [6000, 5000, 4000, 3000]:
        candidates.append(dict(name=f'moped_lmax{high}', covariance='total_noise', lmax=high))
    for low in [280, 580, 1080]:
        candidates.append(dict(name=f'moped_lmin{low}', covariance='total_noise', lmin=low))
    candidates += [dict(name='moped_580_5000', covariance='total_noise', lmin=580, lmax=5000),
                   dict(name='moped_high_ell_taper', covariance='total_noise', taper=True)]
    for arm in ['simulation', 'mask', 'combined']:
        candidates.append(dict(name=f'moped_robust_{arm}', covariance='total_noise', nuisance=arm))
    candidates += [dict(name='moped_robust_combined_weak', covariance='total_noise', nuisance='combined', strength=.1),
                   dict(name='moped_sky_augmented', covariance='total_noise', nuisance='combined', augmentation=True),
                   dict(name='merlin_band50_full', covariance='total_noise', network='merlin')]
    for c in candidates:
        c.setdefault('lmin', 80); c.setdefault('lmax', 7980)
        c.setdefault('network', 'moped'); c.setdefault('strength', 1.)
    return candidates


def fit_transform(noisy, clean, pool):
    """Same noise-scale asinh as Merlin, fitted using training rows only."""
    residual = noisy[pool] - clean[pool]
    scale = 1.4826 * np.median(np.abs(residual - np.median(residual, axis=0)), axis=0)
    scale = np.maximum(scale, 1e-12)
    center = np.median(noisy[pool], axis=0)
    z = np.arcsinh((noisy[pool] - center) / scale)
    return dict(s_center=center, s_scale=scale, z_mean=z.mean(0), z_sd=np.maximum(z.std(0), 1e-12))


def transform(s, params):
    return ((np.arcsinh((s - params['s_center']) / params['s_scale']) - params['z_mean'])
            / params['z_sd']).astype(np.float32)


def add_sky_variation(clean, noisy, ratios, choices):
    """Fiducial-calibrated multiplicative signal augmentation, explicitly an approximation.

    Ratios come ONLY from discovery lightcones/masks. They multiply the signal;
    the original SO residual is retained. This does not simulate the altered
    signal-noise cross term exactly and is not claimed valid over the whole prior.
    Held-out real B12 skies and the fixed-sky prior test must both be reported.
    """
    return noisy + clean * (ratios[choices] - 1.)


def arm_indices(cases, arm, validation=False):
    if arm == 'simulation':
        seeds = VALIDATION_SEEDS if validation else SEEDS
        return [i for i,c in enumerate(cases) if c['mask_seed']==MASKS[0] and c['simulation_seed'] in seeds]
    if arm == 'mask':
        masks = VALIDATION_MASKS if validation else MASKS
        return [i for i,c in enumerate(cases) if c['simulation_seed']==SEEDS[0] and c['mask_seed'] in masks]
    seeds = VALIDATION_SEEDS if validation else SEEDS
    masks = VALIDATION_MASKS if validation else MASKS
    return [i for i,c in enumerate(cases) if c['simulation_seed'] in seeds and c['mask_seed'] in masks]
