"""Fisher forecast of the nine pressure-profile parameters at Battaglia+12 from the SO tSZ spectrum the SBI sees.

Data vector: D_ell of the masked SO split-cross spectrum at all 7,900 multipoles ell = 80..7979 (the Merlin / PCA
inputs before compression), at the B12 parameters. Every simulation shares the one HalfDome lightcone, so the SO map
noise is the only scatter; its mean is the noise-free masked spectrum of the same sky (exactly: the cross spectrum of
two independent noise splits has no noise bias).

Derivatives: central differences of noise-free mocks at B12 +- h, h = 0.5 % of the prior width (mocks/fisher_b12),
and at B12 +- 2h (painted on idark, idark_fisher/deriv) when present; the Fisher matrix uses the Richardson
combination J = (4 J_h - J_2h) / 3 (error O(h^4)), J_h and J_2h are kept as the step-size check. P0 is analytic:
D_ell scales exactly as P0^2 in the engine.

Covariance: residuals r = noisy - noise-free of noise realizations of the B12 sky: the 21 local draws of
mocks/pass1_B12/B12 and the idark draws (idark_fisher/noise, 50 split-seed pairs per job). Model
C = diag(s) R diag(s): s^2 the running mean of the residual variance over +-HALF multipoles, R the Toeplitz matrix of
the Welch autocorrelation (Hann segments of LAGS multipoles, half overlap) of the standardized residuals, positive
semi-definite by construction (it is the Fourier series of a non-negative spectral density). The f_sky 0.4 apodized
cap mask couples neighbouring multipoles (lag-1 correlation 0.55; the alternating pattern carries almost no noise),
so a diagonal covariance is wrong at the per-multipole level.
Calibration (with idark draws): two-fold cross-fitting over the idark jobs. The model fitted without one half
compresses the held-out residuals to t = J^T C^-1 r; their empirical covariance (known zero mean, Hartlap
(n - 10) / n) gives the calibrated Fisher matrix F_cal = M Cov(t)^-1 M with M = J^T C^-1 J, the Fisher information
of the 9 compressed numbers - a lower bound on the information whatever the model error, equal to M if the model is
right. Also chi^2 tests of the held-out residuals (all 7,900 multipoles, and t^T M^-1 t over its 9 dof).

Variants (each a 9 x 9 Fisher matrix in theta units): every multipole (the forecast), bands of width 2..100 and the
SBI's 40 bands (2 ell + 1 weights), Merlin's 75 PCA inputs (the OAS whitening of merlin_unbinned_prepare.py rebuilt
from the same 20 draws, checked against the stored B12 projection), the h and 2h derivatives, and the derivatives
smoothed over +-50 multipoles (removes the fine structure in ell that the one shared sky imprints).
Posteriors (prior widths w = high - low of the 64k flat prior): the Gaussian alone, and the Gaussian times the hard
prior box by rejection sampling, centred on B12 (forecast) or on the linearized maximum-likelihood point of the B12
observation (observed: noise pair 39001001/39001002, the fitted SO mock of every posterior figure).

Reads (--inputs, the local files copied as they are): pass1_B12/B12/{masked_clean_cl,noisy_cross_*}.npy,
fisher_b12/{plan.json,<case>/masked_clean_cl.npy,<case>/case.toml}, merlin/{V.npy,b12.npz,clean.json} (V the 75
kept PCA directions of results_merlin_unbinned/inputs/whitening_pca.npz, b12.npz and clean.json as stored there);
--root: noise_out/n*/B12/{masked_clean_cl,noisy_cross_*}.npy and deriv_out/<name>_{p2,m2}/ from idark_fisher.
Writes to --out: fisher_forecast_b12.json (numbers) and fisher_forecast_b12.npz (Fisher matrices, J, covariance
model, posterior samples).

    python fisher_forecast_b12.py --inputs <dir> --root <dir> --out <dir> [--lags 64] [--half 25]
        [--beam [--beamfix <beamfix root>] --merlin-whitening <whitening_pca.npz> --merlin-b12 <b12.npz>]
--beam: the beam-consistent noise of beamfix (r = B (r - r_nn) + B^2 r_nn with each draw's regenerated noise x noise
cross), the noise of the beam-fixed Merlin inputs (merlin_w420_prepare.py), whose whitening --merlin-whitening gives.

Run on idark (idark_fisher/fisher.pbs, root ~/tsz64k_fisher_b12_20261002) 2026-10-02 with --lags 64 / 256 / 1024 and
--half 5 / 25 / 100: calibrated marginal sd agree to 5 %; --lags 64 is the published one (results_fisher_b12/).
"""
import argparse
import json
import time
import tomllib
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular, toeplitz

LMAX = 7979
ELL = np.arange(80, LMAX + 1)                          # the SBI's multipoles (7,900)
DL = ELL * (ELL + 1) / (2 * np.pi)                     # C_ell -> D_ell
P = len(ELL)
NAMES = ['P0', 'xc', 'beta', 'alpha_m_P0', 'alpha_m_xc', 'alpha_m_beta', 'alpha_z_P0', 'alpha_z_xc', 'alpha_z_beta']
LOW = np.array([1.0, 0.025, 2.8, -0.6, -1.0, -0.2, -6.0, -1.5, -0.5])     # the 64k flat prior
HIGH = np.array([60.0, 4.0, 16.0, 1.5, 0.4, 0.4, 0.5, 3.0, 2.0])
WIDTH = HIGH - LOW
B12 = np.array([18.1, 0.497, 4.35, 0.154, -0.00865, 0.0393, -0.758, 0.731, 0.415])
OBS_PAIR = '39001001_39001002'                         # the fitted B12 observation (left out of every covariance)
SBI_EDGES = np.r_[np.arange(80, 7881, 200), 7980]      # the SBI's 40 bands
BAND_WIDTHS = (2, 5, 10, 20, 50, 100)                  # uniform bands from ell = 80, the last one shorter
MERLIN_SHRINKAGE = 0.9516697772481851                  # results_merlin_unbinned/inputs/clean.json, covariance
SAMPLES = 200_000                                      # posterior samples kept per boxed posterior
SMOOTH = 50                                            # +- multipoles of the 'smoothed derivatives' variant


def spectrum(path):
    """D_ell at ELL of a C_ell file (ell = 0..7979)."""
    return np.load(path)[ELL] * DL


def running_mean(x, half):
    """Running mean over +-half samples, the ends reflected."""
    kernel = np.ones(2 * half + 1) / (2 * half + 1)
    return np.convolve(np.pad(x, half, mode='reflect'), kernel, mode='valid')


def band_matrix(edges):
    """(n_bands, 7900) averaging matrix with 2 ell + 1 weights over [edges[k], edges[k+1])."""
    A = np.zeros((len(edges) - 1, P))
    for k, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        m = (ELL >= a) & (ELL < b)
        A[k, m] = (2 * ELL[m] + 1) / np.sum(2 * ELL[m] + 1)
    return A


# ---------------------------------------------------------------- inputs
def derivatives(inputs, root, d0):
    """J_h, J_2h (None without the idark cases) and the Richardson J, each (7900, 9) in D_ell per theta unit."""
    plan = json.loads((inputs / 'fisher_b12' / 'plan.json').read_text())
    h = np.array(plan['step'])
    np.testing.assert_allclose(plan['fiducial'], B12)
    assert plan['names'] == NAMES

    def central(folder, tags, step, check_cache=True):
        J = np.zeros((P, 9))
        J[:, 0] = 2 * d0 / B12[0]
        for i in range(1, 9):
            pair = []
            for tag, sign in zip(tags, (+1, -1)):
                case_dir = folder / f'{NAMES[i]}_{tag}'
                case = tomllib.loads((case_dir / 'case.toml').read_text())
                expect = B12.copy()
                expect[i] += sign * step[i]
                np.testing.assert_allclose(case['theta'], expect, rtol=0, atol=1e-12)
                if check_cache:
                    assert case['cache']['nodes'] == [256, 128, 64], case['cache']['nodes']
                pair.append(spectrum(case_dir / 'masked_clean_cl.npy'))
            J[:, i] = (pair[0] - pair[1]) / (2 * step[i])
        return J

    J1 = central(inputs / 'fisher_b12', ('p', 'm'), h)
    deriv2 = root / 'deriv_out' if root else None
    if deriv2 is not None and all((deriv2 / f'{n}_{t}' / 'case.toml').exists() for n in NAMES[1:] for t in ('p2', 'm2')):
        J2 = central(deriv2, ('p2', 'm2'), 2 * h)
        return h, J1, J2, (4 * J1 - J2) / 3
    return h, J1, None, J1


def residual_sets(inputs, root, d0, beam=None):
    """Noise residuals about the noise-free spectrum: {'local': (20, 7900), 'idark_<job>': (n, 7900), ...}, the
    observation residual (7900,) and the consistency of each idark job's noise-free B12 spectrum with the local one.

    beam: None (production noise) or dict(bl=(7900,) B_ell, local_nn=<folder>, idark_nn=<folder>) for the
    beam-consistent noise of beamfix: r = B (r - r_nn) + B^2 r_nn, r_nn the regenerated noise x noise cross
    (<folder>/B12_<seed1>_<seed2>.npy) of each draw; the local draws' and the observation's from local_nn
    (beamfix b12draws/nn), the idark draws' from idark_nn (nn_regen/nn)."""
    def fix(r, folder, pair):
        if beam is None:
            return r
        nn = spectrum(folder / f'B12_{pair}.npy')
        return beam['bl'] * (r - nn) + beam['bl'] ** 2 * nn

    b12 = inputs / 'pass1_B12' / 'B12'
    files = sorted(p for p in b12.glob('noisy_cross_*.npy') if OBS_PAIR not in p.name)
    pair = lambda p: p.stem.replace('noisy_cross_', '')                                      # noqa: E731
    local_nn = beam['local_nn'] if beam else None
    sets = {'local': np.stack([fix(spectrum(p) - d0, local_nn, pair(p)) for p in files])}
    obs = fix(spectrum(b12 / f'noisy_cross_{OBS_PAIR}.npy') - d0, local_nn, OBS_PAIR)
    clean_check = {}
    if root is not None:
        for job in sorted((root / 'noise_out').glob('n*')):
            folder = job / 'B12'
            draws = sorted(folder.glob('noisy_cross_*.npy'))
            if not draws or not (folder / 'case.toml').exists():      # job still running: case.toml comes last
                continue
            clean = spectrum(folder / 'masked_clean_cl.npy')
            clean_check[job.name] = float(np.max(np.abs(clean - d0) / d0))
            sets[f'idark_{job.name}'] = np.stack([fix(spectrum(p) - clean, beam['idark_nn'] if beam else None,
                                                      pair(p)) for p in draws])          # own noise-free spectrum
    return sets, obs, clean_check


# ---------------------------------------------------------------- covariance model
def welch_acf(z, lags):
    """Autocorrelation of the rows of z at lags 0..lags-1: Hann segments of length `lags` (half overlap, plus one
    segment ending at the last multipole), averaged over segments and rows. It is the Fourier series of the average
    windowed periodogram, a non-negative spectral density, so its Toeplitz matrices are positive semi-definite."""
    w = np.hanning(lags + 2)[1:-1]
    starts = sorted(set(range(0, z.shape[1] - lags + 1, lags // 2)) | {z.shape[1] - lags})
    acc = np.zeros(lags)
    norm = 0.0
    for s in starts:
        f = np.fft.rfft(z[:, s:s + lags] * w, n=2 * lags, axis=1)
        acc += np.fft.irfft(np.abs(f) ** 2, n=2 * lags, axis=1)[:, :lags].sum(0)
        norm += len(z) * np.sum(w ** 2)
    return acc / norm


def covariance_model(r, lags, half):
    """C = diag(s) R diag(s) from residuals r (n, 7900) of known zero mean. Returns C, s (7900,), rho (lags,)."""
    s = np.sqrt(running_mean(np.mean(r ** 2, 0), half))
    acf = welch_acf(r / s, lags)
    rho = acf / acf[0]
    column = np.zeros(P)
    column[:lags] = rho
    return toeplitz(column) * np.outer(s, s), s, rho


def spectral_density(rho, n=2049):
    """S(omega) = 1 + 2 sum_k rho_k cos(k omega) on omega = 0..pi (white noise: 1)."""
    om = np.linspace(0, np.pi, n)
    return om, 1 + 2 * np.cos(np.outer(om, np.arange(1, len(rho)))) @ rho[1:]


# ---------------------------------------------------------------- Fisher matrices
def fisher(J, C, A=None):
    """J^T C^-1 J of the data (A=None) or of the bands A d (A J, A C A^T)."""
    if A is not None:
        J, C = A @ J, A @ C @ A.T
    factor = cho_factor(C, lower=True)
    return J.T @ cho_solve(factor, J)


def merlin_projection(inputs, draws20, obs_d, whitening=None, b12=None):
    """G (7900, n_pca): Merlin's inputs are s = G^T d. whitening=None: merlin_unbinned_prepare.whitener + PCA, rebuilt
    from the same 20 B12 draws and checked against the stored shrinkage; otherwise the L and V of a whitening_pca.npz
    (e.g. merlin_w420_prepare.py). Both checked against the stored projection of the B12 observation (b12: its
    b12.npz, default inputs/merlin/b12.npz)."""
    if whitening is not None:
        z = np.load(whitening)
        G = solve_triangular(z['L'], z['V'], lower=True, trans='T')
        stored = np.load(b12)['s_obs'][0]
        check = dict(whitening=str(whitening), n_pca=int(G.shape[1]),
                     obs_projection_max_rel_diff=float(np.max(np.abs(G.T @ obs_d - stored)) / np.max(np.abs(stored))))
        assert check['obs_projection_max_rel_diff'] < 1e-6, check
        return G, check
    std = draws20.std(axis=0, ddof=1)
    X = (draws20 - draws20.mean(axis=0)) / std
    n, p = X.shape
    emp = X.T @ X / n                                    # sklearn _oas (assume_centered=True), written out
    alpha = np.mean(emp ** 2)
    mu = np.trace(emp) / p
    shrinkage = min((alpha + mu ** 2) / ((n + 1) * (alpha - mu ** 2 / p)), 1.0)
    corr = (1 - shrinkage) * emp
    corr.flat[::p + 1] += shrinkage * mu
    d = np.sqrt(np.diag(corr))
    L = np.linalg.cholesky(corr / np.outer(d, d) * np.outer(std, std))
    V = np.load(inputs / 'merlin' / 'V.npy')
    G = solve_triangular(L, V, lower=True, trans='T')    # L^-T V
    stored = np.load(inputs / 'merlin' / 'b12.npz')['s_obs'][0]
    check = dict(shrinkage=float(shrinkage), shrinkage_stored=MERLIN_SHRINKAGE,
                 obs_projection_max_rel_diff=float(np.max(np.abs(G.T @ obs_d - stored)) / np.max(np.abs(stored))))
    assert abs(shrinkage - MERLIN_SHRINKAGE) < 1e-10 and check['obs_projection_max_rel_diff'] < 1e-8, check
    return G, check


def inverse_sqrt(Fu):
    """Fisher matrix in prior-width units -> (covariance, a square root S with S S^T = covariance), by eigh: the
    binned and PCA matrices span ~12 decades and defeat inv + cholesky."""
    lam, vec = np.linalg.eigh((Fu + Fu.T) / 2)
    assert lam.min() > 0, f'Fisher matrix not positive definite: {lam.min():.3e}'
    return (vec / lam) @ vec.T, vec / np.sqrt(lam)


def summary(F):
    """Marginal sd (theta units and prior widths), the conditional sd, correlations and eigen-widths in prior widths."""
    Fw = F * np.outer(WIDTH, WIDTH)
    cov_w, _ = inverse_sqrt(Fw)
    sd_w = np.sqrt(np.diag(cov_w))
    lam, vec = np.linalg.eigh(Fw)
    order = np.argsort(lam)[::-1]
    return dict(marginal_sd=(sd_w * WIDTH).tolist(), marginal_sd_prior_widths=sd_w.tolist(),
                conditional_sd_prior_widths=(1 / np.sqrt(np.diag(Fw))).tolist(),
                correlation=(cov_w / np.outer(sd_w, sd_w)).round(4).tolist(),
                eigen_sd_prior_widths=(1 / np.sqrt(lam[order])).tolist(), eigenvectors=vec[:, order].T.round(4).tolist(),
                with_gaussian_prior_sd_prior_widths=np.sqrt(np.diag(inverse_sqrt(Fw + 12 * np.eye(9))[0])).tolist())


# ---------------------------------------------------------------- posteriors
def boxed_samples(F, centre, count, seed, max_proposals=100_000_000, chunk=2_000_000):
    """The Gaussian N(centre, F^-1) restricted to the prior box: (count, 9) samples, the fraction of the Gaussian
    inside the box and the sampler used. Works in u = (theta - low) / width (box [0, 1]^9). Rejection sampling when
    that fraction allows it; otherwise (a likelihood much wider than the box) importance sampling as
    so_nine_fisher.hard_prior_posterior: proposals from the likelihood times a Gaussian of precision 12 about the box
    centre, kept inside the box, reweighted by the inverse of that Gaussian and resampled."""
    rng = np.random.default_rng(seed)
    Fu = F * np.outer(WIDTH, WIDTH)
    c = (centre - LOW) / WIDTH
    _, root = inverse_sqrt(Fu)
    kept, proposed = [], 0
    while sum(len(k) for k in kept) < count and proposed < max_proposals:
        u = c + rng.standard_normal((chunk, 9)) @ root.T
        proposed += chunk
        kept.append(u[np.all((u >= 0) & (u <= 1), axis=1)])
    kept = np.concatenate(kept)
    inside = len(kept) / proposed
    if len(kept) >= count:
        return LOW + WIDTH * kept[:count], inside, 'rejection'
    precision = Fu + 12 * np.eye(9)
    mean = np.linalg.solve(precision, Fu @ c + 12 * 0.5 * np.ones(9))
    _, root = inverse_sqrt(precision)
    parts = []
    for _ in range(10):                                  # 20 M proposals: memory below 1 GB
        u = mean + rng.standard_normal((chunk, 9)) @ root.T
        parts.append(u[np.all((u >= 0) & (u <= 1), axis=1)])
    u = np.concatenate(parts)
    log_w = 6 * np.sum((u - 0.5) ** 2, axis=1)
    w = np.exp(log_w - log_w.max())
    w /= w.sum()
    ess = 1 / np.sum(w ** 2)
    assert ess > 20_000, f'importance sampling ESS {ess:.0f}'
    return LOW + WIDTH * u[rng.choice(len(u), size=count, p=w)], inside, f'importance (ESS {ess:.0f})'


def quantiles(s):
    q = np.quantile(s, [0.16, 0.5, 0.84], axis=0)
    return dict(median=q[1].tolist(), q16=q[0].tolist(), q84=q[2].tolist(),
                width68_over_prior=((q[2] - q[0]) / WIDTH).tolist(), sd_over_prior=(s.std(0) / WIDTH).tolist())


# ---------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--inputs', type=Path, required=True, help='folder with pass1_B12/, fisher_b12/, merlin/')
    p.add_argument('--root', type=Path, default=None, help='idark_fisher run root (noise_out/, deriv_out/)')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--lags', type=int, default=64,
                   help='Welch segment length = lags of the correlation model (64: the only one of 64 / 256 / 1024 '
                        'whose compressed statistics pass the held-out chi^2 test)')
    p.add_argument('--half', type=int, default=25, help='+- multipoles of the running mean of the variance')
    p.add_argument('--beam', action='store_true', help='beam-consistent noise (beamfix): needs the regenerated C_nn')
    p.add_argument('--beamfix', type=Path, default=Path.home() / 'tsz64k_beamfix_20260930',
                   help='beamfix run root: beam_bl.npy, b12draws/nn/ (C_nn of the local draws and the observation)')
    p.add_argument('--merlin-whitening', type=Path, default=None,
                   help="whitening_pca.npz of Merlin's inputs (default: rebuild the original 20-draw whitening)")
    p.add_argument('--merlin-b12', type=Path, default=None, help='b12.npz next to --merlin-whitening (s_obs check)')
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    d0 = spectrum(args.inputs / 'pass1_B12' / 'B12' / 'masked_clean_cl.npy')
    h, J1, J2, J = derivatives(args.inputs, args.root, d0)
    beam = None
    if args.beam:
        beam = dict(bl=np.load(args.beamfix / 'beam_bl.npy')[ELL], local_nn=args.beamfix / 'b12draws' / 'nn',
                    idark_nn=args.root / 'nn_regen' / 'nn')
    assert not (args.beam and args.merlin_whitening is None), \
        'the original Merlin whitening is for the production noise: give --merlin-whitening with --beam'
    sets, obs_r, clean_check = residual_sets(args.inputs, args.root, d0, beam)
    idark = sorted(k for k in sets if k.startswith('idark_'))
    all_r = np.concatenate([sets[k] for k in sets])
    print(f'derivatives: {"h and 2h (Richardson)" if J2 is not None else "h only"}; residuals: '
          f'{ {k: len(v) for k, v in sets.items()} }; idark noise-free vs local max rel diff: {clean_check}', flush=True)
    report = dict(names=NAMES, low=LOW.tolist(), high=HIGH.tolist(), b12=B12.tolist(), step_h=h.tolist(),
                  derivatives='richardson (4 J_h - J_2h) / 3' if J2 is not None else 'central, step h',
                  residuals={k: len(v) for k, v in sets.items()}, idark_clean_vs_local_max_rel=clean_check,
                  model=dict(lags=args.lags, variance_running_mean_half=args.half,
                             noise='beam-consistent (beamfix)' if args.beam else 'production'))

    # ---- the covariance model from all draws; its structure
    C, s, rho = covariance_model(all_r, args.lags, args.half)
    om, S = spectral_density(rho)
    quarters = []
    for q in range(4):
        sl = slice(q * P // 4, (q + 1) * P // 4)
        rq = welch_acf(all_r[:, sl] / s[sl], args.lags)
        rq = rq / rq[0]
        quarters.append(dict(ell=[int(ELL[sl][0]), int(ELL[sl][-1])], rho_1_to_5=rq[1:6].round(4).tolist(),
                             S_pi=float(spectral_density(rq)[1][-1]), S_0=float(spectral_density(rq)[1][0])))
    report['covariance'] = dict(rho_1_to_10=rho[1:11].round(5).tolist(), S_0=float(S[0]), S_pi=float(S[-1]),
                                S_min=float(S.min()), quarters=quarters,
                                min_eigenvalue_R=float(np.linalg.eigvalsh(C / np.outer(s, s)).min()))
    print('covariance: rho_1..5', rho[1:6].round(4).tolist(), f'S(0) {S[0]:.3f} S(pi) {S[-1]:.4f}', flush=True)

    # ---- Fisher matrices
    F = {'every multipole': fisher(J, C)}
    for dl in BAND_WIDTHS:
        F[f'bands of {dl}'] = fisher(J, C, band_matrix(np.r_[np.arange(80, 7980, dl), 7980]))
    F['40 SBI bands'] = fisher(J, C, band_matrix(SBI_EDGES))
    draws20 = sets['local'] + d0
    G, merlin_check = merlin_projection(args.inputs, draws20, obs_r + d0, args.merlin_whitening, args.merlin_b12)
    F['Merlin 75 PCA inputs'] = fisher(G.T @ J, G.T @ C @ G)
    F['every multipole, step h'] = fisher(J1, C)
    if J2 is not None:
        F['every multipole, step 2h'] = fisher(J2, C)
    Js = J.copy()
    Js[:, 1:] = np.stack([running_mean(J[:, i], SMOOTH) for i in range(1, 9)], 1)
    Js[:, 0] = 2 * running_mean(d0, SMOOTH) / B12[0]
    F['every multipole, derivatives smoothed'] = fisher(Js, C)
    report['merlin_projection_check'] = merlin_check
    report['fisher'] = {k: summary(v) for k, v in F.items()}
    for k, v in report['fisher'].items():
        print(f'{k:40s} marginal sd / prior width: ' + ' '.join(f'{x:.3f}' for x in v['marginal_sd_prior_widths']),
              flush=True)

    # ---- step-size check of the derivatives (fine structure: residual about a running mean over +-25)
    if J2 is not None:
        hp = lambda x: x - running_mean(x, 25)                                       # noqa: E731
        report['derivative_check'] = {NAMES[i]: dict(
            rel_l2_diff_h_2h=float(np.linalg.norm(J1[:, i] - J2[:, i]) / np.linalg.norm(J1[:, i])),
            fine_structure_corr_h_2h=float(np.corrcoef(hp(J1[:, i]), hp(J2[:, i]))[0, 1]),
            fine_structure_rms_ratio_2h_over_h=float(np.std(hp(J2[:, i])) / np.std(hp(J1[:, i]))))
            for i in range(1, 9)}

    # ---- where the information sits in the multipole index: the noise-scaled derivatives J / s split into Fourier
    # bands of ell (periods in multipoles); the correlation is Toeplitz, so nearly diagonal in these bands, and the
    # Fisher matrix of the derivatives low-passed below a period cut is (nearly) that of the data smoothed on that scale
    freq = np.fft.rfftfreq(P)                                   # cycles per multipole, 0 .. 1/2
    spectra_J = {k: np.fft.rfft(v / s[:, None], axis=0) for k, v in (('J', J), ('J_h', J1), ('J_2h', J2))
                 if v is not None}
    cuts = {}
    for period in (200, 50, 20, 10, 5, 3):
        keep = freq < 1 / period
        Jl = s[:, None] * np.fft.irfft(spectra_J['J'] * keep[:, None], n=P, axis=0)
        cuts[f'periods above {period}'] = summary(fisher(Jl, C))['marginal_sd_prior_widths']
    report['information_by_period'] = cuts
    floors = {}
    for eps in (1e-5, 1e-4, 1e-3, 1e-2):               # an extra white noise eps s^2: S(omega) floored near eps
        floors[f'{eps:g}'] = summary(fisher(J, C + eps * np.diag(s ** 2)))['marginal_sd_prior_widths']
    report['white_floor_test'] = floors
    if J2 is not None:
        coherence = {}
        for lo_f, hi_f, tag in ((0, 1 / 200, '>200'), (1 / 200, 1 / 20, '20-200'), (1 / 20, 1 / 4, '4-20'),
                                (1 / 4, 0.51, '2-4')):
            keep = (freq >= lo_f) & (freq < hi_f)
            a, b = spectra_J['J_h'][keep, 1:], spectra_J['J_2h'][keep, 1:]
            coherence[f'period {tag}'] = dict(
                coherence=(np.real(np.sum(a * np.conj(b), 0)) / np.sqrt(np.sum(np.abs(a) ** 2, 0)
                                                                      * np.sum(np.abs(b) ** 2, 0))).round(4).tolist(),
                amplitude_2h_over_h=np.sqrt(np.sum(np.abs(b) ** 2, 0) / np.sum(np.abs(a) ** 2, 0)).round(4).tolist())
        report['derivative_coherence_by_period'] = coherence
    print('information by period cut (marginal sd / prior width):', flush=True)
    for k, v in {**cuts, **{f'white floor {e}': v for e, v in floors.items()}}.items():
        print(f'  {k:28s} ' + ' '.join(f'{x:.3f}' for x in v), flush=True)

    # ---- validation and calibration on held-out idark draws (two-fold over the jobs)
    def held_out_tests(r_fit, r_test, label):
        """chi^2 of held-out residuals under the model fitted to r_fit; with >= 30 of them also the calibrated
        Fisher matrix of the compressed statistics (None otherwise)."""
        Cm, _, _ = covariance_model(r_fit, args.lags, args.half)
        factor = cho_factor(Cm, lower=True)
        Ju = J * WIDTH                                       # per prior width: the matrices stay well conditioned
        M = Ju.T @ cho_solve(factor, Ju)
        t = r_test @ cho_solve(factor, Ju)                   # (n, 9) compressed residuals J^T C^-1 r
        n = len(t)
        chi2_full = np.einsum('ij,ji->i', r_test, cho_solve(factor, r_test.T)) / P
        chi2_t = np.einsum('ij,ij->i', t @ inverse_sqrt(M)[0], t) / 9
        s_test = r_test @ G                                  # Merlin's 75 PCA inputs of the held-out residuals
        Cs = G.T @ Cm @ G
        chi2_pca = np.einsum('ij,ij->i', s_test @ np.linalg.inv(Cs), s_test) / G.shape[1]
        out = dict(fit_draws=len(r_fit), test_draws=n, chi2_full_over_dof_mean=float(chi2_full.mean()),
                   chi2_full_sd_expected=float(np.sqrt(2 / P / n)), chi2_compressed_over_dof_mean=float(chi2_t.mean()),
                   chi2_compressed_sd_expected=float(np.sqrt(2 / 9 / n)), chi2_pca_over_dof_mean=float(chi2_pca.mean()),
                   chi2_pca_sd_expected=float(np.sqrt(2 / G.shape[1] / n)))
        F_cal = None
        if n >= 30:
            cov_t = t.T @ t / n                              # known zero mean
            out['eigen_cov_t_over_M'] = np.sort(np.linalg.eigvals(np.linalg.solve(M, cov_t)).real).tolist()
            F_cal = M @ np.linalg.solve(cov_t, M) * (n - 9 - 1) / n / np.outer(WIDTH, WIDTH)   # theta units
            out['calibrated_over_model_marginal_sd'] = (np.array(summary(F_cal)['marginal_sd'])
                                                        / np.array(summary(M / np.outer(WIDTH, WIDTH))['marginal_sd'])).tolist()
            print(f'{label}: chi2/dof full {out["chi2_full_over_dof_mean"]:.4f} (+-{out["chi2_full_sd_expected"]:.4f}), '
                  f'compressed {out["chi2_compressed_over_dof_mean"]:.3f} (+-{out["chi2_compressed_sd_expected"]:.3f}), '
                  f'Merlin PCA inputs {out["chi2_pca_over_dof_mean"]:.3f} (+-{out["chi2_pca_sd_expected"]:.3f}); '
                  'calibrated/model sd ' + ' '.join(f'{x:.3f}' for x in out['calibrated_over_model_marginal_sd']),
                  flush=True)
        return out, F_cal
    calibration, F_cals = {}, []
    if len(idark) >= 2:
        halves = (idark[:len(idark) // 2], idark[len(idark) // 2:])
        for a, b in (halves, halves[::-1]):
            r_fit = np.concatenate([sets['local']] + [sets[k] for k in a])
            r_test = np.concatenate([sets[k] for k in b])
            out, F_cal = held_out_tests(r_fit, r_test, f'fit local+{a[0]}..{a[-1]}, test {b[0]}..{b[-1]}')
            calibration[f'test_{b[0]}..{b[-1]}'] = out
            F_cals.append(F_cal)
        F['every multipole, calibrated'] = np.mean(F_cals, axis=0)
        report['fisher']['every multipole, calibrated'] = summary(F['every multipole, calibrated'])
    else:                       # only the 21 local draws: leave-one-out over them (a weak test, 20 draws)
        loo = [held_out_tests(np.delete(sets['local'], k, 0), sets['local'][k:k + 1], f'leave-one-out {k}')[0]
               for k in range(len(sets['local']))]
        calibration['leave_one_out_local'] = dict(
            chi2_full_over_dof_mean=float(np.mean([o['chi2_full_over_dof_mean'] for o in loo])),
            chi2_compressed_over_dof_mean=float(np.mean([o['chi2_compressed_over_dof_mean'] for o in loo])),
            chi2_pca_over_dof_mean=float(np.mean([o['chi2_pca_over_dof_mean'] for o in loo])), draws=len(loo))
    report['calibration'] = calibration
    # the observation under the model (all draws; the observation is in none of them)
    factor = cho_factor(C, lower=True)
    report['observation_chi2_over_dof'] = float(obs_r @ cho_solve(factor, obs_r) / P)

    # ---- posteriors of the forecast (every multipole): Gaussian alone and times the prior box
    Fm = F['every multipole']
    # linearized maximum-likelihood point of the observation minus B12 (solved per prior width)
    shift = WIDTH * np.linalg.solve(Fm * np.outer(WIDTH, WIDTH), (J * WIDTH).T @ cho_solve(factor, obs_r))
    posts, npz = {}, {}
    for label, centre in (('forecast', B12), ('observed', B12 + shift)):
        samples, inside, method = boxed_samples(Fm, centre, SAMPLES, seed=20261002)
        posts[label] = dict(centre=centre.tolist(), gaussian_mass_inside_box=inside, sampler=method,
                            boxed=quantiles(samples))
        npz[f'boxed_{label}'] = samples.astype(np.float32)
    posts['observed']['shift_over_marginal_sd'] = (shift / np.array(report['fisher']['every multipole']['marginal_sd'])).tolist()
    for key in ('Merlin 75 PCA inputs', '40 SBI bands', 'every multipole, calibrated'):
        if key in F:
            samples, inside, method = boxed_samples(F[key], B12, SAMPLES, seed=20261003)
            posts[f'forecast, {key}'] = dict(gaussian_mass_inside_box=inside, sampler=method, boxed=quantiles(samples))
            npz[f'boxed_forecast_{key.replace(" ", "_").replace(",", "")}'] = samples.astype(np.float32)
    report['posteriors'] = posts
    report['seconds'] = time.time() - t0
    (args.out / 'fisher_forecast_b12.json').write_text(json.dumps(report, indent=1) + '\n')
    np.savez_compressed(args.out / 'fisher_forecast_b12.npz', names=np.array(NAMES), keys=np.array(list(F)),
                        fisher=np.stack(list(F.values())), J=J, J_h=J1, J_2h=J2 if J2 is not None else np.zeros(0),
                        sd=s, rho=rho, d0=d0, low=LOW, high=HIGH, b12=B12, **npz)
    print(f'boxed forecast: Gaussian mass inside the box {posts["forecast"]["gaussian_mass_inside_box"]:.4f}; observed shift / sd',
          np.round(posts['observed']['shift_over_marginal_sd'], 2).tolist())
    print(f'written {args.out} in {report["seconds"]:.0f} s')


if __name__ == '__main__':
    main()
