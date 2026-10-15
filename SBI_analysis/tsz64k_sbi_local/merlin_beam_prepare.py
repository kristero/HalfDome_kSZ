"""Merlin inputs from the beam-consistent spectra and with a fiducial-free noise-only whitening (idark).

The frozen engine beams the signal but adds the beam-deconvolved SO noise without the beam. beamfix/ regenerated each
row's noise x noise cross spectrum C_nn from its seeds; with the stored masked clean C_ss and production noisy C_noisy
the beam-consistent spectrum is, at every multipole (beamfix.correct, as the staged 40-band series uses it),

    C_beam = C_ss + B_ell (C_noisy - C_ss - C_nn) + B_ell^2 C_nn

Three input sets are built in one pass over the 64,440 analysed rows (data/rows_64440 order):
  beam_b12   beam-consistent spectra, whitened as merlin_unbinned_prepare.py does (the covariance of the 20 B12 noise
             draws other than the observation's, made beam-consistent: OAS shrinkage of the correlation, measured
             variances) -> results_merlin_unbinned_beam_asinh_p30            (MERLIN_TAG=_beam_asinh_p30)
  beam_nn    beam-consistent spectra, whitened with the noise-only covariance of the regenerated noise crosses
             B_ell^2 C_nn of the 61,440 training-pool rows (the same OAS estimator as the B12 draws, no signal term;
             the test rows' noise is left out)
             -> results_merlin_unbinned_beam_nnwhite_asinh_p30                (MERLIN_TAG=_beam_nnwhite_asinh_p30)
  prod_nn    the production spectra, whitened with the noise-only covariance of C_nn
             -> results_merlin_unbinned_nnwhite_asinh_p30                     (MERLIN_TAG=_nnwhite_asinh_p30)
The report also gives the noise-only estimate from only the first 1,000 pool realizations, for comparison.
(The production spectra with the B12 whitening are the existing results_merlin_unbinned_asinh_p30.)
Each set: Cholesky factor L of its covariance, the PCA of the whitened noise-free training-pool spectra (top q = 500,
merlin's cut S / S.sum() * 100 > 0.001), projections s = (L^-1 x) V, then merlin_compression.write_features (asinh).
The B12 observation and its 21 noise draws enter every set in that set's noise (production or beam-consistent; their
C_nn regenerated in <beamfix>/b12draws/nn, the observation's also in <beamfix>/nn).
Hard checks (differences over the row's largest band power): the uploaded noisy and clean spectra rebin to the
rows_64440 band powers (1e-9); the corrected spectra rebin to the staged run's beam40 band powers (1e-9) and equal its
per-multipole beam_dl_unbinned (1e-12); the B12
observation's spectrum is the stored one; the observation's two regenerated C_nn agree.

    python merlin_beam_prepare.py
Writes <set inputs>/ (s_noisy.npy, s_clean.npy (64440, n_pca), whitening_pca.npz, b12.npz), the features, and
results_merlin_unbinned_beam/inputs/prepare_whitening.json (checks, covariances, PCA sizes of all sets).
"""
import json
from pathlib import Path
import time

import numpy as np
from scipy.linalg import solve_triangular

HERE = Path(__file__).resolve().parent
BEAMFIX = Path.home() / 'tsz64k_beamfix_20260930'
B12DRAWS = BEAMFIX / 'b12draws'
B12_MOCKS = HERE / 'mocks' / 'pass1_B12' / 'B12'
OBS_PAIR = '39001001_39001002'
ELL = np.arange(80, 7980)
DL = ELL * (ELL + 1) / (2 * np.pi)
EDGES = np.r_[np.arange(80, 7881, 200), 7980]
Q, EPS_PERCENT, CHUNK = 500, 0.001, 4096
# set -> (data, whitening, features folder, inputs folder)
SETS = dict(beam_b12=('beam', 'b12', 'results_merlin_unbinned_beam_asinh_p30', 'results_merlin_unbinned_beam/inputs'),
            beam_nn=('beam', 'nn', 'results_merlin_unbinned_beam_nnwhite_asinh_p30',
                     'results_merlin_unbinned_beam_nnwhite/inputs'),
            prod_nn=('production', 'nn', 'results_merlin_unbinned_nnwhite_asinh_p30',
                     'results_merlin_unbinned_nnwhite/inputs'))
REPORT = HERE / 'results_merlin_unbinned_beam' / 'inputs' / 'prepare_whitening.json'


def rebin40(x):
    w = 2 * ELL + 1
    return np.stack([np.sum(x[..., a - 80:b - 80] * w[a - 80:b - 80], axis=-1) / np.sum(w[a - 80:b - 80])
                     for a, b in zip(EDGES[:-1], EDGES[1:])], axis=-1)


def dl(path):
    cl = np.load(path)
    assert cl.shape == (7980,) and np.isfinite(cl).all(), path
    return cl[ELL] * DL


def correct(clean, noisy, nn, bl):
    """beamfix.correct: C_ss + B (C_noisy - C_ss - C_nn) + B^2 C_nn (D_ell, any leading shape)."""
    return clean + bl * (noisy - clean - nn) + bl ** 2 * nn


def white(L, x):
    return solve_triangular(L, np.asarray(x, dtype=np.float64).T, lower=True, check_finite=False).T


def cholesky(C):
    """Cholesky factor of C; if round-off makes C indefinite, the smallest jitter (relative to its mean diagonal)."""
    for jitter in (0.0, 1e-12, 1e-10, 1e-8):
        try:
            return np.linalg.cholesky(C + jitter * np.mean(np.diag(C)) * np.eye(len(C))), jitter
        except np.linalg.LinAlgError:
            continue
    raise np.linalg.LinAlgError('covariance not positive definite')


def b12_covariance(draws):
    """merlin_unbinned_prepare.whitener on given draws: OAS-shrunk correlation times the measured variances."""
    from sklearn.covariance._shrunk_covariance import _oas
    std = draws.std(axis=0, ddof=1)
    assert np.all(std > 0)
    standardised = (draws - draws.mean(axis=0)) / std
    correlation, shrinkage = _oas(standardised, assume_centered=True)
    correlation /= np.sqrt(np.outer(np.diag(correlation), np.diag(correlation)))
    return correlation * np.outer(std, std), dict(draws=int(len(draws)), oas_correlation_shrinkage=float(shrinkage))


def pca(L, clean, pool):
    """merlin_unbinned_prepare.cmd_clean: PCA of the whitened noise-free pool spectra -> (V, top, kept, mean)."""
    gram = np.zeros((7900, 7900))
    total = np.zeros(7900)
    for start in range(0, len(pool), CHUNK):
        w = white(L, clean[pool[start:start + CHUNK]])
        gram += w.T @ w
        total += w.sum(axis=0)
    mean_white = total / len(pool)
    eigenvalues, eigenvectors = np.linalg.eigh(gram / len(pool) - np.outer(mean_white, mean_white))
    eigenvalues, eigenvectors = np.clip(eigenvalues[::-1], 0, None), eigenvectors[:, ::-1]
    top = np.sqrt(eigenvalues * len(pool))[:Q]
    kept = top / top.sum() * 100 > EPS_PERCENT
    return eigenvectors[:, :Q][:, kept], top, kept, mean_white


def main():
    t0 = time.time()
    bl = np.load(BEAMFIX / 'beam_bl.npy')[ELL]
    row_id = np.load(HERE / 'data' / 'rows_64440' / 'row_id.npy')
    clean40 = np.load(HERE / 'data' / 'rows_64440' / 'clean_dl.npy')
    noisy40 = np.load(HERE / 'data' / 'rows_64440' / 'noisy_dl.npy')
    order = np.load(HERE / 'results_final' / 'split.npz')['order']
    position = {int(r): i for i, r in enumerate(row_id)}
    n = len(row_id)
    clean, noisy_all, beam = (np.full((n, 7900), np.nan) for _ in range(3))
    # the regenerated noise crosses D_nn of the training-pool rows: the noise-only covariance (test rows left out)
    pool_index = {int(q): k for k, q in enumerate(order)}
    nn_pool = np.full((len(order), 7900), np.nan)
    worst = dict(noisy40=0.0, clean40=0.0, beam40=0.0, beam_unbinned=0.0)
    for stage in sorted((BEAMFIX / 'upload').glob('stage*')):
        built = BEAMFIX / 'stages' / stage.name
        beam40_stage, beam_unbinned_stage = {}, {}
        if (built / 'beam40.npy').exists():
            built_rows = np.load(built / 'rows.npy').tolist()
            beam40_stage = dict(zip(built_rows, np.load(built / 'beam40.npy')))
            if (built / 'beam_dl_unbinned.npy').exists():          # the staged build's own per-multipole spectra
                beam_unbinned_stage = dict(zip(built_rows, np.load(built / 'beam_dl_unbinned.npy', mmap_mode='r')))
        for batch in sorted(stage.glob('b*')):
            assert (batch / 'UPLOADED').exists(), f'{batch} is not fully uploaded'
            rows = np.load(batch / 'rows.npy').tolist()
            clean_batch = np.load(batch / 'clean_dl_unbinned.npy')
            for i, r in enumerate(rows):
                if r not in position:
                    continue                                   # a design row outside the 64,440 analysed rows
                p = position[r]
                noisy = dl(batch / 'noisy' / f'{r:05d}.npy')
                nn = dl(BEAMFIX / 'nn' / f'r{r:05d}.npy')
                clean[p], noisy_all[p] = clean_batch[i], noisy
                beam[p] = correct(clean_batch[i], noisy, nn, bl)
                if p in pool_index:
                    nn_pool[pool_index[p]] = nn
                # differences over the row's largest band power: a noise-dominated band of the signed cross spectrum
                # can sit near zero, where a per-band ratio only measures round-off
                worst['noisy40'] = max(worst['noisy40'], float(np.max(np.abs(rebin40(noisy) - noisy40[p]))
                                                                     / np.max(np.abs(noisy40[p]))))
                worst['clean40'] = max(worst['clean40'], float(np.max(np.abs(rebin40(clean[p]) - clean40[p]))
                                                                     / np.max(np.abs(clean40[p]))))
                if r in beam40_stage:
                    b40 = beam40_stage[r]
                    worst['beam40'] = max(worst['beam40'], float(np.max(np.abs(rebin40(beam[p]) - b40))
                                                                       / np.max(np.abs(b40))))
                if r in beam_unbinned_stage:
                    ref = np.asarray(beam_unbinned_stage[r])
                    worst['beam_unbinned'] = max(worst['beam_unbinned'], float(np.max(np.abs(beam[p] - ref))
                                                                               / np.max(np.abs(ref))))
        print(f'{stage.name}: gathered, {time.time() - t0:.0f} s; checks so far {worst}', flush=True)
    assert np.isfinite(clean).all() and np.isfinite(beam).all(), 'rows missing from the uploads or nn/'
    assert np.isfinite(nn_pool).all(), 'pool rows missing from nn/'
    assert worst['noisy40'] < 1e-9 and worst['clean40'] < 1e-9 and worst['beam40'] < 1e-9, worst
    assert worst['beam_unbinned'] < 1e-12, worst
    # the noise-only covariance: the B12 recipe (OAS-shrunk correlation, measured variances) on 61,440 realizations;
    # the same from 1,000 realizations only, for the report
    cov_nn, nn_info = b12_covariance(nn_pool)
    cov_1k, info_1k = b12_covariance(nn_pool[:1000])
    sd_61k, sd_1k = np.sqrt(np.diag(cov_nn)), np.sqrt(np.diag(cov_1k))
    lag1 = lambda C: float(np.mean(np.diag(C, 1) / np.sqrt(np.diag(C)[:-1] * np.diag(C)[1:])))
    comparison_1k = dict(shrinkage_1000=info_1k['oas_correlation_shrinkage'],
                         shrinkage_61440=nn_info['oas_correlation_shrinkage'],
                         sd_ratio_1000_over_61440_median_abs_dev=float(np.median(np.abs(sd_1k / sd_61k - 1))),
                         lag1_correlation_61440=lag1(cov_nn), lag1_correlation_1000=lag1(cov_1k))
    del nn_pool, cov_1k
    print(f'noise-only covariance: {comparison_1k}, {time.time() - t0:.0f} s', flush=True)

    # the B12 observation and its 21 noise draws, in both noises
    import analyze_64k as A
    obs = np.load(A.TEST256 / 'cluster_snapshot' / 'diagnostic_256' / 'observations' / 'Battaglia12.npz')
    clean_b12 = obs['clean_dl_unbinned']
    pairs = [p.stem.replace('noisy_cross_', '') for p in sorted(B12_MOCKS.glob('noisy_cross_*.npy'))]
    assert len(pairs) == 21 and OBS_PAIR in pairs
    k_obs = pairs.index(OBS_PAIR)
    noisy_draws = np.stack([dl(B12_MOCKS / f'noisy_cross_{p}.npy') for p in pairs])
    nn_draws = np.stack([dl(B12DRAWS / 'nn' / f'B12_{p}.npy') for p in pairs])
    beam_draws = correct(clean_b12[None], noisy_draws, nn_draws, bl)
    check_obs_noisy = float(np.max(np.abs(noisy_draws[k_obs] - obs['noisy_dl_unbinned'])
                                   / np.abs(obs['noisy_dl_unbinned'])))
    nn_main = dl(BEAMFIX / 'nn' / f'B12_{OBS_PAIR}.npy')
    check_obs_nn = float(np.max(np.abs(nn_draws[k_obs] - nn_main)) / np.max(np.abs(nn_main)))
    assert check_obs_noisy < 1e-9 and check_obs_nn < 1e-6, (check_obs_noisy, check_obs_nn)
    data = dict(production=(noisy_all, noisy_draws), beam=(beam, beam_draws))

    # the noise-only scatter against the B12 draws' scatter (production noise), for the report
    sd_b12 = np.delete(noisy_draws, k_obs, axis=0).std(axis=0, ddof=1)
    ratio = sd_b12 / sd_61k
    report = dict(rows=int(n), source=str(BEAMFIX), correction='C_ss + B (C_noisy - C_ss - C_nn) + B^2 C_nn',
                  checks=dict(worst, b12_obs_noisy=check_obs_noisy, b12_obs_nn_main_vs_b12draws=check_obs_nn),
                  noise_only=dict(nn_info, rows=int(len(order)), estimate_from_1000_rows=comparison_1k,
                                  b12_sd_over_noise_only_sd={str(l): float(ratio[l - 80])
                                                             for l in (100, 300, 1000, 2000, 3000, 5000, 7000)}),
                  sets={})
    import merlin_compression as MC
    import os
    only = os.environ.get('BEAM_PREPARE_SETS')            # optional comma-separated subset of SETS
    for name, (kind, whitening, features, inputs_dir) in SETS.items():
        if only and name not in only.split(','):
            continue
        spectra, draws = data[kind]
        if whitening == 'b12':
            C, cov_info = b12_covariance(np.delete(draws, k_obs, axis=0))
        else:
            weight = bl ** 2 if kind == 'beam' else np.ones(7900)
            C = cov_nn * np.outer(weight, weight)              # the noise-only crosses of this set's noise
            cov_info = dict(nn_info, rows=int(len(order)),
                            estimator='OAS-shrunk correlation x measured variances of the regenerated C_nn of the pool'
                            + (', times B_ell^2 B_ell^2' if kind == 'beam' else ''))
        L, jitter = cholesky(C)
        cov_info.update(jitter=jitter, diag_sd_range=[float(np.sqrt(C.diagonal().min())),
                                                      float(np.sqrt(C.diagonal().max()))])
        V, top, kept, mean_white = pca(L, clean, order)
        s_clean = np.concatenate([white(L, clean[i:i + CHUNK]) @ V for i in range(0, n, CHUNK)])
        s_noisy = np.concatenate([white(L, spectra[i:i + CHUNK]) @ V for i in range(0, n, CHUNK)])
        s_draws = white(L, draws) @ V
        s_obs = s_draws[k_obs:k_obs + 1]
        out_inputs = HERE / inputs_dir
        out_inputs.mkdir(parents=True, exist_ok=True)
        np.save(out_inputs / 's_noisy.npy', s_noisy)
        np.save(out_inputs / 's_clean.npy', s_clean)
        np.savez(out_inputs / 'whitening_pca.npz', L=L, V=V, singular_values_top=top, kept=kept,
                 train_mean_white=mean_white)
        np.savez(out_inputs / 'b12.npz', s_obs=s_obs, s_draws=s_draws, pairs=np.array(pairs))
        pca_info = dict(q=Q, eps_v_percent=EPS_PERCENT, n_pca=int(kept.sum()), train_rows=int(len(order)))
        report['sets'][name] = dict(data=kind, whitening=whitening, covariance=cov_info, pca=pca_info,
                                    features=features)
        info = dict(source=f'{kind} spectra, all 7,900 multipoles, {whitening} whitening (merlin_beam_prepare.py)',
                    covariance=cov_info, pca=pca_info, n_pca=int(kept.sum()), data_checks=report['checks'])
        MC.write_features(HERE / features, s_noisy, s_obs, s_draws, info, s_clean, 'asinh')
        print(f'{name}: n_pca {int(kept.sum())}, {time.time() - t0:.0f} s', flush=True)
    report['seconds'] = time.time() - t0
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=1) + '\n')
    print(json.dumps(report, indent=1), flush=True)


if __name__ == '__main__':
    main()
