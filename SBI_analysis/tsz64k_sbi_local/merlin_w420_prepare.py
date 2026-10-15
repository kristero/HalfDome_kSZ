"""Merlin inputs from the beam-consistent spectra, whitened with the noise covariance of 420 B12 noise realizations.

Runs on idark in ~/sbi_reruns_20261001/SBI_analysis/tsz64k_sbi_local, next to merlin_beam_prepare.py (read-only
imports: dl, correct, white, cholesky, pca). merlin_beam_prepare.py whitens its 'beam_b12' set with the OAS-shrunk
covariance of the 20 B12 draws (shrinkage 0.95: almost diagonal). Here the B12 noise is the 20 draws plus the 400 new
draws of the Fisher forecast (~/tsz64k_fisher_b12_20261002: noise_out/ painted with paint_mocks.jl, seeds
950001..950800; their noise x noise crosses C_nn regenerated from the seeds in nn_regen/ with beamfix regen_noise.jl),
all made beam-consistent like every row, C_beam = C_ss + B (C_noisy - C_ss - C_nn) + B^2 C_nn.

Covariance: fisher_forecast_b12.covariance_model on the 420 beam-consistent residuals about the noise-free spectrum
(running-mean variances, Toeplitz Welch correlation with 64-multipole Hann segments: the model whose compressed Fisher
statistics pass the held-out chi^2 test), plus a white floor FLOOR x variance: the model's correlation matrix has a
smallest eigenvalue ~1e-6 (the alternating-multipole pattern carries almost no noise), and the floor keeps whitening
from amplifying that one poorly known direction ~1,000-fold. Then, as merlin_unbinned_prepare.py / merlin_beam_prepare.py:
Cholesky factor L, PCA of the whitened noise-free training-pool spectra (top 500, S / S.sum() * 100 > 0.001), s = (L^-1 x) V,
merlin_compression.write_features (asinh inputs).

Checks (asserted): uploaded spectra rebin to the rows_64440 band powers and the corrected ones equal the staged run's
beam_dl_unbinned (as merlin_beam_prepare.py); every new draw's C_nn belongs to its noisy spectrum (slope of
noisy - clean on C_nn at ell >= 6000 within 0.85..1.15; wrong pairs give 0) and, for the hashed tasks, the regenerated
noise maps have the sha256 the painting recorded; the B12 observation is the stored one.

    python merlin_w420_prepare.py          writes results_merlin_unbinned_beam_w420/inputs/ and the features of
                                           results_merlin_unbinned_beam_w420_asinh_p30 (MERLIN_TAG=_beam_w420_asinh_p30)
"""
import json
import sys
import time
import tomllib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FISHER_ROOT = Path.home() / 'tsz64k_fisher_b12_20261002'
sys.path.insert(0, str(FISHER_ROOT))
import merlin_beam_prepare as MBP                                        # noqa: E402
from fisher_forecast_b12 import covariance_model                          # noqa: E402

LAGS, HALF, FLOOR = 64, 25, 1e-3
OBS_PAIR = MBP.OBS_PAIR
INPUTS = HERE / 'results_merlin_unbinned_beam_w420' / 'inputs'
FEATURES = HERE / 'results_merlin_unbinned_beam_w420_asinh_p30'


def gather_rows(bl):
    """Noise-free and beam-consistent spectra (64440, 7900) of the analysed rows, data/rows_64440 order, with the
    checks of merlin_beam_prepare.main."""
    row_id = np.load(HERE / 'data' / 'rows_64440' / 'row_id.npy')
    clean40 = np.load(HERE / 'data' / 'rows_64440' / 'clean_dl.npy')
    noisy40 = np.load(HERE / 'data' / 'rows_64440' / 'noisy_dl.npy')
    position = {int(r): i for i, r in enumerate(row_id)}
    clean, beam = (np.full((len(row_id), 7900), np.nan) for _ in range(2))
    worst = dict(noisy40=0.0, clean40=0.0, beam_unbinned=0.0)
    t0 = time.time()
    for stage in sorted((MBP.BEAMFIX / 'upload').glob('stage*')):
        built = MBP.BEAMFIX / 'stages' / stage.name
        reference = {}
        if (built / 'beam_dl_unbinned.npy').exists():
            reference = dict(zip(np.load(built / 'rows.npy').tolist(),
                                 np.load(built / 'beam_dl_unbinned.npy', mmap_mode='r')))
        for batch in sorted(stage.glob('b*')):
            assert (batch / 'UPLOADED').exists(), f'{batch} is not fully uploaded'
            clean_batch = np.load(batch / 'clean_dl_unbinned.npy')
            for i, r in enumerate(np.load(batch / 'rows.npy').tolist()):
                if r not in position:
                    continue
                p = position[r]
                noisy = MBP.dl(batch / 'noisy' / f'{r:05d}.npy')
                nn = MBP.dl(MBP.BEAMFIX / 'nn' / f'r{r:05d}.npy')
                clean[p] = clean_batch[i]
                beam[p] = MBP.correct(clean_batch[i], noisy, nn, bl)
                worst['noisy40'] = max(worst['noisy40'], float(np.max(np.abs(MBP.rebin40(noisy) - noisy40[p]))
                                                                     / np.max(np.abs(noisy40[p]))))
                worst['clean40'] = max(worst['clean40'], float(np.max(np.abs(MBP.rebin40(clean[p]) - clean40[p]))
                                                                     / np.max(np.abs(clean40[p]))))
                if r in reference:
                    ref = np.asarray(reference[r])
                    worst['beam_unbinned'] = max(worst['beam_unbinned'],
                                                 float(np.max(np.abs(beam[p] - ref)) / np.max(np.abs(ref))))
        print(f'{stage.name}: gathered, {time.time() - t0:.0f} s; checks {worst}', flush=True)
    assert np.isfinite(clean).all() and np.isfinite(beam).all(), 'rows missing from the uploads or nn/'
    assert worst['noisy40'] < 1e-9 and worst['clean40'] < 1e-9 and worst['beam_unbinned'] < 1e-12, worst
    return clean, beam, worst


def b12_draws(bl):
    """Beam-consistent B12 noise realizations: the 21 local draws (the observation's pair included) and the 400 idark
    draws. Returns (pairs21, beam21 (21, 7900), clean_b12 (7900,), residuals400 (400, 7900), checks)."""
    import analyze_64k as A
    obs = np.load(A.TEST256 / 'cluster_snapshot' / 'diagnostic_256' / 'observations' / 'Battaglia12.npz')
    clean_b12 = obs['clean_dl_unbinned']
    pairs = [p.stem.replace('noisy_cross_', '') for p in sorted(MBP.B12_MOCKS.glob('noisy_cross_*.npy'))]
    assert len(pairs) == 21 and OBS_PAIR in pairs
    noisy21 = np.stack([MBP.dl(MBP.B12_MOCKS / f'noisy_cross_{p}.npy') for p in pairs])
    nn21 = np.stack([MBP.dl(MBP.B12DRAWS / 'nn' / f'B12_{p}.npy') for p in pairs])
    k_obs = pairs.index(OBS_PAIR)
    checks = dict(obs_noisy=float(np.max(np.abs(noisy21[k_obs] - obs['noisy_dl_unbinned'])
                                         / np.abs(obs['noisy_dl_unbinned']))))
    assert checks['obs_noisy'] < 1e-9, checks
    beam21 = MBP.correct(clean_b12[None], noisy21, nn21, bl)
    high = MBP.ELL >= 6000                       # noise x noise dominated: the slope of noisy - clean on C_nn is 1
    residuals, slopes, hashes = [], [], dict(checked=0, equal=0)
    for job in sorted((FISHER_ROOT / 'noise_out').glob('n*')):
        folder = job / 'B12'
        case = tomllib.loads((folder / 'case.toml').read_text())
        clean = MBP.dl(folder / 'masked_clean_cl.npy')
        for s1, s2 in case['seeds']:
            label = f'B12_{s1}_{s2}'
            noisy = MBP.dl(folder / f'noisy_cross_{s1}_{s2}.npy')
            nn = MBP.dl(FISHER_ROOT / 'nn_regen' / 'nn' / f'{label}.npy')
            x, y = nn[high], (noisy - clean)[high]
            slopes.append(float(x @ y / (x @ x)))
            regen = tomllib.loads((FISHER_ROOT / 'nn_regen' / 'nn' / f'{label}.toml').read_text())
            if regen.get('noise_sha256'):
                hashes['checked'] += 1
                hashes['equal'] += int(list(regen['noise_sha256']) == list(case['noise_sha256'][f'{s1}_{s2}']))
            residuals.append(MBP.correct(clean, noisy, nn, bl) - clean)
    slopes = np.array(slopes)
    checks.update(new_draws=len(residuals), nn_slope_min=float(slopes.min()), nn_slope_max=float(slopes.max()),
                  nn_slope_median=float(np.median(slopes)), noise_map_hashes=hashes)
    assert len(residuals) == 400 and 0.85 < slopes.min() and slopes.max() < 1.15, checks
    assert hashes['checked'] > 0 and hashes['equal'] == hashes['checked'], checks
    return pairs, beam21, clean_b12, np.stack(residuals), checks


def main():
    t0 = time.time()
    bl = np.load(MBP.BEAMFIX / 'beam_bl.npy')[MBP.ELL]
    pairs, beam21, clean_b12, residuals400, checks = b12_draws(bl)
    print('B12 draws:', checks, flush=True)
    k_obs = pairs.index(OBS_PAIR)
    r_fit = np.concatenate([np.delete(beam21, k_obs, axis=0) - clean_b12, residuals400])    # 420 realizations
    C, s, rho = covariance_model(r_fit, LAGS, HALF)
    C[np.diag_indices_from(C)] += FLOOR * s ** 2
    R = C / np.outer(s, s)
    eig_min = float(np.linalg.eigvalsh(R).min())
    del R
    clean, beam, row_checks = gather_rows(bl)
    L, jitter = MBP.cholesky(C)
    order = np.load(HERE / 'results_final' / 'split.npz')['order']
    V, top, kept, mean_white = MBP.pca(L, clean, order)
    chunk = MBP.CHUNK
    s_clean = np.concatenate([MBP.white(L, clean[i:i + chunk]) @ V for i in range(0, len(clean), chunk)])
    s_noisy = np.concatenate([MBP.white(L, beam[i:i + chunk]) @ V for i in range(0, len(beam), chunk)])
    s_draws = MBP.white(L, beam21) @ V
    s_obs = s_draws[k_obs:k_obs + 1]
    INPUTS.mkdir(parents=True, exist_ok=True)
    np.save(INPUTS / 's_noisy.npy', s_noisy)
    np.save(INPUTS / 's_clean.npy', s_clean)
    np.savez(INPUTS / 'whitening_pca.npz', L=L, V=V, singular_values_top=top, kept=kept, train_mean_white=mean_white)
    np.savez(INPUTS / 'b12.npz', s_obs=s_obs, s_draws=s_draws, pairs=np.array(pairs))
    cov_info = dict(draws=int(len(r_fit)), estimator=f'fisher_forecast_b12.covariance_model (Welch Toeplitz, lags '
                    f'{LAGS}, variance running mean +-{HALF}) of the beam-consistent B12 residuals + white floor '
                    f'{FLOOR} x variance', rho_1_to_5=rho[1:6].round(5).tolist(), min_eigenvalue_R=eig_min,
                    jitter=jitter, diag_sd_range=[float(np.sqrt(C.diagonal().min())), float(np.sqrt(C.diagonal().max()))])
    pca_info = dict(q=MBP.Q, eps_v_percent=MBP.EPS_PERCENT, n_pca=int(kept.sum()), train_rows=int(len(order)))
    info = dict(source='beam-consistent spectra, all 7,900 multipoles, 420-draw B12 whitening (merlin_w420_prepare.py)',
                covariance=cov_info, pca=pca_info, n_pca=int(kept.sum()), data_checks=dict(row_checks, **checks))
    import merlin_compression as MC
    MC.write_features(FEATURES, s_noisy, s_obs, s_draws, info, s_clean, 'asinh')
    report = dict(info, seconds=time.time() - t0)
    (INPUTS / 'prepare_w420.json').write_text(json.dumps(report, indent=1) + '\n')
    print(json.dumps(report, indent=1), flush=True)


if __name__ == '__main__':
    main()
