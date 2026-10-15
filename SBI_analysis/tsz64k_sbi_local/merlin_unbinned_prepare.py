"""Full-resolution inputs for the Merlin compression (merlin_compression.py --source unbinned).

The spectra are D_ell = ell(ell+1) C_ell / 2 pi at all 7,900 multipoles ell = 80..7979. From the collaborator's
complete dataset/rows_65536 (fetched with Globus, one 4.1 GB file at a time through a staging folder) this script
builds everything the unbinned compression needs and keeps only the small projected products:

  1. the noise covariance C of the unbinned spectra and its Cholesky factor L (C = L L^T), with the estimator of
     the 40-band Fisher covariance (so_nine_fisher.conditional_covariance: OAS shrinkage of the correlation
     matrix, the empirical variances kept) applied to the same 20 Battaglia12 SO noise draws, at every multipole;
  2. the whitening w = L^-1 x, under which the noise has unit covariance;
  3. a PCA of the whitened noise-free training spectra, as merlin/preprocessing.py does it: the top q = 500
     singular values S of the centred whitened clean training matrix (torch.pca_lowrank(q=500, center=True)),
     keeping the components with S / S.sum() * 100 > 0.001;
  4. the projection s = (L^-1 x) V of every spectrum onto the kept components V.

Each pass also checks the 4.1 GB file: the 64,440 analysed rows against data/rows_64440 (the 40 bands and the raw
MOPED projections), all rows against the collaborator's own 40 bands (data/rows_65536), and the 1,096 rows they
regenerated against the rows made on idark (data/rows_validation_1096). The first two must agree to 1e-9
(asserted); the regenerated-row differences are only reported.

Reads (paths relative to this folder)
  --file                                 (65536, 7900) float64 D_ell, clean or noisy, rows in rows_65536 order
  data/rows_65536/                       row_id, ell_unbinned, clean_dl and noisy_dl (40 bands, collaborator)
  data/rows_64440/                       row_id, clean_dl, noisy_dl, moped_raw_clean, moped_raw_noisy
  data/rows_validation_1096/             row_id and the same four arrays for the rows regenerated on idark
  results_final/split.npz                order (training pool) and test, as positions in rows_64440
  mocks/pass1_B12/B12/noisy_cross_*.npy  the 21 Battaglia12 noise draws, C_ell at ell = 0..7979
  ../tsz_diagnostic_256_accelerated_20260922/reference/compression_context.npz        the raw MOPED matrix
  ../tsz_diagnostic_256_accelerated_20260922/cluster_snapshot/diagnostic_256/observations/Battaglia12.npz
                                         the B12 observation (noisy pass)

Writes, to results_merlin_unbinned/inputs/
  clean: s_clean.npy (65536, n_pca); whitening_pca.npz (L, V, singular_values_top, kept, train_mean_white);
         clean.json (covariance, PCA and check summaries)
  noisy: s_noisy.npy (65536, n_pca); b12.npz (s_obs (1, n_pca) of the B12 observation, s_draws (21, n_pca) of
         its noise draws); noisy.json (check summaries)

Run clean first (noisy reads whitening_pca.npz):
  merlin_unbinned_prepare.py clean --file <staging>/clean_dl_unbinned.npy     covariance, PCA, clean projections
  merlin_unbinned_prepare.py noisy --file <staging>/noisy_dl_unbinned.npy     noisy projections, B12 observation
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.linalg import solve_triangular

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))                     # SBI_analysis/ on the import path
OUT = HERE / 'results_merlin_unbinned' / 'inputs'        # everything this script writes
SMALL = HERE / 'data' / 'rows_65536'                     # the collaborator's small per-row files of rows_65536
ROWS_64440 = HERE / 'data' / 'rows_64440'                # the 64,440 analysed rows (our own reduction)
VALIDATION = HERE / 'data' / 'rows_validation_1096'      # the 1,096 rows regenerated on idark
B12_MOCKS = HERE / 'mocks' / 'pass1_B12' / 'B12'         # Battaglia12 noise draws of the SO baseline
B12_FITTED = B12_MOCKS / 'noisy_cross_39001001_39001002.npy'   # the draw left out of the covariance estimate
MOPED_CONTEXT = HERE.parent / 'tsz_diagnostic_256_accelerated_20260922' / 'reference' / 'compression_context.npz'
MOPED_KEY = 'Battaglia12_rcond1e-06'                     # the (7900, 9) raw MOPED matrix inside MOPED_CONTEXT
SOURCE = 'collaborator Globus dataset/rows_65536 (tsz_64k_flat_prior_SO_baseline_deproj0_20260926)'
N_ROWS = 65536                                           # rows of the collaborator's unbinned files
N_ELL = 7900                                             # multipoles 80..7979 per row
CHUNK = 2048                                             # rows read from the 4.1 GB file per step (~130 MB)
PROGRESS_EVERY = 8 * CHUNK                               # rows between progress lines
Q = 500                                                  # PCA rank of torch.pca_lowrank in merlin
EPS_PERCENT = 0.001                                      # keep components with S / S.sum() * 100 above this
TOLERANCE = 1e-9                                         # largest relative difference accepted by the checks
BAND_EDGES = np.r_[np.arange(80, 7881, 200), 7980]       # the 40 bands: 39 of width 200, the last 7880..7979


def rebin40(values, ell):
    """Average spectra into the 40 analysis bands, as the rows_64440 and rows_65536 40-band files were made.

    values: (n, n_ell) D_ell at the multipoles ell; ell: (n_ell,) integer multipoles.
    Returns (n, 40): per band, the mean over the multipoles in [low, high) weighted by 2 ell + 1, the number of
    m modes of each multipole."""
    bands = []
    for low, high in zip(BAND_EDGES[:-1], BAND_EDGES[1:]):
        in_band = (ell >= low) & (ell < high)
        bands.append(np.average(values[:, in_band], axis=1, weights=2 * ell[in_band] + 1))
    return np.stack(bands, axis=1)


def rows():
    """Row ids, in rows_65536 positions (row id = position, asserted), of the parts of the analysis.

    Returns (ids64, pool, test, regen): ids64 (64440,) the analysed rows in rows_64440 order; pool (61440,) the
    training pool in the order of results_final/split.npz; test (3000,) the test rows; regen (1096,) the rows
    regenerated on idark, in rows_validation_1096 order."""
    ids64 = np.load(ROWS_64440 / 'row_id.npy')
    split = np.load(HERE / 'results_final' / 'split.npz')
    row_id = np.load(SMALL / 'row_id.npy')
    assert np.array_equal(row_id, np.arange(len(row_id))), 'rows_65536 is not in row order'
    regen = np.load(VALIDATION / 'row_id.npy')
    return ids64, ids64[split['order']], ids64[split['test']], regen


def _b12_draws(paths):
    """The Battaglia12 noise draws at paths as unbinned D_ell: (len(paths), 7900), ell = train_ensemble.ELL."""
    from train_ensemble import ELL, DL
    return np.stack([np.load(p)[ELL] * DL for p in paths])


def whitener():
    """Noise covariance of the unbinned spectra and its Cholesky factor.

    so_nine_fisher.conditional_covariance, computed through sklearn's _oas: sklearn.covariance.oas is the same
    estimate but also forms the pseudo-inverse of the 7,900 x 7,900 matrix, which takes hours and is not needed.
    Uses the 20 draws of mocks/pass1_B12/B12 other than B12_FITTED.
    Returns (L, info): L (7900, 7900) lower triangular with L L^T = C, in D_ell units; info the dict stored as
    'covariance' in clean.json (draws, oas_correlation_shrinkage, diag_sd_range = [min, max] of sqrt(diag C))."""
    from sklearn.covariance._shrunk_covariance import _oas
    draws = _b12_draws([p for p in sorted(B12_MOCKS.glob('noisy_cross_*.npy')) if p != B12_FITTED])
    std = draws.std(axis=0, ddof=1)
    assert np.all(std > 0), 'zero ensemble variance'
    # shrink the correlation of the standardised draws, not the covariance: the variances are kept as measured
    standardised = (draws - draws.mean(axis=0)) / std
    correlation, shrinkage = _oas(standardised, assume_centered=True)
    # OAS normalises by n, the standardisation by n - 1: rescale to a unit diagonal
    correlation /= np.sqrt(np.outer(np.diag(correlation), np.diag(correlation)))
    C = correlation * np.outer(std, std)
    L = np.linalg.cholesky(C)
    diag_sd = np.sqrt(C.diagonal())
    info = dict(draws=len(draws), oas_correlation_shrinkage=shrinkage,
                diag_sd_range=[float(diag_sd.min()), float(diag_sd.max())])
    return L, info


def white(L, x):
    """Whitened spectra L^-1 x (noise covariance 1): x (n, 7900) in D_ell units -> (n, 7900) float64."""
    return solve_triangular(L, np.asarray(x, dtype=np.float64).T, lower=True, check_finite=False).T


def _positions_in_64440(ids64, n_rows):
    """(n_rows,) position of each rows_65536 row in rows_64440, -1 for rows outside the analysis."""
    positions = np.full(n_rows, -1)
    positions[ids64] = np.arange(len(ids64))
    return positions


def _moped_matrix():
    """The raw MOPED compression matrix of the 40-band analysis, (7900, 9), applied to unbinned D_ell."""
    return np.load(MOPED_CONTEXT)[MOPED_KEY]


def _max_difference(values, reference, scale):
    """Largest difference between values and reference (same shape), relative to a scale of the reference.

    scale 'element': |reference| entry by entry (clean spectra, all positive); 'row': the largest |reference| of
    each row (noisy spectra cross zero); 'global': max |values - reference| / max |reference| (MOPED projections).
    Returns a numpy float."""
    if scale == 'element':
        return np.max(np.abs(values - reference) / np.abs(reference))
    if scale == 'row':
        return np.max(np.abs(values - reference) / np.abs(reference).max(axis=1, keepdims=True))
    return np.max(np.abs(values - reference)) / np.max(np.abs(reference))


def _assert_checks(diffs, kind):
    """Print the largest difference of each check in diffs {name: [per-chunk values]} and require < TOLERANCE."""
    for name, chunk_values in diffs.items():
        print(f'  check {kind} {name}: max relative difference {max(chunk_values):.2e}', flush=True)
        assert max(chunk_values) < TOLERANCE, name


def cmd_clean(args):
    """Covariance, whitening, PCA and clean projections from args.file, the (65536, 7900) clean D_ell.

    Pass 1 checks every chunk and accumulates the moments of the whitened training-pool spectra; the PCA follows
    from their covariance; pass 2 projects every row. Writes s_clean.npy, whitening_pca.npz and clean.json."""
    from train_ensemble import ELL
    OUT.mkdir(parents=True, exist_ok=True)
    spectra = np.load(args.file, mmap_mode='r')
    assert spectra.shape == (N_ROWS, N_ELL), spectra.shape
    ell = np.load(SMALL / 'ell_unbinned.npy')
    assert np.array_equal(ell, ELL)
    ids64, pool, _test, regen = rows()
    t0 = time.monotonic()
    L, cov_info = whitener()
    print(f'covariance and Cholesky factor: {time.monotonic() - t0:.0f} s, '
          f'OAS shrinkage {cov_info["oas_correlation_shrinkage"]:.4f}', flush=True)
    in_pool = np.zeros(len(spectra), bool)
    in_pool[pool] = True
    # one-pass moments of the whitened pool spectra: the 4.1 GB file does not fit in memory
    gram = np.zeros((N_ELL, N_ELL))
    white_sum = np.zeros(N_ELL)
    n_train = 0
    matrix = _moped_matrix()
    ours = {k: np.load(ROWS_64440 / f'{k}.npy') for k in ('clean_dl', 'moped_raw_clean')}
    pos64 = _positions_in_64440(ids64, len(spectra))
    theirs40 = np.load(SMALL / 'clean_dl.npy')
    diffs = {}
    for start in range(0, len(spectra), CHUNK):
        x = np.asarray(spectra[start:start + CHUNK], dtype=np.float64)
        chunk_rows = np.arange(start, start + len(x))
        bands40 = rebin40(x, ell)
        diffs.setdefault('their_40', []).append(_max_difference(bands40, theirs40[chunk_rows], 'element'))
        analysed = pos64[chunk_rows] >= 0
        if analysed.any():
            at64 = pos64[chunk_rows[analysed]]
            diffs.setdefault('ours_40', []).append(
                _max_difference(bands40[analysed], ours['clean_dl'][at64], 'element'))
            diffs.setdefault('ours_moped', []).append(
                _max_difference(x[analysed] @ matrix, ours['moped_raw_clean'][at64], 'global'))
        w = white(L, x[in_pool[chunk_rows]])
        gram += w.T @ w
        white_sum += w.sum(0)
        n_train += len(w)
        if start % PROGRESS_EVERY == 0:
            print(f'  pass 1: {start + len(x)}/{len(spectra)} rows, {time.monotonic() - t0:.0f} s', flush=True)
    _assert_checks(diffs, 'clean')
    # covariance of the whitened pool spectra from the accumulated moments: E[w w^T] - E[w] E[w]^T
    train_mean_white = white_sum / n_train
    cov = gram / n_train - np.outer(train_mean_white, train_mean_white)
    # the eigenvectors of the covariance are the right singular vectors of the centred training matrix, whose
    # singular values are sqrt(n_train * eigenvalue); round-off negatives are set to zero
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    descending = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[descending], 0)
    eigenvectors = eigenvectors[:, descending]
    singular_values = np.sqrt(eigenvalues * n_train)
    top = singular_values[:Q]
    # merlin's cut: a component's share of the summed top-q singular values (not of the variance)
    kept = top / top.sum() * 100 > EPS_PERCENT
    V = eigenvectors[:, :Q][:, kept]
    print(f'PCA: {kept.sum()} of the top {Q} components kept; '
          f'first fractions (%) {np.round(100 * top[:6] / top.sum(), 3).tolist()}', flush=True)
    s_clean = np.empty((len(spectra), V.shape[1]))
    for start in range(0, len(spectra), CHUNK):
        s_clean[start:start + CHUNK] = white(L, spectra[start:start + CHUNK]) @ V
    # the regenerated rows: the collaborator's against the idark ones
    idark = {k: np.load(VALIDATION / f'{k}.npy') for k in ('clean_dl', 'moped_raw_clean')}
    regenerated = np.asarray(spectra[regen], dtype=np.float64)
    regen_diff = dict(clean_40=float(_max_difference(rebin40(regenerated, ell), idark['clean_dl'], 'element')),
                      clean_moped=float(_max_difference(regenerated @ matrix, idark['moped_raw_clean'], 'global')))
    print(f'  regenerated rows, collaborator vs idark: {regen_diff}', flush=True)
    np.save(OUT / 's_clean.npy', s_clean)
    np.savez(OUT / 'whitening_pca.npz', L=L, V=V, singular_values_top=top, kept=kept,
             train_mean_white=train_mean_white)
    pca_info = dict(q=Q, eps_v_percent=EPS_PERCENT, n_pca=int(kept.sum()),
                    singular_value_percent_top=(100 * top / top.sum()).tolist(), train_rows=int(n_train),
                    variance_kept=float(np.sum(eigenvalues[:Q][kept]) / np.sum(eigenvalues)))
    info = dict(covariance=cov_info, pca=pca_info, checks={k: float(max(v)) for k, v in diffs.items()},
                regenerated_rows_collaborator_vs_idark=regen_diff, source=SOURCE, seconds=time.monotonic() - t0)
    (OUT / 'clean.json').write_text(json.dumps(info, indent=1) + '\n')
    print(f'clean pass done in {time.monotonic() - t0:.0f} s -> {OUT}', flush=True)


def cmd_noisy(args):
    """Noisy projections from args.file, the (65536, 7900) noisy D_ell, with the whitening and PCA of cmd_clean.

    Also projects the B12 observation and all 21 of its noise draws. Writes s_noisy.npy, b12.npz and noisy.json."""
    import analyze_64k as A
    spectra = np.load(args.file, mmap_mode='r')
    assert spectra.shape == (N_ROWS, N_ELL), spectra.shape
    ell = np.load(SMALL / 'ell_unbinned.npy')
    whitening_pca = np.load(OUT / 'whitening_pca.npz')
    L = whitening_pca['L']
    V = whitening_pca['V']
    ids64, pool, _test, regen = rows()
    matrix = _moped_matrix()
    ours = {k: np.load(ROWS_64440 / f'{k}.npy') for k in ('noisy_dl', 'moped_raw_noisy')}
    pos64 = _positions_in_64440(ids64, len(spectra))
    theirs40 = np.load(SMALL / 'noisy_dl.npy')
    diffs = {}
    t0 = time.monotonic()
    s_noisy = np.empty((len(spectra), V.shape[1]))
    for start in range(0, len(spectra), CHUNK):
        x = np.asarray(spectra[start:start + CHUNK], dtype=np.float64)
        chunk_rows = np.arange(start, start + len(x))
        bands40 = rebin40(x, ell)
        # noisy bands cross zero: differences relative to each row's largest |D_ell|
        diffs.setdefault('their_40', []).append(_max_difference(bands40, theirs40[chunk_rows], 'row'))
        analysed = pos64[chunk_rows] >= 0
        if analysed.any():
            at64 = pos64[chunk_rows[analysed]]
            diffs.setdefault('ours_40', []).append(
                _max_difference(bands40[analysed], ours['noisy_dl'][at64], 'row'))
            diffs.setdefault('ours_moped', []).append(
                _max_difference(x[analysed] @ matrix, ours['moped_raw_noisy'][at64], 'global'))
        s_noisy[start:start + len(x)] = white(L, x) @ V
        if start % PROGRESS_EVERY == 0:
            print(f'  noisy: {start + len(x)}/{len(spectra)} rows, {time.monotonic() - t0:.0f} s', flush=True)
    _assert_checks(diffs, 'noisy')
    idark = {k: np.load(VALIDATION / f'{k}.npy') for k in ('noisy_dl', 'moped_raw_noisy')}
    regenerated = np.asarray(spectra[regen], dtype=np.float64)
    regen_diff = dict(noisy_40=float(_max_difference(rebin40(regenerated, ell), idark['noisy_dl'], 'row')),
                      noisy_moped=float(_max_difference(regenerated @ matrix, idark['moped_raw_noisy'], 'global')))
    print(f'  regenerated rows, collaborator vs idark: {regen_diff}', flush=True)
    obs = np.load(A.TEST256 / 'cluster_snapshot' / 'diagnostic_256' / 'observations' / 'Battaglia12.npz')
    b12_draws = _b12_draws(sorted(B12_MOCKS.glob('noisy_cross_*.npy')))
    np.save(OUT / 's_noisy.npy', s_noisy)
    np.savez(OUT / 'b12.npz', s_obs=white(L, obs['noisy_dl_unbinned'][None]) @ V, s_draws=white(L, b12_draws) @ V)
    info = dict(checks={k: float(max(v)) for k, v in diffs.items()},
                regenerated_rows_collaborator_vs_idark=regen_diff, b12_draws=len(b12_draws),
                seconds=time.monotonic() - t0)
    (OUT / 'noisy.json').write_text(json.dumps(info, indent=1) + '\n')
    print(f'noisy pass done in {time.monotonic() - t0:.0f} s -> {OUT}', flush=True)


def main():
    """Command line: the clean or noisy pass on one unbinned file."""
    parser = argparse.ArgumentParser(description='Full-resolution inputs of merlin_compression.py '
                                                 '--source unbinned (run clean, then noisy).')
    sub = parser.add_subparsers(dest='cmd', required=True)
    for name in ('clean', 'noisy'):
        command = sub.add_parser(name)
        command.add_argument('--file', type=Path, required=True, help=f'the (65536, 7900) {name}_dl_unbinned.npy')
    args = parser.parse_args()
    commands = dict(clean=cmd_clean, noisy=cmd_noisy)
    commands[args.cmd](args)


if __name__ == '__main__':
    main()
