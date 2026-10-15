"""Small arrays for the pipeline explainer figures, from the cluster-only Merlin inputs (runs on idark).

For both beam-consistent input sets (beam_b12: B12 whitening, beam_nn: noise-only whitening) it saves:
  sd               (7900,)       the whitening covariance's standard deviation per multipole (diag of L L^T)
  corr_block       (400, 400)    its correlation matrix for ell 2000..2399 (neighbouring-multipole structure)
  corr_lags        (41,)         the mean correlation at multipole lags 0..40 over ell 80..7979
  singular_values  (500,), kept  the PCA spectrum of the whitened noise-free pool and the components kept
  V6               (7900, 6)     the first six PCA components (as functions of ell)
  white_clean_b12, white_noisy_b12 (7900,)  the whitened B12 spectrum, noise-free and observed (beam-consistent)
  s_noisy, s_clean (4000, 6)     PCA projections of the first 4,000 training-pool rows, first six components
  z_noisy          (4000, 6)     the same after the asinh standardization (network inputs)
  s_center, s_scale, z_mean, z_sd (6,)  the asinh parameters of those components
  theta            (4000, 9)     the parameters of those rows
Writes doc_figure_data/<set>.npz (a few MB each).

    python doc_figure_data.py
"""
from pathlib import Path

import numpy as np

import merlin_beam_prepare as P
import merlin_compression as MC
import analyze_64k as A

SETS = dict(beam_b12=('results_merlin_unbinned_beam_asinh_p30', 'results_merlin_unbinned_beam/inputs'),
            beam_nn=('results_merlin_unbinned_beam_nnwhite_asinh_p30', 'results_merlin_unbinned_beam_nnwhite/inputs'))
ROWS, COMPONENTS, LAGS = 4000, 6, 40
OUT = P.HERE / 'doc_figure_data'


def main():
    OUT.mkdir(exist_ok=True)
    obs = np.load(A.TEST256 / 'cluster_snapshot' / 'diagnostic_256' / 'observations' / 'Battaglia12.npz')
    bl = np.load(P.BEAMFIX / 'beam_bl.npy')[P.ELL]
    nn_b12 = P.dl(P.B12DRAWS / 'nn' / f'B12_{P.OBS_PAIR}.npy')
    b12_beam = P.correct(obs['clean_dl_unbinned'], obs['noisy_dl_unbinned'], nn_b12, bl)
    for name, (folder, inputs) in SETS.items():
        wp = np.load(P.HERE / inputs / 'whitening_pca.npz')
        L, V = wp['L'], wp['V']
        cov_diag = np.einsum('ij,ij->i', L, L)                       # diag(L L^T)
        sd = np.sqrt(cov_diag)
        block = slice(2000 - 80, 2400 - 80)
        Lb = L[block]
        cov_block = Lb @ Lb.T
        corr_block = cov_block / np.outer(sd[block], sd[block])
        # mean correlation at lag k: sum_i (L L^T)_{i,i+k} / (sd_i sd_{i+k}), from rows i and i+k of L
        corr_lags = np.array([np.mean(np.einsum('ij,ij->i', L[:len(L) - k], L[k:]) / (sd[:len(L) - k] * sd[k:]))
                              for k in range(LAGS + 1)])
        f = np.load(P.HERE / folder / 'features_merlin.npz')
        pool = f['pool'][:ROWS]
        s_noisy = np.load(P.HERE / inputs / 's_noisy.npy', mmap_mode='r')
        s_clean = np.load(P.HERE / inputs / 's_clean.npy', mmap_mode='r')
        sn, sc = np.asarray(s_noisy[pool, :COMPONENTS]), np.asarray(s_clean[pool, :COMPONENTS])
        params = {k: f[k][:COMPONENTS] for k in ('s_center', 's_scale', 'z_mean', 'z_sd')}
        z = MC.asinh_standardize(sn, params)
        # the stored network inputs of these rows must be what the transform gives
        check = float(np.max(np.abs(z - f['x_pool'][:ROWS, :COMPONENTS])))
        assert check < 1e-4, check
        np.savez(OUT / f'{name}.npz', sd=sd, corr_block=corr_block.astype(np.float32), corr_lags=corr_lags,
                 singular_values=wp['singular_values_top'], kept=wp['kept'], V6=V[:, :COMPONENTS],
                 white_clean_b12=P.white(L, obs['clean_dl_unbinned'][None])[0],
                 white_noisy_b12=P.white(L, b12_beam[None])[0], s_noisy=sn, s_clean=sc, z_noisy=z,
                 theta=f['theta_pool'][:ROWS], ell=P.ELL, check_z=check, **params)
        print(f'{name}: n_pca {V.shape[1]}, lag-1 corr {corr_lags[1]:.3f}, check {check:.1e}', flush=True)
        del L, V, wp


if __name__ == '__main__':
    main()
