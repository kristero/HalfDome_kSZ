"""Is the Battaglia+12 observation processed exactly like the training rows?

1. settings: observation.toml of the fitted observation vs a dataset row (beam, NSIDE, lmax, mask, noise convention)
2. engine: the local engine reproduces the fitted observation (clean and SO mock) from its parameters and seeds
3. features: the fitted observation among 20 new SO noise draws of the same B12 sky, every one taken through the
   same D_ell, binning, asinh and compression steps as the training rows (MOPED, 40 bins, PCA-9); the features stored
   with the trained networks are recomputed from the reproduced spectrum and must be identical
4. neighbours: a weighted local-linear regression of the training features on theta around B12 predicts the
   noise-averaged features of a B12 sky; it must agree with the 21 noise draws within their scatter
Writes results_logit/obs_processing_audit.json and fig_obs_processing_audit.png.
"""
import argparse
import json
from pathlib import Path
import tomllib

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import analyze_64k as A
from train_ensemble import project_binned, A_rebin, ELL, DL

HERE = Path(__file__).resolve().parent
OBS_SEEDS = (39001001, 39001002)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results', type=Path, default=HERE / 'results_logit')
    p.add_argument('--spectra', type=Path, default=HERE / 'mocks' / 'pass1_B12' / 'B12')
    p.add_argument('--neighbours', type=int, default=400)
    args = p.parse_args()
    report = {}
    # 1. settings
    anchor = tomllib.loads((A.TEST256 / 'cluster_snapshot' / 'tests' / 'anchors' / 'Battaglia12' / 'output4096' / 'observation.toml').read_text())
    row = tomllib.loads((HERE.parent.parent / 'tSZ_64k_flat_prior_SO_baseline_deproj0' / 'reference' / 'test256' / 'batch000' / '00000' / 'observation.toml').read_text())
    keys = ['signal_beam_fwhm_arcmin', 'output_nside', 'ell_max', 'mask_sha256', 'noise_beam_applied', 'split_Nell_multiplier']
    report['settings'] = {k: dict(observation=anchor[k], dataset_row=row[k], same=anchor[k] == row[k]) for k in keys}
    # 2. engine reproduction
    obs = np.load(A.TEST256 / 'cluster_snapshot' / 'diagnostic_256' / 'observations' / 'Battaglia12.npz')
    clean = np.load(args.spectra / 'masked_clean_cl.npy')[ELL] * DL
    files = sorted(args.spectra.glob('noisy_cross_*.npy'))
    fitted = args.spectra / f'noisy_cross_{OBS_SEEDS[0]}_{OBS_SEEDS[1]}.npy'
    new = [f for f in files if f != fitted]
    dl_fit = np.load(fitted)[ELL] * DL
    dl_new = np.stack([np.load(f)[ELL] * DL for f in new])
    report['engine'] = dict(clean_max_rel_diff=float(np.max(np.abs(clean - obs['clean_dl_unbinned']) / np.abs(obs['clean_dl_unbinned']))),
                            so_mock_max_diff_over_signal_scale=float(np.max(np.abs(dl_fit - obs['noisy_dl_unbinned']) /
                                                                            (np.abs(obs['noisy_dl_unbinned']) + np.abs(obs['clean_dl_unbinned'])))),
                            new_noise_draws=len(new))
    # 3-4. features
    figure_rows = []
    for folder in ('noisy', 'noisy_bins40', 'noisy_pca9'):
        d = args.results / folder
        if not (d / 'features.npz').exists():
            continue
        f = np.load(d / 'features.npz'); state = dict(np.load(d / 'transform.npz'))
        if folder == 'noisy':
            state['method'] = 'moped_fixed'
            proj = lambda dl: A.project(state, np.atleast_2d(dl))                            # noqa: E731
        else:
            proj = lambda dl: project_binned(state, A_rebin(np.atleast_2d(dl)))              # noqa: E731
        x_fit = proj(dl_fit)[0]; x_new = proj(dl_new); x_stored = f['x_obs'][0]
        mu, sd = x_new.mean(0), x_new.std(0, ddof=1)
        z_fit = (x_fit - mu) / sd
        z_new = (x_new - mu) / sd
        # leave-one-out z of each new draw for a fair reference distribution of chi2
        chi_new = []
        for k in range(len(x_new)):
            rest = np.delete(x_new, k, axis=0)
            chi_new.append(float(np.sum(((x_new[k] - rest.mean(0)) / rest.std(0, ddof=1)) ** 2)))
        # neighbour prediction from the training rows
        low, high = f['low'], f['high']; t = (f['theta_pool'] - low) / (high - low); t0 = (A.B12 - low) / (high - low)
        dist = np.linalg.norm(t - t0, axis=1); idx = np.argsort(dist)[:args.neighbours]
        h = np.median(dist[idx]); w = np.exp(-0.5 * (dist[idx] / h) ** 2)
        design = np.c_[np.ones(len(idx)), t[idx] - t0]
        W = np.sqrt(w)[:, None]
        coef, *_ = np.linalg.lstsq(design * W, f['x_pool'][idx] * W, rcond=None)
        pred = coef[0]
        resid = f['x_pool'][idx] - design @ coef
        pred_se = np.sqrt(np.sum(w[:, None] * resid ** 2, axis=0) / np.sum(w)) / np.sqrt(np.sum(w) ** 2 / np.sum(w ** 2))
        report[folder] = dict(stored_vs_recomputed_max_abs=float(np.max(np.abs(x_stored - x_fit))),
                              fitted_z_vs_new_draws=z_fit.tolist(), fitted_chi2=float(np.sum(z_fit ** 2)), dof=int(len(z_fit)),
                              new_draws_leave_one_out_chi2=dict(median=float(np.median(chi_new)), p16=float(np.percentile(chi_new, 16)),
                                                                p84=float(np.percentile(chi_new, 84))),
                              neighbour_prediction_minus_noise_mean_over_sd=((pred - mu) / sd).tolist(),
                              neighbour_prediction_se_over_sd=(pred_se / sd).tolist(),
                              neighbours=int(len(idx)), neighbour_median_distance_prior_units=float(h),
                              nearest_neighbour_distance=float(dist[idx[0]]))
        figure_rows.append((folder, x_new, x_fit, pred, pred_se, mu, sd))
    (args.results / 'obs_processing_audit.json').write_text(json.dumps(report, indent=1) + '\n')

    # figure: MOPED features (9) in one row block, PCA-9 in another; 40 bins summarized as z-scores
    plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'cm', 'font.size': 8})
    n = len(figure_rows)
    fig, axes = plt.subplots(n, 1, figsize=(17 / 2.54, 5.2 / 2.54 * n))
    axes = np.atleast_1d(axes)
    for ax, (folder, x_new, x_fit, pred, pred_se, mu, sd) in zip(axes, figure_rows):
        k = np.arange(len(mu))
        z_new = (x_new - mu) / sd
        for j in range(len(z_new)):
            ax.plot(k, z_new[j], 'o', ms=2.2, color='#2a78d6', alpha=0.45, mec='none', label='20 new noise draws of the B12 sky' if j == 0 else None)
        ax.plot(k, (x_fit - mu) / sd, 's', ms=4.5, color='#0b0b0b', label='fitted observation')
        ax.errorbar(k + 0.18, (pred - mu) / sd, yerr=pred_se / sd, fmt='D', ms=3.5, color='#eb6834', elinewidth=0.9,
                    label='training rows near B12 (local-linear prediction)')
        ax.axhspan(-1, 1, color='0.92', zorder=0, lw=0); ax.axhline(0, color='#52514e', lw=0.6)
        ax.set_yscale('symlog', linthresh=3)
        ax.set_xlim(-0.6, len(mu) - 0.4); ax.set_ylabel('(feature - mean) / sd')
        ax.set_title({'noisy': 'MOPED (9)', 'noisy_bins40': '40 band powers', 'noisy_pca9': 'PCA-9 of the 40 bands'}[folder] +
                     f": fitted observation $\\chi^2$ = {report[folder]['fitted_chi2']:.1f} for {len(mu)} features "
                     f"(new draws: median {report[folder]['new_draws_leave_one_out_chi2']['median']:.1f})", fontsize=8)
        ax.grid(True, color='#d9d8d4', lw=0.5); ax.set_axisbelow(True)
    axes[0].legend(frameon=False, fontsize=6.5, ncol=3, loc='lower left')
    axes[-1].set_xlabel('feature index')
    fig.suptitle('Battaglia+12 observation vs training data, feature by feature\n(same D$_\\ell$, binning, asinh and compression; symmetric-log axis, linear within $\\pm3\\sigma$)', fontsize=9)
    fig.tight_layout(); fig.savefig(args.results / 'fig_obs_processing_audit.png', dpi=200); plt.close(fig)
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
