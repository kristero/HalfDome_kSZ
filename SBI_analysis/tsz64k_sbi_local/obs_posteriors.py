"""Posteriors of out-of-training observations for every compression (runs on idark).

Each observation is a masked SO split cross spectrum made with the training rows' post-processing (2 arcmin beam,
NSIDE 4096, f_sky 0.4 cap, SO baseline Deproj-0 noise, lmax 7979; the B12 observation's noise seeds 39001001/39001002):
  B12              Battaglia12.npz of the 256-row test (the usual observation; truth = Battaglia+12)
  L1_m9, Mstar-1sigma, fgas-8sigma   the FLAMINGO hydro tSZ maps (FLAMINGO cosmology; flamingo_tsz) through the
                   same beam, mask and SO noise, as staged for the 256-row test (out of family, no truth)
  Lee22_corrected, Lee22_printed     paint_mocks.jl (the frozen engine) at the Lee22 parameter points of
                   mocks/idark_lee22/lee22_theta.json (truth = those points)
  Lee22_full       the full corrected Lee22 model map (frb_map_generation lee22p_m1) through the operator of the
                   FLAMINGO observations (mocks/idark_lee22_full; out of family, no truth)
For each model the input is built from the observation's spectrum with exactly that model's transform:
  40 bins, MOPED   sbi_final.variant_features of results_final_noise (noise-level asinh scale)
  Merlin, PCA      L^-1 x V (whitening and PCA of merlin_unbinned_prepare.py), then the stored asinh standardization
  raw              the asinh standardization of raw_features.py on all 7,900 multipoles
and the B12 inputs are checked against the stored x_obs of each model. 20,000 posterior samples per (observation,
model), drawn as the evaluations do (torch.manual_seed(2), default_rng(2)).

    python obs_posteriors.py [--raw-N 32000] [--threads 8] [--models Merlin PCA ..] [--merlin-folder DIR] [--out DIR]
Writes <out>/<observation>__<model>.npy (20000, 9) float32 and <out>/summary.json (per pair: median, 16 / 84 %
quantiles; the input checks; the truths). --merlin-folder: the Merlin / PCA networks (default the 61,440-row
asinh run); with MERLIN_PARAM_SPACE=linear (networks trained without the prior box) only Merlin and PCA can run.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
OUT = HERE / 'obs_posteriors'
ELL = np.arange(80, 7980)
DL = ELL * (ELL + 1) / (2 * np.pi)
SAMPLES = 20000
LEE22 = Path.home() / 'tsz64k_lee22_mocks_20261001'
LEE22_FULL = Path('/lustre/work/kristero10/lee22_full_observation_20261001/results/Lee22_corrected_full')
MODELS = ('40 bins', 'MOPED', 'Merlin', 'PCA', 'raw')


def rebin40(dl):
    from train_ensemble import A_rebin
    return A_rebin(np.atleast_2d(dl))


def observations():
    """{name: dict(noisy_dl (40,), noisy_dl_unbinned (7900,), truth or None)}."""
    import analyze_64k as A
    folder = A.TEST256 / 'cluster_snapshot' / 'diagnostic_256' / 'observations'
    obs = {}
    for name in ('Battaglia12', 'L1_m9', 'Mstar-1sigma', 'fgas-8sigma'):
        d = np.load(folder / f'{name}.npz')
        obs['B12' if name == 'Battaglia12' else name] = dict(noisy_dl=d['noisy_dl'], noisy_dl_unbinned=d['noisy_dl_unbinned'],
                                                            truth=A.B12.tolist() if name == 'Battaglia12' else None)
    theta = json.loads((LEE22 / 'lee22_theta.json').read_text())['cases']
    for label, case in theta.items():
        f = LEE22 / 'out' / label / 'noisy_cross_39001001_39001002.npy'
        if not f.exists():
            print(f'{label}: not painted yet', flush=True)
            continue
        unbinned = np.load(f)[ELL] * DL
        obs[label] = dict(noisy_dl=rebin40(unbinned)[0], noisy_dl_unbinned=unbinned, truth=list(case['theta'].values()))
    f = LEE22_FULL / 'masked_noisy_cross_cl.npy'
    if f.exists():
        unbinned = np.load(f)[ELL] * DL
        obs['Lee22_full'] = dict(noisy_dl=rebin40(unbinned)[0], noisy_dl_unbinned=unbinned, truth=None)
    return obs


def model_inputs(raw_N, models=MODELS, merlin_folder=None):
    """{model: (ensemble, function noisy_dl, noisy_dl_unbinned -> (1, n_in) float32 input, stored B12 x_obs)}.

    models: the subset of MODELS to load; merlin_folder: the Merlin / PCA networks (default the full asinh run).
    """
    import argparse as ap
    import sbi_final as SF
    import merlin_compression as MC
    from scipy.linalg import solve_triangular
    from train_ensemble import Ensemble
    import analyze_64k as A
    assert MC.PARAM_SPACE == 'logit' or set(models) <= {'Merlin', 'PCA', 'MOPED'}, \
        'MERLIN_PARAM_SPACE=linear replaces the transforms the logit-space 40-band and raw networks need'
    wp = np.load(HERE / 'results_merlin_unbinned' / 'inputs' / 'whitening_pca.npz')
    L, V = wp['L'], wp['V']
    full = Path(merlin_folder) if merlin_folder else HERE / 'results_merlin_unbinned_asinh_p30'
    fm = dict(np.load(full / 'features_merlin.npz'))
    params = {k: fm[k] for k in ('s_center', 's_scale', 'z_mean', 'z_sd')}
    out = {}
    if '40 bins' in models or 'MOPED' in models:
        maps, _ = SF.variant_features(ap.Namespace(out=HERE / 'results_final_noise'))
        matrix = np.load(A.TEST256 / 'reference' / 'compression_context.npz')['Battaglia12_rcond1e-06']
    if '40 bins' in models:
        ens = Ensemble(HERE / 'results_final_noise' / 'bins40_noisy' / 'N61440', 8)
        out['40 bins'] = (ens, lambda n40, un: np.asarray(maps['bins40_noisy'](None, n40[None], None, None), np.float32),
                          ens.f['x_obs'])
    if 'MOPED' in models:
        # logit: the sbi_final.py ensemble; linear: the merlin_compression 'moped' variant of --merlin-folder, trained
        # on the same noise-level features (moped_noise_features.py); both take the same input map
        ens = (MC.ensemble(Path(merlin_folder), 'moped', 8) if MC.PARAM_SPACE == 'linear' else
               Ensemble(HERE / 'results_final_noise' / 'moped_noisy' / 'N61440', 8))
        out['MOPED'] = (ens, lambda n40, un: np.asarray(maps['moped_noisy'](None, None, un[None] @ matrix, None),
                                                        np.float32), ens.f['x_obs'])

    def merlin_input(n40, un):
        # merlin_unbinned_prepare.white: L^-1 x (no centring), then the PCA basis V, as for the training rows
        s = solve_triangular(L, np.asarray(un, dtype=np.float64)[None].T, lower=True, check_finite=False).T @ V
        return np.asarray(MC.asinh_standardize(s, params), np.float32)
    for name, variant in (('Merlin', 'merlin'), ('PCA', 'pca')):
        if name in models:
            ens = MC.ensemble(full, variant, 8)
            out[name] = (ens, merlin_input, ens.f['x_obs'])
    if 'raw' not in models:
        return out
    fr = dict(np.load(HERE / 'results_merlin_unbinned_asinh_p30' / 'features_raw.npz'))

    def raw_input(n40, un):
        z = (np.arcsinh((np.asarray(un, np.float64)[None] - fr['s_center']) / fr['s_scale']) - fr['z_mean']) / fr['z_sd']
        return z.astype(np.float32)
    raw_folder = HERE / f'results_merlin_unbinned_asinh_p30_N{raw_N}'
    if (raw_folder / 'raw_seed3.pt').exists():
        import os
        os.environ['MERLIN_TRAIN_ROWS'] = str(raw_N)          # the pool the networks were built with
        ens = MC.ensemble(raw_folder, 'raw', 8)
        del os.environ['MERLIN_TRAIN_ROWS']
        out[f'raw (N = {raw_N:,})'] = (ens, raw_input, ens.f['x_obs'])
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--raw-N', type=int, default=32000)
    p.add_argument('--threads', type=int, default=8)
    p.add_argument('--models', nargs='+', choices=MODELS, default=list(MODELS))
    p.add_argument('--merlin-folder', type=Path, default=None)
    p.add_argument('--out', type=Path, default=OUT)
    args = p.parse_args()
    import torch
    torch.set_num_threads(args.threads)
    out_dir = args.out if args.out.is_absolute() else HERE / args.out
    out_dir.mkdir(exist_ok=True)
    obs = observations()
    models = model_inputs(args.raw_N, args.models, args.merlin_folder)
    summary = dict(observations={k: dict(truth=v['truth']) for k, v in obs.items()}, checks={}, posteriors={})
    for model, (ens, make, stored) in models.items():
        x_b12 = make(obs['B12']['noisy_dl'], obs['B12']['noisy_dl_unbinned'])
        # the B12 input rebuilt from its spectrum must be the stored input of the model's evaluation
        summary['checks'][model] = float(np.max(np.abs(x_b12 - np.asarray(stored[:1], np.float32))))
        assert summary['checks'][model] < 1e-4, (model, summary['checks'][model])
        for name, o in obs.items():
            x = make(o['noisy_dl'], o['noisy_dl_unbinned'])
            torch.manual_seed(2)
            samples, _, _ = ens.posterior(x, SAMPLES, rng=np.random.default_rng(2))
            s = samples[:, 0].astype(np.float32)
            safe = model.split(' (')[0].replace(' ', '_')
            np.save(out_dir / f'{name}__{safe}.npy', s)
            q = np.quantile(s, [.16, .5, .84], axis=0)
            summary['posteriors'][f'{name}__{safe}'] = dict(model=model, observation=name, q16=q[0].tolist(),
                                                          median=q[1].tolist(), q84=q[2].tolist())
            print(f'{name} / {model}: median {np.round(q[1], 3).tolist()}', flush=True)
    summary['merlin_folder'] = str(args.merlin_folder) if args.merlin_folder else None
    import merlin_compression as MC
    summary['param_space'] = MC.PARAM_SPACE
    (out_dir / 'summary.json').write_text(json.dumps(summary, indent=1) + '\n')
    print('input checks (max |rebuilt - stored| B12 input):', summary['checks'], flush=True)


if __name__ == '__main__':
    main()
