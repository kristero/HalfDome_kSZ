"""MOPED inputs with the corrected (noise-level) asinh scale for the merlin_compression.py 'moped' variant (idark).

merlin_compression.py prepare writes features_moped.npz from results_final/moped_noisy (the median-scale asinh of
the first runs). This writes the same layout from results_final_noise/moped_noisy/N61440 (the noise-level scale of
sbi_final.py --asinh-scale noise), with the 21 B12 noise draws through the same map, into an output folder whose
features_merlin.npz defines the training order (the pools must agree row by row).

    python moped_noise_features.py results_merlin_unbinned_asinh_p30_lin
"""
import argparse
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('folder', type=Path)
    args = p.parse_args()
    import sbi_final as S
    import analyze_64k as A
    import merlin_compression as MC
    out = args.folder if args.folder.is_absolute() else HERE / args.folder
    mf = np.load(HERE / 'results_final_noise' / 'moped_noisy' / 'N61440' / 'features.npz')
    fm = np.load(out / 'features_merlin.npz')
    assert np.array_equal(mf['pool'], fm['pool']) and np.array_equal(mf['held'], fm['held']), 'pool / test order differ'
    maps, _ = S.variant_features(argparse.Namespace(out=HERE / 'results_final_noise'))
    matrix = np.load(A.TEST256 / 'reference' / 'compression_context.npz')['Battaglia12_rcond1e-06']
    draws = MC.b12_noise_draws()
    x_draws = np.asarray(maps['moped_noisy'](None, None, draws @ matrix, None), np.float32)
    # the B12 observation rebuilt through the same map must be the stored input
    obs = np.load(A.TEST256 / 'cluster_snapshot' / 'diagnostic_256' / 'observations' / 'Battaglia12.npz')
    x_obs = np.asarray(maps['moped_noisy'](None, None, obs['noisy_dl_unbinned'][None] @ matrix, None), np.float32)
    check = float(np.max(np.abs(x_obs - mf['x_obs'][:1])))
    assert check < 1e-5, check
    target = out / 'features_moped.npz'
    if target.is_symlink():
        target.unlink()                                  # the median-scale features linked in by the job script
    np.savez(target, x_pool=mf['x_pool'], x_held=mf['x_held'], x_obs=mf['x_obs'], x_draws=x_draws,
             theta_pool=mf['theta_pool'], theta_held=mf['theta_held'], pool=mf['pool'], held=mf['held'],
             low=mf['low'], high=mf['high'], input_scale=np.array('noise'))
    print(f'wrote {target}: pool {len(mf["pool"])}, held {len(mf["held"])}, B12 check {check:.1e}', flush=True)


if __name__ == '__main__':
    main()
