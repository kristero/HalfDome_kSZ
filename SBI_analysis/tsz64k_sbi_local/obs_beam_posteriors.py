"""FLAMINGO L1_m9 (fiducial) map posteriors from the beam-consistent Merlin / PCA networks (runs on idark).

The observation (flamingo_tsz/process_maps.jl: the Yang+26 L1_m9 lensed y map with the engine's 2' beam, f_sky 0.4 cap
mask and two SO baseline noise splits of seeds 22446 / 22447, the noise without the beam, as in the engine) is made
beam-consistent as the training rows are (merlin_beam_prepare.correct), with the campaign's own masked noise x noise
cross spectrum (cache/masked_noise_cross_cl.npy). Hard check: the noisy - clean excess regresses on that C_nn with
slope ~1 at ell >= 6000 (beamfix.noise_slope; ~0 for an unrelated realization), and the B12 observation with its own
C_nn passes the same test.
  L1_m9          the beam-consistent spectrum, C_ss + B (C_noisy - C_ss - C_nn) + B^2 C_nn
  L1_m9_pixwin   also without the NSIDE 4096 pixel window p_ell: the map is a pixel average, the training signal is
                 evaluated at pixel centres (engine.jl make_signal: map2alm of the NSIDE 8192 point samples, alm2map),
                 so the map's signal carries p_ell^2 and its signal x noise p_ell:
                 C_ss / p^2 + B (C_noisy - C_ss - C_nn) / p + B^2 C_nn   (approximate under the mask)
Each input set maps a spectrum as merlin_beam_prepare.py did (L^-1 x, the PCA basis V, the asinh standardization of
its features_merlin.npz); the B12 observation rebuilt this way must reproduce the stored s_obs (b12.npz) and x_obs.
  beam_b12   results_merlin_unbinned_beam_asinh_p30          B12 whitening
  beam_nn    results_merlin_unbinned_beam_nnwhite_asinh_p30  noise-only whitening
The networks: those folders (bounded, logit space) or their _lin partners (unbounded); MERLIN_PARAM_SPACE is read when
merlin_compression is imported, so run once per space:
    MERLIN_PARAM_SPACE=logit  python obs_beam_posteriors.py [--threads 16]
    MERLIN_PARAM_SPACE=linear python obs_beam_posteriors.py [--threads 16]
Writes obs_posteriors_beam/<obs>__<set>_<bounded|unbounded>_<merlin|pca>.npy (SAMPLES draws, theta units, float32,
untruncated; drawn as posterior_rows.py --b12 draws: torch.manual_seed(2), default_rng(2)), features_<set>.npz (the
network inputs of every observation) and summary_<space>.json (checks, quantiles). The B12 draws, from the stored x_obs,
must equal posterior_rows_whitening/<set>_<space>_<variant>_b12.npy where that exists.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

import merlin_beam_prepare as P
import merlin_compression as MC
import analyze_64k as A

FLAMINGO_NN = Path('/lustre/work/kristero10/flamingo_tsz_comparison_20260914/cache/masked_noise_cross_cl.npy')
FLAMINGO_NN_SHA256 = 'bb78cf2bf07633bee39ce840440362aad219106400e87e511dbe286bde3148a1'
PIXWIN = P.HERE / 'data' / 'pixwin_nside4096.npy'         # healpy.pixwin(4096, lmax=7979)
SETS = dict(beam_b12=('results_merlin_unbinned_beam_asinh_p30', 'results_merlin_unbinned_beam/inputs'),
            beam_nn=('results_merlin_unbinned_beam_nnwhite_asinh_p30', 'results_merlin_unbinned_beam_nnwhite/inputs'))
SAMPLES = 300000
OUT = P.HERE / 'obs_posteriors_beam'
HIGH_ELL = P.ELL >= 6000


def noise_slope(clean, noisy, nn):
    """beamfix.noise_slope: regression coefficient of noisy - clean on C_nn over ell >= 6000 (~1 for the right pair)."""
    d, m = noisy[HIGH_ELL] - clean[HIGH_ELL], nn[HIGH_ELL]
    return float(np.sum(d * m) / np.sum(m * m))


def spectra():
    """The observed D_ell (ell 80..7979) of every observation, and the checks."""
    folder = A.TEST256 / 'cluster_snapshot' / 'diagnostic_256' / 'observations'
    b12, fl = np.load(folder / 'Battaglia12.npz'), np.load(folder / 'L1_m9.npz')
    assert hashlib.sha256(FLAMINGO_NN.read_bytes()).hexdigest() == FLAMINGO_NN_SHA256, FLAMINGO_NN
    nn_fl = P.dl(FLAMINGO_NN)
    nn_b12 = P.dl(P.B12DRAWS / 'nn' / f'B12_{P.OBS_PAIR}.npy')
    bl = np.load(P.BEAMFIX / 'beam_bl.npy')[P.ELL]
    pw = np.load(PIXWIN)[P.ELL]
    checks = dict(slope_L1_m9=noise_slope(fl['clean_dl_unbinned'], fl['noisy_dl_unbinned'], nn_fl),
                  slope_B12=noise_slope(b12['clean_dl_unbinned'], b12['noisy_dl_unbinned'], nn_b12),
                  slope_L1_m9_with_B12_nn=noise_slope(fl['clean_dl_unbinned'], fl['noisy_dl_unbinned'], nn_b12))
    assert abs(checks['slope_L1_m9'] - 1) < 0.15 and abs(checks['slope_B12'] - 1) < 0.15, checks
    clean, noisy = fl['clean_dl_unbinned'], fl['noisy_dl_unbinned']
    x = dict(B12=P.correct(b12['clean_dl_unbinned'], b12['noisy_dl_unbinned'], nn_b12, bl),
             L1_m9=P.correct(clean, noisy, nn_fl, bl),
             L1_m9_pixwin=clean / pw ** 2 + bl * (noisy - clean - nn_fl) / pw + bl ** 2 * nn_fl)
    return x, checks


def features(x, checks):
    """{set: {obs: (1, n_pca) float32 network input}}, with the B12 reconstruction checks added to checks."""
    out = {}
    for name, (folder, inputs) in SETS.items():
        wp = np.load(P.HERE / inputs / 'whitening_pca.npz')
        s = {k: P.white(wp['L'], v[None]) @ wp['V'] for k, v in x.items()}
        del wp
        s_obs = np.load(P.HERE / inputs / 'b12.npz')['s_obs'][:1]
        checks[f'{name}_b12_s_obs'] = float(np.max(np.abs(s['B12'] - s_obs)) / np.max(np.abs(s_obs)))
        f = np.load(P.HERE / folder / 'features_merlin.npz')
        params = {k: f[k] for k in ('s_center', 's_scale', 'z_mean', 'z_sd')}
        z = {k: np.asarray(MC.asinh_standardize(v, params), np.float32) for k, v in s.items()}
        checks[f'{name}_b12_x_obs'] = float(np.max(np.abs(z['B12'] - f['x_obs'][:1])))
        assert checks[f'{name}_b12_s_obs'] < 1e-9 and checks[f'{name}_b12_x_obs'] < 1e-4, checks
        np.savez(OUT / f'features_{name}.npz', **z)
        out[name] = z
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--threads', type=int, default=16)
    args = p.parse_args()
    import torch
    t0 = time.time()
    OUT.mkdir(exist_ok=True)
    space = 'bounded' if MC.PARAM_SPACE == 'logit' else 'unbounded'
    x, checks = spectra()
    z = features(x, checks)
    print(f'{space}: inputs built in {time.time() - t0:.0f} s; checks {checks}', flush=True)
    summary = dict(param_space=MC.PARAM_SPACE, samples=SAMPLES, checks=checks, flamingo_nn=str(FLAMINGO_NN),
                   posteriors={})
    for name, (folder, _) in SETS.items():
        net_folder = P.HERE / (folder + ('_lin' if space == 'unbounded' else ''))
        for variant in ('merlin', 'pca'):
            ens = MC.ensemble(net_folder, variant, args.threads)
            for obs in ('B12', 'L1_m9', 'L1_m9_pixwin'):
                # B12 from the stored input, so that the draws reproduce posterior_rows_whitening exactly
                xin = ens.f['x_obs'][:1] if obs == 'B12' else z[name][obs]
                torch.manual_seed(2)
                samples = ens.posterior(xin, SAMPLES, rng=np.random.default_rng(2))[0][:, 0].astype(np.float32)
                label = f'{obs}__{name}_{space}_{variant}'
                np.save(OUT / f'{label}.npy', samples)
                q = np.quantile(samples, [0.16, 0.5, 0.84], axis=0)
                inside = float(np.mean(np.all((samples >= ens.low) & (samples <= ens.high), axis=1)))
                summary['posteriors'][label] = dict(q16=q[0].tolist(), median=q[1].tolist(), q84=q[2].tolist(),
                                                    inside_prior=inside)
                if obs == 'B12':
                    ref = P.HERE / 'posterior_rows_whitening' / f'{name}_{space}_{variant}_b12.npy'
                    if ref.exists():
                        checks[f'{label}_vs_posterior_rows'] = float(np.max(np.abs(samples - np.load(ref))))
                print(f'{label}: median {np.round(q[1], 3).tolist()} inside {inside:.3f} '
                      f'({time.time() - t0:.0f} s)', flush=True)
    summary['checks'] = checks
    (OUT / f'summary_{space}.json').write_text(json.dumps(summary, indent=1) + '\n')
    print(f'{space}: done in {time.time() - t0:.0f} s; checks {checks}', flush=True)


if __name__ == '__main__':
    main()
