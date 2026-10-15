"""Band-averaged noise scatter of the B12 noise draws against the noise-only scatter of the training pool (idark).

merlin_beam_prepare.py reports single-multipole ratios of the 20 B12 draws' scatter to the pool's, which scatter by
about 1/sqrt(2 x 19) = 16 % each. Here in bands (variances averaged over the band's multipoles, then the square root):
  noisy_over_nn    production noisy B12 draws (signal x noise plus noise x noise) / pool C_nn
  nn_b12_over_nn   the B12 draws' own regenerated C_nn / pool C_nn (the same noise: 1 within the estimate)
from the 20 B12 draws other than the observation's and the first 4,000 training-pool rows' C_nn (production noise;
the beam-consistent versions scale both by B_ell^2 alike).

    python noise_scatter_bands.py
Writes results_merlin_unbinned_beam/inputs/noise_scatter_bands.json.
"""
import json

import numpy as np

import analyze_64k as A
import merlin_beam_prepare as P

POOL_ROWS = 4000
EDGES = np.r_[np.arange(80, 1000, 50), np.arange(1000, 7980, 100), 7980]


def main():
    order = np.load(P.HERE / 'results_final' / 'split.npz')['order']
    row_id = np.load(P.HERE / 'data' / 'rows_64440' / 'row_id.npy')
    nn = np.stack([P.dl(P.BEAMFIX / 'nn' / f'r{int(r):05d}.npy') for r in row_id[order[:POOL_ROWS]]])
    pairs = [p.stem.replace('noisy_cross_', '') for p in sorted(P.B12_MOCKS.glob('noisy_cross_*.npy'))]
    keep = [p for p in pairs if p != P.OBS_PAIR]
    noisy = np.stack([P.dl(P.B12_MOCKS / f'noisy_cross_{p}.npy') for p in keep])
    nn_b12 = np.stack([P.dl(P.B12DRAWS / 'nn' / f'B12_{p}.npy') for p in keep])
    v_nn, v_noisy, v_nnb = nn.var(0, ddof=1), noisy.var(0, ddof=1), nn_b12.var(0, ddof=1)
    obs = np.load(A.TEST256 / 'cluster_snapshot' / 'diagnostic_256' / 'observations' / 'Battaglia12.npz')
    band = lambda v, a, b: float(v[a - 80:b - 80].mean())
    rows = [dict(ell_lo=int(a), ell_hi=int(b), noisy_over_nn=np.sqrt(band(v_noisy, a, b) / band(v_nn, a, b)),
                 nn_b12_over_nn=np.sqrt(band(v_nnb, a, b) / band(v_nn, a, b)),
                 b12_signal_over_nn_sd=band(obs['clean_dl_unbinned'], a, b) / np.sqrt(band(v_nn, a, b)))
            for a, b in zip(EDGES[:-1], EDGES[1:])]
    out = P.REPORT.parent / 'noise_scatter_bands.json'
    out.write_text(json.dumps(dict(b12_draws=len(keep), pool_rows=POOL_ROWS, bands=rows), indent=1) + '\n')
    for r in rows[::8]:
        print(f"{r['ell_lo']:5d}-{r['ell_hi']:5d}  noisy/nn {r['noisy_over_nn']:.3f}  nnB12/nn {r['nn_b12_over_nn']:.3f}")
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
