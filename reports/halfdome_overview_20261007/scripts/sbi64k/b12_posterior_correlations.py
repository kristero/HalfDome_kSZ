"""Parameter-parameter correlations of the B12-mock posteriors (stored samples) and of the Fisher forecast.

Reads only saved arrays (no network is evaluated):
  SBI_analysis/tsz64k_sbi_local/doc_metric_data/beam_bins40_b12.npy                       40 bins, beam-consistent
  SBI_analysis/tsz64k_sbi_local/results_merlin_unbinned_beam_nnwhite_asinh_p30/b12_samples_{merlin,pca}.npy
                                                       Merlin / PCA, beam-consistent noise, noise-only whitening
  SBI_analysis/tsz64k_sbi_local/results_fisher_b12_beam/fisher_forecast_b12.json      Fisher 'every multipole'
                                                       correlation matrix (unboxed Gaussian)
Writes b12_posterior_correlations.json next to this script and prints the pairs with the largest |rho|.
"""
import json
from pathlib import Path

import numpy as np

SBI = Path('/home/cbllover/HalfDome/SBI_analysis/tsz64k_sbi_local')
NAMES = ['P0', 'xc', 'beta', 'alpha_m_P0', 'alpha_m_xc', 'alpha_m_beta', 'alpha_z_P0', 'alpha_z_xc', 'alpha_z_beta']


def main():
    sets = {'40 bins': np.load(SBI / 'doc_metric_data' / 'beam_bins40_b12.npy'),
            'PCA': np.load(SBI / 'results_merlin_unbinned_beam_nnwhite_asinh_p30' / 'b12_samples_pca.npy'),
            'Merlin': np.load(SBI / 'results_merlin_unbinned_beam_nnwhite_asinh_p30' / 'b12_samples_merlin.npy')}
    corr = {k: np.corrcoef(np.asarray(s, np.float64).T) for k, s in sets.items()}
    fisher = json.loads((SBI / 'results_fisher_b12_beam' / 'fisher_forecast_b12.json').read_text())
    corr['Fisher, every multipole (Gaussian, no box)'] = np.array(fisher['fisher']['every multipole']['correlation'])
    pairs = [(i, j) for i in range(9) for j in range(i + 1, 9)]
    # the eight largest |rho| of Merlin, with the same pairs for the others
    top = sorted(pairs, key=lambda p: -abs(corr['Merlin'][p]))[:8]
    out = dict(samples={k: int(len(s)) for k, s in sets.items()},
               top_pairs_by_merlin=[dict(pair=f'{NAMES[i]} - {NAMES[j]}',
                                         **{k: round(float(c[i, j]), 3) for k, c in corr.items()}) for i, j in top],
               matrices={k: np.round(c, 3).tolist() for k, c in corr.items()}, names=NAMES)
    Path(__file__).with_name('b12_posterior_correlations.json').write_text(json.dumps(out, indent=1) + '\n')
    print(out['samples'])
    for row in out['top_pairs_by_merlin']:
        print(row)


if __name__ == '__main__':
    main()
