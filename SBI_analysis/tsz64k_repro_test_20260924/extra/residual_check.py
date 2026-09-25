"""Production residuals (noisy - clean) normalized by their analytic expected scatter.

For a fixed signal s and independent noise splits n1, n2 with spectrum N_l, the pseudo
cross spectrum residual C(ms,mn2) + C(mn1,ms) + C(mn1,mn2) has zero mean and variance
w4 (2 C_s N + N^2) / (2l+1), with C_s the row's own full-sky signal power and w4 the
mask's fourth moment (the pseudo-C_l are treated as independent across l, which gives the
standard band-power variance). z = residual / expected sigma must be ~N(0,1) per bin, and
two different noise draws of the same row must give uncorrelated z vectors.
"""
import json
import sys
from pathlib import Path

import numpy as np

EXTRA = Path(__file__).resolve().parent
sys.path.insert(0, str(EXTRA))
from run_extra import ELL, DL, TEST, WHERE, common, manage  # noqa: E402

W4 = 0.40   # fixed cap mask, f_sky = 0.4 with a 60 arcmin apodized edge (edge ring ~0.9% of the sky)
table = np.loadtxt(common.SO_DIR / 'SO_LAT_Nell_T_atmv1_baseline_fsky0p4_ILC_tSZ.txt', comments='#')
noise = np.zeros(7980)
noise[table[:, 0].astype(int)] = table[:, 1]            # column 2 of the table, as the engine reads it
bins = np.loadtxt(EXTRA / 'noise_check' / 'bins.csv', delimiter=',', ndmin=2)
S1 = np.array([np.sum((2 * np.arange(a, b + 1) + 1) * noise[int(a):int(b) + 1]) for a, b, *_ in bins])
assert np.allclose(S1, bins[:, 2], rtol=1e-12), 'not the noise table the engine used'
edges = np.r_[np.arange(80, 7881, 200), 7980]
weights = [(2 * ELL[(ELL >= a) & (ELL < b)] + 1) for a, b in zip(edges[:-1], edges[1:])]


def expected_sigma(full_sky_signal):
    var_l = W4 * (2 * full_sky_signal[ELL] * noise[ELL] + noise[ELL] ** 2) / (2 * ELL + 1) * DL ** 2
    return np.array([np.sqrt(np.sum((w / w.sum()) ** 2 * var_l[(ELL >= a) & (ELL < b)]))
                     for w, a, b in zip(weights, edges[:-1], edges[1:])])


def test_row(row):
    return TEST / 'diagnostic_256' / 'batches' / f'{row // 4:03d}' / f'{row:05d}'


rows = [r for r in range(16)]
z = dict(new=[], test=[])
for row in rows:
    for key, folder in (('new', WHERE['rows'] / f'{row:05d}'), ('test', test_row(row))):
        clean = np.load(folder / 'masked_clean_cl.npy')[ELL] * DL
        noisy = np.load(folder / 'masked_noisy_cross_cl.npy')[ELL] * DL
        residual = manage.rebin_unbinned((noisy - clean)[None, :], ELL)[0]
        z[key].append(residual / expected_sigma(np.load(folder / 'unmasked_clean_cl.npy')))
z_new, z_test = np.array(z['new']), np.array(z['test'])
same = np.array([np.corrcoef(a, b)[0, 1] for a, b in zip(z_new, z_test)])
between = np.array([np.corrcoef(z_new[i], z_new[j])[0, 1] for i in range(16) for j in range(i)])
result = dict(
    w4=W4, rows=len(rows), bins=z_new.shape[1],
    z_new=dict(mean=float(z_new.mean()), std=float(z_new.std())),
    z_test=dict(mean=float(z_test.mean()), std=float(z_test.std())),
    corr_same_row_new_vs_test=dict(values=[round(float(v), 3) for v in same], mean=float(same.mean()),
                                   std=float(same.std(ddof=1)), sem=float(same.std(ddof=1) / np.sqrt(len(same)))),
    corr_between_rows=dict(pairs=len(between), mean=float(between.mean()), std=float(between.std(ddof=1)),
                           max_abs=float(np.abs(between).max())),
    independent_expectation_std=float(1 / np.sqrt(z_new.shape[1] - 1)),
    mean_z_per_bin_over_32_draws=dict(max_abs=float(np.abs(np.r_[z_new, z_test].mean(0) * np.sqrt(32)).max())))
np.savez(EXTRA / 'results' / 'residuals_normalized.npz', rows=rows, z_new=z_new, z_test=z_test)
common.atomic_json(EXTRA / 'results' / 'residual_check.json', result)
print(json.dumps(result, indent=1))
