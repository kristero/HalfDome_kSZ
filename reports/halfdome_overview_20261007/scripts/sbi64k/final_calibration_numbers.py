"""Calibration and comparison numbers of the final networks, recomputed from the stored per-row data.

Reads SBI_analysis/tsz64k_sbi_local/doc_metric_data/{beam_bins40,beam_nn_pca,beam_nn_merlin}.npz (written on idark by
doc_metric_data.py: per test row the truth, posterior mean, quantiles, log p of the truth, SBC ranks among 999 draws,
the TARP curve and each seed's posterior means). Beam-consistent noise; Merlin and PCA with noise-only whitening;
bounded (logit) 3-seed ensembles trained on 61,440 rows; 3,000 test rows.

Per network: mean log p, mean Pearson r, 68 % and 95 % marginal coverage, interval score, SBC max |ECDF - uniform|
per parameter, the largest TARP deviation from the diagonal. Pairs: the paired differences of doc_build.pairs_table
(same bootstrap seed 7 and 2,000 resamples). Writes final_calibration_numbers.json next to this script.
"""
import json
from pathlib import Path

import numpy as np

D = Path('/home/cbllover/HalfDome/SBI_analysis/tsz64k_sbi_local/doc_metric_data')
LABELS = dict(bins40='beam_bins40', pca='beam_nn_pca', merlin='beam_nn_merlin')
WIDTH = np.array([59, 3.975, 13.2, 2.1, 1.4, 0.6, 6.5, 4.5, 2.5])          # prior ranges (high - low)
DRAWS = 999                                                                # 1,000 // 3 seeds x 3


def load(label):
    return {k: (v.astype(np.float64) if v.dtype.kind == 'f' else v) for k, v in np.load(D / f'{label}.npz').items()}


def pearson(t, m):
    t = t - t.mean(-2, keepdims=True)
    m = m - m.mean(-2, keepdims=True)
    return (t * m).sum(-2) / np.sqrt((t * t).sum(-2) * (m * m).sum(-2))


def interval_score(d):
    t, lo, hi = d['truth'], d['q16'], d['q84']
    s = (hi - lo) + 2 / 0.32 * (lo - t) * (t < lo) + 2 / 0.32 * (t - hi) * (t > hi)
    return (s / WIDTH).mean(1)


def main():
    rows = {m: load(l) for m, l in LABELS.items()}
    out = dict(networks={}, pairs={})
    for m, d in rows.items():
        n = len(d['truth'])
        frac = d['rank'] / DRAWS
        grid = np.linspace(0, 1, 1001)
        sbc = [float(np.max(np.abs(np.searchsorted(np.sort(frac[:, i]), grid, side='right') / n - grid)))
               for i in range(9)]
        dev = d['ecp'] - d['alpha']
        k = int(np.argmax(np.abs(dev)))
        out['networks'][m] = dict(
            rows=n, mean_log_p=float(d['lp'].mean()), seed_log_p=d['lp_seeds'].mean(1).tolist(),
            mean_pearson_r=float(pearson(d['truth'], d['mean']).mean()),
            coverage68=float(np.mean((d['truth'] >= d['q16']) & (d['truth'] <= d['q84']))),
            coverage95=float(np.mean((d['truth'] >= d['q025']) & (d['truth'] <= d['q975']))),
            interval_score68=float(interval_score(d).mean()),
            sbc_max_abs_ecdf_minus_uniform=sbc, sbc_band95=float(1.358 / np.sqrt(n)),
            sbc_params_outside_band=int(np.sum(np.array(sbc) > 1.358 / np.sqrt(n))),
            tarp_max_deviation=float(dev[k]), tarp_max_deviation_at=float(d['alpha'][k]))
    rng = np.random.default_rng(7)
    for a, b in (('bins40', 'merlin'), ('pca', 'merlin'), ('bins40', 'pca')):
        A, B = rows[a], rows[b]
        t, n = A['truth'], len(A['truth'])
        dr = float((pearson(t, B['mean']) - pearson(t, A['mean'])).mean())
        boot = []
        for _ in range(2000):
            i = rng.integers(0, n, n)
            boot.append((pearson(t[i], B['mean'][i]) - pearson(t[i], A['mean'][i])).mean())
        sa, sb = pearson(t[None], A['seed_mean']).mean(1), pearson(t[None], B['seed_mean']).mean(1)
        e_r = float(np.hypot(np.std(boot, ddof=1), np.sqrt(sa.var(ddof=1) / 3 + sb.var(ddof=1) / 3)))
        dl = B['lp'] - A['lp']
        la, lb = A['lp_seeds'].mean(1), B['lp_seeds'].mean(1)
        e_l = float(np.hypot(dl.std(ddof=1) / np.sqrt(n), np.sqrt(la.var(ddof=1) / 3 + lb.var(ddof=1) / 3)))
        di = interval_score(B) - interval_score(A)
        out['pairs'][f'{b} - {a}'] = dict(r=dr, r_err=e_r, log_p=float(dl.mean()), log_p_err=e_l,
                                          interval_score=float(di.mean()),
                                          interval_score_err=float(di.std(ddof=1) / np.sqrt(n)),
                                          rows_where_b_has_higher_log_p=float(np.mean(dl > 0)))
    # the upper edge of the alpha_z_P0 prior (-6, 0.5): bias of the posterior mean and 68 % coverage by truth bin
    out['alpha_z_P0_by_truth'] = {}
    for m, d in rows.items():
        t, mu = d['truth'][:, 6], d['mean'][:, 6]
        inside = (t >= d['q16'][:, 6]) & (t <= d['q84'][:, 6])
        out['alpha_z_P0_by_truth'][m] = {
            f'[{lo}, {hi})': dict(rows=int(np.sum((t >= lo) & (t < hi))),
                                  mean_bias=float(np.mean((mu - t)[(t >= lo) & (t < hi)])),
                                  coverage68=float(np.mean(inside[(t >= lo) & (t < hi)])))
            for lo, hi in ((-6, -3), (-3, -1), (-1, 0), (0, 0.5))}
    # which rows make the wide-posterior peak near 0.9 of the flat width: brightness as doc_figures.signal_strength
    # (noise-free band 10, ell 1880-2080, over the B12 value)
    sbi = D.parent
    test = np.load(sbi / 'results_final' / 'split.npz')['test']
    clean = np.load(sbi / 'data' / 'rows_64440' / 'clean_dl.npy')[test, 9]
    b12 = np.load(sbi.parent / 'tsz_diagnostic_256_accelerated_20260922' / 'cluster_snapshot' / 'diagnostic_256'
                  / 'observations' / 'Battaglia12.npz')['clean_dl'][9]
    faint = clean / b12 < 0.1
    out['wide_rows'] = {}
    for m, d in rows.items():
        w = ((d['q84'] - d['q16']) / (0.6827 * WIDTH)).mean(1)
        wide = w > 0.85
        out['wide_rows'][m] = dict(rows_mean_width_over_flat_above_0p85=int(wide.sum()),
                                   share_fainter_than_0p1_b12=float(np.mean(faint[wide])),
                                   share_faint_all_rows=float(np.mean(faint)))
    # marginal coverage by sky brightness (log10 of signal / B12, the brightness bins of results_doc section 12)
    strength = np.log10(np.clip(clean / b12, 1e-12, None))
    out['coverage_by_brightness'] = {}
    for m, d in rows.items():
        inside = (d['truth'] >= d['q16']) & (d['truth'] <= d['q84'])
        inside95 = (d['truth'] >= d['q025']) & (d['truth'] <= d['q975'])
        out['coverage_by_brightness'][m] = {
            f'log10 [{lo}, {hi})': dict(rows=int(np.sum(sel)), coverage68=float(inside[sel].mean()),
                                        coverage68_P0=float(inside[sel, 0].mean()),
                                        coverage68_alpha_m_P0=float(inside[sel, 3].mean()),
                                        coverage95=float(inside95[sel].mean()))
            for lo, hi in ((-13, -2), (-2, -1), (-1, 0), (0, 1), (1, 2), (2, 3), (3, 20))
            for sel in [(strength >= lo) & (strength < hi)]}
    Path(__file__).with_name('final_calibration_numbers.json').write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps(out['coverage_by_brightness']['merlin'], indent=1))
    print(json.dumps(out['alpha_z_P0_by_truth'], indent=1))
    print(json.dumps(out['wide_rows'], indent=1))
    for m, e in out['networks'].items():
        print(m, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in e.items()
                  if k not in ('sbc_max_abs_ecdf_minus_uniform', 'seed_log_p')},
              'SBC', np.round(e['sbc_max_abs_ecdf_minus_uniform'], 3).tolist())
    for p, e in out['pairs'].items():
        print(p, {k: round(v, 3) for k, v in e.items()})


if __name__ == '__main__':
    main()
