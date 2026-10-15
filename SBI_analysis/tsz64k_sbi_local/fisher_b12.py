"""Local Fisher forecast at Battaglia+12 with the frozen tSZ_64k engine, against the 13 Sept pilot and the NPE.

  fisher_b12.py [--results results_logit] [--fisher mocks/fisher_b12] [--b12 mocks/pass1_B12/B12]

Derivatives: central differences of the masked clean 40-band D_ell (the 64k binning) painted by paint_mocks.jl at
B12 +- h, h = 0.5 % of the 64k prior width; P0 analytic, since D_ell scales exactly as P0^2 in the engine.
Covariance: the 20 new SO noise draws of the B12 sky (the fitted observation excluded), OAS correlation shrinkage
with the measured band variances (the pilot's estimator, so_nine_fisher.conditional_covariance).
Posteriors: the local linear-Gaussian likelihood times a hard uniform prior (the pilot's so_nine_fisher.
hard_prior_posterior), with the pilot's narrow prior and with the 64k prior; plus 1/sqrt(F_ii) (others fixed).
Writes <results>/fisher_b12.json and fig_fisher_b12.png.
"""
import argparse
import json
import sys
from pathlib import Path
import tomllib

import numpy as np
from scipy.linalg import solve_triangular
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import so_nine_fisher as S                                                      # noqa: E402
from train_ensemble import A_rebin, ELL, DL                                      # noqa: E402
from ppc_plots import style, ell_ticks, INK, INK2, GRID                          # noqa: E402

B12 = S.BATTAGLIA12
NAMES = S.PARAMETER_NAMES
LABELS = [r'$P_0$', r'$x_{\rm c}$', r'$\beta$', r'$\alpha_{m,P_0}$', r'$\alpha_{m,x_{\rm c}}$', r'$\alpha_{m,\beta}$',
          r'$\alpha_{z,P_0}$', r'$\alpha_{z,x_{\rm c}}$', r'$\alpha_{z,\beta}$']
LOW64 = np.array([1.0, 0.025, 2.8, -0.6, -1.0, -0.2, -6.0, -1.5, -0.5])
HIGH64 = np.array([60.0, 4.0, 16.0, 1.5, 0.4, 0.4, 0.5, 3.0, 2.0])
PILOT = HERE.parent / 'outputs' / 'so9_fisher_noise16_20260913' / 'results'
OBS_SEEDS = (39001001, 39001002)
BLUE, ORANGE, AQUA, YELLOW = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
EDGES = np.r_[np.arange(80, 7881, 200), 7980]
CENTRES = 0.5 * (EDGES[:-1] + EDGES[1:] - 1)


def band(cl):
    return A_rebin((np.load(cl)[ELL] * DL)[None])[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results', type=Path, default=HERE / 'results_logit')
    p.add_argument('--fisher', type=Path, default=HERE / 'mocks' / 'fisher_b12')
    p.add_argument('--b12', type=Path, default=HERE / 'mocks' / 'pass1_B12' / 'B12')
    p.add_argument('--out', type=Path, default=None, help='output folder (default: --results)')
    p.add_argument('--test-pilot-jacobian', action='store_true', help="code test only: the pilot's derivatives (other binning)")
    args = p.parse_args()
    out = args.out or args.results; out.mkdir(parents=True, exist_ok=True)
    plan = json.loads((args.fisher / 'plan.json').read_text())
    h = np.array(plan['step'])
    np.testing.assert_allclose(plan['fiducial'], B12)
    width64 = HIGH64 - LOW64
    pilot = np.load(PILOT / 'fisher_arrays.npz')
    low_p, high_p = pilot['prior_low'], pilot['prior_high']
    report = dict(step=dict(zip(NAMES, h.tolist())), step_rule=plan['step_rule'])

    # ---------------- mean derivatives ----------------
    d0 = band(args.b12 / 'masked_clean_cl.npy')
    J = np.zeros((40, 9)); curvature = np.full(9, np.nan)
    J[:, 0] = 2 * d0 / B12[0]
    for i in range(1, 9):
        if args.test_pilot_jacobian:
            J[:, i] = pilot['derivatives'][:, i]; continue
        dp, dm = (band(args.fisher / f'{NAMES[i]}_{s}' / 'masked_clean_cl.npy') for s in ('p', 'm'))
        J[:, i] = (dp - dm) / (2 * h[i])
        curvature[i] = float(np.linalg.norm(dp + dm - 2 * d0) / np.linalg.norm(dp - dm))   # second- over first-order change
        for s, th in (('p', +1), ('m', -1)):
            case = tomllib.loads((args.fisher / f'{NAMES[i]}_{s}' / 'case.toml').read_text())
            expect = B12.copy(); expect[i] += th * h[i]
            np.testing.assert_allclose(case['theta'], expect, rtol=0, atol=1e-12)
            assert case['cache']['nodes'] == [256, 128, 64], case['cache']['nodes']
    report['second_order_over_first_order_change'] = dict(zip(NAMES, curvature.tolist()))

    # ---------------- covariance of the 40 bands from the new noise draws ----------------
    fitted = args.b12 / f'noisy_cross_{OBS_SEEDS[0]}_{OBS_SEEDS[1]}.npy'
    draws = np.stack([band(f) for f in sorted(args.b12.glob('noisy_cross_*.npy')) if f != fitted])
    C, shrink = S.conditional_covariance(draws)
    obs = band(fitted)
    report['covariance'] = dict(draws=len(draws), oas_shrinkage=shrink, estimator='so_nine_fisher.conditional_covariance')

    # ---------------- Fisher matrices, eigenmodes ----------------
    F = S.fisher_matrix(J, C)                                           # theta units
    Fw = F * np.outer(width64, width64)                                 # 64k prior-width units
    lam, vec, resolved = S.fisher_modes(Fw)
    Fp = F * np.outer(high_p - low_p, high_p - low_p)
    lam_p, _, _ = S.fisher_modes(Fp)
    conditional = 1 / np.sqrt(np.diag(F))
    unregularised = np.sqrt(np.diag(np.linalg.pinv(F, rcond=1e-14)))
    pilot_conditional = 1 / np.sqrt(np.diag(pilot['fisher_theta']))
    chol = np.linalg.cholesky(C)
    white = solve_triangular(chol, J, lower=True)
    # how much of the beta response the other eight reproduce (GLS fit of the whitened beta column on the others)
    others = [i for i in range(9) if i != 2]
    coef, *_ = np.linalg.lstsq(white[:, others], white[:, 2], rcond=None)
    resid = white[:, 2] - white[:, others] @ coef
    report['fisher'] = dict(
        eigenvalues_64k_prior_width_units=lam.tolist(), sigma_along_eigenvectors_64k_widths=(1 / np.sqrt(lam)).tolist(),
        eigenvectors_64k_widths=vec.tolist(), eigenvalues_pilot_prior_width_units=lam_p.tolist(),
        pilot_eigenvalues_pilot_prior_width_units=pilot['fisher_eigenvalues'].tolist(),
        conditional_sigma_others_fixed=dict(zip(NAMES, conditional.tolist())),
        pilot_conditional_sigma_others_fixed=dict(zip(NAMES, pilot_conditional.tolist())),
        unmarginalised_over_marginalised_no_prior=dict(zip(NAMES, (conditional / unregularised).tolist())),
        marginal_sigma_no_prior=dict(zip(NAMES, unregularised.tolist())),
        beta_response_reproduced_by_other_eight=float(1 - np.sum(resid ** 2) / np.sum(white[:, 2] ** 2)),
        beta_best_combination=dict(zip([NAMES[i] for i in others], (-coef).tolist())))

    # ---------------- linear-Gaussian likelihood x hard uniform prior ----------------
    post, info = {}, {}
    for key, (lo, hi) in (('pilot_prior', (low_p, high_p)), ('prior_64k', (LOW64, HIGH64))):
        for data, residual in (('forecast', np.zeros(40)), ('observed', obs - d0)):
            s, meta = S.hard_prior_posterior(J, C, residual, lo, hi, B12, draws=4_000_000, count=200_000, seed=11)
            post[f'{key}_{data}'], info[f'{key}_{data}'] = s, meta
    npe = np.load(args.results / 'noisy' / 'posterior_Battaglia12.npy').astype(np.float64)
    npe_clean = np.load(args.results / 'clean' / 'posterior_Battaglia12.npy').astype(np.float64)

    def summ(s, w=width64):
        q = np.quantile(s, [.16, .5, .84], axis=0)
        return dict(median=q[1].tolist(), q16=q[0].tolist(), q84=q[2].tolist(), std=s.std(0).tolist(), width68_over_64k_prior=((q[2] - q[0]) / w).tolist())
    report['posteriors'] = {k: dict(summ(v), sampling={kk: vv for kk, vv in info[k].items() if kk in ('importance_ess', 'proposals', 'inside')})
                            for k, v in post.items()}
    report['posteriors']['npe_noisy'] = summ(npe); report['posteriors']['npe_clean'] = summ(npe_clean)
    pc = np.genfromtxt(PILOT / 'parameter_constraints.csv', delimiter=',', names=True, dtype=None, encoding=None)
    report['pilot_table_std'] = {r['parameter']: float(r['std']) for r in pc if r['method'] == 'fiducial_forecast'}

    # widths along the 64k-prior Fisher eigenvectors (prior-width units, centred on B12)
    proj = lambda s: ((s - B12) / width64) @ vec                                              # noqa: E731
    report['std_along_eigenvectors'] = dict(fisher=(1 / np.sqrt(lam)).tolist(), fisher_x_64k_prior=proj(post['prior_64k_forecast']).std(0).tolist(),
                                            npe_noisy=proj(npe).std(0).tolist(), npe_clean=proj(npe_clean).std(0).tolist(),
                                            flat_prior=float(1 / np.sqrt(12)))

    # ---------------- beam error and amplitude offset projected on the parameters ----------------
    sig = lambda f: np.deg2rad(f / 60) / np.sqrt(8 * np.log(2))                               # noqa: E731
    clean_u = np.load(args.b12 / 'masked_clean_cl.npy')[ELL] * DL
    d_beam = A_rebin((clean_u * (np.exp(-ELL * (ELL + 1) * (sig(2.1) ** 2 - sig(2.0) ** 2)) - 1))[None])[0]
    reg = F + 12 * np.diag(1 / width64 ** 2)                     # flat 64k prior as a Gaussian of the same variance
    wb = solve_triangular(chol, d_beam, lower=True)
    shift = np.linalg.solve(reg, white.T @ wb)
    left = wb - white @ shift
    report['beam_plus_0p1_arcmin'] = dict(parameter_shift=dict(zip(NAMES, shift.tolist())),
                                          chi2_of_beam_change=float(wb @ wb), chi2_left_after_refit=float(left @ left))
    np.savez_compressed(out / 'fisher_b12_samples.npz', **{k: v[:50_000].astype(np.float32) for k, v in post.items()}, J=J, C=C, d0=d0)
    (out / 'fisher_b12.json').write_text(json.dumps(report, indent=1) + '\n')
    print(json.dumps({k: report[k] for k in ('second_order_over_first_order_change', 'covariance', 'beam_plus_0p1_arcmin')}, indent=1))
    f = report['fisher']
    print('sigma along eigenvectors (64k widths):', np.round(f['sigma_along_eigenvectors_64k_widths'], 5).tolist())
    print('pilot eigen (pilot widths) new vs old:', np.round(lam_p, 3).tolist(), np.round(pilot['fisher_eigenvalues'], 3).tolist())
    print('conditional sigma new:', np.round(conditional, 4).tolist())
    print('conditional sigma pilot:', np.round(pilot_conditional, 4).tolist())
    print('beta reproduced by the other eight:', f['beta_response_reproduced_by_other_eight'])
    for k in ('pilot_prior_forecast', 'prior_64k_forecast', 'prior_64k_observed', 'npe_noisy'):
        print(k, 'std', np.round(report['posteriors'][k]['std'], 3).tolist(), 'w68/64k', np.round(report['posteriors'][k]['width68_over_64k_prior'], 3).tolist())
    print('std along eigenvectors:', {k: np.round(v, 4).tolist() if isinstance(v, list) else v for k, v in report['std_along_eigenvectors'].items()})

    # ================= figure =================
    style()
    fig, axes = plt.subplots(2, 2, figsize=(19 / 2.54, 17.5 / 2.54), gridspec_kw=dict(hspace=0.95, wspace=0.42))
    (a1, a2), (a3, a4) = axes
    # (a) whitened responses to a 1 % (of the 64k prior width) change, and the beam
    sd = np.sqrt(np.diag(C))
    for i, c, ls in ((2, BLUE, '-'), (1, ORANGE, (0, (5, 1.5))), (0, AQUA, (0, (1.5, 1)))):
        a1.plot(CENTRES, J[:, i] * 0.01 * width64[i] / sd, color=c, lw=1.4, ls=ls, label=f'{LABELS[i]} + 1 % of its prior range')
    combo = (J[:, others] @ coef) * 0.01 * width64[2] / sd
    a1.plot(CENTRES, combo, color=INK, lw=1.0, ls=':', label=r'best mix of the other eight (mimics $\beta$)')
    a1.plot(CENTRES, d_beam / sd, color=YELLOW, lw=1.4, ls=(0, (6, 1.5, 1.5, 1.5)), label="beam 2.0' $\\to$ 2.1'")
    a1.axhline(0, color=INK2, lw=0.6)
    a1.set_xscale('log'); ell_ticks(a1, (200, 1000, 5000)); a1.set_xlabel(r'$\ell$'); a1.set_ylabel(r'$\Delta D_\ell\,/\,\sigma_\ell$ per 200-$\ell$ band')
    a1.set_title(f"(a) band response in SO noise units; the other eight\nreproduce the β response to "
                 f"{100 * np.sqrt(max(1 - f['beta_response_reproduced_by_other_eight'], 0)):.3f} % (rms)", fontsize=8)
    a1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.22), frameon=False, fontsize=6.3, ncol=2)
    # (b) widths along the Fisher eigenvectors
    k = np.arange(1, 10); e = report['std_along_eigenvectors']
    a2.semilogy(k, e['fisher'], 'o', ms=5, color=INK, label=r'Fisher, no prior: $1/\sqrt{\lambda}$')
    a2.semilogy(k + 0.12, e['fisher_x_64k_prior'], 's', ms=4, color=AQUA, label='Fisher $\\times$ 64k prior')
    a2.semilogy(k - 0.12, e['npe_noisy'], 'D', ms=4, color=BLUE, label='NPE, fitted SO mock')
    a2.axhline(e['flat_prior'], color=INK2, ls='--', lw=0.8, label=r'flat 64k prior ($1/\sqrt{12}$)')
    a2.set_xticks(k); a2.set_xlabel('Fisher eigenvector (tightest first)'); a2.set_ylabel('sd along it [64k prior widths]')
    a2.set_title('(b) which parameter combinations\nthe SO spectrum fixes', fontsize=8)
    a2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.22), frameon=False, fontsize=6.3, ncol=2)
    # (c) beta: the pilot prior versus the 64k prior
    bins = np.linspace(LOW64[2], HIGH64[2], 67)
    a3.axvspan(low_p[2], high_p[2], color='#ecebe8', lw=0, zorder=0, label="pilot's prior range (B12 $\\pm$ 20 %)")
    a3.hist(post['pilot_prior_forecast'][:, 2], bins, density=True, histtype='step', color=ORANGE, lw=1.4, label="Fisher $\\times$ pilot's prior")
    a3.hist(post['prior_64k_forecast'][:, 2], bins, density=True, histtype='step', color=AQUA, lw=1.4, label='Fisher $\\times$ 64k prior')
    a3.hist(npe[:, 2], bins, density=True, histtype='step', color=BLUE, lw=1.4, label='NPE, fitted SO mock')
    a3.axvline(B12[2], color=INK, ls='--', lw=0.9, label='Battaglia+12')
    a3.set_xlim(LOW64[2], HIGH64[2]); a3.set_xlabel(r'$\beta$'); a3.set_ylabel('posterior density')
    a3.set_title("(c) β: the pilot's σ = 0.49 is mostly its prior", fontsize=8)
    a3.legend(loc='upper center', bbox_to_anchor=(0.5, -0.22), frameon=False, fontsize=6.3, ncol=2)
    # (d) all nine: 68 % widths over the 64k prior range
    y = np.arange(9)[::-1]
    rows = (('pilot_prior_forecast', ORANGE, 'o', "Fisher $\\times$ pilot's prior (compared earlier)"),
            ('prior_64k_forecast', AQUA, 's', 'Fisher $\\times$ 64k prior'), ('npe_noisy', BLUE, 'D', 'NPE, fitted SO mock'))
    for j, (key, c, m, lab) in enumerate(rows):
        a4.plot(report['posteriors'][key]['width68_over_64k_prior'], y + 0.27 - 0.18 * j, m, ms=4.5, color=c, ls='none', label=lab)
    a4.plot(2 * conditional / width64, y - 0.27, '|', ms=7, mew=1.4, color=INK, ls='none', label=r'Fisher, other eight fixed: $2/\sqrt{F_{ii}}$')
    a4.axvline(0.68, color=INK2, ls='--', lw=0.8, label='flat prior (0.68)')
    a4.set_xscale('log'); a4.set_yticks(y); a4.set_yticklabels(LABELS); a4.set_xlabel('68 % width / 64k prior range')
    a4.set_title('(d) 68 % widths of all nine parameters', fontsize=8)
    a4.legend(loc='upper center', bbox_to_anchor=(0.5, -0.22), frameon=False, fontsize=6.3, ncol=1)
    for ax in axes.ravel():
        ax.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
    fig.suptitle('Fisher forecast at Battaglia+12 with the current tSZ_64k engine (SO baseline Deproj-0, 40 bands, '
                 f'{len(draws)} noise draws, OAS)', fontsize=9, y=0.995)
    fig.savefig(out / 'fig_fisher_b12.png'); plt.close(fig)


if __name__ == '__main__':
    main()
