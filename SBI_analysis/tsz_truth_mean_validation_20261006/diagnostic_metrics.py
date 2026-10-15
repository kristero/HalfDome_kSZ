"""Physical-parameter diagnostics shared by inference and plotting.

Pearson r measures how well posterior means track different truths. It does
not measure unbiased recovery or posterior calibration, so report bias, slope,
RMSE and coverage alongside it. No unbounded samples are clipped to the prior.
"""
import numpy as np

PARAMETERS = ['P0', 'xc', 'beta', 'alpha_m_P0', 'alpha_m_xc',
              'alpha_m_beta', 'alpha_z_P0', 'alpha_z_xc', 'alpha_z_beta']
LABELS = [r'$P_0$', r'$x_c$', r'$\beta$', r'$\alpha_{P,M}$',
          r'$\alpha_{x,M}$', r'$\alpha_{\beta,M}$', r'$\alpha_{P,z}$',
          r'$\alpha_{x,z}$', r'$\alpha_{\beta,z}$']


def pearson_columns(truth, mean):
    """One truth-versus-mean Pearson coefficient per parameter, across cases."""
    a = truth - truth.mean(axis=0)
    b = mean - mean.mean(axis=0)
    denominator = np.sqrt(np.sum(a*a, axis=0) * np.sum(b*b, axis=0))
    if np.any(denominator <= 0):
        raise ValueError('Pearson r needs varying truths and varying estimates')
    return np.sum(a*b, axis=0) / denominator


def summarize(truth, mean, quantiles, low, high, mean_mcse, half_mean):
    """Errors are divided by the training prior width, separately for each theta.

    Quantiles are ordered 2.5%, 16%, 50%, 84%, 97.5%. Regression is in units
    t=(theta-low)/(high-low), so slope=1 and intercept=0 describe identity.
    Posterior means can legitimately shrink toward a prior for weakly measured
    parameters. A slope below one alone is not proof of SBI miscalibration.
    """
    width = high-low
    t = (truth-low)/width
    m = (mean-low)/width
    residual = m-t
    slope = np.sum((t-t.mean(0))*(m-m.mean(0)), axis=0)/np.sum((t-t.mean(0))**2, axis=0)
    q025, q16, _, q84, q975 = quantiles
    coverage68 = np.mean((truth >= q16) & (truth <= q84), axis=0)
    coverage95 = np.mean((truth >= q025) & (truth <= q975), axis=0)
    return dict(
        pearson=pearson_columns(truth, mean),
        normalized_rmse=np.sqrt(np.mean(residual**2, axis=0)),
        normalized_bias=residual.mean(0),
        regression_slope=slope,
        normalized_regression_intercept=m.mean(0)-slope*t.mean(0),
        coverage68=coverage68, coverage95=coverage95,
        coverage68_row_sem=np.sqrt(coverage68*(1-coverage68)/len(truth)),
        coverage95_row_sem=np.sqrt(coverage95*(1-coverage95)/len(truth)),
        median_width68_prior=np.median((q84-q16)/width, axis=0),
        rms_mean_mcse_prior=np.sqrt(np.mean((mean_mcse/width)**2, axis=0)),
        maximum_mean_mcse_prior=np.max(mean_mcse/width, axis=0),
        half_sample_pearson_difference=np.abs(
            pearson_columns(truth, half_mean[0])-pearson_columns(truth, half_mean[1])),
    )


def resample_pearson(truth, means, n_bootstrap=1000, seed=20261006):
    """Paired row bootstrap: all models use the same resampled test cases.

    Intervals describe variation over these reserved parameter/noise cases.
    They are not cosmic-variance errors, training-seed errors, or an assertion
    that low-discrepancy parameter-design rows are independent random draws.
    """
    rng = np.random.default_rng(seed)
    names = list(means)
    result = {name: np.empty((n_bootstrap, truth.shape[1])) for name in names}
    for i in range(n_bootstrap):
        ids = rng.integers(0, len(truth), size=len(truth))
        for name in names:
            result[name][i] = pearson_columns(truth[ids], means[name][ids])
    return result


def self_check():
    """Independent numerical gates for the diagnostics, not scientific validation."""
    rng = np.random.default_rng(941)
    truth = rng.uniform(.1, .9, (400, 9))
    mean = .8*truth + .15
    low = np.zeros(9)
    high = np.ones(9)
    q = np.stack([mean-.3, mean-.15, mean, mean+.15, mean+.3])
    record = summarize(truth, mean, q, low, high, np.zeros_like(truth),
                       np.stack([mean, mean]))
    np.testing.assert_allclose(record['pearson'], 1, atol=1e-12)
    np.testing.assert_allclose(record['regression_slope'], .8, atol=1e-12)
    np.testing.assert_allclose(record['normalized_regression_intercept'], .15, atol=1e-12)
    assert np.all(record['normalized_rmse'] > 0)
    np.testing.assert_allclose(record['coverage68'], 1)
    np.testing.assert_allclose(record['half_sample_pearson_difference'], 0)
    # Changing parameter units must leave normalized metrics and Pearson intact.
    scale = np.arange(1., 10.)
    shift = -np.arange(9.)
    changed = summarize(truth*scale+shift, mean*scale+shift,
                        q*scale+shift, shift, scale+shift,
                        np.zeros_like(truth), np.stack([mean, mean])*scale+shift)
    for name in ['pearson', 'normalized_rmse', 'normalized_bias', 'regression_slope']:
        np.testing.assert_allclose(record[name], changed[name], atol=1e-12)
    try:
        pearson_columns(np.ones_like(truth), mean)
    except ValueError:
        pass
    else:
        raise AssertionError('Constant Battaglia truths must not give a Pearson coefficient')
    print('Diagnostic numerical gates passed; no trained models were evaluated.')


if __name__ == '__main__':
    self_check()
