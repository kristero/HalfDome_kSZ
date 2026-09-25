"""Cluster audit of analytic support, quadrature, thermal scale, and noise metric."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
from scipy.integrate import quad
from scipy.special import betaln, betainc
from scipy.stats import qmc

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "baseline"))
from prior import JointPrior, log_y200
from analytic_prior import AnalyticPrior, B12, evolve


def save(path, data):
    path = Path(path)
    path.parent.mkdir(exist_ok=True, parents=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False)+"\n")


def log_column(x, xc, beta, endpoint=1e5, rtol=1e-10):
    """Positive finite LOS integral in log space, with independent QUADPACK.

    The transform and normalization mirror the mathematical identity in the
    Julia implementation; the adaptive integration algorithm is independent.
    """
    logx, logxc = np.log(x), np.log(xc)
    upper = np.arcsinh(endpoint/x)
    peak = np.clip(logxc+np.log(.7/(beta-.7)), logx, np.log(np.hypot(x, endpoint)))
    def shape(logr):
        return .7*logr+.3*logxc-beta*np.logaddexp(0., logr-logxc)
    normalization = shape(peak)
    def integrand(u):
        return np.exp(shape(logx+np.log(np.cosh(u)))-normalization)
    outcome = quad(integrand, 0, upper, epsabs=0, epsrel=rtol, full_output=1, limit=200)
    value, error, info = outcome[:3]
    if not np.isfinite(value) or value <= 0 or error > 5*rtol*value:
        raise RuntimeError('QUADPACK did not meet the declared tolerance: '+str(outcome[3:]))
    return float(np.log(2)+normalization+np.log(value)), float(error/value), int(info["neval"])


def estimate_covariance(ensemble):
    """Preserve measured variances; regularize only the correlation structure."""
    scale=np.std(ensemble,axis=0,ddof=1)
    if not np.all(scale>0):raise ValueError('A covariance channel has zero variance')
    centered=(ensemble-np.mean(ensemble,axis=0))/scale
    empirical=centered.T@centered/len(ensemble)
    mu=np.trace(empirical)/empirical.shape[0]
    alpha=np.mean(empirical**2)
    denominator=(len(ensemble)+1)*(alpha-mu**2/empirical.shape[0])
    shrinkage=min((alpha+mu**2)/denominator,1.) if denominator>0 else 1.
    regularized=((1-shrinkage)*empirical+shrinkage*mu*np.eye(empirical.shape[0]))
    regularized*=len(ensemble)/(len(ensemble)-1)
    cov=regularized*scale[:,None]*scale[None,:]
    np.testing.assert_allclose(np.diag(cov),np.var(ensemble,axis=0,ddof=1),rtol=1e-12,atol=0)
    return cov,shrinkage,regularized,scale


def covariance(root):
    bundle = np.load(root/'inputs/noise_ensemble.npz')
    sources = json.loads((root/'inputs/noise_sources.json').read_text())
    assert len(bundle['ensemble']) == len(sources) == 64
    expected_edges = np.append(np.arange(80, 7881, 200), 7980)
    ell = np.arange(7980)
    expected_ell = np.array([np.average(ell[a:b], weights=2*ell[a:b]+1)
        for a, b in zip(expected_edges[:-1], expected_edges[1:])])
    for theta in bundle['theta']:
        np.testing.assert_allclose(theta, B12, rtol=1e-6)
    np.testing.assert_allclose(bundle['ell'], expected_ell, rtol=1e-6)
    ensemble = bundle['ensemble'].astype(float)
    cov,shrinkage,normalized,scale=estimate_covariance(ensemble)
    rng=np.random.default_rng(20260919)
    whiteners=[]
    for _ in range(200):
        sample=ensemble[rng.integers(0,len(ensemble),len(ensemble))]
        bootstrap=estimate_covariance(sample)[0]
        whiteners.append(np.linalg.inv(np.linalg.cholesky(bootstrap)))
    np.savez(root/'audit/noise_covariance.npz', covariance=cov, ensemble=ensemble,
             scale=scale, ell=expected_ell,bootstrap_whiteners=np.array(whiteners),
             empirical_covariance=np.cov(ensemble,rowvar=False))
    save(root/'audit/noise_covariance.json', dict(n=64, shrinkage=float(shrinkage),
        condition_number=float(np.linalg.cond(normalized)),
        estimator='OAS-shrunk correlation matrix with unbiased sample variances retained exactly',
        variance_preservation_max_relative_error=float(np.max(abs(np.diag(cov)/np.var(ensemble,axis=0,ddof=1)-1))),
        covariance_bootstrap_samples=200,covariance_bootstrap_seed=20260919,
        scope='Conditional independent SO split-noise at fixed B12 catalogue and signal; excludes cosmic variance, cosmology uncertainty, emulator and model discrepancy.',
        limitation='Reference diagnostic only away from B12: signal-noise covariance depends on the candidate signal.',
        sources=sources))


def support_audit(root):
    old, new = JointPrior(), AnalyticPrior()
    engine = qmc.Sobol(9, scramble=True, seed=20260915)
    unit = engine.random_base2(18)
    old_box = old.from_unit(unit)
    extended = new.low+unit*(new.high-new.low)
    old_accepted = old.contains(old_box)
    analytic_same_box = new.contains(old_box)
    analytic_extended = new.contains(extended)
    old_in_extended = old.contains(extended)
    np.savez_compressed(root/'audit/prior_samples.npz',
        production_prior=old_box[old_accepted], analytic_same_box=old_box[analytic_same_box],
        analytic_extended=extended[analytic_extended], unit=unit)
    # Evaluate every old proxy for all analytic candidates, independently.
    theta = extended[analytic_extended]
    names = ['beta_min', 'beta_max', 'size_min', 'size_max', 'Y200_min', 'Y200_max', 'tail']
    metrics = {name: [] for name in names}
    for start in range(0, len(theta), 512):
        chunk = theta[start:start+512]
        _, xc, beta = evolve(chunk, *old.interpolation_points)
        metrics['beta_min'].extend(beta.min(1)); metrics['beta_max'].extend(beta.max(1))
        metrics['tail'].extend(betainc(beta.min(1)-.7, .7, xc.max(1)/(1e5+xc.max(1))))
        _, xc, beta = evolve(chunk, *old.size_points)
        size = xc/beta/old.reference_size
        metrics['size_min'].extend(size.min(1)); metrics['size_max'].extend(size.max(1))
        ratios = np.exp(log_y200(*evolve(chunk, *old.y_points))-old.reference_log_y)
        metrics['Y200_min'].extend(ratios.min(1)); metrics['Y200_max'].extend(ratios.max(1))
    metrics = {key: np.array(value) for key, value in metrics.items()}
    np.savez_compressed(root/'audit/analytic_proxy_metrics.npz', **metrics)
    failures = dict(beta_lower=metrics['beta_min']<2.8, beta_upper=metrics['beta_max']>50,
        size_lower=metrics['size_min']<.4, size_upper=metrics['size_max']>8,
        Y_lower=metrics['Y200_min']<.003, Y_upper=metrics['Y200_max']>30, LOS_tail=metrics['tail']>.01)
    hull = np.load(root/'inputs/catalogue_logmass_logredshift_hull.npy')
    hull_fail = 0
    for start in range(0, len(extended), 1024):
        t = extended[start:start+1024]
        b = np.exp(np.log(t[:, 2, None])+t[:, 5, None]*hull[None, :, 0]+t[:, 8, None]*hull[None, :, 1])
        hull_fail += int(np.sum(b.min(1)<=2.7))
    save(root/'audit/support.json', dict(proposals=len(unit),
        old_acceptance=float(old_accepted.mean()), analytic_same_box_acceptance=float(analytic_same_box.mean()),
        analytic_extended_acceptance=float(analytic_extended.mean()),
        production_accepted_in_analytic=int(np.sum(new.contains(old_box[old_accepted]))),
        production_total=int(old_accepted.sum()),
        old_support_fraction_of_extended_box=float(old_in_extended.mean()),
        isolated_halo_energy_failures_on_actual_catalogue=hull_fail,
        old_guard_failure_fraction_among_analytic={key: float(value.mean()) for key, value in failures.items()},
        tails={new.config['parameter_order'][i]: dict(accepted_min=float(theta[:, i].min()),
             accepted_max=float(theta[:, i].max()), below_old=int(np.sum(theta[:, i]<old.low[i])),
             above_old=int(np.sum(theta[:, i]>old.high[i]))) for i in range(9)},
        status='Analytic candidate support only; numerical map fidelity not certified'))


def case_audit(root):
    cases = json.loads((root/'inputs/cases.json').read_text())
    pool = np.load(root/'audit/prior_samples.npz')['analytic_extended']
    proxy = np.load(root/'audit/analytic_proxy_metrics.npz')
    stress = {}
    for key in ('beta_max','size_min','size_max','Y200_min','Y200_max','tail'):
        index = np.argmin(proxy[key]) if key.endswith('_min') else np.argmax(proxy[key])
        stress['joint_'+key] = pool[index].tolist()
    rng = np.random.default_rng(20260918)
    for i, index in enumerate(rng.choice(len(pool),32,replace=False)):
        stress['joint_random_'+str(i)] = pool[index].tolist()
    save(root/'inputs/stress_cases.json',stress)
    cases.update(stress)
    old = JointPrior()
    mass, redshift = old.y_points
    records, quadrature = {}, []
    for name, values in cases.items():
        theta = np.array(values)
        p, x, b = evolve(theta, mass, redshift)
        positive_energy = bool(np.all(b>2.7))
        ratios = np.exp(log_y200(p, x, b)-old.reference_log_y) if positive_energy else None
        records[name] = dict(theta=values, production_support=old.contains(theta),
            analytic_support=AnalyticPrior().contains(theta),
            min_catalogue_grid_beta=float(b.min()), max_catalogue_grid_beta=float(b.max()),
            Y200_min=None if ratios is None else float(ratios.min()),
            Y200_max=None if ratios is None else float(ratios.max()))
        for m, z in ((1e13, .1), (1e14, .5), (1e15, 2.), (1e12, 5.),
                     (10**15.7, .001), (1e12, .001), (10**15.7, 5.)):
            p0, xc, beta = [float(v[0]) for v in evolve(theta, np.array([m]), np.array([z]))]
            if beta <= .7:
                continue
            for radius in (.003, .1, 1., 4., 1000.):
                started = time.monotonic()
                try:
                    a, error, neval = log_column(radius, xc, beta)
                    refined, error2, neval2 = log_column(radius, xc, beta, rtol=1e-12)
                    longer, _, _ = log_column(radius, xc, beta, endpoint=2e5)
                except Exception as exc:
                    quadrature.append(dict(case=name,mass=m,z=z,radius=radius,xc=xc,beta=beta,
                                           failed=True,reason=str(exc)))
                    continue
                quadrature.append(dict(case=name, mass=m, z=z, radius=radius, xc=xc, beta=beta,
                    log_column=a, relative_error_estimate=error, evaluations=neval,
                    tightened_relative_change=float(abs(np.expm1(a-refined))),
                    doubled_endpoint_relative_change=float(abs(np.expm1(longer-a))),
                    seconds=time.monotonic()-started, underflows_float64=bool(a<np.log(np.nextafter(0.,1.)))))
        print('Analytic case', name, flush=True)
    save(root/'audit/case_metrics.json', records)
    save(root/'audit/quadrature.json', quadrature)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    args.root.joinpath('audit').mkdir(exist_ok=True)
    covariance(args.root)
    support_audit(args.root)
    case_audit(args.root)
