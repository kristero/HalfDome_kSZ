"""Independent mathematical checks and provenance checks for the study."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import mpmath as mp
import scipy
from scipy.integrate import quad
from scipy.special import betaln, betainc, ndtr
from analytic_prior import AnalyticPrior, B12
from evidence_gate import evaluate, parameter_hash, amplitude_ceiling
from audit import save


def verify(root):
    prior=AnalyticPrior()
    normalization=prior.normalization()
    support=json.loads((root/'audit/support.json').read_text())
    assert abs(normalization-support['analytic_extended_acceptance'])<.001
    fiducial_density=float(prior.log_prob(B12))
    assert np.isfinite(fiducial_density)
    data=np.load(root/'audit/prior_samples.npz')
    assert prior.contains(data['production_prior']).all()
    production_path=root/'inputs/production_theta.npy'
    production=np.load(production_path)
    assert production.shape==(8192,9) and prior.contains(production).all()
    manifest=json.loads((root/'inputs/active_manifest.json').read_text())
    assert hashlib.sha256(production_path.read_bytes()).hexdigest()==manifest['design_sha256']['theta.npy']
    errors=[]
    integral_records=[]
    mp.mp.dps=70
    for xc in (1e-4,.025,.497,4.,100.,1e4):
        for beta in (2.7001,4.35,50.,500.,2500.):
            upper=np.log(1/xc)
            peak=min(upper,np.log(2.7/(beta-2.7)))
            def logarithm(t):return 2.7*t-beta*np.logaddexp(0,t)
            maximum=logarithm(peak)
            value=quad(lambda t: np.exp(logarithm(t)-maximum),min(upper,peak)-60,
                       upper,epsabs=0,epsrel=1e-11,points=[peak] if peak<upper else [])[0]
            numerical=3*np.log(xc)+maximum+np.log(value)
            analytic=3*np.log(xc)+betaln(2.7,beta-2.7)+np.log(betainc(2.7,beta-2.7,1/(1+xc)))
            # Independent arbitrary-precision special function identifies which
            # implementation is wrong when two double-precision routes disagree.
            reference=float(3*mp.log(xc)+mp.log(mp.betainc(mp.mpf('2.7'),
                mp.mpf(beta)-mp.mpf('2.7'),0,1/(1+mp.mpf(xc)))))
            errors.append(abs(np.expm1(numerical-reference)))
            integral_records.append(dict(xc=xc,beta=beta,
                quadrature_relative_error=float(errors[-1]),
                scipy_special_relative_error=float(abs(np.expm1(analytic-reference)))))
    assert max(errors)<1e-9
    special_error=max(row['scipy_special_relative_error'] for row in integral_records)
    integral_validation=dict(scipy_version=scipy.__version__,
        mpmath_digits=70,cases=integral_records,
        scipy_special_exceeds_1e_minus_9_relative_tolerance=special_error>1e-9,
        interpretation='A special-function accuracy warning is recorded separately from the independent quadrature identity check.')
    save(root/'audit/finite_integral_validation.json',integral_validation)
    save(root/('audit/finite_integral_validation_scipy_'+scipy.__version__+'.json'),integral_validation)
    # Missing convergence evidence must never be converted into a pass.
    covariance=np.load(root/'audit/noise_covariance.npz')['covariance']
    missing=evaluate(B12,{},covariance)
    assert not missing['passed'] and len(missing['reasons'])>=6
    assert amplitude_ceiling(18.1,.4,.1)==9.05
    # Verify the projector inequality in a deliberately rank-deficient example.
    rng=np.random.default_rng(8106)
    a=rng.normal(size=(40,6))@rng.normal(size=(6,9))
    q=np.linalg.svd(a,full_matrices=False)[0][:,:6]
    perturbations=rng.normal(size=(100,40))
    projected=np.linalg.norm(perturbations@q,axis=1)
    full=np.linalg.norm(perturbations,axis=1)
    assert np.all(projected<=full*(1+1e-12))
    comparisons={}
    for directory in (root/'maps').glob('*'):
        for path in directory.glob('*/numerics.toml'):
            # Avoid requiring an external TOML package just to check one digest.
            expected='a6c3d64d5ab83e79b6d9b76cccbaf66a708c3e68adce85b9a0d3f610d4100689'
            assert expected in path.read_text()
    current=root/'maps/Battaglia12/historical1/masked_clean_cl.npy'
    original=root/'inputs/original_B12_masked_clean_cl.npy'
    if current.exists() and original.exists():
        discrepancy=float(np.max(abs(np.load(current)[80:]/np.load(original)[80:]-1)))
        assert discrepancy<1e-10
        comparisons['historical_B12_max_relative_difference']=discrepancy
    save(root/'audit/verification.json',dict(status='passed',analytic_normalization=normalization,
        analytic_log_density=fiducial_density,finite_integral_cases=len(errors),
        frozen_production_rows_inside_analytic_support=len(production),
        finite_integral_max_relative_error=max(errors),missing_evidence_fails_closed=True,
        scipy_special_max_relative_error=special_error,
        fisher_projector_rank=6,fisher_inequality_examples=len(perturbations),
        gaussian_budget_interpretation={str(e):dict(mean_shift_KL_nats=e*e/2,
            total_variation=float(2*ndtr(e/2)-1),
            central_68_interval_coverage_loss_percentage_points=float((ndtr(1)-ndtr(-1)-ndtr(1-e)+ndtr(-1-e))*100))
            for e in (.05,.1,.3)},**comparisons,
        limits='Checks mathematical identities, frozen inputs and completed numerical comparisons. Does not certify untested parameter combinations or nonlinear SBI.'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();verify(args.root)
