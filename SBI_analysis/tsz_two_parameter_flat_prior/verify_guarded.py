"""Verify the repaired support, pressure boundary and inference distribution.

These checks specifically catch the former unfiltered-box error and the small
Y200 exclusion that coarse plots can miss. No maps or cluster jobs are run.
"""
import hashlib
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np
from scipy.integrate import quad

from flat_prior import FlatPrior, HERE
from guardrails import B12, evolved, log_y200


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    prior = FlatPrior()
    root = HERE / "outputs_guarded"
    theta = np.load(root / "theta_P0_beta.npy")
    original = np.load(HERE / "outputs/theta_P0_beta.npy")
    assert prior.contains(theta).all()
    assert not prior.contains(original).all()
    full = np.load(root / "theta_nine_parameters.npy")
    np.testing.assert_array_equal(full,prior.expand(theta))

    beta = np.linspace(10.55, prior.beta_limits[1] - 1e-8, 96)
    boundary = prior.amplitude_bounds(beta)[:,0]
    below = np.column_stack((boundary*(1-1e-7),beta))
    above = np.column_stack((boundary*(1+1e-7),beta))
    assert not prior.contains(below).any(), "Faint/steep excluded corner leaked into support"
    assert prior.contains(above).all(), "Valid points adjacent to the pressure boundary were lost"
    assert np.isneginf(prior.log_prob(below)).all()
    np.testing.assert_allclose(prior.log_prob(above),-np.log(prior.area),atol=1e-14,rtol=0)

    # Exact proposed-vs-accepted prefix comparison preserves every permitted
    # point rather than shrinking P0 to a global lower limit of 1.1163.
    larger, ids = prior.sobol(16384, return_proposal_ids=True)
    np.testing.assert_array_equal(theta,larger[:len(theta)])
    np.testing.assert_array_equal(np.load(root/"proposal_ids.npy"),ids[:len(theta)])
    assert np.any((theta[:,0] < 1.1163) & (theta[:,1] < 10.5))

    # Independent radius quadrature checks the incomplete-beta Y expression.
    # Include the excluded and admitted sides of the narrow corner.
    maximum_error = 0.
    cases = np.array([[1.,3.4],[60.,3.4],[1.,10.8],[1.1,10.8],[1.12,10.874]])
    mass, redshift = prior.guards.size_points
    for point in cases:
        values = prior.expand(point)
        for m,z in zip(mass,redshift):
            def direct(theta9):
                p,x,b = [theta9[k]*(m/1e14)**theta9[k+3]*(1+z)**theta9[k+6] for k in range(3)]
                value,error = quad(lambda r:p*x**.3*r**1.7*(1+r/x)**(-b),
                                   0.,1.,epsabs=0,epsrel=2e-11,limit=300)
                assert error < 1e-8*value
                return value
            direct_ratio = direct(values)/direct(B12)
            analytic = float(np.exp(log_y200(*evolved(values,np.array([m]),np.array([z])))
                                    - log_y200(*evolved(B12,np.array([m]),np.array([z]))))[0])
            maximum_error=max(maximum_error,abs(analytic/direct_ratio-1))
    assert maximum_error < 1e-8

    import torch
    torch.manual_seed(8241)
    distribution=prior.as_torch_distribution(dtype=torch.float64)
    samples=distribution.sample((2048,))
    assert samples.shape==(2048,2)
    assert distribution.support.check(samples).all()
    np.testing.assert_allclose(distribution.log_prob(samples).numpy(),prior.log_prob(samples.numpy()),atol=1e-12)
    invalid=np.vstack((below,[[1,2.8],[60,16],[0,4],[np.nan,4]]))
    assert torch.isneginf(distribution.log_prob(invalid)).all()
    assert not distribution.support.check(torch.tensor(invalid)).any()
    assert distribution.sample().shape==(2,)
    assert distribution.sample((0,)).shape==(0,2)
    # The default float32 wrapper also evaluates guards on the actual returned
    # float32 values, so rounding a near-boundary draw cannot bypass the cut.
    float32_prior=prior.as_torch_distribution()
    samples32=float32_prior.sample((1024,))
    assert prior.contains(samples32.numpy()).all()
    assert torch.isfinite(float32_prior.log_prob(samples32)).all()

    report=dict(status="passed",guarded_design_rows=len(theta),all_guarded_rows_pass=True,
                previous_unfiltered_rows_rejected=int((~prior.contains(original)).sum()),
                boundary_pairs=96,both_sides_of_Y_boundary_checked=True,
                accepted_prefix_checked_through=16384,independent_pressure_integrals=len(cases)*len(mass)*2,
                maximum_integral_relative_error=maximum_error,
                torch_float64_draws=2048,torch_float32_draws=1024,
                torch_invalid_support_check="passed",torch_numpy_log_density_agreement="passed",
                source_sha256={p.name:sha256(p) for p in (HERE/"verify_guarded.py",HERE/"torch_prior.py")})
    (root/"verification_guarded.json").write_text(json.dumps(report,indent=2)+"\n")
    artifacts={p.name:sha256(p) for p in root.iterdir() if p.is_file() and p.name!="artifact_sha256.json"}
    (root/"artifact_sha256.json").write_text(json.dumps(artifacts,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
