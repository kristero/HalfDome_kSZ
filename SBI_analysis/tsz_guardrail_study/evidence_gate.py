"""Fail-closed, point-specific numerical admissibility for a future dataset.

This is a definition of a gate, not a claim that the exploration rectangle has
passed it. Never replace a failed production row with a fresh parameter draw.
"""
import hashlib
import numpy as np
from analytic_prior import AnalyticPrior, fidelity_distance

REQUIRED_COMPONENTS = ("interpolation", "pixelization", "los_endpoint")


def amplitude_ceiling(reference_p0, measured_error_sigma, budget=.1):
    """Exact clean-signal scaling at fixed other eight parameters and covariance.

    y is linear in P0, hence every clean bandpower difference scales as P0^2.
    This is a numerical amplitude envelope, not an upper bound on thermal gas
    energy. It requires converged, complete component evidence at the reference
    amplitude; an interpolation-only measurement cannot certify pixelization.
    """
    if reference_p0 <= 0 or measured_error_sigma < 0 or budget <= 0:
        raise ValueError('Amplitudes and budgets must be positive; error nonnegative')
    if not np.isfinite(measured_error_sigma):
        return 0.
    if measured_error_sigma == 0:
        return float('inf')
    return float(reference_p0*np.sqrt(budget/measured_error_sigma))


def parameter_hash(theta):
    return hashlib.sha256(np.asarray(theta, dtype='<f8').tobytes()).hexdigest()


def evaluate(theta, evidence, covariance, budget=.1):
    """Sum measured component distances and successive-reference changes.

    Triangle inequality prevents cancellation between independent error
    sources from masquerading as convergence. Each component must use the
    same 40-bin, beam and mask operator. A 4R200 versus 8R200 change is a
    different physical model and must not be silently substituted here.
    """
    result = dict(passed=False, reasons=[], components={}, budget_sigma=float(budget))
    if not AnalyticPrior().contains(theta):
        result['reasons'].append('outside analytic proposal support')
    if evidence.get('parameter_sha256') != parameter_hash(theta):
        result['reasons'].append('parameter evidence absent or mismatched')
    if not evidence.get('operator_sha256') or not evidence.get('reference_protocol_sha256'):
        result['reasons'].append('operator or reference provenance absent')
    expected_covariance = hashlib.sha256(np.asarray(covariance, dtype='<f8').tobytes()).hexdigest()
    if evidence.get('covariance_sha256') != expected_covariance:
        result['reasons'].append('reference covariance mismatch')
    total = 0.
    for name in REQUIRED_COMPONENTS:
        component = evidence.get(name)
        if component is None:
            result['reasons'].append('missing full-observable test: '+name)
            continue
        arrays = [np.asarray(component.get(key, []), float)
            for key in ('candidate', 'reference', 'refined_reference')]
        if any(value.shape != (40,) or not np.isfinite(value).all() for value in arrays):
            result['reasons'].append('invalid spectra: '+name)
            continue
        error = fidelity_distance(arrays[0], arrays[2], covariance)
        convergence = fidelity_distance(arrays[1], arrays[2], covariance)
        result['components'][name] = dict(error_sigma=error, reference_change_sigma=convergence)
        total += error+convergence
    result['total_measured_error_budget_sigma'] = total
    if total > budget:
        result['reasons'].append('declared numerical error allowance exceeded')
    if evidence.get('reference_is_converged') is not True:
        result['reasons'].append('reference convergence not established')
    result['passed'] = not result['reasons']
    result['scope'] = ('Point-specific empirical gate under the stated reference covariance and convergence protocol; '
                       'not an astrophysical posterior or proof over continuous parameter space.')
    return result
