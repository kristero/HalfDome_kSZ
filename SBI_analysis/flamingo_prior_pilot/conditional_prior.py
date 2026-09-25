"""Optional numerical/outer-energy screen for the proposed prior.

This defines a conditional distribution, not nine independent uniforms.
It does not claim to impose an observational pressure or gas-fraction prior.
The original simulation and SBI bounds are unaffected by this module.
"""
import numpy as np
from scipy.special import betainc

from prior_model import domain_metrics


def admissible(theta, minimum_beta=2.8, maximum_column_tail=.01):
    metrics = domain_metrics(theta)
    beta = metrics["min_beta"]
    xc = metrics["max_xc"]
    good = beta >= minimum_beta
    # Combining maximum core size and minimum beta gives a conservative
    # upper bound, even when their extrema occur at different M,z corners.
    tail = np.ones_like(beta)
    convergent = beta > .7
    tail[convergent] = 1-betainc(.7, beta[convergent]-.7,
                               1e5/(1e5+xc[convergent]))
    return good & (tail <= maximum_column_tail)


def required_beta_amplitude(alpha_m_beta, alpha_z_beta, minimum_beta=2.8):
    """Conditional lower bound on beta0 for the full production grid."""
    mass_factor = np.minimum(.01**np.asarray(alpha_m_beta),
                             (10**1.7)**np.asarray(alpha_m_beta))
    redshift_factor = np.minimum(1.001**np.asarray(alpha_z_beta),
                                 6.**np.asarray(alpha_z_beta))
    return minimum_beta/(mass_factor*redshift_factor)
