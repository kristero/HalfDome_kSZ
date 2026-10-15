#!/usr/bin/env python3
"""Small read-only preflight: fiducial spectrum, derivative units and exact training-row identities."""
import json
import numpy as np

from common import OLD, FISHER, BEAMFIX, ELL, DL, THETA, PARAMETERS


def audit():
    fisher=np.load(FISHER)
    baseline=np.load(OLD/'results/seed_100/mask_12345/clean_cl.npy')[ELL]*DL
    np.testing.assert_allclose(fisher['d0'],baseline,rtol=1e-6,atol=1e-25)
    assert fisher['J'].shape==(7900,9)
    np.testing.assert_allclose(fisher['J'][:,0],2*baseline/THETA[0],rtol=1e-6,atol=1e-25)
    theta=np.load(BEAMFIX/'inputs/theta.npy')
    with np.load(OLD/'models/linear/features_merlin.npz') as f:
        np.testing.assert_array_equal(theta[f['pool']],f['theta_pool'])
        np.testing.assert_array_equal(theta[f['held']],f['theta_held'])
    return dict(status='passed',parameters=PARAMETERS,training_row_identity='exact',
                fiducial_relative_error=float(np.max(np.abs(fisher['d0']/baseline-1))),
                P0_derivative='2 D_ell / P0, exact at fiducial')


if __name__=='__main__':
    print(json.dumps(audit(),indent=2))
