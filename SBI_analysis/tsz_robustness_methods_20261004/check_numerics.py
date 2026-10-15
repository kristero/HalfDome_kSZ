#!/usr/bin/env python3
"""Small mathematical gates: Fisher preservation, nuisance suppression, bands and factorial effects."""
import json
import numpy as np
from scipy.linalg import toeplitz

from common import band_operator, noise_covariance, moped_weights, add_sky_variation
from report import interaction_decomposition


def check():
    rng=np.random.default_rng(75)
    a=band_operator()
    np.testing.assert_allclose(a@np.ones(7900),np.ones(158))
    assert np.all((a!=0).sum(0)==1)
    sd=np.linspace(1,2,7900);rho=np.array([1.,.15,.02])
    actual=noise_covariance(sd,rho,a)
    # Exact element from the full Toeplitz formula without allocating a 7900-square matrix.
    p,q=49,50
    x,y=np.flatnonzero(a[p]),np.flatnonzero(a[q])
    lag=np.abs(x[:,None]-y)
    corr=np.where(lag<len(rho),rho[np.minimum(lag,len(rho)-1)],0)
    expected=np.sum(a[p,x,None]*a[q,y]*sd[x,None]*sd[y]*corr)
    np.testing.assert_allclose(actual[p,q],expected,rtol=1e-12,atol=1e-15)
    j=rng.normal(size=(30,9));c=toeplitz(.2**np.arange(30));width=np.linspace(.2,3,9)
    w,audit=moped_weights(j,c,width)
    full=j.T@np.linalg.solve(c,j)
    compressed=(w.T@j).T@np.linalg.solve(w.T@c@w,w.T@j)
    np.testing.assert_allclose(compressed,full,rtol=1e-10,atol=1e-10)
    nuisance=rng.normal(size=30)
    hardened,_=moped_weights(j,c+100*np.outer(nuisance,nuisance),width)
    assert np.linalg.norm(hardened.T@nuisance)<np.linalg.norm(w.T@nuisance)
    simulation=rng.normal(size=(10,9));mask=rng.normal(size=(10,9))
    med=simulation[:,None]+mask[None]
    parts=interaction_decomposition(med.reshape(100,9),np.ones(9))
    np.testing.assert_allclose(parts[2],0,atol=1e-28)
    clean=rng.uniform(1,2,(8,30));noisy=clean+rng.normal(size=clean.shape)
    np.testing.assert_array_equal(add_sky_variation(clean,noisy,np.ones((2,30)),np.zeros(8,dtype=int)),noisy)
    return dict(status='passed',compression=audit,checks=['complete bands','lag covariance',
                'local Fisher identity','nuisance suppression','factorial interaction','identity augmentation'])


if __name__=='__main__':
    print(json.dumps(check(),indent=2))
