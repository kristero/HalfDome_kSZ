"""Numerical tests: exact joint support and independent frozen-covariance projection."""
import sys
from pathlib import Path
import unittest
from unittest.mock import patch
import argparse
import json
import tempfile

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from analyze import FIDUCIAL,joint_fisher_samples,combine
from calibration import EDGES,digest,write_json
from so_fisher_compression import project_likelihood


class TriangularPrior:
    """Uniform 9D box with a coupled triangular restriction in two coordinates."""
    low=FIDUCIAL-1
    high=FIDUCIAL+1
    config={'log_uniform_indices':[]}

    def contains(self,theta):
        u=theta-FIDUCIAL
        return np.all(abs(u)<=1,axis=1)&(u[:,0]+u[:,1]<=0)


class InferenceTests(unittest.TestCase):
    def test_derivative_assembly_and_observation_exclusion(self):
        rng=np.random.default_rng(71)
        prior=TriangularPrior()
        steps=np.full(9,.002)
        jacobian=rng.normal(size=(40,9))
        mean=np.full(40,2.)
        ensemble=mean+rng.normal(size=(32,40))
        tasks=[dict(kind='noise',theta=FIDUCIAL.tolist(),noise_indices=list(range(32))),
               dict(kind='observation',theta=FIDUCIAL.tolist(),noise_indices=[32])]
        for j in range(9):
            for multiple in (-2,-1,1,2):
                theta=FIDUCIAL.copy();theta[j]+=multiple*steps[j]
                tasks.append(dict(kind='derivative',theta=theta.tolist(),parameter=j,multiple=multiple))
        config=dict(tasks=tasks,steps=steps.tolist(),noise_count=32)
        with tempfile.TemporaryDirectory() as name:
            root=Path(name)
            write_json(root/'manifest.json',config)
            (root/'observations').mkdir()
            np.savez(root/'observations/HalfDome.npz',bin_edges=EDGES,masked_clean_dl40=mean)
            for i,task in enumerate(tasks):
                folder=root/'tasks'/f'task{i:03d}';folder.mkdir(parents=True)
                noise=(ensemble if task['kind']=='noise' else np.full((1,40),1000.)
                       if task['kind']=='observation' else np.empty((0,40)))
                np.savez(folder/'spectra.npz',theta=task['theta'],
                    clean=mean+jacobian@(np.array(task['theta'])-FIDUCIAL),noisy=noise)
                operator=dict(mask_pixel_sha256='mask',noise_table_sha256='noise',stable_los_sha256='painter',
                              noise_pixel_hashes=[f'{i}_{k}' for k in range(2*len(noise))])
                write_json(folder/'complete.json',dict(manifest_sha256=digest(root/'manifest.json'),
                    spectra_sha256=digest(folder/'spectra.npz'),operator=operator))
            args=argparse.Namespace(root=root)
            with patch('analyze.setup',return_value=(config,root,prior)):
                combine(args)
            cal=np.load(root/'comparison/calibration.npz')
            np.testing.assert_allclose(cal['jacobian'],jacobian,rtol=1e-8,atol=1e-10)
            np.testing.assert_array_equal(cal['ensemble'],ensemble)
            np.testing.assert_array_equal(cal['observation'],np.full(40,1000.))
            np.testing.assert_allclose(np.diag(cal['covariance']),ensemble.var(0,ddof=1))

    def test_joint_cut_is_applied_to_fisher_null_modes(self):
        prior=TriangularPrior()
        samples,info=joint_fisher_samples(np.zeros((1,9)),np.eye(1),np.zeros(1),
            prior,draws=300000,count=100000,seed=74)
        self.assertTrue(prior.contains(samples).all())
        u=samples-FIDUCIAL
        np.testing.assert_allclose(u.mean(0)[:2],-1/3,atol=.02)
        np.testing.assert_allclose(u.std(0)[:2],np.sqrt(2/9),atol=.02)
        np.testing.assert_allclose(u.mean(0)[2:],0,atol=.02)
        np.testing.assert_allclose(u.std(0)[2:],1/np.sqrt(3),atol=.02)
        self.assertGreater(info['importance_ess'],5000)

    def test_invertible_asinh_tangent_preserves_full_fisher(self):
        rng=np.random.default_rng(3)
        a=rng.normal(size=(40,40));c=a@a.T+np.eye(40)
        j=rng.normal(size=(40,9));residual=rng.normal(size=40)
        scale=rng.uniform(.1,10,40);mean=rng.normal(size=40)
        standard=rng.uniform(.1,5,40)
        tangent=np.diag(1/(np.sqrt(scale**2+mean**2)*standard))
        full=project_likelihood(j,c,residual,np.eye(40))
        transformed=project_likelihood(j,c,residual,tangent)
        np.testing.assert_allclose(full['fisher'],transformed['fisher'],rtol=1e-11,atol=1e-11)
        np.testing.assert_allclose(full['score'],transformed['score'],rtol=1e-11,atol=1e-11)


if __name__=='__main__':
    unittest.main()
