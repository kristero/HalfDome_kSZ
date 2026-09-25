"""Contract and inference checks using explicitly synthetic data, never sky validation."""
import argparse
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
from scipy.stats import qmc,truncnorm

from export_so_moped_bundle import sha256
from generate_so_fisher_variations import BATTAGLIA12,PARAMETER_NAMES
from prepare_so_nine_fisher_calibration import calibration_design,combine
from run_so_nine_independent_comparison import prepare,validate_calibration,features
from so_nine_fisher import (conditional_covariance,fisher_matrix,hard_prior_posterior,
    load_independent_dataset,moped_weights,read_npz,simulation_signature)
from so_sbi_compression import save_npz,write_json
from so_fisher_compression import fit_linear_pca,project_likelihood


def fixture(root,n=256):
    """A cheap full contract for exercising orchestration, not physical spectra."""
    rng=np.random.default_rng(73)
    config=json.loads((Path(__file__).parent/"so_32k_independent_noise/config.json").read_text())
    config["n_rows"]=n
    low,high=np.array(config["prior_low"]),np.array(config["prior_high"])
    theta=low+qmc.Sobol(d=9,scramble=True,seed=72).random_base2(int(np.log2(n)))*(high-low)
    ell=np.arange(80,7980)
    groups=ell//200
    bins={"ell_unbinned":ell,"ell_binned":np.array([np.average(ell[groups==g],weights=2*ell[groups==g]+1) for g in np.unique(groups)]),
        "bin_ell_min":np.array([ell[groups==g].min() for g in np.unique(groups)]),
        "bin_ell_max":np.array([ell[groups==g].max() for g in np.unique(groups)])}
    jacobian=rng.normal(0,.3,(40,9))/(high-low)
    clean=2+(theta-BATTAGLIA12)@jacobian.T
    x=clean+rng.normal(0,.3,clean.shape)
    generation=dict(config=config,source_sha256={"synthetic.jl":"synthetic"},
        catalogue=dict(size=1),experiment_id="synthetic_contract_only")
    metadata=dict(complete=True,mode="nine_param",product="masked_baseline_noise_cross_deproj0",
        independent_noise_all_rows=True,same_mask_all_rows=True,noise_table_convention="N_ell per split",
        bin_weighting="2ell_plus_1",beam_fwhm_arcmin=2.0,experiment_id=generation["experiment_id"],synthetic_test=True)
    roots=config["noise_seed_bases"]["nine_param"]+65536*np.arange(n,dtype=np.int64)
    data=dict(theta=theta,theta_full=theta,param_names=np.array(PARAMETER_NAMES),full_param_names=np.array(PARAMETER_NAMES),
        prior_low=low,prior_high=high,x=x.astype(np.float32),x_no_noise=clean.astype(np.float32),
        sobol_global_row=np.arange(1,n+1),mask_seed=np.full(n,12345),noise_seed=roots,
        noise_split_seeds=np.column_stack((roots+10101,roots+10102)),metadata_json=np.asarray(json.dumps(metadata)),**bins)
    dataset=root/"dataset.npz"
    save_npz(dataset,**data)
    write_json(root/"complete.json",dict(dataset_sha256=sha256(dataset),metadata=metadata))
    write_json(root/"generation.json",generation)
    cal_seeds=2**50+65536*np.arange(65,dtype=np.int64)
    cal=dict(param_names=np.array(PARAMETER_NAMES),prior_low=low,prior_high=high,fiducial=BATTAGLIA12,
        fiducial_dell=np.full(40,2.),derivatives=jacobian,derivative_small=jacobian,derivative_large=jacobian,
        noise_ensemble=2+rng.normal(0,.3,(64,40)),observation=2+rng.normal(0,.3,40),
        covariance_split_seeds=np.column_stack((cal_seeds[:64]+1,cal_seeds[:64]+2)),
        observation_split_seeds=cal_seeds[64]+np.array([1,2]),source_signature=np.asarray(simulation_signature(generation)),
        dataset_sha256=np.asarray(sha256(dataset)),**bins)
    save_npz(root/"calibration.npz",**cal)
    write_json(root/"calibration_complete.json",dict(complete=True,calibration_sha256=sha256(root/"calibration.npz"),
        observation_excluded_from_covariance=True))
    return argparse.Namespace(dataset=dataset,generation_manifest=root/"generation.json",calibration=root/"calibration.npz",
        expected_rows=n,root=root/"analysis",holdout=8,sizes="128",seed=42,hidden_features=12,num_transforms=2,
        batch_size=32,patience=2,max_epochs=2,posterior_samples=100,fiducial_samples=200,max_proposals=100000)


class FisherTests(unittest.TestCase):
    def test_hard_prior_matches_analytic_truncation(self):
        values,info=hard_prior_posterior(np.array([[1.]]),np.array([[.64]]),np.array([.7]),
            np.array([-.5]),np.array([1.1]),np.array([0.]),draws=150000,count=50000)
        mean,std=truncnorm.stats((-.5-.7)/.8,(1.1-.7)/.8,loc=.7,scale=.8,moments="mv")
        self.assertAlmostEqual(values.mean(),mean,delta=.006)
        self.assertAlmostEqual(values.std(),np.sqrt(std),delta=.006)
        self.assertGreater(info["importance_ess"],10000)

    def test_null_mode_remains_uniform(self):
        values,_=hard_prior_posterior(np.array([[1.,0.]]),np.eye(1),np.zeros(1),
            np.array([-3.,-5.]),np.array([3.,7.]),np.zeros(2),draws=200000,count=100000)
        self.assertAlmostEqual(values[:,1].mean(),1.,delta=.05)
        self.assertAlmostEqual(values[:,1].std(),12/np.sqrt(12),delta=.04)

    def test_moped_preserves_information_with_correlated_noise(self):
        rng=np.random.default_rng(19)
        a=rng.normal(size=(40,40))
        covariance=a@a.T+np.eye(40)
        jacobian=rng.normal(size=(40,9))
        weights,_,error=moped_weights(jacobian,covariance)
        np.testing.assert_allclose(weights.T@covariance@weights,np.eye(9),atol=1e-12)
        np.testing.assert_allclose(fisher_matrix(jacobian,covariance),
            fisher_matrix(weights.T@jacobian,weights.T@covariance@weights),rtol=1e-11,atol=1e-12)
        self.assertLess(error,1e-12)

    def test_oas_preserves_unbiased_variances(self):
        values=np.random.default_rng(8).normal(size=(64,40))*np.linspace(1,4,40)
        covariance,_=conditional_covariance(values)
        np.testing.assert_allclose(np.diag(covariance),values.var(axis=0,ddof=1))

    def test_projected_likelihood_ratios_and_output_scaling(self):
        rng=np.random.default_rng(91)
        a=rng.normal(size=(40,40))
        covariance=a@a.T+np.eye(40)
        jacobian=rng.normal(size=(40,9))
        residual=rng.normal(size=40)
        weights,_,_=moped_weights(jacobian,covariance)
        # Deliberately rescale each compressed coordinate by a different factor.
        weights=weights/np.geomspace(.01,100,9)
        full=project_likelihood(jacobian,covariance,residual,np.eye(40))
        compressed=project_likelihood(jacobian,covariance,residual,weights)
        for delta in rng.normal(size=(20,9)):
            log_full=delta@full['score']-.5*delta@full['fisher']@delta
            log_compressed=delta@compressed['score']-.5*delta@compressed['fisher']@delta
            self.assertAlmostEqual(log_full,log_compressed,places=9)

    def test_pca_variance_is_not_noise_or_parameter_information(self):
        # A large training-scatter direction carries no parameter derivative;
        # discarding the low-variance informative direction loses all Fisher.
        covariance=np.diag([2.,.01])
        jacobian=np.array([[0.],[1.]])
        projected=project_likelihood(jacobian,covariance,np.zeros(2),np.array([[1.],[0.]]))
        self.assertEqual(projected['fisher'].item(),0.)
        self.assertEqual(fisher_matrix(jacobian,covariance).item(),100.)
        self.assertAlmostEqual(projected['relative_fisher_change'].item(),1.)

    def test_pca_uses_only_given_training_rows(self):
        rng=np.random.default_rng(73)
        training=rng.normal(size=(100,40))*np.geomspace(1,100,40)
        transform=fit_linear_pca(training)
        projected=features(training,transform)
        np.testing.assert_allclose(projected.mean(axis=0),0.,atol=1e-7)
        np.testing.assert_allclose(projected.std(axis=0,ddof=1),1.,atol=3e-7)
        np.testing.assert_allclose(transform['input_scale'],training.std(axis=0,ddof=1))

    def test_calibration_has_disjoint_roles(self):
        cfg=json.loads((Path(__file__).parent/"so_32k_independent_noise/config.json").read_text())
        theta,rows=calibration_design(np.array(cfg["prior_low"]),np.array(cfg["prior_high"]),256)
        self.assertEqual(theta.shape,(294,9))
        self.assertEqual(sum(r["kind"]=="covariance" for r in rows),256)
        self.assertEqual(sum(r["kind"]=="observation" for r in rows),1)
        np.testing.assert_allclose(theta[-1],BATTAGLIA12)


class ContractTests(unittest.TestCase):
    def test_calibration_combines_derivatives_and_excludes_observation(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name)
            args=fixture(root)
            source=read_npz(args.calibration)
            theta,rows=calibration_design(source["prior_low"],source["prior_high"],64)
            design=dict(theta=theta,split_seeds=np.arange(2*len(theta)).reshape(-1,2)+2**50,
                **{k:source[k] for k in ("prior_low","prior_high","ell_unbinned","ell_binned","bin_ell_min","bin_ell_max")})
            manifest=dict(rows=rows,source_signature=str(source["source_signature"]),dataset_sha256=str(source["dataset_sha256"]))
            write_json(root/"manifest.json",manifest)
            identity=sha256(root/"manifest.json")
            rng=np.random.default_rng(872)
            for i,t in enumerate(theta):
                folder=root/"rows"/f"row{i+1:04d}"
                folder.mkdir(parents=True)
                clean=2+source["derivatives"]@(t-BATTAGLIA12)
                save_npz(folder/"spectra.npz",clean=clean,noisy=clean+rng.normal(0,.3,40))
                write_json(folder/"complete.json",dict(manifest_sha256=identity,theta=t.tolist(),
                    split_seeds=design["split_seeds"][i].tolist(),spectra_sha256=sha256(folder/"spectra.npz")))
            with patch("prepare_so_nine_fisher_calibration.checked_setup",return_value=(manifest,design,None)):
                combine(argparse.Namespace(root=root))
            cal=read_npz(root/"calibration.npz")
            np.testing.assert_allclose(cal["derivatives"],source["derivatives"],rtol=1e-10,atol=1e-12)
            self.assertEqual(cal["noise_ensemble"].shape,(64,40))
            self.assertFalse(np.intersect1d(cal["covariance_split_seeds"],cal["observation_split_seeds"]).size)

    def test_prepare_matches_splits_and_training_only_scaling(self):
        with tempfile.TemporaryDirectory() as name:
            args=fixture(Path(name))
            prepare(args)
            run=args.root/"N128"
            shared=read_npz(run/"shared.npz")
            self.assertFalse(np.intersect1d(shared["pool_indices"],shared["test_indices"]).size)
            self.assertFalse(np.intersect1d(shared["fit_indices"],shared["validation_indices"]).size)
            data=read_npz(args.root/"data.npz")
            for method in ("bins40","pca","moped"):
                transform=read_npz(run/f"{method}_transform.npz")
                output=features(data["x"],transform)
                np.testing.assert_allclose(output[shared["fit_indices"]].mean(axis=0),0,atol=3e-6)
                np.testing.assert_allclose(output[shared["fit_indices"]].std(axis=0,ddof=1),1,atol=3e-6)
                likelihood=read_npz(run/f"{method}_likelihood.npz")
                covariance=read_npz(args.root/"calibration.npz")["covariance"]
                np.testing.assert_allclose(likelihood["covariance"],
                    likelihood["matrix"].T@covariance@likelihood["matrix"])
            np.testing.assert_array_equal(shared["low"],data["prior_low"])
            with self.assertRaisesRegex(ValueError,"fresh analysis root"):
                prepare(args)

    def test_rejects_missing_rows_and_reused_noise(self):
        with tempfile.TemporaryDirectory() as name:
            args=fixture(Path(name))
            with self.assertRaisesRegex(ValueError,"32768"):
                load_independent_dataset(args.dataset,args.generation_manifest)
            data=read_npz(args.dataset)
            data["noise_split_seeds"][1,0]=data["noise_split_seeds"][0,1]
            save_npz(args.dataset,**data)
            complete=json.loads(args.dataset.with_name("complete.json").read_text())
            complete["dataset_sha256"]=sha256(args.dataset)
            write_json(args.dataset.with_name("complete.json"),complete)
            with self.assertRaises(AssertionError):
                load_independent_dataset(args.dataset,args.generation_manifest,256)

    def test_rejects_observation_covariance_leakage(self):
        with tempfile.TemporaryDirectory() as name:
            args=fixture(Path(name))
            data,generation=load_independent_dataset(args.dataset,args.generation_manifest,256)
            cal=read_npz(args.calibration)
            cal["observation_split_seeds"]=cal["covariance_split_seeds"][0]
            save_npz(args.calibration,**cal)
            write_json(args.calibration.with_name("calibration_complete.json"),dict(complete=True,
                calibration_sha256=sha256(args.calibration),observation_excluded_from_covariance=True))
            with self.assertRaisesRegex(ValueError,"share a noise split"):
                validate_calibration(args.calibration,data,generation,args.dataset)


if __name__=="__main__":
    unittest.main()
