"""Fast numerical and plotting checks; these do not train a production NPE."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

import run_so_sbi_compression_comparison as runner
from so_sbi_compression import (
    FIDUCIAL, METHODS, PARAM_NAMES, asinh_coordinates, check_pair, finish_transform,
    fit_asinh, fit_moped, fit_pca, load_dataset, metrics_from_samples, moped_basis,
    pearson_columns, project, save_npz, split_rows, write_json,
)


def synthetic_data(n=1200):
    rng = np.random.default_rng(42)
    low, high = FIDUCIAL - 1, FIDUCIAL + 1
    theta = rng.uniform(low, high, (n, 9))
    jacobian = rng.normal(size=(40, 9))
    clean = (theta - FIDUCIAL) @ jacobian.T * 1e-12
    noisy = clean + rng.normal(scale=.3e-12, size=clean.shape)
    metadata = json.dumps(dict(statistic="weighted mean of linear D_ell; x equals binned D_ell",
                               bin_weighting="2ell+1"))
    common = dict(theta=theta.astype(np.float32), prior_low=low, prior_high=high,
                  param_names=np.asarray(PARAM_NAMES), sobol_global_row=rng.permutation(n) + 1,
                  ell_binned=np.arange(40), bin_ell_min=np.arange(40)*200 + 80,
                  bin_ell_max=np.arange(40)*200 + 279, metadata_json=np.asarray(metadata))
    return (dict(common, x=noisy, product=np.asarray("masked_baseline_noise_cross_deproj0")),
            dict(common, x=clean, product=np.asarray("masked_no_noise")))


class CompressionTests(unittest.TestCase):
    def test_best_validation_weights_restored_at_epoch_cap(self):
        try:
            import torch
        except ImportError:
            self.skipTest("Checkpoint restoration check requires Torch, but not SBI")
        estimator = torch.nn.Linear(2, 1)
        best = {key: value.detach().clone() for key, value in estimator.state_dict().items()}
        inference = SimpleNamespace(_best_model_state_dict=best)
        with torch.no_grad():
            estimator.weight.add_(10)
        result = runner.restore_best_validation_weights(estimator, inference)
        self.assertTrue(result["returned_weights_differed_from_best"])
        for key, value in estimator.state_dict().items():
            self.assertTrue(torch.equal(value, best[key]))
        result = runner.restore_best_validation_weights(estimator, inference)
        self.assertFalse(result["returned_weights_differed_from_best"])
        with self.assertRaisesRegex(RuntimeError, "best-validation state"):
            runner.restore_best_validation_weights(estimator, SimpleNamespace())
        with self.assertRaises(RuntimeError):
            runner.restore_best_validation_weights(estimator, SimpleNamespace(
                _best_model_state_dict={"wrong_layer": torch.ones(1)}))

    def test_raw_acceptance_requires_all_parameters_inside(self):
        from diagnose_so_compression_acceptance import summarize_raw
        raw = np.full((4, 9), .5)
        raw[1, 0] = -.1
        raw[2, 1] = 1.1
        raw[3, [0, 1]] = [-.1, 1.1]
        result = summarize_raw(raw, np.zeros(9), np.ones(9))
        self.assertEqual(result["accepted_count"], 1)
        self.assertEqual(result["acceptance"], .25)
        self.assertEqual(result["per_parameter"][0]["below"], .5)
        self.assertEqual(result["per_parameter"][1]["above"], .5)
        raw[0, 0] = np.nan
        with self.assertRaises(ValueError):
            summarize_raw(raw, np.zeros(9), np.ones(9))

    def test_split_preserves_prior_and_original_holdout(self):
        noisy, _ = synthetic_data()
        noisy["theta"][[2, 1199], 0] = noisy["prior_high"][0] + 1
        s = split_rows(noisy["theta"], noisy["prior_low"], noisy["prior_high"], 10, .1, 3)
        np.testing.assert_array_equal(s["excluded"], [2, 1199])
        np.testing.assert_array_equal(s["test"], np.arange(1190, 1199))
        self.assertEqual(len(set(s["fit"]) & set(s["validation"])), 0)
        self.assertEqual(len(set(s["test"]) & set(s["pool"])), 0)
        self.assertNotIn(2, s["pool"])

    def test_pair_checks_identity_not_just_shape(self):
        noisy, clean = synthetic_data()
        check_pair(noisy, clean)
        clean["sobol_global_row"] = clean["sobol_global_row"][::-1]
        with self.assertRaisesRegex(ValueError, "sobol_global_row"):
            check_pair(noisy, clean)

    def test_lower_median_and_signed_asinh(self):
        x = np.array([[-1., -2.], [3., 4.], [5., 6.], [7., 8.]])
        base = fit_asinh(x)
        np.testing.assert_array_equal(base["scale"], [3, 4])
        y = asinh_coordinates(x, base)
        np.testing.assert_allclose(y.mean(axis=0), 0, atol=1e-15)
        np.testing.assert_allclose(y.std(axis=0, ddof=1), 1)

    def test_pca_full_rank_reconstruction_and_transform(self):
        noisy, _ = synthetic_data()
        fit = noisy["x"][:1000]
        base = fit_asinh(fit)
        values = asinh_coordinates(fit, base)
        pca = fit_pca(values, 40)
        recovered = ((values - pca["center"]) @ pca["matrix"]) @ pca["matrix"].T + pca["center"]
        np.testing.assert_allclose(recovered, values, atol=1e-13)
        transform = finish_transform(base, values, pca["matrix"], pca["center"])
        projected = project(fit, transform)
        np.testing.assert_allclose(projected.std(axis=0, ddof=1), 1, rtol=1e-6)
        np.testing.assert_allclose(project(fit[:1], transform), projected[:1], atol=1e-6)

    def test_moped_fisher_and_rank_deficiency(self):
        rng = np.random.default_rng(4)
        a = rng.normal(size=(40, 40))
        covariance = a @ a.T + np.eye(40)
        derivatives = rng.normal(size=(40, 9))
        derivatives[:, -1] = 2*derivatives[:, 0]
        m = moped_basis(derivatives, covariance, 1e-8)
        self.assertEqual(m["matrix"].shape, (40, 8))
        np.testing.assert_allclose(m["compressed_covariance"], np.eye(8), atol=1e-13)
        np.testing.assert_allclose(m["fisher"], m["compressed_fisher"], atol=1e-12)

    def test_local_moped_fit_is_finite(self):
        noisy, clean = synthetic_data()
        base = fit_asinh(noisy["x"][:1000])
        m = fit_moped(asinh_coordinates(noisy["x"][:1000], base),
                      asinh_coordinates(clean["x"][:1000], base), noisy["theta"][:1000],
                      noisy["prior_high"] - noisy["prior_low"], 800, .05, 1e-6)
        self.assertLess(m["local_indices"].max(), 1000)
        self.assertTrue(np.isfinite(m["matrix"]).all())
        np.testing.assert_allclose(m["compressed_covariance"], np.eye(9), atol=1e-12)

    def test_metric_normalizes_before_averaging(self):
        samples = np.array([[1., 20.], [3., 60.], [5., 100.]])
        metric = metrics_from_samples(samples, np.array([1., 20.]),
                                      np.array([0., 0.]), np.array([10., 100.]))
        np.testing.assert_allclose(metric["normalized_error_prior"], [.2, .4])
        np.testing.assert_allclose(metric["pull"], [1., 1.])
        self.assertAlmostEqual(np.sqrt(np.mean(metric["normalized_error_prior"]**2)), np.sqrt(.1))
        np.testing.assert_allclose(pearson_columns(samples, samples), [1, 1])

    def test_bounded_sampling_old_and_new_interfaces(self):
        try:
            import torch
        except ImportError:
            self.skipTest("Sampling adapter check requires Torch, but not SBI")

        class OldFlow:
            def sample(self, num_samples, context):
                return torch.ones((1, num_samples, 3))

        class NewFlow:
            def sample(self, sample_shape, condition):
                return torch.ones((*sample_shape, 1, 3))

        class OutsideFlow:
            def sample(self, num_samples, context):
                return torch.full((1, num_samples, 3), 100.)

        config = dict(posterior_samples=10, max_proposals=23, sampling_seconds=30)
        for model in (OldFlow(), NewFlow()):
            samples, proposals, acceptance = runner.bounded_samples(
                model, np.zeros(40), np.zeros(3), np.full(3, 2), config)
            self.assertEqual(samples.shape, (10, 3))
            self.assertEqual(proposals, 23)
            self.assertEqual(acceptance, 1)
        with self.assertRaisesRegex(RuntimeError, "accepted=0, proposals=23"):
            runner.bounded_samples(OutsideFlow(), np.zeros(40), np.zeros(3), np.full(3, 2), config)

    def test_prepare_and_plot_without_sbi(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            noisy, clean = synthetic_data()
            save_npz(root / "noisy.npz", **noisy)
            save_npz(root / "clean.npz", **clean)
            load_dataset(root / "noisy.npz")
            import argparse
            args = argparse.Namespace(
                output_root=root / "experiment", dataset=root / "noisy.npz", clean_dataset=root / "clean.npz",
                holdout_last_n=10, pca_components=9, moped_local_n=800, covariance_shrinkage=.05,
                moped_rcond=1e-6, hidden_features=64, num_transforms=6, training_batch_size=1024,
                stop_after_epochs=20, max_num_epochs=200, validation_fraction=.1, posterior_samples=100,
                max_proposals=10000, sampling_seconds=30., seed=42, skip_corner=True,
            )
            runner.prepare(args)
            runner.prepare(args)  # Unchanged preparation is reusable.
            config = json.loads((args.output_root / "experiment.json").read_text())
            shared = runner.load_npz(args.output_root / "shared.npz")
            rng = np.random.default_rng(33)
            for method in METHODS:
                run = args.output_root / method
                write_json(run / "training_complete.json", dict(experiment_id=config["experiment_id"],
                                                                training_seconds=1))
                for idx in shared["test_indices"]:
                    center = .9*shared["theta"][idx] + .1*FIDUCIAL
                    samples = rng.uniform(center - .05, center + .05, (100, 9))
                    save_npz(run / f"evaluation/profiles/row{idx}.npz", samples=samples,
                             truth=shared["theta"][idx], method=np.asarray(method),
                             experiment_id=np.asarray(config["experiment_id"]))
                write_json(run / "evaluation/evaluation_complete.json", dict(experiment_id=config["experiment_id"]))
            runner.summarize(args, config, shared)
            self.assertTrue((args.output_root / "summary/summary_complete.json").is_file())
            self.assertEqual(len(list((args.output_root / "summary").glob("*.png"))), 11)
            args.pca_components = 8
            with self.assertRaisesRegex(ValueError, "differs"):
                runner.prepare(args)


if __name__ == "__main__":
    unittest.main()
