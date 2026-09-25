"""Small numerical checks; no production scientific result is implied."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import run_so_two_param_compression_convergence as pipeline
from audit_so_two_param_combined_spectra import binned_dell
from so_sbi_compression import FIDUCIAL, PARAM_NAMES, save_npz


def fixture(n=180, fixed=True):
    rng = np.random.default_rng(37)
    low, high = np.array([1.81, 3.48]), np.array([34.39, 5.22])
    theta = rng.uniform(low, high, (n, 2)).astype(np.float32)
    full = np.tile(FIDUCIAL, (n, 1)).astype(np.float32)
    full[:, [0, 2]] = theta
    jac = rng.normal(size=(2, 40))
    clean = (3+(theta-pipeline.TRUTH)/(high-low) @ jac)*1e-12
    noisy = clean + rng.normal(size=clean.shape)*.03e-12
    native = np.arange(80, 7980)
    _, obs = binned_dell(np.ones(7980)*1e-19)
    meta = dict(complete=True, statistic="weighted mean of linear D_ell",
        bin_weighting="2ell_plus_1", same_mask_all_rows=True,
        independent_noise_all_rows=not fixed, noise_seed_base=1000000,
        beam_applied_to_signal=True, beam_fwhm_arcmin=2.0)
    bmin = np.array([native[native//200 == j].min() for j in range(40)])
    bmax = np.array([native[native//200 == j].max() for j in range(40)])
    return dict(theta=theta, theta_full=full, x=noisy, x_no_noise=clean,
        prior_low=low, prior_high=high, param_names=np.array(pipeline.TARGETS),
        full_param_names=np.array(PARAM_NAMES), ell_binned=(bmin+bmax)/2,
        ell_unbinned=native, bin_ell_min=bmin, bin_ell_max=bmax,
        sobol_global_row=np.arange(1, n+1),
        noise_seed=np.full(n, 12345) if fixed else np.arange(1000001, 1000001+n),
        mask_seed=np.full(n, 12345), product=np.asarray(pipeline.PRODUCT),
        metadata_json=np.asarray(json.dumps(meta)), obs=obs, obs_theta=pipeline.TRUTH,
        obs_theta_full=FIDUCIAL, obs_source=np.asarray("synthetic Battaglia12 fixture"),
        obs_mask_seed=np.asarray(12345), obs_noise_seed=np.asarray(12345 if fixed else 3000001))


class ConvergenceTests(unittest.TestCase):
    def test_fixed_noise_requires_explicit_opt_in(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"data.npz"
            save_npz(path, **fixture())
            with self.assertRaisesRegex(ValueError, "Independent"):
                pipeline.load_data(path)
            data, _ = pipeline.load_data(path, fixed_noise=True)
            self.assertEqual(data["theta"].shape, (180, 2))

    def test_repeated_independent_seed_rejected(self):
        d = fixture(fixed=False)
        d["noise_seed"][2] = d["noise_seed"][1]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"data.npz"
            save_npz(path, **d)
            with self.assertRaises(AssertionError):
                pipeline.load_data(path)

    def test_wrong_nuisance_parameter_rejected(self):
        d = fixture()
        d["theta_full"][5, 1] += .2
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"data.npz"
            save_npz(path, **d)
            with self.assertRaises(AssertionError):
                pipeline.load_data(path, fixed_noise=True)

    def test_signed_binning(self):
        ell, x = binned_dell(-np.ones(7980))
        self.assertEqual(x.shape, (40,))
        self.assertTrue(np.all(x < 0))
        native = np.arange(80, 200)
        self.assertAlmostEqual(ell[0], np.average(native, weights=2*native+1))
        self.assertAlmostEqual(x[0], np.average(-native*(native+1)/(2*np.pi), weights=2*native+1))

    def test_metric_normalizes_before_averaging(self):
        truth = np.zeros((3, 2))
        mean = np.array([[2., 1.], [4., 3.], [5., 8.]])
        std = np.array([[1., 3.], [2., 1.], [5., 4.]])
        width = np.array([10., 20.])
        stats = pipeline.compute_statistics(truth, mean, std, width)
        self.assertAlmostEqual(stats["aggregate_rmse_prior"], np.sqrt(np.mean((mean/width)**2)))
        self.assertAlmostEqual(stats["aggregate_rmse_std"], np.sqrt(np.mean((mean/std)**2)))
        self.assertNotAlmostEqual(stats["aggregate_rmse_std"], np.sqrt(np.mean(mean**2))/std.mean())

    def test_train_only_compression_and_reuse(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            path, root = folder/"data.npz", folder/"output"
            save_npz(path, **fixture())
            args = pipeline.parser().parse_args(["setup", "--data", str(path), "--root", str(root),
                "--sizes", "128", "--holdout", "20", "--fixed-noise"])
            pipeline.setup(args)
            config = json.loads((root/"experiment.json").read_text())
            data = pipeline.read_npz(root/"shared.npz")
            pipeline.prepare_size(root, 128, config, data)
            fit = pipeline.read_npz(root/"N128/shared.npz")["fit_indices"]
            self.assertFalse(np.intersect1d(fit, data["test_indices"]).size)
            transform = pipeline.read_npz(root/"N128/moped_transform.npz")
            altered = dict(data)
            altered["x"] = data["x"].copy()
            unused = np.setdiff1d(np.arange(len(data["theta"])), fit)
            altered["x"][unused] *= 1e8
            pipeline.prepare_size(folder/"other", 128, config, altered)
            again = pipeline.read_npz(folder/"other/N128/moped_transform.npz")
            for key in transform:
                np.testing.assert_array_equal(transform[key], again[key], err_msg=key)
            pipeline.prepare_size(root, 128, config, data)

    def test_max_size_reserves_holdout(self):
        self.assertEqual(pipeline.parse_sizes("256,512,max", 31768), [256, 512, 31768])
        with self.assertRaises(ValueError):
            pipeline.parse_sizes("32768", 31768)


if __name__ == "__main__":
    unittest.main()
