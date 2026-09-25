"""Fast checks, with no Julia execution or full-sky maps."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from audit_so_compression_baseline import matched_metrics
from so_moped_local import (
    JULIA_PARAMETERS, posterior_samples, prepare_context, sampling_diagnostics,
    simulator_command, truth_vector,
)
from so_sbi_compression import FIDUCIAL, PARAM_NAMES


class LocalMopedTests(unittest.TestCase):
    def test_simulator_contract(self):
        command = simulator_command(Path.cwd(), "/tmp/test-observation", "/tmp/catalogue.h5",
                                    "/tmp/baseline.txt", FIDUCIAL)
        settings = dict(s.split("=", 1) for s in command[5:])
        for name, value in zip(JULIA_PARAMETERS, FIDUCIAL):
            self.assertEqual(float(settings[name]), value)
        self.assertEqual(settings["nside"], "4096")
        self.assertEqual(settings["so_noise_deprojections"], "0")
        self.assertEqual(settings["mask_seed"], "12345")
        self.assertEqual(settings["noise_seed"], "3000001")
        self.assertEqual(settings["gaussian_beam_fwhm_arcmin"], "2.0")
        self.assertEqual(settings["apply_gaussian_beam"], "true")
        self.assertEqual(settings["reuse_existing_cache"], "false")
        self.assertEqual(settings["sobol_row"], "0")

    def test_reject_changed_mask_and_bad_theta(self):
        args = (Path.cwd(), "/tmp/out", "/tmp/catalogue", "/tmp/noise", FIDUCIAL)
        with self.assertRaises(ValueError):
            simulator_command(*args, mask_seed=2)
        params = dict(zip(PARAM_NAMES, FIDUCIAL))
        bounds = dict(low=FIDUCIAL-1, high=FIDUCIAL+1)
        np.testing.assert_array_equal(truth_vector(params, bounds), FIDUCIAL)
        params["P0"] += 2
        with self.assertRaises(ValueError):
            truth_vector(params, bounds)

    def test_signed_binning_and_saved_projection(self):
        ell = np.arange(80, 120, dtype=np.float32)
        contract = dict(ell_unbinned=ell, bin_ell_min=ell, bin_ell_max=ell,
                        metadata_json=np.asarray(json.dumps({"bin_weighting": "2ell_plus_1"})))
        transform = dict(scale=np.ones(40), mean=np.zeros(40), std=np.ones(40),
                         matrix=np.eye(40)[:, :9], projection_center=np.zeros(40),
                         output_mean=np.zeros(9), output_std=np.ones(9))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"cl.npy"
            cl = -np.ones(40, dtype=np.float32)
            np.save(path, cl)
            x, context = prepare_context(path, contract, transform)
        expected = cl*ell*(ell+1)/(2*np.pi)
        np.testing.assert_allclose(x, expected, rtol=2e-7)
        self.assertTrue(np.all(x < 0))
        np.testing.assert_allclose(context, np.arcsinh(expected[:9]), rtol=2e-7)

    def test_matched_rows(self):
        old, new = [], []
        for i in range(5):
            for j, param in enumerate(PARAM_NAMES):
                truth, mean, std = float(i+j), float(i+j+.2), .3
                old.append(dict(n_train=100, test_index=i, param=param, theta_true=truth,
                    posterior_mean=mean, posterior_std=std, pull=(mean-truth)/std,
                    prior_width=2., error_over_prior_range=(mean-truth)/2.))
                for method in ("bins40", "pca", "moped"):
                    if i == 4 and method == "pca":
                        continue
                    new.append(dict(method=method, test_index=i, param=param, truth=truth,
                        mean=mean, std=std, pull=(mean-truth)/std, normalized_error_prior=(mean-truth)/2.))
        result, _, indices = matched_metrics(pd.DataFrame(old), pd.DataFrame(new), 100)
        self.assertEqual(indices, [0, 1, 2, 3])
        np.testing.assert_allclose(result.pearson_r, 1.)
        damaged = pd.DataFrame(new)
        damaged.loc[0, "truth"] += 1
        with self.assertRaises(AssertionError):
            matched_metrics(pd.DataFrame(old), damaged, 100)

    def test_notebook_code_syntax(self):
        for name in ("moped_gnfw_local.ipynb", "moped_fixed_noise_diagnostic.ipynb"):
            notebook = json.loads(Path(__file__).with_name(name).read_text())
            for i, cell in enumerate(notebook["cells"]):
                if cell["cell_type"] == "code":
                    compile("".join(cell["source"]), f"{name}:cell{i}", "exec")

    def test_zero_acceptance_stops_before_expensive_sampling(self):
        contract = dict(low=np.zeros(9), high=np.ones(9),
                        reference_context=np.zeros((4, 9)))
        draws = [np.full((20, 9), 2.), np.full((20, 9), .5)]
        with tempfile.TemporaryDirectory() as folder:
            with patch("diagnose_so_compression_acceptance.draw_raw", side_effect=draws):
                with patch("run_so_sbi_compression_comparison.bounded_samples") as bounded:
                    with self.assertRaisesRegex(RuntimeError, "0/20 raw draws"):
                        posterior_samples(None, np.full(9, 100.), contract,
                                          count=10, pilot_count=20, diagnostics_dir=folder)
                    bounded.assert_not_called()
            report = json.loads((Path(folder) / "sampling_preflight.json").read_text())
        self.assertEqual(report["accepted_count"], 0)
        self.assertEqual(report["training_control"]["acceptance"], 1.)
        self.assertAlmostEqual(report["zero_count_95pct_upper_bound"], 1-.05**(1/20))

    def test_supported_pilot_does_not_need_control(self):
        contract = dict(low=np.zeros(9), high=np.ones(9),
                        reference_context=np.zeros((4, 9)))
        with patch("diagnose_so_compression_acceptance.draw_raw",
                   return_value=np.full((20, 9), .5)) as draw:
            report = sampling_diagnostics(None, np.zeros(9), contract, pilot_count=20)
        draw.assert_called_once()
        self.assertEqual(report["acceptance"], 1.)
        self.assertNotIn("training_control", report)

    def test_paired_residual_diagnostic(self):
        from diagnose_so_moped_observation import analyze_arrays
        residual = np.arange(12, dtype=float).reshape(4, 3)
        clean = np.ones_like(residual) * 100.
        clean_obs = np.ones(3) * 25.
        obs = clean_obs + residual.mean(0) + 2 * residual.std(0, ddof=1)
        result = analyze_arrays(clean + residual, clean, obs, clean_obs)
        np.testing.assert_allclose(result["new_residual_z"], 2.)


if __name__ == "__main__":
    unittest.main()
