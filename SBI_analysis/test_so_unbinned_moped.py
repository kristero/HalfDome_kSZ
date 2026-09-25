"""Numerical checks for raw-spectrum alignment and streamed MOPED fitting."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from prepare_validate_battaglia12_sbi_observation import make_cl_to_dl_matrix
from so_sbi_compression import FIDUCIAL, PARAM_NAMES, fit_asinh, project
from so_unbinned_moped import (dell_values, fit_scaling, fit_unbinned, open_raw,
                               project_observation, validate_rebin, write_projected)


class UnbinnedTests(unittest.TestCase):
    def test_sampling_budget_resume_preserves_scientific_configuration(self):
        import run_so_unbinned_moped_comparison as runner
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "moped40").mkdir()
            (root / "moped40/stage_failures.json").write_text(json.dumps(dict(errors=["previous attempt"])))
            (root / "prepare_complete.json").touch()
            original = dict(experiment_id="fixture", max_proposals=200000,
                            sampling_seconds=120., posterior_samples=2000)
            (root / "experiment.json").write_text(json.dumps(original))
            np.savez(root / "shared.npz", theta=np.zeros((1, 9)))
            command = ["runner", "evaluate", "--root", str(root), "--method", "moped40"]
            with patch.object(runner, "evaluate") as evaluate, patch.object(runner, "fixed_observation"):
                with patch("sys.argv", command + ["--max-proposals", "2000000", "--sampling-seconds", "300"]):
                    runner.main()
                with patch("sys.argv", command):
                    runner.main()
                for call in evaluate.call_args_list:
                    effective = call.args[2]
                    self.assertEqual(effective["max_proposals"], 2000000)
                    self.assertEqual(effective["posterior_samples"], 2000)
                    self.assertEqual(effective["experiment_id"], original["experiment_id"])
                with patch("sys.argv", command + ["--max-proposals", "100"]):
                    with self.assertRaisesRegex(ValueError, "reference limits"):
                        runner.main()
            self.assertEqual(json.loads((root / "experiment.json").read_text()), original)
            self.assertEqual(json.loads((root / "moped40/stage_failures.json").read_text())["errors"], [])

    def test_block_statistics_match_dense_and_ignore_heldout(self):
        rng = np.random.default_rng(71)
        raw = rng.normal(size=(90, 21)).astype(np.float32) * 1e-12
        ell = np.arange(80, 101)
        fit = rng.permutation(75)[:60]
        with contextlib.redirect_stdout(io.StringIO()):
            result = fit_scaling(raw, ell, fit, feature_block=5)
            expected = fit_asinh(dell_values(raw, ell, fit))
            raw[75:] *= 1e9
            unchanged = fit_scaling(raw, ell, fit, feature_block=8)
        for key in expected:
            np.testing.assert_allclose(result[key], expected[key], rtol=2e-14, atol=1e-15)
            np.testing.assert_allclose(result[key], unchanged[key], rtol=2e-14, atol=1e-15)

    def test_moped_preserves_fisher_and_streamed_observation(self):
        rng = np.random.default_rng(8)
        ell = np.arange(80, 112)
        theta = FIDUCIAL + rng.uniform(-1, 1, (1000, 9))
        jacobian = rng.normal(size=(9, len(ell)))
        clean = ((theta - FIDUCIAL) @ jacobian) * 1e-12
        noisy = clean + rng.normal(size=clean.shape) * .4e-12
        fit = rng.permutation(900)
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            transform, diagnostic = fit_unbinned(noisy, clean, ell, theta, fit,
                FIDUCIAL-1, FIDUCIAL+1, local_n=700, feature_block=7)
            path = Path(directory) / "projected.npy"
            transform = write_projected(noisy, ell, transform, fit, path, block_rows=73)
            result = np.load(path)
        self.assertEqual(transform["matrix"].shape, (len(ell), 9))
        self.assertTrue(set(diagnostic["local_dataset_indices"]).issubset(set(fit)))
        np.testing.assert_allclose(diagnostic["compressed_covariance"], np.eye(9), atol=2e-12)
        np.testing.assert_allclose(diagnostic["compressed_fisher"], diagnostic["fisher"], atol=2e-11)
        expected = project(dell_values(noisy, ell, slice(None)), transform)
        np.testing.assert_allclose(result, expected, rtol=2e-6, atol=2e-6)
        np.testing.assert_allclose(project_observation(noisy[-1], ell, transform), result[-1], atol=2e-6)
        np.testing.assert_allclose(result[fit].mean(axis=0), 0, atol=1e-7)
        np.testing.assert_allclose(result[fit].std(axis=0, ddof=1), 1, rtol=1e-6)

    def test_raw_metadata_and_rebin_detect_permuted_or_changed_rows(self):
        rng = np.random.default_rng(9)
        raw = rng.normal(size=(15, 80)).astype(np.float32) * 1e-12
        ell = np.arange(80, 160)
        minimum, maximum = ell[::2], ell[1::2]
        matrix = make_cl_to_dl_matrix(ell, minimum, maximum, "2ell_plus_1")
        theta = (FIDUCIAL + rng.normal(size=(len(raw), 9))).astype(np.float32)
        ids = rng.permutation(len(raw))
        product = np.asarray("masked_baseline_noise_cross_deproj0")
        prepared = dict(theta=theta, sobol_global_row=ids, product=product,
            x=raw @ matrix, bin_ell_min=minimum, bin_ell_max=maximum,
            metadata_json=np.asarray(json.dumps(dict(bin_weighting="2ell_plus_1"))))
        with tempfile.TemporaryDirectory() as directory:
            path, metadata = Path(directory) / "raw.npy", Path(directory) / "metadata.npz"
            np.save(path, raw)
            def save_ids(row_ids):
                np.savez(metadata, theta=theta.astype(np.float64), theta_columns=PARAM_NAMES,
                         ell=ell, sobol_global_row=row_ids, product=product)
            save_ids(ids)
            opened, _ = open_raw(path, metadata, prepared, 80, 159)
            check = validate_rebin(opened, ell, prepared, block_rows=4)
            self.assertEqual(check["rows_checked"], len(raw))
            save_ids(ids[::-1])
            with self.assertRaises(AssertionError):
                open_raw(path, metadata, prepared, 80, 159)
            changed = raw.copy()
            changed[9, 20] *= 2
            with self.assertRaisesRegex(ValueError, "mismatch"):
                validate_rebin(changed, ell, prepared, block_rows=4)


if __name__ == "__main__":
    unittest.main()
