from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    from prepare_planck_no_noise_dataset import (
        DEFAULT_DATASET_SIZES,
        DEFAULT_OBSERVED_DL,
        DEFAULT_SOURCE_DATASET,
        build_planck_no_noise_dataset,
        parse_dataset_sizes,
        repo_root,
        validate_dataset_sizes,
        write_dataset,
    )
except ModuleNotFoundError:
    from SBI_analysis.prepare_planck_no_noise_dataset import (
        DEFAULT_DATASET_SIZES,
        DEFAULT_OBSERVED_DL,
        DEFAULT_SOURCE_DATASET,
        build_planck_no_noise_dataset,
        parse_dataset_sizes,
        repo_root,
        validate_dataset_sizes,
        write_dataset,
    )


class PlanckNoNoiseSetupTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = repo_root()
        cls.source_dataset = cls.root / DEFAULT_SOURCE_DATASET
        cls.observed_dl = cls.root / DEFAULT_OBSERVED_DL
        cls.payload, cls.summary = build_planck_no_noise_dataset(cls.source_dataset, cls.observed_dl)

    def test_planck_binned_dataset_contract(self) -> None:
        self.assertEqual(self.payload["theta"].shape, (100_000, 9))
        self.assertEqual(self.payload["x"].shape, (100_000, 16))
        self.assertEqual(self.payload["obs"].shape, (16,))
        self.assertEqual(self.payload["ell"].shape, (16,))
        self.assertFalse(bool(self.payload["noise_enabled"]))
        self.assertEqual(str(self.payload["noise_mode"]), "none")

    def test_planck_bin_edges_are_not_so_edges(self) -> None:
        np.testing.assert_allclose(self.payload["bin_ell_min"][[0, -1]], [21.0, 1085.0])
        np.testing.assert_allclose(self.payload["bin_ell_max"][[0, -1]], [26.0, 1410.0])
        self.assertLess(float(self.payload["ell"][-1]), 1500.0)

    def test_default_dataset_size_sweep_fits_available_rows(self) -> None:
        sizes = parse_dataset_sizes(DEFAULT_DATASET_SIZES)
        self.assertEqual(sizes, [1024, 2048, 4096, 8192, 16384, 32768, 50_000, 70_000, 85_000, 100_000])
        validate_dataset_sizes(sizes, int(self.summary["n_rows"]))

    def test_written_npz_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "planck_binned16_no_noise_sbi_run.npz"
            write_dataset(output, self.payload, self.summary)
            with np.load(output, allow_pickle=True) as data:
                self.assertIn("prior_low", data.files)
                self.assertIn("prior_high", data.files)
                self.assertNotIn("prior", data.files)
                self.assertEqual(data["x"].shape[1], 16)
                np.testing.assert_allclose(data["obs"], self.payload["obs"])


if __name__ == "__main__":
    unittest.main()
