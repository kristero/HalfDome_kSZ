import csv
import json
import tempfile
from pathlib import Path
import unittest

import numpy as np

from generate import BUNDLE, NAMES, validate_config, validate_noise_tables, seeds
from make_designs import make_designs
from plot_sobol_designs import read_table, design_summary


class SobolPlotChecks(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((BUNDLE / "config.json").read_text())
        self.config["n_rows"] = 32
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name)/"designs"
        make_designs(self.config, self.folder)

    def test_uniform_counts_and_fixed_columns(self):
        for mode in ("two_param", "nine_param"):
            result = design_summary(read_table(self.folder/(mode+".csv"), 32), mode, self.config, bins=8)
            self.assertEqual(result["n_unique_rows"], 32)
            for param in result["parameters"]:
                if param["varying"]:
                    self.assertEqual(param["min_bin_count"], 4)
                    self.assertEqual(param["max_bin_count"], 4)
                    self.assertLessEqual(param["max_abs_cdf_minus_uniform"], 1/32 + 1e-12)
                else:
                    self.assertIsNone(param["max_abs_cdf_minus_uniform"])

    def test_duplicate_and_out_of_prior_rejected(self):
        theta = read_table(self.folder/"nine_param.csv", 32)
        duplicate = theta.copy()
        duplicate[-1] = duplicate[0]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            design_summary(duplicate, "nine_param", self.config)
        theta[0, 0] = self.config["prior_high"][0] + 1
        with self.assertRaisesRegex(ValueError, "Out-of-prior"):
            design_summary(theta, "nine_param", self.config)

    def test_column_order_rejected(self):
        bad = self.folder/"bad.csv"
        with bad.open("w", newline="") as f:
            csv.writer(f).writerow(list(reversed(NAMES)))
        with self.assertRaisesRegex(ValueError, "column order"):
            read_table(bad, 32)

    def test_large_preset_preserves_physics_and_streams(self):
        default = json.loads((BUNDLE / "config.json").read_text())
        large = json.loads((BUNDLE / "config_524288_all_noise.example.json").read_text())
        self.assertEqual(validate_config(large), 8388608)
        validate_noise_tables(large)
        for key in default:
            if key not in {"n_rows", "design_dir", "noise_cases", "deprojections"}:
                self.assertEqual(default[key], large[key])
        self.assertEqual(seeds(default, "two_param", 32768), seeds(large, "two_param", 32768))


if __name__ == "__main__":
    unittest.main()
