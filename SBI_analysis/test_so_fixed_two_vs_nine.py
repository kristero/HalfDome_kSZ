"""Metric and identity tests; no SBI dependencies or posterior sampling."""
import unittest
import numpy as np
import pandas as pd

from compare_so_fixed_two_vs_nine import statistics, validate_rows


class ComparisonTests(unittest.TestCase):
    def rows(self):
        return pd.DataFrame(dict(series=["two_bins40"]*2, method=["bins40"]*2,
            n_train=[256]*2, test_index=[0, 1], param=["P0"]*2,
            truth=[2., 6.], mean=[3., 9.], std=[1., 10.],
            pull=[1., .3], error_prior=[.1, .3], prior_low=[0., 0.], prior_high=[10., 10.]))

    def test_normalize_before_rms(self):
        row = statistics(self.rows(), {"P0": 20.}).iloc[0]
        self.assertAlmostEqual(row.rmse_std, np.sqrt((1.+.09)/2))
        self.assertAlmostEqual(row.rmse_prior_native, np.sqrt(5)/10)
        self.assertAlmostEqual(row.rmse_prior_common, np.sqrt(5)/20)
        self.assertAlmostEqual(row.pearson_r, 1.)

    def test_separate_series_never_pair_rows(self):
        first = self.rows()
        second = first.copy()
        second["series"] = "nine_bins40"
        second["test_index"] += 1000
        stats = statistics(pd.concat([first, second]), {"P0": 10.})
        self.assertEqual(len(stats), 2)
        self.assertTrue((stats.n_test == 2).all())

    def test_validate_rows_and_stored_normalization(self):
        result = validate_rows(self.rows(), ["P0"], [0, 1], np.array([[2.], [6.]]), [0.], [10.])
        self.assertEqual(len(result), 2)
        bad = self.rows()
        bad["error_prior"] *= 2
        with self.assertRaises(AssertionError):
            validate_rows(bad, ["P0"], [0, 1], np.array([[2.], [6.]]), [0.], [10.])

    def test_missing_duplicate_wrong_truth_rejected(self):
        for rows in (self.rows().iloc[:1], pd.concat([self.rows(), self.rows()])):
            with self.assertRaises(ValueError):
                validate_rows(rows, ["P0"], [0, 1], np.array([[2.], [6.]]), [0.], [10.])
        with self.assertRaises(AssertionError):
            validate_rows(self.rows(), ["P0"], [0, 1], np.array([[6.], [2.]]), [0.], [10.])

    def test_zero_std_rejected_and_constant_prediction_not_perfect(self):
        frame = self.rows()
        frame["std"] = 0.
        with self.assertRaises(ValueError):
            validate_rows(frame, ["P0"], [0, 1], np.array([[2.], [6.]]), [0.], [10.])
        frame = self.rows()
        frame["mean"] = 4.
        self.assertTrue(np.isnan(statistics(frame, {"P0": 10.}).iloc[0].pearson_r))


if __name__ == "__main__":
    unittest.main()
