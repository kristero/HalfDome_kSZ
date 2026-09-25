"""Small format checks for the real byte-order mismatch encountered on COSMA."""
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np

from download_maps import validate


class MapFormatTests(unittest.TestCase):
    def check_fixture(self, values, expected_shape):
        with tempfile.TemporaryDirectory(prefix="flamingo_format_test_") as directory:
            # Confirm the automatically cleaned directory is confined to the
            # intended temporary root, including on the Windows workstation.
            Path(directory).resolve().relative_to(Path(tempfile.gettempdir()).resolve())
            path = Path(directory) / "map.hdf5"
            with h5py.File(path, "w") as handle:
                handle["data"] = values
            return validate(path, dict(file_bytes=path.stat().st_size,
                                       shape=expected_shape, dtype="<f8"))

    def test_both_float64_byte_orders_preserve_values_and_sign(self):
        values = np.array([-3e-7, 1e-6, 2e-6, 8e-6])
        reports = [self.check_fixture(values.astype(dtype), [4]) for dtype in ("<f8", ">f8")]
        for report in reports:
            self.assertEqual(report["minimum"], values.min())
            self.assertEqual(report["maximum"], values.max())
            self.assertEqual(report["mean"], values.mean())
            self.assertTrue(report["all_pixels_finite"])
        self.assertEqual(reports[0]["rms"], reports[1]["rms"])

    def test_wrong_shape_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "dtype or shape"):
            self.check_fixture(np.zeros((2, 2), dtype="f8"), [4])

    def test_float32_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "dtype or shape"):
            self.check_fixture(np.zeros(4, dtype="f4"), [4])

    def test_nonfinite_map_is_rejected(self):
        for invalid in (np.nan, np.inf, -np.inf):
            with self.assertRaisesRegex(ValueError, "Nonfinite"):
                self.check_fixture(np.array([0., invalid], dtype="f8"), [2])


if __name__ == "__main__":
    unittest.main()
