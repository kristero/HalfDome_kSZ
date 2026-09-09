"""Rerun isolation and transfer-status tests; no PBS jobs or SSH connections."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from prepare_so_compression_bestval_rerun import prepare_rerun
from so_sbi_compression import METHODS, write_json


class RerunTests(unittest.TestCase):
    def make_source(self, root):
        root.mkdir()
        write_json(root / "experiment.json", dict(experiment_id="fixture"))
        np.savez(root / "shared.npz", test_indices=np.array([10, 11]))
        for name in ("pca_diagnostics.npz", "moped_diagnostics.npz"):
            np.savez(root / name, diagnostic=np.ones(1))
        for method in METHODS:
            np.savez(root / f"{method}_transform.npz", matrix=np.eye(2))
            np.save(root / f"{method}_x.npy", np.zeros((12, 2)))
        write_json(root / "bins40/training_complete.json", dict(
            experiment_id="fixture", converged_by_early_stopping=True))
        write_json(root / "bins40/evaluation/evaluation_complete.json", dict(experiment_id="fixture"))
        (root / "bins40/evaluation/profiles").mkdir()
        for idx in (10, 11):
            np.savez(root / f"bins40/evaluation/profiles/row{idx}.npz", samples=np.ones((4, 9)))
        write_json(root / "pca/evaluation/old_result.json", dict(do_not_copy=True))

    @patch("prepare_so_compression_bestval_rerun.subprocess.check_output", return_value="testcommit\n")
    def test_copies_control_but_not_old_compressed_models(self, _):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            self.make_source(source)
            target = root / "new"
            prepare_rerun(source, target)
            self.assertFalse((target / "pca").exists())
            self.assertFalse((target / "moped").exists())
            self.assertTrue((target / "bins40/evaluation/profiles/row10.npz").is_file())
            self.assertEqual((source / "experiment.json").read_bytes(), (target / "experiment.json").read_bytes())
            self.assertFalse((source / "shared.npz").samefile(target / "shared.npz"))
            meta = json.loads((target / "rerun_provenance.json").read_text())
            self.assertEqual(meta["retrained_methods"], ["pca", "moped"])
            with self.assertRaises(FileExistsError):
                prepare_rerun(source, target)
            with self.assertRaises(ValueError):
                prepare_rerun(source, source / "nested")


if __name__ == "__main__":
    unittest.main()
