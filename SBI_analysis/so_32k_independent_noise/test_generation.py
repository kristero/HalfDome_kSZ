import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import generate as g
from make_designs import make_designs


class GenerationChecks(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((g.BUNDLE / "config.json").read_text())

    def test_all_actual_split_seeds_unique(self):
        values = g.validate_config(self.config)
        self.assertEqual(values, 131072)
        seeds = [s for mode in g.MODES for r in range(1, self.config["n_rows"]+1)
                 for s in g.seeds(self.config, mode, r)[1:]]
        self.assertEqual(len(set(seeds)), values)

    def test_larger_design_all_products_stable_seeds(self):
        original = g.seeds(self.config, "nine_param", 32000)
        self.config.update(n_rows=524288, noise_cases=["goal", "baseline"], deprojections=[2, 0])
        self.assertEqual(g.validate_config(self.config), 8388608)
        self.assertEqual(g.seeds(self.config, "nine_param", 32000, "baseline_deproj0"), original)
        values = [s for m in g.MODES for r in (1, 2, 32768, 32769, 524288)
                  for p in g.products(self.config) for s in g.seeds(self.config, m, r, p)[1:]]
        self.assertEqual(len(values), len(set(values)))
        self.config.update(n_rows=32768, sequence_offset=32768)
        self.assertEqual(g.seeds(self.config, "two_param", 1),
                         g.seeds(dict(self.config, sequence_offset=0), "two_param", 32769))

    def test_noise_tables_and_invalid_selection(self):
        self.config.update(noise_cases=["baseline", "goal"], deprojections=[0, 2])
        g.validate_noise_tables(self.config)
        self.config["deprojections"] = [0, 0]
        with self.assertRaises(ValueError):
            g.validate_config(self.config)

    def test_multi_product_command(self):
        self.config.update(noise_cases=["baseline", "goal"], deprojections=[0, 2])
        cmd = g.command(self.config, "two_param", 1, Path("design"), Path("halos"), Path("raw"), Path("cache"), "julia", 2)
        self.assertIn("so_noise_deprojections=0,2", cmd)
        self.assertIn("save_goal_noise_cross_cl=true", cmd)

    def test_extend_sobol_preserves_prefix_and_row_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            small, large = dict(self.config, n_rows=8), dict(self.config, n_rows=16)
            make_designs(small, directory/"small")
            make_designs(large, directory/"large")
            continuation = dict(small, sequence_offset=8)
            make_designs(continuation, directory/"next")
            sources = {mode: directory/"large"/(mode+".csv") for mode in g.MODES}
            make_designs(continuation, directory/"extracted", sources=sources)
            for mode in g.MODES:
                read = lambda name: np.loadtxt(directory/name/(mode+".csv"), delimiter=",", skiprows=1)
                np.testing.assert_array_equal(read("small"), read("large")[:8])
                np.testing.assert_array_equal(read("next"), read("large")[8:])
                np.testing.assert_array_equal(read("extracted"), read("next"))
            large["design_dir"] = str(directory/"large")
            g.load_designs(large)
            with self.assertRaisesRegex(ValueError, "row-offset"):
                g.load_designs(dict(large, sequence_offset=16))

    def test_adjacent_seed_bug_rejected(self):
        self.config["noise_seed_stride"] = 1
        self.assertEqual(g.seeds(self.config, "two_param", 1)[2],
                         g.seeds(self.config, "two_param", 2)[1])
        with self.assertRaises(ValueError):
            g.validate_config(self.config)

    def test_cross_dataset_collision_rejected(self):
        self.config["noise_seed_bases"]["nine_param"] = self.config["noise_seed_bases"]["two_param"]
        with self.assertRaises(ValueError):
            g.validate_config(self.config)

    def test_designs_and_fixed_parameters(self):
        designs = g.load_designs(self.config)
        self.assertEqual(designs["two_param"][1].shape, (32768, 9))
        self.assertEqual(designs["nine_param"][1].shape, (32768, 9))
        self.assertEqual(np.flatnonzero(np.ptp(designs["two_param"][1], axis=0) > 0).tolist(), [0, 2])
        self.assertTrue(np.all(np.ptp(designs["nine_param"][1], axis=0) > 0))

    def test_noise_only_baseline_and_fixed_mask(self):
        args = g.command(self.config, "nine_param", 19, Path("design.csv"), Path("halos.h5"),
                         Path("raw"), Path("cache"), "julia", 26)
        self.assertIn("save_goal_noise_cross_cl=false", args)
        self.assertIn("so_noise_deprojections=0", args)
        self.assertIn("mask_seed=12345", args)
        self.assertIn(f"noise_seed={g.seeds(self.config, 'nine_param', 19)[0]}", args)
        self.assertIn("save_no_noise_cl=true", args)

    def test_inherited_seed_cannot_override_command(self):
        with patch.dict("os.environ", {"TSZ_NOISE_SEED": "12345", "TSZ_MASK_SEED": "4"}):
            env = g.simulator_environment(26)
        self.assertNotIn("TSZ_NOISE_SEED", env)
        self.assertNotIn("TSZ_MASK_SEED", env)
        self.assertEqual(env["JULIA_NUM_THREADS"], "26")

    def test_signed_binning(self):
        values = g.bin_dell(-np.ones(7980), self.config)
        self.assertEqual(values.shape, (40,))
        self.assertTrue(np.all(values < 0))

    def test_checkpoints_resume_and_corruption(self):
        config = copy.deepcopy(self.config)
        config.update(n_rows=2, noise_cases=["baseline", "goal"], deprojections=[0, 2])
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            args = SimpleNamespace(root=folder, worker=0, workers=1, seconds=100,
                row_timeout=1, catalogue=folder/"catalogue", julia="not-used", threads=2, max_rows=0)
            designs = {mode: (folder/(mode+".csv"), np.tile(config["fiducial"], (2, 1))) for mode in g.MODES}
            def simulate(cmd, **kwargs):
                options = dict(s.split("=", 1) for s in cmd if "=" in s)
                out = Path(options["output_dir"])
                kwargs["stdout"].write(f"Mask seed: 12345\nNoise seed: {options['noise_seed']}\n")
                kwargs["stdout"].write("Actual beam FWHM: 2.0\nPainter: ring_locked\n")
                seed = int(options["noise_seed"])
                for product, (case, d) in g.products(config).items():
                    offset = 10000*(d+1) + {"baseline": 100, "goal": 200}[case]
                    kwargs["stdout"].write(f"Actual split seeds {product}: {seed+offset+1},{seed+offset+2}\n")
                    np.save(out/f"mock_masked_{case}_noise_cross_cl_deproj{d}_lmax7979.npy", -np.ones(7980))
                kwargs["stdout"].write("Actual theta: " + json.dumps(config["fiducial"]) + "\n")
                for d in config["deprojections"]:
                    np.save(out/f"mock_masked_no_noise_cl_deproj{d}_lmax7979.npy", np.ones(7980))
            with patch.object(g.subprocess, "run", side_effect=simulate) as process:
                g.work(args, config, designs, {"experiment_id": "fixture"})
                self.assertEqual(process.call_count, 4)
                g.work(args, config, designs, {"experiment_id": "fixture"})
                self.assertEqual(process.call_count, 4)
            args.stage = "combine"
            g.status_or_combine(args, config, designs, {"experiment_id": "fixture"})
            for mode, n_params in (("two_param", 2), ("nine_param", 9)):
                with np.load(folder/mode/"prepared/dataset.npz") as data:
                    self.assertEqual(data["theta"].shape, (2, n_params))
                    self.assertEqual(data["x"].shape, (2, 40))
                    self.assertTrue(np.all(data["x"] < 0))
                    for p in g.products(config):
                        self.assertEqual(data[f"noise_split_seeds_{p}"].shape, (2, 2))
                        np.testing.assert_array_equal(data[f"x_{p}"], data["x"])
                    columns = [0, 2] if n_params == 2 else list(range(9))
                    np.testing.assert_array_equal(data["theta"], data["theta_full"][:, columns])
            marker = folder/"two_param/raw/row00001/complete.json"
            clean_path = json.loads(marker.read_text())["spectra"]["clean"]["file"]
            np.save(marker.parent/clean_path, np.zeros(7980))
            with self.assertRaisesRegex(ValueError, "Changed/truncated"):
                g.verify_row(folder/"two_param/raw/row00001", config, "two_param", 1,
                    designs["two_param"][1][0], "fixture")


if __name__ == "__main__":
    unittest.main()
