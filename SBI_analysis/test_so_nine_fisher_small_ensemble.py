"""Regression checks for split reuse in the historical Battaglia12 ensemble."""
import unittest

from run_so_nine_fisher_small_ensemble import select_disjoint, split_seeds


class NoiseSplitTests(unittest.TestCase):
    def test_adjacent_roots_share_one_split(self):
        self.assertEqual(len(set(split_seeds(20001)) & set(split_seeds(20002))), 1)

    def test_pilot_excludes_observation_and_reused_splits(self):
        records = [dict(seed=s) for s in range(20001,20065)]
        observation, selected, _, available = select_disjoint(records,20001,16)
        self.assertEqual(available,31)
        self.assertEqual([r["seed"] for r in selected],list(range(20003,20035,2)))
        splits = [s for r in [observation]+selected for s in split_seeds(r["seed"])]
        self.assertEqual(len(splits),len(set(splits)))

    def test_rejects_more_independent_rows_than_available(self):
        with self.assertRaisesRegex(ValueError,"Only 31"):
            select_disjoint([dict(seed=s) for s in range(20001,20065)],20001,32)


if __name__ == "__main__":
    unittest.main()
