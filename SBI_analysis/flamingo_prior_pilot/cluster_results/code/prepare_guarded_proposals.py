"""Prepare bounded replacements while slow native-grid trials reach their limits."""
import argparse
import json
from pathlib import Path

import numpy as np

from forward_proposals import CatalogueForward
from prior_model import bin_cl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--campaign", type=Path, required=True)
    args = parser.parse_args()
    forward = CatalogueForward(args.root, args.campaign)
    # These names follow the already-running explicit failed/extreme trials.
    # Existing proposals are never overwritten.
    for variant, role, iteration in (("L1_m9", "fiducial", 4),
                                      ("fgas-8sigma", "other_feedback", 3)):
        name = "fit_{}_iter{}".format(variant, iteration)
        if (args.root / "proposals" / (name+".json")).exists():
            continue
        state = json.loads((args.root / ("progress_"+role+".json")).read_text())
        best = state["fits"][variant]["best"]
        theta = np.array(best["theta"])
        measured = bin_cl(np.load(args.root / "results" / best["name"] / "masked_clean_cl.npy"))[1]
        target = bin_cl(np.load(args.campaign / "results" / variant / "masked_clean_cl.npy"))[1]
        forward.fit(target, name, start=theta, correction=measured/forward.raw(theta),
                    multistart=False, trust_radius=.12)
    print("Guarded follow-up proposals ready for independent full-map checks", flush=True)


if __name__ == "__main__":
    main()
