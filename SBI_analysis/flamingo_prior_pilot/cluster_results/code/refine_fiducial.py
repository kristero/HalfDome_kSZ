"""Validate the fiducial secant proposal and update its completed fit record."""
import argparse
import json
from pathlib import Path

import numpy as np

from prior_model import bin_cl, comparison_metrics, save_json
from run_map import full_map


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--campaign", type=Path, required=True)
    args = parser.parse_args()
    name = "fit_L1_m9_secant0"
    proposal = json.loads((args.root / "proposals" / (name+".json")).read_text())
    output = full_map(args.root, args.campaign, name, proposal["theta"])
    target = bin_cl(np.load(args.campaign / "results/L1_m9/masked_clean_cl.npy"))[1]
    prediction = bin_cl(np.load(output / "masked_clean_cl.npy"))[1]
    record = dict(name=name, theta=proposal["theta"], **comparison_metrics(prediction, target),
                  proposal_rms_fractional=proposal["rms_fractional"])
    path = args.root / "pilot_results_fiducial.json"
    state = json.loads(path.read_text())
    fit = state["fits"]["L1_m9"]
    if not any(item["name"] == name for item in fit["iterations"]):
        fit["iterations"].append(record)
    if record["rms_fractional"] < fit["best"]["rms_fractional"]:
        fit["best"] = record
    fit["meets_accuracy_target"] = fit["best"]["rms_fractional"] <= .02 and fit["best"]["max_fractional"] <= .05
    save_json(path, state)
    save_json(args.root / "results/L1_m9_fit.json", fit)
    print("FULL-MAP secant refinement: RMS {:.3%}, max {:.3%}".format(
        record["rms_fractional"], record["max_fractional"]), flush=True)


if __name__ == "__main__":
    main()
