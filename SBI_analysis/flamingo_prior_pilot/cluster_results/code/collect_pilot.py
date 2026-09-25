"""Merge disjoint cluster runs after all three PBS jobs have finished."""
import argparse
import json
from pathlib import Path

from prior_model import save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    result = None
    inputs = []
    for label in ("fiducial", "other_feedback", "extremes"):
        path = args.root / ("pilot_results_"+label+".json")
        data = json.loads(path.read_text())
        assert data["status"] == "pilot_forward_runs_finished"
        if result is None:
            result = {key: value for key, value in data.items() if key not in ("fits", "extremes", "elapsed_seconds")}
            result.update(fits={}, extremes={})
        assert not (set(result["fits"]) & set(data["fits"]))
        assert not (set(result["extremes"]) & set(data["extremes"]))
        result["fits"].update(data["fits"])
        result["extremes"].update(data["extremes"])
        inputs.append(str(path))
    assert len(result["fits"]) == 3 and len(result["extremes"]) == 6
    result["merged_run_files"] = inputs
    save_json(args.root / "pilot_results.json", result)
    print("Merged three feedback fits and six independent full-map stress tests", flush=True)


if __name__ == "__main__":
    main()
