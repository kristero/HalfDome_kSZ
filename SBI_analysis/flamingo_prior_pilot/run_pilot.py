"""Fit clean FLAMINGO spectra, repaint proposals, and exercise prior extremes.

All reported fits use full Nside=4096 maps. The fast catalogue calculation
only proposes parameters; repeated full maps calibrate its local discrepancy.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from forward_proposals import CatalogueForward
from prior_model import FIDUCIAL, NAMES, bin_cl, comparison_metrics, save_json
from run_map import full_map

VARIANTS = ("L1_m9", "fgas-8sigma", "Mstar-1sigma")


def extreme_design():
    # These examples isolate shape limits and two coupled evolutions. Other
    # single edges, all pairs and all corners are covered by the pressure audit.
    changes = {
        "xc_high": {"xc": 4.},
        "beta_low": {"beta": 2.8},
        "beta_high": {"beta": 16.},
        "wide_steep": {"xc": 4., "beta": 16.},
        "compact_steep_evolving": {"xc": .1, "beta": 16., "alpha_z_beta": 1.5},
        "mass_redshift_amplitude": {"alpha_m_P0": 1.5, "alpha_z_P0": -4.5},
    }
    result = {}
    for name, updates in changes.items():
        theta = FIDUCIAL.copy()
        for key, value in updates.items():
            theta[NAMES.index(key)] = value
        result[name] = theta
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--max-iterations", type=int, default=5)
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=VARIANTS)
    parser.add_argument("--skip-extremes", action="store_true")
    parser.add_argument("--extremes-only", action="store_true")
    parser.add_argument("--run-label", default="all")
    args = parser.parse_args()
    root, campaign = args.root, args.campaign
    suffix = "" if args.run_label == "all" else "_"+args.run_label
    progress_path = root / ("progress"+suffix+".json")
    began = time.monotonic()
    forward = CatalogueForward(root, campaign)
    forward.validate_projection()
    state = dict(fit_statistic="Equal-weight log residuals of 40 clean, masked, beam-smoothed D_ell bins",
                 approximation_role="Parameter proposals only; fit quality below always uses repainted full maps",
                 target_rms_fractional=.02, target_max_fractional=.05, fits={}, extremes={})
    save_json(progress_path, state)
    for variant in ([] if args.extremes_only else args.variants):
        target = bin_cl(np.load(campaign / "results" / variant / "masked_clean_cl.npy"))[1]
        correction = forward.correction.copy()
        theta = FIDUCIAL.copy()
        iterations = []
        best = None
        for iteration in range(args.max_iterations):
            name = "fit_{}_iter{}".format(variant, iteration)
            old_status_path = root / "results" / name / "run_status.json"
            if old_status_path.exists():
                old_status = json.loads(old_status_path.read_text())
                if old_status["exit_code"] == 124:
                    iterations.append(dict(name=name, theta=old_status["theta"],
                        status="runtime_limit", elapsed_seconds=old_status["elapsed_seconds"]))
                    print("Retaining timed-out candidate as a range-limit finding: "+name, flush=True)
                    continue
            proposal_path = root / "proposals" / (name+".json")
            if proposal_path.exists():
                proposal = json.loads(proposal_path.read_text())
                theta = np.array(proposal["theta"])
            else:
                theta, proposal = forward.fit(target, name, start=theta,
                    correction=correction, multistart=(iteration == 0),
                    trust_radius=None if iteration == 0 else .12)
            try:
                output = full_map(root, campaign, name, theta)
            except RuntimeError:
                failure = json.loads(old_status_path.read_text())
                if failure["exit_code"] != 124:
                    raise
                iterations.append(dict(name=name, theta=theta.tolist(),
                    status="runtime_limit", elapsed_seconds=failure["elapsed_seconds"]))
                theta = FIDUCIAL.copy() if best is None else np.array(best["theta"])
                if best is not None:
                    clean = bin_cl(np.load(root / "results" / best["name"] / "masked_clean_cl.npy"))[1]
                    correction = clean/forward.raw(theta)
                continue
            clean = bin_cl(np.load(output / "masked_clean_cl.npy"))[1]
            record = dict(name=name, theta=theta.tolist(), **comparison_metrics(clean, target),
                          proposal_rms_fractional=proposal["rms_fractional"])
            iterations.append(record)
            if best is None or record["rms_fractional"] < best["rms_fractional"]:
                best = record
            state["fits"][variant] = dict(best=best, iterations=iterations,
                meets_accuracy_target=best["rms_fractional"] <= .02 and best["max_fractional"] <= .05)
            save_json(progress_path, state)
            print("FULL-MAP {}: RMS {:.3%}, max {:.3%}".format(name,
                  record["rms_fractional"], record["max_fractional"]), flush=True)
            if record["rms_fractional"] <= .02 and record["max_fractional"] <= .05:
                break
            # Local response calibration is re-anchored to the actual new map.
            # The following optimizer remains an approximation until repainted.
            correction = clean/forward.raw(theta)
        if best is None:
            raise RuntimeError("No completed fit candidate for "+variant)
        save_json(root / "results" / (variant+"_fit.json"), state["fits"][variant])

    for name, theta in ({} if args.skip_extremes else extreme_design()).items():
        name = "extreme_"+name
        try:
            output = full_map(root, campaign, name, theta)
            clean = bin_cl(np.load(output / "masked_clean_cl.npy"))[1]
            record = dict(theta=theta.tolist(), status="full_map_completed",
                **comparison_metrics(clean, forward.fiducial_full),
                power_ratio_min=float(min(clean/forward.fiducial_full)),
                power_ratio_max=float(max(clean/forward.fiducial_full)))
        except (RuntimeError, ValueError) as exc:
            # A failed extreme is a scientific/numerical finding, not a reason
            # to lose the remaining independent stress tests.
            record = dict(theta=theta.tolist(), status="failed", error=str(exc))
            print("Extreme failed: {}: {}".format(name, exc), flush=True)
        state["extremes"][name] = record
        save_json(progress_path, state)
    state.update(status="pilot_forward_runs_finished", elapsed_seconds=time.monotonic()-began)
    save_json(root / ("pilot_results"+suffix+".json"), state)
    print("All pilot forward runs finished", flush=True)


if __name__ == "__main__":
    main()
