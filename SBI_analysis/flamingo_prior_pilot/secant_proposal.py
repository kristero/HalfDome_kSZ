"""Propose a parameter blend using two measured full-map spectra."""
import argparse
import json
from pathlib import Path

import numpy as np

from prior_model import HIGH, LOW, bin_cl, comparison_metrics, from_unit, save_json, to_unit


def make_proposal(root, campaign, variant, first, second, destination):
    path = root / "proposals" / (destination+".json")
    if path.exists():
        return json.loads(path.read_text())
    target = bin_cl(np.load(campaign / "results" / variant / "masked_clean_cl.npy"))[1]
    spectra, theta = [], []
    for name in (first, second):
        directory = root / "results" / name
        receipt = json.loads((directory / "full_map_complete.json").read_text())
        theta.append(np.array(receipt["theta"]))
        spectra.append(bin_cl(np.load(directory / "masked_clean_cl.npy"))[1])
    direction = np.log(spectra[1]/spectra[0])
    desired = np.log(target/spectra[0])
    centred = direction-direction.mean()
    fraction = float(np.clip(np.dot(centred, desired-desired.mean())/np.dot(centred, centred), 0, 1))
    log_amplitude_squared = float(np.mean(desired-fraction*direction))
    parameters = from_unit((1-fraction)*to_unit(theta[0])+fraction*to_unit(theta[1]))
    parameters[0] *= np.exp(log_amplitude_squared/2)
    assert np.all(parameters >= LOW) and np.all(parameters <= HIGH)
    estimated = spectra[0]*np.exp(fraction*direction+log_amplitude_squared)
    proposal = dict(theta=parameters.tolist(), **comparison_metrics(estimated, target),
        fit_type="Local log-spectrum secant between measured full maps; requires new full-map validation",
        source_full_maps=[first, second], second_map_fraction=fraction,
        amplitude_scale=float(np.exp(log_amplitude_squared/2)), nfev=0, elapsed_seconds=0.,
        success=True, message="Analytic interpolation proposal; not a globally optimized parameter fit",
        regularization_coefficient=0.)
    save_json(path, proposal)
    print("{}: blend {:.4f}, estimated RMS {:.3%}, max {:.3%}".format(
        destination, fraction, proposal["rms_fractional"], proposal["max_fractional"]), flush=True)
    return proposal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--campaign", type=Path, required=True)
    args = parser.parse_args()
    make_proposal(args.root, args.campaign, "L1_m9",
                  "fit_L1_m9_iter2", "fit_L1_m9_iter4", "fit_L1_m9_secant0")
    make_proposal(args.root, args.campaign, "fgas-8sigma",
                  "fit_fgas-8sigma_iter0", "fit_fgas-8sigma_iter1", "fit_fgas-8sigma_iter4")


if __name__ == "__main__":
    main()
