"""Propose the nearby Mstar variant from a validated fiducial full map.

At fixed shape and evolution, y is linear in P0 and clean power is quadratic.
Only the scalar amplitude is optimized here; the main driver still repaints
this candidate before accepting it as a fit to the other feedback variant.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from prior_model import HIGH, LOW, bin_cl, comparison_metrics, save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--campaign", type=Path, required=True)
    args = parser.parse_args()
    destination = args.root / "proposals/fit_Mstar-1sigma_iter0.json"
    if destination.exists():
        print("Mstar proposal already exists; preserving it")
        return
    source = json.loads((args.root / "results/L1_m9_fit.json").read_text())["best"]
    source_spectrum = bin_cl(np.load(args.root / "results" / source["name"] / "masked_clean_cl.npy"))[1]
    target = bin_cl(np.load(args.campaign / "results/Mstar-1sigma/masked_clean_cl.npy"))[1]
    scale = float(np.exp(.5*np.mean(np.log(target/source_spectrum))))
    theta = np.array(source["theta"])
    theta[0] *= scale
    assert np.all(theta >= LOW) and np.all(theta <= HIGH)
    proposal = dict(theta=theta.tolist(), **comparison_metrics(scale**2*source_spectrum, target),
        fit_type="Exact P0 rescaling of validated L1_m9 map; requires Mstar full-map confirmation",
        source_full_map=source["name"], amplitude_scale=scale, nfev=0, elapsed_seconds=0.,
        success=True, message="Analytic minimum for amplitude at fixed source shape; not a nine-parameter optimum",
        regularization_coefficient=0.)
    temporary = destination.with_suffix(".temporary.json")
    save_json(temporary, proposal)
    if destination.exists():
        temporary.unlink()
        print("Mstar fitting began while preparing seed; preserving its proposal")
        return
    temporary.replace(destination)
    print("Mstar seed RMS {:.3%}, max {:.3%}; full repaint still required".format(
        proposal["rms_fractional"], proposal["max_fractional"]))


if __name__ == "__main__":
    main()
