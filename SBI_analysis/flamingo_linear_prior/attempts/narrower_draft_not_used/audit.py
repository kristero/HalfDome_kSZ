"""Validate uniform coverage, full-box inclusion and finite-pressure integrals."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.integrate import quad

from prior import UniformPrior, digest, log_y200, write_json
from worker import verify_frozen


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = verify_frozen(root)
    prior = UniformPrior(manifest["prior"])
    theta = np.load(root / "design/theta.npy")
    assert prior.contains(theta).all()
    larger, indices = prior.sobol(524288)
    np.testing.assert_array_equal(theta, larger[:len(theta)])
    np.testing.assert_array_equal(np.load(root / "design/proposal_index.npy"), indices[:len(theta)])
    # Exact equal marginal bin counts are guaranteed at these powers of two.
    bins = 32
    histograms = [np.histogram(theta[:, k], np.linspace(prior.low[k], prior.high[k], bins + 1))[0]
                  for k in range(9)]
    if len(theta) & (len(theta) - 1) == 0 and len(theta) >= bins:
        assert all(np.all(h == len(theta) // bins) for h in histograms)
    old = json.loads((root / "inputs/old_metadata.json").read_text())["verified_bundle"]
    old_low, old_high = np.array(old["prior_low"]), np.array(old["prior_high"])
    assert np.all(prior.low <= old_low) and np.all(prior.high >= old_high)
    in_old = ((theta >= old_low) & (theta <= old_high)).all(1)
    errors = []
    for xc in (.02, .14, .5, 3., 300.):
        for beta in (1.8, 2.6, 2.7, 2.70000001, 3., 50., 110.):
            direct = quad(lambda r: xc**.3 * r**1.7 * (1 + r / xc)**(-beta),
                          0., 1., epsabs=0., epsrel=1e-10, limit=300)[0]
            actual = np.exp(log_y200(1., xc, beta))
            errors.append(float(abs(actual / direct - 1)))
    assert max(errors) < 1e-8
    # These scalar checks diagnose pressure extremes; they do not reject rows.
    metrics = prior.metrics(prior.corners())
    report = dict(passed=True, manifest_sha256=digest(root / "manifest.json"), rows=len(theta),
                  prefix_preserved_through_524288=True, independent_linear_target=True,
                  rectangular_support=True, rejection_used=False,
                  histogram_bins=bins, histogram_counts=[h.tolist() for h in histograms],
                  entire_old_box_included=True, actual_rows_inside_entire_old_box=int(in_old.sum()),
                  old_box_fraction_of_new_box=float(np.prod((old_high-old_low)/(prior.high-prior.low))),
                  finite_integral_checks=len(errors), maximum_finite_integral_relative_error=max(errors),
                  box_certificate=prior.certify_box(),
                  corner_metric_ranges={key: [float(v.min()), float(v.max())] for key, v in metrics.items()},
                  full_map_validation="Required separately; this audit is not a simulation preflight")
    write_json(root / "audit/design_check.json", report)
    print(json.dumps({k:v for k,v in report.items() if k != "histogram_counts"}, indent=2))


if __name__ == "__main__":
    main()
