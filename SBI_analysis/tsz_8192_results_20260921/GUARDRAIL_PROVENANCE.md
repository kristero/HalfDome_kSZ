# Historical guardrails: clarification and primary evidence

The recent eight fast individual-LOS probes do not invalidate the earlier
full-cache/full-map timeouts. The explanation that discussed only those fast
probes was incomplete. These are different parameter/radius samples and
different scopes of computation.

## Recorded failures

- `../flamingo_prior_pilot/cluster_results/results/fit_L1_m9_iter1/run_status.json`:
  exit code 124 after 1200.0265 seconds, job 597349.idark.
- `../flamingo_prior_pilot/cluster_results/results/extreme_compact_steep_evolving/run_status.json`:
  exit code 124 after 1200.0165 seconds, job 597356.idark. This case had
  beta amplitude 16, beta redshift exponent 1.5, and xc amplitude 0.1.
- These are measured 20-minute timeouts, not proof of an infinite loop. The
  watchdog covered an entire map job, so its duration is not a scalar-LOS timing.
- `../flamingo_prior_pilot/cluster_results/audit/quadrature_cost_summary.json`
  records 2 native evaluation-budget hits among 926 probes. Both are outside
  the painted 4 R200 disc. Each used 4123 evaluations under a requested budget
  of 4096; normalized integration used 19 evaluations at each point.
- The underlying `audit/quadrature_cost/quadrature_cost_dense.csv` places them
  at M=10^15.7 solar masses, z=5 and projected radii approximately
  4.22e6 and 5.62e6 R200. Native relative error estimates were 9.18e-10 and
  5.00e-10 versus requested 1e-12. This diagnoses a floating-point tail
  precision problem in the overextended interpolation domain. It does not
  establish that each timeout spent every second at these exact two nodes.
- `../flamingo_extended_prior/cluster_results/independent_noise_20260915/preflight/quality_gate.json`
  records a successful repaired steep-boundary map in 270.78 seconds, with
  19756 nonpositive cache entries floored to the smallest positive Float64.
  This is a distinct control, not a matched before/after timing of either
  timed-out pilot. The original floor-underflow failure is also reproduced
  separately in this report's stock-cache experiment.

## What the old restrictions meant

The authoritative old support is `../flamingo_linear_prior/prior.json`.
Its justification was documented in `../flamingo_extended_prior/README.md`;
the later audit is `../tsz_guardrail_study/METHODS.md`, section
"Audit of each existing cut".

| Old restriction | Provenance and limitation |
|---|---|
| Evolved beta >= 2.8 | Margin above beta > 2.7 for finite untruncated thermal energy. The extra 0.1 is discretionary, and a finite sphere does not require this inequality. |
| Evolved beta <= 50 | Engineering cap informed by expensive steep trials. No measured sharp breakdown at 50 and no physical upper bound. Earlier proposal searches used a temporary upper cap of 40. |
| Relative xc/beta between 0.4 and 8 | Conservative size proxy. It contains neither angular-diameter distance nor pixel geometry, and is not a demonstrated HEALPix accuracy threshold. |
| Y200/B12 between 0.003 and 30 | Chosen signal/thermal-content envelope. Neither endpoint is a singularity or universal pressure-only physical bound. |
| Missing central LOS column <= 1% | Explicit tolerance choice for approximating an untruncated model. Not equivalent to a bound on the final noise-weighted observable. |
| Positive finite cache; convergence checks | Actual numerical requirements. Keep these, with a safe treatment of negligible underflow tails. |
| 1200-second watchdog | A compute budget. Keep failure records and the assigned row; do not silently redraw to obtain a faster sample. |

## Consequence for the revised prior

The failures were real, while several particular exclusion thresholds were
heuristic and stronger than necessary. Fixing normalized LOS quadrature and
the cache floor addresses demonstrated implementation failures without
requiring the old beta ceiling. Spherical truncation is an additional physical
model change and must be distinguished from those numerical repairs.

Retire unsupported parameter cuts only with validation of the repaired
forward model. Keep positivity, finite values, integration accuracy, resource
accounting and observable-level interpolation/pixel convergence checks. A fast
scalar integral is not evidence that every full-cache/full-map parameter
combination is valid. No prior or simulator code was changed in this
clarification.

This clarification uses the locally archived primary run records, inspected
on 21 September. A live SSH attempt to read the old cluster log tails timed
out; no current queue status is asserted here.
