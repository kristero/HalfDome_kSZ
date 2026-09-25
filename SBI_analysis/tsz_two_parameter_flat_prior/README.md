# Guarded two-parameter tSZ prior

The active version is `tsz_P0_beta_guarded_v2`. It applies every existing
production support guard to the two free parameters, with the seven other
parameters fixed to the user-confirmed Battaglia12 values. Current artifacts
are in `outputs_guarded/`. The previous unfiltered proposal-only version is
superseded: its sources are in `archive_unfiltered_v1/`, and its original
figures/design remain in `outputs/` for provenance.

## Prior and physics

Start with independent linear-uniform proposals P0 in [1,60] and beta in
[2.8,16], then retain a proposal only if **all** original joint guards pass.
The joint density is constant on that retained region and zero outside.
It is a fixed-seven-parameter slice, not a marginal of the nine-parameter prior.
FLAMINGO does not set the support.

The pressure model is

```
A(M,z) = A0*(M200c/1e14 Msun)^alpha_m*(1+z)^alpha_z, A = P0, xc, beta
Pth/P200 = P0*q^(-0.3)*(1+q)^(-beta), q = r/(xc*R200c)
Pe = 0.5176*Pth
```

P0 controls pressure amplitude and beta the radial decline. The raw Battaglia
beta has asymptotic outer slope -(beta+0.3). The core amplitude is 0.497.
The fixed mass exponents for P0, xc, beta are 0.154, -0.00865, 0.0393;
the redshift exponents are -0.758, 0.731, 0.415.

## Applied guardrails and resulting support

`guardrails.py` and `guardrails.json` are unchanged, checksum-verified copies
from the frozen nine-parameter production campaign. They require:

1. Evolved beta in [2.8,50] throughout log10(M/Msun)=[12,15.7], z=[0.001,5].
2. Relative (xc/beta)/(xc_B12/beta_B12) in [0.4,8] throughout the catalogue
   rectangle log10(M/Msun)=[12.8167,15.5781], z=[0.0021,3.8555].
3. Finite spherical Y200/Y200_B12 in [0.003,30] on the original 9x9 grid,
   uniform in log(M) and log(1+z).
4. Conservative missing central LOS column beyond 1e5 R200c at most 1%.

With seven parameters fixed at B12, the allowed envelope is approximately
P0 in [1,60] and beta0 in [3.3541145392,10.875]. Exact floating-point boundaries
are derived in `flat_prior.py`; only roundoff-sized inward adjustments are
used when equality would otherwise fail the original numerical inequalities.

There is also a small excluded corner at low P0 and high beta. At P0=1,
beta0 must be at most about 10.50589594. At the largest beta, P0 must be at
least about 1.11630558. We retain the curved boundary rather than raising the
minimum P0 globally and dropping valid points elsewhere.

At fixed beta, Y200 is exactly proportional to P0. If F_i(beta) is Y200/B12
on grid point i evaluated at P0=1, the pressure cut gives

```
P0_lower(beta) = max(1, max_i[0.003/F_i(beta)])
P0_upper(beta) = min(60, min_i[30/F_i(beta)])
```

The density normalization is the integral of P0_upper-P0_lower over the
permitted beta interval: area about 443.711026685, or 56.9736809% of the
original proposal rectangle. Both marginals are close to uniform; their small
departures follow from this corner cut. The plots integrate the actual support
cross-sections rather than drawing independent uniform approximations.

## Reference points

FLAMINGO markers are P0/beta projections of the saved cosmology-corrected
nine-parameter clean-spectrum fits. Their other fit parameters differed from
B12. They are not fresh two-parameter fits or posterior confidence regions.
All displayed projected points pass this guarded two-parameter prior.
Purple dashed lines show the projected bounds of the same verified old
nine-parameter, 40-bin SBI prior used in the preceding comparisons.

## Generation and inference

```bash
python make_plots.py
```

This creates an **8192-point accepted parameter design**, expanded nine-column
parameters, proposal IDs, exact support boundary, PNG/vector-PDF figures,
CSV references, source hashes and validation in `outputs_guarded/`.
It does not generate maps, submit cluster jobs, or change the active 8k
nine-parameter run. Failed production simulations must not be redrawn simply
to select easier profiles.

For a larger parameter design:

```bash
python make_plots.py --count 524288 --output outputs_guarded_524288
```

The accepted Sobol prefix is count-independent. Rejection removes exact
digital-net balance; the design targets the uniform guarded joint density.

```python
from flat_prior import FlatPrior
prior = FlatPrior()
theta = prior.sample(1000, seed=12345)  # IID accepted draws; [P0, beta]
full_theta = prior.expand(theta)       # original nine-parameter order
log_density = prior.log_prob(theta)    # normalized; -inf for rejected values
torch_prior = prior.as_torch_distribution()  # optional PyTorch adapter
```

Do not replace this prior with a BoxUniform over its envelope: that would
include the excluded integrated-pressure corner. `torch_prior.py` evaluates
the same original guards for sampling, support and log_prob.

## Verification and limitations

The generator requires all 8192 points to pass both the original 9x9 guard
and a denser 33x33 Y200 check. It checks support against the frozen reference,
fixed values, accepted-prefix stability, rejected old corners, normalization
by independent integrations in both coordinate directions, and input hashes.
`verify_guarded.py` adds targeted boundary and PyTorch consistency checks.

These are guardrail and parameter-design checks. The six preflight maps of
the earlier campaign do not constitute a new full-map test of this entire
two-parameter slice. The inherited finite-Y grid, pixel sampling and projected
4R200c cutoff limitations remain. This work adds no further exclusions.

## Files

Modified: `prior.json`, `flat_prior.py`, `make_plots.py`, `README.md`.
Added: `guardrails.py`, `guardrails.json`, `torch_prior.py`,
`verify_guarded.py`, `outputs_guarded/`, and the unfiltered source archive.
The original nine-parameter source and cluster campaign are unchanged.
