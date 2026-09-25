# Broad HalfDome prior with linear sampling and compatible profile reuse

This is the active replacement requested after the user rejected narrowing the
prior around FLAMINGO. The complete previous broad parameter ranges and every
joint feasibility cut are preserved. Only the two previously log-uniform base
coordinates, P0 and xc, change to linear-uniform. FLAMINGO is used for plot
overlays and coverage checks, not to set these ranges.

| Parameter | Lower | Upper | Base density |
|---|---:|---:|---|
| P0 | 1 | 60 | uniform in physical value |
| xc | 0.1 | 4 | uniform in physical value |
| beta | 2.8 | 16 | uniform in physical value |
| alpha_m_P0 | -0.2 | 1.5 | uniform |
| alpha_m_xc | -0.6 | 0.4 | uniform |
| alpha_m_beta | -0.2 | 0.4 | uniform |
| alpha_z_P0 | -4.5 | 0.5 | uniform |
| alpha_z_xc | -1.5 | 2 | uniform |
| alpha_z_beta | -0.5 | 1.5 | uniform |

These are proposal bounds. The target prior is this product density conditioned
on the unchanged beta, size, finite-Y200 and central-column-tail restrictions.
Consequently the joint density is constant on the allowed region, but individual
marginals remain correlated and nonuniform. The linear-proposal diagnostic
accepted approximately 2.18% of proposals. This is the blue-curve version from
the previous comparison, not an independent uniform box or a Gaussian prior.

## Physics and numerical contract

For A = P0, xc or beta,
`A(M,z)=A0*(M200c/1e14 Msun)^alpha_m*(1+z)^alpha_z`.
The pressure is `Pth/P200=P0*q^(-0.3)*(1+q)^(-beta)`, with
`q=r/(xc*R200c)` and `Pe=0.5176*Pth`. Masses are physical M200c.

The original full catalogue, redshift range, 4 R200c projected painting radius,
NSIDE=4096, pixel sampling, 2-arcmin beam, 40 spectrum bins, fsky=0.4 and mask
apodization are preserved. The same scaled finite line-of-sight quadrature,
positive interpolation-cache floor, and independent SO noise algorithm are
copied unchanged from the previous run. The 108 independent column checks
continue to run for every painted row. No failed row is replaced silently.

`prior.json` retains the exact previous support: beta(M,z) in [2.8,50] over the
interpolation domain; relative xc/beta in [0.4,8] over the catalogue enclosing
domain; finite Y200/B12 in [0.003,30] on the original 9x9 grid; and the 1%
central-column tail limit. These cuts still exclude parts of the old SBI box.
Neither the plot nor this replacement claims full uniform coverage of that box.

## Reusing previous simulations without selecting on completion time

On the common support, the old density is proportional to `1/(P0*xc)` and the
new density is constant. `design.py` independently thins the entire frozen old
design with retention probability `P0*xc/(60*4)`. The draw is a deterministic
hash of the old row index with an explicit domain-separated salt. It is applied
before inspecting completed files, including old points that have not yet been
simulated. Thus simulation runtime does not choose the target parameters.

Selected old points occupy every eighth output position until that pool is
exhausted. Remaining positions use fresh scrambled Sobol proposals with seed
20260916 and the unchanged rejection rules. Old and fresh sources therefore
target the same conditional density. Both are randomized space-filling designs;
neither is claimed to retain exact Sobol digital-net balance after rejection.

`restart.py --action import` copies a selected old row only if its exact
parameters, physical code, completed status and split-noise seeds match the new
design. It verifies the copied arrays bit for bit and records the original
source path, source SHA-256, generation manifest and row index in the new HDF5
file. Available old rows that were not selected remain in the previous run.
Selected old points without completed simulations are generated normally.

Reused rows retain their original noise-seed row IDs. Fresh rows use
`1048576 + new_row_index`; preflight IDs remain in the separate 524288 namespace.
The master seed remains 12345. The algorithm and SO spectrum are unchanged.
`check_noise_seeds.py` checks collisions across the complete potential 524k fresh
range, the selected old pool, all preflight rows, and observation seeds.

## Run and validation

The active cluster root is
`/lustre/work/kristero10/halfdome_flamingo_linear_8192_20260915`.
Preparation freezes code, design, previous-design provenance and observations.

```bash
PY=/home/anaconda3/bin/python3
RUN=/lustre/work/kristero10/halfdome_flamingo_linear_8192_20260915

$PY code/prepare.py --root "$RUN" --count 8192
$PY "$RUN/code/audit.py" --root "$RUN"
$PY "$RUN/code/check_noise_seeds.py" --root "$RUN"
$PY "$RUN/code/restart.py" --root "$RUN" --action retire
$PY "$RUN/code/restart.py" --root "$RUN" --action import
$PY "$RUN/code/restart.py" --root "$RUN" --action preflight
# Once the three preflight jobs finish:
$PY "$RUN/code/validate.py" --root "$RUN"
$PY "$RUN/code/restart.py" --root "$RUN" --action production
$PY "$RUN/code/restart.py" --root "$RUN" --action status
```

Retirement selects only this user's jobs with the exact previous RUN_ROOT. All
completed old files and all old scientific inputs are checked before and after
cancellation and remain in their original directory. Production is submitted
only after the fresh six-map operator preflight passes. It uses 64 chunks in
6 mini and 16 mini_B dependency chains (maximum 22 workers), with 26 CPUs,
64 GB and 23:59:00 per worker. A collector depends on every terminal job.
The lower-priority mini_B queue may leave some workers waiting for resources.

## Scaling to 524288

Pass `--count 524288` to `prepare.py` with a fresh run root. The original frozen
old proposal pool, thinning salt, slot pattern, fresh Sobol stream, validation
split IDs and noise IDs are independent of requested count. Thus the first 8192
rows and their noise realizations remain the same. The software supports this
size; no 524k simulation run is submitted by the 8k workflow.

## Figures and inference handoff

`plot_prior.py` reads the frozen design, exact old SBI bounds and saved
cosmology-corrected FLAMINGO fits. It exports all nine marginals, all 36 parameter
pairs, the original three projections, exact numeric values, and provenance.
Every coordinate is plotted in physical linear units. FLAMINGO points are
effective clean-spectrum matches, not posterior confidence regions.

The collector exports clean and independent-noise spectra, theta, validation
split, source-old-row indices and noise IDs. Future SBI should use
`sbi_prior.ExtendedPrior(config, normalization=audit["normalizing_mass_Z"])`.
Its log density is constant on the permitted region and includes the estimated
normalization from the audit, with reported uncertainty. An unconditioned
BoxUniform or the previous trained posterior would describe a different prior.
New compression/training and inference validation remain separate tasks.
