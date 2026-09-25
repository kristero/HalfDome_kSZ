# Resolution and speed preflight

Raw NSIDE 8192 is the candidate for catalogue painting. NSIDE 4096 remains a
candidate if the retained data vector is sufficiently insensitive to its
rendering error. NSIDE 16384 is an accuracy reference only, using the already
submitted spherical controls. All these maps are beam-filtered and synthesized
onto the same NSIDE 4096 observation grid. The prepared 256-row diagnostic is
not started by this workflow.

## What is submitted

One 2-CPU, 8-GiB runtime gate, four 26-CPU, 64-GiB workers, and one dependent
analysis job. Each worker has a 23:59 walltime; the gate has 30 minutes and the
analysis has two hours. Workers share a finite, locked list of 80 controls:

| Controls | Purpose |
|---|---|
| 2 anchors | Battaglia12 and historical FLAMINGO-fit parameters; raw 4096/8192 paired within each task; 128 independent split-noise pairs per anchor |
| 8 extreme maps | Compact, extended/shallow, combined tails, and high-amplitude tails, each at 4096 and 8192 |
| 4 refined caches | Double all three interpolation axes for the two anchors and the compact/extended cases |
| 64 derivative maps | Two anchors, eight shape/evolution parameters, two central-difference step sizes, two signs; all at 8192 |
| 2 performance maps | Remove unused temporary map; compare 26 and 13 application threads |

The P0 derivative is analytic: y is proportional to P0, so the clean power
derivative is 2 D_ell/P0. The eight other derivatives use 0.5% and 0.25% of the
declared prior width. This removes eight unnecessary full-map derivative jobs.
The historical FLAMINGO fit is a stress-test location, not a newly inferred
spherical-model fit. These controls do not retrain a nine-parameter SBI model.

The gate reruns the small Julia and Python checks using the archived cluster
runtime before any full-catalogue worker is released. Controls are never
silently redrawn after a failure. Source/settings changes require a fresh root.
An interrupted noise sequence retains completed draws; the catalogue map is
repainted on resume. A scheduler timeout can leave controls incomplete; the
report exposes this and the worker can resume the same frozen design later.

```bash
python submit.py --submit    # local machine with SSH/scp; submit once
python fetch.py              # retrieve only small outputs and scheduler status
```

The source snapshot and each accepted PBS identifier are saved on the cluster
before the next submission. There is no dataset-generation submission path.

## How 4096 versus 8192 is decided

The relevant statistic is a shift in the actual compressed observation,

    Delta t = t(D_4096) - t(D_8192)
    d^2 = Delta t^T Cov(t)^(-1) Delta t.

It is measured relative to the scatter of one noisy observation, not the much
smaller uncertainty of the average of many mocks. Two complementary compressors
are evaluated:

1. The existing frozen, signed-asinh unbinned MOPED transform, without refitting
   it separately to either resolution. This answers what that transform notices.
2. A spherical-model local linear-D_ell MOPED/Fisher subspace constructed from
   the new derivatives. This tests parameter-sensitive directions of the new
   model. It does not certify a full nonlinear posterior or all nine individual
   marginal constraints. Weak directions and rank-cut sensitivity are reported.

Both use individual multipoles, ell=80..7979. The ell-max sweep is 1000, 1500,
2000, 3000, 4000, 5000, 6000 and 7979. Restricting existing MOPED weights to a
smaller interval does not constitute retraining the old SBI network. The local
MOPED is constructed at each cutoff and its retained information is reported.
Covariance uses an OAS shrinkage operator implemented by a small Woodbury solve,
without allocating a dense 7900-by-7900 inverse.

Each anchor has 128 independently generated noise pairs, with identical seeds
at the two resolutions. Draws 0..63 estimate covariance, and draws 64..127
validate the compressed scale/paired shift. The two split noises are independent.
The covariance is conditional on a fixed sky: cosmic variance, foreground
residuals and model discrepancy are excluded. Two splits each retain the
historical tabulated SO N_ell, multiplier 1. A half-depth observing split with
twice that N_ell is a separate survey-noise choice, not silently substituted.

A small shift at lower ell-max could justify 4096 if the discarded high-ell
data carry acceptably little information for the desired constraints. A small
shift in the old compressor alone is insufficient: it can discard directions
that become informative under the new model. Conversely a large raw-spectrum
distance does not imply an equally large parameter bias. No universal numerical
tolerance or NSIDE requirement is imposed by these scripts.

## Why a 2 arcmin beam does not settle it

The continuous LOS integral is computed before painting. The current renderer
then samples that column at pixel centres. A compact halo can be missed or
oversampled depending on its position relative to those centres. Applying a
beam to that sampled map cannot reconstruct the continuous flux that was lost.

For a Gaussian, B_ell^2 = exp[-ell(ell+1) sigma^2], with
sigma = FWHM/sqrt(8 ln 2). At 2 arcmin, B_ell^2 is 0.783 at ell=2000,
0.376 at 4000, 0.111 at 6000 and 0.0205 at 7979. The useful intermediate
multipoles can therefore retain rendering errors despite the suppressed tail.
See the [healpy Gaussian-beam definition](https://healpy.readthedocs.io/en/latest/generated/healpy.sphtfunc.gauss_beam.html).

The [HEALPix harmonic-analysis recommendations](https://healpix.sourceforge.io/html/fac_anafast.htm)
distinguish band-limited, already smoothed skies from unband-limited input, for
which sampling introduces aliasing. A 4096 output grid near ell-max=8000 is not
in itself disallowed; the question here is accuracy of the unsmoothed halo
painting that precedes it. Merely increasing NSIDE also need not resolve every
extreme compact profile.

Reanalysis of the completed spherical B12 controls gives:

| Last included multipole | Conditional binned-noise distance | Maximum included band difference |
|---:|---:|---:|
| 1879 | 0.669 | 0.605% |
| 2879 | 2.489 | 1.270% |
| 3879 | 6.436 | 2.369% |
| 4879 | 8.865 | 4.148% |
| 7979 | 9.410 | 17.501% |

These use complete existing bins and the saved 64-draw conditional B12 noise
covariance. They are not unbinned MOPED results or posterior biases. The main
change accumulates before the most strongly suppressed tail. This can coexist
with an earlier negligible low-ell resolution test. See
`plots/initial_resolution_cutoffs.pdf` and `results/initial_evidence.json`.

## Numerical changes and established tests

The source snapshot uses the HalfDome spherical-truncation update: gas is
integrated only inside 4 R200c, not throughout a cylinder. The cache stores the
chord-mean pressure, restores the exact chord after interpolation, and uses
log(theta), log(z), log10(M) coordinates. Default dimensions remain
512 x 256 x 128, padding 256. The four refinement controls use
1024 x 512 x 256: eight times as many nodes. This is a convergence test, not an
automatic replacement of the default interpolator.

This update adds a finite 4096-evaluation quadrature budget and checks the
returned error against rtol=1e-10; failures are explicit. It does not change the
physical pressure normalization. With l=x sinh(u), the radial Jacobian is
r=x cosh(u). If g=log[r p(r)] and c is its maximum, the numerical integral
K=integral exp(g-c) du is rescaled back as J=exp(log(2K)+c). At x=0 a log-radius
integral handles the integrable cusp. The peak scaling is undone exactly in
the formula; it never rescales the halo's physical P0.

The bounded implementation passed 10,180 independent reference columns, with
maximum relative error 7.26e-13, and 36,864 projections spanning all 512 prior
corners. An intentionally unresolved positive integrand triggers the failure
path. These scalar checks are not proof of full-map convergence.

Previous CLASS-SZ pressure-profile comparisons agreed to 4.3e-15 after matching
mass and parameter conventions. That check does not establish equality of a
halo-model spectrum and a particular catalogue. Previous sparse HEALPix tests
also compared continuous pre-beam convolution and child-pixel integration;
they are retained as evidence, not claimed to be a completed full-catalogue
replacement. A full-catalogue positive pre-beam renderer remains a separate
development route if raw-grid refinement proves inadequate. The failed naive
radial-FFT pre-beam method is not adopted here.

## Speed and resource measurements

The completed four-thread clean B12 controls took 958.3 seconds at 4096 and
2558.5 seconds at 8192, with peak RSS 9.43 and 19.91 GiB. That is 2.67 times the
walltime for this operator, not an eight-fold scaling rule or a noisy-row
benchmark. A single Float64 map takes 1.5 GiB at 4096 and 6 GiB at 8192.

The new controls time cache construction, allocation, catalogue/painting,
map-to-alm, beam, output synthesis, masked analysis and each noise realization
separately. Full jobs record peak RSS. Benchmarks compare application threads
and spectra, not just walltime; a speedup is accepted only if spectra agree.

Implemented candidates:

- Remove an unused full-sky temporary map when mass/bin-map outputs are off.
  This saves one 6-GiB allocation at 8192. The actual entrypoint's small-map
  outputs agreed exactly; full-map spectrum and RSS comparisons are queued.
- Reuse masked signal alms for repeated noise draws. Linearity gives
  C(s+n1,s+n2)=C(s,s)+C(s,n1)+C(s,n2)+C(n1,n2). The independent small test
  agreed with the original map route to 1.94e-16, and the first full-size draw
  is checked again. This primarily accelerates repeated-noise validation;
  it is not asserted to speed a fresh one-noise-pair row by the same factor.
- Use the exact P0 derivative instead of repeatedly painting amplitude changes.

Next options, guided by measured phase timings: reuse catalogue geometry and
mass/redshift transformations across parameter changes; keep a Julia process
alive for several models; precompute mass/redshift interpolation coordinates
per halo; investigate ring-lock contention. Six Float64 geometry arrays for
85 million haloes already cost about 3.8 GiB. A halo-to-pixel association cache
may be much larger and needs a size audit before materialization. A dedicated
amplitude-only iteration can rescale an existing clean map, with noise added
afterward, because y is linear in P0. These are proposals, not measured speedups.

## File inventory

- `make_plan.py`, `plan.json`: parameter points, cutoffs and resource-independent test definitions.
- `spherical_truncation_profiles.jl`: isolated spherical wrapper with bounded, checked quadrature.
- `fullsky_test.jl`: frozen spherical cache and point painter from the previous preflight.
- `control.jl`: timed maps, paired resolutions, noise algebra and allocation benchmark.
- `worker.py`, `worker.pbs`: locked, resumable catalogue controls and PBS resources.
- `analyze.py`, `analyze.pbs`, `unbinned_moped.py`: unbinned cutoff/MOPED, reference and timing analysis.
- `test_quadrature.jl`, `test_noise_algebra.jl`, `test_load.jl`, `test_workflow.py`,
  `test_analysis.py`, `verify_gate.py`, `gate.pbs`: local and cluster-runtime verification.
- `submit.py`, `fetch.py`: frozen deployment, journaled submissions and small-result retrieval.
- `initial_evidence.py`: completed-control cutoff evidence and its plot.
- `bootstrap.py`, `design_tests.py`, `projection.py`, `inputs/`: source snapshots and independent references.

All files are in this new isolated directory. Existing production sources are
not rewritten by this validation workflow. `manifest.json` lists exact hashes,
and `external_dependencies.json` records the archived runtime/source dependencies.
