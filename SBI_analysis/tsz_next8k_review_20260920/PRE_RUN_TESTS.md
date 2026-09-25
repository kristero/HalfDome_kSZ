# Tests for the next diagnostic tSZ dataset

No new production run has been submitted. Current statuses distinguish completed
measurements from tests that must still be run. See REVIEW.md for the physics.

## Before starting the new 8k

### P01 - Freeze the complete forward model

Pin HalfDome and XGPaint sources, spherical radius, physical M200c/h conversion, cosmology, pressure-electron factor, catalogue, redshift range, raw/output NSIDE, beam, mask, noise prescription and bin edges.

**Pass:** One machine-readable manifest; intentional differences from the old run enumerated. Do not mix the truncation comparison h=0.6774 with the SBI h=0.68.

**Current status:** Source review completed; new-run manifest not frozen

### P02 - Spherical LOS and integrated-Y identities

Compare chord projection with independent quadrature, a uniform-pressure sphere, and 2pi integral b*y(b) db versus the 3D pressure-volume integral. Probe centre, grazing chords and outside support.

**Pass:** Agreement at 1e-8 relative where well scaled; absolute error bounds near zero; exact zero outside the sphere. This is a numerical identity tolerance.

**Current status:** B12 wrapper and separate finite-radius tests exist; broad-prior integrated-Y suite required

### P03 - Remove hidden dependence on the old LOS integral

Use the inner model's prepared amplitude rather than inner(theta200)/old_column. Exercise beta<=0.7, tiny xc, high beta and all cache corners.

**Pass:** No old beta assertion, 0/0 normalization, NaN or quadrature stall. Any underflow is bounded in observable units.

**Current status:** Issue identified in wrapper; proposed production change not applied

### P04 - Interpolation and boundary reconstruction

Cache positive chord-mean values, multiply by the exact chord outside interpolation, compare linear-z and log-z grids and two refinements over adversarial points plus at least 10000 random columns.

**Pass:** Observable error plus measured reference change below chosen budget on full-map controls; convergence near the moving sphere boundary documented.

**Current status:** Old-cylinder log-z tests passed selected points; spherical extended-prior tests pending

### P05 - Flux-conserving treatment of compact haloes

Compare point painting, pixel-integrated or pre-beam painting at4096, and raw8192/16384 controls. Randomize subpixel halo positions. Apply the beam once; include its wings beyond the physical sphere.

**Pass:** Integrated Y and retained multipoles stable under position/resolution changes; no pixel hit/miss bias. Full-map numerical budget passes at tested points.

**Current status:** Old painter resolution defect measured; alternative renderer not implemented

### P06 - Joint-extreme and catalogue-scale fidelity

Use B12, updated clean FLAMINGO fits, each tail, and 16-32 joint stress cases; compare clean spectra with the same beam/mask at two resolutions, with selected16384 references.

**Pass:** No accepted point silently dropped; failures resolved or support/density explicitly redesigned. A sphere4 versus sphere8 test is model sensitivity, not numerical convergence.

**Current status:** Previous cylinder tests available; new spherical model pending

### P07 - Matched timing and memory pilot

Measure cold/warm cache, painting, transforms, noise, masking, compression and I/O separately at4096 and8192 with identical settings, threads and repeat count.

**Pass:** Per-row p50/p95/p99 and peak RSS recorded; memory-limited concurrency and524288-row storage estimated from measurements.

**Current status:** Matched clean eight-thread timings exist; complete proposed pipeline pending

### D01 - Flat-prior density and failure accounting

Declare independent physical uniforms or a correlated flat-marginal density. Test all nine marginals, pair coverage, normalization and sampler/log_prob agreement. Keep sampling independent of FLAMINGO fit quality.

**Pass:** No undocumented rejection, timeout filtering or Gaussian/log weighting. Exact ranges retained unless a documented support revision is chosen.

**Current status:** Continuous correlated beta sampler verified; full nine-parameter spherical prior not certified

### D02 - Noise splits and observation parity

Unique row/split seeds; independent train/test/observation seeds; matched beam/noise conventions; split auto/cross spectra and shared foreground residual tests.

**Pass:** Means/covariances agree with the declared mock. Decide idealized table draws versus physical observing splits; do not silently double the full ILC table.

**Current status:** Old independent-noise design verified; new operator parity pending

### D03 - SBI serialization, units and row alignment

Round-trip known cases through stored clean/noisy spectra, signed asinh transform, compression and inference; verify parameter order, D_l/C_l factors and array indexing.

**Pass:** Exact row labels and reconstruction within numerical precision; negative cross bins retained. Every test row has a recorded outcome.

**Current status:** New-pipeline regression suite pending

### D04 - Compression fit isolation and rank

Fit scaling/PCA/MOPED only on optimization rows; compare finite-difference and local-regression derivatives, covariance whitening and singular values at B12 and additional anchors.

**Pass:** No test/observation leakage; no hidden regularization manufacturing nine constrained directions. bins40 retained as a reference.

**Current status:** Previous local Fisher rank six is a warning, not a rank certificate for the new model

### D05 - Resume and scale reproducibility

Generate the same small prefix with count8192 and524288; simulate interruption/resume, failed rows and imported outputs.

**Pass:** Identical prefix theta and seed IDs; checksums prevent reuse across geometry/beam changes; writes are atomic and no incomplete row counts as success.

**Current status:** Old workflow tested; new geometry identity must be added

### I01 - Noiseless positive controls

Fit P0 only and P0-beta at fixed remaining parameters against direct likelihood/grid controls; add noise progressively using a small isolated pilot.

**Pass:** Known truths recovered in identifiable settings; disagreement traced before nine-parameter production.

**Current status:** Required new pilot

### I02 - Prior-only and shuffled-data controls

Evaluate the constant prior-mean predictor and prior density; train the same architecture after shuffling parameter-spectrum pairings.

**Pass:** Correctly paired data improve held-out density scores and identifiable parameter recovery over both controls. Weak parameters may legitimately remain prior-like.

**Current status:** Prior-mean baseline computed in this review; shuffled training pending

## Use the diagnostic 8k to evaluate

### I03 - Repeated-noise truth recovery

Use16-32 known parameter settings spanning interior and tails, with16-32 independent noise realizations each. Keep all copies of a clean sky in the same split.

**Pass:** Report conditional bias/coverage per truth and parameter, with binomial intervals; no focus solely on B12 or pooled correlation.

**Current status:** Planned for new diagnostic dataset

### I04 - SBC plus information-sensitive validation

Run at least256 prior-predictive SBC cases and posterior predictive checks; assess conditional coverage, observation dependence and posterior-to-prior information gain.

**Pass:** Calibration assessed with finite-sample/simultaneous uncertainty; SBC alone is insufficient because a prior-only predictor can pass it.

**Current status:** Not done for the completed8k analysis

### I05 - Nested learning curves and estimator robustness

Use fixed held-out truths and nested training sizes256,512,1024,2048,4096 and the remaining pool; compare bins40 and MOPED with multiple training seeds.

**Pass:** Improvement survives seed variation and exceeds prior/shuffled controls; report failures and per-parameter metrics with uncertainty.

**Current status:** Old curves exist; rerun required after model/prior changes

### I06 - FLAMINGO is an out-of-family observation test

Reprocess with exactly the new observation operator, refit clean effective parameters if desired, and check spectral support/posterior predictive residuals before interpreting parameters.

**Pass:** No claim of parameter coverage using effective FLAMINGO gNFW fits as truths. Cosmology/diffuse-gas/model discrepancies are distinguished from network error.

**Current status:** Old observations exist; new-model comparison pending

## Before scaling to 524k

### I07 - Decision to scale

Combine physics, rendering, prior, noise, calibration, information and resource results from the diagnostic8k run.

**Pass:** Launch larger production only after declared numerical budgets, inference controls and operational cost limits pass; extra samples cannot repair a wrong forward model or unidentifiable statistic.

**Current status:** Future decision

