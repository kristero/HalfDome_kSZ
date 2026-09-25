# Catalogue reuse and cache-resolution experiment

This directory contains an isolated experiment. It does not submit the 256-row
dataset. The user requested performance tests before starting that dataset.

The observable remains all 7,900 unbinned multipoles, ell=80..7979, a 2 arcmin
Gaussian beam, output NSIDE4096 and the same fsky=0.4 mask. Raw NSIDE8192 is the
candidate. NSIDE16384 is used only as an accuracy reference. The intermediate
signal harmonic transform still extends to 3*4096-1, as in the validated code.

## What is being changed and why it is valid

`benchmark.jl` has three painting modes. `reread` scans the full catalogue once
per parameter combination, with the validated painter. `read_once` reads each
original chunk once and calls that same painter for every combination. `fused`
also computes the sky direction, R200 angular radius, intersected rings/pixels,
angular separation and logarithmic interpolation coordinates once, then updates
each combination's independent map with its own pressure interpolation.

For fixed cosmology and M200c, the halo mass, redshift, position and R200 do not
depend on the nine pressure parameters. The finite 4 R200 sphere is also shared.
Thus the change is a reuse of identical geometry, not a change of gas pressure,
halo population, beam or observable. This sharing would need revision for a
cosmology-varying dataset or a parameter-dependent painting radius.

Only one catalogue chunk is held at a time. All 85,224,251 selected halos remain
in the calculation. Reading all five Float64 fields for the whole catalogue
would occupy about 3.17 GiB before derived geometry and temporary copies; it is
not needed here. One raw Float64 NSIDE8192 map occupies exactly 6 GiB, so four
simultaneous maps occupy 24 GiB before caches and harmonic workspaces. Four-map
experiments request 64 GiB. Peak RSS is measured, not inferred from this count.

Sharing the catalogue does not justify sharing noise. These clean controls use
the already measured independent split-noise ensembles solely as an error
scale. A future batched dataset must retain each row's own two noise seeds.

## Cache experiments

The axes remain log(theta), log(z) and log10(M/Msun), with identical endpoints,
padding 256, cubic B-spline interpolation of log chord-mean pressure, exact
restoration of the sphere chord, a 1e-300 floor and the same bounded normalized
LOS quadrature. Only the number of interpolation nodes changes.

| Test | theta nodes | z nodes | mass nodes | Values array only |
| --- | ---: | ---: | ---: | ---: |
| Default | 512 | 256 | 128 | 128 MiB |
| Half theta | 256 | 256 | 128 | 64 MiB |
| Half z | 512 | 128 | 128 | 64 MiB |
| Half mass | 512 | 256 | 64 | 64 MiB |
| Half all axes | 256 | 128 | 64 | 16 MiB |
| Quarter all axes | 128 | 64 | 32 | 2 MiB |

Each full-catalogue test includes Battaglia12, the existing fiducial FLAMINGO
fit, the compact/faint extreme and the extended/shallow bright extreme. All
spectra are compared both with the validated default cache and the previous
doubled cache. The additional scalar test uses these four, two combined-tail
cases and the first 16 saved flat-prior Sobol points, with 1,024 off-grid probes
per point per grid. Those are cache probes, not dataset rows.

The primary accuracy check projects the full unbinned spectral difference onto
the local spherical-model MOPED directions at both anchors, using each test
case's own independent SO noise ensemble. Relative singular cutoffs 1e-3 and
1e-6 expose weak-mode sensitivity. A 0.1-noise-unit budget is an engineering
target for numerical error, not a physical prior cut. Passing these tests does
not establish global posterior coverage; the diagnostic must still test that.

## Remaining rendering questions being pursued

The raw 8192 four-map control is also averaged into actual HEALPix parent pixels
at 4096. The bright/shallow and Battaglia12 16384 reference controls provide
8192 and 4096 parent averages. Four versus sixteen fine-pixel samples per 4096
pixel can then be compared. This is finite equal-area quadrature, not exact
analytic integration, and very small cores can still need a better reference.
Building a fine map and then averaging it is an accuracy reference, not itself
a faster production route. It will test the target behavior for a future painter
that integrates directly into coarse pixels without allocating the fine map.

Pixel averaging has a pixel-window response distinct from centre sampling. The
code saves both the averaged-map result and a version with the isotropic
HEALPix pixel window removed before applying the 2 arcmin beam. The latter is
an approximation to test, not an automatically adopted correction. See the
[HEALPix definition](https://healpix.sourceforge.io/html/intro_Pixel_window_functions.htm).
The numerical gate checks unity preservation, total-flux conservation, nested
parent indexing against independent sky-coordinate assignment, and original
versus shared-painter map agreement. No second Gaussian beam is applied.

The noise-weighted default-cache check exposed a 0.40--0.46 noise-unit error
for the bright/shallow model. An additional [edge-continuation experiment](edge_extension/README.md)
tests removal of a derivative discontinuity in the cache's unused exterior.
It preserves physical pressure and the exact sphere boundary. This is a separate
numerical hypothesis, not an adopted production change.

The live catalogue controls also exposed a long final-chunk tail with low CPU
utilization. `work_balance/` tests greedy scheduling of 256-halo blocks instead
of large static contiguous groups, using the same arithmetic, map locks and
default cache. Its gate compares 16- and 256-halo blocks against the original
painter; a full-catalogue spectrum comparison follows. This isolates scheduling
from the interpolation change. Snapshot evidence is in `results/cluster_snapshot.json`;
per-thread SSH inspection was unavailable, so the imbalance diagnosis uses PBS
CPU-time increments and chunk timings, not a captured thread trace.

## Execution and evidence

`plan.json` is the finite 13-control task list. `run.py` verifies frozen source
hashes, claims each task using atomic mkdir and records failures without
discarding or replacing parameter combinations. Four static worker shards
avoid duplicate work. The reference job waits for the speed workers; the
analysis runs after the workers and reference finish. The optional follow-up
audit has its own source manifest and jobs. All PBS scripts use LF newlines.

Use Windows Python and the configured `idark` SSH alias:

```
python fetch.py
```

Do not call `submit.py --submit` again for this root. Inspect the saved
`results/submission.json` instead. Frozen producer files cannot be edited under
running jobs. New results are saved under `results/` and `plots/`; `fetch.py`
downloads spectra and small evidence only, verifying SHA256 hashes.

## Files added

New implementation: `benchmark.jl`, `gate.jl`, `prepare.py`, `run.py`,
`analyze.py`, `submit.py`, `pixel_windows.py`, `fetch.py`, `report_progress.py`
and this README.
New audit: `followup/cache_probe.jl`, `followup/baseline_cache.py`,
`followup/baseline_cache_v2.py`, `followup/launch.py`, `followup/prepare_submit.py`.
Additional isolated tests: `edge_extension/` and `work_balance/` entrypoints,
gates, analysis and deployment scripts; read-only `snapshot_cluster.py`.
Frozen copies: `fullsky_test.jl`, `spherical_truncation_profiles.jl`,
`external_dependencies.json`. Generated inputs, plans, manifests, PBS files,
submission journals and measurement outputs are confined to this directory.
No existing production or prepared-dataset source files are edited.
