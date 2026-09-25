# Catalogue reuse: verified progress

Snapshot: 2026-09-21T13:59:29.066361+00:00

The 256-row dataset is held. All tests retain ell=80..7979 and unbinned MOPED.

## Completed checks

Shared versus original painter: maximum relative map L2 error 1.98e-31 on 256 synthetic halos and four pressure models at NSIDE128. This is a small-map equivalence test; full-catalogue parity is reported separately below. Geometry is identical across those models.

HEALPix nested-child tests passed: all child directions select the expected parent, a constant map remains constant and equal-area averaging preserves integrated flux.

Scalar cache audit: 22 parameter combinations, 3 grids, 1024 common off-grid points per combination and grid (67,584 comparisons). The normalized finite-sphere quadrature remains unchanged.

| Grid | Largest error / central y of that halo | Median cache build [s] |
| --- | ---: | ---: |
| 512 x 256 x 128 | 0.00108855 | 8.573 |
| 256 x 128 x 64 | 0.00108858 | 1.016 |
| 128 x 64 x 32 | 0.0030185 | 0.116 |

These are pointwise errors over the complete cache domain, including low-mass/low-z corners outside the actual catalogue. They include the historical angular floor and do not replace beam-, mask- and noise-weighted spectrum comparisons.

![Scalar cache accuracy](plots/cache_probe_accuracy.png)

## Newly completed observable-level cache check

Default versus doubled cache at the same raw NSIDE8192, with the same beam and mask. Each model uses its own 64 held-out SO split-noise draws. Values below are local linear MOPED shifts in noise units, across the two anchor compressors; not posterior biases.

| Model | Five retained directions | Nine retained directions |
| --- | ---: | ---: |
| Battaglia12 | 2.439e-05 to 2.679e-05 | 2.634e-05 to 2.965e-05 |
| FL_L1_m9 | 0.0001975 to 0.0001976 | 0.0002008 to 0.0002028 |
| compact | 8.891e-14 to 8.934e-14 | 8.967e-14 to 9.401e-14 |
| extended_shallow | 0.4046 to 0.4046 | 0.4237 to 0.456 |

The bright/shallow model fails the provisional 0.1-noise-unit interpolation budget even at the current default cache. This is a renderer-accuracy question; no parameter combination has been removed from the flat prior. The smaller-cache decision must use the full-sky results, and the bright case may need a different grid.

![Noise-weighted cache accuracy](plots/cache_noise_accuracy.png)

## Full-catalogue controls

Completed: 2/13.

| Task | Catalogue + painting [s / sky] | Maximum relative spectrum error vs old 8192 | Largest MOPED shift |
| --- | ---: | ---: | ---: |
| shared_geometry_four_profiles | 405.73 | 7.627e-16 | 6.718e-12 |
| half_theta | 409.35 | 3.174e-05 | 3.076 |

Completed half-theta test (256 x 256 x 128), relative to the old default cache:

| Model | Relative spectrum norm change | Largest tested MOPED shift |
| --- | ---: | ---: |
| Battaglia12 | 9.8001e-07 | 0.00020279 |
| FL_L1_m9 | 2.0729e-06 | 0.0012067 |
| compact | 8.2638e-07 | 1.6825e-12 |
| extended_shallow | 3.1744e-05 | 3.0757 |

Reading mass/redshift and positions took 7.324 s, selection 1.615 s, and painting 1627.475 s for all four maps. Reading alone is a small fraction of this workload. The large last-chunk tail is the motivation for the separately gated greedy-block scheduling experiment.

Spectrum differences for altered grids or NSIDE are approximation changes, not a pure refactoring-equivalence criterion. Full process wall time includes Julia startup, compilation, cache creation, map allocation, transforms and, in selected controls, additional pixel-averaging experiments. Clean-signal timings exclude per-row noise.

## Still pending

- Complete serial/read-once/shared-geometry runtime comparison and independent repeat.
- One-axis and combined cache reductions, checked over the full multipole range.
- Full-catalogue parent-pixel comparisons and bright/Battaglia12 16384 references.
- Global renderer error across the prior and held-out SBI calibration after the numerical tests.

No lowering of ellmax, narrowing of priors, or dataset submission has been used to obtain speed.

The original auxiliary baseline-analysis attempt failed from a Python module-name collision. The corrected auxiliary entrypoint completed; its old log is retained. The simulation jobs and physical model were unaffected.

Additional tests now running: the cache-exterior continuation that removes a derivative kink, and greedy scheduling of small halo blocks. Their small numerical gates passed. The continuation preserves interior values bitwise and the physical support; greedy scheduling agreed with the original painter to below 9e-17 in relative map norm on 1024 synthetic halos. Full-catalogue validation of these two additions remains pending.

Implementation, physics, file list and cluster commands: [README](README.md). Numeric evidence: [progress.json](results/progress.json). The completed four-child pixel experiment is reported separately in [PIXEL_PROGRESS.md](PIXEL_PROGRESS.md); it is not yet a converged coarse-pixel renderer.
