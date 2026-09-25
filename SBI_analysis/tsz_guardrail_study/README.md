# tSZ guardrail study

Start with [RESULTS.md](RESULTS.md) for measured findings and
[METHODS.md](METHODS.md) for equations, assumptions, guard-by-guard decisions
and primary references. Figures are supplied as editable-text PDF and PNG in
[plots](plots). Machine-readable measurements and provenance are in `audit/`.

The active 8,192-row production dataset has **not been changed**. This study
does not certify the extended candidate for a new production run. In particular,
the original point-sampling renderer exhibits a substantial resolution change
even for Battaglia12. The candidate's `numerical_certification` is explicitly
false; missing convergence evidence cannot pass the numerical gate.

## Scientific choices

* The proposal is uniform in the physical values of all nine parameters.
  Conditioning on finite untruncated halo energy makes the three beta-related
  marginals nonuniform; no Gaussian or log-normal weights are used.
* The six extended limits are experimental coverage choices, not claimed
  singularities or observational limits. FLAMINGO fit values are overlays.
* Numerical accuracy is measured in the original 40 clean, masked, beam-smoothed
  bandpowers with a specified SO-noise covariance. A numerical budget of 0.1
  is an explicit precision requirement; 0.05 and 0.3 are reported for sensitivity.
* FIRAS mean-y measurements remain independent diagnostics. They are not
  silently imposed on an exploratory simulation prior.
* A failure to resolve a continuous profile is a renderer issue. A weak signal
  or a poorly identified parameter is not, by itself, an unphysical model.

## Reproduce the analysis

Cluster study root:

```text
/lustre/work/kristero10/tsz_guardrail_study_20260915
```

Source lives in `code/` on the cluster and directly in this local directory.
Dependencies and paths are deliberately pinned to the existing campaign in
`protocol.json`; this is a reproducible workspace study, not a standalone
installation package. Python needs NumPy, SciPy, Matplotlib and mpmath. The
catalogue and pixel diagnostics additionally use the campaign's compatible
Astropy/healpy stack. Map experiments use its Julia environment and XGPaint.

Refresh tables, validation and plots from completed experiments:

```bash
/home/anaconda3/bin/python3 \
  /lustre/work/kristero10/tsz_guardrail_study_20260915/code/finalize.py \
  --root /lustre/work/kristero10/tsz_guardrail_study_20260915
```

Run or resume a paired map experiment in an isolated study allocation:

```bash
qsub -v STUDY_CASES=Battaglia12,STUDY_GRIDS=historical1 \
  /lustre/work/kristero10/tsz_guardrail_study_20260915/code/run_maps.pbs
```

The PBS default is eight cores, 64 GiB and 23:59:00 in `mini2` for this study.
The 16384 sampling references explicitly request 128 GiB. Successful cases
are skipped on resumption; per-case locks prevent duplicate writers. A failed
case retains its assigned parameter vector and error, rather than being
replaced by a more convenient draw. The active production directory is never
used as a study output root.

`run_diagnostics.pbs` recomputes the support, quadrature, sparse-pixel and
mean-y audits. These are real calculations on the cluster, not unit tests.
For local review, `fetch.py` retrieves the diagnostic arrays and spectra while
leaving the large map/cache files on the cluster. `finalize.py --root <directory>`
then refreshes the report. Keep the initial snapshot and original comparison
inputs; do not recreate them from a later production version.

## Files added

| Source file | Purpose |
|---|---|
| `protocol.json` | Parameter order, proposed bounds, numerical budgets and frozen operator paths. |
| `prepare.py` | Creates initial source/input snapshots and specified parameter examples. |
| `analytic_prior.py` | Analytic support, physical-volume sampling, normalized density and covariance distance. |
| `evidence_gate.py` | Point-specific numerical admissibility and exact fixed-covariance amplitude scaling. |
| `audit.py` | Prior-volume scan, covariance/bootstrap construction, evolved guard metrics and LOS stress tests. |
| `pixel_experiment.py` | HEALPix random-position flux/power experiments and effective-area diagnostic. |
| `mean_y.py` | Independent full-catalogue thermal-energy/mean-y diagnostic with refinement checks. |
| `los_support.py`, `los_support.pbs` | Endpoint sensitivity on catalogue hull vertices and occupied-bin means. |
| `map_experiment.jl` | Complete-catalogue interpolation and raw-pixel sampling comparisons. |
| `run_maps.py` | Isolated execution, source snapshots, locks, timing, memory and failure records. |
| `run_maps.pbs` | Cluster allocation for clean map experiments. |
| `run_diagnostics.pbs` | Cluster allocation for the analytic and sparse-pixel diagnostics. |
| `refresh_results.pbs` | Cluster allocation to refresh the measured report and figures. |
| `summarize.py` | Publication figures, spectra comparisons and conditional amplitude example. |
| `verify.py` | Independent integral identities, missing-evidence rejection and frozen-reference reproduction. |
| `finalize.py` | Results, completion state and artifact hashes. |
| `fetch.py` | Retrieves small scientific outputs without downloading full map caches. |
| `METHODS.md` | Detailed physical derivations, evidence and limitations for manuscript use. |
| `README.md` | Scientific choices, reproduction instructions and file index. |

Generated files include `RESULTS.md`, `artifact_manifest.json`,
`changed_files.txt`, `inputs/`, `audit/`, `maps/`, `plots/` and `logs/`.
The three files in `baseline/` are frozen copies of the pre-existing prior
and normalized LOS implementation, not edits to those original files.

## Limits of the evidence

Agreement at two numerical resolutions is a measured convergence step, not
a proof of the continuum limit. The reference SO covariance is conditional on
one Battaglia12 catalogue and excludes cosmic variance and model discrepancy.
Its bootstrap ranges measure sensitivity to the finite noise ensemble, not
the full uncertainty of an arbitrary new SBI posterior. FLAMINGO markers are
effective nine-parameter spectrum fits with an approximate cosmology
correction; they are not nine independently measured halo-pressure constraints.
These qualifications should accompany figures or numerical sigma values used
in a paper.
