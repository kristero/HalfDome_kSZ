# Files created for the tSZ robustness experiment

All research changes for this request are isolated in
`SBI_analysis/tsz_robustness_methods_20261004/` locally and
`/lustre/work/kristero10/tsz_robustness_methods_20261004/` on idark.
The existing seed/mask experiment, original trained models, catalogues, and
painting source are preserved.

## Scientific calculation and inference

| File | Purpose |
| --- | --- |
| `common.py` | Parameter contract, band definitions, transformations, covariance and sky-template helpers. |
| `prepare_idark.py` | Freeze inputs, prepare the 10-by-10 plan, and reuse the seed-100 spectra. |
| `validate_grid.py` | Verify all 100 observations and their fixed-noise and baseline identities. |
| `pipeline.py` | Reconstruct beam-corrected training spectra, prepare compression, train and evaluate the main candidates. |
| `report.py` | Preliminary ranking, parameter intervals, scale response, and simulation/mask/interaction figures. |
| `bounded.py` | Train and evaluate prior-bounded companions without changing the primary unbounded ranking. |
| `merlin_checks.py` | Matched learned-compression cut controls and final comparison of all 24 methods. |
| `final_report.py` | Apply both marginal and joint calibration diagnostics before choosing the final methods. |
| `joint_truth.py` | Measure the joint physical-parameter posterior density rank of the known Battaglia12 truth. |
| `final_prior_test.py` | Test the selected methods on the 1,500 reserved parameter rows, without re-ranking. |
| `signal_audit.py` | Export measured clean-spectrum variation across simulations, masks, and their full grid. |
| `publication_plots.py` | Format tradeoff and parameter-width figures, and export both parameter spaces with Monte Carlo diagnostics; leave inference unchanged. |

## Execution and checks

| File | Purpose |
| --- | --- |
| `dispatch.py` | Select available cluster resources, submit jobs, and monitor completion within the submission cap. |
| `submit_extensions.py` | Submit the additional matched-Merlin and final-comparison jobs with dependencies. |
| `check_numerics.py` | Numerical checks of covariance projection, local Fisher preservation and factorial decomposition. |
| `freeze_code.py` | Hash the scientific source used by the main run. |
| `audit_inputs.py` | Check source data and fiducial identities. |
| `review.py` | Report progress and verify final artifacts, finite intervals, and model hashes. |
| `paint.pbs` | One painting per simulation; produce all ten masked observations from that painting. |
| `pipeline.pbs` | Main compression and inference job. |
| `merlin_checks.pbs` | Additional learned-compression controls. |
| `combine.pbs` | Final comparison, reserved-data tests, and verification. |

## Documentation and generated records

`README.md`, `EXTENSIONS.md`, and this `FILES_CREATED.md` describe the physical
contract, methods, limitations, and file responsibilities.

Generated run records include `plan.json`, `jobs.json`, `extension_jobs.json`,
`input_sha256.json`, `code_freeze.json`, `status.json`, and verification records.
`final_results/INTERPRETATION.md` explains measured recommendations, physics,
and limitations. `final_results/` also contains the verified report copy,
exact archived source snapshot, and current guides.
Derived spectra, features, models, posterior samples, plots, and CSV/JSON
diagnostics are generated under the experiment's data, candidates, results,
signal-audit, and report directories. They are research outputs rather than
edits to the original training data.

The cluster's `code/paint_seed.jl` is an unchanged copy of the completed
seed/mask experiment's painter. The final authoritative report is
`comparison/report/`; completion requires both `comparison/COMPLETE.json`
and `comparison/VERIFIED.json`. The parent report is preliminary.
