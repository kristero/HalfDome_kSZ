FLAMINGO snapshot-to-stack fits: code walkthrough, 8 October 2026

Start with FLAMINGO_snapshot_to_stack_fits_walkthrough.pdf (12 pages).
It explains the executed radii234_contamination_L1_m9_0077_20261008 study,
not a new simulation run or a completed global mass/redshift calibration.

Created files in this separate directory:
  build_walkthrough.py       Report and illustrative stack-plot builder.
  walkthrough_examples.py   Read-only profile rebuild and single-stack refit CLI.
  verify_walkthrough.py     Source hashes, PDF checks and render/contact-sheet QA.
  README.txt                This file.
  FLAMINGO_snapshot_to_stack_fits_walkthrough.pdf
  stack_fit_example.png and stack_fit_example.svg
  example_verification.json Six local pressure/electron 2/3/4R fit checks.
  walkthrough_verification.json  Provenance, SHA256 and visual QA status.

Run from this directory in WSL:
  /home/cbllover/miniconda3/bin/python -B walkthrough_examples.py --bin-low 14.5 --radius 4
  /home/cbllover/miniconda3/bin/python -B walkthrough_examples.py --bin-low 14.5 --radius 3 --field electrons

The CLI prints JSON. It does not write files, fetch particles or submit jobs.
It rebuilds profiles from existing additive shell sums, checks the saved stack,
refits with the original optimizer and shows one correlated parameter draw.
The two physical-profile checks allow 2e-8 relative differences across physical
constant/software environments; the actual largest local difference is recorded.
Stored-profile refits use a separate, tighter 1e-11 coefficient comparison.

Rebuild this documentation only:
  /home/cbllover/miniconda3/bin/python -B build_walkthrough.py
  /home/cbllover/miniconda3/bin/python -B verify_walkthrough.py
Inspect all rendered pages under HalfDome/tmp/pdfs/procedure_walkthrough_20261008
before using verify_walkthrough.py --reviewed. Rebuilding resets the QA status.

All report source paths refer to the completed sibling study. Some pipeline
settings and array sizes are hardcoded; study_config.json alone is insufficient
to retarget that analysis. A new scientific run must have a fresh output root.
Do not run fit_study.py or extraction main() inside the frozen completed study
just to explore the saved data: use walkthrough_examples.py instead.

The whole-halo bootstrap describes empirical sensitivity of mean-fit parameters.
It is distinct from the selected individual-halo coefficient distribution, and
neither establishes a fully calibrated covariance or stochastic painting model.
The guide documents current limitations, including unresolved low-mass bins,
star-forming gas with zero native Y/ne, overlapping all-gas spheres and one z.

No source code or outputs in the completed sibling study were modified.
No new cluster jobs, full-snapshot downloads or snapshot particle reads occurred.
