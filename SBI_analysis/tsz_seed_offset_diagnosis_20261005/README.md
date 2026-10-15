# Seed-offset diagnosis

This follow-up reuses the frozen 100-observation grid and trained models in
`/lustre/work/kristero10/tsz_robustness_methods_20261004`.
Its primary inference is the original unbounded Merlin model; the bounded model
and both retrained 50-multipole-band versions provide controls.

Read `REPORT.md` for the physical interpretation, and start with
`plots/mechanism_summary.pdf` and `plots/fingerprint_per_seed.pdf`.
The CSV files contain every diagnostic posterior interval. These are controlled
spectral perturbations, not additional independent physical skies.

The two main scripts are readable collections of spectrum transformations:

- `diagnose.py` separates simulation-induced clean and signal–noise differences,
  tests scale blocks, smooth/fine components, reversed changes and interpolation,
  and evaluates the saved networks. It also checks observation reconstruction,
  feature support and local pressure-derivative geometry.
- `extended_controls.py` repeats the smoothing test for each simulation, separates
  clean/signal–noise/noise–noise terms for masks, and checks joint-only averages.
- `audit_inputs.py` streams checksums of the inputs actually consumed. Raw halo
  catalogues are identified by path, byte size and saved painting metadata;
  their full contents are not rehashed because no new painting is performed.
- `make_report.py` writes tables, the physics report, the summary figure and the
  final checked artifact manifest. It can run locally after downloading outputs.

For a fresh cluster rerun, use a fresh output directory, adjust the explicit PBS
root paths, submit `diagnose.pbs`, then `extended_controls.pbs` with an `afterok`
dependency. Run `audit_inputs.py` and `make_report.py` after completion. PBS jobs
request 26 CPUs and use `mini2`; another eligible queue may be selected if it has
available resources. All walltimes are below 24 hours.

`finalize.pbs` records the small incremental job used in this investigation. Its
append option adds only two mean-mask controls and preserves the earlier draws;
it detects already completed controls on a repeat invocation.

`FINAL_COMPLETE.json` and `VERIFIED.json` are the final delivery manifests.
`COMPLETE.json` and `EXTENDED_COMPLETE.json` record earlier stages; their mutable
job-ledger entries can precede later submissions. The final manifest supersedes
those earlier ledger snapshots. The final report describes the repaired report
formatting failure after the third job's inference controls completed.

All seven Python/PBS source files and this README were newly created. Previous
HalfDome maps, trained weights and reports were not edited. Bytecode caches are
discarded; compact numerical outputs are retained for reproducibility.
