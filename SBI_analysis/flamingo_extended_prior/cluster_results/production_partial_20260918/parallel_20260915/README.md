# Increased parallelism for the existing 8k production run

The user authorized more concurrent dataset-generation jobs on 2026-09-15.
This changes scheduling only. It does not change the frozen generation code,
prior, catalogue, cosmology, map definition, parameters, row IDs or noise seeds.

Production root:
`/lustre/work/kristero10/halfdome_flamingo_extended_8192_20260915`.

The new scheduling tool and audit are outside the frozen `code/` directory:
`tools/parallel_20260915/` within that production root. A local copy of the audit
is kept in this directory's `cluster_results/` subdirectory.

## Queue allocation

The live PBS configuration allows six running jobs per user in `mini` and
sixteen in the lower-priority `mini_B` queue. Both queues use GroupA nodes.
The normal queue has priority 20; the background queue has priority -10.
The new schedule therefore has six normal lanes and sixteen background lanes,
with an upper bound of 22 concurrent production jobs. PBS decides how many can
actually start as CPU and memory resources become available.

Every worker retains 26 allocated CPUs, 26 Julia threads, 64 GB requested memory,
and a 23:59:00 walltime. GroupB had no free 26-CPU slot when inspected; no jobs
were moved to `mini2`. The expanded schedule was observed with eleven production
jobs running: six in `mini` and five in `mini_B`, with eleven additional lane
heads queued for CPUs. Read `cluster_results/audit/status.json` for the timestamped
snapshot; this is not a live status display.

## What the tool changed

`increase_parallelism.py` first verified the generation manifest and quality
gate. It checked that chunks 0 through 3 were already running and that all other
production jobs were held by their dependencies. It then:

1. Placed user holds on the old pending schedule, leaving running jobs alone.
2. Submitted 60 replacement chunk jobs and one replacement collector under
   user holds, using the existing frozen PBS scripts.
3. Verified all resource requests and the new dependency graph. Each of the
   64 chunks occurs once; the 22 disjoint lanes cover every chunk.
4. Removed only the old held jobs, descendants first, then released the new
   jobs. The four original running jobs retained their job IDs and progress.
5. Rechecked hashes of all frozen inputs and all 90 row files that existed at
   the start of the transaction. Every hash matched.

The original running jobs are `597627.idark` through `597630.idark`. Replacement
production jobs are recorded individually in `cluster_results/audit/schedule.json`.
The new collector is **597775.idark**, dependent on all 22 terminal jobs.
The original `production_submission.json` is retained as historical provenance;
the schedule in this audit supersedes its pending job IDs and collector ID.

## Why this preserves the physics and noise

Every worker executes the same pressure model, line-of-sight integration,
painting, beam, mask and spectrum calculation. Scheduling changes which rows
are processed simultaneously, not their physical definitions.

Noise seeds are derived from the immutable row ID and split, rather than the
worker number or execution order. A row therefore keeps its assigned SO noise
draw when moved to a different scheduling lane. The existing chunk locks and
atomic row publication remain active, and no duplicate chunk jobs were released.
The frozen manifest and all per-row numerical checks remain enforced.

## Files

- `increase_parallelism.py`: scheduling-only driver with dry-run default,
  staged replacement, immediate action journaling and pre-retirement rollback.
- `README.md`: this operational and scientific description.
- `cluster_results/audit/plan.json`: original schedule, queue settings and hash snapshot.
- `cluster_results/audit/schedule.json`: submitted job IDs, dependency graph, actions
  and verification results.
- `cluster_results/audit/status.json`: observed running, queued and dependent jobs.

The tool deliberately refuses to overwrite an existing transaction and checks
the specific four-lane starting state. It is not a general resubmission command
for a later stage of the run. If a failure occurs after retiring old jobs,
inspect the recorded transaction before taking recovery action; never blindly
resubmit all chunks.
