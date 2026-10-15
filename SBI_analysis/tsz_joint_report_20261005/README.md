# HalfDome Merlin: combined seed/mask study and Pearson report

The final report combines the completed 10 x 10 robustness experiment, its
directional-offset diagnosis, and the newly computed posterior Pearson test.
All scientific inference was run on idark. The report was typeset and visually
checked locally and is also delivered to the cluster root below.

Cluster root: `/lustre/work/kristero10/tsz_joint_report_20261005`.
Local root: `SBI_analysis/tsz_joint_report_20261005`.
Final report: `output/pdf/HalfDome_Merlin_seed_mask_and_Pearson_report.pdf`.

## Exact meaning of the requested smooth control

This uses the same saved input as the **simulation** column of the earlier
"actual, clean, smooth, fine" panels. The original seed-100/mask-12345 noisy
observation is compared with that baseline plus the Gaussian-smoothed fractional
mean clean difference of simulations 102-118 at mask 12345. The plotted
`smooth_clean_5` control is primary; `smooth_clean_50` is secondary. Both inputs
are reused directly from the frozen diagnostic spectra, with their algebraic
reconstruction and transformed features checked independently (maximum error 0).

This does not smooth seed 100 itself or the complete noisy spectrum. It leaves
the original fine pattern and saved corrected SO noise residual intact. It is
a diagnostic changed input to the **original frozen Merlin**, not the retrained
50-multipole-band model and not a new calibrated inference method.

## New result

The original/sigma5/sigma50 unbounded physical-parameter correlations are:

| Pair | Original | sigma5 | sigma50 |
|---|---:|---:|---:|
| P0-beta | -0.33747 | -0.27660 | -0.31664 |
| P0-xc | -0.64529 | -0.63155 | -0.64224 |
| xc-beta | +0.81307 | +0.79119 | +0.80354 |

The largest primary change is P0-beta, +0.06087 with approximate sampling MCSE
0.00426. Mean 68% width ratios are 1.01224 and 0.99867. The main degeneracies
remain close. The bounded comparison and an inside-prior sensitivity support
the same response direction. MCSE does not include model or training uncertainty.

`pearson_pairs.csv` contains all 144 pair/control/model rows; `pearson_linear.json`
and `pearson_logit.json` contain full matrices, quantiles, physical-density truth
ranks, prior-tail fractions and sampling precision. The primary correlation
uses all 90,000 draws, without prior clipping. Plot arrays retain 4,500 draws;
the full posterior arrays remain on idark under `data/`.

## Execution and reproduction

One new job, `600652.idark`, ran on the free node ansys1 through queue mini2,
requesting 26 CPUs and 12 GB. Application exit status was 0. Wall time was
39.39 seconds and peak RSS 1,187,648 KiB. PBS history is disabled on the server;
completion is verified by the application resource log, finished shell log,
completion marker and hashes, rather than an unavailable PBS historical record.
No repainting or retraining was performed for this new test.

To reproduce the inference on idark, use the existing modern SBI environment:

```bash
cd /lustre/work/kristero10/tsz_joint_report_20261005
/home/kristero10/envs/sbi27/bin/python baseline_pearson.py --threads 26
```

For production, submit `pearson.pbs` after checking current free-node/queue
availability. Do not submit duplicate jobs while one is active. `jobs.json`
records the actual job and the 20-job limit.

To rebuild the document, first obtain the inference JSON/NPZ products and the
delivered evidence directory. In a Python environment with NumPy, SciPy,
Matplotlib, Pillow, ReportLab and pypdf:

```bash
python report_figures.py
python build_report.py
pdftoppm -r 100 -png output/pdf/HalfDome_Merlin_seed_mask_and_Pearson_report.pdf tmp/pdfs/final
```

The builder embeds real math renderings and extracts code directly from delivered
source files. `source_excerpts.json` records line ranges and source SHA256 values.
`page_manifest.json` lists the 36 pages. `collect_evidence.py` can refresh the
local evidence copy from the frozen previous deliveries without changing them.
`--prior-draft` builds only the earlier sections for layout review; it is not a
final scientific delivery.

## Checks, evidence and limits

- Independent Gaussian-draw correlation and fixed-residual numerical gates pass.
- Original and both saved-control features match the previous diagnosis exactly.
- Pearson symmetry, unit diagonal, finite range and output SHA256 checks pass.
- Both ensembles use physical parameter coordinates; joint truth tests include
  the coordinate Jacobian for samples and truth.
- Every PDF page is rendered and reviewed; equations, source boxes, matrices and
  captions are checked for readability and overflow.
- The full-grid recommendation is the retrained band50 Merlin under a 25%
  **geometric mean** width budget. Worst-parameter increases can be much larger.
- Fixed-sky parameter/noise validation, fixed-truth sky sensitivity and posterior
  truth containment are different tests. None alone certifies calibration across
  unseen halo realizations throughout the prior.

Frozen predecessors are `tsz_robustness_methods_20261004` and
`tsz_seed_offset_diagnosis_20261005`. Their completion identities are recorded in
the PDF and evidence manifest. Previous maps, trained models, results and reports
were not overwritten. `FILES_CREATED.md` is the complete inventory for this work.

