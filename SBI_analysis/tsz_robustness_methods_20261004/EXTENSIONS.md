# Matched Merlin controls and final combined ranking

The original 17-candidate run is augmented with six matched Merlin controls:
lower cuts at ℓ=280, 580, 1080; upper cuts near ℓ=5000 and 4000; and approximate
discovery-sky signal augmentation. These use the same learned Merlin head,
SO noise-only whitening, pressure prior, training split and three training
seeds as the full-range 50-multipole-band Merlin control. Comparing these
controls isolates the effect of cutting scales within one compression
architecture. The MOPED cut controls separately isolate cuts within MOPED.

The extra job reuses all 100 validated observations and the existing banded
training spectra. No halo is painted again. The approximation in signal
augmentation is the same B12-template transfer across pressure parameters
described in README.md. For the augmented Merlin control, the transformed
clean signal is subtracted when estimating the asinh scale, so that scale
remains the SO residual MAD; the fixed MOPED augmented candidate uses total
signal-plus-noise variation for that scale. These are invertible input
transforms, but can affect the finite network's training accuracy.

`merlin_checks.py screen` prepares/trains/evaluates the six controls in one
26-CPU PBS job. `merlin_checks.py combine` waits through a PBS afterok dependency
on that job and the original pipeline, then ranks **23 new candidates plus
the original Merlin** together. Selection remains restricted to the original
6 × 5 discovery set; neither validation factor is used for tuning. It refines
selected posteriors to 90,000 samples and makes their prior-bounded companions.

The study uses **12 submitted jobs total**: nine paintings, the original
inference, matched Merlin controls, and the dependent final comparison.
`jobs.json` and `extension_jobs.json` record them; `submit_extensions.py`
checks the combined submission count against the user-authorized cap of 20.

Final artifacts are in:

`/lustre/work/kristero10/tsz_robustness_methods_20261004/comparison/report/`

The parent `report/` remains the completed preliminary 18-method comparison.
The `comparison/COMPLETE.json` marker hashes the final 24-method report;
`comparison/VERIFIED.json` checks artifacts, finite posterior intervals, and
trained model hashes. Source data/models are reused through directory links,
without changing the physical catalogue or painting contract.

For progress, run `code/review.py`; for final verification run
`code/review.py --root <cluster-root>/comparison --verify` with the sbi27 Python.
The parent job can complete before the final dependent comparison; the final
comparison marker is the completion criterion for this request.

Final selection screens both marginal and joint random-reference diagnostics.
The user-confirmed recommendation uses a **hard 25% cap** on the geometric mean
of the 68% width ratios across discovery cases and parameters, with the worst
individual-parameter increase reported separately. Eligible methods must
improve both truth displacement and sky scatter over original Merlin. Within
that cap, the method with the lowest quadrature discrepancy is recommended.
If none qualifies, the baseline is retained with an explicit no-improvement
flag. The preliminary soft width-penalty score is superseded in the final
comparison; alternatives under 10% and 50% budgets are also recorded.
The 1,500 even-indexed held-out parameter rows are validation for this screen;
the 1,500 odd-indexed rows are reserved for the final selected-method test and
do not cause re-ranking. Joint theta-space HPD levels of B12 truth are measured
using 6,000 new draws per sky, with the correct parameter-transform Jacobian,
to distinguish a displaced marginal median from joint density exclusion.

Sky validation holds factors out of covariance/template fitting and selection.
Some single-factor validation cases were visible in the previous 19-observation
study that motivated this experiment; this is not an untouched blind analysis.
The held-out 4 × 5 joint combinations are newly constructed here. Any signal
augmentation recommendation is limited to the tested B12 skies and the stated
approximation, rather than a calibration claim across physical new skies at
arbitrary pressure parameters.

`publication_plots.py` replaces overlapping tradeoff annotations with short
labels for the original and recommended methods and a family legend. It also
shows all nine parameter-wise width ratios, so the average-width cap cannot
hide the cost to an individual parameter. This runs after the final bounded
parameter test and before the final report hashes are recorded.
