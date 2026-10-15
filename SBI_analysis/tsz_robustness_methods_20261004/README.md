# SO tSZ simulation and mask robustness on idark

This experiment compares Gaussian HalfDome simulations 100, 102, …, 118 with
mask seeds 12345–12354, a full 10 × 10 grid. Simulation 100 with mask 12345 is
the original data-generation baseline. Pressure is Battaglia12, M200c, all
available redshifts, with exactly the physical selection and finite spherical
projection of the completed seed/mask pipeline.

The two fixed noise draws are the trained split cross-spectrum of **one SO
baseline deproj0 prescription**. They are held fixed for every sky. The same
Gaussian-beam correction and signed pseudo-spectrum enter every observation.
Changing simulation and mask measures spatial and simulation variation,
conditional on those noise draws; it does not measure independent-noise coverage.

## Execution

The cluster root is `/lustre/work/kristero10/tsz_robustness_methods_20261004`.
Code is copied into `code/`, and `prepare_idark.py` freezes source hashes, the
scientific plan, and nine catalogue tasks. The ten completed masks of simulation
100 are reused. Each other simulation is painted once in one PBS job; all ten
masks are then applied to that painting. No catalogue or full-sky map is saved
again. `validate_grid.py` checks hashes, fixed noise, and the original same-seed
baseline spectrum before inference can start.

`dispatch.py` inspects free CPUs and memory in GroupA and GroupB and submits to
`mini`, `mini2`, or `mini_B`. It performs no numerical science on the login node.
Nine painting jobs plus one checkpointed inference job require ten submissions;
the dispatcher has a hard limit of twenty and stops on failed completion checks.
Jobs request 26 CPUs, 64 GiB for painting or 48 GiB for inference, and 23:59:00.
The new data footprint is estimated below 6 GiB; pipeline memory below 10 GiB,
excluding the separately reserved Julia painting memory. Existing results stay
in their original directory.

Inside the inference job, `pipeline.py` reconstructs all 64,440 corrected training
spectra from the original uploaded clean, production, and noise-only data. It
stores 158 bands of width 50, using weights proportional to 2ℓ+1 within each
band. The original 61,440-row training pool, 3,000-row held-out parameter test,
prior, parameter order, three training seeds, MAF, and training settings are
preserved. Changing compression requires retraining; zeroing inputs in a frozen
Merlin network would not be a matched scale-cut estimator.

All 17 candidates are compared with the original uncut unbounded Merlin model:

- Noise-only and full-noise-covariance MOPED controls over the full ℓ range.
- Upper cuts near ℓ=6000, 5000, 4000, 3000; lower cuts at 280, 580, 1080;
  and a combined 580–5000 cut. Exact inward-rounded band limits are recorded.
- Smooth high-ℓ covariance inflation, reducing small-scale weight without a cut.
- Simulation-, mask-, and joint-variation-aware MOPED; a weaker joint variant.
- Approximate sky-signal augmentation with joint-variation-aware MOPED.
- A learned Merlin head on all 158 noise-whitened bands.

Unbounded inference is primary. `bounded.py` trains prior-bounded companions for
the selected methods and evaluates the saved original bounded Merlin model,
using the same summaries and without selecting a second set of winners.

## Physics and compression

For a mean spectrum μ(θ), let J = ∂μ/∂θ at Battaglia12, and C be a chosen
beam-consistent covariance. With C = LLᵀ and a QR decomposition of L⁻¹J,
MOPED weights W = L⁻ᵀQ give nine summaries Wᵀd with unit covariance. Their
local mean-response Fisher matrix equals JᵀC⁻¹J when all nine derivative
directions are kept. Parameter-prior widths merely stabilize the QR basis.
This is local information preservation for the specified binned data and C,
not a guarantee of global SBI accuracy or of information retention relative
to the original unbinned input.

Empirical discovery-sky covariance S gives C_eff = C + λS. This is a finite
nuisance-covariance version of suppressing seed/mask directions in the score:
variations that are large between discovery skies receive less weight. It can
reduce sensitivity to rare-halo selection and mask coupling. It can also
reduce genuine pressure information; the posterior-width comparison measures
that cost. A high-ℓ taper inflates C smoothly above ℓ=4000 instead. It neither
assumes nor guarantees that small scales are the actual source of the bias.

Changing the weighting covariance changes the projection. It does not by itself
make an SBI estimator trained on the original fixed sky marginalize over new-sky
uncertainty. Only the augmented candidate changes that training distribution;
all unaugmented candidates are tested for robustness without claiming such
marginalization.

The augmented candidate perturbs each training signal by a randomly selected
discovery B12 signal-ratio template and retains its SO residual. This assumes
those ratios transfer across the pressure prior and approximates the changed
signal-noise cross term. It is an experimental correction, checked on real
unused B12 skies and on fixed-sky parameter tests, not a validated general
cosmological simulator. No posterior is translated to the known truth.

References: [MOPED](https://arxiv.org/abs/astro-ph/9911102),
[nuisance-hardened compression](https://arxiv.org/abs/1903.01473),
[random-reference posterior tests](https://arxiv.org/abs/2302.03026).

## Selection and deliverables

Discovery uses simulations 100–110 and masks 12345–12349. Validation separately
uses simulations 112–118 at mask 12345, masks 12350–12354 on simulation 100,
and their unused 4 × 5 intersection. Neither validation factor enters the
nuisance covariance or method selection. All 100 observations are also reported.

`report.py` ranks truth displacement, posterior-median scatter, and 68% width
cost. Displacement/scatter are measured in the fixed original baseline half
width so artificial broadening cannot erase them. The Pareto front, one stated
25%-width-cost compromise, and choices under 10%, 25%, and 50% width budgets
make the tradeoff inspectable. A prior-wide marginal-coverage screen is
diagnostic; a broad or poorly calibrated posterior is not a successful fix.

`report/` contains PDF/PNG figures for each arm, the balanced simulation/mask/
interaction variance decomposition, compression scale response, CSV posterior
intervals and parameter diagnostics, and `results.md`. Original and selected
methods receive 90,000 posterior samples per sky, others 12,000. Numerical
Monte Carlo precision, outside-prior mass, held-out parameter coverage and
random-reference joint ranks are saved. `COMPLETE.json` hashes the report only
after unbounded and bounded analyses finish.

`check_numerics.py` checks the local Fisher identity, nuisance suppression,
complete bands, covariance lags, factorial decomposition, and identity
augmentation. These numerical gates do not constitute scientific validation.
