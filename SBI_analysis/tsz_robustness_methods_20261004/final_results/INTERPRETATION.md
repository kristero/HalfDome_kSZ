# SO tSZ: simulation and mask robustness

The best of the **24 tested methods** under the user-confirmed **25% average-width
budget** is retrained, noise-only-whitened Merlin on **158 bands of width 50
multipoles**, retaining **80 ≤ ℓ ≤ 7979**. It is selected for simulation, mask,
and combined comparisons. It makes no multipole cut, but averages away detail
within each band and changes compression relative to the original PCA inputs.

Ten Gaussian simulations, 100, 102, …, 118, are crossed with masks 12345–12354.
The baseline is simulation 100 with mask 12345. Each simulation is painted
once and reused for all masks. Every observation preserves the trained
cross-spectrum: two fixed split-noise draws of the single SO baseline deproj0
prescription, with the trained Gaussian-beam correction.

## Measured performance

Selection uses six simulations and five masks. The tests below hold four
simulation factors and five mask factors out of covariance/template fitting
and method selection. Their joint intersection has 20 cases. Some single-factor
cases were visible in the earlier study; this is not an untouched blind analysis.

| Held-out comparison | Cases | Truth-RMS reduction | Median-scatter reduction | Average width increase | Worst parameter width increase |
| --- | ---: | ---: | ---: | ---: | ---: |
| Simulation | 4 | 33.0% | 68.2% | 24.7% | 59.1% |
| Mask | 5 | 18.1% | 78.7% | 17.9% | 55.4% |
| Combined | 20 | 30.0% | 71.5% | 22.6% | 74.4% |

On all ten simulations, ten masks on simulation 100, and the full 100-case grid,
average width increases are 24.7%, 17.4%, and 21.5%; median-scatter reductions
are approximately 90%, 85%, and 79%. These full-set results include discovery.

The true nine-parameter Battaglia12 vector lies inside the selected model's
**68% joint physical-parameter highest-density region on all 100 skies**, in
both parameter spaces. The original unbounded model excludes truth from its
95% joint region on every held-out sky. Marginal medians of `xc` and `beta`
remain displaced, and truth remains outside their 68% marginal intervals.
All nine truths are inside the selected model's 95% marginal intervals.

The prior-bounded companion confirms stability and joint truth recovery, but
its held-out width increases are 27.0%, 17.1%, and 25.9% for simulation, mask,
and combined cases. It exceeds the budget in two comparisons. These ratios
reference the original bounded model; the unbounded model remains primary.
Bounded results are a companion check without a second selection.

No qualifying discovery improvement was found within a 10% width budget.
With 50%, joint-variation-aware MOPED lowers truth displacement further,
but its full-set width costs are about 40% for simulation and 37% for combined
cases; the weak joint variant costs about 37% for masks. These methods have
more median scatter than the selected banded Merlin.

## Metric definitions

Let `m_ci` and `w_ci` be the posterior median and 84th-minus-16th-percentile
width for case `c`, parameter `i`. The fixed reference `h_i` is half the original
model's baseline 68% width, `theta_i*` is truth, and `mbar_i` is the mean median
across cases. Truth displacement and sky scatter are

\[
T=\sqrt{\langle[(m_{ci}-\theta_i^*)/h_i]^2\rangle_{c,i}},\qquad
A=\sqrt{\langle[(m_{ci}-\bar m_i)/h_i]^2\rangle_{c,i}}.
\]

Increasing a candidate's widths cannot reduce these metrics because the
denominator stays fixed. The average width ratio is the geometric mean

\[
R=\exp\langle\log(w_{ci}/w_{ci}^{\rm original})\rangle_{c,i}.
\]

The reference width in this ratio is the original model on the same case.
Within R ≤ 1.25, selection minimizes the quadrature of T and A, requires both
to improve, and applies the stated marginal and joint diagnostic screen.
An average cap permits larger losses for individual parameters.

The joint truth test measures posterior mass at density higher than the density
at truth. A level below 0.68 means truth is inside the 68% highest-density
region. Density uses physical parameter coordinates and the correct transform
Jacobian. Displaced marginal medians can therefore coexist with truth in a
high-density joint region; separate medians do not describe the correlated
distribution.

## Physics and coded methods

Compton-y measures electron pressure integrated along the line of sight:

\[
y=\frac{\sigma_T}{m_e c^2}\int P_e\,dl.
\]

`sigma_T` is the Thomson cross-section, `m_e` the electron mass, `c` the speed
of light, `P_e` physical electron pressure, and `l` physical distance. The
Battaglia12 painter uses supplied M200c masses converted from Msun/h, a
physical mass threshold of 10^12 Msun, all available redshifts, and the
training pipeline's finite 4R200c sphere. R200c encloses mean density 200 times
the critical density. The nine parameters describe normalization, core scale,
outer slope, and their mass/redshift scalings.

Rare massive halos contribute strongly to pressure power. Simulation phases
change which halos occur; masks change which are observed and couple angular
modes. Measured relative spectrum scatter is largest below ℓ≈580: 10.7%
across simulations and 15.0% across masks, versus less than 0.7% above ℓ≈2980.
These are spatial/simulation variations conditional on fixed noise draws.

Banded Merlin averages the signed D_ell = ell(ell+1)C_ell/(2pi) with weights
proportional to 2ell+1 inside each band. It uses projected SO noise-only
covariance, existing asinh preprocessing, and the same learned-head/flow
architecture, training pool, prior, parameter order, and three training seeds.
Use its **new ensemble and saved transform together**. The experiment tests
this full method; it does not isolate averaging alone from changed whitening,
projection, and retraining relative to the original model.

For MOPED, J contains local spectrum derivatives and C is the weighting
covariance. With C=LL^T and a QR decomposition of L^-1 J, weights W=L^-T Q
give summaries W^T d. All nine derivative directions are retained. This
preserves the local mean-response Fisher matrix J^T C^-1 J of the selected
binned data and covariance. It does not guarantee global posterior fidelity
or retention of parameter-dependent covariance information.
[MOPED paper](https://arxiv.org/abs/astro-ph/9911102).

Variation-aware weights use C_eff=C+lambda S: S is empirical discovery-sky
covariance and lambda sets suppression strength. The high-ell taper inflates
covariance smoothly on small scales. These change projection weights;
explicit sky uncertainty enters the training distribution only in the
separate approximate augmentation trials. The cumulative-response figure is
a derivative-response norm, not an additive Fisher-information fraction.

Balanced full-grid decomposition assigns original median variation to
simulation (26.3%), mask (31.8%), and interaction (41.9%). For banded Merlin,
the fractions are 12.4%, 73.1%, and 14.4%. Interaction includes nonlinear
inference response; overlapping masks make these descriptive fractions.

## Validation and limits

The selected unbounded model's reserved 1,500-row parameter test gives
68% marginal coverages of 63.3–66.9%, 95% coverages of 95.7–99.1%, and joint
random-reference ranks of 66.4% and 96.1%. The bounded model gives marginal
ranges of 68.8–72.7% and 94.5–96.1%, and joint ranks of 69.9% and 94.5%.
They pass the broad screen but retain calibration errors. Neither exact
calibration nor unbiased marginal medians is established.

Fixed-sky reserved-row mean log density at truth is lower for banded Merlin:
-5.48 versus -3.01 for original unbounded Merlin. Robustness has an information
cost beyond retaining the full multipole range. This is the best tested
compromise, not proof of minimum possible information loss.

Selected posteriors use 90,000 draws per sky; joint truth ranks use 6,000 new
draws. Mean median Monte Carlo error is about 0.005–0.006 of an original
baseline half width. Reserved parameter rows do not re-rank the methods.
The final report has 58 verified artifacts, and 78 model files were hashed
on idark. LOCAL_VERIFIED.json checks delivered source/report hashes, finite
ordered quantiles, and variance decomposition. Temporary review copies and
copied Python caches were removed.

Main figures: [joint truth](report/joint_truth_linear.pdf),
[simulation](report/posteriors_simulation.pdf),
[mask](report/posteriors_mask.pdf), [combined](report/posteriors_combined.pdf),
and [width costs](report/parameter_width_cost.pdf).
Both parameter spaces and Monte Carlo errors are in
[selected_posterior_diagnostics.csv](report/selected_posterior_diagnostics.csv).

Cluster report: `/lustre/work/kristero10/tsz_robustness_methods_20261004/comparison/report/`.
Primary model: `/lustre/work/kristero10/tsz_robustness_methods_20261004/candidates/merlin_band50_full/`.
Current guides are in `guide/`; `source_snapshot/` preserves exact archived
cluster sources, including the original frozen documentation.
