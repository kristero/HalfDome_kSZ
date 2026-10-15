tSZ FIT-RADIUS AND FINITE-CUT CONTRIBUTIONS

Run from this directory with the validated science Python:

    /home/cbllover/miniconda3/bin/python compute_halo_terms.py
    /home/cbllover/miniconda3/bin/python check_halo_terms.py
    /home/cbllover/miniconda3/bin/python plot_contributions.py
    /home/cbllover/miniconda3/bin/python check_bootstrap_linearity.py
    /home/cbllover/miniconda3/bin/python bootstrap_nonlinear.py

These programs read existing compact fit parameters, the saved full lightcone
count histogram, and saved CLASS ingredients. They do not rerun CLASS, read
snapshot particles, stream the large halo catalogue, or submit cluster jobs.

SCENARIOS AND OUTPUTS

tsz_halo_terms.npz contains all 16 combinations of fit radius and projected
sphere radius, each selected from 1, 2, 3 and 4 R200c. Its labels, cases and ell
arrays define the axes. Scenario index is 4*(fit_radius-1)+(cut_radius-1).

    a = np.load("tsz_halo_terms.npz")
    fit_only = a["cl_total"][[3, 7, 11, 15]]
    truncation_only = a["cl_total"][[12, 13, 14, 15]]
    reference = a["cl_total"][15]  # fit through 4R, project through 4R

The cl_1h, cl_linear_2h and cl_total arrays are Compton-y autocorrelation
spectra C_l^yy, with no beam, pixel window, instrumental noise or observing
frequency. The dl_*_uK2 arrays and absolute-power figures show

    T_CMB^2 D_l^yy,  D_l^yy = l(l+1) C_l^yy/(2*pi),

with T_CMB = 2.7255 million microkelvin. This is a temperature equivalent with
frequency factor g_nu = 1, not a frequency-specific tSZ temperature spectrum.
For DeltaT_nu = T_CMB*g_nu*y, multiply these powers by g_nu^2.

mean_compton_y is the zero-mode mean. A Limber C_0 is not reported.

The CSV tsz_fit_cut_matrix.csv gives the spectra and matched-baseline changes.
The cl_1h_by_mass array retains each original catalogue mass cell, summed over
redshift. cl_1h_by_z and cl_linear_2h_by_z retain the redshift contributions.
cl_linear_2h_cross_mass retains a symmetric group-by-group matrix. Its full
sum, including both off-diagonal halves, equals the full linear 2-halo term.
Do not add only its diagonal: that would discard correlations across masses.

PRESSURE, NORMALIZATION AND EVOLUTION

The fitted dimensionless profile is

    F(x) = Pe/P200 = 0.5176*P0*(x/xc)^(-0.3)*(1+x/xc)^(-beta),
    x = r/R200c,
    P200 = G*M200c*(200*rho_critical)*fb/(2*R200c).

The inner slope is fixed. P0, xc and beta are the fitted coefficients; beta
is not itself the final outer power-law slope, which is -(beta+0.3).
rho_critical is the physical critical density; fb is Omega_b/Omega_m. The
lightcone uses h=0.68, Omega_m=0.31, Omega_b=0.049. Physical masses are Msun.
Header-derived native electron pressure has no additional hydrogen factor.

Each native 0.2-dex mass bin has a constant dimensionless shape. The physical
P200 and R200 are evaluated at each actual catalogue-cell mass. For every
fit/cut scenario, fitted shapes are used only in the identical common support
13.5 <= log10(M/Msun) <= 15.388243... . Outside this range the same published
B12 profile at the actual mass is used. No interpolation smears these edges.

Only z=0 profiles were measured. All lightcone scenarios assume

    F(M,z,x) = F(M,0,x) * F_B12(M,z,x)/F_B12(M,0,x).

This preserves the published relative redshift response. It does not measure
redshift evolution of the new FLAMINGO parameters. The profile convention and
normalization are transferred between two slightly different cosmologies.

1-HALO AND LINEAR 2-HALO FORMULAS

For each saved mass/redshift count cell, y_l is the finite-sphere transform

    y_l = sigma_T/(m_e*c^2) * P200*R200^3/DA^2
          * 4*pi * integral_0^Rcut x^2*F(x)*sinc(q*x) dx,
    q = (l+0.5)*R200/DA,  sinc(t) = sin(t)/t.

DA is the angular-diameter distance; sigma_T is the Thomson cross section;
m_e is the electron mass; c is the speed of light. Physical units are used
consistently by the cached Ingredients.geometry function.

    C_l^1h = sum_cells N_cell * |y_l|^2/(4*pi),
    C_l^linear2h = sum_z P_linear(l,z)/V_z
                  * [sum_mass N_cell*b(M,z)*y_l/(4*pi)]^2.

N_cell is the full-sky halo count. V_z is the comoving shell volume per
steradian. b is the saved halo bias. The native CLASS power is converted from
(Mpc/h)^3 to Mpc^3 before applying this expression. Squaring the complete
mass sum retains all cross-mass pairs. Both terms use the same profile;
the 1-halo term emphasizes squared halo signals, whereas the 2-halo term
contains the bias-weighted correlated amplitude.

All-gas stacks already include some correlated environment and neighboring
halos. Adding this linear 2-halo term can double count environmental gas.
The result is therefore a matched sensitivity forecast, not a unique gas
1-halo/2-halo decomposition or a hydrodynamic tSZ truth comparison. The
no_other control removes gas bound to other halos but does not remove all
correlated diffuse gas. The flat-sky profile kernel and Limber approximation
are not independently validated at low multipoles here.

WHAT THE CONTROLS MEAN

shape_matched_cl_* rescales each halo/scenario to the integrated Y of the
fit4/cut4 reference at the SAME mass and redshift. This isolates angular
redistribution from flux changes. Equality is guaranteed at q=0, not at an
arbitrary low l: very nearby halos can remain resolved even at l=100.

taper_cl_* applies a cosine window to the outermost 10 percent of each
sphere, holding the 4R fit fixed. It changes the total signal. The separately
saved taper_Ymatched_cl_* restores the corresponding sharp-cut integrated Y
and therefore tests edge shape at fixed flux. A taper is a numerical control,
not a physically preferred halo boundary.

independent_checks.npz contains a direct analytic-profile 1024-node radial
refinement at l=100, 1000 and 3000, as well as the other-bound-gas, expanded
1R coefficient-bound and published-B12 controls. The baseline calculation
uses an exact oscillatory integral of a 256-node piecewise-linear profile.
The separate adaptive-quadrature gate includes q values through 1000 and
normalizes absolute error by the zero mode so Fourier zeros do not cause
misleading percentage divergences.

positive_power_tail_control.npz tests the 530 exact zero cells in the saved
CLASS power cache. They lie at high l in 20 very-low-z rows. The alternative
fills those absent tails using the last three positive logP/log(l+0.5) points
of each row and then interpolates positive logP. This tests sensitivity to
missing cache tails; it is not a new CLASS calculation or a validated high-k
power spectrum. Its receipt gives effects on both 2-halo and total power.

NONLINEAR BOOTSTRAP AND ITS LIMITATIONS

Use bootstrap_nonlinear.npz for the final conditional uncertainty at l=100,
1000 and 3000. It evaluates all 128 complete paired coefficient vectors for
all 16 fit/cut scenarios and every original catalogue count cell. A radial
singular-value decomposition (SVD) accelerates the nonlinear profile functions
without linearizing their parameter dependence. Its maximum pointwise profile
error is below 1e-7, and direct transforms of all original radial draws at
representative actual count cells independently verify the kernels. The
unbootstrapped point prediction exactly reproduces the main spectra. The
bootstrap_nonlinear_verification.json receipt records these numerical gates.

The cl_*_draws arrays have axes [16 scenarios, 128 draws, 3 multipoles].
The paired_*_change_percent arrays are 100*(draw_case/draw_reference - 1).
Their percentile arrays have axes [16th/50th/84th percentile, scenario, ell].
Always use the complete paired vector across parameters and fit radii. The
same bootstrap draw labels preserve correlations between the compared fits.

These remain errors of the fitted MEAN conditional on the sampled halos.
The calculation treats different mass bins as independent and fixes both the
catalogue and published fallback. It does not capture physical halo scatter,
cosmic variance, uncertainty in evolution, ownership, bias, cosmology,
fallback profiles or profile-model inadequacy. Small populations cannot
bootstrap halos or environments that were never observed.

FIRST-ORDER CONTROL: DO NOT USE AS THE FINAL INTERVAL

bootstrap_linear_response.npz uses all 128 paired mean-profile bootstrap
coefficient vectors. It propagates a FIRST-ORDER response in ln(P0), ln(xc)
and ln(beta), retaining their covariance and the matched fit-radius pairing.
Different mass bins are treated as independent. Fixed fallback halos carry
no fitted error. These are conditional errors of mean-profile forecasts.
These first-order arrays are not nonlinear bootstrap spectra, per-halo scatter,
spatial/cosmic variance, or uncertainty in evolution, gas ownership, bias,
cosmology, fallback profiles or model inadequacy. Boundary/degeneracy effects
are not captured by a linear response and must not be treated as a rigorous
bound on the physical tSZ signal. The nonlinear spectral bootstrap above
supersedes these arrays as the final mean-fit uncertainty.

check_bootstrap_linearity.py directly evaluates all 128 nonlinear radial
profiles at representative mass pivots, redshifts and Fourier arguments. It
finds significant coefficient-space curvature in the rarest mass bin: the
first-order zero-mode standard deviation can exceed the actual nonlinear
value by more than a factor of two. This is why the final calculation keeps
the full nonlinear coefficient vectors rather than propagating independent
marginal errors or a linear covariance approximation.

The root directory contains a separate empirical amplitude-scatter control.
It addresses a different question from uncertainty of the fitted mean.
