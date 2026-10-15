Conditional lightcone analytic observables
=========================================

lightcone_analytic.py reads root/profile_scenarios.npz, the retained full-sky
HalfDome count histogram (421 redshift x 238 log-mass cells), and saved CLASS
cosmological/bias/linear-power ingredients. It never invokes CLASS or streams
the particle or halo catalogue. It uses one BLAS thread.

The profile input is dimensionless F0(M,x) at z=0. Redshift evolution is the
declared assumption F(M,z,x) = F0(M,x) Fpublished(M,z,x)/Fpublished(M,0,x).
Published B12 and B16 profiles supply this response. The main mass range is
13.3 <= log10(M200c/Msun) <= 15.388; published fallback outside it is defined
by the parent scenario builder. A separate unseen-tail stress case is labelled.
All profiles use each halo cell's own physical mass, radius, critical density
and pressure normalization in the declared lightcone cosmology, h=.68,
Omega_m=.31, Omega_b=.049. This is a dimensionless-profile transfer assumption
from FLAMINGO, not a newly measured FLAMINGO redshift calibration.

The output includes baseline 2/3/4R and each supplied scenario at 4R:

  y_l = (sigma_T / m_e c^2) P200 R200^3 / D_A^2 times the dimensionless
        spherical Fourier integral, with the physical Mpc/metre conversion.
  tau_l = sigma_T f_b rho_c R200^3 / (mu_XGP D_A^2) times that integral.
  C_l^yy,self = sum_h |y_l,h|^2/(4 pi).
  C_l^kSZ,self = sum_h (v_los,h/c)^2 |tau_l,h|^2/(4 pi).

These are catalogue one-halo/self terms evaluated at histogram bin centres.
The tSZ linear two-halo estimate uses the SAME catalogue abundance per shell,
the saved Tinker bias, and saved P_lin. It squares the sum over all masses
before integration, retaining cross-mass terms. It uses physical Mpc volumes
and converts native CLASS (Mpc/h)^3 power to Mpc^3 with division by h^3.

kSZ two-halo is deliberately absent: density bias alone is not a momentum
model. The parent compares actual maps with the matched catalogue self term.
Such a map-total minus self residual is signed in one realization and needs
pixel/flat-sky closure checks before being interpreted as a two-halo spectrum.
Internal gas motions, diffuse momentum and reionization remain absent.

For zsource=1, the code integrates the expected halo-only mean DM using the
observer factor 1/(1+z). The additional exp(-lambda_hit) is an independent-centre
Poisson footprint model. It is not the zero-hit fraction of clustered FRB rays,
and no full DM PDF is generated here. Parent all-sky pixels and matched120k rays
provide the actual painted halo PDF for the chosen finite angular resolution.

The Fourier transform analytically integrates a piecewise-linear radial table:
closed sin/cos antiderivatives avoid missing rapid low-z oscillations; a moment
series avoids cancellation at small q. The zero-mode uses q=0. Spectral modes
use the flat-sky q=(ell+1/2)R200/D_A. A direct quadrature comparison records the
radial-interpolation error; a separate 48-point published-B12 comparison uses
retained CLASS y_l values without new CLASS calls or amplitude adjustments.

The initial kSZ self term uses retained coarse sum(v_los^2) histograms allocated
to fine cells by halo counts, preserving each represented coarse-cell sum.
This approximation was replaced by the parent catalogue pass's matched fine
sum(v_los^2). apply_fine_velocity.py reweights the saved tau kernels and also
recomputes the kSZ bootstrap derivatives and exact-bin numerical control.
fine_velocity_replay_verification.json records closure, cost and the effect.

Empirical shape-scatter factors are supported only between adjacent finite
measurements. Near Fourier zero crossings the input factors are intentionally
NaN. finite_scatter_interpolation() treats those intervals as unsupported;
it does not bridge them or turn missing variance into invalid sky spectra.
repair_scatter_from_cache.py reweights the saved cells cheaply, records the
power coverage and verifies every variance increment is finite/nonnegative.

Outputs
-------
analytic_spectra.npz: sparse ell, labels, C_l self/linear2h, D_l and cell terms.
  D_l^yy is dimensionless; D_l^kSZ,self is in microkelvin squared.
  ell=0 is retained only for integrated moments; do not plot it as a measured
  sky auto-spectrum after monopole removal.
tau_kernel_cache.npz: float32 tau_l for occupied fine catalogue cells, case
  labels and indices; float32 rounding is negligible compared with model error.
analytic_DM_moments.json: mean halo DM and independent-centre hit quantities.
analytic_metadata.json: formulas, units, scope, approximations, resources, hashes.
analytic_verification.json: finite/nonnegative self/linear power and input gates.
class_profile_reference_gate.json: published B12 versus retained CLASS y_l.
figures/*.png and *.svg: radius dependence and high-mass sensitivity.

Reproduction from this folder, using the repository's science Python:
  python lightcone_analytic.py
  python extend_analytic.py
  python apply_fine_velocity.py
  python repair_scatter_from_cache.py
  python verify_lightcone_analytic.py
  python finalize_analytic.py
The first command creates all current scenarios, including the matched binning
control. The extension can update an existing control without duplicating it.
apply_fine_velocity.py expects the retained parent histogram in lightcone_maps.
The final scatter replay is deliberately cheap; unsupported factors add no
variance and its verification records the supported fraction of self power.

All-gas profiles include material associated with neighbours and the diffuse
environment. Adding an analytic two-halo term to them can double-count that
environment. The no_other case is a sensitivity control, not a unique ownership
partition. These products are conditional painting predictions, not independent
hydrodynamic sky validation, gas resolution convergence or likelihood errors.
