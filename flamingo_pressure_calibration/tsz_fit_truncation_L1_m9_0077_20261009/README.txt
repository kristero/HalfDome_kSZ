FLAMINGO tSZ: separate pressure-fit extent and spherical truncation

RESULTS
- Report: output/pdf/tsz_fit_radius_and_truncation.pdf (34 pages).
  Pages1-2 are the concise numerical summary; remaining pages give diagnostics,
  equations, code examples, resources and interpretation.
- gallery.html: all figures, with PNG and SVG links.
- halo_model/tsz_fit_cut_matrix.csv: all16 fit/cut cases and83 multipoles.
- fit_inputs/coefficients.csv: native per-bin coefficients and bootstrap limits.
- paired_meanfit_uncertainty.csv: nominal and nonlinear paired spectral intervals.
- complete_tsz_radius_analysis.json, ARTIFACTS.json and SHA256SUMS: delivery evidence.

WHAT WAS CODED
1. Native pressure fits through1R, using the original shell-volume estimator,
   resolution mask, diagonal log-profile objective and coefficient bounds.
   Original2/3/4R coefficients and128 paired bootstrap draws remain bit-identical.
2. All16 Rfit x Rcut lightcone halo-model spectra. The primary fit-only cases are
   (1,4),(2,4),(3,4),(4,4); truncation-only uses (4,1),(4,2),(4,3),(4,4).
3. Separate one-halo and LINEAR two-halo terms, including cross-mass pairs,
   redshift/mass contributions, integrated-Y matching, taper and environment tests.
4. Full nonlinear128-draw profile-bootstrap propagation atell100/1000/3000.
   An SVD radial basis compresses the actual curves; it does not linearize their
   parameters. First-order propagation is retained as a failed-curvature diagnostic.
5. A same-painter random-sign angular self/distinct check on652254 halos,
   M>=1e14Msun, z<=1, NSIDE256; independent NSIDE512 total-map convergence.
6. A complete421868-central z=0 periodic-box source-power and projected-box
   y-power check for all16 cases, using physical wavenumber rather than sky ell.
7. An independent mean-one pressure-amplitude scatter control from measured
   within-bin integrated-pressure moments; not a physical gNFW parameter sampler.
8. Publication panels, report rendering, image QA, input/artifact hash checks,
   reproducible source inventory and temporary-file cleanup.

PHYSICS CONTRACT
F=Pe/P200=0.5176*P0*(x/xc)^(-0.3)*(1+x/xc)^(-beta), x=r/R200c.
P200=G*M200c*200*rho_crit*(Omega_b/Omega_m)/(2*R200c).
Native particle electron pressure already includes its header unit conversion.
Do not multiply it by a second hydrogen fraction or h conversion.
The0.5176 belongs to the published B12 gas-to-electron model convention.
The central cusp is integrable: its fixed exponent0.3 is below1 for a line
integral and below3 for a volume integral.

The shared calibrated domain is13.5<=log10(M200c/Msun)<=15.388243.
The13.3-13.5 bin has four strict1R shells, fewer than the six-shell minimum.
Identical published fallback is used outside the common domain in all variants.
The fitted F is constant within each original0.2dex bin; physical P200 andR200c
use actual catalogue-cell masses. No cross-bin interpolation mixes fitted shapes.

Lightcone evolution is ASSUMED from the relative published B12 redshift response:
F(M,z,x)=F(M,0,x)*Fpub(M,z,x)/Fpub(M,0,x).
The lightcone cosmology is h=.68,Omega_m=.31,Omega_b=.049; this differs slightly
from the FLAMINGO normalization used for the original dimensionless calibration.
Angular arrays are Compton-y Cl. Legacy dl_*_uK2 is T_CMB^2*ell(ell+1)Cl/(2pi),
with spectral factorg_nu=1. Multiply by g_nu^2 for frequency-dependent temperature.

One-halo uses sum(N*|y_ell|^2)/(4pi). Linear two-halo uses a bias-weighted complete
mass sum squared, multiplied by P_linear/shell_volume_per_sr. Cross-mass terms
must remain. This is not a unique ownership decomposition of simulation gas:
all-gas pressure around a central can already include neighbouring halos and
filaments. Additive painting plus another environment component can double count.

For the independent sky check, random signs cancel cross-halo terms in expectation.
Its distinct contribution is total minus same-painter estimated self, can be signed,
and includes finite-catalogue clustering. It is distinct from the LINEAR forecast.

Bootstrap intervals measure conditional uncertainty of the sampled mean fit.
They omit cosmic/spatial covariance, fallback/evolution/model uncertainty and
missing halo diversity. Ten rotations are views of existing halos, not new halos.
A realistic individual-halo painting distribution needs a joint correlated scatter
model; neither independent parameter error bars nor the mean-fit bootstrap supplies it.

REPRODUCTION FROM EXISTING LOCAL DATA
Run from this directory with /home/cbllover/miniconda3/bin/python.
No particle downloads or cluster submissions are needed.

python fit_inputs/prepare_fit_inputs.py
python fit_inputs/plot_fit_diagnostics.py
python fit_inputs/verify_fit_inputs.py
python halo_model/compute_halo_terms.py
python halo_model/check_halo_terms.py
python halo_model/check_bootstrap_linearity.py
python halo_model/bootstrap_nonlinear.py
python halo_model/plot_contributions.py
python empirical_check/angular_self_distinct.py
python snapshot_check/analyze_snapshot.py
python snapshot_check/verify_snapshot.py
python scatter_control.py
python plot_paired_uncertainty.py --selection fit
python plot_paired_uncertainty.py --selection cut
python plot_projection_pairs.py
python build_report_spec.py
python publish_report.py
python verify_delivery.py --render
# Inspect every rendered page and refresh final_visual_qa.json before finalizing.
python build_gallery.py
python verify_delivery.py --finalize
sha256sum --check SHA256SUMS

RESOURCE SCOPE
No new snapshot particles, simulation downloads, remote cluster connections or
PBS jobs were used. The main halo-model drivers use one thread; independent sky
and snapshot FFT checks use at most eight. Per-stage receipts give measured RAM
and timing; resource_usage.json adds live local capacity at finalization.
Report-authoring time is not included in numerical-driver CPU totals.
Temporary full-page rasters and Python caches inside this NEW root are removed
after visual review. Scientific NPZ/CSV products, figure pairs and QA contacts remain.

FILES EDITED
No pre-existing study, pipeline, input or report files were edited.
All newly created Python source files are listed below; ARTIFACTS.json lists every
retained new source/data/figure/report file and its checksum.

build_gallery.py
build_report_spec.py
empirical_check/angular_self_distinct.py
fit_inputs/plot_fit_diagnostics.py
fit_inputs/prepare_fit_inputs.py
fit_inputs/verify_fit_inputs.py
halo_model/bootstrap_nonlinear.py
halo_model/check_bootstrap_linearity.py
halo_model/check_halo_terms.py
halo_model/compute_halo_terms.py
halo_model/finalize_halo_model.py
halo_model/plot_contributions.py
plot_paired_uncertainty.py
plot_projection_pairs.py
publish_report.py
scatter_control.py
snapshot_check/analyze_snapshot.py
snapshot_check/verify_snapshot.py
verify_delivery.py
