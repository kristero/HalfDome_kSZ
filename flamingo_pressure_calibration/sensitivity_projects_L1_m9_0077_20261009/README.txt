FLAMINGO L1_m9 snapshot 0077: three sensitivity projects
======================================================

The six final PDFs are in output/pdf: one concise and one detailed report per
project. reports_manifest.json records their hashes and page counts. The final
verification receipt distinguishes measured profiles, conditional painted
fields, numerical checks and scientific limitations.

What was reused
---------------
The frozen radii234_contamination_L1_m9_0077_20261008 extraction has 4,375
centrals, native-header units, periodic distances, gas membership, 36 shells,
finite 2/3/4R200c measurements and matched swiftsimio extraction checks. It was
not edited or rerun. This new study reads its compact products, 421,868 SOAP
central metadata entries, and the already-local HalfDome lightcone catalogue.
No full snapshot gas particles were read or downloaded in this study.

Source map
----------
analysis_common.py                  Physical kernels, normalization and weights
fetch_box_metadata.py               Metadata-only STRW SOAP access and hashes
01_radius_scatter/analyze_radius.py Fit radius vs cutoff, constraints and scatter
02_high_mass/analyze_high_mass.py    Nested pooling, leave-one-out and bounds
03_mass_binning/analyze_binning.py   Actual-mass pivots, weights and cross-validation
box_observables/analyze_box.py       Full central-catalogue FFT/self/distinct powers
build_profile_scenarios.py           Shared positive z=0 shape curves and controls
build_column_tables.py               Finite-sphere chord and volume integrals
analytic_observables/*.py            Conditional angular self/linear2h diagnostics
lightcone_maps/paint_lightcone.py    All-halo sky maps and historical FRB rays
frb_poisson_control.py              Independent-centre matched DM control
lightcone_maps/plot_lightcone.py     Spectra, PDFs, maps and numerical comparisons
integrate_lightcone_reports.py       Report adapter using completed numerical outputs
report_content.py                   Project 01/02 physics and editable prose
03_mass_binning/report_content_binning.py   Project 03 report content
publish_reports.py                  Six PDF layouts with rendered mathematics
verify_reports.py                   Every-page raster and text-geometry checks

Definitions that must survive edits
----------------------------------
1. Mass is physical M200c. Native FLAMINGO fields use their own physical-CGS
   attributes, without an extra h factor. HalfDome catalogue mass is divided by
   h=0.68. FLAMINGO's plan baryon fraction includes the separate neutrino term.
2. Fit radius selects shells. Cutoff radius limits the projected/Fourier sphere.
   They are independent settings, and the compact project01 matrix has all nine
   combinations. The main sky tests use native Rfit=Rcut for 2/3/4R200c.
3. Native particle Compton-Y areas give electron pressure; native n_e*m/rho gives
   electron count. Do not reapply a hydrogen fraction to native n_e. The B16
   painting conversion uses the explicitly documented mu_XGP convention.
4. All gas is retained in extraction; native star-forming Y/n_e fields are zero.
   Recently AGN-heated gas remains in the primary instantaneous-signal target.
5. Bootstrap refits estimate sensitivity of the empirical mean. Individual halo
   scatter is distinct and must be sampled jointly across profile coefficients.
6. A gas membership class is not a halo-model self/distinct decomposition. The
   painted box explicitly computes the same-halo sum and distinct-centre cross
   terms; all-gas spheres still overlap and are not a conserved physical gas field.
7. All lightcone predictions assume F(M,z,x)=F0(M,x)*Fpub(M,z,x)/Fpub(M,0,x).
   Published profiles are used below logM=13.3 and above measured logM=15.38824,
   except the explicit last-measured-tail stress case. These are extrapolations.
8. Binning alternatives freeze x<0.518507 to the same native core and are scored
   against binning_constant_outer, with the same fit mask/radial representation.
9. FRB columns include the observer-frame 1/(1+z) factor and only halos at z<=1.
   The exact historical 120,000 directions (seed42) are retained. All sky pixels
   means every HEALPix2048 pixel, not a continuum of infinitesimal sightlines.
10. kSZ uses each halo's coherent line-of-sight bulk velocity. Internal gas
    motions, diffuse gas and reionization are outside the computed model.

Reading the products
--------------------
Use dict(np.load(path)) when several arrays are needed: repeated access to a
compressed NpzFile repeatedly decompresses arrays. All axes and units are
documented in project JSON/README files. Figures are supplied in PNG/SVG pairs.

Example (run from this directory with the validated science Python):

    import numpy as np
    from analysis_common import HERE
    data = dict(np.load(HERE / '01_radius_scatter/radius_sensitivity_arrays.npz'))
    print(data['measured_kernels'].shape)  # halo, Rcut, field, physical k

The completed sky painter refuses to overwrite complete.json. Preserve the
dated output root and use a fresh root for new experiments. Do not interpret a
successful numerical gate as agreement with a direct hydrodynamic sky field.

Cluster access
--------------
The configured Windows OpenSSH alias is `ssh strw-maas`, for nagainis through
the STRW gateway. Simulation root: /net/hypernova/data2/FLAMINGO. This study's
snapshot and SOAP paths are retained in the frozen metadata_plan.json. idark
was attempted but unreachable; zero PBS jobs were submitted. No new login or
MFA configuration was changed by this study.
