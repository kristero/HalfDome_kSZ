FLAMINGO 2/3/4 R200c comparison, snapshot 0077, L1_m9
====================================================

This isolated study keeps earlier outputs and production sources unchanged.
The source of truth is the full snapshot and SOAP-HBT membership on STRW/maas.
Main extraction is direct HDF5 cell streaming. Forty-five independent matched
swiftsimio spatial cutouts audit particle selection and linear shell sums.

Scientific target
-----------------
All gas, including recent AGN heating, is the primary electron-pressure target.
The native Compton-Y and electron-density fields are zero for star-forming gas,
as stated in their dataset headers. "All gas" keeps all particles but inherits
this observable convention; no unresolved ISM electron signal is reconstructed.
Central, own-satellite, other-halo and unbound membership components are disjoint.
"Unbound" does not mean uncontaminated by the correlated environment. All-gas
spheres overlap and cannot be assumed to be additive painting templates.

Selection: physical log10(M200c/Msun) bins 12.5,12.7,...,15.5. Three hundred
centrals are randomly chosen per bin below 14.5; all higher-mass centrals remain.
The seed and exact selected IDs are saved. All radii use the same population.

Entry points
------------
  prepare_study.py       metadata-only selection, membership map and cost plan
  control.py            archive failed attempt, deploy, gate, launch, status, fetch
  extract_study.py      bounded/resumable remote extraction and SWIFT audits
  fit_study.py          common-mask fits, controls, paired bootstraps, halo scatter
  interpret_results.py  supported-bin summaries and derived core-excluded map errors
  verify_study.py       old-high-mass reproduction and scientific numerical gates
  plot_study.py         profiles, RMS, coefficients, gas/electron/pressure residuals
  build_viewer.py       embeds an offline radial/spherical explorer
  build_report.py       PDF with measured results and full-analysis proposal

Run with a science Python environment (numpy, scipy, h5py, matplotlib, Plotly,
reportlab, pypdf). swiftsimio is required only for the independent remote route.
Use python -B to avoid bytecode caches. The control helper uses the preconfigured
strw-maas Windows OpenSSH alias through WSL. It never stores passwords or MFA.
Remote Python is an existing venv; paths are explicit in control.py.

Typical inspection/reproduction after extraction:
  python -B control.py status
  python -B control.py fetch
  python -B fit_study.py
  python -B interpret_results.py
  python -B verify_study.py
  python -B plot_study.py
  python -B build_viewer.py
  python -B build_report.py

measure.py owns component assignment, nested spheres, flux-conserving pixel
histograms and smoothing-length histograms. projection_model.py integrates a
gNFW field through a finite sphere and averages it over the matching pixels.
models.py owns the positive parameter bounds, shell integration and optimizer.
pilot_physics.py and the dependencies preserve the earlier estimators.
swift_reader.py is the inherited independent implementation; only swift_halo
is invoked by this study. remote_control.py supplies the SSH transport function;
use control.py for this study's CLI, not the inherited transport CLI.

Array contracts
---------------
measurements.npz:
  sums[halo, component, estimator, radial_shell] -- linear physical sums
  pressure/electron/gas_components[halo, component, radial_shell]
  projected_profiles[halo, orientation, sphere, component, field, annulus]
projection_stacks.npz:
  map_sums[mass_bin, morphology_group, orientation, sphere, component,
           field, x_pixel, y_pixel]
  groups: all, disturbed; orientations: original z, mean of ten cameras
  spheres: 2R,3R,4R; fields: pressure,electrons,gas; extent +/-4R,64 pixels
  divide stack sums by the appropriate halo count before plotting
  the less-disturbed map is (all sum - disturbed sum)/(Nall - Ndisturbed)
components: central bound, own satellites bound, other halos bound, unbound
estimators: see pilot_physics.SUM_NAMES, in that exact order
products/*_parameters.npz:
  parameters[halo, sphere, field, coefficient] -- fields pressure,electrons
  accepted[halo,sphere,field]; status -1=unsupported,0=unstable,1=accepted
  bootstrap[sphere,field,draw,coefficient] -- mean-fit sensitivity
  control_bootstrap[control,sphere,field,draw,coefficient]
  controls: no_other, no_satellites_or_other
  Pair pressure/electron coefficients from the SAME halo and radius; do not
  independently sample marginal coefficients. See report for selection biases.

Scientific cautions
-------------------
Lower-mass fits that fail strict common resolution support are diagnostic only.
The empirical diagonal fit uses halo SD / mean with a 5% floor, not a full
likelihood or uncertainty on the mean. Bootstrap percentiles ignore spatial
and cosmic-variance correlations. Individual-fit scatter is resolution and
stability selected and is not deconvolved intrinsic scatter. Ten views do not
increase the independent sample size. Particle-centre maps conserve flux but
are not SPH-kernel reconstructions. A single z=0 test cannot fit z exponents.
Profile figures hatch the excluded inner region; dashed model segments are
extrapolations beyond each actual fit mask. Solid segments cover fitted shells.
The additional core-excluded map metrics use existing fits and measured maps;
they do not require new particle reads or parameter refits.

The measured recommendation is a 4R acquisition cutout. For a complete all-gas
4R target, 4R fitting reduces supported-bin held-out pressure RMS from median
20.36% (2R) to 15.61% (4R), and wins nine of ten held-out bins. Inner fidelity is
better with the 2R fit. A production painter should fit separate inner/outer
components and enforce unique gas ownership; overlapping all-gas spheres are
not independent additive templates. See the PDF and products/interpretation.json.

Verification and preservation
----------------------------
complete.json is the remote extraction completion marker. executed_sources.json
pins the deployed extraction sources. Scientific checks and remote input SHA256
receipts distinguish numerical completeness from scientific validity. The final
inventory lists every created file and hash. attempts/ retains the first dtype
failure and its resource evidence. Completed per-file products are resumable
until final merged products have been independently verified; only then may
their redundant intermediates be removed with a cleanup receipt.
Resume assumes unchanged selection, inputs, unit conventions and physics code.
Use a NEW output directory for any scientific change; per-file cache filenames
alone do not encode a new selection or membership definition. This completed
comparison starts fresh and does not reuse particle products from earlier studies.
