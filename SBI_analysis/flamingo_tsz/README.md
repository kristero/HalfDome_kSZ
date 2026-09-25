# FLAMINGO tSZ comparison on idark

The user approved the cluster download and resource budget on 2026-09-14
and selected the verified **40-bin, nine-parameter, fixed-noise MOPED/SBI
model**. All three approved FLAMINGO files have been downloaded and validated
on `idark`. Their total size is exactly 4,556,709,696 bytes; all pixels are
finite, and `download_complete.json` records the file checksums and statistics.
The map/inference campaign **completed successfully**, job `597329.idark`,
on 2026-09-14. Its `campaign_complete.json` and comparison artifact manifest
are present, and all 37 primary result-file checksums passed local verification.
The fresh HalfDome control reproduces all 40 historical input bins exactly
and yields 10,000 posterior samples. Each FLAMINGO map failed the original
prior sampling diagnostic, with 0/20,000 accepted network proposals.

Read [RUN_REPORT.md](RUN_REPORT.md) for the measured spectra, inference
limitations, physics, resource use, code inventory and reproduction commands.
The raw maps remain on the cluster; small plots, tables, spectra and logs are
in `cluster_results/`. The allocation was 64 GB RAM; the measured maximum
production RSS was **10.97 GiB**, and workspace use was approximately
**11.4 GiB**, below the approved 20-GiB budget.

## Implementation and runtime matching

`download_maps.py` checks the approval and frozen metadata manifest, downloads
only the three selected files, supports resuming incomplete transfers, checks
byte counts and the Float64 dataset shape, scans the map in bounded chunks,
and records checksums. The actual HDF5 dtype is **big-endian `>f8`**, whereas
the hdfstream metadata endpoint describes its native representation as `<f8`.
The initial strict byte-order check stopped the first download job; the
corrected check preserves the same floating-point values and records both
representations. No file was substituted or downloaded twice.

`process_maps.jl` includes the existing HalfDome operator source files, without
editing them. It builds one cached mask and two SO noise maps, verifies their
pixel hashes, paints a fresh HalfDome Battaglia12 control, and processes that
control and the three FLAMINGO inputs using the same arrays. It saves clean,
noisy and noise-only spectra, runtime/source provenance, and the actual
HalfDome catalogue redshift support. A small regression check verifies that
the memory-saving in-place map addition matches the original helper exactly.

`run_campaign.py` runs these stages sequentially under the 64-GB/26-CPU PBS
allocation, records per-stage logs, exit codes and peak RSS, and stops before
another map if RSS approaches the allocation. `compare_inference.py` reuses
the historical bundle-loading, binning, MOPED and bounded SBI-sampling helpers.
It verifies 32 reference model log probabilities and the saved HalfDome
posterior before evaluating new observations. It retains support failures as
explicit results, continues the other models, and produces PNG/CSV/JSON/NPZ
comparison artifacts. All original model parameters, priors and transforms
are retained.

The cluster's Julia 1.6.2 and 1.10.4 generate **different** MersenneTwister
draws from the saved observation's Julia 1.12.2, even for identical seeds.
Therefore `stage_runtime.py` packages a pinned dependency closure and the
exact XGPaint source from the reference environment, and `setup_runtime.pbs`
installs an isolated Julia 1.12.2 under the campaign. Its binary archive is
checked against the official SHA256 list. No existing cluster environment
is modified. The cluster's older OS also requires Julia's bundled C++ library
for its 7z helper; that library path is restricted to Julia subprocesses.
The production job requires the reference mask fingerprint and matching
noise/beam probe values before building full-resolution maps.

Execution wrappers are `run_download.pbs`, `setup_runtime.pbs`, and
`run_campaign.pbs`. `approval.json` records the user's approval and exact
three-map scope; `source_manifest.json` and `runtime_source_manifest.json`
record the staged source hashes. Runtime software and intermediate files
remain within the approved 20-GiB campaign workspace budget.

## Verified downloads and resource request

The live catalogue contains these ready-made, integrated Compton-y maps:

| FLAMINGO model | File size, bytes | Decimal GB |
| --- | ---: | ---: |
| `L1_m9` (fiducial) | 1,520,700,487 | 1.520700487 |
| `fgas-8sigma` | 1,516,150,395 | 1.516150395 |
| `Mstar-1sigma` | 1,519,858,814 | 1.519858814 |
| **Total** | **4,556,709,696** | **4.556709696** |

The total is **4.244 GiB**. Each input has an HDF5 dataset `data` with shape
`(201326592,)` and Float64 dtype: one uncompressed map occupies **1.5 GiB**.
This is `12 * 4096**2` HEALPix pixels. Ring ordering and dimensionless
Compton-y units are documented by FLAMINGO; these files have no dataset
attributes recording those conventions.

Approved cluster request: **20 GiB workspace, 64 GB RAM, 26 CPUs, queue
`mini`, walltime `23:59:00`**, processing one feedback model at a time.
The preflight estimated process peak was **24-32 GiB**; the subsequently
measured maximum production RSS was **10.97 GiB**. The
64-GB request includes library, harmonic-transform and allocation overhead.
Each map job recorded peak RSS before the next map ran.
The selected model bundle itself is below 1 MiB; map operations dominate RAM.

Workspace budgeting includes all three original HDF5 inputs, up to 4.5 GiB
for a reusable Float64 mask and two noise maps, one optional 1.5-GiB working
map, spectra, posterior samples, figures and margin. Retain the raw inputs
but avoid saving every intermediate map. A harmonic coefficient array at
ell_max=7979 occupies 509,507,040 bytes; Gaussian smoothing also uses the
original code's default harmonic limit, which must not be lowered merely
to save memory. Workspace capacity reported by `df` is not a quota check.

The documented integrated-map directories returned HTTP 404 for
`fgas+2sigma`, `fgas-2sigma`, `fgas-4sigma`,
`Mstar-1sigma_fgas-4sigma`, `Jet`, and `Jet_fgas-4sigma`.
Do not silently substitute another observable or another cosmology for
these models. Shell-based construction would be a separate proposal:
60 Float64 Compton-y shells at this resolution contain **90 GiB per model**
before transport compression, even when streamed without retaining all shells.
That alternative is outside the approved 4.56-GB download scope and would need
its own verified manifest and consistent rotation/lensing treatment.

## Processing contract

1. Download only the three manifest-listed `lensed_tSZ_rot_same_rot.hdf5`
   files into the cluster workspace. Check their advertised byte counts,
   record SHA256 hashes, and validate the full maps for finite values.
2. Use dimensionless y throughout. Apply the same 2-arcmin Gaussian beam
   to the signal with `tSZ_visuals/output.jl`'s smoothing implementation.
   This has harmonic response
   `B_l = exp[-l(l+1) sigma_b^2 / 2]`, where
   `sigma_b = FWHM / sqrt(8 ln 2)`. Do not add a frequency conversion.
3. Reuse the mask and noise functions in
   `tSZ_visuals/run_halfdome_fullsky_so_noise.jl`. The mask is a randomly
   oriented cap with support f_sky=0.4 and a 60-arcmin cosine apodization.
   Its mean squared weight differs from 0.4; preserve the actual mask.
4. Use SO **baseline, deprojection 0**, root seed **12345** and mask seed
   **12345**. The original deterministic offsets give noise-split seeds
   **22446 and 22447**. Use the original Julia RNG/Healpix implementation,
   not a NumPy RNG given the same integer. Record runtime versions and
   compare generated mask/noise hashes for every model and the HalfDome
   control. Matching seed integers alone does not establish matching arrays.
5. Add native SO Compton-y noise to the beamed signal, without smoothing
   the noise a second time. Use noise and spectrum ell_max=7979 and
   niter=0. The existing SO input file's checksum matches the saved
   HalfDome observation request:
   `04cba7b20e06002b1a6af67e853aa643fbd3b5decea78f99e480cc7cc6b42f64`.
6. Form the raw masked split cross-spectrum of `W(By+n1)` and `W(By+n2)`.
   Preserve the training estimator: no extra division by f_sky, beam
   deconvolution, noise-bias subtraction, or mode-coupling correction.
   Also save clean and noise diagnostics to explain the differences.
7. Apply the bundle's exact 40 bins spanning ell=80-7979, with `(2ell+1)`
   weighting of linear `D_l = l(l+1) C_l / (2*pi)`. Preserve signed bins.
   Use the saved asinh scale, centering/standardization and MOPED matrix
   via `so_sbi_compression.project`; never refit on FLAMINGO.
8. Load the saved best-validation SBI checkpoint and verify its reference
   log probabilities. Use all nine parameters in the saved order and the
   original prior. Run bounded support/acceptance diagnostics first, then
   obtain posterior samples only if valid. Save failures as diagnostics;
   do not widen the prior or rescale a context to obtain a contour.
9. Compare clean/noisy spectra, signed binned differences, compressed
   coordinates, and supported posterior samples with the verified
   fixed-seed HalfDome Battaglia12 reference. Compare FLAMINGO variants
   against their own fiducial as well. Preserve existing HalfDome figures.

## Physics and interpretation

Thermal SZ traces integrated electron pressure,
`y = sigma_T / (m_e c^2) integral P_e dl_phys`. Feedback changes the
amount, temperature and spatial distribution of gas, changing y and its
angular spectrum. The fgas and Mstar labels refer to shifts in the
observational calibration targets; they are not direct numerical changes
to a Battaglia pressure parameter or Gaussian errors on the inferred model.

The FLAMINGO files are precomputed **lensed**, rotated, integrated maps.
The L1 variants extend to z=3 in the source publication. Recently
AGN-heated gas is excluded from the underlying Compton-y shells. Keep
these source conventions explicit. The maps must not be described as a
matched halo-only realization of HalfDome, whose reference paints a
Battaglia profile onto haloes above 1e12 Msun at fixed cosmology. A
redshift/cosmology or high-z amplitude correction is not automatically
authorized: verify the reference lightcone support and report the mismatch.
No empirical rescaling is part of this campaign.

Even using the same noise maps, the cross-spectrum contains
`C_ss + C_s,n2 + C_n1,s + C_n1,n2`. FLAMINGO changes the signal phases,
so the signal-noise terms change. The selected historical SBI model was
trained using fixed noise and a HalfDome signal family. Its MOPED weights
are sensitive to directions with little training residual scatter.
Consequently a FLAMINGO spectrum can fall outside learned support even
when the uncompressed spectrum looks similar. Any returned parameters
are effective HalfDome/Battaglia interpretations of FLAMINGO, not measured
FLAMINGO input parameters or established SO noise-marginalized uncertainties.
Comparing variants at common FLAMINGO cosmology helps isolate feedback
changes; comparing directly with HalfDome includes other simulation differences.

## Code, evidence and reproduction

- `preflight.py` is the new metadata/resource checker. It queries only
  directory listings and HDF5 metadata, caps responses at 1 MiB, rejects
  HDF5 dataset bodies, checks the existing cluster model hashes, and records
  the saved HalfDome reference provenance. It neither downloads simulation
  files nor submits jobs.
- `metadata_manifest.json` is its successful cluster output. The completed
  scan received 3,939 bytes of successful metadata responses, checked all
  nine variant paths, and verified every selected model-bundle checksum.
- This `README.md` records the approved download scope, resources, processing
  contract and scientific limitations. No existing analysis file
  was edited.

An earlier exploratory request used `max_data_size=0`, the parameter name
shown in the site's prose API page. The server ignored that spelling and
started streaming map bytes into local RAM. That process was stopped and
no simulation file was saved; the byte count was not measured. The live
viewer uses **`data_size_limit=0`**, which the new script uses together
with its independent response-size cap. The successful cluster preflight
is separate from that interrupted exploratory request.

Cluster staging directory:
`/lustre/work/kristero10/flamingo_tsz_comparison_20260914`.
The model is
`/lustre/work/kristero10/battaglia12_nine_fisher_comparison_models_20260913/moped`,
experiment `35fbe0a27078c4b34d8aac9aee2b95fb4e2e4c977dc720ac516dd91176c302d6`.
The existing reference is
`/lustre/work/kristero10/so9_fisher_moped524k_20260913/fixed_noise_observation`.

```bash
/home/anaconda3/bin/python3 \
  /lustre/work/kristero10/flamingo_tsz_comparison_20260914/code/preflight.py \
  --output /lustre/work/kristero10/flamingo_tsz_comparison_20260914/preflight/metadata_manifest.json \
  --bundle /lustre/work/kristero10/battaglia12_nine_fisher_comparison_models_20260913/moped \
  --halfdome-observation /lustre/work/kristero10/so9_fisher_moped524k_20260913/fixed_noise_observation
```

Validation is complete: focused input-validation tests passed locally and on
the cluster; the matching-runtime operator probe passed; all four full map
stages ran; the fresh HalfDome inputs matched the saved reference; bundle
checksums and 32 reference log probabilities passed; and small result
artifacts passed transfer checksums. Three primary plots and the additional
proposal-support diagnostic were visually inspected. The run is a completed
comparison with failed FLAMINGO original-prior sampling, not a claim of
calibrated FLAMINGO posterior constraints.

## Sources

- [FLAMINGO integrated-map documentation](https://dataweb.cosma.dur.ac.uk:8443/flamingo/lightcones/integrated_lightcones.html#integrated-thermal-sz-maps)
- [Feedback model names and paths](https://dataweb.cosma.dur.ac.uk:8443/flamingo/simulations/hydro.html)
- [Shell redshift coverage](https://dataweb.cosma.dur.ac.uk:8443/flamingo/lightcones/healpix_shell_redshifts.html)
- [Live viewer implementation](https://dataweb.cosma.dur.ac.uk:8443/flamingo/_static/viewer.js)
- [Yang et al. integrated-map methodology](https://academic.oup.com/mnras/article/548/2/stag625/8571439)
- HalfDome `SO_MOPED_FIXED_NOISE_DIAGNOSIS.md` and
  `SO_NINE_FISHER_MOPED524K_OVERLAY.md`, checked in the current workspace.
