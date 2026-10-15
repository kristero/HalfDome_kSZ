# Lee22 normalization, FRB PDFs, and finite-sphere kSZ comparison

Results: `../outputs/lee22_pdf_ksz_comparison_20260922/`.

The latest user instruction sets **kSZ to 3R200c**. FRB PDFs retain **1R200c**,
source redshift 1, 120,000 random rays, seed 42, and physical M200c in
`[1e13, 1e14) Msun`. All new products use a separate output directory.

## Scientific implementation

`compare_dm.py` reads the retained HalfDome, TNG300-1-Dark, and Ralf ray arrays.
The two Lee22 versions have the same profile shape. Their electron-density
ratio is constant, `n_e,paper / n_e,previous = 1.6621465692764925`, established
by the preceding Figure 5 audit. Multiplying every previous-model ray by this
factor and making a new histogram exactly implements the paper normalization.
PDFs are conditional on positive DM; zero-hit probabilities are saved separately.
Percentages are signed, displayed within +/-20%, with full values in the CSV.
Residual bins require at least 10 counts in each compared sample.

`paint_ksz.jl` uses the existing, unmodified `lee2022_frb_dm_profile.jl` and the
installed XGPaint Battaglia16 AGN density profile. It paints

```text
Theta(n) = DeltaT/T = -sigma_T * sum_h [v_los,h/c * integral n_e,h dl_phys]
```

For impact parameter b, the finite physical line of sight has half-length
`sqrt((3 R200c)^2-b^2)`. Thus 3R200c truncates the three-dimensional sphere,
including the line of sight. It is not just a larger projected aperture applied
to an infinite column. There is no observer-DM factor `1/(1+z)` in optical depth.
Peculiar velocities are read in km/s. Halo bulk velocity is applied to all
electrons in each spherical profile; no internal gas velocity field is added.

Four NSIDE=4096 RING maps share pixel geometry: Lee22 previous and Battaglia16,
each for the restricted selection and all 85,224,251 catalogue halos. A cache
stores only the dimensionless chord mean. Radius, amplitude, and the exact
grazing-edge square root are evaluated separately. Workers own disjoint HEALPix
rings, avoiding overlapping writes. The paper-normalized Lee22 map is an exact
constant multiple, so its power is obtained by multiplying by 2.7627312177576138.

`compute_ksz.py` calls the established full-CLASS runner with B16 AGN, f_free=0.9,
Tinker08 M200c mass function, matched physical mass/redshift limits, and a
three-dimensional 3R200c cutoff. This is an independent **B16** prediction; it
does not claim CLASS-SZ implements Lee22. Map spectra use healpy anafast,
lmax=6000, iter=3, and monopole subtraction, one map at a time. CLASS-SZ arrays
already contain dimensionless D_ell. Both predictions are converted using
`T_CMB = 2.7255e6 microkelvin`; only map C_ell need the factor ell(ell+1)/(2pi).

`extract_websky.py` extracts the orange halo-only markers directly from the
original vector figure supplied with Stein et al. (2020), arXiv:2001.08787v1.
The plotted reference retains the paper's selection and compensation. It has
not been recalculated with the new HalfDome cuts or cutoff.

`plot_ksz.py` creates a two-selection comparison and a separate full-catalogue
figure with a linear vertical axis. Displayed bandpowers are (2ell+1)-weighted
means of D_ell. Signed B16/CLASS percentages are shown in +/-20% panels with no
arrows; the saved CSV is not clipped. `aperture_diagnostics.jl` calculates
per-halo volume-integrated electron counts, not kSZ power ratios.

## Reproduction

From `/home/cbllover/HalfDome` in WSL, use the following stages. The plotting
scripts can also run under Windows Python with access to the WSL checkout.
The reference vector figure is retained, so extraction does not need a download.

```bash
export HDF5_USE_FILE_LOCKING=FALSE
export JULIA_DEPOT_PATH=/home/kn18001/.julia
export PATH_TO_CLASS_SZ_DATA=/home/kn18001/class_sz_data_directory
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=20
PYTHON=/home/kn18001/.conda/envs/class_sz/bin/python
JULIA=/home/kn18001/.juliaup/bin/julia
SOURCE=frb_map_generation/lee22_pdf_ksz_comparison
RESULT=frb_map_generation/outputs/lee22_pdf_ksz_comparison_20260922

$PYTHON $SOURCE/compare_dm.py
$PYTHON $SOURCE/prepare_ksz.py
$JULIA --threads=20 $SOURCE/paint_ksz.jl $RESULT/ksz/settings.toml $RESULT/ksz/bounds.toml --validate-only
# Painting refuses to overwrite completed maps. Use a fresh output directory
# and fresh chord tables when changing the physical configuration.
$JULIA --threads=20 $SOURCE/paint_ksz.jl $RESULT/ksz/settings.toml $RESULT/ksz/bounds.toml
OMP_NUM_THREADS=4 TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=1 $PYTHON $SOURCE/compute_ksz.py theory
$PYTHON $SOURCE/compute_ksz.py spectra
$JULIA $SOURCE/aperture_diagnostics.jl $RESULT/ksz/settings.toml
$PYTHON $SOURCE/extract_websky.py
$PYTHON $SOURCE/plot_ksz.py
```

PyMuPDF is needed for vector extraction. CLASS-SZ 0.1.70.post40 was restored in
the existing Python environment using official PyPI distributions and CPU-only
TensorFlow dependencies. Package versions, download URLs, and checksums are in
the result's `software/` directory. Full CLASS, not a neural surrogate, produced
the spectra. Download installers and temporary rendering images were removed
after successful verification; production maps and useful provenance remain.

## Limits of the comparison

- Adopted painting/theory cosmology retains the FRB contract: h=0.68,
  Omega_m=0.31, Omega_b=0.049, sigma8=0.8159, n_s=0.9667. The HalfDome simulation
  was generated at h=0.6774, Omega_m=0.3089, Omega_b=0.0486. This small mismatch
  remains; the comparison is not a test using precisely the simulation cosmology.
- The low end of the restricted bin is below Lee22's h-scaled mass calibration.
  Full HalfDome additionally extends below/above its mass range and above z=2.
  The previous high-mass shape freeze at 10^14.8/h Msun is retained; density
  amplitude continues its fitted mass law. Extension to 3R200c also extrapolates
  beyond the fitted radial interval ending near 1.34R200c.
- Websky's halo-only component uses M200c>1e13 Msun and theta200c>0.5 arcmin,
  extends to z<4.6, and subtracts a uniform halo counterterm at overdensity 3.
  Our halo maps have no such compensation. This reference is not an
  apples-to-apples error denominator, particularly on large angular scales.
- Halo-model CLASS uses an ensemble mass function and velocity statistics;
  HalfDome uses one discrete halo/velocity realization. Scatter, nonspherical
  gas, internal gas motions, diffuse gas, and reionization are not modeled here.
- Maps use pixel-centre sampling with no beam or pixel-window deconvolution.
  The painter and projection were validated, but a separate higher-resolution
  spectrum convergence study was not performed. Do not interpret percent-level
  residuals as a unique test of the density normalization.
- PDF MAE is a descriptive histogram difference with a count cut, not a
  covariance-weighted goodness-of-fit statistic or uncertainty on the mean.

For numerical results, measured resources, and verification evidence, see
`RUN_REPORT.md`, `completion.json`, and `SHA256SUMS` in the output directory.
