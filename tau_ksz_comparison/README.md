# Local HalfDome tau and kSZ comparison

From **Windows PowerShell**, run:

```powershell
wsl.exe -d myrootfs -u kn18001 -- bash /home/cbllover/HalfDome/tau_ksz_comparison/run_local.sh
```

From an existing **myrootfs WSL terminal logged in as kn18001**, run:

```bash
bash /home/cbllover/HalfDome/tau_ksz_comparison/run_local.sh
```

The defaults in `settings.toml` select the existing development checkout at
`/home/kn18001/.julia/dev/XGPaint`, the installed scientific Python environment,
20 CPU threads, NSIDE=4096, and lmax=12287. The launcher does not install packages.
It computes the theory, paints both maps, computes their spectra, and saves:

```
/home/cbllover/HalfDome/tau_ksz_comparison/outputs/nside4096/tau_ksz_comparison.png
/home/cbllover/HalfDome/tau_ksz_comparison/outputs/nside4096/tau_ksz_comparison.pdf
```

The same directory contains `maps.h5`, `map_spectra.npz`,
`class_sz_spectra.npz`, `comparison_arrays.npz`, `profile_table.h5`,
`catalog_bounds.toml`, the effective settings, source SHA256 hashes, complete
CLASS-SZ input parameters, and `painting.log`. The NPZ files store numeric arrays
without requiring pickle. Completed stages are reused on a subsequent identical
run; interrupted map painting restarts from the beginning of the catalogue.
An output directory is rejected if its settings, catalogue timestamp/size, or
scientific source files have changed. Choose `--output /new/output/directory` in
that case. Existing XGPaint source and inherited profile caches are not edited.

## Physics and numerical conventions

The input is `/home/cbllover/HalfDome/lightcone_100.hdf5`. Every valid halo is used,
without an additional lower/upper mass cut or redshift selection. A complete
scan of this local file found **85,224,251 valid halos**, covering
**z=0.00214159--3.85542774** and **physical M200c=6.58228e12--3.79976e15 Msun**.
"All" means the finite support of this catalogue, not unobserved halos of
arbitrarily small mass or beyond the lightcone. CLASS-SZ integrates over that
same measured mass/redshift rectangle using the Tinker08 M200c mass function;
it represents an ensemble prediction, not a fit to this catalogue's counts.

The cosmology is the published HalfDome cosmology: h=0.6774, Omega_m=0.3089,
Omega_b=0.0486, sigma8=0.8159, n_s=0.9667, with massless neutrinos.
See the [HalfDome simulation paper](https://arxiv.org/abs/2407.17462).
Masses in the catalogue are interpreted as Msun/h and divided by h for XGPaint;
CLASS-SZ's mass integration bounds are expressed in Msun/h. `Velocity` is used
as peculiar velocity in km/s. The direction comes from `Position`; its common
distance-unit factor cancels in the direction and radial velocity.

For each halo, the existing XGPaint BattagliaTauProfile computes

    tau(theta) = sigma_T integral n_e dl_physical
    (DeltaT/T)(theta) = -tau(theta) v_los/c.

Its B16 AGN profile is proportional to

    rho_gas(x) = f_b rho_crit(z) rho0 (x/xc)^gamma
                 [1+(x/xc)^alpha]^[-(beta+gamma)/alpha],
    x = r/R200c, xc=0.5, gamma=-0.2,
    rho0 = 4000 (M200c/1e14 Msun)^0.29 (1+z)^(-0.66),
    alpha = 0.88 (M200c/1e14 Msun)^(-0.03) (1+z)^0.19,
    beta = 3.83 (M200c/1e14 Msun)^0.04 (1+z)^(-0.025).

The current local implementation includes the baryon fraction and a factor
0.9 in its free-electron density. CLASS-SZ uses `gas_profile=B16`, AGN mode,
the same baryon fraction, and `f_free=0.9`. CLASS-SZ's default electron molecular
weight is 1.14; XGPaint uses its explicit hydrogen/helium/electron mass formula.
This small remaining convention difference is not removed by a fitted amplitude.

**The default cutoff geometries differ:** XGPaint integrates the native profile
along the line of sight and then stops painting at projected angular radius
4 theta200c. CLASS-SZ truncates the three-dimensional gas density at 4 R200c.
Both are left in their native form. Equal cutoff numbers therefore do not make
the two transforms mathematically identical. The figure tests the existing
implementation under the requested settings; exact agreement with another
person's image cannot be asserted without their original inputs and conventions.

XGPaint's existing profile is tabulated in scaled angular radius, log(1+z),
and log10(M200c). Mass and redshift bounds come from the complete scan. This
avoids the hardcoded bounds of inherited interpolators. Off-grid and corner
values are compared against direct XGPaint evaluation; production requires
less than 1% maximum relative error across 150 checks. Radii below 1e-7 R200c
use the central table value. This is a numerical cusp regularization, not a
halo mass/redshift cut. The painter uses XGPaint's ring/disc queries and assigns
disjoint rings to workers, preventing lost updates where profiles overlap.

Spectra use full-sky `healpy.anafast` after subtracting each map's monopole.
The upper panels show 1e7 D_l(tau) and 1e13 D_l(DeltaT/T), with
D_l=l(l+1)C_l/(2pi). The lower panels show the map/theory C_l ratios, equivalent
to D_l ratios at the same multipole. CLASS-SZ's `cl_ee()` and `cl_ksz()` already
return dimensionless D_l; this factor must not be applied a second time.
The [CLASS-SZ source](https://github.com/CLASS-SZ/class_sz) documents these
accessors and the B16 implementation. Ratios use positive log-space PCHIP
interpolation only inside the theory's multipole range.

There is no instrumental beam, added noise, sky mask, amplitude adjustment,
shot-noise subtraction, or pixel-window deconvolution. Both maps are halo-only;
diffuse gas and patchy reionization are absent. Pixel-centre sampling and the
finite catalogue can affect agreement, especially near the resolution limit.
kSZ also weights the realized coherent halo velocities, while CLASS-SZ uses
ensemble halo-model velocity statistics. A deficit at low ell must be measured,
not imposed to resemble the supplied picture.

## Other useful commands

Extend to lmax=24575, matching the far-right extent of the supplied upper panels:

```bash
bash /home/cbllover/HalfDome/tau_ksz_comparison/run_local.sh --nside 8192
```

The two Float64 maps require approximately 3 GiB at NSIDE=4096 or 12 GiB at
NSIDE=8192, before catalogue chunks and harmonic-transform working memory.
The higher resolution substantially increases runtime and peak memory.

For a quick software check with the first 20,000 real catalogue halos:

```bash
bash /home/cbllover/HalfDome/tau_ksz_comparison/run_local.sh --smoke --threads 4
```

This deliberately incomplete, low-resolution result is labelled SMOKE and
cannot validate scientific agreement. Use `--stage scan`, `--stage theory`,
`--stage maps`, `--stage spectra`, or `--stage plot` to run an individual stage.
Use the same config and output flags when revisiting an existing run.

To change figure styling and regenerate the PNG/PDF from completed spectra:

```bash
bash /home/cbllover/HalfDome/tau_ksz_comparison/run_local.sh --stage plot
```

This plot-only command uses the settings recorded with the saved spectra and
preserves the original scientific provenance. It does not rerun map painting,
CLASS-SZ, or harmonic transforms. The legend labels are `XGPaint` and `CLASS-SZ`;
CLASS-SZ is drawn as a black dashed line.

The additional single-panel halo-kSZ comparison uses the red halo-curve style
and axis limits of the supplied Websky reference image: logarithmic ell=70--5500,
linear D_ell=0.1--2.6 microkelvin squared. Generate it from saved spectra with:

```bash
/home/kn18001/.conda/envs/class_sz/bin/python \
  /home/cbllover/HalfDome/tau_ksz_comparison/plot_ksz_halo_reference.py
```

It creates `ksz_halo_reference_comparison.png` and `.pdf` in the run directory,
plus separate numerical arrays and metadata. XGPaint is red and CLASS-SZ B16
1h+2h is black dashed. The conversion is
`D_ell[microK^2] = D_ell[DeltaT/T] * (T_CMB[K] * 1e6)^2`, using the saved CLASS
temperature or its default 2.7255 K. The XGPaint curve is the measured HalfDome
halo spectrum, not digitized Websky data or an amplitude fit. The existing
four-panel figures, spectra, maps, and their provenance are preserved.

The numerical painter regression compares the new accumulation routine against
XGPaint's direct serial painter, including overlapping halos and polar/seam cases:

```bash
JULIA_NUM_THREADS=4 /home/kn18001/.juliaup/bin/julia \
  --project=/home/kn18001/.julia/environments/v1.12 \
  /home/cbllover/HalfDome/tau_ksz_comparison/test_painting.jl
```
