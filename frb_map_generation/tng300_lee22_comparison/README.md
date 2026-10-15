# TNG300-1-Dark / HalfDome Lee22 halo-DM comparison

This workflow paints halo-only dispersion measures to source redshift 1 with
the existing Lee22 Table A2 no-concentration implementation. It keeps 120,000
unique NSIDE 4096 sightlines and seed 42. Both simulations use exactly the same
sky directions; their halo realizations are independent.

The profile source `../lee2022_frb_dm_profile.jl` is **unchanged**. Its SHA256 is
`1fe10765ba8e38ae42c0cf1d20630babff0b8b10aa2a41e1a617bdbc50ae8806`, identical to
the successful 2026-09-17 comparison. The selection uses `M200c` in physical
solar masses, and gas is confined to a **sphere** of radius `R200c`.

## Physical and numerical choices

The retained conventions are `normalization=xgpaint_ne2d`, `n0_pivot=mcut`,
`concentration_mode=none`, and `redshift_scaling=physical`. The wrapper supplies
`P0=200*n0` and `beta_XGPaint=alpha*beta_prime-gamma`; XGPaint's own gas-to-electron
conversion is unchanged. This is the empirically successful convention in the
prior comparison, not an independent resolution of the normalization ambiguity
in Lee22's printed equation 9.

For projected impact parameter `b<R200c`, the observed halo contribution is

```text
DM_halo = 2/(1+z_h) * integral_0^sqrt(R200c^2-b^2) n_e(sqrt(b^2+l^2)) dl .
```

The existing profile code handles the length/electron-density unit conversion to
`pc cm^-3`. The chord goes continuously to zero at the halo boundary. There is
no diffuse IGM, FRB-host, or Milky Way component and no additional scatter.
The smooth fitted profile is continued below its fitted inner radius of
`0.04 R200c`, as in the established implementation; this is distinct from the
strict restriction on halo mass requested here.

The original angular interpolation has appreciable relative errors for some
grazing rays at low redshift. The new native-TNG cache failed a direct-integral
check. Therefore the final comparison evaluates the **unchanged spherical chord
function directly for both simulations**, using the existing pixel query and
mass-window accumulator. Eighty checks per cosmology compare the calculation
with tighter quadrature and verify the XGPaint density mapping. The saved
`numerical_audit.json` compares the cached and direct HalfDome calculations on
identical halos and rays; mean changes are about 1.3%, with unchanged hit masks.

TNG uses `h=0.6774`, `Omega_m=0.3089`, `Omega_b=0.0486`; HalfDome retains
`h=0.68`, `Omega_m=0.31`, `Omega_b=0.049`. The Lee22 mass pivot and critical
density are evaluated in each simulation's native cosmology. No profile
amplitude is fitted to Ralf's data or rescaled to force agreement.
The DMO particle masses represent the total matter density, so their `M200c`
is used as the total halo mass. XGPaint applies the baryon fraction in the gas
normalization once; no extra baryon-fraction multiplier is introduced.

## Mass selections and Ralf reference

The common physical mass support is the intersection of the Lee22 fit ranges
`1e13 <= M200c/(Msun/h) < 10^14.8` under the two h values:

| Window | M200c interval in physical Msun |
|---|---|
| supported_low | `[1.4762326542663123e13, 1e14)` |
| supported_high | `[1e14, 9.278784477649915e14)` |
| supported_all | `[1.4762326542663123e13, 9.278784477649915e14)` |

The first two windows are disjoint. Their per-ray DMs must sum to the third.
There is no low-mass extrapolation or shape clipping within this selection.

Ralf's available arrays do not provide these thresholds. A per-ray total for
`1e13-1e14 Msun` cannot be narrowed to `1.476e13-1e14 Msun` without knowing the
individual foreground halo contributions. Cutting the ray DM, subtracting
histograms, or rescaling the reference is not a valid substitute. The matched
PDF and percentage figure therefore compares TNG300-1-Dark with HalfDome.
A separate contextual figure includes Ralf's `1e13-1e14 Msun` curve and states
the differing mass cuts explicitly. It does not give mass-matched residuals to
Ralf. An exact three-way comparison needs finer reference data.

## TNG catalogue and lightcone

The user approved the public FoF/SUBFIND catalogue. It differs from the Rockstar
halo catalogue used in Lee22's calibration and can give different centres,
counts, and masses even at the same spherical-overdensity definition.

Only `Group_M_Crit200`, `GroupPos`, and `Group_R_Crit200` are requested from the
official TNG API. The key is read from `TNG_API_KEY` and sent in an authentication
header; it is never saved in a result or source file. The complete mass field
is read for every snapshot before making the mass selection. No completeness
assumption about mass ordering is used.

The API field dumps are contiguous, uncompressed HDF5. Position/radius reads
verify this layout and the actual dataset offset, then receive the byte prefix
through the largest selected row. This avoids transferring unneeded low-mass
rows. The temporary file's unreceived rows are never used. Receipts distinguish
full-field SHA256s from received-prefix SHA256s. Exact comparison with a fully
downloaded snapshot 99 control is recorded separately. Compact catalogues keep
the original halo IDs and have their own complete SHA256s. Redundant raw staging
files are deleted after validating the compact products.

The unit conversions are

```text
M200c [Msun] = Group_M_Crit200 * 1e10 / h_TNG
position [comoving Mpc] = GroupPos / (1000*h_TNG)
R200c [physical Mpc] = Group_R_Crit200 / (1000*h_TNG*(1+z_snapshot)).
```

Each snapshot's radii are checked against
`M200c = (4*pi/3) * 200*rho_critical(z) * R200c^3`.

All snapshots 50 through 99 are used. Comoving radial shell boundaries lie
halfway between snapshot distances. The observer is a seed-42 uniform position
in the TNG box; box orientation and observer stay fixed through the shells.
Periodic tiling covers the full sky with no radial gaps and no per-shell random
reorientation. Halo centres select shell membership; a selected halo contributes
its full sphere, as in the HalfDome painter. Halo positions/masses are taken from
the nearest snapshot; the gas model is evaluated at continuous lightcone redshift.
The lightcone records snapshot, original halo ID, periodic tile, mass and position.

This is a halo-centre lightcone and smooth-profile painting, not Ralf's gas-cell
ray tracing. The finite TNG box repeats structure and lacks larger-scale modes.
One observer and one periodic realization do not measure cosmic variance.

## Running and outputs

### Matched physical mass bin 13--14 and signed 20% panels

The requested follow-up uses **1e13 <= M200c/Msun < 1e14** for all three
catalogues, matching `DMhalo_r200_mass13to14.npy` from Ralf. This explicitly
extends Lee22 below its calibrated mass limit, approximately 1.476e13 Msun.
It is a separate case and retains the original calibrated-range results.
The output root is `frb_map_generation/outputs/tng300_lee22_mass13to14_20260921/`.

From this workflow directory, with `TNG_API_KEY` configured for the download:

```sh
export TNG_LEE22_CASE=mass13to14
python compare_mass13to14.py --prepare
python download_tng.py --snapshots 50:99 --workers 3
python build_lightcone.py
python run_tng.py --halfdome --ray-configuration ../outputs/tng300_lee22_mass13to14_20260921/ray_configuration.h5
python run_tng.py --ray-configuration ../outputs/tng300_lee22_mass13to14_20260921/ray_configuration.h5
python compare_mass13to14.py
```

For a remote authenticated transfer, `download_tng.py --api-key-stdin` accepts
the key from one stdin line without putting it in a file or command argument.
The common sky directions are copied from the previous run; only the halo
mass selection changes. The painter accepts arbitrary configured mass windows
and retains the original profile, chord integral, and XGPaint accumulator.

The new three-way plot gives **100*(p_HD/p_Ralf-1)** and
**100*(p_TNG/p_Ralf-1)**, followed by **100*(p_HD/p_TNG-1)**. Each percentage
axis spans **-20% to +20%**. Values outside these limits are left unchanged
and clipped by the axes, with **no arrows or replacement markers**. The CSV
retains the full signed values. As before, both histogram bins must contain
at least ten rays; the densities condition on positive DM.

### Original calibrated-range case

Run Python analysis and Julia painting in the HalfDome WSL environment. Downloads
can run under Windows Python if WSL DNS is unavailable. From this directory:

```text
python preflight.py --offline --public-page ../../frb_map_generation/outputs/tng300_lee22_1r200c_20260921/tng300_dark_download_page.html
python download_tng.py --snapshots 50:99 --workers 3
python download_tng.py --verify-prefix
python run_halfdome.py --run
python run_tng.py --halfdome
python audit_numerics.py
python validate_geometry.py
python build_lightcone.py
python run_tng.py
python compare.py
```

`run_halfdome.py` creates the cached control and shared sightlines.
`run_tng.py --halfdome` creates the direct-integral HalfDome result used in the
final comparison. Existing scientific outputs are protected from overwrite.
The local runners cap address space at 12 GiB and record `/usr/bin/time -v`.
No PBS submission is included or performed. Any future cluster run approaching
40 GB must be reported to the user first.

Outputs are under `../outputs/tng300_lee22_1r200c_20260921/` relative to this
directory (absolute project root `/home/cbllover/HalfDome`).
The final products are `comparison.json`, `statistics.csv`, `pdf_bins.csv`, and
PDF/PNG/SVG figures in `plots/`. Per-ray HDF5 catalogues retain zero-DM rays.
Plotted PDFs are conditional on positive DM; underflow/overflow and zero mass
are counted separately. Percentage bins require at least ten rays in both
samples. The descriptive KS distance is reported without an independence-based
significance claim.

## Sources

- [Official TNG300-1-Dark catalogue and cosmology](https://www.tng-project.org/data/downloads/TNG300-1-Dark/)
- [Official TNG units and field definitions](https://www.tng-project.org/data/docs/specifications/)
- [Official API authentication and single-field downloads](https://www.tng-project.org/data/docs/api/)
- [Lee et al. 2022](https://arxiv.org/abs/2205.01710)
- [Rockstar catalogue access information](https://aust427.github.io/releases.html)
- [Konietzka et al. 2025 ray-tracing methods](https://arxiv.org/html/2507.07090v2)
- Repository implementation history: `../WHAT_CHANGED_FOR_TNG_AGREEMENT_20260917.md`.
