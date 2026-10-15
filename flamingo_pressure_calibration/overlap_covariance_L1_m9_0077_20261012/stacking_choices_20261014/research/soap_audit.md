# Using SOAP instead of particles

FLAMINGO L1_m9 (L1000N1800) HYDRO_FIDUCIAL, SOAP-HBT `halo_properties_0077.hdf5` (z = 0). I accessed it read-only on maas at nice 19 on 2026-10-14. I compared it with the study's particle pass (4,588 centrals, inclusion weights).

All numbers come from files in `research/soap/`. The main ones are `soap_audit.json` (inventory, radii, validation, stacks, cost), `soap_read_timing.json`, `nfw_aperture_mapping.json` and `soap_vs_particle_stacks.png`. The scripts that produced them are in `research/soap/scripts/`.

## Verdict

**SOAP stores no radial profiles.**
- I searched all 2,113 dataset names for profile, radial, bin and shell: nothing matched.
- What SOAP has is cumulative (aperture) sums at a fixed set of radii.

**Its numbers are exact.** At identical radii they reproduce our particle sums to the level of its lossy compression (about ±0.05%).

**A SOAP-only stack is a coarse version of the particle stack:**
- **Pressure:** 8 cumulative points between 0.27 and 3.3 R200c.
- **Gas density:** up to 17 points between 0.01 and 9 R200c.
- **Accuracy:** about 1-3% per shell when compared over the same radial intervals.
- **What it misses:** pressure inside about 0.3 R200c, and the 3.3-4 R200c region that our calibration fits and cuts at.

**Speed:** it is about 100 times cheaper than a particle pass and covers every central. It is also the only gas information for the feedback variants away from z = 0 and 0.15.

## 1. What SOAP-HBT stores (inventory)

**The file.**
- Size and contents: 36.7 GB; 15,329,653 subhalo rows; 2,113 datasets.
- 15 top-level groups: BoundSubhalo, Cells, Code, Cosmology, ExclusiveSphere, Header, InclusiveSphere, InputHalos, Parameters, PhysicalConstants, ProjectedAperture, SO, SOAP, SWIFT, Units.
- Storage: chunks of 1,000 rows; every float is stored with a lossy 9-bit-mantissa filter (`FMantissa9` / `DMantissa9`, i.e. ≤ 2^-10 relative rounding). Particle counts are lossless.
- Masks: the SO groups 2500c, 1000c, 100c, 50c, BN98 and 5xR500c, the 500 / 1000 / 3000 kpc spheres and all projected apertures are filled only for subhaloes with ≥ 100 bound particles. 200c, 200m, 500c and the 10-300 kpc spheres are filled for every subhalo.
- Coverage: in `SO/200_crit`, GasMass is non-zero in 11,197,001 rows and ComptonY in 3,795,818 rows.

**Gas-related content by aperture family:**

| Family | Radii | Particles counted | Gas datasets (per aperture) |
|---|---|---|---|
| `SO/*` (9) | 2500c, 1000c, 500c, 200c, 100c, BN98 (Δc = 101.8 at z = 0), 200_mean (Δc = 60.9), 50c, 5xR_500c | all particles in the sphere (inclusive) | `SORadius`, `TotalMass`, `GasMass`, `HotGasMass` (T > 1e5 K), **`ComptonY`**, **`ComptonYWithoutRecentAGNHeating`**, `GasTemperature` (mass-weighted; also `WithoutCoolGas`, `WithoutRecentAGNHeating`, both), `GasComptonYTemperature`(+noAGN), `SpectroscopicLikeTemperature`(+noAGN), `XRayLuminosity` / `InRestframe` / `XRayPhotonLuminosity` (3 bands, +noAGN), `ThermalEnergyGas`, `KineticEnergyGas`, `DopplerB` (kSZ towards the first light-cone observer), `NumberOfGasParticles`, metal / Fe / O fractions, `MassFractionSatellites` / `External`. Core-excised (r > 0.15 R500c) variants exist for 500c only. |
| `InclusiveSphere/*` (8) | 10, 30, 50, 100, 300, 500, 1000, 3000 kpc, **physical** (z = 2 check below) | all particles | `GasMass`, `GasTemperature` (mass-weighted, +noAGN), `NumberOfGasParticles`, `StarFormingGasMass`, metal fractions, `HalfMassRadiusGas`, `KineticEnergyGas`. **No ComptonY, no X-ray.** |
| `ExclusiveSphere/*` (8) | same radii | particles bound to this subhalo only | same datasets as inclusive |
| `ProjectedAperture/*/proj{x,y,z}` (4) | 10, 30, 50, 100 kpc | bound, projected | `GasMass`, `NumberOfGasParticles`, gas velocity dispersion, half-mass radius; no ComptonY |
| `BoundSubhalo` | none (HBT bound set) | bound | `GasMass`, `GasTemperature` (4 variants), `StarFormingGasMass`, `HalfMassRadiusGas`, ... |

**Missing everywhere:** an electron number, a hydrogen mass or fraction (only metal fractions), any per-phase mass other than `HotGasMass` (SO only) and `StarFormingGasMass`, and any radial binning.

**Units.**
- Conversion factors are given by the `Conversion factor to physical CGS` attributes:
  - masses: 1e10 Msun;
  - ComptonY: 9.52e48 cm² = 1 Mpc²;
  - temperatures: K;
  - X-ray luminosities: 6.44e33 erg/s.
- `SORadius` is stored comoving (a-exponent 1, `Value stored as physical` = 0). Masses, Y and temperatures are stored physical.

**Apertures are physical kpc.** At z = 2 (0037, a = 1/3), the largest `HalfMassRadiusGas` in the 10 kpc spheres is 0.02988 cMpc = 9.96 physical kpc. A 10 comoving-kpc aperture could not exceed 3.3 physical kpc.

**Other files have the same layout.** HYDRO_STRONG_AGN 0077 (37.0 GB, 15,345,294 rows) and HYDRO_FIDUCIAL 0037 (z = 2, 32.4 GB, 17,523,788 rows) have identical dataset names, dtypes and shapes. The only differences are in `Parameters/RecentlyHeatedGasFilter`:

| File | ΔT_AGN [K] | T_min - T_max [K] | a_limit (15 Myr) |
|---|---|---|---|
| FIDUCIAL 0077 | 8.80e7 | 8.80e6 - 1.756e8 | 0.99896 |
| STRONG_AGN 0077 | 1.20e8 | 1.20e7 - 2.40e8 | 0.99896 |
| FIDUCIAL 0037 | 8.80e7 | 8.80e6 - 1.756e8 | 0.33229 |

The "recent AGN" cut therefore scales with each variant's heating temperature.

## 2. Which radii the apertures probe (x = r / R200c)

Measured medians (16-84%) for the study haloes at z = 0:

| Aperture | 12.5-12.7 | 12.9-13.1 | 13.3-13.5 | 13.7-13.9 | 14.1-14.3 | 14.5-14.7 | 14.9-15.1 | 15.3-15.5 |
|---|---|---|---|---|---|---|---|---|
| R200c [Mpc] | 0.328 | 0.449 | 0.613 | 0.832 | 1.125 | 1.520 | 2.066 | 2.754 |
| 2500c | 0.326 (0.30-0.34) | 0.312 | 0.301 | 0.292 | 0.283 | 0.280 (0.25-0.30) | 0.272 | 0.281 |
| 1000c | 0.495 | 0.482 | 0.473 | 0.466 | 0.460 | 0.459 | 0.452 | 0.466 |
| 500c | 0.674 | 0.666 | 0.656 | 0.655 | 0.650 | 0.651 | 0.649 | 0.656 |
| 100c | 1.34 | 1.34 | 1.36 | 1.36 | 1.36 | 1.35 | 1.35 | 1.36 |
| BN98 | 1.33 | 1.33 | 1.35 | 1.35 | 1.35 | 1.34 | 1.34 | 1.35 |
| 200m | 1.64 | 1.66 | 1.68 | 1.67 | 1.67 | 1.66 | 1.65 | 1.67 |
| 50c | 1.78 | 1.80 | 1.82 | 1.82 | 1.81 | 1.80 | 1.79 | 1.81 |
| 5xR500c | 3.37 (3.25-3.43) | 3.33 | 3.28 | 3.28 | 3.25 | 3.26 (3.13-3.33) | 3.24 | 3.28 |
| 10 kpc | 0.031 | 0.022 | 0.016 | 0.012 | 0.009 | 0.007 | 0.005 | 0.004 |
| 30 kpc | 0.091 | 0.067 | 0.049 | 0.036 | 0.027 | 0.020 | 0.015 | 0.011 |
| 50 kpc | 0.15 | 0.11 | 0.082 | 0.060 | 0.044 | 0.033 | 0.024 | 0.018 |
| 100 kpc | 0.31 | 0.22 | 0.16 | 0.12 | 0.089 | 0.066 | 0.048 | 0.036 |
| 300 kpc | 0.91 | 0.67 | 0.49 | 0.36 | 0.27 | 0.20 | 0.15 | 0.11 |
| 500 kpc | 1.52 | 1.11 | 0.82 | 0.60 | 0.44 | 0.33 | 0.24 | 0.18 |
| 1000 kpc | 3.05 | 2.23 | 1.63 | 1.20 | 0.89 | 0.66 | 0.48 | 0.36 |
| 3000 kpc | 9.14 | 6.69 | 4.89 | 3.61 | 2.67 | 1.97 | 1.45 | 1.09 |

**z = 0.**
- SO apertures sit at nearly fixed x with 5-10% scatter, set by concentration.
- R100c and R_BN98 coincide within 1%, so the 9 SO apertures give 8 distinct radii.
- An NFW model with each bin's median SOAP c200c reproduces the measured medians to 1% for 2500c-500c and 0.3-3% for the outer apertures (`nfw_aperture_mapping.json`).

**z = 2 (NFW model, not measured).**
- Ω_m(z) = 0.92, so Δ_BN98 = 171 and Δ_200m = 184 (in units of ρ_crit). R_BN98 and R200m fall at x ≈ 1.07 and 1.04, i.e. on top of R200c.
- That leaves about 7 distinct SO radii.
- At fixed M200c, R200c is 0.48 times its z = 0 value, so the kpc spheres move to about twice the x listed above.

## 3. Validation against the particle pass

**Exact radii** (R200c, R500c, 5R500c). SOAP (`selected_halos.npz`) is compared with the particle aperture sums in `apertures.npz` (all gas, inclusive):

| Ratio SOAP / particles (all 4,588) | median | 16-84% | 0.5-99.5% |
|---|---|---|---|
| Y(<R200c) | 1.0000014 | 0.99946 - 1.00055 | 0.9960 - 1.0034 |
| Y(<R500c) | 0.99998 | 0.99941 - 1.00055 | 0.9954 - 1.0036 |
| Y(<5R500c) | 1.000008 | 0.99951 - 1.00049 | 0.9980 - 1.0023 |
| M_gas(<R200c) | 0.99998 | 0.99938 - 1.00060 | 0.9953 - 1.0043 |
| M_gas(<R500c) | 0.99998 | 0.99931 - 1.00064 | 0.9940 - 1.0041 |
| Y_noAGN(<R200c) | 1.0000056 | 0.99946 - 1.00054 | 0.9960 - 1.0035 |

- The ±0.055% width is the 9-bit mantissa rounding.
- Above 10^13.5 the 0.5-99.5% range is within ±0.3%. Below 10^13.5 it is ±1%, from low-gas haloes.
- The single worst haloes are off by 12% at R200c and 22% at R500c; these are haloes with few gas particles. This is the same exception documented in SYNTHESIS.md.

**All other apertures.** The particle cumulative is interpolated at each halo's own radius (monotone PCHIP in ln x through the 67 X200 edges).
- The interpolation itself scatters by 0.5% at 16-84% (checked against the exact R500c sums: median 1.00004, 0.9952-1.0059).
- For every SO aperture, and for x ≥ 0.025, SOAP/particle medians are 0.9990-1.0001 for Y, Y_noAGN and M_gas in all three mass groups. The 16-84% widths equal the interpolation scatter.
- **Exclusive spheres** against central-bound particles (owner 0): medians 0.9995-1.0002.
- **Inclusive spheres** against all particles: medians 0.9984-1.0003.
- The definitions therefore match:
  - SO and InclusiveSphere include everything in the sphere (satellites, neighbours, unbound gas).
  - ExclusiveSphere and BoundSubhalo include only gas bound to the central subhalo; satellites' own bound gas is excluded.
- Spheres below 0.025 R200c (10 and 30 kpc for M ≳ 10^13.5) cannot be checked: our grid does not resolve them.

**Definition differences you must handle yourself.**

| Item | What SOAP does | Consequence |
|---|---|---|
| Recent AGN | Y_noAGN only in SO apertures. The filter is a > a_limit (15 Myr) and T within [0.1, 10^0.3] × ΔT_AGN, so it is variant dependent. GasMass and the kpc spheres have no noAGN version. | Recently heated gas carries 1.6% (R2500c) / 0.70% (R1000c) / 0.41% (R500c) / 0.22% (R200c) / 0.08% (5R500c) of the Y-weighted total at 10^12.5-13.5. It carries ≤ 0.06% above 10^13.5 and ≤ 0.006% above 10^14.5. The per-halo median is 0. |
| Inclusive vs exclusive | SO and Inclusive = all gas; Exclusive = bound to the central only. | The only owner split available is M_gas(inclusive) − M_gas(exclusive), i.e. satellites + neighbours + unbound, and only in kpc spheres. No excision, partition or overlapper classes. |
| Radii | SORadius is comoving; kpc apertures are physical. | Apply a^1 to SORadius at z > 0. |
| Temperature cuts | GasMass and ComptonY: none. HotGasMass and `WithoutCoolGas`: T > 1e5 K. Core excision: 500c only. | |
| Electrons | Not stored. | Converting GasMass with full ionisation and X = 0.76 overestimates the particle electrons: +1.4-1.7% (R200c-5R500c, ≥ 10^13.5), +2.0-2.3% (R500c), +3.5-4.6% (R2500c), +2.4-9.4% in groups. From the particle sums, about 1-1.4% of this is composition (X_H = 0.74-0.745 instead of 0.76 outside 0.3 R200c). Ionisation (neutral and star-forming gas) is negligible outside 0.3 R200c but 6-25% in cores below 0.1 R200c. |
| Pressure in kpc spheres | Not stored. | The proxy Y ≈ σ_T k M_gas T_mw / (m_e c² μ_e m_p), μ_e = 1.136, is high by +1.1% (5R500c, 3000 kpc), +1.3-1.8% (R200c, 1000 kpc), +2.2-2.4% (300 kpc), +2-4% (R2500c). Its scatter is ±0.1-0.5% in clusters and several % in groups. It is calibratable against the SO ComptonY. |
| Compression | 9-bit mantissa | 0.05% rms per value; negligible for stacks |

## 4. A SOAP-only stacked profile (tested at z = 0 on the same haloes and weights)

**Method.**
- **Normalisation.** Per halo, Ỹ(<r) = Y_noAGN · m_e c²/σ_T / (P200 R200c³) and M̃(<r) = M_gas / (f_b M200c), with the study's P200 and f_b = 0.1588.
- **Mean aperture position.** For each 0.2-dex bin, the inclusion-weighted mean aperture position x̄ is used. Apertures closer than 5% (SO) or 10% (SO + kpc) to their neighbour are merged: R100c is dropped next to R_BN98.
- **Estimator A:** difference of the stacked cumulatives between consecutive x̄, divided by the shell volume.
- **Estimator B:** mean over haloes of each halo's own shell (its own radii), placed on the x̄ edges.
- **Reference:** the particle stack on the 67-shell grid, averaged over exactly the same [x̄_a, x̄_b] interval.
- **Usable shell:** median ≥ 50 gas particles per halo in the shell, and both edges ≥ 0.025 R200c (except the inner integral [0, x_1]).

**Pressure shells** (log M = 13.7-13.9, 324 haloes; ratio = SOAP / particle over the same interval):

| Shell | x | P/P200 (particles) | A | B | SOAP/particles at same per-halo radii |
|---|---|---|---|---|---|
| 0 → 2500c (integral) | 0-0.29 | 1.75 | 1.029 | 0.972 | 0.998 |
| 2500c → 1000c | 0.29-0.46 | 0.73 | 0.991 | 0.979 | 1.001 |
| 1000c → 500c | 0.46-0.65 | 0.354 | 0.986 | 0.989 | 0.999 |
| 500c → 200c | 0.65-1.00 | 0.138 | 1.001 | 0.999 | 1.001 |
| 200c → BN98 | 1.00-1.35 | 0.0504 | 1.026 | 0.993 | 1.001 |
| BN98 → 200m | 1.35-1.69 | 0.0214 | 1.027 | 0.974 | 1.000 |
| 200m → 50c | 1.69-1.83 | 0.0126 | 1.032 | 0.957 | 0.996 |
| 50c → 5xR500c | 1.83-3.25 | 0.00359 | 0.955 | 0.988 | 1.001 |

**Summary over all bins:**

| log M200c | P usable / shells | P max \|A−1\| | P max \|B−1\| | P median \|A−1\| / \|B−1\| | Forward model: R2500c / max other | ρ (SO+kpc) usable, x range | ρ max \|B−1\| (median) |
|---|---|---|---|---|---|---|---|
| 12.5-12.7 | 1/8 | 0.032 | 0.002 | 0.032 / 0.002 | 0.980 / 0.046 | 1/11, 1.8-3.0 | 0.005 |
| 12.9-13.1 | 5/8 | 0.072 | 0.033 | 0.036 / 0.011 | 1.017 / 0.029 | 5/13, 0.66-3.3 | 0.010 (0.007) |
| 13.1-13.3 | 7/8 | 0.044 | 0.020 | 0.027 / 0.013 | 1.021 / 0.015 | 6/13, 0.30-3.3 | 0.015 (0.007) |
| 13.5-13.7 | 8/8 | 0.034 | 0.034 | 0.023 / 0.016 | 1.030 / 0.012 | 7/11, 0.14-3.3 | 0.011 (0.003) |
| 13.9-14.1 | 8/8 | 0.044 | 0.046 | 0.023 / 0.020 | 1.025 / 0.008 | 9/12, 0.05-3.1 | 0.021 (0.007) |
| 14.3-14.5 | 8/8 | 0.033 | 0.037 | 0.016 / 0.019 | 1.026 / 0.006 | 12/15, 0.04-3.2 | 0.010 (0.004) |
| 14.5-14.7 | 8/8 | 0.041 | 0.038 | 0.018 / 0.021 | 1.028 / 0.006 | 11/14, 0.03-3.2 | 0.019 (0.006) |
| 14.9-15.1 | 8/8 | 0.036 | 0.049 | 0.013 / 0.020 | 1.034 / 0.005 | 9/13, 0.05-3.2 | 0.023 (0.004) |

The full table for all 15 bins, every shell, both estimators and both fields is in `soap_audit.json` → `stack_summary` / `stacks_full`. The figure is `soap_vs_particle_stacks.png`.

**What the comparison shows.**
- **SOAP equals particles at the same radii.** At each halo's own radius, the SOAP and particle stacks agree within 0.3% above 10^13.7. Below 10^13 the difference reaches 2%, from interpolating a few particles. All remaining error comes from compressing per-halo SO radii into one x̄ per bin.
- **The R2500c point is biased in both estimators.** Haloes with larger R2500c/R200c (more concentrated) also have more concentrated gas.
  - Estimator A puts +2-3% into the inner integral and −4% into the outermost shell, and up to ±7-10% into the narrow 200m→50c shell (Δx/x = 8%).
  - Estimator B is 2-4% low in the inner integral and beyond R200c.
- **Forward modelling is the clean way to use SOAP.** Predict each halo's aperture integrals from a profile model at its own radii, then fit. For the inclusion-weighted mean of the fine stacked profile evaluated at each halo's own radii:
  - SOAP agrees within ≤ 1% at every SO aperture except R2500c (+2.1 to +3.4% above 10^12.9; the concentration correlation above);
  - fixed-kpc apertures agree within ≤ 0.5%;
  - at 10^12.5-12.7 the outer SO apertures deviate by up to 4.6%.
- **Gas density does better than pressure.** Combining SO and InclusiveSphere masses with per-halo shells (B) gives 9-12 usable shells above 10^13.3 from x ≈ 0.03-0.16 out to 3.2-3.6 R200c. The median deviation is 0.3-0.7% and the maximum 2.3% (3.8% in the 6-halo top bin). Estimator A fails on narrow mixed shells (up to 18%).

**What a SOAP-only profile cannot constrain.**
- **Pressure inside R2500c (x ≲ 0.27-0.33):** only the integral is available. That integral holds 1.5% (10^12.6), 3.3% (10^13.2), 7.6% (10^13.8), 13% (10^14.4) and 16% (10^15) of Y(<4 R200c).
- **Pressure beyond 5R500c (x ≈ 3.25):** no information. The 3.25-4 R200c region holds 31% (10^12.6), 15% (10^13.2), 8% (10^13.8), 5% (10^14.4) and 3% (10^15) of Y(<4 R200c), and our calibration fits and cuts at 4 R200c.
- **Gas mass beyond 3.25 R200c:** reachable only via the 1000 / 3000 kpc spheres, i.e. 3000 kpc for M200c ≳ 10^13.7 (x = 3.6 at 13.7-13.9).
- **Below about 10^13:** only 1-5 pressure shells have ≥ 50 particles per halo. The 2500c-500c apertures hold a median of 25-100 gas particles at 10^12.5-13.5, so stacks there are particle-limited, as in the particle analysis.
- **Choices it cannot make:**
  - owner, excision or partition splits (only inclusive − exclusive gas mass);
  - phase splits beyond `HotGasMass` (SO only);
  - a min-particles-per-bin choice (only the 100-bound-particle mask);
  - electrons (mass only);
  - pressure in kpc spheres (needs the M·T proxy, +1-4%);
  - an AGN-excluded gas mass.
- **What it does provide:** per-halo values, so the halo-to-halo scatter of aperture integrals is available (e.g. to calibrate a lognormal amplitude).

## 5. Cost of a SOAP-only route

**Measured on maas (single process, nice 19):**
- Full columns (15,329,653 rows):
  - `SO/200_crit/ComptonY`: 4.57 s (15.6 MB stored, 123 MB logical);
  - `SO/200_crit/GasMass`: 4.83 s (28.2 MB stored, 61 MB logical).
- 53 columns at the 4,588 study rows: 101.8 s (1.92 s per column). These rows touch about 25% of the 1,000-row chunks, so reading all rows costs only about 2.5 times more.
- Metadata:
  - full attribute and storage-size walk: 231.6 s;
  - quick listing of a warm file: 8-9 s;
  - the first (cold) listing of the two other files took 7 min 47 s in total.
- This audit read about 290 MB of compressed column data in total: about 243 MB of chunks touched (estimated from the chunk layout), 44 MB in the two full-column reads, and two small z = 2 columns.

**Estimates.** Time per column is taken as the measured 4.7 s, scaled by SOAP file size relative to 0077. Cold-file metadata latency (up to minutes) is excluded.

| Column set | Columns | Stored data per z=0 file | Per z=0 snapshot | 71 fiducial SOAP files (58.9 z=0-equivalents by size) | One variant, 8 SOAP files (7.4 eq.) | All 8 feedback variants |
|---|---|---|---|---|---|---|
| Minimal: IsCentral, M200c, 9 SO × (SORadius, Y_noAGN, GasMass), 8 InclusiveSphere GasMass | 37 | 0.70 GB | about 2.9 min | about 2.8 h, 41 GB | about 21 min, 5.2 GB | about 2.8 h, 41 GB |
| Extended: + Y (with AGN), particle counts, T_mw (SO and kpc) | 80 | 1.33 GB | about 6.3 min | about 6.1 h, 78 GB | about 46 min, 9.8 GB | about 6.2 h, 78 GB |

**Comparison with the particle pass at z = 0:**
- particles: 66 min wall, 6 workers, 5.5 CPU-h, 308 GB logical, 4,588 haloes;
- SOAP: about 3 min, 1 process, 0.7 GB, and it covers all centrals (421,868 above 10^12.5).
- Memory is negligible: one column at a time, reduced to the central rows.
- Over the COSMA hdfstream link (0.3-6 MB/s, quoted from the context, not measured here), 0.7 GB per snapshot would take about 2-40 min.
- **Variants have no other gas data away from z = 0 and 0.15.** The feedback variants have SOAP-HBT at 0017-0077 but no membership files, and full snapshots only at 0074/0077. SOAP is therefore their only gas information at the other redshifts.

## Recommendation

1. **Keep particles for the profile shape.** Use the z = 0 particle stacks (and L2p8 reduced snapshots) for the shape inside 0.3 R200c and for 3.3-4 R200c, where SOAP is blind.
2. **Use SOAP as cheap, exact integral constraints:**
   - forward-model Y(<R_Δ) at the 8 distinct SO radii and M_gas(<r) at SO + kpc radii, for every halo at its own radius;
   - use this to re-fit or validate the amplitude and mid-radius parameters at all 71 fiducial snapshots and in the feedback variants;
   - expected bias is ≤ 1%, except +2-3% for the R2500c point, so down-weight or model it.
3. **Do not difference stacked SO cumulatives at mean radii.** That gives 3-10% shell errors. If shells are needed, use per-halo shells (B), which give ≤ 2.3% for gas density and 2-6% for pressure.
4. **Electrons and AGN gas:**
   - convert gas mass to electrons with the particle-measured composition and ionisation factors above;
   - take Y_noAGN directly from SOAP, remembering that the AGN filter scales with each variant's ΔT_AGN.