## COSMA data service (hdfstream)

**What was done.** I listed the whole `/FLAMINGO` tree on `https://dataweb.cosma.dur.ac.uk:8443` (anonymous `hdfstream.open('cosma','/')`, hdfstream 0.0.28) and read HDF5 headers or attributes. That was every reduced snapshot (1,330 files), every full-snapshot virtual file (422), and 3 cell-count tables. I made one timed 24 MB data read and fetched 20 documentation pages. In total about 35 MB was downloaded from COSMA. To get the reduced-snapshot halo selection of one variant, I also ran a read-only query against the same SOAP-HBT catalogues on STRW maas: 194 MB logical, at nice 19. The COSMA copy has the same row counts (15,329,653 rows at z = 0; 17,516,877 for STRONG_AGN at z = 2). Inventory: `research/cosma/cosma_inventory.json`. Figures: `cosma_availability.png` and `downsampled_noise.png`.

### 1. What is on the service

| Box | Runs (COSMA names) | Size |
|---|---|---|
| L1_m8 | L1_m8, L1_m8_DMO | 196 TB |
| L1_m9 | L1_m9 (fiducial) + 18 hydro variants + 10 DMO runs | 769 TB |
| L1_m10 | L1_m10, L1_m10_DMO | 30 TB |
| L2p8_m9 | L2p8_m9, L2p8_m9_DMO | 601 TB |
| L5p6_m10, L11p2_m11 | DMO only | 71 / 61 TB |
| L2p8_m8 | FLAMINGO-10K (DMO) | 901 TB |

**Variant names.** These were checked against the `RunName` in the snapshot headers.

| COSMA | STRW name |
|---|---|
| fgas+2sigma | HYDRO_WEAK_AGN |
| fgas-2sigma | HYDRO_STRONG_AGN |
| fgas-4sigma | HYDRO_STRONGER_AGN |
| fgas-8sigma | HYDRO_STRONGEST_AGN |
| Mstar-1sigma | HYDRO_STRONG_SUPERNOVA |
| Mstar-1sigma_fgas-4sigma | HYDRO_STRONGER_AGN_STRONG_SUPERNOVA |
| Jet | HYDRO_JETS (RunName HYDRO-JETS-AGN-non-fixed-dir) |
| Jet_fgas-4sigma | HYDRO_STRONG_JETS |
| Planck | HYDRO_PLANCK |
| PlanckNu0p24Var / PlanckNu0p24Fix / PlanckNu0p48Fix | HYDRO_PLANCK_LARGE_NU_VARY / LARGE_NU_FIXED / LARGER_NU_FIXED |
| LS8 | LOW-SIGMA8 |
| LS8_fgas-8sigma | HYDRO_LOW_SIGMA8_STRONGEST_AGN |
| L1_m9_extraDM | HYDRO_FIDUCIAL_HiResDM |
| NoCooling | HYDRO_ADIABATIC |
| PlanckDCDM12 / PlanckDCDM24 | HYDRO-PLANCK-DCDM-12 / -24 |

**Particle and catalogue products by run** (output numbers, then redshift):

| Run | Full snapshots (each includes membership + recomputed X-ray) | Reduced | 1% downsampled | SOAP-HBT | Halo lightcone | HEALPix shells |
|---|---|---|---|---|---|---|
| L1_m9 fiducial | **all 78** (0000-0077, z = 15-0), 350-545 GB each, 64 files | not served (the SOAP flag exists: 7,234 haloes at z = 0) | not served | 0-77 | 2 observers | nside 4096 + 16384 |
| 8 feedback variants | 13 outputs: 9, 13, 17, 37, 47, 57, 62, 67, 69, 71, 73, 75, 77 (z = 5, 4, 3, 2, 1.5, 1, 0.75, 0.5, 0.4, 0.3, 0.2, 0.1, 0). 369-548 GB each; Jet runs have 128 files and 406-593 GB | all 78 | all 78 | all 78 | yes | yes |
| Planck, PlanckNu*, LS8, LS8_fgas-8sigma, L1_m9_extraDM (916 GB per snapshot) | same 13 | all 78 | all 78 | all 78 | yes | yes |
| NoCooling | same 13 | all 78 | all 78 | all 78 | no | yes |
| PlanckDCDM12/24 | same 13, **no membership** | no | no | no (announced) | no | yes |
| L2p8_m9 | 18 outputs: 10, 14, 18, 38, 48, 58, 63, 66, 68, 70-78 (z = 5 … 0.05, 0), 8.4-12.2 TB each | all 79 (0000-0078) | all 79 | all 79 | 8 observers | yes |
| L1_m8 | 14 outputs (z = 5, 4, 3, 2, 1.5, 1, 0.75, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0), 3.1-4.4 TB each | all 79 | not served | all 79 | yes | yes |
| L1_m10 | all 78 (48-71 GB each) | no | no | 0-77 | yes | yes |

**Compared with STRW maas.** Maas has HYDRO_FIDUCIAL full snapshots at 10 outputs and the variants only at 0074 and 0077 (plus 0000 for the AGN variants), with no variant membership files. COSMA adds:
- all 78 fiducial outputs;
- 13 full outputs per variant, all with membership files (GroupNr_bound, HaloCatalogueIndex, Rank_bound, FOFGroupIDs in the virtual file);
- reduced and 1% downsampled snapshots at all 78 outputs for every variant.

COSMA does **not** have variant output 0074 (z = 0.15). Its neighbours are 0073 (z = 0.2) and 0075 (z = 0.1).

**Other products.** These are useful for validation, not for 3D profiles.
- HEALPix shells: the per-shell wrapper `lightcone0.shell_N.0.hdf5` links to `swift_lightcone0…` with `ComptonY`, `DM`, `DopplerB`, `SmoothedGasMass` and X-ray maps. Each map is float64: 201,326,592 pixels at nside 4096, 3,221,225,472 pixels (25.8 GB) at nside 16384.
- Halo lightcones (`lightcone_halos_NNNN.hdf5`): HaloCentre, Redshift, SOAPIndex, BoundSubhalo/TotalMass, and an nside-16 pixel index. The SOAPIndex was corrected on 2026-10-02 (service warning).
- Integrated maps: yang26 (rotated), mcdonald26 (ROSAT), broxterman24 (lensing).

### 2. Reduced snapshots

**Selection and contents** (service documentation, confirmed by measurement):
- In each 0.05-dex bin of M200c above 10^13 Msun, 200 random haloes are kept, or all of them if the bin has fewer. All flagged haloes are centrals.
- The file holds **all particles inside R100c of the flagged haloes and nothing else.** For STRONG_AGN, the SOAP `SO/100_crit/NumberOfGasParticles` summed over the flagged haloes is:
  - z = 0: 153,008,062, against 153,007,798 in the header;
  - z = 2: 13,794,712, against 13,792,774.
  The excess of 264 particles at z = 0 is consistent with a few overlapping spheres counted twice.
- R100c/R200c = 1.355 (5-95%: 1.32-1.42) at z = 0 and 1.344 at z = 2. R100c/R500c = 2.08.
- All full-snapshot gas fields are stored. They include Coordinates, Masses, Densities, Pressures, Temperatures, ElectronNumberDensities, ComptonYParameters, LastAGNFeedbackScaleFactors, SmoothingLengths, SmoothedElementMassFractions, the recomputed X-ray luminosities, GroupNr_bound, GroupNr_all, Rank_bound, FOFGroupIDs and SOIDs.
- The file is cell-ordered (32^3 cells for L1_m9, 64^3 for L2p8) with `Cells/OffsetsInFile`, so one pass in cell order is possible.

**Haloes per 0.2-dex bin of log M200c** (L1_m9 from the SOAP flag on maas; L2p8 from the local catalogue caches in `notebook_cache/`):

| Sample | 13.0 | 13.2 | 13.4 | 13.6 | 13.8 | 14.0 | 14.2 | 14.4 | 14.6 | 14.8 | 15.0 | ≥15.2 | Total |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| L1_m9 STRONG_AGN z = 0 | 795 | 800 | 800 | 800 | 799 | 803 | 794 | 806 | 538 | 194 | 66 | 13 | 7,213 |
| L1_m9 fiducial z = 0 (flag only) | 799 | 801 | 798 | 800 | 800 | 807 | 789 | 804 | 557 | 197 | 67 | 13 | 7,234 |
| L1_m9 STRONG_AGN z = 2 | 800 | 796 | 804 | 685 | 228 | 44 | 7 | 0 | – | – | – | – | 3,364 |
| L2p8 z = 0 | 800 | 799 | 801 | 800 | 798 | 802 | 800 | 801 | 798 | 801 | 791 | 356 | 9,151 |
| L2p8 z = 0.5 | 800 | 800 | 801 | 798 | 800 | 800 | 800 | 804 | 796 | 741 | 195 | 38 | 8,173 |
| L2p8 z = 1 | 800 | 800 | 800 | 800 | 800 | 800 | 800 | 801 | 508 | 92 | 7 | 0 | 7,008 |
| L2p8 z = 2 | 800 | 800 | 801 | 799 | 799 | 746 | 183 | 19 | 2 | 0 | – | – | 4,949 |
| L2p8 z = 3 | 800 | 798 | 802 | 624 | 79 | 10 | 0 | – | – | – | – | – | 3,113 |

Median gas particles inside R100c per halo, L1_m9 STRONG_AGN z = 0, by bin from 13.0 upwards: 448, 923, 2,072, 4,319, 8,439, 15,012, 25,875, 42,777. The fiducial values are 549, 1,212, 2,626, … At z = 2 a 10^13 halo holds 1,332.

**Gas particles in the reduced files** (from headers):

| Run | z = 3 | z = 2 | z = 1 | z = 0.5 | z = 0.3 | z = 0 | All outputs |
|---|---|---|---|---|---|---|---|
| L2p8_m9 | 12M | 39M | 134M | 268M | 343M | 485M | 7.86e9 (79 outputs) |
| L1_m9 feedback variants (range) | 3-4M | 12-15M | 41-53M | 78-97M | 99-122M | 134M (fgas-8σ) to 164M (fgas+2σ) | 2.31e9 to 2.89e9 per variant (78 outputs) |
| L1_m8 | 30M | 111M | 397M | 741M | 934M | 1,265M | 2.19e10 |

**What this means for the profile choices**
- **Truncation and fit radius.** The data stop at R100c ≈ 1.35 R200c. The recommended cut = fit = 4 R200c is impossible from reduced data alone. In the study's L1_m9 truth (SYNTHESIS §2.1), the 2-4R shells carry 4% of tSZ at ℓ = 100, 25% of the kSZ one-halo term, and 37% of mean DM.
- **Mass bins.** About 800 haloes per 0.2-dex bin from 10^13 to 10^14.6 (z = 0) or to 10^13.6 (z = 2), which is 2.7 times the current 300 per bin, so the error of the stack mean falls by a factor 1.6. There are **no haloes below 10^13**, while the current calibration starts at 10^12.5.
- **Ownership.** Inside R100c every owner choice is possible (GroupNr_bound and GroupNr_all are stored). Beyond R100c there are no particles, so neighbours and A2-ii (which matters only beyond 2R) cannot be done.
- **Recently-AGN-heated gas.** The cut is possible (LastAGNFeedbackScaleFactors and Temperatures are stored).
- **Minimum particles per bin.** Same particle resolution as the full snapshot; the limit is the low gas content of haloes near 10^13 at z = 0.

### 3. 1% downsampled snapshots

- Gas fields are only Coordinates, Masses (×100), Velocities and ComptonYParameters (float32). There are 52.5M gas particles at z = 0 and 56.8M at z = 2 for fgas-2sigma (headers), plus 32^3 cell metadata.
- Missing: the AGN flag, temperature, electron density, smoothing lengths and membership. So no AGN cut and no bound-membership ownership; geometric A2-ii and "all gas" still work.
- Electrons from gas mass: from the local z = 0 pass beyond 1.35 R200c, n_e equals 0.990-0.996 × the fully-ionised value, and X_H is 0.743-0.746 for haloes above 10^13 (`electron_ratio.json`). A fixed mass-to-electron factor is therefore accurate to about 1% in the outskirts.

**Noise test.** I compared the 1% Bernoulli sampling error of the stacked shell sum, sqrt((1-p)/p Σy²)/Σy, with the halo-to-halo error of the mean. Both come from the local 4,588-halo pass (`measurements_owner.npz`, quantities y, y2, count; all owners and phases).

| log M200c (n) | Shell r/R200c | 1% sampling error: y / electrons | Halo-scatter error: y / electrons |
|---|---|---|---|
| 12.5-12.7 (300) | 0.7-1.0 | 16% / 11% | 5.1% / 3.4% |
| | 2-3 | 8.0% / 4.1% | 11% / 3.5% |
| 13.0-13.2 (312) | 0.7-1.0 | 5.1% / 4.3% | 2.8% / 1.9% |
| | 2-3 | 2.9% / 1.9% | 5.7% / 2.2% |
| 13.5-13.7 (300) | 0.2-0.37 | 4.1% / 3.8% | 2.5% / 2.3% |
| | 2-3 | 1.6% / 1.1% | 5.3% / 1.9% |
| 14.0-14.2 (311) | 0.7-1.0 | 0.9% / 0.8% | 1.5% / 0.9% |
| ≥14.5 (1,375) | 0.05-0.1 | 1.1% / 1.0% | 4.2% / 2.2% |

Above 10^13, beyond about 1.35-2 R200c, the downsampled noise is at or below the halo-scatter error. Inside R200c below 10^14 it is 1.5-10 times larger. So **reduced data (r < R100c, full resolution) stitched to downsampled data (r > R100c, 1%)** gives 4R stacks for every variant at all 78 outputs above 10^13 Msun. The stitched outer part lacks an AGN cut and bound membership (the SYNTHESIS finds the AGN cut changes observables by ≤ 0.26%).

### 4. Download speed and cost

**Timing test.** `L1_m9/fgas-2sigma/snapshots_reduced/flamingo_0077.hdf5`, `PartType0/Coordinates[0:1,000,000]` (24.0 MB float64): **2.77 s, 8.67 MB/s** (2026-10-14 07:27 UTC). The response was chunked msgpack with **no Content-Encoding**: 24,000,056 bytes on the wire for 24,000,000 logical, so transfer volume = logical bytes. Opening the file's metadata took 0.57 s and 49 kB. Header reads cost 1.8 kB and 0.1-0.5 s each.

This is one sample. Earlier notes measured 0.31 MB/s locally (2026-09-27; 0.37 MB/s with 6 streams) and about 12 MB/s on idark (2026-09-25). The service asks for at most 2-4 simultaneous requests.

**Bytes per gas particle**
- 44 B: Coordinates, ComptonY, Temperatures, LastAGN, SmoothingLengths (pressure only).
- 64 B: the above plus ElectronNumberDensities, Masses, Densities and int32 GroupNr_bound (reduced files).
- 68 B: the same with int64 membership (full snapshots).
- 140 B: the field set of the study's z = 0 pass.
- Downsampled: 32 B, or 44 B with velocities.

| Case | Gas rows | B/particle | GB | Hours at 0.31 MB/s | At 8.7 MB/s | At 12 MB/s |
|---|---|---|---|---|---|---|
| (a) L2p8 reduced z = 0 (9,151 haloes) | 485M | 44 / 64 | 21.3 / 31.0 | 19 / 28 | 0.68 / 0.99 | 0.49 / 0.72 |
| (a) L1_m9 STRONG_AGN reduced z = 0 (7,213 haloes) | 153M | 64 | 9.8 | 8.8 | 0.31 | 0.23 |
| (a) same, z = 2 (3,364 haloes) | 14M | 64 | 0.9 | 0.8 | 0.03 | 0.02 |
| (b) L2p8 reduced, all 79 outputs | 7.86e9 | 44 / 64 | 346 / 503 | 310 / 451 | 11.1 / 16.1 | 8.0 / 11.6 |
| (c) one L1_m9 feedback variant, reduced, all 78 outputs | 2.63e9 (mean of 8) | 64 | 168 | 151 | 5.4 | 3.9 |
| (c) 8 feedback variants, reduced, 13 outputs each | 6.98e9 | 64 | 447 | 400 | 14.3 | 10.3 |
| (c) one variant, downsampled, 13 outputs | 715M (derived: 1% of header counts) | 32 | 22.9 | 20.5 | 0.73 | 0.53 |
| (c) **8 variants, reduced + downsampled, 13 outputs** | | | **~630** | **564** | **20.2** | **14.6** |
| (d) full z = 0, cells of 4R spheres, 25/bin + all >10^14.5 (1,625 haloes) | 983M (18.7%) | 68 | 67 | 60 | 2.1 | 1.6 |
| (d) full z = 0, 300/bin + all >10^14.5 (4,375 haloes) | 1.64e9 (31%) | 68 | 111 | 100 | 3.6 | 2.6 |
| (d) full z = 0, the study's cells (10,247) and field set | 2.20e9 (42%) | 140 | 308 | 276 | 9.9 | 7.1 |
| (d) full z = 0, all gas | 5.27e9 | 68 | 359 | 321 | 11.5 | 8.3 |
| (d) 8 variants × 13 full outputs, 300/bin (z = 0 cell fraction assumed) | 1.7e11 | 68 | 11,589 | 10,384 | 371 | 268 |

**Why full snapshots are expensive over the link.** Particles are ordered only by 31.25 Mpc top-level cell: 23k-1.08M gas each, 161k on average at z = 0. So a halo costs whole cells. An isolated 10^13 halo needs one cell (median 2.4e5 rows, about 16 MB at 68 B), and a 10^14.5 halo needs two (7.2e5 rows). Thinning the sample barely helps: 1,625 haloes still need 18.7% of the gas.

Cell geometry comes from `selected_halos.npz` and the COSMA `Cells/Counts`. My geometric 4R union is 7,132 cells; the study pass read 10,247 cells.

### 5. Recommendations (COSMA side)

1. **Feedback variants at z > 0.15** (the only source of them). Do one pass in cell order (`request_slices`) per reduced file for r < R100c, plus one pass per downsampled file for R100c-4R. Do the reduced-file pass at 64 B/particle, which keeps the AGN cut, electrons, volumes and membership. Find flagged haloes from the 61 MB `SOAP/IncludedInReducedSnapshot` column per L1_m9 output, or the cell-index trick for L2p8.
   - Cost: about 79 GB per variant for the 13 full-snapshot outputs (about 630 GB for all 8; 14.6 h at 12 MB/s). Run it on idark, not on the local link (24 days at 0.31 MB/s).
   - Limits: haloes above 10^13 only; the outer part has no AGN cut and no bound membership.
2. **Fiducial redshift evolution.** COSMA has all 78 L1_m9 outputs, but streaming a 4R sample costs about 110 GB per output. Prefer the 10 outputs local to maas, or L2p8 reduced plus downsampled data (79 outputs, about 800 haloes per bin up to 10^15).
3. **Full-snapshot 4R stacks for variants** (8 × 13 outputs, about 11.6 TB) are not realistic over hdfstream. They need on-site access (a COSMA account) or the maas copies (0074 and 0077 only).
4. **Use the SOAP-HBT SO apertures as free checks** at every output and variant. They cover 2500c, 1000c, 500c, 200c, 100c, 50c and 5xR500c, with ComptonY and ComptonYWithoutRecentAGNHeating; SOAP is about 37 GB per L1_m9 output, and a single column is 61 MB. Also use them as outer integral constraints beyond R100c.
5. **Projected stacks** of the nside-16384 ComptonY and DM shells around halo-lightcone positions are possible for every run with a halo lightcone. This route is untested; it gives 2D profiles including line-of-sight structure and is not a replacement for 3D profiles.