# Speed-ups and reuse of saved profiles

**Scope.**
- Inventory of every saved per-halo profile product: local, STRW maas, idark records, and the COSMA data service.
- Data products that avoid full particle reads.
- Code-level speed-ups for new full-snapshot passes, ranked by how much each saves.

All files cited below are in `stacking_choices_20261014/research/speedups/`, and the machine-readable inventory is `speedups_inventory.json`. I wrote and ran every script there. Remote work was read-only:
- **maas:** directory listings, HDF5 headers, filter parameters and the 32,768-cell index (about 0.75 MB).
- **COSMA:** directory and header listings.
- **idark:** did not answer within 40 s (`banner exchange: Connection timed out`), so it was skipped.

## Key findings

1. **The z = 0 master product already makes every stacking choice free.**
   - `measurements_owner.npz` stores linear shell **sums** (not ratios) on a fine grid: 4,588 haloes x 17 owners x 3 phases x 16 quantities x 67 X200 shells to 4 R200c, plus 40 X500 shells to 6 R500c.
   - Building a new profile variant and its 15-bin stack takes **0.06-4.0 s**, compared with **66 min / 5.5 CPU-h** for a new particle pass (`bench_restack.json`).
2. **Nothing comparable exists for z > 0 or for the feedback variants.**
   - Every other saved product is pressure-only, and each is limited in one way:
     - SOAP 7-aperture shells;
     - 89 + 307 particle haloes from idark (pressure-only, 32-33 shells to 4 R200c);
     - L2p8 reduced snapshots, which stop at R100c.
   - maas holds only copies of local files.
3. **Halo sampling does not save I/O.**
   - The 2026-10-12 pass needed 41.7% of the gas rows but decompressed **4,559 of 5,028 HDF5 chunks (90.7%)**, 2.17 times the rows it needed. The cause is 1,048,576-row chunks against cells of about 215k rows.
   - All haloes above 10^14 touch 98.9% of the chunks; even the 1,375 haloes above 10^14.5 touch 59%.
   - So a complete sample of all 421,868 centrals above 10^12.5 costs only about 10% more decompression than our 4,588-halo sample.
4. **Decompression cost is dominated by the HDF5 nbit filter.**
   - The lossy fields use nbit + shuffle + deflate(4) + fletcher32. Decoding them costs 108-121 ns per row on the local machine, against 10-23 ns for the same values stored as plain floats.
   - SmoothedElementMassFractions (9 nbit columns) costs 975 ns per row. A 40-48 MiB chunk cache did not keep its chunk, so cell slices decompressed it again: 2,600 ns per row against 990 ns with an 80 MiB cache.
   - This per-field model reproduces the measured maas cost to within 13% (maas/local = 0.87).

## 1. Saved per-halo profile products

All rows are FLAMINGO L1_m9 HYDRO_FIDUCIAL at snapshot 0077 (z = 0), except where the "Run" column says otherwise. Sizes were measured on disk; axes come from `npz_schema_scan.json` (782 npz files scanned).

**Product family 1. This study's sums (2026-10-12 maas pass):**
- **Sample "4,588 centrals":**
  - 300 random per 0.2-dex bin in 12.5-14.5;
  - 213 annulus-only additions;
  - all 1,375 above 10^14.5;
  - inclusion weights 1/p are in `selected_halos.npz`.

| Product (under `flamingo_pressure_calibration/`) | Size | Per-halo axes | Sample | Radial grid | Reusable for |
|---|---|---|---|---|---|
| `overlap_covariance_.../measurements_owner.npz` | 157.5 MB | owner200 [17 owners, 3 phases (recent_agn, xray_selected, other), 16 quantities, 67]; owner500 [.., 40]; aux [2 AGN, H mass / He mass / fully ionised electrons, S]; hs [S, 384] smoothing-length histograms | 4,588 centrals | X200: 67 shells 0-4 R200c (contains the radii234 36 and Lee22 27 edges). X500: 40 shells 0-6 R500c (contains B24's 30) | All choices: mass bins, truncation (any edge up to 4 R200c / 6 R500c), AGN in/out, all ownership variants, estimators (geometric, Q_V, Q_V_y, Q_M, Q_X, nT_X), normalisation, minimum particles per shell (count + hs), mean/median/weighted stacks, covariance |
| `.../measurements_partition.npz` | 9.8 MB | [owner (central, painted neighbour), bound to other painted, AGN, 5 reduced quantities, 67] | same | X200 | The recommended A2-ii recipe for any bins, truncation or AGN choice |
| `.../measurements_excision.npz` | 71.4 MB | [4 variants, 4 zones, 3 members, 2 AGN, 5, 67] | same | X200 | Excision kappa and placebo tests |
| `.../measurements_legacy.npz` | 21.5 MB | [4 components, 11 SUM_NAMES, 36] | same; rows 0-4374 bit-identical to radii234 | radii234 36 shells | Reproducing radii234 |
| `.../apertures.npz` | 2.5 MB | [4 apertures (R200c, R500c, 5 R500c, sphere), 2 populations, 7 quantities]; inertia; overlapper integrals | same | apertures | Y(<R), N_e(<R) scatter amplitudes; SOAP checks |
| `.../census_population.npz` | 58.4 MB | class, bin and inclusion probability of all 421,868 centrals above 10^12.5 | population | - | Weights; planning complete samples |

**Product family 2. Earlier maas passes (sums, ratios and maps):**

| Product | Size | Per-halo axes | Sample | Radial grid | Reusable for |
|---|---|---|---|---|---|
| `radii234_contamination_.../measurements.npz` | 68.9 MB | sums [4 components, 11, 36]; projected_profiles [2 views, spheres 2/3/4 R200c, 4, 3, 36] | 4,375 (subset of the 4,588) | 36 shells 0-4 R200c | Superseded by owner200. Still the only per-halo projected profiles at 2/3/4 R200c |
| `radii234_.../projection_stacks.npz` | 31.4 MB | per bin: 64x64 maps, 11 views x 36 annuli | 15 bins | - | Projected stacks |
| `scatter_r4_bins02_.../full_profiles.npz` | 70.7 MB | [35 shells]: sums + ratios; 64x64 y and gas maps | 1,375 haloes above 10^14.5 | 35 shells 0-4 R200c | High-mass maps; AGN choice only through y_noagn |
| `scatter_r4_.../orientation_products.npz` | 7.5 MB | 48^3 voxel stacks; 10 rotated views | top 2 bins | - | Orientation scatter |
| `mass14p5_15_...`, `highest_mass_...`, `profile_test_.../full_profiles.npz` | 65.8 / 4.3 / 1.7 MB | [32 shells] | 1,295 / 80 / 40 | 0-3 R200c | Superseded (same haloes now go to 4 R) |

**Product family 3. Derived per-halo ratios, stacks and 3D cubes:**

| Product | Size | Per-halo axes | Sample | Radial grid | Reusable for |
|---|---|---|---|---|---|
| `B_covariance/results/profiles_fit.npz` (+ `_lee22`, `_old`) | 36.7 MB (+3.4, +4.6) | ratio profiles: P/E/G for geo, AGN kept, A2i, A2ii, QV, QM, QX, nTX, bound/excision variants; T_M/T_V/T_X | 4,588 | 35 F/B shells (Lee22: 27; old: 36) | Restacking the stored variants only. These are ratios, so they cannot be rebinned radially exactly |
| `F_observables/results/profiles.npz`, `importance_F.npz` | 11.4 + 4.4 MB | P/E (main, a2i, a2ii, agn_kept) [35]; integrals within 1/2/4 R; per-bin covariance [15, 35, 35]; observable importance | 4,588 | 35 shells | Observable-weighted fits; scatter |
| `G_physical_outer/results/` | 34.3 MB | fits, evaluate [4706 x 68] | - | - | Model comparison only |
| `D_literature/results/stacks.npz` | 3.4 MB | per recipe and bin: mean, median, q16, q84, sd | 15 bins | 36 shells | Literature recipe comparisons |
| `observable_residual_maps_.../voxel_mass_extension_20261013/` | 243 MB | per 0.2-dex bin: 7 fields (pressure, gas, electrons, y, kSZ, aligned kSZ, DM) on 64^3 (centre) + 48^3 (full) float32 | stacks per bin | 3D cube | Stacked 3D or projected profiles at any radial binning. Stacks only, no per-halo scatter |
| `.../measured_observable_maps.npz` | 7.3 MB | [15 bins, 10 maps, 4 cuts (1-4 R), 64, 64] | stacks | - | Projected stacks |
| `.../shell_correlations.npz` | 1.4 MB | covariance and correlation [35, 35] per field / owner / bin / truncation | - | 35 shells | Covariance-weighted fits |

**Product family 4. Other redshifts and runs:**

| Product | Size | Contents | Sample | Radial grid | Run | Reusable for / not for |
|---|---|---|---|---|---|---|
| `cluster_results/profiles/` | 21.3 MB | `soap_NNNN.npz`: pressure in shells between 7 SOAP apertures; `particles_*.npz`: pe, pe_raw, counts | SOAP: ≤ 64 per 0.25 dex per snapshot (99,306 rows; 1,438 at 0077). Particles: 89 haloes at z = 0, 1, 2, 3 | SOAP: 2500c..5xR500c. Particles: 32 shells 0.03-4 R200c | idark 2026-09-21 via COSMA, all 78 outputs | Coarse pressure evolution with z. Not for electrons or ownership |
| `refinement_20260921/cluster_results/profiles/` | 68.9 MB | per halo: pe/pe_raw [33 shells], projected y [3 views], 256^2 y maps, phase diagram, 5,000 sampled particles | 307 haloes, 10^12-10^15.5, z = 0-5 | 33 shells 0-4 R200c | idark via COSMA | Pressure shape with z (small samples). Not for electrons |
| `notebook_cache/stacks/` | 84.7 MB | `L2p8_m9_NNNN_stack_products.npz`: y3_raw / y3_noagn / n3 [31 shells], hsml histograms, 64^2 maps; `_fullsnap_products` | 510-855 centrals above 10^13 per snapshot at z = 0-4; fullsnap 157 / 113 haloes at 10^12.4-13.8 | 31 shells to 1.6 R200c (data end at R100c ≈ 1.35 R200c); fullsnap 37 shells to 3.2 R200c | L2p8_m9 | Inner pressure with z. Pressure only; not beyond R100c |
| `notebook_cache/particles/` | 53.2 MB | per-particle caches | 109 L2p8 z = 0 haloes | - | L2p8_m9 0078 | Any estimator inside R100c for those haloes |

**What a new stacking choice costs from these products.**
- Measured with `analysis.variants.profile` on all 4,588 haloes:
  - geometric P or n_e: 0.06-0.07 s;
  - Q_V: 0.15 s;
  - A2-ii partition: 0.33 s;
  - X500 grid: 2.3 s;
  - first call, including lazy loading: 4.0 s.
- A 15-bin weighted stack takes 3 ms. Peak RAM was 3.2 GiB.

**maas** (`maas_listing.json`, read-only walk of `/home/nagainis/HalfDome*`):
- Every profile product there is a copy of a local file (same names and sizes), all for L1_m9 0077.
- The only things not held locally:
  - the per-chunk-file partial sums of the 2026-10-12 pass (`file_products/file_000..063.npz`, about 290 MB, already merged into `measurements_*.npz`);
  - `gate_file/file_056.npz`;
  - `swift_ids/swift_check_ids.npz` (8.3 MB);
  - Test F/G work caches in `HalfDome_runtime/` (`basis_T.npz`, 198 MB).
- Nothing exists there for other snapshots or variants.

**idark** was not reachable. From local records (READMEs and delivery records), it holds:
- `flamingo_pressure_neighbours_20260921/profiles/all_haloes_NNNN.npz`: SOAP 9-aperture cumulative Y, Y_noAGN and N_gas, plus shell pressures, for **every** central with at least 1,000 gas particles in R200c. That is 1,757,339 halo-snapshot records over 71 snapshots at z = 0-6 (53,234 at z = 0). Only the JSON receipts are local.
- The full particle cutouts behind the refinement run.
- The L2p8 catalogue cache (`flamingo_reduced_cache`).
- The planned `flamingo_l2p8_reduced_20260928` pipeline. No results from it were ever fetched, so whether it completed is unverified.

No FLAMINGO-provided radial-profile product appears in the COSMA listing of `L1_m9/L1_m9`.

## 2. Data products that avoid full particle reads

These come from `cosma_listing*.json` (read-only) and the task facts.

| Product | Where / coverage | Can give | Cannot give | Cost |
|---|---|---|---|---|
| **SOAP-HBT apertures** (details are audited separately) | COSMA: all 78 outputs, fiducial and every hydro variant except DCDM. maas: 0007-0077 (fiducial), 8 outputs (variants). Apertures: 9 SO (2500c, 1000c, 500c, 200c, 100c, 50c, BN98, 200m, 5xR500c), 8 inclusive/exclusive spheres (10 kpc-3 Mpc), 4 projected (10-100 kpc) | Cumulative Y and Y without recent AGN heating, gas mass, N_gas, temperatures for **all** haloes. Coarse shells of up to about 9 + 8 points. Shell Y matched particle sums to a median of 0.18% (idark run) | Fine shells; anything beyond 5 R500c (≈ 3.3 R200c); ownership splits; per-shell resolution masks | About 1 GB of columns per snapshot (census read 155 s on maas) |
| **Reduced snapshots** | COSMA: L1_m8, L2p8 and every L1_m9 **variant** at all 78 outputs. The fiducial L1_m9 has none on COSMA. fgas-8sigma at z = 0: 134.2M gas (2.6%), all gas fields plus GroupNr_bound and GroupNr_all | Full particle physics inside R100c of the flagged haloes: all estimators, AGN flag, membership | 2-4 R200c (which carries 37% of mean DM and 25% of kSZ at ell = 100); low-mass completeness (capped at 200 haloes per 0.05 dex, assumed to be the same rule as L2p8) | About 8.6 GB at 64 B per particle → about 12 min at idark's 12 MB/s |
| **Downsampled snapshots (1% of gas)** | COSMA: every variant, all 78 outputs. maas: fiducial z = 0. Gas fields: Coordinates, Masses, Velocities, ComptonYParameters only | Stacked y profiles of complete samples. Particles within 4 R per 0.2-dex bin: 1.0-1.8e6 for 12.5-14.5, 0.04-0.95e6 above 14.5 (counts x 1%, `entries_estimate.json`). Also stacked large-scale 2-halo y | n_e, densities (SPH volume), AGN flag, smoothing lengths; per-halo profiles (about 275 particles per 10^14 halo within 4 R, about 0.6 within R200c at 10^12.6) | About 2.3 GB per snapshot |
| **Lightcone HEALPix shells and halo lightcone** | COSMA: nside 4096 and 16384; ComptonY and DM per shell (L1_m9: 60 shells to z = 3); `halo_lightcone` lightcone0/1; Yang+26 integrated maps (rotated) | Projected stacked y and DM profiles around lightcone haloes, continuous in z, including the 2-halo term and projection; a direct observable target | 3D shells, ownership, AGN split; resolution 0.86' (nside 4096) | 1.61 GB per map per shell at nside 4096; about 166 GB for y + DM to z = 3 (memory note); idark only |
| **Power spectra** | COSMA: 14 types per output, including power_pressure, power_gas, power_matter-pressure, power_gas-matter | Integral checks of painted profiles through the halo model | Profiles | Negligible |
| **Full snapshots over hdfstream** | COSMA: fiducial at **all 78 outputs**; every hydro variant at **13 outputs** (z = 5, 4, 3, 2, 1.5, 1, 0.75, 0.5, 0.4, 0.3, 0.2, 0.1, 0), with GroupNr_bound / HaloCatalogueIndex linked into PartType0 | Everything a maas pass gives, at more z and variants (maas has 9 useful fiducial outputs and variants only at 0074/0077, without membership) | - | Logical bytes are shipped: 64 B x 5.27e9 = 337 GB per z = 0 snapshot → about 7.8 h at 12 MB/s (idark), weeks locally. Practical only for small samples |

## 3. Why the current pass costs what it costs (measured)

**Measured cost of the 2026-10-12 pass** (`complete.json`, `chunk_amplification.json`):
- 66.4 min wall and 5.54 CPU-h with 6 workers, which gave 5.0 effective cores.
- 2.198e9 candidate rows, which is 9.08 µs of CPU per needed row.
- About 5.29e9 rows of chunks were actually decompressed, which is **3.77 µs per decompressed row**.
- The particle-halo work (about 2.5e8 entries x 2.5 µs plus 2.2e9 candidates x 0.07 µs, from the STEP1 benchmarks) is about 800 s, or 4% of the CPU.

**HDF5 layout** (`maas_filter_parameters.json`):
- Chunks of 1,048,576 rows.
- Lossy fields use nbit (18 or 21 significant bits) + shuffle + deflate level 4 + fletcher32.
- Coordinates and Velocities use scale-offset (D-scale 5 and 1).
- Membership files use 10,000-row chunks and store 0.12 B per particle.

**Per-field stored bytes and local decode cost:**

| Field | Stored B/particle | Local decode, ns/row |
|---|---|---|
| SmoothedElementMassFractions | 14.6 | 975 |
| Coordinates | 9.8 | 159 |
| XrayLuminosities | 4.2 | ~364 (3 x n_e) |
| Velocities | 4.9 | 71 |
| SmoothingLengths | 3.1 | 8 |
| ParticleIDs | 2.9 | 16 |
| Masses | 2.7 | 13 |
| ElectronNumberDensities | 2.1 | 121 |
| ComptonY, Densities, Temperatures, Pressures | 2.1 each | 108-111 |
| StarFormationRates | 0.5 | ~111 |
| LastAGNFeedbackScaleFactors | 0.1 | ~111 |

- The same values stored without nbit decode at 10 ns (f4) and 23 ns (f8).
- Read sets:
  - our pass: 48.5 B stored / 140 B logical per particle;
  - minimal set: 24.2 / 64.
- Decode model for the pass as run: 4,156 ns/row, including about 2.72x re-decompression of SmoothedElementMassFractions and about 1.11x from reopening files every 16 cells.
  - The measured maas cost is 0.87 of this, so the model is consistent.
  - SmoothedElementMassFractions makes up 64% of this model.
  - The pass needs it only for H/He diagnostics, not for the recommended P or n_e estimators.

**How much of the snapshot a sample decompresses** (`sample_vs_chunks.json`, 4 R200c spheres):

| Sample | Rows needed | HDF5 chunks touched |
|---|---|---|
| 32 per bin + all above 14.5 | 24.6% | 64.8% |
| 256 per bin + all above 14.5 | 38.1% | 87.2% |
| Our 4,588 | 41.7% | 90.7% |
| All above 10^14 | 64.1% | 98.9% |
| All above 10^13.5 | 91.6% | 100% |

For complete samples, the particle-halo entries within 4 R (extrapolated with the inclusion weights) are 1.68e9 above 10^12.5 and 1.40e9 above 10^13, against 2.54e8 in our sample.

## 4. Ranked speed-ups

| # | Speed-up | Estimated factor | Basis |
|---|---|---|---|
| 1 | Keep per-halo **shell sums on a fine grid** and derive every choice from them (done for z = 0; repeat for every new snapshot or variant) | 0.06-4 s per variant instead of a 66-min pass (10^3-10^5x) | `bench_restack.json`, `complete.json` |
| 2 | **Complete samples** instead of halo sampling | 92x more haloes for about +10% decompression | 90.7% of chunks already touched; above 10^14 alone 98.9% |
| 3 | **Read only the needed fields**: Coordinates, Masses, Densities, ComptonY, ElectronNumberDensities, Temperatures, LastAGN, SmoothingLengths (+ membership if wanted). Drop SmoothedElementMassFractions (use SOAP or fixed X_H for the n_H diagnostic), XrayLuminosities, Pressures, SFR, ParticleIDs | Decode 3.1x cheaper (5.5x against the pass as run); stored bytes 48.5 → 24.2 B/particle | Per-field benchmark plus maas stored sizes. Q_X and nT_X estimators need the X-ray fields; keep them only if those estimators are wanted |
| 4 | **Chunk cache ≥ 80 MiB per dataset, or whole-chunk sequential streaming** | Measured 1.56x (80 vs 8 MiB, 60 cells). Cell slices with too small a cache cost 5.0x (Coordinates, 8 MiB) and 2.6x (SmoothedElementMassFractions, 40-48 MiB) | `memory_diagnostic.json`, `bench_hdf5_filters.json`. Streaming with explicit buffers also avoids the 3.6-3.8 GiB peak RSS that the 80 MiB cache caused |
| 5 | **Direct chunk decoding** (`read_direct_chunk` + zlib + numpy unshuffle + numba nbit unpack) | 3.4-4.5x on nbit fields, bit-exact; minimal-set decode 758 → 337 ns/row | `bench_direct_chunk.json`. Fletcher32 was not verified; scale-offset Coordinates were not reimplemented |
| 6 | **Spatial hashing or trees** for halo-particle matching | Complete sample: 1.89e11 brute-force comparisons = 1.8-3.7 CPU-h → about 0.1-0.3 CPU-h (**estimate**) | `resource_preflight.json`; 33.6 ns per comparison (`bench_particle_work.json`) |
| 7 | **Reduced accumulators** (bincount over [halo, owner group, phase, 67 shells] x 6 quantities, 38.6 kB per halo in f8) | About 12x per entry (0.21 vs 2.5 µs); 1.68e9 entries: 4,193 s → 358 s | `bench_particle_work.json`, STEP1 benchmark |
| 8 | **Skip membership** when the recipe does not need it (all gas + A2-ii needs only SOAP centres) | Small CPU gain (8 of 64 B logical). The real gain is that the maas variant snapshots, which have no membership files, become usable | `SYNTHESIS.md` recipe; maas probe |
| 9 | **Several snapshots per job** | Saves census (155 s) + prepare (2-5 min) + gate (345 s): ≤ 15% of a 66-min pass, 40-50% once a pass takes about 15 min | `census_summary.json`, `gate_file.json` |
| 10 | **Workers** on the shared 48-thread node | 6 → 5.0 effective cores (83%); 8 → 5.9-6.0 (74-75%) with background load about 25. Use 6-8, not more | Three `complete.json` files |
| 11 | **float32 storage** in scaled units | Halves product size (owner200 is 437 kB per halo in f8); no CPU gain. Accumulate in f64 (n_e V ≈ 1e70 overflows f32) | STEP1 deviations |

**Scenarios per z = 0 snapshot** (`speedups_inventory.json`; decode model calibrated to maas, 5.0 effective cores):

| Scenario | Haloes | CPU-h | Wall |
|---|---|---|---|
| A. Measured pass (14 fields, cell slices, 40 MiB cache) | 4,588 | 5.54 | 66 min |
| B. Same code, complete sample above 10^12.5 | 421,868 | about 10.1 | about 2.0 h |
| C. Streaming + minimal fields + spatial hash + reduced accumulators | 421,868 | about 1.3 | about 15 min |
| D. C + direct nbit decoding | 421,868 | about 0.7 | about 9 min |

Scenario C needs 127 GB of stored bytes read in about 15 min, about 140 MB/s. Our pass implied about 64 MB/s, and the maas NFS rate has never been measured. A gate step should measure it before relying on C or D.

**Implications for many snapshots and variants:**
- The 9 useful maas fiducial outputs at design C come to about 12 CPU-h, plus about 7% for the higher gas counts at z ≥ 1.
- The 17 maas variant passes (8 feedback variants x 0074/0077, plus HYDRO_PLANCK 0077) require the membership-free design.
- COSMA's 13-output variant snapshots are reachable only through about 337 GB per snapshot of hdfstream transfer. The practical COSMA route is reduced snapshots (inner ≤ 1.35 R200c) plus SOAP and downsampled data for the outskirts and amplitudes.