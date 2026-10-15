# Step 1 (maas census + particle pass): how to run

Plan: `/root/.claude/plans/there-has-been-multiple-indexed-lantern.md`, Step 1 / 1.1 / 1.2.

Status (2026-10-12):
- The code is ready and tested locally: `tests/test_results.json` (11/11) and the end-to-end synthetic
  rehearsal `tests/rehearsal_results.json` (G1-G7 passed on a mini-FLAMINGO).
- The G4 probe has run on maas (`probe_g4.json`).
- Nothing is deployed. Deploy only after Checkpoint 1.

Local Python: `/home/cbllover/miniconda3/envs/flamingo/bin/python -B`. Remote: `ssh.exe strw-maas`
(ProxyJump `strw`). The login shell is tcsh, so every remote action is a Python program on stdin.

## Run sequence

| # | Command | Remote work | Outputs | Gate |
|---|---|---|---|---|
| 1 | `python -B control.py deploy` | creates the remote directory, writes the whitelist | `executed_sources.json` | hashes equal local |
| 2 | `python -B control.py probe` (optional) | re-runs the read-only probe | `probe_g4.json` | G4 |
| 3 | `python -B control.py census` | `census.py`, detached and polled (5-10 min, about 1 GB SOAP read, <= 6 GB RAM) | `census_pairs.npz`, `census_population.npz`, `census_summary.json`, `census_g3.json` | G3 |
| 4 | edit `study_config.json` `enlargement.apply` (true/false), then `control.py deploy` | - | - | decision |
| 5 | `python -B control.py prepare` | `prepare_study.py` (2-5 min; SOAP rows, cell metadata, shapes) | `selected_halos.npz`, `overlap_tables.npz`, `metadata_plan.json` | G4 re-check, budget <= 488 GB, own satellites = radii234 |
| 6 | `python -B control.py gate` | `extract_study.py --gate-cell` | `one_cell_gate.json` | G5 |
| 7 | `python -B control.py gate_file` | `extract_study.py --gate-file 56` | `gate_file.json` (peak RSS, projected minutes) | G6 |
| 8 | `python -B control.py launch`, then `python -B control.py status` | `extract_study.py` (8 workers) | `file_products/`, merged products, `complete.json` | - |
| 9 | `python -B control.py fetch`, then `python -B verify_study.py --manifest` | - | `verification.json`, `SHA256SUMS`, `ARTIFACTS.json` | G1-G7 |

- Failure handling: `python -B control.py archive_failed --job <census|prepare|gate|gate_file|launch>` moves
  the receipt, log and failure files to `attempts/attempt_NN` on both sides.
- File products are resumable: a relaunch skips completed `file_NNN.npz`. A scientific change needs a new
  study directory.
- `control.py` refuses to:
  - start a job whose prerequisites or earlier gates are missing or failed;
  - start a job while another one is running;
  - redeploy after `launch`.

### Expected resources

- **Read (logical).** radii234 read 2.14e9 rows x 112 B = 240 GB. The new pass reads 140 B per particle (+4 B
  SFR, +24 B XrayLuminosities float64[3]) over the cells of max(4 R200c, 6 R500c) spheres: about 300-330 GB
  for the old sample. The stratified enlargement adds the cells of the added haloes, estimated at tens of GB.
  `prepare_study.py` refuses anything above 488 GB.
- **Wall time.** 55-90 min estimated (radii234: 71 min, decompression-bound with an 8 MiB cache; now 80 MiB).
  - G6 projects it from file 56.
  - Particle work: about 2.5 us per selected particle-halo entry and 0.07 us per candidate (benchmarked
    locally); radii234 had 2.5e8 entries within 4R.
  - Hard stop at 300 min.
- **Memory.**
  - Workers have RLIMIT_AS 7 GiB. HDF5 caches take up to about 1.1 GiB per worker; big selections are
    processed in 1M-particle batches.
  - The tree stops at 64 GiB RSS.
  - Merge in the parent: about 5 GB (4,375 haloes) to 9 GB (8,000).
- **Remote disk.** About 1-2 GB of `file_products/` plus 2-4 GB of compressed merged products.

## Files

| File | Role |
|---|---|
| `pilot_physics.py`, `measure.py`, `models.py`, `projection_model.py`, `swift_reader.py`, `check_fresh_mask.py`, `remote_control.py`, `dependencies/{common,flamingo_io,catalogue_stage}.py` | frozen radii234 copies; SHA256 = radii234 `executed_sources.json` = `final_source_manifest.json` (G1, `input_hashes.json`) |
| `inputs/radii234_{selected_halos.npz,metadata_plan.json,particle_hashes.json}` | read-only radii234 inputs (old sample order, old (halo, cell) pairs, 45 SWIFT-check hashes) |
| `study_config.json` | every run parameter (grids, classes, budgets, limits, seeds) |
| `grids.py` | X200 / X500, sub-grid indices, `to_old`, `to_lee22`, `to_b24` (G2) |
| `census.py` | bucketed periodic KD-tree census (local dry run and maas `main()`) |
| `prepare_study.py` | sample, SOAP columns, owner/excision/placebo/partition tables, cells, budget, G4 re-check |
| `measure_overlap.py` | per-halo accumulators (contracts below); `legacy_sums` = frozen sums path, verbatim |
| `extract_study.py` | worker pass, G5 (`--gate-cell`), G6 (`--gate-file N`), merge, resource monitor |
| `geometry.py` | lens / ball-shell volumes, union-of-spheres excised fraction, offset-template shell average |
| `verify_study.py` | gates G1-G7 |
| `control.py`, `probe_g4.py` | transport, jobs, fetch; G4 probe |
| `tests/run_tests.py`, `tests/rehearsal_e2e.py` | unit tests; end-to-end rehearsal (`OVERLAP_STUDY_DIR` points the scripts at the fixture) |

Environment overrides, for local tests only:
- `OVERLAP_STUDY_DIR`: data/config directory (default: this directory);
- `OVERLAP_BATCH`: particles per accumulate call (default 1,000,000).

## Definitions implemented

- **Overlapper of central c.** Any other subhalo j with mu = M_j / M200c,c >= 0.03 and d < 2 R200c,c + R_j.
  - M_j is M200c for centrals and the HBT bound mass for satellites.
  - R_j is R200c for centrals, or (3 M_j / (4 pi 200 rho_c))^(1/3) for satellites.
  - `reach1` means d < R200c,c + R_j.
  - Own satellite: SOAP/HostHaloIndex = c's SOAP row.
- **Satellites of a neighbouring-central overlapper** take that central's owner class (its group).
- **Main definition.** mu >= 0.1, f = 2. The halo classes are CLEAN, CONT-1R (some overlapper reaches inside 1R)
  and CONT-annulus (reaches 1-2R only).
- **Enlargement.** Below 14.5, seeded stratified additions until each (bin, class) has >= 50 haloes. Inclusion
  probability = n_selected / N of the (bin, class) stratum; weight = 1 / probability. Without enlargement:
  300 / N_bin below 14.5 and 1 above.
- **Phases.**
  - recent AGN: SOAP filter, a_last >= 0.998956 and 8.80e6 <= T <= 1.756e8 K;
  - X-ray-selected: StarFormationRates <= 0, T >= 1e5 K and not AGN. Probe: SFR > 0 is star-forming;
    negative values store -a of the last star formation;
  - other: everything else.
- **Excision spheres.** Overlappers with mu >= 0.1, in two variants: all, and neighbours only (the
  keep-satellites flavour).
  - Placebo, for CLEAN haloes only: the full excision configuration of a random CONT partner in the same bin
    (`placebo_partner`), scaled by R200c,clean / R200c,partner.
  - `placebo_iso` applies a uniform random rotation.
  - `placebo_nbr` rotates the partner's dominant sphere onto the direction of the nearest central with
    mu >= 0.1 (filament proxy), with a random roll.
  - Zones use s = min_k |r - d_k| / R_k with edges 0.5 / 1 / 2. kappa excision removes the zones with
    s < kappa.
- **Partition (A2 ii).** A particle belongs to a painted neighbour (centrals with M200c >= 6.6e12 within
  d < r_ext,c + (r_ext,c / R200c,c) R200c,n) if min_n |r - d_n| / R200c,n < r / R200c,c.
  - `bound_to_other_painted` (A2 i) means bound to a subhalo whose host has M200c >= 6.6e12, excluding the
    central and its own satellites.
- **Extraction sphere.** r <= r_ext = max(4 R200c, 6 R500c), via an exact periodic test after a per-axis
  pre-filter.
  - The legacy subset uses radii234's own (halo, cell) pairs and its exact 4 R200c rule, so the legacy sums
    are bit-identical.

## Array contracts (merged products; n = rows of `selected_halos.npz`, 0-4374 = radii234 in its order)

Grids (`grids.py`):
- X200 has 68 edges / 67 shells in r/R200c (0..4). It contains the radii234 36-shell edges bitwise (indices
  `OLD_INDEX`) and Lee22's 27 shells.
- X500 has 41 edges / 40 shells in r/R500c (0..6). It contains B24's 30 shells.
- Use `grids.to_old` / `to_lee22` / `to_b24` (np.add.reduceat) to rebin linear sums.

| File : array | Shape / dtype | Axes |
|---|---|---|
| `measurements_legacy.npz:sums` | [n, 4, 11, 36] f8 | radii234 components (central, own satellites, other haloes, unbound) x `pilot_physics.SUM_NAMES` x old shells; rows 0-4374 `np.array_equal` radii234 |
| `measurements_owner.npz:owner200` | [n, 17, 3, 16, 67] f8 | owner x phase x quantity x X200 shell |
| `measurements_owner.npz:owner500` | [n, 17, 3, 16, 40] f8 | same on X500 (r/R500c <= 6) |
| `measurements_owner.npz:aux200 / aux500` | [n, 2, 3, 67 / 40] f8 | not-AGN / AGN x (H mass, He mass, fully ionised electrons) |
| `measurements_owner.npz:hs200 / hs500` | [n, 67 / 40, 384] u4 | smoothing-length histograms of h a / R200c (R500c), edges `hs_edges` = geomspace(1e-4, 1e2, 385) |
| `measurements_excision.npz:excision` | [n, 4, 4, 3, 2, 5, 67] f8 | variant (all, neighbours_only, placebo_iso, placebo_nbr) x zone (s<0.5, 0.5-1, 1-2, >=2) x member (central-bound, overlapper-bound, rest) x AGN (no, yes) x reduced quantity x shell |
| `measurements_partition.npz:partition` | [n, 2, 2, 2, 5, 67] f8 | owner (central, painted neighbour) x bound to another painted halo (no, yes) x AGN x reduced quantity x shell |
| `apertures.npz:apertures` | [n, 4, 2, 7] f8 | aperture (R200c, R500c, 5 R500c, whole sphere) x population (all gas, central-bound) x (count, mass, y, y_noagn, L_X ROSAT, L_X ROSAT no-AGN, electrons) |
| `apertures.npz:inertia` | [n, 2, 10] f8 | (all gas, central-bound) inside 1 R200c: sum m, m x, m y, m z, m xx, m yy, m zz, m xy, m xz, m yz (R200c units) |
| `apertures.npz:overlapper` | [n_ov, 2, 2, 6] f8 | per overlapper (rows follow `overlap_tables.npz` `ov_ptr`): (own bound, group bound) x (r <= 2 R200c, whole sphere) x (count, mass, y, y_noagn, electrons, electron_kT) |
| `apertures.npz:id_hash / id_count` | [n, 2, 2] u8 / [n, 2] i8 | (4 R200c rule, whole sphere) x (sum, xor) of splitmix64(ParticleIDs) |
| `apertures.npz:diagnostics` | [n, 10] i8 | `measure_overlap.DIAGNOSTICS` |
| `particle_hashes.json` | 45 entries | SHA256 of sorted legacy-subset IDs of radii234's SWIFT-check haloes |

- The 17 owner codes are (`measure_overlap.OWNERS`):
  - 0 central-bound;
  - 1-12 overlapper-bound: 1 + (mu class x 2 + [annulus only]) x 2 + [neighbour], with mu classes
    [0.03, 0.1), [0.1, 0.3), >= 0.3;
  - 13 other own satellites;
  - 14 other bound, painted host;
  - 15 other bound, not painted;
  - 16 unbound.
  - Legacy components: central = 0; own satellites = odd codes 1-11 + 13; other haloes = even codes 2-12 +
    14 + 15; unbound = 16.
- The 16 quantities (`measure_overlap.QUANTITIES`, units in `UNITS`): count, mass, volume (m/rho), y,
  electrons (n_e V), electron_kT, gas_pressure_volume, mass_pe, volume_pe2, mass_t, volume_t, mass_rho,
  lx_rosat, lx_pe, lx_t, y2.
  - Pe = y / V x m_e c^2 / sigma_T is the particle electron pressure [Pa].
  - The reduced set is count, mass, volume, y, electrons.
- Per-halo inputs (`selected_halos.npz`):
  - identity and SO properties; SOAP Y / Y_noAGN at 200c, 500c and 5xR500c; X-ray at 500c and 5xR500c;
    optional concentration and formation proxies (missing ones are listed in `metadata_plan.json`);
  - role (0 = radii234, 1 = addition), inclusion probability, weight, census class, nearest-massive offset;
  - cell pairs with their legacy flag.
- Tables (`overlap_tables.npz`):
  - CSR owner tables, overlappers (ov_*), excision spheres (exc_ptr indexed by halo x 4 + variant) and
    partition neighbours (pn_*);
  - placebo partner, rotations and scale;
  - `excision_volume_fraction` [n, 4, 3 kappa, 67], the geometric excised fraction per shell, from
    `geometry.union_shell_fraction`.

## Deviations from the plan text, and decisions to confirm at Checkpoint 1

- **Mass floor.** 9.4e10 Msun instead of 9.5e10. 0.03 x 10^12.5 = 9.49e10, so the plan's floor could drop
  mu = 0.03 overlappers of the lightest centrals.
- **float64 blocks.** Excision and partition blocks are float64, not float32: float32 overflows n_e V (~1e70).
  Size: about 5 GB uncompressed for 4,375 haloes.
- **G7 SOAP median criterion.** Implemented as |median fractional difference| < 1e-4. radii234 has
  median -3.5e-6 but median |difference| = 3.8e-4, so "median |diff| < 1e-4" read literally would already fail.
- **X-ray source.** XrayLuminosities is read from the chunk files. The G4 probe found no auxiliary link on
  L1_m9 0077 (unlike L2p8): chunk and top-level values are byte-identical; column 2 is ROSAT.
- **No swiftsimio cutouts.** They are not repeated (about 4.5 GB). G7 compares the 45 legacy-subset ID
  hashes with radii234's instead.
- **Enlargement.** Additions are stratified by (bin, class), not uniform within a bin, with stored weights.
  `enlargement.apply` must be set explicitly.
- **RLIMIT_AS.** 7 GiB may be tight on file 56 (radii234 peak 4.45 GiB with an 8 MiB cache; now +~1 GiB of
  80 MiB caches). G6 checks peak RSS < 0.9 x the limit before launch.
