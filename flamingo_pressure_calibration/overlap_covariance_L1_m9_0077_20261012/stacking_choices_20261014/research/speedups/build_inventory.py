"""Assemble speedups_inventory.json: saved per-halo profile products (local / maas / idark records / COSMA), data
products that avoid full particle reads, and a ranked list of code-level speed-ups with their quantitative basis.
Every number is read from a file named in its 'source' field or computed here from such files."""
import json, os, glob
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = '/home/cbllover/HalfDome/flamingo_pressure_calibration'
STUDY = ROOT + '/overlap_covariance_L1_m9_0077_20261012'
J = lambda p: json.load(open(p))
scan = J(os.path.join(HERE, 'npz_schema_scan.json'))
def size(rel):
    p = os.path.join(ROOT, rel)
    if os.path.isdir(p):
        return int(sum(os.path.getsize(os.path.join(dp, f)) for dp, dn, fn in os.walk(p) for f in fn))
    return int(os.path.getsize(p)) if os.path.exists(p) else None
def members(rel):
    v = scan.get(rel)
    return {k: s for k, (s, t) in v['members'].items()} if v else None

G = {'owner200': '[halo 4588, owner 17, phase 3 (recent_agn, xray_selected, other), quantity 16, X200 shell 67] float64 SUMS',
     'owner500': '[halo 4588, owner 17, phase 3, quantity 16, X500 shell 40] float64 SUMS',
     'aux': '[halo, AGN 2, (H mass, He mass, fully ionised electrons) 3, shell] float64 SUMS',
     'hs': '[halo, shell, 384] uint32 histograms of h a / R200c (R500c), edges geomspace(1e-4, 1e2, 385)'}
SAMPLE_OC = ('4,588 L1_m9 centrals at z=0: 300 random per 0.2-dex M200c bin in 12.5-14.5 (seed 20261008) + 213 '
             'annulus-only additions (seed 20261012) + all 1,375 above 10^14.5; inclusion weights 1/p in selected_halos.npz')
ALL_CHOICES = ['mass bins (any edges; re-weight with 1/p)', 'truncation (any grid edge <= 4 R200c / 6 R500c)',
               'recent-AGN gas in/out (phase axis)', 'ownership: all gas, own satellites, 12 overlapper classes, other bound, unbound',
               'estimator: geometric, Q_V, Q_V_y, Q_M, Q_X, nT_X; T_M/T_V/T_X diagnostics', 'normalisation (P200, R200c or R500c grids)',
               'minimum particles per shell (count quantity) and smoothing-length masks (hs histograms)', 'mean/median/weighted stacks, scatter, covariance']
local = [
 {'path': 'overlap_covariance_L1_m9_0077_20261012/measurements_owner.npz', 'kind': 'per-halo shell SUMS (master product)',
  'schema': G, 'sample': SAMPLE_OC, 'radial_grid': 'X200: 67 shells 0-4 R200c (contains the radii234 36-shell and Lee22 27-shell edges); X500: 40 shells 0-6 R500c (contains B24 30 shells)',
  'quantities': 'count, mass, volume (m/rho), y, electrons (n_e V), electron_kT, gas_pressure_volume, mass_pe, volume_pe2, mass_t, volume_t, mass_rho, lx_rosat, lx_pe, lx_t, y2',
  'run': 'L1_m9 HYDRO_FIDUCIAL snapshot 0077 (z=0), maas pass 2026-10-12', 'reusable_for': ALL_CHOICES,
  'not_reusable_for': ['radii beyond 4 R200c / 6 R500c', 'shells finer than the 67/40 grids', 'other redshifts or feedback/cosmology variants', 'projected maps', 'haloes outside the sample (complete samples)']},
 {'path': 'overlap_covariance_L1_m9_0077_20261012/measurements_partition.npz', 'kind': 'per-halo shell SUMS, A2-ii scaled-distance partition',
  'schema': '[halo 4588, owner (central, painted neighbour) 2, bound to other painted (no, yes) 2, AGN 2, reduced quantity (count, mass, volume, y, electrons) 5, X200 shell 67] float64',
  'sample': SAMPLE_OC, 'radial_grid': 'X200 67 shells 0-4 R200c', 'run': 'L1_m9 0077', 'reusable_for': ['recommended painting recipe (all gas + A2-ii) for any mass bins / truncation / AGN choice', 'A2-i vs A2-ii comparison']},
 {'path': 'overlap_covariance_L1_m9_0077_20261012/measurements_excision.npz', 'kind': 'per-halo shell SUMS, overlapper excision zones',
  'schema': '[halo 4588, variant (all, neighbours_only, placebo_iso, placebo_nbr) 4, zone (s<0.5, 0.5-1, 1-2, >=2) 4, member (central-bound, overlapper-bound, rest) 3, AGN 2, reduced quantity 5, X200 shell 67] float64',
  'sample': SAMPLE_OC, 'radial_grid': 'X200 67 shells', 'run': 'L1_m9 0077', 'reusable_for': ['excision kappa 0.5/1/2, placebo controls', 'overlap tests (Test A)']},
 {'path': 'overlap_covariance_L1_m9_0077_20261012/measurements_legacy.npz', 'kind': 'radii234-compatible sums', 'schema': '[halo 4588, component (central, own satellites, other haloes, unbound) 4, SUM_NAMES 11, old shell 36] float64; rows 0-4374 bit-identical to radii234',
  'sample': SAMPLE_OC, 'radial_grid': 'radii234 36 shells 0-4 R200c', 'run': 'L1_m9 0077', 'reusable_for': ['reproduction of the radii234 / Lee22-style results']},
 {'path': 'overlap_covariance_L1_m9_0077_20261012/apertures.npz', 'kind': 'per-halo aperture integrals', 'schema': 'apertures [halo, aperture (R200c, R500c, 5R500c, sphere) 4, population (all gas, central-bound) 2, (count, mass, y, y_noagn, L_X ROSAT, L_X no-AGN, electrons) 7]; inertia [halo, 2, 10]; overlapper [6518, 2, 2, 6]',
  'sample': SAMPLE_OC, 'radial_grid': 'apertures only', 'run': 'L1_m9 0077', 'reusable_for': ['Y(<R) / N_e(<R) amplitudes for scatter models', 'SOAP cross-checks', 'shape (inertia) splits']},
 {'path': 'overlap_covariance_L1_m9_0077_20261012/selected_halos.npz', 'kind': 'sample definition', 'schema': 'per halo SOAP identity, M200c/M500c/radii, centres, inclusion probability/weights, census class, concentrations, (halo, cell) pairs',
  'sample': SAMPLE_OC, 'run': 'L1_m9 0077', 'reusable_for': ['weights for any re-binning', 'cell lists for re-reads']},
 {'path': 'overlap_covariance_L1_m9_0077_20261012/census_population.npz', 'kind': 'population census', 'schema': 'pool of 421,868 centrals >= 10^12.5: bin, class, inclusion probability, max mu; host M200c by HBT id (32.2M)',
  'sample': 'all 421,868 centrals >= 10^12.5 at z=0', 'run': 'L1_m9 0077', 'reusable_for': ['inclusion weights', 'abundance weighting', 'planning complete samples']},
 {'path': 'radii234_contamination_L1_m9_0077_20261008/measurements.npz', 'kind': 'per-halo shell SUMS + derived ratios (older pass)',
  'schema': 'sums [halo 4375, component 4, SUM_NAMES 11 (count, mass, y, y_noagn, electrons, electron_kT, volume, gas_pressure_volume, hydrogen_mass, helium_mass, fully_ionized_electrons), shell 36]; projected_profiles [halo, orientation 2, sphere (2,3,4 R200c) 3, component 4, field 3, annulus 36]; plus per-halo ratio profiles',
  'sample': '4,375 centrals (the first 4,375 rows of the overlap study; same seed)', 'radial_grid': '36 shells 0.025-4 R200c (+ centre bin)', 'run': 'L1_m9 0077, maas 2026-10-08 (71 min, 8 workers)',
  'reusable_for': ['membership-component choices (4 components)', 'AGN in/out (y vs y_noagn only)', 'projected profiles for 2/3/4 R200c spheres'], 'superseded_by': 'measurements_owner.npz (contains all its rows and sums)'},
 {'path': 'radii234_contamination_L1_m9_0077_20261008/projection_stacks.npz', 'kind': 'stacked projected maps', 'schema': 'map_sums [bin 15, 2, 2, 3, 4, 3, 64, 64]; view_profile_sums [bin 15, view 11, 3, 4, 3, 36]',
  'sample': 'radii234 4,375', 'run': 'L1_m9 0077', 'reusable_for': ['projected (2D) stacked profiles, view-to-view scatter']},
 {'path': 'scatter_r4_bins02_L1_m9_0077_20261008/full_profiles.npz', 'kind': 'per-halo shell sums + ratios + 64x64 y and gas maps',
  'schema': '[halo 1375, shell 35] for count, mass, volume, y, y_noagn, electrons, electron_kT, gas_pressure_volume, hydrogen/helium mass, fully_ionized_electrons, hsml median + ratios; y_map/gas_map [1375, 64, 64]',
  'sample': 'all 1,375 centrals >= 10^14.5', 'radial_grid': '35 shells 0.025-4 R200c (radii234 edges without 2.0)', 'run': 'L1_m9 0077, maas 2026-10-08 (36.8 min)',
  'reusable_for': ['high-mass per-halo profiles and maps (all gas only)', 'AGN in/out via y_noagn']},
 {'path': 'scatter_r4_bins02_L1_m9_0077_20261008/orientation_products.npz', 'kind': 'voxel stacks + rotated maps', 'schema': 'pressure/gas voxel stacks [2 bins, 48^3] float32; rotated_y_maps [27 haloes, 10 views, 64, 64]',
  'sample': '15.1-15.5 bins', 'run': 'L1_m9 0077', 'reusable_for': ['3D stacked fields of the two most massive bins', 'orientation scatter']},
 {'path': 'mass14p5_15_centrals_L1_m9_0077_20261007/cluster_results/full_profiles.npz', 'kind': 'per-halo shell sums + ratios + maps', 'schema': '[halo 1295, shell 32] (same keys as scatter_r4) + 64x64 maps',
  'sample': 'all 1,295 centrals 14.5-15', 'radial_grid': '32 shells 0-3 R200c', 'run': 'L1_m9 0077 (2026-10-07)', 'reusable_for': ['3R-truncated reproductions'], 'superseded_by': 'scatter_r4 full_profiles (same haloes to 4R)'},
 {'path': 'highest_mass_centrals_L1_m9_0077_20261007/cluster_results/full_profiles.npz', 'kind': 'per-halo shell sums + ratios + maps', 'schema': '[halo 80, shell 32]', 'sample': 'all 80 centrals >= 10^15',
  'radial_grid': '32 shells 0-3 R200c', 'run': 'L1_m9 0077', 'superseded_by': 'scatter_r4 full_profiles'},
 {'path': 'profile_test_L1_m9_0077_20261007/cluster_results/full_profiles.npz', 'kind': 'pilot per-halo profiles', 'schema': '[halo 40, shell 32]', 'sample': '40 pilot centrals', 'radial_grid': '32 shells 0-3 R200c', 'run': 'L1_m9 0077', 'superseded_by': 'measurements_owner.npz'},
 {'path': 'overlap_covariance_L1_m9_0077_20261012/B_covariance/results/profiles_fit.npz', 'kind': 'per-halo RATIO profiles (derived)',
  'schema': '[halo 4588, shell 35] for P/E/G variants: geo, geo_agnkept, A2i, A2ii, QV, QVy, QM, QX, nTX, bound_all/keep, exc_all/keep; T_M, T_V, T_X; C_P, C_rho; retained fractions',
  'sample': SAMPLE_OC, 'radial_grid': 'F/B 35-shell grid 0.025-4 R200c', 'run': 'L1_m9 0077',
  'reusable_for': ['re-stacking the stored variants with other mass bins / truncation (ratios; cannot be re-binned radially exactly)']},
 {'path': 'overlap_covariance_L1_m9_0077_20261012/B_covariance/results/profiles_lee22.npz', 'kind': 'per-halo ratio profiles', 'schema': '[4588, 27] P/E geo, geo_agnkept', 'radial_grid': 'Lee22 27 shells', 'sample': SAMPLE_OC, 'run': 'L1_m9 0077'},
 {'path': 'overlap_covariance_L1_m9_0077_20261012/F_observables/results/profiles.npz', 'kind': 'per-halo ratio profiles + integrals + per-bin covariance',
  'schema': 'P_/E_{main, a2i, a2ii, agn_kept} [4588, 35]; *_int1R/2R/4R [4588]; SPE_/SEE_* [15, 35, 35]', 'sample': SAMPLE_OC, 'radial_grid': 'F 35-shell grid to 4 R200c', 'run': 'L1_m9 0077',
  'reusable_for': ['observable-driven fits, scatter (Y/N_e within 1/2/4R)']},
 {'path': 'overlap_covariance_L1_m9_0077_20261012/F_observables/results/importance_F.npz', 'kind': 'observable importance maps (Jacobians)', 'schema': 'see file', 'sample': 'per bin', 'run': 'L1_m9 0077 + HalfDome painting kernels', 'reusable_for': ['weighting shells by observable importance (any new stack on the same grid)']},
 {'path': 'overlap_covariance_L1_m9_0077_20261012/G_physical_outer/results', 'kind': 'fits and evaluations of physical outer models', 'schema': 'fits.npz, evaluate.npz [4706 labels, 68 observables], f_base/*', 'sample': SAMPLE_OC, 'run': 'L1_m9 0077', 'reusable_for': ['model comparison, not re-stacking']},
 {'path': 'overlap_covariance_L1_m9_0077_20261012/D_literature/results/stacks.npz', 'kind': 'stacks per literature recipe', 'schema': 'f<recipe>__<bin>__{mean, median, q16, q84, sd, nfinite, mask} [36]', 'sample': 'per bin', 'run': 'L1_m9 0077', 'reusable_for': ['B12/A10/B24/Lee22 recipe comparisons']},
 {'path': 'observable_residual_maps_L1_m9_0077_20261012/voxel_mass_extension_20261013', 'kind': '3D stacked voxel cubes per 0.2-dex bin',
  'schema': 'bin_XX.npz: center [7 fields, 64^3] + full [7 fields, 48^3] float32; fields pressure, gas, electrons, compton_y, ksz, ksz_aligned, frb_dm', 'sample': 'per bin (e.g. 300 haloes in 13.5-13.7)', 'run': 'L1_m9 0077',
  'reusable_for': ['stacked 3D/projected profiles at any radial binning inside the cube (stack only, no per-halo scatter)']},
 {'path': 'observable_residual_maps_L1_m9_0077_20261012/measured_observable_maps.npz', 'kind': 'stacked projected maps', 'schema': 'measured [15 bins, 10 maps, 4 cuts (1-4 R200c), 64, 64]', 'sample': 'per bin', 'run': 'L1_m9 0077', 'reusable_for': ['projected stacked y, tau, kSZ, gas profiles at 4 truncations']},
 {'path': 'observable_residual_maps_L1_m9_0077_20261012/shell_correlations_20261013/shell_correlations.npz', 'kind': 'shell covariance/correlation matrices', 'schema': '<field>__<owner>__b<bin>__t<trunc>__{cov, corr, corr_with_integral, n, sd_integral} [35, 35]', 'sample': 'per bin', 'run': 'L1_m9 0077', 'reusable_for': ['covariance-weighted fits for truncations 1-4R']},
 {'path': 'cluster_results/profiles', 'kind': 'SOAP 7-aperture shell pressure profiles + 89 particle profiles (idark run 2026-09-21, fetched)',
  'schema': 'soap_NNNN.npz: per halo r_inner/r_outer/pe/pe_raw/shell_ngas/y_cumulative [n, 7 apertures]; particles_SSSS_ID.npz: pe/pe_raw/shell_ngas/sum_y [32 shells 0.03-4 R200c] + 7 catalogue apertures',
  'sample': 'soap: <= 64 random haloes per 0.25-dex bin per snapshot, 99,306 rows over 78 snapshot files (0077: 1,438); particles: 89 haloes at 0017/0037/0057/0077 (16/24/24/25)', 'radial_grid': 'SOAP 2500c,1000c,500c,200c,100c,50c,5xR500c shells; particles 32 log shells 0.03-4 R200c',
  'run': 'L1_m9 all 78 outputs (z=0-15), COSMA hdfstream', 'reusable_for': ['pressure-only redshift coverage z=0-6 (coarse)', 'AGN in/out (ComptonY vs WithoutRecentAGNHeating)'], 'not_reusable_for': ['electrons', 'ownership', 'fine shells inside R2500c']},
 {'path': 'refinement_20260921/cluster_results/profiles', 'kind': 'per-halo particle pressure profiles + projections (idark, fetched)',
  'schema': 'projected_SSSS_ID.npz: pe/pe_raw/shell_ngas/valid [33 shells 0-4 R200c], projected_y [3 views, 33], y_maps [3, 256, 256], phase_y [72, 70], 5,000 sampled particles; concentration_SSSS.npz per snapshot',
  'sample': '307 haloes 10^12-10^15.5 at snapshots 0009/0010/0013/0017/0037/0057/0077 (9/23/21/43/57/67/87)', 'radial_grid': '33 shells 0-4 R200c', 'run': 'L1_m9 z=0-5, COSMA hdfstream',
  'reusable_for': ['redshift-evolution checks of the pressure shape (small samples)', 'AGN in/out (pe vs pe_raw)'], 'not_reusable_for': ['electrons', 'ownership splits', 'population stacks']},
 {'path': 'notebook_cache/stacks', 'kind': 'L2p8_m9 reduced-snapshot per-halo shell sums + maps',
  'schema': 'L2p8_m9_NNNN_stack_products.npz: y3_raw/y3_noagn/n3/np [halo, 31 shells 0-1.6 R200c], hsml_hist [halo, 31, 60], map_raw/map_noagn [halo, 64, 64], SOAP aperture columns; *_fullsnap_products.npz: [halo, 37 shells 0-3.2 R200c] from full-snapshot cells',
  'sample': '510-855 centrals >= 10^13 per snapshot (random until 0.5 M particles per 0.2-dex bin, 10-200 haloes) at z = 0, 0.5, 1, 1.5, 2, 3, 4; full-snapshot: 157 (z=0) and 113 (z=1) haloes 10^12.4-13.8',
  'radial_grid': '31 shells to 1.6 R200c (data end at R100c ~1.35 R200c); 37 shells to 3.2 R200c (fullsnap)', 'run': 'L2p8_m9 snapshots 0014-0078, COSMA hdfstream (2026-09-27)',
  'reusable_for': ['pressure (y) inner profiles vs z (AGN in/out)', 'resolution (hsml) masks'], 'not_reusable_for': ['electrons (not stored)', 'outskirts beyond R100c (reduced data)', 'ownership']},
 {'path': 'notebook_cache/particles', 'kind': 'per-halo particle caches (L2p8 reduced)', 'schema': 'xyz, r, mass, density, hsml, T, ne, pe, y, last_agn_a, recently_heated, group_nr_bound per particle', 'sample': '109 files (L2p8 0078 haloes, ~1e4 particles each)', 'run': 'L2p8_m9 0078', 'reusable_for': ['any estimator for those haloes inside R100c']},
]
for e in local:
    e['bytes'] = size(e['path'])
    e['members_scanned'] = members(e['path']) if e['path'].endswith('.npz') else None

maas = J(os.path.join(HERE, 'maas_listing.json'))
maas_unique = [{'path': p.replace('/home/nagainis/', ''), 'bytes': s} for p, s, t in maas['data_files']
               if ('file_products/' in p or 'gate_file/' in p or 'swift_ids' in p or 'basis_T' in p or 'pooled_correlation' in p or 'models_main' in p or 'selection.npz' in p)]
maas_entry = {'listed': '2026-10-14, read-only os.walk of /home/nagainis/HalfDome* (venvs skipped)', 'roots': maas['roots'], 'dir_totals_files_bytes': maas['dir_totals'],
              'finding': 'All profile products on maas are copies of local files (same names/sizes), all for L1_m9 0077. Not held locally: the per-chunk-file partial sums of the 2026-10-12 pass (file_products/file_000..063.npz, ~290 MB, merged into measurements_*.npz), gate_file/file_056.npz, swift_ids/swift_check_ids.npz and a few F/G work files in HalfDome_runtime (198 MB basis_T.npz).',
              'not_held_locally': maas_unique}
cosma = {'listing_files': ['cosma_listing.json', 'cosma_listing2.json', 'cosma_listing3.json', 'cosma_listing4.json']}
c3 = J(os.path.join(HERE, 'cosma_listing3.json')); c4 = J(os.path.join(HERE, 'cosma_listing4.json'))
cosma['L1_m9_fiducial_full_snapshots'] = len(c3['snapshots']['L1_m9'])
cosma['L1_m9_variant_full_snapshots'] = {k: [s.split('_')[1] for s in v] for k, v in c3['snapshots'].items() if k != 'L1_m9' and isinstance(v, list)}
cosma['variant_snapshot_redshifts'] = {k: v['z'] for k, v in c4['variant_snapshots'].items()}
cosma['variant_snapshot_membership_in_PartType0'] = all(v['has_membership_field'] for v in c4['variant_snapshots'].values())
cosma['fiducial_PartType0_fields'] = c3['hdr_0077']['PartType0_fields']
cosma['variant_reduced_0077'] = {'npart': c4['snapshots_reduced']['npart'], 'PartType0': c4['snapshots_reduced']['PartType0']}
cosma['variant_downsampled_0077'] = {'npart': c4['snapshots_downsampled']['npart'], 'PartType0': c4['snapshots_downsampled']['PartType0']}
cosma['L2p8_downsampled_0078'] = {'npart': c3['L2p8_ds_0078_npart'], 'fields': c3['L2p8_ds_0078']}
cosma['soap_groups'] = c4['soap_groups']; cosma['soap_SO_apertures'] = c4['soap_SO_apertures']; cosma['soap_other_apertures'] = c4['soap_other_apertures']
cosma['power_spectra_types'] = c4['L1_m9_power_spectra_types']; cosma['integrated_maps'] = c4['L1_m9_integrated_maps']

idark = {'reachable_2026_10_14': False, 'attempt': "ssh.exe -o BatchMode=yes -o ConnectTimeout=25 idark 'ls ...' -> 'banner exchange: Connection timed out' within 40 s; skipped as instructed",
         'known_roots_from_local_records': [
  {'root': '/lustre/work/kristero10/flamingo_pressure_20260921', 'holds': 'SOAP 7-aperture pressure profiles for 78 L1_m9 snapshots + 89 particle profiles; compact copies local in cluster_results/', 'source': 'README.md, cluster_results/reports/RUN_REPORT.md'},
  {'root': '/lustre/work/kristero10/flamingo_pressure_neighbours_20260921', 'holds': 'profiles/all_haloes_NNNN.npz: SOAP 9-aperture (2500c..5xR500c, BN98, 200m) cumulative Y / Y_noAGN / N_gas and shell pressures for every central with >= 1000 gas particles in R200c: 1,757,339 halo-snapshot records in 71 non-empty snapshots (z=0-6.04; z=0: 53,234 haloes); only JSON receipts are local', 'source': 'neighbour_control_20260921/DELIVERY_RECORD.md, catalogues.py, cluster_results/profiles/all_haloes_0077.json'},
  {'root': '/lustre/work/kristero10/flamingo_pressure_refinement_20260921', 'holds': 'per-halo projected particle profiles (307 local copies), concentrations; full particle positions remain on the cluster', 'source': 'refinement_20260921/README.md'},
  {'root': '/lustre/work/kristero10/flamingo_pressure_resolution_review_20260922', 'holds': 'resolution review (pressure) products; only code/qa local', 'source': 'resolution_review_20260922/README.md'},
  {'root': '/lustre/work/kristero10/flamingo_reduced_cache', 'holds': 'L2p8_m9 reduced-snapshot catalogue caches', 'source': 'memory note flamingo-reduced-snapshots-l2p8'},
  {'root': '/lustre/work/kristero10/flamingo_l2p8_reduced_20260928', 'holds': 'planned L2p8 reduced-snapshot pipeline (10 outputs z=0-3, ~105 GB); no fetched results exist locally, so completion is unverified', 'source': 'l2p8_reduced_calibration/README.md, memory note flamingo-l2p8-cluster-pipeline'}]}

ca = J(os.path.join(HERE, 'chunk_amplification.json'))
sv = J(os.path.join(HERE, 'sample_vs_chunks.json'))
ee = J(os.path.join(HERE, 'entries_estimate.json'))
bf = J(os.path.join(HERE, 'bench_hdf5_filters.json'))['fields']
bd = J(os.path.join(HERE, 'bench_direct_chunk.json'))
bp = J(os.path.join(HERE, 'bench_particle_work.json'))
br = J(os.path.join(HERE, 'bench_restack.json'))
comp = J(STUDY + '/complete.json'); gate = J(STUDY + '/gate_file.json'); memd = J(STUDY + '/memory_diagnostic.json')
r234 = J(ROOT + '/radii234_contamination_L1_m9_0077_20261008/complete.json'); sc4 = J(ROOT + '/scatter_r4_bins02_L1_m9_0077_20261008/complete.json')
pre = J(ROOT + '/mass12p5_15p5_r4_preflight_L1_m9_0077_20261008/resource_preflight.json')

# measured per-row costs
rows_decompressed = ca['pass_20261012']['decompressions_by_cache_model']['cache40MiB_reopen16_chunk36MiB'] * ca['chunk_rows']
cpu = comp['sampled_cpu_seconds']
measured = {'pass_wall_s': comp['wall_seconds'], 'pass_cpu_s': cpu, 'workers': comp['workers'], 'effective_cores': cpu / comp['wall_seconds'],
            'candidate_rows': ca['pass_20261012']['rows_needed'], 'cpu_us_per_candidate_row': 1e6 * cpu / ca['pass_20261012']['rows_needed'],
            'chunk_rows_decompressed_est': rows_decompressed, 'cpu_us_per_decompressed_row': 1e6 * cpu / rows_decompressed,
            'radii234': {'wall_s': r234['wall_seconds'], 'cpu_s': r234['sampled_cpu_seconds'], 'workers': r234['workers'], 'effective_cores': r234['sampled_cpu_seconds'] / r234['wall_seconds']},
            'scatter_r4': {'wall_s': sc4['wall_seconds'], 'cpu_s': sc4['sampled_tree_cpu_seconds'], 'workers': sc4['workers'], 'effective_cores': sc4['sampled_tree_cpu_seconds'] / sc4['wall_seconds']},
            'gate_file_56': gate, 'memory_diagnostic_60_cells': {k: {'wall': v['wall'], 'peak_rss_GiB': v['peak_rss_GiB']} for k, v in memd.items()}}
# per-field decode model (local benchmark, ns per row; FMantissa9-like fields without own benchmark use ComptonY's value)
nb = bf['ComptonY_FMantissa9']['ns_per_row_whole_read']; db = bf['ElectronNumberDensities_DMantissa9']['ns_per_row_whole_read']
cost = {'Coordinates': bf['Coordinates_DScale5']['ns_per_row_whole_read'], 'SmoothingLengths': bf['SmoothingLengths_f4_lossless']['ns_per_row_whole_read'],
        'Masses': bf['Masses_f4_lossless']['ns_per_row_whole_read'], 'Densities': bf['Densities_FMantissa9']['ns_per_row_whole_read'], 'ComptonYParameters': nb,
        'ElectronNumberDensities': db, 'Temperatures': nb, 'LastAGNFeedbackScaleFactors': nb, 'ParticleIDs': bf['ParticleIDs_u8']['ns_per_row_whole_read'],
        'SmoothedElementMassFractions': bf['SEMF_FMantissa9_x9']['ns_per_row_whole_read'], 'Pressures': nb, 'StarFormationRates': nb, 'XrayLuminosities': 3 * db,
        'Velocities': bf['Velocities_DScale1_f4']['ns_per_row_whole_read'], 'membership:GroupNr_bound': bf['ParticleIDs_u8']['ns_per_row_whole_read']}
amp_semf = ca['pass_20261012']['decompressions_by_cache_model']['cache8MiB_reopen0_chunk36MiB'] / ca['pass_20261012']['distinct_chunks_touched']
reopen = ca['pass_20261012']['decompressions_by_cache_model']['cache40MiB_reopen16_chunk24MiB'] / ca['pass_20261012']['distinct_chunks_touched']
sets = ca['read_sets']
def model(names, semf_amp=1.0, other_amp=1.0):
    return sum(cost[n] * (semf_amp if n == 'SmoothedElementMassFractions' else other_amp) for n in names)
pass_names = sets['pass_20261012 (13 fields + membership)']['fields']
min_names = sets['minimal y + n_e, AGN flag, resolution, ownership']['fields']
model_pass_as_run = model(pass_names, amp_semf, reopen)
model_pass_clean = model(pass_names)
model_min = model(min_names)
calib = (cpu - 800.0) / rows_decompressed * 1e9 / model_pass_as_run   # maas ns per local-model ns; 800 s = particle work estimate
nbit_names = [n for n in min_names if n in ('Densities', 'ComptonYParameters', 'ElectronNumberDensities', 'Temperatures', 'LastAGNFeedbackScaleFactors')]
direct_speed = np.mean([v['speedup'] for v in bd.values()])
model_min_direct = model_min - sum(cost[n] for n in nbit_names) * (1 - 1 / direct_speed)
decode = {'per_field_ns_per_row_local': cost, 'semf_amplification_at_40MiB_cache': amp_semf, 'reopen16_amplification_other_fields': reopen,
          'model_pass_as_run_ns_per_row': model_pass_as_run, 'model_pass_no_amplification_ns_per_row': model_pass_clean, 'model_minimal_ns_per_row': model_min,
          'model_minimal_direct_chunk_ns_per_row': model_min_direct, 'maas_over_local_calibration': calib,
          'note': 'local synthetic benchmark (bench_hdf5_filters.json); fields without their own benchmark reuse the FMantissa9 (ComptonY) or DMantissa9 (n_e) cost; calibration = measured maas CPU per decompressed row (minus ~800 s particle work) / model of the pass as run'}
N0 = ca['gas_rows_total']
ent = ee['population_entries_4R_above_12.5']
def scen(ns_row_local, n_rows, entries, us_entry, comparisons=0.0, ns_comp=0.0, extra_s=0.0):
    dec = n_rows * ns_row_local * calib * 1e-9
    part = entries * us_entry * 1e-6 + comparisons * ns_comp * 1e-9 + extra_s
    return {'decode_cpu_h': dec / 3600, 'particle_cpu_h': part / 3600, 'total_cpu_h': (dec + part) / 3600,
            'wall_min_at_measured_effective_cores': (dec + part) / measured['effective_cores'] / 60}
comparisons_complete = pre['alternatives']['all_centrals_12p5_15p5']['candidate_particle_halo_comparisons']
scenarios = {
  'A_measured_pass_20261012 (4,588 haloes, 14 fields, cell slices, 40 MiB cache)': {'total_cpu_h': cpu / 3600, 'wall_min': comp['wall_seconds'] / 60, 'haloes': 4588},
  'B_same_code_complete_sample_>=12.5 (421,868 haloes)': dict(scen(model_pass_as_run, N0, ent, 2.5, comparisons_complete, 70.0), haloes=421868,
      basis='decode of all chunks (sample already touched 90.7%); 2.5 us/entry and 0.07 us/candidate comparison from STEP1_README benchmarks; 1.89e11 comparisons from resource_preflight.json'),
  'C_streaming_minimal_fields_reduced_accumulators_spatial_hash (complete >=12.5)': dict(scen(model_min, N0, ent, bp['accumulate_reduced']['ns_per_entry'] * 1e-3, 0, 0, extra_s=0.2 * 3600), haloes=421868,
      basis='minimal field set decode model, whole-chunk reads (no amplification), 0.21 us/entry reduced bincount accumulators (bench_particle_work.json), 0.2 CPU-h assumed for spatial hashing (ESTIMATE, not benchmarked)'),
  'D_C_plus_direct_chunk_nbit_decoding': dict(scen(model_min_direct, N0, ent, bp['accumulate_reduced']['ns_per_entry'] * 1e-3, 0, 0, extra_s=0.2 * 3600), haloes=421868,
      basis='C with the nbit fields decoded by read_direct_chunk + zlib + numba (mean speed-up %.2f, bench_direct_chunk.json)' % direct_speed)}
ranked = [
 {'rank': 1, 'item': 'Store per-halo shell SUMS on a fine grid (done: measurements_owner.npz) and derive every stacking choice from them',
  'factor': 'a new variant costs %.2f-%.1f s instead of a %.0f-min / %.1f CPU-h particle pass (~10^3-10^5 x)' % (min(c['profile_seconds'] for c in br['cases']), max(c['profile_seconds'] for c in br['cases']), comp['wall_seconds'] / 60, cpu / 3600),
  'basis': 'bench_restack.json (analysis.variants.profile on 4,588 haloes) vs complete.json'},
 {'rank': 2, 'item': 'Complete halo samples instead of sampling, when reading whole files anyway',
  'factor': '%.0fx more haloes (421,868 vs 4,588) for ~+%.0f%% decompression; the 4,588-halo sample already decompressed %.1f%% of all HDF5 chunks' % (421868 / 4588, 100 * (ca['total_hdf5_chunks_per_dataset'] / ca['pass_20261012']['distinct_chunks_touched'] - 1), 100 * ca['pass_20261012']['touched_chunks_fraction_of_all']),
  'basis': 'chunk_amplification.json, sample_vs_chunks.json (all >= 10^14 alone touch 98.9% of chunks; >= 10^14.5 touch 59%)'},
 {'rank': 3, 'item': 'Read only the fields the recommended estimators need (drop SmoothedElementMassFractions, XrayLuminosities, Pressures, StarFormationRates, ParticleIDs)',
  'factor': '%.1fx less decode CPU without the cache problem, %.1fx with it (model %.0f -> %.0f ns/row; stored %.1f -> %.1f B/particle, logical 140 -> 64 B)' % (model_pass_clean / model_min, model_pass_as_run / model_min, model_pass_as_run, model_min, sets['pass_20261012 (13 fields + membership)']['stored_B_per_particle'], sets['minimal y + n_e, AGN flag, resolution, ownership']['stored_B_per_particle']),
  'basis': 'bench_hdf5_filters.json per-field costs; maas_cell_index_and_field_storage.json stored bytes; SmoothedElementMassFractions alone = %.0f%% of the decode model as run' % (100 * cost['SmoothedElementMassFractions'] * amp_semf / model_pass_as_run)},
 {'rank': 4, 'item': 'Chunk cache >= 80 MiB per dataset, or read whole 1,048,576-row chunks sequentially (streaming)',
  'factor': 'measured 1.56x (80 vs 8 MiB, 60 cells); benchmark: cell slices with a cache smaller than the chunk cost 5.0x (Coordinates, 8 MiB) and 2.6x (SmoothedElementMassFractions, 40-48 MiB) more than whole chunks',
  'basis': 'memory_diagnostic.json; bench_hdf5_filters.json; chunk_amplification.json (8 MiB cache -> 2.72 decompressions per touched chunk)'},
 {'rank': 5, 'item': 'Decode nbit/shuffle/deflate chunks directly (h5py read_direct_chunk + zlib + numpy unshuffle + numba bit unpack) instead of the HDF5 filter pipeline',
  'factor': '%.1f-%.1fx on nbit fields (bit-exact on synthetic data); minimal-set decode model %.0f -> %.0f ns/row' % (min(v['speedup'] for v in bd.values()), max(v['speedup'] for v in bd.values()), model_min, model_min_direct),
  'basis': 'bench_direct_chunk.json; fletcher32 not verified in the benchmark (adds a little); scale-offset Coordinates not re-implemented'},
 {'rank': 6, 'item': 'Halo-particle matching by spatial hashing / trees instead of testing every candidate row against every halo of its cells',
  'factor': 'complete sample: %.2e brute-force comparisons x 34-70 ns = %.1f-%.1f CPU-h; hashing to O(10^10) comparisons ~0.1-0.3 CPU-h (ESTIMATE)' % (comparisons_complete, comparisons_complete * 34e-9 / 3600, comparisons_complete * 70e-9 / 3600),
  'basis': 'resource_preflight.json comparisons; bench_particle_work.json (33.6 ns/comparison brute force)'},
 {'rank': 7, 'item': 'Reduced accumulators (np.bincount on [halo, owner group, phase, shell] x few quantities) instead of the full 17-owner/excision/partition blocks',
  'factor': '~12x per entry (0.21 vs 2.5 us); complete sample 1.68e9 entries: %.0f s -> %.0f s' % (ent * 2.5e-6, ent * bp['accumulate_reduced']['ns_per_entry'] * 1e-9),
  'basis': 'bench_particle_work.json; STEP1_README (2.5 us per selected entry); entries_estimate.json'},
 {'rank': 8, 'item': 'Skip membership (GroupNr_bound) when the recipe does not need it (all gas + A2-ii needs only SOAP centres)',
  'factor': 'small CPU gain (8 of 64 B/particle logical; 0.12 B/particle stored); main gain: maas variant snapshots have no membership files, COSMA links it in PartType0',
  'basis': 'maas_cell_index_and_field_storage.json; SYNTHESIS.md recommendation table; task context (maas variants lack membership)'},
 {'rank': 9, 'item': 'Several snapshots per job / shared setup', 'factor': 'saves census (155 s), prepare (2-5 min) and gate (345 s) per snapshot: <= 15% of a 66-min pass, ~40-50% once a pass takes ~15 min',
  'basis': 'census_summary.json, gate_file.json, STEP1_README'},
 {'rank': 10, 'item': 'Worker count on the shared 48-thread maas node', 'factor': '6 workers -> %.1f effective cores (%.0f%%), 8 workers -> %.1f-%.1f (%.0f-%.0f%%); load ~25 before start; more than 8 is not advised' % (
      measured['effective_cores'], 100 * measured['effective_cores'] / 6, measured['radii234']['effective_cores'], measured['scatter_r4']['effective_cores'], 100 * measured['radii234']['effective_cores'] / 8, 100 * measured['scatter_r4']['effective_cores'] / 8),
  'basis': 'complete.json files of the three passes'},
 {'rank': 11, 'item': 'float32 storage (scaled units) of the per-halo sums', 'factor': 'halves product size (owner200 437 kB/halo in float64); no CPU gain (accumulation < 5% of CPU); keep float64 accumulation (n_e V ~1e70 overflows float32)',
  'basis': 'STEP1_README deviations; bench_particle_work.json'},
]
scenarios['C_streaming_minimal_fields_reduced_accumulators_spatial_hash (complete >=12.5)']['stored_GB_read'] = sets['minimal y + n_e, AGN flag, resolution, ownership']['stored_B_per_particle'] * N0 / 1e9
scenarios['C_streaming_minimal_fields_reduced_accumulators_spatial_hash (complete >=12.5)']['implied_storage_MB_per_s'] = scenarios['C_streaming_minimal_fields_reduced_accumulators_spatial_hash (complete >=12.5)']['stored_GB_read'] * 1e3 / (60 * scenarios['C_streaming_minimal_fields_reduced_accumulators_spatial_hash (complete >=12.5)']['wall_min_at_measured_effective_cores'])
measured['pass_stored_GB_est'] = sets['pass_20261012 (13 fields + membership)']['stored_B_per_particle'] * rows_decompressed / 1e9
measured['pass_storage_MB_per_s_est'] = measured['pass_stored_GB_est'] * 1e3 / comp['wall_seconds']
inv = {'generated': '2026-10-14', 'scope': 'saved FLAMINGO per-halo profile products, data products avoiding particle reads, code speed-ups',
       'local_products': local, 'maas': maas_entry, 'idark': idark, 'cosma_service': cosma, 'measured_costs': measured,
       'decode_model': decode, 'chunk_amplification': {k: ca[k] for k in ('pass_20261012', 'radii234', 'total_hdf5_chunks_per_dataset', 'chunk_rows')},
       'read_sets_bytes_per_particle': {k: {kk: vv for kk, vv in v.items() if kk != 'fields'} for k, v in sets.items()},
       'sample_vs_chunks': sv, 'entries_estimate': {k: v for k, v in ee.items() if k != 'bins'}, 'scenarios_per_z0_snapshot': scenarios, 'ranked_speedups': ranked,
       'benchmarks': ['bench_hdf5_filters.json', 'bench_direct_chunk.json', 'bench_particle_work.json', 'bench_restack.json']}
json.dump(inv, open(os.path.join(HERE, 'speedups_inventory.json'), 'w'), indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
print('calibration maas/local', round(calib, 3))
print('decode model ns/row: as run %.0f, clean %.0f, minimal %.0f, minimal direct %.0f' % (model_pass_as_run, model_pass_clean, model_min, model_min_direct))
for k, v in scenarios.items():
    print(k, {kk: (round(vv, 2) if isinstance(vv, float) else vv) for kk, vv in v.items() if kk != 'basis'})
for r in ranked:
    print(r['rank'], r['item'][:70], '|', r['factor'])
print('C io', scenarios['C_streaming_minimal_fields_reduced_accumulators_spatial_hash (complete >=12.5)']['stored_GB_read'], scenarios['C_streaming_minimal_fields_reduced_accumulators_spatial_hash (complete >=12.5)']['implied_storage_MB_per_s'])
print('measured', {k: (round(v, 3) if isinstance(v, float) else v) for k, v in measured.items() if not isinstance(v, dict)})
