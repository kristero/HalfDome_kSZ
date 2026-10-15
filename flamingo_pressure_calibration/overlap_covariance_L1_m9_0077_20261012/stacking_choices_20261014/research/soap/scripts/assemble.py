"""Assemble ../soap_audit.json: inventory, radii, validation, stacks summary, AGN fractions, read cost."""
import json
import numpy as np
OUT = '/home/cbllover/HalfDome/flamingo_pressure_calibration/overlap_covariance_L1_m9_0077_20261012/stacking_choices_20261014/research/soap/'
STUDY = '/home/cbllover/HalfDome/flamingo_pressure_calibration/overlap_covariance_L1_m9_0077_20261012/'
inv = json.load(open(OUT + 'inventory_gas_fid0077.json'))
quick = json.load(open(OUT + 'inventory_raw_quick.json'))
full = json.load(open(OUT + 'inventory_raw_fid0077.json'))
p1 = json.load(open(OUT + 'soap_audit_part1.json'))
p2 = json.load(open(OUT + 'soap_audit_part2.json'))
tim = json.load(open(OUT + 'soap_read_timing.json'))
nfw = json.load(open(OUT + 'nfw_aperture_mapping.json'))

# ---- exact-aperture validation (particle sums at the identical radius, apertures.npz) vs SOAP (selected_halos)
H = np.load(STUDY + 'selected_halos.npz')
APX = np.load(STUDY + 'apertures.npz')['apertures']
lm = np.log10(H['m200c_msun'])
groups = {'12.5-13.5': lm < 13.5, '13.5-14.5': (lm >= 13.5) & (lm < 14.5), '>=14.5': lm >= 14.5, 'all': np.ones(len(lm), bool)}


def st(soap, part, m):
    ok = m & (part > 0) & np.isfinite(soap)
    r = soap[ok] / part[ok]
    return {'n': int(ok.sum()), 'median': float(np.median(r)), 'p16': float(np.percentile(r, 16)), 'p84': float(np.percentile(r, 84)),
            'p0.5': float(np.percentile(r, 0.5)), 'p99.5': float(np.percentile(r, 99.5)), 'frac_abs_gt_0p3pct': float(np.mean(np.abs(r - 1) > 0.003))}


exact = {}
for j, (ap, tag) in enumerate([('R200c', '200'), ('R500c', '500'), ('5R500c', '5x500')]):
    exact[ap] = {g: {'Y': st(H[f'y_{tag}_mpc2'], APX[:, j, 0, 2], m), 'Ynoagn': st(H[f'y_noagn_{tag}_mpc2'], APX[:, j, 0, 3], m),
                     'Mgas': st(H[f'gasmass_{tag}_msun'], APX[:, j, 0, 1], m)} for g, m in groups.items()}

# ---- layout comparison
fds = full['datasets']
layout = {}
for tag in ('strong_agn0077', 'fid0037'):
    ds = quick[tag]['datasets']
    layout[tag] = {'path': quick[tag]['path'], 'bytes': quick[tag]['bytes'], 'header': quick[tag]['header'],
                   'n_datasets': len(ds), 'only_in_fid0077': sorted(set(fds) - set(ds)), f'only_in_{tag}': sorted(set(ds) - set(fds)),
                   'dtype_or_shape_differences': [k for k in set(fds) & set(ds) if fds[k]['dtype'] != ds[k][0] or fds[k]['shape'][1:] != ds[k][1][1:]],
                   'recent_agn_filter': quick[tag]['agn_filter'], 'listing_seconds_warm': quick[tag]['seconds']}
layout['fid0077_recent_agn_filter'] = full['meta']['Parameters/RecentlyHeatedGasFilter']

# ---- cost
sizes = quick['fid_soap_sizes']
z0 = sizes['halo_properties_0077.hdf5']
variant_snaps = ['0017', '0027', '0037', '0047', '0057', '0067', '0075', '0077']
SO = ['2500_crit', '1000_crit', '500_crit', '200_crit', '100_crit', 'BN98', '200_mean', '50_crit', '5xR_500_crit']
KPC = ['10kpc', '30kpc', '50kpc', '100kpc', '300kpc', '500kpc', '1000kpc', '3000kpc']
minimal = ['InputHalos/IsCentral', 'SO/200_crit/TotalMass'] + [f'SO/{a}/{q}' for a in SO for q in ('SORadius', 'ComptonYWithoutRecentAGNHeating', 'GasMass')] + [f'InclusiveSphere/{k}/GasMass' for k in KPC]
extended = minimal + [f'SO/{a}/{q}' for a in SO for q in ('ComptonY', 'NumberOfGasParticles', 'GasTemperature')] + [f'InclusiveSphere/{k}/{q}' for k in KPC for q in ('GasTemperature', 'NumberOfGasParticles')]
t_full = np.mean([v['seconds'] for k, v in tim['timing'].items() if k.startswith('full_')])
cost = {'measured': {'full_column_reads': {k: v for k, v in tim['timing'].items() if k.startswith('full_')},
                     'study_rows_53_columns_s': tim['timing']['study_rows_total_s'],
                     'study_rows_per_column_s_mean': float(np.mean([v['seconds'] for v in tim['columns'].values()])),
                     'metadata_walk_full_inventory_s': full['elapsed'],
                     'metadata_walk_quick_warm_s': {t: quick[t]['seconds'] for t in ('strong_agn0077', 'fid0037')},
                     'note_cold_walk': 'a first (failed at JSON-dump) quick walk of the same two files took 7 min 47 s wall in total (cold metadata); the repeat took 8-9 s per file'},
        'column_sets': {}}
nz0 = sum(sizes.values()) / z0
nvar = sum(sizes[f'halo_properties_{s}.hdf5'] for s in variant_snaps) / z0
for name, cols in (('minimal', minimal), ('extended', extended)):
    stored = sum(fds[c]['storage_bytes'] for c in cols)
    per = len(cols) * t_full
    cost['column_sets'][name] = {'n_columns': len(cols), 'stored_MB_z0': stored / 1e6, 'est_seconds_z0': per,
                                 'fiducial_71_snapshots': {'z0_equivalents_by_file_size': nz0, 'est_hours': per * nz0 / 3600, 'est_GB_read': stored * nz0 / 1e9},
                                 'per_variant_8_snapshots': {'z0_equivalents_by_file_size_(fiducial_sizes_as_proxy)': nvar, 'est_hours': per * nvar / 3600, 'est_GB_read': stored * nvar / 1e9},
                                 'all_8_variants': {'est_hours': 8 * per * nvar / 3600, 'est_GB_read': 8 * stored * nvar / 1e9}}
cost['columns_minimal'] = minimal
cost['assumption'] = 'time per column = mean of the two measured full-column reads at z = 0 (4.7 s), scaled by SOAP file size relative to 0077; I/O-bound, one process, nice 19; excludes cold-file metadata latency'
cost['soap_file_sizes_GB'] = {k: v / 1e9 for k, v in sorted(sizes.items())}

# ---- stack summary
summary = {}
for b, s in p2['stacks']['P'].items():
    u = np.array(s['usable_shell']); r = p2['stacks']['rho_SO_plus_kpc'][b]; ur = np.array(r['usable_shell'])
    summary[b] = {'bin': s['bin'], 'n_halos': s['n_halos'],
                  'P_usable': int(u.sum()), 'P_shells': len(u),
                  'P_max_abs_A': float(np.max(np.abs(np.array(s['ratio_A_to_fine'])[u] - 1))) if u.any() else None,
                  'P_max_abs_B': float(np.max(np.abs(np.array(s['ratio_B_to_fine'])[u] - 1))) if u.any() else None,
                  'P_forward_model_R2500c': s['forward_model_ratio'][0],
                  'P_forward_model_max_dev_other': float(np.max(np.abs(np.array(s['forward_model_ratio'][1:]) - 1))),
                  'rho_usable': int(ur.sum()), 'rho_shells': len(ur),
                  'rho_max_abs_A': float(np.max(np.abs(np.array(r['ratio_A_to_fine'])[ur] - 1))) if ur.any() else None,
                  'rho_max_abs_B': float(np.nanmax(np.abs(np.array(r['ratio_B_to_fine'])[ur] - 1))) if ur.any() else None,
                  'coverage': p2['coverage'][b]}

out = {'description': 'SOAP-HBT audit for 1D gas profiles, FLAMINGO L1_m9 HYDRO_FIDUCIAL 0077 (z = 0); read-only maas access 2026-10-14',
       'inventory': {'path': inv['path'], 'file_bytes': inv['file_bytes'], 'n_rows': inv['n_rows'], 'n_datasets': inv['n_datasets'],
                     'top_groups': inv['top_groups'], 'apertures': inv['apertures'],
                     'profile_like_datasets': inv['profile_like_datasets'], 'chunk_rows': 1000,
                     'gas_datasets_by_family': {f: {n: {k: e[k] for k in ('dtype', 'shape1', 'units', 'a_exp', 'cgs', 'description', 'lossy')} | {'n_apertures': len(e['groups']), 'apertures': e['groups'], 'stored_MB_mean': float(np.mean(e['stored_bytes'])) / 1e6}
                                                    for n, e in items.items()} for f, items in inv['gas_datasets_by_family'].items()},
                     'group_masks': {k: v['attrs'] for k, v in full['groups'].items() if k.split('/')[0] in ('SO', 'ExclusiveSphere', 'InclusiveSphere', 'ProjectedAperture') and v['attrs']},
                     'aperture_units_check_z2': tim['z2_aperture_check']},
       'layout_other_files': layout,
       'aperture_x_by_bin_measured_z0': p1['aperture_x_by_bin'],
       'aperture_x_nfw_model': nfw,
       'validation_exact_apertures': exact,
       'validation_interpolated_all_apertures': p1['validation'],
       'particle_interpolation_check': p1['particle_interpolation_check'],
       'soap_recent_agn_fraction_of_Y': p1['soap_recent_agn_fraction'],
       'stack_summary': summary,
       'stacks_full': p2['stacks'],
       'cost': cost}
json.dump(out, open(OUT + 'soap_audit.json', 'w'), indent=1, default=float)
print(json.dumps(exact['R200c']['all']), json.dumps(exact['R500c']['all']), json.dumps(exact['5R500c']['all']))
for g in ('12.5-13.5', '13.5-14.5', '>=14.5'):
    print(g, {ap: {q: (round(exact[ap][g][q]['median'], 6), round(exact[ap][g][q]['p16'], 5), round(exact[ap][g][q]['p84'], 5), round(exact[ap][g][q]['p0.5'], 4), round(exact[ap][g][q]['p99.5'], 4)) for q in ('Y', 'Mgas')} for ap in exact})
print(json.dumps(cost['column_sets'], indent=1))
print({t: (layout[t]['only_in_fid0077'], layout[t]['dtype_or_shape_differences']) for t in ('strong_agn0077', 'fid0037')})
