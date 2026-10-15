"""Summarise the unit-harness outputs (unit_harness.cpp) for the group_particles boundary check.

Reads  <OUT>/unit/<label>_{targeted,population}_*.{json,csv}
Writes <OUT>/unit/analysis_summary.json and prints the tables.

Checks
 1. per group class and code path: pairs dropped / spurious / never searched / double counted / wrong radial bin
 2. mechanism: every wrong periodic_to_add is +1 where -1 is needed, and the dropped pairs are exactly the
    true pairs whose minimum image lies across a low face (x=0, y=0 or z=0)
 3. single-low-face groups: lost fraction of a shell at radius r vs the spherical-cap fraction (1 - d/r)/2
 4. population of uniformly placed groups: stacked lost fraction vs x = r/R200c vs the law 3 r / (4 Bsize)
"""
import json, sys, os
import numpy as np

OUT = sys.argv[1] if len(sys.argv) > 1 else '/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_boundary_check_20261007'
U = os.path.join(OUT, 'unit')
B = 205000.0
res = {}

def load(label, mode):
    with open(os.path.join(U, f'{label}_{mode}_summary.json')) as f:
        return json.load(f)

VARS = ['hint_sorted_c3fe9b8_default', 'plain_sorted_HEAD_default', 'early_return_hint', 'naive_all_particles']
BAD = ['dropped_cell_not_searched_gross', 'dropped_by_distance_gross', 'spurious_gross', 'double_counted', 'wrong_bin_gross']
PREC = ['dropped_cell_not_searched_at_R', 'dropped_by_distance_at_R', 'spurious_at_R', 'wrong_bin_at_edge']

for label in ['c3fe9b8', 'c3fe9b8_perfix', '6c6d486']:
    d = load(label, 'targeted')
    out = {'n_groups': d['n_groups'], 'n_particles': d['n_particles'],
           'particle_coords_equal_B': d['particle_coords_equal_B'], 'particle_coords_equal_0': d['particle_coords_equal_0'],
           'classes': {}}
    print(f'\n=== targeted, library tree {label}: {d["n_groups"]} groups, {d["n_particles"]} particles '
          f'({d["particle_coords_equal_B"]} coordinates == B, {d["particle_coords_equal_0"]} == 0)')
    print(f'{"class":26s} {"true pairs":>10s} | ' + ' | '.join(f'{v[:18]:>18s}' for v in VARS))
    for cls, c in d['classes'].items():
        row = {}
        cells = []
        for v in VARS:
            bad = {k: c[v][k] for k in BAD if c[v][k]}
            prec = {k: c[v][k] for k in PREC if c[v][k]}
            row[v] = {'gross': bad, 'float_precision': prec, 'lost_fraction': c[v]['dropped_by_distance_gross'] / max(c['n_true_pairs'], 1)}
            cells.append(f'{sum(bad.values()):8d}/{sum(prec.values()):<3d} {100*row[v]["lost_fraction"]:5.1f}%')
        m = c['hint_mechanism']
        row['mechanism'] = m
        row['n_true_pairs'] = c['n_true_pairs']
        out['classes'][cls] = row
        print(f'{cls:26s} {c["n_true_pairs"]:10d} | ' + ' | '.join(f'{s:>18s}' for s in cells))
    m = d['classes']['ALL']['hint_mechanism']
    print('  hint values attached to searched cells (x axis: -1, 0, +1):', m['range_hint_counts_axis_x_minus1_0_plus1'])
    print('  visits with a wrong hint:', m['visits_with_wrong_hint'], ' of which all "+1 where -1 needed":',
          m['wrong_hint_all_plus1_where_minus1_needed'], ' other patterns:', m['wrong_hint_other_pattern'])
    print('  true pairs reached across a low face:', m['true_pairs_reached_across_low_face'],
          ' dropped by the hint path:', m['of_which_hint_dropped'],
          ' hint drops not across a low face:', m['hint_dropped_not_across_low_face'],
          ' accepted despite wrong hint:', m['hint_accepted_despite_wrong_hint'])
    for v in VARS:
        print(f'  {v}: float-precision entries', {k: d['classes']['ALL'][v][k] for k in PREC})
    res[f'targeted_{label}'] = out
    print('  (cells: gross/float-precision discrepancies, lost fraction of true pairs)')

# ---- 3. spherical-cap test on single-low-face groups (c3fe9b8 tree), own isotropic cloud particles only
meta = load('c3fe9b8', 'targeted')
base = meta['n_background'] + meta['n_special_lattice']
ncl = meta['n_cloud']
p = np.genfromtxt(os.path.join(U, 'c3fe9b8_targeted_lowface_pairs.csv'), delimiter=',', names=True)
pid = p['pid'].astype(np.int64)
own = (pid >= base) & ((pid - base) // ncl == p['gidx'].astype(np.int64))
snapped = own & (((pid - base) % ncl) % 50 == 0)          # every 50th cloud particle was put on a face plane
iso = own & ~snapped
u = p['r'] / np.maximum(p['d_face'], 1e-6)
edges = np.array([0, 0.5, 1.0, 1.25, 1.5, 2, 3, 5, 10, 30, 1e9])
cap = []
print('\n=== single-low-face groups (c3fe9b8), own isotropic cloud: lost fraction of true pairs vs r/d (d = distance to the face)')
print(f'{"r/d bin":>14s} {"pairs":>8s} {"lost (hint)":>11s} {"lost (plain)":>12s} {"cap (1-d/r)/2 mean":>18s} {"binomial sigma":>14s}')
for a, b in zip(edges[:-1], edges[1:]):
    s_ = iso & (u >= a) & (u < b)
    n = int(s_.sum())
    if n == 0:
        continue
    lost_h = 1 - p['acc_hint'][s_].mean()
    lost_p = 1 - p['acc_plain'][s_].mean()
    pred = np.mean(np.clip((1 - 1 / u[s_]) / 2, 0, None))
    sig = np.sqrt(max(pred * (1 - pred), 1e-12) / n)
    cap.append({'r_over_d_lo': a, 'r_over_d_hi': b if b < 1e8 else None, 'n_pairs': n,
                'lost_fraction_hint': lost_h, 'lost_fraction_plain': lost_p, 'cap_fraction_prediction': pred, 'binomial_sigma': sig})
    print(f'{a:6.2f}-{b:<7.4g} {n:8d} {lost_h:11.4f} {lost_p:12.4f} {pred:18.4f} {sig:14.4f}')
across = p['across_low_face'] > 0
res['lowface_cap_test'] = {'bins': cap, 'n_pairs_all': int(len(p)), 'n_pairs_own_isotropic': int(iso.sum()),
                           'pairs_across_face': int(across.sum()),
                           'pairs_across_face_dropped_by_hint': int(((p['acc_hint'] == 0) & across).sum()),
                           'pairs_not_across_face_dropped_by_hint': int(((p['acc_hint'] == 0) & ~across).sum()),
                           'pairs_dropped_by_plain': int((p['acc_plain'] == 0).sum())}
print('  all pairs of these groups: across the face', res['lowface_cap_test']['pairs_across_face'], ', dropped by hint path',
      res['lowface_cap_test']['pairs_across_face_dropped_by_hint'], ', hint drops not across the face',
      res['lowface_cap_test']['pairs_not_across_face_dropped_by_hint'], ', dropped by plain path', res['lowface_cap_test']['pairs_dropped_by_plain'])

# ---- 4. population stack
for label in ['c3fe9b8', 'c3fe9b8_perfix']:
    s = np.genfromtxt(os.path.join(U, f'{label}_population_stack_vs_x.csv'), delimiter=',', names=True)
    g = np.genfromtxt(os.path.join(U, f'{label}_population_groups.csv'), delimiter=',', names=True, dtype=None, encoding=None)
    d = load(label, 'population')
    nt, nh, npl = s['n_true'], s['n_acc_hint'], s['n_acc_plain']
    law = 3.0 * s['sum_r_true_ckpc_h'] / (4.0 * B) / np.maximum(nt, 1)       # expected lost fraction, 3 r / (4 B)
    xs = 0.5 * (s['x_lo'] + s['x_hi'])
    rows = []
    print(f'\n=== population ({label}): {len(g)} uniformly placed groups, 1e13-10^14.2 Msun/h, z in {{0,...,2}}, '
          f'mean R200c = {g["R200c_ckpc_h"].mean():.1f} ckpc/h; groups with a low face within 2.5 R200c: '
          f'{int((g["n_low_faces_within_R"] > 0).sum())} ({100 * np.mean(g["n_low_faces_within_R"] > 0):.2f}%)')
    print(f'{"x range":>12s} {"true pairs":>10s} {"lost hint":>10s} {"lost plain":>10s} {"3r/(4B)":>9s} {"across low face":>15s}')
    for lo, hi in [(0.0, 0.04), (0.04, 0.1), (0.1, 0.3), (0.3, 0.6), (0.6, 1.0), (1.0, 1.34), (1.34, 2.0), (2.0, 2.5)]:
        sel = (s['x_lo'] >= lo - 1e-9) & (s['x_hi'] <= hi + 1e-9)
        T = nt[sel].sum()
        if T == 0:
            continue
        r = {'x_lo': lo, 'x_hi': hi, 'n_true': int(T), 'lost_hint': 1 - nh[sel].sum() / T, 'lost_plain': 1 - npl[sel].sum() / T,
             'law_3r_over_4B': (3.0 * s['sum_r_true_ckpc_h'][sel].sum() / (4.0 * B)) / T,
             'across_low_face_fraction': s['n_true_across_low_face'][sel].sum() / T}
        rows.append(r)
        print(f'{lo:5.2f}-{hi:<5.2f} {int(T):10d} {100*r["lost_hint"]:9.3f}% {100*r["lost_plain"]:9.3f}% {100*r["law_3r_over_4B"]:8.3f}% '
              f'{100*r["across_low_face_fraction"]:14.3f}%')
    res[f'population_{label}'] = {'n_groups': len(g), 'mean_R200c': float(g['R200c_ckpc_h'].mean()),
                                  'frac_groups_low_face_within_R': float(np.mean(g['n_low_faces_within_R'] > 0)),
                                  'bins': rows,
                                  'gross_discrepancies': {v: {k: d['classes']['ALL'][v][k] for k in BAD} for v in VARS}}

with open(os.path.join(U, 'analysis_summary.json'), 'w') as f:
    json.dump(res, f, indent=1, default=float)
print('\nwritten', os.path.join(U, 'analysis_summary.json'))
