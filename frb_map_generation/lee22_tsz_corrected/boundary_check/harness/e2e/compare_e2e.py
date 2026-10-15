"""Compare y_prof.cpp outputs (built against different group_particles trees) with a brute-force
minimum-image calculation on the synthetic snapshot written by make_synthetic_tng.py.

For every group the brute force reproduces y_prof.cpp's float32 arithmetic (double -> float
coordinates, |dx| periodic distance, 0.5 log(Rsq), 128 log bins 0.03-2.5 R200c) and gives
  expected_correct : all particles within 2.5 R200c (minimum image)
  expected_bug     : the same minus the pairs whose minimum image lies across a low face
                     (x_p - x_g > B/2 on some axis), the prediction of the PER() sign error
Writes <out>/e2e_summary.json and <out>/e2e_groups.csv.
"""
import argparse, json, os
import numpy as np
from scipy.spatial import cKDTree

ap = argparse.ArgumentParser()
ap.add_argument('--data', required=True)
ap.add_argument('--snap', default='099')
ap.add_argument('--run', action='append', required=True, help='name=output_dir')
ap.add_argument('--out', required=True)
a = ap.parse_args()

T = np.load(os.path.join(a.data, 'truth.npz'))
B = float(T['B']); f32 = np.float32; B32 = f32(B)
gpos, gR, gcls, classes = T['gpos'], T['gR'], T['gcls'], [str(c) for c in T['classes']]
X32 = T['X'].astype(f32)                         # convert_coords: (coord_t)(double)
mass, u, xe = T['mass'].astype(np.float64), T['u'].astype(np.float64), T['xe'].astype(np.float64)
XH, gamma = 0.76, 5.0 / 3.0
Yp = 4.0 * xe * XH / (1.0 + 3.0 * XH + 4.0 * XH * xe) * (gamma - 1.0) * mass * u
N = 128
Ng = len(gpos)

Xw = X32.astype(np.float64); Xw[Xw >= B] -= B     # tree needs [0, B); 205000.0f is the face x = 0
tree = cKDTree(Xw, boxsize=B)

def shell_vol(logRmin, dlogR):
    lmin, dl = float(logRmin), float(dlogR)
    v = np.empty(N)
    v[0] = 4 * np.pi / 3 * np.exp(3 * lmin)
    i = np.arange(1, N)
    v[1:] = 4 * np.pi / 3 * (np.exp(3 * (lmin + dl * i)) - np.exp(3 * (lmin + dl * (i - 1))))
    return v

exp_n = np.zeros((Ng, N), np.int64); exp_n_bug = np.zeros((Ng, N), np.int64)
exp_P = np.zeros((Ng, N)); exp_P_bug = np.zeros((Ng, N))
n_true = np.zeros(Ng, np.int64); n_across = np.zeros(Ng, np.int64)
for g in range(Ng):
    R200 = gR[g]
    R = f32(2.5) * R200; Rsq = R * R
    c = gpos[g].astype(np.float64).copy(); c[c >= B] -= B
    idx = np.array(tree.query_ball_point(c, r=float(R) + 0.1), dtype=np.int64)
    if idx.size == 0:
        continue
    d = np.abs(X32[idx] - gpos[g][None, :])                       # float32, as abs_periodic_dist
    d = np.where(d > f32(0.5) * B32, B32 - d, d).astype(f32)
    Rsq_f = (d[:, 0] * d[:, 0] + d[:, 1] * d[:, 1]) + d[:, 2] * d[:, 2]
    keep = ~(Rsq_f > Rsq)
    idx, Rsq_f = idx[keep], Rsq_f[keep]
    across = ((X32[idx].astype(np.float64) - gpos[g].astype(np.float64)[None, :]) > 0.5 * B).any(axis=1)
    logRmin = np.log(f32(0.03) * R200); logRmax = np.log(f32(2.50) * R200)
    dlogR = (logRmax - logRmin) / f32(N - 1)
    with np.errstate(divide='ignore'):
        logR = f32(0.5) * np.log(Rsq_f)
    ok = ~(logR > logRmax)
    b = np.where(logR < logRmin, 0, 1 + np.floor(((logR - logRmin) / dlogR).astype(f32)).astype(np.int64))
    ok &= b < N
    b, idx_ok, across_ok = b[ok], idx[ok], across[ok]
    n_true[g] = ok.sum(); n_across[g] = across_ok.sum()
    sv = shell_vol(logRmin, dlogR)
    exp_n[g] = np.bincount(b, minlength=N)
    exp_n_bug[g] = np.bincount(b[~across_ok], minlength=N)
    exp_P[g] = np.bincount(b, weights=Yp[idx_ok], minlength=N) / sv
    exp_P_bug[g] = np.bincount(b[~across_ok], weights=Yp[idx_ok][~across_ok], minlength=N) / sv

summary = {'n_groups': int(Ng), 'n_particles': int(len(X32)),
           'n_particle_coords_equal_B_after_float_conversion': int((X32 == B32).sum()),
           'true_pairs_in_bins': int(n_true.sum()), 'true_pairs_across_low_face': int(n_across.sum()), 'runs': {}}
rows = {}
for spec in a.run:
    name, d = spec.split('=', 1)
    nump = np.fromfile(os.path.join(d, f'grp_num_part_prof_{a.snap}.bin'), dtype=np.uint64).reshape(-1, N).astype(np.int64)
    pres = np.fromfile(os.path.join(d, f'grp_pressure_prof_{a.snap}.bin'), dtype=np.float64).reshape(-1, N)
    Rout = np.fromfile(os.path.join(d, f'grp_R200c_{a.snap}.bin'), dtype=np.float64)
    assert len(Rout) == Ng and np.allclose(Rout, gR), 'group order/selection differs'
    dn = nump - exp_n
    dn_bug = nump - exp_n_bug
    tot = nump.sum(1)
    run = {'pairs_counted': int(tot.sum()),
           'pairs_missing_vs_correct': int(-(dn.sum())),
           'groups_with_any_count_difference_vs_correct': int((np.abs(dn).sum(1) > 0).sum()),
           'abs_bin_count_differences_vs_correct': int(np.abs(dn).sum()),
           'abs_bin_count_differences_vs_bug_prediction': int(np.abs(dn_bug).sum()),
           'max_rel_pressure_diff_vs_correct_in_groups_without_count_diff': None,
           'by_class': {}}
    clean = np.abs(dn).sum(1) == 0
    with np.errstate(invalid='ignore', divide='ignore'):
        rel = np.where(exp_P > 0, np.abs(pres / exp_P - 1), np.where(pres == 0, 0, np.inf))
    run['max_rel_pressure_diff_vs_correct_in_groups_without_count_diff'] = float(np.nanmax(rel[clean])) if clean.any() else None
    for ci, cname in enumerate(classes):
        s = gcls == ci
        lost = 1 - tot[s].sum() / max(exp_n[s].sum(), 1)
        frac_g = np.where(exp_n[s].sum(1) > 0, 1 - tot[s] / np.maximum(exp_n[s].sum(1), 1), 0)
        run['by_class'][cname] = {'n_groups': int(s.sum()), 'pairs_expected': int(exp_n[s].sum()), 'pairs_counted': int(tot[s].sum()),
                                  'lost_fraction': float(lost), 'max_group_lost_fraction': float(frac_g.max()),
                                  'pairs_across_low_face': int(n_across[s].sum()),
                                  'abs_bin_diff_vs_correct': int(np.abs(dn[s]).sum()),
                                  'abs_bin_diff_vs_bug_prediction': int(np.abs(dn_bug[s]).sum())}
    summary['runs'][name] = run
    rows[name] = tot
    print(f'\n== {name}: counted {run["pairs_counted"]} of {int(exp_n.sum())} pairs; missing {run["pairs_missing_vs_correct"]}; '
          f'|bin diff| vs correct {run["abs_bin_count_differences_vs_correct"]}, vs bug prediction {run["abs_bin_count_differences_vs_bug_prediction"]}; '
          f'max rel. pressure diff in count-clean groups {run["max_rel_pressure_diff_vs_correct_in_groups_without_count_diff"]}')
    for cname, v in run['by_class'].items():
        print(f'   {cname:20s} groups {v["n_groups"]:3d} expected {v["pairs_expected"]:8d} counted {v["pairs_counted"]:8d} '
              f'lost {100*v["lost_fraction"]:6.2f}% (max per group {100*v["max_group_lost_fraction"]:5.1f}%), '
              f'across-low-face {v["pairs_across_low_face"]:7d}, |diff| vs correct {v["abs_bin_diff_vs_correct"]:7d}, vs bug pred {v["abs_bin_diff_vs_bug_prediction"]:4d}')

os.makedirs(a.out, exist_ok=True)
with open(os.path.join(a.out, 'e2e_summary.json'), 'w') as f:
    json.dump(summary, f, indent=1)
with open(os.path.join(a.out, 'e2e_groups.csv'), 'w') as f:
    names = list(rows)
    f.write('gidx,cls,x,y,z,R200c,n_expected_correct,n_expected_bug,' + ','.join(f'n_{k}' for k in names) + '\n')
    for g in range(Ng):
        f.write(f'{g},{classes[gcls[g]]},{gpos[g,0]:.4f},{gpos[g,1]:.4f},{gpos[g,2]:.4f},{gR[g]:.4f},{exp_n[g].sum()},{exp_n_bug[g].sum()},'
                + ','.join(str(rows[k][g]) for k in names) + '\n')
print('\nwritten', a.out)
