"""Expected size of the group_particles PER() boundary bug for Lee et al. 2022's TNG300 profiles.

Mechanism (measured with unit_harness.cpp and the real y_prof.cpp, see analyze_unit.py / e2e):
a (halo, particle) pair is dropped iff the particle is reached across a low box face (x=0, y=0 or
z=0) from the halo.  For a halo at distance d < r from one low face, the shell at radius r loses
the spherical-cap fraction (1 - d/r)/2; for halos placed uniformly in the box (box faces are
arbitrary planes with respect to the large-scale structure) the expected lost fraction of the shell
is  E[f](r) = P(some n_i r > x_i)  ~= 3 r / (4 B)  for r << B.
The mean (stacked) profile of Lee22, <P_e/P_200> in a (M, z, c) bin of N halos, is therefore
biased by -E[f](x R200c) with a halo-sampling scatter of sqrt(Var f / N) ~= sqrt(r / (4 B N)).

Writes <OUT>/estimate/lee22_bias_table.csv and lee22_bias_estimate.json.
"""
import json, os, sys
import numpy as np

OUT = sys.argv[1] if len(sys.argv) > 1 else '/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_boundary_check_20261007'
os.makedirs(os.path.join(OUT, 'estimate'), exist_ok=True)
B = 205000.0                       # ckpc/h, TNG300-1
Om, OL, RHOC0 = 0.3089, 0.6911, 2.775e-8
rng = np.random.default_rng(1)

def r200c_com(M, z):                # M in Msun/h -> comoving ckpc/h
    E2 = Om * (1 + z) ** 3 + OL
    return np.cbrt(3 * (M / 1e10) / (4 * np.pi * 200 * RHOC0 * E2)) * (1 + z)

def mc_lost_fraction(r, n=4_000_000):
    """Exact expectation (union of up to three caps) for uniform halo positions: Monte Carlo."""
    # only halos with some x_i < r can lose anything; sample x_i uniform on [0, B) via the near-face slab
    v = rng.normal(size=(n, 3)); v /= np.linalg.norm(v, axis=1)[:, None]
    x = rng.uniform(0, B, size=(n, 3))
    return np.mean(((x + r * v) < 0).any(axis=1))

def mc_lost_fraction_slab(r, n=4_000_000):
    """Same expectation with variance reduction: x_i ~ U(0, r) in a random subset, reweighted."""
    # P(lost) = sum over nonempty subsets S of faces within r ... evaluate by conditioning on each
    # coordinate being < r with probability p = r/B (independent), directions isotropic
    p = r / B
    near = rng.uniform(size=(n, 3)) < p
    x = np.where(near, rng.uniform(0, r, size=(n, 3)), np.inf)
    v = rng.normal(size=(n, 3)); v /= np.linalg.norm(v, axis=1)[:, None]
    return np.mean(((x + r * v) < 0).any(axis=1))

xs = [0.04, 0.1, 0.3, 0.5, 1.0, 1.34, 2.5]
zs = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.5, 2.0]   # TNG full snapshots up to z = 2
logMs = [13.0, 13.1, 13.5, 14.0, 14.1, 14.2]
rows = []
for lm in logMs:
    for z in zs:
        R = r200c_com(10 ** lm, z)
        row = {'log10M200c_Msun_h': lm, 'z': z, 'R200c_ckpc_h': R, 'R_search_2p5R200c_cells': 2.5 * R / (B / 64)}
        for x in xs:
            r = x * R
            row[f'frac_halos_affected_x{x}'] = 1 - (1 - r / B) ** 3
            row[f'mean_bias_x{x}'] = -0.75 * r / B
        rows.append(row)

# Monte Carlo check of the first-order law (exact union of caps), at a few radii
mc = []
for r in [300.0, 1000.0, 3000.0, 6000.0]:
    e = mc_lost_fraction_slab(r)
    mc.append({'r_ckpc_h': r, 'mc_expected_lost_fraction': e, 'law_3r_over_4B': 0.75 * r / B})

# halo-sampling scatter of a bin mean, N halos
scatter = []
for N in [30, 100, 300, 1000, 3000]:
    for r in [300.0, 1000.0, 2000.0]:
        scatter.append({'N_halos': N, 'r_ckpc_h': r, 'mean_bias': -0.75 * r / B,
                        'std_of_bias_over_halo_samples': np.sqrt(r / (4 * B * N)),
                        'one_halo_on_a_low_face_bias_at_r_gg_d': -0.5 / N})

with open(os.path.join(OUT, 'estimate', 'lee22_bias_table.csv'), 'w') as f:
    keys = list(rows[0])
    f.write(','.join(keys) + '\n')
    for r in rows:
        f.write(','.join(f'{r[k]:.6g}' for k in keys) + '\n')
json.dump({'Bsize_ckpc_h': B, 'law': 'mean lost fraction of a shell at radius r = 3 r / (4 Bsize) (first order)',
           'monte_carlo_check': mc, 'bin_mean_scatter': scatter,
           'rows': rows}, open(os.path.join(OUT, 'estimate', 'lee22_bias_estimate.json'), 'w'), indent=1)

print('Monte Carlo (union of caps, uniform positions) vs 3r/(4B):')
for m in mc:
    print(f"  r = {m['r_ckpc_h']:7.0f} ckpc/h: MC {100*m['mc_expected_lost_fraction']:.4f}%  law {100*m['law_3r_over_4B']:.4f}%")
print('\nExpected bias of the stacked mean profile (percent) and fraction of halos affected within 1.34 R200c:')
print(f'{"log M":>6s} {"z":>4s} {"R200c":>7s} {"R/acell":>7s} | ' + ' '.join(f'x={x:<5}' for x in xs) + ' | aff(1.34) aff(2.5)')
for r in rows:
    if r['z'] in (0.0, 1.0, 2.0) and r['log10M200c_Msun_h'] in (13.0, 13.5, 14.0, 14.2):
        print(f"{r['log10M200c_Msun_h']:6.1f} {r['z']:4.1f} {r['R200c_ckpc_h']:7.1f} {r['R_search_2p5R200c_cells']:7.3f} | "
              + ' '.join(f"{100*r[f'mean_bias_x{x}']:7.3f}" for x in xs)
              + f" | {100*r['frac_halos_affected_x1.34']:6.2f}%  {100*r['frac_halos_affected_x2.5']:6.2f}%")
print('\nscatter of a bin mean over halo samples (percent):')
for s in scatter:
    if s['r_ckpc_h'] == 1000.0:
        print(f"  N={s['N_halos']:5d} r=1000 ckpc/h: mean {100*s['mean_bias']:.3f}%  std {100*s['std_of_bias_over_halo_samples']:.3f}%  "
              f"one halo on a face: {100*s['one_halo_on_a_low_face_bias_at_r_gg_d']:.2f}%")
# extreme radii of the search sphere in cell units (the R/acell and Rsq/acell^2 numbers)
for lm, z in [(13.0, 0.0), (15.0, 2.0)]:
    R = 2.5 * r200c_com(10 ** lm, z)
    print(f'R = 2.5 R200c for 10^{lm} at z={z}: {R:.1f} ckpc/h -> R_normalized = R/acell = {R/(B/64):.4f}, '
          f'Rsq_normalized = Rsq/acell^2 = {(R/(B/64))**2:.4f} (= 4096 R^2/B^2)')
