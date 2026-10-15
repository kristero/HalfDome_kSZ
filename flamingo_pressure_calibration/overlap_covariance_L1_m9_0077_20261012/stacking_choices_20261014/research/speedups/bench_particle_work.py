"""Local benchmark of the per-particle work of a profile pass on synthetic particles: (a) per-halo brute-force distance
tests over all candidate particles of the halo's cells, (b) one cKDTree per region + ball queries, (c) accumulation of
per-halo shell sums (np.bincount) for a reduced layout [halo, 4 owner groups, 3 phases, 67 shells] x 6 quantities."""
import json, os, time
import numpy as np
from scipy.spatial import cKDTree
HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(3)
L = 62.5                      # Mpc (2x2x2 cells of 31.25 Mpc)
NP = 8 * 161000               # mean gas rows per cell at z=0 (5.27e9 / 32768)
pos = rng.uniform(0, L, (NP, 3))
# haloes: masses drawn from the z=0 central population above 10^12.5 (census counts per 0.2-dex bin), 13 per cell
counts = np.array([156466, 97982, 61956, 40038, 25741, 16389, 10217, 6302, 3561, 1841, 895, 326, 127, 21, 6], float)
lo = 12.5 + 0.2 * np.arange(15)
nh = 8 * 13
b = rng.choice(15, nh, p=counts / counts.sum())
logm = lo[b] + 0.2 * rng.random(nh)
R200 = (10 ** logm / (4 / 3 * np.pi * 200 * 1.287e11)) ** (1 / 3)   # Mpc, rho_c = 1.287e11 Msun/Mpc^3
cen = rng.uniform(0, L, (nh, 3))
r4 = 4 * R200
res = {'particles': NP, 'haloes': nh}
# (a) brute force: every halo tests every particle in the region (periodic minimum image)
t0 = time.perf_counter(); entries_a = 0
for h in range(nh):
    d = pos - cen[h]; d -= L * np.round(d / L)
    r2 = np.einsum('ij,ij->i', d, d)
    entries_a += int((r2 < r4[h] ** 2).sum())
ta = time.perf_counter() - t0
res['brute_force'] = {'seconds': ta, 'comparisons': nh * NP, 'ns_per_comparison': 1e9 * ta / (nh * NP), 'entries': entries_a}
# (b) tree on particles, ball query per halo
t0 = time.perf_counter(); tree = cKDTree(pos, boxsize=L); tb_build = time.perf_counter() - t0
t0 = time.perf_counter(); lists = tree.query_ball_point(cen, r4, workers=1); tb_q = time.perf_counter() - t0
entries_b = sum(len(x) for x in lists)
res['kdtree'] = {'build_seconds': tb_build, 'query_seconds': tb_q, 'entries': entries_b,
                 'ns_per_particle_build': 1e9 * tb_build / NP, 'ns_per_entry_query': 1e9 * tb_q / max(entries_b, 1)}
assert entries_a == entries_b
# (c) accumulation of reduced per-halo shell sums
edges = np.load('/home/cbllover/HalfDome/flamingo_pressure_calibration/overlap_covariance_L1_m9_0077_20261012/measurements_owner.npz')['x200']
hid = np.concatenate([np.full(len(x), i) for i, x in enumerate(lists)]).astype(np.int64)
pid = np.concatenate([np.asarray(x, np.int64) for x in lists])
w = rng.random((6, len(pid)))
owner = rng.integers(0, 4, len(pid)); phase = rng.integers(0, 3, len(pid))
t0 = time.perf_counter()
d = pos[pid] - cen[hid]; d -= L * np.round(d / L)
x = np.sqrt(np.einsum('ij,ij->i', d, d)) / R200[hid]
shell = np.searchsorted(edges, x, side='right') - 1
ok = (shell >= 0) & (shell < len(edges) - 1)
key = ((hid * 4 + owner) * 3 + phase) * (len(edges) - 1) + shell
nb = nh * 4 * 3 * (len(edges) - 1)
sums = np.stack([np.bincount(key[ok], weights=w[q][ok], minlength=nb) for q in range(6)])
tc = time.perf_counter() - t0
res['accumulate_reduced'] = {'seconds': tc, 'entries': int(len(pid)), 'ns_per_entry': 1e9 * tc / len(pid),
                             'layout': '[halo, 4 owner groups, 3 phases, 67 X200 shells] x 6 quantities, float64'}
res['per_halo_bytes_reduced_layout_f8'] = 4 * 3 * 67 * 6 * 8
res['per_halo_bytes_owner200_f8'] = 17 * 3 * 16 * 67 * 8
json.dump(res, open(os.path.join(HERE, 'bench_particle_work.json'), 'w'), indent=1)
print(json.dumps(res, indent=1))
