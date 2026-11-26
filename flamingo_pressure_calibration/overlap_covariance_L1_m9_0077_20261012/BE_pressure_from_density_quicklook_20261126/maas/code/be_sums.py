"""BE step 1: Y200-weighted electron and pressure stacks on the exact BD residual geometry (STRW maas, read-only inputs).

Reads only E/{a2ii,all,hb}, P/{a2ii,all,hb} and small halo columns of the production fitting_inputs.hdf5 files listed in the
BD statistics request (bd code/flamingo_request.json: the files and 109 mass x redshift bins of the FLAMINGO pressure fit of
record). Never reads particles or maps. Writes only under --output.

Per snapshot s, block b (0.2-dex bin of M200c, membership m >= 10^low, m < 10^high, as bd_flamingo_sums.py), spatial fold k:
  w_h       haloes/weight (complete census: 1)
  Y_h       p200 r200c^3 sum_c P^a2ii_hc V_c(<1), V_c(<1) = (min(e_c+1,1)^3 - min(e_c,1)^3)/3  (production Y200 weight;
            identical to bd_flamingo_sums.py, so the pressure stacks reproduce the BD fold sums exactly)
  u_h       = w_h Y_h  (the weight of every Y-weighted stack)
  S^{F,q}_bkc = sum_h u_h F^q_hc,  F in {P, E}, q in {a2ii, all, hb}; P = P_e/P200c, E = n_e mu_XGP/(rho_crit f_b) of halo h
  U^{F,q}_bkc = sum_h w_h F^q_hc  (unit-weight stacks, for the check against the BC step-2 atlas)
  W_bk = sum_h u_h, N_bk = sum_h w_h, n_bk = number of haloes.
Per residual i of the B32 set (rows of results/flamingo_residuals.npz 'B32__*', in file order) and F in {P, E} (all gas):
  v_hi = R_i . F^all_h   (R_i: volume mean over [lo_i, hi_i] for shells, 4 pi int_0^hi x^2 F dx for core / group integrals,
                          evaluated on the stored cells assuming a uniform value inside each cell, as FlamingoSet)
  m_i = sum u v / sum u,  f_i = sqrt(sum u (v - m)^2 / sum u) / m_i   (fractional Y-weighted halo scatter; for P this is
                          the definition of the record's s_i before its 0.05 floor and fallbacks)
"""
import argparse
import hashlib
import json
import os
import resource
import time
from pathlib import Path

os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
import h5py  # noqa: E402
import numpy as np  # noqa: E402
import psutil  # noqa: E402

FIELDS = [('P', 'a2ii'), ('P', 'all'), ('P', 'hb'), ('E', 'a2ii'), ('E', 'all'), ('E', 'hb')]


def guard(started, limit_min):
    if time.monotonic()-started > limit_min*60:
        raise TimeoutError('Duration stop reached')
    rss = psutil.Process().memory_info().rss/2**30
    if rss > 3.0:
        raise MemoryError('RSS stop (3 GiB) reached')
    if psutil.virtual_memory().available < 8*2**30:
        raise MemoryError('Less than 8 GiB free on the node')
    return rss


def operator(fine, lo, hi, kind):
    """Rows of R: shells -> volume-mean weights over [lo, hi]; integrals -> 4 pi/3 (cell volume inside [0, hi])."""
    R = np.zeros((len(lo), len(fine)-1))
    for i in range(len(lo)):
        a = lo[i] if kind[i] == 'shell' else 0.
        v = np.maximum(np.minimum(fine[1:], hi[i])**3-np.maximum(fine[:-1], a)**3, 0.)
        R[i] = v/v.sum() if kind[i] == 'shell' else 4*np.pi/3*v
    return R


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('request', type=Path)
    parser.add_argument('residuals', type=Path, help='BD results/flamingo_residuals.npz')
    parser.add_argument('residual_meta', type=Path, help='BD results/flamingo_residuals.json')
    parser.add_argument('output', type=Path)
    parser.add_argument('--minutes', type=float, default=120.)
    args = parser.parse_args()
    os.nice(19)
    resource.setrlimit(resource.RLIMIT_AS, (int(3.5*2**30), int(3.5*2**30)))
    started = time.monotonic()
    request = json.loads(args.request.read_text())
    meta = json.loads(args.residual_meta.read_text())
    blocks_meta = meta['blocks']
    with np.load(args.residuals) as Z:
        rb, rlo, rhi, rkind = Z['B32__block'], Z['B32__lo'], Z['B32__hi'], Z['B32__kind'].astype(str)
    arrays, receipt, peak = {}, dict(snapshots=[], inputs=dict(request=str(args.request), residuals=str(args.residuals))), 0.
    stat = {F: dict(S0=np.zeros(len(rb)), S1=np.zeros(len(rb)), S2=np.zeros(len(rb))) for F in ('P', 'E')}
    for source in request['snapshots']:
        path = Path(source['basis'])
        digest = hashlib.sha256()
        blocks = source['blocks']
        # residual rows of this snapshot's blocks
        bidx = {b['label']: [i for i, m in enumerate(blocks_meta) if int(m['snapshot']) == int(source['snapshot']) and m['label'] == b['label']]
                for b in blocks}
        with h5py.File(path, 'r') as f:
            fine = f['edges'][:]
            ncell = len(fine)-1
            inner = np.maximum(np.minimum(fine[1:], 1.)**3-np.minimum(fine[:-1], 1.)**3, 0)/3
            ops = {}
            for b in blocks:
                ids = bidx[b['label']]
                if len(ids) != 1:
                    raise ValueError(f"block {source['snapshot']} {b['label']} matches {len(ids)} residual blocks")
                rows = np.flatnonzero(rb == ids[0])
                ops[b['label']] = (rows, operator(fine, rlo[rows], rhi[rows], rkind[rows]))
            acc = {b['label']: dict(W=np.zeros(4), N=np.zeros(4), n=np.zeros(4, np.int64),
                                     **{f'S_{F}_{q}': np.zeros((4, ncell)) for F, q in FIELDS},
                                     **{f'U_{F}_{q}': np.zeros((4, ncell)) for F, q in FIELDS}) for b in blocks}
            size = len(f['haloes/m200c'])
            for first in range(0, size, 4096):
                last = min(first+4096, size)
                prof = {(F, q): f[f'{F}/{q}'][first:last] for F, q in FIELDS}
                for (F, q), a in prof.items():
                    if np.any(~np.isfinite(a)) or np.any(a < 0):
                        raise ValueError(f'Invalid saved values {F}/{q} in {path}')
                digest.update(np.ascontiguousarray(prof[('E', 'all')]).tobytes())
                mass = f['haloes/m200c'][first:last]
                y = f['haloes/p200'][first:last]*f['haloes/r200c'][first:last]**3*(prof[('P', 'a2ii')] @ inner)
                wh = f['haloes/weight'][first:last]
                u = wh*y
                fold = f['haloes/fold'][first:last] % 4
                for b in blocks:
                    chosen = (mass >= 10**b['log_mass_low']) & (mass < 10**b['log_mass_high'])
                    if not chosen.any():
                        continue
                    a = acc[b['label']]
                    for k in range(4):
                        t = chosen & (fold == k)
                        if not t.any():
                            continue
                        a['W'][k] += u[t].sum()
                        a['N'][k] += wh[t].sum()
                        a['n'][k] += int(t.sum())
                        for F, q in FIELDS:
                            a[f'S_{F}_{q}'][k] += u[t] @ prof[(F, q)][t]
                            a[f'U_{F}_{q}'][k] += wh[t] @ prof[(F, q)][t]
                    rows, R = ops[b['label']]
                    for F in ('P', 'E'):
                        v = prof[(F, 'all')][chosen] @ R.T
                        uu = u[chosen][:, None]
                        stat[F]['S0'][rows] += uu[:, 0].sum()
                        stat[F]['S1'][rows] += (uu*v).sum(axis=0)
                        stat[F]['S2'][rows] += (uu*v**2).sum(axis=0)
                peak = max(peak, guard(started, args.minutes))
        for b in blocks:
            a = acc[b['label']]
            if int(a['n'].sum()) != int(b['n_haloes']):
                raise ValueError(f"Halo count {a['n'].sum()} != {b['n_haloes']} for {source['snapshot']} {b['label']}")
            key = f"{source['snapshot']}__{b['label']}"
            for name, value in a.items():
                arrays[f'{key}__{name}'] = value
        arrays[f"{source['snapshot']}__edges"] = fine
        receipt['snapshots'].append(dict(snapshot=source['snapshot'], basis=str(path), E_all_sha256=digest.hexdigest(),
                                         haloes=int(size), blocks=len(blocks)))
        print(source['snapshot'], f'{time.monotonic()-started:.0f} s', flush=True)
    for F in ('P', 'E'):
        S0, S1, S2 = stat[F]['S0'], stat[F]['S1'], stat[F]['S2']
        m = S1/S0
        arrays[f'resid_{F}_mean'] = m
        arrays[f'resid_{F}_frac_scatter'] = np.sqrt(np.maximum(S2/S0-m**2, 0.))/m
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output/'be_sums.npz', **arrays)
    receipt.update(seconds=time.monotonic()-started, peak_rss_GiB=peak, nice=19, threads=1)
    (args.output/'be_sums.json').write_text(json.dumps(receipt, indent=1)+'\n')


if __name__ == '__main__':
    main()
