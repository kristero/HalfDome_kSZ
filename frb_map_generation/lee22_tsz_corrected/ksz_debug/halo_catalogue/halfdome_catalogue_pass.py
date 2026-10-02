"""One chunked pass over the HalfDome lightcone_100.hdf5 halo catalogue (2026-10-02, ksz_debug/halo_catalogue):
column sanity (Selection, Weight, Value, Aemit vs redshift), M200c/M200m ratio, |Position| vs chi(z), velocity rms per z,
duplicate MinID / (Task, ID) candidates, and (log10 M200m [Msun], z) / (log10 M200c, z) histograms with summed v_los^2, |v|^2."""
import h5py, numpy as np, time, sys
CAT = '/home/cbllover/HalfDome/lightcone_100.hdf5'
OUT = '/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_ksz_debug_20261002/halo_catalogue/halfdome_catalogue_pass.npz'
H = 0.68; C = 299792.458
za = np.linspace(0, 6, 600001)
def chi_of(om):
    E = np.sqrt(om * (1 + za) ** 3 + 1 - om)
    return np.concatenate([[0], np.cumsum(0.5 * (1 / E[1:] + 1 / E[:-1]) * np.diff(za))]) * C / 100.0   # Mpc/h
OMS = [0.30, 0.305, 0.3097, 0.31, 0.315, 0.32]
chis = {om: chi_of(om) for om in OMS}
LOGM_EDGES = np.arange(11.0, 16.0001, 0.025); Z_EDGES = np.arange(0.0, 4.0001, 0.025)
nm, nz = len(LOGM_EDGES) - 1, len(Z_EDGES) - 1
hist = {k: np.zeros((nm, nz)) for k in ['n_c', 'v2_c', 'n_m', 'v2_m', 'v3_m']}
ZB = np.arange(0, 4.01, 0.1); nzb = len(ZB) - 1
acc = {k: np.zeros(nzb) for k in ['n', 'vlos2', 'v3d2', 'ratio', 'ratio2', 'lnratio_dev']}
dres = {om: np.zeros((nzb, 3)) for om in OMS}      # sum res, sum res^2, sum |res|>1
ratio_hist = np.zeros((nzb, 200)); ratio_edges = np.linspace(0.4, 1.2, 201)
f = h5py.File(CAT, 'r'); N = f['redshift'].shape[0]
minid = np.empty(N, np.int64); taskid = np.empty(N, np.int64)
bad = dict(sel_false=0, weight_ne1=0, value_ne1=0, aemit_maxdev=0.0, m200c_gt_m200m=0, nonfinite=0, m200c_le0=0)
wsum = 0.0; weights_seen = set(); values_seen = set()
lengths_min = 10**9
t0 = time.time(); CH = 2_000_000
for s in range(0, N, CH):
    e = min(N, s + CH); sl = slice(s, e)
    z = f['redshift'][sl].astype(np.float64); a = f['Aemit'][sl].astype(np.float64)
    mc = f['halo_mass_m200c'][sl] / H; mm = f['halo_mass_m200m'][sl] / H
    pos = f['Position'][sl].astype(np.float64); vel = f['Velocity'][sl].astype(np.float64)
    sel = f['Selection'][sl]; w = f['Weight'][sl]; val = f['Value'][sl]
    minid[sl] = f['MinID'][sl]; tk = f['Task'][sl].astype(np.int64); idd = f['ID'][sl].astype(np.int64)
    taskid[sl] = tk * (1 << 32) + idd
    L = f['Length'][sl]; lengths_min = min(lengths_min, int(L.min()))
    bad['sel_false'] += int(np.count_nonzero(~sel)); bad['weight_ne1'] += int(np.count_nonzero(w != 1)); bad['value_ne1'] += int(np.count_nonzero(val != 1))
    weights_seen.update(np.unique(w)[:20].tolist()); values_seen.update(np.unique(val)[:20].tolist())
    bad['aemit_maxdev'] = max(bad['aemit_maxdev'], float(np.max(np.abs(1 / a - 1 - z) / (1 + z))))
    bad['m200c_gt_m200m'] += int(np.count_nonzero(mc > mm)); bad['m200c_le0'] += int(np.count_nonzero(mc <= 0))
    bad['nonfinite'] += int(np.count_nonzero(~np.isfinite(mc) | ~np.isfinite(z) | ~np.isfinite(pos).all(1) | ~np.isfinite(vel).all(1)))
    r = np.sqrt((pos ** 2).sum(1)); vlos = (pos * vel).sum(1) / r; v3 = (vel ** 2).sum(1)
    jz = np.clip(np.searchsorted(ZB, z, 'right') - 1, 0, nzb - 1)
    ratio = mc / mm
    np.add.at(acc['n'], jz, 1); np.add.at(acc['vlos2'], jz, vlos ** 2); np.add.at(acc['v3d2'], jz, v3)
    np.add.at(acc['ratio'], jz, ratio); np.add.at(acc['ratio2'], jz, ratio ** 2)
    ir = np.clip(np.searchsorted(ratio_edges, ratio, 'right') - 1, 0, 199); np.add.at(ratio_hist, (jz, ir), 1)
    for om in OMS:
        res = r - np.interp(z, za, chis[om])
        np.add.at(dres[om][:, 0], jz, res); np.add.at(dres[om][:, 1], jz, res ** 2); np.add.at(dres[om][:, 2], jz, np.abs(res) > 1.0)
    j = np.searchsorted(Z_EDGES, z, 'right') - 1
    for tag, m in (('c', mc), ('m', mm)):
        i = np.searchsorted(LOGM_EDGES, np.log10(m), 'right') - 1
        ok = (i >= 0) & (i < nm) & (j >= 0) & (j < nz)
        np.add.at(hist['n_' + tag], (i[ok], j[ok]), 1); np.add.at(hist['v2_' + tag], (i[ok], j[ok]), vlos[ok] ** 2)
        if tag == 'm': np.add.at(hist['v3_m'], (i[ok], j[ok]), v3[ok])
    print(f'{e}/{N}  {time.time() - t0:.0f} s', flush=True)
# duplicates
def dup_info(key):
    o = np.argsort(key, kind='stable'); ks = key[o]; d = np.nonzero(ks[1:] == ks[:-1])[0]
    return o, ks, d
o, ks, d = dup_info(minid)
n_dup_minid = len(d)
pairs = np.stack([o[d], o[d + 1]], 1) if len(d) else np.zeros((0, 2), int)
print('duplicate MinID pairs', n_dup_minid, flush=True)
o2, ks2, d2 = dup_info(taskid); n_dup_taskid = len(d2)
print('duplicate (Task,ID) pairs', n_dup_taskid, flush=True)
# inspect duplicate MinID pairs: separation, dz, masses
pinfo = []
if len(pairs):
    take = pairs[:min(len(pairs), 2_000_000)]
    idx = np.unique(take.ravel())
    P = np.empty((len(idx), 3)); Z = np.empty(len(idx)); M = np.empty(len(idx)); V = np.empty((len(idx), 3))
    # read in sorted order by blocks
    for b in range(0, len(idx), 200000):
        ii = idx[b:b + 200000]
        P[b:b + len(ii)] = f['Position'][ii]; Z[b:b + len(ii)] = f['redshift'][ii]; M[b:b + len(ii)] = f['halo_mass_m200c'][ii] / H
        V[b:b + len(ii)] = f['Velocity'][ii]
    ia = np.searchsorted(idx, take[:, 0]); ib = np.searchsorted(idx, take[:, 1])
    sep = np.sqrt(((P[ia] - P[ib]) ** 2).sum(1)); dzp = np.abs(Z[ia] - Z[ib])
    sepmod = np.abs(((P[ia] - P[ib]) + 1875.0) % 3750.0 - 1875.0); sepmod = np.sqrt((sepmod ** 2).sum(1))
    pinfo = dict(sep=sep, dz=dzp, ma=M[ia], mb=M[ib], za=Z[ia], zb=Z[ib], sepmod=sepmod, ia=take[:, 0], ib=take[:, 1],
                 dv=np.sqrt(((V[ia] - V[ib]) ** 2).sum(1)))
np.savez(OUT, logm_edges=LOGM_EDGES, z_edges=Z_EDGES, **{'hist_' + k: v for k, v in hist.items()},
         zb=ZB, **{'acc_' + k: v for k, v in acc.items()}, ratio_hist=ratio_hist, ratio_edges=ratio_edges,
         oms=np.array(OMS), dres=np.stack([dres[om] for om in OMS]), n_dup_minid=n_dup_minid, n_dup_taskid=n_dup_taskid,
         lengths_min=lengths_min, **{'bad_' + k: v for k, v in bad.items()},
         weights_seen=np.array(sorted(weights_seen)), values_seen=np.array(sorted(values_seen)),
         **({'dup_' + k: v for k, v in pinfo.items()} if len(pairs) else {}))
print(bad, 'weights', sorted(weights_seen)[:10], 'values', sorted(values_seen)[:10], 'Lmin', lengths_min)
print('done', time.time() - t0)
