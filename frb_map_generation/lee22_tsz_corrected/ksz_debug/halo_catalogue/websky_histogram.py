"""Full-catalogue (log10 M200m [Msun], z) histogram of the WebSky halos.pksc (Stein+2020) with summed v_los^2 and |v|^2,
for the HalfDome-vs-WebSky halo catalogue comparison (lee22_ksz_debug_20261002/halo_catalogue, 2026-10-02).
Format (WebSky docs): int32 N, float32 RTHmax, float32 redshiftbox, N x 10 float32 (x,y,z [comoving Mpc], vx,vy,vz [km/s],
R_TH [comoving Mpc], x_lag,y_lag,z_lag). M200m = 4pi/3 rho_m,0(comoving) R_TH^3, rho_m = 2.775e11 Omega_m h^2 Msun/Mpc^3.
Redshift from chi with Omega_m = 0.31, h = 0.68 (flat LCDM). Usage: python3.8 websky_histogram.py <pksc> <out.npz>"""
import sys, time, numpy as np
path, out = sys.argv[1], sys.argv[2]
OM, H = 0.31, 0.68
C = 299792.458
za = np.linspace(0, 6, 600001)
Ez = np.sqrt(OM * (1 + za) ** 3 + 1 - OM)
chia = np.concatenate([[0], np.cumsum(0.5 * (1 / Ez[1:] + 1 / Ez[:-1]) * np.diff(za))]) * C / (100 * H)   # Mpc
RHO_M = 2.775e11 * OM * H ** 2
LOGM_EDGES = np.arange(11.0, 16.0001, 0.025); Z_EDGES = np.arange(0.0, 4.7001, 0.025)
shape = (len(LOGM_EDGES) - 1, len(Z_EDGES) - 1)
cnt = np.zeros(shape); v2 = np.zeros(shape); v3 = np.zeros(shape)
chi_edges = np.arange(0, 8001, 10.0); chi_cnt = np.zeros(len(chi_edges) - 1)
t0 = time.time()
with open(path, 'rb') as f:
    N = int(np.fromfile(f, np.int32, 1)[0]); RTHmax, zbox = np.fromfile(f, np.float32, 2)
    print('N', N, 'RTHmax', RTHmax, 'redshiftbox', zbox, flush=True)
    chunk = 10_000_000; done = 0; sample = []
    while done < N:
        n = min(chunk, N - done)
        a = np.fromfile(f, np.float32, 10 * n).reshape(n, 10).astype(np.float64)
        chi = np.sqrt((a[:, :3] ** 2).sum(1))
        z = np.interp(chi, chia, za)
        m = 4 * np.pi / 3 * RHO_M * a[:, 6] ** 3
        vlos = (a[:, :3] * a[:, 3:6]).sum(1) / np.maximum(chi, 1e-30)
        v3d2 = (a[:, 3:6] ** 2).sum(1)
        lm = np.log10(m)
        i = np.searchsorted(LOGM_EDGES, lm, 'right') - 1; j = np.searchsorted(Z_EDGES, z, 'right') - 1
        ok = (i >= 0) & (i < shape[0]) & (j >= 0) & (j < shape[1])
        np.add.at(cnt, (i[ok], j[ok]), 1); np.add.at(v2, (i[ok], j[ok]), vlos[ok] ** 2); np.add.at(v3, (i[ok], j[ok]), v3d2[ok])
        chi_cnt += np.histogram(chi, chi_edges)[0]
        sample.append(a[::2000, :].astype(np.float32))   # copy: a view would keep the whole chunk alive
        done += n
        print(f'{done}/{N} halos  outside-grid {np.count_nonzero(~ok)}  logM range {lm.min():.3f}-{lm.max():.3f}  z max {z.max():.3f}  '
              f'{time.time() - t0:.0f} s', flush=True)
np.savez(out, logm200m_edges=LOGM_EDGES, z_edges=Z_EDGES, count=cnt, sum_vlos2_kms2=v2, sum_v3d2_kms2=v3, N=N, RTHmax=RTHmax,
         redshiftbox=zbox, chi_edges_mpc=chi_edges, chi_count=chi_cnt, sample_every2000=np.concatenate(sample).astype(np.float32),
         note='WebSky halos.pksc; M200m [Msun] = 4pi/3 2.775e11 Om h^2 R_TH^3; z from chi (Om 0.31, h 0.68); v km/s')
print('done', time.time() - t0, flush=True)
