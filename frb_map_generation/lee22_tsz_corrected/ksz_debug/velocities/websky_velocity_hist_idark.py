"""Full pass over the WebSky halo catalogue (halos.pksc) on idark: per (log10 M200m [Msun], z) cell, the halo count,
sum v_los^2 [km^2/s^2], sum |v|^2 and sum v_los (2026-10-02, ksz_debug/velocities). Also the same histogram for the
Stein+2020 halo-map selection proxy theta_200m > 0.5 arcmin (R200m from M200m; physical R / D_A).
Cosmology Om = 0.31, h = 0.68 (WebSky). M200m = 4pi/3 rho_m R_TH^3, rho_m = 2.775e11 Om h^2 Msun/Mpc^3."""
import numpy as np, sys, os, time
fn = "/home/kristero10/sims/halos.pksc"
out = sys.argv[1]
h, Om = 0.68, 0.31
rho_m = 2.775e11 * Om * h**2
c = 299792.458
zg = np.linspace(0, 6, 60001)
Ez = np.sqrt(Om * (1 + zg) ** 3 + 1 - Om)
chig = np.concatenate([[0], np.cumsum(0.5 * (1 / Ez[1:] + 1 / Ez[:-1]) * np.diff(zg))]) * c / (100 * h)   # Mpc
LE = np.arange(12.0, 15.8001, 0.05); ZE = np.arange(0.0, 4.7001, 0.05)
nM, nZ = len(LE) - 1, len(ZE) - 1
H = {k: np.zeros((nM, nZ)) for k in ("count", "sum_vlos2", "sum_v2", "sum_vlos", "count_th", "sum_vlos2_th")}
CH = 5_000_000
t0 = time.time()
with open(fn, "rb") as f:
    N = int(np.fromfile(f, dtype=np.int32, count=1)[0]); np.fromfile(f, dtype=np.float32, count=2)
    done = 0
    while done < N:
        n = min(CH, N - done)
        d = np.fromfile(f, dtype=np.float32, count=10 * n).reshape(n, 10)
        x = d[:, :3].astype(np.float64); v = d[:, 3:6].astype(np.float64); R = d[:, 6].astype(np.float64)
        r = np.sqrt((x * x).sum(1)); z = np.interp(r, chig, zg)
        vlos = (x * v).sum(1) / r
        M = 4 * np.pi / 3 * rho_m * R**3
        lm = np.log10(M)
        im = np.floor((lm - LE[0]) / 0.05).astype(np.int64); iz = np.floor(z / 0.05).astype(np.int64)
        ok = (im >= 0) & (im < nM) & (iz >= 0) & (iz < nZ)
        idx = im[ok] * nZ + iz[ok]
        for key, w in (("count", None), ("sum_vlos2", vlos[ok] ** 2), ("sum_v2", (v[ok] ** 2).sum(1)), ("sum_vlos", vlos[ok])):
            H[key] += np.bincount(idx, weights=w, minlength=nM * nZ).reshape(nM, nZ)
        # theta_200m > 0.5 arcmin proxy: R200m(phys) = (3M/(4pi 200 rho_m (1+z)^3))^(1/3); D_A = r/(1+z)
        R200m_phys = (3 * M / (4 * np.pi * 200 * rho_m * (1 + z) ** 3)) ** (1 / 3)
        th = R200m_phys / (r / (1 + z)) * (180 / np.pi * 60)
        okt = ok & (th > 0.5)
        idt = im[okt] * nZ + iz[okt]
        H["count_th"] += np.bincount(idt, minlength=nM * nZ).reshape(nM, nZ)
        H["sum_vlos2_th"] += np.bincount(idt, weights=vlos[okt] ** 2, minlength=nM * nZ).reshape(nM, nZ)
        done += n
        print("%d / %d  %.0f s" % (done, N, time.time() - t0), flush=True)
np.savez(out, logm200m_edges=LE, z_edges=ZE, **H)
print("saved", out)
