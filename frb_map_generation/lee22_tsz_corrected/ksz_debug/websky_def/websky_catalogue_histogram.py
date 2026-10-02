"""Full WebSky halos.pksc pass (idark): (log10 M200c [Msun], z) histograms of counts and sum (v_rad/c)^2, the same
quantities the HalfDome painter writes, so the catalogue one-halo kSZ term can be computed for WebSky.
WebSky conventions (readhalos.py, Stein+2020 Sec 3.1.1): M200m = 4pi/3 rho_m R_TH^3, rho_m = 2.775e11 Om h^2 Msun/Mpc^3;
M200c from M200m assuming NFW with c200m = 7 (paper: 'NFW profile with a fixed concentration, c = 7').
Also stores the M200m histogram and per-z velocity moments."""
import numpy as np, sys, time
path = sys.argv[1]; out = sys.argv[2]
Om, h = 0.31, 0.68; OL = 1 - Om; C = 299792.458
rho_m = 2.775e11 * Om * h**2
za = np.linspace(0, 6, 100000); dz = za[1] - za[0]
chia = np.cumsum(C / (100*h*np.sqrt(Om*(1+za)**3 + OL))) * dz     # same as WebSky cosmology.py
def zofchi(chi): return np.interp(chi, chia, za)
# NFW c200m=7 -> M200c/M200m as a function of z (depends only on Omega_m(z))
def mfun(x): return np.log1p(x) - x/(1+x)
cm = 7.0
zt = np.linspace(0, 5, 501); ratio = np.empty_like(zt); rratio = np.empty_like(zt)
xs = np.linspace(0.5, 7.0, 200001)
for i, z in enumerate(zt):
    Omz = Om*(1+z)**3 / (Om*(1+z)**3 + OL)
    target = mfun(cm)/cm**3/Omz              # m(x)/x^3 = m(c)/(c^3 Omega_m(z)),  x = R200c/rs
    g = mfun(xs)/xs**3 - target
    j = np.where(np.diff(np.sign(g)))[0][0]
    x = xs[j] - g[j]*(xs[j+1]-xs[j])/(g[j+1]-g[j])
    ratio[i] = mfun(x)/mfun(cm); rratio[i] = x/cm
LOGM = np.arange(11.5, 15.7 + 1e-9, 0.025); ZE = np.arange(0, 4.75 + 1e-9, 0.025)
nM, nZ = len(LOGM)-1, len(ZE)-1
cnt = np.zeros((nM, nZ)); v2 = np.zeros((nM, nZ)); cntm = np.zeros((nM, nZ)); v2m = np.zeros((nM, nZ))
vz_n = np.zeros(nZ); vz_r2 = np.zeros(nZ); vz_3d2 = np.zeros(nZ)
f = open(path, 'rb'); N = int(np.fromfile(f, count=3, dtype=np.int32)[0]); f.seek(12)
chunk = 10_000_000; done = 0; t0 = time.time()
while done < N:
    n = min(chunk, N - done)
    c = np.fromfile(f, count=10*n, dtype=np.float32).reshape(n, 10).astype(np.float64)
    x, y, zz, vx, vy, vz, R = (c[:, k] for k in range(7))
    chi = np.sqrt(x*x + y*y + zz*zz); vr = (x*vx + y*vy + zz*vz)/chi
    z = zofchi(chi); M200m = 4*np.pi/3*rho_m*R**3; M200c = M200m*np.interp(z, zt, ratio)
    iz = np.floor(z/0.025).astype(int); ok = (iz >= 0) & (iz < nZ)
    for M, H, V in ((M200c, cnt, v2), (M200m, cntm, v2m)):
        im = np.floor((np.log10(M) - 11.5)/0.025).astype(int); m = ok & (im >= 0) & (im < nM)
        lin = im[m]*nZ + iz[m]
        H += np.bincount(lin, minlength=nM*nZ).reshape(nM, nZ)
        V += np.bincount(lin, weights=(vr[m]/C)**2, minlength=nM*nZ).reshape(nM, nZ)
    vz_n += np.bincount(iz[ok], minlength=nZ); vz_r2 += np.bincount(iz[ok], weights=vr[ok]**2, minlength=nZ)
    vz_3d2 += np.bincount(iz[ok], weights=(vx*vx+vy*vy+vz*vz)[ok], minlength=nZ)
    done += n
    print(f"{done}/{N} rows, {time.time()-t0:.0f} s, last M200m {M200m[-1]:.3e}", flush=True)
np.savez(out, logm_edges=LOGM, z_edges=ZE, count_m200c=cnt, sum_vr_over_c_sq_m200c=v2, count_m200m=cntm,
         sum_vr_over_c_sq_m200m=v2m, z_table=zt, m200c_over_m200m=ratio, r200c_over_r200m=rratio,
         vz_count=vz_n, vz_sum_vr2=vz_r2, vz_sum_v3d2=vz_3d2, N=N)
print("done", time.time()-t0)
