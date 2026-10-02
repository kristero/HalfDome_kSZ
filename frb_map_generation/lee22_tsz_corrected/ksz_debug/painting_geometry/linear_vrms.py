"""Linear-theory LOS (1-D) velocity rms vs z for the HalfDome cosmology, unsmoothed and smoothed on the Lagrangian
radius of a halo (top hat), to compare with the catalogue's per-cell v_los rms (halo_mass_redshift_histogram.h5)."""
import numpy as np, camb, h5py
h = 0.68; Ob = 0.049; Oc = 0.261
pars = camb.set_params(H0=100 * h, ombh2=Ob * h**2, omch2=Oc * h**2, ns=0.965, As=2.1e-9, WantTransfer=True)
zs = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0]
pars.set_matter_power(redshifts=zs, kmax=50.0)
res = camb.get_results(pars)
s8 = res.get_sigma8_0(); scale = (0.81 / s8) ** 2
kh, zz, pk = res.get_matter_power_spectrum(minkh=1e-4, maxkh=50, npoints=2000)
k = kh * h; P = pk * scale / h**3            # 1/Mpc, Mpc^3
Om = Ob + Oc
rho_m0 = Om * 2.775e11 * h**2                # Msun/Mpc^3
f = h5py.File("/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_ksz_20261002/maps/halo_mass_redshift_histogram.h5")
le = f["logm_edges"][:]; ze = f["z_edges"][:]; cnt = f["count"][:]; v2 = f["sum_vlos_over_c_squared"][:]
C = 299792.458
for i, z in enumerate(zz):
    a = 1 / (1 + z); E = np.sqrt(Om * (1 + z)**3 + 1 - Om); fz = (Om * (1 + z)**3 / E**2) ** 0.55
    aH = 100 * h * E * a
    out = []
    for lm in (None, 13.0, 13.5, 14.0):
        W = 1.0
        if lm is not None:
            R = (3 * 10**lm / (4 * np.pi * rho_m0)) ** (1 / 3); x = k * R
            W = 3 * (np.sin(x) - x * np.cos(x)) / x**3
        s2 = (fz * aH) ** 2 * np.trapz(P[i] * W**2, k) / (2 * np.pi**2) / 3.0   # 1-D
        out.append(np.sqrt(s2))
    iz = np.searchsorted(ze, z + 1e-9) - 1; iz = max(iz, 0)
    sl = slice(max(iz - 2, 0), iz + 3)
    cat = {}
    for lo, hi in ((12.8, 13.25), (13.25, 13.75), (13.75, 14.25)):
        mm = (le[:-1] >= lo) & (le[:-1] < hi)
        n = cnt[sl][:, mm].sum(); cat[(lo, hi)] = np.sqrt(v2[sl][:, mm].sum() / max(n, 1)) * C
    print("z=%.2f  linear 1-D vrms: matter %.0f, R_L(1e13) %.0f, R_L(10^13.5) %.0f, R_L(1e14) %.0f km/s | catalogue v_los rms "
          "logM 12.8-13.25: %.0f, 13.25-13.75: %.0f, 13.75-14.25: %.0f km/s" % (z, *out, *cat.values()))
