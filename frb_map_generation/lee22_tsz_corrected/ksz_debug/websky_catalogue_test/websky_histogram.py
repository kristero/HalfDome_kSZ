# Stream the WebSky halo catalogue (halos.pksc) and build the same (log10 M200c [Msun], z) histogram as the HalfDome kSZ
# painter (count and sum (v_los/c)^2), for several M200m -> M200c conversions, plus diagnostics (2026-10-02, ksz_debug).
# Format (WebSky docs / Stein+2020): int32 N, float32 RTHmax, float32 redshiftbox, then N x 10 float32:
#   x, y, z [comoving Mpc], vx, vy, vz [km/s], R_TH [comoving Mpc, Lagrangian], x_lag, y_lag, z_lag.
# M200m = 4 pi/3 rho_m R_TH^3 with rho_m = 2.775e11 Omega_m h^2 Msun/Mpc^3 (WebSky readme). Observer at the origin.
# Usage: python3.8 websky_histogram.py <halos.pksc> <outdir> [chunk] [max_halos]
import sys, os, time, json
import numpy as np
from scipy.interpolate import RegularGridInterpolator

fn, outdir = sys.argv[1], sys.argv[2]
chunk = int(sys.argv[3]) if len(sys.argv) > 3 else 20_000_000
max_halos = int(sys.argv[4]) if len(sys.argv) > 4 else 0
os.makedirs(outdir, exist_ok=True)

OM, OB, H = 0.31, 0.049, 0.68
OL = 1.0 - OM
C_KMS = 299792.458
RHO_M0 = 2.775e11 * OM * H**2                     # comoving mean matter density, Msun/Mpc^3
RHO_C0 = 2.775e11 * H**2                          # critical density today, Msun/Mpc^3 (physical)

# z(chi): trapezoid integral of c/H on a fine grid (WebSky readme uses the same background, cumsum on 0..6)
zg = np.linspace(0.0, 6.0, 600001)
Hz = 100 * H * np.sqrt(OM * (1 + zg)**3 + OL)
chig = np.concatenate([[0.0], np.cumsum(0.5 * (C_KMS / Hz[1:] + C_KMS / Hz[:-1]) * np.diff(zg))])
def z_of_chi(chi):
    return np.interp(chi, chig, zg)
def E2(z):
    return OM * (1 + z)**3 + OL
def omz(z):
    return OM * (1 + z)**3 / E2(z)

# ---- M200m -> M200c conversions
def mu(x):
    return np.log1p(x) - x / (1 + x)
def solve_x(c, q):
    """x = R200c/R200m for NFW with concentration c (wrt R200m) and q = rho_c/rho_m = 1/Omega_m(z): mu(cx)/x^3 = q mu(c)."""
    lo = np.full(np.shape(c), 1e-3); hi = np.full(np.shape(c), 1.0)
    t = q * mu(c)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        f = mu(c * mid) / mid**3 - t
        lo = np.where(f > 0, mid, lo); hi = np.where(f > 0, hi, mid)
    return 0.5 * (lo + hi)

ztab = np.linspace(0.0, 6.0, 1201)
q = 1.0 / omz(ztab)
# (a) NFW, c200m = 7 fixed (Stein+2020 Sec 3.1.1: "NFW profile with a fixed concentration, c = 7")
x7 = solve_x(np.full_like(ztab, 7.0), q)
ratio_c7m = mu(7.0 * x7) / mu(7.0)                      # M200c/M200m
# (b) NFW, c200c = 7 fixed (alternative reading of the same sentence): R200m = y R200c, mu(7y)/y^3 = Omega_m(z) mu(7)... solve
def solve_y(c, om):
    lo = np.full(np.shape(om), 1.0); hi = np.full(np.shape(om), 10.0)
    t = om * mu(c)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        f = mu(c * mid) / mid**3 - t
        lo = np.where(f > 0, mid, lo); hi = np.where(f > 0, hi, mid)
    return 0.5 * (lo + hi)
y7 = solve_y(7.0, omz(ztab))
ratio_c7c = mu(7.0) / mu(7.0 * y7)                      # M200c/M200m
# (c) NFW with Duffy+08 c200m(M200m, z) (full sample, 200m: 10.14 (M/2e12 h^-1)^-0.081 (1+z)^-1.01), as SOConvertNFW.jl
lc = np.linspace(np.log(1.0), np.log(40.0), 400)
CC, ZZ = np.meshgrid(np.exp(lc), ztab, indexing="ij")
ratio_duffy_tab = mu(CC * solve_x(CC, 1.0 / omz(ZZ))) / mu(CC)
duffy_itp = RegularGridInterpolator((lc, ztab), ratio_duffy_tab, bounds_error=True)
def c_duffy(m200m, z):
    return 10.14 * (m200m / (2e12 / H))**(-0.081) * (1 + z)**(-1.01)
# (d) polynomial M200m/M200c(z) in websky_halos_test_kSZ.jl
PA = (1.3595873806301997, -0.49815455039058704, 0.3014644154503205, -0.08294138910919961, 0.0083985355523884)
def poly_ratio(z):
    return PA[0] + PA[1] * z + PA[2] * z**2 + PA[3] * z**3 + PA[4] * z**4
# (e) Omega_m(z)^0.35 in Websky_to_hdf5.jl

def m200c_all(m200m, z):
    out = {}
    out["c7m"] = m200m * np.interp(z, ztab, ratio_c7m)
    out["c7c"] = m200m * np.interp(z, ztab, ratio_c7c)
    c = np.clip(c_duffy(m200m, z), 1.0, 40.0)
    out["duffy"] = m200m * duffy_itp(np.stack([np.log(c), z], axis=-1))
    out["poly"] = m200m / poly_ratio(z)
    out["om035"] = m200m * omz(z)**0.35
    out["m200m"] = m200m                                 # no conversion (for the mass function / bracketing)
    return out
CONV = ("c7m", "c7c", "duffy", "poly", "om035", "m200m")

LOGM_EDGES = np.round(np.arange(12.0, 15.7 + 1e-9, 0.025), 6); Z_EDGES = np.round(np.arange(0.0, 4.0 + 1e-9, 0.025), 6)
NM, NZ = len(LOGM_EDGES) - 1, len(Z_EDGES) - 1
hist = {k: {s: np.zeros(NM * NZ) for s in ("count", "v2", "vtot2", "count_sel", "v2_sel")} for k in CONV}
NZF = 200; zf_edges = np.linspace(0.0, 5.0, NZF + 1)   # fine z diagnostics up to z=5 (all halos)
zdiag = {s: np.zeros(NZF) for s in ("count", "vlos2", "vtot2", "count_m13", "vlos2_m13", "vtot2_m13")}
outside = {k: {"below_mmin": 0, "above_mmax": 0, "z_above_4": 0} for k in CONV}
stats = {"xmin": np.inf, "xmax": -np.inf, "ymin": np.inf, "ymax": -np.inf, "zmin": np.inf, "zmax": -np.inf,
         "chimin": np.inf, "chimax": -np.inf, "sumx": 0.0, "sumy": 0.0, "sumz": 0.0, "sum_vx": 0.0, "sum_vy": 0.0, "sum_vz": 0.0,
         "Rmin": np.inf, "Rmax": -np.inf, "octants": [0] * 8, "n": 0, "sum_disp2": 0.0}
sub = []                                                   # every 1000th halo: M200m, z, chi, vlos, |v|
ARCMIN = np.pi / 180 / 60

with open(fn, "rb") as f:
    N = int(np.fromfile(f, dtype=np.int32, count=1)[0]); hdr = np.fromfile(f, dtype=np.float32, count=2)
    size = os.path.getsize(fn)
    print("N", N, "RTHmax", hdr[0], "redshiftbox", hdr[1], "file size", size, "expected", 12 + 40 * N, flush=True)
    assert size == 12 + 40 * N
    Nuse = N if max_halos == 0 else min(N, max_halos)
    t0 = time.time(); done = 0
    while done < Nuse:
        n = min(chunk, Nuse - done)
        a = np.fromfile(f, dtype=np.float32, count=10 * n).reshape(n, 10)
        x, y, zc = (a[:, i].astype(np.float64) for i in range(3))
        vx, vy, vz = (a[:, i].astype(np.float64) for i in range(3, 6))
        R = a[:, 6].astype(np.float64)
        disp2 = ((a[:, 0] - a[:, 7]).astype(np.float64)**2 + (a[:, 1] - a[:, 8]).astype(np.float64)**2 + (a[:, 2] - a[:, 9]).astype(np.float64)**2)
        del a
        chi = np.sqrt(x * x + y * y + zc * zc)
        z = z_of_chi(chi)
        vlos = (x * vx + y * vy + zc * vz) / chi
        vtot2 = vx * vx + vy * vy + vz * vz
        m200m = 4 * np.pi / 3 * RHO_M0 * R**3
        st = stats
        st["xmin"] = min(st["xmin"], x.min()); st["xmax"] = max(st["xmax"], x.max())
        st["ymin"] = min(st["ymin"], y.min()); st["ymax"] = max(st["ymax"], y.max())
        st["zmin"] = min(st["zmin"], zc.min()); st["zmax"] = max(st["zmax"], zc.max())
        st["chimin"] = min(st["chimin"], chi.min()); st["chimax"] = max(st["chimax"], chi.max())
        st["sumx"] += x.sum(); st["sumy"] += y.sum(); st["sumz"] += zc.sum()
        st["sum_vx"] += vx.sum(); st["sum_vy"] += vy.sum(); st["sum_vz"] += vz.sum()
        st["Rmin"] = min(st["Rmin"], R.min()); st["Rmax"] = max(st["Rmax"], R.max()); st["n"] += n
        st["sum_disp2"] += disp2.sum()
        octv = np.bincount((x > 0).astype(int) * 4 + (y > 0).astype(int) * 2 + (zc > 0).astype(int), minlength=8)
        st["octants"] = [int(o + p) for o, p in zip(st["octants"], octv)]
        izf = np.floor(z / (5.0 / NZF)).astype(np.int64); okf = (izf >= 0) & (izf < NZF)
        zdiag["count"] += np.bincount(izf[okf], minlength=NZF)
        zdiag["vlos2"] += np.bincount(izf[okf], weights=vlos[okf]**2, minlength=NZF)
        zdiag["vtot2"] += np.bincount(izf[okf], weights=vtot2[okf], minlength=NZF)
        m13 = okf & (m200m > 1e13)
        zdiag["count_m13"] += np.bincount(izf[m13], minlength=NZF)
        zdiag["vlos2_m13"] += np.bincount(izf[m13], weights=vlos[m13]**2, minlength=NZF)
        zdiag["vtot2_m13"] += np.bincount(izf[m13], weights=vtot2[m13], minlength=NZF)
        sel = slice(None, None, 1000)
        sub.append(np.stack([m200m[sel], z[sel], chi[sel], vlos[sel], np.sqrt(vtot2[sel])], axis=1))
        iz = np.floor(z / 0.025 + 1e-12).astype(np.int64)
        vc2 = (vlos / C_KMS)**2; vt2 = vtot2 / C_KMS**2
        rhoc_z = RHO_C0 * E2(z)
        for k, mc in m200c_all(m200m, z).items():
            lm = np.log10(mc)
            im = np.floor((lm - 12.0) / 0.025 + 1e-12).astype(np.int64)
            ok = (im >= 0) & (im < NM) & (iz >= 0) & (iz < NZ)
            outside[k]["below_mmin"] += int((im < 0).sum()); outside[k]["above_mmax"] += int((im >= NM).sum())
            outside[k]["z_above_4"] += int(((iz >= NZ) & (im >= 0)).sum())
            idx = im[ok] * NZ + iz[ok]
            hist[k]["count"] += np.bincount(idx, minlength=NM * NZ)
            hist[k]["v2"] += np.bincount(idx, weights=vc2[ok], minlength=NM * NZ)
            hist[k]["vtot2"] += np.bincount(idx, weights=vt2[ok], minlength=NM * NZ)
            # WebSky halo-map selection (Stein+2020 Sec 4.4.2): M200c > 1e13 Msun and r200c subtending > 0.5 arcmin
            r200c_phys = (3 * mc / (4 * np.pi * 200 * rhoc_z))**(1 / 3)
            theta = r200c_phys * (1 + z) / chi
            s = ok & (mc > 1e13) & (theta > 0.5 * ARCMIN)
            idx = im[s] * NZ + iz[s]
            hist[k]["count_sel"] += np.bincount(idx, minlength=NM * NZ)
            hist[k]["v2_sel"] += np.bincount(idx, weights=vc2[s], minlength=NM * NZ)
        done += n
        print(f"  {done}/{Nuse} halos  z {z.min():.3f}-{z.max():.3f}  M200m {m200m.min():.3e}-{m200m.max():.3e}  "
              f"elapsed {time.time() - t0:.0f} s", flush=True)

out = os.path.join(outdir, "websky_halo_mass_redshift_histograms.npz")
save = {"logm_edges": LOGM_EDGES, "z_edges": Z_EDGES, "zf_edges": zf_edges, "ratio_ztab": ztab, "ratio_c7m": ratio_c7m,
        "ratio_c7c": ratio_c7c, "subsample_every1000": np.concatenate(sub)}
for k in CONV:
    for s in hist[k]:
        save[f"{k}_{s}"] = hist[k][s].reshape(NM, NZ)
for s in zdiag:
    save[f"zdiag_{s}"] = zdiag[s]
np.savez(out, **save)
stats["N_header"] = N; stats["RTHmax"] = float(hdr[0]); stats["redshiftbox"] = float(hdr[1]); stats["outside"] = outside
stats["mean_pos"] = [stats["sumx"] / stats["n"], stats["sumy"] / stats["n"], stats["sumz"] / stats["n"]]
stats["mean_vel"] = [stats["sum_vx"] / stats["n"], stats["sum_vy"] / stats["n"], stats["sum_vz"] / stats["n"]]
stats["rms_eulerian_minus_lagrangian_Mpc"] = float(np.sqrt(stats["sum_disp2"] / stats["n"]))
stats["z_at_chimax"] = float(z_of_chi(stats["chimax"]))
with open(os.path.join(outdir, "websky_catalogue_stats.json"), "w") as g:
    json.dump({k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in stats.items()}, g, indent=1, default=float)
print(json.dumps({k: v for k, v in stats.items() if k not in ("octants",)}, default=float, indent=1))
print("octants", stats["octants"])
for k in CONV:
    print(k, "count in histogram", int(hist[k]["count"].sum()), "selected", int(hist[k]["count_sel"].sum()), outside[k])
print("saved", out, "total", time.time() - t0, "s")
