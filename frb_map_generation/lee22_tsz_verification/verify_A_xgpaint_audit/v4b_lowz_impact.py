"""Verifier A, check 4: impact of the production-interpolator error for nearby halos (z < 0.1) on the painted maps.
For every catalogue halo with z < 0.1: y_itp(x) from v4a (production caches, painter arithmetic) vs my direct LOS y(x) (vlib,
validated against XGPaint direct quadrature to ~1e-9). Outputs: integrated Y excess per z bin, and C_ell of the stored map
vs the stored map with the excess D = y_itp - y_direct removed (NSIDE 1024 pixel centres; stored map ud_graded 4096 -> 1024)."""
import json, sys, time
import numpy as np, healpy as hp
from astropy.io import fits
sys.path.insert(0, "/home/cbllover/HalfDome/frb_map_generation/lee22_tsz_verification/verify_A_xgpaint_audit")
import vlib, h5py
S = "/tmp/claude-0/-home-cbllover-HalfDome/015746ae-94b7-42cb-8dc5-0e8c33dc8490/scratchpad"
O = "/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_tsz_verification_20260929/verify_A_xgpaint_audit"
R = "/home/cbllover/HalfDome/frb_map_generation/outputs"
MAPS = {"b12": f"{R}/tsz_frb_1m_z1_r200c_apertures/maps/battaglia12_fiducial_halfdome_compton_y_allz_nside4096_m200c_r200cx4.fits",
        "lee22": f"{R}/tsz_dm_cross_updated_20260917/inputs/lee22_noconc_pressure_compton_y_allz_nside4096_m200c_r200cx4.fits"}
MEAN_Y = {"b12": 1.0415762787005558e-6, "lee22": 4.561758653063625e-7}   # provenance map_mean_y
t0 = time.time()
mz = np.fromfile(f"{S}/lowz_mz.bin", dtype="<f8").reshape(-1, 2); idx = np.fromfile(f"{S}/lowz_idx.bin", dtype="<i8")
N = len(idx); m, z = mz[:, 0], mz[:, 1]
xs = np.exp(np.linspace(np.log(1e-3), np.log(3.999), 128))
with h5py.File("/home/cbllover/HalfDome/lightcone_100.hdf5", "r") as f:
    order = np.argsort(idx); P = f["Position"][idx[order]]      # h5py needs increasing indices
pos = np.empty_like(P); pos[order] = P
u = pos.astype(np.float64); u /= np.linalg.norm(u, axis=1)[:, None]
res = {}
NS = 1024
for mod, cosm in (("b12", vlib.Cos(neff=3.04)), ("lee22", vlib.Cos(neff=3.046))):
    raw = np.fromfile(f"{S}/lowz_itp_{mod}.bin", dtype="<f8").reshape(N, 130)
    t200, tmax, yi = raw[:, 0], raw[:, 1], raw[:, 2:]
    yd = np.empty_like(yi)
    for i0 in range(0, N, 500):
        sl = slice(i0, min(i0 + 500, N))
        p0, xc, beta = (vlib.b12_params if mod == "b12" else vlib.lee_params)(m[sl], z[sl])
        rho = cosm.rho_crit(z[sl]); r200 = cosm.r200_m(m[sl], z[sl])
        amp = vlib.SIGMA_T / (vlib.ME * vlib.C ** 2) * (200 * vlib.G * m[sl] * vlib.MSUN * rho * vlib.FB / 2) * p0
        Xg = np.broadcast_to(xs[None, :], (len(amp), 128)).ravel()
        I = vlib.los_fast(Xg, np.repeat(xc, 128), np.repeat(beta, 128)).reshape(len(amp), 128)
        yd[sl] = amp[:, None] * I
    th = xs[None, :] * t200[:, None]
    w = 2 * np.pi * np.sin(th)
    Yi = np.trapz(w * yi, th, axis=1); Yd = np.trapz(w * yd, th, axis=1)
    Ytot_map = MEAN_Y[mod] * 4 * np.pi
    bins = [0, 0.005, 0.01, 0.0206, 0.05, 0.1]
    tab = {}
    for a, b in zip(bins[:-1], bins[1:]):
        s = (z >= a) & (z < b)
        tab[f"z{a}-{b}"] = dict(N=int(s.sum()), Y_itp_sr=float(Yi[s].sum()), Y_direct_sr=float(Yd[s].sum()),
                               ratio=float(Yi[s].sum() / Yd[s].sum()), excess_frac_of_map_total=float((Yi[s].sum() - Yd[s].sum()) / Ytot_map))
    tab["all_z<0.1_excess_frac_of_map_total_Y"] = float((Yi.sum() - Yd.sum()) / Ytot_map)
    print(mod, json.dumps(tab, indent=1)); sys.stdout.flush()
    # paint D = y_itp - y_direct at NSIDE 1024 pixel centres
    D = np.zeros(hp.nside2npix(NS))
    lx = np.log(xs)
    for i in range(N):
        pix = hp.query_disc(NS, u[i], tmax[i])
        if len(pix) == 0:
            continue
        v = np.array(hp.pix2vec(NS, pix)).T
        tt = np.arccos(np.clip(v @ u[i], -1, 1))
        ok = tt < tmax[i]
        xx = np.clip(tt[ok] / t200[i], xs[0], xs[-1])
        D[pix[ok]] += np.interp(np.log(xx), lx, yi[i] - yd[i])
    with fits.open(MAPS[mod], memmap=True) as h:
        full = np.array(h[1].data["PIXELS"], dtype=np.float64)
    M1024 = hp.ud_grade(full, NS, order_in="RING"); del full
    lmax = 2000
    cl_s = hp.anafast(M1024, lmax=lmax, iter=1); cl_c = hp.anafast(M1024 - D, lmax=lmax, iter=1)
    ell = np.arange(lmax + 1)
    sel = {L: float(cl_s[L - 5:L + 6].mean() / cl_c[L - 5:L + 6].mean()) for L in (10, 20, 30, 50, 100, 200, 300, 500, 1000, 1500, 2000 - 6)}
    print(mod, "C_ell(stored)/C_ell(corrected):", sel, "mean y stored/corrected", M1024.mean() / (M1024 - D).mean()); sys.stdout.flush()
    res[mod] = dict(Y_budget=tab, cl_stored_over_corrected=sel, mean_y_stored_over_corrected=float(M1024.mean() / (M1024 - D).mean()),
                    D_max=float(D.max()), D_min=float(D.min()))
    np.savez_compressed(f"{O}/v4b_cl_{mod}.npz", ell=ell, cl_stored=cl_s, cl_corrected=cl_c)
    del D, M1024
json.dump(dict(res=res, elapsed_s=time.time() - t0), open(f"{O}/v4b_lowz_impact.json", "w"), indent=1)
print("done", time.time() - t0)
