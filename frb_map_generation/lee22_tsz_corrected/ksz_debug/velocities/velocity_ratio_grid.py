"""Smooth WebSky/HalfDome <v_los^2> ratio at matched (M200m, z) mapped onto the painter's (log10 M200c [Msun], z) histogram
grid (2026-10-02, ksz_debug/velocities). Writes vel2_ratio_ws_over_hd.h5 (Julia layout (logm, z)) for one_halo_velocity_reweight.jl.
Fit: ln(sigma_WS/sigma_HD) = poly2 in x = log10 M200m - 13.5 and z (with x z term), cells weighted by min(N_WS, N_HD)."""
import numpy as np, h5py
O = "/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_ksz_debug_20261002/velocities/"
w = np.load(O + "websky_velocity_hist.npz"); d = np.load(O + "halfdome_velocity_hist.npz")
LE = w["logm200m_edges"]; ZE = w["z_edges"]
lmc_cent = 0.5 * (LE[1:] + LE[:-1]); zc = 0.5 * (ZE[1:] + ZE[:-1])
nw, sw = w["count"], w["sum_vlos2"]; nh, sh = d["m200m_count"], d["m200m_sum_vlos2"]
ok = (nw > 20) & (nh > 20) & (zc[None, :] < 4.0)
X = np.broadcast_to(lmc_cent[:, None] - 13.5, nw.shape)[ok]; Z = np.broadcast_to(zc[None, :], nw.shape)[ok]
y = 0.5 * np.log((sh[ok] / nh[ok]) / (sw[ok] / nw[ok])) * -1          # ln(sigma_WS / sigma_HD)
wt = np.sqrt(np.minimum(nw[ok], nh[ok]))
A = np.c_[np.ones_like(X), X, X**2, Z, Z**2, X * Z]
coef, *_ = np.linalg.lstsq(A * wt[:, None], y * wt, rcond=None)
print("fit coefficients ln(sWS/sHD):", coef.round(4))
model = lambda x, z: np.exp(np.c_[np.ones_like(x), x, x**2, z, z**2, x * z] @ coef)
res = y - np.log(model(X, Z)); print("weighted rms residual of ln ratio: %.4f" % np.sqrt(np.average(res**2, weights=wt**2)))
for lm in (12.9, 13.25, 13.75, 14.25, 14.75, 15.1):
    print("log10 M200m %.2f: sigma_HD/sigma_WS at z=0.1,0.5,1,2,3 =" % lm, (1 / model(np.full(5, lm - 13.5), np.array([0.1, 0.5, 1, 2, 3]))).round(3))
# painter grid: log10 M200c [Msun] 12.0:0.025:15.7, z 0:0.025:4
with h5py.File("/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_ksz_20261002/maps/halo_mass_redshift_histogram.h5", "r") as f:
    le_p = f["logm_edges"][:]; ze_p = f["z_edges"][:]
lm_p = 0.5 * (le_p[1:] + le_p[:-1]); z_p = 0.5 * (ze_p[1:] + ze_p[:-1])
# HalfDome mean log10 M200m in M200c cells (0.05 grid) -> offset M200m - M200c as a function of (M200c, z)
nc = d["m200c_count"]; off = np.where(nc > 0, d["m200c_sum_log10m200m"] / np.maximum(nc, 1) - lmc_cent[:, None], np.nan)
# smooth offset fit (poly in M200c and z)
okc = nc > 5
Xc = np.broadcast_to(lmc_cent[:, None] - 13.5, nc.shape)[okc]; Zc = np.broadcast_to(zc[None, :], nc.shape)[okc]
Ac = np.c_[np.ones_like(Xc), Xc, Xc**2, Zc, Zc**2, Zc**3, Xc * Zc]
cc, *_ = np.linalg.lstsq(Ac * np.sqrt(nc[okc])[:, None], off[okc] * np.sqrt(nc[okc]), rcond=None)
offm = lambda x, z: np.c_[np.ones_like(x), x, x**2, z, z**2, z**3, x * z] @ cc
print("M200m-M200c offset [dex] at log10 M200c=13.5: z=0.1,1,2,3 ->", offm(np.zeros(4), np.array([0.1, 1, 2, 3])).round(3))
LMp, Zp = np.meshgrid(lm_p, z_p, indexing="ij")
lm200m = LMp + offm((LMp - 13.5).ravel(), Zp.ravel()).reshape(LMp.shape)
ratio_v2 = model((lm200m - 13.5).ravel(), Zp.ravel()).reshape(LMp.shape) ** 2      # <v^2>_WS / <v^2>_HD
with h5py.File(O + "vel2_ratio_ws_over_hd.h5", "w") as f:
    f["ratio_v2"] = ratio_v2.T          # h5py (z, logm) -> Julia (logm, z)
    f["logm_edges"] = le_p; f["z_edges"] = ze_p
    f.attrs["note"] = "<v_los^2>_WebSky / <v_los^2>_HalfDome at matched M200m and z, on the painter (log10 M200c[Msun], z) grid"
with h5py.File("/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_ksz_20261002/maps/halo_mass_redshift_histogram.h5", "r") as f:
    cnt = f["count"][:].T; v2 = f["sum_vlos_over_c_squared"][:].T
print("count-weighted mean ratio_v2 = %.4f ; (v^2)-weighted = %.4f" % ((ratio_v2 * cnt).sum() / cnt.sum(), (ratio_v2 * v2).sum() / v2.sum()))
