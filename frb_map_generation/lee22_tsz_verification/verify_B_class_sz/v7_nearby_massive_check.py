"""Accuracy check of the v2 binned catalogue 1h for the halos that dominate the Lee22 low-ell power
(M200c >= 10^14.8 Msun/h, z < 0.1): (a) per-halo y_ell at the exact (m, z) instead of the bin mean, (b) the X-grid /
table kernel against a direct adaptive quadrature for the nearest such halo."""
import json, sys
import numpy as np, h5py
from scipy.integrate import quad
from scipy.special import j0
sys.path.insert(0, "/home/cbllover/HalfDome/frb_map_generation/lee22_tsz_verification/verify_B_class_sz")
import v2_catalogue_poisson as v2
OUT = "/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_tsz_verification_20260929/verify_B_class_sz/"
with h5py.File(v2.CAT, "r") as f:
    m = f["halo_mass_m200c"][:]; z = f["redshift"][:].astype(float)
s = (m >= 10 ** 14.8) & (z < 0.1)
mh, zz = m[s], z[s]; del m, z
print("N halos above clip at z<0.1:", mh.size, "z range", zz.min(), zz.max())
Mp = mh / v2.H; r = v2.R200(Mp, zz); th = r / v2.D_A_cm(zz); amp = v2.SIGMA_T / v2.MEC2 * v2.P200(Mp, zz) * r
out = {"N": int(mh.size)}
for name, pars in [("b12", v2.b12_params(Mp, zz)), ("lee22", v2.lee_params(mh, zz))]:
    yl, _ = v2.yell(pars[0], pars[1], pars[2], th, amp)
    c = (yl ** 2).sum(0) / (4 * np.pi)
    out[name + "_exact_positions"] = (c.reshape(v2.ELLS.size, v2.FR.size) * v2.FW[None, :]).sum(1).tolist()
v2res = json.load(open(OUT + "v2_catalogue_poisson.json"))["cl_1h_catalogue"]
for name in ["b12", "lee22"]:
    out[name + "_binned (v2)"] = v2res[name + "_high_zlt0.1"]
    out[name + " exact/binned"] = (np.array(out[name + "_exact_positions"]) / np.array(v2res[name + "_high_zlt0.1"])).tolist()
# direct quadrature for the nearest halo
i = int(np.argmin(zz)); P0, xc, beta = [np.atleast_1d(p)[i] for p in v2.lee_params(mh, zz)]
Sig = lambda X: 2 * quad(lambda s_: (np.sqrt(X * X + s_ * s_) / xc) ** -0.3 * (1 + np.sqrt(X * X + s_ * s_) / xc) ** -beta, 0, np.inf, limit=400)[0]
Xg = np.linspace(1e-6, 4, 4001); Sg = np.array([Sig(x) for x in Xg])
dir_ = []
for ell in [100.0, 300.0, 1000.0]:
    q = (ell + 0.5) * th[i]
    dir_.append(2 * np.pi * th[i] ** 2 * amp[i] * P0 * np.trapz(Xg * Sg * j0(q * Xg), Xg))
tab = v2.yell(np.array([P0]), np.array([xc]), np.array([beta]), th[i:i + 1], amp[i:i + 1])[0][0].reshape(v2.ELLS.size, v2.FR.size)[1:4, 1]
out["nearest halo"] = {"m_msunh": float(mh[i]), "z": float(zz[i]), "theta200_rad": float(th[i]),
                       "kernel/direct at ell 100,300,1000": (tab / np.array(dir_)).tolist()}
print(json.dumps(out, indent=1))
json.dump(out, open(OUT + "v7_nearby_massive_check.json", "w"), indent=1)
