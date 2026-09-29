"""Independent Tinker08 one-halo C_ell^yy (my own profile transform and integration) on CLASS-SZ's own tabulated
dn/dlnM (b12_paint_all.npz: tab_z, tab_lnm_msunh, tab_dndlnm_h3mpc3, tab_da_mpc, tab_H_over_c_invmpc), to test
(a) CLASS-SZ's default vs high-precision 1h at ell < 300, (b) the Tinker08 halo counts (1436 above 10^14.8 Msun/h,
1.021e8 in total), (c) Lee22/B12 and the above-clip share in the Tinker08 model.
Profiles: v2_catalogue_poisson (B12 x 0.5176, Lee22 no-c as painted, 4 R200c projected disc, infinite LOS, flat sky,
k = (ell+1/2)/D_A). CLASS-SZ uses a 3-D sphere truncated at 4 R200c, so mine/CLASS-SZ should be the aperture/sphere
factor (~1.01 at ell 100, 1.00 at ell >= 1000) if CLASS-SZ is converged.
Also: per-mass-bin CLASS-SZ decomposition from the workstream's mbin runs (medprec)."""
import json, sys, time
import numpy as np
sys.path.insert(0, "/home/cbllover/HalfDome/frb_map_generation/lee22_tsz_verification/verify_B_class_sz")
import v2_catalogue_poisson as v2

OUT = "/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_tsz_verification_20260929/verify_B_class_sz/"
CS = "/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_tsz_verification_20260929/B_class_sz/class_sz/"
d = np.load(CS + "b12_paint_all.npz", allow_pickle=True)
z = d["tab_z"]; lnm = d["tab_lnm_msunh"]; dn = d["tab_dndlnm_h3mpc3"] * v2.H ** 3          # Mpc^-3 per ln m
da = d["tab_da_mpc"]; hc = d["tab_H_over_c_invmpc"]
chi = da * (1 + z)
dVdzdO = chi ** 2 / hc                                                                  # Mpc^3 / sr
print("D_A check (mine/CLASS-SZ):", (v2.D_A_cm(z) / v2.MPC / da)[[0, 50, 100, 200]])
mh = np.exp(lnm)
Z, MH = np.meshgrid(z, mh, indexing="ij")
Zf, MHf = Z.ravel(), MH.ravel(); Mp = MHf / v2.H
r200 = v2.R200(Mp, Zf); th = r200 / v2.D_A_cm(Zf); amp = v2.SIGMA_T / v2.MEC2 * v2.P200(Mp, Zf) * r200
t0 = time.time()
Y = {}
for name, pars in [("b12", v2.b12_params(Mp, Zf)), ("lee22", v2.lee_params(MHf, Zf))]:
    yl = np.empty((Mp.size, v2.ELL_EVAL.size))
    for s in range(0, Mp.size, 200):
        sl = slice(s, s + 200)
        yl[sl] = v2.yell(pars[0][sl], pars[1][sl], pars[2][sl], th[sl], amp[sl])[0]
    Y[name] = yl.reshape(z.size, mh.size, -1)
    print(name, "%.1f s" % (time.time() - t0))

def onehalo(name, mlo=0, mhi=np.inf, zlo=0, zhi=np.inf):
    w = ((mh >= mlo) & (mh < mhi)).astype(float)
    # exact mass-bin edges via fractional weights at the boundary node spacing is not attempted; nodes inside the bin
    integ_m = np.trapz(dn[:, :, None] * Y[name] ** 2 * w[None, :, None], lnm, axis=1)     # (z, ell)
    wz = ((z >= zlo) & (z < zhi)).astype(float)
    c = np.trapz(dVdzdO[:, None] * integ_m * wz[:, None], z, axis=0)
    return (c.reshape(v2.ELLS.size, v2.FR.size) * v2.FW[None, :]).sum(1)

def cs_band(name, term):
    dd = np.load(CS + name + ".npz", allow_pickle=True)
    out = []
    for l in v2.ELLS:
        li = np.arange(int(np.ceil(0.9 * l)), int(np.floor(1.1 * l)) + 1)
        out.append(np.mean(np.exp(np.interp(np.log(li), np.log(dd["ell"]), np.log(dd[term])))))
    return np.array(out)

res = {"ells": v2.ELLS.tolist(), "note": __doc__}
b1 = onehalo("b12"); l1 = onehalo("lee22")
clip = 10 ** 14.8
res["my T08 1h aperture / CLASS-SZ hiprec 1h (B12)"] = (b1 / cs_band("b12_paint_all_hiprec", "cl_1h")).tolist()
res["my T08 1h aperture / CLASS-SZ medprec 1h (B12)"] = (b1 / cs_band("b12_paint_all_medprec", "cl_1h")).tolist()
res["my T08 1h aperture / CLASS-SZ default 1h (B12)"] = (b1 / cs_band("b12_paint_all", "cl_1h")).tolist()
cs_l1 = cs_band("lee22_paint_low_medprec", "cl_1h") + cs_band("lee22_paint_high_medprec", "cl_1h")
cs_l1d = cs_band("lee22_paint_low", "cl_1h") + cs_band("lee22_paint_high", "cl_1h")
res["my T08 1h aperture / CLASS-SZ medprec 1h (Lee22 low+high)"] = (l1 / cs_l1).tolist()
res["my T08 1h aperture / CLASS-SZ default 1h (Lee22 low+high)"] = (l1 / cs_l1d).tolist()
b2 = cs_band("b12_paint_all_hiprec", "cl_2h"); l2 = cs_band("lee22_paint_low_medprec", "cl_2h") + cs_band("lee22_paint_high_medprec", "cl_2h")
res["Lee22/B12, my T08 1h + CLASS-SZ 2h"] = ((l1 + l2) / (b1 + b2)).tolist()
bh = onehalo("b12", mlo=clip); lh = onehalo("lee22", mlo=clip)
res["above-clip share of Lee22/B12-1, my T08 (1h)"] = ((lh - bh) / (b1 + b2)).tolist()
bhz = onehalo("b12", mlo=clip, zhi=0.1); lhz = onehalo("lee22", mlo=clip, zhi=0.1)
res["above-clip & z<0.1 share, my T08 (1h)"] = ((lhz - bhz) / (b1 + b2)).tolist()
# halo counts (full sky)
Nz = 4 * np.pi * np.trapz(dVdzdO[:, None] * dn, z, axis=0)            # per ln m
res["T08 N total (my integration of CLASS-SZ dn/dlnM)"] = float(np.trapz(Nz, lnm))
sel = lnm >= np.log(clip)
# add the partial interval between clip and the next node by interpolation
lg = np.concatenate([[np.log(clip)], lnm[sel]]); Ng = np.interp(lg, lnm, Nz)
res["T08 N above 10^14.8 Msun/h"] = float(np.trapz(Ng, lg))
# CLASS-SZ mass-bin runs (medprec): (L22 - B12)(bin) / B12 total, 1h + 2h per bin
btot = cs_band("b12_paint_all_hiprec", "cl_1h") + cs_band("b12_paint_all_hiprec", "cl_2h")
mb = {}
for i in range(6):
    Lb = cs_band(f"lee22_paint_mbin{i}_medprec", "cl_1h") + cs_band(f"lee22_paint_mbin{i}_medprec", "cl_2h")
    Bb = cs_band(f"b12_paint_mbin{i}_medprec", "cl_1h") + cs_band(f"b12_paint_mbin{i}_medprec", "cl_2h")
    mb[f"mbin{i}"] = ((Lb - Bb) / btot).tolist()
    mb[f"mbin{i} L22/B12 1h"] = (cs_band(f"lee22_paint_mbin{i}_medprec", "cl_1h") / cs_band(f"b12_paint_mbin{i}_medprec", "cl_1h")).tolist()
res["CLASS-SZ mbin runs (medprec), (L22-B12)(bin)/B12 total"] = mb
json.dump(res, open(OUT + "v5_t08_onehalo.json", "w"), indent=1)
for k, v in res.items():
    if isinstance(v, list):
        print(f"{k:62s}" + "".join(f"{x:9.3f}" for x in v))
    elif isinstance(v, float):
        print(k, v)
for k, v in mb.items():
    print(f"   {k:30s}" + "".join(f"{x:9.3f}" for x in v))
