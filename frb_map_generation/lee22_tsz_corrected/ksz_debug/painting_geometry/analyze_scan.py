#!/usr/bin/env python3
"""Truncation x compensation x cut scan vs the WebSky halo kSZ (reads onehalo_scan_amplitudes.npz)."""
import numpy as np

OUT = "/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_ksz_debug_20261002/painting_geometry"
KSZ = "/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_ksz_20261002"
T2 = 2.7255e6**2
d = np.load(OUT + "/onehalo_scan_amplitudes.npz")
ells, XS, sph = d["ells"], d["XS"], list(d["sph"])
a_h, a_s, Ne = d["a_halo"], d["a_sph"], d["Ne"]
w = d["sum_v2"] / (4 * np.pi)
lm, z, th = d["logm"], d["z"], d["theta200"]
pre = ells * (ells + 1) / (2 * np.pi) * T2
wx, wy = np.loadtxt(KSZ + "/websky_halo_stein2020_fig6_digitized_binned.csv", delimiter=",").T
web = lambda L: np.exp(np.interp(np.log(L), np.log(wx), np.log(wy)))
S = np.load(KSZ + "/spectra/ksz_spectra.npz")
el_m = S["ell"].astype(float)
dmap = el_m * (el_m + 1) * S["cl_b16"] / (2 * np.pi) * T2
LS = np.array([300, 500, 1000, 2000, 3000, 5000.0])
iX4 = int(np.argmin(abs(XS - 4)))
sph4 = (w[:, None] * a_h[:, iX4] ** 2).sum(0) * pre
at = lambda arr, L: np.exp(np.interp(np.log(L), np.log(ells), np.log(np.maximum(arr, 1e-300))))
m2h = np.array([dmap[(el_m >= 0.93 * L) & (el_m < 1.07 * L)].mean() / at(sph4, L) for L in LS])
m2h_i = lambda L: np.exp(np.interp(np.log(L), np.log(LS), np.log(m2h)))
fit_l = np.geomspace(520, 5000, 40)
LW = (600, 1000, 2000, 3000, 5000)
NE_RATIO2 = (0.88 / (0.9 * 0.88 * 1.13636 / 1.1369)) ** 2   # WebSky eq. 3.15 n_e (0.88 rho/m_p) vs XGPaint (0.9/1.1369 rho/m_p), squared
masks = {"all": np.ones(len(lm), bool), "webskycut": (lm >= 13.0) & (th > 0.5 / 60 * np.pi / 180)}
rows = []
for mname, mk in masks.items():
    for iX, X in enumerate(XS):
        for comp in ["none"] + sph:
            if comp == "none":
                amp = a_h[mk, iX]
            else:
                amp = a_h[mk, iX] - a_s[mk, sph.index(comp)]
            dl = (w[mk, None] * amp**2).sum(0) * pre
            for nef, nelab in ((1.0, "xgpaint_ne"), (NE_RATIO2, "websky_ne")):
                r = np.array([at(dl, L) * nef * m2h_i(L) / web(L) for L in fit_l])
                lr = np.log10(r)
                rows.append((mname, X, comp, nelab, np.sqrt(np.mean(lr**2)), np.std(lr), 10 ** np.mean(lr),
                             [at(dl, L) * nef * m2h_i(L) / web(L) for L in LW], [at(dl, L) * nef for L in LS]))
lines = []
P = lambda s: (print(s), lines.append(s))
P("NE_RATIO2 (WebSky 0.88 rho/m_p vs XGPaint 0.9 x 0.88) = %.4f" % NE_RATIO2)
P("map/one-halo factors used (ell %s): %s" % (LS.astype(int), np.round(m2h, 3)))
P("columns: mask X comp ne | rms log10(ratio) 520-5000 | shape-only std log10 | mean ratio | ratio at ell 600/1000/2000/3000/5000 | D_ell(1h) at 300/500/1000/2000/3000/5000")
for mname in masks:
    for nelab in ("xgpaint_ne", "websky_ne"):
        P("=== mask %s, %s: no compensation, truncation scan" % (mname, nelab))
        for r in rows:
            if r[0] == mname and r[3] == nelab and r[2] == "none":
                P("  X=%.2f none     | %.3f | %.3f | %.2f | %s | %s" % (r[1], r[4], r[5], r[6], " ".join("%.2f" % v for v in r[7]), " ".join("%.4f" % v for v in r[8])))
        P("=== mask %s, %s: WebSky compensation (M200m, 3 rho_m), truncation scan" % (mname, nelab))
        for r in rows:
            if r[0] == mname and r[3] == nelab and r[2] == "M200m_3":
                P("  X=%.2f M200m_3  | %.3f | %.3f | %.2f | %s | %s" % (r[1], r[4], r[5], r[6], " ".join("%.2f" % v for v in r[7]), " ".join("%.4f" % v for v in r[8])))
        best = sorted([r for r in rows if r[0] == mname and r[3] == nelab], key=lambda r: r[4])[:6]
        P("  best 6 by rms: " + "; ".join("X=%.2f %s rms %.3f (ratios %s)" % (r[1], r[2], r[4], "/".join("%.2f" % v for v in r[7])) for r in best))
# electron budget vs X (count-weighted, z<1)
k = z < 1.0
P("")
P("N_e(<X R200c) / (f_b M200c), count-weighted z<1, all masses: " + " ".join("X=%.2f:%.2f" % (X, np.average(Ne[k, i] / (0.049 / 0.31 * 10**lm[k] * 1.98847e30 * 0.9 / (9.1093837015e-31 + 0.8636363636 * 1.67262192369e-27 + 0.0681818 * 4 * 1.67262192369e-27)), weights=d["count"][k])) for i, X in enumerate(XS)))
open(OUT + "/scan_tables.txt", "w").write("\n".join(lines) + "\n")
