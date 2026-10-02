#!/usr/bin/env python3
"""Tables for the painting/geometry investigation: one-halo kSZ variants vs WebSky halo (Stein+20 Fig. 6), cuts, and the
(M, z) cells that drive the HalfDome excess. Reads onehalo_variants_percell.npz (onehalo_variants.py)."""
import numpy as np
import h5py

OUT = "/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_ksz_debug_20261002/painting_geometry"
KSZ = "/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_ksz_20261002"
T2 = 2.7255e6**2
d = np.load(OUT + "/onehalo_variants_percell.npz")
ells = d["ells"]
meta = d["meta"]               # logM, z, count, sum v2, theta200 [rad], Ne(<4)/(f_b M), Ne(<1)/(f_b M), M200m/M200c
lm, z, cnt, sv2, th = meta[:, 0], meta[:, 1], meta[:, 2], meta[:, 3], meta[:, 4]
pre = ells * (ells + 1) / (2 * np.pi) * T2
wx, wy = np.loadtxt(KSZ + "/websky_halo_stein2020_fig6_digitized_binned.csv", delimiter=",").T
web = lambda L: np.exp(np.interp(np.log(L), np.log(wx), np.log(wy)))

# map/one-halo of the production b16 map (two-halo + discreteness), no pixel window (point-sampled map)
S = np.load(KSZ + "/spectra/ksz_spectra.npz")
el_m = S["ell"].astype(float)
dmap = el_m * (el_m + 1) * S["cl_b16"] / (2 * np.pi) * T2
def dmap_at(L):
    k = (el_m >= 0.93 * L) & (el_m < 1.07 * L)
    return dmap[k].mean()
LS = (300, 500, 1000, 2000, 3000, 5000)
sph4 = d["sph4"].sum(0) * pre
at = lambda arr, L: float(np.exp(np.interp(np.log(L), np.log(ells), np.log(np.maximum(arr, 1e-300)))))
m2h = {L: dmap_at(L) / at(sph4, L) for L in LS}
m2h_interp = lambda L: float(np.exp(np.interp(np.log(L), np.log(LS), np.log([m2h[x] for x in LS]))))

lines = []
P = lambda s: (print(s), lines.append(s))
P("Production B16 map D_ell [uK^2] (anafast, no pixwin): " + "  ".join("l%d=%.4f" % (L, dmap_at(L)) for L in LS))
P("Python one-halo sph4:                              " + "  ".join("l%d=%.4f" % (L, at(sph4, L)) for L in LS))
P("map / one-halo:                                    " + "  ".join("l%d=%.3f" % (L, m2h[L]) for L in LS))
P("WebSky halo (digitized):                           " + "  ".join("l%d=%.4f" % (L, web(L)) for L in LS[1:]))
P("")

masks = {
    "all halos": np.ones(len(lm), bool),
    "M200c>1e13": lm >= 13.0,
    "theta200c>0.5'": th > 0.5 / 60 * np.pi / 180,
    "M200c>1e13 & theta200c>0.5' (WebSky stated cut)": (lm >= 13.0) & (th > 0.5 / 60 * np.pi / 180),
}
variants = [("sph1", "sphere 1 R200c"), ("sph2", "sphere 2 R200c"), ("sph3", "sphere 3 R200c"), ("sph4", "sphere 4 R200c (production)"),
            ("proj4", "projected 4 R200c (XGPaint default)"),
            ("comp3_m200m", "sphere 4 - uniform sphere (M200m, 3 rho_m)  [WebSky compensation]"),
            ("comp4_m200m", "sphere 4 - uniform sphere (M200m, 4 rho_m)"),
            ("comp3_m200c", "sphere 4 - uniform sphere (M200c, 3 rho_m)"),
            ("comp3_m200m_ne088", "as comp3_m200m with WebSky n_e = 0.88 rho/m_p (x1.111 tau)")]
LW = (600, 1000, 2000, 3000, 5000)
fit_l = np.geomspace(520, 5000, 30)
P("one-halo D_ell [uK^2] (x map/1h factor in brackets ratio)   |  ratio to WebSky at ell=600/1000/2000/3000/5000 (one-halo x map/1h)  | rms log10 ratio 520-5000")
for mname, mk in masks.items():
    P("== mask: %s  (halos %d of %d)" % (mname, cnt[mk].sum(), cnt.sum()))
    for key, lab in variants:
        tot = d[key][mk].sum(0) * pre
        vals = " ".join("%.4f" % at(tot, L) for L in LS)
        corr = lambda L: at(tot, L) * m2h_interp(L)
        rat = " ".join("%.2f" % (corr(L) / web(L)) for L in LW)
        rms = np.sqrt(np.mean(np.log10([corr(L) / web(L) for L in fit_l]) ** 2))
        P("  %-62s %s | %s | %.3f" % (lab, vals, rat, rms))
P("")

# electron budget of the truncated B16 profile
for lo, hi in ((12.8, 13.25), (13.25, 13.75), (13.75, 14.25), (14.25, 14.75), (14.75, 15.7)):
    for zl, zh in ((0, 0.5), (0.5, 1.5), (1.5, 4)):
        k = (lm >= lo) & (lm < hi) & (z >= zl) & (z < zh)
        if cnt[k].sum() == 0:
            continue
        wv = cnt[k]
        P("N_e(<4R200c)/(f_b M200c) logM %.2f-%.2f z %.1f-%.1f: %.3f   N_e(<R200c)/(f_b M): %.3f   M200m/M200c: %.3f" %
          (lo, hi, zl, zh, np.average(meta[k, 5], weights=wv), np.average(meta[k, 6], weights=wv), np.average(meta[k, 7], weights=wv)))
P("(N_e normalised to f_b M x XGPaint 0.9/1.1365 m_p electrons per kg)")
P("")

# (4) which (M, z) cells drive the power and the excess
mb = [12.8, 13.0, 13.5, 14.0, 14.5, 15.0, 15.7]
zb = [0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
for L in (1000, 3000, 5000):
    il = np.argmin(abs(ells - L))
    base = d["sph4"][:, il]
    comp = d["comp3_m200m"][:, il]
    P("ell=%d: fraction of sph4 one-halo power per (log M200c rows, z cols)   z bins %s" % (ells[il], zb))
    for i in range(len(mb) - 1):
        row = []
        for j in range(len(zb) - 1):
            k = (lm >= mb[i]) & (lm < mb[i + 1]) & (z >= zb[j]) & (z < zb[j + 1])
            row.append(base[k].sum() / base.sum())
        P("   logM %.1f-%.1f: " % (mb[i], mb[i + 1]) + " ".join("%.3f" % r for r in row))
    P("ell=%d: fraction of the excess (sph4 - compensated) per cell" % ells[il])
    exc = base - comp
    for i in range(len(mb) - 1):
        row = []
        for j in range(len(zb) - 1):
            k = (lm >= mb[i]) & (lm < mb[i + 1]) & (z >= zb[j]) & (z < zb[j + 1])
            row.append(exc[k].sum() / exc.sum())
        P("   logM %.1f-%.1f: " % (mb[i], mb[i + 1]) + " ".join("%.3f" % r for r in row))
    # median angular size of the power
    o = np.argsort(th)
    cs = np.cumsum(base[o]) / base.sum()
    P("ell=%d: power-weighted theta200c quartiles [arcmin]: %.2f %.2f %.2f; power from theta200c < 0.5': %.3f; from z<0.5: %.3f; from M>1e14.5: %.3f" % (
        ells[il], *(th[o][np.searchsorted(cs, q)] * 180 / np.pi * 60 for q in (0.25, 0.5, 0.75)),
        base[th < 0.5 / 60 * np.pi / 180].sum() / base.sum(), base[z < 0.5].sum() / base.sum(), base[lm >= 14.5].sum() / base.sum()))
    P("")
open(OUT + "/analysis_tables.txt", "w").write("\n".join(lines) + "\n")
