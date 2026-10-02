"""(4) Which (M, z) cells carry the power removed by (a) truncating at 1.75 instead of 4 R200c, (b) the WebSky compensation."""
import numpy as np
OUT = "/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_ksz_debug_20261002/painting_geometry"
d = np.load(OUT + "/onehalo_scan_amplitudes.npz")
ells, XS = d["ells"], d["XS"]; lm, z, th = d["logm"], d["z"], d["theta200"]; w = d["sum_v2"] / (4 * np.pi)
i4 = int(np.argmin(abs(XS - 4))); i175 = int(np.argmin(abs(XS - 1.75)))
p4 = w[:, None] * d["a_halo"][:, i4] ** 2
p175 = w[:, None] * d["a_halo"][:, i175] ** 2
pc = w[:, None] * (d["a_halo"][:, i4] - d["a_sph"][:, 0]) ** 2
mb = [12.8, 13.5, 14.0, 14.5, 15.7]; zb = [0, 0.25, 0.5, 1.0, 1.5, 4.0]
for L in (1000, 3000, 5000):
    il = np.argmin(abs(ells - L))
    print("ell=%d: total ratio sph1.75/sph4 = %.3f, comp/sph4 = %.3f" % (ells[il], p175[:, il].sum() / p4[:, il].sum(), pc[:, il].sum() / p4[:, il].sum()))
    print("   per-bin ratio sph1.75/sph4 [rows log M %s; cols z %s]" % (mb, zb))
    for i in range(len(mb) - 1):
        r = []
        for j in range(len(zb) - 1):
            k = (lm >= mb[i]) & (lm < mb[i + 1]) & (z >= zb[j]) & (z < zb[j + 1])
            r.append(p175[k, il].sum() / p4[k, il].sum() if p4[k, il].sum() > 0 else np.nan)
        print("     ", " ".join("%.2f" % x for x in r))
    print("   per-bin share of the removed power (sph4 - sph1.75)")
    ex = p4[:, il] - p175[:, il]
    for i in range(len(mb) - 1):
        r = []
        for j in range(len(zb) - 1):
            k = (lm >= mb[i]) & (lm < mb[i + 1]) & (z >= zb[j]) & (z < zb[j + 1])
            r.append(ex[k].sum() / ex.sum())
        print("     ", " ".join("%.3f" % x for x in r))
    o = np.argsort(th); 
    for nm, arr in (("sph4", p4[:, il]), ("removed by trunc", ex), ("removed by comp", p4[:, il] - pc[:, il])):
        cs = np.cumsum(arr[o]) / arr.sum()
        print("   %-18s median theta200c of the power: %.2f arcmin" % (nm, th[o][np.searchsorted(cs, 0.5)] * 180 / np.pi * 60))
