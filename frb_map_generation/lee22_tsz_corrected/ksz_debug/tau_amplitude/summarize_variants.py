"""Summary table: HalfDome one-halo D_ell for each tau-convention variant, ratio to base and to the digitized WebSky halo curve,
plus the painted-map ratio to WebSky (map spectrum from spectra/ksz_spectra.npz, +-5% ell averaging)."""
import glob, os
import numpy as np
OUT = "/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_ksz_debug_20261002/tau_amplitude"
BASE = "/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_ksz_20261002"
L = np.array([520, 750, 1000, 1500, 2000, 3000, 4000, 4700])
ws = np.loadtxt(f"{BASE}/websky_halo_stein2020_fig6_digitized_binned.csv", delimiter=",")
wsi = np.interp(np.log(L), np.log(ws[:, 0]), ws[:, 1])
d = np.load(f"{BASE}/spectra/ksz_spectra.npz"); cl = d["cl_b16"]; ell = np.arange(len(cl))
Dm = ell * (ell + 1) * cl / 2 / np.pi * 2.7255e6 ** 2
Dmap = np.array([Dm[int(l * 0.95):int(l * 1.05)].mean() for l in L])
rows = []
base = np.load(f"{OUT}/onehalo_base.npz")
Db = base["ell"] * (base["ell"] + 1) * base["cl"] / 2 / np.pi * 2.7255e6 ** 2
f = lambda e, D: np.exp(np.interp(np.log(L), np.log(e), np.log(D)))
with open(f"{OUT}/variant_summary.csv", "w") as io:
    io.write("# variant, then for ell=" + " ".join(map(str, L)) + ": D_1h[uK^2] | variant/base | variant/WebSky | map-equivalent/WebSky (map x variant/base)\n")
    io.write("map_b16," + ",".join(f"{x:.4f}" for x in Dmap) + ",|," + ",".join(f"{x:.3f}" for x in Dmap / wsi) + "\n")
    print("ell            ", " ".join(f"{l:6d}" for l in L))
    print("WebSky halo    ", " ".join(f"{x:6.3f}" for x in wsi))
    print("map b16        ", " ".join(f"{x:6.3f}" for x in Dmap), " map/WS:", " ".join(f"{x:.2f}" for x in Dmap / wsi))
    for p in sorted(glob.glob(f"{OUT}/onehalo_*.npz")):
        v = os.path.basename(p)[8:-4]
        q = np.load(p); D = q["ell"] * (q["ell"] + 1) * q["cl"] / 2 / np.pi * 2.7255e6 ** 2
        Di = f(q["ell"], D); r = Di / f(base["ell"], Db)
        io.write(v + "," + ",".join(f"{x:.4f}" for x in Di) + ",|," + ",".join(f"{x:.3f}" for x in r) + ",|," +
                 ",".join(f"{x:.3f}" for x in Di / wsi) + ",|," + ",".join(f"{x:.3f}" for x in Dmap * r / wsi) + "\n")
        print(f"{v:22s} 1h/base " + " ".join(f"{x:5.3f}" for x in r) + " | map-equiv/WS " + " ".join(f"{x:5.2f}" for x in Dmap * r / wsi))
