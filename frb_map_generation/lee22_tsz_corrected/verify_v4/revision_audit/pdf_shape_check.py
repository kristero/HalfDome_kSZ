"""Low-DM and high-DM behaviour of the corrected density vs TNG's true 1e13-1e14 decade; TNG 1e12-1e14 hit fraction with and
without round-off clipping (as tng_windows.tng_exact does it). Summary numbers only."""
import json
from pathlib import Path
import numpy as np, h5py
T = Path("/home/cbllover/HalfDome/frb_catalog_comparison_outputs/haloDM-20260819T104507Z-1-001/haloDM")
O = Path("/home/cbllover/HalfDome/frb_map_generation/outputs")
zt = np.load(T / "redshifts_DMhalo.npy")
def tng(name, zs):
    a = np.load(T / f"DMhalo_r200_mass{name}.npy", mmap_mode="r"); i = int(np.flatnonzero(np.isclose(zt, zs))[0])
    return np.asarray(a[i] if a.shape[0] == len(zt) else a[:, i], float)
def rays(tag, zs):
    s = ("%.1f" % zs).replace(".", "p"); n = f"zsrc{s}_nside4096_nrays120000_allhalos_sphere1p0_m200c_{tag}_seed42"
    d = (O / "lee22_pdf_zseries_20261001/rays" / n) if zs > 1 else ((O / "lee22_corrected_20260930/rays" / n) if tag == "lee22_phys" else (O / "lee22_tngcal_20260929/rays" / n))
    with h5py.File(d / "halfdome_uniform_fixedz_foreground_mass_histograms.h5", "r") as h:
        labs = [v.decode() if isinstance(v, bytes) else str(v) for v in h["window_label"][:]]
        e = np.asarray(h["pdf_bin_edges_pc_cm3"][:], float); c = np.asarray(h["pdf_count"][:])
    if c.shape[0] != len(labs): c = c.T
    return e, c[labs.index("m1e13_to_1e14")].astype(float)
out = {}
for zs in (0.1, 0.5, 1.0, 1.5, 2.0, 3.0):
    dec = tng("10to13", zs) - tng("10to12", zs); dec = dec[dec > 1e-6]
    r = {}
    for tag in ("lee22_phys", "b16"):
        e, c = rays(tag, zs); cen = np.sqrt(e[1:] * e[:-1]); tc, _ = np.histogram(dec, bins=e)
        f = lambda cnt, sel: cnt[sel].sum() / cnt.sum()
        lo = cen < 10; hi = cen > 200; hi3 = cen > 300
        r[tag] = dict(frac_lt10=(round(f(c, lo), 4), round(f(tc, lo), 4), round(f(c, lo) / f(tc, lo), 3)),
                      frac_gt200=(round(f(c, hi), 4), round(f(tc, hi), 4), round(f(c, hi) / f(tc, hi), 3),
                                  round(f(c, hi) / f(tc, hi) * np.sqrt(1 / max(c[hi].sum(), 1) + 1 / max(tc[hi].sum(), 1)), 3)),
                      frac_gt300=(round(f(c, hi3), 4), round(f(tc, hi3), 4), round(f(c, hi3) / f(tc, hi3), 3)),
                      n_tng_gt200=int(tc[hi].sum()), n_hd_gt200=int(c[hi].sum()))
    out[zs] = r; print(zs, json.dumps(r))
# 1e12-1e14 at z_s = 1
d12 = tng("12to14", 1.0); d14 = tng("10to14", 1.0) - tng("10to13", 1.0)
d14c = np.where(d14 > 1e-9, d14, 0.0)
raw = d12 - d14c
print("TNG [1e12,1e14) hit: helper-style (value != 0) %.3f%%, clipped >1e-6 %.3f%%, negatives %.3f%%, |v|<1e-9 nonzero %.3f%%" % (
    100 * np.mean(raw != 0), 100 * np.mean(raw > 1e-6), 100 * np.mean(raw < 0), 100 * np.mean((raw != 0) & (np.abs(raw) < 1e-9))))
json.dump(out, open("/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_verify_v4_20261001/revision_audit/pdf_shape_check.json", "w"), indent=1)
