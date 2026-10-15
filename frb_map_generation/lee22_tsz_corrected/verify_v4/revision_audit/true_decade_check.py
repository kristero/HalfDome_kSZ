"""Independent check of the v4 true-decade FRB numbers (no tng_windows.py, no plot scripts).
TNG arrays read directly with np.load; HalfDome values from the run summary CSVs (exact per-ray means) and the HDF5 histograms.
Prints summary statistics only; no catalogue data are written."""
import csv, json
from pathlib import Path
import numpy as np, h5py

R = Path("/home/cbllover/HalfDome")
T = R / "frb_catalog_comparison_outputs/haloDM-20260819T104507Z-1-001/haloDM"
O = R / "frb_map_generation/outputs"
OUT = O / "lee22_verify_v4_20261001/revision_audit"
zt = np.load(T / "redshifts_DMhalo.npy")
A = lambda name: np.load(T / f"DMhalo_r200_mass{name}.npy", mmap_mode="r")
def tng(name, zs):
    a = A(name); i = int(np.flatnonzero(np.isclose(zt, zs))[0])
    return np.asarray(a[i] if a.shape[0] == len(zt) else a[:, i], float)

def rundir(tag, zs):
    s = ("%.1f" % zs).replace(".", "p")
    name = f"zsrc{s}_nside4096_nrays120000_allhalos_sphere1p0_m200c_{tag}_seed42"
    if zs > 1.0: return O / "lee22_pdf_zseries_20261001/rays" / name
    if tag == "lee22_phys": return O / "lee22_corrected_20260930/rays" / name
    return O / "lee22_tngcal_20260929/rays" / name

def summ(tag, zs, label):
    with open(rundir(tag, zs) / "halfdome_uniform_fixedz_foreground_mass_histograms_summary.csv") as f:
        for r in csv.DictReader(f):
            if r["label"] == label:
                h = 1 - float(r["zero_fraction"]); return dict(hit=h, mph=float(r["dm_mean_pc_cm3"]) / h, mean=float(r["dm_mean_pc_cm3"]))
    raise KeyError(label)

def hist(tag, zs, label):
    with h5py.File(rundir(tag, zs) / "halfdome_uniform_fixedz_foreground_mass_histograms.h5", "r") as h:
        labs = [v.decode() if isinstance(v, bytes) else str(v) for v in h["window_label"][:]]
        e = np.asarray(h["pdf_bin_edges_pc_cm3"][:], float); c = np.asarray(h["pdf_count"][:])
    if c.shape[0] != len(labs): c = c.T
    return e, c[labs.index(label)].astype(float)

res = {}
ZS = (0.1, 0.5, 1.0, 1.5, 2.0, 3.0)
for zs in ZS:
    d10_12, d10_13, d10_14, d10_15, d13_14, d12_14 = (tng(n, zs) for n in ("10to12", "10to13", "10to14", "10to15", "13to14", "12to14"))
    ident = float(np.max(np.abs(d10_14 - d10_12 - d13_14)))
    dec = d10_13 - d10_12; dec[dec < 1e-6] = 0
    dec14 = d10_14 - d10_13; dec14[dec14 < 1e-6] = 0
    dec15 = d10_15 - d10_14; dec15[dec15 < 1e-6] = 0
    r = dict(identity_resid=ident, neg_min=float((d10_13 - d10_12).min()),
             tng_dec_hit=float((dec > 0).mean()), tng_dec_mph=float(dec[dec > 0].mean()),
             tng_arr_hit=float((d13_14 > 0).mean()), tng_arr_mph=float(d13_14[d13_14 > 0].mean()),
             tng_1415_hit=float((dec14 > 0).mean()), tng_1516_hit=float((dec15 > 0).mean()))
    if zs == 1.0:
        t12 = d12_14 - dec14; t12[t12 < 1e-6] = 0
        r["tng_12_14_true_hit"] = float((t12 > 0).mean()); r["tng_12to14_array_hit"] = float((d12_14 > 0).mean())
        r["hd_phys_12_14_hit"] = summ("lee22_phys", zs, "m1e12_to_1e14")["hit"]
        r["hd_phys_12_16_hit"] = summ("lee22_phys", zs, "m1e12_to_1e16")["hit"] if True else None
    for tag in ("lee22_phys", "lee22_xgpnative", "b16") + (("lee22_tngcal",) if zs <= 1 else ()):
        s = summ(tag, zs, "m1e13_to_1e14")
        r[tag] = dict(hit=s["hit"], mph=s["mph"], ratio_dec=s["mph"] / r["tng_dec_mph"], ratio_arr=s["mph"] / r["tng_arr_mph"])
    # PDF comparison corrected vs TNG decade, low-DM and high-DM bins
    e, c = hist("lee22_phys", zs, "m1e13_to_1e14")
    tc, _ = np.histogram(dec[dec > 0], bins=e)
    p = c / (c.sum() * np.diff(e)); q = tc / (tc.sum() * np.diff(e)); cen = np.sqrt(e[1:] * e[:-1])
    def band(lo, hi):
        k = (cen >= lo) & (cen < hi) & (c >= 20) & (tc >= 20)
        if not k.any(): return None
        rr = p[k] / q[k]; return [round(float(rr.min()), 3), round(float(rr.max()), 3)], round(float(p[k].sum() / q[k].sum()), 3)
    r["pdf_ratio_lt10"] = band(0, 10); r["pdf_ratio_1_10"] = band(1, 10); r["pdf_ratio_3_10"] = band(3, 10)
    r["pdf_ratio_10_100"] = band(10, 100); r["pdf_ratio_gt200"] = band(200, 1e5)
    # integrated tail fractions above 300 pc/cm3
    hd_tail = c[cen > 300].sum() / c.sum(); tng_tail = tc[cen > 300].sum() / tc.sum()
    r["tail_frac_gt300"] = (float(hd_tail), float(tng_tail))
    # KS on binned CDFs
    r["KS_binned"] = float(np.max(np.abs(np.cumsum(c) / c.sum() - np.cumsum(tc) / tc.sum())))
    res[zs] = r
    print(zs, json.dumps(r, default=str))
json.dump(res, open(OUT / "true_decade_check.json", "w"), indent=1, default=str)
