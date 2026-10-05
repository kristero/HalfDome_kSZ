"""Compare two painter output dirs: 5 maps + histogram. Usage: compare.py <ref_dir> <test_dir> <nside> [label]"""
import sys, re
import numpy as np, h5py
from astropy.io import fits

VARIANTS = ["b16_phys", "lee22corr_phys", "lee22pipe_phys", "b16_websky", "lee22corr_websky"]
ref, test, nside = sys.argv[1], sys.argv[2], int(sys.argv[3])
label = sys.argv[4] if len(sys.argv) > 4 else f"{test} vs {ref}"


def read_map(d, v):
    with fits.open(f"{d}/maps/halfdome_ksz_{v}_nside{nside}.fits", memmap=False) as h:
        col = h[1].columns[0].name
        return np.asarray(h[1].data[col], dtype=np.float64).ravel()


print(f"=== {label}")
worst = 0.0
for v in VARIANTS:
    a = read_map(ref, v); b = read_map(test, v)
    assert a.size == b.size == 12 * nside**2, (a.size, b.size)
    d = b - a
    maxrel = np.max(np.abs(d)) / np.max(np.abs(a))
    l2 = np.sqrt(np.sum(d**2)) / np.sqrt(np.sum(a**2))
    sr = np.sum(b) / np.sum(a)
    nz_a = np.count_nonzero(a); nz_b = np.count_nonzero(b)
    nz_mismatch = np.count_nonzero((a != 0) != (b != 0))
    nexact = np.count_nonzero(d == 0)
    # pointwise relative deviation on pixels with |a| above 1e-6 max|a|
    m = np.abs(a) > 1e-6 * np.max(np.abs(a))
    prel = np.max(np.abs(d[m]) / np.abs(a[m])) if m.any() else 0.0
    worst = max(worst, maxrel)
    print(f"  {v:18s} max|d|/max|ref|={maxrel:.3e}  relL2={l2:.3e}  sum ratio-1={sr-1:+.3e}  "
          f"max pointwise rel (|ref|>1e-6 max)={prel:.3e}  bit-identical px={nexact}/{a.size}  "
          f"nonzero ref/test={nz_a}/{nz_b} support mismatch={nz_mismatch}")
print(f"  WORST max|d|/max|ref| over variants = {worst:.3e}")

with h5py.File(f"{ref}/maps/halo_mass_redshift_histogram.h5") as fa, h5py.File(f"{test}/maps/halo_mass_redshift_histogram.h5") as fb:
    for k in ["count", "sum_vlos_over_c_squared", "sum_v2_log10m", "sum_v2_z", "websky_count",
              "websky_sum_vlos_over_c_squared", "websky_sum_v2_log10m", "websky_sum_v2_z"]:
        A = fa[k][...].astype(np.float64); B = fb[k][...].astype(np.float64)
        dd = B - A
        print(f"  hist {k:32s} total ref={A.sum():.10g} test={B.sum():.10g}  max|d|={np.max(np.abs(dd)):.3e}  "
              f"max|d|/max|ref|={np.max(np.abs(dd))/max(np.max(np.abs(A)),1e-300):.3e}  bins differing={np.count_nonzero(dd)}")


def counters(d):
    txt = open(f"{d}/run.log").read()
    rows = re.findall(r"rows (\S+)\s+halos=(\d+)\s+pixel updates=(\d+)\s+websky-selected=(\d+)\s+exact-lowz=(\d+)\s+elapsed=([\d.]+) s", txt)
    done = re.findall(r"done in ([\d.]+) s", txt)
    return rows, done


for name, d in (("ref", ref), ("test", test)):
    rows, done = counters(d)
    with h5py.File(f"{d}/maps/halo_mass_redshift_histogram.h5") as f:
        wsn = int(f["websky_count"][...].sum()); hn = int(f["count"][...].sum())
    last = rows[-1]
    print(f"  {name}: halos={last[1]} pixel_updates={last[2]} websky-selected={last[3]} exact-lowz={last[4]} "
          f"rows-elapsed={[r[5] for r in rows]} done_in={done}  hist count total={hn}  websky_count total={wsn}  "
          f"websky_count - websky-selected = {wsn - int(last[3])}")
