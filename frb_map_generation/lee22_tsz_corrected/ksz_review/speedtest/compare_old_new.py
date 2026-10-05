"""Compare OLD vs NEW kSZ painter outputs of the idark speed test (maps, histograms, counters, timings). 2026-10-02."""
import argparse, json, os, re
import numpy as np, h5py, healpy as hp

VARIANTS = ["b16_phys", "lee22corr_phys", "lee22pipe_phys", "b16_websky", "lee22corr_websky"]
CASES = [("lowz", 1024), ("mid", 4096)]

ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); args = ap.parse_args()
root = args.root
out = {}


def prov(tag):
    d = {}
    with open(os.path.join(root, "runs", tag, "maps", "provenance.txt")) as f:
        for line in f:
            k, _, v = line.rstrip("\n").partition("="); d[k] = v
    return d


def loginfo(tag):
    txt = open(os.path.join(root, "logs", tag + ".log")).read()
    rows = re.findall(r"pixel updates=(\d+)\s+websky-selected=(\d+)\s+exact-lowz=(\d+)\s+elapsed=([\d.]+) s", txt)
    done = re.findall(r"done in ([\d.]+) s", txt)
    t = open(os.path.join(root, "logs", tag + "_time.txt")).read()
    wall = re.search(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): (\S+)", t).group(1)
    secs = sum(float(x) * 60 ** i for i, x in enumerate(reversed(wall.split(":"))))
    rss = int(re.search(r"Maximum resident set size \(kbytes\): (\d+)", t).group(1))
    cpu = re.search(r"Percent of CPU this job got: (\S+)", t).group(1)
    ut = float(re.search(r"User time \(seconds\): (\S+)", t).group(1))
    st = float(re.search(r"System time \(seconds\): (\S+)", t).group(1))
    return dict(pixel_updates=int(rows[-1][0]), websky_selected=int(rows[-1][1]), exact_lowz=int(rows[-1][2]),
                paint_loop_s_last_chunk_line=float(rows[-1][3]), paint_plus_save_s=float(done[-1]) if done else None,
                wall_s=secs, max_rss_gb=rss / 1024 ** 2, cpu_percent=cpu, user_s=ut, sys_s=st)


for case, nside in CASES:
    o, n = "old_" + case, "new_" + case
    if not all(os.path.exists(os.path.join(root, "runs", t, "maps", "provenance.txt")) for t in (o, n)):
        print(f"!! {case}: missing provenance for {o} or {n}; skipped"); continue
    res = {"nside": nside}
    po, pn = prov(o), prov(n)
    lo, ln = loginfo(o), loginfo(n)
    res["timing"] = {"old": lo, "new": ln,
                     "speedup_paint_loop_provenance": float(po["elapsed_seconds"]) / float(pn["elapsed_seconds"]),
                     "old_paint_loop_provenance_s": float(po["elapsed_seconds"]), "new_paint_loop_provenance_s": float(pn["elapsed_seconds"]),
                     "speedup_total_wall": lo["wall_s"] / ln["wall_s"]}
    res["counters"] = {k: {"old": po.get(k), "new": pn.get(k), "equal": po.get(k) == pn.get(k)}
                       for k in ["halos_read", "websky_selected_halos", "exact_lowz_halos", "vlos_rms_kms", "theta_min_rad", "julia_version", "threads"]}
    res["counters"]["pixel_updates_log"] = {"old": lo["pixel_updates"], "new": ln["pixel_updates"], "equal": lo["pixel_updates"] == ln["pixel_updates"]}
    res["counters"]["websky_histogram_halos_new"] = pn.get("websky_histogram_halos")
    # histograms
    hres = {}
    with h5py.File(os.path.join(root, "runs", o, "maps", "halo_mass_redshift_histogram.h5"), "r") as fo, \
         h5py.File(os.path.join(root, "runs", n, "maps", "halo_mass_redshift_histogram.h5"), "r") as fn:
        assert sorted(fo.keys()) == sorted(fn.keys()), (list(fo.keys()), list(fn.keys()))
        for k in fo.keys():
            a, b = fo[k][()], fn[k][()]
            hres[k] = {"identical": bool(np.array_equal(a, b)), "max_abs_diff": float(np.max(np.abs(a - b))) if a.size else 0.0,
                       "total_old": float(np.sum(a)), "total_new": float(np.sum(b))}
        hres["attrs_equal"] = {k: str(fo.attrs[k]) == str(fn.attrs[k]) for k in fo.attrs}
    res["histogram"] = hres
    # maps
    mres = {}
    for v in VARIANTS:
        fname = f"halfdome_ksz_{v}_nside{nside}.fits"
        a = hp.read_map(os.path.join(root, "runs", o, "maps", fname), dtype=np.float64)
        b = hp.read_map(os.path.join(root, "runs", n, "maps", fname), dtype=np.float64)
        d = b - a
        nzo, nzn = a != 0, b != 0
        rel = np.abs(d[nzo]) / np.abs(a[nzo])
        mres[v] = {"max_abs_old": float(np.max(np.abs(a))), "max_abs_diff_over_max_abs_old": float(np.max(np.abs(d)) / np.max(np.abs(a))),
                   "rms_diff_over_rms_old": float(np.sqrt(np.mean(d ** 2)) / np.sqrt(np.mean(a ** 2))),
                   "max_pixel_rel_diff_nonzero": float(rel.max()) if rel.size else 0.0,
                   "p99999_pixel_rel_diff": float(np.quantile(rel, 0.99999)) if rel.size else 0.0,
                   "sum_new_over_sum_old_minus1": float(np.sum(b) / np.sum(a) - 1),
                   "pixels_bit_identical_frac": float(np.mean(a == b)), "pixels_differing": int(np.sum(a != b)),
                   "nonzero_old": int(nzo.sum()), "nonzero_new": int(nzn.sum()), "support_mismatch": int(np.sum(nzo != nzn)),
                   "rms_old": float(np.sqrt(np.mean(a ** 2))), "rms_new": float(np.sqrt(np.mean(b ** 2)))}
        del a, b, d, nzo, nzn, rel
    res["maps"] = mres
    out[case] = res

json.dump(out, open(os.path.join(root, "compare_old_new.json"), "w"), indent=1)
for case, res in out.items():
    t = res["timing"]
    print(f"== {case} NSIDE {res['nside']}")
    print(f"  paint loop (provenance elapsed_seconds): old {t['old_paint_loop_provenance_s']:.1f} s  new {t['new_paint_loop_provenance_s']:.1f} s  "
          f"speed-up {t['speedup_paint_loop_provenance']:.2f}x")
    print(f"  total wall (/usr/bin/time): old {t['old']['wall_s']:.1f} s  new {t['new']['wall_s']:.1f} s  speed-up {t['speedup_total_wall']:.2f}x; "
          f"max RSS old {t['old']['max_rss_gb']:.2f} GB new {t['new']['max_rss_gb']:.2f} GB; CPU% old {t['old']['cpu_percent']} new {t['new']['cpu_percent']}")
    for k, c in res["counters"].items():
        print(f"  {k}: {c}")
    for k, h in res["histogram"].items():
        print(f"  hist {k}: {h}")
    for v, m in res["maps"].items():
        print(f"  map {v}: max|d|/max|old|={m['max_abs_diff_over_max_abs_old']:.3e} rms(d)/rms(old)={m['rms_diff_over_rms_old']:.3e} "
              f"max pix rel={m['max_pixel_rel_diff_nonzero']:.3e} sum ratio-1={m['sum_new_over_sum_old_minus1']:.3e} "
              f"differing={m['pixels_differing']} nonzero old/new={m['nonzero_old']}/{m['nonzero_new']} support mismatch={m['support_mismatch']}")
