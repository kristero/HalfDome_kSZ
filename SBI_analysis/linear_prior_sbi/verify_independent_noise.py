#!/usr/bin/env python3
"""Verify that the 8192-row linear-prior run used independent SO noise per row and per split.

Checks (all on the frozen run directory):
1. design seeds == SHA256 recipe for the stored noise row IDs; all 16384 seeds distinct; no observation seed.
2. every packed row's provenance: seeds match the design row, seed1 != seed2, noise pixel hashes distinct
   across all rows and splits (each row received its own two noise maps), mask hash constant.
3. dataset statistics: r = noisy_cross - clean per bin; with independent noise the row mean of r is
   consistent with zero (z-scores O(1)) and residual vectors of different rows are uncorrelated.
   Fixed noise would give a common n1*n2 component: |z| >> 1 at high ell and cross-row correlation ~ 1.
"""
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np, h5py

def split_seeds(row_id, master_seed):
    out = []
    for split in (1, 2):
        message = "halfdome-so-v1|%d|%d|%d" % (master_seed, row_id, split)
        out.append(int.from_bytes(hashlib.sha256(message.encode("ascii")).digest()[:8], "big") >> 1)
    return out

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", type=Path, required=True); ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(); root = a.root; a.out.mkdir(parents=True, exist_ok=True)
    run = json.loads((root / "run_config.json").read_text()); master = run["noise_master_seed"]
    ids = np.load(root / "dataset/noise_seed_row_ids.npy"); seeds = np.load(root / "dataset/noise_split_seeds.npy")
    expected = np.array([split_seeds(int(i), master) for i in ids], dtype=np.int64)
    report = {"rows": int(len(ids)), "master_seed": master}
    report["design_seeds_match_sha256_recipe"] = bool(np.array_equal(seeds, expected))
    report["row_ids_unique"] = bool(len(np.unique(ids)) == len(ids))
    report["all_16384_seeds_unique"] = bool(len(np.unique(seeds)) == seeds.size)
    report["observation_seeds_22446_22447_absent"] = bool(not np.isin(seeds, [22446, 22447]).any())
    # per-row provenance from packed chunks
    n = run["count"]; count = (n + run["chunk_size"] - 1) // run["chunk_size"]
    sha1, sha2, masks, mism, reused, seed_ids_ok = [], [], set(), 0, 0, 0
    keys_seen = None
    for c in range(count):
        with h5py.File(root / "chunks" / ("chunk_%05d.h5" % c), "r") as h:
            for key, row in h.items():
                idx = int(key); st = json.loads(row.attrs["status_json"])
                if keys_seen is None: keys_seen = sorted(st.keys())
                noise = st["noise"]
                if "reuse_provenance_json" in row.attrs: reused += 1
                if list(noise["split_seeds"]) != [int(seeds[idx, 0]), int(seeds[idx, 1])]: mism += 1
                if int(noise["seed_row_id"]) == int(ids[idx]): seed_ids_ok += 1
                sha1.append(noise["noise1_pixel_sha256"]); sha2.append(noise["noise2_pixel_sha256"]); masks.add(noise["mask_pixel_sha256"])
    allsha = sha1 + sha2
    report.update(rows_read=len(sha1), status_keys=keys_seen, rows_with_reuse_provenance=reused,
                  rows_with_seed_mismatch=mism, rows_with_seed_row_id_match=seed_ids_ok,
                  distinct_noise_pixel_hashes=len(set(allsha)), expected_distinct=2 * len(sha1),
                  split1_equals_split2_rows=int(sum(a == b for a, b in zip(sha1, sha2))), distinct_mask_hashes=len(masks))
    # dataset statistics
    clean = np.load(root / "dataset/masked_clean_dl40.npy"); noisy = np.load(root / "dataset/masked_noisy_cross_dl40.npy")
    r = noisy - clean; N = len(r)
    mean_z = r.mean(0) / (r.std(0, ddof=1) / np.sqrt(N))
    # cross-row correlation of residual vectors (high-ell half, noise dominated), random pairs
    rng = np.random.default_rng(0); i = rng.integers(0, N, 20000); j = rng.integers(0, N, 20000); ok = i != j
    hi = slice(20, 40); ri, rj = r[i[ok]][:, hi], r[j[ok]][:, hi]
    ri = (ri - ri.mean(1, keepdims=True)) / ri.std(1, keepdims=True); rj = (rj - rj.mean(1, keepdims=True)) / rj.std(1, keepdims=True)
    corr = (ri * rj).mean(1)
    # what a fixed noise realization would look like: reuse the same n1*n2 for all rows -> correlate r with its median pattern
    common = np.median(r, axis=0)
    report.update(residual_mean_zscore_per_bin=[float(v) for v in mean_z], residual_mean_zscore_max_abs=float(np.abs(mean_z).max()),
                  cross_row_residual_correlation_mean=float(corr.mean()), cross_row_residual_correlation_std=float(corr.std()),
                  cross_row_residual_correlation_expected_std_if_independent=float(1 / np.sqrt(20)),
                  negative_noisy_bin_fraction=float((noisy < 0).mean()))
    passed = (report["design_seeds_match_sha256_recipe"] and report["all_16384_seeds_unique"] and mism == 0
              and report["distinct_noise_pixel_hashes"] == report["expected_distinct"] and report["split1_equals_split2_rows"] == 0
              and report["residual_mean_zscore_max_abs"] < 4 and abs(corr.mean()) < 0.05)
    report["passed"] = bool(passed)
    (a.out / "noise_independence_check.json").write_text(json.dumps(report, indent=1))
    # figure
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    edges = np.load(root / "dataset/bin_edges.npy"); lc = 0.5 * (edges[:-1] + edges[1:])
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    ax[0].axhspan(-2, 2, color="#cde2fb", alpha=0.6, lw=0); ax[0].plot(lc, mean_z, "o-", color="#2a78d6", ms=4)
    ax[0].axhline(0, color="#52514e", lw=0.8); ax[0].set_xlabel(r"multipole $\ell$"); ax[0].set_ylabel(r"mean of (noisy $-$ clean) / SEM"); ax[0].set_title("Row-mean residual per bin (independent noise: |z| ~ 1)", fontsize=10)
    ax[1].hist(corr, bins=60, color="#2a78d6", alpha=0.8); ax[1].axvline(0, color="#52514e", lw=0.8)
    ax[1].set_xlabel(r"correlation of residual vectors, random row pairs ($\ell>4000$ bins)"); ax[1].set_ylabel("pairs"); ax[1].set_title(f"mean {corr.mean():+.4f}, std {corr.std():.3f} (fixed noise would give ~1)", fontsize=10)
    sd = r.std(0, ddof=1); ax[2].plot(lc, 1e12 * sd, "o-", color="#eb6834", ms=4, label="row scatter of noisy $-$ clean")
    ax[2].plot(lc, 1e12 * np.abs(r.mean(0)), "s-", color="#52514e", ms=3, label="|row mean| of noisy $-$ clean")
    ax[2].set_yscale("log"); ax[2].set_xlabel(r"multipole $\ell$"); ax[2].set_ylabel(r"$10^{12}\,D_\ell$"); ax[2].legend(fontsize=9, frameon=False); ax[2].set_title("Noise scatter vs residual mean", fontsize=10)
    for x in ax: x.grid(alpha=0.25)
    fig.suptitle(f"Independent-noise check, {N} rows: seeds match SHA256 recipe = {report['design_seeds_match_sha256_recipe']}, "
                 f"{report['distinct_noise_pixel_hashes']}/{report['expected_distinct']} distinct noise maps, PASS = {passed}", fontsize=10.5)
    fig.tight_layout(); fig.savefig(a.out / "noise_independence_check.png", dpi=170); fig.savefig(a.out / "noise_independence_check.pdf")
    print(json.dumps({k: v for k, v in report.items() if k != "residual_mean_zscore_per_bin"}, indent=1))

if __name__ == "__main__":
    main()
