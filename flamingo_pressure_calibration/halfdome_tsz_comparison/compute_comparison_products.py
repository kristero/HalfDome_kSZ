"""Spectra, one-point statistics and cutouts of the HalfDome and FLAMINGO Compton-y maps.

Inputs (NSIDE 4096, RING):
  HalfDome, painted with XGPaint inside 4 R200c spheres (point samples at pixel centres):
    output/halfdome_y_flamingo_sphere4_nside4096.fits                   FLAMINGO joint fit (with SOAP shells)
    output/halfdome_y_flamingo_particle_shells_sphere4_nside4096.fits   FLAMINGO particle-shells-only fit
    ../../truncation_comparison/radius_series/output_local/halfdome_y_sphere4_nside4096.fits   Battaglia+12
  FLAMINGO L2p8_m9 lightcone 0, Yang+26 lensed integrated y map (particles deposited into pixels):
    flamingo_maps/L2p8_m9_lc0_lensed_tSZ_rot_same_rot_nside4096.npy

All spectra: healpy.anafast(lmax=8192, iter=0). Stored raw; the FLAMINGO map's pixel window is
divided out downstream (painted maps are point samples and carry no pixel window).
Output: products/comparison_products.npz (+ metadata JSON). Run with the halfdome env.
"""
import json
import time
from pathlib import Path

import healpy as hp
import numpy as np

HERE = Path(__file__).resolve().parent
NSIDE, LMAX = 4096, 8192
MAPS = {
    "halfdome_flamingo": HERE / "output/halfdome_y_flamingo_sphere4_nside4096.fits",
    "halfdome_flamingo_particle_shells": HERE / "output/halfdome_y_flamingo_particle_shells_sphere4_nside4096.fits",
    "halfdome_b12": HERE.parents[1] / "truncation_comparison/radius_series/output_local/halfdome_y_sphere4_nside4096.fits",
    "flamingo_lc0": HERE / "flamingo_maps/L2p8_m9_lc0_lensed_tSZ_rot_same_rot_nside4096.npy",
    "flamingo_lc1": HERE / "flamingo_maps/L2p8_m9_lc1_lensed_tSZ_rot_same_rot_nside4096.npy",
}
LOG_Y_EDGES = np.linspace(-11, -3, 161)
CUTOUTS = [(60.0, 30.0), (200.0, -45.0)]          # (lon, lat) degrees of two 8x8 deg patches
CUTOUT_SIZE_DEG, CUTOUT_PIX = 8.0, 480


def load(path):
    if path.suffix == ".npy":
        return np.load(path, mmap_mode="r")
    return hp.read_map(path, dtype=np.float64)


def ready(path):
    """A downloaded .npy map is complete only when its sidecar JSON says so."""
    if not path.exists():
        return False
    if path.suffix == ".npy":
        state = path.with_suffix(".json")
        return state.exists() and json.loads(state.read_text()).get("complete", False)
    return True


def main():
    out = HERE / "products"
    out.mkdir(exist_ok=True)
    result, meta = {"ell": np.arange(LMAX + 1)}, {"nside": NSIDE, "lmax": LMAX, "maps": {}}
    for name, path in MAPS.items():
        cache = out / f"map_{name}.npz"
        if path.exists() and cache.exists() and cache.stat().st_mtime > path.stat().st_mtime:
            c = np.load(cache, allow_pickle=False)
            for key in ("cl", "hist", "cutouts"):
                result[f"{key}_{name}"] = c[key]
            meta["maps"][name] = json.loads(str(c["meta"]))
            print(name, "from cache", flush=True)
            continue
        if not ready(path):
            print("not ready", path)
            continue
        t0 = time.time()
        m = np.asarray(load(path), dtype=np.float64)
        assert m.size == hp.nside2npix(NSIDE), (name, m.size)
        finite = np.isfinite(m)
        if not finite.all():
            m = np.where(finite, m, 0.0)
        cl = hp.anafast(m, lmax=LMAX, iter=0)
        positive = m[m > 0]
        hist = np.histogram(np.log10(positive), LOG_Y_EDGES)[0]
        cut = [hp.gnomview(m, rot=(lon, lat), xsize=CUTOUT_PIX, reso=CUTOUT_SIZE_DEG * 60 / CUTOUT_PIX,
                           return_projected_map=True, no_plot=True).data for lon, lat in CUTOUTS]
        result[f"cl_{name}"] = cl
        result[f"hist_{name}"] = hist
        result[f"cutouts_{name}"] = np.array(cut, dtype=np.float32)
        meta["maps"][name] = {"path": str(path), "mean_y": float(m.mean()), "std_y": float(m.std()),
                              "zero_fraction": float(np.mean(m <= 0)), "negative_fraction": float(np.mean(m < 0)),
                              "negative_to_positive_sum": float(-m[m < 0].sum() / m[m > 0].sum()),
                              "min_y": float(m.min()), "max_y": float(m.max()), "finite_fraction": float(finite.mean()),
                              "percentiles_1_50_99_99.9": np.percentile(m, [1, 50, 99, 99.9]).tolist(),
                              "seconds": round(time.time() - t0, 1)}
        np.savez_compressed(cache, cl=cl, hist=hist, cutouts=result[f"cutouts_{name}"], meta=json.dumps(meta["maps"][name]))
        print(name, json.dumps(meta["maps"][name]), flush=True)
        del m
    result["log_y_edges"] = LOG_Y_EDGES
    result["pixwin"] = hp.pixwin(NSIDE, lmax=LMAX)
    meta["cutouts_lonlat_deg"] = CUTOUTS
    meta["cutout_size_deg"] = CUTOUT_SIZE_DEG
    np.savez_compressed(out / "comparison_products.npz", **result)
    (out / "comparison_products.json").write_text(json.dumps(meta, indent=1) + "\n")
    print("written", out / "comparison_products.npz")


if __name__ == "__main__":
    main()
