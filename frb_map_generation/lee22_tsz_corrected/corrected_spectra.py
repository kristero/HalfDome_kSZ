#!/usr/bin/env python3
"""Spectra for the corrected-Lee22 tSZ maps and the bookkeeping-corrected Lee22 DM maps (idark, 2026-09-30; run from
the repository root; Python 3.6 / healpy 1.15). Conventions as fullsky_tsz_dm_comparison.py and tngcal_spectra.py:
maps mean-subtracted, map2alm lmax 8192 with 3 Jacobi iterations, no pixel-window deconvolution, cross monopole set
to 0, Cauchy-Schwarz bound checked. Also compares the low-z-fixed b12/lee22p maps with the production maps pixel by
pixel. Output: frb_map_generation/outputs/lee22_corrected_20260930/spectra/corrected_spectra.npz with cl_y_<y>,
cl_dm_<model>_<survey>, cl_y_<y>__dm_<model>_<survey> and metadata_json."""
import argparse
import hashlib
import json
from pathlib import Path

import healpy as hp
import numpy as np

OUTPUTS = Path("frb_map_generation/outputs")
OUT = OUTPUTS / "lee22_corrected_20260930"
NEW_Y = ("b12", "lee22p", "lee22p_m1", "lee22p_m1_b12hi", "lee22p_m1_calib")
PRODUCTION_Y = {
    "b12": OUTPUTS / "tsz_frb_1m_z1_r200c_apertures/maps/battaglia12_fiducial_halfdome_compton_y_allz_nside4096_m200c_r200cx4.fits",
    "lee22p": OUTPUTS / "tsz_dm_cross_updated_20260917/inputs/lee22_noconc_pressure_compton_y_allz_nside4096_m200c_r200cx4.fits",
}
DM_MAPS = {  # model: directory holding <model>_<survey>.fits
    "b16_sphere1": OUTPUTS / "tsz_dm_fullsky_20260918/maps",
    "lee22_noconc_sphere1": OUTPUTS / "tsz_dm_fullsky_20260918/maps",
    "lee22_noconc_sphere1_calib": OUTPUTS / "tsz_dm_fullsky_20260918/maps",
    "lee22_noconc_tngcal_sphere1": OUTPUTS / "lee22_tngcal_20260929/fullsky/maps",
    "lee22_noconc_tngcal_sphere1_calib": OUTPUTS / "lee22_tngcal_20260929/fullsky/maps",
    "lee22_noconc_a3fn_sphere1": OUT / "fullsky/maps",
    "lee22_noconc_a3fn_sphere1_calib": OUT / "fullsky/maps",
    "lee22_noconc_phys_sphere1": OUT / "fullsky/maps",
    "lee22_noconc_phys_sphere1_calib": OUT / "fullsky/maps",
}
SURVEYS = {"planck": 71, "act": 31}
KERNELS = OUTPUTS / "tsz_dm_fullsky_20260918/kernels"


def new_y_path(name):
    return OUT / "ymaps/{}_compton_y_allz_nside4096_m200c_r200cx4_lowzexact.fits".format(name)


def sha256(path):
    h = hashlib.sha256()
    with open(str(path), "rb") as fh:
        for block in iter(lambda: fh.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def provenance(path):
    out = {}
    for line in Path(str(path)).read_text().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def read(path):
    m = hp.read_map(str(path), dtype=np.float64, verbose=False)
    if hp.get_nside(m) != 4096 or not np.all(np.isfinite(m)) or np.any(m < 0):
        raise ValueError("Invalid full-sky map: " + str(path))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmax", type=int, default=8192)
    ap.add_argument("--iter", type=int, default=3)
    a = ap.parse_args()
    target = OUT / "spectra/corrected_spectra.npz"
    target.parent.mkdir(parents=True, exist_ok=True)
    alms, arrays = {}, {"ell": np.arange(a.lmax + 1)}
    meta = {"lmax": a.lmax, "map2alm_iterations": a.iter, "pixel_window": "not deconvolved", "maps": {}, "lowz_fix": {}}
    for key in NEW_Y:
        path = new_y_path(key)
        prov = provenance(path.with_name(path.stem + "_provenance.txt"))
        if prov.get("catalog_rows_scanned") != prov.get("catalog_total_rows") or prov.get("beam") != "none":
            raise ValueError("Incomplete or beamed y map: " + str(path))
        m = read(path)
        if key in PRODUCTION_Y:   # low-z fix: pixel-level difference from the production map
            old = read(PRODUCTION_Y[key])
            d = m - old
            changed = np.abs(d) > 1e-12 * old.max()
            meta["lowz_fix"][key] = {"pixels_changed_fraction": float(changed.mean()), "mean_new": float(m.mean()),
                                     "mean_production": float(old.mean()), "mean_ratio": float(m.mean() / old.mean()),
                                     "max_abs_diff": float(np.abs(d).max()), "max_production": float(old.max())}
            print("low-z fix {}: {}".format(key, meta["lowz_fix"][key]), flush=True)
            old -= old.mean()
            alms["yprod_" + key] = hp.map2alm(old, lmax=a.lmax, iter=a.iter, pol=False)
            del old, d, changed
        mean = float(m.mean())
        m -= mean
        print("map2alm y {} (mean y {:.4e})".format(key, mean), flush=True)
        alms["y_" + key] = hp.map2alm(m, lmax=a.lmax, iter=a.iter, pol=False)
        del m
        meta["maps"]["y_" + key] = {"path": str(path), "sha256": sha256(path), "mean": mean, "provenance": prov}
    digest = {s: sha256(KERNELS / (s + "_sources.csv")) for s in SURVEYS}
    for model, folder in DM_MAPS.items():
        for survey, count in SURVEYS.items():
            path = folder / "{}_{}.fits".format(model, survey)
            prov = provenance(path.with_name(path.stem + "_provenance.txt"))
            if (prov.get("catalog_rows_scanned") != prov.get("catalog_total_rows") or prov.get("source_kernel_sha256") != digest[survey]
                    or int(prov.get("source_count", -1)) != count or prov.get("profile_label") != model):
                raise ValueError("DM map provenance mismatch: " + str(path))
            m = read(path)
            mean = float(m.mean())
            m -= mean
            print("map2alm DM {} {} (mean DM {:.3f} pc/cm3)".format(model, survey, mean), flush=True)
            alms["dm_{}_{}".format(model, survey)] = hp.map2alm(m, lmax=a.lmax, iter=a.iter, pol=False)
            del m
            meta["maps"]["dm_{}_{}".format(model, survey)] = {"path": str(path), "sha256": sha256(path), "mean": mean}
    for key, alm in alms.items():
        arrays["cl_" + key] = hp.alm2cl(alm)
    for ykey in NEW_Y:
        for model in DM_MAPS:
            for survey in SURVEYS:
                cross = hp.alm2cl(alms["y_" + ykey], alms["dm_{}_{}".format(model, survey)])
                cross[0] = 0.0
                ay, ad = arrays["cl_y_" + ykey], arrays["cl_dm_{}_{}".format(model, survey)]
                if not np.all(np.isfinite(cross)) or np.any(cross[1:] ** 2 > ay[1:] * ad[1:] * (1 + 1e-9)):
                    raise ValueError("Invalid cross spectrum {} x {} {}".format(ykey, model, survey))
                arrays["cl_y_{}__dm_{}_{}".format(ykey, model, survey)] = cross
    arrays["metadata_json"] = np.asarray(json.dumps(meta))
    np.savez_compressed(str(target), **arrays)
    print("Saved " + str(target), flush=True)


if __name__ == "__main__":
    main()
