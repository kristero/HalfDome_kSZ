#!/usr/bin/env python3
"""y x DM spectra of the TNG-calibrated Lee22 kernel-weighted DM maps (run from the repository root on idark;
Python 3.6 / healpy 1.15 compatible). Same conventions as fullsky_tsz_dm_comparison.py `spectra`: maps mean-subtracted,
map2alm lmax 8192 with 3 Jacobi iterations, no pixel-window deconvolution, monopole of every cross set to 0, and the
Cauchy-Schwarz bound checked against the two autos. Output: frb_map_generation/outputs/lee22_tngcal_20260929/spectra/
tngcal_spectra.npz with cl_y_<ykey>__dm_<model>_<survey>, cl_dm_<model>_<survey>, cl_y_<ykey> and metadata_json."""
import argparse
import hashlib
import json
from pathlib import Path

import healpy as hp
import numpy as np

OUTPUTS = Path("frb_map_generation/outputs")
OUT = OUTPUTS / "lee22_tngcal_20260929"
Y_MAPS = {
    "b12": OUTPUTS / "tsz_frb_1m_z1_r200c_apertures/maps/battaglia12_fiducial_halfdome_compton_y_allz_nside4096_m200c_r200cx4.fits",
    "lee22p": OUTPUTS / "tsz_dm_cross_updated_20260917/inputs/lee22_noconc_pressure_compton_y_allz_nside4096_m200c_r200cx4.fits",
    "lee22p_calib": OUTPUTS / "tsz_dm_fullsky_20260918/maps/lee22_noconc_pressure_compton_y_calib_zmax2p0_nside4096_m200c_r200cx4.fits",
}
DM_MODELS = ("lee22_noconc_tngcal_sphere1", "lee22_noconc_tngcal_sphere1_calib")
SURVEYS = {"planck": 71, "act": 31}
KERNELS = OUTPUTS / "tsz_dm_fullsky_20260918/kernels"


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


def load_centered(path):
    m = hp.read_map(str(path), dtype=np.float64, verbose=False)
    if hp.get_nside(m) != 4096 or not np.all(np.isfinite(m)) or np.any(m < 0):
        raise ValueError("Invalid full-sky map: " + str(path))
    mean = float(np.mean(m))
    m -= mean
    return m, mean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmax", type=int, default=8192)
    ap.add_argument("--iter", type=int, default=3)
    a = ap.parse_args()
    target = OUT / "spectra/tngcal_spectra.npz"
    target.parent.mkdir(parents=True, exist_ok=True)
    alms, arrays = {}, {"ell": np.arange(a.lmax + 1)}
    meta = {"lmax": a.lmax, "map2alm_iterations": a.iter, "pixel_window": "not deconvolved", "maps": {}}
    for key, path in Y_MAPS.items():
        prov = provenance(path.with_name(path.stem + "_provenance.txt"))
        if prov.get("catalog_rows_scanned") != prov.get("catalog_total_rows") or prov.get("beam") != "none":
            raise ValueError("Incomplete or beamed y map: " + str(path))
        m, mean = load_centered(path)
        print("map2alm y {} (mean y {:.4e})".format(key, mean), flush=True)
        alms["y_" + key] = hp.map2alm(m, lmax=a.lmax, iter=a.iter, pol=False)
        del m
        meta["maps"]["y_" + key] = {"path": str(path), "sha256": sha256(path), "mean": mean}
    digest = {s: sha256(KERNELS / (s + "_sources.csv")) for s in SURVEYS}
    for model in DM_MODELS:
        for survey, count in SURVEYS.items():
            path = OUT / "fullsky/maps/{}_{}.fits".format(model, survey)
            prov = provenance(path.with_name(path.stem + "_provenance.txt"))
            if (prov.get("catalog_rows_scanned") != prov.get("catalog_total_rows") or prov.get("source_kernel_sha256") != digest[survey]
                    or int(prov.get("source_count", -1)) != count or prov.get("profile_label") != model):
                raise ValueError("DM map provenance mismatch: " + str(path))
            m, mean = load_centered(path)
            print("map2alm DM {} {} (mean DM {:.3f} pc/cm3)".format(model, survey, mean), flush=True)
            alms["dm_{}_{}".format(model, survey)] = hp.map2alm(m, lmax=a.lmax, iter=a.iter, pol=False)
            del m
            meta["maps"]["dm_{}_{}".format(model, survey)] = {"path": str(path), "sha256": sha256(path), "mean": mean,
                                                              "provenance": prov}
    for key, alm in alms.items():
        arrays["cl_" + key] = hp.alm2cl(alm)
    for ykey in Y_MAPS:
        for model in DM_MODELS:
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
