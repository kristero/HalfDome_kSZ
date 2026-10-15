"""H0 (local): export the per-halo inputs of Test H from the validated stacking cache (stacking_choices_20261014/cache,
read through core/stackcore.py, read-only) into compact files under H_sensitivity/inputs/ that are mirrored to maas.

Per halo (4,588 centrals, rows in STUDY/selected_halos.npz order = Test F's halo_rows), on the X200 grid (67 shells,
0-4 R200c; x = r/R200c):
  P_<sel>  pressure P_e/P200, geometric estimator (sum of particle y over the shell volume), recently AGN-heated gas
           removed;  key 'P|geometric|removed|<A2ii|all>|all|x200|P200'
  E_<sel>  electrons n_e mu_XGP/(rho_c f_b), same estimator, AGN selection and ownership; key
           'E|geometric|removed|<A2ii|all>|all|x200|rhoc_fb'
  sel = a2ii (A2-ii painting partition: each gas particle belongs to the painted halo, M200c >= 6.6e12 Msun, in
        which it has the smallest r/R200c; own satellites and sub-threshold groups stay with the host) and
        main (all gas, satellites kept: Test F's 'main' selection).
  counts / hsml  gas-particle count and upper-median smoothing length h a/R200c per shell (resolution inputs).
Per engine mass bin (15 bins of 0.2 dex in M200c, 12.5-15.5): the resolution masks (analysis.masks semantics through
stackcore.resolution_detail; base = strict if the bin is supported, else diagnostic).
Halo table: weight (inclusion weight), mass_bin, logm200c, logm500c, m200c, r200c, fold, class_main, strata.
Mass-binning schemes of stackcore (edges and variable) for H5.
Checks written to inputs/h0_meta.json: cache x200 edges vs f_common.X200, and the volume-weighted rebinning of every
exported array to Test F's 35 shells against F_observables/results/profiles.npz (must agree to ~1e-15).
Usage: python -B h_inputs.py   (local; reads ~40 MB of the cache)
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CORE = ROOT / "stacking_choices_20261014" / "core"
sys.path.insert(0, str(CORE))
import stackcore as sc  # noqa: E402

KEYS = {("P", "a2ii"): sc.make_key("P", selection="A2ii", grid="x200"),
        ("E", "a2ii"): sc.make_key("E", selection="A2ii", grid="x200"),
        ("P", "main"): sc.make_key("P", selection="all", grid="x200"),
        ("E", "main"): sc.make_key("E", selection="all", grid="x200")}
SCHEMES = ("M200c_0.1", "M200c_0.2", "M200c_0.3", "M200c_0.5", "M500c_0.2", "M500c_0.5")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def main():
    sys.path.insert(0, str(ROOT / "F_observables"))
    import os
    import tempfile
    os.environ.setdefault("F_RESULTS", tempfile.mkdtemp(prefix="h0_fcommon_"))   # f_common's dirs (unused)
    import f_common as FC                       # only for X200 (and its edge construction); writes nothing else
    ex = sc.edges("x200")
    x200 = np.asarray(FC.X200, float)
    tab = sc.halo_table()
    n = len(tab["row"])
    assert np.array_equal(tab["row"], np.arange(n))
    out = {"x200_edges": x200, "cache_x200_edges": ex}
    meta = {"cache": str(sc.CACHE), "keys": {f"{f}_{s}": k for (f, s), k in KEYS.items()},
            "x200_edges_max_abs_diff_cache_vs_f_common": float(np.max(np.abs(ex - x200))), "files": {}}
    for (f, s), key in KEYS.items():
        v = np.asarray(sc.per_halo(key), float)
        info = sc.variant(key)
        assert v.shape == (n, 67)
        out[f"{f}_{s}"] = v
        meta["files"][f"{f}_{s}"] = {"file": info["file"], "sha256": sha(sc.CACHE / info["file"]),
                                     "units": info.get("units"), "description": info.get("description"),
                                     "nan": int(np.isnan(v).sum()), "negative": int((v < 0).sum()),
                                     "zero": int((v == 0).sum())}
    out["counts"] = np.asarray(sc.counts("x200"), float)
    out["hsml"] = np.asarray(sc.hsml_x("x200"), float)
    for k in ("weight", "mass_bin", "logm200c", "logm500c", "m200c", "r200c", "m500c", "r500c", "fold", "class_main",
              "concentration_200c", "eps_over_r200c", "block"):
        out[k] = np.asarray(tab[k])
    out["strata"] = np.asarray(tab["mass_bin"]) * 3 + np.asarray(tab["class_main"])
    # resolution masks per engine bin (15 bins), analysis.masks semantics
    edges = np.asarray(tab["mass_edges"], float)
    nb = len(edges) - 1
    base = np.zeros((nb, 67), bool)
    strict = np.zeros_like(base)
    diag = np.zeros_like(base)
    supported = np.zeros(nb, bool)
    resolved_fraction = np.zeros((nb, 67))
    for b in range(nb):
        rows = np.flatnonzero(tab["mass_bin"] == b)
        if not len(rows):
            continue
        d = sc.resolution_detail(rows, "x200")
        strict[b], diag[b] = d["strict"], d["diagnostic"]
        supported[b] = int(np.sum(d["strict"] & (ex[1:] <= 2.0))) >= 6
        base[b] = strict[b] if supported[b] else diag[b]
        resolved_fraction[b] = d["resolved_fraction"]
    out.update(base_mask=base, strict_mask=strict, diagnostic_mask=diag, supported=supported,
               resolved_fraction=resolved_fraction, mass_edges=edges)
    # stackcore's fit-grid masks reproduce Test F's (sanity check of the semantics)
    fpr = np.load(ROOT / "F_observables" / "results" / "profiles.npz")
    ef = sc.edges("fit")
    same = []
    for b in range(nb):
        rows = np.flatnonzero(tab["mass_bin"] == b)
        if not len(rows):
            continue
        d = sc.resolution_detail(rows, "fit")
        sup = int(np.sum(d["strict"] & (ef[1:] <= 2.0))) >= 6
        bm = d["strict"] if sup else d["diagnostic"]
        same.append(bool(np.array_equal(bm, fpr["base_mask"][b])))
    meta["fit_grid_base_mask_equals_test_F"] = same
    # rebinning parity with Test F's per-halo profiles
    idx = np.array([int(np.argmin(np.abs(x200 - e))) for e in FC.F_EDGES])
    vol = 4 * np.pi / 3 * np.diff(x200**3)
    worst = {}
    for (f, s) in KEYS:
        reb = np.add.reduceat(out[f"{f}_{s}"] * vol, idx[:-1], axis=1) / np.add.reduceat(vol, idx[:-1])
        ref = fpr[f"{f}_{s}"]
        nz = ref != 0
        worst[f"{f}_{s}"] = {"max_rel": float(np.max(np.abs(reb[nz] / ref[nz] - 1))),
                             "max_abs_where_F_is_zero": float(np.max(np.abs(reb[~nz]))) if (~nz).any() else 0.0}
    meta["rebinned_vs_test_F_profiles"] = worst
    # mass-binning schemes (H5)
    schemes = {}
    for name in SCHEMES:
        sd = sc.scheme_definition(name)
        bins = sc.mass_bins(name)
        schemes[name] = {"variable": sd["variable"], "edges": [float(e) for e in sd["edges"]], "units": sd["units"],
                         "bins": [{"label": b["label"], "lo": b["lo"], "hi": b["hi"], "N": b["N"],
                                   "N_eff": b["N_eff"], "mean_log_mass": b["mean_log_mass"],
                                   "mean_logm200c": b["mean_logm200c"]} for b in bins]}
    meta["schemes"] = schemes
    meta["selection_definitions"] = {s: sc.variant(KEYS[("P", s)]).get("selection_label") for s in ("a2ii", "main")}
    meta["n_haloes"] = n
    path = HERE / "inputs" / "h0_profiles_x200.npz"
    path.parent.mkdir(exist_ok=True)
    np.savez_compressed(path, **out)
    meta["output"] = {"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size}
    (HERE / "inputs" / "h0_meta.json").write_text(json.dumps(meta, indent=1, default=float) + "\n")
    print(json.dumps({"rebinning": worst, "fit_mask_parity": all(same), "edges": meta["x200_edges_max_abs_diff_cache_vs_f_common"],
                      "bytes": meta["output"]["bytes"]}, indent=1))


if __name__ == "__main__":
    main()
