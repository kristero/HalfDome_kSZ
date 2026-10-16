"""J0 (local): export the per-halo inputs of Test J from the validated stacking cache (stacking_choices_20261014/cache,
read through core/stackcore.py, read-only) and the exact owned particle counts from the particle-pass products
(measurements_partition.npz, measurements_owner.npz; read-only, streamed) into one compact file
inputs/j0_profiles_x200.npz (mirrored to maas) plus inputs/j0_meta.json.

All 4,588 FLAMINGO L1_m9 z = 0 centrals (rows in STUDY/selected_halos.npz order), all 15 engine mass bins 12.5-15.5,
on the X200 grid (67 shells, 0-4 R200c; x = r/R200c):
  P_a2ii   P_e/P200 (geometric estimator, recently AGN-heated gas removed, A2-ii partition)
           key 'P|geometric|removed|A2ii|all|x200|P200'
  E_a2ii   n_e mu_XGP/(rho_c f_b), same estimator / AGN cut / A2-ii      key 'E|geometric|removed|A2ii|all|x200|rhoc_fb'
  E_hb     the same for host-bound gas (central + own satellites, HBT)   key 'E|geometric|removed|host_bound|all|x200|rhoc_fb'
  E_all    the same for all gas in the sphere (Test F/H 'main')          key 'E|geometric|removed|all|all|x200|rhoc_fb'
  count_all   gas particles per shell, every owner and phase (cache counts_x200.npy; includes recent-AGN gas)
  count_a2ii  gas particles per shell owned by the halo in the A2-ii partition with the AGN cut (partition block:
              partition owner = central, painted_bound both, AGN = not_recent_agn, reduced quantity 'count')
  count_hb    host-bound gas particles per shell with the AGN cut (owner block: owners central_bound + own satellites
              = analysis.newdata.LEGACY_COMPONENT_OWNERS[0] + [1], phases xray_selected + other, quantity 'count')
  hsml     upper-median SPH smoothing length h a / R200c per shell (cache hsml_x200.npy; NaN in empty shells)
  halo table: weight (inclusion weight), mass_bin, logm200c, logm500c, m200c, r200c, m500c, r500c, fold, class_main,
              concentration_200c, eps_over_r200c, block, strata (= mass_bin * 3 + class_main), mass_edges
Checks (inputs/j0_meta.json): P_a2ii, E_a2ii and E_all bitwise equal to Test H's export (H_sensitivity/inputs/
h0_profiles_x200.npz: P_a2ii, E_a2ii, E_main); the owned counts are zero exactly where the profiles are zero; the
owned counts never exceed count_all.
Usage: OMP_NUM_THREADS=2 python -B j_inputs.py   (local; streams the 2 GB owner block row by row, < 0.3 GB RAM)
"""
from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CORE = ROOT / "stacking_choices_20261014" / "core"
sys.path.insert(0, str(CORE))
import stackcore as sc  # noqa: E402

KEYS = {"P_a2ii": sc.make_key("P", selection="A2ii", grid="x200"),
        "E_a2ii": sc.make_key("E", selection="A2ii", grid="x200"),
        "E_hb": sc.make_key("E", selection="host_bound", grid="x200"),
        "E_all": sc.make_key("E", selection="all", grid="x200")}
H0 = ROOT / "H_sensitivity" / "inputs" / "h0_profiles_x200.npz"
H0_NAME = {"P_a2ii": "P_a2ii", "E_a2ii": "E_a2ii", "E_all": "E_main"}


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def stream_rows(npz_path, member, take):
    """Stream a C-ordered float array [n, ...] from an npz member row by row; returns np.stack([take(row)])."""
    with zipfile.ZipFile(npz_path) as zf, zf.open(member) as fh:
        version = np.lib.format.read_magic(fh)
        shape, fortran, dtype = (np.lib.format.read_array_header_1_0(fh) if version == (1, 0)
                                 else np.lib.format.read_array_header_2_0(fh))
        assert not fortran
        row_shape = shape[1:]
        row_bytes = int(np.prod(row_shape)) * dtype.itemsize
        out = []
        for _ in range(shape[0]):
            buf = fh.read(row_bytes)
            assert len(buf) == row_bytes
            out.append(take(np.frombuffer(buf, dtype=dtype).reshape(row_shape)))
        return np.stack(out), shape, str(dtype)


def main():
    sys.path.insert(0, str(ROOT))
    from analysis import newdata as ND                   # owner-code bookkeeping only (read-only)
    schema = json.loads((ROOT / "schema.json").read_text())
    names = schema["names"]
    q_count = names["quantities"].index("count")
    r_count = names["reduced"].index("count")
    phases_removed = [names["phases"].index(p) for p in ("xray_selected", "other")]
    hb_owners = list(ND.LEGACY_COMPONENT_OWNERS[0]) + list(ND.LEGACY_COMPONENT_OWNERS[1])
    tab = sc.halo_table()
    n = len(tab["row"])
    assert np.array_equal(tab["row"], np.arange(n))
    out = {"x200_edges": np.asarray(sc.edges("x200"), float)}
    meta = {"cache": str(sc.CACHE), "keys": KEYS, "files": {}, "n_haloes": n}
    for name, key in KEYS.items():
        v = np.array(sc.per_halo(key), float)
        assert v.shape == (n, 67)
        out[name] = v
        info = sc.variant(key)
        meta["files"][name] = {"file": info["file"], "sha256": sha(sc.CACHE / info["file"]), "units": info.get("units"),
                               "selection_label": info.get("selection_label"), "nan": int(np.isnan(v).sum()),
                               "negative": int((v < 0).sum()), "zero": int((v == 0).sum())}
    out["count_all"] = np.array(sc.counts("x200"), float)
    out["hsml"] = np.array(sc.hsml_x("x200"), float)
    # exact owned counts (AGN cut) from the particle-pass products
    part, pshape, pdt = stream_rows(ROOT / "measurements_partition.npz", "partition.npy",
                                    lambda r: r[0, :, 0, r_count, :].sum(axis=0))
    out["count_a2ii"] = np.asarray(part, float)
    own, oshape, odt = stream_rows(ROOT / "measurements_owner.npz", "owner200.npy",
                                   lambda r: r[hb_owners][:, phases_removed, q_count, :].sum(axis=(0, 1)))
    out["count_hb"] = np.asarray(own, float)
    meta["owned_counts"] = {
        "count_a2ii": {"source": "measurements_partition.npz:partition", "shape": pshape, "dtype": pdt,
                       "index": "[:, partition_owner=central, painted_bound=all, agn=not_recent_agn, reduced=count, :]",
                       "sha256": sha(ROOT / "measurements_partition.npz")},
        "count_hb": {"source": "measurements_owner.npz:owner200", "shape": oshape, "dtype": odt,
                     "owners": [names["owners"][o] for o in hb_owners], "phases": ["xray_selected", "other"],
                     "quantity": "count", "sha256": sha(ROOT / "measurements_owner.npz")}}
    for k in ("weight", "mass_bin", "logm200c", "logm500c", "m200c", "r200c", "m500c", "r500c", "fold", "class_main",
              "concentration_200c", "eps_over_r200c", "block"):
        out[k] = np.asarray(tab[k])
    out["strata"] = np.asarray(tab["mass_bin"]) * 3 + np.asarray(tab["class_main"])
    out["mass_edges"] = np.asarray(tab["mass_edges"], float)
    # ---- checks
    chk = {}
    with np.load(H0) as h0:
        for name, hname in H0_NAME.items():
            chk[f"{name}_bitwise_equal_H0_{hname}"] = bool(np.array_equal(out[name], h0[hname]))
        chk["x200_edges_equal_H0_cache_edges"] = bool(np.array_equal(out["x200_edges"], h0["cache_x200_edges"]))
        chk["x200_edges_max_abs_diff_vs_f_common_X200"] = float(np.max(np.abs(out["x200_edges"] - h0["x200_edges"])))
        chk["weights_equal_H0"] = bool(np.array_equal(out["weight"], h0["weight"]))
        chk["hsml_equal_H0"] = bool(np.array_equal(np.nan_to_num(out["hsml"], nan=-1), np.nan_to_num(h0["hsml"], nan=-1)))
        chk["count_all_equal_H0"] = bool(np.array_equal(out["count_all"], h0["counts"]))
    # (a positive count with a zero profile is gas without free electrons / thermal pressure, e.g. cold
    #  star-forming particles; a zero count with a non-zero profile would be an inconsistency)
    for cname, pname in (("count_a2ii", "E_a2ii"), ("count_a2ii", "P_a2ii"), ("count_hb", "E_hb")):
        c, v = out[cname], out[pname]
        chk[f"{cname}_zero_iff_{pname}_zero"] = {"count0_value_nonzero": int(np.sum((c == 0) & (v != 0))),
                                                 "count_pos_value_zero": int(np.sum((c > 0) & (v == 0)))}
    for cname in ("count_a2ii", "count_hb"):
        chk[f"{cname}_le_count_all_violations"] = int(np.sum(out[cname] > out["count_all"] + 1e-9))
        chk[f"{cname}_integer_valued"] = bool(np.all(out[cname] == np.round(out[cname])))
    meta["checks"] = chk
    meta["haloes_per_engine_bin"] = np.bincount(out["mass_bin"], minlength=15).tolist()
    path = HERE / "inputs" / "j0_profiles_x200.npz"
    path.parent.mkdir(exist_ok=True)
    np.savez_compressed(path, **out)
    meta["output"] = {"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size}
    (HERE / "inputs" / "j0_meta.json").write_text(json.dumps(meta, indent=1, default=float) + "\n")
    print(json.dumps({"checks": chk, "per_bin": meta["haloes_per_engine_bin"], "bytes": meta["output"]["bytes"]},
                     indent=1, default=float))


if __name__ == "__main__":
    main()
