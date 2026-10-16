"""L stage 'inputs' (local, light): the per-halo arrays that the stacking cache does not hold.

1. The A2-ii complement split by the partition block's painted-bound flag (measurements_partition.npz, read through the
   study library analysis.variants.profile with two partition selections, the same code path as the cache's A2-ii):
     p10 = partition owner 'painted_neighbour' (r/R200c smaller to a painted neighbour), not bound to another painted
           halo (owner 14);   p11 = the same, bound to another painted halo.
   For P, E and G on the X200 grid.  Gate: all = a2ii + p10 + p11 per halo-shell (cache arrays).
   Why: host-bound gas is never 'bound to another painted halo', so per halo-shell
        max(0, hb - p10) <= (hb AND a2ii) <= min(hb, a2iia2i)
   brackets the nested bound-in-A2-ii quantity, which is not stored (the partition block has no owner axis).
2. Owned particle counts of the central-bound and host-bound selections (recent AGN removed) per halo-shell, streamed
   from measurements_owner.npz:owner200 one halo at a time (the 2.0 GB member is never held in memory).  Gate: the
   host-bound counts equal Test J's J0 export count_hb exactly.
3. R200m per halo: SOAP SO/200_mean/SORadius (research/soap/soap_study_rows.npz, joined on soap_row), physical Mpc.
4. From Test J's J0 export (bitwise the cache's): count_all, count_a2ii, count_hb, hsml.
Output: inputs/l_inputs.npz (compressed), inputs/l_inputs_meta.json (sources with SHA256, gates).
"""
from __future__ import annotations

import zipfile

import l_common as LC
import numpy as np

from analysis import newdata, variants as V


def stream_owner_counts(path, owner_sets, quantity_index, phases):
    """Sum of owner200[h, owners, phases, quantity, :] per halo for each owner set, streamed from the npz member."""
    with zipfile.ZipFile(path) as zf, zf.open("owner200.npy") as fh:
        version = np.lib.format.read_magic(fh)
        shape, fortran, dtype = np.lib.format._read_array_header(fh, version)
        assert not fortran and dtype == np.float64, (fortran, dtype)
        n, n_own, n_ph, n_q, n_s = shape
        per = n_own * n_ph * n_q * n_s
        out = {k: np.zeros((n, n_s)) for k in owner_sets}
        for h in range(n):
            buf = fh.read(per * 8)
            assert len(buf) == per * 8
            a = np.frombuffer(buf, dtype=np.float64).reshape(n_own, n_ph, n_q, n_s)
            sub = a[:, list(phases), quantity_index, :].sum(axis=1)      # [owners, shells]
            for k, owners in owner_sets.items():
                out[k][h] = sub[list(owners)].sum(axis=0)
    return out, shape


def main():
    t0 = LC.time.perf_counter()
    meta = {"sources": {}, "gates": {}, "definitions": __doc__}
    d = newdata.load(LC.STUDY)
    # ---------------------------------------------------------------- 1. partition complement
    sel10 = V.Selection("partition_nbr_not_painted_bound", owners=None, partition_owner=(1,), painted_bound=(0,),
                        label="A2-ii complement (painted neighbour owns), not bound to another painted halo")
    sel11 = V.Selection("partition_nbr_painted_bound", owners=None, partition_owner=(1,), painted_bound=(1,),
                        label="A2-ii complement (painted neighbour owns), bound to another painted halo")
    lib = {"P": "pressure", "E": "electrons", "G": "gas"}
    out = {}
    for f, name in lib.items():
        out[f"{f}_p10"] = V.profile(d, name, "geometric", "removed", sel10, "x200").values
        out[f"{f}_p11"] = V.profile(d, name, "geometric", "removed", sel11, "x200").values
        a2ii_lib = V.profile(d, name, "geometric", "removed", "partition_owned", "x200").values
        a2ii = np.asarray(LC.SC.per_halo(LC.key(f, "a2ii")))
        allg = np.asarray(LC.SC.per_halo(LC.key(f, "all")))
        a2iia2i = np.asarray(LC.SC.per_halo(LC.key(f, "a2iia2i")))
        scale = np.maximum(np.abs(allg), 1e-300)
        meta["gates"][f"{f}_library_a2ii_equals_cache"] = float(np.max(np.abs(a2ii_lib - a2ii)))
        clo = np.abs(a2ii + out[f"{f}_p10"] + out[f"{f}_p11"] - allg)
        meta["gates"][f"{f}_all_eq_a2ii_p10_p11_max_rel"] = float(np.max(np.where(allg > 0, clo / scale, clo)))
        p01 = a2ii - a2iia2i                                       # A2-ii and bound to another painted halo
        meta["gates"][f"{f}_min_a2ii_minus_a2iia2i"] = float(np.min(p01))
        inner = LC.X[1:] <= 1.0 + 1e-12
        meta["gates"][f"{f}_complement_inside_R200c_max"] = float(np.max(np.abs(out[f"{f}_p10"][:, inner]) +
                                                                  np.abs(out[f"{f}_p11"][:, inner])))
    d.release()
    meta["sources"]["measurements_partition.npz"] = LC.sha256(LC.STUDY / "measurements_partition.npz")
    # ---------------------------------------------------------------- 2. owner counts (streamed)
    names = d.names
    q_count = list(names["quantities"]).index("count")
    phases = [list(names["phases"]).index(p) for p in ("xray_selected", "other")]      # recent AGN removed
    cb = V.selection("central_bound").owners
    hb = V.selection("host_bound").owners
    owner_sets = {"cb": tuple(np.flatnonzero(cb)), "hb": tuple(np.flatnonzero(hb))}
    LC.log("streaming owner200 counts ...")
    cnt, shape = stream_owner_counts(LC.STUDY / "measurements_owner.npz", owner_sets, q_count, phases)
    meta["owner200_shape"] = list(shape)
    meta["owner_sets"] = {k: [int(i) for i in v] for k, v in owner_sets.items()}
    meta["sources"]["measurements_owner.npz"] = LC.sha256(LC.STUDY / "measurements_owner.npz")
    j0 = np.load(LC.J_DIR / "inputs" / "j0_profiles_x200.npz")
    meta["sources"]["J_halo_dm_lowmass/inputs/j0_profiles_x200.npz"] = LC.sha256(LC.J_DIR / "inputs" / "j0_profiles_x200.npz")
    meta["gates"]["count_hb_equals_J0_max_abs"] = float(np.max(np.abs(cnt["hb"] - j0["count_hb"])))
    meta["gates"]["count_cb_le_hb"] = bool(np.all(cnt["cb"] <= cnt["hb"] + 1e-9))
    out["count_cb"] = cnt["cb"]
    out["count_hb_stream"] = cnt["hb"]
    for k in ("count_all", "count_a2ii", "count_hb", "hsml"):
        out[k] = j0[k]
    # J0 profile arrays equal the cache (gate)
    for k, (f, s) in {"P_a2ii": ("P", "a2ii"), "E_a2ii": ("E", "a2ii"), "E_hb": ("E", "hb"), "E_all": ("E", "all")}.items():
        meta["gates"][f"J0_{k}_equals_cache"] = float(np.max(np.abs(j0[k] - np.asarray(LC.SC.per_halo(LC.key(f, s))))))
    # ---------------------------------------------------------------- 3. R200m from SOAP
    sp = LC.STUDY / "stacking_choices_20261014" / "research" / "soap" / "soap_study_rows.npz"
    s = np.load(sp)
    tab = LC.SC.halo_table()
    idx = np.searchsorted(s["soap_row_sorted"], tab["soap_row"])
    assert np.all(s["soap_row_sorted"][idx] == tab["soap_row"])
    out["r200m_mpc"] = np.asarray(s["SO__200_mean__SORadius"][idx], float)
    out["r100c_mpc"] = np.asarray(s["SO__100_crit__SORadius"][idx], float)
    meta["sources"]["research/soap/soap_study_rows.npz"] = LC.sha256(sp)
    meta["r200m_over_r200c_percentiles_0_16_50_84_100"] = np.percentile(out["r200m_mpc"] / tab["r200c"],
                                                                        [0, 16, 50, 84, 100]).tolist()
    meta["gates"]["r200m_gt_r200c"] = bool(np.all(out["r200m_mpc"] > tab["r200c"]))
    meta["sources"]["cache/halo_table.npz"] = LC.sha256(LC.CACHE / "halo_table.npz")
    for f in ("P", "E", "G"):
        for sname in ("all", "a2ii", "hb", "cb", "a2iia2i"):
            fn = LC.SC.variant(LC.key(f, sname))["file"]
            meta["sources"][f"cache/{fn}"] = LC.sha256(LC.CACHE / fn)
    for k in LC.WEIGHTED_KEYS.values():
        fn = LC.SC.variant(k)["file"]
        meta["sources"][f"cache/{fn}"] = LC.sha256(LC.CACHE / fn)
    np.savez_compressed(LC.INPUT_FILE, **{k: np.asarray(v, np.float64) for k, v in out.items()})
    meta["output_sha256"] = LC.sha256(LC.INPUT_FILE)
    meta["seconds"] = LC.time.perf_counter() - t0
    meta["peak_rss_gb"] = LC.rss_gb()
    LC.jdump(meta, LC.INPUT_META)
    LC.log("inputs done", meta["gates"], f"{meta['seconds']:.0f} s, rss {meta['peak_rss_gb']:.2f} GB")


if __name__ == "__main__":
    main()
