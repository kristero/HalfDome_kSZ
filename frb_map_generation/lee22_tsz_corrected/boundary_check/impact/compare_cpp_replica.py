#!/usr/bin/env python
"""Compare the C++ group_particles outputs (y_prof / ne_prof, builds buggy / plain / perfix / naive) of one run
directory with the Python replica (camels_bug_replica.py) for the same translation.

Usage: python compare_cpp_replica.py REPLICA.npz SHIFT_INDEX RUNDIR SNAPNUM OUT.json
Per variant: groups and M200c order identical?; per (group, shell) count mismatches against the replica's fixed
(plain/perfix/naive) or buggy (buggy) pair set; total pairs; max relative difference of the shell sums where the
counts agree; the C++ buggy-vs-plain pair deficit compared with the replica's low-face pair count.
"""
import json
import os
import sys

import numpy as np

rep_f, s_idx, run, snap, out_json = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5]
R = np.load(rep_f)
NB = 128


def load(v):
    d = os.path.join(run, v)
    M = np.fromfile(f"{d}/output/grp_M200c_{snap}.bin", "f8")
    P200 = np.fromfile(f"{d}/output/grp_P200c_{snap}.bin", "f8")
    pe = np.fromfile(f"{d}/output/grp_pressure_prof_{snap}.bin", "f8").reshape(-1, NB)
    npe = np.fromfile(f"{d}/output/grp_num_part_prof_{snap}.bin", "u8").reshape(-1, NB).astype(np.int64)
    Mn = np.fromfile(f"{d}/ne_prof_results_Jun22/grp_M200c.bin", "f8")
    ne = np.fromfile(f"{d}/ne_prof_results_Jun22/grp_ne_prof.bin", "f8").reshape(-1, NB)
    nne = np.fromfile(f"{d}/ne_prof_results_Jun22/grp_num_part_prof.bin", "u8").reshape(-1, NB).astype(np.int64)
    return dict(M=M, Mn=Mn, P200=P200, pe=pe, npe=npe, ne=ne, nne=nne)


res = {"replica": os.path.basename(rep_f), "shift_index": s_idx, "shift": R["shifts"][s_idx].tolist(), "run": run}
variants = [v for v in ("buggy", "plain", "perfix", "naive") if os.path.isdir(os.path.join(run, v))]
C = {v: load(v) for v in variants}
for v in variants:
    c = C[v]
    if v == "buggy":
        n_rep, pe_rep, ne_rep = R["bug_n"][s_idx], R["bug_pe"][s_idx] / R["vol"], R["bug_ne"][s_idx] / R["vol"]
    else:
        n_rep, pe_rep, ne_rep = R["fix_n"], R["fix_pe"] / R["vol"], R["fix_ne"] / R["vol"]
    same_groups = bool(np.array_equal(c["M"], R["M200_code"]) and np.array_equal(c["Mn"], R["M200_code"]))
    r = {"n_groups": int(len(c["M"])), "groups_and_order_identical": same_groups,
         "P200_max_rel_diff": float(np.max(np.abs(c["P200"] / R["P200_code"] - 1)))}
    for q, ncpp, vcpp, vrep in (("pe", c["npe"], c["pe"], pe_rep), ("ne", c["nne"], c["ne"], ne_rep)):
        mism = ncpp != n_rep
        agree = (~mism) & (ncpp > 0)
        rel = np.abs(vcpp[agree] / vrep[agree] - 1)
        r[q] = {"pairs_cpp": int(ncpp.sum()), "pairs_replica": int(n_rep.sum()),
                "shells_with_count_mismatch": int(mism.sum()), "shells_total": int(mism.size),
                "sum_abs_count_mismatch": int(np.abs(ncpp - n_rep).sum()),
                "max_rel_diff_where_counts_agree": float(rel.max()) if rel.size else 0.0}
    res[v] = r
if "buggy" in C and "plain" in C:
    lost_cpp = C["plain"]["npe"].sum(1) - C["buggy"]["npe"].sum(1)
    lost_rep = R["fix_n"].sum(1) - R["bug_n"][s_idx].sum(1)
    res["lost_pairs"] = {"cpp_total": int(lost_cpp.sum()), "replica_total": int(lost_rep.sum()),
                         "groups_with_loss_cpp": int((lost_cpp > 0).sum()), "groups_with_loss_replica": int((lost_rep > 0).sum()),
                         "max_abs_group_diff": int(np.abs(lost_cpp - lost_rep).max()),
                         "cpp_buggy_minus_plain_negative_shells": int(((C["buggy"]["npe"] - C["plain"]["npe"]) > 0).sum())}
for a, b in (("plain", "perfix"), ("plain", "naive")):
    if a in C and b in C:
        dn = np.abs(C[a]["npe"] - C[b]["npe"]).sum() + np.abs(C[a]["nne"] - C[b]["nne"]).sum()
        m = C[b]["pe"] > 0
        res[f"{a}_vs_{b}"] = {"sum_abs_count_diff": int(dn),
                              "max_rel_pe": float(np.max(np.abs(C[a]["pe"][m] / C[b]["pe"][m] - 1))) if m.any() else 0.0}
with open(out_json, "w") as f:
    json.dump(res, f, indent=1)
print(json.dumps(res, indent=1))
