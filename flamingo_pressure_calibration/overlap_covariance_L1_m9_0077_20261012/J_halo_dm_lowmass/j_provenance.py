"""Provenance checks quoted by the J1 report (local, numpy only, read-only inputs; writes post/provenance_checks.json).

Q1 (why 12.5-13.5 was not calibrated):
  * radii234 (2026-10-08) halo sample: radii234_contamination_L1_m9_0077_20261008/selected_halos.npz (halo_index,
    mass_bin) per 0.2-dex bin; the current sample of this study: ../selected_halos.npz (the overlap particle pass);
    a radii234 halo is 'in the current sample' if its halo_index (SOAP row) is there and in the same mass bin;
  * the 2026-10-09 truncation study's common-support rule (tsz_fit_truncation_L1_m9_0077_20261009/fit_inputs/
    prepare_fit_inputs.py, 'common = strict & np.all(stable, axis=1) & (masks[:,0].sum(axis=1)>=6)'), its output
    common_support.json (strict_resolution_supported, stable_all_radii, fit_flags[b].strict_1R_shells) and radii234's
    products/study_summary.json bins[b].fit_status are read and copied here per bin.
D1 = all gas inside R200c: inputs/j0_profiles_x200.npz E_a2ii against E_all on the 50 X200 shells inside R200c, every halo
  (relative difference over the halo-shells where E_all != 0; the zero patterns must be equal).
Usage: python -B j_provenance.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
FPC = STUDY.parent
R234 = FPC / "radii234_contamination_L1_m9_0077_20261008"
TRUNC = FPC / "tsz_fit_truncation_L1_m9_0077_20261009" / "fit_inputs"
LABELS = ["12.5-12.7", "12.7-12.9", "12.9-13.1", "13.1-13.3", "13.3-13.5"]


def main():
    out = {"definitions": __doc__, "q1": {}, "a2ii_equals_all_gas_inside_R200c": {}}
    a = np.load(R234 / "selected_halos.npz")
    b = np.load(STUDY / "selected_halos.npz")
    assert np.array_equal(a["mass_edges"], b["mass_edges"])
    cur = dict(zip(b["halo_index"].tolist(), b["mass_bin"].tolist()))
    prep = json.loads((HERE / "results" / "prepare.json").read_text())
    cs = json.loads((TRUNC / "common_support.json").read_text())
    st = json.loads((R234 / "products" / "study_summary.json").read_text())
    src = (TRUNC / "prepare_fit_inputs.py").read_text().splitlines()
    rule_line = [i + 1 for i, line in enumerate(src) if re.search(r"common = strict & np\.all\(stable", line)]
    per = {}
    for k, lab in enumerate(LABELS):
        ids = a["halo_index"][a["mass_bin"] == k].tolist()
        n_cur = int(np.sum(b["mass_bin"] == k))
        kept = sum(1 for h in ids if cur.get(h) == k)
        ff = cs["fit_flags"][k]
        assert ff["mass_edges"] == [float(x) for x in lab.split("-")], (ff["mass_edges"], lab)
        per[lab] = {"radii234_haloes": len(ids), "radii234_haloes_in_current_sample": kept, "current_haloes": n_cur,
                    "added_by_overlap_particle_pass": n_cur - kept, "J1_prepare_n_haloes": prep["J1a_by_bin"][lab]["n_haloes"],
                    "common_support": bool(cs["common_support"][k]),
                    "strict_resolution_supported": bool(cs["strict_resolution_supported"][k]),
                    "strict_resolved_shells_inside_R200c": int(ff["strict_1R_shells"]),
                    "stable_fit_at_1R_2R_3R_4R": [bool(x) for x in cs["stable_all_radii"][k]],
                    "radii234_fit_status": st["bins"][k]["fit_status"]}
    tot = {key: sum(v[key] for v in per.values()) for key in ("radii234_haloes", "radii234_haloes_in_current_sample",
                                                               "current_haloes", "added_by_overlap_particle_pass")}
    out["q1"] = {"per_bin": per, "totals": tot, "common_support_rule_line": rule_line,
                 "common_support_rule": src[rule_line[0] - 1].strip() if rule_line else None,
                 "minimum_1R_shells": cs["minimum_1R_shells"],
                 "checks": {"current_counts_equal_J1_prepare": all(v["current_haloes"] == v["J1_prepare_n_haloes"]
                                                                   for v in per.values()),
                            "every_radii234_halo_in_current_sample": tot["radii234_haloes"] == tot["radii234_haloes_in_current_sample"]}}
    j0 = np.load(HERE / "inputs" / "j0_profiles_x200.npz")
    X = np.asarray(j0["x200_edges"], float)
    n1 = int(np.sum(X[1:] <= 1.0 + 1e-12))
    Ea, Eall = np.asarray(j0["E_a2ii"], float)[:, :n1], np.asarray(j0["E_all"], float)[:, :n1]
    nz = Eall != 0
    out["a2ii_equals_all_gas_inside_R200c"] = {
        "n_haloes": int(Ea.shape[0]), "n_shells_inside_R200c": n1, "outer_edge": float(X[n1]),
        "n_halo_shells": int(Ea.size), "n_halo_shells_nonzero": int(nz.sum()),
        "max_rel_difference": float(np.max(np.abs(Ea[nz] / Eall[nz] - 1))),
        "zero_patterns_equal": bool(np.array_equal(Ea == 0, Eall == 0)),
        "max_abs_where_all_gas_zero": float(np.max(np.abs(Ea[~nz]), initial=0.0)),
        "first_shell_outside_R200c_max_rel_difference": float(np.max(np.abs(
            np.asarray(j0["E_a2ii"], float)[:, n1][np.asarray(j0["E_all"], float)[:, n1] != 0]
            / np.asarray(j0["E_all"], float)[:, n1][np.asarray(j0["E_all"], float)[:, n1] != 0] - 1)))}
    (HERE / "post").mkdir(exist_ok=True)
    (HERE / "post" / "provenance_checks.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({"q1_totals": tot, "q1_checks": out["q1"]["checks"], "rule_line": rule_line,
                      "a2ii_vs_all": out["a2ii_equals_all_gas_inside_R200c"]}, indent=1))


if __name__ == "__main__":
    main()
