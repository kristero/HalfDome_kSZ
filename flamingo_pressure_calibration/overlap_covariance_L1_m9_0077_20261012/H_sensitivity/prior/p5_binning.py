"""P5: earlier results on mass binning and binning choice (read only).

Sources
* B_covariance/results/b2_stability_pairs.csv (Test B2): permutation tests of the per-halo ln F correlation structure
  between adjacent 0.2-dex bins (min-p over metrics).
* stacking_choices_20261014/groups/mass_binning/work/res_schemes.json, res_topbins.json (stacking-choices workflow,
  2026-10-14; machine-readable results, no written report yet): first-order impact of other binning schemes and of
  top-bin pooling on the observables, propagated with F's importance maps from the measured stacks (no fits).
Outputs: tables/binning_stability_pairs.csv, binning_scheme_impacts.csv, binning_topbins.csv, binning_assignment.csv,
p5_summary.json.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import prior_common as C


def main():
    S = {}
    st = pd.read_csv(C.B_RESULTS / "b2_stability_pairs.csv")
    pcol = [c for c in st.columns if c.startswith("p_minp")] or [c for c in st.columns if "minp" in c]
    st = st[(st.status != "undetermined")]
    keep = ["grid", "sample", "field", "bins", "shells", "N_a", "N_b", "rank_deficient", "frobenius_rms",
            "null95_frobenius_rms", "max_angle_deg", "null95_max_angle_deg"] + pcol
    st[keep].to_csv(C.TABLES / "binning_stability_pairs.csv", index=False, float_format="%.4g")
    summ = []
    for (grid, sample, field), g in st.groupby(["grid", "sample", "field"]):
        p = g[pcol[-1]]
        summ.append({"grid": grid, "sample": sample, "field": field, "pairs_tested": len(g),
                     "pairs_p_lt_0.05": int((p < 0.05).sum()), "p_column": pcol[-1],
                     "differing_pairs": "; ".join(g.loc[p < 0.05, "bins"].tolist())})
    S["stability"] = summ

    rs = json.load(open(C.MB_WORK / "res_schemes.json"))
    rows = []
    for scheme, v in rs["scheme_impacts_smooth_painting"].items():
        row = {"scheme": scheme, "extrapolated_bins": ";".join(v["P"].get("extrapolated_bins", []))}
        for f in ("P", "E"):
            for o, x in v[f]["impact"].items():
                row[f"{o}_pct"] = x["pct"]
                row[f"{o}_err_pct"] = x["err_pct"]
        rows.append(row)
    pd.DataFrame(rows).to_csv(C.TABLES / "binning_scheme_impacts.csv", index=False, float_format="%.4g")
    S["scheme_impacts_smooth_painting"] = rows
    S["scheme_impacts_note"] = rs["notes"]["scheme_impacts_smooth_painting"]
    arow = []
    for k, v in rs["painting_assignment_estimate"].items():
        arow.append({"scheme|assignment": k, **{f"vs_truth_{a}": b for a, b in v["vs_truth_pct"].items()},
                     **{f"vs_smooth_M200c_0.1_{a}": b for a, b in v["vs_smooth_M200c_0.1_pct"].items()}})
    pd.DataFrame(arow).to_csv(C.TABLES / "binning_assignment.csv", index=False, float_format="%.4g")
    S["painting_assignment_estimate"] = arow
    S["painting_assignment_note"] = rs["notes"]["painting_assignment_estimate"]

    rt = json.load(open(C.MB_WORK / "res_topbins.json"))
    trow = []
    for k, v in rt["impacts"].items():
        row = {"variant": k}
        for f in v:
            for o, x in v[f].items():
                row[f"{o}_pct"] = x["pct"]
                row[f"{o}_err_pct"] = x["err_pct"]
        trow.append(row)
    pd.DataFrame(trow).to_csv(C.TABLES / "binning_topbins.csv", index=False, float_format="%.4g")
    S["topbins"] = trow
    S["topbins_note"] = rt["notes"]
    # within-bin mass trend: how much of the per-halo scatter a 0.2-dex bin's mass spread explains
    tv = []
    for f in ("P", "E"):
        for b in rs["mass_trend"][f]:
            share = b.get("trend_var_share") or {}
            vals = [x for x in share.values() if x is not None]
            if vals:
                tv.append({"field": f, "bin": b["bin"], "trend_var_share_max_over_bands": float(np.max(vals))})
    S["trend_var_share"] = tv
    C.write_json(C.PRIOR / "p5_summary.json", S)
    print(pd.DataFrame(summ).to_string())
    print(pd.DataFrame(rows).round(2).to_string())


if __name__ == "__main__":
    main()
