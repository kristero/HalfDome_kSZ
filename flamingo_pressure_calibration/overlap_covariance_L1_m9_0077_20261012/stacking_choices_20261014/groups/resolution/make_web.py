"""web_variants.json of the 'resolution' group (website format of stacking_choices_20261014).

Reference records are recomputed with stackcore.stack_bin (identical in every group).  Mask-only settings carry the
reference values with their own 'resolved' flags (P, E); per-halo settings carry their own stacks (P, E, G, T_e, K);
'fit4_*' variants are the B12 (P) / B16 (E) fits to 4 R200c with each setting's inner limit (value = shell-averaged
model, err = paired-bootstrap SD of the model, q16/q84 null, resolved = shells used by the fit).  Bins where the
setting leaves fewer than 6 shells (analysis.fitting.MIN_SHELLS) have no fit: their records are present with null
values and resolved = 0, so every variant has a record for every bin of the scheme.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import res_common as R  # noqa: E402

sc = R.sc
ARR = np.load(R.GROUP / "work" / "arrays.npz")
EDGES = sc.edges(R.GRID)
XV = R.x_volume(EDGES)
BINS = sc.mass_bins(R.SCHEME)
MASK_ONLY = ("strict", "diagnostic", "frac", "none")
FIT_WEB = ("ref", "mc0", "mc100", "mc500", "hf1", "hf3", "frac95", "per_halo", "none")
KEY_TEMPLATE = "{field}|{estimator}|removed|all|all|fit|{norm}  (estimator geometric for P, E, G; weighted for T_e, K)"


def lst(a, ints=False):
    a = np.asarray(a)
    if ints:
        return [int(v) for v in a]
    return [R.sig(v) for v in a]


def rec(vid, field, b, value, err, q16, q84, n_used, resolved):
    return {"variant": vid, "field": field, "scheme": R.SCHEME, "bin": b["label"], "logm": R.sig(b["mean_log_mass"]),
            "N": int(b["N"]), "N_eff": R.sig(b["N_eff"]), "x_unit": "R200c", "edges": lst(EDGES), "x": lst(XV),
            "value": lst(value), "err": lst(err), "q16": lst(q16), "q84": lst(q84),
            "n_used": lst(n_used, ints=True), "resolved": lst(np.asarray(resolved).astype(int), ints=True)}


def settings_block(o):
    d = {"key_template": KEY_TEMPLATE, "statistic": "mean", "rule": o["rule"], "min_count": o["min_count"],
         "softening_factor": o["softening_factor"], "hsml_factor": o["hsml_factor"], "min_edge": o["min_edge"],
         "scheme": R.SCHEME, "sample": "all haloes of the bin (4,588 FLAMINGO L1_m9 z = 0 centrals, inclusion weights)"}
    if "frac_min" in o:
        d["frac_min"] = o["frac_min"]
    return d


def main():
    variants, records = [], []
    # ------------------------------------------------------------------ reference (core driver)
    ref = R.SETTING_BY_SHORT["ref"]
    variants.append({"id": "ref", "dimension": "Reference", "label": "Reference", "description": ref["description"],
                     "fields": list(R.FIELDS_WEB_ALL), "scheme": R.SCHEME, "grid": R.GRID,
                     "settings": settings_block(ref["opts"])})
    for f in R.FIELDS_WEB_ALL:
        key = R.field_key(f)
        for b in BINS:
            r = sc.stack_bin(key, b)
            records.append(rec("ref", f, b, r["value"], r["err"], r["q16"], r["q84"], r["n_used"], r["resolved"]))
    # ------------------------------------------------------------------ stack variants
    for s in R.SETTINGS:
        if s["short"] == "ref":
            continue
        o = s["opts"]
        fields = ("P", "E") if o["rule"] in MASK_ONLY else R.FIELDS_WEB_ALL
        variants.append({"id": s["id"], "dimension": s["dimension"], "label": s["label"],
                         "description": s["description"], "fields": list(fields), "scheme": R.SCHEME,
                         "grid": R.GRID, "settings": settings_block(o)})
        for f in fields:
            for b in BINS:
                p = f"stack|{s['short']}|{f}|{b['label']}|"
                records.append(rec(s["id"], f, b, ARR[p + "value"], ARR[p + "err"], ARR[p + "q16"], ARR[p + "q84"],
                                   ARR[p + "n_used"], ARR[p + "resolved"]))
    # ------------------------------------------------------------------ fitted-profile variants
    for short in FIT_WEB:
        s = R.SETTING_BY_SHORT[short]
        vid = f"{R.GROUP_NAME}:fit4_{short}"
        variants.append({
            "id": vid, "dimension": "Resolution: B12/B16 fit to 4 R200c (inner limit from the setting)",
            "label": f"fit 4R: {s['label']}",
            "description": (f"B12 (pressure) or B16 (electrons) fitted to the stack's kept shells out to 4 R200c with the "
                            f"'{s['label']}' setting (radii234 diagonal objective), shown as shell averages; shells "
                            f"inside the first kept edge are the extrapolated model (resolved = 0). Bins with fewer "
                            f"than 6 kept shells have no fit (empty record)."),
            "fields": ["P", "E"], "scheme": R.SCHEME, "grid": R.GRID,
            "settings": dict(settings_block(s["opts"]), fit="B12 (P) / B16 (E), objective (a), outer limit 4 R200c")})
        for f in ("P", "E"):
            for b in BINS:
                key = f"fit|{short}|{f}|4|{b['label']}|"
                sp = f"stack|{short}|{f}|{b['label']}|"
                nan = np.full(len(XV), np.nan)
                if key + "model" not in ARR:        # no fit in this bin: empty record (null values, resolved = 0)
                    records.append(rec(vid, f, b, nan, nan, nan, nan, ARR[sp + "n_used"], np.zeros(len(XV), int)))
                    continue
                v = ARR[sp + "value"]
                used = ARR[sp + "resolved"].astype(bool) & (EDGES[1:] <= 4 + 1e-12) & np.isfinite(v) & (v > 0)
                records.append(rec(vid, f, b, ARR[key + "model"], ARR[key + "model_err"], nan, nan,
                                   ARR[sp + "n_used"], used))
    out = {"group": R.GROUP_NAME, "variants": variants, "records": records}
    path = R.GROUP / "web_variants.json"
    path.write_text(json.dumps(out, separators=(",", ":"), allow_nan=False))
    print(f"{len(variants)} variants, {len(records)} records, {path.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
