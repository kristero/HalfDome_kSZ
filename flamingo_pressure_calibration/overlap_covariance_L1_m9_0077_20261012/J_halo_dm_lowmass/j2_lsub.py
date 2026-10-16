"""Test J2: why objective (L) has no fit in some bins (local, numpy only, < 0.2 GB; reads inputs/j2_fitdata.npz and
results_j2/fits.json; writes post_j2/J2_L_substitutions.json).

j2_fits.one(...) gives no (L) fit when the Hartlap factor H = (N_h - p - 2)/(N_h - 1) is undefined (N_h <= p + 2) or
when p < 4, and analysis.fitting.fit_general returns no fit when p < max(MIN_SHELLS, n_par + 1) = max(6, 4) = 6
(MIN_SHELLS read from analysis/fitting.py; n_par = 3 for B16).  N_h is the number of haloes of the bin and p the number
of fit shells that pass the bin's fit mask, lie inside R_fit (outer edge <= R_fit) and have F > 0 in every halo of the
bin (Lee22's ln F covariance needs that).
Such bins are painted with their objective-(a) fit ('L*').  This script recomputes N_h and p for every (definition,
R_fit, bin) with the same rule, and checks that (i) the bins without an (L) fit are exactly the ones in results_j2/
fits.json with a null (L) record and (ii) for every existing (L) fit N_h and p equal the fit record's 'N' and 'p'.
Usage: python -B j2_lsub.py
"""
from __future__ import annotations

import re

import j2_common as C
import numpy as np


def main():
    d = C.load_npz(C.J2_INPUT, keys=["f35_edges", "mass_bin", "E_a2ii_35", "E_hb_35", "mask35_a2ii", "mask35_hb"])
    F35 = np.asarray(d["f35_edges"], float)
    mb = d["mass_bin"]
    fits = C.read_json(C.J2_RESULTS / "fits.json")["fits"]
    min_shells = int(re.search(r"^MIN_SHELLS = (\d+)", (C.ROOT / "analysis" / "fitting.py").read_text(), re.M).group(1))
    p_min = max(4, min_shells, 3 + 1)
    out, mismatches = {}, []
    for D, spec in C.DEFS.items():
        sel = spec["sel"]
        vals = d[f"E_{sel}_35"]
        for rf in spec["rfit"]:
            ie = int(np.argmin(np.abs(F35 - rf)))
            assert abs(F35[ie] - rf) < 1e-6, (rf, F35[ie])
            within = F35[1:] <= F35[ie] + 1e-12
            rows_out = {}
            for b in C.J_BINS:
                v = vals[mb == b]
                okL = np.all(np.isfinite(v) & (v > 0), axis=0)
                base = np.asarray(d[f"mask35_{sel}"][b], bool)
                p = int(np.sum(base & within & okL))
                n_h = int(v.shape[0])
                reason = None
                if n_h <= p + 2:
                    reason = "N_h <= p + 2"
                elif p < p_min:
                    reason = f"p < {p_min}"
                rec = fits.get(f"{D}|L|{rf:g}|{b}")
                if (rec is None) != (reason is not None):
                    mismatches.append({"key": f"{D}|L|{rf:g}|{b}", "reason": reason, "record_is_null": rec is None})
                if rec is not None and (int(rec["N"]) != n_h or int(rec["p"]) != p):
                    mismatches.append({"key": f"{D}|L|{rf:g}|{b}", "N_h": n_h, "p": p, "rec_N": rec["N"],
                                       "rec_p": rec["p"]})
                if reason is not None:
                    rows_out[C.bin_label(b)] = {"engine_bin": b, "N_h": n_h, "p": p,
                                                "p_mask_and_Rfit_only": int(np.sum(base & within)), "reason": reason}
            out[f"{D}|Rf{rf:g}"] = rows_out
    res = {"definitions": __doc__, "p_min": p_min, "min_shells_analysis_fitting": min_shells, "missing_L_fits": out,
           "mismatches_vs_fits_json": mismatches, "check_passed": not mismatches}
    C.write_json(C.HERE / "post_j2" / "J2_L_substitutions.json", res)
    for k, v in out.items():
        print(k, {b: (r["N_h"], r["p"], r["reason"]) for b, r in v.items()})
    print("mismatches:", mismatches)


if __name__ == "__main__":
    main()
