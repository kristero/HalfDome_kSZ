"""Mass-binning group: does a curvature-aware interpolant remove the node-spacing (chord) error of wide bins?

Continuous painting (mbcommon.continuous_paint) of the calibrated haloes through the M200c_0.1 / 0.2 / 0.3 / 0.5 node
stacks with ln F (a) linear and (b) monotone-cubic (PCHIP) in log M200c, compared with the true bin means and with the
0.2-dex nodes painted the same way.  Reads work/scheme_stacks.npz (run_schemes.py).
Output: work/res_interp_test.json.
"""
import numpy as np

from mbcommon import CAL_LABELS, OBS_FIELDS, REF_SCHEME, WORK, continuous_paint, dump, impacts, sc

ST = np.load(WORK / "scheme_stacks.npz")
cal_bins = [sc.find_bin(REF_SCHEME, lab) for lab in CAL_LABELS]
WIDTHS = ("M200c_0.1", "M200c_0.2", "M200c_0.3", "M200c_0.5")
res = {"notes": (
    "continuous painting of the 10 calibrated bins' haloes at their own log M200c through each scheme's node stacks "
    "(nodes at the inclusion-weighted mean log M200c); interp 'linear' = ln F linear in log M between nodes, 'pchip' = "
    "monotone cubic Hermite (PCHIP) of ln F in log M; both extrapolate linearly beyond the end nodes.  vs_truth: painted "
    "bin means vs the true bin means (reference stacks); vs_0.2dex_same_interp: vs the 0.2-dex nodes painted with the "
    "same interpolant.  Per cent, first order (sc.impact_from_stacks); err = paired global-bootstrap SD (census design)."),
       "linear": {}, "pchip": {}}
for interp in ("linear", "pchip"):
    for field in OBS_FIELDS:
        labels = list(ST[f"{field}__{REF_SCHEME}__labels"])
        idx = [labels.index(c) for c in CAL_LABELS]
        Fr = ST[f"{field}__{REF_SCHEME}__value"][idx]
        Br = np.moveaxis(ST[f"{field}__{REF_SCHEME}__boot"][idx], 0, 1)
        painted = {}
        for S in WIDTHS:
            Pv, Pb, _ = continuous_paint(ST[f"{field}__{S}__m200"], ST[f"{field}__{S}__value"],
                                         ST[f"{field}__{S}__boot"], cal_bins, interp=interp)
            painted[S] = (Pv, Pb)
            res[interp].setdefault(S, {})[field] = {"vs_truth": impacts(Pv, Fr, Pb, Br, field)}
        P0, P0b = painted[REF_SCHEME]
        for S in WIDTHS:
            res[interp][S][field]["vs_0.2dex_same_interp"] = impacts(*painted[S][:1], P0, painted[S][1], P0b, field)
    print("done", interp, flush=True)
dump(res, WORK / "res_interp_test.json")
