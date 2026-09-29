#!/usr/bin/env python3
"""Validate the Python halo model (halo_model.py) against CLASS-SZ on Battaglia12 (all masses, and the two mass
pieces split at 10^14.8/h) and on the exactly representable Lee22 pieces (low, high, calib), 1h and 2h separately.
Output: OUT/halo_model/validation.json and validation_curves.npz."""
import json, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, "/home/cbllover/HalfDome/frb_map_generation/lee22_tsz_verification/B_class_sz")
import common as C  # noqa: E402
import halo_model as HM  # noqa: E402

OUTD = Path(C.OUT) / "halo_model"; OUTD.mkdir(parents=True, exist_ok=True)
CS = Path(C.OUT) / "class_sz"
ELLS = (30, 100, 300, 1000, 3000, 6000)

CASES = [  # (class_sz run, python model, kwargs)
    ("b12_paint_all", "b12", {}),
    ("b12_paint_low", "b12", dict(m_hi=C.LEE_CLIP_HINV)),
    ("b12_paint_high", "b12", dict(m_lo=C.LEE_CLIP_HINV)),
    ("lee22_paint_low", "lee22", dict(m_hi=C.LEE_CLIP_HINV)),
    ("lee22_paint_high", "lee22", dict(m_lo=C.LEE_CLIP_HINV)),
    ("lee22_paint_calib", "lee22", dict(m_lo=C.LEE_CALIB_MMIN_HINV, m_hi=C.LEE_CLIP_HINV, z_hi=C.LEE_CALIB_ZMAX)),
    ("lee22m1_paint_low", "lee22_m1", dict(m_hi=C.LEE_CLIP_HINV)),
    ("lee22_paint_high_noclip", "lee22_noclip", dict(m_lo=C.LEE_CLIP_HINV)),
]

if __name__ == "__main__":
    only = sys.argv[1:]
    ing = HM.Ingredients()
    res, curves = {}, {}
    for run, model, kw in CASES:
        if only and run not in only:
            continue
        f = CS / f"{run}.npz"
        if not f.exists():
            print("missing", f); continue
        d = np.load(f)
        t0 = time.time()
        r = HM.halo_model_cl(ing, model, kind="sphere", X=4.0, **kw)
        dt = time.time() - t0
        ell = d["ell"]
        rr = {}
        for L in ELLS:
            i = int(np.argmin(np.abs(ell - L)))
            rr[str(L)] = dict(ell=float(ell[i]), ratio_1h=float(r["cl_1h"][i] / d["cl_1h"][i]),
                              ratio_2h=float(r["cl_2h"][i] / d["cl_2h"][i]),
                              ratio_tot=float((r["cl_1h"][i] + r["cl_2h"][i]) / (d["cl_1h"][i] + d["cl_2h"][i])))
        s = (ell >= 30) & (ell <= 6000)
        tot = (r["cl_1h"] + r["cl_2h"]) / (d["cl_1h"] + d["cl_2h"])
        rr["max_abs_dev_tot_ell30_6000"] = float(np.max(np.abs(tot[s] - 1)))
        rr["max_abs_dev_1h_ell30_6000"] = float(np.max(np.abs(r["cl_1h"][s] / d["cl_1h"][s] - 1)))
        rr["seconds"] = dt
        res[run] = rr
        curves[run + "__py_1h"] = r["cl_1h"]; curves[run + "__py_2h"] = r["cl_2h"]
        curves[run + "__cs_1h"] = d["cl_1h"]; curves[run + "__cs_2h"] = d["cl_2h"]
        print(run, model, f"{dt:.0f}s", {k: (v["ratio_1h"], v["ratio_2h"]) if isinstance(v, dict) else v for k, v in rr.items()},
              flush=True)
    tag = "_".join(only) if only else "all"
    json.dump(res, open(OUTD / f"validation_{tag}.json", "w"), indent=1)
    np.savez(OUTD / f"validation_curves_{tag}.npz", ell=ing.ell, **curves)
