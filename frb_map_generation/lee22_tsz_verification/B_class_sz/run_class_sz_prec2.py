#!/usr/bin/env python3
"""Third CLASS-SZ batch (2026-09-29 continuation; run after run_class_sz_extra.py, one classy_sz process at a time).

Finding of run_class_sz_extra.py: at the default precision of the 2026-09-25 truncation-series dictionary
(n_z_pressure_profile = 100 over ln(1+z) from z = 0.0021, i.e. first table nodes at z = 0.0021 and 0.016) CLASS-SZ's
1-halo term is too low at ell < 150 (by 25-35 % at ell = 30, 3 % (B12) to 15 % (Lee22 low piece) at ell = 100),
because y_ell at fixed ell is interpolated across z nodes whose angular scale R200/D_A differs by 7x. The full
high-precision run (HIPREC, ~8 min per run) agrees with the Python halo model to < 1 % at ell >= 100.

This batch:
  1. MEDPREC (only the pressure-profile table refined: n_z_pressure_profile = 400, n_m_pressure_profile = 150) for B12
     all and Lee22 low; accepted if within ACCEPT of the corresponding HIPREC run at 30 <= ell <= 6000, else HIPREC is
     used for step 2;
  2. the chosen precision for every run that enters the map-level comparison: B12 low/high, Lee22 high/calib,
     M1 low/high/calib; with MEDPREC also the mass bins and redshift bins of the CLASS-SZ decomposition check.
Outputs OUT/class_sz/<name>_<prec>.npz; OUT/class_sz/precision_choice.json
"""
import json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, "/home/cbllover/HalfDome/frb_map_generation/lee22_tsz_verification/B_class_sz")
import common as C  # noqa: E402
import run_class_sz as R  # noqa: E402
import run_class_sz_extra as X  # noqa: E402

MEDPREC = {"n_z_pressure_profile": 400, "n_m_pressure_profile": 150}
ACCEPT = 0.01
OUTD = Path(C.OUT) / "class_sz"


def with_prec(spec, prec, tag):
    s = dict(spec)
    s["name"] = spec["name"] + "_" + tag
    s["prof"] = dict(spec["prof"], **prec)
    s.pop("extract", None); s.pop("tables", None)
    return s


def maxdev(a, b):
    da, db = np.load(OUTD / f"{a}.npz"), np.load(OUTD / f"{b}.npz")
    ell = da["ell"]; s = (ell >= 30) & (ell <= 6000)
    ta, tb = da["cl_1h"] + da["cl_2h"], db["cl_1h"] + db["cl_2h"]
    return float(np.max(np.abs(ta[s] / tb[s] - 1)))


if __name__ == "__main__":
    base = {s["name"]: s for s in R.runs()}
    for nm in ("b12_paint_all", "lee22_paint_low"):
        R.run_one(with_prec(base[nm], MEDPREC, "medprec"))
    dev = {nm: maxdev(nm + "_medprec", nm + "_hiprec") for nm in ("b12_paint_all", "lee22_paint_low")}
    use_med = all(v < ACCEPT for v in dev.values())
    prec, tag = (MEDPREC, "medprec") if use_med else (X.HIPREC, "hiprec")
    json.dump(dict(max_abs_dev_medprec_vs_hiprec_ell30_6000=dev, accept=ACCEPT, chosen=tag, MEDPREC=MEDPREC,
                   HIPREC=X.HIPREC), open(OUTD / "precision_choice.json", "w"), indent=1)
    print("medprec vs hiprec:", dev, "-> using", tag, flush=True)
    todo = ["b12_paint_low", "b12_paint_high", "lee22_paint_high", "lee22_paint_calib",
            "lee22m1_paint_low", "lee22m1_paint_high", "lee22m1_paint_calib"]
    if tag == "medprec":           # bins only at the cheap precision (HIPREC would take ~8 min x 30 runs)
        todo += [n for n in base if "_mbin" in n or "_zbin" in n]
    for nm in todo:
        R.run_one(with_prec(base[nm], prec, tag))
    print("ALL DONE", flush=True)
