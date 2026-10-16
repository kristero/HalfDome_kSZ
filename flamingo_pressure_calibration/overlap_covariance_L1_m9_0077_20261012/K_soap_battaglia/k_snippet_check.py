"""Check that report/xgpaint_refit_snippet.jl runs as written with the validated XGPaint and that the models it builds
give the same 3D profiles as k_fit_core for the same fits (local, small).

  python -B k_snippet_check.py [--snippet report/xgpaint_refit_snippet.jl] [--out report/xgpaint_snippet_check.json]
Runs Julia 1.12.2 (/home/kn18001/.julia/juliaup/julia-1.12.2+0.x64.linux.gnu/bin/julia, depot /home/kn18001/.julia,
XGPaint dev checkout branch 'cluster') as the depot owner: `include(snippet)`, then for every `model_<label>` defined,
XGPaint.get_params(model, M Msun, z) and P0 x [break factor] x generalized_nfw(x, x_c, alpha, beta_int, gamma) at
M200c = 10^12.8, 10^13.5, 10^14, 10^15 Msun, z = 0, 0.5, 1, 2 and x = 0.1, 0.5, 1, 2, 3.  Python side: the same fit
from results/k_params.json through k_fit_core.law_profile (P: divided by 0.5176, since XGPaint's P0 x gNFW is the
thermal pressure P_th/P200; G: rho_gas/(f_b rho_c)).  Pass: the snippet runs and max relative difference < 1e-7
(the snippet prints 10 significant digits).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import k_fit_core as K  # noqa: E402

JULIA = "/home/kn18001/.julia/juliaup/julia-1.12.2+0.x64.linux.gnu/bin/julia"
DEPOT = "/home/kn18001/.julia"
MASSES = (10 ** 12.8, 10 ** 13.5, 1e14, 1e15)
ZS = (0.0, 0.5, 1.0, 2.0)
XS = (0.1, 0.5, 1.0, 2.0, 3.0)

DRIVER = r'''
using XGPaint, Unitful, UnitfulAstro
snippet = ARGS[1]
ok = try
    include(snippet)
    println("SNIPPET_OK")
    true
catch err
    println("SNIPPET_ERROR: ", replace(sprint(showerror, err)[1:min(end, 600)], "\n" => " "))
    false
end
if ok
    names_ = [n for n in names(Main; all=true) if startswith(String(n), "model_")]
    for n in names_
        model = getfield(Main, n)
        for M in MASSES, z in ZS
            p = XGPaint.get_params(model, M * XGPaint.M_sun, z)
            bf = (model isa XGPaint.BreakModel && M < model.M_break) ? (M / model.M_break)^model.alpha_break : 1.0
            vals = [p.P₀ * bf * XGPaint.generalized_nfw(x, p.xc, p.α, p.β, p.γ) for x in XS]
            println("VAL ", String(n), " ", M, " ", z, " ", join(string.(vals), " "))
        end
    end
end
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snippet", default=str(HERE / "report" / "xgpaint_refit_snippet.jl"))
    ap.add_argument("--params", default=str(HERE / "results" / "k_params.json"))
    ap.add_argument("--out", default=str(HERE / "report" / "xgpaint_snippet_check.json"))
    a = ap.parse_args()
    work = HERE / "parity_work"
    work.mkdir(exist_ok=True)
    os.chmod(work, 0o777)
    drv = work / "snippet_driver.jl"
    head = (f"const MASSES = [{', '.join(repr(float(m)) for m in MASSES)}]\n"
            f"const ZS = [{', '.join(repr(float(z)) for z in ZS)}]\nconst XS = [{', '.join(repr(float(x)) for x in XS)}]\n")
    drv.write_text(head + DRIVER)
    snip = work / "snippet_under_test.jl"
    snip.write_text(Path(a.snippet).read_text())
    for f in (drv, snip):
        os.chmod(f, 0o666)
    cmd = ["runuser", "-u", "kn18001", "--", "env", f"JULIA_DEPOT_PATH={DEPOT}", "JULIA_NUM_THREADS=1", JULIA,
           "--startup-file=no", str(drv), str(snip)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    text = r.stdout
    sp = Path(a.snippet).resolve()
    res = {"snippet": str(sp.relative_to(HERE)) if sp.is_relative_to(HERE) else str(sp), "julia": JULIA, "xgpaint": "/home/kn18001/.julia/dev/XGPaint (branch cluster)",
           "returncode": r.returncode, "runs": "SNIPPET_OK" in text, "stderr_tail": r.stderr[-1500:] if r.returncode else ""}
    err = re.search(r"SNIPPET_ERROR: (.*)", text)
    if err:
        res["error"] = err.group(1)
    # which fit each model_<label> is: the comment line '# <label>: <key>  (sigma_d ...' just above it
    labels = dict(re.findall(r"^# (\S+): (\S+\|\S+\|\S+)\s+\(sigma_d", Path(a.snippet).read_text(), flags=re.M))
    P = json.loads(Path(a.params).read_text())["fits"]
    worst, per = 0.0, {}
    for line in text.splitlines():
        if not line.startswith("VAL "):
            continue
        parts = line.split()
        name, M, z = parts[1], float(parts[2]), float(parts[3])
        xv = np.array([float(v) for v in parts[4:]])
        key = labels.get(name[len("model_"):])
        if key is None:
            continue
        f = P[key]
        form = "B12" if f["field"] in ("P", "Pagn") else "B16"
        law = K.Law(form, f["parametrization"])
        F = K.law_profile(law, np.array(f["theta"]), np.array(XS), M, z)
        if form == "B12":
            F = F / K.E_OVER_TH
        d = float(np.max(np.abs(xv / F - 1)))
        per.setdefault(name, {"fit": key, "max_rel_diff": 0.0, "n": 0})
        per[name]["max_rel_diff"] = max(per[name]["max_rel_diff"], d)
        per[name]["n"] += len(XS)
        worst = max(worst, d)
    res.update(models=per, max_rel_diff=worst, masses=list(MASSES), redshifts=list(ZS), x=list(XS),
               passed=bool(res["runs"] and per and worst < 1e-7))
    Path(a.out).write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
