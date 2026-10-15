"""Local supplement: DM-PDF quantiles of the scatter-aware (lognormal amplitude) predictions that the maas pass
did not include, for the candidate painting recipes.  Light: deterministic shell averages only (no per-halo
truth), 2 workers, the locally rebuilt frozen production basis (results_local/work/basis_T_frozen.npz, f7).
Reproducibility check: grid_a_Rf4_Rc4 (deterministic and lognormal 1R) against the maas quantiles.
Output: results_local/scatter_pdf.json, results_local/scatter_pdf_errors.csv.
"""
from __future__ import annotations

import csv
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.setdefault("F_MODE", "full")
os.environ.setdefault("F_WORKERS", "2")
os.environ["F_RESULTS"] = str(HERE / "results")
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

import f_common as FC  # noqa: E402

FC.WORK = HERE / "results_local" / "work"
import f_pdf as FP  # noqa: E402

MODELS = ("flex_env_d", "flex_env_a", "flex_gnfw5_d", "flex_gnfw5_a", "grid_e_Rf4_Rc4", "grid_f_Rf4_Rc4",
          "flex_spline_a", "trunc_flux")
CHECK = "grid_a_Rf4_Rc4"


def gauss_hermite(sigma2, n=9):
    t, w = np.polynomial.hermite.hermgauss(n)
    s = np.sqrt(max(sigma2, 0.0))
    return np.exp(s * np.sqrt(2) * t - sigma2 / 2), w / np.sqrt(np.pi)


def main():
    t0 = time.perf_counter()
    mm = np.load(FC.PRODUCTS / "models_main.npz")
    labels = list(mm["labels"])
    tr = np.load(FC.PRODUCTS / "truth.npz")
    ts = FC.read_json(FC.PRODUCTS / "truth_summary.json")
    Qt = np.asarray(ts["pdf"]["truth_main"]["quantiles"], float)
    mom = {"lognormal_1R": tr["lognormal_main_1R"], "lognormal_4R": tr["lognormal_main_4R"]}
    ent = FP.Entries(42)
    hid = {}
    for m in (CHECK,) + MODELS:
        E = mm["E"][labels.index(m)]
        if m == CHECK:
            h = ent.new_hist(f"{m}|deterministic")
            hid[(m, "deterministic")] = h
            for c in range(FC.N_CAL):
                ent.add(c, E[c], h, 1.0)
        for name, mo in mom.items():
            if m == CHECK and name != "lognormal_1R":
                continue
            h = ent.new_hist(f"{m}|{name}")
            hid[(m, name)] = h
            for c in range(FC.N_CAL):
                amp, wq = gauss_hermite(mo[c, 1])
                ent.add(c, amp[:, None] * E[c][None, :], h, wq)
    ent.freeze()
    res = FP.run(ent, "T_frozen", include_fallback=True)
    q = {key: res["summaries"][h]["quantiles"] for key, h in hid.items()}
    # reproducibility against the maas values
    maas = {}
    with open(FC.TABLES / "F_errors_long.csv") as handle:
        for r in csv.DictReader(handle):
            if r["model"] == CHECK and r["observable"].startswith("dm_q"):
                maas[(r["prediction"], r["observable"])] = float(r["value"])
    dev = []
    for pred in ("deterministic", "lognormal_1R"):
        for p, name in enumerate(FC.Q_NAMES):
            dev.append(abs(q[(CHECK, pred)][p] / maas[(pred, f"dm_{name}")] - 1))
    rows = []
    for (m, pred), qq in q.items():
        for p, name in enumerate(FC.Q_NAMES):
            rows.append({"model": m, "prediction": pred, "observable": f"dm_{name}", "value": float(qq[p]),
                         "truth": float(Qt[p]), "rel_error": float(qq[p] / Qt[p] - 1)})
    FC.write_csv(HERE / "results_local" / "scatter_pdf_errors.csv", rows)
    FC.write_json(HERE / "results_local" / "scatter_pdf.json", {
        "models": list(MODELS), "check_model": CHECK, "reproducibility_max_abs_vs_maas": float(max(dev)),
        "timing": res["timing"], "wall_seconds": time.perf_counter() - t0,
        "note": "lognormal amplitude mixture, 9 Gauss-Hermite nodes per calibrated bin, sigma from per-halo Y / N_e "
                "moments (1R, 4R apertures); fallback haloes deposited in this pass"})
    FC.log("scatter pdf done", max(dev), time.perf_counter() - t0)


if __name__ == "__main__":
    main()
