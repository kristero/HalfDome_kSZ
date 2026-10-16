"""Add the convergence status of every fit to the parameter file (local; no refit).

  python -B k_params_status.py

Reads the maas-produced parameter file (kept unchanged as attempts/run3_20261015/k_params_maas.json; copied there on the
first call) and results/k7_fit_audit.json (k_fit_audit.py), and writes results/k_params.json with, for every fit, a
'fit_status' entry (converged, least-squares status of each sigma_d iteration, chi^2/nu at the quoted sigma_d, the sigma_d
that gives chi^2/nu = 1 at the final theta, usable flag and note) and the 'electron_factors' block that the maas file's
B16 convention refers to (from results/k4_electron_factors.json).  Every maas field is kept unchanged.  The
results/SHA256SUMS line of k_params.json is updated and the change is recorded in results/k7_params_provenance.json.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = HERE / "results"
ORIG = HERE / "attempts" / "run3_20261015" / "k_params_maas.json"
MAAS_SHA = "4471f4f47e546df69088a90fcadb3e0cecf69cc540f8cc1a4dbcdd94962400e4"   # results/SHA256SUMS line written on maas

NOT_USABLE = {
    "G|R1|free": "not usable: least squares stopped at max_nfev = 600 in every sigma_d iteration (not converged), x_c at its "
                 "upper bound 10 at 1e14 Msun and gamma down to -2.99 at the quadrature nodes (degenerate)",
}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    if not ORIG.exists():
        assert sha(R / "k_params.json") == MAAS_SHA, "results/k_params.json is not the maas file"
        shutil.copy2(R / "k_params.json", ORIG)
    assert sha(ORIG) == MAAS_SHA
    P = json.loads(ORIG.read_text())
    A = json.loads((R / "k7_fit_audit.json").read_text())["convergence"]
    for key, fit in P["fits"].items():
        a = A[key]
        fit["fit_status"] = {
            "converged": a["converged"],
            "least_squares_status_per_sigma_iteration": [it["status"] for it in a["iterations"]],
            "nfev_per_sigma_iteration": [it["nfev"] for it in a["iterations"]],
            "nfev_final_solve": a["nfev_final_solve"], "max_nfev": a["max_nfev"],
            "chi2_nu_at_quoted_sigma_d": a["chi2_nu_at_reported_sigma_d"],
            "sigma_d_chi2nu1_at_final_theta": a["sigma_d_chi2nu1_at_final_theta"],
            "usable": key not in NOT_USABLE,
            "note": NOT_USABLE.get(key, "converged (scipy least_squares success); chi^2/nu = 1 at the quoted sigma_d" if a["converged"]
                                   else "not converged"),
        }
    # the maas file's B16 convention refers to 'electron_factors', which it did not contain: add them (k4_electron_factors.json)
    P["electron_factors"] = json.loads((R / "k4_electron_factors.json").read_text())
    P["conventions"]["fit_status"] = ("added locally by k_params_status.py from results/k7_fit_audit.json (status codes: scipy "
                                      "least_squares; 0 = stopped at max_nfev, 1-4 = converged); all other fields are the maas output")
    out = R / "k_params.json"
    out.write_text(json.dumps(P, indent=1) + "\n")
    new = sha(out)
    lines = (R / "SHA256SUMS").read_text().splitlines()
    lines = [f"{new}  k_params.json" if ln.endswith("  k_params.json") else ln for ln in lines]
    (R / "SHA256SUMS").write_text("\n".join(lines) + "\n")
    (R / "k7_params_provenance.json").write_text(json.dumps({
        "file": "results/k_params.json", "maas_sha256": MAAS_SHA, "maas_copy": str(ORIG.relative_to(HERE)),
        "local_sha256": new, "added": "fits[*].fit_status, conventions.fit_status, electron_factors (= results/k4_electron_factors.json)", "script": "k_params_status.py",
        "when": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "note": "results/SHA256SUMS (written on maas) has its k_params.json line replaced by the local sha256; every other line is the maas one"},
        indent=1) + "\n")
    print("wrote", out, new)


if __name__ == "__main__":
    main()
