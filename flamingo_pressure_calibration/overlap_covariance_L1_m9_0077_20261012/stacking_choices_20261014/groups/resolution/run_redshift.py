"""Redshift dependence of the resolution criteria (group 'resolution').

Softening convention (FLAMINGO L1_m9 / L2p8_m9 run parameters, read from probe_g4.json snapshot_describe and
metadata_plan.json): eps_phys = min(a x 0.0223 cMpc, 0.0057 pMpc), i.e. fixed at 5.7 physical kpc for z < 2.91
and fixed in comoving units (22.3 ckpc) above.

1. Analytic: at fixed M200c, R200c ~ E(z)^(-2/3) (physical), so eps/R200c grows as E(z)^(2/3) while eps is
   physical; the reference criterion x >= 4 eps / R200c is evaluated for the L1_m9 bins at z = 0.5, 1, 2, 3.
2. Empirical: the L2p8_m9 stack products (same particle mass as L1_m9; notebook_cache/stacks, z = 0-4, 31 shells
   to 1.6 R200c) store per-halo shell counts and smoothing-length histograms (h a / R200c, 0.05 dex). The same
   per-halo criteria (count >= 20; x_in >= 2 x upper-median h; x_in >= max(0.04, 4 eps(z)/R200c)) give the first
   kept shell per 0.2-dex bin and redshift (strict = every halo; frac95 = >= 95% of haloes; the samples are
   subsampled to <= 200 haloes per bin, so strict is quoted with N).
Writes work/redshift.json.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import res_common as R  # noqa: E402

sc = R.sc
STACKS = Path("/home/cbllover/HalfDome/flamingo_pressure_calibration/notebook_cache/stacks")
EPS_C, EPS_MAX = 0.0223, 0.0057
OMEGA_M = 0.306


def softening(a):
    return min(a * EPS_C, EPS_MAX)


def E(z):
    return np.sqrt(OMEGA_M * (1 + z)**3 + 1 - OMEGA_M)


def probe_softening():
    d = json.loads((R.STUDY / "probe_g4.json").read_text())["snapshot_describe"]
    return {"softening_comoving_mpc": d["softening_comoving_mpc"],
            "softening_max_physical_mpc": d["softening_max_physical_mpc"], "source": d["softening_source"],
            "metadata_plan_softening_physical_mpc": json.loads((R.STUDY / "metadata_plan.json").read_text())[
                "softening_physical_mpc"]}


def analytic():
    tab = sc.halo_table()
    e = sc.edges(R.GRID)
    out = {}
    for z in (0.0, 0.5, 1.0, 2.0, 3.0, 4.0):
        a = 1 / (1 + z)
        fac = softening(a) / EPS_MAX * E(z)**(2 / 3)          # eps(z)/R200c(z) / (eps0/R200c(0)) at fixed M
        res = {}
        for b in sc.mass_bins(R.SCHEME):
            eps_r = tab["eps_over_r200c"][b["rows"]] * fac
            lim = max(0.04, 4 * float(eps_r.max()))
            idx = np.flatnonzero(e[:-1] >= lim)
            res[b["label"]] = {"softening_limit_x": lim, "first_edge_softening_only": float(e[idx[0]])}
        out[f"{z:g}"] = {"eps_over_R_factor_vs_z0": fac, "E_z": float(E(z)), "bins": res}
    return out


def empirical():
    out = {}
    for f in sorted(glob.glob(str(STACKS / "L2p8_m9_*_stack_products.npz"))):
        d = np.load(f)
        z = float(d["redshift"])
        a = float(d["scale_factor"])
        edges = d["edges"]
        x = edges[:-1]
        hed = d["hsml_log_edges"]
        hist = d["hsml_hist"].astype(float)                       # [n, S, 60]
        cum = np.cumsum(hist, axis=2)
        tot = cum[..., -1:]
        idx = np.argmax(cum >= tot / 2, axis=2)
        h_up = np.where(tot[..., 0] > 0, 10**hed[idx + 1], np.nan)  # upper edge of the median bin (conservative)
        n3 = d["n3"]
        r200 = d["cat_r_200_crit_mpc"]
        eps_r = softening(a) / r200
        with np.errstate(invalid="ignore"):
            c_soft = x[None, :] >= np.maximum(0.04, 4 * eps_r[:, None])
            c_count = n3 >= 20
            c_hsml = x[None, :] >= 2 * h_up
        res_all = c_soft & c_count & c_hsml
        logm = np.log10(d["cat_m200c_msun"])
        bins = {}
        for k in range(len(d["bin_edges"]) - 1):
            lo, hi = float(d["bin_edges"][k]), float(d["bin_edges"][k + 1])
            rows = np.flatnonzero((logm >= lo) & (logm < hi))
            if len(rows) < 10:
                continue
            res = res_all[rows]
            frac = res.mean(axis=0)

            def fe(m):
                i = np.flatnonzero(m)
                return float(x[i[0]]) if len(i) else None

            crit = {}
            for name, c in (("count", c_count), ("softening", c_soft), ("smoothing_length", c_hsml)):
                crit[name] = {"strict": fe(c[rows].all(axis=0)), "frac95": fe(c[rows].mean(axis=0) >= 0.95)}
            s_ref = int(np.searchsorted(x, 0.1))                    # shell starting at ~0.1 R200c
            bins[f"{lo:.1f}-{hi:.1f}"] = {
                "N": int(len(rows)), "strict_first_edge": fe(res.all(axis=0)),
                "frac95_first_edge": fe(frac >= 0.95), "frac50_first_edge": fe(frac >= 0.5),
                "by_criterion": crit,
                "median_count_at_x0.1": float(np.median(n3[rows, s_ref])), "x_ref": float(x[s_ref]),
                "median_x_over_h_at_x0.1": float(np.nanmedian(x[s_ref] / h_up[rows, s_ref])),
                "eps_over_R200c_median": float(np.median(eps_r[rows]))}
        out[f"{z:g}"] = {"snapshot": int(d["snapshot"]), "a": a, "softening_phys_mpc": softening(a), "bins": bins}
    return out


def main():
    res = {"softening": probe_softening(), "analytic_L1m9_bins": analytic(), "L2p8_empirical": empirical(),
           "notes": ["eps_phys = min(a eps_c, eps_max) = 5.7 pkpc for z < 2.91 (a > 0.2556); 22.3 a ckpc above",
                     "analytic: at fixed M200c, eps/R200c ~ E(z)^(2/3) below z = 2.91 (estimate, self-similar)",
                     "L2p8 'upper median' h = upper edge of the 0.05-dex median bin; L1_m9 cache uses 0.0156-dex bins",
                     "L2p8 bins are subsampled (<= 200 haloes), so 'strict' is less strict than with the full sample"]}
    R.write_json(R.GROUP / "work" / "redshift.json", res, n=5)
    for z, v in res["L2p8_empirical"].items():
        print(z, {k: (b["N"], b["strict_first_edge"], b["frac95_first_edge"], round(b["median_x_over_h_at_x0.1"], 2),
                      b["median_count_at_x0.1"]) for k, b in v["bins"].items()})
    for z, v in res["analytic_L1m9_bins"].items():
        print(z, round(v["eps_over_R_factor_vs_z0"], 3), {k: round(b["softening_limit_x"], 3) for k, b in v["bins"].items()})


if __name__ == "__main__":
    main()
