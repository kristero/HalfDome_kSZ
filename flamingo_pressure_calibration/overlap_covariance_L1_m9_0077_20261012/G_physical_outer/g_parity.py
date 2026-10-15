"""G stage 'parity': the rebuilt basis and the DM-PDF machinery reproduce Test F's truth.

1. fast_deposit (g_pdf) equals f_pdf._deposit on random input (max |difference| must be 0).
2. Basis: the moment observables of the measured truth (per-halo second moments, main and A2-ii) and of the truth
   mean, recomputed with the rebuilt basis, against Test F's truth.npz (Test F's maas basis).
3. DM PDF: one pass with the published-fallback haloes included gives the fallback slot used by every later pass
   (results/work/pdf_fallback_g.npz) and the quantiles of the deterministic truth means (main, A2-ii), compared
   with Test F's truth_summary.json.
Output: results/parity.json.
"""
from __future__ import annotations

import f_basis as FB
import g_common as GC
import g_pdf as GP
import numpy as np

import f_pdf as FP  # noqa: E402


def main():
    timer = GC.Timer()
    out = {}
    out["deposit_identity_max_abs"] = GP.deposit_identity_check()
    pr = dict(np.load(GC.FBASE / "profiles.npz"))
    tr = dict(np.load(GC.FBASE / "truth.npz"))
    ts = GC.read_json(GC.FBASE / "truth_summary.json")
    B = FB.Basis("T")
    cal = GC.CAL_BINS
    worst = {}
    for sel in ("main", "a2ii"):
        P, E = pr[f"meanP_{sel}"][cal], pr[f"meanE_{sel}"][cal]
        S = dict(SPP=pr[f"SPP_{sel}"][cal], SEE=pr[f"SEE_{sel}"][cal], SPE=pr[f"SPE_{sel}"][cal])
        for tag, o in (("truth", B.observables(P, E, **S)), ("truthmean", B.observables(P, E))):
            for k, v in o.items():
                ref = tr.get(f"{tag}_{sel}__{k}")
                if ref is None:
                    continue
                with np.errstate(divide="ignore", invalid="ignore"):
                    rel = np.abs(np.asarray(v) / np.asarray(ref) - 1)
                worst[f"{tag}_{sel}__{k}"] = float(np.nanmax(rel))
    out["basis_vs_F_truth_max_rel"] = max(worst.values())
    out["basis_vs_F_truth_by_key"] = worst
    GC.log("basis parity", out["basis_vs_F_truth_max_rel"])
    # ---------------- PDF: fallback slot + truth-mean quantiles
    GP.install()
    ent = FP.Entries(B.ns)
    labels = []
    for sel in ("main", "a2ii"):
        h = ent.new_hist(f"truthmean_{sel}")
        labels.append(f"truthmean_{sel}")
        for c in range(GC.N_CAL):
            ent.add(c, pr[f"meanE_{sel}"][cal[c]], h, 1.0)
    ent.freeze()
    res = FP.run(ent, "T", include_fallback=True)
    np.savez_compressed(GC.WORK / "pdf_fallback_g.npz", hist=res["fallback"][0], overflow=res["fallback"][1])
    out["pdf_timing"] = res["timing"]
    out["truthmean_quantiles"] = {}
    for lab, s in zip(res["labels"], res["summaries"]):
        ref = np.asarray(ts["pdf"][lab]["quantiles"], float)
        q = np.asarray(s["quantiles"], float)
        out["truthmean_quantiles"][lab] = {"G": q, "F": ref, "max_rel": float(np.max(np.abs(q / ref - 1)))}
    out["pdf_vs_F_max_rel"] = max(v["max_rel"] for v in out["truthmean_quantiles"].values())
    fpath = GC.FBASE / "work" / "pdf_fallback.npz"
    if fpath.exists():                                      # smoke: F's own truth stage wrote one
        fb = np.load(fpath)
        out["fallback_vs_F_smoke_max_abs"] = float(np.max(np.abs(fb["hist"] - res["fallback"][0])))
    out["passed"] = bool(out["deposit_identity_max_abs"] == 0.0 and out["basis_vs_F_truth_max_rel"] < 1e-8
                         and out["pdf_vs_F_max_rel"] < 1e-8)
    out["seconds"] = timer.mark("total")
    GC.write_json(GC.RESULTS / "parity.json", out)
    GC.log("parity done", out["passed"], round(out["seconds"], 1))
    return out


if __name__ == "__main__":
    main()
