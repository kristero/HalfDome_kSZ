"""J1 revision 2026-10-16 (local, numpy only, < 1 GB): the DM-quantile importance maps of results/sens.npz rebuilt from
the whole-atom Mecke Jacobians of the maas run results_mecke/ (j_mecke_atom.py), and everything j_sens.py derives from
them (sum rules, zero-beyond-boundary checks, tolerances, peaks, mass shares, radial bands, the J1d tables).

  python -B j_mecke_update.py check   re-derive the quantile entries from J1's stored Jacobians (results/truth.npz
                                      Jq|*, the half-atom formula) with this module's code and compare them with the
                                      stored sens.npz / sens_summary.json / tables (must be identical: this module
                                      reproduces j_sens.py's quantile code exactly); writes nothing
  python -B j_mecke_update.py apply   archive the four J1 sens products once (post/superseded_halfatom_20261016/, the
                                      maas originals) and rewrite them from the archived originals with the whole-atom
                                      Jacobians of D0-D3 (idempotent); update their lines in results/SHA256SUMS; write
                                      post/mecke_update.json (old / new SHA256, sum rules, map changes, outer-slab
                                      first-order checks)
Definitions (j_sens.py): importance of quantile Q_p, I[b, s] = (dQ_p/d ln F_{b,s}) / Q_p, b = engine mass bin (the
fallback haloes above 10^15.388 in their own engine bins), s = fine X200 shell; 'imp' = per-halo subsample maps (Mecke
references C14|D|T1|truth), 'impdet' = truth-mean maps (references C14|D|T1|truthmean).  Only entries with 'dm_q' in
their key change; the moment maps, truncation curves and radii are untouched.
Outer-slab first-order check: removing the outermost fine shells [x_k, R_D] of every calibrated halo changes the
truth-mean quantile by dQ_exact = Q(R_D) - Q(x_k) (sens.npz trunc_all__D__dmq_mean, exact PDF passes); to first order
dQ_Mecke = Q sum_{b calibrated, s >= k} impdet[b, s].
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
MECKE = HERE / "results_mecke"
ARCH = HERE / "post" / "superseded_halfatom_20261016"
LOG = HERE / "post" / "mecke_update.json"
FILES = ("sens.npz", "sens_summary.json", "tables/J1d_importance_shells.csv", "tables/J1d_mass_shares.csv",
         "tables/J1d_bands.csv")
DEFS = {"D0": 4.0, "D1": 1.0, "D2": 2.0, "D3": 4.0}
REPLACE = ("D0", "D1", "D2", "D3")
Q_NAMES = ("q16", "q50", "q84", "q975", "q999")
BANDS = ((0.0, 0.2335), (0.2335, 0.5185), (0.5185, 1.0), (1.0, 2.0), (2.0, 4.0))
BAND_LABELS = ("<0.233R", "0.233-0.519R", "0.519-1R", "1-2R", "2-4R")
DX = 0.125


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def plain(v):                                     # f_common.plain
    if isinstance(v, dict):
        return {str(k): plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [plain(x) for x in v]
    if isinstance(v, np.ndarray):
        return plain(v.tolist())
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return f if np.isfinite(f) else None
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


def fmt(v):                                       # f_common._fmt (csv cells)
    if v is None:
        return ""
    if isinstance(v, (float, np.floating)):
        return "" if not np.isfinite(v) else f"{float(v):.8g}"
    if isinstance(v, np.integer):
        return str(int(v))
    return str(v)


def read_csv(path):
    with open(path, newline="") as fh:
        r = csv.DictReader(fh)
        return r.fieldnames, list(r)


def write_csv_text(header, rows):
    buf = io.StringIO(newline="")
    w = csv.DictWriter(buf, fieldnames=header)
    w.writeheader()
    for r in rows:
        w.writerow(r)
    return buf.getvalue()


def geometry(S):
    X = np.asarray(S["x_edges"], float)
    n_x = len(X) - 1
    dln = np.asarray(S["dlnx"], float)
    mid = np.sqrt(np.maximum(X[:-1], 1e-6) * X[1:])
    band_of = np.searchsorted(np.array([b[1] for b in BANDS]), mid, side="left")
    cal = [int(b) for b in S["j_bins"]]
    cal_mask = np.zeros(16, bool)
    cal_mask[cal] = True
    low = [b for b in cal if b < 5]
    labels = [str(x) for x in S["engine_bin_labels"]]
    kcut = {c: int(np.argmin(np.abs(X - c))) for c in (1.0, 2.0, 4.0)}
    return dict(X=X, n_x=n_x, dln=dln, mid=mid, band_of=band_of, cal=cal, cal_mask=cal_mask, low=low, labels=labels,
                kcut=kcut)


def quantile_entries(d, jac, ts, g):
    """j_sens.main's quantile code for definition d.  jac: {'imp': (Jq, Jqfb), 'impdet': (Jq, Jqfb)} with Jq
    [5, n_cal, 75], Jqfb [5, 16, 67].  Returns (npz entries, summary entries {section: {key: value}}, csv rows)."""
    tag, cut = f"C14|{d}|T1", DEFS[d]
    kB = g["kcut"][cut]
    out, summ = {}, {k: {} for k in ("sum_rules", "checks", "tolerance", "peaks", "shares", "bands")}
    rows = {"shells": [], "shares": [], "bands": []}
    maps = {}
    for kind, lab in (("imp", f"{tag}|truth_subsample"), ("impdet", f"{tag}|truthmean")):
        Jq, Jqfb = jac[kind]
        Q = np.asarray(ts["pdf"][lab.replace("_subsample", "")]["quantiles"], float)
        for p, qn in enumerate(Q_NAMES):
            M = np.array(Jqfb[p], float, copy=True)
            for c, b in enumerate(g["cal"]):
                M[b] += Jq[p, c, :g["n_x"]]
            norm = M / Q[p]
            out[f"{kind}__{d}__dm_{qn}"] = norm.astype(np.float32)
            summ["sum_rules"][f"{kind}__{d}__dm_{qn}"] = float(abs(norm.sum() - 1))
            summ["checks"][f"zero_beyond_boundary__{kind}__{d}__dm_{qn}"] = float(np.max(np.abs(norm[:, kB:]))) \
                if kB < g["n_x"] else 0.0
            if kind == "imp":
                maps[f"dm_{qn}"] = norm
    X, mid, dln, cal_mask = g["X"], g["mid"], g["dln"], g["cal_mask"]
    for name, m in maps.items():
        k, at = 1, "scalar"
        per_shell = m[cal_mask].sum(axis=0)
        dens = m.sum(axis=0) / dln
        with np.errstate(divide="ignore"):
            tol_shell = 0.01 / np.abs(k * per_shell)
            tol_01dex = 0.01 / np.abs(k * (per_shell / dln) * 0.1 * np.log(10))
        out[f"tol01dex__{d}__{name}@{at}"] = tol_01dex.astype(np.float32)
        inside_ = np.arange(g["n_x"]) < kB
        j = int(np.argmin(np.where(inside_, tol_01dex, np.inf)))
        summ["tolerance"][f"{d}__{name}@{at}"] = {
            "min_tol_per_0.1dex": float(tol_01dex[j]), "at_x": float(mid[j]),
            "tol_per_0.1dex_at": {f"{r:g}": float(np.interp(np.log(r), np.log(mid), tol_01dex))
                                  for r in (0.1, 0.3, 0.5, 1.0, 1.5, 2.0, 3.0, 3.8) if r < cut}}
        jp = int(np.argmax(dens))
        summ["peaks"][f"{d}__{name}@{at}"] = {"x_peak_density": float(mid[jp]), "peak_density_per_lnx": float(dens[jp])}
        shares = m.sum(axis=1)
        sh = {g["labels"][b]: float(shares[b]) for b in range(16)}
        sh["calibrated_engine_bins"] = float(shares[cal_mask].sum())
        sh["low_mass_12.7-13.5"] = float(shares[g["low"]].sum()) if g["low"] else 0.0
        summ["shares"][f"{d}__{name}@{at}"] = sh
        bsum = np.array([m[:, g["band_of"] == q].sum() for q in range(len(BANDS))])
        summ["bands"][f"{d}__{name}@{at}"] = dict(zip(BAND_LABELS, bsum.tolist()))
        for b in range(16):
            rows["shares"].append(dict(definition=d, observable=name, ell=at, mass_bin=g["labels"][b],
                                       calibrated=bool(cal_mask[b]), share=float(shares[b])))
        for q, lab in enumerate(BAND_LABELS):
            rows["bands"].append(dict(definition=d, observable=name, ell=at, band=lab, importance=float(bsum[q])))
        for s in range(kB):
            rows["shells"].append(dict(definition=d, observable=name, ell=at, x_lo=X[s], x_hi=X[s + 1],
                                       importance_calibrated=float(per_shell[s]), importance_all=float(m[:, s].sum()),
                                       density_per_lnx=float(dens[s]), tol_shell=float(tol_shell[s]),
                                       tol_per_0p1dex=float(tol_01dex[s])))
    return out, summ, rows


CSV_OF = {"shells": "tables/J1d_importance_shells.csv", "shares": "tables/J1d_mass_shares.csv",
          "bands": "tables/J1d_bands.csv"}


def rebuild(src, jacobians, defs):
    """New contents of the five files: the products in directory ``src`` with the quantile entries of ``defs``
    rebuilt from ``jacobians[d]``.  Returns {file: bytes or dict} and the per-definition entries."""
    S = dict(np.load(src / "sens.npz", allow_pickle=False))
    ss = json.loads((src / "sens_summary.json").read_text())
    ts = json.loads((RES / "truth_summary.json").read_text())
    g = geometry(S)
    csvs = {k: read_csv(src / v) for k, v in CSV_OF.items()}
    per = {}
    for d in defs:
        out, summ, rows = quantile_entries(d, jacobians[d], ts, g)
        per[d] = (out, summ, rows)
        for k, v in out.items():
            assert k in S and S[k].shape == v.shape and S[k].dtype == v.dtype, k
            S[k] = v
        for sec, ent in summ.items():
            for k, v in ent.items():
                assert k in ss[sec], (sec, k)
                ss[sec][k] = v
        for kind, (header, old_rows) in csvs.items():
            new = [{h: fmt(r.get(h)) for h in header} for r in rows[kind]]
            idx = [i for i, r in enumerate(old_rows) if r["definition"] == d and r["observable"].startswith("dm_q")]
            assert len(idx) == len(new) and idx == list(range(idx[0], idx[0] + len(idx))), (kind, d, len(idx), len(new))
            old_rows[idx[0]:idx[0] + len(idx)] = new
    buf = io.BytesIO()
    np.savez_compressed(buf, **S)
    files = {"sens.npz": buf.getvalue(), "sens_summary.json": (json.dumps(plain(ss), indent=1) + "\n").encode()}
    for kind, (header, rows_) in csvs.items():
        files[CSV_OF[kind]] = write_csv_text(header, rows_).encode()
    return files, per, S


def j1_jacobians():
    tr = np.load(RES / "truth.npz", allow_pickle=False)
    return {d: {"imp": (tr[f"Jq|C14|{d}|T1|truth_subsample"], tr[f"Jqfb|C14|{d}|T1|truth_subsample"]),
                "impdet": (tr[f"Jq|C14|{d}|T1|truthmean"], tr[f"Jqfb|C14|{d}|T1|truthmean"])} for d in DEFS}


def whole_jacobians():
    z = np.load(MECKE / "mecke_atom.npz", allow_pickle=False)
    return {d: {"imp": (z[f"Jq_whole|C14|{d}|T1|truth_subsample"], z[f"Jqfb_whole|C14|{d}|T1|truth_subsample"]),
                "impdet": (z[f"Jq_whole|C14|{d}|T1|truthmean"], z[f"Jqfb_whole|C14|{d}|T1|truthmean"])} for d in DEFS}


def compare(src, files, S_new):
    """Equality of rebuilt products with the files in src: npz arrays bitwise, json values, csv text."""
    out = {}
    old = dict(np.load(src / "sens.npz", allow_pickle=False))
    same = lambda x, y: np.array_equal(x, y, equal_nan=x.dtype.kind == "f")     # noqa: E731
    out["sens.npz_arrays_differing"] = [k for k in old if not same(old[k], S_new[k])]
    out["sens.npz_bytes_equal"] = files["sens.npz"] == (src / "sens.npz").read_bytes()
    for f in FILES[1:]:
        out[f"{f}_bytes_equal"] = files[f] == (src / f).read_bytes()
    out["passed"] = bool(not out["sens.npz_arrays_differing"] and all(v for k, v in out.items()
                                                                       if k.endswith("_bytes_equal") and k != "sens.npz_bytes_equal"))
    return out


def outer_slab(S, ts, d, g, kinds=("impdet",)):
    """First-order outer-slab check of the truth-mean maps against the exact truncation passes."""
    kB = g["kcut"][DEFS[d]]
    tq = np.asarray(S[f"trunc_all__{d}__dmq_mean"], float)
    Q = np.asarray(ts["pdf"][f"C14|{d}|T1|truthmean"]["quantiles"], float)
    res = {}
    for p, qn in enumerate(Q_NAMES):
        if Q[p] <= DX:
            continue
        imp = np.asarray(S[f"impdet__{d}__dm_{qn}"], float)
        rows = []
        for k in (kB - 1, kB - 2, kB - 3):
            exact = tq[kB, p] - tq[k, p]
            first = Q[p] * imp[g["cal_mask"]][:, k:kB].sum()
            rows.append({"R_t": float(g["X"][k]), "R_D": DEFS[d], "exact_dQ": float(exact), "mecke_first_order_dQ": float(first),
                         "ratio": float(first / exact) if exact else None})
        res[qn] = rows
    return res


def cmd_check():
    src = ARCH if ARCH.exists() else RES
    files, per, S_new = rebuild(src, j1_jacobians(), tuple(DEFS))
    res = compare(src, files, S_new)
    res["source"] = str(src.relative_to(HERE))
    print(json.dumps(res, indent=1))
    return res


def cmd_apply():
    ARCH.mkdir(parents=True, exist_ok=True)
    (ARCH / "tables").mkdir(exist_ok=True)
    archived = {}
    for f in FILES:
        if not (ARCH / f).exists():
            shutil.copy2(RES / f, ARCH / f)
        archived[f] = sha(ARCH / f)
    # the archive must be the maas originals (results/SHA256SUMS lines as written by the J1 run, or as logged)
    log_old = json.loads(LOG.read_text()) if LOG.exists() else {}
    orig = log_old.get("maas_original_sha256") or {}
    if not orig:
        sums = dict(reversed(line.split("  ", 1)) for line in (RES / "SHA256SUMS").read_text().splitlines() if line.strip())
        orig = {f: sums[f] for f in FILES}
    assert all(archived[f] == orig[f] for f in FILES), "archive is not the maas original"
    chk_files, _, S_chk = rebuild(ARCH, j1_jacobians(), tuple(DEFS))
    reproduce = compare(ARCH, chk_files, S_chk)
    assert reproduce["passed"], reproduce
    files, per, S_new = rebuild(ARCH, whole_jacobians(), REPLACE)
    old_S = dict(np.load(ARCH / "sens.npz", allow_pickle=False))
    for f, data in files.items():
        (RES / f).write_bytes(data)
    new_sha = {f: sha(RES / f) for f in FILES}
    # results/SHA256SUMS: only the lines of the five rebuilt files change
    lines = (RES / "SHA256SUMS").read_text().splitlines()
    out_lines = []
    for line in lines:
        if not line.strip():
            continue
        dig, rel = line.split("  ", 1)
        out_lines.append(f"{new_sha[rel]}  {rel}" if rel in new_sha else line)
    (RES / "SHA256SUMS").write_text("\n".join(out_lines) + "\n")
    ts = json.loads((RES / "truth_summary.json").read_text())
    g = geometry(S_new)
    summ = json.loads((MECKE / "mecke_atom_summary.json").read_text())
    par = json.loads((MECKE / "mecke_parity.json").read_text())
    change = {}
    for d in DEFS:
        for kind in ("imp", "impdet"):
            for qn in Q_NAMES:
                k = f"{kind}__{d}__dm_{qn}"
                a, b = np.asarray(old_S[k], float), np.asarray(S_new[k], float)
                change[k] = {"sum_old": float(a.sum()), "sum_new": float(b.sum()),
                             "max_abs_change_over_max": float(np.max(np.abs(b - a)) / np.max(np.abs(a)))
                             if np.max(np.abs(a)) > 0 else None,
                             "atom_term_min": float(np.min(b - a)),
                             "quantile": float(ts["pdf"][f"C14|{d}|T1|" + ("truth" if kind == "imp" else "truthmean")]
                                               ["quantiles"][Q_NAMES.index(qn)])}
    slabs = {d: {"whole_atom": outer_slab(S_new, ts, d, g), "half_atom_J1": outer_slab(old_S, ts, d, g)} for d in DEFS}
    log = {"time": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "definitions": __doc__,
           "maas_run": "results_mecke (j_mecke_atom.py via run_test_j2.py stages j1base,mecke)",
           "maas_parity_passed": par["passed"], "replaced_definitions": list(REPLACE),
           "files": {f: {"old_sha256": orig[f], "new_sha256": new_sha[f]} for f in FILES},
           "maas_original_sha256": orig, "archive": str(ARCH.relative_to(HERE)),
           "reproduce_stored_from_J1_jacobians": reproduce,
           "map_change": change, "outer_slab_first_order": slabs,
           "mecke_sum_rule_minus_1": {d: v["sum_rule_minus_1"] for d, v in summ["per_definition"].items()},
           "pmf0": {d: v["pmf0"] for d, v in summ["per_definition"].items()}}
    LOG.write_text(json.dumps(plain(log), indent=1) + "\n")
    print(json.dumps({"files": log["files"], "reproduce": reproduce["passed"],
                      "slab_q50": {d: [r["ratio"] for r in slabs[d]["whole_atom"].get("q50", [])] for d in DEFS}},
                     indent=1))


if __name__ == "__main__":
    {"check": cmd_check, "apply": cmd_apply}[sys.argv[1]]()
