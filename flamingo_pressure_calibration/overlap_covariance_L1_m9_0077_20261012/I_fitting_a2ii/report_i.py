"""Test I final local analysis: merges maas run 1 (results/) and the supplement run 2 (results/supp/) into the final
tables, figures, report numbers and verification.json.  Local, light (numpy / matplotlib; < 1 GB RAM).

  python -B report_i.py
Reads  results/tables/I_scores.csv, results/supp/tables/I_scores.csv, results/supp/{supp_items,basis_rebuild}.json,
       results/{fits35,finefits,lvariants,ladder_selection_summary}.json, results/tables/I2_ladder_*.csv,
       results/evaluate.npz (fine model shells at R_cut = 4, A2-ii stacks, family weights), inputs/i0_variants.npz,
       Test G's all-gas ladder (G_physical_outer/results/tables/G_selection_*.csv, selection_summary.json).
Writes results/tables/I*_final*.csv (listed in report_numbers.json), figures/fig_I*.png, report_numbers.json,
       verification.json (updated: the checks of make_figures.py are kept, the supplement checks added).

Conventions (as i_common / i_fine):
  family RMS     sqrt(mean_O (O_model / O_truth - 1)^2) over the family's primary members against Test H's A2-ii
                 scatter-inclusive truth: tSZ = C_l^yy total at l = 100, 300, 1000, 3000; kSZ = 1-halo C_l at the same
                 l; tSZxDM = C_l^{y,DM} total at the same l; DM = mean, variance and the 16/50/84/97.5/99.9 %
                 quantiles of the halo DM PDF (z_s = 1).
  prediction     S1 (lognormal amplitude scatter from the A2-ii per-halo integrals; the painting HalfDome would use)
                 for every family where it exists; the DM PDF was run for S1 only where R_cut = R_fit or R_cut = 4,
                 so off-diagonal DM entries are det (one profile per bin) and are flagged 'det'.
  sensitive-part S_fam = sqrt( sum_{c,s} w_fam[c,s] d[c,s]^2 / sum_{c,s} w_fam[c,s] ), d = ln(model/stack) inside
                 R_cut and d = -1 beyond (removed signal); w_fam = |Test H A2-ii importance| (family mean over members
                 and l = 100-3000).  S_in = sqrt((S^2 - c)/(1 - c)), c = weight share beyond R_cut (the in-profile part).
  profile-only   prof_fam = family RMS of the det painting against Test H's TRUTH MEAN (the A2-ii bin-mean profiles,
  error          one per bin, painted the same way): (O_det(model) / O_truthmean - 1).  It isolates the profile-shape
                 (and truncation) error from the scatter model.  Exact identity per observable:
                   1 + e_S1 = (1 + e_det,mean) (1 + f_S1),  f_S1 = [O_S1(M) / O_det(M)] / [O_truth / O_truthmean] - 1,
                 f_S1 = the error of the lognormal-amplitude (S1) scatter model (ratio of the S1 scatter boost to the
                 true one), nearly independent of the painted profile M: its median over the R_cut = 4 paintings is the
                 'S1 floor' (I4_s1_floor.csv).
  profile RMS    (I1_profile_rms_summary.csv) = Test F's / G's fit diagnostic 'rms_percent' (models.fit_profile):
                 100 sqrt(mean_s (S_s/m_s - 1)^2) over the fitted (resolved, x_{s+1} <= R_fit) shells of one bin,
                 unweighted, i.e. the RMS of (model/stack - 1), NOT of ln(model/stack).
Changes after the first independent verification (2026-10-14 21:xx; pre-fix copy in attempts/pre_fix_verifier1_20261014/):
  profile-only errors and the S1 floor (I4_s1_floor.csv), interpolated quantile radii and the calibrated bins' share of
  the signed importance (I4_importance_regions.csv, I4_calibrated_signal_share.csv), the free-gamma slope table
  (I2_gamma_slopes.csv), M_cut / fixed-M_cut table (I2_mcut_summary.csv), non-monotone R_cut curves
  (I4_rcut_nonmonotone.csv), unresolved-core weight (I4_unresolved_core.csv), fig_I1 (DM: S1 where run, det marked),
  fig_I2 (selections per panel), new fig_I1_profile_vs_S1.png.
Changes after the second independent verification (2026-10-14 ~22:00; pre-fix copy in
attempts/pre_fix_verifier2_20261014/):
  per-halo resolution rule: maas run 3 (i_perhalo.py, results/perhalo/) refits the variant on its OWN resolved shells
  (runs 1-2 fitted it on the strict shells, where it equals A2-ii by construction) and paints the per-halo-rule fine
  stack itself -> I5_perhalo_rule.csv, I5_perhalo_rule_bins.csv, fig_I5_perhalo_rule.png; fig_I5 shows the own-shell
  variant (the strict-shell one is flagged 'identical to the reference by construction' in I5_variants_final.csv) and
  has rotated column labels; I5_stack_differences.csv states the shells compared and adds the per-halo own shells;
  exact S1 scatter-model error of the target profile (the A2-ii stack painted with S1) next to the median estimate;
  unresolved-core weight per bin (I4_unresolved_core_by_bin.csv); observable truncation radii from Test H's
  trunc_all__a2ii__* curves (I4_observable_truncation_radii.csv) next to the |importance|-weight radii; compact
  Lee22 laws vs per-bin forms at R_fit = R_cut = 4 (I4_compact_vs_perbin_Rf4Rc4.csv); non-monotone R_cut count split
  into ladder and I1 curves; fig_I2 points offset horizontally per R_fit so that coincident stars are visible.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RES = HERE / "results"
SUPP = RES / "supp"
TAB = RES / "tables"
FIG = HERE / "figures"
GRES = ROOT / "G_physical_outer" / "results"
sys.path.insert(0, str(HERE))
OKABE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#000000", "#999999"]
MARK = ["o", "s", "^", "D", "v", "P", "X", "*"]
FAMS = ("tSZ", "kSZ", "DM", "tSZxDM")
FAM_FIELD = {"tSZ": "P", "kSZ": "E", "DM": "E", "tSZxDM": "E"}       # tSZxDM: electron-side models (P side also kept)
FAM_TXT = {"tSZ": "tSZ $C_\\ell^{yy}$ (total)", "kSZ": "kSZ $C_\\ell$ (1-halo)", "DM": "FRB DM PDF",
           "tSZxDM": "tSZ$\\times$DM $C_\\ell$ (total)"}
R_FITS_LAD = (1.34, 2.0, 3.0, 4.0)
R_CUTS = (1.0, 1.34, 1.5, 2.0, 3.0, 4.0)


def rcsv(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def wcsv(path, rows):
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def s_in(S, c):
    S, c = fnum(S), fnum(c)
    if not (np.isfinite(S) and np.isfinite(c)) or c >= 1:
        return np.nan
    return float(np.sqrt(max(S**2 - c, 0.0) / (1 - c)))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ============================================================================ loading
def load_scores():
    r1 = rcsv(TAB / "I_scores.csv")
    r2 = rcsv(SUPP / "tables" / "I_scores.csv") if (SUPP / "tables" / "I_scores.csv").exists() else []
    k1 = {(r["label"], r["prediction"]): r for r in r1}
    parity, merged = [], list(r1)
    for r in r2:
        key = (r["label"], r["prediction"])
        if key in k1:
            parity.append((r, k1[key]))
        else:
            r["source"] = "supp"
            merged.append(r)
    for r in r1:
        r["source"] = "run1"
    return merged, parity


def parity_stats(parity):
    worst, n, worst_key = 0.0, 0, None
    for new, old in parity:
        for col, v in new.items():
            if col.startswith(("rms_", "S_", "lin_", "err_")) and v not in ("", None) and old.get(col) not in ("", None):
                x, y = fnum(v), fnum(old[col])
                if np.isfinite(x) and np.isfinite(y) and y != 0:
                    d = abs(x / y - 1)
                    n += 1
                    if d > worst:
                        worst, worst_key = d, f"{new['label']}|{new['prediction']}|{col}"
    return {"rows": len(parity), "values": n, "max_rel": worst, "worst": worst_key}


def index(rows):
    """{label: {prediction: row}}"""
    out = {}
    for r in rows:
        out.setdefault(r["label"], {})[r["prediction"]] = r
    return out


def headline(by_pred, fam, prefer=None):
    """(value, prediction) of a family: S1 where it exists (DM: where the S1 PDF was run), else det; ``prefer`` forces
    one prediction (used for DM curves over R_cut, where only det exists off the diagonal)."""
    for p in ((prefer,) if prefer else ("S1", "det")):
        r = by_pred.get(p)
        if r is not None and r.get(f"rms_{fam}") not in (None, ""):
            v = fnum(r[f"rms_{fam}"])
            if np.isfinite(v):
                return v, p
    return np.nan, None


def score(by_pred, fam):
    r = by_pred.get("det") or by_pred.get("S1")
    if r is None or r.get(f"S_{fam}") in (None, ""):
        return np.nan, np.nan, np.nan
    S, c = fnum(r[f"S_{fam}"]), fnum(r[f"cutshare_{fam}"])
    return S, c, s_in(S, c)


def det_profile(by_pred, fam):
    """(det vs scatter-inclusive truth, det vs truth mean = profile-only error) family RMS of one painting."""
    r = by_pred.get("det")
    if r is None:
        return np.nan, np.nan
    return fnum(r.get(f"rms_{fam}")), fnum(r.get(f"rms_{fam}_vs_mean"))


FAM_MEMBERS = {"tSZ": ("yytot_100", "yytot_300", "yytot_1000", "yytot_3000"),
               "kSZ": ("ksz1h_100", "ksz1h_300", "ksz1h_1000", "ksz1h_3000"),
               "DM": ("dm_mean", "dm_var", "dm_q16", "dm_q50", "dm_q84", "dm_q975", "dm_q999"),
               "tSZxDM": ("ydmtot_100", "ydmtot_300", "ydmtot_1000", "ydmtot_3000")}


def s1_floor(long_rows):
    """f_S1 = (1 + e_S1) / (1 + e_det,mean) - 1 per observable over the R_cut = 4 paintings with both predictions."""
    acc = {}
    for r in long_rows:
        if "|Rc4" not in r["label"]:
            continue
        acc.setdefault((r["label"], r["family"], r["observable"]), {})[r["prediction"]] = r
    vals = {}
    for (lab, fam, obs), d in acc.items():
        if "S1" in d and "det" in d:
            e1, em = fnum(d["S1"]["error"]), fnum(d["det"]["error_vs_mean"])
            if np.isfinite(e1) and np.isfinite(em):
                field = "P" if ("|P|" in lab) else "E"
                vals.setdefault((fam, obs, field), []).append((1 + e1) / (1 + em) - 1)
    rows = []
    for (fam, obs, field), v in sorted(vals.items()):
        v = np.asarray(v)
        rows.append({"family": fam, "observable": obs, "painted_field": field, "n_paintings": len(v),
                     "median_pct": 100 * float(np.median(v)), "min_pct": 100 * float(v.min()),
                     "max_pct": 100 * float(v.max())})
    fam_rms = {}
    for fam, mem in FAM_MEMBERS.items():
        field = "P" if fam == "tSZ" else "E"
        med = [r["median_pct"] for r in rows if r["family"] == fam and r["observable"] in mem
               and r["painted_field"] == field]
        if med:
            fam_rms[fam] = float(np.sqrt(np.mean(np.square(med))))
    return rows, fam_rms


# ============================================================================ main
def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False,
                         "axes.spines.right": False})
    FIG.mkdir(exist_ok=True)
    nums = {"_sources": {}}
    ver_path = HERE / "verification.json"
    ver = json.loads(ver_path.read_text()) if ver_path.exists() else {"mode": "full", "checks": {}}
    rows, parity = load_scores()
    by = index(rows)
    meta = {r["label"]: r for r in rows}
    # ------------------------------------------------------------------ maas run 3 (per-halo rule, results/perhalo/)
    PHD = RES / "perhalo"
    ph_rows = rcsv(PHD / "tables" / "I_scores.csv") if (PHD / "tables" / "I_scores.csv").exists() else []
    k_all = {(r["label"], r["prediction"]): r for r in rows}
    ph_parity = [(r, k_all[(r["label"], r["prediction"])]) for r in ph_rows if (r["label"], r["prediction"]) in k_all]
    ph_new = [r for r in ph_rows if (r["label"], r["prediction"]) not in k_all]
    for r in ph_new:
        r["source"] = "perhalo_run3"
    ph_by = index(ph_new)
    ph_meta = {r["label"]: r for r in ph_new}
    phf = json.loads((PHD / "phfits.json").read_text()) if (PHD / "phfits.json").exists() else None
    ver["checks"]["perhalo_run3_reproduces_run1_paintings"] = parity_stats(ph_parity)
    if phf is not None:
        ver["checks"]["perhalo_run3_strict_mask_refits_vs_run1"] = phf["parity"]
    # ------------------------------------------------------------------ supplement checks
    supp_items = json.loads((SUPP / "supp_items.json").read_text())["items"] if (SUPP / "supp_items.json").exists() else {}
    basis = json.loads((SUPP / "basis_rebuild.json").read_text()) if (SUPP / "basis_rebuild.json").exists() else {}
    ver["checks"]["supp_basis_rebuilt_parity_truthmean_max_rel"] = basis.get("parity_truthmean_a2ii_max_rel")
    ver["checks"]["supp_basis_rebuilt_parity_n"] = basis.get("parity_truthmean_a2ii_n")
    ver["checks"]["supp_run1_paintings_reproduced"] = parity_stats(parity)
    lfb = sorted({tuple(v["fallback_a_bin_labels"]) for v in supp_items.values() if v.get("kind") == "I1_Lfb"})
    ver["checks"]["supp_Lfb_fallback_bins"] = [list(x) for x in lfb]
    ver["checks"]["n_score_rows_merged"] = len(rows)
    # parity of I1 family RMS with Test G's A2-ii scores (make_figures.py's I1_vs_G_scores.csv)
    pg = rcsv(TAB / "I1_vs_G_scores.csv") if (TAB / "I1_vs_G_scores.csv").exists() else []
    out = {}
    for obj in ("a", "d"):
        for fam in ("tSZ", "kSZ", "DM"):
            d = [(abs(fnum(r[f"rms_{fam}_I"]) - fnum(r[f"rms_{fam}_G"])), fnum(r[f"rms_{fam}_I"]), fnum(r[f"rms_{fam}_G"]),
                  r["R_fit"], r["R_cut"], r["prediction"]) for r in pg if r["objective"] == obj
                 and r.get(f"rms_{fam}_G") not in (None, "") and r.get(f"rms_{fam}_I") not in (None, "")]
            if d:
                d.sort(reverse=True)
                out[f"{obj}|{fam}"] = {"n": len(d), "max_abs_diff_pts": round(100 * d[0][0], 3), "at": d[0][3:],
                                       "I": round(100 * d[0][1], 3), "G": round(100 * d[0][2], 3),
                                       "median_abs_diff_pts": round(100 * float(np.median([x[0] for x in d])), 4)}
    ver["checks"]["I1_vs_G_family_rms_abs_diff_percentage_points"] = out
    ver["checks"]["I1_vs_G_note"] = ("G: 35-shell basis and Test F truth; I: fine 67+8-shell A2-ii basis and Test H truth. "
                                     "Objective (a) fits are identical; (d) differs by construction (I weights = Test H "
                                     "A2-ii importance, G weights = Test F importance).")
    fits35 = json.loads((RES / "fits35.json").read_text())
    nL ={f: sum(1 for k, v in fits35["fits"].items() if k.startswith(f"{f}|L|") and v is None) for f in "PE"}
    ver["checks"]["I1_L_fits_missing_bins_per_field"] = nL
    # ================================================================== I1 final table and heat maps
    i1 = []
    for lab, bp in by.items():
        m = meta[lab]
        if m["block"] != "I1":
            continue
        row = {"label": lab, "grid": m["grid"], "field": m["field"], "objective": m["objective"],
               "R_fit": fnum(m["R_fit"]), "R_cut": fnum(m["R_cut"])}
        fams = ("tSZ", "tSZxDM") if m["field"] == "P" else ("kSZ", "DM", "tSZxDM")
        for fam in fams:
            v, p = headline(bp, fam)
            row[f"rms_{fam}"], row[f"pred_{fam}"] = v, p
            if fam == "DM":
                row["rms_DM_det"] = fnum(bp.get("det", {}).get("rms_DM"))
            S, c, si = score(bp, fam if not (fam == "tSZxDM") else "tSZxDM")
            row[f"S_{fam}"], row[f"cutshare_{fam}"], row[f"Sin_{fam}"] = S, c, si
            row[f"rmsdet_{fam}"], row[f"prof_{fam}"] = det_profile(bp, fam)
        i1.append(row)
    i1.sort(key=lambda r: (r["grid"], r["field"], r["objective"], r["R_fit"], r["R_cut"]))
    wcsv(TAB / "I1_scores_final.csv", i1)
    nums["_sources"]["I1"] = "results/tables/I1_scores_final.csv"

    def i1get(grid, field, obj, rf, rc, col):
        for r in i1:
            if r["grid"] == grid and r["field"] == field and r["objective"] == obj and abs(r["R_fit"] - rf) < 1e-9 \
                    and abs(r["R_cut"] - rc) < 1e-9:
                return r.get(col, np.nan)
        return np.nan

    R_FIT1 = sorted({r["R_fit"] for r in i1})
    R_CUT1 = sorted({r["R_cut"] for r in i1})
    objs = [("f35", "a", "(a) diagonal, 5% floor"), ("f35", "d", "(d) importance-weighted"),
            ("f35", "Lfb", "(L) Lee22 lognormal, cov. of mean*"), ("x200", "a", "(a) on the fine 67-shell grid")]
    for kind, col, title, fname in (
            ("rms", "rms", "family RMS error vs the A2-ii truth [%] (S1 prediction; DM: S1 where the S1 DM PDF was "
                           "run (R_cut = R_fit or 4), det elsewhere, marked d)", "fig_I1_rfit_rcut_family_rms.png"),
            ("S", "S", "sensitive-part score $S_{fam}$ [%] (importance-weighted RMS of ln(model/stack); "
                       "removed shells count as -100%)", "fig_I1_rfit_rcut_score.png")):
        famrows = [("tSZ", "P"), ("tSZxDM", "P"), ("kSZ", "E"), ("DM", "E")] if kind == "rms" else \
            [("tSZ", "P"), ("kSZ", "E"), ("DM", "E")]
        fig, axs = plt.subplots(len(famrows), len(objs), figsize=(12.5, 2.55 * len(famrows) + 0.8),
                                constrained_layout=True)
        norm = LogNorm(vmin=0.3, vmax=60)
        for i, (fam, f) in enumerate(famrows):
            for j, (g, o, otxt) in enumerate(objs):
                ax = axs[i, j]
                Z = np.full((len(R_FIT1), len(R_CUT1)), np.nan)
                Pd = np.zeros_like(Z, dtype=bool)
                for p, rf in enumerate(R_FIT1):
                    for q, rc in enumerate(R_CUT1):
                        Z[p, q] = 100 * fnum(i1get(g, f, o, rf, rc, f"{col}_{fam}"))
                        if kind == "rms" and fam == "DM":
                            Pd[p, q] = i1get(g, f, o, rf, rc, "pred_DM") == "det"
                im = ax.imshow(Z, origin="lower", cmap="Blues", norm=norm, aspect="auto")
                for (p, q), v in np.ndenumerate(Z):
                    if np.isfinite(v):
                        ax.text(q, p, f"{v:.1f}" + ("d" if Pd[p, q] else ""), ha="center", va="center", fontsize=7,
                                color="white" if v > 8 else "black")
                ax.set_xticks(range(len(R_CUT1)), [f"{x:g}" for x in R_CUT1])
                ax.set_yticks(range(len(R_FIT1)), [f"{x:g}" for x in R_FIT1])
                ax.grid(False)
                if i == len(famrows) - 1:
                    ax.set_xlabel("$R_{cut}$ [$R_{200c}$]")
                if j == 0:
                    ax.set_ylabel(f"{fam}\n$R_{{fit}}$ [$R_{{200c}}$]")
                if i == 0:
                    ax.set_title(f"{'B12/B16'} {otxt}", fontsize=8)
        fig.colorbar(im, ax=axs, shrink=0.5, label="%" if kind == "rms" else "$S_{fam}$ [%]")
        fig.suptitle(f"I1 B12 (pressure) / B16 (electrons) fits to the A2-ii stacks: {title}\n"
                     "* (L): bins 15.1-15.3 and 15.3-15.5 have N <= p+2 haloes (no Lee22 precision) and are painted "
                     "with the objective-(a) fit of the same $R_{fit}$", fontsize=9)
        fig.savefig(FIG / fname, dpi=150)
        plt.close(fig)
    # key I1 numbers
    for f, fams in (("P", ("tSZ", "tSZxDM")), ("E", ("kSZ", "DM", "tSZxDM"))):
        for o in ("a", "d", "Lfb"):
            for rf in (1.0, 1.34, 2.0, 3.0, 4.0):
                for rc in (1.0, 2.0, 3.0, 4.0):
                    for fam in fams:
                        v = i1get("f35", f, o, rf, rc, f"rms_{fam}")
                        if np.isfinite(v):
                            nums[f"I1|{f}|{o}|Rf{rf:g}|Rc{rc:g}|rms_{fam}"] = round(v, 5)
                        s = i1get("f35", f, o, rf, rc, f"S_{fam}")
                        if np.isfinite(s):
                            nums[f"I1|{f}|{o}|Rf{rf:g}|Rc{rc:g}|S_{fam}"] = round(s, 4)
    for fam, f in (("tSZ", "P"), ("kSZ", "E"), ("DM", "E")):
        for rf in (1.0, 2.0, 4.0):
            nums[f"I1|{f}|x200a|Rf{rf:g}|Rc4|rms_{fam}"] = round(fnum(i1get("x200", f, "a", rf, 4.0, f"rms_{fam}")), 5)
    # parameters of the L fits vs (a)
    dl = []
    for key, v in fits35["fits"].items():
        f, o, rf, c = key.split("|")
        if o != "L" or v is None:
            continue
        a = fits35["fits"].get(f"{f}|a|{rf}|{c}")
        if a is None:
            continue
        d = np.log(np.asarray(v["parameters"]) / np.asarray(a["parameters"]))
        dl.append({"field": f, "R_fit": rf, "bin": int(c), "N": v.get("N"), "p_shells": v.get("p"),
                   "hartlap": v.get("hartlap"), "chi2nu_L": v.get("chi2_per_dof"),
                   **{f"dlnp{i}_L_minus_a": float(d[i]) for i in range(3)}})
    wcsv(TAB / "I1_L_vs_a_parameters.csv", dl)
    prms = []
    for f in "PE":
        for o in ("a", "d", "L"):
            for rf in (1.0, 1.34, 1.5, 2.0, 3.0, 4.0):
                v = [fits35["fits"].get(f"{f}|{o}|{rf:g}|{c}") for c in range(10)]
                r = [x["rms_percent"] for x in v if x]
                if r:
                    # rms_percent = 100 sqrt(mean_s (S_s/m_s - 1)^2) over the fitted shells (models.fit_profile)
                    prms.append({"field": f, "objective": o, "R_fit": rf, "n_bins_fitted": len(r),
                                 "definition": "100*sqrt(mean over fitted shells of (model/stack-1)^2), per bin",
                                 "profile_rms_ratio_pct_median": float(np.median(r)),
                                 "profile_rms_ratio_pct_min": float(min(r)),
                                 "profile_rms_ratio_pct_max": float(max(r))})
                    nums[f"I1|{f}|{o}|Rf{rf:g}|profile_rms_ratio_minus1_pct_median_min_max"] = [
                        round(float(np.median(r)), 2), round(min(r), 2), round(max(r), 2)]
    wcsv(TAB / "I1_profile_rms_summary.csv", prms)
    # ------------------------------------------------------------------ profile-only error vs S1, and the S1 floor
    long_rows, seen = [], set()
    for p in (TAB / "I_errors_long.csv", SUPP / "tables" / "I_errors_long.csv"):
        if p.exists():
            for r in rcsv(p):
                key = (r["label"], r["prediction"], r["observable"])
                if key not in seen:
                    seen.add(key)
                    long_rows.append(r)
    floor_rows, floor_fam = s1_floor(long_rows)
    wcsv(TAB / "I4_s1_floor.csv", floor_rows)
    nums["I4|S1_floor_family_rms_pct (ESTIMATE: median over R_cut=4 paintings)"] = {k: round(v, 3)
                                                                                   for k, v in floor_fam.items()}
    nums["I4|S1_floor_by_observable_median_pct"] = {f"{r['family']}|{r['observable']}|{r['painted_field']}":
                                                    [round(r["median_pct"], 3), round(r["min_pct"], 3),
                                                     round(r["max_pct"], 3), r["n_paintings"]] for r in floor_rows}
    for f, fams in (("P", ("tSZ", "tSZxDM")), ("E", ("kSZ", "DM", "tSZxDM"))):
        for o in ("a", "d", "Lfb"):
            for rf in (1.0, 1.34, 2.0, 3.0, 4.0):
                for fam in fams:
                    for col in ("prof", "rmsdet"):
                        v = i1get("f35", f, o, rf, 4.0, f"{col}_{fam}")
                        if np.isfinite(fnum(v)):
                            nums[f"I1|{f}|{o}|Rf{rf:g}|Rc4|{col}_{fam}"] = round(fnum(v), 5)
    fig, axs = plt.subplots(1, 4, figsize=(15, 4.4), constrained_layout=True)
    for ax, (fam, f) in zip(axs, (("tSZ", "P"), ("kSZ", "E"), ("DM", "E"), ("tSZxDM", "E"))):
        for n, (o, otxt) in enumerate((("a", "(a)"), ("d", "(d)"), ("Lfb", "(L*)"))):
            xs = [rf for rf in R_FIT1]
            ys = [100 * fnum(i1get("f35", f, o, rf, 4.0, f"rms_{fam}")) for rf in xs]
            yp = [100 * fnum(i1get("f35", f, o, rf, 4.0, f"prof_{fam}")) for rf in xs]
            ax.plot(xs, ys, "-", marker=MARK[n], color=OKABE[n], ms=5, lw=1.3,
                    label=f"B12/B16 {otxt}: S1 vs truth")
            ax.plot(xs, yp, "--", marker=MARK[n], color=OKABE[n], ms=5, lw=1.0, mfc="white",
                    label=f"B12/B16 {otxt}: det vs truth mean (profile only)")
        if fam in floor_fam:
            ax.axhline(floor_fam[fam], color="0.3", lw=1.0, ls=":")
            ax.text(0.03, 0.03, f"dotted: S1 scatter-model floor {floor_fam[fam]:.2f}% (estimate)",
                    transform=ax.transAxes, fontsize=7, color="0.25", va="bottom")
        ax.set_yscale("log")
        ax.set_ylim(0.1, 30)
        ax.set_xlabel("$R_{fit}$ [$R_{200c}$]  ($R_{cut}$ = 4)")
        ax.set_ylabel("family RMS error [%]")
        ax.set_title(FAM_TXT[fam] + (" (B16, electron side)" if fam == "tSZxDM" else
                                     " (B12 pressure fits)" if fam == "tSZ" else " (B16 electron fits)"), fontsize=8.5)
    h, l = axs[1].get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncol=3, fontsize=7)
    fig.suptitle("I1: S1 prediction vs the scatter-inclusive truth (solid) and det vs the truth mean (dashed: the "
                 "profile-shape error alone), B12/B16 fits to A2-ii, $R_{cut}=4$", fontsize=9.5)
    fig.savefig(FIG / "fig_I1_profile_vs_S1.png", dpi=150)
    plt.close(fig)
    for f in "PE":
        for rf in ("1", "2", "4"):
            ch = [x["chi2nu_L"] for x in dl if x["field"] == f and x["R_fit"] == rf]
            if ch:
                nums[f"I1|{f}|L|Rf{rf}|chi2nu_L_range"] = [round(min(ch), 2), round(max(ch), 2)]
    # ================================================================== I2 ladder figure (A2-ii vs all gas)
    lad = rcsv(TAB / "I2_ladder_models.csv")
    gl = rcsv(GRES / "tables" / "G_selection_models.csv")
    summ = json.loads((RES / "ladder_selection_summary.json").read_text())
    gsumm = json.loads((GRES / "selection_summary.json").read_text())
    fig, axs = plt.subplots(2, 2, figsize=(12, 8.6), constrained_layout=True)
    for i, f in enumerate(("P", "E")):
        for j, (col, lab) in enumerate((("chi2nu_floor", r"best $\chi^2/\nu$ at this k (floor likelihood)"),
                                        ("dBIC_floor", r"best $\Delta$BIC at this k (floor; 0 plotted at 0.3)"))):
            ax = axs[i, j]
            for n, rf in enumerate(R_FITS_LAD):
                off = (0.86, 0.95, 1.05, 1.16)[n]       # horizontal offset per R_fit (coincident stars visible)
                for src, ls, tag, ch in ((lad, "-", "A2-ii", summ["chosen"]), (gl, ":", "all gas (G)", gsumm["chosen"])):
                    rr = [x for x in src if x["field"] == f and x["dset"] == "mass" and abs(fnum(x["R_fit"]) - rf) < 1e-9
                          and np.isfinite(fnum(x.get(col)))]
                    ks = sorted({int(fnum(x["k"])) for x in rr})
                    yv = [min(fnum(x[col]) for x in rr if int(fnum(x["k"])) == k) for k in ks]
                    if col == "dBIC_floor":
                        yv = [max(y, 0.3) for y in yv]
                    sel = ch.get(f"{f}|mass|{rf:g}", {})
                    ax.plot([k * off for k in ks], yv, ls=ls, marker=MARK[n], ms=5, lw=1.3 if tag == "A2-ii" else 1.0,
                            color=OKABE[n],
                            mfc=OKABE[n] if tag == "A2-ii" else "white",
                            label=f"$R_{{fit}}$={rf:g}, {tag}")
                    if sel and tag == "A2-ii":
                        x = [x for x in rr if x["model"] == sel["model"]]
                        if x:
                            y = fnum(x[0][col])
                            ax.plot([int(fnum(x[0]["k"])) * off], [max(y, 0.3) if col == "dBIC_floor" else y], marker="*",
                                    ms=11, color=OKABE[n], mec="black", mew=0.7, ls="none", zorder=5)
            ax.set_yscale("log")
            ax.set_xscale("log")
            ax.set_xticks([4, 6, 8, 10, 15, 20, 30], ["4", "6", "8", "10", "15", "20", "30"])
            ax.minorticks_off()
            ax.set_xlim(3.2, 46)
            ax.set_xlabel("number of free parameters k (10 calibrated mass bins together)")
            ax.set_ylabel(lab)
            ax.axhline(6 if col == "dBIC_floor" else 1, color="0.35", lw=0.8, ls="--")
            fld = 'pressure (tSZ)' if f == 'P' else 'electron density (kSZ, DM)'
            src_ch = summ["chosen"] if j == 0 else gsumm["chosen"]
            it = [f"$R_{{fit}}$={rf:g}: {src_ch.get(f'{f}|mass|{rf:g}', {}).get('model')} "
                  f"(k {src_ch.get(f'{f}|mass|{rf:g}', {}).get('k')})" for rf in R_FITS_LAD]
            selt = ", ".join(it[:2]) + ",\n" + ", ".join(it[2:])
            if j == 0:
                ax.set_title(f"{fld}, mass bins: best $\\chi^2/\\nu$ per k\n"
                             f"stars = selected on A2-ii: {selt}", fontsize=8)
            else:
                ax.set_title(f"{fld}, mass bins: best $\\Delta$BIC per k (stars as on the left)\n"
                             f"all gas (Test G) selected, for comparison: {selt}", fontsize=8)
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncol=4, fontsize=7,
               title="colour/marker = $R_{fit}$ [$R_{200c}$]; filled, solid = A2-ii (Test I); open, dotted = all gas "
                     "(Test G).\nStars = the model selected on A2-ii (fewest k with dBIC <= 6 and CV within the best "
                     "model's fold scatter); names in each row's left title.\nPoints are offset horizontally by "
                     "factors 0.86/0.95/1.05/1.16 for R_fit = 1.34/2/3/4 so that coincident points stay visible.",
               title_fontsize=7)
    fig.savefig(FIG / "fig_I2_ladder_chi2_dbic_vs_k.png", dpi=150)
    plt.close(fig)
    for key, ch in summ["chosen"].items():
        nums[f"I2|selected|{key}"] = {k: ch[k] for k in ("model", "k", "chi2nu_floor", "rule_satisfied", "best_BIC",
                                                       "best_CV")}
        g = gsumm["chosen"].get(key, {})
        nums[f"I2|selected_allgas|{key}"] = {k: g.get(k) for k in ("model", "k", "chi2nu_floor")}
    sig = {}
    for k, v in summ["sigma_floor"].items():
        sig[k] = {"sigma_ref": v["sigma_ref"], "reference_outer": v["reference_outer"],
                  "by_outer": {o: w["sigma_d"] for o, w in v["by_outer"].items()}}
    nums["I2|sigma_floor"] = sig
    nums["_sources"]["I2"] = ("results/ladder_selection_summary.json; results/tables/I2_ladder_ladder.csv, "
                              "I2_ladder_models.csv, I2_ladder_vs_allgas.csv")
    # parsimony ladder compact table (mass and massc) A2-ii next to all gas
    lt = rcsv(TAB / "I2_ladder_ladder.csv")
    glt = rcsv(GRES / "tables" / "G_selection_ladder.csv")
    comp = []
    for r in lt:
        g = [x for x in glt if x["field"] == r["field"] and x["dset"] == r["dset"] and x["R_fit"] == r["R_fit"]
             and x["k"] == r["k"]]
        comp.append({"field": r["field"], "dset": r["dset"], "R_fit": r["R_fit"], "k": r["k"],
                     "best_A2ii": r["model"], "chi2nu_floor_A2ii": fnum(r["chi2nu_floor"]),
                     "dBIC_A2ii": fnum(r["dBIC_floor"]), "pass_bic": r["pass_bic"], "pass_cv": r["pass_cv"],
                     "selected_A2ii": r["selected"], "best_allgas": g[0]["model"] if g else "",
                     "chi2nu_floor_allgas": fnum(g[0]["chi2nu_floor"]) if g else np.nan,
                     "dBIC_allgas": fnum(g[0]["dBIC_floor"]) if g else np.nan})
    wcsv(TAB / "I2_parsimony_ladder_A2ii_vs_allgas.csv", comp)
    # chi2/nu under the three likelihoods of key models (A2-ii, mass)
    keym = []
    for x in lad:
        if x["dset"] != "mass":
            continue
        if x["inner"] in ("PB", "PL1", "PL3", "BPLall", "PL3g", "PL3gM") or x["model"] in \
                [c["model"] for k, c in summ["chosen"].items() if k.startswith(x["field"] + "|mass|")]:
            keym.append({k: x.get(k) for k in ("field", "R_fit", "model", "k", "chi2nu_stat", "chi2nu_floor",
                                                "chi2nu_L22", "dBIC_floor", "AIC_floor", "BIC_floor", "cv_chi2",
                                                "pass_bic", "pass_cv") if k in x})
    wcsv(TAB / "I2_key_models_likelihoods.csv", keym)
    # ------------------------------------------------------------------ what the free gamma does (shape at fixed mass)
    # F_in(x) = A (x/x_c)^gamma (1 + x/x_c)^-beta  =>  dlnF/dlnx = gamma - beta (x/x_c) / (1 + x/x_c)
    # mass laws (PL): ln Q(M) = Q_0 + Q_am ln(M / 10^14.5 Msun) for Q in {A, x_c, beta} (theta stores ln values);
    # PL3gM: gamma(M) = g0 (M / 10^14.5)^g_am.
    lres = json.loads(str(np.load(RES / "ladder_selection.npz")["results_json"]))
    lby = {(r["field"], r["dset"], float(r["R_fit"]), r["model"]): r for r in lres}
    xs_sl = (0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 4.0)
    gam_rows = []
    for f in "PE":
        for rf in R_FITS_LAD:
            for mname in ("PL3+O0", "PL3g+O0", "PL3gM+O0", "PL3+O2", "PL3g+O2"):
                r = lby.get((f, "mass", rf, mname))
                if r is None or "floor" not in r["fits"]:
                    continue
                th = dict(zip(r["names"], r["fits"]["floor"]["theta"]))
                for logm in (13.6, 14.0, 14.5, 15.0, 15.4):
                    lm = (logm - 14.5) * np.log(10.0)
                    xc = float(np.exp(th["xc_0"] + th.get("xc_am", 0.0) * lm))
                    beta = float(np.exp(th["beta_0"] + th.get("beta_am", 0.0) * lm))
                    g = float(th["g0"] * np.exp(th.get("g_am", 0.0) * lm)) if "g0" in th else -0.3
                    row = {"field": f, "R_fit": rf, "model": mname, "log10M200c": logm, "gamma": g, "x_c": xc,
                           "beta": beta, "asymptotic_slope_gamma_minus_beta": g - beta}
                    for xv in xs_sl:
                        row[f"slope_at_x{xv:g}"] = g - beta * (xv / xc) / (1 + xv / xc)
                    gam_rows.append(row)
    wcsv(TAB / "I2_gamma_slopes.csv", gam_rows)
    for r in gam_rows:
        if r["field"] == "E" and r["R_fit"] == 4.0 and r["log10M200c"] == 14.5 and r["model"] in ("PL3+O0", "PL3g+O0"):
            nums[f"I2|E|Rf4|{r['model']}|pivot14.5"] = {k: round(v, 3) for k, v in r.items()
                                                        if isinstance(v, float) and k not in ("R_fit", "log10M200c")}
    # ------------------------------------------------------------------ break mass and Lee22's fixed M_cut
    mc_rows = []
    for r in lres:
        if "floor" in r["fits"] and "log10Mcut" in r["names"]:
            mc_rows.append({"field": r["field"], "dset": r["dset"], "R_fit": float(r["R_fit"]), "model": r["model"],
                            "log10Mcut_free_Msun": float(r["fits"]["floor"]["theta"][r["names"].index("log10Mcut")])})
    for m in mc_rows:
        x = [y for y in lad if y["field"] == m["field"] and y["dset"] == m["dset"]
             and abs(fnum(y["R_fit"]) - m["R_fit"]) < 1e-9 and y["model"] == m["model"]]
        if x:
            m["dBIC_floor"], m["pass_bic"], m["pass_cv"] = fnum(x[0]["dBIC_floor"]), x[0]["pass_bic"], x[0]["pass_cv"]
    l22_rows = [y for y in lad if "_L22" in y["model"]]
    for y in l22_rows:
        mc_rows.append({"field": y["field"], "dset": y["dset"], "R_fit": fnum(y["R_fit"]), "model": y["model"],
                        "log10Mcut_fixed_Msun": summ["lee22_mcut_msun"][y["field"] + ("_c" if y["dset"] == "massc" else "")],
                        "dBIC_floor": fnum(y["dBIC_floor"]), "pass_bic": y["pass_bic"], "pass_cv": y["pass_cv"]})
    wcsv(TAB / "I2_mcut_summary.csv", mc_rows)
    mc_sum = {}
    for f in "PE":
        fr = [m["log10Mcut_free_Msun"] for m in mc_rows if m["field"] == f and m["dset"] == "mass"
              and "log10Mcut_free_Msun" in m]
        if fr:
            mc_sum[f"{f}|mass|free_log10Mcut_range"] = [round(min(fr), 3), round(max(fr), 3)]
            fa = [m["log10Mcut_free_Msun"] for m in mc_rows if m["field"] == f and m["dset"] == "mass"
                  and "log10Mcut_free_Msun" in m and m["model"].startswith("BPLall+")]
            mc_sum[f"{f}|mass|free_log10Mcut_range_BPLall"] = [round(min(fa), 3), round(max(fa), 3)]
        for ds in ("mass", "massc"):
            d = [m for m in mc_rows if m["field"] == f and m["dset"] == ds and "log10Mcut_fixed_Msun" in m]
            if d:
                mc_sum[f"{f}|{ds}|fixed_L22_dBIC_range"] = [round(min(m["dBIC_floor"] for m in d), 2),
                                                            round(max(m["dBIC_floor"] for m in d), 2)]
                best = {}
                for m in d:
                    best[m["R_fit"]] = min(best.get(m["R_fit"], np.inf), m["dBIC_floor"])
                mc_sum[f"{f}|{ds}|fixed_L22_best_dBIC_per_Rfit"] = {f"{k:g}": round(v, 2) for k, v in sorted(best.items())}
                passing = [f"{m['model']}@{m['R_fit']:g} dBIC {m['dBIC_floor']:.2f} cv {m['pass_cv']}" for m in d
                           if m["pass_bic"] in ("True", True)]
                mc_sum[f"{f}|{ds}|fixed_L22_passing_BIC"] = passing
    sel_mc = {}
    for key, ch in summ["chosen"].items():
        x = [m for m in mc_rows if f"{m['field']}|{m['dset']}|{m['R_fit']:g}" == key and m["model"] == ch["model"]
             and "log10Mcut_free_Msun" in m]
        if x:
            sel_mc[key] = round(x[0]["log10Mcut_free_Msun"], 3)
    mc_sum["selected_models_free_log10Mcut"] = sel_mc
    mc_sum["lee22_mcut_msun (converted, h = 0.6774)"] = summ["lee22_mcut_msun"]
    nums["I2|Mcut"] = mc_sum
    # ================================================================== I3 / I4: ladder paintings at every cut
    roles_of = {}
    for lab, rec in supp_items.items():
        if rec.get("kind") == "I3_cut":
            roles_of[lab] = rec.get("roles")
    i3 = []
    for lab, bp in by.items():
        m = meta[lab]
        if m["block"] != "I3":
            continue
        row = {"label": lab, "field": m["field"], "role": m["role"], "roles": ";".join(r for r in roles_of.get(lab, [m["role"]]) if r),
               "model": m["model"], "dset": m["dset"], "k": int(fnum(m["k"])), "R_fit": fnum(m["R_fit"]),
               "R_cut": fnum(m["R_cut"]), "source": m.get("source")}
        for fam in (("tSZ", "tSZxDM") if m["field"] == "P" else ("kSZ", "DM", "tSZxDM")):
            v, p = headline(bp, fam)
            row[f"rms_{fam}"], row[f"pred_{fam}"] = v, p
            S, c, si = score(bp, fam)
            row[f"S_{fam}"], row[f"cutshare_{fam}"], row[f"Sin_{fam}"] = S, c, si
            row[f"rmsdet_{fam}"], row[f"prof_{fam}"] = det_profile(bp, fam)
        i3.append(row)
    i3.sort(key=lambda r: (r["field"], r["R_fit"], r["k"], r["model"], r["R_cut"]))
    wcsv(TAB / "I3_ladder_observables_final.csv", i3)
    nums["_sources"]["I3"] = "results/tables/I3_ladder_observables_final.csv"

    # model table for I4: rows = models, columns = (R_fit, R_cut in {R_fit, 4})
    def i3pick(field, role, rf, rc):
        c = [r for r in i3 if r["field"] == field and abs(r["R_fit"] - rf) < 1e-9 and abs(r["R_cut"] - rc) < 1e-9
             and (role in r["roles"].split(";") or r["role"] == role)]
        return c[0] if c else None

    ROLES = (("PL1", "PL1"), ("PL3", "PL3"), ("gamma", "best free-$\\gamma$"), ("BPLall", "BPLall"),
             ("cbest", "best +c (bin-mean c)"), ("PB", "PB (per bin)"), ("selected", "selected"))
    table = []
    for fam in FAMS:
        f = FAM_FIELD[fam]
        for o, otxt in (("a", "B12/B16 (a)"), ("d", "B12/B16 (d)"), ("Lfb", "B12/B16 (L*)")):
            for rf in R_FITS_LAD:
                for cut in ("fit", "4"):
                    rc = rf if cut == "fit" else 4.0
                    v = i1get("f35", f, o, rf, rc, f"rms_{fam}")
                    s = i1get("f35", f, o, rf, rc, f"S_{fam}")
                    table.append({"family": fam, "model": otxt, "role": f"I1{o}", "k": 30, "R_fit": rf, "R_cut": rc,
                                  "rms": v, "S": s, "pred": i1get("f35", f, o, rf, rc, f"pred_{fam}"),
                                  "prof": i1get("f35", f, o, rf, rc, f"prof_{fam}")})
        for role, rtxt in ROLES:
            for rf in R_FITS_LAD:
                for cut in ("fit", "4"):
                    rc = rf if cut == "fit" else 4.0
                    r = i3pick(f, role, rf, rc)
                    if r is None:
                        continue
                    table.append({"family": fam, "model": f"{rtxt}: {r['model']}" + (" [c cells]" if r["dset"] == "massc" else ""),
                                  "role": role, "k": r["k"], "R_fit": rf, "R_cut": rc, "rms": r[f"rms_{fam}"],
                                  "S": r[f"S_{fam}"], "pred": r[f"pred_{fam}"], "prof": r[f"prof_{fam}"]})
    wcsv(TAB / "I4_model_table.csv", table)
    for t in table:
        nums[f"I4|table|{t['family']}|{t['model']}|Rf{t['R_fit']:g}|Rc{t['R_cut']:g}"] = {
            "k": t["k"], "rms_pct": round(100 * fnum(t["rms"]), 3), "pred": t["pred"],
            "S_pct": round(100 * fnum(t["S"]), 2), "profile_only_pct": round(100 * fnum(t["prof"]), 3)}
    # compact Lee22 laws vs the per-bin forms at R_fit = R_cut = 4 (S1 vs truth and profile-only), per family
    cvp = []
    for fam in FAMS:
        t4 = [t for t in table if t["family"] == fam and t["R_fit"] == 4.0 and t["R_cut"] == 4.0]
        ref = {t["role"]: t for t in t4 if t["role"].startswith("I1")}
        seen_m = set()
        for t in sorted(t4, key=lambda t: fnum(t["rms"])):
            if t["model"] in seen_m:
                continue
            seen_m.add(t["model"])
            row = {"family": fam, "model": t["model"], "role": t["role"], "k": t["k"],
                   "per_bin": t["role"].startswith("I1") or t["role"] == "PB" or
                   (t["role"] == "selected" and t["model"].split(": ")[-1].startswith("PB")),
                   "rms_S1_pct": 100 * fnum(t["rms"]), "profile_only_pct": 100 * fnum(t["prof"]),
                   "S_pct": 100 * fnum(t["S"]), "pred": t["pred"]}
            for o in ("a", "d"):
                r0 = ref.get(f"I1{o}")
                row[f"minus_B_{o}_pp"] = 100 * (fnum(t["rms"]) - fnum(r0["rms"])) if r0 else np.nan
            cvp.append(row)
    wcsv(TAB / "I4_compact_vs_perbin_Rf4Rc4.csv", cvp)
    nums["I4|compact_vs_perbin_Rf4Rc4 (rms_S1_pct, profile_only_pct, minus_B_a_pp, minus_B_d_pp)"] = {
        f"{r['family']}|{r['model']}": [round(r["rms_S1_pct"], 3), round(r["profile_only_pct"], 3),
                                        round(r["minus_B_a_pp"], 3), round(r["minus_B_d_pp"], 3)] for r in cvp}
    # fits that stop at R_fit = 1.34 extrapolated to R_cut = 4 (electrons)
    ext = []
    seen_ext = set()
    for r in i3:
        if r["field"] == "E" and abs(r["R_fit"] - 1.34) < 1e-9 and r["R_cut"] == 4.0 \
                and (r["model"], r["dset"]) not in seen_ext:
            seen_ext.add((r["model"], r["dset"]))
            ext.append({"model": r["model"] + (" [c cells]" if r["dset"] == "massc" else ""), "k": r["k"],
                        "kSZ_pct": 100 * r["rms_kSZ"], "DM_pct": 100 * r["rms_DM"], "S_kSZ_pct": 100 * r["S_kSZ"],
                        "S_DM_pct": 100 * r["S_DM"], "prof_kSZ_pct": 100 * fnum(r["prof_kSZ"]),
                        "prof_DM_pct": 100 * fnum(r["prof_DM"])})
    for o in ("a", "d", "Lfb"):
        ext.append({"model": f"B16 ({o})", "k": 30, "kSZ_pct": 100 * i1get("f35", "E", o, 1.34, 4.0, "rms_kSZ"),
                    "DM_pct": 100 * i1get("f35", "E", o, 1.34, 4.0, "rms_DM"),
                    "S_kSZ_pct": 100 * i1get("f35", "E", o, 1.34, 4.0, "S_kSZ"),
                    "S_DM_pct": 100 * i1get("f35", "E", o, 1.34, 4.0, "S_DM"),
                    "prof_kSZ_pct": 100 * fnum(i1get("f35", "E", o, 1.34, 4.0, "prof_kSZ")),
                    "prof_DM_pct": 100 * fnum(i1get("f35", "E", o, 1.34, 4.0, "prof_DM"))})
    wcsv(TAB / "I4_Rfit134_extrapolated_to_Rcut4.csv", ext)
    nums["I4|E|Rfit1.34_Rcut4"] = {e["model"]: {k: (round(v, 2) if isinstance(v, float) else v) for k, v in e.items()
                                                if k != "model"} for e in ext}
    nums["_sources"]["I4"] = "results/tables/I4_model_table.csv, I4_rcut_curves.csv, I4_minimal_adequate_final.csv"
    # figure: heat-map table per family (rows: role; columns: R_fit x {R_cut = R_fit, 4})
    role_rows = [("I1a", "B12/B16 (a), k=30"), ("I1d", "B12/B16 (d), k=30"), ("I1Lfb", "B12/B16 (L*), k=30")] + \
                [(r, t) for r, t in ROLES]
    cols = [(rf, cut) for cut in ("fit", "4") for rf in R_FITS_LAD]
    fig, axs = plt.subplots(2, 2, figsize=(14, 9.2), constrained_layout=True)
    for ax, fam in zip(axs.ravel(), FAMS):
        Z = np.full((len(role_rows), len(cols)), np.nan)
        ylab = []
        for i, (role, rtxt) in enumerate(role_rows):
            names = set()
            ks = set()
            for j, (rf, cut) in enumerate(cols):
                rc = rf if cut == "fit" else 4.0
                c = [t for t in table if t["family"] == fam and t["role"] == role and abs(t["R_fit"] - rf) < 1e-9
                     and abs(t["R_cut"] - rc) < 1e-9]
                if c:
                    Z[i, j] = 100 * fnum(c[0]["rms"])
                    if not role.startswith("I1"):
                        names.add(c[0]["model"].split(": ")[1].replace(" [c cells]", "c"))
                        ks.add(c[0]["k"])
            if role.startswith("I1"):
                ylab.append(rtxt)
            else:
                ylab.append(f"{rtxt.split(':')[0]} ({'/'.join(sorted(names))}; k {'/'.join(str(k) for k in sorted(ks))})"
                            if names else rtxt)
        ax.imshow(Z, cmap="Blues", norm=LogNorm(vmin=0.3, vmax=60), aspect="auto")
        for (p, q), v in np.ndenumerate(Z):
            if np.isfinite(v):
                ax.text(q, p, f"{v:.1f}", ha="center", va="center", fontsize=7, color="white" if v > 8 else "black")
        ax.set_xticks(range(len(cols)), [f"{rf:g}\n{'=R_fit' if cut == 'fit' else '4'}" for rf, cut in cols], fontsize=7)
        ax.set_yticks(range(len(role_rows)), ylab, fontsize=6.5)
        ax.axvline(3.5, color="black", lw=1.2)
        ax.grid(False)
        ax.set_xlabel("$R_{fit}$ / $R_{cut}$ [$R_{200c}$]   (left block: $R_{cut}=R_{fit}$; right block: $R_{cut}=4$)")
        ax.set_title(f"{FAM_TXT[fam]}: RMS error vs A2-ii truth [%] "
                     f"({'pressure' if FAM_FIELD[fam] == 'P' else 'electron'} models; S1)", fontsize=8.5)
    fig.suptitle("I3/I4: B12/B16 per-bin fits and the Lee22 ladder (fitted to A2-ii, painted on the fine A2-ii basis)",
                 fontsize=9.5)
    fig.savefig(FIG / "fig_I4_model_table.png", dpi=150)
    plt.close(fig)
    # figure: error and score vs k at R_fit = R_cut = 4 (filled) and R_fit = 2, R_cut = 4 (open), labelled
    fig, axs = plt.subplots(2, 4, figsize=(15, 7.4), constrained_layout=True)
    for j, fam in enumerate(FAMS):
        for i, col in enumerate(("rms", "S")):
            ax = axs[i, j]
            for n, rf in enumerate((4.0, 2.0)):
                pts = [t for t in table if t["family"] == fam and abs(t["R_fit"] - rf) < 1e-9 and t["R_cut"] == 4.0
                       and t["role"] != "selected" and np.isfinite(fnum(t[col]))]
                pts.sort(key=lambda t: (t["k"], fnum(t[col])))
                for npt, t in enumerate(pts):
                    lab = t["model"].split(": ")[-1].replace(" [c cells]", "c") if not t["role"].startswith("I1") \
                        else t["model"].replace("B12/B16 ", "")
                    ax.plot(t["k"] * (1.0 if rf == 4.0 else 1.04), 100 * fnum(t[col]), marker="o", ms=6,
                            color=OKABE[0] if rf == 4.0 else OKABE[1], mfc=OKABE[0] if rf == 4.0 else "white", ls="none")
                    if rf == 4.0:
                        ax.annotate(lab, (t["k"], 100 * fnum(t[col])), xytext=(5, 3 if npt % 2 == 0 else -9),
                                    textcoords="offset points",
                                    fontsize=6, color="0.15")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xticks([4, 6, 8, 10, 15, 20, 30], ["4", "6", "8", "10", "15", "20", "30"])
            ax.minorticks_off()
            ax.set_xlim(3.5, 45)
            ax.set_xlabel("free parameters k")
            ax.set_ylabel("family RMS error [%]" if col == "rms" else "sensitive-part score $S_{fam}$ [%]")
            ax.set_title(f"{FAM_TXT[fam]}", fontsize=8.5)
            if col == "rms":
                for thr in (1, 2, 5):
                    ax.axhline(thr, color="0.5", lw=0.7, ls="--")
    axs[0, 0].plot([], [], "o", color=OKABE[0], label="$R_{fit}=4$, $R_{cut}=4$ (labelled)")
    axs[0, 0].plot([], [], "o", color=OKABE[1], mfc="white", label="$R_{fit}=2$, $R_{cut}=4$")
    axs[0, 0].legend(fontsize=7, loc="lower left")
    fig.suptitle("I4: observable error (top) and sensitive-part score (bottom) vs number of parameters "
                 "(B12/B16 per-bin fits: k = 30)", fontsize=9.5)
    fig.savefig(FIG / "fig_I4_error_score_vs_k.png", dpi=150)
    plt.close(fig)
    # R_cut curves for key models (full cut grid from the supplement)
    curves = []
    keyroles = (("I1a", "B12/B16 (a)"), ("I1d", "B12/B16 (d)"), ("PB", "PB"),
                ("PL3", "PL3"), ("gamma", "free gamma"), ("BPLall", "BPLall"))
    for fam in FAMS:
        f = FAM_FIELD[fam]
        for role, rtxt in keyroles:
            for rf in (2.0, 4.0):
                for rc in R_CUTS:
                    if role.startswith("I1"):
                        o = role[2:]
                        lab = [r["label"] for r in i1 if r["grid"] == "f35" and r["field"] == f and r["objective"] == o
                               and abs(r["R_fit"] - rf) < 1e-9 and abs(r["R_cut"] - rc) < 1e-9]
                        if not lab:
                            continue
                        v, pr = headline(by[lab[0]], fam, "det" if fam == "DM" else "S1")
                        s = i1get("f35", f, o, rf, rc, f"S_{fam}")
                        mdl, k = rtxt, 30
                    else:
                        r = i3pick(f, role, rf, rc)
                        if r is None:
                            continue
                        v, pr = headline(by[r["label"]], fam, "det" if fam == "DM" else "S1")
                        s, mdl, k = r[f"S_{fam}"], r["model"], r["k"]
                    if np.isfinite(fnum(v)):
                        curves.append({"family": fam, "role": role, "model": mdl, "k": k, "R_fit": rf, "R_cut": rc,
                                       "rms": v, "pred": pr, "S": s})
    wcsv(TAB / "I4_rcut_curves.csv", curves)
    # non-monotone truncation curves: the error at a larger R_cut exceeds the error at a smaller one (cancellation
    # between the removed outskirts and an in-profile error of the opposite sign)
    nonmono = []
    series = {}
    for x in curves:
        series.setdefault(("I4", x["family"], x["role"], x["model"], x["R_fit"]), []).append((x["R_cut"], fnum(x["rms"]),
                                                                                         x["pred"]))
    for r in i1:
        if r["grid"] != "f35":
            continue
        for fam in (("tSZ", "tSZxDM") if r["field"] == "P" else ("kSZ", "DM", "tSZxDM")):
            col = "rms_DM_det" if fam == "DM" else f"rms_{fam}"
            pr = "det" if fam == "DM" else r[f"pred_{fam}"]
            series.setdefault(("I1", fam, f"{r['field']}|{r['objective']}", f"{'B12' if r['field'] == 'P' else 'B16'}"
                               f" ({r['objective']})", r["R_fit"]), []).append((r["R_cut"], fnum(r[col]), pr))
    for key, pts in series.items():
        pts = sorted(p for p in pts if np.isfinite(p[1]))
        rc = [p[0] for p in pts]
        v = [p[1] for p in pts]
        worst = 0.0
        where = None
        for i in range(len(v)):
            for j in range(i + 1, len(v)):
                if v[j] - v[i] > worst:
                    worst, where = v[j] - v[i], (rc[i], rc[j], v[i], v[j])
        if worst > 5e-4:                     # > 0.05 percentage points
            nonmono.append({"source": key[0], "family": key[1], "role": key[2], "model": key[3], "R_fit": key[4],
                            "R_cut_low": where[0], "R_cut_high": where[1], "rms_pct_at_low": 100 * where[2],
                            "rms_pct_at_high": 100 * where[3], "increase_pct_points": 100 * worst,
                            "rms_pct_at_Rcut4": 100 * v[rc.index(4.0)] if 4.0 in rc else np.nan,
                            "min_rms_pct": 100 * min(v), "R_cut_of_min": rc[int(np.argmin(v))],
                            "prediction": pts[0][2]})
    nonmono.sort(key=lambda r: -r["increase_pct_points"])
    wcsv(TAB / "I4_rcut_nonmonotone.csv", nonmono)
    # count distinct curves: the I4 curves of the I1 fits (roles I1a / I1d) duplicate I1 series
    nums["I4|rcut_nonmonotone_count"] = {
        "ladder_curves": sum(1 for r in nonmono if r["source"] == "I4" and not r["role"].startswith("I1")),
        "I1_curves": sum(1 for r in nonmono if r["source"] == "I1"),
        "I4_curves_of_I1_fits_(duplicates_of_I1_curves)": sum(1 for r in nonmono if r["source"] == "I4"
                                                               and r["role"].startswith("I1"))}
    nums["I4|rcut_nonmonotone_examples"] = [{k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}
                                            for r in nonmono[:12]]
    ver["checks"]["rcut_nonmonotone_curves"] = nums["I4|rcut_nonmonotone_count"]
    fig, axs = plt.subplots(1, 4, figsize=(15, 5.6), constrained_layout=True)
    for ax, fam in zip(axs, FAMS):
        for n, (role, rtxt) in enumerate(keyroles):
            for rf, ls in ((4.0, "-"), (2.0, "--")):
                c = sorted([x for x in curves if x["family"] == fam and x["role"] == role and x["R_fit"] == rf],
                           key=lambda x: x["R_cut"])
                if not c:
                    continue
                sel4 = summ["chosen"].get(f"{FAM_FIELD[fam]}|mass|4", {}).get("model")
                name = rtxt if role.startswith("I1") else \
                    f"{rtxt} ({c[0]['model']}{', selected at R_fit 4' if c[0]['model'] == sel4 else ''})"
                ax.plot([x["R_cut"] for x in c], [100 * fnum(x["rms"]) for x in c], ls=ls, marker=MARK[n], ms=4,
                        color=OKABE[n], lw=1.2, label=(f"{name}" if rf == 4.0 else None))
        for thr in (1, 2, 5):
            ax.axhline(thr, color="0.5", lw=0.7, ls=":")
        ax.set_yscale("log")
        ax.set_xlabel("truncation radius $R_{cut}$ [$R_{200c}$]")
        ax.set_ylabel("family RMS error vs A2-ii truth [%]")
        ax.set_title(f"{FAM_TXT[fam]}" + (" (det prediction)" if fam == "DM" else " (S1 prediction)"), fontsize=8.5)
        ax.legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False,
                  title="solid: $R_{fit}=4$; dashed: $R_{fit}=2$ (same model family)", title_fontsize=6.5)
    fig.suptitle("I4: how far to truncate - observable error vs $R_{cut}$ for fits to $R_{fit}$ = 4 and 2 "
                 "(electron models for kSZ, DM, tSZxDM)", fontsize=9.5)
    fig.savefig(FIG / "fig_I4_rcut_curves.png", dpi=150)
    plt.close(fig)
    # minimal adequate model per family (all I1 f35 objectives + ladder, every R_fit / R_cut evaluated)
    cand = []
    for r in i1:
        if r["grid"] != "f35":
            continue
        for fam in FAMS:
            if FAM_FIELD[fam] != r["field"]:
                continue
            cand.append({"family": fam, "model": f"{'B12' if r['field'] == 'P' else 'B16'}({r['objective']})", "k": 30,
                         "R_fit": r["R_fit"], "R_cut": r["R_cut"], "rms": r[f"rms_{fam}"], "pred": r[f"pred_{fam}"],
                         "S": r[f"S_{fam}"], "prof": r[f"prof_{fam}"]})
    for r in i3:
        for fam in FAMS:
            if FAM_FIELD[fam] != r["field"]:
                continue
            cand.append({"family": fam, "model": r["model"] + (" [c cells]" if r["dset"] == "massc" else ""), "k": r["k"],
                         "R_fit": r["R_fit"], "R_cut": r["R_cut"], "rms": r[f"rms_{fam}"], "pred": r[f"pred_{fam}"],
                         "S": r[f"S_{fam}"], "prof": r[f"prof_{fam}"]})
    mins = []
    for fam in FAMS:
        for thr in (0.01, 0.02, 0.05):
            c = [x for x in cand if x["family"] == fam and np.isfinite(fnum(x["rms"])) and x["rms"] <= thr]
            best_any = min([x for x in cand if x["family"] == fam and np.isfinite(fnum(x["rms"]))],
                           key=lambda x: x["rms"])
            if not c:
                mins.append({"family": fam, "threshold": thr, "model": "none", "best_overall": best_any["model"],
                             "best_overall_rms": best_any["rms"], "best_overall_R_fit": best_any["R_fit"],
                             "best_overall_R_cut": best_any["R_cut"]})
                continue
            b = min(c, key=lambda x: (x["k"], x["R_fit"], x["R_cut"]))
            sr = min(c, key=lambda x: (x["R_fit"], x["R_cut"], x["k"]))
            mins.append({"family": fam, "threshold": thr, "model": b["model"], "k": b["k"], "R_fit": b["R_fit"],
                         "R_cut": b["R_cut"], "rms": b["rms"], "pred": b["pred"], "S": b["S"],
                         "smallest_range_model": sr["model"], "smallest_range_k": sr["k"],
                         "smallest_range_R_fit": sr["R_fit"], "smallest_range_R_cut": sr["R_cut"],
                         "smallest_range_rms": sr["rms"], "best_overall": best_any["model"],
                         "best_overall_rms": best_any["rms"], "best_overall_R_fit": best_any["R_fit"],
                         "best_overall_R_cut": best_any["R_cut"], "n_adequate": len(c)})
    wcsv(TAB / "I4_minimal_adequate_final.csv", mins)
    # stricter variant: profile-only error (det vs truth mean) <= threshold AND S1 error <= threshold, R_cut = 4 only
    # (no truncation cancellation; no help from the S1 scatter-model floor)
    mins_p = []
    for fam in FAMS:
        for thr in (0.01, 0.02, 0.05):
            c = [x for x in cand if x["family"] == fam and x["R_cut"] == 4.0 and np.isfinite(fnum(x["prof"]))
                 and np.isfinite(fnum(x["rms"])) and fnum(x["prof"]) <= thr and fnum(x["rms"]) <= thr]
            if not c:
                mins_p.append({"family": fam, "threshold": thr, "model": "none"})
                continue
            b = min(c, key=lambda x: (x["k"], x["R_fit"], fnum(x["prof"])))
            sr = min(c, key=lambda x: (x["R_fit"], x["k"], fnum(x["prof"])))
            mins_p.append({"family": fam, "threshold": thr, "model": b["model"], "k": b["k"], "R_fit": b["R_fit"],
                           "R_cut": 4.0, "rms_S1": b["rms"], "profile_only": b["prof"], "S": b["S"],
                           "smallest_R_fit_model": sr["model"], "smallest_R_fit_k": sr["k"],
                           "smallest_R_fit": sr["R_fit"], "smallest_R_fit_rms_S1": sr["rms"],
                           "smallest_R_fit_profile_only": sr["prof"], "n_adequate": len(c)})
    wcsv(TAB / "I4_minimal_adequate_profile_Rcut4.csv", mins_p)
    nums["I4|minimal_adequate_profile_Rcut4"] = [{k: (round(v, 5) if isinstance(v, float) else v) for k, v in m.items()}
                                                 for m in mins_p]
    nums["I4|minimal_adequate"] = [{k: (round(v, 5) if isinstance(v, float) else v) for k, v in m.items()} for m in mins]
    # where the weight is (family weights) and where models fail (signed residual by region, R_fit = R_cut = 4)
    ev = np.load(RES / "evaluate.npz")
    xe = ev["x_edges"]
    import i_common as IC
    regions = ((0.0, 0.3), (0.3, 0.7), (0.7, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 3.0), (3.0, 4.0))
    wrows = []
    for fam in ("tSZ", "kSZ", "DM", "tSZxDM_P", "tSZxDM_E"):
        w = ev[f"w__{fam}"]
        tot = w.sum()
        row = {"family": fam}
        for a, b in regions:
            sel = (xe[:-1] >= a - 1e-9) & (xe[1:] <= b + 1e-9)
            row[f"share_{a:g}-{b:g}"] = float(w[:, sel].sum() / tot)
        cum = np.cumsum(w.sum(axis=0)) / tot
        cum0 = np.concatenate([[0.0], cum])
        for q in (0.5, 0.9, 0.95, 0.99):
            # 'x_at_q' (kept name) = OUTER EDGE of the shell in which the cumulative share first reaches q;
            # '_interp' = linear interpolation of the cumulative share in x inside that shell
            row[f"x_at_{int(q * 100)}pct"] = float(xe[1:][np.searchsorted(cum, q)])
            row[f"x_at_{int(q * 100)}pct_interp"] = float(np.interp(q, cum0, xe))
        bs = w.sum(axis=1) / tot
        for c in range(len(bs)):
            row[f"bin{c}_share"] = float(bs[c])
        wrows.append(row)
    wcsv(TAB / "I4_importance_regions.csv", wrows)
    nums["I4|importance_regions"] = {r["family"]: {k: round(v, 4) for k, v in r.items() if k != "family"} for r in wrows}
    nums["I4|importance_regions_note"] = ("shares are normalised within the 10 calibrated bins (|importance|, ell "
                                          "100-3000); x_at_q = outer shell edge where the cumulative share first "
                                          "reaches q, x_at_q_interp = interpolated inside the shell")
    # what fraction of each observable the calibrated bins carry at all (signed importance, Test H sens.npz)
    hs = np.load(ROOT / "H_sensitivity" / "results" / "sens.npz")
    cal = np.asarray(hs["cal_bins"], int)
    ell_h = np.asarray(hs["ell"], float)
    ok_ell = (ell_h >= 100 - 1e-9) & (ell_h <= 3000 + 1e-9)
    cs_rows = []
    for nm in ("yytot", "ksz1h", "ydmtot_P", "ydmtot_E", "dm_mean", "dm_var", "dm_q16", "dm_q50", "dm_q84", "dm_q975",
               "dm_q999"):
        I = np.asarray(hs[f"imp__a2ii__{nm}"], float)
        if I.ndim == 3:
            tot_all = I.sum(axis=(0, 1))[ok_ell]
            tot_cal = I[cal].sum(axis=(0, 1))[ok_ell]
            cs_rows.append({"observable": nm, "ell_range": "100-3000", "sum_all_bins_min": float(tot_all.min()),
                            "sum_all_bins_max": float(tot_all.max()), "calibrated_share_min": float(tot_cal.min()),
                            "calibrated_share_max": float(tot_cal.max())})
        else:
            cs_rows.append({"observable": nm, "ell_range": "", "sum_all_bins_min": float(I.sum()),
                            "sum_all_bins_max": float(I.sum()), "calibrated_share_min": float(I[cal].sum()),
                            "calibrated_share_max": float(I[cal].sum())})
    wcsv(TAB / "I4_calibrated_signal_share.csv", cs_rows)
    nums["I4|calibrated_bins_signed_importance_share"] = {r["observable"]: [round(r["calibrated_share_min"], 3),
                                                                           round(r["calibrated_share_max"], 3)]
                                                          for r in cs_rows}
    ver["checks"]["signed_importance_sum_all_bins_minmax"] = [min(r["sum_all_bins_min"] for r in cs_rows),
                                                              max(r["sum_all_bins_max"] for r in cs_rows)]
    # weight in shells inside the first resolved F shell of each bin (not fitted: the model is extrapolated there)
    z0 = np.load(HERE / "inputs" / "i0_variants.npz")
    f35e = np.asarray(fits35["f_edges"])
    un_rows = []
    first_res = {}
    for fam, fld in (("tSZ", "P"), ("kSZ", "E"), ("DM", "E")):
        res = z0[f"res_a2ii_{fld}"].astype(bool)
        first_res[fld] = [round(float(f35e[np.argmax(res[c])]), 4) for c in range(res.shape[0])]
        w = ev[f"w__{fam}"]
        unres = np.zeros_like(w, dtype=bool)
        for c in range(w.shape[0]):
            unres[c] = xe[1:] <= f35e[np.argmax(res[c])] + 1e-9
        core = (xe[1:] <= 0.3 + 1e-9)[None, :] & np.ones_like(unres)
        share_un = float(w[unres].sum() / w.sum())
        share_core_res = float(w[core & ~unres].sum() / w.sum())
        for lab in [k[4:] for k in ev.files if k.startswith("M4__") and "|Rf4|Rc4" in k
                    and (k.startswith("M4__I3") or k.startswith("M4__I1|f35|"))]:
            f_l = lab.split("|")[2] if lab.startswith("I1") else lab.split("|")[1]
            if f_l != fld:
                continue
            M, m = ev[f"M4__{lab}"], ev[f"mstack_{fld}"]
            pos = (m > 0) & (M > 0)
            d = np.log(np.where(pos, M, 1.0) / np.where(pos, m, 1.0))
            ok_u, ok_r = unres & pos, core & ~unres & pos
            un_rows.append({"family": fam, "label": lab, "weight_share_unresolved": share_un,
                            "weight_share_resolved_x_lt_0.3": share_core_res,
                            "mean_ln_ratio_unresolved": float(np.sum(w[ok_u] * d[ok_u]) / w[ok_u].sum()),
                            "mean_ln_ratio_resolved_x_lt_0.3": float(np.sum(w[ok_r] * d[ok_r]) / w[ok_r].sum())})
    wcsv(TAB / "I4_unresolved_core.csv", un_rows)
    nums["I4|first_resolved_shell_inner_edge_x_per_bin"] = first_res
    nums["I4|unresolved_core"] = {f"{r['label']}|{r['family']}": {k: round(v, 4) for k, v in r.items()
                                                                  if isinstance(v, float)} for r in un_rows}
    # the same unresolved-core weight split per calibrated bin (share of the family total over all ten bins)
    cal_lab = ["13.5-13.7", "13.7-13.9", "13.9-14.1", "14.1-14.3", "14.3-14.5", "14.5-14.7", "14.7-14.9",
               "14.9-15.1", "15.1-15.3", "15.3-15.5"]
    ub_rows = []
    for fam, fld in (("tSZ", "P"), ("kSZ", "E"), ("DM", "E")):
        res = z0[f"res_a2ii_{fld}"].astype(bool)
        w = ev[f"w__{fam}"]
        for c in range(w.shape[0]):
            x_in = float(f35e[np.argmax(res[c])])
            un = xe[1:] <= x_in + 1e-9
            ub_rows.append({"family": fam, "bin": c, "log10M200c_bin": cal_lab[c], "first_resolved_inner_edge_x": x_in,
                            "n_unresolved_fine_shells": int(un.sum()),
                            "weight_share_of_family_total": float(w[c, un].sum() / w.sum()),
                            "bin_weight_share": float(w[c].sum() / w.sum())})
    wcsv(TAB / "I4_unresolved_core_by_bin.csv", ub_rows)
    nums["I4|unresolved_core_by_bin_pct (family: per-bin shares, bins 0-2 sum, all-bin sum)"] = {
        fam: [[round(100 * r["weight_share_of_family_total"], 3) for r in ub_rows if r["family"] == fam],
              round(100 * sum(r["weight_share_of_family_total"] for r in ub_rows if r["family"] == fam and r["bin"] < 3), 3),
              round(100 * sum(r["weight_share_of_family_total"] for r in ub_rows if r["family"] == fam), 3)]
        for fam in ("tSZ", "kSZ", "DM")}
    nums["I4|first_resolved_inner_edge_x_per_bin (A2-ii strict, fit grid)"] = [
        round(r["first_resolved_inner_edge_x"], 4) for r in ub_rows if r["family"] == "tSZ"]
    # observable truncation radii (Test H sens.npz trunc_all__a2ii__<obs>[R_t(, ell)]: every calibrated halo of the
    # A2-ii truth truncated at R_t = each fine edge, fallback haloes unchanged, scatter-inclusive moments).
    # retained(R_t) = O(R_t) / O(4 R200c); R_q = first R_t with retained >= q (linear in x between edges);
    # lost(R_t) = 1 - retained(R_t); linear estimate of lost(R_t) = k_O * (signed importance of the calibrated
    # bins beyond R_t) (first order in the removed profile; k_O = 2 for C_l and the DM variance, 1 for the DM mean).
    xe_h = np.asarray(hs["x_edges"], float)
    trows = []
    for nm, k_o in (("yytot", 2), ("ksz1h", 2), ("ydmtot_Ecut", 1), ("ydmtot_Pcut", 1), ("dm_mean", 1), ("dm_var", 2)):
        cur = np.asarray(hs[f"trunc_all__a2ii__{nm}"], float)
        imp_nm = {"ydmtot_Ecut": "ydmtot_E", "ydmtot_Pcut": "ydmtot_P"}.get(nm, nm)
        I = np.asarray(hs[f"imp__a2ii__{imp_nm}"], float)[cal]
        for e in ((100, 300, 1000, 3000) if cur.ndim == 2 else (None,)):
            if e is None:
                c_, Ie = cur, I
            else:
                li = int(np.argmin(np.abs(ell_h - e)))
                c_, Ie = cur[:, li], I[..., li]
            ret = c_ / c_[-1]
            row = {"observable": nm, "ell": e if e is not None else "", "k_O": k_o}
            for q in (0.5, 0.9, 0.95, 0.99):
                kq = int(np.argmax(ret >= q))
                row[f"R{int(round(100 * q))}"] = float(xe_h[0]) if kq == 0 else \
                    float(np.interp(q, [ret[kq - 1], ret[kq]], [xe_h[kq - 1], xe_h[kq]]))
            for rt in (1.0, 1.1516, 1.5, 2.0, 3.0):
                kt = int(np.argmin(np.abs(xe_h - rt)))
                beyond = Ie[:, kt:].sum()
                row[f"lost_pct_at_{rt:g}"] = 100 * float(1 - ret[kt])
                row[f"lost_pct_linear_at_{rt:g}"] = 100 * float(k_o * beyond)
            trows.append(row)
    wcsv(TAB / "I4_observable_truncation_radii.csv", trows)
    nums["I4|observable_truncation_radii (R50,R90,R95,R99; lost % at 1,1.15,1.5,2,3)"] = {
        f"{r['observable']}|{r['ell']}": [[round(r[f"R{q}"], 3) for q in (50, 90, 95, 99)],
                                          [round(r[f"lost_pct_at_{t:g}"], 2) for t in (1.0, 1.1516, 1.5, 2.0, 3.0)]]
        for r in trows}
    resid = []
    keylabs = [k for k in ev.files if k.startswith("M4__") and ("|Rf4|Rc4" in k) and
               (k.startswith("M4__I3") or k.startswith("M4__I1|f35|"))]
    for k in keylabs:
        lab = k[4:]
        f = lab.split("|")[2] if lab.startswith("I1") else lab.split("|")[1]
        m = ev[f"mstack_{f}"]
        for fam in (("tSZ",) if f == "P" else ("kSZ", "DM")):
            reg = IC.score_regions(ev[k], m, ev[f"w__{fam}"], xe, 4.0, regions=regions)
            row = {"label": lab, "family": fam}
            for rg, d in reg.items():
                row[f"wshare_{rg}"] = d["weight_share"]
                row[f"mean_d_{rg}"] = d["mean_d"]
                row[f"rms_d_{rg}"] = d["rms_d"]
            resid.append(row)
    wcsv(TAB / "I4_residual_by_region.csv", resid)
    nums["I4|residual_by_region_weighted_mean_ln_pct (R_fit=R_cut=4)"] = {
        f"{r['label']}|{r['family']}": {rg: round(100 * fnum(r[f"mean_d_{rg}"]), 2) for rg in
                                        [f"{a:g}-{b:g}" for a, b in regions]} for r in resid}
    # residual profile figure (no twin axes: residual panels above, importance panels below)
    show = [0, 4, 8]
    fig, axs = plt.subplots(4, 3, figsize=(13, 11), constrained_layout=True, sharex=True,
                            gridspec_kw={"height_ratios": [2.2, 1, 2.2, 1]})
    xm = np.sqrt(np.maximum(xe[:-1], 0.5 * xe[1]) * xe[1:])
    cal_labels = ["13.5-13.7", "13.7-13.9", "13.9-14.1", "14.1-14.3", "14.3-14.5", "14.5-14.7", "14.7-14.9",
                  "14.9-15.1", "15.1-15.3", "15.3-15.5"]
    for blk, f in enumerate(("P", "E")):
        picks = [(f"I1|f35|{f}|a|Rf4|Rc4", f"{'B12' if f == 'P' else 'B16'} (a)"),
                 (f"I1|f35|{f}|d|Rf4|Rc4", f"{'B12' if f == 'P' else 'B16'} (d)")]
        seen_models = set()
        for role in ("selected", "PB", "PL3", "gamma", "BPLall", "cbest"):
            r = i3pick(f, role, 4.0, 4.0)
            if r is not None and f"M4__{r['label']}" in ev.files and (r["model"], r["dset"]) not in seen_models:
                seen_models.add((r["model"], r["dset"]))
                picks.append((r["label"], f"{role}: {r['model']}" + (" (c cells, bin-mean c)" if r["dset"] == "massc" else "")))
        for j, c in enumerate(show):
            ax = axs[2 * blk, j]
            m = ev[f"mstack_{f}"][c]
            for n, (lab, name) in enumerate(picks):
                M = ev[f"M4__{lab}"][c]
                ok = (m > 0) & (M > 0)
                ax.plot(xm[ok], np.log(M[ok] / m[ok]), color=OKABE[n], lw=1.3, label=name,
                        ls="-" if n < 4 else "--")
            ax.axhline(0, color="black", lw=0.7)
            ax.set_xscale("log")
            ax.set_ylim(-0.5, 0.5)
            ax.set_ylabel("ln(model / A2-ii stack)")
            ax.set_title(f"{'pressure' if f == 'P' else 'electron density'}, log M200c {cal_labels[c]} "
                         f"($R_{{fit}}=R_{{cut}}=4$)", fontsize=8.5)
            axw = axs[2 * blk + 1, j]
            dl = np.log(xe[1:] / np.maximum(xe[:-1], 0.5 * xe[1]))
            for n2, fam in enumerate(("tSZ",) if f == "P" else ("kSZ", "DM")):
                w = ev[f"w__{fam}"]
                axw.step(xm, w[c] / dl / w.sum(), where="mid", color=["0.25", OKABE[3]][n2], lw=1.2,
                         label=f"{fam} importance")
            axw.set_ylabel("importance /\nd ln x (share)")
            axw.set_xscale("log")
            if j == 0:
                axw.legend(fontsize=7, loc="upper left")
        h, l = axs[2 * blk, 0].get_legend_handles_labels()
        axs[2 * blk, 2].legend(h, l, fontsize=6.5, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    for ax in axs[-1]:
        ax.set_xlabel("$x = r / R_{200c}$")
    fig.suptitle("I4: residuals of key fits on the A2-ii stacks (upper rows) and where the observables' weight is "
                 "(lower rows; Test H A2-ii importance of the bin, per unit ln x, as a share of the family total)",
                 fontsize=9)
    fig.savefig(FIG / "fig_I4_residual_profiles.png", dpi=150)
    plt.close(fig)
    # ================================================================== I5 stacking choices
    i5 = []
    for src_by, src_meta in ((by, meta), (ph_by, ph_meta)):
        for lab, bp in src_by.items():
            m = src_meta[lab]
            if m["block"] != "I5" or str(m.get("variant", "")).startswith("stack_"):
                continue
            row = {"label": lab, "variant": m["variant"], "field": m["field"], "model": m["model"],
                   "R_fit": fnum(m["R_fit"]), "R_cut": fnum(m["R_cut"]), "k": int(fnum(m["k"])),
                   "selected": m.get("selected"), "source": m.get("source")}
            for fam in (("tSZ", "tSZxDM") if m["field"] == "P" else ("kSZ", "DM", "tSZxDM")):
                v, p = headline(bp, fam)
                row[f"rms_{fam}"], row[f"pred_{fam}"] = v, p
                S, c, si = score(bp, fam)
                row[f"S_{fam}"] = S
                row[f"rmsdet_{fam}"], row[f"prof_{fam}"] = det_profile(bp, fam)
            row["note"] = ("identical to the A2-ii reference by construction: fitted on the strict shells, where the "
                           "per-halo stack equals A2-ii; superseded by perhalo_own") if m["variant"] == "perhalo" else \
                ("per-halo rule fitted on its own resolved shells (maas run 3, results/perhalo)"
                 if m["variant"] == "perhalo_own" else "")
            i5.append(row)
    wcsv(TAB / "I5_variants_final.csv", i5)
    VARS = ["a2ii", "a2ii_agn", "allgas", "a2i", "unweighted", "median", "m05", "perhalo_own", "xray"]
    VTXT = {"a2ii": "A2-ii (reference)", "a2ii_agn": "A2-ii, recent-AGN gas kept", "allgas": "all gas (no partition)",
            "a2i": "A2-i partition", "unweighted": "A2-ii, unweighted mean", "median": "A2-ii, median",
            "m05": "A2-ii, 0.5-dex bins", "perhalo": "A2-ii, per-halo rule on strict shells (= reference)",
            "perhalo_own": "A2-ii, per-halo resolution rule\n(fitted on its own resolved shells)",
            "xray": "all gas, X-ray phase only"}
    fig, axs = plt.subplots(1, 4, figsize=(16, 6.0), constrained_layout=True)
    for ax, fam in zip(axs, FAMS):
        f = FAM_FIELD[fam]
        mods = []
        for r in i5:
            if r["field"] == f and r["R_cut"] == 4.0:
                key = (r["model"], r["R_fit"])
                if key not in mods:
                    mods.append(key)
        mods.sort(key=lambda t: (t[0] not in ("B12", "B16"), t[0], t[1]))
        Z = np.full((len(VARS), len(mods)), np.nan)
        for i, v in enumerate(VARS):
            for j, (mdl, rf) in enumerate(mods):
                c = [r for r in i5 if r["variant"] == v and r["field"] == f and r["model"] == mdl and r["R_fit"] == rf
                     and r["R_cut"] == 4.0]
                if c:
                    Z[i, j] = 100 * fnum(c[0][f"rms_{fam}"])
        ax.imshow(Z, cmap="Blues", norm=LogNorm(vmin=0.3, vmax=60), aspect="auto")
        for (p, q), v in np.ndenumerate(Z):
            ax.text(q, p, "-" if not np.isfinite(v) else f"{v:.1f}", ha="center", va="center", fontsize=7,
                    color="white" if np.isfinite(v) and v > 8 else "black")
        ax.set_xticks(range(len(mods)), [f"{m}, $R_{{fit}}$={rf:g}" for m, rf in mods], fontsize=7, rotation=40,
                      ha="right", rotation_mode="anchor")
        ax.set_yticks(range(len(VARS)), [VTXT[v] for v in VARS], fontsize=7)
        ax.grid(False)
        ax.set_title(f"{FAM_TXT[fam]} [%]", fontsize=8.5)
    fig.suptitle("I5: stack used for the fit -> family RMS error against the A2-ii painting target "
                 "($R_{cut}$ = 4, S1; '-' = per-bin law not defined on the 0.5-dex binning)", fontsize=9.5)
    fig.supxlabel("columns: model fitted to the variant's stack, and its $R_{fit}$ [$R_{200c}$] (B12/B16: objective "
                  "(a); Lee22 laws: diagonal weights).  The per-halo rule row is fitted on its own resolved shells "
                  "(maas run 3); its strict-shell version equals the reference by construction and is not shown.",
                  fontsize=8)
    fig.savefig(FIG / "fig_I5_stacking_choices.png", dpi=150)
    plt.close(fig)
    for r in i5:
        if r["R_cut"] != 4.0:
            continue
        for fam in (("tSZ",) if r["field"] == "P" else ("kSZ", "DM")):
            nums[f"I5|{r['variant']}|{r['field']}|{r['model']}|Rf{r['R_fit']:g}|rms_{fam}"] = round(fnum(r[f"rms_{fam}"]), 5)
            nums[f"I5|{r['variant']}|{r['field']}|{r['model']}|Rf{r['R_fit']:g}|prof_{fam}"] = round(fnum(r[f"prof_{fam}"]), 5)
            nums[f"I5|{r['variant']}|{r['field']}|{r['model']}|Rf{r['R_fit']:g}|S_{fam}"] = round(fnum(r[f"S_{fam}"]), 4)
    nums["_sources"]["I5"] = ("results/tables/I5_variants_final.csv, I5_parameter_shifts_summary.csv, "
                              "I5_stack_differences.csv, I5_perhalo_rule.csv, I5_perhalo_rule_bins.csv")
    # parameter shifts of B12 / B16 (objective a) relative to A2-ii, R_fit = 4 and 2 (run 1 table + run 3 own shells)
    sh = rcsv(TAB / "I5_parameter_shifts.csv")
    if phf is not None:
        for key, rec in phf["B"].items():
            f, rf, i = key.split("|")
            ref = fits35["variants"].get(f"a2ii|{f}|{rf}|{i}")
            if rec is None or ref is None:
                continue
            d = np.log(np.asarray(rec["parameters"], float) / np.asarray(ref["parameters"], float))
            sh.append({"variant": "perhalo_own", "field": f, "R_fit": rf, "bin": i,
                       **{f"dlnp{j}": float(d[j]) for j in range(3)}})
    pnames = {"P": ("P0", "x_c", "beta"), "E": ("rho0", "alpha", "beta")}
    shs = []
    for v in VARS[1:] + ["perhalo"]:
        for f in "PE":
            for rf in ("2", "4"):
                rr = [x for x in sh if x["variant"] == v and x["field"] == f and x["R_fit"] == rf]
                if not rr:
                    continue
                d = np.array([[fnum(x[f"dlnp{i}"]) for i in range(3)] for x in rr])
                shs.append({"variant": v, "field": f, "R_fit": rf, **{f"max_abs_dln_{pnames[f][i]}": float(np.abs(d[:, i]).max())
                                                                     for i in range(3)},
                            **{f"median_dln_{pnames[f][i]}": float(np.median(d[:, i])) for i in range(3)}})
    wcsv(TAB / "I5_parameter_shifts_summary.csv", shs)
    fig, axs = plt.subplots(2, 3, figsize=(13, 6.8), constrained_layout=True, sharex=True)
    for i, f in enumerate("PE"):
        for j in range(3):
            ax = axs[i, j]
            for n, v in enumerate([v for v in VARS[1:] if v != "m05"]):
                rr = sorted([x for x in sh if x["variant"] == v and x["field"] == f and x["R_fit"] == "4"],
                            key=lambda x: int(x["bin"]))
                if rr:
                    ax.plot([int(x["bin"]) for x in rr], [fnum(x[f"dlnp{j}"]) for x in rr], marker=MARK[n], ms=4,
                            color=OKABE[n], lw=1.0, label=VTXT[v].replace("\n", " "))
            ax.axhline(0, color="black", lw=0.7)
            ax.set_title(f"{'B12' if f == 'P' else 'B16'} {pnames[f][j]}: ln(p_variant / p_A2-ii), $R_{{fit}}=4$",
                         fontsize=8.5)
            ax.set_xticks(range(10), cal_labels, rotation=45, ha="right", fontsize=6.5)
            if i == 1:
                ax.set_xlabel("log10 M200c [Msun] bin")
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncol=4, fontsize=7)
    fig.savefig(FIG / "fig_I5_parameter_shifts.png", dpi=150)
    plt.close(fig)
    # stack differences of the variants (input side)
    z = np.load(HERE / "inputs" / "i0_variants.npz")
    f35 = np.asarray(fits35["f_edges"])
    sd = []
    for v in ["a2ii_agn", "allgas", "a2i", "unweighted", "median", "perhalo", "xray"]:
        for f in "PE":
            a, b = z[f"mean_a2ii_{f}"], z[f"mean_{v}_{f}"]
            ok = (a > 0) & (b > 0) & np.isfinite(a) & np.isfinite(b) & z[f"res_a2ii_{f}"].astype(bool)
            d = np.where(ok, np.log(np.where(ok, b, 1) / np.where(ok, a, 1)), np.nan)
            row = {"variant": v, "field": f, "compared_on": "A2-ii strict-resolved shells of the 35-shell fit grid"}
            for lo, hi in ((0.0, 1.0), (1.0, 2.0), (2.0, 4.0)):
                sel = (f35[:-1] >= lo - 1e-9) & (f35[1:] <= hi + 1e-9)
                dd = d[:, sel]
                row[f"median_abs_dln_{lo:g}-{hi:g}"] = float(np.nanmedian(np.abs(dd)))
                row[f"max_abs_dln_{lo:g}-{hi:g}"] = float(np.nanmax(np.abs(dd)))
            sd.append(row)
    for f in "PE":                  # the per-halo rule where it differs: shells it resolves and the strict rule does not
        a, b = z[f"mean_a2ii_{f}"], z[f"mean_perhalo_{f}"]
        add = z[f"res_perhalo_{f}"].astype(bool) & ~z[f"res_a2ii_{f}"].astype(bool) & (a > 0) & (b > 0) \
            & np.isfinite(a) & np.isfinite(b)
        d = np.where(add, np.log(np.where(add, b, 1) / np.where(add, a, 1)), np.nan)
        row = {"variant": "perhalo (own shells)", "field": f,
               "compared_on": "shells resolved by the per-halo rule but not by the strict rule (all at x < 0.33): the "
                              "per-halo stack averages only the haloes that resolve the shell, A2-ii all haloes"}
        for lo, hi in ((0.0, 1.0), (1.0, 2.0), (2.0, 4.0)):
            sel = (f35[:-1] >= lo - 1e-9) & (f35[1:] <= hi + 1e-9)
            dd = d[:, sel]
            fin = np.isfinite(dd)
            row[f"median_abs_dln_{lo:g}-{hi:g}"] = float(np.median(np.abs(dd[fin]))) if fin.any() else np.nan
            row[f"max_abs_dln_{lo:g}-{hi:g}"] = float(np.max(np.abs(dd[fin]))) if fin.any() else np.nan
        sd.append(row)
    wcsv(TAB / "I5_stack_differences.csv", sd)
    # ------------------------------------------------------------------ the per-halo resolution rule (maas run 3)
    if phf is not None and (HERE / "inputs" / "i0_perhalo_x200.npz").exists():
        zp = np.load(HERE / "inputs" / "i0_perhalo_x200.npz")
        xph = np.asarray(zp["x_edges_x200"], float)
        evp = np.load(PHD / "evaluate.npz")
        prb = []
        for f in "PE":
            a, b = z[f"mean_a2ii_{f}"], z[f"mean_perhalo_{f}"]
            own, st = zp[f"own_fit_{f}"].astype(bool), zp[f"strict_fit_{f}"].astype(bool)
            for c in range(10):
                add = own[c] & ~st[c]
                with np.errstate(divide="ignore", invalid="ignore"):
                    lr = np.log(b[c] / a[c])
                add_x = zp[f"own_{f}"][c].astype(bool) & ~zp[f"strict_{f}"][c].astype(bool)   # fine-grid added
                row = {"field": f, "bin": c, "log10M200c_bin": cal_lab[c], "N_haloes": int(zp["N"][c]),
                       "fit_shells_strict": int(st[c].sum()), "fit_shells_own": int(own[c].sum()),
                       "x_inner_strict": float(f35[np.argmax(st[c])]), "x_inner_own": float(f35[np.argmax(own[c])]),
                       "n_haloes_min_on_added_shells": int(zp[f"nused_fit_{f}"][c][add].min()) if add.any() else "",
                       "weight_frac_min_on_added_shells": float(zp[f"wfrac_fit_{f}"][c][add].min()) if add.any() else "",
                       "ln_perhalo_over_a2ii_max_on_added": float(np.nanmax(lr[add])) if add.any() else "",
                       "ln_perhalo_over_a2ii_min_on_added": float(np.nanmin(lr[add])) if add.any() else "",
                       "fine_shells_added": int(add_x.sum())}
                for fam in (("tSZ",) if f == "P" else ("kSZ", "DM")):
                    w = ev[f"w__{fam}"]
                    row[f"weight_share_added_{fam}"] = float(w[c, add_x].sum() / w.sum())
                # B12 / B16 (a), R_fit 4: own-shell fit vs the A2-ii fit, ln ratio of the fine-shell model averages
                k_own, k_ref = f"M4__I5|perhalo_own|{f}|B|Rf4|Rc4", f"M4__I1|f35|{f}|a|Rf4|Rc4"
                if k_own in evp.files and k_ref in ev.files:
                    M1, M0 = evp[k_own][c], ev[k_ref][c]
                    okm = (M1 > 0) & (M0 > 0)
                    lfr = np.where(okm, np.log(np.where(okm, M1, 1.0) / np.where(okm, M0, 1.0)), np.nan)
                    xin, xout = xe[1:] <= 0.33 + 1e-9, xe[:-1] >= 1.0 - 1e-9
                    row["Rf4_fit_max_ln_ratio_x_lt_0.33"] = float(np.nanmax(lfr[xin]))
                    row["Rf4_fit_max_abs_ln_ratio_x_1_to_4"] = float(np.nanmax(np.abs(lfr[xout])))
                prb.append(row)
        wcsv(TAB / "I5_perhalo_rule_bins.csv", prb)
        nums["I5|perhalo_rule_bins"] = {f"{r['field']}|{r['log10M200c_bin']}": {k: (round(v, 4) if isinstance(v, float)
                                                                                   else v) for k, v in r.items()
                                                                               if k not in ("field", "bin",
                                                                                            "log10M200c_bin")}
                                        for r in prb}
        for fam in ("tSZ", "kSZ", "DM"):
            nums[f"I5|perhalo_rule_added_shells_weight_share_pct|{fam}"] = round(
                100 * sum(r.get(f"weight_share_added_{fam}", 0.0) for r in prb), 3)
        # observables: the stacks themselves (no fit) and the own-shell fits vs the A2-ii reference fits
        prr = []

        def addrow(kind, model, rf, bp_ref, bp_ph, field):
            row = {"comparison": kind, "model": model, "R_fit": rf, "field": field}
            for fam in (("tSZ", "tSZxDM") if field == "P" else ("kSZ", "DM", "tSZxDM")):
                for tag, bp in (("ref", bp_ref), ("perhalo", bp_ph)):
                    if bp is None:
                        continue
                    v, p = headline(bp, fam)
                    row[f"{tag}_rms_S1_{fam}_pct"] = 100 * v
                    row[f"{tag}_profile_only_{fam}_pct"] = 100 * det_profile(bp, fam)[1]
                    row[f"{tag}_S_{fam}_pct"] = 100 * score(bp, fam)[0]
            prr.append(row)

        for f in "PE":
            addrow("stack itself (no fit): A2-ii vs per-halo rule", "stack", "", ph_by.get(f"I5|stack_a2ii|{f}|stack|Rc4"),
                   ph_by.get(f"I5|stack_perhalo|{f}|stack|Rc4"), f)
            for lab in sorted(l for l in ph_by if l.startswith(f"I5|perhalo_own|{f}|") and l.endswith("|Rc4")):
                ref_lab = lab.replace("perhalo_own", "a2ii")
                mdl = ph_meta[lab]["model"]
                addrow("fit: A2-ii reference vs per-halo rule on its own shells", mdl, fnum(ph_meta[lab]["R_fit"]),
                       by.get(ref_lab), ph_by.get(lab), f)
        wcsv(TAB / "I5_perhalo_rule.csv", prr)
        nums["I5|perhalo_rule"] = [{k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()} for r in prr]
        # exact S1 scatter-model error of the target profile: the A2-ii stack painted with S1 against the truth
        ex = {}
        for f, fams in (("P", ("tSZ", "tSZxDM")), ("E", ("kSZ", "DM", "tSZxDM"))):
            bp = ph_by.get(f"I5|stack_a2ii|{f}|stack|Rc4")
            if bp is None:
                continue
            for fam in fams:
                ex[f"{fam}|{f}"] = {"S1_vs_truth_pct": round(100 * headline(bp, fam, "S1")[0], 4),
                                    "det_vs_truthmean_pct": round(100 * det_profile(bp, fam)[1], 6)}
        nums["I4|S1_error_of_target_profile_EXACT (A2-ii stack painted with S1; det vs truth mean must be 0)"] = ex
        ver["checks"]["stack_a2ii_det_vs_truthmean_family_rms_max"] = max(
            (abs(v["det_vs_truthmean_pct"]) / 100 for v in ex.values()), default=None)
        # figure: the per-halo rule (stack ratio, resolving weight fraction, fitted-profile ratios)
        fig, axs = plt.subplots(2, 2, figsize=(12.5, 8.4), constrained_layout=True)
        cmap = plt.get_cmap("viridis")
        xm35 = np.sqrt(np.maximum(f35[:-1], 0.5 * f35[1]) * f35[1:])
        xm_f = np.sqrt(np.maximum(xe[:-1], 0.5 * xe[1]) * xe[1:])
        for c in range(10):
            col = cmap(c / 9)
            a, b = z["mean_a2ii_E"][c], z["mean_perhalo_E"][c]
            own, st = zp["own_fit_E"][c].astype(bool), zp["strict_fit_E"][c].astype(bool)
            ok = own & (a > 0) & (b > 0)
            axs[0, 0].plot(xm35[ok], np.log(b[ok] / a[ok]), color=col, lw=1.3, marker="o", ms=2.5, label=cal_lab[c])
            axs[0, 0].plot([f35[np.argmax(st)]], [0], marker="|", ms=12, mew=2, color=col)
            okw = zp["wfrac_fit_E"][c] > 0
            axs[0, 1].plot(xm35[okw], zp["wfrac_fit_E"][c][okw], color=col, lw=1.3, label=cal_lab[c])
            for ax, f, lab0 in ((axs[1, 0], "E", "B16"), (axs[1, 1], "P", "B12")):
                k_own, k_ref = f"M4__I5|perhalo_own|{f}|B|Rf4|Rc4", f"M4__I1|f35|{f}|a|Rf4|Rc4"
                if k_own in evp.files and k_ref in ev.files:
                    M1, M0 = evp[k_own][c], ev[k_ref][c]
                    okm = (M1 > 0) & (M0 > 0)
                    ax.plot(xm_f[okm], np.log(M1[okm] / M0[okm]), color=col, lw=1.3, label=cal_lab[c])
        axs[0, 0].set_ylabel("ln(per-halo-rule stack / A2-ii stack), electrons")
        axs[0, 0].set_title("stacked electron density: per-halo rule vs strict rule (A2-ii)\n"
                            "on the shells the per-halo rule resolves; ticks at 0 = first strict-resolved shell",
                            fontsize=8.5)
        axs[0, 1].set_ylabel("inclusion-weight fraction of the bin's haloes\nthat resolve the shell")
        axs[0, 1].set_title("haloes entering each shell under the per-halo rule\n(strict rule: a shell is used only "
                            "where this is 1)", fontsize=8.5)
        axs[1, 0].set_ylabel("ln(B16 fit to per-halo stack / B16 fit to A2-ii)")
        axs[1, 0].set_title("electron density: B16 objective (a), $R_{fit}$ = 4, own shells vs strict shells",
                            fontsize=8.5)
        axs[1, 1].set_ylabel("ln(B12 fit to per-halo stack / B12 fit to A2-ii)")
        axs[1, 1].set_title("pressure: B12 objective (a), $R_{fit}$ = 4, own shells vs strict shells", fontsize=8.5)
        for ax in axs.ravel():
            ax.set_xscale("log")
            ax.set_xlabel("$x = r / R_{200c}$")
            ax.axhline(0, color="black", lw=0.7)
        axs[0, 1].set_ylim(0, 1.05)
        h, l = axs[0, 0].get_legend_handles_labels()
        fig.legend(h, l, loc="outside right center", fontsize=7, title="log10 M200c [Msun]", title_fontsize=7)
        fig.suptitle("I5: the per-halo resolution rule (each halo enters only the shells it resolves) on the 10 "
                     "calibrated bins (maas run 3)", fontsize=9.5)
        fig.savefig(FIG / "fig_I5_perhalo_rule.png", dpi=150)
        plt.close(fig)
    # ================================================================== verification and numbers
    figs = sorted(p.name for p in FIG.glob("fig_I*.png"))
    ver["checks"]["figures"] = figs
    ver["checks"]["final_tables"] = sorted(p.name for p in TAB.glob("I*_final*.csv")) + \
        ["I4_model_table.csv", "I4_rcut_curves.csv", "I4_importance_regions.csv", "I4_residual_by_region.csv",
         "I2_parsimony_ladder_A2ii_vs_allgas.csv", "I2_key_models_likelihoods.csv", "I1_L_vs_a_parameters.csv",
         "I5_parameter_shifts_summary.csv", "I5_stack_differences.csv", "I4_s1_floor.csv",
         "I4_calibrated_signal_share.csv", "I4_unresolved_core.csv", "I4_rcut_nonmonotone.csv",
         "I4_Rfit134_extrapolated_to_Rcut4.csv", "I2_gamma_slopes.csv", "I2_mcut_summary.csv",
         "I4_minimal_adequate_profile_Rcut4.csv", "I4_compact_vs_perbin_Rf4Rc4.csv", "I4_unresolved_core_by_bin.csv",
         "I4_observable_truncation_radii.csv", "I5_perhalo_rule.csv", "I5_perhalo_rule_bins.csv"]
    ver["checks"]["S1_floor_family_rms_pct_estimate"] = floor_fam
    ver["checks"]["profile_rms_definition"] = ("I1_profile_rms_summary.csv = RMS of (model/stack - 1) over the fitted "
                                               "shells (models.fit_profile rms_percent), not of ln(model/stack)")
    ver["checks"]["fixes_after_independent_verification_1"] = [
        "profile RMS relabelled (ratio - 1, not ln)",
        "quantile radii: shell-edge and interpolated values; calibrated-bin share of the signed importance added",
        "profile-only error (det vs truth mean) and S1 scatter-model floor added; R_cut = 4 errors are not pure "
        "profile-shape errors",
        "free-gamma mechanism re-derived from the fitted parameters (I2_gamma_slopes.csv)",
        "fixed-M_cut dBIC ranges on mass and concentration cells (I2_mcut_summary.csv)",
        "non-monotone R_cut curves listed (I4_rcut_nonmonotone.csv)",
        "fig_I1: DM cells S1 where run, det marked 'd'; fig_I2: per-panel selected models; new fig_I1_profile_vs_S1",
        "no numerical result of the maas runs changed: no rerun needed"]
    ver["checks"]["fixes_after_independent_verification_2"] = [
        "compact Lee22 laws vs per-bin forms stated per field (I4_compact_vs_perbin_Rf4Rc4.csv): worse for tSZ "
        "(BPLall+O2 +1.5 pp, PL1/PL3/PL3g +5.2 to +9.1 pp vs B12 (a)); for electrons PL3/PL3g/BPLall+O0 beat B16 (a) "
        "and match or beat PB; only B16 (d) is better",
        "S1-floor statement: tSZ and kSZ near the floor, DM limited by the profile shape (profile-only 0.49% vs floor "
        "0.18%); exact S1 error of the target profile added (A2-ii stack painted with S1, maas run 3)",
        "unresolved core: inner edges of all ten bins and per-bin shares (I4_unresolved_core_by_bin.csv)",
        "per-halo resolution rule: run 3 refits it on its own resolved shells and paints its stack "
        "(I5_perhalo_rule*.csv, fig_I5_perhalo_rule.png); the strict-shell variant is flagged as the reference by "
        "construction; fig_I5 column labels rotated",
        "weight radii vs observable radii: observable truncation radii from Test H's trunc_all__a2ii__* curves "
        "(I4_observable_truncation_radii.csv)",
        "objective (d): its advantage on the observables is largely in-sample (weights = first-order responses)",
        "non-monotone R_cut count: 7 ladder curves + 38 I1 curves (the 8th I4 curve duplicates I1 B12 (d) R_fit 4)",
        "fig_I2: points offset horizontally per R_fit (coincident PB+O2 stars at k 34 now visible)",
        "maas run 3 (per-halo rule) added; runs 1-2 results unchanged"]
    ok = ver["checks"].get("supp_basis_rebuilt_parity_truthmean_max_rel")
    pr = ver["checks"]["supp_run1_paintings_reproduced"]["max_rel"]
    ver["gates"] = {
        "selection_rule_reapplied_matches": bool(ver["checks"].get("selection_rule_reapplied_matches")),
        "I1_objective_a_parameters_equal_TestG_A2ii (<1e-6)": (ver["checks"].get("I1_f35_objective_a_params_vs_G_max_rel")
                                                              or 1) < 1e-6,
        "I0_A2ii_stacks_equal_TestG (<1e-12)": max(ver["checks"]["I0_variant_a2ii_vs_G_stacks_max_rel"].values()) < 1e-12,
        "H_inputs_used_equal_local_H_results": all(ver["checks"].get("H_inputs_used_equal_local_H_results", {}).values()),
        "supp_basis_parity (<1e-8)": ok is not None and ok < 1e-8,
        "supp_reproduces_run1 (<1e-6)": pr < 1e-6,
        "cutshare_zero_at_Rcut4": bool(ver["checks"].get("cutshare_zero_at_Rcut4")),
        "scores_finite": ver["checks"].get("scores_finite_fraction") == 1.0,
        "perhalo_run3_reproduces_run1 (<1e-6)": ver["checks"].get("perhalo_run3_reproduces_run1_paintings", {})
        .get("rows", 0) > 0 and ver["checks"]["perhalo_run3_reproduces_run1_paintings"]["max_rel"] < 1e-6,
        "perhalo_strict_refits_equal_run1 (<1e-6)": phf is not None and
        phf["parity"]["B_strict_vs_run1_params_max_rel"] < 1e-6 and phf["parity"]["lee_strict_vs_run1_theta_max_abs"] < 1e-6,
        "stack_a2ii_det_equals_truthmean (<1e-8)": (ver["checks"].get("stack_a2ii_det_vs_truthmean_family_rms_max")
                                                    is not None) and
        ver["checks"]["stack_a2ii_det_vs_truthmean_family_rms_max"] < 1e-8,
    }
    ver["all_gates_passed"] = all(ver["gates"].values())
    ver["inputs_sha256"] = {n: sha(TAB / n) for n in ("I_scores.csv",)}
    if (SUPP / "tables" / "I_scores.csv").exists():
        ver["inputs_sha256"]["supp/tables/I_scores.csv"] = sha(SUPP / "tables" / "I_scores.csv")
    if (PHD / "tables" / "I_scores.csv").exists():
        ver["inputs_sha256"]["perhalo/tables/I_scores.csv"] = sha(PHD / "tables" / "I_scores.csv")
    ver_path.write_text(json.dumps(ver, indent=1, default=str) + "\n")
    (HERE / "report_numbers.json").write_text(json.dumps(nums, indent=1, default=str) + "\n")
    print(json.dumps(ver["gates"], indent=1), "all", ver["all_gates_passed"])


if __name__ == "__main__":
    main()
