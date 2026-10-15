"""H6: sensitivity of the observables to the B12 / B16 parameters fitted to the A2-ii stacks (fine grid).

Fits (h_fits): B12 (pressure: P0, x_c, beta) and B16 (electrons: rho0, alpha, beta) per calibrated bin, objective (a),
R_fit = 4 and 2 R200c, painted with a hard cut at R_cut = 4 R200c and S1 lognormal amplitude scatter (Test F / G:
sigma^2 = ln <X^2>/<X>^2 of the per-halo X = Y(<4R), N_e(<4R) of the bin, A2-ii; cross <Y N>/(<Y><N>)).
Quantities (bin c, parameter k, theta_k = ln p_k, fine shell s, observable O, k_O = 2 for quadratic, 1 for linear):
  parameter sensitivity   g_O[c, k] = d ln O / d theta_k = sum_s (dO/d ln F_cs) (d ln S_cs / d theta_k) / O, at the
                          fitted painting (exact in-basis Jacobians; DM quantiles: Mecke Jacobians of the painting)
  Fisher-type information I_fam[c] = (1/n_O) sum_{O in family} g_O g_O^T  [3 x 3]: a shift delta of ln theta changes
                          the family's RMS fractional error by sqrt(delta^T I delta); eigenvectors = parameter
                          combinations the family constrains (largest eigenvalue) or ignores (smallest)
  fit Fisher              F_fit[c] = J^T W J (J = d ln S / d theta on the fit shells, W = diag sigma_a^-2)
  fit influence           H[c, k, s] = d theta_k / d ln m_cs = [(J^T W J)^-1 J^T W]_ks  (m = the stacked mean)
  fit-transfer kernel     T_O[c, s] = sum_k (dO/d theta_k) H[c, k, s] / (k_O O): how a change of the stack in shell s
                          reaches O through the fit; compared with the direct kernel D_O[c, s] = (dO/d ln F_cs)/(k_O O)
                          at the truth (= the H3 importance of the calibrated bins).  For a stack perturbation
                          delta_s, dO/O = k_O sum_s T delta_s (fit route) vs k_O sum_s D delta_s (true response).
  bootstrap               200 (smoke 20) stratified halo resamples per bin (stackcore's seeds and strata: paired with
                          the stacking-choice study), the same resample for pressure and electrons; each replicate is
                          refitted and painted (S1 moments fixed at the full-sample values) -> parameter covariances and
                          the observable scatter (moment observables exact; DM quantiles first order through the
                          painting's Mecke Jacobians); also the scatter of painting the replicate stack itself.
Outputs: results/params.npz, results/params_summary.json, tables/H6_*.csv.
"""
from __future__ import annotations

import types

import h_common as HC
import h_fits as HF
import h_pdf as HP
import h_truth as HT
import numpy as np

FC, FB = HC.FC, HC.FB
import g_physics as PH  # noqa: E402  (Test G, read-only: scatter_tables)

R_FITS = (4.0, 2.0)
N_BOOT = 20 if HC.SMOKE else 200
STACK_SEED = 20261012                 # stackcore STUDY_SEED; bin seed = + 1000 x engine bin index
FAMILY_OBS = {
    "P": {"tSZ": [("yytot", e) for e in (100, 300, 1000, 3000)],
          "tSZxDM": [("ydmtot_P", e) for e in (100, 300, 1000, 3000)]},
    "E": {"kSZ": [("ksz1h", e) for e in (100, 300, 1000, 3000)],
          "DM": [("dm_mean", None), ("dm_var", None)] + [(f"dm_{q}", None) for q in HC.Q_NAMES],
          "tSZxDM": [("ydmtot_E", e) for e in (100, 300, 1000, 3000)]},
}
K_OBS = {"yytot": 2, "ydmtot_P": 1, "ydmtot_E": 1, "ksz1h": 2, "dm_mean": 1, "dm_var": 2}
TRUTH_KEY = {"yytot": "yytot", "ydmtot_P": "ydmtot", "ydmtot_E": "ydmtot", "ksz1h": "ksz1h", "dm_mean": "dm_mean",
             "dm_var": "dm_var"}
REGIONS = (("<1R", 0.0, 1.0), ("1-2R", 1.0, 2.0), ("2-4R", 2.0, 4.0))


def s1_moments(Pm, Em, s1):
    o = lambda a, b: np.einsum("cs,ct->cst", a, b)                                     # noqa: E731
    return dict(P=Pm, E=Em, SPP=np.exp(s1[:, 0])[:, None, None] * o(Pm, Pm),
                SEE=np.exp(s1[:, 1])[:, None, None] * o(Em, Em), SPE=s1[:, 2][:, None, None] * o(Pm, Em))


def obs_index(name, at):
    i = None if at is None else int(np.flatnonzero(FC.ELL == at)[0])
    return i


def fit_all(pr, field, r_fit, rows_by_bin=None, starts=None, means=None):
    """Objective-(a) fits of every calibrated bin (optionally on given mean / SD replicates)."""
    out = []
    for c in range(FC.N_CAL):
        if means is None:
            bd = HF.bin_data(pr, field, "a2ii", c)
        else:
            m, sd, base = means[c]
            bd = types.SimpleNamespace(mean=m, sd=sd, base=base, family=HF.FAMILY[field],
                                       mask=lambda r, base=base: base & FC.shells_within(FC.F_EDGES, r))
        out.append(HF.fit(bd, r_fit, start=None if starts is None else starts[c]))
    return out


def model_from_fits(field, fits):
    return np.array([HF.model_shells(field, f["parameters"], cut=4.0) for f in fits])


def dlnS(field, fits):
    D = np.array([HF.dlnS_dlntheta(field, f["parameters"]) for f in fits])          # [c, 75, 3]
    D[:, HC.N_X:] = 0.0                                                                # cut at 4 R200c
    return D


def boot_positions(pr, c, n_boot):
    rows = HT.bin_rows(pr, c)
    b = FC.CAL_BINS[c]
    return rows, FC.S.stratified_bootstrap_rows(pr["strata"][rows], n_boot, STACK_SEED + 1000 * b)


_BOOT: dict = {}


def _boot_bin(c):
    """All bootstrap refits of one calibrated bin (pressure and electrons share the resample); inputs in _BOOT
    (inherited by the forked workers)."""
    pr, fits0, n_boot = _BOOT["pr"], _BOOT["fits0"], _BOOT["n_boot"]
    rows, pos = boot_positions(pr, c, n_boot)
    w = pr["weight"][rows]
    vals = {f: np.nan_to_num(pr[f"{f}_a2ii"][rows], nan=0.0) for f in ("P", "E")}
    base0 = pr["base_mask"][FC.CAL_BINS[c]]
    res = {f"{f}|{r:g}": np.full((n_boot, 3), np.nan) for f in ("P", "E") for r in R_FITS}
    means = {f: np.zeros((n_boot, HC.N_X)) for f in ("P", "E")}
    for i in range(n_boot):
        p = pos[i]
        for f in ("P", "E"):
            m, sd, _ = FC.S.weighted_stack(vals[f][p], w[p])
            means[f][i] = m
            base = base0 & (m > 0)
            bd = types.SimpleNamespace(mean=m, sd=sd, base=base, family=HF.FAMILY[f],
                                       mask=lambda r, base=base: base & FC.shells_within(FC.F_EDGES, r))
            for r in R_FITS:
                fit = HF.fit(bd, r, start=fits0[f"{f}|{r:g}"][c]["parameters"])
                if fit is not None:
                    res[f"{f}|{r:g}"][i] = fit["parameters"]
    return c, res, means


def main():
    timer = HC.Timer()
    pr = HC.load_npz(FC.PRODUCTS / "profiles.npz")
    tr = HC.load_npz(HC.RESULTS / "truth.npz")
    ts = HC.read_json(HC.RESULTS / "truth_summary.json")
    fallback = np.load(HC.WORK / "pdf_fallback_h.npz")
    fallback = (fallback["hist"], float(fallback["overflow"]))
    s1, _ = PH.scatter_tables(pr, "a2ii", cuts=(4.0,))
    out = {"s1": s1, "x_edges": HC.X200}
    summary = {"definitions": __doc__, "fits": {}, "painting_errors": {}, "sensitivity": {}, "fisher": {},
               "transfer": {}, "bootstrap": {}}
    truth = dict(ts["observables"]["truth_a2ii"])
    for p, qn in enumerate(HC.Q_NAMES):
        truth[f"dm_{qn}"] = float(ts["pdf"]["truth_a2ii"]["quantiles"][p])
    # ---------------------------------------------------------------- fits
    fits = {}
    for field in ("P", "E"):
        for r in R_FITS:
            f = fit_all(pr, field, r)
            fits[f"{field}|{r:g}"] = f
            key = f"{field}_Rf{r:g}"
            out[f"params__{key}"] = np.array([x["parameters"] for x in f])
            out[f"model__{key}"] = model_from_fits(field, f)
            H = np.zeros((FC.N_CAL, 3, HC.N_X))
            neff = np.zeros((FC.N_CAL, 3))
            Ffit = np.zeros((FC.N_CAL, 3, 3))
            for c, x in enumerate(f):
                H[c], neff[c] = HF.influence_full(x)
                Jw = x["jacobian_lnS"] / x["sigma"][:, None]
                Ffit[c] = Jw.T @ Jw
            out[f"influence__{key}"] = H
            out[f"fisher_fit__{key}"] = Ffit
            summary["fits"][key] = {"parameters": out[f"params__{key}"], "names": HF.PARAMS[field],
                                    "chi2_per_dof": [x["chi2_per_dof"] for x in f],
                                    "rms_percent": [x["rms_percent"] for x in f],
                                    "accepted": [bool(x["accepted"]) for x in f],
                                    "n_shells": [int(x["n_shells"]) for x in f],
                                    "influence_n_eff": neff,
                                    "influence_by_region": {lab: H[:, :, (HC.X200[:-1] >= a - 1e-9)
                                                                     & (HC.X200[1:] <= b + 1e-9)].sum(axis=2)
                                                            for lab, a, b in REGIONS}}
    HC.log("fits", timer.mark("fits"))
    # ---------------------------------------------------------------- the fitted painting, its PDF and Mecke Jacobians
    paint = {}
    specs = []
    for r in R_FITS:
        Pm, Em = out[f"model__P_Rf{r:g}"], out[f"model__E_Rf{r:g}"]
        mom = s1_moments(Pm, Em, s1)
        paint[r] = mom
        specs.append((f"S1|{r:g}", HP.s1_entries(Em, s1[:, 1])))
        specs.append((f"det|{r:g}", HP.det_entries(Em)))
    res, _, _ = HP.run_specs(specs, "HA", fallback=fallback, keep_density=True, label="pdf params")
    items = [(f"S1|{r:g}", res[f"S1|{r:g}"], HP.s1_entries(paint[r]["E"], s1[:, 1]), True) for r in R_FITS]
    mk, _ = HP.run_mecke(items, "HA", fallback, label="mecke params")
    HC.log("pdf/mecke", timer.mark("pdf"))
    # ---------------------------------------------------------------- bootstrap refits (forked pool; no basis loaded yet)
    _BOOT.clear()
    _BOOT.update(pr=pr, fits0={k: [{"parameters": x["parameters"]} for x in v] for k, v in fits.items()}, n_boot=N_BOOT)
    boot = {k: np.full((N_BOOT, FC.N_CAL, 3), np.nan) for k in fits}
    bmeans = {f: np.zeros((N_BOOT, FC.N_CAL, HC.N_X)) for f in ("P", "E")}
    HC.mem("params before the bootstrap fork")
    for c, res_c, means_c in FC.pool_map(_boot_bin, list(range(FC.N_CAL)), ordered=False):
        for k in boot:
            boot[k][:, c] = res_c[k]
        for f in ("P", "E"):
            bmeans[f][:, c] = means_c[f]
    HC.log("bootstrap fits", timer.mark("boot"))
    # ---------------------------------------------------------------- the A2-ii basis and the H3 direct kernels (after every fork)
    HC.mem("params after forks")
    B = FB.Basis("HA")
    sens = HC.load_npz(HC.RESULTS / "sens.npz", keys=lambda k: k.startswith("imp__a2ii__"))
    for r in R_FITS:
        mom = paint[r]
        o = B.scalars(B.observables(**mom))
        for p, qn in enumerate(HC.Q_NAMES):
            o[f"dm_{qn}"] = float(res[f"S1|{r:g}"]["quantiles"][p])
        odet = B.scalars(B.observables(mom["P"], mom["E"]))
        for p, qn in enumerate(HC.Q_NAMES):
            odet[f"dm_{qn}"] = float(res[f"det|{r:g}"]["quantiles"][p])
        summary["painting_errors"][f"Rf{r:g}_Rc4"] = {
            "S1": {k: o[k] / truth[k] - 1 for k in o if k in truth and truth[k]},
            "det": {k: odet[k] / truth[k] - 1 for k in odet if k in truth and truth[k]}}
        J = B.jacobians(**mom)
        Jq, Jqfb = mk[f"S1|{r:g}"]
        Q = np.asarray(res[f"S1|{r:g}"]["quantiles"], float)
        Obs = B.observables(**mom)
        D = {"P": dlnS("P", fits[f"P|{r:g}"]), "E": dlnS("E", fits[f"E|{r:g}"])}
        H = {"P": out[f"influence__P_Rf{r:g}"], "E": out[f"influence__E_Rf{r:g}"]}
        for field, fams in FAMILY_OBS.items():
            G_all = {}
            for fam, lst in fams.items():
                G = np.zeros((len(lst), FC.N_CAL, 3))
                T = np.zeros((len(lst), FC.N_CAL, HC.N_X))
                Dk = np.zeros((len(lst), FC.N_CAL, HC.N_X))
                for j, (name, at) in enumerate(lst):
                    if name.startswith("dm_q"):
                        p = HC.Q_NAMES.index(name[3:])
                        dOdlnF = Jq[p][:, :HC.N_FINE]
                        Ov, kO = Q[p], 1
                        direct = sens[f"imp__a2ii__{name}"][FC.CAL_BINS]          # [10, 67]
                    else:
                        i = obs_index(name, at)
                        dOdlnF = J[name] if i is None else J[name][..., i]
                        tkey = TRUTH_KEY[name]
                        Ov = Obs[tkey] if i is None else Obs[tkey][i]
                        kO = K_OBS[name]
                        imp = sens[f"imp__a2ii__{name}"]
                        direct = (imp[FC.CAL_BINS] if i is None else imp[FC.CAL_BINS][..., i])
                    g = np.einsum("cs,csk->ck", dOdlnF[:, :HC.N_FINE], D[field]) / Ov
                    G[j] = g
                    T[j] = np.einsum("ck,cks->cs", g, H[field]) / kO
                    Dk[j] = direct
                    lab = f"{name}@{at}" if at is not None else name
                    out[f"g__Rf{r:g}__{field}__{lab}"] = g
                    out[f"transfer__Rf{r:g}__{field}__{lab}"] = T[j]
                    out[f"direct__{field}__{lab}"] = Dk[j]
                    reg = {}
                    for rl, a, b in REGIONS:
                        m = (HC.X200[:-1] >= a - 1e-9) & (HC.X200[1:] <= b + 1e-9)
                        reg[rl] = {"transfer": float(T[j][:, m].sum()), "direct": float(Dk[j][:, m].sum())}
                    reg["total"] = {"transfer": float(T[j].sum()), "direct": float(Dk[j].sum())}
                    summary["transfer"][f"Rf{r:g}|{field}|{lab}"] = reg
                    summary["sensitivity"][f"Rf{r:g}|{field}|{lab}"] = {
                        HC.CAL_LABELS[c]: dict(zip(HF.PARAMS[field], g[c].tolist())) for c in range(FC.N_CAL)}
                G_all[fam] = G
                # Fisher-type information per bin
                I = np.einsum("jck,jcl->ckl", G, G) / len(lst)
                ev, evec = np.linalg.eigh(I)
                out[f"fisher_obs__Rf{r:g}__{field}__{fam}"] = I
                Ff = out[f"fisher_fit__{field}_Rf{r:g}"]
                rows = {}
                for c in range(FC.N_CAL):
                    evf, vecf = np.linalg.eigh(Ff[c])
                    v1 = evec[c][:, -1]
                    rows[HC.CAL_LABELS[c]] = {
                        "eigenvalues": ev[c][::-1].tolist(),
                        "tolerance_lnparam_1pct_along_eigvecs": (0.01 / np.sqrt(np.maximum(ev[c][::-1], 1e-300))).tolist(),
                        "most_constrained_combination": dict(zip(HF.PARAMS[field], (v1 * np.sign(v1[np.argmax(np.abs(v1))])).tolist())),
                        "least_constrained_combination": dict(zip(HF.PARAMS[field], evec[c][:, 0].tolist())),
                        "fit_best_combination": dict(zip(HF.PARAMS[field], vecf[:, -1].tolist())),
                        "fit_worst_combination": dict(zip(HF.PARAMS[field], vecf[:, 0].tolist())),
                        "alignment_obs_best_vs_fit_eigvecs_desc": np.abs(vecf[:, ::-1].T @ v1).tolist()}
                summary["fisher"][f"Rf{r:g}|{field}|{fam}"] = rows
        HC.log(f"sensitivity Rf{r:g}", timer.mark(f"sens{r}"))
    # ---------------------------------------------------------------- finite-difference check of the transfer kernel
    fd = {}
    for r in R_FITS:
        for field, region in (("E", (2.0, 4.0)), ("P", (0.1051, 0.5185))):
            delta = 0.02
            m = (HC.X200[:-1] >= region[0] - 1e-9) & (HC.X200[1:] <= region[1] + 1e-9)
            means = []
            for c in range(FC.N_CAL):
                bd = HF.bin_data(pr, field, "a2ii", c)
                mm = bd.mean * np.where(m, 1 + delta, 1.0)
                means.append((mm, bd.sd * np.where(m, 1 + delta, 1.0), bd.base))
            f2 = fit_all(pr, field, r, means=means, starts=[x["parameters"] for x in fits[f"{field}|{r:g}"]])
            M2 = model_from_fits(field, f2)
            mom = dict(paint[r])
            if field == "P":
                mom = s1_moments(M2, paint[r]["E"], s1)
                name, i, kO = "yytot", obs_index("yytot", 1000), 2
                T = out[f"transfer__Rf{r:g}__P__yytot@1000"]
            else:
                mom = s1_moments(paint[r]["P"], M2, s1)
                name, i, kO = "ksz1h", obs_index("ksz1h", 1000), 2
                T = out[f"transfer__Rf{r:g}__E__ksz1h@1000"]
            o0 = B.observables(**paint[r])[name][i]
            o1 = B.observables(**mom)[name][i]
            pred = kO * float(T[:, m].sum()) * delta
            fd[f"Rf{r:g}|{field}|{name}@1000|+{delta:g} in {region[0]:g}-{region[1]:g}R"] = {
                "refit_dlnO": float(np.log(o1 / o0)), "transfer_first_order": pred}
    summary["transfer_finite_difference_check"] = fd
    HC.log("fd check", timer.mark("fd"))
    # ---------------------------------------------------------------- bootstrap: parameter covariances, observable scatter
    for k, v in boot.items():
        f, r = k.split("|")
        out[f"boot_params__{f}_Rf{r}"] = v
        lv = np.log(v)
        cov = np.array([np.cov(lv[:, c][np.isfinite(lv[:, c]).all(axis=1)].T) for c in range(FC.N_CAL)])
        out[f"boot_cov_lnparams__{f}_Rf{r}"] = cov
        summary["bootstrap"][f"{f}_Rf{r}"] = {
            "n_ok": [int(np.isfinite(lv[:, c]).all(axis=1).sum()) for c in range(FC.N_CAL)],
            "sigma_lnparam": np.sqrt(np.einsum("ckk->ck", cov)).tolist(),
            "corr": [(cv / np.sqrt(np.outer(np.diag(cv), np.diag(cv)))).tolist() for cv in cov]}
    # observable scatter per replicate (S1 moments fixed); DM quantiles first order (Mecke at the painting)
    sel_keys = [f"yytot_{e}" for e in (100, 300, 1000, 3000)] + [f"ksz1h_{e}" for e in (100, 300, 1000, 3000)] + \
               ["dm_mean", "dm_var"] + [f"ydmtot_{e}" for e in (100, 300, 1000, 3000)]
    for r in R_FITS:
        base = B.scalars(B.observables(**paint[r]))
        Jq, _ = mk[f"S1|{r:g}"]
        Q = np.asarray(res[f"S1|{r:g}"]["quantiles"], float)
        rel = {k: [] for k in sel_keys + [f"dm_{q}" for q in HC.Q_NAMES]}
        for i in range(N_BOOT):
            pP, pE = boot[f"P|{r:g}"][i], boot[f"E|{r:g}"][i]
            if not (np.isfinite(pP).all() and np.isfinite(pE).all()):
                continue
            Pm = np.array([HF.model_shells("P", p, cut=4.0) for p in pP])
            Em = np.array([HF.model_shells("E", p, cut=4.0) for p in pE])
            o = B.scalars(B.observables(**s1_moments(Pm, Em, s1)))
            for k in sel_keys:
                rel[k].append(o[k] / base[k] - 1)
            with np.errstate(divide="ignore", invalid="ignore"):
                dl = np.where(paint[r]["E"] > 0, np.log(Em / np.where(paint[r]["E"] > 0, paint[r]["E"], 1)), 0.0)
            for p, qn in enumerate(HC.Q_NAMES):
                rel[f"dm_{qn}"].append(float(np.sum(Jq[p][:, :HC.N_FINE] * dl) / Q[p]))
        out[f"boot_obs_rel__Rf{r:g}"] = np.array([rel[k] for k in rel])
        summary["bootstrap"][f"observables_Rf{r:g}"] = {k: {"sd": float(np.std(v, ddof=1)), "mean": float(np.mean(v))}
                                                         for k, v in rel.items() if len(v) > 1}
        summary["bootstrap"][f"observable_keys"] = list(rel)
    # painting the replicate stack itself (no fit), S1 moments fixed
    base = B.scalars(B.observables(**s1_moments(HT.cal_moments(pr, "a2ii")["P"], HT.cal_moments(pr, "a2ii")["E"], s1)))
    rel = {k: [] for k in sel_keys}
    for i in range(N_BOOT):
        o = B.scalars(B.observables(**s1_moments(bmeans["P"][i], bmeans["E"][i], s1)))
        for k in sel_keys:
            rel[k].append(o[k] / base[k] - 1)
    summary["bootstrap"]["observables_stack_no_fit"] = {k: {"sd": float(np.std(v, ddof=1))} for k, v in rel.items()}
    HC.log("bootstrap observables", timer.mark("boot_obs"))
    np.savez_compressed(HC.RESULTS / "params.npz", **out)
    summary["seconds"] = timer.mark("total")
    HC.write_json(HC.RESULTS / "params_summary.json", summary)
    rows = []
    for key, reg in summary["transfer"].items():
        r, field, lab = key.split("|")
        rows.append(dict(R_fit=r, field=field, observable=lab,
                         **{f"{k}_{t}": v[t] for k, v in reg.items() for t in ("transfer", "direct")}))
    HC.write_csv(HC.TABLES / "H6_transfer.csv", rows)
    rows = []
    for key, d in summary["sensitivity"].items():
        r, field, lab = key.split("|")
        for b, g in d.items():
            rows.append(dict(R_fit=r, field=field, observable=lab, mass_bin=b, **{f"dlnO_dln_{k}": v for k, v in g.items()}))
    HC.write_csv(HC.TABLES / "H6_parameter_sensitivity.csv", rows)
    HC.log("params done", summary["seconds"])
    return summary


if __name__ == "__main__":
    main()
