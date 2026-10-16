"""Report text of Test J2 ('Halo component of DM and low-mass calibration: fits') generated from the fetched and
post-processed results (no hand transcription).

Every number is produced by Test J1's registry class (j_report.Reg): it reads one field of a results file, applies one
declared operation and formats the value; the registry is written to post_j2/report_numbers_j2.json and verify_j2.py
gate G11 re-resolves every entry and checks that its text appears in the report.  Ranges, extremes and rankings are
computed from the data in this module (min / max / sorting), never typed in.
Outputs: post_j2/J2_report.txt (Markdown with LaTeX), post_j2/report_numbers_j2.json, post_j2/key_numbers_j2.json,
post_j2/recommendations_j2.json.
Usage: python -B j2_report.py   (run j2_lsub.py first; run verify_j2.py after it, then this module once more so that
the G11 row quotes the gate's result for the final text)
"""
from __future__ import annotations

import json

import j2_common as C
from j2_report_text import DEFINITIONS
from j_report import Reg

HERE = C.HERE
SUM, SC, LM = "post_j2/J2_summary.json", "post_j2/J2_scores.csv", "post_j2/J2_lowmass.csv"
FP, LP, EP = "results_j2/fits_parity.json", "results_j2/ladder_parity.json", "results_j2/eval_parity.json"
MAN, DONE, LS = "results_j2/run_manifest.json", "results_j2/test_J_failed.json", "results_j2/ladder_summary.json"
FITS, META, VER = "results_j2/fits.json", "inputs/j2_fitdata_meta.json", "verification_j2.json"
J1TS, J1P, CR = "results/truth_summary.json", "post/j_post.json", "CLUSTER_RUN.json"
LSJ = "post_j2/J2_L_substitutions.json"
DEFS = ("D0", "D1", "D2", "D3")
DNAME = {"D0": "D0 (A2-ii gas to 4 R200c)", "D1": "D1 (A2-ii gas in the sphere r <= R200c)",
         "D2": "D2 (A2-ii gas in r <= 2 R200c)", "D3": "D3 (host-bound gas to 4 R200c)"}
STAT_LAB = {"dm_mean": "mean", "dm_var": "variance", "q16": "Q16", "q50": "Q50", "q84": "Q84", "q975": "Q97.5",
            "q999": "Q99.9"}
LOWB = ("12.7-12.9", "12.9-13.1", "13.1-13.3", "13.3-13.5")
KSZ_RECIPE = "lm|C14|B12a+B16dIT"   # Test I's published fits in 13.5-15.5 + J2's fits in 12.7-13.5 (run 2)
TESTI_PUB = "lm|C10|B12a+B16dIT"    # Test I's painting exactly as published (fallback below 10^13.5)
DI_REFIT = "lm|C14|B12a+B16dI"      # Test I's weight formula on J1's C14 maps, refitted in all 14 bins
DONE1, CR1 = "results_j2_run1/test_J_done.json", "results_j2_run1/run_manifest.json"
DONE2 = "results_j2_run2/test_J_done.json"
DIV = "post_j2/J2_summary.json"
V10 = "verify_independent_Test_J2/out_r2/v2_s10_dI_effect_estimate.json"


def pretty(label):
    t = label.split("|")
    if t[0] in ("a", "rc"):
        obj = {"L": "L*"}.get(t[3], t[3])
        return f"B16 per bin ({obj}), R_fit = {t[4][2:]}" + (", RC fits in 12.7-13.5" if t[0] == "rc" else "")
    if t[0] == "l":
        return f"{t[2]} ({'mass' if t[3] == 'mass' else 'concentration'} cells, R_fit = {t[4][2:]})"
    if t[0] == "ref":
        return "the stack itself"
    return label


SCJ, LMJ = "post_j2/J2_scores.json", "post_j2/J2_lowmass.json"
SS1 = "results/sens_summary.json"
ILS = "../I_fitting_a2ii/results/ladder_selection_summary.json"
J1PREP = "results/prepare.json"
IRN = "../I_fitting_a2ii/report_numbers.json"
RP = "post_j2/J2_recommended_parameters.json"
STK = "results/stacks.npz"
MUPD = "post/mecke_update.json"


class Reg2(Reg):
    def absmax(self, srcs, fmt, scale=1.0):
        vals = [abs(self.get(f, k)) for f, k in srcs]
        return self._rec("absmax", srcs, fmt, scale, 0.0, max(vals) * scale)


def rc_other_txt(R):
    """'< 0.1 pp' when the largest RC change of the other D0-D2 paintings is below 0.1 percentage points."""
    if R.get(SUM, ["rc_max_abs_change_rms_DM_D0_D2_except_D0a"]) < 0.001:
        return "< 0.1 pp"
    return f"<= {R.v(SUM, ['rc_max_abs_change_rms_DM_D0_D2_except_D0a'], '.2f', 100)} pp"


def join_and(items):
    items = list(items)
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def build():
    R = Reg2()
    S = R.load(SUM)
    SCD = R.load(SCJ)
    L = []
    w = L.append

    def e(lab, field, fmt=".2f"):
        return R.v(SCJ, [lab, field], fmt, 100)

    def pair(lab):
        return f"{e(lab, 'rms_DM')} ({e(lab, 'prof_DM')})"


    def rec(D, kind):
        return S["recommendations"][D][kind]["label"]

    def worst_stat(lab, D):
        return R.absmax([(SCJ, [lab, f"e_{k}"]) for k in C.dm_family(D)], ".2f", 100)

    def which_worst(lab, D):
        return STAT_LAB[max(C.dm_family(D), key=lambda kk: abs(SCD[lab][f"e_{kk}"]))]

    def mcut(D, key):
        return R.v(SUM, ["ladder", "log10Mcut_free", D, key], ".2f")

    def km(D, key, field, fmt=".2f"):
        return R.v(SUM, ["ladder", "key_models", D, key, field], fmt)

    def lm(obj, pred, obs, field, fmt=".1f"):
        return R.v(LMJ, [f"{obj}|{pred}|{obs}", field], fmt, 100)

    def xh_seq():
        return " / ".join(R.v(STK, ["x_h", b], ".2f") for b in (1, 2, 3, 4))

    def ksz_factor(term, which):
        """D0/D3 kSZ D_ell factor at the multipole where it is smallest ('min') or largest ('max')."""
        k = J1["kSZ_Dl_uK2_C14"]
        ells = (100, 300, 1000, 3000)
        fac = {x: k["D0"][f"{term}_{x}"] / k["D3"][f"{term}_{x}"] for x in ells}
        x = (min if which == "min" else max)(fac, key=fac.get)
        txt = R.r(J1P, ["kSZ_Dl_uK2_C14", "D0", f"{term}_{x}"], J1P, ["kSZ_Dl_uK2_C14", "D3", f"{term}_{x}"], ".1f")
        return f"{txt} (ell = {x})"

    def ksz_ratio_rng(term):
        return R.rng([(J1P, ["kSZ_D3_over_D0", "C14", f"{term}_{x}"]) for x in (100, 300, 1000, 3000)], ".3f")

    J1 = R.load(J1P)
    S2 = R.load(CR)["j2_maas_runs"][1]

    def worst_of(lab):
        return max(SCD[lab]["rms_DM"], SCD[lab]["prof_DM"])

    def best_o0_compact(D):
        c = [lab for lab, r in SCD.items() if r["D"] == D and r["block"] == "J2b" and r.get("dset") == "mass"
             and lab.split("|")[2].endswith("+O0") and int(r["k"]) <= 14 and r.get("rms_DM") is not None]
        return min(c, key=worst_of)
    p = {D: rec(D, "primary") for D in DEFS}
    cp = {D: rec(D, "compact") for D in DEFS}
    # ================================================================== header
    w("## Halo component of DM and low-mass calibration: fits (Test J2)")
    w("")
    w("*FLAMINGO L1_m9, z = 0 (snapshot 0077). The four halo-electron definitions come from Test J1:*")
    w("- *D0: A2-ii gas to 4 R200c;*")
    w("- *D1: A2-ii gas in the 3D sphere r <= R200c;*")
    w("- *D2: the same in r <= 2 R200c;*")
    w("- *D3: gas bound (HBT) to the halo or its own satellites, to 4 R200c.*")
    w("")
    w("*They are calibrated in all 14 painted mass bins 12.7-15.5 (configuration C14; bins in log10(M200c/Msun)). "
      "Every fitted form is painted with J1's fine basis and spherical-chord projection and compared with J1's truth of "
      "the same definition. Every number is generated by `j2_report.py` from the file named in brackets. "
      "`post_j2/report_numbers_j2.json` lists each number with its source file and field, and gate G11 of "
      "`verify_j2.py` re-resolves all of them. Paths are relative to `J_halo_dm_lowmass/`. Errors are in per cent and "
      "written 'S1 (profile-only)': the first is the S1 painting against the scatter-inclusive truth, the second (in "
      "brackets) is the det painting against the truth mean (J2.2). Estimates are labelled ESTIMATE.*")
    w("")
    # ================================================================== J2.0
    w("### J2.0 Short answer")
    w("")
    w("Recommended painting per definition: the fitted painting with the smallest max(S1 error, profile-only error) "
      "of the DM family (rule in J2.2) [post_j2/J2_summary.json: recommendations; post_j2/J2_scores.json]:")
    w("")
    w("| definition | recommended painting (k parameters) | R_fit [R200c] | R_cut [R200c] | DM family error [%] | "
      "largest single DM statistic error, S1 [%] | S1 floor [%] | compact law (k): DM family error [%] |")
    w("|---|---|---|---|---|---|---|---|")
    for D in DEFS:
        pk, ck = R.v(SCJ, [p[D], "k"], ".0f"), R.v(SCJ, [cp[D], "k"], ".0f")
        rf, rcut = R.v(SCJ, [p[D], "R_fit"], "g"), R.v(SCJ, [p[D], "R_cut"], "g")
        comp = "the same" if cp[D] == p[D] else f"{cp[D].split('|')[2]} ({ck}): {pair(cp[D])}"
        name = pretty(p[D]).split(", R_fit")[0].split(" cells")[0].replace(" (mass", ", mass cells")
        w(f"| {DNAME[D]} | {name} ({pk}) | {rf} | {rcut} | {pair(p[D])} | {worst_stat(p[D], D)} "
          f"({which_worst(p[D], D)}) | {e(f'ref|{D}|stack', 'rms_DM')} | {comp} |")
    w("")
    w("- 'B16 per bin (L*)' uses Lee22's covariance-of-the-mean likelihood (objective L). In bins where (L) has no fit "
      "the objective-(a) fit is painted instead (J2.3).")
    w("- 'compact law': the candidate ladder law with k <= 14 that minimises the same max(S1, profile-only) DM error; "
      "for D0 it carries the outer option O2 (see below, and J2.5).")
    w("- The largest single statistic is given as |e_O|; its sign is in J2.3.")
    w("- The S1 floor is the error of the definition's own stack painted with the S1 scatter model, i.e. what the "
      "scatter model alone causes when the profile is exact. It is not a lower bound: a fitted profile's errors can "
      "partly cancel the scatter model's, which is why the rule also uses the profile-only error.")
    w("")
    w("**Which definition for which purpose:**")
    w("- *Comparisons with TNG / Konietzka's within-R200 catalogue and HalfDome's SphericalChordDMProfile: use D1.*")
    w(f"  - Paint {pretty(p['D1'])}, cut in 3D at 1 R200c: {pair(p['D1'])}%.")
    w(f"  - Its S1 error lies below D1's S1 floor ({e('ref|D1|stack', 'rms_DM')}%) through partial cancellation, "
      f"and its profile-only error is only {e(p['D1'], 'prof_DM')}%. D1's S1 error is therefore set by the lognormal "
      f"amplitude-scatter model, not by the profile. Example, the DM variance: floor {e('ref|D1|stack', 'e_dm_var')}%, "
      f"profile-only {e(p['D1'], 'p_dm_var')}%, S1 {e(p['D1'], 'e_dm_var')}%; Q99.9: floor "
      f"{e('ref|D1|stack', 'e_q999')}%, profile-only {e(p['D1'], 'p_q999')}%, S1 {e(p['D1'], 'e_q999')}%.")
    w(f"  - Below 10^13.5 the gas inside the hydrodynamic-resolution radius x_h is resolved only as an integral (J1): "
      f"x_h = {xh_seq()} R200c in 12.7-12.9 / 12.9-13.1 / 13.1-13.3 / 13.3-13.5. In 12.7-12.9 the D1 sphere is "
      f"therefore constrained only through its gas content; in 12.9-13.5 the shape between x_h and R200c is "
      f"resolved. J1 could not test the simulated gas content of these haloes for resolution convergence.")
    w("- *Total halo-associated gas (the painting partition to 4 R200c; HalfDome's tSZ/kSZ target): use D0.*")
    w(f"  - Paint {pretty(p['D0'])}: {pair(p['D0'])}%.")
    nd = rec('D0', 'primary_not_d')
    w(f"  - Objective (d) is tuned in-sample, as in Test I. The best painting without an importance-weighted "
      f"objective ((d), dI or dIT), {pretty(nd)}, gives {pair(nd)}%.")
    if not nd.split("|")[2].endswith("+O0"):
        o0 = best_o0_compact("D0")
        w(f"  - That law carries the truncated two-halo outer option ({nd.split('|')[2].split('+')[1]}). Its DM "
          f"accuracy comes from residuals of opposite sign at 2-3 and 3-4 R200c (100 ln(model/stack) = "
          f"{e(nd, 'dmean_DM_2-3', '+.1f')} and {e(nd, 'dmean_DM_3-4', '+.1f')}, DM-importance-weighted) that cancel in the one-point DM "
          f"statistics. It paints the DM-field C_ell to {e(nd, 'rms_DMcl')}% and the kSZ 1-halo term to "
          f"{e(nd, 'rms_kSZ')}%. The best law without an outer option and with k <= 14, {pretty(o0)}, gives "
          f"{pair(o0)}% in DM, {e(o0, 'rms_DMcl')}% in the DM-field C_ell and {e(o0, 'rms_kSZ')}% in kSZ; "
          f"the per-bin PB+O0 gives {pair('l|D0|PB+O0|mass|Rf4')}%.")
    w(f"  - {R.v(SS1, ['bands', 'D0__dm_mean@scalar', '2-4R'], '.1f', 100)}% of D0's mean DM comes from 2-4 R200c, "
      f"mostly unbound gas (J1).")
    w(f"- *Gas within 2 R200c (intermediate): use D2*, {pretty(p['D2'])}: {pair(p['D2'])}%.")
    d3all = [lab for lab, r in SCD.items() if r["D"] == "D3" and r["block"] in ("J2a", "J2b")
             and r.get("dset") != "massc"]
    w(f"- *Physically bound gas: D3.* None of the allowed forms follows it to better than "
      f"{R.agg('min', [(SCJ, [lab, 'rms_DM']) for lab in d3all], '.1f', 100)}% (S1).")
    w(f"  - The best of all {len(d3all)} candidate D3 paintings (every B16 per-bin painting and every mass-cell "
      f"ladder law, J2.2) is {pretty(p['D3'])}, {R.v(SCJ, [p['D3'], 'k'], '.0f')} parameters: {pair(p['D3'])}%. The "
      f"best per-bin B16 painting ({pretty(rec('D3', 'best_B16')).replace('B16 per bin ', 'objective ')}) gives "
      f"{pair(rec('D3', 'best_B16'))}%.")
    w("  - The bound-gas profile drops steeply between 1 and 2 R200c. B16 cannot follow that (J2.3, "
      "`figures_j2/J2_residual_profiles`); the gNFW laws approximate it only by running towards the limit of an "
      "exponential truncation (J2.5).")
    w(f"  - For a better D3, either paint the measured per-bin D3 stacks themselves (S1 error "
      f"{e('ref|D3|stack', 'rms_DM')}%), or add a truncation term without the two-halo term (a new form, not tested "
      f"here).")
    w("")
    w("**Low-mass calibration** (D0 target, Test I's recipe; J2.4) [post_j2/J2_scores.json: lm|...]. Family errors "
      "S1 against J1's 14-bin truth:")
    w("")
    w("| painting | DM family [%] | DM-field C_ell [%] | kSZ 1-halo [%] | kSZ 2-halo [%] | tSZ [%] |")
    w("|---|---|---|---|---|---|")
    for lab, name in ((TESTI_PUB, "Test I as published: its fits in 13.5-15.5, published fallback below 10^13.5 (C10)"),
                      (KSZ_RECIPE, "**recommended:** Test I's fits unchanged in 13.5-15.5 + new fits in 12.7-13.5 "
                                   "(C14, 'dIT')"),
                      (DI_REFIT, "alternative: refit of all 14 bins with Test I's weight formula on J1's C14 maps "
                                 "(C14, 'dI')")):
        w(f"| {name} | {e(lab, 'rms_DM')} | {e(lab, 'rms_DMcl')} | {e(lab, 'rms_kSZ')} | {e(lab, 'rms_kSZ2h')} | "
          f"{e(lab, 'rms_tSZ')} |")
    w("")
    w("**Lee22 ladder on 14 bins** (J2.5):")
    w(f"- With 12.7-13.5 included, one power law in mass no longer fits. For D0, PL3g+O0 (Test I's selection on 10 "
      f"bins, floor chi^2/nu {R.v(ILS, ['chosen', 'E|mass|4', 'chi2nu_floor'], '.2f')}) now has "
      f"{km('D0', 'PL3g+O0|Rf4', 'chi2nu_floor')}.")
    sel_txt = []
    for D in DEFS:
        for rf in C.DEFS[D]["lad_rfit"]:
            ch = S["per_definition"][D]["ladder_chosen"][f"E|mass|{rf:g}"]
            sel_txt.append(f"{D} (R_fit {rf:g}): {ch['model']} (k = {R.v(LS, ['chosen', D, f'E|mass|{rf:g}', 'k'], '.0f')})")
    w("- G's rule selects (mass cells): " + "; ".join(sel_txt) + f". The D0 break lies at log10 M = "
      f"{mcut('D0', 'BPLall+O0|mass|Rf4')}.")
    w(f"- Every one of the {R.agg('sum', [(SUM, ['ladder', 'completeness', k, 'n_painted']) for k in S['ladder']['completeness']], '.0f')} "
      f"fitted mass-cell laws is now painted and evaluated (gate G13), so the minimal adequate models of J2.6 are "
      f"established over the complete ladder.")
    w("")
    w("**kSZ** (J2.7):")
    w(f"- The recipe stays as it is: A2-ii gas to 4 R200c, Test I's published fits unchanged in 13.5-15.5, and the "
      f"same recipe (B12 (a) pressure, B16 (d) electrons with Test I's weight formula, R_fit = R_cut = 4 R200c) newly "
      f"fitted in 12.7-13.5 (recommendation 1). Against the 14-bin truth it reproduces the kSZ 1-halo family to "
      f"{e(KSZ_RECIPE, 'rms_kSZ')}% and the 2-halo family to {e(KSZ_RECIPE, 'rms_kSZ2h')}% (S1).")
    w(f"- Bound-only gas carries {ksz_ratio_rng('ksz1h')} of the 1-halo kSZ and {ksz_ratio_rng('ksz2h')} of the "
      f"2-halo kSZ at ell = 100-3000 (J1, computed from the bound-only stacks). A bound-only fit would describe that "
      f"different signal, and the allowed forms reproduce its 1-halo term only to "
      f"{R.v(SUM, ['kSZ_D3_best', 'rms_kSZ'], '.1f', 100)}% at best.")
    w("")
    # ================================================================== J2.1
    w("### J2.1 What was run, where, and checks")
    w("")
    S3 = R.load(CR)["j2_maas_runs"][2]
    w("**Three maas runs** [CLUSTER_RUN.json: j2_maas_runs; results_j2/run_manifest.json: invocations]. All through "
      "`cluster_tests.py run J --workers 8 --rss-stop 24 --driver-args=\"--j2 ...\"` with 8 workers, nice 19, "
      "ionice best-effort 7; run_test_j.py hands '--j2' to run_test_j2.py.")
    w(f"- **Run 1** (launched 2026-10-15 19:28:47 EEST; stages j1base, fits, ladder, eval): exit code 0, wall "
      f"{R.v(DONE1, ['wall_seconds'], ',.0f')} s, peak tree RSS {R.v(DONE1, ['peak_tree_rss_GiB'], '.2f')} GiB "
      f"[results_j2_run1/test_J_done.json]. It painted {R.n('results_j2_run1/evaluate_items.json', ['items'])} "
      f"items, of the ladder only the selected and best-BIC laws.")
    w(f"- **Run 2** (launched {S2['launched']}; stages j1base and eval only, re-using run 1's fits and ladder "
      f"unchanged): exit code {R.v(DONE2, ['returncode'], '.0f')}, wall {R.v(DONE2, ['wall_seconds'], ',.0f')} s, peak "
      f"tree RSS {R.v(DONE2, ['peak_tree_rss_GiB'], '.2f')} GiB, minimum machine memory available "
      f"{R.v(DONE2, ['min_machine_available_GiB'], '.1f')} GiB [{DONE2}].")
    w(f"  - Why: the review of run 1 found that (i) only 7-8 of the 13-39 mass-cell ladder laws per definition and "
      f"R_fit had been painted, so 'fewest k' claims were not established, and (ii) the painting called 'Test I's "
      f"recipe' had re-fitted 13.5-15.5 with weights from J1's C14 maps instead of using Test I's fits.")
    w(f"  - Run 2 paints all {R.n('results_j2_run2/evaluate_items.json', ['items'])} items: run 1's items, every other "
      f"mass-cell ladder law, and the 'dIT' paintings (Test I's own fits in 13.5-15.5 + the new fits in 12.7-13.5). "
      f"The new items are deposited in their own DM-PDF groups after run 1's, and every run-1 painting is reproduced "
      f"exactly (gate G15: largest relative difference "
      f"{R.v(VER, ['gates', 'G15_run1_reproduced', 'max_rel_difference'], '.1f')} over "
      f"{R.v(VER, ['gates', 'G15_run1_reproduced', 'n_values_compared'], ',.0f')} values).")
    st2_ = ["invocations", 2, "stages"]
    w(f"- **Run 3** (launched {S3['launched']}; stages j1base, fits and eval, re-using run 1's ladder unchanged, then "
      f"J1's stage mecke_checks): the J2 stages completed (return codes "
      f"{R.v(MAN, st2_ + ['j1base', 'returncode'], '.0f')} / {R.v(MAN, st2_ + ['fits', 'returncode'], '.0f')} / "
      f"{R.v(MAN, st2_ + ['eval', 'returncode'], '.0f')}; eval parity passed); wall "
      f"{R.v(DONE, ['wall_seconds'], ',.0f')} s, peak tree RSS {R.v(DONE, ['peak_tree_rss_GiB'], '.2f')} GiB, minimum "
      f"machine memory available {R.v(DONE, ['min_machine_available_GiB'], '.1f')} GiB [results_j2/test_J_failed.json; "
      "results_j2/run_manifest.json: invocations[2]]. The wrapper reports exit code "
      f"{R.v(DONE, ['returncode'], '.0f')} only because the J1 stage mecke_checks, run after eval in the same job, was "
      "stopped by hand after its first version of the per-halo noise pass proved far too slow; that stage was rewritten "
      "and re-run on its own (J1's maas run 4, CLUSTER_RUN.json j1_mecke_runs[1]).")
    w("  - Why: J1's DM-quantile importance maps were corrected (Mecke maps with the whole DM = 0 atom, J1 revision of "
      "2026-10-16). They enter the DM-family weights of objective (d) of every definition and the DM half of dI. "
      "`j2_inputs.py` rebuilt inputs/j2_fitdata.npz from J1's corrected sens.npz; run 3 refitted every per-bin fit "
      f"(the ladder does not use these weights) and re-evaluated all {R.n('results_j2/evaluate_items.json', ['items'])} "
      "paintings.")
    w(f"  - Gate G17: {R.v(VER, ['gates', 'G17_run2_reproduced', 'n_unchanged_items'], '.0f')} paintings whose painted "
      f"shells are unchanged are reproduced exactly (largest relative difference "
      f"{R.v(VER, ['gates', 'G17_run2_reproduced', 'max_rel_difference_unchanged'], '.1f')} over "
      f"{R.v(VER, ['gates', 'G17_run2_reproduced', 'n_values_compared'], ',.0f')} values); the "
      f"{R.v(VER, ['gates', 'G17_run2_reproduced', 'n_changed_items'], '.0f')} paintings with new fits are exactly "
      "the objective-(d) paintings (incl. RC) and D0's dI / dIT paintings [verification_j2.json: G17].")
    w("")
    w("| run | stage | wall [s] | what |")
    w("|---|---|---|---|")
    st0, st1, st2 = ["invocations", 0, "stages"], ["invocations", 1, "stages"], ["invocations", 2, "stages"]
    w(f"| 1 | j1base | {R.v(MAN, st0 + ['j1base', 'wall_seconds'], '.0f')} | J1's prepare, basis and fallback re-run "
      f"into results_j2/j1base (the 14-bin fine bases JA/JB in node-local scratch) |")
    w(f"| 1 | fits | {R.v(MAN, st0 + ['fits', 'wall_seconds'], '.1f')} | B16 / B12 per bin, objectives (a)/(d)/(L)/dI, "
      f"RC variant (Test G's configuration, 35-shell grid) |")
    w(f"| 1 | ladder | {R.v(MAN, st0 + ['ladder', 'wall_seconds'], '.0f')} | Lee22 ladder per definition, plus the "
      f"Test I parity run |")
    w(f"| 1 | eval | {R.v(MAN, st0 + ['eval', 'wall_seconds'], '.0f')} | "
      f"{R.n('results_j2_run1/evaluate_items.json', ['items'])} paintings, det and S1, DM-PDF passes |")
    w(f"| 2 | j1base | {R.v(MAN, st1 + ['j1base', 'wall_seconds'], '.0f')} | the same rebuild (the run-1 scratch had "
      f"been deleted) |")
    w(f"| 2 | eval | {R.v(MAN, st1 + ['eval', 'wall_seconds'], '.0f')} | "
      f"{R.n('results_j2_run2/evaluate_items.json', ['items'])} paintings against J1's truth |")
    w(f"| 3 | j1base | {R.v(MAN, st2 + ['j1base', 'wall_seconds'], '.0f')} | the same rebuild |")
    w(f"| 3 | fits | {R.v(MAN, st2 + ['fits', 'wall_seconds'], '.1f')} | every per-bin fit, with the corrected DM "
      f"weights |")
    w(f"| 3 | eval | {R.v(MAN, st2 + ['eval', 'wall_seconds'], '.0f')} | "
      f"{R.n('results_j2/evaluate_items.json', ['items'])} paintings against J1's truth |")
    w("")
    w("- **Inputs:** `j2_inputs.py` (local, numpy only) writes inputs/j2_fitdata.npz (run 1 and 2's version: "
      "`results_j2_run2/j2_fitdata_run2.npz`). Code and inputs were uploaded with `j2_cluster.py upload`; before each "
      "launch every file J2 reads and every J2 module was identical on maas (`post_j2/remote_check.json` for run 3, "
      "`results_j2_run2/remote_check_run2.json` for run 2, `results_j2_run1/remote_check_run1.json` for run 1). Before "
      "runs 2 and 3 the remote ladder_*.npz and ladder_summary.json (and before run 2 fits.json) were checked equal "
      "(SHA256) to the local copies of run 1. Run 2's results are kept in `results_j2_run2/`.")
    w(f"- **Scratch:** the bases lived in node-local /tmp/nagainis_halfdome_testJ "
      f"({R.v(CR, ['j2_maas_runs', 0, 'tmp_free_GB_at_launch'], '.1f')} / "
      f"{R.v(CR, ['j2_maas_runs', 1, 'tmp_free_GB_at_launch'], '.1f')} / "
      f"{R.v(CR, ['j2_maas_runs', 2, 'tmp_free_GB_at_launch'], '.1f')} GB free at the three launches). It was deleted "
      f"after each run's final fetch ({R.v(CR, ['j2_maas_runs', 0, 'scratch_bytes_deleted'], '.2f', 1e-9)}, "
      f"{R.v(CR, ['j2_maas_runs', 1, 'scratch_bytes_deleted'], '.2f', 1e-9)} and "
      f"{R.v(CR, ['j2_maas_runs', 2, 'scratch_bytes_deleted'], '.2f', 1e-9)} GB) "
      "[CLUSTER_RUN.json: j2_maas_runs[*].scratch_deleted].")
    w("- **Provenance of run_test_j.py:** J1 ran on maas with sha 719caec... (`post/remote_code_check.json`). The "
      "'--j2' hook was added to it at 19:20 EEST, before J2's first run (every J2 run used the edited file, sha "
      "a97f845...). run_test_j2.py gained the stage 'mecke' (J1's Mecke revision) before run 3, which ran with that "
      "version (gate G1). J1's SHA256SUMS is regenerated by J1's own revision (`j_sums.py`, logged in "
      "`post/sums_update_j1.json`); the earlier updates of its shared lines by J2 are logged in "
      "`post_j2/sums_update_j2.json`.")
    w("- **Local post-processing** (numpy only, < 1 GB): `j2_post.py`, `j2_params.py`, `j2_lsub.py` (why (L) has no "
      "fit in some bins), `j2_figures.py`, `j2_report.py`, `verify_j2.py`, `j2_sums.py`.")
    w("")
    G = R.load(VER)["gates"] if (HERE / VER).exists() else {}
    g11 = G.get("G11_report_numbers", {})

    def ok(name):
        return "passed" if G.get(name, {}).get("passed") else "NOT PASSED / not yet run"

    w("**Gates** [verification_j2.json]:")
    w("")
    w("| gate | check | result |")
    w("|---|---|---|")
    w(f"| G0 | 35-shell inputs equal Test F/G's; x_h, softening, counts, merged groups equal J1's | rebinning "
      f"{R.v(META, ['checks', 'E_a2ii_35_vs_G_f35_max_rel'], '.1e')} relative; the rest bitwise equal |")
    w(f"| G1 | remote provenance: inputs and code on maas equal the local files (run 3) | {ok('G1_remote_provenance')} |")
    w(f"| G2 | J1 rebuild on maas: stacks equal J1's array by array; fallback parity | stacks bitwise equal; "
      f"fallback {R.v('results_j2/j1base/fallback_parity.json', ['JA', 'max_abs_over_max'], '.1e')} |")
    w(f"| G3 | J2's (a) fits of D0-D2, (L) fits of D0 and B12 (a) fits of 13.5-15.5 equal Test I's fits35.json | "
      f"{R.v(FP, ['max_overall'], '.1e')} relative. The dI fits are not covered: they differ (J2.4) |")
    w(f"| G4 | J2's ladder driver on Test I's 10 bins reproduces Test I's E ladder at R_fit 4 | floor chi^2 "
      f"{R.v(LP, ['max_rel_vs_TestI_I2_ladder_models_E_Rfit4', 'chi2_floor'], '.1e')}, sigma_d "
      f"{R.v(LP, ['max_rel_vs_TestI_I2_ladder_models_E_Rfit4', 'sigma_floor'], '.1e')}, CV "
      f"{R.v(LP, ['max_rel_vs_TestI_I2_ladder_models_E_Rfit4', 'cv_chi2'], '.1e')} (all "
      f"{R.v(LP, ['models_compared'], '.0f')} models); stat chi^2 "
      f"{R.v(VER, ['gates', 'G4_ladder_parity_test_I', 'max_rel_converged_fits', 'stat'], '.1e')} for the fits that "
      f"converged inside their bounds in both runs; same selected models (PL3g+O0, L22best+O0). Fits that stopped at "
      f"G's 400-evaluation limit or on a bound depend on their start and are listed separately. |")
    w(f"| G5 | each definition's stack painted det = J1's truth mean | moments and DM quantiles "
      f"{R.v(EP, ['D0', 'max_rel_all'], '.1f')} (bitwise) for D0-D3 |")
    g6t = G.get("G6_pdf_vs_moments", {}).get("variance_deviation_above_1e-4_explained_by_tail", [])
    g6k = lambda i, f: (VER, ["gates", "G6_pdf_vs_moments", "variance_deviation_above_1e-4_explained_by_tail", i, f])  # noqa: E731
    g6p = {}
    for i, t in enumerate(g6t):
        lab, pred = t["painting"].rsplit("|", 1)
        g6p.setdefault(lab, {})[pred] = i
    g6parts = []
    for lab, preds in sorted(g6p.items()):
        order = [x for x in ("S1", "det") if x in preds]
        sl = lambda f, fmt: " / ".join(R.absmax([g6k(preds[x], f)], fmt) for x in order)   # noqa: E731
        g6parts.append(f"({lab}) in its {' and '.join(order)} prediction{'s' if len(order) > 1 else ''} deviates by "
                       f"{sl('rel_dev_variance', '.1e')} in the variance; its PDF{'s' if len(order) > 1 else ''} put"
                       f"{'' if len(order) > 1 else 's'} {sl('mass_beyond_grid', '.1e')} of probability beyond the grid "
                       f"end (8192 pc cm^-3), a lower bound of {sl('deficit_lower_bound_rel', '.1e')} on the deficit")
    g6x = (f"; {len(g6p)} painting{'s' if len(g6p) > 1 else ''} " + "; ".join(g6parts)) if g6t else ""
    w(f"| G6 | DM-PDF mean / variance vs their moment forms, every painting | "
      f"{R.v(VER, ['gates', 'G6_pdf_vs_moments', 'max_rel_mean'], '.1e')} / "
      f"{R.v(VER, ['gates', 'G6_pdf_vs_moments', 'max_rel_variance_others'], '.1e')}{g6x} |")
    w(f"| G7 | every painting has det and S1 observables and DM quantiles | {ok('G7_completeness')} |")
    w(f"| G8 | results_j2/SHA256SUMS after the fetch | {ok('G8_results_checksums')} |")
    w(f"| G9 | the stacks themselves score S = 0 | {ok('G9_stack_scores_zero')} |")
    w(f"| G10 | node-local scratch deleted after each run's final fetch | {ok('G10_scratch_deleted')} |")
    w(f"| G11 | every number of this section re-resolved from its source and found in the text | "
      f"{'passed' if g11.get('passed') else 'not yet run on this text'} |")
    w(f"| G12 | the bins without an (L) fit follow from the rule N_h <= p + 2 or p < 6 and equal the substituted bins "
      f"of every (L*) painting | {ok('G12_L_substitutions')} |")
    w(f"| G13 | every fitted mass-cell ladder law is painted, at every ladder radius | {ok('G13_ladder_complete')} |")
    w(f"| G14 | the dIT paintings use Test I's fits35.json parameters bitwise in 13.5-15.5 and J2's in 12.7-13.5; their "
      f"shells equal the exact shell averages of those parameters | {ok('G14_dIT_construction')} (largest relative "
      f"shell difference {R.v(VER, ['gates', 'G14_dIT_construction', 'max_rel_shell_difference'], '.1e')}) |")
    w(f"| G15 | every run-1 painting reproduced by run 2 (observables, quantiles, shells, CDFs) | "
      f"{ok('G15_run1_reproduced')} |")
    w(f"| G17 | run 3 reproduces every run-2 painting whose painted shells are unchanged; the changed ones are exactly "
      f"the objective-(d) paintings and D0's dI / dIT paintings | {ok('G17_run2_reproduced')} |")
    w(f"| G16 | J1's SHA256SUMS verifies (re-hashed by J1's revision, `post/sums_update_j1.json`) | "
      f"{ok('G16_checksum_manifests')} |")
    w("")
    # ================================================================== J2.2
    sub = {"core": R.v(J1P, ["derived_for_report", "treatment_largest_change", "abs_change"], ".2f", 100),
           "atom": R.v(J1TS, ["crossings", "1R", "p_no_crossing"], ".1f", 100),
           "rebin": R.v(META, ["checks", "E_a2ii_35_vs_G_f35_max_rel"], ".1e"),
           "strict": " / ".join(R.v(SUM, ["strict_rule_first_edge_low_bins", lab], ".2f") for lab in LOWB),
           "xh": xh_seq(),
           "pmin": R.v(LSJ, ["p_min"], ".0f"),
           "ebar": R.v(STK, ["E_cosmic"], ".4f"),
           "dimax": R.v(SUM, ["dI_vs_TestI", "max_rel_param"], ".2f", 100),
           "lfirst": " / ".join(R.v(SUM, ["first_fitted_edge_low_bins", f"D0|L|4|{b}", "first_fitted_edge"], ".2f")
                                for b in (1, 2, 3, 4)),
           "mfirst": " / ".join(R.v(SUM, ["first_fitted_edge_low_bins", f"D0|L|4|{b}", "mask35_first_edge"], ".3f")
                                for b in (1, 2, 3, 4)),
           "nlad": R.agg("sum", [(SUM, ["ladder", "completeness", k, "n_painted"])
                                 for k in S["ladder"]["completeness"]], ".0f")}
    txt = DEFINITIONS
    for k, v in sub.items():
        txt = txt.replace(f"@@{k}@@", v)
    w(txt.strip("\n"))
    w("")
    return R, L, S, SCD, e, pair, rec, worst_stat, which_worst, mcut, km, lm, p, cp, ksz_factor, ksz_ratio_rng, xh_seq


def sections_fits(R, L, S, SCD, e, pair, rec, worst_stat, which_worst, mcut, km, lm, p, cp, ksz_factor,
                  ksz_ratio_rng, xh_seq):
    w = L.append
    # ================================================================== J2.3
    w("### J2.3 J2a: B16 per mass bin for each definition")
    w("")
    w("All paintings are cut at the definition's boundary; k = 42 (3 parameters x 14 bins). Objectives (J2.2): (a), "
      "(d), (L*), and for D0 dI and dIT. Columns: DM family error; "
      "DM-field C_ell and y x DM family errors (ell = 100-3000); the kSZ 1-halo family error of the painted electrons; "
      "and the sensitive-part score in the fitted region S_DM,fit (J2.2, J2.6) [post_j2/J2_scores.json; "
      "post_j2/J2_scores.csv]. 'RC': the RC fits in 12.7-13.5.")
    w("")
    for D in DEFS:
        w(f"**{DNAME[D]}** (S1 floor {e(f'ref|{D}|stack', 'rms_DM')}%):")
        w("")
        w("| R_fit | objective | DM family [%] | DM-field C_ell [%] | y x DM [%] | kSZ 1-halo [%] | S_DM,fit [%] |")
        w("|---|---|---|---|---|---|---|")
        labs = [lab for lab, r in SCD.items() if r["D"] == D and r["block"] == "J2a"]
        labs.sort(key=lambda lab: (lab.startswith("rc"), float(SCD[lab]["R_fit"]),
                                   ("a", "d", "dI", "dIT", "L").index(SCD[lab]["objective"])))
        for lab in labs:
            r = SCD[lab]
            obj = {"dI": "dI (Test I's formula, C14 maps, refit)", "dIT": "dIT (Test I's fits + dI below 13.5)",
                   "L": "L*"}.get(r["objective"], r["objective"])
            if lab.startswith("rc"):
                obj += ", RC"
            w(f"| {R.v(SCJ, [lab, 'R_fit'], 'g')} | {obj} | {pair(lab)} | {e(lab, 'rms_DMcl')} | {e(lab, 'rms_ydm')} | "
              f"{e(lab, 'rms_kSZ')} | {e(lab, 'S_DM_fit', '.1f')} |")
        w("")
    w("Per DM statistic, the recommended painting of each definition, S1 (profile-only) in % "
      "[post_j2/J2_scores.json: e_*, p_*]:")
    w("")
    w("| statistic | " + " | ".join(f"{D}: {pretty(p[D])}" for D in DEFS) + " |")
    w("|---|" + "---|" * len(DEFS))
    for k in ("dm_mean", "dm_var", "q16", "q50", "q84", "q975", "q999"):
        cells = []
        for D in DEFS:
            if k not in C.dm_family(D):
                cells.append("in the atom (not used)")
            else:
                cells.append(f"{e(p[D], 'e_' + k)} ({e(p[D], 'p_' + k)})")
        w(f"| {STAT_LAB[k]} | " + " | ".join(cells) + " |")
    w("")
    a0 = "a|D0|B16|a|Rf4"
    w("Findings:")
    w(f"- **D0 needs R_fit = 4**, as in Test I. B16 (d) gives {pair('a|D0|B16|d|Rf2')} / {pair('a|D0|B16|d|Rf3')} / "
      f"{pair('a|D0|B16|d|Rf4')}% at R_fit = 2 / 3 / 4.")
    w(f"  - At R_fit 4, objective (a) gives {pair(a0)}% and (L*) {pair('a|D0|B16|L|Rf4')}% (the covariance of the "
      f"mean weights the inner shells, as in Test I).")
    w(f"  - Test I's recipe: dIT (Test I's published fits in 13.5-15.5, unchanged, + dI fits in 12.7-13.5) gives "
      f"{pair('a|D0|B16|dIT|Rf4')}% on the 14 bins. Test I reported "
      f"{R.v(IRN, ['I1|E|d|Rf4|Rc4|rms_DM'], '.2f', 100)} ({R.v(IRN, ['I1|E|d|Rf4|Rc4|prof_DM'], '.2f', 100)})% for "
      f"the same fits on its 10 bins, where the haloes below 10^13.5 were painted identically in truth and model "
      f"[I_fitting_a2ii/report_numbers.json]. With Test I's fits unchanged above 10^13.5, the increase comes from the "
      f"bins 12.7-13.5, which are now part of both truth and model (exact, gate G14).")
    w(f"  - The dI refit of all 14 bins gives {pair('a|D0|B16|dI|Rf4')}%. Its weights come from J1's C14 maps, "
      f"so in 13.5-15.5 it moves Test I's fits by up to {R.v(SUM, ['dI_vs_TestI', 'max_rel_param'], '.2f', 100)}% in "
      f"the parameters (J2.4); it is not 'identical to Test I' in those bins (gate G3 does not cover dI).")
    lo, hi = "a|D1|B16|L|Rf1.34", "a|D1|B16|a|Rf1.34"
    d1labs = [lab for lab, r in SCD.items() if r["D"] == "D1" and r["block"] == "J2a"]
    w(f"- **D1 is limited by the scatter model.** Every per-bin B16 painting lies within "
      f"{R.rng([(SCJ, [lab, 'rms_DM']) for lab in d1labs], '.2f', 100)}% (S1), around the floor of "
      f"{e('ref|D1|stack', 'rms_DM')}%. The profile-only errors "
      f"({R.rng([(SCJ, [lab, 'prof_DM']) for lab in d1labs], '.2f', 100)}%) separate the forms.")
    w(f"  - (L*) is best at both R_fit. Fitting the A2-ii gas to 1.34 R200c and cutting at 1 R200c "
      f"({pair(lo)}%) is better than stopping the fit at 1 R200c ({pair('a|D1|B16|L|Rf1')}%).")
    w(f"  - Objective (d) brings nothing for D1 ({pair('a|D1|B16|d|Rf1')}% against (a) {pair('a|D1|B16|a|Rf1')}% "
      f"at R_fit 1).")
    w(f"- **D2:** R_fit = 2 is needed (R_fit 1.34: {pair('a|D2|B16|a|Rf1.34')}% for (a)). At R_fit 2, (L*) gives "
      f"{pair('a|D2|B16|L|Rf2')}%, (a) {pair('a|D2|B16|a|Rf2')}% and (d) {pair('a|D2|B16|d|Rf2')}%. Fitting further "
      f"out does not beat (L*) at R_fit 2: at R_fit 3, (a) gives {pair('a|D2|B16|a|Rf3')}% and (L*) "
      f"{pair('a|D2|B16|L|Rf3')}%.")
    w("- **D3: the B16 form fails for bound gas.**")
    w(f"  - With the S400 masks the best per-bin painting is (d) at R_fit 4: {pair('a|D3|B16|d|Rf4')}% (its Q16 error "
      f"{e('a|D3|B16|d|Rf4', 'e_q16')}%). Its RC variant gives {pair('rc|D3|B16|d|Rf4')}%, the best per-bin B16 "
      f"painting of D3 overall. Objective (a) gives {pair('a|D3|B16|a|Rf4')}% and (L*) {pair('a|D3|B16|L|Rf4')}%.")
    w(f"  - The fits' own profile RMS (model/stack - 1 over the fitted shells) reaches "
      f"{R.agg('max', [('results_j2/fits.json', ['fits', f'D3|a|4|{b}', 'rms_percent']) for b in (1, 2, 3, 4)], '.0f')}% "
      f"in 12.7-13.5 for (a) at R_fit 4 [results_j2/fits.json: fits.D3|a|4|*.rms_percent].")
    w("  - B16 has one outer power law and cannot drop steeply at 1-2 R200c, so it is low near R200c and far too high "
      "at 2-4 R200c (`figures_j2/J2_residual_profiles`).")
    # ---- (L*) substitutions, from post_j2/J2_L_substitutions.json
    LS_ = R.load(LSJ)["missing_L_fits"]
    hi_bins = ("15.1-15.3", "15.3-15.5")
    every = all(all(b in LS_[k] for b in hi_bins) for k in LS_)
    nh = " and ".join(R.v(LSJ, ["missing_L_fits", "D0|Rf4", b, "N_h"], ".0f") for b in hi_bins)
    pmin_hi = R.agg("min", [(LSJ, ["missing_L_fits", k, b, "p"]) for k in LS_ for b in hi_bins], ".0f")
    w(f"- **(L*) substitutions** [results_j2/evaluate_items.json: a_substituted_bins; post_j2/J2_L_substitutions.json]. "
      f"These bins have no (L) fit and are painted with their (a) fits:")
    assert every, "15.1-15.5 expected without (L) fit at every R_fit"
    w(f"  - 15.1-15.5 at every R_fit of every definition: only {nh} haloes, while at least {pmin_hi} shells are "
      f"fitted (N_h <= p + 2, so the Hartlap factor H = (N_h - p - 2)/(N_h - 1) is non-positive).")
    low_txt = []
    for D in DEFS:
        rfs = [k for k in LS_ if k.startswith(D + "|")]
        per = {}
        for k in rfs:
            for b, r in LS_[k].items():
                if b not in hi_bins:
                    per.setdefault(b, []).append(k.split("|Rf")[1])
        if not per:
            continue
        all_rf = [k.split("|Rf")[1] for k in rfs]
        parts = []
        for b in sorted(per):
            when = "every R_fit" if sorted(per[b]) == sorted(all_rf) else "R_fit " + " and ".join(per[b])
            ps = sorted({LS_[f'{D}|Rf{x}'][b]['p'] for x in per[b]})
            parts.append(f"{b} at {when} (p = {', '.join(str(x) for x in ps)})")
        low_txt.append(f"{D}: {'; '.join(parts)}")
    w(f"  - Too few shells with gas in every halo (p < {R.v(LSJ, ['p_min'], '.0f')}): " + "; ".join(low_txt) + ".")
    ev = R.load("results_j2/evaluate_items.json")["items"]
    sub_d1 = next(it["a_substituted_bins"] for it in ev if it["label"] == p["D1"])
    w(f"  - For the recommended D1 painting ({pretty(p['D1'])}) the substituted bins are {join_and(sub_d1)}.")
    w(f"- **RC fits** (resolved shape + core integral in 12.7-13.5 instead of the S400 shells) change the S1 DM error "
      f"of D0-D2 by at most {R.v(SUM, ['rc_max_abs_change_rms_DM_D0_D2'], '.2f', 100)} pp (D0 (a): "
      f"{pair('rc|D0|B16|a|Rf4')} vs {pair('a|D0|B16|a|Rf4')}%), the others by "
      f"{rc_other_txt(R)} (e.g. D1 (a): "
      f"{pair('rc|D1|B16|a|Rf1')} vs {pair('a|D1|B16|a|Rf1')}%) [post_j2/J2_summary.json: rc_change]. For D3 the "
      f"B16 failure dominates ((d): {pair('rc|D3|B16|d|Rf4')} vs {pair('a|D3|B16|d|Rf4')}%).")
    w(f"  - The RC grid has only {R.v('results_j2/fits.json', ['rc', 'D1|a|1|1', 'n_points'], '.0f')} shells for D1 in "
      f"12.7-12.9: no fit, so the S400 fit is painted there.")
    w(f"  - For D1 in 12.9-13.1 and D2 in 12.7-12.9 the RC fit is degenerate: rho0 runs to its upper bound "
      f"({R.v('results_j2/fits.json', ['rc', 'D1|a|1|2', 'parameters', 0], '.0e')}) with alpha = "
      f"{R.v('results_j2/fits.json', ['rc', 'D1|a|1|2', 'parameters', 1], '.2f')}-"
      f"{R.v('results_j2/fits.json', ['rc', 'D2|a|2|1', 'parameters', 1], '.2f')}, i.e. a steep inner power law. "
      f"`analysis.fitting` flags these fits as not accepted, but they are painted as they are (their profile RMS is "
      f"{R.rng([('results_j2/fits.json', ['rc', k, 'rms_percent']) for k in ('D1|a|1|2', 'D1|d|1|2', 'D2|a|2|1', 'D2|d|2|1')], '.1f')}%) "
      f"[results_j2/fits.json: rc]. Every non-RC fit is accepted.")
    w("")
    # ================================================================== J2.4
    w("### J2.4 J2a: the newly calibrated bins with Test I's recipe (D0 target)")
    w("")
    w("Test I's recipe: pressure B12 objective (a); electrons B16 objective (d) with the weights "
      "w = 0.5 w_kSZ + 0.5 w_DM (J2.2); R_fit = R_cut = 4 R200c. The recommended painting **dIT** keeps Test I's "
      "published fits in 13.5-15.5 unchanged (`I_fitting_a2ii/results/fits35.json`: E|d|4, P|a|4). Test I has no fits "
      "in 12.7-13.5, and Test H's importance maps (the source of Test I's weights) do not cover these bins. There the "
      "weights come from J1's C14 maps and the fitted shells from the S400 masks. Parameters of the new fits "
      "[results_j2/fits.json: fits.P|a|4|b, fits.D0|dI|4|b]; P0 in P200 units, rho0 in E units, x_c in R200c; "
      "profile RMS: the RMS of model/stack - 1 over the fitted shells:")
    w("")
    w("| bin | B12: P0 | x_c | beta | profile RMS [%] | B16 (dI): rho0 | alpha | beta | profile RMS [%] | "
      "fitted shells P / E |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    FJ = "results_j2/fits.json"
    for b in (1, 2, 3, 4):
        kp, ke = f"P|a|4|{b}", f"D0|dI|4|{b}"
        w(f"| {C.bin_label(b)} | {R.v(FJ, ['fits', kp, 'parameters', 0], '.3f')} | "
          f"{R.v(FJ, ['fits', kp, 'parameters', 1], '.3f')} | {R.v(FJ, ['fits', kp, 'parameters', 2], '.2f')} | "
          f"{R.v(FJ, ['fits', kp, 'rms_percent'], '.1f')} | {R.v(FJ, ['fits', ke, 'parameters', 0], '.1f')} | "
          f"{R.v(FJ, ['fits', ke, 'parameters', 1], '.3f')} | {R.v(FJ, ['fits', ke, 'parameters', 2], '.2f')} | "
          f"{R.v(FJ, ['fits', ke, 'rms_percent'], '.1f')} | {R.v(FJ, ['fits', kp, 'n_shells'], '.0f')} / "
          f"{R.v(FJ, ['fits', ke, 'n_shells'], '.0f')} |")
    w("")
    w(f"- The low-mass electron profiles are shallow outside: beta = "
      f"{R.rng([(FJ, ['fits', f'D0|dI|4|{b}', 'parameters', 2]) for b in (1, 2, 3, 4)], '.2f')} in 12.7-13.5 against "
      f"{R.v('../I_fitting_a2ii/results/fits35.json', ['fits', 'E|d|4|5', 'parameters', 2], '.2f')} in 14.5-14.7 "
      f"(Test I). Only "
      f"{R.rng([(J1PREP, ['J1a_by_bin', b, 'share_of_Ne_lt4R_inside_R200c', 'A2ii']) for b in LOWB], '.0f', 100)}% "
      f"of their A2-ii electrons inside 4 R200c lie inside R200c (J1).")
    w("")
    w("**Test I's fits against the dI refit in 13.5-15.5** [post_j2/J2_summary.json: dI_vs_TestI]. dI applies Test I's "
      "weight formula to J1's C14 maps, whose responses include the newly calibrated bins; its weights in 13.5-15.5 "
      "therefore differ from Test I's, and so do its fits:")
    w("")
    w("| bin | Test I: rho0 | alpha | beta | dI refit: rho0 | alpha | beta | largest parameter change [%] |")
    w("|---|---|---|---|---|---|---|---|")
    for b in range(5, 15):
        lab = C.bin_label(b)
        pb = ["dI_vs_TestI", "per_bin", lab]
        w(f"| {lab} | {R.v(DIV, pb + ['TestI_E_d_4', 0], '.1f')} | {R.v(DIV, pb + ['TestI_E_d_4', 1], '.3f')} | "
          f"{R.v(DIV, pb + ['TestI_E_d_4', 2], '.3f')} | {R.v(DIV, pb + ['J2_dI', 0], '.1f')} | "
          f"{R.v(DIV, pb + ['J2_dI', 1], '.3f')} | {R.v(DIV, pb + ['J2_dI', 2], '.3f')} | "
          f"{R.v(DIV, pb + ['max_rel_param'], '.2f', 100)} |")
    w("")
    fd = ["dI_vs_TestI", "families"]
    w(f"- Exact effect on the 14-bin paintings (S1), dIT against dI: DM family "
      f"{R.v(DIV, fd + [KSZ_RECIPE, 'rms_DM'], '.2f', 100)} vs {R.v(DIV, fd + [DI_REFIT, 'rms_DM'], '.2f', 100)}%, "
      f"kSZ 1-halo {R.v(DIV, fd + [KSZ_RECIPE, 'rms_kSZ'], '.2f', 100)} vs "
      f"{R.v(DIV, fd + [DI_REFIT, 'rms_kSZ'], '.2f', 100)}%, kSZ 2-halo "
      f"{R.v(DIV, fd + [KSZ_RECIPE, 'rms_kSZ2h'], '.2f', 100)} vs {R.v(DIV, fd + [DI_REFIT, 'rms_kSZ2h'], '.2f', 100)}%, "
      f"tSZ {R.v(DIV, fd + [KSZ_RECIPE, 'rms_tSZ'], '.2f', 100)} vs {R.v(DIV, fd + [DI_REFIT, 'rms_tSZ'], '.2f', 100)}%. "
      + ("Keeping Test I's fits is at least as accurate as the refit in all four families. "
         if all(R.load(DIV)["dI_vs_TestI"]["families"][KSZ_RECIPE][k] <= R.load(DIV)["dI_vs_TestI"]["families"][DI_REFIT][k] + 1e-6
                for k in ("rms_DM", "rms_kSZ", "rms_kSZ2h", "rms_tSZ")) else "")
      + f"The exact values agree with the first-order ESTIMATE made in the review of run 1 (DM "
      f"{R.v(V10, ['DM_family_with_TestI_high_fits_ESTIMATE_pct'], '.2f')}, kSZ 1-halo "
      f"{R.v(V10, ['kSZ_family_with_TestI_high_fits_ESTIMATE_pct'], '.2f')}, kSZ 2-halo "
      f"{R.v(V10, ['kSZ2h_family_with_TestI_high_fits_ESTIMATE_pct'], '.2f')}%; "
      f"`verify_independent_Test_J2/out_r2/v2_s10_dI_effect_estimate.json`).")
    w("")
    w("Effect of replacing the published fallback by the new fits (both fields painted; S1; C10 = Test I as "
      "published, i.e. its fits in 13.5-15.5 and the published B12/B16 below 10^13.5; C14 = the same plus the new fits "
      "in 12.7-13.5, i.e. dIT; truth = J1's A2-ii stacks of all 14 bins) [post_j2/J2_lowmass.json, keys "
      "'dIT|S1|<observable>']:")
    w("")
    w("| observable | C14 / C10 - 1 [%] | C10 (Test I) vs truth [%] | C14 (dIT) vs truth [%] |")
    w("|---|---|---|---|")
    obs = ([(f"yytot_{x}", f"tSZ C_ell^yy, ell = {x}") for x in (100, 300, 1000, 3000)] + [("ymean", "mean y")]
           + [(f"ksz1h_{x}", f"kSZ 1-halo, ell = {x}") for x in (100, 300, 1000, 3000)]
           + [(f"ksz2h_{x}", f"kSZ 2-halo, ell = {x}") for x in (100, 1000, 3000)]
           + [("dm_mean", "mean DM"), ("dm_var", "DM variance")]
           + [(q, f"DM {STAT_LAB[q]}") for q in ("q16", "q50", "q84", "q975", "q999")]
           + [(f"dmcltot_{x}", f"DM-field C_ell, ell = {x}") for x in (100, 1000, 3000)]
           + [(f"ydmtot_{x}", f"y x DM, ell = {x}") for x in (100, 1000, 3000)])
    for k, lab in obs:
        w(f"| {lab} | {lm('dIT', 'S1', k, 'change_C14_over_C10')} | {lm('dIT', 'S1', k, 'err_C10_fit')} | "
          f"{lm('dIT', 'S1', k, 'err_C14_fit')} |")
    w("")
    LMD = R.load(LMJ)
    ch = {k: LMD[f"dIT|S1|{k}"]["change_C14_over_C10"] for k, _ in obs}
    names = dict(obs)
    big = sorted([k for k in ch if abs(ch[k]) >= 0.05], key=lambda k: -abs(ch[k]))
    rest = [k for k in ch if k not in big and not k.startswith("yytot")]
    w("- Calibrating 12.7-13.5 changes these observables by 5% or more (largest first): "
      + "; ".join(f"{names[k]} {R.v(LMJ, [f'dIT|S1|{k}', 'change_C14_over_C10'], '+.1f', 100)}%" for k in big) + ".")
    w(f"- Every other observable of the table moves by at most "
      f"{R.absmax([(LMJ, [f'dIT|S1|{k}', 'change_C14_over_C10']) for k in rest], '.1f', 100)}%, and tSZ C_ell^yy by "
      f"at most {R.absmax([(LMJ, [f'dIT|S1|yytot_{x}', 'change_C14_over_C10']) for x in (100, 300, 1000, 3000)], '.1f', 100)}%.")
    w(f"- Family errors against the truth, C10 -> C14 [post_j2/J2_scores.json: {TESTI_PUB}, {KSZ_RECIPE}]: DM "
      f"{e(TESTI_PUB, 'rms_DM')} -> {e(KSZ_RECIPE, 'rms_DM')}%, DM-field C_ell "
      f"{e(TESTI_PUB, 'rms_DMcl')} -> {e(KSZ_RECIPE, 'rms_DMcl')}%, kSZ 1-halo "
      f"{e(TESTI_PUB, 'rms_kSZ')} -> {e(KSZ_RECIPE, 'rms_kSZ')}%, kSZ 2-halo "
      f"{e(TESTI_PUB, 'rms_kSZ2h')} -> {e(KSZ_RECIPE, 'rms_kSZ2h')}%, tSZ "
      f"{e(TESTI_PUB, 'rms_tSZ')} -> {e(KSZ_RECIPE, 'rms_tSZ')}%.")
    w(f"- With objective (a) for the electrons instead (J2's (a) fits equal Test I's E|a|4 in 13.5-15.5, gate G3): DM "
      f"{e('lm|C10|B12a+B16a', 'rms_DM')} -> {e('lm|C14|B12a+B16a', 'rms_DM')}%.")
    w("")


def sections_ladder(R, L, S, SCD, e, pair, rec, worst_stat, which_worst, mcut, km, lm, p, cp, ksz_factor,
                    ksz_ratio_rng, xh_seq):
    w = L.append
    # ================================================================== J2.5
    w("### J2.5 J2b: the Lee22 ladder per definition")
    w("")
    w("Test G's selection machinery on each definition's own 35-shell stacks (14 bins; mass cells and concentration "
      "cells), with G's rule (the fewest k with dBIC <= 6 and CV within the best-CV model's fold scatter) "
      "[results_j2/ladder_summary.json: chosen, sigma_floor; results_j2/tables/J2b_ladder_models_<D>.csv]:")
    w("")
    w("| definition | R_fit | cells | selected (k) | chi^2/nu floor | dBIC | rule satisfied | best BIC | best CV | "
      "sigma_d [%] |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for D in DEFS:
        for rf in C.DEFS[D]["lad_rfit"]:
            for ds in ("mass", "massc"):
                key = f"E|{ds}|{rf:g}"
                ch = S["per_definition"][D]["ladder_chosen"][key]
                w(f"| {D} | {rf:g} | {ds} | {ch['model']} ({R.v(LS, ['chosen', D, key, 'k'], '.0f')}) | "
                  f"{R.v(LS, ['chosen', D, key, 'chi2nu_floor'], '.2f')} | {R.v(LS, ['chosen', D, key, 'dBIC'], '.1f')} | "
                  f"{'yes' if ch['rule_satisfied'] else 'no'} | {ch['best_BIC']} | {ch['best_CV']} | "
                  f"{R.v(LS, ['sigma_floor', D, f'{ds}|{rf:g}', 'sigma_ref'], '.2f', 100)} |")
    w("")
    w("Key mass-cell models (chi^2/nu under the stat / floor / L22 likelihoods; dBIC; CV - CV_best and the fold "
      "scatter sqrt(K) SD_folds of the best-CV model, J2.2; DM family error S1 (profile-only) of the painting at "
      "R_cut = boundary) [post_j2/J2_summary.json: ladder.key_models; post_j2/J2_scores.json]:")
    w("")
    w("| definition | model (k) | chi^2/nu stat | floor | L22 | dBIC | CV - CV_best | fold scatter | DM family [%] |")
    w("|---|---|---|---|---|---|---|---|---|")
    for D in DEFS:
        for rf in C.DEFS[D]["lad_rfit"]:
            for m in ("PL3+O0", "PL3g+O0", "BPLall_L22+O0", "BPLall+O0", "PB+O0"):
                key = f"{m}|Rf{rf:g}"
                if key not in S["ladder"]["key_models"].get(D, {}):
                    continue
                lab = f"l|{D}|{m}|mass|Rf{rf:g}"
                l22 = S["ladder"]["key_models"][D][key].get("chi2nu_L22")
                l22t = km(D, key, "chi2nu_L22") if l22 is not None else "no cell usable"
                err = pair(lab) if lab in SCD else "not painted"
                w(f"| {D}, R_fit {rf:g} | {m} ({km(D, key, 'k', '.0f')}) | {km(D, key, 'chi2nu_stat')} | "
                  f"{km(D, key, 'chi2nu_floor')} | {l22t} | {km(D, key, 'dBIC_floor', '.1f')} | "
                  f"{km(D, key, 'cv_minus_best', '.0f')} | {km(D, key, 'cv_scatter_best', '.0f')} | {err} |")
    w("")
    w("AIC, BIC, the per-fold CV chi^2 and the parameters of every model are in "
      "results_j2/tables/J2b_ladder_models_<D>.csv; the parsimony ladders (best chi^2 per k) in "
      "results_j2/tables/J2b_ladder_ladder_<D>.csv and `figures_j2/J2_ladder_chi2_vs_k`.")
    w("")
    w("**DM accuracy against the number of parameters, over the complete mass-cell ladder** (run 2). Every fitted "
      "mass-cell law is painted (gate G13). The table lists, per definition and ladder radius, the laws that are more "
      "accurate in DM than every law with fewer parameters, i.e. the front of max(S1, profile-only) DM family error "
      "against k (the quantity of the recommendation rule, J2.2). Columns: the law; its DM family error S1 "
      "(profile-only); its floor chi^2/nu and dBIC; the number of fitted laws with the same k "
      "[post_j2/J2_summary.json: ladder.dm_by_k]:")
    w("")
    w("| definition (R_fit) | k | law | DM family [%] | chi^2/nu floor | dBIC | laws with this k |")
    w("|---|---|---|---|---|---|---|")
    for key, perk in S["ladder"]["dm_by_k"].items():
        run = float("inf")
        for i, ent in enumerate(perk):
            if ent["max_S1_profile"] < run:
                run = ent["max_S1_profile"]
                pk = ["ladder", "dm_by_k", key, i]
                c2 = R.v(SUM, pk + ["chi2nu_floor"], ".2f") if ent.get("chi2nu_floor") is not None else "-"
                db = R.v(SUM, pk + ["dBIC_floor"], ".1f") if ent.get("dBIC_floor") is not None else "-"
                w(f"| {key.replace('|Rf', ' (')}) | {R.v(SUM, pk + ['k'], '.0f')} | {ent['model']} | "
                  f"{R.v(SUM, pk + ['rms_DM'], '.2f', 100)} ({R.v(SUM, pk + ['prof_DM'], '.2f', 100)}) | {c2} | {db} | "
                  f"{R.v(SUM, pk + ['n_models_with_k'], '.0f')} |")
    w("")
    w("DM errors of the selected mass-cell models at R_cut = boundary: S1 vs the truth, det vs the truth, det vs the "
      "truth mean (profile-only) [post_j2/J2_scores.json: rms_DM, rms_DM_det_vs_truth, prof_DM]:")
    w("")
    w("| definition | selected model (R_fit) | S1 vs truth [%] | det vs truth [%] | det vs truth mean [%] |")
    w("|---|---|---|---|---|")
    for D in DEFS:
        for lab, r in S["per_definition"][D]["ladder_selected"].items():
            if "|mass|" not in lab:
                continue
            w(f"| {D} | {lab.split('|')[2]} ({lab.split('|')[4][2:]}) | {e(lab, 'rms_DM')} | "
              f"{e(lab, 'rms_DM_det_vs_truth')} | {e(lab, 'prof_DM')} |")
    w("")
    w("Findings:")
    w(f"- **Parity with Test I.** On Test I's 10 bins the J2 driver reproduces Test I's E ladder at R_fit 4 "
      f"(floor chi^2 to {R.v(LP, ['max_rel_vs_TestI_I2_ladder_models_E_Rfit4', 'chi2_floor'], '.0e')} relative) "
      f"and its selections (PL3g+O0 on mass cells, L22best+O0 on concentration cells) [results_j2/ladder_parity.json; "
      f"verification_j2.json G4].")
    w(f"- **Adding 12.7-13.5 breaks the single power laws.** For D0 the floor chi^2/nu of PL3+O0 / PL3g+O0 is "
      f"{km('D0', 'PL3+O0|Rf4', 'chi2nu_floor')} / {km('D0', 'PL3g+O0|Rf4', 'chi2nu_floor')} (dBIC "
      f"{km('D0', 'PL3+O0|Rf4', 'dBIC_floor', '.0f')} / {km('D0', 'PL3g+O0|Rf4', 'dBIC_floor', '.0f')}), against "
      f"{R.v(ILS, ['chosen', 'E|mass|4', 'chi2nu_floor'], '.2f')} for PL3g+O0 on Test I's 10 bins.")
    w(f"  - A break in mass is needed: log10 M_cut = {mcut('D0', 'BPLall+O0|mass|Rf4')} for D0 (BPLall+O0), "
      f"{mcut('D1', 'BPLall+O0|mass|Rf1')} for D1, {mcut('D2', 'BPLall+O0|mass|Rf2')} for D2 and "
      f"{mcut('D3', 'BPLall+O0|mass|Rf2')} for D3 at R_fit 2 [post_j2/J2_summary.json: ladder.log10Mcut_free].")
    w(f"  - Fixing the break at Lee22's value (BPLall_L22, J2.2) costs dBIC "
      f"{km('D0', 'BPLall_L22+O0|Rf4', 'dBIC_floor', '.0f')} for D0.")
    ch0 = S["per_definition"]["D0"]["ladder_chosen"]["E|mass|4"]
    w(f"- **D0:** G's rule picks {ch0['model']} (k = {R.v(LS, ['chosen', 'D0', 'E|mass|4', 'k'], '.0f')}) by BIC. "
      f"No model passes both criteria (the per-bin PB+O0 is best in CV), so rule_satisfied = no.")
    w(f"  - DM errors: BPLall+O0 {pair('l|D0|BPLall+O0|mass|Rf4')}%, PB+O0 {pair('l|D0|PB+O0|mass|Rf4')}%, the "
      f"per-bin B16 (d) {pair(p['D0'])}%.")
    w(f"  - The systematic floor is sigma_d = {R.v(LS, ['sigma_floor', 'D0', 'mass|4', 'sigma_ref'], '.1f', 100)}% "
      f"per shell (Test I on 10 bins: "
      f"{R.v(ILS, ['sigma_floor', str(('E', 'mass', 4.0)), 'sigma_ref'], '.1f', 100)}%).")
    OV = S["ladder"]["outer_vs_O0"]

    def outer_txt(key):
        D, rf = key.split("|Rf")
        ov = OV[key]
        ex = ov["inners_where_an_outer_option_beats_O0"]
        bi, bo = ov["best_overall_by_BIC"]
        o_labs = [f"l|{D}|{i}+{o}|mass|Rf{rf}" for i, v in ov["per_inner"].items() for o in ("O1", "O2") if o in v]
        z_labs = [f"l|{D}|{i}+O0|mass|Rf{rf}" for i, v in ov["per_inner"].items() if "O0" in v]
        exs = ("for every inner law except " + join_and(ex) + ", where " +
               "; ".join(f"{i}+{ov['per_inner'][i]['best_by_BIC']} has dBIC "
                         f"{R.v(SUM, ['ladder', 'outer_vs_O0', key, 'per_inner', i, ov['per_inner'][i]['best_by_BIC'], 'dBIC_floor'], '.0f')} "
                         f"against {R.v(SUM, ['ladder', 'outer_vs_O0', key, 'per_inner', i, 'O0', 'dBIC_floor'], '.0f')} "
                         f"for {i}+O0" for i in ex)) if ex else "for every inner law"
        return (f"O0 has the lowest BIC {exs}; the best law overall by BIC is {bi}+{bo}. In DM the best law with "
                f"an outer option reaches {R.agg('min', [(SCJ, [lab, 'rms_DM']) for lab in o_labs], '.2f', 100)}% (S1), "
                f"the best O0 law {R.agg('min', [(SCJ, [lab, 'rms_DM']) for lab in z_labs], '.2f', 100)}%")

    w(f"  - Outer options (two-halo term O1, truncated O2), D0 at R_fit 4: {outer_txt('D0|Rf4')} "
      f"[post_j2/J2_summary.json: ladder.outer_vs_O0].")
    w(f"- **D1 and D2:** the per-bin law PB+O0 is selected (k = 42; sigma_d = "
      f"{R.v(LS, ['sigma_floor', 'D1', 'mass|1', 'sigma_ref'], '.2f', 100)}% and "
      f"{R.v(LS, ['sigma_floor', 'D2', 'mass|2', 'sigma_ref'], '.2f', 100)}%).")
    w(f"  - The best compact law by BIC, BPLall+O0, has dBIC {km('D1', 'BPLall+O0|Rf1', 'dBIC_floor', '.0f')} (D1) and "
      f"{km('D2', 'BPLall+O0|Rf2', 'dBIC_floor', '.0f')} (D2); every other law with k < 42 has a larger dBIC "
      f"[results_j2/tables/J2b_ladder_models_<D>.csv].")
    w(f"  - In DM, BPLall+O0 gives {pair('l|D1|BPLall+O0|mass|Rf1')}% (D1) and {pair('l|D2|BPLall+O0|mass|Rf2')}% "
      f"(D2); the front above lists the laws that do better with fewer parameters.")
    w(f"- **D3:** sigma_d = {R.v(LS, ['sigma_floor', 'D3', 'mass|2', 'sigma_ref'], '.0f', 100)}% at R_fit 2 and "
      f"{R.v(LS, ['sigma_floor', 'D3', 'mass|4', 'sigma_ref'], '.0f', 100)}% at R_fit 4: no gNFW law, with or without "
      f"the outer options, follows the bound gas shell by shell.")
    c32, c34 = (S["per_definition"]["D3"]["ladder_chosen"][f"E|mass|{x}"]["model"] for x in ("2", "4"))
    w(f"  - G's rule picks {c32} at R_fit 2 ({pair(f'l|D3|{c32}|mass|Rf2')}%) and {c34} at R_fit 4 "
      f"({pair(f'l|D3|{c34}|mass|Rf4')}%).")
    w(f"  - Outer options, D3 at R_fit 2: {outer_txt('D3|Rf2')}. At R_fit 4: {outer_txt('D3|Rf4')}. Bound gas has no "
      f"two-halo term by construction; where O2 wins by BIC, it does so through its truncation factor.")
    import csv as _csv
    p3 = p["D3"]
    if p3.startswith("l|"):
        m3, rf3 = p3.split("|")[2], float(p3.split("|")[4][2:])
        mrow = [r for r in _csv.DictReader(open(HERE / "results_j2" / "tables" / "J2b_ladder_models_D3.csv"))
                if r["dset"] == "mass" and r["model"] == m3 and abs(float(r["R_fit"]) - rf3) < 1e-9][0]
        bnd = [x for x in (mrow["at_bound_floor"] or "").split(";") if x]
        xc = [(RP, ["D3", "primary", "per_bin", "x_c", i]) for i in range(14)]
        be = [(RP, ["D3", "primary", "per_bin", "beta", i]) for i in range(14)]
        RPD = R.load(RP)["D3"]["primary"]["per_bin"]
        w(f"  - The recommended D3 law, {m3} at R_fit {rf3:g}, has "
          + (f"{len(bnd)} parameter(s) at a bound of G's fit ({', '.join(bnd)})" if bnd else
             "no parameter at a bound of G's fit")
          + f" [results_j2/tables/J2b_ladder_models_D3.csv: at_bound_floor]. Its per-bin x_c = "
          f"{R.agg('min', xc, '.1f')} to {R.agg('max', xc, '.1e')} R200c and beta = {R.agg('min', be, '.0f')} to "
          f"{R.agg('max', be, '.1e')} [post_j2/J2_recommended_parameters.json].")
        if min(RPD["beta"]) > 10:
            w(f"  - With beta >> 1 the factor (1 + x/x_c)^-beta approaches exp(-beta x/x_c): the law runs towards an "
              f"exponentially truncated power law x^-0.3 exp(-x/x_e) with e-folding length x_e = x_c/beta = "
              f"{R.r(RP, ['D3', 'primary', 'per_bin', 'x_c', 13], RP, ['D3', 'primary', 'per_bin', 'beta', 13], '.2f')}-"
              f"{R.r(RP, ['D3', 'primary', 'per_bin', 'x_c', 0], RP, ['D3', 'primary', 'per_bin', 'beta', 0], '.2f')} "
              f"R200c (15.3-15.5 and 12.7-12.9). Bound gas wants a truncation, which is why B16 fails.")
        else:
            w("  - Its outer slopes are steep but finite; B16, with one outer power law fixed by the inner "
              "parametrisation, cannot follow the drop between 1 and 2 R200c (J2.3).")
    w(f"- **Concentration cells** select the per-cell law PBcell for D1 and D2 (rule satisfied). For D0 and D3 they "
      f"select c-dependent L22best laws with rule_satisfied = no. Painted at the bin-mean concentration, the latter "
      f"give {pair('l|D0|L22best+O0|massc|Rf4')}% (D0) and {pair('l|D3|L22best+O0|massc|Rf2')}% (D3, R_fit 2). "
      f"HalfDome has no concentrations, so they bring nothing; they are not candidates of the recommendation rule.")
    w("")
    # ================================================================== J2.6
    w("### J2.6 J2c: which model captures the sensitive part, and the role of the low-mass bins")
    w("")
    w("Sensitive-part scores in the fitted region (x >= x_m, the inner edge of the first shell of each bin's 35-shell "
      "fit mask; J2.2), in % [post_j2/J2_scores.json: S_*_fit]. S_DM,low / S_DM,high: the bins 12.7-13.5 / "
      "13.5-15.5 alone.")
    w("")
    w("| definition | painting (k) | S_DM | S_mean | S_var | S_Q50 | S_Q84 | S_Q99.9 | S_DM,low | S_DM,high | "
      "DM family [%] |")
    w("|---|---|---|---|---|---|---|---|---|---|---|")
    rows = {"D0": [p["D0"], "a|D0|B16|a|Rf4", "l|D0|PB+O0|mass|Rf4", "l|D0|BPLall+O0|mass|Rf4",
                   "l|D0|PL3g+O0|mass|Rf4"],
            "D1": [p["D1"], "a|D1|B16|a|Rf1", "l|D1|PB+O0|mass|Rf1", "l|D1|BPLall+O0|mass|Rf1",
                   "l|D1|PL3+O0|mass|Rf1"],
            "D2": [p["D2"], "a|D2|B16|a|Rf2", "l|D2|PB+O0|mass|Rf2", "l|D2|BPLall+O0|mass|Rf2",
                   "l|D2|PL3+O0|mass|Rf2"],
            "D3": [p["D3"], "a|D3|B16|d|Rf4", "l|D3|PB+O0|mass|Rf2", "l|D3|BPLall+O0|mass|Rf2",
                   "l|D3|PL3+O0|mass|Rf2"]}
    for D in DEFS:
        if cp[D] not in rows[D]:
            rows[D].append(cp[D])
        for lab in rows[D]:
            w(f"| {D} | {pretty(lab)} ({R.v(SCJ, [lab, 'k'], '.0f')}) | {e(lab, 'S_DM_fit', '.1f')} | "
              f"{e(lab, 'S_mean_fit', '.1f')} | {e(lab, 'S_var_fit', '.1f')} | {e(lab, 'S_q50_fit', '.1f')} | "
              f"{e(lab, 'S_q84_fit', '.1f')} | {e(lab, 'S_q999_fit', '.1f')} | {e(lab, 'S_DM_low_fit', '.1f')} | "
              f"{e(lab, 'S_DM_high_fit', '.1f')} | {pair(lab)} |")
    w("")
    nc = {D: S["per_definition"][D]["n_candidates"] for D in DEFS}
    w("Minimal adequate model (fewest k among the candidates with both the S1 and the profile-only DM family error at "
      "or below the threshold; candidates: " + "; ".join(
          f"{D} {R.v(SUM, ['per_definition', D, 'n_candidates', 'B16_per_bin'], '.0f')} B16 + "
          f"{R.v(SUM, ['per_definition', D, 'n_candidates', 'ladder_mass_cells'], '.0f')} mass-cell laws"
          for D in DEFS) + ") [post_j2/J2_summary.json: per_definition.<D>.minimal_adequate]:")
    w("")
    w("| definition | <= 1% | <= 2% | <= 5% | <= 10% |")
    w("|---|---|---|---|---|")
    for D in DEFS:
        cells = []
        for t in ("0.01", "0.02", "0.05", "0.1"):
            m = S["per_definition"][D]["minimal_adequate"][t]
            if m is None:
                cells.append("none")
            else:
                cells.append(f"{pretty(m['label'])} (k {R.v(SUM, ['per_definition', D, 'minimal_adequate', t, 'k'], '.0f')}): "
                             f"{R.v(SUM, ['per_definition', D, 'minimal_adequate', t, 'rms_DM'], '.2f', 100)} "
                             f"({R.v(SUM, ['per_definition', D, 'minimal_adequate', t, 'prof_DM'], '.2f', 100)})")
        w(f"| {D} | " + " | ".join(cells) + " |")
    w("")
    ties = []
    for D in DEFS:
        for t, labs in S["per_definition"][D]["minimal_adequate_ties"].items():
            shown = S["per_definition"][D]["minimal_adequate"][t]["label"]
            others = [x for x in labs if x != shown]
            if others:
                ties.append(f"{D} <= {float(t) * 100:g}%: " + ", ".join(pretty(x) for x in others))
    w("- Other candidates that also pass with the same minimal k (the table shows the one with the lowest S1 error): "
      + ("; ".join(ties) if ties else "none") + ".")
    w("")
    allD = {D: [lab for lab, r in SCD.items() if r["D"] == D and r["block"] in ("J2a", "J2b")
                and r.get("dset") != "massc"] for D in DEFS}

    def ma(D, t):
        m = S["per_definition"][D]["minimal_adequate"][t]
        if m is None:
            return None
        return (f"{pretty(m['label'])} (k = {R.v(SUM, ['per_definition', D, 'minimal_adequate', t, 'k'], '.0f')}, "
                f"{R.v(SUM, ['per_definition', D, 'minimal_adequate', t, 'rms_DM'], '.2f', 100)} "
                f"({R.v(SUM, ['per_definition', D, 'minimal_adequate', t, 'prof_DM'], '.2f', 100)})%)")

    def ma_sentence(D):
        parts = []
        for t, name in (("0.01", "1%"), ("0.02", "2%"), ("0.05", "5%"), ("0.1", "10%")):
            m = ma(D, t)
            parts.append(f"<= {name}: {m}" if m else f"<= {name}: none")
        return "; ".join(parts)
    core = [(SUM, ["per_definition", D, "core", "DM_weight_share_inside_xmin"]) for D in DEFS]
    w("Findings:")
    w(f"- **The score must be read in the fitted region.** The unfitted cores (inside the first fitted shell: the "
      f"unresolved cores of every bin, and the central spike inside the softening limit in 12.7-13.5) carry only "
      f"{R.rng(core, '.1f', 100)}% of the DM weight. They still dominate S over all shells (D0 recommended: "
      f"{e(p['D0'], 'S_DM', '.1f')}% with the cores, {e(p['D0'], 'S_DM_fit', '.1f')}% without), because the forms "
      f"are extrapolated inward there and ln(model/stack) is large. J1 showed that the core shape moves every DM "
      f"statistic by at most {R.v(J1P, ['derived_for_report', 'treatment_largest_change', 'abs_change'], '.2f', 100)}%, "
      f"so these residuals are harmless.")
    lowS0 = min(allD["D0"], key=lambda lab: SCD[lab]["S_DM_fit"])
    w(f"- **D0:** {pretty(lowS0)} has the lowest S_DM,fit of all D0 candidates "
      f"({R.agg('min', [(SCJ, [lab, 'S_DM_fit']) for lab in allD['D0']], '.1f', 100)}%); B16 (d) and BPLall+O0 have "
      f"{e(p['D0'], 'S_DM_fit', '.1f')}% and {e('l|D0|BPLall+O0|mass|Rf4', 'S_DM_fit', '.1f')}%. In DM, the "
      f"recommended {pretty(p['D0'])} is best ({pair(p['D0'])}% against {pair(lowS0)}% for {pretty(lowS0)}).")
    w(f"  - The single power laws leave {e('l|D0|PL3g+O0|mass|Rf4', 'S_DM_fit', '.1f')}% (PL3g+O0) and DM errors of "
      f"{e('l|D0|PL3g+O0|mass|Rf4', 'rms_DM', '.1f')}%.")
    w(f"  - Minimal adequate (complete ladder): {ma_sentence('D0')}.")
    d1pb = [lab for lab in allD["D1"] if lab.startswith(("a|", "rc|")) or "|PB+O0|" in lab]
    w(f"- **D1:** every per-bin form (B16 with every objective and R_fit, its RC variant, PB+O0) leaves S_DM,fit <= "
      f"{R.agg('max', [(SCJ, [lab, 'S_DM_fit']) for lab in d1pb], '.1f', 100)}% inside R200c. The compact BPLall+O0 "
      f"leaves {e('l|D1|BPLall+O0|mass|Rf1', 'S_DM_fit', '.1f')}% (low-mass bins "
      f"{e('l|D1|BPLall+O0|mass|Rf1', 'S_DM_low_fit', '.1f')}%) and PL3+O0 {e('l|D1|PL3+O0|mass|Rf1', 'S_DM_fit', '.1f')}% "
      f"(low-mass bins {e('l|D1|PL3+O0|mass|Rf1', 'S_DM_low_fit', '.1f')}%).")
    w(f"  - No D1 painting reaches 1% in S1 (lowest {R.agg('min', [(SCJ, [lab, 'rms_DM']) for lab in allD['D1']], '.2f', 100)}%). "
      f"Minimal adequate: {ma_sentence('D1')}.")
    d2_1 = [lab for lab in allD["D2"] if SCD[lab]["rms_DM"] <= 0.01 and SCD[lab]["prof_DM"] <= 0.01]
    w(f"- **D2:** {len(d2_1)} candidate painting(s) reach 1% in both S1 and profile-only error "
      f"({', '.join(pretty(x) for x in d2_1) if d2_1 else 'none'}). Minimal adequate: {ma_sentence('D2')}.")
    w(f"- **D3:** every candidate leaves S_DM,fit >= "
      f"{R.agg('min', [(SCJ, [lab, 'S_DM_fit']) for lab in allD['D3']], '.1f', 100)}% (lowest S1 DM error "
      f"{R.agg('min', [(SCJ, [lab, 'rms_DM']) for lab in allD['D3']], '.2f', 100)}%). Minimal adequate: "
      f"{ma_sentence('D3')}.")
    pb = [lab for D in ("D0", "D1", "D2") for lab in rows[D][:3]]
    w(f"- **Role of the low-mass bins.** They carry "
      f"{R.v(SUM, ['per_definition', 'D0', 'core', 'DM_weight_share_low_bins'], '.0f', 100)}% (D0), "
      f"{R.v(SUM, ['per_definition', 'D1', 'core', 'DM_weight_share_low_bins'], '.0f', 100)}% (D1), "
      f"{R.v(SUM, ['per_definition', 'D2', 'core', 'DM_weight_share_low_bins'], '.0f', 100)}% (D2) and "
      f"{R.v(SUM, ['per_definition', 'D3', 'core', 'DM_weight_share_low_bins'], '.0f', 100)}% (D3) of the DM-family "
      f"weight [post_j2/J2_summary.json: per_definition.<D>.core].")
    w(f"  - (i) Their unresolved inner shape does not affect the choice. The RC fits (resolved shape + core integral) "
      f"change the DM errors of D0-D2 by <= {R.v(SUM, ['rc_max_abs_change_rms_DM_D0_D2'], '.2f', 100)} pp, and no "
      f"RC painting beats the recommended painting of its definition (J2.3). For the per-bin forms of D0-D2 in the "
      f"table, S_DM,low,fit = {R.rng([(SCJ, [lab, 'S_DM_low_fit']) for lab in pb], '.1f', 100)}% and "
      f"S_DM,high,fit = {R.rng([(SCJ, [lab, 'S_DM_high_fit']) for lab in pb], '.1f', 100)}%: the low-mass bins are "
      f"fitted about as well as the massive ones.")
    w("  - (ii) Their different, much shallower outskirts (J2.4) are what removes the single power laws and makes "
      "the per-bin forms or a mass break necessary (J2.5).")
    w(f"  - (iii) For D1, the gas inside x_h = {xh_seq()} R200c (12.7-12.9 / ... / 13.3-13.5) is resolved only as "
      f"an integral (J1). In 12.7-12.9 a D1 fit therefore constrains the gas content inside R200c, not its shape; in "
      f"12.9-13.5 it also constrains the shape between x_h and R200c.")
    w("")
    # ================================================================== J2.7
    w("### J2.7 J2d: kSZ with host-bound gas only (exact; recipe unchanged)")
    w("")
    w("J1's bound-only kSZ, computed from the bound-only stacks themselves ('exact', unlike the first-order estimate "
      "of the stacking-choices study): C14, same recipe as D0, i.e. catalogue velocities for the 1-halo term and the "
      "linear approximation for the 2-halo term. D_ell = ell(ell+1) C_ell/(2 pi) of the kSZ temperature anisotropy, "
      "in muK^2 [post/j_post.json: kSZ_Dl_uK2_C14, kSZ_D3_over_D0.C14]:")
    w("")
    w("| ell | D0 1-halo | D3 1-halo | D3/D0 1-halo | D0 2-halo | D3 2-halo | D3/D0 2-halo |")
    w("|---|---|---|---|---|---|---|")
    for x in (100, 300, 1000, 3000):
        w(f"| {x} | {R.v(J1P, ['kSZ_Dl_uK2_C14', 'D0', f'ksz1h_{x}'], '.4f')} | "
          f"{R.v(J1P, ['kSZ_Dl_uK2_C14', 'D3', f'ksz1h_{x}'], '.4f')} | "
          f"{R.v(J1P, ['kSZ_D3_over_D0', 'C14', f'ksz1h_{x}'], '.3f')} | "
          f"{R.v(J1P, ['kSZ_Dl_uK2_C14', 'D0', f'ksz2h_{x}'], '.4f')} | "
          f"{R.v(J1P, ['kSZ_Dl_uK2_C14', 'D3', f'ksz2h_{x}'], '.4f')} | "
          f"{R.v(J1P, ['kSZ_D3_over_D0', 'C14', f'ksz2h_{x}'], '.3f')} |")
    w("")
    kb = S["kSZ_D3_best"]["label"]
    w("- **The kSZ recipe stays as it is**: A2-ii gas to 4 R200c; Test I's published fits unchanged in 13.5-15.5 "
      "(pressure B12 (a), electrons B16 (d), R_fit = R_cut = 4 R200c), and the same recipe newly fitted in 12.7-13.5 "
      "('dIT', J2.4, recommendation 1).")
    w(f"  - Against the 14-bin truth it reproduces the kSZ 1-halo family to {e(KSZ_RECIPE, 'rms_kSZ')}% "
      f"(profile-only {e(KSZ_RECIPE, 'prof_kSZ')}%) and the 2-halo family to {e(KSZ_RECIPE, 'rms_kSZ2h')}% (S1).")
    w(f"  - The dI refit of all 14 bins gives {e(DI_REFIT, 'rms_kSZ')}% and {e(DI_REFIT, 'rms_kSZ2h')}%; the D0 DM "
      f"recommendation ({pretty(p['D0'])}, DM-only weights) gives {e(p['D0'], 'rms_kSZ')}% and "
      f"{e(p['D0'], 'rms_kSZ2h')}%. The latter is a DM painting, not a change of the kSZ recipe.")
    w(f"- **Bound-only gas is a different signal.** D3/D0 = {ksz_ratio_rng('ksz1h')} for the 1-halo term and "
      f"{ksz_ratio_rng('ksz2h')} for the 2-halo term over ell = 100-3000 (table).")
    w(f"- **It would also be fitted less accurately.** The best D3 painting for kSZ, {pretty(kb)}, reproduces the "
      f"bound-only kSZ 1-halo term to {e(kb, 'rms_kSZ')}% (its DM error is {e(kb, 'rms_DM', '.1f')}%); the "
      f"recommended D3 painting reproduces it to {e(p['D3'], 'rms_kSZ')}%.")
    w("")


def sections_end(R, L, S, SCD, e, pair, rec, worst_stat, which_worst, mcut, km, lm, p, cp, ksz_factor,
                 ksz_ratio_rng, xh_seq):
    w = L.append
    RPJ = R.load(RP)
    SCD_ = SCD

    def worst_of(lab):
        return max(SCD_[lab]["rms_DM"], SCD_[lab]["prof_DM"])

    def best_o0_compact(D):
        c = [lab for lab, r in SCD_.items() if r["D"] == D and r["block"] == "J2b" and r.get("dset") == "mass"
             and lab.split("|")[2].endswith("+O0") and int(r["k"]) <= 14 and r.get("rms_DM") is not None]
        return min(c, key=worst_of)
    # ================================================================== J2.8
    w("### J2.8 Recommendations")
    w("")
    w(f"1. **Calibrate every painted mass (C14).** Keep Test I's published fits in 13.5-15.5 unchanged and add, in "
      f"12.7-13.5, fits with Test I's recipe: pressure B12 (a), electrons B16 (d) with Test I's weight formula "
      f"evaluated on J1's C14 maps, R_fit = R_cut = 4 R200c, S400 masks (parameters in J2.4; painting 'dIT'). "
      f"Against the 14-bin truth this brings, relative to Test I as published (fallback below 10^13.5), the DM family "
      f"error from {e(TESTI_PUB, 'rms_DM')}% to {e(KSZ_RECIPE, 'rms_DM')}%, the kSZ 1-halo error from "
      f"{e(TESTI_PUB, 'rms_kSZ')}% to {e(KSZ_RECIPE, 'rms_kSZ')}%, the kSZ 2-halo error from "
      f"{e(TESTI_PUB, 'rms_kSZ2h')}% to {e(KSZ_RECIPE, 'rms_kSZ2h')}% and the tSZ error from "
      f"{e(TESTI_PUB, 'rms_tSZ')}% to {e(KSZ_RECIPE, 'rms_tSZ')}% (exact paintings, J2.4).")
    w(f"   - Re-fitting 13.5-15.5 as well (dI) gives {e(DI_REFIT, 'rms_DM')} / {e(DI_REFIT, 'rms_kSZ')} / "
      f"{e(DI_REFIT, 'rms_kSZ2h')} / {e(DI_REFIT, 'rms_tSZ')}% (DM / kSZ 1-halo / kSZ 2-halo / tSZ) and would change "
      f"published parameters by up to {R.v(SUM, ['dI_vs_TestI', 'max_rel_param'], '.2f', 100)}%; it is not needed.")
    ev = R.load("results_j2/evaluate_items.json")["items"]
    sub_d1 = next(it["a_substituted_bins"] for it in ev if it["label"] == p["D1"])
    w(f"2. **D1, for TNG / Konietzka within-R200 comparisons and HalfDome's SphericalChordDMProfile:** B16 per bin, "
      f"objective (L*) [(a) in {join_and(sub_d1)}], fitted to the A2-ii stacks to R_fit = 1.34 R200c, painted inside "
      f"the 3D sphere R_cut = 1 R200c.")
    w(f"   - Expected DM errors {pair(p['D1'])}% (family); largest single statistic {worst_stat(p['D1'], 'D1')}% "
      f"({which_worst(p['D1'], 'D1')}); mean DM {e(p['D1'], 'e_dm_mean')}%, median {e(p['D1'], 'e_q50')}%.")
    w(f"   - The S1 error is set by the scatter model (profile-only {e(p['D1'], 'prof_DM')}%, S1 floor "
      f"{e('ref|D1|stack', 'rms_DM')}%), so a better per-halo scatter model, not a better profile, would improve D1.")
    w(f"   - Compact alternative: {pretty(cp['D1'])} (k = {R.v(SCJ, [cp['D1'], 'k'], '.0f')}), {pair(cp['D1'])}%.")
    w(f"3. **D0, total halo-associated gas (painting partition to 4 R200c):** B16 per bin, objective (d), "
      f"R_fit = R_cut = 4 R200c: {pair(p['D0'])}% (in-sample tuned).")
    w(f"   - Without an importance-weighted objective ((d), dI or dIT): {pretty(rec('D0', 'primary_not_d'))} gives "
      f"{pair(rec('D0', 'primary_not_d'))}% and B16 (a) {pair('a|D0|B16|a|Rf4')}%. The kSZ recipe of "
      f"recommendations 1 and 7 (dIT) gives {pair('a|D0|B16|dIT|Rf4')}%.")
    o0c = best_o0_compact("D0")
    w(f"   - Compact, by the rule: {pretty(cp['D0'])} (k = {R.v(SCJ, [cp['D0'], 'k'], '.0f')}), {pair(cp['D0'])}% "
      f"in the one-point DM statistics, but {e(cp['D0'], 'rms_DMcl')}% in the DM-field C_ell and "
      f"{e(cp['D0'], 'rms_kSZ')}% in kSZ (cancelling outer residuals, J2.0). For maps use {pretty(o0c)} "
      f"(k = {R.v(SCJ, [o0c, 'k'], '.0f')}): {pair(o0c)}% in DM, {e(o0c, 'rms_DMcl')}% in the DM-field C_ell, "
      f"{e(o0c, 'rms_kSZ')}% in kSZ.")
    w(f"   - R_fit < 4 costs accuracy: B16 (d) at R_fit 3 gives {pair('a|D0|B16|d|Rf3')}%.")
    w(f"4. **D2, gas within 2 R200c:** {pretty(p['D2'])}, R_cut = 2 R200c: {pair(p['D2'])}%; compact "
      f"{pretty(cp['D2'])} (k = {R.v(SCJ, [cp['D2'], 'k'], '.0f')}) {pair(cp['D2'])}%.")
    w(f"5. **D3, bound gas:** within the allowed forms use {pretty(p['D3'])}, cut at 4 R200c: {pair(p['D3'])}% "
      f"(mean DM {e(p['D3'], 'e_dm_mean')}%, median {e(p['D3'], 'e_q50')}%).")
    w(f"   - Do not use per-bin B16 for D3 ({pair(rec('D3', 'best_B16'))}% at best).")
    w(f"   - For percent-level D3, paint the measured per-bin stacks (S1 error {e('ref|D3|stack', 'rms_DM')}%) or add "
      f"a truncation term (not tested).")
    w("   - The per-bin parameters of every recommended painting, compact law and best B16 painting are in "
      "`post_j2/J2_recommended_parameters.json` / `.csv` (B16: rho0, alpha, beta per bin; ladder laws: the law "
      "parameters and A, x_c, beta, gamma per bin at the bin-mean mass; for O0 laws the rebuilt shells equal the "
      f"painted ones to {R.agg('max', [(RP, [D, k, 'rebuild_vs_painted_max_rel']) for D in DEFS for k in ('primary', 'compact') if RPJ[D][k].get('rebuild_vs_painted_max_rel') is not None], '.0e')}; "
      "for the D0 compact O2 law, which includes the two-halo term, the exact painted shells are given as well).")
    ov0 = S["ladder"]["outer_vs_O0"]["D0|Rf4"]
    w(f"6. **Outer options (O1/O2) and concentration laws** (J2.5). The outer options are not needed to fit the "
      f"stacks: for D0, O0 has the lowest BIC for every inner law except "
      f"{join_and(ov0['inners_where_an_outer_option_beats_O0'])} (power laws in the amplitude only, which fit badly "
      f"with any outer option), and the best D0 law by BIC is {'+'.join(ov0['best_overall_by_BIC'])} (e.g. BPLall+O2: "
      f"floor chi^2/nu {km('D0', 'BPLall+O2|Rf4', 'chi2nu_floor')} against "
      f"{km('D0', 'BPLall+O0|Rf4', 'chi2nu_floor')} for BPLall+O0). In the one-point DM statistics the D0 O2 laws are "
      f"nevertheless the most accurate compact laws, through cancelling outer residuals (recommendation 3); for "
      f"maps prefer O0. They add the two-halo (field) term explicitly, which a halo-only component (D1-D3) should not "
      f"contain; bound gas has no two-halo term. HalfDome has no concentrations, so the concentration laws bring "
      f"nothing.")
    w(f"7. **kSZ:** keep the recipe, i.e. A2-ii gas to 4 R200c with Test I's published fits in 13.5-15.5 and the new "
      f"fits in 12.7-13.5 (recommendation 1): kSZ 1-halo error {e(KSZ_RECIPE, 'rms_kSZ')}%, 2-halo "
      f"{e(KSZ_RECIPE, 'rms_kSZ2h')}%.")
    w(f"   - A bound-only kSZ would be a different signal: {ksz_factor('ksz1h', 'min')} to "
      f"{ksz_factor('ksz1h', 'max')} times smaller in the 1-halo term and {ksz_factor('ksz2h', 'min')} to "
      f"{ksz_factor('ksz2h', 'max')} times smaller in the 2-halo term over ell = 100-3000 (J2.7).")
    w(f"   - It would also be fitted less accurately ({e(S['kSZ_D3_best']['label'], 'rms_kSZ')}% at best for the "
      f"1-halo term).")
    w(f"8. **Caveats.**")
    w("   - (d), dI and dIT are tuned in-sample (their weights are responses of the same observables to the same "
      "truth); (L*) mixes (L) and (a) bins.")
    w("   - The ladder laws were fitted only at the ladder radii (D0: R_fit 4; D1: 1; D2: 2; D3: 2 and 4); the "
      "minimal adequate models and the front of J2.5 are complete for these radii.")
    w(f"   - D1 below 10^13.5 rests on the simulated gas content inside x_h ({xh_seq()} R200c; J1), whose resolution "
      f"convergence needs a higher-resolution run.")
    qd1 = [(META, ["checks", "abs_importance_sum_quantile_maps_D1", f"dm_{q}"])
           for q in ("q50", "q84", "q975", "q999")]
    qd0 = [(META, ["checks", f"abs_importance_sum_quantile_maps_{D}", f"dm_{q}"])
           for D in ("D0", "D2", "D3") for q in ("q16", "q50", "q84", "q975", "q999")]
    w(f"   - The DM quantile weights use J1's Mecke maps with the whole DM = 0 atom (J1 revision of 2026-10-16): the "
      f"|importance| maps of D1's Q50-Q99.9 sum to {R.rng(qd1, '.2f')}, those of D0, D2 and D3 to "
      f"{R.rng(qd0, '.2f')} [inputs/j2_fitdata_meta.json: checks.abs_importance_sum_quantile_maps_*]. D1's Q16 lies "
      "inside the atom and is not used. The earlier half-atom maps (D1 sums "
      + R.rng([(MUPD, ["map_change", f"imp__D1__dm_{q}", "sum_old"]) for q in ("q50", "q84", "q975", "q999")], ".2f")
      + f" [{MUPD}: map_change]) had down-weighted D1's quantiles; the (d) fits of every definition were refitted "
      "with the corrected weights (J2.1, run 3). The refit changed no recommendation: the primary and compact labels "
      "of every definition and the DM errors of J2.0 are the same as in run 2; only D1's importance-weighted "
      "sensitive-part scores moved, as expected from the new weights (post_j2/J2_summary.json against "
      "results_j2_run2/J2_summary_run2.json).")
    w("   - All errors are at z = 0 with the published redshift response and refer to FLAMINGO L1_m9.")
    w("")
    # ================================================================== J2.9
    w("### J2.9 Figures (`figures_j2/`, PNG and SVG)")
    w("")
    w("- `J2_dm_error_vs_k` (main): DM family error against the number of parameters per definition, S1 (top) and "
      "profile-only (bottom); B16 per bin by objective (including dI and dIT for D0), RC variant, every mass-cell "
      "ladder law, the concentration laws (open grey squares, not candidates), G's selections, S1 floor; the "
      "recommended painting and the compact law are labelled.")
    w("- `J2_residual_profiles` (main): DM-importance-weighted ln(model/stack) of B16 (a), (d), (L*) at the boundary, "
      "the selected ladder law and the recommended painting, per definition and mass group, with the DM importance per "
      "unit ln x (grey) and, dotted, the innermost first fitted edge of the 35-shell fit masks among the panel's bins "
      "(in every panel that of its most massive bin).")
    w("- `J2_dm_pdf_selected` (main): halo-DM PDFs (dP/d ln DM, z_s = 1) of the recommended painting and the compact "
      "law (S1) against J1's truth, and CDF(model) - CDF(truth) in percentage points.")
    w("- `J2_lowmass_fallback_vs_calibrated` (main): change of every observable of the J2.4 table when the new fits "
      "replace the published fallback in 12.7-13.5 (Test I's fits unchanged above), and the errors against the truth "
      "with and without.")
    w("- Supplementary: `J2_ladder_chi2_vs_k` (parsimony ladders, Test I's D0 ladder for comparison), "
      "`J2_score_vs_error` (S_DM against the DM error), `J2_rfit_dependence` (DM error against R_fit).")
    w("")


def main():
    out = build()
    R, L = out[0], out[1]
    sections_fits(*out)
    sections_ladder(*out)
    sections_end(*out)
    text = "\n".join(L) + "\n"
    post = HERE / "post_j2"
    (post / "J2_report.txt").write_text(text)
    (post / "report_numbers_j2.json").write_text(json.dumps({"n": len(R.items), "items": R.items}, indent=1) + "\n")
    S, SCD = out[2], out[3]
    keyn = []
    for D in DEFS:
        lab = S["recommendations"][D]["primary"]["label"]
        keyn.append({"name": f"{D}: recommended painting DM family error S1 (profile-only) [%]",
                     "value": f"{100 * SCD[lab]['rms_DM']:.2f} ({100 * SCD[lab]['prof_DM']:.2f})",
                     "source": f"post_j2/J2_scores.json: {lab}.rms_DM / prof_DM", "painting": pretty(lab)})
    k = SCD[KSZ_RECIPE]
    keyn.append({"name": "kSZ recipe unchanged (Test I's fits 13.5-15.5 + new fits 12.7-13.5, C14): kSZ 1-halo / "
                         "2-halo family error S1 [%]",
                 "value": f"{100 * k['rms_kSZ']:.2f} / {100 * k['rms_kSZ2h']:.2f}",
                 "source": f"post_j2/J2_scores.json: {KSZ_RECIPE}.rms_kSZ / rms_kSZ2h",
                 "painting": "dIT: Test I's published B12 (a) / B16 (d) fits in 13.5-15.5 (fits35.json) + J2's B12 (a) "
                             "/ B16 (dI) fits in 12.7-13.5, R_fit = R_cut = 4"})
    t = SCD[TESTI_PUB]
    keyn.append({"name": "Test I as published vs 14-bin truth -> with new low-mass fits: DM / kSZ 1h / kSZ 2h / tSZ "
                         "family error S1 [%]",
                 "value": " / ".join(f"{100 * t[x]:.2f} -> {100 * k[x]:.2f}"
                                     for x in ("rms_DM", "rms_kSZ", "rms_kSZ2h", "rms_tSZ")),
                 "source": f"post_j2/J2_scores.json: {TESTI_PUB}, {KSZ_RECIPE}"})
    (post / "key_numbers_j2.json").write_text(json.dumps(keyn, indent=1) + "\n")
    recs = dict(S["recommendations"])
    recs["kSZ"] = {"rule": "user decision 2026-10-15: keep the kSZ recipe (A2-ii gas to 4 R200c) unchanged",
                   "label": KSZ_RECIPE, "description": "Test I's published fits in 13.5-15.5 (I_fitting_a2ii/results/"
                   "fits35.json E|d|4, P|a|4) unchanged; in 12.7-13.5 pressure B12 (a) and electrons B16 (d) with Test "
                   "I's weight formula (0.5 kSZ + 0.5 DM) on J1's C14 maps (results_j2/fits.json P|a|4|b, D0|dI|4|b), "
                   "R_fit = R_cut = 4 R200c",
                   "rms_kSZ": k["rms_kSZ"], "rms_kSZ2h": k["rms_kSZ2h"], "prof_kSZ": k["prof_kSZ"],
                   "rms_DM": k["rms_DM"], "prof_DM": k["prof_DM"], "rms_tSZ": k["rms_tSZ"]}
    (post / "recommendations_j2.json").write_text(json.dumps(recs, indent=1) + "\n")
    missing = [it["text"] for it in R.items if it["in_report"] and it["text"] not in text]
    print(f"report: {len(text)} characters, {len(R.items)} registered numbers; not found in the text: {len(missing)}")


if __name__ == "__main__":
    main()
