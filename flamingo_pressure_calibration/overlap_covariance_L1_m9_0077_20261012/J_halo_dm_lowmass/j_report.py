"""Report text of Test J1 generated from the fetched results (no hand transcription).

Every number in the report is produced by the registry below: it reads one or more fields of a results file
(results/, post/, inputs/, verification.json, CLUSTER_RUN.json), applies one declared operation and formats the value.
The registry is written to post/report_numbers.json (text, value, operation, sources, format); verify_j.py gate G13
re-resolves every entry from its sources and checks that the formatted text appears in the report.
Operations: value  (a - minus) * scale;  ratio  (a / b - minus) * scale;  min / max / sum over the sources, then
(agg - minus) * scale;  len  the length of a list field.
Outputs: post/J1_report.txt (Markdown text with LaTeX), post/report_numbers.json, post/key_numbers.json.
Usage: python -B j_report.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PREP, TS, SS, POST = "results/prepare.json", "results/truth_summary.json", "results/sens_summary.json", "post/j_post.json"
PAR, VER, ST, DONE = "results/parity.json", "verification.json", "results/stacks.npz", "results/test_J_done.json"
MAN, BAS, CR, UP, META = "results/run_manifest.json", "results/basis.json", "CLUSTER_RUN.json", "post/upload_J.json", \
    "inputs/j0_meta.json"
LCM = "post/lightcone_mass_z.json"
PROV = "post/provenance_checks.json"
R234 = "../../radii234_contamination_L1_m9_0077_20261008/products/study_summary.json"
CSUP = "../../tsz_fit_truncation_L1_m9_0077_20261009/fit_inputs/common_support.json"
MUPD = "post/mecke_update.json"
MDONE, MMAN = "results_mecke/test_J_done.json", "results_mecke/run_manifest.json"
LOW = ("12.7-12.9", "12.9-13.1", "13.1-13.3", "13.3-13.5")
CAL = ("12.7-12.9", "12.9-13.1", "13.1-13.3", "13.3-13.5", "13.5-13.7", "13.7-13.9", "13.9-14.1", "14.1-14.3",
       "14.3-14.5", "14.5-14.7", "14.7-14.9", "14.9-15.1", "15.1-15.3", "15.3-15.5")
HIGH = ("14.5-14.7", "14.7-14.9", "14.9-15.1", "15.1-15.3", "15.3-15.5")
DEFS = ("D0", "D1", "D2", "D3")


class Reg:
    def __init__(self):
        self.items, self._cache = [], {}
        self.context = "report"

    def load(self, f):
        if f not in self._cache:
            p = HERE / f
            self._cache[f] = dict(np.load(p, allow_pickle=False)) if f.endswith(".npz") else json.loads(p.read_text())
        return self._cache[f]

    def raw(self, f, keys):
        o = self.load(f)
        for k in keys:
            o = o[k]
        return o

    def get(self, f, keys):
        return float(self.raw(f, keys))

    def _rec(self, op, srcs, fmt, scale, minus, value):
        txt = format(value, fmt)
        if txt.startswith("-0") and set(txt[1:]) <= set("0.,"):
            txt = txt[1:]
        self.items.append({"text": txt, "value": value, "op": op, "sources": [[f, list(k)] for f, k in srcs],
                           "format": fmt, "scale": scale, "minus": minus, "in_report": self.context == "report"})
        return txt

    def v(self, f, keys, fmt, scale=1.0, minus=0.0):
        return self._rec("value", [(f, keys)], fmt, scale, minus, (self.get(f, keys) - minus) * scale)

    def r(self, fa, ka, fb, kb, fmt, scale=1.0, minus=0.0):
        return self._rec("ratio", [(fa, ka), (fb, kb)], fmt, scale, minus,
                         (self.get(fa, ka) / self.get(fb, kb) - minus) * scale)

    def agg(self, op, srcs, fmt, scale=1.0, minus=0.0):
        vals = [self.get(f, k) for f, k in srcs]
        a = {"min": min, "max": max, "sum": sum}[op](vals)
        return self._rec(op, srcs, fmt, scale, minus, (a - minus) * scale)

    def n(self, f, keys):
        return self._rec("len", [(f, keys)], "d", 1.0, 0.0, len(self.raw(f, keys)))

    def rng(self, srcs, fmt, scale=1.0, minus=0.0, sep="-"):
        f1, f2 = (fmt, fmt) if isinstance(fmt, str) else fmt
        return f"{self.agg('min', srcs, f1, scale, minus)}{sep}{self.agg('max', srcs, f2, scale, minus)}"

    def seq(self, f, keyfun, labels, fmt, scale=1.0, minus=0.0, sep=" / "):
        return sep.join(self.v(f, keyfun(lab), fmt, scale, minus) for lab in labels)


def build(R: Reg):
    L = []
    w = L.append
    J1a = lambda lab, *k: ("J1a_by_bin", lab) + k                     # noqa: E731
    pdef = lambda cfg_d, s: ("definitions", cfg_d, s)                   # noqa: E731
    rat = lambda d, s: ("definitions_ratio_to_D0_C14", d, s)            # noqa: E731
    mvf = lambda d, s: ("measured_over_fallback_C14_over_C10", d, s)     # noqa: E731
    pct = dict(scale=100.0, minus=1.0)
    bandkey = lambda d, name, band: ("bands", f"{d}__{name}", band)      # noqa: E731
    radk = lambda d, name, k: ("radii", d, name, k)                      # noqa: E731
    lowshare = lambda d, name: ("shares", f"{d}__{name}", "low_mass_12.7-13.5")   # noqa: E731
    lc = ("lightcone",)
    G = R.load(VER)["gates"]
    n_gates = len(G)
    gate_names = sorted(G, key=lambda k: int(k.split("_")[0][1:]))

    w("## Halo component of DM and low-mass calibration: sensitivity (Test J1)")
    w("")
    w("*FLAMINGO L1_m9, z = 0 (snapshot 0077), A2-ii painting partition with recently AGN-heated gas removed, "
      "geometric shell estimator, inclusion-weighted mean, 0.2-dex M200c bins, Test H's fine grid (67 X200 shells to "
      "4 R200c + 8 extension shells). Every number below is generated by `j_report.py` from the files named in "
      "brackets; `post/report_numbers.json` lists each number with its source file, field and operation "
      "(gate G13 re-resolves all of them).*")
    w("- Paths are relative to `J_halo_dm_lowmass/`. `results/` holds the outputs fetched from STRW maas, "
      "`post/j_post.json` exact local functions of them, `inputs/j0_meta.json` the J0 export checks.")
    w("- Estimates are labelled ESTIMATE. Configurations: C14 = all 14 painted bins 12.7-15.5 calibrated (new); "
      "C10 = only 13.5-15.388 calibrated, published B16/B12 fallback below (the configuration of Tests F/H); "
      "C0 = published profiles everywhere.")
    w("")
    # ------------------------------------------------------------------------------------------------ J.0
    w("### J.0 Answers")
    w("")
    w("**Q1: why was DM from haloes below 10^13.5 not calibrated?**")
    w("")
    LOW5 = ("12.5-12.7",) + LOW
    q1 = lambda lab, k: ("q1", "per_bin", lab, k)                              # noqa: E731
    w("*Cause: the per-halo strict resolution rule, not missing haloes.*")
    w("- The engine's calibrated support comes from `kernels/engine.py:315 support_definition`. It reads "
      "`common_support` from the 2026-10-09 truncation study (`tsz_fit_truncation_L1_m9_0077_20261009/fit_inputs/"
      "profile_shapes.npz`, the same array as `common_support.json` there), which fitted the radii234 profile set "
      "of 2026-10-08. Tests F and H kept that support (`F_observables/f_common.py:152 CAL_BINS`).")
    w(f"- radii234 already held {R.seq(R234, lambda k: ('bins', k, 'halos'), range(5), '.0f')} haloes in the five "
      f"bins 12.5-12.7 ... 13.3-13.5 ({R.v(PROV, ('q1', 'totals', 'radii234_haloes'), ',.0f')} in all; "
      f"{R.v(PROV, ('q1', 'totals', 'radii234_haloes_in_current_sample'), ',.0f')} of them are in the current sample, "
      "same SOAP row and mass bin). The overlap particle pass added only "
      f"{R.seq(PROV, lambda lab: q1(lab, 'added_by_overlap_particle_pass'), LOW5, '.0f')} per bin, giving the "
      f"{R.seq(PREP, lambda lab: J1a(lab, 'n_haloes'), LOW5, '.0f')} haloes used here [{R234}: bins[0..4].halos; "
      f"{PROV}: q1; {PREP}: J1a_by_bin.*.n_haloes].")
    w("- The truncation study's common-support rule excluded them (`tsz_fit_truncation_L1_m9_0077_20261009/fit_inputs/"
      f"prepare_fit_inputs.py:{R.v(PROV, ('q1', 'common_support_rule_line', 0), '.0f')}`, `common = strict & "
      "np.all(stable, axis=1) & (masks[:,0].sum(axis=1)>=6)`). A bin is calibrated only if (i) at least 6 shells inside "
      "2 R200c are strict-resolved, i.e. resolved by every halo of the bin (radii234's per-halo rule: x >= "
      "max(0.04, 4 epsilon/R200c), >= 20 gas particles and x >= 2 h_SPH/R200c in every halo), (ii) the B12 fits at 1, "
      f"2, 3 and 4 R200c are all stable, and (iii) at least {R.v(CSUP, ('minimum_1R_shells',), '.0f')} strict-resolved "
      "shells lie inside R200c:")
    w(f"  - 12.5-13.3: {R.agg('max', [(CSUP, ('fit_flags', k, 'strict_1R_shells')) for k in range(4)], '.0f')} "
      "strict-resolved shells inside R200c in each of the four bins, and "
      f"{R.seq(R234, lambda k: ('bins', k, 'strict_shell_count_to_2R'), range(4), '.0f')} inside 2 R200c "
      f"[{CSUP}: fit_flags[0..3].strict_1R_shells; {R234}: bins[0..3].strict_shell_count_to_2R, fit_status "
      "'diagnostic only; strict 2R support absent'];")
    w(f"  - 13.3-13.5: {R.v(R234, ('bins', 4, 'strict_shell_count_to_2R'), '.0f')} strict-resolved shells inside "
      f"2 R200c but only {R.v(CSUP, ('fit_flags', 4, 'strict_1R_shells'), '.0f')} inside R200c, and an unstable 1R fit "
      f"[{CSUP}: fit_flags[4], stable_all_radii[4]].")
    w("- So lightcone haloes below 10^13.5 were painted with the published B16 (electrons) / B12 (pressure) profiles "
      "although their FLAMINGO stacks existed: the low-mass haloes are resolved only as stacks, not halo by halo "
      "(J.3).")
    w("")
    w("*Fix.* Test J calibrates every painted mass, bins 12.7-12.9 to 15.3-15.5 (14 bins, configuration C14), "
      "for pressure and electrons:")
    w(f"- engine bin 12.5-12.7 holds {R.v(PREP, lc + ('engine_bin_12.5-12.7_haloes',), '.0f')} painted lightcone "
      f"haloes; the smallest painted halo has log10(M200c/Msun) = {R.v(PREP, lc + ('min_logm200c_painted',), '.3f')} "
      f"[{PREP}: lightcone], but only in the lightcone redshift shells at z >= "
      f"{R.v(LCM, ('columns_below_13_z_le_1', 0, 'z_min_occupied_shell_lower_edge'), '.2f')} [{LCM}];")
    lcm = lambda *k: ("bin_12.7-12.9",) + k                                # noqa: E731
    col = lambda i, k: ("columns_below_13_z_le_1", i, k)                    # noqa: E731
    w(f"- at z <= 1 (the DM range, sources at z_s = 1) the smallest painted halo has log10 M = "
      f"{R.v(LCM, ('min_logm200c_painted_z_le_1',), '.3f')}, and the catalogue is incomplete below about 10^12.90: "
      f"the z <= 1 haloes per {R.v(LCM, col(4, 'width_dex'), '.4f')}-dex mass column centred at "
      f"10^{R.v(LCM, col(4, 'logm200c_centre'), '.3f')} / 10^{R.v(LCM, col(5, 'logm200c_centre'), '.3f')} / "
      f"10^{R.v(LCM, col(6, 'logm200c_centre'), '.3f')} / 10^{R.v(LCM, col(7, 'logm200c_centre'), '.3f')} number "
      f"{R.v(LCM, col(4, 'count_z_le_1'), ',.0f')} / {R.v(LCM, col(5, 'count_z_le_1'), ',.0f')} / "
      f"{R.v(LCM, col(6, 'count_z_le_1'), ',.0f')} / {R.v(LCM, col(7, 'count_z_le_1'), ',.0f')}, i.e. "
      f"{R.v(LCM, col(4, 'completeness_z_le_1_ESTIMATE'), '.2f', 100)} / "
      f"{R.v(LCM, col(5, 'completeness_z_le_1_ESTIMATE'), '.1f', 100)} / "
      f"{R.v(LCM, col(6, 'completeness_z_le_1_ESTIMATE'), '.0f', 100)} / "
      f"{R.v(LCM, col(7, 'completeness_z_le_1_ESTIMATE'), '.1f', 100)}% of a power law fitted to the z <= 1 columns "
      "at 10^12.9-10^13.1 (ESTIMATE). The four lower columns, centred at "
      f"10^{R.v(LCM, col(0, 'logm200c_centre'), '.3f')} to 10^{R.v(LCM, col(3, 'logm200c_centre'), '.3f')}, are "
      f"painted only beyond z = {R.v(LCM, col(0, 'z_min_occupied_shell_lower_edge'), '.2f')} to "
      f"{R.v(LCM, col(3, 'z_min_occupied_shell_lower_edge'), '.2f')}, respectively "
      f"[{LCM}: columns_below_13_z_le_1, completeness_ESTIMATE];")
    w(f"- so at z <= 1 bin 12.7-12.9 holds only {R.v(LCM, lcm('haloes_z_le_1'), ',.0f')} haloes "
      f"({R.v(LCM, lcm('haloes_z_le_1_over_bin_12.9-13.1'), '.3f')} times the {R.v(PREP, lc + ('haloes_per_engine_bin_z_le_1', 2), ',.0f')} "
      f"of 12.9-13.1), with mean log10 M = {R.v(LCM, lcm('mean_logm200c_z_le_1'), '.3f')} (all z: "
      f"{R.v(LCM, lcm('mean_logm200c_all_z'), '.3f')}); this is why it carries only "
      f"{R.v(POST, ('per_bin_C14', 'D0', 'dm_mean_share', 0), '.2f', 100)}% of D0's mean DM in C14 "
      f"[{POST}: per_bin_C14.D0.dm_mean_share]. The lightcone histogram and its per-bin counts are the ones the maas "
      f"run used (SHA256 and counts equal [{LCM}: parity_with_results_prepare_lightcone]);")
    w(f"- only the {R.v(PREP, lc + ('haloes_above_15.388',), '.0f')} lightcone haloes above 10^15.388 keep the "
      f"fallback; they carry {R.v(POST, ('per_bin_C14', 'D0', 'fallback_dm_mean_share'), '.3f', 100)}% of D0's mean DM "
      f"[{POST}: per_bin_C14.D0.fallback_dm_mean_share];")
    w("- inside 12.7-13.5 the per-halo strict resolution mask is replaced by the stack-level rule of J.2.")
    w("")
    w("*What the fallback got wrong.* Measured A2-ii divided by published at each FLAMINGO halo's own mass, in "
      "12.7-12.9 / 12.9-13.1 / 13.1-13.3 / 13.3-13.5 [" + PREP + ": J1a_by_bin]:")
    w(f"- N_e(<R200c): {R.seq(PREP, lambda b: J1a(b, 'Ne_lt1R_ratio_A2ii_over_B16'), LOW, '.2f')};")
    w(f"- N_e(<2 R200c): {R.seq(PREP, lambda b: J1a(b, 'Ne_lt2R_ratio_A2ii_over_B16'), LOW, '.2f')};")
    w(f"- N_e(<4 R200c): {R.seq(PREP, lambda b: J1a(b, 'Ne_lt4R_ratio_A2ii_over_B16'), LOW, '.2f')};")
    w(f"- pressure, Y(<R200c)/B12: {R.seq(PREP, lambda b: J1a(b, 'Y_lt1R_ratio_A2ii_over_B12'), LOW, '.2f')}.")
    w("")
    w(f"*Effect on D0 (A2-ii to 4R) of replacing the fallback (C14/C10 - 1)* [{POST}: "
      "measured_over_fallback_C14_over_C10.D0]:")
    w(f"- mean DM {R.v(POST, mvf('D0', 'dm_mean'), '+.1f', **pct)}%;")
    w(f"- variance {R.v(POST, mvf('D0', 'dm_var'), '+.1f', **pct)}%;")
    w(f"- Q16 {R.v(POST, mvf('D0', 'q16'), '+.1f', **pct)}%, Q50 {R.v(POST, mvf('D0', 'q50'), '+.1f', **pct)}%, "
      f"Q84 {R.v(POST, mvf('D0', 'q84'), '+.1f', **pct)}%, Q97.5 {R.v(POST, mvf('D0', 'q975'), '+.1f', **pct)}%, "
      f"Q99.9 {R.v(POST, mvf('D0', 'q999'), '+.1f', **pct)}%;")
    w(f"- DM-field C_ell {R.v(POST, mvf('D0', 'dmcltot_1000'), '+.1f', **pct)}% at ell = 1000 and "
      f"{R.v(POST, mvf('D0', 'dmcltot_3000'), '+.1f', **pct)}% at ell = 3000.")
    w("")
    w(f"*Effect inside a 1 R200c sphere (D1, C14/C10 - 1):* mean DM {R.v(POST, mvf('D1', 'dm_mean'), '+.1f', **pct)}%, "
      f"variance {R.v(POST, mvf('D1', 'dm_var'), '+.1f', **pct)}%, Q50 {R.v(POST, mvf('D1', 'q50'), '+.1f', **pct)}%, "
      f"Q84 {R.v(POST, mvf('D1', 'q84'), '+.1f', **pct)}%.")
    w("")
    w(f"*Weight of the newly calibrated bins (C14, D0)* [{SS}: shares.D0__*.low_mass_12.7-13.5]: 12.7-13.5 carries "
      f"{R.v(SS, lowshare('D0', 'dm_mean@scalar'), '.1f', 100)}% of the mean DM, "
      f"{R.v(SS, lowshare('D0', 'dm_var@scalar'), '.1f', 100)}% of the variance, "
      f"{R.v(SS, lowshare('D0', 'dm_q50@scalar'), '.0f', 100)}% of Q50 and "
      f"{R.v(SS, lowshare('D0', 'dm_q16@scalar'), '.0f', 100)}% of Q16 (importance shares, J.2).")
    w("")
    w("**Q2: why does painting to 5R add so much, and what is the halo component?**")
    w("")
    w("*The 4-5R shell is large and its gas is mostly not bound to the halo* [" + PREP + ": J1a_by_bin.*.bands, "
      "share_of_Ne_lt4R_inside_R200c]:")
    w(f"- the 4-5R shell has (5^3 - 4^3)/4^3 = {R.v(POST, ('derived_for_report', 'shell_volume_ratio_4to5R_over_0to4R'), '.3f')} "
      "times the volume of the 0-4R sphere;")
    b364 = lambda lab: J1a(lab, "bands", "3.6342-4R", "A2ii_mean_over_cosmic")     # noqa: E731
    w(f"- in the outermost fine shell, 3.63-4R, the A2-ii electron density is "
      f"{R.rng([(PREP, b364(lab)) for lab in LOW], '.1f')} times the cosmic mean in 12.7-13.5, "
      f"{R.v(PREP, b364('13.5-13.7'), '.1f')} times at 13.5-13.7 and "
      f"{R.rng([(PREP, b364(lab)) for lab in HIGH], '.1f')} times above 10^14.5 (cosmic mean "
      f"E-bar = {R.v(ST, ('E_cosmic',), '.4f')}, J.2);")
    w(f"- host-bound gas there amounts to {R.rng([(PREP, J1a(lab, 'bands', '3.6342-4R', 'hb_over_A2ii')) for lab in CAL], '.1f', 100)}% "
      "of the A2-ii electrons, so at most that share of them is bound (HBT) to the halo or its own satellites "
      "(non-nested selections: host-bound gas is not a subset of A2-ii); the bound gas there is at "
      f"{R.rng([(PREP, J1a(lab, 'bands', '3.6342-4R', 'hb_mean_over_cosmic')) for lab in CAL], '.2f')} times the "
      "cosmic mean;")
    w(f"- only {R.v(PREP, J1a('12.7-12.9', 'share_of_Ne_lt4R_inside_R200c', 'A2ii'), '.0f', 100)}% (12.7-12.9) to "
      f"{R.agg('max', [(PREP, J1a(lab, 'share_of_Ne_lt4R_inside_R200c', 'A2ii')) for lab in CAL], '.0f', 100)}% "
      "(15.1-15.3, the maximum) of the A2-ii electrons inside 4R lie inside R200c; above 10^14.5 the share is "
      f"{R.rng([(PREP, J1a(lab, 'share_of_Ne_lt4R_inside_R200c', 'A2ii')) for lab in HIGH], '.0f', 100)}%.")
    w("")
    w(f"*So D0 is a halo-plus-local-field partition, not a halo component* [{SS}: bands, radii.D0]:")
    w(f"- {R.v(SS, bandkey('D0', 'dm_mean@scalar', '2-4R'), '.1f', 100)}% of D0's mean DM comes from 2-4R; its "
      f"R50 is {R.v(SS, radk('D0', 'dm_mean@scalar', 'R50'), '.2f')} R200c and its R90 "
      f"{R.v(SS, radk('D0', 'dm_mean@scalar', 'R90'), '.2f')} R200c;")
    w(f"- cutting every calibrated halo at 1 R200c loses {R.v(SS, radk('D0', 'dm_mean@scalar', 'loss_at_1R'), '.0f', 100)}% "
      "of it. The 'DM not converged at 4R' result of Test H describes this field-inclusive target.")
    w("")
    w(f"*The three halo-component definitions (C14)* [{POST}: definitions, definitions_ratio_to_D0_C14; {TS}: crossings]:")
    w("- D1, A2-ii gas in the 3D sphere r <= R200c:")
    w(f"  - mean {R.v(POST, pdef('C14|D1', 'dm_mean'), '.1f')} pc cm^-3 = {R.v(POST, rat('D1', 'dm_mean'), '.3f')} x D0;")
    w(f"  - Q50 {R.v(POST, pdef('C14|D1', 'q50'), '.1f')} pc cm^-3 = {R.v(POST, rat('D1', 'q50'), '.3f')} x D0;")
    w(f"  - variance {R.v(POST, rat('D1', 'dm_var'), '.3f')} x D0;")
    w(f"  - {R.v(TS, ('crossings', '1R', 'p_no_crossing'), '.1f', 100)}% of sightlines to z_s = 1 cross no R200c "
      f"sphere (lambda = {R.v(TS, ('crossings', '1R', 'lambda_total'), '.3f')} crossings per sightline);")
    aeq = lambda k: ("a2ii_equals_all_gas_inside_R200c", k)                  # noqa: E731
    w(f"  - inside R200c the A2-ii partition equals all gas for every one of the {R.v(PROV, aeq('n_haloes'), ',.0f')} "
      f"haloes (largest relative difference {R.v(PROV, aeq('max_rel_difference'), '.1e')} over "
      f"{R.v(PROV, aeq('n_halo_shells'), ',.0f')} halo-shells, identical zero pattern) [{PROV}: "
      "a2ii_equals_all_gas_inside_R200c], so D1 is the all-gas component within R200c, like Konietzka's catalogue.")
    w(f"- D2, A2-ii gas in r <= 2 R200c: mean {R.v(POST, rat('D2', 'dm_mean'), '.3f')} x D0, Q50 "
      f"{R.v(POST, rat('D2', 'q50'), '.3f')} x, variance {R.v(POST, rat('D2', 'dm_var'), '.3f')} x.")
    w("- D3, host-bound gas to 4 R200c:")
    w(f"  - mean {R.v(POST, pdef('C14|D3', 'dm_mean'), '.1f')} pc cm^-3 = {R.v(POST, rat('D3', 'dm_mean'), '.3f')} x D0, "
      f"Q50 {R.v(POST, rat('D3', 'q50'), '.3f')} x, variance {R.v(POST, rat('D3', 'dm_var'), '.3f')} x;")
    w(f"  - R90 of its mean DM is {R.v(SS, radk('D3', 'dm_mean@scalar', 'R90'), '.2f')} R200c, and only "
      f"{R.v(SS, bandkey('D3', 'dm_mean@scalar', '2-4R'), '.1f', 100)}% of its mean comes from 2-4R.")
    w("")
    w(f"*Bound gas beyond the boundaries (calibrated bins 12.7-15.5)* [{PREP}: J1a_by_bin.*.host_bound_*]:")
    w(f"- {R.rng([(PREP, J1a(lab, 'host_bound_fraction_of_hb_lt4R_in', '2-4R')) for lab in CAL], '.0f', 100)}% of the "
      "host-bound electrons inside 4R lie at 2-4R (this is the bound gas that D2 drops);")
    w(f"- beyond 4R: {R.rng([(PREP, J1a(lab, 'host_bound_beyond_4R_ESTIMATE', 'fraction_of_hb_lt4R')) for lab in CAL], ('.2f', '.1f'), 100)}% "
      "of the bound electrons inside 4R (ESTIMATE: power law fitted at 3-4R, extrapolated to infinity).")
    w("")
    w("**Q3: kSZ with bound gas only (exact; kSZ recipe unchanged)**")
    w("")
    kz = lambda cfg, k: ("kSZ_D3_over_D0", cfg, k)                         # noqa: E731
    ells = (100, 300, 1000, 3000)
    w(f"Ratio host-bound (D3) / A2-ii to 4R (D0) in C14 at ell = 100 / 300 / 1000 / 3000 [{POST}: kSZ_D3_over_D0.C14]:")
    for term, name in (("ksz1h", "1-halo"), ("ksz2h", "2-halo (approximate)"), ("ksztot", "total")):
        w(f"- {name}: " + " / ".join(R.v(POST, kz("C14", f"{term}_{e}"), ".3f") for e in ells) + ";")
    w("")
    fo = lambda k, f: ("first_order_vs_exact_C10_hostbound_over_allgas", k, f)   # noqa: E731
    w(f"Exact versus the earlier first-order estimate, host-bound / all gas in C10 (the estimate's configuration) "
      f"[{POST}: first_order_vs_exact_C10_hostbound_over_allgas]:")
    w("")
    w("| ell | first-order 1-halo change | exact 1-halo change |")
    w("|---|---|---|")
    for e in (100, 1000, 3000):
        w(f"| {e} | {R.v(POST, fo(f'kSZ_1h_ell{e}', 'first_order_pct'), '+.1f')}% | "
          f"{R.v(POST, fo(f'kSZ_1h_ell{e}', 'exact_pct'), '+.1f')}% |")
    w("")
    w(f"First order overstates the large changes of this quadratic observable. Mean DM agrees "
      f"({R.v(POST, fo('DM_mean', 'first_order_pct'), '+.2f')}% vs {R.v(POST, fo('DM_mean', 'exact_pct'), '+.2f')}%) "
      "because it is linear in the profile.")
    w("")
    # ------------------------------------------------------------------------------------------------ J.1
    w("### J.1 What was run, where, and checks")
    w("")
    w("**Local steps.**")
    w(f"- J0 (`j_inputs.py`) exported, for all {R.v(PREP, ('n_haloes',), ',.0f')} haloes and all 15 bins 12.5-15.5, "
      "the cache arrays E (A2-ii, host_bound, all gas), P (A2-ii), counts and smoothing lengths, and the exact owned "
      "particle counts from the partition and owner blocks (`inputs/j0_profiles_x200.npz`).")
    w(f"- P_a2ii, E_a2ii and E_all are bitwise equal to Test H's export [{META}: checks].")
    w("- Local post-processing (`j_post.py`, `j_local_mean.py`, `j_lightcone_mass.py`, `j_mecke_update.py`, "
      "`j_provenance.py`, `make_figures.py`, `j_report.py`, `verify_j.py`, `j_sums.py`) reads only the fetched files; "
      "`j_lightcone_mass.py` also reads the frozen lightcone histogram, the same file the maas run read (gate G10), and "
      "`j_provenance.py` the radii234 and 2026-10-09 truncation-study records quoted in Q1 and the J0 export.")
    w("")
    stg = lambda s: ("invocations", 0, "stages", s, "wall_seconds")          # noqa: E731
    w(f"**maas run 1** [{CR}: maas_runs[0]; {DONE}; {MAN}]:")
    w(f"- host maas.strw.leidenuniv.nl, launched 2026-10-15 13:51 EEST through `cluster_tests.py run J`;")
    w(f"- {R.v(DONE, ('workers',), '.0f')} workers, nice {R.v(DONE, ('nice',), '.0f')}, ionice "
      f"best-effort 7, rss-stop {R.v(DONE, ('rss_stop_GiB',), '.0f')} GiB;")
    w(f"- exit code {R.v(DONE, ('returncode',), '.0f')}, wall {R.v(DONE, ('wall_seconds',), ',.0f')} s;")
    w(f"- peak tree RSS {R.v(DONE, ('peak_tree_rss_GiB',), '.2f')} GiB; minimum machine memory available "
      f"{R.v(DONE, ('min_machine_available_GiB',), '.1f')} GiB.")
    w("")
    w("| Stage | Wall (s) | Notes |")
    w("|---|---|---|")
    w(f"| prepare | {R.v(MAN, stg('prepare'), '.1f')} | J1a stacks and the low-mass rule |")
    w(f"| basis | {R.v(MAN, stg('basis'), '.1f')} | fine bases JA (A2-ii) and JB (host-bound), 14 bins; parent "
      f"{R.v(BAS, ('peak_rss_parent_GiB',), '.2f')} GiB, largest child {R.v(BAS, ('peak_rss_largest_child_GiB',), '.2f')} GiB |")
    w(f"| fallback | {R.v(MAN, stg('fallback'), '.1f')} | published pieces of the uncalibrated haloes |")
    w(f"| truth | {R.v(MAN, stg('truth'), '.1f')} | moments, DM PDFs, Mecke quantile Jacobians |")
    w(f"| sens | {R.v(MAN, stg('sens'), '.1f')} | importance maps, truncation curves, shares |")
    w("")
    w("**Upload.**")
    w("- The full `cluster_tests.py mirror` failed after 25.6 min when outbound ssh from this machine dropped "
      "(blocked 11:16-13:45 EEST) [" + CR + ": mirror_attempts].")
    w(f"- The J files were then uploaded with `j_upload.py`, a targeted SHA256-verified upload as Tests I and K did: "
      f"{R.n(UP, ('uploaded',))} files uploaded (10.36 MB), {R.n(UP, ('identical',))} already identical, "
      f"{R.v(UP, ('files_checked',), '.0f')} checked [{UP}].")
    w("- The compute modules on maas equal the local ones (`post/remote_code_check.json`, gate G11).")
    w("")
    w("**Scratch.** The fine bases lived in node-local `/tmp/nagainis_halfdome_testJ` (2.20 GB), deleted after the "
      "final fetch (2026-10-15 16:04:55 EEST) [" + CR + ": maas_runs[0].scratch_deleted].")
    w("")
    MR = R.load(CR)["j1_mecke_runs"][0]
    mst = lambda s_: ("invocations", 0, "stages", s_, "wall_seconds")      # noqa: E731
    w(f"**maas run 2 (2026-10-16): Mecke quantile maps with the whole DM = 0 atom** [{CR}: j1_mecke_runs[0]; "
      f"{MDONE}; {MMAN}]:")
    w(f"- launched {MR['launched']} through `cluster_tests.py run J --workers 8 --rss-stop 24 --driver-args=\"--j2 "
      "--results J_halo_dm_lowmass/results_mecke --stages j1base,mecke --scratch /tmp/nagainis_halfdome_testJ\"` "
      "(run_test_j.py hands '--j2' to run_test_j2.py; its stage 'mecke' runs `j_mecke_atom.py`); "
      f"{R.v(MDONE, ('workers',), '.0f')} workers, nice {R.v(MDONE, ('nice',), '.0f')}, ionice best-effort 7, "
      f"rss-stop {R.v(MDONE, ('rss_stop_GiB',), '.0f')} GiB;")
    w(f"- exit code {R.v(MDONE, ('returncode',), '.0f')}, wall {R.v(MDONE, ('wall_seconds',), ',.0f')} s (j1base "
      f"{R.v(MMAN, mst('j1base'), '.0f')} s: J1's prepare, basis and fallback rebuilt into results_mecke/j1base; "
      f"mecke {R.v(MMAN, mst('mecke'), '.0f')} s); peak tree RSS {R.v(MDONE, ('peak_tree_rss_GiB',), '.2f')} GiB; "
      f"minimum machine memory available {R.v(MDONE, ('min_machine_available_GiB',), '.1f')} GiB;")
    w("- per definition D0-D3 it re-ran the reference DM PDFs C14|D|T1 (truth and truth mean; equal to J1's, largest "
      f"relative quantile difference {R.v(VER, ('gates', 'G14_mecke_whole_atom_revision', 'reference_quantiles_max_rel'), '.1e')}) "
      "and computed the Mecke Jacobians twice on J1's entries: with J1's formula (equal to J1's results/truth.npz, "
      f"largest difference {R.v(VER, ('gates', 'G14_mecke_whole_atom_revision', 'half_atom_jacobians_max_rel'), '.1e')} "
      "of the largest element) and with the whole atom (J.2) [results_mecke/mecke_parity.json];")
    w("- `j_mecke_update.py` (local) first re-derived the stored quantile entries from J1's Jacobians byte for byte, "
      "then rebuilt them from the whole-atom Jacobians: the quantile maps, sum rules, tolerances, peaks, mass shares and "
      "bands in results/sens.npz and sens_summary.json and the three J1d tables (originals in "
      f"`post/superseded_halfatom_20261016/`; old and new SHA256 in {MUPD}; the five lines of results/SHA256SUMS updated). "
      "Moment maps, truncation curves and radii are unchanged;")
    w(f"- code and inputs uploaded with `j2_cluster.py upload` and checked equal on maas before the launch "
      f"(`post/remote_check_mecke.json`); the scratch ({R.v(CR, ('j1_mecke_runs', 0, 'scratch_bytes_deleted'), '.2f', 1e-9)} GB) "
      f"was deleted after the final fetch [{CR}: j1_mecke_runs[0].scratch_deleted].")
    w("")
    J2M = "results_j2/run_manifest.json"
    MCD, MCM = "results_mecke_checks/test_J_done.json", "results_mecke_checks/run_manifest.json"
    mcs = lambda s_: ("invocations", 0, "stages", s_, "wall_seconds")       # noqa: E731
    w(f"**maas runs 3 and 4 (2026-10-16): checks of the revised maps** [{CR}: j1_mecke_runs[1]; {MCD}; {MCM}], stage "
      "'mecke_checks' (`j_mecke_atom.py --checks`):")
    w("- its first version ran after J2's eval in the wrapper job of Test J2's run 3 and was stopped after "
      f"{R.v(J2M, ('invocations', 2, 'stages', 'mecke_checks', 'wall_seconds'), '.0f')} s [{J2M}: invocations[2]; "
      f"{CR}: j2_maas_runs[2].stage_stopped]: its noise part, one f_pdf Mecke set per subsampled halo, costs about 30 "
      "times a normal Mecke pass. Rewritten (`noise_pass`: the per-halo scalar sums with f_pdf's column construction; "
      "equal to the first version to rounding in a local smoke run), it ran on its own as run 4:")
    w(f"- run 4: launched {R.load(CR)['j1_mecke_runs'][1]['launched']}, same command form with --results and "
      f"--mecke-out J_halo_dm_lowmass/results_mecke_checks and --stages j1base,mecke_checks; exit code "
      f"{R.v(MCD, ('returncode',), '.0f')}, wall {R.v(MCD, ('wall_seconds',), ',.0f')} s (j1base "
      f"{R.v(MCM, mcs('j1base'), '.0f')} s, mecke_checks {R.v(MCM, mcs('mecke_checks'), '.0f')} s); peak tree RSS "
      f"{R.v(MCD, ('peak_tree_rss_GiB',), '.2f')} GiB; the scratch "
      f"({R.v(CR, ('j1_mecke_runs', 1, 'scratch_bytes_deleted'), '.2f', 1e-9)} GB) was deleted after the final fetch; "
      "outputs in `results_mecke_checks/` (SHA256SUMS);")
    w("- finite differences: per definition the truth-mean profiles of the calibrated bins scaled by 1 +- 0.05 in one "
      "radial band at a time (the five bands of J.5 and the outermost fine shell), each painted through the DM PDF; the "
      f"unscaled pass reproduces J1's truth-mean quantiles exactly [results_mecke_checks/mecke_fd.json];")
    w("- subsample noise: per subsampled halo (up to 64 per bin) and quantile the sum over shells of its whole-atom "
      "Mecke Jacobian; 2000 bootstrap draws within the subsampled bins; the per-halo sums reproduce the subsample-map "
      f"sums of run 2 to {R.v(VER, ('gates', 'G14_mecke_whole_atom_revision', 'noise_per_halo_sums_vs_main_pass_max_abs'), '.1e')} "
      "(float32 reference density) [results_mecke_checks/mecke_noise.json].")
    w("")
    g = lambda *k: ("gates",) + k                                           # noqa: E731
    q = lambda name: ("gates", "G4_sum_rules", "quantiles", name, "deviation")   # noqa: E731
    w(f"**Gates** [{VER}; {n_gates} gates, " + ("all passed" if all(G[k]["passed"] for k in G) else "NOT all passed")
      + f": {gate_names[0].split('_')[0]}-{gate_names[-1].split('_')[0]}]:")
    w("")
    w("| Gate | Check | Result |")
    w("|---|---|---|")
    w("| G0 | J0 inputs equal Test H's export; owned counts consistent | bitwise equal |")
    w(f"| G1 | fallback reconstruction vs the basis' own fallback totals | "
      f"{R.v(VER, g('G1_fallback_parity', 'JA'), '.1e')} |")
    w(f"| G2 | Test H parity, C10/D0 = H's A2-ii truth | moments {R.v(VER, g('G2_test_H_parity', 'moments_max_rel'), '.1e')}; "
      f"DM-PDF quantiles {R.v(VER, g('G2_test_H_parity', 'pdf_max_rel'), '.1e')}; deposit identity "
      f"{R.v(VER, g('G2_test_H_parity', 'deposit_identity_max_abs'), '.0f')} |")
    g3 = [(VER, g('G3_per_bin_parity_with_H', 'max_rel', k)) for k in R.raw(VER, g('G3_per_bin_parity_with_H', 'max_rel'))]
    w(f"| G3 | per-bin own terms in the 10 common bins vs H's sens.npz | <= {R.agg('max', g3, '.1e')} |")
    QA = ("q16", "q50", "q84", "q975", "q999")
    qapp = lambda kind, d: [(VER, q(f"{kind}__{d}__dm_{x}")) for x in QA          # noqa: E731
                            if R.raw(VER, ("gates", "G4_sum_rules", "quantiles", f"{kind}__{d}__dm_{x}", "applicable"))]
    slab = lambda d, k="ratio": ("gates", "G4_sum_rules", "outer_slab_q50", d, k)   # noqa: E731
    w(f"| G4 | importance sum rules | moment observables: max deviation "
      f"{R.v(VER, g('G4_sum_rules', 'moments_max_deviation'), '.1e')} (threshold 1e-10). Mecke quantile maps with the "
      "whole DM = 0 atom, D0-D3, every quantile outside the atom (all but D1's Q16; threshold max(0.05, 3 sigma) for "
      "the per-halo subsample maps, sigma = bootstrap SD of the map sum over the subsampled haloes "
      "[results_mecke_checks/mecke_noise.json]; 0.01 for the truth-mean maps): subsample max deviation "
      + " / ".join(R.agg("max", qapp("imp", d), ".3f") for d in DEFS) + " (D0 / D1 / D2 / D3); truth-mean max "
      "deviation " + " / ".join(R.agg("max", qapp("impdet", d), ".1e") for d in DEFS) + ". Radial distribution: "
      "central finite differences (eps = 0.05) of the truth-mean quantiles per radial band and for the outermost fine "
      "shell against Q times the band sum of the truth-mean map (threshold 3%, Q16-Q97.5 outside the atom, bands with "
      f">= 5% of the map; {R.v(VER, g('G4_sum_rules', 'finite_difference', 'n_applicable'), '.0f')} pairs): largest "
      f"|ratio - 1| = {R.v(VER, g('G4_sum_rules', 'finite_difference', 'max_abs_ratio_minus_1_whole_atom'), '.3f')} "
      f"(J1's half-atom maps: {R.v(VER, g('G4_sum_rules', 'finite_difference', 'max_abs_ratio_minus_1_half_atom_J1'), '.2f')}); "
      f"Q99.9 (reported, not gated): {R.v(VER, g('G4_sum_rules', 'finite_difference', 'q999_max_abs_ratio_minus_1_whole_atom'), '.3f')} "
      f"over {R.v(VER, g('G4_sum_rules', 'finite_difference', 'q999_n'), '.0f')} pairs |")
    w(f"| G5 | truncation curves | end point = truth to {R.v(VER, g('G5_truncation', 'endpoint_max_rel'), '.1e')}; "
      "exactly flat beyond each boundary; zero importance beyond it; cumulative curves reach exactly 1 at each "
      "boundary |")
    w(f"| G6 | low-mass rule | every stacked shell >= 400 particles (minimum "
      f"{R.agg('min', [(VER, g('G6_low_mass_rule', 'min_stacked_count', f'{lab}|E_a2ii')) for lab in LOW], '.0f')} "
      f"for A2-ii); merging conserves each halo's integral to "
      f"{R.v(VER, g('G6_low_mass_rule', 'merge_conservation_max_rel'), '.1e')}; "
      f"{R.v(VER, g('G6_low_mass_rule', 'groups_straddling_1R_2R_xh'), '.0f')} merged shells straddle 1R, 2R or x_h |")
    w(f"| G7 | PDF mean/variance vs moment forms | <= {R.v(VER, g('G7_pdf_vs_moments', 'max_abs_rel'), '.1e')} |")
    w(f"| G8 | spherical chord (J.2) | engine vs chord integral, same nodes "
      f"{R.v(VER, g('G8_spherical_chord', 'engine_vs_chord_integral_same_nodes_max_rel'), '.1e')}; unit chain "
      f"{R.v(VER, g('G8_spherical_chord', 'engine_DM_vs_HalfDome_formula_SI_max_rel'), '.1e')} |")
    w(f"| G9 | D1/D2 PDF atom vs exp(-lambda) | relative difference {R.v(VER, g('G9_zero_hit_probability', 'D1', 'rel'), '.1e')} "
      f"(D1), {R.v(VER, g('G9_zero_hit_probability', 'D2', 'rel'), '.1e')} (D2) |")
    w(f"| G10 | configuration | {R.n(VER, g('G10_configuration', 'calibrated_bins'))} calibrated bins; engine bin 12.5-12.7 "
      f"holds {R.v(VER, g('G10_configuration', 'engine_bin_12.5-12.7_haloes'), '.0f')} painted haloes; the z <= 1 "
      "lightcone-mass summary (`j_lightcone_mass.py`) reads the histogram of the maas run (SHA256 equal; per-bin counts "
      f"differ by {R.v(VER, g('G10_configuration', 'lightcone_mass_z_parity', 'counts_z_le_1_max_abs_diff'), '.0f')}) |")
    w("| G11 | remote code check | compute modules equal |")
    w(f"| G12 | mean DM of all evaluations recomputed locally from ell = 0 kernels (`j_local_mean.py`) | "
      f"{R.v(VER, g('G12_local_mean_dm_recomputation', 'max_rel'), '.1e')} over "
      f"{R.v(VER, g('G12_local_mean_dm_recomputation', 'n_evaluations'), '.0f')} evaluations |")
    g13 = G.get("G13_report_numbers")
    w("| G13 | every number of this report re-resolved from its source field and found in the text "
      "(`post/report_numbers.json`) | " + ("passed" if g13 and g13["passed"] else "not yet evaluated") + " |")
    g14 = lambda *k: ("gates", "G14_mecke_whole_atom_revision") + k              # noqa: E731
    w("| G14 | Mecke whole-atom revision (maas run 2) | " + ("passed" if G.get("G14_mecke_whole_atom_revision", {}).get("passed")
                                                          else "NOT PASSED") +
      f": reference PDFs and J1's half-atom Jacobians reproduced "
      f"({R.v(VER, g14('reference_quantiles_max_rel'), '.1e')} / {R.v(VER, g14('half_atom_jacobians_max_rel'), '.1e')}); "
      "the local rebuild reproduced J1's stored sens products byte for byte; rebuilt files and archive match the logged "
      "digests; toy compound Poisson with lambda = "
      f"{R.v(VER, g14('toy', 'lambda'), '.3f')}: whole-atom map sums "
      + " / ".join(R.v(VER, g14('toy', 'whole_atom_sum', i), '.3f') for i in range(3))
      + " at Q50 / Q84 / Q97.5, J1's half-atom formula "
      + " / ".join(R.v(VER, g14('toy', 'half_atom_sum', i), '.3f') for i in range(3)) + " |")
    w("")
    msr = lambda d, kind, lab, i: (MUPD, ("mecke_sum_rule_minus_1", d, f"{kind}|C14|{d}|T1|{lab}", i))   # noqa: E731
    mch = lambda key, k: (MUPD, ("map_change", key, k))                                            # noqa: E731
    sl = lambda d, which, k="ratio", i=0: (MUPD, ("outer_slab_first_order", d, which, "q50", i, k))   # noqa: E731
    w("**Quantile sum rules and the DM = 0 atom (revised 2026-10-16).** Every quantile outside the atom is homogeneous "
      "of degree 1 in the profile, so its exact importance map sums to 1 whatever the atom. J1's first maps did not: the "
      "Mecke code keeps only half of the CIC atom at DM = 0 (J.2), so they kept only half of the atom's term, "
      "$\\pi_0$ times the $\\lambda$-weighted single-halo DM density at $Q_p$, and missed the other half. D1, with "
      f"{R.v(POST, pdef('C14|D1', 'p_atom'), '.1f', 100)}% of sightlines in the atom, had Q50-Q99.9 maps summing to "
      + R.rng([mch(f"imp__D1__dm_{x}", "sum_old") for x in QA[1:]], ".2f")
      + " (subsample; truth mean "
      + R.rng([(TS, ("mecke_sum_rule", "C14|D1|T1|truthmean", i)) for i in (1, 2, 3, 4)], ".2f", 1.0, -1.0)
      + f"); D2 ({R.v(POST, pdef('C14|D2', 'p_atom'), '.2f', 100)}%) had a Q16 truth-mean map summing to "
      f"{R.v(TS, ('mecke_sum_rule', 'C14|D2|T1|truthmean', 0), '.3f', 1.0, -1.0)}. The missing term has its own radial "
      "distribution, so the shapes were biased too: removing D1's outermost fine shell "
      f"({R.v(*sl('D1', 'half_atom_J1', 'R_t'), '.2f')}-1 R200c) changes the truth-mean Q50 by "
      f"{R.v(*sl('D1', 'half_atom_J1', 'exact_dQ'), '.2f')} pc cm^-3 (exact PDF passes), and the old map predicted "
      f"{R.v(*sl('D1', 'half_atom_J1', 'ratio'), '.2f')} of it (D0, no atom: "
      f"{R.v(*sl('D0', 'half_atom_J1', 'ratio'), '.3f')}). With the whole atom [{MUPD}: mecke_sum_rule_minus_1, "
      "outer_slab_first_order]:")
    NZ = "results_mecke_checks/mecke_noise.json"
    w("- D1's Q50-Q99.9 maps sum to "
      + R.rng([msr("D1", "whole", "truth_subsample", i) for i in (1, 2, 3, 4)], ".3f", 1.0, -1.0)
      + " (per-halo subsample; bootstrap SD of the sums "
      + R.rng([(NZ, ("per_definition", "D1", "sigma_boot", i)) for i in (1, 2, 3, 4)], ".3f")
      + f" [{NZ}]) and " + R.rng([msr("D1", "whole", "truthmean", i) for i in (1, 2, 3, 4)], ".3f", 1.0, -1.0)
      + " (truth mean, no subsampling); D2's Q16 maps sum to "
      + f"{R.v(*msr('D2', 'whole', 'truth_subsample', 0), '.3f', 1.0, -1.0)} / {R.v(*msr('D2', 'whole', 'truthmean', 0), '.3f', 1.0, -1.0)};")
    fdk = lambda k: ("gates", "G4_sum_rules", "finite_difference_detail", k, "ratio_whole_atom")    # noqa: E731
    fdh = lambda k: ("gates", "G4_sum_rules", "finite_difference_detail", k, "ratio_half_atom_J1")  # noqa: E731
    fds = lambda *k: ("gates", "G4_sum_rules", "finite_difference") + k                              # noqa: E731
    so2 = lambda d, k: (VER, fds("outermost_shell_q50_second_order", d, k))                     # noqa: E731
    q999_where = "{}, band {}".format(*R.raw(VER, fds("q999_worst_pair")).split("|")[:2])
    pq = lambda d: R.r(MUPD, ("outer_slab_first_order", d, "whole_atom", "q50", 0, "exact_dQ"), TS,   # noqa: E731
                       ("pdf", f"C14|{d}|T1|truthmean", "quantiles", 1), ".1f", 100)
    w(f"- the outer-slab check now gives {R.v(*sl('D1', 'whole_atom'), '.3f')} for D1 and "
      f"{R.v(*sl('D2', 'whole_atom'), '.3f')} for D2 (D0 {R.v(*sl('D0', 'whole_atom'), '.3f')}, D3 "
      f"{R.v(*sl('D3', 'whole_atom'), '.3f')}). Removing a whole shell is a finite change, so this check includes the "
      "second-order response, and only D1's is sizeable. The size of the change is not the reason: removing the "
      f"outermost shell moves Q50 by {pq('D1')}% for D1, by {pq('D0')}% for D0 "
      f"({R.v(*sl('D0', 'whole_atom', 'R_t'), '.2f')}-{R.v(*sl('D0', 'whole_atom', 'R_D'), '.0f')} R200c) and by "
      f"{pq('D2')}% for D2 ({R.v(*sl('D2', 'whole_atom', 'R_t'), '.2f')}-"
      f"{R.v(*sl('D2', 'whole_atom', 'R_D'), '.0f')} R200c), and D0 and D2 stay linear. The curvature is D1's own: "
      "with Q' and Q'' the first and second central differences of Q50 in the shell's profile scale factor s at s = 1 "
      "(runs at s = 1 +- 0.05; Q' is dQ50/d ln F of the shell), "
      f"Q''/Q' = {R.v(*so2('D1', 'd2Q_over_dQ'), '.3f')} for D1 against {R.v(*so2('D0', 'd2Q_over_dQ'), '.3f')} for "
      f"D0 and {R.v(*so2('D2', 'd2Q_over_dQ'), '.3f')} for D2. D1 is the definition with the large atom, and its median "
      f"({R.v(TS, ('pdf', 'C14|D1|T1|truthmean', 'quantiles', 1), '.1f')} pc cm^-3) lies just above it. To second "
      f"order the removal changes D1's Q50 by Q' - Q''/2 = {R.v(*so2('D1', 'removal_dQ_second_order'), '.3f')} pc cm^-3, "
      f"close to the exact {R.v(*so2('D1', 'removal_dQ_exact'), '.3f')} (first order: "
      f"{R.v(*so2('D1', 'removal_dQ_first_order'), '.3f')}) [{VER}: G4 finite_difference."
      "outermost_shell_q50_second_order]. The first-order test proper is the central finite difference itself: for that "
      f"shell of D1 the map's dQ50/d ln F is {R.v(VER, fdk('D1|outermost_shell|q50'), '.3f')} of the "
      f"finite-difference value (J1's half-atom map: {R.v(VER, fdh('D1|outermost_shell|q50'), '.3f')} of it) "
      "[results_mecke_checks/mecke_fd.json];")
    w("- over every radial band (and the outermost shell) of every definition, the whole-atom truth-mean maps match the "
      f"finite differences to {R.v(VER, fds('max_abs_ratio_minus_1_whole_atom'), '.1f', 100)}% (largest |ratio - 1| over "
      f"{R.v(VER, fds('n_applicable'), '.0f')} band-quantile pairs, Q16-Q97.5, bands with >= 5% of the map; D1 "
      f"{R.v(VER, fds('D1', 'max_abs_ratio_minus_1_whole_atom'), '.1f', 100)}%), against "
      f"{R.v(VER, fds('max_abs_ratio_minus_1_half_atom_J1'), '.0f', 100)}% for J1's half-atom maps (G4). Q99.9 is "
      "outside the gate's Q16-Q97.5 range and reported only: its maps match to "
      f"{R.v(VER, fds('q999_max_abs_ratio_minus_1_whole_atom'), '.1f', 100)}% (largest |ratio - 1| over its "
      f"{R.v(VER, fds('q999_n'), '.0f')} pairs, at {q999_where}), also within the 3% tolerance;")
    w("- D0 and D3 have almost no atom (" + f"{R.v(POST, pdef('C14|D0', 'p_atom'), '.1e')} and "
      f"{R.v(POST, pdef('C14|D3', 'p_atom'), '.1e')}): their maps change by at most "
      + R.agg("max", [mch(f"{k}__D0__dm_{x}", "max_abs_change_over_max") for k in ("imp", "impdet") for x in QA], ".1e")
      + " and " + R.agg("max", [mch(f"{k}__D3__dm_{x}", "max_abs_change_over_max") for k in ("imp", "impdet") for x in QA], ".1e")
      + " of their largest element;")
    w("- D1's Q16 lies inside the atom (it does not move when the profiles change), so it has no map; the per-halo "
      "subsample maps scatter by a few per cent (G4). The quantile radii of J.5 come from the exact PDF truncation "
      "passes and did not change.")
    w("")
    # ------------------------------------------------------------------------------------------------ J.2
    w("### J.2 Definitions and equations")
    w("")
    w("**Measured profiles (FLAMINGO, z = 0).** With $x = r/R_{200c}$:")
    w("$$E(x)=\\frac{n_e(x)\\,\\mu_{\\rm XGP}}{f_b^{\\rm F}\\,\\rho_c^{\\rm F}},\\qquad "
      "P(x)=\\frac{P_e(x)}{P_{200}},\\qquad P_{200}=\\frac{G\\,M_{200c}\\,200\\,\\rho_c^{\\rm F}\\,f_b^{\\rm F}}{2R_{200c}}$$")
    w("- $n_e$: free-electron number density [m^-3]; $P_e$: electron pressure [Pa];")
    w("- $\\mu_{\\rm XGP}=(m_e+2m_p/1.76)/0.9\\approx1.2632\\,m_p$: XGPaint's gas mass per electron [kg];")
    w(f"- $f_b^{{\\rm F}}=\\Omega_b/\\Omega_m$ = {R.v(PAR, ('chord', 'f_b_study'), '.5f')}: FLAMINGO's cosmic baryon "
      f"fraction [{PAR}: chord.f_b_study]; $\\rho_c^{{\\rm F}}$: FLAMINGO's critical density at z = 0 [kg m^-3];")
    w("- $G$: Newton's constant; $M_{200c}$ [kg] and $R_{200c}$ [m]: mass and radius enclosing 200 $\\rho_c^{\\rm F}$;")
    w("- $E$ and $P$ are dimensionless. A shell value $F_{h,s}$ ($F$ = $E$ or $P$) is the volume average of halo $h$ "
      "in fine shell $s=[x_s,x_{s+1}]$.")
    w("")
    w("**Painting in the engine (HalfDome lightcone, observer at z = 0).** A lightcone halo of mass $M$ at redshift "
      "$z$ in calibrated bin $c$ is painted with the bin's z = 0 stack and the published redshift response:")
    w("$$n_e(r)=n_{\\rm u}(z)\\,E_c(x)\\,e_E(M,z,x),\\qquad n_{\\rm u}(z)=\\frac{f_b^{\\rm HD}\\rho_c^{\\rm HD}(z)}"
      "{\\mu_{\\rm XGP}},\\qquad e_E(M,z,x)=\\frac{F^{\\rm B16}(M,z,x)}{F^{\\rm B16}(M,0,x)}$$")
    w("$$F^{\\rm B16}(M,z,x)=A\\,(x/0.5)^{-0.2}\\big[1+(x/0.5)^{\\alpha}\\big]^{-(\\beta-0.2)/\\alpha},\\quad "
      "A=4000\\,m^{0.29}(1+z)^{-0.66},\\ \\alpha=0.88\\,m^{-0.03}(1+z)^{0.19},\\ \\beta=3.83\\,m^{0.04}(1+z)^{-0.025}$$")
    w(f"- $f_b^{{\\rm HD}}=0.049/0.31$ = {R.v(PAR, ('chord', 'f_b_engine_(lightcone_analytic.FB)'), '.5f')}: HalfDome's "
      "baryon fraction used by the engine (`lightcone_analytic.FB`), not FLAMINGO's; at fixed $M$ the painted electron "
      f"count is therefore {R.v(POST, ('derived_for_report', 'f_b_engine_over_f_b_study'), '.4f')} times the "
      "FLAMINGO-measured one (the stacks fix $E$, i.e. the electron density relative to $f_b\\rho_c$);")
    w("- $\\rho_c^{\\rm HD}(z)$: HalfDome's critical density at the halo redshift (saved CLASS tables, h = 0.68) "
      "[kg m^-3]; $R_{200c}$ of the lightcone halo follows from $M$ and $\\rho_c^{\\rm HD}(z)$;")
    w("- $n_{\\rm u}(z)$: the electron-density unit [m^-3];")
    w("- $e_E$: the declared redshift response (dimensionless; $e_E=1$ at z = 0). It rescales the z = 0 stack by the "
      "published B16 evolution at the halo's own mass, so every DM number depends on it;")
    w("- $F^{\\rm B16}$: the published Battaglia (2016) gas-density profile in units of $f_b\\rho_c$; "
      "$m=M/(10^{14}\\,M_\\odot)$; coefficients `pilot_physics.B16` "
      "(`radii234_contamination_L1_m9_0077_20261008/pilot_physics.py:21`), used by "
      "`lightcone_analytic.published_profile`;")
    w("- pressure is painted the same way: $P_e=P_{200}^{\\rm HD}(M,z)\\,P_c(x)\\,e_P(M,z,x)$ with "
      "$e_P=F^{\\rm B12}(M,z,x)/F^{\\rm B12}(M,0,x)$, $F^{\\rm B12}=0.5176\\,P_0\\,(x/x_c)^{-0.3}(1+x/x_c)^{-\\beta}$, "
      "$P_0=18.1\\,m^{0.154}(1+z)^{-0.758}$, $x_c=0.497\\,m^{-0.00865}(1+z)^{0.731}$, "
      "$\\beta=4.35\\,m^{0.0393}(1+z)^{0.415}$ (0.5176 converts thermal to electron pressure), and $P_{200}^{\\rm HD}$ "
      "the expression above with $f_b^{\\rm HD}$ and $\\rho_c^{\\rm HD}(z)$.")
    w("- Uncalibrated (fallback) haloes are painted with $F^{\\rm B16}$ / $F^{\\rm B12}$ themselves.")
    w("")
    w("**A2-ii partition.** A gas particle at position $\\mathbf r_i$ belongs to the painted halo "
      "$h^*=\\arg\\min_h|\\mathbf r_i-\\mathbf r_h|/R_{200c,h}$ over the halo and its painted neighbours "
      "($M_{200c}\\ge6.6\\times10^{12}\\,M_\\odot$). **Host-bound:** gas bound (HBT) to the central or to its own "
      "satellites, at any radius to 4 R200c; it is not a subset of A2-ii.")
    w("")
    w(f"**Cosmic mean in E units.** $\\bar E=\\kappa_E\\,\\Omega_m$ = {R.v(ST, ('E_cosmic',), '.4f')} [{ST}: E_cosmic]:")
    GP = "../G_physical_outer/results/prepare.json"
    w(f"- $\\kappa_E$ = {R.v(GP, ('normalisation', 'KAPPA_E_measured_2_4R'), '.4f')}: electrons per unit gas mass in E "
      "units, i.e. $\\mu_{\\rm XGP}$ divided by the true gas mass per free electron, measured at 2-4 R200c "
      "(`G_physical_outer/results/prepare.json`: normalisation.KAPPA_E_measured_2_4R);")
    w(f"- $\\Omega_m=\\Omega_b/f_b^{{\\rm F}}$ = {R.v(GP, ('normalisation', 'Omega_m = Omega_b/f_b'), '.3f')}: FLAMINGO's "
      "matter density parameter.")
    w("")
    w("**Stack.**")
    w("$$\\bar F_{b,s}=\\frac{\\sum_{h\\in b}w_hF_{h,s}}{\\sum_{h\\in b}w_h}$$")
    w("- $b$: 0.2-dex M200c bin; $s$: fine shell; $w_h$: inclusion weight of halo $h$ (inverse inclusion probability "
      "of its stratum); $F_{h,s}$: the halo's shell average of $P$ or $E$ (dimensionless).")
    w("")
    w("**Stack-level rule (12.7-13.5 only).** Inside each segment $[0,x_h]$, $[x_h,1]$, $[1,2]$, $[2,4]$, "
      "consecutive fine shells are merged greedily from the inside out into a group $G$ until")
    w("$$N_{b,G}=\\sum_{h\\in b}\\sum_{s\\in G}n_{h,s}\\ \\ge\\ 400 .$$")
    w("- $n_{h,s}$: gas particles of halo $h$ in shell $s$ owned by the selection, AGN-heated gas excluded "
      "(A2-ii counts for $P$ and $E$, host-bound counts for D3); $N_{b,G}$: the stack's particle count in group $G$;")
    w("- $1/\\sqrt{400}$ = 5% is the Poisson error of the stacked particle sum; a remainder below 400 at the end of a "
      "segment joins the previous group;")
    w("- per halo, the merged value $F_{h,G}=\\sum_{s\\in G}V_sF_{h,s}/\\sum_{s\\in G}V_s$, with "
      "$V_s=\\tfrac{4\\pi}{3}(x_{s+1}^3-x_s^3)$ the dimensionless shell volume, is assigned to every fine shell of $G$ "
      "(treatment T1, baseline). Bins >= 13.5 keep the raw fine shells, as in Test H.")
    w("")
    w("**Hydrodynamic-resolution radius.** $x_h$ = the first fine edge with "
      "$x_s\\ge\\mathrm{median}_h(2h_{h,s}/R_{200c})$, where $h_{h,s}$ is the upper-median SPH smoothing length of "
      "the halo's gas in shell $s$. Inside $x_h$ the radial shape is unresolved but the integral is measured.")
    w("")
    w("**Treatments of the unresolved cores** (all three keep each halo's measured integral inside $x_h$):")
    w("- T2: inside $x_h$, $F_{h,s}=S_s\\,X_h(<x_h)/\\sum_{s<k_h}V_sS_s$, with $S_s$ the published shell average "
      "(B16 for $E$, B12 for $P$) inclusion-weighted over the bin, $k_h$ the index of the shell starting at $x_h$, "
      "and $X_h(<x_h)=\\sum_{s<k_h}V_sF_{h,s}$ the halo's measured integral inside $x_h$;")
    w("- T3: raw fine shells (no merging);")
    w(f"- S82: bin 12.7-12.9 built only from its {R.v(PREP, ('subset_12.82', 'n'), '.0f')} haloes with "
      f"M200c >= 6.6e12 Msun; their mean log M is {R.v(PREP, ('subset_12.82', 'mean_logm'), '.3f')}, against "
      f"{R.v(PREP, J1a('12.7-12.9', 'mean_logm200c_flamingo'), '.3f')} for the full bin and "
      f"{R.v(PREP, J1a('12.7-12.9', 'mean_logm200c_lightcone'), '.3f')} for the bin's lightcone haloes at all z and "
      f"{R.v(LCM, ('bin_12.7-12.9', 'mean_logm200c_z_le_1'), '.3f')} at z <= 1 [{LCM}]. S82 is therefore still below "
      f"the painted z <= 1 masses (smallest {R.v(LCM, ('min_logm200c_painted_z_le_1',), '.3f')}).")
    w("")
    w("**Integrals.**")
    w("$$X(<R)=\\sum_{x_{s+1}\\le R}V_s\\bar F_s,\\qquad \\frac{N_e(<R)\\,\\mu_{\\rm XGP}}{f_b^{\\rm F}M_{200c}}"
      "=\\frac{X_E(<R)}{800\\pi/3}=f_e(<R)$$")
    w("- $R$: radius in units of R200c; $X_E$ ($X_P$): the integral of $E$ ($P$) in units of "
      "$f_b^{\\rm F}\\rho_c^{\\rm F}R_{200c}^3/\\mu_{\\rm XGP}$ ($P_{200}R_{200c}^3$); $N_e(<R)$: electrons inside $R$; "
      "the identity uses $M_{200c}=200\\rho_c^{\\rm F}\\tfrac{4\\pi}{3}R_{200c}^3$; $f_e$: electrons in units of "
      "the cosmic-baryon-fraction expectation. $Y(<R)=X_P(<R)$ in units of $P_{200}R_{200c}^3$.")
    w("")
    w("**Halo-component definitions.** D0: A2-ii to 4 R200c. D1: $F^{\\rm D1}_s=F^{\\rm A2ii}_s\\,"
      "\\mathbb 1[x_{s+1}\\le1]$, a 3D cut. D2: the same cut at $x=2$. D3: host-bound to 4 R200c. The fallback haloes "
      "above 10^15.388 are cut at the same radius as the definition.")
    w("")
    w("**Projection of a 3D-cut profile.**")
    w("$$\\Sigma(b)=2\\int_0^{\\sqrt{X^2-b^2}}E\\big(\\sqrt{b^2+l^2}\\big)\\,e_E\\,dl,\\qquad "
      "\\mathrm{DM}(b)=\\frac{R_{200c}\\,n_{\\rm u}(z)}{1+z}\\,\\Sigma(b)=\\frac{2R_{200c}}{1+z}"
      "\\int_0^{\\sqrt{X^2-b^2}}n_e\\big(\\sqrt{b^2+l^2}\\big)\\,dl$$")
    w("- $b$: impact parameter in units of R200c; $l$: line-of-sight coordinate in units of R200c; $X$: sphere radius "
      "(1, 2 or 4); $\\Sigma$: dimensionless column; DM in pc cm^-3 (observer frame, factor $1/(1+z)$); "
      "$R_{200c}n_{\\rm u}(z)/(1+z)$ is the column scale (pc cm^-3 per unit $\\Sigma$).")
    w(f"- This is HalfDome's SphericalChordDMProfile formula (`frb_map_generation/lee2022_frb_dm_profile.jl:681-751`). "
      f"Gate G8, B16 profile at log M = {R.v(PAR, ('chord', 'logm200c'), '.1f')}, z = {R.v(PAR, ('chord', 'z'), '.1f')}, "
      f"X = {R.v(PAR, ('chord', 'sphere_radius_R200c'), '.0f')} [{PAR}: chord]: the engine column (exact projection "
      "over the basis segments inside X) equals the chord integral of the same node profile to "
      f"{R.v(PAR, ('chord', 'engine_vs_chord_integral_same_nodes_max_rel'), '.1e')} and the smooth B16 to "
      f"{R.v(PAR, ('chord', 'engine_vs_chord_integral_smooth_B16_max_rel'), '.1e')}; column scale x Sigma equals the "
      f"formula in SI units to {R.v(PAR, ('chord', 'engine_DM_vs_HalfDome_formula_SI_max_rel'), '.1e')}; the column "
      f"scale there is {R.v(PAR, ('chord', 'column_scale_pc_cm3_per_unit_Sigma'), '.4f')} pc cm^-3, computed with "
      "$f_b^{\\rm HD}$.")
    w("")
    w("**Fourier transforms of a halo** (multipole $\\ell$; flat-sky, Limber):")
    w("$$\\tau_\\ell(M,z)=\\sigma_T\\,n_{\\rm u}(z)\\,\\frac{R_{200c}^3}{d_A^2}\\,4\\pi\\int_0^{X}x^2E(x)\\,e_E(M,z,x)"
      "\\,\\frac{\\sin(qx)}{qx}\\,dx,\\qquad q=\\frac{(\\ell+1/2)R_{200c}}{d_A}$$")
    w("- $\\tau_\\ell$: transform of the halo's optical-depth map (dimensionless); $\\sigma_T$: Thomson cross-section "
      "[m^2]; $d_A$: angular-diameter distance [m]; $X$: outer radius of the definition; $q$: dimensionless wavenumber.")
    w("- DM: $\\mathrm{DM}_\\ell=s_{\\rm DM}(z)\\,\\tau_\\ell$ with $s_{\\rm DM}(z)=1/[\\sigma_T\\,u_{\\rm DM}\\,(1+z)]$ for "
      "$z\\le1$ and 0 beyond (sources at $z_s=1$); $u_{\\rm DM}=3.0857\\times10^{22}$ m^-2 is 1 pc cm^-3.")
    w("- tSZ: $y_\\ell$ is the same transform of $\\sigma_T P_e/(m_ec^2)$ ($m_e$ electron mass, $c$ speed of light).")
    w("- In the fine basis every transform is linear in the shell values: $\\tau_\\ell=\\sum_s E_{c,s}\\,g_{s,\\ell}(M,z)$, "
      "with $g_{s,\\ell}$ the transform of a unit step profile in shell $s$ (including $e_E$).")
    w("")
    w("**Moments.**")
    w("$$\\langle\\mathrm{DM}\\rangle=\\sum_c\\sum_sE_{c,s}\\,m_{c,s}+\\mathrm{fb},\\qquad "
      "\\mathrm{Var(DM)}=\\sum_c\\sum_{s,t}Q_c[s,t]\\,\\langle E_sE_t\\rangle_c+\\mathrm{fb}$$")
    w("$$C_\\ell^{1h}=\\sum_c\\sum_{s,t}K_c[s,t,\\ell]\\,\\langle F_sF_t\\rangle_c+\\mathrm{fb},\\qquad "
      "C_\\ell^{2h}=\\sum_z\\frac{P_{\\rm lin}(k_\\ell,z)}{V_z}\\,a_z\\Big[W_{\\rm fb}(z,\\ell)+\\sum_{c,s}F_{c,s}W_{c,s}(z,\\ell)\\Big]^2$$")
    w("- $c$: calibrated bin; cells: lightcone (z, M) histogram cells with $N$ haloes over the full sky; fb: the same "
      "sums over the fallback haloes;")
    w("- $m_{c,s}=\\sum_{{\\rm cells}\\in c,\\,z\\le1}\\frac{N}{4\\pi}s_{\\rm DM}(z)\\,g_{s,0}$: mean DM per unit $E$ in "
      "shell $s$ [pc cm^-3];")
    w("- $Q_c[s,t]=\\sum_{{\\rm cells}\\in c}\\sum_{b}\\lambda_b\\,\\mathrm{DM}_s(b)\\,\\mathrm{DM}_t(b)$: compound-Poisson "
      "(Campbell) variance kernel [(pc cm^-3)^2], $\\mathrm{DM}_s(b)$ the DM at impact node $b$ of a unit profile in "
      "shell $s$, $\\lambda_b=\\frac{N}{4\\pi}(R_{200c}/d_A)^2\\,\\Delta A_b$ the expected number of haloes per sightline "
      "with impact parameter in node $b$ ($\\Delta A_b$ its area in units of $R_{200c}^2$);")
    w("- $\\langle F_sF_t\\rangle_c$: inclusion-weighted per-halo second moments (scatter-inclusive);")
    w("- $K_c[s,t,\\ell]=\\sum_{{\\rm cells}\\in c}\\frac{N}{4\\pi}\\,g_{s,\\ell}g_{t,\\ell}$ for tSZ ($y$ transforms), "
      "the DM field ($s_{\\rm DM}^2$ times the $\\tau$ transforms) and y x DM; for the kSZ 1-halo term $N$ is replaced by "
      "$\\sum_{h\\in{\\rm cell}}(v_{{\\rm LOS},h}/c)^2$ with the catalogue line-of-sight velocities $v_{\\rm LOS}$;")
    w("- $k_\\ell=(\\ell+1/2)/\\chi(z)$: Limber wavenumber, $\\chi$ the comoving distance; $P_{\\rm lin}$: CLASS linear "
      "matter power spectrum [Mpc^3]; $V_z=(\\chi_{\\rm hi}^3-\\chi_{\\rm lo}^3)/3$: comoving volume of lightcone "
      "redshift shell $z$ per steradian [Mpc^3 sr^-1], $\\chi_{\\rm lo}$ and $\\chi_{\\rm hi}$ the comoving distances "
      "of its edges (`kernels/engine.py:494`), matching the per-steradian halo counts $N/4\\pi$ in $W$;")
    w("- $W_{c,s}(z,\\ell)=\\sum_{M\\in c}\\frac{N}{4\\pi}\\,b_h(M,z)\\,g_{s,\\ell}(M,z)$: 2-halo window per unit shell value "
      "(Tinker bias $b_h$); $W_{\\rm fb}$: the same sum over the fallback haloes with their published profiles;")
    w("- $a_z$ = 1 (tSZ), $\\sigma^2_{v,{\\rm lin}}(z)/c^2$ (kSZ, approximate 2-halo term; $\\sigma_{v,{\\rm lin}}$ the linear "
      "line-of-sight velocity dispersion), $s_{\\rm DM}(z)^2$ (DM field), $s_{\\rm DM}$ on one side only for y x DM;")
    w("- units: kSZ $C_\\ell$ in $(\\Delta T/T)^2$, $D_\\ell=\\ell(\\ell+1)C_\\ell T_{\\rm CMB}^2/2\\pi$ in muK^2 with "
      "$T_{\\rm CMB}$ = 2.7255 K; DM-field $C_\\ell$ in (pc cm^-3)^2; y x DM in pc cm^-3.")
    w("")
    w("**DM PDF** ($z_s$ = 1, observer frame). Compound Poisson: every lightcone cell and impact node $b$ adds intensity "
      "$\\lambda_b$ at $\\mathrm{DM}(b)$ (cloud-in-cell on a 0.125 pc cm^-3 grid of $2^{16}$ points); "
      "$p=\\mathrm{IFFT}[\\exp(\\mathrm{FFT}\\,\\Lambda-\\Lambda_{\\rm tot})]$, with $\\Lambda$ the deposited intensity "
      "histogram and $\\Lambda_{\\rm tot}$ its sum. Quantiles use the mid-cell convention: the first cell is "
      "DM < 0.0625 pc cm^-3, so a quantile of 0.0625 means 'inside the no-crossing atom'.")
    w("")
    w("**Halo crossings per sightline** [" + TS + ": crossings].")
    w("$$\\lambda_X=\\sum_{z\\le1}\\sum_M N\\,\\frac{1-\\cos(XR_{200c}/d_A)}{2},\\qquad P(\\text{no crossing})=e^{-\\lambda_X}$$")
    w(f"- $X R_{{200c}}/d_A$: angular radius of the sphere [rad]. "
      f"$\\lambda$ = {R.v(TS, ('crossings', '1R', 'lambda_total'), '.3f')} / "
      f"{R.v(TS, ('crossings', '2R', 'lambda_total'), '.3f')} / {R.v(TS, ('crossings', '4R', 'lambda_total'), '.2f')} "
      "for X = 1 / 2 / 4.")
    w("")
    w("**Importance.**")
    w("$$I_O[b,s]=\\frac{1}{k_O\\,O}\\frac{\\partial O}{\\partial\\ln F_{b,s}},\\qquad \\sum_{b,s}I_O[b,s]=1$$")
    w("- $O$: an observable; $k_O$ = 2 for quadratic observables (variance, $C_\\ell$), 1 for means, quantiles and each "
      "side of y x DM; $\\partial/\\partial\\ln F_{b,s}$ scales every halo of bin $b$ in shell $s$ by $1+\\epsilon$; the "
      "sum rule is Euler's theorem for homogeneous functions.")
    w("- Quantiles (Mecke): $\\frac{\\partial Q_p}{\\partial\\ln F_{b,s}}=\\frac{1}{p(Q_p)}\\sum\\lambda_{b'}\\,"
      "\\frac{\\partial\\mathrm{DM}_{b'}}{\\partial\\ln F_{b,s}}\\,p(Q_p-\\mathrm{DM}_{b'})$, summed over cells and impact "
      "nodes $b'$; $Q_p$ the quantile; $p$ the full PDF of the summed DM, including the atom $\\pi_0\\,\\delta(\\mathrm{DM})$ "
      "of sightlines that cross no halo ($\\pi_0=e^{-\\lambda}$). The atom contributes $\\pi_0$ times the "
      "$\\lambda$-weighted density of single-halo DM values at $Q_p$ (a halo whose own DM equals $Q_p$ on a sightline that "
      "crosses nothing else); with it the identity $\\sum\\lambda\\,\\mathrm{DM}\\,p(Q-\\mathrm{DM})=Q\\,p(Q)$ holds and "
      "the map of every quantile outside the atom sums to 1 (every sightline DM is linear in the profile, so "
      "$Q_p(\\alpha F)=\\alpha Q_p(F)$).")
    w("- On the CIC grid (step $\\Delta$ = 0.125 pc cm^-3) the atom is the node-0 mass $p_0\\Delta$, a triangle of half-width "
      "$\\Delta$ around DM = 0. The engine's and Test F's Mecke code evaluate $p(u)$ as `np.interp(u, grid, density, left=0)` "
      "(`kernels/engine.py:_quantile_jacobian`, `F_observables/f_pdf.py`), which keeps only its half at $u\\ge0$. J1 now "
      "uses the whole-atom density $p_w(u)=p(u)+p_0\\max(0,1-|u|/\\Delta)\\,[u<0]$, computed with f_pdf's own code and a "
      "reference whose density is shifted by one node (a zero at DM = 0) and whose quantiles are shifted by $+\\Delta$ "
      "(`j_mecke_atom.py`; f_pdf and the engine are unchanged). A quantile inside the atom (D1's Q16) does not move: its "
      "map is not defined.")
    w("")
    w("**Truncation curves.**")
    w("$$T_O(R_t)=\\frac{O(R_t)}{O(R_D)},\\qquad C_O(R_t)=\\frac{O(R_t)-O(0)}{O(R_D)-O(0)}$$")
    w("- $O(R_t)$: every calibrated halo cut in 3D at $R_t$, fallback unchanged; $R_D$: the definition's boundary; "
      "$O(0)$: no calibrated halo; $R_X$: first radius where $C_O=X$; loss at $R_t$ = $1-T_O(R_t)$.")
    w("")
    w("**Tolerance per 0.1 dex.**")
    w("$$\\mathrm{tol}(x)=\\frac{0.01}{|k_O\\,i(x)\\,0.1\\ln10|},\\qquad i(x)=\\frac{\\sum_{b\\,\\rm cal}I_O[b,s]}{\\Delta\\ln x_s}$$")
    w("- the coherent fractional error of all calibrated stacks over a 0.1-dex range around $x$ that changes $O$ by 1%; "
      "$i(x)$: importance per unit $\\ln x$ in the shell containing $x$, $\\Delta\\ln x_s$ its width.")
    w("")
    # ------------------------------------------------------------------------------------------------ J.3
    w("### J.3 Low-mass stacks vs the published fallback (J1a)")
    w("")
    w(f"Sources: {PREP}: J1a_by_bin and lightcone; {ST}. Columns: FLAMINGO haloes in the stack; lightcone haloes "
      "with z <= 1; $x_h$; outer edge of the last merged A2-ii group inside R200c; $f_e(<1R)$ for A2-ii / B16 / "
      "host-bound; A2-ii/B16 electrons inside 1, 2, 4 R200c; Y(<1R) A2-ii/B12; share of the A2-ii electrons inside 4R "
      "that lie inside R200c.")
    w("")
    w("| bin (log M200c) | haloes | lightcone haloes z<=1 | x_h | merged out to | f_e(<1R) A2-ii / B16 / bound | "
      "A2-ii/B16 at 1R / 2R / 4R | Y(<1R) A2-ii/B12 | A2-ii share inside R200c |")
    w("|---|---|---|---|---|---|---|---|---|")
    for i, lab in enumerate(LOW):
        w(f"| {lab} | {R.v(PREP, J1a(lab, 'n_haloes'), '.0f')} | "
          f"{R.v(PREP, lc + ('haloes_per_engine_bin_z_le_1', i + 1), ',.0f')} | {R.v(PREP, J1a(lab, 'x_h'), '.2f')} | "
          f"{R.v(PREP, J1a(lab, 'merge_E_a2ii', 'outer_edge_of_last_merged_group_inside_R200c'), '.2f')}R | "
          f"{R.v(PREP, J1a(lab, 'Ne_lt1R_over_fbM', 'A2ii'), '.3f')} / {R.v(PREP, J1a(lab, 'Ne_lt1R_over_fbM', 'B16'), '.3f')} / "
          f"{R.v(PREP, J1a(lab, 'Ne_lt1R_over_fbM', 'host_bound'), '.3f')} | "
          f"{R.v(PREP, J1a(lab, 'Ne_lt1R_ratio_A2ii_over_B16'), '.2f')} / {R.v(PREP, J1a(lab, 'Ne_lt2R_ratio_A2ii_over_B16'), '.2f')} / "
          f"{R.v(PREP, J1a(lab, 'Ne_lt4R_ratio_A2ii_over_B16'), '.2f')} | {R.v(PREP, J1a(lab, 'Y_lt1R_ratio_A2ii_over_B12'), '.2f')} | "
          f"{R.v(PREP, J1a(lab, 'share_of_Ne_lt4R_inside_R200c', 'A2ii'), '.1f', 100)}% |")
    w("")
    w("- **Particle counts.** The median halo has " + " / ".join(
        f"{R.v(PREP, J1a(lab, 'median_particles_per_fine_shell_inside_R200c', 'A2ii_per_shell_median_range', 0), '.0f')}-"
        f"{R.v(PREP, J1a(lab, 'median_particles_per_fine_shell_inside_R200c', 'A2ii_per_shell_median_range', 1), '.0f')}"
        for lab in LOW) + " A2-ii particles per fine shell inside R200c (four bins in order). Without merging, "
      + R.seq(PREP, lambda b: J1a(b, 'merge_E_a2ii', 'fine_shells_below_400_unmerged_T3'), LOW, '.0f')
      + " fine shells would hold fewer than 400 stacked particles. After merging the smallest stacked A2-ii shells "
      "hold " + R.seq(PREP, lambda b: J1a(b, 'merge_E_a2ii', 'min_stacked_count'), LOW, '.0f')
      + " particles and the host-bound ones " + R.seq(PREP, lambda b: J1a(b, 'merge_E_hb', 'min_stacked_count'), LOW, '.0f') + ".")
    w("- **Resolution radii.** The median 2h/R200c at 2R is "
      + R.seq(PREP, lambda b: J1a(b, 'median_2h_over_R200c_at_2R'), LOW, '.2f')
      + "; the gravitational-softening limit 4 epsilon/R200c is "
      + R.rng([(PREP, J1a(lab, 'softening_limit_4eps_over_R')) for lab in LOW], '.3f') + ". So in 12.7-12.9 the SPH "
      f"kernel at 2R is wider than R200c, and $x_h$ reaches {R.v(PREP, J1a('12.7-12.9', 'x_h'), '.2f')} R200c: inside "
      "R200c only the integral is measured.")
    inn = lambda lab, k: ("derived_for_report", "innermost_shell_low_bins", lab, k)   # noqa: E731
    w("- **Central spike.** The innermost fine shell, x < 0.025, is not merged (it already holds "
      + R.seq(POST, lambda b: inn(b, 'stacked_count_a2ii'), LOW, '.0f') + " stacked A2-ii particles). Its mean E is "
      + R.seq(POST, lambda b: inn(b, 'E_over_B16'), LOW, '.1f') + " times B16 and its mean P "
      + R.seq(POST, lambda b: inn(b, 'P_over_B12'), LOW, '.1f') + " times B12, but it lies inside the softening "
      "limit and holds only " + R.seq(POST, lambda b: inn(b, 'share_of_Ne_lt_R200c'), LOW, '.1f', 100)
      + f"% of N_e(<R200c) [{POST}: derived_for_report.innermost_shell_low_bins].")
    w("- **Where the fallback is wrong.** B16 puts " + R.seq(PREP, lambda b: J1a(b, 'share_of_Ne_lt4R_inside_R200c', 'B16'), LOW, '.0f', 100)
      + "% of its electrons inside 4R within R200c, the A2-ii stacks " + R.seq(PREP, lambda b: J1a(b, 'share_of_Ne_lt4R_inside_R200c', 'A2ii'), LOW, '.0f', 100)
      + "%: the measured low-mass gas is far more extended than B16, so the fallback overestimates the electrons "
      "inside R200c (ratios in the table), while inside 4R the A2-ii/B16 ratios are only "
      + R.rng([(PREP, J1a(lab, 'Ne_lt4R_ratio_A2ii_over_B16')) for lab in LOW], '.2f') + ".")
    w("")
    w(f"Per-bin mean DM of the newly calibrated bins, measured (C14) / fallback (C10), and the same for the per-bin "
      f"DM variance [{POST}: per_bin_measured_vs_fallback]:")
    w("")
    w("| definition | mean DM ratio, 12.7-12.9 / 12.9-13.1 / 13.1-13.3 / 13.3-13.5 | variance ratio, same bins | "
      "low-bin share of the total mean DM, C10 -> C14 |")
    w("|---|---|---|---|")
    pm = lambda d: (lambda b: ("per_bin_measured_vs_fallback", d, b, "ratio_mean"))       # noqa: E731
    pv = lambda d: (lambda b: ("per_bin_measured_vs_fallback", d, b, "ratio_var"))        # noqa: E731
    for d in DEFS:
        sl = ("per_bin_measured_vs_fallback", d, "sum_low_bins")
        w(f"| {d} | {R.seq(POST, pm(d), LOW, '.2f')} | {R.seq(POST, pv(d), LOW, '.2f')} | "
          f"{R.v(POST, sl + ('share_of_C10_total_mean_from_low_bins_fallback',), '.1f', 100)}% -> "
          f"{R.v(POST, sl + ('share_of_C14_total_mean_from_low_bins_measured',), '.1f', 100)}% |")
    w("")
    w("For D3 the fallback column is the published B16 profile of all gas, so the D3 ratios also measure bound-only "
      "versus all gas in those bins.")
    w("")
    # ------------------------------------------------------------------------------------------------ J.4
    w("### J.4 DM statistics per definition (C14, treatment T1)")
    w("")
    w(f"Values with the ratio to D0 in brackets [{POST}: definitions, definitions_ratio_to_D0_C14; {TS}: pdf, evaluations]. "
      "DM in pc cm^-3 (observer frame, z_s = 1); variance in (pc cm^-3)^2; DM-field C_ell in (pc cm^-3)^2; y x DM in "
      "pc cm^-3.")
    w("")
    w("| statistic | D0: A2-ii to 4R | D1: A2-ii, r <= R200c | D2: A2-ii, r <= 2R200c | D3: host-bound to 4R |")
    w("|---|---|---|---|---|")
    rows = (("mean DM", "dm_mean", ".1f"), ("variance", "dm_var", ",.0f"), ("Q16", "q16", ".1f"), ("Q50 (median)", "q50", ".1f"),
            ("Q84", "q84", ".1f"), ("Q97.5", "q975", ".1f"), ("Q99.9", "q999", ".1f"),
            ("DM-field C_ell, ell = 1000", "dmcltot_1000", ".4f"), ("DM-field C_ell, ell = 3000", "dmcltot_3000", ".5f"),
            ("y x DM C_ell, ell = 1000", "ydmtot_1000", ".3e"))
    atoms = R.load(POST)["quantiles_in_zero_atom"]
    for name, key, fmt in rows:
        cells = []
        for d in DEFS:
            if key in atoms[f"C14|{d}"]:
                cells.append("0 (inside the atom)")
                continue
            val = R.v(POST, pdef(f"C14|{d}", key), fmt)
            cells.append(val if d == "D0" else f"{val} ({R.v(POST, rat(d, key), '.3f')})")
        w(f"| {name} | " + " | ".join(cells) + " |")
    pa = []
    for d in DEFS:
        val = R.get(POST, pdef(f"C14|{d}", "p_atom"))
        pa.append(R.v(POST, pdef(f"C14|{d}", "p_atom"), ".1e" if val < 1e-3 else ".4f"))
    w("| P(DM < 0.0625), no-crossing atom | " + " | ".join(pa) + " |")
    w("")
    w("- The quantile ratios rise towards 1 from Q16 to Q99.9: the high tail comes from sightlines through halo cores, "
      "which all definitions share, while the low and middle quantiles come from the outskirts that D1-D3 remove.")
    w(f"- D3/D0 for the DM-field C_ell rises from {R.v(POST, rat('D3', 'dmcltot_100'), '.3f')} at ell = 100 to "
      f"{R.v(POST, rat('D3', 'dmcltot_3000'), '.3f')} at ell = 3000; D1/D0 from {R.v(POST, rat('D1', 'dmcltot_100'), '.3f')} "
      f"to {R.v(POST, rat('D1', 'dmcltot_3000'), '.3f')}: small scales are dominated by the inner halo.")
    w("")
    w(f"**Replacing the fallback** (C14/C10, same definition) [{POST}: measured_over_fallback_C14_over_C10]:")
    w("")
    w("| statistic | D0 | D1 | D2 | D3 |")
    w("|---|---|---|---|---|")
    for name, key in (("mean DM", "dm_mean"), ("variance", "dm_var"), ("Q16", "q16"), ("Q50", "q50"), ("Q84", "q84"),
                      ("Q97.5", "q975"), ("Q99.9", "q999"), ("DM C_ell, ell = 1000", "dmcltot_1000"),
                      ("DM C_ell, ell = 3000", "dmcltot_3000")):
        cells = []
        for d in DEFS:
            if key in atoms[f"C14|{d}"] and key in atoms[f"C10|{d}"]:
                cells.append("both in the atom")
            else:
                cells.append(R.v(POST, mvf(d, key), ".3f"))
        w(f"| {name} | " + " | ".join(cells) + " |")
    w("")
    w(f"- For D3 the C10 reference paints the low-mass bins with the all-gas B16 fallback, so D3's C14/C10 mixes the "
      "calibration with the bound-only selection.")
    w(f"- Against the published profiles everywhere (C0), the full calibration changes D0's mean DM by "
      f"{R.v(POST, ('measured_over_published_C14_over_C0', 'D0', 'dm_mean'), '+.1f', **pct)}% and its variance by "
      f"{R.v(POST, ('measured_over_published_C14_over_C0', 'D0', 'dm_var'), '+.1f', **pct)}%, and D1's mean by "
      f"{R.v(POST, ('measured_over_published_C14_over_C0', 'D1', 'dm_mean'), '+.1f', **pct)}% "
      f"[{POST}: measured_over_published_C14_over_C0].")
    w("")
    # ------------------------------------------------------------------------------------------------ J.5
    w("### J.5 Where each DM statistic comes from (J1d)")
    w("")
    w(f"Importance shares by radial band (sums of $I_O$ over all bins, in %), C14, T1 [{SS}: bands]. Quantile maps are "
      "the Mecke per-halo-subsample maps with the whole DM = 0 atom (J.2); they sum to 1 within subsample noise (G4), "
      "except D1's Q16, which lies inside the atom.")
    w("")
    w("| definition | statistic | <0.233R | 0.233-0.519R | 0.519-1R | 1-2R | 2-4R |")
    w("|---|---|---|---|---|---|---|")
    bands = ("<0.233R", "0.233-0.519R", "0.519-1R", "1-2R", "2-4R")
    for d in DEFS:
        for name, lab in (("dm_mean@scalar", "mean"), ("dm_var@scalar", "variance"), ("dm_q50@scalar", "Q50"),
                          ("dm_q84@scalar", "Q84"), ("dmcltot@1000", "C_ell 1000")):
            w(f"| {d} | {lab} | " + " | ".join(R.v(SS, bandkey(d, name, bb), ".1f", 100) for bb in bands) + " |")
    w("")
    w(f"Radii (in R200c) where the cumulative truncation curve reaches 50% and 90%, and the loss when every calibrated "
      f"halo is cut at 1 R200c [{SS}: radii; quantiles from the bin-mean PDF truncation passes 'truthmean']:")
    w("")
    w("| definition | mean: R50 / R90 / loss at 1R | variance: R50 / R90 | Q50: R50 / R90 | Q84: R50 / R90 | "
      "C_ell 1000: R50 / R90 |")
    w("|---|---|---|---|---|---|")
    for d in DEFS:
        loss1 = R.v(SS, radk(d, 'dm_mean@scalar', 'loss_at_1R'), '.1f', 100)
        w(f"| {d} | {R.v(SS, radk(d, 'dm_mean@scalar', 'R50'), '.2f')} / {R.v(SS, radk(d, 'dm_mean@scalar', 'R90'), '.2f')} / {loss1}% | "
          f"{R.v(SS, radk(d, 'dm_var@scalar', 'R50'), '.2f')} / {R.v(SS, radk(d, 'dm_var@scalar', 'R90'), '.2f')} | "
          f"{R.v(SS, radk(d, 'dm_q50@truthmean', 'R50'), '.2f')} / {R.v(SS, radk(d, 'dm_q50@truthmean', 'R90'), '.2f')} | "
          f"{R.v(SS, radk(d, 'dm_q84@truthmean', 'R50'), '.2f')} / {R.v(SS, radk(d, 'dm_q84@truthmean', 'R90'), '.2f')} | "
          f"{R.v(SS, radk(d, 'dmcltot@1000', 'R50'), '.2f')} / {R.v(SS, radk(d, 'dmcltot@1000', 'R90'), '.2f')} |")
    w("")
    w(f"- D0's median DM loses {R.v(SS, radk('D0', 'dm_q50@truthmean', 'loss_at_1R'), '.0f', 100)}% when the haloes are "
      f"cut at 1R and {R.v(SS, radk('D0', 'dm_q50@truthmean', 'loss_at_2R'), '.0f', 100)}% when cut at 2R; D3's median "
      f"loses {R.v(SS, radk('D3', 'dm_q50@truthmean', 'loss_at_1R'), '.0f', 100)}% and "
      f"{R.v(SS, radk('D3', 'dm_q50@truthmean', 'loss_at_2R'), '.1f', 100)}%.")
    w("")
    w(f"Share of each statistic carried by the newly calibrated bins 12.7-13.5 (importance shares, %) [{SS}: shares]:")
    w("")
    w("| definition | mean | variance | Q16 | Q50 | Q84 | Q97.5 | C_ell 1000 | kSZ 1h, ell = 1000 |")
    w("|---|---|---|---|---|---|---|---|---|")
    for d in DEFS:
        cells = []
        for name in ("dm_mean@scalar", "dm_var@scalar", "dm_q16@scalar", "dm_q50@scalar", "dm_q84@scalar",
                     "dm_q975@scalar", "dmcltot@1000", "ksz1h@1000"):
            if d == "D1" and name == "dm_q16@scalar":
                cells.append("atom")
            else:
                cells.append(R.v(SS, lowshare(d, name), ".1f", 100))
        w(f"| {d} | " + " | ".join(cells) + " |")
    w("")
    w(f"Per mass bin (each bin's own compound-Poisson terms, additive over bins) [{POST}: per_bin_C14; also "
      "`post/J_per_bin.csv`]:")
    w("")
    w("| bin | D0 mean DM (pc cm^-3) | D0 share of mean | D1/D0 | D2/D0 | D3/D0 | D0 share of variance | variance D1/D0 | "
      "variance D3/D0 |")
    w("|---|---|---|---|---|---|---|---|---|")
    for i, lab in enumerate(CAL):
        pbk = lambda d, k: ("per_bin_C14", d, k, i)                 # noqa: E731
        w(f"| {lab} | {R.v(POST, pbk('D0', 'dm_mean'), '.2f')} | {R.v(POST, pbk('D0', 'dm_mean_share'), '.1f', 100)}% | "
          f"{R.v(POST, pbk('D1', 'dm_mean_ratio_to_D0'), '.3f')} | {R.v(POST, pbk('D2', 'dm_mean_ratio_to_D0'), '.3f')} | "
          f"{R.v(POST, pbk('D3', 'dm_mean_ratio_to_D0'), '.3f')} | {R.v(POST, pbk('D0', 'dm_var_share'), '.1f', 100)}% | "
          f"{R.v(POST, pbk('D1', 'dm_var_ratio_to_D0'), '.3f')} | {R.v(POST, pbk('D3', 'dm_var_ratio_to_D0'), '.3f')} |")
    w("")
    w(f"- The D1/D0 mean-DM ratio rises from {R.v(POST, ('per_bin_C14', 'D1', 'dm_mean_ratio_to_D0', 0), '.3f')} in "
      f"12.7-12.9 to {R.agg('max', [(POST, ('per_bin_C14', 'D1', 'dm_mean_ratio_to_D0', i)) for i in range(14)], '.3f')} "
      "at the top: the low-mass haloes' gas lies mostly outside R200c.")
    w("")
    w(f"Profile tolerance (J.2), minimum over the inside of each definition, fractional error per 0.1 dex that moves "
      f"the statistic by 1% [{SS}: tolerance]:")
    w("")
    w("| definition | mean: tol at x | variance: tol at x | Q50: tol at x |")
    w("|---|---|---|---|")
    tk = lambda d, n, k: ("tolerance", f"{d}__{n}", k)                 # noqa: E731
    for d in DEFS:
        w(f"| {d} | " + " | ".join(
            f"{R.v(SS, tk(d, n, 'min_tol_per_0.1dex'), '.3f')} at {R.v(SS, tk(d, n, 'at_x'), '.2f')}"
            for n in ("dm_mean@scalar", "dm_var@scalar", "dm_q50@scalar")) + " |")
    w("")
    w(f"D0's mean and median are most sensitive in its outermost fine shell (x = "
      f"{R.v(SS, tk('D0', 'dm_mean@scalar', 'at_x'), '.2f')}), i.e. to field gas; the most sensitive radii of D1-D3 lie "
      f"at {R.rng([(SS, tk(d, n, 'at_x')) for d in ('D1', 'D2', 'D3') for n in ('dm_mean@scalar', 'dm_var@scalar', 'dm_q50@scalar')], '.2f')} R200c.")
    w("")
    # ------------------------------------------------------------------------------------------------ J.6
    w("### J.6 Treatment of the unresolved cores (T2, T3, S82)")
    w("")
    wc = ("derived_for_report", "treatment_largest_change")
    w(f"Relative change of every DM statistic (mean, variance, Q16-Q99.9, DM-field C_ell, y x DM) against T1, C14 "
      f"[{POST}: treatment_sensitivity_rel_change, derived_for_report.treatment_largest_change]:")
    w(f"- the largest change is {R.v(POST, wc + ('change',), '+.3f', 100)}% (D1 variance under T2); "
      f"{R.v(POST, ('derived_for_report', 'treatment_changes_above_1pct_count'), '.0f')} changes exceed 1%;")
    bv = lambda t: ("derived_for_report", "treatment_largest_change_by_variant", t, "abs_change")   # noqa: E731
    w(f"- largest absolute change per variant: T2 {R.v(POST, bv('T2'), '.2f', 100)}%, T3 (raw fine shells) "
      f"{R.v(POST, bv('T3'), '.3f', 100)}%, S82 (only haloes above the painting threshold in 12.7-12.9) "
      f"{R.v(POST, bv('S82'), '.3f', 100)}%.")
    w("")
    w("**What this does and does not test.** T1, T2 and T3 all keep each halo's measured electron count inside $x_h$; "
      "they differ only in how that count is distributed inside $x_h$ (the core shape). The result is that the DM "
      "statistics do not depend on the core shape below $x_h$. It does not test whether the measured low-mass gas "
      "content, N_e(<x_h) and N_e(<R200c), or its split across R200c, is converged with mass resolution. That split is "
      f"what drives D1: in 12.7-12.9 $x_h$ = {R.v(PREP, J1a('12.7-12.9', 'x_h'), '.2f')} R200c and the median 2h at 2R is "
      f"{R.v(PREP, J1a('12.7-12.9', 'median_2h_over_R200c_at_2R'), '.2f')} R200c, and the low-mass bins carry "
      f"{R.v(POST, ('per_bin_measured_vs_fallback', 'D1', 'sum_low_bins', 'share_of_C14_total_mean_from_low_bins_measured'), '.1f', 100)}% "
      f"of D1's mean DM in C14 (D1 mean changed by {R.v(POST, mvf('D1', 'dm_mean'), '+.1f', **pct)}% against the "
      "fallback). The mean DM is unchanged by construction under T2 "
      f"({R.v(POST, ('treatment_sensitivity_rel_change', 'D1|T2', 'dm_mean'), '+.1e')} relative for D1). A resolution test "
      "needs a higher-resolution run (J.8).")
    w("")
    # ------------------------------------------------------------------------------------------------ J.7
    w("### J.7 kSZ with host-bound gas only (exact, Q3)")
    w("")
    w(f"Same recipe as D0 (catalogue velocities for the 1-halo term, linear approximate 2-halo term); only the gas "
      f"selection changes. $D_\\ell$ in muK^2, C14 [{POST}: kSZ_Dl_uK2_C14, kSZ_D3_over_D0]:")
    w("")
    w("| ell | D0 1h | D3 1h | D3/D0 1h | D0 2h | D3 2h | D3/D0 2h | D0 total | D3 total | D3/D0 total | "
      "D3/D0 1h in C10 |")
    w("|---|---|---|---|---|---|---|---|---|---|---|")
    dl = lambda d, k: ("kSZ_Dl_uK2_C14", d, k)                     # noqa: E731
    for e in ells:
        w(f"| {e} | {R.v(POST, dl('D0', f'ksz1h_{e}'), '.4f')} | {R.v(POST, dl('D3', f'ksz1h_{e}'), '.4f')} | "
          f"{R.v(POST, kz('C14', f'ksz1h_{e}'), '.3f')} | {R.v(POST, dl('D0', f'ksz2h_{e}'), '.4f')} | "
          f"{R.v(POST, dl('D3', f'ksz2h_{e}'), '.4f')} | {R.v(POST, kz('C14', f'ksz2h_{e}'), '.3f')} | "
          f"{R.v(POST, dl('D0', f'ksztot_{e}'), '.4f')} | {R.v(POST, dl('D3', f'ksztot_{e}'), '.4f')} | "
          f"{R.v(POST, kz('C14', f'ksztot_{e}'), '.3f')} | {R.v(POST, kz('C10', f'ksz1h_{e}'), '.3f')} |")
    w("")
    w("- Bound-only gas removes "
      + " / ".join(R.v(POST, kz("C14", f"ksz1h_{e}"), ".0f", -100, 1.0) for e in ells)
      + "% of the 1-halo kSZ and " + " / ".join(R.v(POST, kz("C14", f"ksz2h_{e}"), ".0f", -100, 1.0) for e in ells)
      + "% of the 2-halo kSZ at ell = 100 / 300 / 1000 / 3000: the 2-halo term, which weights the total electron count "
      "of each halo, loses most.")
    w("- The bound-only loss is larger in C14 than in C10: in C10 the low-mass haloes are painted with the same "
      "all-gas B16 fallback in D0 and D3, which dilutes the ratio; in C14 they get their own bound-gas stacks, which "
      "hold little gas (f_e(<1R) bound " + R.seq(PREP, lambda b: J1a(b, 'Ne_lt1R_over_fbM', 'host_bound'), LOW, '.3f')
      + " in 12.7-12.9 / 12.9-13.1 / 13.1-13.3 / 13.3-13.5).")
    w("")
    w(f"First order vs exact, host-bound / all gas, C10 [{POST}: first_order_vs_exact_C10_hostbound_over_allgas]:")
    w("")
    w("| statistic | first-order change | exact change |")
    w("|---|---|---|")
    for k, lab in (("kSZ_1h_ell100", "kSZ 1h, ell = 100"), ("kSZ_1h_ell1000", "kSZ 1h, ell = 1000"),
                   ("kSZ_1h_ell3000", "kSZ 1h, ell = 3000"), ("DM_mean", "mean DM"), ("DM_var", "DM variance"),
                   ("DM_q16", "DM Q16"), ("DM_q50", "DM Q50"), ("DM_q84", "DM Q84"),
                   ("DM_Cl_total_ell1000", "DM-field C_ell, ell = 1000")):
        w(f"| {lab} | {R.v(POST, fo(k, 'first_order_pct'), '+.1f')}% | {R.v(POST, fo(k, 'exact_pct'), '+.1f')}% |")
    w("")
    w("A bound-only kSZ fit would need only the compact bound profile (R90 of the kSZ 1-halo term at ell = 1000: "
      f"{R.v(SS, radk('D3', 'ksz1h@1000', 'R90'), '.2f')} R200c for D3 against "
      f"{R.v(SS, radk('D0', 'ksz1h@1000', 'R90'), '.2f')} R200c for D0), but it would describe a different signal; the "
      "kSZ recipe stays A2-ii to 4R.")
    w("")
    # ------------------------------------------------------------------------------------------------ J.8
    w("### J.8 FLAMINGO L1_m8 option (cost ESTIMATE only, nothing downloaded)")
    w("")
    l8 = ("L1_m8_cost_ESTIMATE",)
    cases = list(R.load(POST)["L1_m8_cost_ESTIMATE"]["cases_full_snapshot"])
    w(f"[{POST}: L1_m8_cost_ESTIMATE; COSMA inventory and audit in `stacking_choices_20261014/research/`]")
    w(f"- L1_m8 has {R.v(POST, l8 + ('gas_particle_mass_ratio_m9_over_m8',), '.0f')} times better gas mass resolution; "
      "ESTIMATE: 8 times more particles per shell at fixed mass, and $x_h$ roughly halves.")
    w(f"- **Reduced z = 0 snapshot**: {R.v(POST, l8 + ('reduced_z0', 'gas_rows_header'), ',.0f')} gas rows, about "
      f"{R.v(POST, l8 + ('reduced_z0', 'GB_at_64B'), '.0f')} GB at 64 B per row. It holds only haloes >= 10^13 Msun "
      "and only r < R100c, about 1.35 R200c (ESTIMATE of R100c/R200c). That covers the whole D1 extent (r <= R200c), "
      "and 1-1.35 R200c of D0/D2/D3, for 13.1-13.3, 13.3-13.5 and the upper half of 12.9-13.1. It can therefore test "
      "whether N_e(<R200c) and the split of the gas across R200c are resolution-converged in those bins. It does not "
      "cover 12.7-13.0, nor D0/D2/D3 beyond about 1.35 R200c.")
    w(f"- **Full z = 0 snapshot**: {R.v(POST, l8 + ('L1_m8_z0_snapshot_TB',), '.1f')} TB in total; the 4R cells of "
      f"the low-mass sample alone are about {R.v(POST, l8 + ('cases_full_snapshot', cases[0], 'GB'), '.0f')} GB "
      f"({R.v(POST, l8 + ('cases_full_snapshot', cases[0], 'hours', 'cosma_timed_8.67_MBps'), '.0f')} h at the timed "
      f"COSMA rate of 8.67 MB/s; {R.v(POST, l8 + ('cases_full_snapshot', cases[0], 'hours', 'local_0.31_MBps'), '.0f')} h "
      f"over the local link), and all painted bins about {R.v(POST, l8 + ('cases_full_snapshot', cases[1], 'GB'), '.0f')} GB "
      "(ESTIMATE; L1_m8's cell layout was not checked). Not on maas.")
    w("")
    # ------------------------------------------------------------------------------------------------ figures
    w("### J.9 Figures (`figures/`, PNG and SVG)")
    w("")
    w("- `J1_dm_pdf_definitions`: (a) halo-DM PDF per ln DM for D0-D3 with P(DM < 0.0625), (b) cumulative "
      "distributions, (c) every DM statistic relative to D0.")
    w("- `J1_importance_vs_radius`: importance per unit ln x for each definition and DM statistic (mean, variance, "
      "Q50, Q99.9), by mass group (five groups, light to dark blue, each with its own dash pattern); each quantile "
      "panel states the sum of its Mecke map with the whole DM = 0 atom (per-halo subsample map, "
      "[results/sens.npz: imp__D*__dm_q*]) and of the truth-mean map [post/mecke_update.json].")
    w("- `J1_lowmass_stacks_vs_fallback`: low-mass A2-ii E and P stacks (T1 merged and T3 raw fine shells) and the "
      "host-bound E stack (T1) against the published B16/B12, with x_h (grey band) and the gravitational-softening "
      "limit 4 epsilon/R200c (dotted vertical line).")
    w("- `J1_bound_vs_a2ii_profiles`: host-bound vs A2-ii electron profiles per mass bin with the cosmic mean.")
    w("- `J1_ksz_bound_vs_a2ii`: kSZ D_ell for D0 and D3 and their ratio (C14 and C10, first-order markers).")
    w("- `J1_cumulative_vs_radius`: cumulative truncation curves C_O(R_t) per definition.")
    w("- Supplementary: `J1_mass_shares`, `J1_treatment_sensitivity`, `J1_measured_vs_fallback`.")
    w("")
    # ------------------------------------------------------------------------------------------------ recommendations
    w("### J.10 Recommendations")
    w("")
    recs = recommendations(R)
    for i, rr in enumerate(recs, 1):
        w(f"{i}. {rr}")
    w("")
    return "\n".join(L) + "\n", recs


def recommendations(R):
    pct = dict(scale=100.0, minus=1.0)
    out = []
    out.append("Paint every halo above the 6.6e12 Msun threshold with calibrated stacks (C14). The fallback is wrong "
               "in shape in 12.7-13.5: inside R200c the measured electrons are only "
               + R.rng([(PREP, ("J1a_by_bin", lab, "Ne_lt1R_ratio_A2ii_over_B16")) for lab in LOW], ".2f")
               + " of B16, and these bins carry "
               + R.v(SS, ("shares", "D0__dm_mean@scalar", "low_mass_12.7-13.5"), ".1f", 100) + "% of D0's mean DM.")
    out.append("For a halo-only DM comparable to Konietzka's TNG within-R200 catalogue and to HalfDome's "
               "SphericalChordDMProfile, use D1 (A2-ii gas in the 3D sphere r <= R200c, chord-limited line of sight): "
               "mean " + R.v(POST, ("definitions", "C14|D1", "dm_mean"), ".1f") + " pc cm^-3, median "
               + R.v(POST, ("definitions", "C14|D1", "q50"), ".1f") + " pc cm^-3, and "
               + R.v(TS, ("crossings", "1R", "p_no_crossing"), ".1f", 100) + "% of sightlines to z_s = 1 with no halo "
               "contribution.")
    out.append("Do not use D0 (A2-ii to 4R) or the 5R extension as a 'halo component': "
               + R.v(SS, ("bands", "D0__dm_mean@scalar", "2-4R"), ".1f", 100) + "% of D0's mean DM comes from 2-4R, "
               "where the gas is mostly unbound field gas. Use D0 only for total halo-plus-local-field budgets; D2 "
               "(r <= 2 R200c, mean " + R.v(POST, ("definitions_ratio_to_D0_C14", "D2", "dm_mean"), ".3f")
               + " x D0) is the intermediate choice.")
    out.append("For the physically bound halo gas use D3 (host-bound to 4R): mean "
               + R.v(POST, ("definitions_ratio_to_D0_C14", "D3", "dm_mean"), ".3f") + " x D0, compact (R90 of the mean "
               + R.v(SS, ("radii", "D3", "dm_mean@scalar", "R90"), ".2f") + " R200c); truncating it at 4R is safe "
               "(bound gas beyond 4R <= "
               + R.agg("max", [(PREP, ("J1a_by_bin", lab, "host_bound_beyond_4R_ESTIMATE", "fraction_of_hb_lt4R"))
                               for lab in CAL], ".1f", 100) + "% of that inside 4R, ESTIMATE).")
    out.append("The treatment of the unresolved cores does not matter for DM: T1/T2/T3 and S82 move every DM statistic "
               "by at most " + R.v(POST, ("derived_for_report", "treatment_largest_change", "abs_change"), ".2f", 100)
               + "%. But these treatments keep each halo's measured gas inside x_h, so they test only the core shape, "
               "not whether the low-mass gas content and its split across R200c are converged with resolution. D1 "
               "depends on that split (x_h = " + R.v(PREP, ("J1a_by_bin", "12.7-12.9", "x_h"), ".2f")
               + " R200c in 12.7-12.9; low-mass bins carry "
               + R.v(POST, ("per_bin_measured_vs_fallback", "D1", "sum_low_bins",
                            "share_of_C14_total_mean_from_low_bins_measured"), ".1f", 100)
               + "% of D1's mean DM). Before relying on D1 below 10^13.5, compare N_e(<R200c) and N_e(<1.35 R200c) of "
               "the 10^13-10^13.5 stacks with the L1_m8 reduced z = 0 snapshot (about "
               + R.v(POST, ("L1_m8_cost_ESTIMATE", "reduced_z0", "GB_at_64B"), ".0f")
               + " GB, ESTIMATE), which covers the whole D1 extent there; 12.7-13.0 would need the full snapshot.")
    out.append("Keep the kSZ recipe (A2-ii to 4R). Bound-only gas would remove "
               + R.v(POST, ("kSZ_D3_over_D0", "C14", "ksz1h_1000"), ".0f", -100, 1.0)
               + "% of the 1-halo kSZ at ell = 1000 and "
               + R.v(POST, ("kSZ_D3_over_D0", "C14", "ksz2h_1000"), ".0f", -100, 1.0)
               + "% of the 2-halo term; the earlier first-order estimate overstated the 1-halo change (exact "
               + R.v(POST, ("first_order_vs_exact_C10_hostbound_over_allgas", "kSZ_1h_ell100", "exact_pct"), "+.1f")
               + "% vs first-order "
               + R.v(POST, ("first_order_vs_exact_C10_hostbound_over_allgas", "kSZ_1h_ell100", "first_order_pct"), "+.1f")
               + "% at ell = 100, C10).")
    return out


def key_numbers(R):
    R.context = "key_numbers"
    K = []

    def add(name, f, keys, fmt, scale=1.0, minus=0.0, unit=""):
        K.append({"name": name, "value": R.v(f, keys, fmt, scale, minus) + unit, "source": f"{f}: {'.'.join(map(str, keys))}"})
    for d, lab in (("D0", "A2-ii to 4R"), ("D1", "A2-ii, r <= R200c"), ("D2", "A2-ii, r <= 2R200c"), ("D3", "host-bound to 4R")):
        add(f"mean halo DM, {d} ({lab}), C14 [pc cm^-3]", POST, ("definitions", f"C14|{d}", "dm_mean"), ".1f")
        add(f"median halo DM Q50, {d}, C14 [pc cm^-3]", POST, ("definitions", f"C14|{d}", "q50"), ".1f")
    for d in ("D1", "D2", "D3"):
        add(f"{d}/D0 mean DM", POST, ("definitions_ratio_to_D0_C14", d, "dm_mean"), ".3f")
        add(f"{d}/D0 DM variance", POST, ("definitions_ratio_to_D0_C14", d, "dm_var"), ".3f")
    add("P(no R200c crossing to z_s = 1) [%]", TS, ("crossings", "1R", "p_no_crossing"), ".1f", 100)
    add("D0 mean DM change from calibrating 12.7-13.5 (C14/C10 - 1) [%]", POST,
        ("measured_over_fallback_C14_over_C10", "D0", "dm_mean"), "+.1f", 100, 1)
    add("D0 DM variance change (C14/C10 - 1) [%]", POST, ("measured_over_fallback_C14_over_C10", "D0", "dm_var"), "+.1f", 100, 1)
    add("D1 mean DM change (C14/C10 - 1) [%]", POST, ("measured_over_fallback_C14_over_C10", "D1", "dm_mean"), "+.1f", 100, 1)
    add("share of D0 mean DM from 12.7-13.5 [%]", SS, ("shares", "D0__dm_mean@scalar", "low_mass_12.7-13.5"), ".1f", 100)
    add("share of D0 mean DM from 2-4 R200c [%]", SS, ("bands", "D0__dm_mean@scalar", "2-4R"), ".1f", 100)
    for lab in LOW:
        add(f"N_e(<R200c) A2-ii/B16, {lab}", PREP, ("J1a_by_bin", lab, "Ne_lt1R_ratio_A2ii_over_B16"), ".3f")
        add(f"Y(<R200c) A2-ii/B12, {lab}", PREP, ("J1a_by_bin", lab, "Y_lt1R_ratio_A2ii_over_B12"), ".3f")
    for e in (100, 300, 1000, 3000):
        add(f"kSZ 1h host-bound/A2-ii, C14, ell = {e}", POST, ("kSZ_D3_over_D0", "C14", f"ksz1h_{e}"), ".3f")
    for e in (100, 1000, 3000):
        add(f"kSZ 2h host-bound/A2-ii, C14, ell = {e}", POST, ("kSZ_D3_over_D0", "C14", f"ksz2h_{e}"), ".3f")
    add("largest DM change from the core treatment (D1 variance, T2) [%]", POST,
        ("derived_for_report", "treatment_largest_change", "change"), "+.3f", 100)
    add("maas wall time [s]", DONE, ("wall_seconds",), ",.0f")
    add("maas peak tree RSS [GiB]", DONE, ("peak_tree_rss_GiB",), ".2f")
    add("D1 Q50 Mecke map sum, whole DM = 0 atom (truth mean)", MUPD,
        ("mecke_sum_rule_minus_1", "D1", "whole|C14|D1|T1|truthmean", 1), ".3f", 1.0, -1.0)
    add("D1 Q50 Mecke map sum, J1's half-atom formula (truth mean)", MUPD,
        ("mecke_sum_rule_minus_1", "D1", "half|C14|D1|T1|truthmean", 1), ".3f", 1.0, -1.0)
    add("D1 Q50 outer-slab check (Mecke / exact), whole atom", MUPD,
        ("outer_slab_first_order", "D1", "whole_atom", "q50", 0, "ratio"), ".3f")
    add("maas run 2 (Mecke revision) wall time [s]", MDONE, ("wall_seconds",), ",.0f")
    return K


def main():
    R = Reg()
    text, recs = build(R)
    K = key_numbers(R)
    missing = [it["text"] for it in R.items if it["in_report"] and it["text"] not in text]
    out = HERE / "post"
    (out / "J1_report.txt").write_text(text)
    (out / "report_numbers.json").write_text(json.dumps({"n": len(R.items), "items": R.items}, indent=1) + "\n")
    (out / "key_numbers.json").write_text(json.dumps(K, indent=1) + "\n")
    (out / "recommendations.json").write_text(json.dumps(recs, indent=1) + "\n")
    print(f"wrote post/J1_report.txt ({len(text)} chars), {len(R.items)} registered numbers, "
          f"{len(K)} key numbers; registered texts not found in the report: {len(missing)}")


if __name__ == "__main__":
    main()
