# BE: electron pressure from the electron-density law and a temperature model (26 Nov 2026)

Question (user, 2026-11-26): following Schneider et al. 2025 (arXiv:2507.07892; Eq. 12 is their hot-gas density, Sec. 6
derives pressure from hydrostatic equilibrium + the Shaw+10 non-thermal fraction with no extra parameters), can the
pressure profiles be predicted from the density fit with a simple temperature profile, so that density + pressure need
fewer than 9 + 9 law parameters (+ the 2-halo amplitude)?

Two parts:
1. `maas/`: the full test on STRW maas (exact BD B32 residual set, fitted electron law, held-out outputs, 2-halo).
2. The quick look of the morning (top-level `look.py`, `tests.py`, `hse_hybrid.py`; measured electron stacks, unweighted).

Independent verification of part 1: PASSED (`maas/verify_independent/VERIFY.md`; details at the end of this file).
Step 3 (section 0): independent check PASSED (`maas/verify_independent/step3/VERIFY_STEP3.md`). The verifier wrote its own
code from the docstrings and ran it on maas.
- The existing fits' L_P, L_out4 and L_out6 agree to <= 2.5e-6 relative with this study and with the losses saved in the
  BD files. The outer RMS per band agrees to 1e-16.
- The joint and separate temperature-law models + 2-halo agree to <= 3e-7 (L_P) and 1e-5 (L_E).
- The split fit of the exponential temperature law is converged: refitting the amplitude with the law fixed changes it
  by 7.8e-8, and refitting the law lowers L_P + L_E by 1.3e-8.
- The parameter counts are confirmed.
- Not checked: RMS inside B32 by group, the extension ratios, the physicality numbers and the hydrostatic 2-halo fits.

## 0. Step 3: against the existing pressure fit with a free 2-halo amplitude (26 Nov, later)
User: "Have to be compared to best pressure fit (with A_2h free amplitude). Also include 2-halo for temperature. Use the
existing fit for pressure."

- References: the existing BD fits, evaluated with their saved parameters and no refit (`maas/code/be_compare.py`). Their
  recomputed L_P equals the saved L_in to <= 1.8e-15. The reference in the figures is the fiducial (user, 20 Nov):
  constant alpha, A_2h(z) free, amplitude from the shells beyond B32 to 6 R200c
  (BD `explore2h/corner_1h_vs_1h2h_20261120/runs/B_2h_rout6.json`). The table also has the alpha(z) fit of record
  and the 16-parameter mass-break law.
- Electron-law models with the same 2-halo term (Vikram+17 template, exclusion 0.75, ln A_2h = a + b lz + c lz^2) use
  the BD split procedure, alternated to convergence. The law (+ electron loss for joint fits) is fitted inside B32 and
  the amplitude on the outer shells.
- New temperature variant T7x: exponential outer decline exp[-(x/x_e)^c_e]. It replaces T7's (1 + x/x_t)^-c, whose
  exponent sat at its bound. Fitted c_e = 0.93, x_e = 0.77.

| model (2-halo amplitude from shells to 6 R200c) | parameters: electron + pressure + amplitude | L_P inside B32 | L_out to 6 R200c | RMS inside: 10^12.7-13.1 / 13.1-13.5 / 13.5-14.3 / >= 14.3 | RMS beyond B32: 1.5-2 / 2-3 / 3-4 / 4-6 R200c |
|---|---|---|---|---|---|
| existing pressure fit, constant alpha (fiducial) | 9 + 10 + 3 = 22 | 14.33 | 1869 | 0.581 / 0.269 / 0.072 / 0.088 | 0.090 / 0.097 / 0.117 / 0.116 |
| existing pressure fit, alpha(z) of record | 9 + 9 + 3 = 21 | 12.34 | 3736 | 0.599 / 0.280 / 0.067 / 0.073 | 0.183 / 0.177 / 0.138 / 0.186 |
| existing pressure fit with mass break (amplitude to 4 R200c) | 9 + 16 + 3 = 28 | 2.00 | 4119 | 0.148 / 0.089 / 0.035 / 0.031 | 0.157 / 0.176 / 0.188 / 0.249 |
| electron law x temperature law T7x + 2-halo, joint | 9 + 7 + 3 = 19 | 7.68 | 4279 | 0.247 / 0.081 / 0.072 / 0.067 | 0.207 / 0.186 / 0.141 / 0.115 |
| electron law x temperature law T7 + 2-halo, joint | 9 + 7 + 3 = 19 | 7.74 | 4559 | 0.247 / 0.081 / 0.072 / 0.067 | 0.214 / 0.200 / 0.147 / 0.119 |
| hydrostatic + core term + 2-halo, joint | 9 + 6 + 3 = 18 | 5.56 | 49273 | 0.227 / 0.096 / 0.060 / 0.065 | 0.887 / 0.307 / 0.464 / 0.563 |

Source: `maas/out/be_compare.json: results[name]`; all rows (amplitude to 4 R200c, separate fits, A_2h(z)) are in
`maas/tables_step3.md`. RMS = root mean square of ln(model/all-gas stack) over bins with >= 100 haloes. Beyond B32 the
RMS is over shells of all such bins, by the shell's outer edge.

Reading:
- Against the fiducial pressure fit, the joint temperature-law model is better inside B32 (L_P 7.68 vs 14.33) with
  three fewer parameters (16 + 2-halo vs 19 + 2-halo). The gain is mostly in groups (RMS 0.25 vs 0.58 below 10^13.1),
  and clusters improve slightly (0.067 vs 0.088).
- It is worse beyond B32: L_out 4279 vs 1869, RMS 0.21 / 0.19 / 0.14 at 1.5-2 / 2-3 / 3-4 R200c vs 0.09 / 0.10 / 0.12.
  The 4-6 R200c band is equal (0.115 vs 0.116).
- Beyond B32 it matches the alpha(z) fit of record (L_out 3736, RMS 0.18 / 0.18 / 0.14) and the mass-break law
  (L_out 4119).
- Its A_2h(z) follows the fiducial's (0.61 / 1.63 / 2.34 at z = 0 / 1 / 3 vs 0.52 / 1.69 / 2.82).
- Electron cost: L_E 15.82 vs 15.15 (+4 %).
- The mass-break pressure law is far better inside B32 (2.00) but costs 28 parameters with the electron law.
- The hydrostatic model with the 2-halo term is the best inside (5.56) and fails beyond B32 (L_out 49273).
- Figures: `maas/figures/compare_profiles_2h` (six mass x redshift points to 6 R200c, total and 1-halo part),
  `compare_rms_2h`, `compare_amplitude`.

## 1. Full test (maas)

### Setup
- Data: the BD residual set, read through `bd_mass.Data` (read-only BD code on maas): region 0 = the 1,494 residuals of the
  FLAMINGO B32 pressure fit of record (1,385 shells, 69 core integrals, 40 group integrals; 109 mass x redshift bins,
  10 outputs z = 0-3), region 1 = 4 R200c shells beyond B32, region 2 = 4-6 R200c cells. Targets: Y200-weighted all-gas
  stacks.
- New step-1 product `be_sums.npz`: Y200-weighted (same weights as the pressure targets) and unit-weight stacks of
  E = n_e mu_XGP/(rho_crit f_b) and P = P_e/P200c for the halo component, all gas and host-bound gas, per fold, from the
  production `fitting_inputs.hdf5` files listed in BD `code/flamingo_request.json`. It also holds the per-residual
  Y-weighted halo scatter of E and P.
- Identity checks (`out/be_fits.json: checks`): the pressure targets rebuilt from `be_sums.npz` equal the BD targets to
  2.2e-16. The recomputed pressure scatter s_i matches the record's `B32__s` with median relative difference 3.4e-4
  (maximum 0.054). The refitted 9-parameter pressure law gives L_P = 12.0797, and the BD record value is 12.08.
  The 2-halo split run reproduces the BD record: L_in = 12.34 and A_2h(z = 0) = 0.50 for alpha(z) held, 14.33 and 0.48 for
  one constant alpha (`out/be_twohalo.json`; BD `explore2h/FINAL_RECIPE.md` section 2).
- Models (definitions with every symbol in the docstring of `maas/code/be_fit.py`):
  - electron law (production B16 form): E = n0 (x/0.5)^-0.2 [1 + (x/0.5)^a_e]^(-(beta_e - 0.2)/a_e). The three quantities
    n0, a_e and beta_e each have the form exp(lnA + am ln(M/1e14 Msun) + az ln(1+z)), giving 9 parameters.
  - pressure from electrons: P = E tau / C_T, where tau = T_e/T200, T200 = 0.59 m_p G M200c/(2 R200c k_B) and C_T = 428.21.
    The temperature laws are:
    - T7: ln tau = ln T0 + am lm + az lz - c ln(1 + x/x_t) + ln[(x/x_cc)^2/(1 + (x/x_cc)^2)], with ln x_cc = c0 + bm lm
      (7 parameters; the last term is a cool-core / multiphase dip);
    - T5: T7 without the dip;
    - T8: T7 with a redshift term in x_cc.
  - hydrostatic: P = 0.5176 (1 - f_nth) K int_x^x_out E m / s^2 ds, where m is the NFW mass profile with Duffy+08
    c200c(M, z), P_tot(x_out) = 0, and f_nth = a (1+z)^b x^n capped at 0.95 (H3, 3 parameters). H6 adds the same core
    dip with ln x_cc = c0 + bm lm + bz lz (6 parameters).
  - fits: 'separate' = electron law from the electron loss alone, then the temperature parameters from the pressure loss;
    'joint' = all parameters from L_P + L_E.
  - losses: L_P = sum over region 0 of w_i ln^2(model/stack) with the record weights (mix_Y_lin_SO_90, sum 2701). L_E is
    the same with electron scatter weights (W0 b_i/s_E,i^2, normalised to the same total).

### Results
The full tables (generated from the JSON files, with source fields) are in `maas/tables_generated.md`. Key numbers:

| model | density + pressure parameters | L_P inside B32 | L_E | held-out L_P (10 outputs) | RMS ln P, 10^12.7-13.1 / 13.1-13.5 / 13.5-14.3 / >= 14.3 | Y200 model/data, 10^13, z = 0 |
|---|---|---|---|---|---|---|
| electron law + pressure law of record (alpha(z) held) | 9 + 9 = 18 | 12.08 | 15.15 | 12.91 | 0.592 / 0.277 / 0.066 / 0.073 | 2.03 |
| electron law + pressure law, one constant alpha | 9 + 10 = 19 | 14.09 | 15.15 | 15.72 | 0.574 / 0.266 / 0.072 / 0.088 | 1.95 |
| electron law x temperature law T7, separate | 9 + 7 = 16 | 9.50 | 15.15 | 13.44 | 0.241 / 0.092 / 0.080 / 0.073 | 0.89 |
| electron law x temperature law T7, joint | 9 + 7 = 16 | 7.75 | 15.88 | 8.75 | 0.245 / 0.084 / 0.072 / 0.067 | 0.94 |
| hydrostatic H3, separate | 9 + 3 = 12 | 13.24 | 15.15 | 15.21 | 0.539 / 0.295 / 0.089 / 0.067 | 1.24 |
| hydrostatic + core term H6, separate | 9 + 6 = 15 | 9.24 | 15.15 | 11.93 | 0.274 / 0.112 / 0.072 / 0.079 | 0.97 |
| hydrostatic + core term H6, joint | 9 + 6 = 15 | 5.66 | 16.24 | 6.64 | 0.198 / 0.102 / 0.064 / 0.065 | 1.01 |

Sources: `maas/out/be_fits.json: results[name].{L_P, L_E, bands_P, Y200}`; `maas/out/be_heldout.json: results[name].total_L_P_test`.
Held-out electron loss: electron law alone 19.39, joint T7 17.79, joint H6 17.92 (`be_heldout.json`).

1. Inside the cut, pressure from the electron law with 3-7 extra parameters beats the 9-parameter pressure law.
   - The joint fits halve the pressure loss (5.7-7.8 vs 12.1) and stay ahead on held-out outputs (6.6-8.8 vs 12.9).
   - The gain is in groups: RMS 0.20-0.25 vs 0.59 below 10^13.1, and Y200 within 11 % instead of x2 at 10^13, z = 0.
     Clusters are equal (RMS 0.065-0.08).
   - The cost is +5-7 % electron loss in sample (15.9-16.2 vs 15.15). Out of sample, the joint electron laws are better
     (held-out L_E 17.8-17.9 vs 19.4).
   - The model must reproduce the measured stack temperature, so groups gain from the cool-core term. Without it
     (T5, H3) groups stay as poor as with the pressure law.
2. At the high-redshift end, both new models predict held-out z = 2 and 3 worse than the pressure law
   (`be_heldout.json`: z = 2: 0.96 and 0.73 vs 0.68; z = 3: 0.46 and 0.50 vs 0.24).
3. Beyond B32 the two routes differ completely (`tables_generated.md` Tables 3 and 4; figure `pressure_profiles`).
   - Temperature law (T7): smooth and monotonic in all 109 bins, with log slope -5.5 [-7.5, -4.3] at 3-4 R200c (record law
     -5.8). For clusters the law / halo-component ratio is 0.95 / 0.68 / 0.44 at 1.5-2 / 2-3 / 3-4 R200c (record law
     0.84 / 0.60 / 0.41). With the 2-halo term, the outer RMS is 0.10-0.21 (record 0.09-0.22).
   - Hydrostatic (H6): the fitted non-thermal fraction rises like (1+z) x^2.7. It is 0.22 at R200c at z = 0, and
     reaches the 0.95 cap at about 1.3 R200c for z >= 1. Beyond B32 the pressure therefore drops by about a factor 10
     with a visible kink, to the level of the host-bound stack (0.12-0.25 of the halo component). Even with the 2-halo
     term, the outer RMS is 0.3-1.0.
   - A smooth uncapped form, f = 1 - exp(-a (1+z)^b x^n), drives the extended pressure to zero (only 75 of 109 bins
     decline monotonically).
   - Integrating the hydrostatic pressure to 3, 4 or 6 R200c changes L_P only from 8.94 to 9.43 (separate H6).
4. The implied non-thermal fraction of haloes >= 10^13.5 (figure `nonthermal_fraction`) rises from about 0 to 0.35-0.47
   at 1.3 R200c (z = 0), steeper than the Shaw+10 form used by Schneider+25. It is negative (-0.05 to -0.13) at small
   radius for z = 0 clusters, where the NFW Duffy+08 mass profile underestimates the central hydrostatic pressure.
5. For comparison: a richer pressure-only law, the BD smooth mass-break law (16 parameters), reaches L_P = 2.00 with the
   2-halo term (BD `explore2h/mass_weights/final/best_sbplzz_held.json: losses.best`). With the electron law that is
   9 + 16 = 25 parameters in total.

### Reading
- Within the "< 18 + 2-halo" budget, the candidate for painting is the joint electron law x temperature law (16 parameters).
  It is better than the pressure law of record inside the cut, on held-out outputs (except z = 2-3), and in groups. It
  extends smoothly to 4 R200c, near the record law.
- The hydrostatic route (15 parameters) is the best interior description and the most physical one, but as fitted it
  cannot be painted beyond B32. Its non-thermal fraction would need a form that does not extrapolate steeply, for example
  one that is held fixed beyond the cut or saturates.
- Caveats:
  - T7's outer temperature exponent c sits at its upper bound (10) in the joint fit, so (1 + x/x_t)^-c acts as
    exp(-1.32 x). An exponential outer form would remove that redundancy.
  - The joint fit changes the electron law, so the kSZ / FRB-DM laws would change too (in sample +5 %, out of sample better).
  - The painter has no n_e x T form yet, so there is no tSZ map test.
  - Electron targets use the pressure-fit rows (B32 of pressure), not the electron B32.

### Figures (`maas/figures/`, rendered on maas)
- `pressure_profiles`: six mass x redshift points, x^3 P to 4 R200c with the host-bound to halo-component band.
- `residual_rms`: RMS inside the cut by mass group, for pressure and electron density.
- `temperature_profiles`: stack temperature 428 P/n_e against the joint temperature law, z = 0, 1, 2.
- `nonthermal_fraction`: implied 1 - P_stack/P_hydrostatic against the fitted and Shaw+10 forms.
- `heldout`: held-out pressure loss per output.
- `../figures_sep/` on maas holds the same figures for the separate fits.

### Files and provenance
- maas runtime: `/net/schelde/data2/nagainis/HalfDome_runtime/BE_pressure_from_density_20261126/` (code, out, logs,
  receipts, figures). Local copies: `maas/code`, `maas/out/*.json`, `maas/figures`, `maas/receipts`, `maas/logs`.
- Receipts: one JSON per job (host, times, exit status, peak RSS, code and BD-code hashes, output hashes).
  `maas/receipts/out_sha256_remote.txt` lists the remote output checksums. The local JSON copies match them.
- `be_fit.py` was extended (models H3e/H6e and task fits2) while the held-out and 2-halo jobs ran. Those jobs had already
  imported the earlier version, and the existing code paths are unchanged (`maas/logs/code_update_fits2.txt` gives both hashes).
- Jobs: sums (6.4 min), fits (5 min), held-out and 2-halo in parallel, fits2, plots. All at nice 19, idle I/O, one thread,
  peak RSS < 1 GiB. The user's production extraction was running on maas at the same time.

## 2. Quick look (morning, superseded by part 1)
It used the BC step-2 atlas unit-weight stacks (`atlas_{P,E}.npz: prof_a2ii`), the measured electron profile (no law)
and unweighted shells. Results are in `tests_output.txt`: hydrostatic + fitted f_nth gave 4.5-4.9 % RMS for >= 10^13.5;
the 7-parameter temperature law gave 0.13 for groups and 0.04-0.06 for clusters. Part 1 replaces it.

## Glossary (code names used in files only)
plaw_held / plaw_const = pressure law of record with alpha(z) held / one constant alpha; electron_law = 9-parameter B16
electron law; T5 / T7 / T8 = temperature laws; H3 / H3c / H6 = hydrostatic models (H3c: free concentration factor; H6:
with core term); suffix e = smooth uncapped non-thermal fraction; xout3/xout6 = P_tot = 0 at 3 / 6 R200c (default 4);
sep / joint = separate / joint fits; rout4 / rout6 = 2-halo amplitude shells to 4 / 6 R200c.

## Verification status
PASSED (26 Nov 2026). A separate agent recomputed the results on maas with its own code, written from the docstrings only
(`maas/verify_independent/VERIFY.md`, `scripts/`, `out/`; identical copies on maas under `verify_independent/`).
- Constants C_T = 428.214, K_HSE = 0.0089957 and PE_FRAC = 0.517647 were rederived; both unit equations were checked in SI.
- Model values for the pressure law of record, joint T7, separate H6, joint H6 and joint H6 to 6 R200c agree with the
  study to <= 9.5e-5 relative. The losses agree to <= 8.4e-5 (joint H6: L_P 5.656548 vs 5.656373), and the Y200
  ratios to <= 2.5e-5.
- The pressure law of record gives L_P = 12.079708 vs 12.079656. With the BD targets, the BD record value 12.0799522 is
  reproduced to 2e-15.
- Data: the electron and pressure targets equal the volume means of the stored sums to 4.8e-15. The unit-weight
  halo-component electron stacks equal the independent BC step-2 atlas in all 109 bins and 7,630 shells (max 1.1e-15),
  and the electron weights recompute exactly.
- The held-out totals equal the sums of their per-output entries. An independent leave-one-out refit of the pressure
  law gives 12.914220.
- Stack temperature, clusters >= 10^14 at z = 0: 1.23-1.55 (0.1 R200c), 0.73-0.91 (0.5 R200c), 0.46-0.54 (R200c).

Points raised by the verifier:
- The 109 shells beyond B32 that straddle the 0.95 cap of the non-thermal fraction are integrated without splitting at
  the cap, which gives errors of up to 0.53 % (joint H6). They do not enter the inside-B32 losses, but they do enter the
  hydrostatic 2-halo fits.
- This study fits the all-gas pressure stack; the BD record target inside B32 is the halo-component stack. They differ
  by <= 7.9e-4, which changes L_P by 2.7e-4.
- Joint T7's c_out sits at its bound (10). The x_out = 4 hydrostatic models reach exactly zero pressure at 4 R200c by
  construction.
- Not verified: that the T7 / H6 fits are global minima, the per-halo electron scatter behind the electron weights, and
  the 2-halo fits (which reproduce the BD record for the pressure law, see Setup).
