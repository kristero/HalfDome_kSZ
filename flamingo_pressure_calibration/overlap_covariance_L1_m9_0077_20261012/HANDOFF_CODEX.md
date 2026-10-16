# Handoff: FLAMINGO-calibrated halo profiles for painting HalfDome

Written 2026-10-16 by Claude (Opus) for the next agent, for example Codex. It covers four things:

- what has been done;
- where everything is;
- what is still open;
- how to extend the calibration to a full-particle run over all snapshots.

Read sections 1 and 2 first, then the report of the test you continue. Every number quoted here comes from the named report or results file. Re-check any number before you rely on it.

`STUDY` = `/home/cbllover/HalfDome/flamingo_pressure_calibration/overlap_covariance_L1_m9_0077_20261012` (this directory).

---

## 1. Goal and the user's decisions (binding)

**Goal.** Calibrate 1D halo profiles on FLAMINGO L1_m9 (HYDRO_FIDUCIAL, 1 Gpc, m9) and paint them on the HalfDome halo catalogue with XGPaint:

- electron pressure for tSZ;
- electron density for kSZ and FRB DM.

Painted haloes have M200c ≥ 6.6e12 Msun.

**Decisions, in order of importance.**

1. **Paint haloes, not the field.** The user does not want to reproduce field-gas effects in kSZ or FRB DM.
2. **Fit inside a halo truncation radius; paint to 4R.** The truncation radius R_t only says which data constrain the halo-profile fit. The fit uses **all gas** inside R_t, as observational fits do. HalfDome then paints the **fitted law extrapolated to 4 R200c**. Never recommend painting only inside R_t.
3. **User's truncation rule (B32).** In each mass bin, for pressure and electrons separately, R_t is where the bound fraction of the stacked gas per shell drops to 32%. "Bound" means HBT-bound to the central or to its own satellites, the cache selection `host_bound`.
   - A second reading, C68 (the radius enclosing 68% of the bound gas), was tested and rejected (L1).
   - Alternative rules were requested and implemented (L1).
4. **Ownership of shared gas: the A2-ii painting partition is the main stack.** Each particle goes to the painted halo (M200c ≥ 6.6e12) in which it has the smallest r/R200c. Own satellites stay with the host. Inside 1.5 R200c, A2-ii and all gas are identical to better than 0.12%.
5. **All 14 painted mass bins are calibrated (C14)**, from 10^12.7 to 10^15.5. The stack-level resolution rule from Test J is used for 12.7-13.5.
6. **Redshift.** Everything so far is at z = 0, except the SOAP refit (Test K). The next extension goes to the most sensitive redshifts, z ≈ 0.1-1.
7. **Cluster etiquette.** Heavy work runs on STRW maas through `cluster_tests.py`:
   - at most 8 workers, nice 19, ionice 7;
   - memory and load guards;
   - node-local scratch for large intermediates;
   - never modify shared or other users' data;
   - the user said "use ssh strw".
8. **Protect the user's laptop.** It has 64 GB RAM and C: is about 99% full. The WSL disk lives on C:.
   - no headless Chrome except through `stacking_choices_20261014/tools/safe_chrome.sh`;
   - no searches over `/` or `/mnt/c`;
   - local Python with 2 threads and less than 4 GB RAM;
   - keep local files small.
9. **Verify everything.** Every test so far had an implementation, then an independent adversarial verification, then fixes. Keep that standard.

---

## 2. Status of every test (z = 0 unless stated)

| Test | Folder | Report | Status | Main result |
|---|---|---|---|---|
| Particle pass | `.` (`measure_overlap.py`, `extract_study.py`, `census.py`, `prepare_study.py`) | `README.md`, `STEP1_README.md` | done 2026-10-12 | 4,588 centrals from 10^12.5 to 10^15.5 (inclusion-weighted). Per-halo linear shell sums: 17 owner classes × 3 phases × 16 quantities × 67 shells to 4 R200c, plus 40 shells to 6 R500c (`measurements_owner.npz` on maas) |
| A-G | `A_overlap` … `G_physical_outer` | each `REPORT.md`; `SYNTHESIS.md` (A-G) | done | Overlap and ownership, covariance (B), sigmoid truncation (C), literature recipes (D), observable engine (F), physical outer models and Lee22 ladder on all gas (G) |
| Stacking choices | `stacking_choices_20261014/` | `groups/*/SECTION.md`, `research/*.md` | done, verified | Every user choice and its impact. The core is `core/stackcore.py` with the per-halo cache `cache/` (722 variants) |
| H | `H_sensitivity` | `REPORT.md` | done | Fine-grid sensitivity of tSZ, kSZ and DM to the A2-ii stack to 4R. This target includes field gas (see L) |
| I | `I_fitting_a2ii` | `REPORT.md` | done | B12/B16 per bin are best for tSZ. Compact Lee22 laws: BPLall+O0 (k = 10) for electrons |
| J | `J_halo_dm_lowmass` | `REPORT_J1.md`, `REPORT_J2.md` | done, verified | Low-mass calibration C14. Halo-DM definitions: D1 (R200c sphere), D2 (2R sphere), D3 (bound gas). Exact bound-only kSZ |
| K | `K_soap_battaglia` | `REPORT.md` | done, verified | B12/B16 Battaglia laws refitted to SOAP at all 71 outputs. `results/k_params.json`; fitter `k_fit_core.py` |
| **L1** | `L_halo_truncation` | `REPORT_L1.md` | done; 3 verification rounds; last 5 text slips fixed by hand | **B32 radius ≈ 1.35-1.48 R200c at every mass; global x_t = 1.43 R200c** (0.86 R200m, 2.21 R500c) |
| **L2** | `L_halo_truncation` (`l2_*.py`, `run_test_l2.py`, `results_l2/`) | not yet | **in progress** (started 2026-10-16 17:35) | Sensitivity, extrapolation and weights for "fit inside R_t, paint to 4R" (maas) |
| **L3** | `L_halo_truncation` | not yet | **pending** | Fits: B12/B16, Lee22 ladder and the SOAP laws, against the bracket |

**The exact task definitions for L2 and L3** are the prompts in `stacking_choices_20261014/workflow_scripts/halo_truncation.js`:

- the constants `CONTEXT`, `CLARIFY`, `L2` and `L3`, and the verification checklist `VERIFY`;
- `CLARIFY` holds the user's paint-to-4R correction and overrides any older framing.

### Key L1 numbers (`L_halo_truncation/REPORT_L1.md`, `results/radii.json`, `global.json`)

- **B32 per bin.** 1.35-1.45 R200c (P) and 1.37-1.48 (E) for 10^12.9-10^15.3; 1.25 (P) in 12.7-12.9. Bootstrap half-widths are 0.006-0.04.
- **Global radius.** The signal-weighted global radius is 1.429 (P) and 1.435 (E), common 1.432 R200c. A global cut instead of per-bin cuts changes the observables by 0.17-0.57% RMS per family.
- **Alternative rules.**
  - 50% bound (B50): 1.20 / 1.24.
  - Budget matching (Req: painted gas inside R = all bound gas): 1.20 / 1.33. It reproduces D3's mean DM to 1.002.
  - Steepest gas slope: 2.2 → 1.3 with mass.
  - Two-halo floor (E): 1.7-2.7.
  - D3 R90: 1.40-2.06.
  - Entropy maximum: 2.3-3.1, only above 10^14.5.
  - C68: 0.75 / 1.03, rejected.
- **What the extrapolation controls.** For kSZ and DM, 0.28-0.61 of the painted 1-halo or linear terms depends on the extrapolation beyond x_t. The published shapes, extrapolated beyond x_t, sit at bracket position t (0 = bound gas, 1 = all A2-ii gas): B16 t = 0.25-0.97 (0.25-0.39 below 10^13.5, 0.83-0.97 above 10^13.9), B12 up to 1.21 (`results/extrap.json`).
- **Codex reference painting.** The HalfDome 4R painting has FLAMINGO/HalfDome map ratios of 1.315 / 2.462 / 1.248 (tSZ / kSZ / DM at ell ≈ 1025).
  - "Keep fractions" for a hard 1.43R cut (0.94 / 0.60 / 0.72) are **not** the painting plan.

### Key earlier numbers worth knowing

- **Bound share of A2-ii electrons by radius** (Test J, `results/prepare.json`): 84-95% at 0-1R, 31-39% at 1-2R, 4-15% at 2-3R, 1-7% at 3-4R. 44.5% of D0's mean DM comes from 2-4R.
- **Low-mass bins were mis-painted before C14.** The published-profile fallback was wrong there: measured N_e(<R200c) is 0.27-0.55 of B16. The 12.7-13.5 bins carry 43% of the mean DM.
- **Test K recommended laws.**
  - tSZ: P R3 `brk_amp` (XGPaint BreakModel).
  - kSZ/DM: G R3 `free`, or `brk_all` / `orig` as smooth alternatives.
  - **These parameters need the validated XGPaint checkout** (`/home/kn18001/.julia/dev/XGPaint`, branch `cluster`, commit 1660e08). The copy in `julia_env/dev/XGPaint.jl` hard-codes the published laws and normalises τ differently.
  - For `brk_amp`, the tabulated A is P0_amp: P0(1e14, z = 0) = 3.728.
  - R1 fits (inside R200c) are now the relevant SOAP fits for "fit inside R_t ≈ 1.43R". R2 (to about 1.8R) brackets it.
- **Lee22 criticism** (Test B):
  - The GLS covariance on fine grids is near-singular (condition number up to 1,500).
  - It is only meaningful where the model is adequate; beyond 2R it worsens the observables.
  - Parameter stability against R_fit diagnoses model adequacy (plateau to about 1.5-2R), not the truncation radius.

---

## 3. What to finish (priority order)

1. **L2, if not finished.** Read `L_halo_truncation/run_test_l2.py`, `CLUSTER_RUN.json`, `results_l2*`, `verification_l2.json` and `cluster_tests.py status`. Continue from them and do not relaunch a running job. Deliverables, per the `L2` + `CLARIFY` prompts:
   - importance of shells inside R_t for the 4R-painted observables;
   - how well β, x_c and α are constrained from inside R_t (bootstrap + Fisher);
   - the share of each observable from R_t-4R;
   - the extrapolation bracket (bound gas below, all gas above) per bin;
   - the comparison with the Codex HalfDome shell fractions and FLAMINGO map ratios;
   - an independent verification.
2. **L3.** Fits inside R_t, painted to 4R, and scored against:
   - (a) the bracket;
   - (b) D0;
   - (c) D3.

   Fit with:
   - B12/B16 per bin with objectives (a), (d) and (L);
   - the Lee22 ladder with outer option O0;
   - the Test K SOAP laws (R1/R2).

   Also re-evaluate the stacking choices inside the cut. The final recommendation per probe is: rule and radius, model, R_fit, objective, expected errors, and which earlier recommendations it replaces.
3. **Website.** Add the L pages and update the recipe and every page that says "fit and cut at 4R" for kSZ/DM.
   - Site: `stacking_choices_20261014/site/`. Content is `content/*.txt` in Markdown with front matter. Build with `build_site.py --check`; `--syntax` lists the markup.
   - The published copy (a Claude artifact) can only be republished from a Claude session. Codex can build the site locally and the user can open `site/index.html`.
4. **Push to GitHub** (`kristero/HalfDome_kSZ`, branch `cluster`; public repo).
   - Push code and reports only, never data.
   - The SSH key fails from WSL, so push over https: `git -c credential.helper='!gh auth git-credential' push https://github.com/kristero/HalfDome_kSZ.git cluster`.
   - Previous pushes: 7a060f8 (pipelines and reports), 58399ed (J, K).
5. **Before painting with refitted parameters,** quantify the change of the HalfDome tSZ/kSZ spectra with the HalfDome painter. Codex's `halfdome_sz_contributions_20261015` machinery on idark is the natural tool.

---

## 4. Extending to all snapshots with full particles (the production run)

The design is in `stacking_choices_20261014/site/content/snapshots.txt` (sections "The recommended pipeline", "What to store per snapshot", "What it costs") and `research/speedup_inventory.md`. The core idea is to touch particles once per snapshot and store per-halo **linear shell sums** (never ratios), so that every stacking choice, truncation rule and fit can be redone in seconds.

**Changes needed for the truncation analysis** (not in the earlier layout):

1. **Owner split for B32.** The B32 rule needs the bound-to-own-host split per shell. Extend the proposed owner groups ({A2-ii central side, A2-ii neighbour side} × {bound to another painted halo: no/yes}) with **"bound to this host (central or own satellite)"**. In practice, keep these classes:
   - host-bound;
   - unbound-in-A2-ii;
   - other-host-bound-in-A2-ii;
   - outside A2-ii.

   This needs HBT membership, which exists for the fiducial run, not for the variants on maas.
2. **Extend the stored grid beyond 4R** (edges at 4.5, 5 and 6 R200c), so the extrapolation bracket can be measured rather than assumed.
3. **Keep both normalisations.** Keep R200c, and store R200m and R500c per halo, so the rule can be tested in other units at higher z.

**Where the data are:**

- **STRW maas,** `/net/hypernova/data2/FLAMINGO/L1000N1800/HYDRO_FIDUCIAL/`.
  - Full snapshots plus HBT membership at outputs 0000, 0037, 0047, 0057, 0061, 0067, 0071, 0073, 0074 and 0077: z = 15, 2, 1.5, 1, 0.8, 0.5, 0.3, 0.2, 0.15 and 0.
  - SOAP-HBT at all outputs.
  - Variants only at 0074 and 0077, without membership.
- **COSMA hdfstream service** (read-only over https, streamed best from idark at about 9-12 MB/s): all 78 fiducial full snapshots; variants at 13 outputs plus reduced/downsampled data at all outputs.

**Cost** (estimates from the site; re-measure with a one-chunk-file gate first):

- about 1.3 CPU-h and 15 min wall per lean maas pass;
- about 11 CPU-h for the 8 useful fiducial outputs besides z = 0;
- about 2 GB of stored sums per snapshot for a complete sample above 10^12.5.

**Fitting the full-particle results:**

- `K_soap_battaglia/k_fit_core.py` (data-source agnostic: per-halo aperture sums or shell profiles) and the laws in `k_params.json` allow a like-for-like comparison with the SOAP fits.
- Compare profiles and aperture integrals, not single parameters: P0, x_c and β are correlated up to 0.99.

**Redshift priority.** Most of the HalfDome tSZ, kSZ and DM power at ell ≈ 1000 comes from z ≈ 0.1-1 (Codex `halfdome_sz_contributions_20261015/*_redshift_fractions.csv`). Do 0074, 0071, 0067, 0061 and 0057 first (z = 0.15, 0.3, 0.5, 0.8, 1).

---

## 5. Infrastructure and how to run

- **Local Python:** `OMP_NUM_THREADS=2 LD_LIBRARY_PATH=/home/cbllover/miniconda3/envs/halfdome/lib /home/cbllover/miniconda3/envs/halfdome/bin/python -B`.
- **maas from WSL:** `ssh.exe strw-maas`, a ProxyJump through `strw` (user nagainis, Windows config `C:/Users/kn18001/.ssh/config`). Codex also used an isolated config, `C:/Users/kn18001/.ssh/strw_flamingo_20261015.conf`.
  - No scheduler: run plain background processes through the wrapper.
  - 48 threads; often about 24 load from another user; about 178 GiB free.
- **`cluster_tests.py` (in STUDY):** `mirror`, `run <TEST> --workers 8 --rss-stop N --driver-args="..."`, `status`, `fetch <TEST>`, `stop`.
  - The `TESTS` dict maps letters A-L to folders.
  - Remote runtime tree: `/home/nagainis/HalfDome_runtime/overlap_covariance_tests/HalfDome/...`, home quota **3.9 GB**.
  - Put large intermediates in node-local `/tmp/nagainis_halfdome_test<X>` and delete them after the fetch. Codex used `/data1/nagainis/...`, which has more space.
  - Remote venv: `/home/nagainis/HalfDome/flamingo_pressure_calibration/profile_test_L1_m9_0077_20261007/venv/bin/python`.
- **Pitfalls met before:**
  - the home quota filled up;
  - tree-RSS guards double-count copy-on-write pages of forked workers;
  - the mirror upload is slow and dies when STRW drops; STRW had an outage on 2026-10-15 from 11:16 to about 13:40;
  - `while pgrep -f` matches itself;
  - `grep -c` exits 1 at 0 matches;
  - HDF5 chunk caches inflate RSS, so reopen files and set `MALLOC_MMAP_THRESHOLD_`.
- **Stacking core:** `stacking_choices_20261014/core/stackcore.py` (API in `API_SUMMARY.txt`). Keys look like `'E|geometric|removed|host_bound|all|x200|rhoc_fb'`. Selections include `all`, `A2ii`, `host_bound`, `central_bound`, `no_other`, …; grids include `x200` (67 shells to 4R), `fit` (35) and `lee22`.
- **Do not edit** the folders of earlier tests or the particle-pass products. Import earlier modules read-only and configure them in your own process, as H, J, K and L do.

---

## 6. Open questions to raise with the user

- Which halo-component definition becomes the HalfDome default for DM: fit inside B32 R_t and paint to 4R (L), the R200c sphere (D1, matches the TNG/Konietzka comparison), or bound gas only (D3)?
- Whether the outer extrapolation should be anchored by the bound-gas lower edge or by all gas, per probe (L3 decides with the bracket).
- The FLAMINGO-vs-HalfDome map comparison needs FLAMINGO halo-only maps (the Codex particle stage), because the released FLAMINGO maps include field gas.
