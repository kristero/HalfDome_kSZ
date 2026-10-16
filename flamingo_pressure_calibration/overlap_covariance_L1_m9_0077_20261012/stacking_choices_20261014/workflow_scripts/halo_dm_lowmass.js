export const meta = {
  name: 'halo-dm-and-low-mass',
  description: 'Test J: calibrate the 5e12-3e13 Msun bins and redo the DM sensitivity and fits for three halo-component definitions (R200c sphere, 2R200c sphere, bound gas) on STRW maas; exact kSZ bound-only comparison; update the report site',
  phases: [
    { title: 'J1 sensitivity', detail: 'inputs, low-mass stacks, fine bases, truths and importance for the three halo definitions (maas)' },
    { title: 'Verify J1' },
    { title: 'J2 fits', detail: 'B16 and Lee22 ladder per halo definition, low-mass calibration effect, kSZ bound-only exact (maas)' },
    { title: 'Verify J2' },
    { title: 'Site', detail: 'new pages (halo DM, low-mass calibration) and updates of the existing pages' },
    { title: 'Site verify' },
    { title: 'Site fix' },
  ],
}

const STUDY = '/home/cbllover/HalfDome/flamingo_pressure_calibration/overlap_covariance_L1_m9_0077_20261012'
const OUT = STUDY + '/stacking_choices_20261014'
const SITE = OUT + '/site'
const J = STUDY + '/J_halo_dm_lowmass'
const PY = 'OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 LD_LIBRARY_PATH=/home/cbllover/miniconda3/envs/halfdome/lib /home/cbllover/miniconda3/envs/halfdome/bin/python -B'
const CHROME = OUT + '/tools/safe_chrome.sh'

const LIMITS = `LOCAL MACHINE LIMITS (strict; the user's laptop hit 100% RAM and disk on 2026-10-14): heavy compute ONLY on STRW maas; local Python only for smoke tests on small subsets, figures and tables, with ${PY} (2 threads), RAM < 4 GB; keep local files small (C: is nearly full; the WSL disk lives on C:); never search / or /mnt/c; never start chrome.exe directly — for page tests use ${CHROME} 'shot <png> <html> [W,H] [--dark]' / 'dom <html>' (one at a time) and run '${CHROME} cleanup' at the end. Do not write .md files (the harness may block them); return prose in the structured output.`

const CONTEXT = `CONTEXT
- Project: calibrate 1D stacked electron-pressure (tSZ) and electron-density (kSZ, FRB DM) profiles of FLAMINGO L1_m9 haloes (snapshot 0077, z = 0) to paint the HalfDome halo catalogue (XGPaint; painted haloes have M200c >= 6.6e12 Msun). STUDY=${STUDY}. The main stack is the A2-ii painting partition (each gas particle belongs to the painted halo in which it has the smallest r/R200c). Recently AGN-heated gas removed, geometric shell estimator, inclusion-weighted mean, 0.2-dex M200c bins.
- Done before (read first): ${STUDY}/H_sensitivity/REPORT.md (Test H: fine-grid sensitivity at the A2-ii target, fine bases with 67 X200 shells to 4 R200c + 8 extension shells; code h_*.py, run_test_h.py), ${STUDY}/I_fitting_a2ii/REPORT.md (Test I: B12/B16 refits with objectives (a)/(d)/(L) and the Lee22 ladder on A2-ii; code i_*.py), ${STUDY}/G_physical_outer/REPORT.md (Lee22 selection machinery g_selection, g_fit, g_models), ${STUDY}/F_observables/REPORT.md (engine, families, DM PDF via f_pdf, z_s = 1), SYNTHESIS.md. Stacking core ${OUT}/core/stackcore.py with per-halo cache ${OUT}/cache (keys 'field|estimator|agn|selection|phase|grid|norm', e.g. 'E|geometric|removed|A2ii|all|x200|rhoc_fb' and 'E|geometric|removed|host_bound|all|x200|rhoc_fb'; API in ${OUT}/core/API_SUMMARY.txt). The website ${SITE} (build_site.py, content/*.txt) reports all of this.
- THE USER'S NEW QUESTIONS AND DECISIONS (2026-10-15):
  (Q1) "Why DM from haloes below 10^13.5 is not calibrated? It should be calibrated for halos > 5*10^12." Facts: the engine support (kernels/engine.py:315 support_definition; F_observables/f_common.py:152 CAL_BINS) marks only 13.5-15.388 as calibrated because the original radii234 extraction had only haloes above 10^13.5; the later particle pass added 1,600 haloes in 12.5-13.5 (321/332/322/317/308 per 0.2-dex bin) that are in the cache but were never calibrated; the published B16/B12 fallback paints them. They carry 42% of the halo mean DM (engine bins 12.7-12.9/12.9-13.1/13.1-13.3/13.3-13.5: 0.7/14.0/14.0/14.2%; H_sensitivity/results/sens_summary.json shares a2ii__dm_mean@scalar) and 10-18% of kSZ 1h. The strict resolution mask trusts only shells from 2.18/1.58/1.03/0.54 R200c outward in these bins; the median halo in 12.7-12.9 has 0-7 gas particles per fine shell inside R200c (gas particle mass ~1e9 Msun) and median smoothing length h ~ 0.65 R200c at 2R. DECISION: calibrate every painted mass, i.e. the bins 12.7-12.9 (contains the 6.6e12 painting threshold), 12.9-13.1, 13.1-13.3, 13.3-13.5 in addition to 13.5-15.5, for pressure and electrons.
  (Q2) "Why painting out to 5R gives such contribution? It seems there is too much field component taken into account, even though I only care about the halo component in this case." Facts: the 4-5R shell has 95% of the volume of the 0-4R sphere; the A2-ii electron density at 3.6-4R is 3-6x the cosmic mean (E-unit mean 0.3348) and the extrapolations put 1.2-4.8x the mean at 4-4.5R; the share of A2-ii electrons bound (HBT) to the host or its own satellites is 84-95% at 0-1R, 31-37% at 1-2R, 4-14% at 2-3R, 1-7% at 3-4R; only 11% (12.7-12.9) to 43-46% (>=14.5) of the A2-ii electrons inside 4R lie inside R200c. So the 'DM not converged at 4R' result of Test H describes a field-inclusive target. The user's HalfDome DM pipeline uses a spherical R200c halo component (frb_map_generation/LIKE_FOR_LIKE_SPHERICAL_R200C_20260916.md: chord-limited line of sight inside a 3D sphere, like Konietzka's TNG within-R200 catalogue). DECISION: compare THREE halo-component definitions for DM: (D1) A2-ii gas inside the 3D sphere r <= R200c; (D2) A2-ii gas inside r <= 2 R200c; (D3) bound gas only: gas HBT-bound to the halo or its own satellites (cache selection 'host_bound'), at all radii to 4R (state how much bound gas lies beyond 2R and 4R). Reference for comparison: the current target (D0) = A2-ii gas to 4R. Sphere truncation = a 3D cut of the profile followed by the line-of-sight projection, equal to the HalfDome SphericalChordDMProfile (check this equivalence on one profile against frb_map_generation's spherical chord code or its formula DM = (2 R200c/(1+z)) int_0^{sqrt(X^2-x^2)} n_e(sqrt(x^2+l^2)) dl).
  (Q3) kSZ: "Keep as it is [A2-ii gas to 4R], but compare what are the differences if only the halo bound particles are used (if already done, skip that)". Facts: only a first-order estimate exists (groups/selection/results.json variants 'selection:host_bound' impact E: kSZ 1h -79.4/-57.7/-21.8% at ell 100/1000/3000, all-gas reference, calibrated bins 13.5-15.5 only; first order overstates large changes of quadratic observables). DECISION: compute the exact kSZ (1h at ell 100/300/1000/3000 and 2h) with bound-only gas versus A2-ii to 4R, including the new low-mass bins; keep the kSZ recipe unchanged.
- CLUSTER EXECUTION (the user requires it): heavy stages run on STRW maas through ${STUDY}/cluster_tests.py (read its docstring and H_sensitivity/CLUSTER_RUN.json, write_cluster_run.py, run_test_h.py: mirror / run <TEST> --workers 8 --rss-stop N --driver-args=... / status / fetch / stop). You may add ONE entry 'J' to the TESTS dict (the only allowed edit outside ${J}). The remote runtime tree /home/nagainis/HalfDome_runtime/overlap_covariance_tests is ours (home quota 3.9 GB: big intermediates such as fine bases (~2.4 GB) go to node-local scratch /tmp/nagainis_halfdome_testJ, deleted after the final fetch of Test J; record it). Remote real study directories and /net/hypernova are read-only. Check maas load/memory first; 8 workers (user's instruction), nice 19, the wrapper's guards (H attempt 3 ran at ~6 GiB tree RSS with --rss-stop 24); poll every 5 minutes with a loop that terminates on the wrapper's done/failed status (never 'while pgrep -f ...'); fetch when done. If you are a resumed agent, first check cluster_tests.py status and the files in ${J} and continue from them; never launch a duplicate job.
- Write only in ${J} (and the one TESTS entry). Import F/G/H/I/kernels/analysis/core modules read-only, configured in your process only (as H did for the fine grid; here also the calibrated support must be overridden to 12.7-15.388 in your process). Every number you report must come from your results files (cite file/field); label estimates. Explain every equation: every symbol, its meaning and units. Figures readable, labelled with units, colour-blind-safe.`

const RESULT_SCHEMA = {
  type: 'object',
  properties: {
    summary: { type: 'string' },
    report_markdown: { type: 'string', description: 'complete, well-structured section text (markdown with LaTeX $...$/$$...$$); every equation followed by the definition of every symbol; tables; figure paths' },
    key_numbers: { type: 'array', items: { type: 'object', properties: { name: { type: 'string' }, value: { type: 'string' }, source: { type: 'string' } }, required: ['name', 'value', 'source'] } },
    figures: { type: 'array', items: { type: 'object', properties: { path: { type: 'string' }, caption: { type: 'string' }, main: { type: 'boolean' } }, required: ['path', 'caption', 'main'] } },
    recommendations: { type: 'array', items: { type: 'string' } },
    files_written: { type: 'array', items: { type: 'string' } },
    run_record: { type: 'string' },
    verification: { type: 'string' },
    caveats: { type: 'array', items: { type: 'string' } },
  },
  required: ['summary', 'report_markdown', 'key_numbers', 'figures', 'recommendations', 'files_written', 'run_record', 'verification', 'caveats'],
}
const VERIFY_SCHEMA = {
  type: 'object',
  properties: {
    passed: { type: 'boolean' },
    checks: { type: 'array', items: { type: 'object', properties: { name: { type: 'string' }, passed: { type: 'boolean' }, detail: { type: 'string' } }, required: ['name', 'passed', 'detail'] } },
    wrong_claims: { type: 'array', items: { type: 'string' } },
    fix_instructions: { type: 'string' },
  },
  required: ['passed', 'checks', 'wrong_claims', 'fix_instructions'],
}

const J1_PROMPT = `TASK (Test J1 — low-mass stacks and DM sensitivity for three halo-component definitions). Create ${J}/ (run_test_j.py with stages, smoke mode, CLUSTER_RUN.json via write_cluster_run.py, results/, figures/, verification.json, SHA256SUMS; same conventions as H_sensitivity). Stages:
J0 inputs: export from the cache, on the x200 grid for ALL 15 bins (12.5-15.5): E (A2ii, host_bound, all) and P (A2ii) per-halo profiles, counts, smoothing lengths, halo table; compact npz under ${J}/inputs/.
J1a low-mass stacks: for 12.7-12.9 ... 13.3-13.5 build the A2-ii (P, E) and host_bound (E) stacks. Replace the per-halo strict mask with a STACK-LEVEL rule suited to stacked particle sums: merge fine shells inside a stack-dependent radius into coarse shells so that each stacked shell holds >= 400 particles summed over the bin's haloes (5% Poisson), and report the hydrodynamic-resolution limit separately (median 2h/R200c per shell; shells inside it are 'unresolved shape' but their integrated electron count is still measured). Show and quantify: measured stacks vs the published fallback (B16 electrons, B12 pressure) the engine uses now in these bins (profile ratios; N_e(<R200c), N_e(<2R), Y(<R200c)); how the DM observables change when the fallback is replaced by the measured stacks (D0 and D1-D3 targets); the sensitivity of the DM observables to the treatment of the unresolved inner part (e.g. measured coarse shells vs a B16 shape renormalised to the measured N_e inside the resolved radius): if it changes a DM statistic by more than 1%, say so clearly. Also give the FLAMINGO L1_m8 option (8x better mass resolution; COSMA data service, see ${OUT}/research/cosma_service_audit.md) as an estimate of cost only — do not download.
J1b fine bases on maas with the calibrated support extended to 12.7-15.388 (engine bin 12.5-12.7 holds no painted haloes; check). Parity with Test H where the configuration coincides (13.5-15.388 bins, D0 target) to <= 1e-6.
J1c truths for D0 (A2-ii to 4R), D1 (A2-ii, sphere 1 R200c), D2 (A2-ii, sphere 2 R200c), D3 (host_bound, all radii to 4R): DM mean, variance, quantiles 16/50/84/97.5/99.9 (f_pdf, z_s = 1, scatter-inclusive as in H), DM-field C_ell, tSZ x DM (E side); also the kSZ 1h (ell 100/300/1000/3000) and 2h for D0 and D3 (exact bound-only kSZ comparison, Q3).
J1d sensitivity per definition at the finest radial resolution: normalised importance maps (F1 convention, k = 2 for quadratic observables), per-shell contributions, cumulative signal vs radius inside each definition's boundary, R50/R90/R95, mass shares (now including the low-mass calibrated bins and the remaining fallback), profile tolerance per shell. Answer explicitly: where does each DM statistic come from under each definition, and how much do the three definitions differ (mean DM, variance, quantiles, PDF shape; numbers and ratios D1/D0, D2/D0, D3/D0), per mass bin and in total.
Outputs: results (npz/json/csv), figures (main: DM PDF and statistics for D0-D3; importance vs radius per definition with mass bins; low-mass stacks vs fallback; bound vs A2-ii electron profiles per mass bin with the cosmic-mean line; kSZ bound-only vs A2-ii), verification.json (parities, sum rules, cumulative curves reach 1 at each boundary, low-mass rule checks), and the report section 'Halo component of DM and low-mass calibration: sensitivity' with explicit answers to Q1-Q3.`

const J2_PROMPT = (j1) => `TASK (Test J2 — fits for the three halo-component definitions and the low-mass bins; uses Test J1 outputs in ${J}). J1 returned:\n${JSON.stringify({ summary: j1 && j1.summary, key_numbers: j1 && j1.key_numbers, files: j1 && j1.files_written, caveats: j1 && j1.caveats }).slice(0, 9000)}\nModels: ONLY B16 per mass bin (electrons; B12 for pressure in the low-mass bins) and the Lee22 parametrisation ladder from G's g_selection (as used in Test I: PB, PL1, PL2, PL3, BPLa, BPLs, BPLall with free and with Lee22's M_cut, PL3g, PL3gM, the +c concentration models; outer option O0 inside a boundary, and O0/O1/O2 for D0 and D3 where the profile extends to 4R) — no new model families. Data: the D1, D2, D3 stacks (and D0 for reference) for the calibrated bins 12.7-15.5 (now 14 bins).
J2a B16 per bin for each definition: R_fit = the definition's boundary (1 R200c for D1, 2 for D2, 4 for D3 and D0; also R_fit smaller than the boundary where useful), objectives (a), (d) (importance from J1 for that definition), (L); predictions det and S1 against each definition's truth at R_cut = the boundary; DM family RMS and per-statistic errors (mean, variance, quantiles), DM-field C_ell, tSZ x DM; the sensitive-part score (importance-weighted RMS of ln(model/stack), same formula as Test I). Also B12 (pressure) and B16 (electrons, D0 target) for the four new low-mass bins with the Test I recipe, and the change of tSZ / kSZ / DM when these calibrated fits replace the published fallback.
J2b Lee22 ladder per definition (mass bins; concentration cells where available), likelihoods stat/floor/L22, chi^2/nu, AIC, BIC, dBIC, spatial CV, G's selection rule; the selected model per definition, and its DM errors (det, S1) at R_cut = boundary. Compare with Test I's A2-ii (D0) ladder.
J2c Which model and fit range best captures the sensitive part of the electron profile for each definition and DM statistic; the minimal adequate model per definition; how the low-mass bins (where the inner shape is unresolved) affect the choice.
J2d kSZ (unchanged recipe A2-ii to 4R): add the exact bound-only comparison from J1 to the report (kSZ 1h ell 100/300/1000/3000 and 2h; D3 vs D0), and say what a bound-only kSZ fit would change, briefly.
Outputs: results tables (csv/npz), figures (main: DM error vs number of parameters per definition; residual profiles of key models per definition with the importance weight; the selected models' DM PDFs vs truth; low-mass fallback-vs-calibrated effect), verification.json, and the report section 'Halo component of DM and low-mass calibration: fits' with a clear recommendation per definition (model, R_fit, R_cut, expected DM errors) and a short statement of which definition to use for which purpose (TNG/Konietzka within-R200 comparisons; total halo-associated gas; bound gas), tied to the numbers.`

const VERIFY = (name, dir, res) => `TASK: independently verify ${name} in ${dir} (another agent's work). Be adversarial. Its output:\n${JSON.stringify({ summary: res && res.summary, key_numbers: res && res.key_numbers, recommendations: res && res.recommendations, run_record: res && res.run_record, verification: res && res.verification, caveats: res && res.caveats }).slice(0, 9000)}\nReport text (check every quantitative claim):\n${(res && res.report_markdown || '').slice(0, 18000)}\nChecks (write scripts under ${dir}/verify_independent_${name.replace(/\W+/g, '_')}/): (1) recompute at least 8 headline numbers from the saved results/inputs with your own code (e.g. a D1/D0 mean-DM ratio, a per-bin low-mass N_e(<R200c) vs the fallback, an R90 under a definition, a fitted parameter set and its chi^2/nu, a selected ladder model, a DM error, the exact bound-only kSZ change); (2) check that the sphere truncation equals the HalfDome chord-limited spherical DM on at least one profile; (3) check the low-mass stack-level rule (particle totals per shell) and the stated resolution limits from the inputs; (4) conservation/consistency (importance sum rules, cumulative curves, parity with Test H where configurations coincide); (5) that the heavy runs happened on maas (receipts, logs, CLUSTER_RUN.json, fetched results with SHA256SUMS) and nothing outside ${dir} was modified except the TESTS entry; (6) Read 3-4 main figures; (7) every equation's symbols are defined and recommendations follow from the numbers. passed = true only if no wrong claim remains; give precise fix instructions. ${LIMITS}`

const runWithVerify = async (name, dir, prompt, implPhase, verPhase) => {
  let res = await agent(`${CONTEXT}\n\n${LIMITS}\n\n${prompt}`, { label: `implement:${name}`, phase: implPhase, schema: RESULT_SCHEMA })
  let ver = null
  for (let round = 0; round < 3; round++) {
    ver = await agent(`${CONTEXT}\n\n${VERIFY(name, dir, res)}`, { label: `verify:${name}:${round}`, phase: verPhase, schema: VERIFY_SCHEMA })
    if (ver && ver.passed) break
    if (round === 2) { log(`${name}: still ${ver ? ver.wrong_claims.length : '?'} problems after 3 verifications`); break }
    log(`${name}: verification round ${round} failed (${ver ? ver.wrong_claims.length : '?'} problems) — fixing`)
    res = await agent(`${CONTEXT}\n\n${LIMITS}\n\nTASK: fix ${name} in ${dir} according to the verifier (rerun on maas if results change; regenerate figures, verification.json, SHA256SUMS) and return the complete corrected outputs.\nVerifier:\n${JSON.stringify(ver).slice(0, 10000)}\nPrevious report text:\n${(res && res.report_markdown || '').slice(0, 14000)}`, { label: `fix:${name}:${round}`, phase: implPhase, schema: RESULT_SCHEMA })
  }
  return { res, ver }
}

phase('J1 sensitivity')
const J1 = await runWithVerify('Test J1', J, J1_PROMPT, 'J1 sensitivity', 'Verify J1')
phase('J2 fits')
const J2 = await runWithVerify('Test J2', J, J2_PROMPT(J1.res), 'J2 fits', 'Verify J2')

// ---------------------------------------------------------------- site
const SITE_RESULT = {
  type: 'object',
  properties: {
    files: { type: 'array', items: { type: 'string' } },
    pages: { type: 'array', items: { type: 'object', properties: { page: { type: 'string' }, title: { type: 'string' }, summary: { type: 'string' } }, required: ['page', 'title', 'summary'] } },
    key_messages: { type: 'array', items: { type: 'string' } },
    notes: { type: 'string' },
    problems: { type: 'array', items: { type: 'string' } },
  },
  required: ['files', 'pages', 'key_messages', 'notes', 'problems'],
}
const CHECK = {
  type: 'object',
  properties: {
    passed: { type: 'boolean' },
    problems: { type: 'array', items: { type: 'object', properties: { page: { type: 'string' }, where: { type: 'string' }, problem: { type: 'string' }, fix: { type: 'string' }, severity: { type: 'string', enum: ['blocker', 'major', 'minor'] } }, required: ['page', 'where', 'problem', 'fix', 'severity'] } },
    notes: { type: 'string' },
  },
  required: ['passed', 'problems', 'notes'],
}
const SITE_BASE = `${LIMITS}\n\nSITE: ${SITE}. Pages are Markdown in ${SITE}/content/<page>.txt with a front-matter block (title, short, section, order, summary); built by ${PY} ${SITE}/build_site.py (--check fails on broken links, missing figures, math or equation-block errors; --syntax prints the exact syntax: equation blocks with latex/meaning/symbols/code, fig: and code: and explorer: links, callouts). Read 2-3 existing pages (index.txt, sensitivity.txt, fitting.txt) to match the style: 'In short' block first, plain precise English, every equation in an equation block with every symbol defined (meaning, units) and a code link, figures read before use and captioned with what to look at, numbers exactly from the source files with the source cited. Write only under ${SITE}/ (content/, and build outputs).\nTest J results:\nJ1 summary: ${(J1.res && J1.res.summary || '').slice(0, 4000)}\nJ2 summary: ${(J2.res && J2.res.summary || '').slice(0, 4000)}\nJ files: ${J}/ (results/, figures/, verification.json, CLUSTER_RUN.json, j_*.py).`

phase('Site')
const pageNew = await agent(`${SITE_BASE}\n\nTASK (new pages): write content/halo_dm.txt (section 'sensitivity' or a new section 'halo' if the builder supports adding it cleanly; title like 'The halo component of DM'), with detail pages as needed (e.g. halo_dm_fits.txt), and content/lowmass.txt ('Calibrating 5e12-3e13 Msun haloes'). They must answer the user's questions directly at the top: why the low-mass haloes were not calibrated before and what changes now (numbers); why the 5R extension gave +7.7% (field gas: volume, density relative to the cosmic mean, bound fractions) and what each halo-component definition (R200c sphere, 2R200c sphere, bound gas) gives for the DM statistics, with the recommended model, fit range and cut per definition; and the exact kSZ bound-only comparison. Full J1 report:\n${(J1.res && J1.res.report_markdown || '').slice(0, 45000)}\nFull J2 report:\n${(J2.res && J2.res.report_markdown || '').slice(0, 45000)}\nJ figures: ${JSON.stringify([...(J1.res && J1.res.figures || []), ...(J2.res && J2.res.figures || [])]).slice(0, 6000)}\nAlso write content/glossary_halo.json for new terms. Build with --check and fix your pages' errors.`, { label: 'site:new-pages', phase: 'Site', schema: SITE_RESULT })

const pageUpd = await agent(`${SITE_BASE}\n\nTASK (update existing pages): the new pages are ${JSON.stringify(pageNew && pageNew.pages || []).slice(0, 3000)} (key messages: ${JSON.stringify(pageNew && pageNew.key_messages || []).slice(0, 4000)}). Update the existing content so nothing is stale or contradicts Test J: index.txt ('In short', the recommended recipe, the open questions — the 43% uncalibrated mean DM below 10^13.5 is now addressed; the 5R statement must say it is field gas; add the halo-component choice for DM with links), sensitivity.txt and sens_truncation.txt (qualify 'DM not converged at 4R' as a property of the field-inclusive A2-ii-to-4R target; link the halo-component page), fitting.txt and fit_observables.txt (add the per-definition recommendation for DM and the low-mass bins, linking J2), choices.txt and choices_selection.txt (bound-only membership: point to the exact kSZ/DM comparison), about.txt (Test J run records on maas, verification status), glossary.json (merge glossary_halo.json). Search all content/*.txt for statements that Test J changes (e.g. '43%', 'below 10^13.5', 'not converged', '5 R200c', 'fallback', '+7.7') and fix each with the new numbers. Rebuild with --check. List every change.`, { label: 'site:update-pages', phase: 'Site', schema: SITE_RESULT })

phase('Site verify')
const SITE_VERIFIERS = [
  { key: 'numbers', prompt: `Check NUMBERS on halo_dm*, lowmass and every changed passage of index, sensitivity, sens_truncation, fitting, fit_observables, choices, choices_selection, about against the Test J results files (recompute where needed). Also check for stale statements anywhere in the site that Test J contradicts.` },
  { key: 'completeness', prompt: `Check that the site now answers the user's three new questions clearly and correctly: (Q1) why haloes below 10^13.5 were not calibrated and what calibrating 12.7-13.5 changes; (Q2) why the 5R extension added so much DM (field gas) and how the three halo-component definitions compare, with a recommendation per definition; (Q3) the exact kSZ bound-only comparison. Also readability: is the answer near the top, are the equations' symbols defined, do the recommendations follow from the numbers.` },
  { key: 'browser', prompt: `Check RENDERING with the safe Chrome helper (one at a time, at most 12 screenshots, then cleanup): halo_dm, lowmass, the detail pages, index, sensitivity at 1400 px; one dark-mode shot. Run a link checker over all HTML (local hrefs and anchors exist). Read every PNG: KaTeX errors, broken images, overflowing tables, unreadable figures.` },
]
const vAgent = (v, extra) => agent(`${SITE_BASE}\n\n${extra || ''}TASK (verify, adversarial, independent): ${v.prompt} passed = true only if no blocker or major problem remains.`, { label: `site:verify:${v.key}`, phase: 'Site verify', schema: CHECK })
let checks = await Promise.all(SITE_VERIFIERS.map(v => vAgent(v)))

phase('Site fix')
for (let round = 0; round < 2; round++) {
  const open = checks.flatMap((c, i) => (c && c.problems || []).filter(p => p.severity !== 'minor' || round === 0).map(p => ({ ...p, check: SITE_VERIFIERS[i].key })))
  if (!open.length) break
  log(`site fix round ${round}: ${open.length} problems`)
  await agent(`${SITE_BASE}\n\nTASK (fix): correct these problems in ${SITE}/content/*.txt (or build_site.py), checking each against the sources; rebuild with --check; re-screenshot affected pages with the safe Chrome helper (at most 8) and Read them; cleanup at the end.\nProblems:\n${JSON.stringify(open).slice(0, 30000)}`, { label: `site:fix:${round}`, phase: 'Site fix', schema: SITE_RESULT })
  const failed = SITE_VERIFIERS.map((v, i) => ({ v, i })).filter(({ i }) => checks[i] && !checks[i].passed)
  const re = await Promise.all(failed.map(({ v, i }) => vAgent(v, `RE-VERIFY after fixes. Previously found problems:\n${JSON.stringify(checks[i].problems).slice(0, 12000)}\nCheck each is fixed, then repeat the check.\n`)))
  failed.forEach(({ i }, k) => { checks[i] = re[k] })
  if (checks.every(c => c && c.passed)) break
}

return { J1, J2, pageNew, pageUpd, checks: checks.map((c, i) => ({ key: SITE_VERIFIERS[i].key, passed: c && c.passed, problems: c && c.problems })) }
