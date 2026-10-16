export const meta = {
  name: 'soap-battaglia-refit',
  description: 'Test K: refit the Battaglia B12 (pressure) and B16 (gas density) parametrizations to FLAMINGO L1_m9 fiducial SOAP-HBT apertures at all 71 outputs on STRW maas (3 fit ranges x original / mass-break / free-shape), compare with the published parameters and profiles, add to the report site',
  phases: [
    { title: 'K fit', detail: 'SOAP reads on maas, per-bin data, global fits, uncertainties, comparisons' },
    { title: 'Verify K' },
    { title: 'Site', detail: 'parameter page, interactive profile comparison, links' },
    { title: 'Site verify' },
    { title: 'Site fix' },
  ],
}

const STUDY = '/home/cbllover/HalfDome/flamingo_pressure_calibration/overlap_covariance_L1_m9_0077_20261012'
const OUT = STUDY + '/stacking_choices_20261014'
const SITE = OUT + '/site'
const K = STUDY + '/K_soap_battaglia'
const PY = 'OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 LD_LIBRARY_PATH=/home/cbllover/miniconda3/envs/halfdome/lib /home/cbllover/miniconda3/envs/halfdome/bin/python -B'
const CHROME = OUT + '/tools/safe_chrome.sh'
const XGP = '/home/cbllover/HalfDome/julia_env/dev/XGPaint.jl/src'

const LIMITS = `LOCAL MACHINE LIMITS (strict; the user's laptop hit 100% RAM and disk on 2026-10-14): heavy reads and fits ONLY on STRW maas; local Python only for smoke tests on small subsets, figures and tables, with ${PY} (2 threads), RAM < 4 GB; keep local files small (C: is nearly full; the WSL disk lives on C:; target < 300 MB for everything fetched); never search / or /mnt/c; never start chrome.exe directly — page tests only with ${CHROME} 'shot <png> <html> [W,H] [--dark]' / 'dom <html>' (one at a time), then '${CHROME} cleanup'. Do not write .md files (the harness may block them); return prose in the structured output.`

const CONTEXT = `CONTEXT
- Project: profiles of electron pressure (tSZ) and gas/electron density (kSZ, FRB DM) for painting the HalfDome halo catalogue with XGPaint (painted haloes M200c >= 6.6e12 Msun). Earlier work (read as needed): ${STUDY}/SYNTHESIS.md; ${OUT}/research/soap_audit.md and research/soap/ (SOAP-HBT inventory: no radial profiles; SO apertures 2500c, 1000c, 500c, 200c, 100c, BN98, 200m, 50c, 5xR500c at x = r/R200c ~ 0.28, 0.47, 0.65, 1, 1.35, 1.34, 1.66, 1.8, 3.3 with ComptonY, ComptonYWithoutRecentAGNHeating, GasMass, NumberOfGasParticles, SORadius (COMOVING); InclusiveSphere 10/30/50/100/300/500/1000/3000 PHYSICAL kpc with GasMass only; some SO groups and the large kpc spheres are filled only for subhaloes with >= 100 bound particles; floats are stored with a lossy 9-bit mantissa; SOAP sums match the particle pass to +-0.05% at identical radii; the R2500c point is biased +2-3% when stacked at the mean radius because of the concentration correlation, so forward-model every halo at its own aperture radii; scripts in research/soap/scripts/ show how the files were read on maas, including paths); ${STUDY}/I_fitting_a2ii/REPORT.md (Test I: per-bin B12/B16 fits were best at z = 0; the Lee22 ladder: PL3 = all gNFW parameters as single power laws in mass (the Battaglia form at fixed z) gave tSZ 9.3% / kSZ 1.1% / DM 1.8% errors, BPLall (broken power law in mass) tSZ 2.2%; the code i_*.py and G's g_models/g_fit/g_selection hold the B12/B16/gNFW forms and fitters to reuse read-only); ${STUDY}/G_physical_outer/report/Test_G_report.tex (B12/B16 equations).
- THE USER'S REQUEST (2026-10-15): "For the fiducial L1_9, fit the B12/B16 parameters for SOAP. I want to have updated parameters for the Battaglia parametrization. Include the results in the website. I want to compare with original parameters and profiles for different masses. Later, I will do a separate run on full particles, and compare the constrained parameters." DECISIONS: (a) all 71 fiducial SOAP-HBT outputs (every redshift available); (b) three fit ranges side by side: within R200c (SO 2500c, 1000c, 500c, 200c and the kpc spheres with x <= 1), within about 1.8 R200c (adds 100c, BN98, 200m, 50c and kpc spheres to x <= 1.8), within 3.3 R200c (adds 5xR500c and kpc spheres to x <= 3.3; note: SOAP counts all gas in each sphere, so beyond about 2 R200c neighbour gas is included, unlike the A2-ii partition); (c) three parametrizations: ORIGINAL Battaglia form (B12: P/P200 = P0 (x/x_c)^gamma [1+(x/x_c)^alpha]^(-beta), alpha = 1, gamma = -0.3 fixed, each of P0, x_c, beta = A (M200c/1e14 Msun)^alpha_m (1+z)^alpha_z; B16: rho_gas/(rho_c(z) f_b) = rho0 (x/x_c)^gamma [1+(x/x_c)^alpha]^(-(beta+gamma)/alpha), x_c = 0.5, gamma = -0.2 fixed, each of rho0, alpha, beta = A M^alpha_m (1+z)^alpha_z; check the exact published forms, P200 definition and units in Battaglia et al. 2012 / 2016 and in XGPaint's implementation ${XGP}/profiles_y.jl, profiles_tau.jl, util.jl), a MASS-BREAK variant (each parameter a broken power law in mass with a common or per-parameter break mass, Lee22 BPLall style, same (1+z) laws), and a FREE-SHAPE variant (original laws plus free gamma for B12 and free x_c and gamma for B16, constant or with their own laws — state which); (d) run now on maas with 8 workers (Test J is running there in parallel; keep within the guards).
- Defaults (state them in the report): central haloes with M200c >= 10^12.7 Msun at each output; Y without recently AGN-heated gas for pressure (Y with it as a check); GasMass for B16 (a gas-density profile; no electron conversion in the fit; give the particle-measured gas-to-electron correction factors from the SOAP audit separately for DM use); the inclusion-free mean over all centrals in each (0.1- or 0.2-dex mass bin, output) cell, forward-modelled at every halo's own aperture radii (use a random subsample of up to ~500 haloes per cell for the forward model if needed, with weights; state it); bootstrap errors over haloes per cell plus a model-discrepancy floor sigma_d (chi^2/nu = 1) in the global fit; pivot M = 1e14 Msun h-free; masks: an aperture is used for a halo only if it is filled and holds enough gas particles (state the threshold, e.g. >= 50); report how many haloes and cells enter per output.
- CLUSTER EXECUTION (user requirement): reads and fits run on STRW maas through ${STUDY}/cluster_tests.py (read its docstring and an earlier test's CLUSTER_RUN.json / write_cluster_run.py / run_test_*.py; mirror / run <TEST> --workers 8 --rss-stop N --driver-args=... / status / fetch / stop). Add ONE entry 'K' to the TESTS dict (another agent may add 'J' concurrently: re-read the file right before editing and add only 'K'). The remote runtime tree /home/nagainis/HalfDome_runtime/overlap_covariance_tests is ours, but its home quota is 3.9 GB and Test J also uses it: keep per-halo intermediates in node-local scratch /tmp/nagainis_halfdome_testK and fetch only reduced products (< 300 MB); delete the scratch after the final fetch and record it. /net/hypernova and the remote study directories are read-only. 8 workers, nice 19, the wrapper's guards; poll every 5 minutes with a loop that terminates on the wrapper's done/failed status (never 'while pgrep -f ...'). If you are a resumed agent, check cluster_tests.py status and ${K} first; never launch a duplicate job.
- Write only in ${K} (and the TESTS entry). Every number you report must come from your results files (cite file/field). Explain every equation with every symbol, meaning and units.`

const RESULT_SCHEMA = {
  type: 'object',
  properties: {
    summary: { type: 'string' },
    report_markdown: { type: 'string' },
    parameter_tables: { type: 'string', description: 'markdown tables: published vs refitted parameters (A, alpha_m, alpha_z for each parameter, with 1-sigma uncertainties) for every field x fit range x parametrization, in XGPaint units' },
    key_numbers: { type: 'array', items: { type: 'object', properties: { name: { type: 'string' }, value: { type: 'string' }, source: { type: 'string' } }, required: ['name', 'value', 'source'] } },
    figures: { type: 'array', items: { type: 'object', properties: { path: { type: 'string' }, caption: { type: 'string' }, main: { type: 'boolean' } }, required: ['path', 'caption', 'main'] } },
    recommendations: { type: 'array', items: { type: 'string' } },
    files_written: { type: 'array', items: { type: 'string' } },
    run_record: { type: 'string' },
    verification: { type: 'string' },
    caveats: { type: 'array', items: { type: 'string' } },
  },
  required: ['summary', 'report_markdown', 'parameter_tables', 'key_numbers', 'figures', 'recommendations', 'files_written', 'run_record', 'verification', 'caveats'],
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

const K_PROMPT = `TASK (Test K — Battaglia B12/B16 refit to FLAMINGO L1_m9 fiducial SOAP-HBT at all outputs). Create ${K}/ (run_test_k.py with stages and a smoke mode, CLUSTER_RUN.json via write_cluster_run.py, results/, figures/, verification.json, SHA256SUMS; same conventions as H_sensitivity and I_fitting_a2ii). Stages:
K0 inventory on maas: list the 71 fiducial SOAP-HBT files, their redshifts and halo counts above 10^12.7 (read headers only); a smoke read of one file.
K1 reads on maas (8 workers, one file per worker): for every output, the centrals with M200c >= 10^12.7: M200c, R200c, M500c, R500c, concentration if present, and for the 9 SO apertures SORadius (convert comoving to physical with a), ComptonYWithoutRecentAGNHeating, ComptonY, GasMass, NumberOfGasParticles, plus the 8 InclusiveSphere GasMass and NumberOfGasParticles (physical kpc). Normalise per halo: Ytilde(<r) = Y_noAGN m_e c^2/sigma_T / (P200 R200c^3) and Mtilde(<r) = M_gas/(f_b M200c), with P200 and rho_c(z) in the B12 convention (state it); store per-halo reduced arrays in the scratch, and per-cell (mass bin, output) means, bootstrap errors and the per-halo radius distributions (or the subsample used for forward modelling) as compact results.
K2 per-cell free fits (diagnostics): in each (0.2-dex mass bin, output) cell with enough haloes, fit B12 (P0, x_c, beta) and B16 (rho0, alpha, beta) freely, forward-modelled at the haloes' own radii, for each of the 3 fit ranges: these show the actual mass and redshift trends of each parameter and where a power law fails.
K3 global fits for each field x fit range x parametrization (original; mass break; free shape): the 9 (or more) law parameters with uncertainties (bootstrap over haloes, refitting; >= 100 replicates if affordable on maas, otherwise the Fisher/Hessian estimate, stated), parameter correlations, chi^2/nu, sigma_d, AIC/BIC, residual maps (data/model - 1 per aperture vs mass and z), and the information each fit range adds.
K4 comparisons: (i) parameter tables published vs refitted (B12 2012 Table/Eq. values; B16 2016 values; also XGPaint's defaults if they differ) in XGPaint units, with a ready-to-paste Julia/Python snippet; (ii) profile comparisons P/P200 and rho/(rho_c f_b) vs x for M200c = 10^12.8, 10^13, 10^13.5, 10^14, 10^14.5, 10^15 Msun at z = 0, 0.5, 1, 2, 3 (and the highest z with data): published vs refitted (each fit range and parametrization) vs the SOAP data points (cell means at the mean aperture radii, labelled as such) and, at z = 0, vs our particle stacks (A2-ii and all gas, fit grid; stacking cache ${OUT}/cache via core/stackcore.py) and the Test I per-bin fits; (iii) aperture integrals Y(<R500c), Y(<R200c), M_gas(<R200c) predicted by each law vs SOAP per (M, z) cell; (iv) the gas-to-electron correction factors for DM use; (v) a machine-readable parameter file (JSON) with every fit, its convention, units, data range (masses, redshifts, radii) and uncertainties, designed so the user's later full-particle run can be fitted with the SAME code and compared parameter by parameter: provide k_fit_core.py (data-source agnostic: accepts per-halo aperture integrals or per-halo shell profiles) and a short usage note in the report.
K5 smoke and parity: the forward model integrates the 3D profile in a sphere; check it against direct numerical integration and against the particle cumulative at z = 0 on a few haloes; check the normalisation and P200 convention by evaluating XGPaint's Battaglia profile (Julia: read ${XGP}/profiles_y.jl and profiles_tau.jl; local Julia per the user's setup if needed, small single-halo evaluations only) with the published parameters against your Python implementation (agreement <= 1e-6 relative), so the refitted parameters can be pasted into XGPaint directly.
Outputs: results (json/npz/csv; reduced only), figures (main: parameter vs mass and vs z with the per-cell free fits and the laws, published law in grey; profile grid published vs refitted at the listed masses and redshifts; residual maps; aperture-integral comparison), verification.json, and the report section 'Battaglia parameters refitted to FLAMINGO (SOAP)' with the parameter tables and recommendations (which fit range and parametrization to use for tSZ, kSZ, DM; how large the change from the published parameters is in profile and in Y/M_gas; known limits: SOAP's inclusive apertures include neighbour gas beyond ~2R, the core inside ~0.27 R200c is constrained only by the R2500c integral, few gas particles below ~10^13 Msun, physical kpc apertures at high z).`

const VERIFY = (res) => `TASK: independently verify Test K in ${K} (another agent's work). Be adversarial. Its output:\n${JSON.stringify({ summary: res && res.summary, key_numbers: res && res.key_numbers, recommendations: res && res.recommendations, run_record: res && res.run_record, verification: res && res.verification, caveats: res && res.caveats }).slice(0, 9000)}\nParameter tables:\n${(res && res.parameter_tables || '').slice(0, 8000)}\nReport text:\n${(res && res.report_markdown || '').slice(0, 16000)}\nChecks (scripts under ${K}/verify_independent/): (1) the published B12/B16 parameter values and forms quoted are correct (check against the papers' equations as implemented in XGPaint ${XGP} and any copies in the repo) and the refitted parameters are in the same convention and units (evaluate one published and one refitted profile both ways); (2) refit at least two global fits yourself from the fetched per-cell data (e.g. B12 original within R200c and B16 original within 1.8 R200c) and compare parameters and chi^2/nu; (3) recompute 6 other headline numbers (a per-cell mean, a residual, a Y(<R500c) ratio, a parameter uncertainty, the halo count at one output, a profile ratio published/refitted at one mass and z); (4) check the forward model at each halo's own radii, comoving-to-physical radius conversion and the kpc apertures at high z; (5) the runs happened on maas (receipts, logs, CLUSTER_RUN.json, SHA256SUMS) and nothing outside ${K} was modified except the TESTS entry; (6) Read 3-4 main figures; (7) every equation's symbols defined; recommendations follow from the numbers. passed = true only if no wrong claim remains; give precise fix instructions. ${LIMITS}`

phase('K fit')
let res = await agent(`${CONTEXT}\n\n${LIMITS}\n\n${K_PROMPT}`, { label: 'implement:Test K', phase: 'K fit', schema: RESULT_SCHEMA })
let ver = null
for (let round = 0; round < 3; round++) {
  ver = await agent(`${CONTEXT}\n\n${VERIFY(res)}`, { label: `verify:Test K:${round}`, phase: 'Verify K', schema: VERIFY_SCHEMA })
  if (ver && ver.passed) break
  if (round === 2) { log(`Test K: still ${ver ? ver.wrong_claims.length : '?'} problems after 3 verifications`); break }
  log(`Test K: verification round ${round} failed (${ver ? ver.wrong_claims.length : '?'} problems) — fixing`)
  res = await agent(`${CONTEXT}\n\n${LIMITS}\n\nTASK: fix Test K in ${K} according to the verifier (rerun on maas if results change; regenerate figures, parameter JSON, verification.json, SHA256SUMS) and return the complete corrected outputs.\nVerifier:\n${JSON.stringify(ver).slice(0, 10000)}\nPrevious report text:\n${(res && res.report_markdown || '').slice(0, 14000)}`, { label: `fix:Test K:${round}`, phase: 'K fit', schema: RESULT_SCHEMA })
}

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
const SITE_BASE = `${LIMITS}\n\nSITE: ${SITE}. Pages are Markdown in ${SITE}/content/<page>.txt with a front-matter block (title, short, section, order, summary); built by ${PY} ${SITE}/build_site.py (--check fails on broken links, missing figures, math or equation-block errors; --syntax prints the exact syntax: equation blocks with latex/meaning/symbols/code, fig: and code: and explorer: links, callouts). Read 2-3 existing pages (index.txt, fitting.txt, snap_soap.txt) to match the style: 'In short' block first, plain precise English, every equation in an equation block with every symbol defined (meaning, units) and a code link, figures read before use and captioned with what to look at, numbers exactly from the source files. IMPORTANT: another workflow (Test J) may edit content files at the same time: never rewrite an existing page wholesale; re-read a file right before each edit and make small targeted edits; create new pages freely. Write only under ${SITE}/.\nTest K summary: ${(res && res.summary || '').slice(0, 5000)}\nTest K files: ${K}/ (results/, figures/, parameter JSON, verification.json, CLUSTER_RUN.json, k_*.py).`

phase('Site')
const pages = await agent(`${SITE_BASE}\n\nTASK (pages): (1) write content/battaglia_soap.txt (section 'fitting'; title like 'Battaglia parameters refitted to FLAMINGO (SOAP)'): the answer first (the recommended refitted parameters per field and fit range, how much they differ from the published B12/B16 values, and in which masses and radii the profiles change most), then the parameter tables (published vs refitted with uncertainties, every fit range and parametrization), the profile comparisons at several masses and redshifts, the trends with mass and redshift, the aperture-integral checks, the z = 0 comparison with the particle stacks and Test I, the limits of SOAP, and how the user's later full-particle run will be compared (the parameter JSON and k_fit_core.py), with equation blocks for both forms and every law; detail pages as needed (e.g. battaglia_soap_details.txt). (2) Build an interactive comparison page ${SITE}/battaglia_compare.html (self-contained, vendored ${SITE}/vendor/plotly.min.js, data in ${SITE}/data/battaglia_params.js generated by a small script ${SITE}/build_battaglia_data.py from the Test K parameter JSON and the SOAP cell means): the user chooses field (pressure / gas density), mass (slider or list, 10^12.7-10^15.5), redshift, fit range and parametrization (multi-select), and sees the published and refitted profiles (and x^2 F or x^3 F), their ratio panel, the SOAP cell-mean points of the nearest cell, and a table of the parameter values at that (M, z); light/dark theme like explorer.html, URL state, works from file:// and as a published artifact; link it from battaglia_soap.txt, from fitting.txt and from the nav if the builder supports external pages (check how explorer.html is linked). Test it with the safe Chrome helper (a few screenshots). (3) Small targeted edits: index.txt (one line in the recipe or results pointing to the refitted parameters), snap_soap.txt (a link: SOAP was used for the refit), about.txt (Test K run record), glossary.json if needed. Rebuild with --check.\nFull Test K report:\n${(res && res.report_markdown || '').slice(0, 45000)}\nParameter tables:\n${(res && res.parameter_tables || '').slice(0, 15000)}\nFigures: ${JSON.stringify(res && res.figures || []).slice(0, 6000)}`, { label: 'site:battaglia-pages', phase: 'Site', schema: SITE_RESULT })

phase('Site verify')
const SITE_VERIFIERS = [
  { key: 'numbers', prompt: `Check every number and parameter value on battaglia_soap*.html and in battaglia_compare.html's data against the Test K results files (and the published B12/B16 values against the papers' forms as implemented in XGPaint ${XGP}); check the equation blocks (forms, P200 convention, units, every symbol defined).` },
  { key: 'interactive', prompt: `Test battaglia_compare.html with the safe Chrome helper (at most 10 screenshots: default, pressure and density at 10^13 / 10^14 / 10^15 Msun and z = 0 / 1 / 2, each fit range, the mass-break and free-shape variants, dark mode, a narrow view via 'dom' if a screenshot cannot go below 500 px): curves drawn, ratio panel correct (published or chosen reference = 1), parameter table matches the JSON, URL state works, no console errors visible in a status line, no network requests. Also screenshot battaglia_soap.html and run a link check over all HTML.` },
  { key: 'completeness', prompt: `Check that the pages answer the user's request completely and readably: updated Battaglia parameters for the fiducial L1_m9 from SOAP (all outputs, three fit ranges, original / mass-break / free-shape), comparison with the original parameters and with profiles at different masses (and redshifts), clear recommendations, limits stated, and a clear path for comparing the user's later full-particle run.` },
]
const vAgent = (v, extra) => agent(`${SITE_BASE}\n\n${extra || ''}TASK (verify, adversarial, independent): ${v.prompt} passed = true only if no blocker or major problem remains.`, { label: `site:verify:${v.key}`, phase: 'Site verify', schema: CHECK })
let checks = await Promise.all(SITE_VERIFIERS.map(v => vAgent(v)))

phase('Site fix')
for (let round = 0; round < 2; round++) {
  const open = checks.flatMap((c, i) => (c && c.problems || []).filter(p => p.severity !== 'minor' || round === 0).map(p => ({ ...p, check: SITE_VERIFIERS[i].key })))
  if (!open.length) break
  log(`site fix round ${round}: ${open.length} problems`)
  await agent(`${SITE_BASE}\n\nTASK (fix): correct these problems (re-check each against the sources); rebuild with --check; re-test affected pages with the safe Chrome helper (at most 8 screenshots) and Read them; cleanup at the end.\nProblems:\n${JSON.stringify(open).slice(0, 30000)}`, { label: `site:fix:${round}`, phase: 'Site fix', schema: SITE_RESULT })
  const failed = SITE_VERIFIERS.map((v, i) => ({ v, i })).filter(({ i }) => checks[i] && !checks[i].passed)
  const re = await Promise.all(failed.map(({ v, i }) => vAgent(v, `RE-VERIFY after fixes. Previously found problems:\n${JSON.stringify(checks[i].problems).slice(0, 12000)}\nCheck each is fixed, then repeat the check.\n`)))
  failed.forEach(({ i }, k) => { checks[i] = re[k] })
  if (checks.every(c => c && c.passed)) break
}

return { K: { res, ver }, pages, checks: checks.map((c, i) => ({ key: SITE_VERIFIERS[i].key, passed: c && c.passed, problems: c && c.problems })) }
