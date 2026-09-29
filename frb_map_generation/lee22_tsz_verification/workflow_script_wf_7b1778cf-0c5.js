export const meta = {
  name: 'lee22-tsz-profile-verification',
  description: 'Verify B12/Lee22 tSZ (and Lee22 density) profiles vs XGPaint, CLASS-SZ, FLAMINGO and TNG (CAMELS)',
  phases: [
    { title: 'Build', detail: 'XGPaint audit, CLASS-SZ, FLAMINGO, CAMELS-TNG comparisons in parallel' },
    { title: 'Verify', detail: 'adversarial re-check of each workstream' },
    { title: 'Synthesize', detail: 'concise figures + report from verified results' },
  ],
}

const ROOT = '/home/cbllover/HalfDome'
const CODE = ROOT + '/frb_map_generation/lee22_tsz_verification'
const OUT = ROOT + '/frb_map_generation/outputs/lee22_tsz_verification_20260929'

const COMMON = `
CONTEXT (shared by all workstreams; date 2026-09-29; repo ${ROOT}, branch cluster).
The user asked: "Also comparison can be made with FLAMINGO profiles. So there is a change in XGPaint code, which would change the
tSZ profiles? I want to verify the tSZ profiles (also with the Lee22). Do it with class-SZ and also the FLAMINGO tSZ. If possible,
compare directly to IllustrisTNG tSZ or pressure profiles." Earlier they said "Use the cluster for computations".

Background already established (do not re-derive; read ${ROOT}/frb_map_generation/LEE22_NORMALIZATION_FORENSICS_20260929.md, 1 page,
and skim ${ROOT}/frb_map_generation/outputs/lee22_normalization_forensics_20260928/FINAL_REPORT.md where needed):
- Lee et al. 2022 (arXiv:2205.01710, MNRAS 517, 5303; fits to TNG300, 20 snapshots z<=2, 1e13-10^14.8 h^-1 Msun, 0.04-1.34 R200c).
  Density nbar = n_e/n200 = n0 (x/xc')^gamma [1+(x/xc')^alpha]^(-beta'), n200 = 200 rho_cr(z) f_b/(X_H m_p); pressure same form,
  P/P200, P200 = 200 G M200 rho_cr f_b/(2 R200) (physical R200, rho_cr(z) physical). alpha=1, gamma=-0.3; no-c Table 7 (pressure) /
  Table 8 (density); broken power law in M at Mcut = 10^13.61 h^-1 Msun (density) / see provenance for pressure.
- Forensic finding M1: Lee22's public extraction code (github leanderthiele/group_particles, examples/ne_prof.cpp, y_prof.cpp,
  PR #1 by bkklee) divides cell sums by COMOVING (ckpc/h)^3 shell volumes (from Group_R_Crit200) but normalizes by the PHYSICAL
  rho_crit(Time) = rho_crit_0 (Omega0/a^3 + OmegaLambda); P200 uses the comoving R200. Prediction: printed density fit too low by
  (1+z)^-3 and printed pressure fit too low by (1+z)^-2 relative to the truth, i.e. true = printed x (1+z)^3 (density), x (1+z)^2
  (pressure). Measured against TNG (Konietzka within-R200 halo DM): density needs C(z) = 0.2841790493880425 (1+z)^2.483190184608004
  (held at z=2 above z=2), relative to the literal eq. 9 normalization. M2: an extra constant density factor F_n ~ 3-8 (unknown origin).
  The pressure a^2 prediction has NOT been tested against simulations yet (open question 3 in the concise report).
- Production conventions: HalfDome cosmology Omega_b=0.049, Omega_c=0.261, h=0.68 (check sigma8, n_s in existing class_sz scripts).
  Catalogue ${ROOT}/lightcone_100.hdf5 (85.2M halos, masses halo_mass_m200c in Msun/h; convert /h). XGPaint electron-to-thermal
  pressure P_e = 0.5176 P_th (Battaglia12 convention). The Lee22 y map converts the Lee22 P_e/P200 fit with XGPaint P0 = P0_Lee/0.5176
  and beta_XG = alpha beta - gamma (see provenance file below).
- y maps (NSIDE 4096, 4 R200c XGPaint projected aperture, halos 0<z<=... see provenance):
  Battaglia12: ${ROOT}/frb_map_generation/outputs/tsz_frb_1m_z1_r200c_apertures/maps/battaglia12_fiducial_halfdome_compton_y_allz_nside4096_m200c_r200cx4.fits
  Lee22 (no-c, Table 7): path given as UPDATED/inputs/lee22_noconc_pressure_compton_y_allz_nside4096_m200c_r200cx4.fits in
  ${ROOT}/frb_map_generation/fullsky_tsz_dm_comparison.py (resolve UPDATED there); calibrated-range variant and provenance:
  ${ROOT}/frb_map_generation/outputs/tsz_dm_fullsky_20260918/maps/lee22_noconc_pressure_compton_y_calib_zmax2p0_nside4096_m200c_r200cx4{.fits,_provenance.txt}.
  Auto spectra already exist: ${ROOT}/frb_map_generation/outputs/tsz_dm_fullsky_20260918/spectra/fullsky_spectra.npz keys cl_y_b12,
  cl_y_lee22p, cl_y_lee22p_calib (map2alm lmax 8192, iter 3; check its metadata for pixel-window convention).
- XGPaint source: dev checkout /home/kn18001/.julia/dev/XGPaint (branch cluster). Its git log since 2026-09-24 shows only
  07fc686 (print-format/typo fixes in src/profiles.jl), 35f79ad (restore N_logtheta), 1660e08 (tests). Today's only code change is an
  opt-in density option redshift_scaling=:tng_calibrated in ${ROOT}/frb_map_generation/lee2022_frb_dm_profile.jl (HalfDome repo, FRB DM /
  density only, not XGPaint, not pressure).
- Validated Python formulas (Lee22 n and P, B12 P_th and P_e, B16, Arnaud10 UPP; transcription checked to 6e-16 against the TeX):
  ${ROOT}/frb_map_generation/lee22_normalization_forensics/C_pressure_consistency/lee22_models.py (import it; do not modify it). A halo-model
  helper exists in .../F_sz_budget/szhm.py.
- Earlier results to reuse: tSZ truncation series (${ROOT}/truncation_comparison/, radius_series/, TSZ_TRUNCATION_RADIUS_SERIES_20260925.md:
  B12 map / CLASS-SZ(x_out=N) = 0.90-1.00 over ell 100-3500, a mass-function offset); HalfDome vs FLAMINGO tSZ
  (${ROOT}/flamingo_pressure_calibration/halfdome_tsz_comparison/, README section "HalfDome painted with the FLAMINGO fit"; products/ holds
  per-map spectra caches; B12 painted/FLAMINGO lc0 = 0.70/0.69/0.69 at ell 1000/2000/3000); FLAMINGO L2p8_m9 pressure stacks
  (${ROOT}/flamingo_pressure_calibration/: cluster_results/, notebook_cache/stacks/, notebook_cache/particles/, flamingo_stacks.py,
  stack_products.py, README.md, FLAMINGO_L2p8_preliminary_stacks.ipynb).

TOOLS / ENVIRONMENT:
- Local Python: /home/cbllover/miniconda3/envs/halfdome/bin/python (h5py, matplotlib, numpy; export
  LD_LIBRARY_PATH=/home/cbllover/miniconda3/envs/halfdome/lib before importing scipy). Check whether healpy is available; if not, find
  another env (e.g. the class_sz env) rather than installing.
- Local Julia (for production-profile checks): /home/kn18001/.julia/juliaup/julia-1.12.2+0.x64.linux.gnu/bin/julia with
  JULIA_DEPOT_PATH=/home/kn18001/.julia (default env resolves XGPaint from the dev checkout). Never call the juliaup launcher as root.
- CLASS-SZ: PYTHONPATH=/home/kn18001/.local/lib/python3.11/site-packages PATH_TO_CLASS_SZ_DATA=/home/kn18001/class_sz_data_directory
  CUDA_VISIBLE_DEVICES="" /home/kn18001/.conda/envs/class_sz/bin/python ... Only ONE class_sz process at a time on this machine
  (parallel imports segfault). Parameter names: x_outSZ, class_sz_verbose; template ${ROOT}/truncation_comparison/compute_class_sz_references.py
  and radius_series/compute_class_sz_radius_series.py.
- Cluster idark (PBS): from WSL always use /mnt/c/Windows/System32/OpenSSH/ssh.exe -o BatchMode=yes idark "..." and scp.exe (Windows binaries:
  copy files through the Windows temp dir, e.g. scp.exe idark:/remote/file 'C:\\Users\\kn18001\\AppData\\Local\\Temp\\name' then cp
  /mnt/c/Users/kn18001/AppData/Local/Temp/name <local>; for uploads the reverse). Pipe ssh output through tr -d '\\r'. User kristero10;
  work dir /lustre/work/kristero10/ (create your own subdir named lee22_tsz_verification_20260929_<workstream>). Queue mini, e.g.
  #PBS -q mini, -l select=1:ncpus=8:mem=48gb, walltime as needed; max 6 running jobs per user and 4 are already used by other jobs of this
  session, so run at most ONE PBS job at a time yourself. Python on idark: /home/anaconda3/bin/python3.8 (h5py 3.3, numpy 1.24,
  matplotlib 3.3; check scipy); /home/kristero10/.conda/envs/SZ/bin/python is Python 3.6 with healpy. The login node has internet
  (curl to users.flatironinstitute.org works); check before assuming compute nodes do. Lustre is 96% full (125 TB free): keep downloads
  < 60 GB and delete raw snapshots when done.
- Local machine: 20 cores / 31 GB shared with other work: keep local jobs light (<= 8 threads, < 12 GB RAM, < 15 min each). Heavy
  computation (raw snapshot processing, map painting, big downloads, particle streaming) goes to idark.

HARD RULES: do not modify existing files outside your own new directories (${CODE}/<workstream>/ for code, ${OUT}/<workstream>/ for
outputs; create them). In particular do not touch lee2022_frb_dm_profile.jl, generator/PBS files, lee22_models.py, XGPaint, or other
sessions' directories (frb_map_generation/tng300_lee22_comparison/, lee22_fig5_audit/, lee22_pdf_ksz_comparison/) - reading is fine.
Never modify git config, never commit or push. Never write wait loops of the form 'while pgrep -f "<pattern>"' (they match themselves
and never exit); wait on a PID, a file, or poll qstat instead. Do not delete anything you did not create. Be explicit about units and
conventions (physical vs comoving, h factors, M200c definitions, P_e vs P_th, f_b, mu_e, X_H) in every comparison - these conventions
are exactly what went wrong in Lee22. Colours for any figure: data black, Battaglia12/Battaglia16 #0072B2, Lee22 as painted/production
#E69F00, Lee22 corrected #009E73, Lee22 printed literal #D55E00, other #CC79A7; render each figure and look at it (Read the PNG) to fix
overlaps. Write a short README.md in your output dir listing files, conventions and results.`

const RESULT_SCHEMA = {
  type: 'object',
  properties: {
    summary: { type: 'string', description: '5-15 sentence summary of what was done and found' },
    key_numbers: { type: 'array', items: { type: 'object', properties: {
      name: { type: 'string' }, value: { type: 'string' }, context: { type: 'string' } }, required: ['name', 'value', 'context'] } },
    claims: { type: 'array', items: { type: 'object', properties: {
      claim: { type: 'string' }, evidence: { type: 'string' }, how_to_reproduce: { type: 'string' },
      confidence: { type: 'string', enum: ['high', 'medium', 'low'] } }, required: ['claim', 'evidence', 'how_to_reproduce', 'confidence'] } },
    figures: { type: 'array', items: { type: 'string' } },
    data_products: { type: 'array', items: { type: 'string', description: 'absolute path + one-line description' } },
    caveats: { type: 'array', items: { type: 'string' } },
    open_items: { type: 'array', items: { type: 'string' } },
  },
  required: ['summary', 'key_numbers', 'claims', 'figures', 'data_products', 'caveats', 'open_items'],
}

const VERDICT_SCHEMA = {
  type: 'object',
  properties: {
    checks: { type: 'array', items: { type: 'object', properties: {
      claim: { type: 'string' }, verdict: { type: 'string', enum: ['confirmed', 'refuted', 'uncertain'] },
      evidence: { type: 'string' }, corrected_statement: { type: 'string' } }, required: ['claim', 'verdict', 'evidence'] } },
    missed_issues: { type: 'array', items: { type: 'string' } },
    overall: { type: 'string' },
  },
  required: ['checks', 'missed_issues', 'overall'],
}

const STREAMS = [
  {
    key: 'A_xgpaint_audit',
    prompt: `WORKSTREAM A - XGPaint / painted-profile audit and production profile tables.
Goals:
1. Answer precisely: did any code change (XGPaint or HalfDome) alter the tSZ profiles or y maps? Identify which XGPaint commit / profile
   code / interpolator cache painted the Battaglia12 y map and the Lee22 y map (provenance files, logs, run scripts; Lee22 interpolator
   cache lee22_noconc_pressure_m200c_logm15p7_tsz_interpolator.jld2 in frb_map_generation/outputs/halfdome_frb_inputs/), and diff the
   XGPaint pressure code (src/profiles*.jl, anything in the y path) between that commit and HEAD. State whether today's
   tng_calibrated density option can reach any tSZ path (grep all callers).
2. Verify the painted 2D profiles against independent line-of-sight integration: evaluate the production y(theta) interpolators (B12 and
   Lee22, exactly as used by the painter, including the 4 R200c projected aperture and mass/h conversion) at log10 M200c/Msun in
   {13, 13.5, 14, 14.5, 15}, z in {0.1, 0.5, 1, 2}, 30 angles from 0.02 to 4 R200c; compare with a Python/Julia LOS integral of the
   3D P_e from lee22_models.py (b12_Pe_over_P200 and lee_Pbar with the stated XGPaint mapping) with y = sigma_T/(m_e c^2) int P_e dl.
   Report max relative differences; explain any systematic offset (interpolation, aperture, 0.5176 factor).
3. Export shared tables for the other workstreams and the synthesis: 3D P_e(r) (keV cm^-3, physical) and P_e/P200, and n_e(r) (cm^-3,
   physical) and n_e/n200, at log10 M200c/Msun 12.5..15.5 step 0.25, z in {0, 0.25, 0.5, 0.75, 1, 1.5, 2, 3}, 60 radii 0.01-5 R200c,
   for: B12 (XGPaint), Lee22 pressure as painted, Lee22 pressure x(1+z)^2 (M1 bookkeeping hypothesis), B16 n_e (XGPaint), Lee22 n_e
   literal, x0.6016 (production xgpaint_ne2d), x C(z) (tng_calibrated). Save as ${OUT}/A_xgpaint_audit/profile_tables.npz (+ a JSON
   describing arrays and units). Cross-check the Lee22 density numbers against the Julia production lee2022_frb_dm_profile.jl (load it
   in Julia read-only via include in a scratch script) for a few points.
4. Re-read group_particles examples/y_prof.cpp (fetch from GitHub; cite lines) and confirm or correct the code-level pressure
   bookkeeping factor (is it (1+z)^-2 exactly? which quantity - cell Density, InternalEnergy, volumes, R200 - carries which a-power?).
   Give the exact corrected-pressure hypothesis to test.
This workstream runs locally (light).`,
  },
  {
    key: 'B_class_sz',
    prompt: `WORKSTREAM B - CLASS-SZ verification of the Battaglia12 and Lee22 tSZ.
Goals:
1. Profile level: compare CLASS-SZ's B12 pressure profile (3D P_e or P_th vs r, and its ell-space y_ell(M,z) or projected y) with XGPaint's
   B12 (lee22_models.b12_Pe_over_P200 reproduces XGPaint; verify that with the XGPaint Julia code for a couple of points if cheap) at
   log10 M200c/Msun 13-15, z 0.1-2. Check CLASS-SZ's conventions (P_e vs P_th, 0.5176, M200c vs M500c, h units, f_b) from its source/docs.
2. Map level, B12: CLASS-SZ C_ell^yy (1h + 2h) in HalfDome cosmology with the HalfDome catalogue's mass and redshift bounds and x_outSZ=4,
   against cl_y_b12 from fullsky_spectra.npz (apply the right pixel-window convention; maps are point samples). Reuse the existing
   truncation-series CLASS-SZ setup and results; state which mass function is used and quote the catalogue-vs-mass-function offset.
3. Lee22 with CLASS-SZ: find how to represent the Lee22 no-c pressure fit in CLASS-SZ (inspect the installed classy_sz source: generalized
   B12 parameters such as P0_B12, xc_B12, beta_B12, alpha_m_*, alpha_z_*, alphap_m_* (above a mass cut), mcut, c-dependence; or a custom /
   tabulated profile). Match exactly what was painted (P_e fit converted as in the provenance, shape clip above 10^14.8 h^-1 Msun, P0
   extrapolated with the high-mass slope). If CLASS-SZ cannot represent it exactly, split the mass range (one-halo terms add over mass
   ranges) and/or validate an independent Python halo-model integrator on B12 against CLASS-SZ (<2%) and use it for Lee22. Compute
   C_ell^yy(Lee22 as painted) and compare with cl_y_lee22p and cl_y_lee22p_calib. Also give the mass and redshift decomposition of the
   Lee22/B12 ratio at ell = 100, 300, 1000, 3000 (which halos cause the 1.7-2.7x low-ell excess?).
4. Also compute C_ell^yy for the Lee22 pressure x(1+z)^2 bookkeeping hypothesis (M1) to show how much the tSZ would change if the pressure
   fit carries the same error as the density.
Only one class_sz process at a time. Runs locally (class_sz tSZ runs take ~20 s).`,
  },
  {
    key: 'C_flamingo',
    prompt: `WORKSTREAM C - FLAMINGO pressure AND gas-density profiles and FLAMINGO tSZ, against B12/B16 and Lee22.
Goals:
1. Pressure: collect the FLAMINGO L2p8_m9 stacked electron-pressure profiles (all available snapshots z=0..3 and mass bins; prefer the
   full-statistics idark pipeline results in flamingo_pressure_calibration/cluster_results/ if complete, else the preliminary stacks) and
   compare P_e/P200 vs r/R200c with B12, Lee22 as painted, and Lee22 x(1+z)^2 (M1 hypothesis), in FLAMINGO's cosmology (P200 and R200c
   computed consistently in each simulation's own cosmology; state the FLAMINGO P_e definition and any X_H/mu differences). Quantify
   per (M, z): log ratio model/FLAMINGO at 0.1, 0.3, 0.5, 1 R200c and Y(<R200c) (spherical) ratio, and the REDSHIFT TREND at fixed mass
   (fit dlog(P/P200)/dlog(1+z) at fixed x for FLAMINGO vs each model) - this is the test of the Lee22 pressure (1+z)^-2 bookkeeping.
2. Density: find or compute FLAMINGO gas / electron density profiles (n_e/n200 or rho_gas/(f_b rho_cr)) and gas fraction inside R200c
   vs z for the same stacks. Check existing products first (stack products, local particles/*.npz caches carry masses/positions?). If they
   must be computed, do it on idark (streaming from COSMA is ~12 MB/s there vs 0.3 MB/s locally; reuse flamingo_pressure_calibration code
   and its hdfstream setup, see README and flamingo_reduced_io.py; the cluster venv of 2026-09-21). Compare with B16, Lee22 literal,
   x0.6016 and x C(z); fit the redshift trend of n_e/n200 at fixed x vs the Lee22 printed trend - independent test of the density
   bookkeeping in a different simulation (FLAMINGO is not TNG, so absolute levels differ; the redshift trend relative to self-similar
   scaling is what matters).
3. tSZ maps: extend the existing HalfDome-vs-FLAMINGO y-power comparison (halfdome_tsz_comparison/products) with the Lee22 y map
   (cl_y_lee22p; same conventions as the B12 products there: pixel window, lmax, cosmology correction factor F from CLASS-SZ already
   in the README - do not rerun class_sz) so that D_ell for B12, FLAMINGO-fit-painted, Lee22 and FLAMINGO lc0 (and lc1 spread) appear
   together with ratios.
Prefer existing products; heavy computation on idark.`,
  },
  {
    key: 'D_tng_camels',
    prompt: `WORKSTREAM D - direct IllustrisTNG-physics pressure and density profiles from CAMELS (public), against Lee22/B12/B16.
TNG300 particle data are NOT accessible (no TNG API key; the API returns 403). CAMELS IllustrisTNG CV set (27 boxes of 25 Mpc/h, 256^3,
fiducial TNG model; gas-cell mass ~1.3e7 h^-1 Msun, close to TNG300-1) is public:
  raw: https://users.flatironinstitute.org/~camels/Sims/IllustrisTNG/CV/CV_<i>/snapshot_NNN.hdf5 (2.66 GB each) and groups_NNN.hdf5 (16 MB),
       new 91-snapshot numbering (090 = z 0); find the redshift table (extra_files/ or the header).
  precomputed 3D profiles (Moser+22 illstack): https://users.flatironinstitute.org/~camels/Profiles/IllustrisTNG/CV/CV_<i>/IllustrisTNG_CV_<i>_NNN.hdf5
       (0.7 MB; old 34-snapshot numbering, 033 = z 0?; datasets Profiles (4, Nhalo, 25), n (4, Nhalo, 25), r (25), Group_M_Crit200,
       Group_R_Crit200, GroupPos?, ...). One z=0 file is already at
       /tmp/claude-0/-home-cbllover-HalfDome/015746ae-94b7-42cb-8dc5-0e8c33dc8490/scratchpad/camels/IllustrisTNG_CV_0_033.hdf5.
Goals:
1. Establish the precomputed-profile conventions (which 4 quantities, units, comoving vs physical r and density, h factors, how pressure
   is defined (thermal or electron), radial bins) from the CAMELS documentation (camels.readthedocs.io, the Moser+22 paper / illstack
   code) AND verify them by computing profiles yourself from raw snapshots on idark for CV_0 at z~0 and z~1 (download on the idark login
   node or a job if compute nodes have internet; delete raw snapshots afterwards). Direct computation: gas cells within 5 R200c of
   GroupPos (periodic box), n_e = ElectronAbundance * X_H * rho / m_p with rho = Density converted to PHYSICAL units (a^-3 h^2), T from
   InternalEnergy and ElectronAbundance (mu = 4/(1+3X_H+4X_H x_e) m_p), P_e = n_e k_B T, spherical shells in PHYSICAL radius, volume
   weighting by cell volume Masses/Density (and note star-forming cells). Treat these direct profiles as the ground truth; use the
   precomputed ones for more boxes only if they agree.
2. Build TNG-physics stacks for all 27 CV boxes (precomputed profiles, if validated; otherwise direct computation for as many boxes as a
   <= 60 GB download budget allows, e.g. 4-6 boxes at z = 0, 0.5, 1, 2) in mass bins log10 M200c/Msun 12.5-13, 13-13.5, 13.5-14 (and
   higher if any) at z ~ 0, 0.5, 1, 1.5, 2: n_e/n200 and P_e/P200 vs r/R200c (Lee22 definitions, physical, own cosmology Om 0.3 Ob 0.049
   h 0.6711), gas (and electron) fraction inside R200c, and Y(<R200c)/(sigma_T/m_e c^2 P200 R200^3 ...) in consistent units.
3. Compare with Lee22 printed density and pressure, Lee22 density x C(z), Lee22 pressure x (1+z)^2 (M1 hypothesis), B12, B16: log
   ratios at 0.1/0.3/0.5/1 R200c, and the redshift trend at fixed mass and x. This is the direct test of M1 for both the density and the
   pressure and of the constant F_n: report the measured n_e/n200 and P_e/P200 normalization vs z relative to the printed Lee22 values.
4. Emulate Lee22's bookkeeping on the same CAMELS cells: shell sums divided by comoving (ckpc/h)^3 volumes from Group_R_Crit200,
   normalized by physical rho_crit(z) n200 and P200 with comoving R200, then fit/compare: does the emulation reproduce the redshift
   dependence (and level?) of the printed Lee22 fits? Read the group_particles code (github leanderthiele/group_particles,
   examples/ne_prof.cpp, y_prof.cpp) to emulate exactly which quantities they sum.
Mention the caveats: CAMELS boxes are small (few halos above 1e13.5), TNG model at 256^3/25 Mpc/h (not TNG300 volume), different
cosmology; halo-to-halo scatter; star-forming gas treatment. Heavy parts on idark.`,
  },
]

const results = await pipeline(
  STREAMS,
  s => agent(`${s.prompt}\n\nCode dir: ${CODE}/${s.key}/  Output dir: ${OUT}/${s.key}/\n${COMMON}\n\nReturn the structured result; every claim must name the file/command that reproduces it.`,
    { label: `build:${s.key}`, phase: 'Build', schema: RESULT_SCHEMA }),
  (res, s) => res ? agent(`You are an adversarial verifier for workstream ${s.key} of a tSZ/Lee22 profile verification. The workstream's task was:\n${s.prompt}\n\nIts result (JSON):\n${JSON.stringify(res)}\n\nTry to REFUTE its load-bearing claims and key numbers: re-derive at least the 3-5 most important ones independently (write your own short scripts in ${CODE}/verify_${s.key}/ and outputs in ${OUT}/verify_${s.key}/; do not modify the workstream's files). Hunt especially for convention errors (physical vs comoving, h factors, a-powers, P_e vs P_th and 0.5176, M200c vs M500c, f_b/mu_e/X_H, pixel windows, mass/redshift selection mismatches, stacking weights, cosmology mismatch). Look at its figures (Read the PNGs) for errors in labels/units. Default to 'uncertain' when you cannot check. Keep compute light; the class_sz rule (one process at a time) applies - the class_sz workstream has finished before you start only if you are verifying B_class_sz, otherwise do NOT run class_sz.\n${COMMON}`,
    { label: `verify:${s.key}`, phase: 'Verify', schema: VERDICT_SCHEMA }).then(v => ({ key: s.key, result: res, verdict: v })) : null,
)

const ok = results.filter(Boolean)
log(`verified workstreams: ${ok.map(r => r.key).join(', ')}`)

phase('Synthesize')
const synth = await agent(`Synthesize the verified results of a tSZ / Lee22 profile verification into concise figures and a short report.
Inputs (JSON of each workstream result + its adversarial verdict):\n${JSON.stringify(ok)}\n
Use only claims that were confirmed or not refuted; where a verifier refuted or corrected something, use the corrected statement and say so.
Produce in ${OUT}/synthesis/ (code in ${CODE}/synthesis/):
 - fig_tsz1_pressure_profiles.{png,pdf}: P_e/P200 vs r/R200c at z ~ 0, 0.5, 1, 2 for 2-3 mass bins: TNG (CAMELS direct), FLAMINGO stacks, B12
   (= XGPaint = CLASS-SZ), Lee22 as painted, Lee22 x(1+z)^2. Ratio panels to the simulations.
 - fig_tsz2_redshift_trend.{png,pdf}: normalization vs z at fixed mass for pressure (and density): the direct test of the Lee22 bookkeeping
   (TNG, FLAMINGO, printed Lee22, corrected Lee22, B12/B16, and the Lee22-bookkeeping emulation on CAMELS if available).
 - fig_tsz3_yy_power.{png,pdf}: D_ell^yy of the HalfDome B12 and Lee22 maps, CLASS-SZ (B12, Lee22 as painted, Lee22 x(1+z)^2), FLAMINGO lc0 (+lc1),
   with ratio panel.
 - optionally fig_tsz4_density.{png,pdf}: gas density / gas fraction vs z (TNG CAMELS, FLAMINGO, Konietzka-implied, B16, Lee22 literal / x0.6016 / x C(z)).
Reuse the workstreams' data products (profile_tables.npz etc.); do not recompute heavy things. Figure style: white background, DejaVu Sans,
large readable fonts, colours as specified below, direct labels where lines are close, legends not covering data, consistent axis ranges;
render and inspect every PNG (Read it) and iterate until there are no overlaps or clipped labels.
 - ${ROOT}/frb_map_generation/LEE22_TSZ_VERIFICATION_20260929.md: at most ~1.5 pages. Sections: Bottom line (3-5 sentences; answer "is there
   a change in XGPaint that changes the tSZ profiles?" explicitly), 1. Is the painting right (XGPaint vs independent LOS vs CLASS-SZ), 2. B12
   and Lee22 vs simulations (TNG/CAMELS, FLAMINGO) - profiles and redshift trend; verdict on the Lee22 pressure (1+z)^-2 bookkeeping,
   3. y power (maps vs CLASS-SZ vs FLAMINGO), 4. What changes if Lee22 pressure is corrected, 5. Caveats and open items. A table of key
   numbers. Reference figures by relative path. Write for a cosmologist; no filler.
${COMMON}
Return a JSON-free plain-text summary (<= 25 lines) of the bottom line, the key numbers, the figure paths and anything refuted.`,
  { label: 'synthesis', phase: 'Synthesize' })

return { synthesis: synth, streams: ok.map(r => ({ key: r.key, summary: r.result.summary, overall: r.verdict.overall, refuted: r.verdict.checks.filter(c => c.verdict === 'refuted') })) }
