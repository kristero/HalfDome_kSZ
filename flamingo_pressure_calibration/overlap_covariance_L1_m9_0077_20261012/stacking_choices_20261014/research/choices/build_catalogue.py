"""Write choices_catalogue.json: every user-facing choice of the extraction and stacking code that affects the 1D stacks.

Paths in 'where' are relative to the study root
/home/cbllover/HalfDome/flamingo_pressure_calibration/overlap_covariance_L1_m9_0077_20261012 unless they start with '../'
(= /home/cbllover/HalfDome/flamingo_pressure_calibration/).  Numbers in 'evidence' come from quantify_choices.json (this
directory, recomputed from the stored products), from the run records named there, or from the study reports.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
Q = json.loads((HERE / "quantify_choices.json").read_text())
C = []


def add(group, name, where, current, alternatives_local, alternatives_new_pass, why, evidence="", pipelines_differ=""):
    C.append({"group": group, "name": name, "where": where, "current": current,
              "alternatives_local": alternatives_local, "alternatives_new_pass": alternatives_new_pass, "why": why,
              "evidence": evidence, "pipelines_differ": pipelines_differ})


G1, G2, G3, G4, G5, G6 = ("1 sample & catalogue", "2 extraction geometry", "3 particle selection",
                          "4 quantities & estimators", "5 radial grids", "6 normalisations")
G7, G8, G9, G10, G11 = ("7 resolution criteria", "8 stacking statistic & weights", "9 mass binning",
                        "10 fitting & truncation", "11 performance knobs")

# ============================================================================ 1 sample & catalogue
add(G1, "Simulation, snapshot, halo finder",
    "study_config.json:2-8",
    "L1_m9 HYDRO_FIDUCIAL (L1000N1800), snapshot 0077 (z = 0), SOAP-HBT halo_properties_0077 + HBT membership_0077",
    "none (the products hold one snapshot of one run)",
    "other snapshots with full particles on maas: 0000, 0037, 0047, 0057, 0061, 0067, 0071, 0073, 0074 (z = 15 ... 0.15); "
    "feedback variants only at 0074/0077 and without membership files (ownership would need a SOAP-based reconstruction); "
    "L2p8_m9 / L1_m8 reduced snapshots (R100c only) over hdfstream",
    "Redshift evolution and feedback dependence of the profiles cannot be inferred from one z = 0 snapshot; HalfDome paints a lightcone.",
    "", "L2p8 pipeline: 10 snapshots z = 0-3 of L2p8_m9 reduced (../l2p8_reduced_calibration/config.json:4); fullsnap plan: "
        "L2p8_m9 11 snapshots + L1_m9/L1_m8 resolution pairs (../fullsnap_calibration/config.json:6-11).")
add(G1, "Central definition",
    "census.py:270-271 (SOAP/HostHaloIndex < 0)",
    "central = SOAP row without a host (radii234 convention)",
    "identical to InputHalos/IsCentral on this snapshot (0 disagreements, census_summary.json soap_consistency)",
    "satellites as profile centres (HalfDome paints no satellites, so not needed)",
    "HalfDome's RFoF catalogue paints distinct haloes only; satellite gas belongs to its host's profile.",
    "census_summary.json: iscentral_disagrees_with_host = 0, host_not_central = 0",
    "L2p8 uses InputHalos/IsCentral (../l2p8_reduced_calibration/catalogue_stage.py:61): same set at L1_m9 z = 0.")
add(G1, "Mass and radius definition",
    "census.py:274-276; prepare_study.py:262 (SO/200_crit/TotalMass, SORadius); SO/500_crit read at prepare_study.py:41",
    "M200c / R200c (all particle types inside the SOAP 200c sphere); M500c / R500c stored for the X500 grid",
    "M500c / R500c (X500 grid, B24 / A10 / self-similar-500 normalisations); host-only mass (M200c minus satellites, Test A) "
    "from the overlapper table; bin by M500c (Test D B24 recipe)",
    "M200m, Mvir, BN98 or any other SO definition as the scaling radius (the shells are stored in R200c and R500c units only)",
    "The scaling radius sets which profiles look self-similar; HalfDome's catalogue is in M200c, so M200c/R200c is the painting convention.",
    "R500c/R200c median 0.656 (5-95%: 0.606-0.682), quantify_choices.json extraction_sphere")
add(G1, "Mass range of the sample",
    "study_config.json:9-26; census.py:321-324",
    "12.5 <= log10 M200c < 15.5 (15 bins of 0.2 dex); 421,868 centrals in the pool",
    "any sub-range; the calibrated (supported) range is 13.5-15.5 with log M <= 15.388 (F_observables/f_common.py:146-154)",
    "below 10^12.5 (needs a new pass; resolution there is poor: no 0.2-dex bin below 13.3 passes the strict mask)",
    "HalfDome paints above 6.6e12 Msun; DM and kSZ get 42-50% of mean DM from haloes below 10^13.5, where the published fallback is still used (SYNTHESIS 2.1).",
    "census_summary.json pool_centrals = 421,868")
add(G1, "Sampling per mass bin",
    "study_config.json:27-37 (old_sample); inputs/radii234_selected_halos.npz",
    "300 random centrals per 0.2-dex bin below 10^14.5 (seed 20261008), all centrals above 10^14.5",
    "sub-sampling (fewer haloes; bootstrap / CV folds exist), restricting to the 4,375 radii234 rows",
    "more haloes per bin (each extra halo costs its cells: ~307.8 GB logical for 4,588 haloes, census_summary.json); "
    "complete samples below 10^14.5",
    "Sets the statistical error of each stack (N_eff ~300 per bin below 14.5) and the size of the read.",
    "sample per bin 308-332 below 14.5; 895/326/127/21/6 above (metadata_plan.json sample_per_bin)",
    "fullsnap plan: random 2^3-cell blocks (2% of blocks) + complete above 10^14.3 capped at 50 per 0.05 dex "
    "(../fullsnap_calibration/config.json:13-17); L2p8: every halo in the reduced snapshot (<= 200 per 0.05 dex above 1e13).")
add(G1, "Stratified enlargement",
    "study_config.json:56-70; census.py:228-258 (enlargement_plan)",
    "applied: below 10^14.5 every (bin, census class) stratum topped up to >= 50 haloes; 213 annulus-only additions (seed 20261012)",
    "drop the 213 additions (role == 1 in selected_halos.npz) and use the radii234 sample alone",
    "other strata (e.g. by concentration or environment), larger targets",
    "Gives enough contaminated (CONT-annulus) haloes for Test A; it makes halo weights non-uniform within a bin.",
    "census_summary.json proposed_additions = 213; class counts per bin there")
add(G1, "Census class definition (overlap)",
    "study_config.json:38-50; census.py:37, 161-166, 337-338",
    "overlapper = other subhalo with mu = M_j/M200c >= 0.1 reaching inside f = 2 R200c (d < 2 R200c + R_j); classes clean / cont_reach1R / cont_annulus_only",
    "mu >= 0.03 / 0.1 / 0.3 and kind all / own satellites / neighbours / neighbour centrals (newdata.py:259-302)",
    "mu < 0.03 or reach f > 2 (overlapper lists stop there)",
    "Defines the strata for the inclusion weights and the clean/contaminated sub-samples; calibrating on clean haloes biases yy by +0.7 to +7.9% (SYNTHESIS 2.2).",
    "1,650 of 4,588 centrals overlapped (SYNTHESIS 2.2)")
add(G1, "Inclusion weights",
    "prepare_study.py:284-295; analysis/newdata.py:242-244",
    "w_h = 1 / inclusion probability of the (bin, class) stratum (Horvitz-Thompson); 1 above 10^14.5",
    "equal weights; weights x Y200 (B12 convention, D_literature/td_recipes.py:133-139); per-bin re-normalisation for any new bin definition (weights stay valid for any sub-selection)",
    "-",
    "With the enlargement, equal weights over-represent the CONT-annulus stratum (neighbours in 1-2R) and bias the outer stack high.",
    "equal-weight / inclusion-weighted stack - 1, largest per-bin deviation in 13.5-14.5 (equal weights read high): pressure 0.4-0.8% inside 1R, 2.1-5.3% at 1-2R, 3.4-9.4% at 2-4R; "
    "electrons +1.7..4.3% at 2-4R (quantify_choices.json; F_observables/results/profiles.npz). Weight max/min within a bin 1.56-1.88.",
    "radii234, L2p8, flamingo_stacks and fullsnap all use equal halo weights (radii234: correct for its uniform sample; "
    "../radii234_contamination_L1_m9_0077_20261008/fit_study.py:114).")
add(G1, "Census mass floor and painting threshold",
    "study_config.json:88-91; census.py:39-42",
    "census tree keeps subhaloes with M_j >= 9.4e10 Msun; 'painted' = host M200c >= 6.6e12 Msun",
    "none for the floor; the painted flag enters owner codes 14/15 and the A2 rules",
    "a different painting threshold (owner 14 vs 15 split and the A2-ii neighbour list are fixed in the products)",
    "The painting threshold decides which neighbour gas would be painted twice (A2-i/A2-ii).",
    "")

# ============================================================================ 2 extraction geometry
add(G2, "Extraction sphere",
    "study_config.json:95-97; prepare_study.py:281-282; census.py:357",
    "r <= r_ext = max(4 R200c, 6 R500c) per halo",
    "owner200 blocks: r <= 4 R200c only (measure_overlap.py:177, 207); owner500: r <= 6 R500c (measure_overlap.py:181-182, 209); "
    "any outer radius <= 4 R200c (X200) or <= 6 R500c (X500) that is a stored edge",
    "anything beyond 4 R200c / 6 R500c (e.g. 5 R200c, 6 R200c or the 2-halo regime) needs a new pass with a larger sphere",
    "The outer extent fixes the largest truncation radius and the 2-4R gas that carries 37% of mean DM, 64% of DM Q16 and 25% of kSZ 1h at ell = 100 (SYNTHESIS 2.4).",
    "6 R500c > 4 R200c for 26.5% of haloes; r_ext/R200c median 4.00, max 4.24; 6 R500c = 3.94 R200c (median) (quantify_choices.json)",
    "L2p8 reduced: R100c only (../l2p8_reduced_calibration/particle_stage.py:83-88, 150); flamingo_stacks: R100c; fullsnap: 5 R200c (../fullsnap_calibration/config.json:18).")
add(G2, "Profile centre",
    "census.py:272; dependencies/flamingo_io.py:73 (InputHalos/HaloCentre)",
    "SOAP InputHalos/HaloCentre (the halo finder's centre)",
    "none for the profiles; per-halo |CoM - centre| stored (selected_halos.npz centre_offset_R200c) for selections / diagnostics",
    "re-centring on the centre of mass, the gas density peak or the X-ray / SZ peak",
    "Mis-centring flattens the inner profile in the stack; HalfDome paints at its own halo centres (N-body), so the choice should mimic theirs.",
    "centre offset |CoM - centre|/R200c: median 0.087, 90th pct 0.184, 63% > 0.07 (quantify_choices.json)")
add(G2, "Periodic wrapping and sphere test",
    "extract_study.py:108-120 (select_sphere; 1e-9 tolerance at :110); measure_overlap.py:175-177",
    "minimum-image offsets; per-axis pre-filter then exact |delta|^2 <= r^2; particles exactly at the outer edge go to the last shell (grids.py:59-61)",
    "-", "-",
    "Correctness only (no physics choice); verified bit-identical against radii234 (gates G5/G7).", "")
add(G2, "Cell padding",
    "study_config.json:98; prepare_study.py:310-314; dependencies/catalogue_stage.py:51-76",
    "each SWIFT top-level cell box enlarged by 0.1 cell size (3.1 Mpc) when testing sphere-cell overlap",
    "-", "smaller padding (fewer cells read) after checking drift; larger for safety",
    "Particles drift slightly outside their cell between tree rebuilds; padding guarantees completeness of the sphere.",
    "10,247 cells, 2.198e9 candidate rows (metadata_plan.json)",
    "same 0.1 in L2p8 and fullsnap configs.")
add(G2, "Physical vs comoving units",
    "dependencies/flamingo_io.py:50-59; measure_overlap.py:178-181; prepare_study.py:381",
    "coordinates and smoothing lengths in comoving Mpc, converted with a; x = |delta| a / R200c (physical); softening = min(a eps_com, eps_max,phys) = 0.0057 Mpc at z = 0",
    "-", "-",
    "No effect at z = 0 (a = 1); matters for any z > 0 pass (softening and R200c are physical).",
    "metadata_plan.json: scale_factor 1.0, softening_physical_mpc 0.0057")

# ============================================================================ 3 particle selection
add(G3, "Recently AGN-heated gas: definition",
    "dependencies/flamingo_io.py:623-630 (SOAP Parameters/RecentlyHeatedGasFilter); measure_overlap.py:196; metadata_plan.json recent_agn_filter",
    "a_last >= 0.998956 (15 Myr at z = 0) and 8.80e6 <= T <= 1.756e8 K (SOAP's own filter, same as ComptonYWithoutRecentAGNHeating)",
    "none (only the phase flag is stored)",
    "other windows (e.g. 30-50 Myr), other temperature bands, or a cut on MaximalTemperatures",
    "B24's cut; it removes artificially hot, just-heated particles from the pressure.",
    "LastAGNFeedbackScaleFactors is stored with the HalfFloat lossy filter (probe_g4.json); float16 values passing the cut near a = 1 are 0.99902, 0.99951, 1.0, "
    "so the effective window is ~17.5 Myr instead of 15 Myr (estimate from float16 arithmetic, quantify_choices.json recent_agn_filter); SOAP uses the same stored values.")
add(G3, "Recently AGN-heated gas: removed or kept",
    "analysis/variants.py:159-167, 320 (agn='removed' default); F_observables/f0_profiles.py:19-20",
    "removed for pressure AND electrons (phases xray_selected + other)",
    "agn='kept' (all three phases); per-phase sums for any quantity on X200/X500; AGN axis on excision/partition blocks",
    "-",
    "Only unresolved cores of low-mass groups move; keeping it changes observables by <= 0.04% (F), <= 0.26% (D).",
    "kept/removed - 1 on resolved (base) shells, bins 13.5-15.5: pressure <= 0.05%, electrons <= 0.17% (quantify_choices.json)",
    "Electrons: recently heated gas has real free electrons, so removing it from n_e is a convention borrowed from pressure (effect <= 0.17%).")
add(G3, "Star-forming and cold gas",
    "measure_overlap.py:197-198; study_config.json:109-110",
    "no explicit exclusion; star-forming (SFR > 0) and T < 1e5 K gas fall in phase 'other' and are kept",
    "phase 'xray_selected' alone (SFR <= 0, T >= 1e5 K, not AGN) via only_xray (variants.py:159-162)",
    "a cut on T or density (e.g. T > 1e6 K only), or removing cold gas from the electrons",
    "Star-forming gas has y = 0 by construction (SWIFT), so pressure is unaffected; it contributes electrons according to its stored ionisation.",
    "star-forming particles with non-zero y: 0 of 503,296 (apertures.npz diagnostics, quantify_choices.json)")
add(G3, "X-ray-selected phase",
    "study_config.json:102-110; extract_study.py:131; measure_overlap.py:198",
    "SFR <= 0, T >= 1e5 K, not recent AGN; L_X = XrayLuminosities column 2 (ROSAT band)",
    "Q_X, nT_X estimators and T_X diagnostics on this phase",
    "eROSITA bands (columns 0, 1) or a different temperature threshold",
    "Needed only for X-ray-weighted literature recipes (B24 Q_X, A10-like n_e T_X), which are strongly biased for tSZ (D: +80/+42/+23%).", "")
add(G3, "Gas ownership classes (membership)",
    "measure_overlap.py:46-57, 132-155; prepare_study.py:117-165 (groups at :148-151); extract_study.py:176-182",
    "HBT GroupNr_bound per particle -> 17 owner codes: central-bound, 12 overlapper classes (mu class x reach x own/neighbour), other own satellites, other bound (painted / not painted host), unbound",
    "any union of owner codes per halo (variants.selection); radii234 components; per-halo masks (A2-i)",
    "FOF membership (FOFGroupIDs), a different mass cut for 'painted', exclusive membership by a SOAP SO radius",
    "Which gas a painted profile should contain; gas bound to other painted haloes would be painted twice.",
    "y shares of the stacked profile (AGN removed), bins 13.5-15.5: inside 0.1-1R central-bound 84-96%, own satellites 0.1-4.2%, unbound 4-12%; "
    "1-2R central 30-38%, unbound 59-64%; 2-4R unbound 72-89%, own satellites 3.8-14.4%, other-bound 5.0-13.2% (painted hosts 4.8-12.9%) (quantify_choices.json)")
add(G3, "Gas selection used for the stacks",
    "analysis/variants.py:60-61, 92-124, 321 (default 'all'); F_observables/f0_profiles.py:19-20",
    "'all' (every particle in the sphere, satellites kept) = 'main'; painting target A2-ii",
    "no_other, no_satellites_or_other, host_bound, central_bound, remove_overlappers(mu, reach, flavour), A2-i, A2-ii, A2-ii and not other-painted",
    "-",
    "Calibrating on 'central_bound' would drop 62-70% of the 1-2R y (central-bound share 30-38%); 'all' paints the 2-4R neighbour gas twice; A2-ii fixes the double counting (2.4-7.4% of Y(<4R), SYNTHESIS 1).",
    "", "fullsnap default 'exclude_other_haloes' (../fullsnap_calibration/config.json:30) = our 'no_other': also removes gas bound to unpainted small haloes; "
        "L2p8 has no membership at all (all gas inside R100c).")
add(G3, "A2-i rule (bound to another painted halo)",
    "analysis/variants.py:110-116, 198-220 (bound-gas majority 0.5 at :218)",
    "remove owner 14 and every neighbour-overlapper owner code whose overlappers are painted (bound-gas-weighted majority)",
    "owner-14-only variant (include_neighbour_overlappers=False)", "-",
    "Removes gas bound to other painted haloes; leaves their unbound atmospheres.", "")
add(G3, "A2-ii rule (scaled-distance partition)",
    "measure_overlap.py:217-226; census.py:9-11 (neighbour list)",
    "a particle belongs to a painted neighbour (M200c >= 6.6e12) if min_n |r - d_n|/R200c,n < r/R200c,c",
    "partition_owned, partition_owned_not_other_painted (X200 grid, reduced quantities only)",
    "a different distance metric (e.g. in R500c or physical units) or a softer (weighted) split",
    "Painting-consistent ownership: equals ALL to <= 1.4% inside 2R and removes the twice-painted 2-4R gas.",
    "SYNTHESIS 1: removes 2.4-7.4% of Y(<4R), 5-7.6% of N_e(<4R)")
add(G3, "Excision around overlappers",
    "analysis/variants.py:33-36, 246-262, 352-358, 401-402; measure_overlap.py:60-61, 211-216; study_config.json:71-87",
    "not used for the painting calibration",
    "variants all / neighbours_only / placebo_iso / placebo_nbr; kappa 0.5 / 1 / 2; members all / no_overlapper_bound / central_bound; min_retained 0.5",
    "other kappa values or excision of mu < 0.1 structures",
    "Overshoots on real haloes by 6-16% in Y(<1R) because it also cuts host gas (SYNTHESIS 2.2).", "")
add(G3, "Template subtraction of overlappers",
    "analysis/variants.py:38-46, 529-567",
    "not used for the painting calibration",
    "model or measured amplitude, truncation 1 R_j, kinds all / neighbour centrals / neighbours / own satellites",
    "-",
    "Leaves the overlappers' unbound atmospheres (+3.5-5% residual, SYNTHESIS 2.2).", "")

# ============================================================================ 4 quantities & estimators
add(G4, "Pressure source",
    "measure_overlap.py:75, 190-193; dependencies/flamingo_io.py:55",
    "per-particle ComptonYParameters (SWIFT, physical Mpc^2); P_e = y/V m_e c^2/sigma_T",
    "n_e k T (electron_kT, T_e = T); SPH gas pressure (Pressures x V, total gas pressure)",
    "a two-temperature or non-equilibrium electron temperature",
    "y is what the tSZ map measures; SOAP's ComptonY uses the same per-particle values (aperture sums agree to ~1e-5, D).", "")
add(G4, "Electron count",
    "measure_overlap.py:189-191, 228-231; pilot_physics.py:2-5",
    "ElectronNumberDensities x (m/rho) (SWIFT's stored ionisation state)",
    "fully ionised H + He: m/m_p (X + Y/2) (aux200/aux500); gas mass / mu_XGP (the B16 convention)",
    "metal electrons (not in the fully ionised estimate)",
    "DM and kSZ count free electrons; using gas density with XGPaint's mu_XGP gives ~9-10% fewer electrons (D).",
    "n_e/fully-ionised on resolved shells (13.5-15.5): 97.0-99.7% inside 0.1R, 99.2-100.2% at 0.1-1R, 99.8-100.0% at 1-2R, 99.2-99.8% at 2-4R (quantify_choices.json)")
add(G4, "Pressure estimator (shell statistic)",
    "analysis/variants.py:57-58, 320, 361-386",
    "geometric: sum y / V_shell (volume average; tSZ/DM-correct)",
    "Q_V (sum V Pe / sum V), Q_V_y, Q_M (mass-weighted), Q_X (L_X-weighted), nT_X",
    "-",
    "Only the geometric estimator integrates to the observed Y; weighted particle averages are clumping-biased.",
    "B: Q_V +3-5% inside 1R, +18-25% at 3-4R; Q_M x1.6-2.6 at 1-2R; Q_X +17-36% at 0.1-0.5R (SYNTHESIS 2.3)")
add(G4, "Volume estimator",
    "analysis/variants.py:266-269 (geometric) vs 366-373, 393-396 (SPH volume m/rho)",
    "exact geometric shell volume 4pi/3 (r_hi^3 - r_lo^3)",
    "sum of SPH volumes m/rho of the particles in the shell",
    "kernel-weighted volume deposition (SPH smoothing of each particle over shells)",
    "SPH volumes do not tile space; dividing by them inflates densities and pressures where the filling fraction < 1.",
    "sum(m/rho)/V_shell on resolved shells (13.5-15.5): 94.9-98.5% inside 0.1R, 95.2-97.7% at 0.1-1R, 90.9-95.4% at 1-2R, 76.0-90.7% at 2-4R (quantify_choices.json)")
add(G4, "Particle deposition (no kernel smoothing)",
    "measure_overlap.py:183-184, 206-209 (each particle's full value in the shell of its centre)",
    "nearest-shell (point) deposition",
    "-", "SPH-kernel deposition over shells (a particle spread over r +- h)",
    "Point deposition is unbiased for integrals but noisy and slightly sharpened where shells are thinner than h; the 2h resolution cut removes those shells.", "")
add(G4, "Electron and gas density estimators",
    "analysis/variants.py:57-58, 387-400",
    "electrons: geometric sum n_e V / V_shell; gas: geometric",
    "electrons Q_V; gas Q_V, Q_M (sum m rho / sum m)",
    "-",
    "Mass-weighted densities are clumping-biased (x7-10 at 0.1-0.5R, x300-390 at 1-2R, SYNTHESIS 2.3).", "")
add(G4, "Temperature weightings (diagnostics)",
    "analysis/variants.py:432-451",
    "not used in the stacks",
    "T_M (mass), T_V (volume), T_X (L_X); clumping factors <rho^2>/<rho>^2 and <Pe^2>/<Pe>^2",
    "spectroscopic-like temperature, T_e != T",
    "Needed for X-ray-route recipes and for the outer-temperature terms of Test G.", "")
add(G4, "Order of operations",
    "analysis/variants.py:7-8, 333 (linear sums rebinned before any ratio)",
    "per halo: rebin linear sums, then divide by the geometric shell volume and the halo's normalisation",
    "ratio-of-sums stacks (sum over haloes of y / sum of V P200) can be formed from the same sums",
    "-",
    "Averaging ratios on fine shells and then rebinning would weight thin shells wrongly; the current order is exact.", "")

# ============================================================================ 5 radial grids
add(G5, "X200 storage grid",
    "grids.py:14 (edges); measurements_owner.npz x200",
    "67 shells over 0-4 R200c: union of the radii234 36 edges, Lee22's 27 log shells (0.04-1.34) and {1.5, 1.75, 2.5}",
    "any exact sub-grid (np.add.reduceat), e.g. old 36, Lee22 27, F 35, custom (newdata.py:150-168)",
    "edges that are not in the 68 stored edges (e.g. a finer core grid, 0.03 R200c) or beyond 4 R200c",
    "Every fit / cut radius must be a stored edge; sub-grids are exact only for linear sums.",
    "thinnest storage shells are 0.24% wide (grids.py docstring); analyses use sub-grids")
add(G5, "X500 storage grid",
    "grids.py:15",
    "40 shells over 0-6 R500c (B24's 30 log shells 0.01-3 plus 0.03, 0.15, 1, 3.5-6)",
    "B24 30-shell grid, any exact sub-grid; normalisations B24 / A10 / self-similar-500",
    "edges outside the 41 stored edges",
    "R500c-scaled literature comparisons (A10, B24).", "")
add(G5, "F grid (fit grid of Tests B, F, G)",
    "F_observables/f_common.py:86-103",
    "35 shells to 4 R200c: radii234 log edges (0.069 dex) inside 0.84 R200c plus 1, 1.15, 1.34, 1.5, 1.75, 2, 2.18, 2.5, 3, 3.30, 3.63, 4",
    "any other sub-grid of X200", "-",
    "Every requested R_fit / R_cut (1, 1.34, 1.5, 1.75, 2, 2.5, 3, 4) is an exact edge; no slivers.", "")
add(G5, "Lee22 grid", "grids.py:18", "27 log shells over 0.04-1.34 R200c (exact sub-grid)", "yes (stored)", "-",
    "Reproduces Lee22's covariance figure; on it the sampling covariance is near-singular (adjacent rho 0.90-0.94, condition 220-1,500, SYNTHESIS 2.3).", "")
add(G5, "radii234 grid", "pilot_physics.py:24", "36 shells: 0, 31 log edges 0.025-3, 1, 2, 3-4 (4 log)", "yes (bitwise subset)", "-",
    "Legacy parity (measurements_legacy.npz is bit-identical to radii234).", "")
add(G5, "Model extension shells", "F_observables/f_common.py:104-107", "7 shells 4-12 R200c for models only (the truth stops at 4R)",
    "-", "measured profiles beyond 4 R200c", "Lets painted models extend past the calibration range (e.g. outer power law / 2-halo terms).", "")
add(G5, "Smoothing-length histogram binning", "measure_overlap.py:73, 233-240",
    "h a / R200c (R500c) in 384 log bins over 1e-4-1e2 (0.0156 dex) per shell, all gas",
    "upper median, any quantile from the histogram; rebinned exactly to any sub-grid",
    "per owner or per phase h distributions",
    "Feeds the 2h resolution criterion; the upper-median bin edge is conservative by <= 0.0156 dex.",
    "hs200/hs500 under/overflow: 0 (apertures.npz diagnostics)",
    "L2p8: 70 bins over 10^-3-10^0.5 (0.05 dex), lower median at the bin centre (../l2p8_reduced_calibration/config.json:9, fit_stage.py:39-46).")

# ============================================================================ 6 normalisations
add(G6, "Pressure normalisation P200",
    "pilot_physics.py:30-34; analysis/newdata.py:199-201; analysis/variants.py:276-281",
    "P200 = G M200c (200 rho_c) f_b / (2 R200c) with the halo's own M200c, R200c and f_b = 0.158823",
    "B24 P_e,500 = n_e,500 k T500 (f_B 0.159, mu 0.59, mu_e 1.14); A10 P500; self-similar-500 (newdata.py:203-230)",
    "-",
    "XGPaint's Battaglia profiles are written in P/P200 vs r/R200c; any other normalisation must be converted before painting.",
    "rho_c from halo M/R vs cosmology: 1.08e-5 relative (census_summary.json g3)")
add(G6, "Density normalisation",
    "pilot_physics.py:17-19; analysis/variants.py:306-316",
    "electrons: n_e mu_XGP / (rho_c f_b), mu_XGP = (m_e + 2 m_p/(1+0.76))/0.9 = 1.2632 m_p; gas: rho / (rho_c f_b)",
    "B24 n_e,500 for X500 grids; any constant can be re-applied (per-halo linear sums are stored)",
    "-",
    "mu_XGP is the XGPaint tau-profile convention; electrons calibrated in these units paint the right n_e, gas density does not (-9 to -10% N_e, D).", "")
add(G6, "Baryon fraction", "analysis/newdata.py:103 (metadata_plan.json baryon_fraction)",
    "f_b = Omega_b / Omega_m,total = 0.0486/0.306 = 0.158823", "B24 fixed 0.159 (0.11% different)", "-",
    "Only a constant factor in P200 and the density units; must match what the painter assumes.", "")
add(G6, "Scaling variable", "analysis/variants.py:266-269 (x in units of the grid radius)",
    "x = r/R200c (X200) or r/R500c (X500)", "R500c scaling on X500", "r/R200m, physical radius, r/R_sp",
    "Self-similarity and the HalfDome catalogue convention (M200c).", "")

# ============================================================================ 7 resolution criteria
add(G7, "Minimum inner edge",
    "analysis/masks.py:18, 42", "x_inner >= 0.04", "any value (masks.resolution_masks(min_edge=...))", "-",
    "Excludes the unresolved core even in the most massive haloes; binding only in 15.1-15.5.",
    "min_edge 0.01 changes the first resolved edge only in 15.1-15.3 (0.040 -> 0.034) and 15.3-15.5 (0.040 -> 0.025) (quantify_choices.json)")
add(G7, "Softening criterion",
    "analysis/masks.py:18, 42; prepare_study.py:381", "x_inner >= 4 eps / R200c, eps = 0.0057 Mpc physical",
    "any factor", "-", "Gravitational softening scale; never the binding criterion here.",
    "softening factor 2.8 instead of 4: no change in any bin (quantify_choices.json)",
    "same 4 eps in L2p8 (config.json:12) and fullsnap (config.json:47), but they use the comoving 0.0223 Mpc capped at 0.0057 Mpc physical.")
add(G7, "Minimum particles per halo-shell",
    "analysis/masks.py:18, 43", "count >= 20 gas particles (all owners and phases) in every halo of the bin",
    "any value", "-",
    "Shot noise of a single halo-shell; at 20 it never binds because the 2h criterion is stricter.",
    "first resolved inner edge with 20 vs 32: identical in every bin; with 100: one to two shells further out "
    "(13.5-13.7: 0.321 -> 0.519; 14.5-14.7: 0.090 -> 0.123) (quantify_choices.json)",
    "L2p8 and fullsnap use 32 (../l2p8_reduced_calibration/config.json:10; ../fullsnap_calibration/config.json:48); "
    "fullsnap counts the POOLED particles over all haloes of the stack, not per halo (../fullsnap_calibration/profiles.py:611, 621).")
add(G7, "Smoothing-length criterion",
    "analysis/masks.py:18, 44-45; analysis/io.py:240-248", "x_inner >= 2 x upper-median (h a / R200c) of the halo-shell",
    "any factor (1 = kernel-smoothed core, 3 = conservative); any quantile from hs200/hs500", "-",
    "SPH smooths structure below ~h; this is the binding criterion in every supported bin.",
    "first resolved inner edge (current, F grid): 0.321, 0.274, 0.170, 0.145, 0.105, 0.090, 0.065, 0.056, 0.040, 0.040 for 13.5-13.7 ... 15.3-15.5; "
    "factor 1 moves it in by 1-2 shells and makes 13.1-13.3 supported; factor 3 moves it out by 1-2 shells and makes 13.3-13.5 unsupported (quantify_choices.json)",
    "L2p8: median (not upper median) at the bin centre, per halo; fullsnap: pooled median per 0.05-dex fine bin, then 'outward of the last failing shell'.")
add(G7, "Strict / diagnostic / base masks and support",
    "analysis/masks.py:46-53", "strict = every halo of the bin resolved; supported = >= 6 strict shells with outer edge <= 2; base = strict if supported else diagnostic (geometric for every halo, >= 200 particles in total, particles in >= 50% of haloes)",
    "any quantile rule (resolved_fraction is returned), per-halo masks, other support thresholds", "-",
    "A common shell set keeps the same haloes in every shell of a stack (no selection bias); 'every halo' becomes stricter as N grows.",
    "Unsupported bins 12.5-13.3 fall back to the diagnostic mask, whose first shell (0.105-0.233) lies inside the supported 13.3-13.5 bin's (0.519); there the per-shell median is up to 98% below the mean in 0.1-1R (many empty halo-shells) (quantify_choices.json). Not used for painting (F/G use the published fallback below 13.5).")
add(G7, "Per-halo vs common shell masks",
    "analysis/masks.py:43-46 (per-halo 'resolved' is computed but stacks use the common mask); F_observables/f_fit.py:108",
    "common (strict) mask per bin; stack over all haloes on those shells",
    "per-halo masks (mean over the resolved haloes of each shell)", "-",
    "A per-halo mask changes the halo set from shell to shell; better-resolved (denser) haloes dominate the partially resolved shells.",
    "mean over resolved haloes only / mean over all haloes in partially resolved shells (0.04-0.32 R200c), bins 13.5-15.1: +1.6% to +104% (quantify_choices.json)",
    "L2p8 uses per-halo masks (../l2p8_reduced_calibration/fit_stage.py:49-56); flamingo_stacks/fullsnap use per-halo 'complete' masks at the outer edge (R100c) and pooled resolution masks.")
add(G7, "Fit-shell mask", "analysis/masks.py:56-62; analysis/fitting.py:33", "base & outer edge <= cut & mean > 0; >= 6 shells (MIN_SHELLS)",
    "any cut, any minimum", "-", "Zero-pressure shells are excluded, never log-filled.", "",
    "fullsnap min_fit_shells 3 (../fullsnap_calibration/config.json:36).")

# ============================================================================ 8 stacking statistic & weights
add(G8, "Stack statistic",
    "analysis/stacking.py:33-44 (weighted_stack); F_observables/f0_profiles.py:25-28; F_observables/f_fit.py:107",
    "inclusion-weighted arithmetic mean of per-halo normalised profiles (mean of ratios)",
    "median, geometric (log-) mean (stacking.py:29-30, 47-54), weighted median (D_literature/td_recipes.py:214), Y200-weighted mean, ratio of summed sums",
    "-",
    "The painted map is a sum over haloes, so the mean (not the median) is what C_ell and mean DM need.",
    "median/mean - 1 on resolved shells, 13.5-15.5: pressure -2.7..-56% at 2-4R and -24..+6% at 1-2R; electrons -18..+2% at 2-4R; geometric mean -3.8..-49% at 2-4R (quantify_choices.json). Y200 weighting: +1.6..+3.4% on yy (D).")
add(G8, "Scatter statistic and N_eff",
    "analysis/stacking.py:33-44", "reliability-weighted SD (1 - sum w^2 correction), N_eff = 1/sum w_norm^2",
    "unweighted SD, percentiles (16/84) per shell, log-covariance of per-halo profiles (covariance.py:59-73)", "-",
    "Sets the diagonal fit errors (objective (a)) and the painted scatter model.",
    "N_eff of the inclusion weights: 299.6-325.3 per bin below 14.5 (quantify_choices.json)")
add(G8, "Bootstrap of the stack",
    "analysis/stacking.py:14-15, 66-79; F_observables/f_common.py:113-114; analysis/covariance.py:88-109",
    "halo bootstrap, 1,000 draws (F), seed 20261012 (+ per bin); 50 draws for the shrinkage intensity",
    "stratified (within census class) bootstrap, delta-method covariance, spatial jackknife (4^3 blocks)",
    "-",
    "Spatial jackknife errors are 1.16-1.26x the halo bootstrap for pressure parameters and 1.11x per shell at 2-4R (SYNTHESIS 2.3): bootstrap intervals beyond 2R are lower limits.", "",
    "radii234: 128 draws (fit_study.py:28); L2p8: 200 linearised replicates; fullsnap: 200 (stack) / 100 (fit).")
add(G8, "Covariance shrinkage and pooling",
    "analysis/covariance.py:173-197, 279-338; analysis/pipeline.py:57, 76; F_observables/f_fit.py:126-139, 142-166",
    "Schaefer-Strimmer-type shrinkage towards the pooled correlation of neighbouring bins, intensity from 50 resamples; lam = 1 if N < 3p; eigenvalue floor 1e-10",
    "no shrinkage, factor-analysis covariance, all-bin pooling (only acceptable for pressure), Higham projection", "-",
    "The ln-mean covariance is near-singular on fine grids (condition 220-1,500 on Lee22's grid); only GLS fits use it.", "")
add(G8, "CV folds (spatial)", "F_observables/f0_profiles.py:22, 114-123", "4^3 blocks of 250 Mpc, 8 folds of 8 blocks (seed 20261012)",
    "other block sizes / fold counts", "-", "Out-of-fold validation of the fitted profiles.", "")

# ============================================================================ 9 mass binning
add(G9, "Bin width and edges",
    "study_config.json:9-26; F_observables/f_common.py:146-154",
    "0.2 dex in log10 M200c [Msun], 12.5-15.5; calibrated bins 13.5-15.5 (10 bins) with log M <= 15.388",
    "any edges (per-halo profiles + per-halo weights exist): merged 0.4-dex bins, 0.1-dex bins (~150 haloes below 14.5), Lee22 h^-1 Msun bins (analysis/io.py:307-318), "
    "B24 0.5-dex M500c bins (D_literature/td_recipes.py:141-174)",
    "-",
    "Narrow bins track the mass trend of the shape; wide bins reduce noise. Top bins are small (127, 21, 6 haloes).",
    "N per bin: 308-332 below 14.5; 895 / 326 / 127 / 21 / 6 above (metadata_plan.json)",
    "L2p8: 0.25-dex bins centred on multiples of 0.25 (np.round, ../l2p8_reduced_calibration/common.py:178-180); fullsnap/flamingo_stacks: 0.2 dex with 0.02-dex mass nodes for the model.")
add(G9, "Pooling of the top bins",
    "B_covariance/bcommon.py:12-13, 103-104",
    "Test B: >= 14.9 pooled (154 haloes) with a per-shell linear log M trend; Test F/G: 15.1-15.3 (N = 21) and 15.3-15.5 (N = 6) fitted individually",
    "either", "-",
    "The top bins dominate C_ell^yy; the two conventions differ by 0.5-1.3% in C_ell^yy at 2R (SYNTHESIS 2.3 'open difference').", "")
add(G9, "Mass assigned to a stack",
    "F_observables/f_fit.py:109 (bin centre); kernels/engine.py:332-345 (shape constant within a bin)",
    "bin centre for mass-trend fits; the painted shape is constant within each 0.2-dex bin",
    "population-weighted mean log M, median, or the model averaged over the bin's haloes (as L2p8/fullsnap do)", "-",
    "Only matters for mass-trend models (B12 power laws, Lee22 ladder).",
    "population-weighted mean log10 M lies 0.004-0.026 dex below the centre (0.047 in 15.3-15.5) (quantify_choices.json)")

# ============================================================================ 10 fitting & truncation (after stacking)
add(G10, "Profile family",
    "../radii234_contamination_L1_m9_0077_20261008/models.py:9-17; analysis/families.py; F_observables/f_fit.py:14-20; G_physical_outer/g_models.py",
    "B12 (pressure; gamma = -0.3, alpha = 1) and B16 (electrons; xc = 0.5, gamma = -0.2); recommended: + physical outer terms (G)",
    "gnfw5, b12env/b16env, splines, Lee22 broken power laws, physical outer models", "-",
    "B12/B16 describe the stack only to 2R (sigma_d 12% at 4R, SYNTHESIS 2.4).", "")
add(G10, "Fit radius and cut radius",
    "F_observables/f_common.py:109-111; G_physical_outer/g_common.py:83-84",
    "R_fit = R_cut = 4 R200c recommended (Pareto-optimal for tSZ, kSZ, DM, tSZ x DM)",
    "1, 1.34, 1.5, 1.75, 2, 2.5, 3, 4 R200c (every one an exact F edge)", "cuts beyond 4 R200c",
    "A 2R cut costs tSZ -5%, kSZ -26%, DM -22% (SYNTHESIS 1).",
    "stacked integrals (resolved shells): Y(<2R)/Y(<4R) = 70-90%, N_e(<2R)/N_e(<4R) = 54-69% in 13.5-15.5 (quantify_choices.json)")
add(G10, "Truncation shape", "F_observables/f_fit.py:204-224", "hard cut", "0.05-dex logistic (centred or flux-matched)", "-",
    "<= 0.21% on C_ell at 4R (Test C).", "")
add(G10, "Objective and error floor",
    "analysis/fitting.py:193-205 (floor 0.05), 33 (MIN_SHELLS 6), 95-97 (acceptance); analysis/pipeline.py:1-30",
    "(a) diagonal sigma = max(SD/mean, 5%) on ln(mean); importance-diagonal (d) for two-component forms",
    "(b) bootstrap SD, (c) GLS + sigma_d, (d)-(g) importance / metric / hybrid / extra residuals", "-",
    "GLS beyond 2R becomes an unweighted fit of the tail (P0 x1.45 at 4R) and is the worst for observables (SYNTHESIS 1).", "",
    "fullsnap / flamingo_stacks: sigma_log = hypot(err/mean, 0.03) (../fullsnap_calibration/profiles.py:793; ../flamingo_stacks.py:292); "
    "L2p8: log of the arithmetic mean vs the same-halo model mean, groups weighted equally per (z, mass bin), no per-shell errors (../l2p8_reduced_calibration/fit_stage.py:108-111, 137).")

# ============================================================================ 11 performance knobs
add(G11, "Fields read and bytes per particle",
    "prepare_study.py:35-36, 326-357; dependencies/flamingo_io.py:50-59; study_config.json:99-100",
    "13 datasets + membership = 140 B/particle logical (Coordinates 24, ElementMassFractions 36, XrayLuminosities 24, ElectronNumberDensities 8, ParticleIDs 8, 8 x 4 B, membership 8)",
    "-",
    "a lean pass for P + n_e + AGN flag + resolution + ownership needs Coordinates, SmoothingLengths, Masses, Densities, ComptonY, ElectronNumberDensities, Temperatures, LastAGN, membership = 64 B logical (46%)",
    "The pass is decompression-bound; every HDF5 chunk of a read dataset is decompressed whole (1,048,576 rows per chunk; 2-D datasets carry all columns).",
    "metadata_plan.json field_schema; probe_g4.json chunk shapes; 307.8 GB logical for 2.198e9 rows")
add(G11, "Workers and politeness",
    "study_config.json:112, 119-135; extract_study.py:329-333, 499-500",
    "6 workers (load-adaptive: <= 0.75 x cores - load, >= 2), nice 19, ionice best-effort 7",
    "-", "more workers if the shared host is idle",
    "Shared host (maas) without a scheduler; wall time scales ~1/workers until I/O saturates.",
    "complete.json: 66.4 min wall, 5.47 CPU-h, peak tree RSS 8.8 GB, load 24.5 at start")
add(G11, "HDF5 chunk cache and file reopening",
    "study_config.json:116-117, 137-138; extract_study.py:48-50, 166-177",
    "40 MiB per dataset (holds one chunk of the largest dataset, 36 MiB), reopen files every 16 cells",
    "-", "larger cache (80 MiB raised the open-file footprint, study_config.json chunk_cache_note)",
    "Selected cells are read in file order; consecutive cells share chunks only if the cache holds them.", "")
add(G11, "Memory guards", "study_config.json:113-115, 139-143; extract_study.py:76-78, 336-369, 503-504",
    "RLIMIT_AS 8 GiB per worker, tree RSS stop 48 GiB, system free-memory floor 12%, 300-min wall stop, fixed malloc mmap threshold",
    "-", "-", "Protects the shared host; G6 measured 1.52 GiB RSS per worker (gate_file.json).", "")
add(G11, "Read budget", "study_config.json:101; prepare_study.py:357-361", "refuse plans above 488 GB logical", "-", "-",
    "Caps a single pass (actual 307.8 GB).", "")
add(G11, "Batch size", "extract_study.py:51", "1,000,000 particles per accumulate call", "-", "-",
    "Bounds temporaries in cluster cells; no effect on results (linear sums).", "")
add(G11, "Read path", "extract_study.py:157 (read_from='chunks')", "chunk files directly (no virtual top-level file)",
    "-", "-", "Avoids the virtual-dataset layer; G4 checked chunk and top-level values are byte-identical on L1_m9 0077.", "")

out = {"description": "Catalogue of user-facing choices that affect the 1D stacked profiles (overlap_covariance_L1_m9_0077_20261012)",
       "path_convention": "relative to the study root unless prefixed '../' (= flamingo_pressure_calibration/)",
       "fields": ["group", "name", "where", "current", "alternatives_local", "alternatives_new_pass", "why", "evidence",
                  "pipelines_differ"],
       "n_choices": len(C), "choices": C}
(HERE / "choices_catalogue.json").write_text(json.dumps(out, indent=1) + "\n")
# plain list as requested (same objects)
(HERE / "choices_catalogue_list.json").write_text(json.dumps(C, indent=1) + "\n")
print(len(C), "choices;", sorted({c["group"] for c in C}))
