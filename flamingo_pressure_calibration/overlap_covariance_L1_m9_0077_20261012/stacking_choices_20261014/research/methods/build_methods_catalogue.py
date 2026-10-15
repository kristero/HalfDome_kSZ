"""Write methods_catalogue.json (list of ways to obtain 1D stacked gas profiles) and literature_table.json.

Every number below comes from (i) methods_numbers.json (this directory, script methods_numbers.py),
(ii) the study reports (SYNTHESIS.md, A/B/D/F/G REPORT.md), (iii) the read-only SOAP listing
soap_0077_group_listing.json, or (iv) the cited papers (checked on arXiv on 2026-10-14). Estimates are
labelled as such.
"""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent

M = []


def add(category, name, measures, inputs, cost, pros, cons, literature, relevance, numbers=None, status=None):
    M.append(dict(category=category, name=name, measures=measures, inputs=inputs, cost=cost, pros=pros,
                  cons=cons, literature=literature, relevance_for_painting=relevance,
                  key_numbers=numbers or [], status_in_this_project=status or ""))


# ------------------------------------------------------------------ (a) per-halo 3D estimators
add("a_estimator", "Geometric shell estimator, particle-centre binning",
    "Volume average of a linear quantity in a spherical shell: <P_e> = (m_e c^2/sigma_T) sum_i y_i / V_shell, "
    "<n_e> = sum_i n_e,i V_i / V_shell, <rho_gas> = sum_i m_i / V_shell. Exactly the quantity whose line-of-sight "
    "integral gives Compton-y, tau/kSZ and DM.",
    ["particle Coordinates, ComptonYParameters (or ElectronNumberDensities x m/rho, or Masses)",
     "halo centre and R_Delta (SOAP)", "radial grid edges in r/R_Delta (x_min, x_max, number, spacing)",
     "gas selection (phases, ownership)", "normalisation (P200, rho_c f_b, ...)"],
    "One pass over all gas inside max(4 R200c, 6 R500c): this project's z = 0 pass for 4,588 centrals took 66 min wall, "
    "6 workers, 5.5 CPU-h, 308 GB logical read, 2.2e9 rows (42% of the 5.27e9 gas particles); decompression-bound.",
    ["linear sums: merge exactly over files/haloes and rebin exactly to any coarser grid",
     "conserves Y and N_e (particle Y(<R500c) equals SOAP to a median 2e-5)",
     "the tSZ/kSZ/DM-correct quantity; no SPH volume deficit"],
    ["shot noise where a shell holds few particles -> needs a resolution mask",
     "ignores the SPH kernel extent (matters only where H >~ r, i.e. in shells the mask already rejects)"],
    ["this project (radii234, Tests A-G)", "Lee et al. 2022 (arXiv:2205.01710): A = V(bin)^-1 sum V_i A_i on TNG cells"],
    "Primary route: what XGPaint paints is exactly this volume average.",
    ["SOAP positive control: Y(<R500c) median offset +2.0e-5, 98.5% within 0.3% (D)"],
    "used (reference estimator)")

add("a_estimator", "SPH-volume-weighted estimator Q_V = sum q_i V_i / sum V_i, V_i = m_i/rho_i",
    "Volume-weighted mean of particle values, normalised by the summed SPH volumes instead of the geometric volume.",
    ["as geometric + Densities", "particle quantity q (P_e = n_e k T)"],
    "Same pass; one extra float field.",
    ["standard in the FLAMINGO X-ray paper, so directly comparable to B24",
     "less sensitive to the number of particles in a shell"],
    ["summed SPH volumes fill only part of the shell -> biased high for tSZ/DM",
     "Q_V/geometric: +3-5% (<1R200c), +6-7% (1-2R), +13-14% (2-3R), +18-25% (3-4R) (B)",
     "through the fits: C_ell^yy +6 to +8%, mean DM +4.3% (B)"],
    ["Braspenning et al. 2024 (arXiv:2312.08277), eq. 8"],
    "Not for painting; only for like-for-like comparison with B24-style X-ray profiles.",
    ["aggregate filling sum(m/rho)/V_shell for log M200c >= 13.9: 0.94-0.97 at 0.09-1 R200c, 0.85-0.86 at 2.5-3 R200c, 0.77-0.79 at 3.6-4 R200c (methods_numbers B)"],
    "available (variants Q_V)")

add("a_estimator", "Mass-weighted estimator Q_M = sum q_i m_i / sum m_i",
    "Mass-weighted mean (the natural temperature definition; SOAP GasTemperature is mass-weighted).",
    ["Masses, the particle quantity (T, P_e, rho)"], "Same pass.",
    ["right estimator for mass-weighted temperature T_mw profiles", "robust to low particle numbers"],
    ["dense clumps dominate: pressure +13-38% at 0.5-1R, x1.6-2.6 at 1-2R, x7-20 at 2-3R; mass-weighted density x7-10 at 0.1-0.5R, x300-390 at 1-2R (B)",
     "B24 Q_M recipe painted on HalfDome: yy 1h +29/+16/+8% at ell = 100/1000/3000 (D)"],
    ["Braspenning et al. 2024 (arXiv:2312.08277), eq. 10", "Arnaud et al. 2010 (arXiv:0910.1234) simulated profiles used mass-weighted T"],
    "Never for P or n_e painting; acceptable for T_mw (outer temperature terms, Test G).",
    status="available (variants Q_M, diagnostic T_M)")

add("a_estimator", "Emission (X-ray) weighting Q_X and spectroscopic-like temperature",
    "L_X-weighted particle averages (Q_X = sum q_i L_X,i / sum L_X,i, ROSAT 0.5-2 keV) or T_sl with weights "
    "w_i = rho_i m_i T_i^(-3/4) (Mazzotta et al. 2004); pressure as n_e x T_X.",
    ["XrayLuminosities (ROSAT column), Temperatures, Densities; X-ray phase cut (no SF gas, T > 1e5 K, no recent AGN)"],
    "Same pass; +24 B/particle for XrayLuminosities (float64[3]).",
    ["what X-ray observations measure; needed to compare with REXCESS/A10-type data"],
    ["Q_X pressure +17-36% at 0.1-0.5R, x1.3-2.4 at 0.5-1R, x3-10 at 1-2R (B)",
     "particle L_X-weighted T_X is NOT spectroscopic-like: +8.8% median at R500c vs SOAP T_sl (+20-30% in groups) (D)",
     "B24 Q_X recipe painted on HalfDome: yy +80/+42/+23% (D)",
     "n_e kT_X route: within 1% inside 0.5R but +10-42% at 2-3R (B)"],
    ["Mazzotta et al. 2004 (arXiv:astro-ph/0404425)", "Braspenning et al. 2024 (arXiv:2312.08277), eq. 9 (fiducial)"],
    "Not for painting.", status="available (Q_X, nT_X, T_X)")

add("a_estimator", "Synthetic X-ray spectroscopy and hydrostatic scaling",
    "Per-particle APEC spectra, single-temperature fits in 3D radial bins, R500 from hydrostatic modelling; profiles of n_e, T, P, K.",
    ["particle T, rho, metal abundances, a spectral code, instrument response"],
    "Much more expensive than shell sums (spectral fits per bin per halo).",
    ["like-for-like with X-ray samples (includes spectroscopic and hydrostatic biases)"],
    ["measures biased X-ray quantities, not the true volume average; R500,hse != true R500"],
    ["Le Brun et al. 2014 (arXiv:1312.5462)", "McCarthy et al. 2014 (arXiv:1312.5341)", "McCarthy et al. 2017 (arXiv:1603.02702)"],
    "Not for painting.", status="not done")

add("a_estimator", "SPH-kernel deposition onto shells or a 3D grid",
    "Each particle's quantity is spread with its kernel (Wendland C2 in SWIFT, support H = 1.936492 h) over the shells "
    "or voxels it overlaps, then divided by the geometric volume.",
    ["SmoothingLengths, kernel definition, positions; padding of >= H beyond the extraction radius"],
    "Analytic sphere-shell overlap: the 67 x 96 x 68 transfer matrix built here in 1.6 s; deposition onto 3D grids costs ~N_particles x (H/cell)^3.",
    ["correct where the kernel is larger than the shell (cores of groups)", "smooth, conserves totals",
     "the convention used to make SPH maps (McCarthy et al. 2014; Osato & Nagai 2023 particle pasting)"],
    ["in shells the resolution mask accepts it changes stacks by <= 1.2% for log M200c >= 13.5 (<= 2.4% pressure, <= 4.2% electrons at 12.9-13.1)",
     "in rejected cores it moves shells by -78% to +78%: those radii are not measured by either method",
     "edge effect: the last shell before r_ext loses 5-28% unless particles beyond r_ext are also deposited"],
    ["McCarthy et al. 2014 (arXiv:1312.5341)", "Osato & Nagai 2023 (arXiv:2201.02632)"],
    "Not needed for 1D calibration at resolved radii; needed for maps and for any statement about cores.",
    ["estimate from the stored per-shell smoothing-length histograms (methods_numbers C)"],
    "estimated only (no particle re-read)")

# ------------------------------------------------------------------ stacking statistics
add("a_statistic", "Mean of per-halo normalised profiles (equal or inclusion weights)",
    "<f>_bin = sum_h w_h f_h(x) / sum_h w_h, f = P/P200 or n_e/n_200 on a scaled grid.",
    ["per-halo linear sums", "per-halo scales", "weights (1/inclusion probability for stratified samples)"],
    "Negligible after the pass (all statistics here: 14 s for 4,588 haloes incl. 300 bootstraps).",
    ["painting the mean profile reproduces the mean of linear observables (mean y, mean DM)",
     "no zero-occupancy problem", "simple bootstrap errors"],
    ["outer shells dominated by rare neighbour-rich haloes: bootstrap half-width 5-9% at 2.5-3R, 9-19% at 3.6-4R (pressure, ~320 haloes per 0.2 dex)",
     "carries no scatter: one mean profile per bin leaves 1-halo tSZ 1.5-7.3% low (F)"],
    ["Lee et al. 2022 (arXiv:2205.01710)", "Planck Int. V (arXiv:1207.4061)", "this project"],
    "Recommended statistic (with a separate lognormal scatter model).", status="used (reference)")

add("a_statistic", "Median of per-halo profiles",
    "Weighted 50th percentile of f_h(x) in each shell.",
    ["per-halo profiles"], "Negligible.",
    ["robust to outliers; standard in X-ray work (A10, B24, Le Brun, McCarthy, Kay 2024)"],
    ["undershoots the mean where the halo distribution is skewed: pressure -6 to +2% in resolved shells inside 1R, -5 to -12% at 1.34-1.75R, -24 to -34% at 2.5-3R, -39 to -55% at 3.6-4R (log M200c 13.5-15.5)",
     "electrons -5 to -17% at 2-4R",
     "zero where most haloes have no particle in a shell (12.9-13.1: -99% at 0.09-0.13R)",
     "as a painting input: -7.4 to -7.6% on Y(<4R), about -11% on the 2-halo term at ell = 100 (D)"],
    ["Arnaud et al. 2010 (arXiv:0910.1234)", "Braspenning et al. 2024 (arXiv:2312.08277)", "Le Brun et al. 2014 (arXiv:1312.5462)"],
    "Do not paint medians.", status="tested (D, methods_numbers A)")

add("a_statistic", "Log-mean (geometric mean) / lognormal model",
    "exp(<ln f>) over haloes with f > 0; Lee22 fit the mean with a lognormal likelihood.",
    ["per-halo profiles"], "Negligible.",
    ["natural for lognormal scatter; Nagai et al. 2007 averaged log quantities"],
    ["below the mean by the variance term: pressure -1 to -2% at 0.3-1R, -7 to -9% at 1.34-1.75R, -23 to -27% at 2.5-3R, -31 to -49% at 3.6-4R",
     "undefined for empty shells; dropping zeros biases it high (electrons at 12.9-13.1, 0.09-0.13R: +41%)"],
    ["Nagai et al. 2007 (arXiv:astro-ph/0703661)", "Lee et al. 2022 (arXiv:2205.01710)"],
    "Not as the painted amplitude; useful for scatter modelling.", status="computed here (methods_numbers A)")

add("a_statistic", "Aggregate (particle) stacking",
    "Pool all particles of all haloes in scaled coordinates. Normalising each particle by its host's scale gives the M200c-weighted mean of f (V_shell ~ R^3 ~ M); "
    "the raw ratio sum(y)/sum(V P200) is the M^(5/3)-weighted mean.",
    ["particle positions in r/R_Delta, host scale"], "Same pass; no per-halo storage needed.",
    ["no empty-shell problem; works with very sparse data (downsampled snapshots)"],
    ["implicit mass weighting depends on the bin width: within +-2% of the mean in 0.2-dex bins (-4% at 3.6-4R in 13.5-13.7); M^(5/3) weighting up to -7% (13.5-13.7, 3.6-4R) and +4% (14.9-15.5 core)",
     "no per-halo scatter or covariance"],
    ["(generic)"], "Acceptable in narrow bins; prefer per-halo means.", status="computed here (methods_numbers A)")

add("a_statistic", "Y-weighted stacking (Battaglia et al. 2012)",
    "sum_h Y200_h f_h / sum_h Y200_h.",
    ["per-halo Y200 (SOAP ComptonY)"], "Negligible.",
    ["weights haloes roughly as they enter the tSZ signal"],
    ["not the mean profile: +2 to +8% at 0.09-0.28R and 0 to +5% at 0.28-0.61R (resolved shells, log M200c 13.5-15.5), within +-3% beyond; +18 to +31% at 12.9-13.1",
     "B12's Y200 weighting adds +1.6/+2.4/+3.4% to yy 1h at ell = 100/1000/3000 (D)",
     "double counts the Y scatter if scatter is painted separately"],
    ["Battaglia et al. 2012 (arXiv:1109.3711)"], "Do not use with a separate scatter model.", status="tested (D, methods_numbers A)")

add("a_statistic", "Clumping-robust estimators (per-shell median of particle values, clump excision)",
    "Bulk-ICM value that discounts dense clumps; clumping factors C_rho = <rho^2>/<rho>^2, C_P = <P^2>/<P>^2.",
    ["per-particle values or second moments (volume_pe2, mass_rho stored)"], "Same pass.",
    ["isolates the smooth ICM for X-ray comparisons"],
    ["clumps carry real y and electrons: removing them biases tSZ/DM low", "pressure clumping 1.04 at 0.1R, 2.8 at 1.75R (B)",
     "excision of overlappers overshoots Y(<1R) by 6-16% on real haloes (A)"],
    ["Nagai & Lau 2011 (arXiv:1103.0280)"], "Not for painting.", status="diagnostics available (clumping factors, excision blocks)")

# ------------------------------------------------------------------ (b) radius / normalisation / binning
add("b_scaling", "Radius scaling: R200c, R500c, R200m, Rvir (BN98), splashback",
    "Choice of the radius that defines x = r/R_Delta (and which mass enters the scale).",
    ["SO radii from SOAP (2500c, 1000c, 500c, 200c, 100c, 50c, 200m, BN98, 5xR500c available)"],
    "Free once per-halo radii are known; changing the base grid requires a new particle binning (here X200 and X500 grids were stored).",
    ["R500c reduces the within-bin scatter of the inner pressure (sd ln P at ~0.25 R200c: 0.17 vs 0.27)",
     "R200m/splashback make DM outskirts more self-similar (Diemer & Kravtsov 2014; Lau et al. 2015)"],
    ["R200c gives smaller scatter at >= 0.5 R200c (pressure sd ln P 0.12 vs 0.19 at ~0.55R, 0.25 vs 0.40 at ~1R, 0.44 vs 0.54 at ~1.7R)",
     "neither R200c nor R500c removes the mass trend (d ln<P>/d log10 M: +0.52/+0.51 at ~0.25R200c, -0.59/-0.58 at ~1.7R200c)",
     "the painted catalogue must provide the same radius; converting needs a concentration"],
    ["Nagai et al. 2007 (arXiv:astro-ph/0703661)", "Diemer & Kravtsov 2014 (arXiv:1401.1216)", "Lau et al. 2015 (arXiv:1411.5361)"],
    "Use the catalogue's own mass definition (HalfDome M200c) for the grid.", status="R200c and R500c tested (methods_numbers E)")

add("b_scaling", "Pressure and density normalisation, self-similar redshift scaling",
    "P200 = G M200 200 rho_c f_b/(2 R200) (B12); P500 = n_g,500 k T500 (N07); A10 P500 with an extra (M500/3e14)^0.12; B24 P_e,500 (f_B 0.159, mu 0.59, mu_e 1.14); "
    "density in rho_c (B16), n200 = 200 rho_c f_b/(X_H m_p) (Lee22), XGPaint n_e mu_XGP/(rho_c f_b).",
    ["M_Delta, R_Delta, rho_c(z), f_b, mu, mu_e conventions"], "Free.",
    ["removes most of the M^(2/3) E(z)^(8/3) dependence"],
    ["a convention mismatch is a systematic: B16 gas density painted with mu_XGP gives -9% electrons, kSZ -12.6%, mean DM -5.3% (D)",
     "A10's universal M^0.12 shape: yy 1h +29% at ell = 100 (D)",
     "cosmic mean of XGPaint electron units is 0.335 = 1.094 Omega_m, not Omega_m (G)"],
    ["Nagai et al. 2007 (arXiv:astro-ph/0703661)", "Arnaud et al. 2010 (arXiv:0910.1234)", "Battaglia et al. 2012 (arXiv:1109.3711)", "Battaglia 2016 (arXiv:1607.02442)", "Braspenning et al. 2024 (arXiv:2312.08277)"],
    "Normalise exactly as the painter does (P200 and XGPaint electron units).", status="used")

add("b_scaling", "Mass definition, bin width, haloes per bin, bin-mass assignment",
    "Bins in M200c or M500c, in Msun or h^-1 Msun, width 0.05-0.5 dex, with a minimum halo count; the stack is assigned to the bin centre, the mean mass or the mean of log M.",
    ["SO masses, bin edges, selection (complete or random per bin), inclusion weights"], "Free after the pass; more haloes per bin cost particle I/O.",
    ["narrow bins (0.2 dex) keep the implicit weighting (mean vs M-weighted) within +-2%"],
    ["wide bins mix shapes: B24's 0.5-dex M500c bins cost -0.8% Y(<4R), -2.2% 2-halo at ell = 100; 83% of the 13.5-13.7 engine bin's haloes come from the 13.0-13.5 M500c bin (D)",
     "Lee22's > 29-halo rule drops the top bins (17 and 2 haloes) (D)",
     "stratified samples need inclusion weights (here 300 random per 0.2 dex below 10^14.5)"],
    ["Lee et al. 2022 (arXiv:2205.01710)", "Braspenning et al. 2024 (arXiv:2312.08277)", "Kugel et al. 2023 (arXiv:2306.05492): adaptive bins with >= 10 objects"],
    "0.2-dex M200c bins (or a continuous mass model) on the painter's mass definition.", status="used")

add("b_scaling", "Stacking by secondary properties (concentration, accretion rate, formation time, environment)",
    "Split each mass bin by c200, Gamma = dlogM/dloga, formation snapshot, satellite fraction or neighbour mass.",
    ["SOAP Concentration, HBT+ formation/LastMaxMass histories, neighbour catalogue"], "Free after the pass.",
    ["explains scatter: core amplitude correlates with ln c200c (+0.41 to +0.90); 2-4R amplitude with neighbour mass within 4R (+0.34 to +0.51)"],
    ["only useful if the painted catalogue carries the same property (HalfDome RFoF has none); concentration terms did not help painting (G)"],
    ["Lee et al. 2022 (arXiv:2205.01710): concentration quartiles", "Diemer & Kravtsov 2014 (arXiv:1401.1216)", "Lau et al. 2015 (arXiv:1411.5361)"],
    "Diagnostic only for HalfDome.", status="correlations measured (shell_correlations_20261013/physical_drivers.json)")

# ------------------------------------------------------------------ (c) selection
add("c_selection", "Gas-phase cuts: recently AGN-heated, star-forming, cold gas",
    "Remove particles heated by AGN within 15 Myr (SOAP filter), star-forming particles, or T < 1e5 K gas.",
    ["SOAP RecentlyHeatedGasFilter: last AGN event a >= 0.998956 (15 Myr at z = 0) and 10^-1..10^0.3 x Delta T_AGN = 8.80e6-1.756e8 K (Delta T_AGN = 8.80e7 K)",
     "StarFormationRates (> 0 = star-forming), Temperatures"],
    "Free inside the pass (phase axis stored).",
    ["matches FLAMINGO's own y maps and B24"],
    ["recent-AGN cut changes every observable by <= 0.04% (F), <= 0.26% in any literature recipe (D); only unresolved cores move (+73/+28/+4/+0.6% innermost shell when kept, log M200c 13.5-15.1)",
     "SF + cold gas removal matters only in group cores: up to +17% (0.15-0.5 R500c) and +190% (0.08-0.15 R500c) in Q_V at 1e13-1e13.5 M500c (D)",
     "in this pass all 503,296 star-forming particles have y = 0 (apertures diagnostics)"],
    ["Braspenning et al. 2024 (arXiv:2312.08277)", "Schaye et al. 2023 (arXiv:2306.04024)"],
    "Remove recent-AGN gas (default); keep everything else; the choice is immaterial at resolved radii.", status="used")

add("c_selection", "Ownership / membership of gas",
    "All gas in the sphere (SO convention) vs FoF members vs gas bound to the central (HBT+/SUBFIND) vs host-bound (central + own satellites) vs exclusive spheres vs a partition between painted neighbours (A2-ii).",
    ["membership files (GroupNr_bound), SOAP HostHaloIndex, neighbour catalogue above the painting threshold"],
    "Membership adds 8 B/particle and the membership files (not available for the feedback variants).",
    ["A2-ii avoids double painting: equals ALL to <= 1.4% inside 2R, removes 2.4-7.4% of Y(<4R) and 5-7.6% of N_e(<4R) beyond (A)"],
    ["removing satellites biases low (P0 -7 to -26%); clean-only samples bias yy +0.7 to +7.9% (A)",
     "own satellites >= 10% M200c within 2R in 21-31% of centrals"],
    ["Han et al. 2018 HBT+ (arXiv:1708.03646)", "McGibbon et al. 2025 SOAP (arXiv:2507.22669)", "Le Brun et al. 2014 (bound substructure gas excluded)"],
    "All gas with own satellites, A2-ii partition against painted neighbours.", status="used (A2-ii)")

add("c_selection", "Centring",
    "Potential minimum / most-bound particle (HBT+/SOAP HaloCentre), shrinking-sphere centre of mass (B12), most-bound black hole (McCarthy 2025), BCG or X-ray peak (observations).",
    ["SOAP InputHalos/HaloCentre, BoundSubhalo/CentreOfMass, BH positions"], "Free.",
    ["potential centre is stable and matches N-body halo finders"],
    ["centre of mass of the bound subhalo lies 0.087 R200c (median; 84th pct 0.15-0.16, 99th 0.31-0.35) from the most-bound particle at every mass (selected_halos)",
     "observational centres (BCG, X-ray peak) add miscentring that must be forward-modelled"],
    ["Battaglia et al. 2012 (arXiv:1109.3711)", "Gupta et al. 2017 (arXiv:1612.05266)", "McCarthy et al. 2025 (arXiv:2410.19905)"],
    "Use the same centre definition as the painted catalogue (HalfDome RFoF centres; offset not tested here).", status="most-bound particle used")

add("c_selection", "Resolution criteria (inner radius, particles per shell, haloes per bin)",
    "Masks that drop shells inside a few softenings or smoothing lengths, with too few particles, or too few haloes.",
    ["softening (5.7 physical kpc for L1_m9 = 0.0027-0.017 R200c over 10^12.5-10^15.2)", "per-shell particle counts and smoothing-length histograms"],
    "Free.",
    ["this project: x_in >= max(0.04, 4 eps/R), >= 20 gas particles, x_in >= 2 x upper-median h; B24: >= 16% of haloes with >= 1 particle; Lee22: > 29 haloes per bin"],
    ["first resolved shell 0.52 R200c at 13.3-13.5 down to 0.040 at 15.1-15.5 (D)",
     "a 2h -> 1h change moves Y(<4R) by -5% (10^13), -3% (10^14), +3.5% (10^15) in the L2p8 reduced test (memory note, 2026-09-27)"],
    ["Power et al. 2003 (arXiv:astro-ph/0201544)", "Braspenning et al. 2024 (arXiv:2312.08277)", "Lee et al. 2022 (arXiv:2205.01710)"],
    "Keep the 3-part mask; extrapolate the fitted form inward (unresolved radii carry <= 0.7% of C_ell in D).", status="used")

add("c_selection", "Extraction radius and truncation",
    "How far out particles are collected (r_ext) and where the painted profile is cut (hard edge or taper).",
    ["r_ext (here max(4 R200c, 6 R500c)), truncation radius and shape"], "Particle I/O grows ~ r_ext^3 at fixed halo number.",
    ["cut = fit = 4 R200c is the only Pareto-optimal choice for tSZ/kSZ/DM (F)"],
    ["a 2R cut costs tSZ -5%, kSZ -26%, DM -22% (F)", "0.05-dex sigmoid vs hard cut <= 0.21% (C)",
     "reduced snapshots stop at R100c ~ 1.35 R200c; SOAP's outermost aperture is 5 R500c ~ 3.2-3.4 R200c"],
    ["Battaglia et al. 2012 (6 R500 Gaussian taper)", "Arnaud et al. 2010 (Y to 5 R500)"],
    "4 R200c.", status="used")

# ------------------------------------------------------------------ (d) projected / observable space
add("d_projected", "2D annuli from projected maps (snapshot projections or lightcone HEALPix shells)",
    "Projected y, DM or kSZ temperature in annuli around halo centres.",
    ["maps (FLAMINGO lightcone shells: Delta z = 0.05 to z = 3, HEALPix NSIDE 4096 per-shell DM and ComptonY maps on the service, native 16384 for kSZ)", "halo catalogue with sky positions"],
    "Per-shell map 1.61 GB per field at NSIDE 4096; 60 shells to z = 3 (memory note); stacking itself is cheap.",
    ["what observations measure; includes projection, asphericity and 2-halo gas"],
    ["projection depth matters: a +-1 R200c slab misses 1-6% of the projected pressure at 0.1 R200c, 6-14% at 0.5 R200c and 18-26% at R200c (log M 13.5-15.5); electrons 4-15%, 12-24% and 28-35%",
     "gas beyond 2 R200c supplies 6-10% (P) and 12-15% (n_e) of the projected signal at R200c",
     "pixel 0.86 arcmin at NSIDE 4096 = 98/236/323 kpc at z = 0.1/0.3/0.5 (estimate), vs R200c = 0.72 Mpc at 10^13.6",
     "the 2-halo term beyond the extraction sphere is included and must be modelled"],
    ["Schaye et al. 2023 (arXiv:2306.04024)", "McCarthy et al. 2025 (arXiv:2410.19905)", "Le Brun et al. 2015 (arXiv:1501.05666)"],
    "Validation/closure test of the painted maps, not the calibration itself.", status="projection numbers computed (methods_numbers D)")

add("d_projected", "Deprojection (onion peeling, Abel inversion, regularised deconvolution)",
    "Recover 3D profiles from projected annuli assuming spherical symmetry.",
    ["projected profile, outer boundary, beam, regularisation"], "Cheap.",
    ["exact in the noiseless spherical case when the outer boundary encloses all gas"],
    ["outer boundary at 2 R200c while gas extends to 4 R200c: shells inside 0.42 R200c biased <= +0.2% (P) / <= +0.9% (n_e), +0.1-1.2% / +0.4-2.8% at 0.42-1R, +2-4% / +5-7% at 1-1.34R, last shell (1.75-2R) x2.05-2.26 / x2.38-2.53",
     "boundary at 1.34 R200c: +2.8-6.5% (P) / +6.7-13.5% (n_e) at 0.6-1R",
     "piecewise-constant shells on 16 coarse annuli alone: -6.6 to +9.6% shell-to-shell oscillation",
     "noise amplification, beam deconvolution and asphericity come on top"],
    ["Planck Int. V 2013 (arXiv:1207.4061), method of Croston et al. 2006", "Arnaud et al. 2010 (arXiv:0910.1234)"],
    "Not needed in simulations (3D is available).", status="tested on stacks (methods_numbers D)")

add("d_projected", "Forward-modelled observable-space stacks (beam, CAP filter, galaxy/halo selection, miscentring, 2-halo)",
    "Cumulative compensated-aperture-photometry (CAP) profiles of y or kSZ around selected galaxies, beam-convolved.",
    ["lightcone maps, galaxy catalogue (stellar-mass cut), beam (e.g. ACT DR5 150 GHz), CAP radii (1-6 arcmin)"],
    "Map I/O per redshift shell; McCarthy 2025 found 3 shells (galaxy shell +- 1) converge.",
    ["like-for-like with ACT/SPT/Planck stacks"],
    ["mixes masses, satellites (+20% kSZ from satellites), 2-halo and beam; cannot be inverted uniquely into a 3D profile"],
    ["Amodeo et al. 2021 (arXiv:2009.05558)", "Schaan et al. 2021 (arXiv:2009.05557)", "Hill et al. 2018 (arXiv:1706.03753)", "McCarthy et al. 2025 (arXiv:2410.19905)", "Siegel et al. 2026 (arXiv:2608.00164)"],
    "Closure test of the painted HalfDome sky against observations.", status="not done")

# ------------------------------------------------------------------ (e) catalogue shortcuts and saved products
add("e_shortcut", "SOAP spherical-overdensity apertures (cumulative values at up to 9 radii)",
    "Differences of cumulative ComptonY, DopplerB, GasMass, HotGasMass between consecutive SO radii give coarse shells.",
    ["SOAP SO/{2500c,1000c,500c,200c,100c,50c,200m,BN98,5xR500c}: ComptonY(+WithoutRecentAGNHeating), DopplerB, GasMass, HotGasMass, GasTemperature (mass-weighted, variants), GasComptonYTemperature, SpectroscopicLikeTemperature, XRayLuminosity, SORadius"],
    "Reading one float64 column for all 15.3M SOAP rows is 122 MB; no particle I/O.",
    ["exact integrals at those radii (particle Y matches SOAP to ~1e-5)", "available for many snapshots and every feedback/cosmology variant (SOAP 0017-0077 for variants)"],
    ["only ~8 distinct radii at ~0.27-0.32, 0.45-0.50, 0.65-0.68, 1.0, 1.33-1.37 (100c ~ BN98), 1.63-1.70 (200m), 1.76-1.85 (50c), 3.23-3.35 (5 R500c) R200c (NFW estimate; R500c measured 0.646-0.671)",
     "nothing beyond ~3.3 R200c and no core shells", "no electron-number property (GasMass/mu_e assumes full ionisation)",
     "all particles in the sphere (no ownership options); properties masked below 100 bound particles; ComptonY lossy-compressed (DMantissa9)"],
    ["McGibbon et al. 2025 (arXiv:2507.22669)", "Kugel et al. 2023 (arXiv:2306.05492): aperture gas fractions", "McCarthy et al. 2017 (arXiv:1603.02702): Y(<5 r500)"],
    "Best fast route for redshift and feedback-variant trends of integrals (Y, gas mass) and to constrain an outer amplitude; not for shapes inside 0.3 R200c or beyond 3.3 R200c.",
    status="listing read (soap_0077_group_listing.json); radii estimated (soap_aperture_radii_nfw_estimate.json)")

add("e_shortcut", "SOAP fixed-radius spheres and projected apertures",
    "Inclusive (all particles) and exclusive (bound only) spheres at 10, 30, 50, 100, 300, 500, 1000, 3000 kpc; projected apertures at 10-100 kpc.",
    ["SOAP InclusiveSphere/ExclusiveSphere/ProjectedAperture groups"], "Same as SO columns.",
    ["physical-radius profiles of gas mass and mass-weighted T without particles"],
    ["no ComptonY or DopplerB in these groups; radii fixed in kpc, so their r/R200c varies with mass"],
    ["McGibbon et al. 2025 (arXiv:2507.22669)"], "Minor: gas-mass profile checks for groups.", status="listing read")

add("e_shortcut", "FLAMINGO reduced and downsampled snapshots",
    "Reduced: all particles within R100c (~1.35 R200c) of <= 200 haloes per 0.05-dex M200c bin above 1e13 Msun (L2p8_m9 and L1_m8 on the COSMA service). Downsampled: 1% of the gas with Coordinates, Masses, Velocities, ComptonYParameters.",
    ["hdfstream access; SOAP IncludedInReducedSnapshot flag"],
    "L2p8 reduced z = 0: 485M gas particles (~44 B each read in cell order); local link 0.3-6 MB/s, so stream on a cluster.",
    ["many redshifts (all 79 outputs for L2p8), larger box, no 545 GB full snapshot needed"],
    ["reduced data stop at R100c: cannot calibrate the 1.35-4 R200c range that carries most of DM and kSZ",
     "1% downsampling: expected fraction of haloes with >= 20 particles per coarse shell is 0 inside 0.6 R200c at 13.5-13.7 and <= 17% beyond; per-halo profiles impossible below ~10^14.3 (methods_numbers F); no electron or temperature fields"],
    ["Schaye et al. 2023 (arXiv:2306.04024)"], "Useful for redshift trends of the inner profile; not for 4R calibration.", status="used earlier (L2p8 pilot, 2026-09)")

add("e_shortcut", "Re-using this project's saved particle-pass products",
    "measurements_owner.npz etc.: per-halo linear sums [4,588 haloes x 17 owners x 3 phases x 16 quantities x 67 shells to 4 R200c] plus X500 grid, partition, excision, apertures and smoothing-length histograms.",
    ["the study products (read-only)"], "Seconds: every number in methods_numbers.json was recomputed in ~14 s on one core.",
    ["every estimator in (a), every phase/ownership choice in (c), any sub-grid of the 67 edges, any statistic and binning"],
    ["z = 0, HYDRO_FIDUCIAL, L1_m9 only; centrals >= 10^12.5; stratified sample (300 per 0.2 dex below 10^14.5)"],
    ["this project (STEP1_README.md)"], "Fastest route for every z = 0 choice; new snapshots need a new pass.", status="used")

add("e_shortcut", "Halo-model fits to stacked observables",
    "Fit GNFW/OBB profile parameters (plus 2-halo amplitude) to projected, beam-convolved stacks.",
    ["stacked data or mock stacks, halo mass distribution, beam, filter, 2-halo model"], "Cheap per likelihood call (MCMC).",
    ["works on data; gives effective profiles for a galaxy-selected population"],
    ["parameters degenerate; profile tied to the assumed form and mass distribution"],
    ["Amodeo et al. 2021 (arXiv:2009.05558)", "Hill et al. 2018 (arXiv:1706.03753)", "Pandey et al. 2022 (arXiv:2108.01601)"],
    "Not a calibration source for FLAMINGO-based painting.", status="not done")

add("e_shortcut", "Published fitting functions and emulators as the profile source",
    "Use a parametrised profile calibrated elsewhere (B12, B16, A10, Lee22, HMx, picasso) instead of measuring.",
    ["the halo catalogue's M, z (and c for Lee22/picasso)"], "Free.",
    ["no simulation access needed; smooth in M and z"],
    ["inherits the source simulation and conventions: on HalfDome relative to the FLAMINGO-calibrated reference, B12 +4.4 to +5.6% yy, A10 +29% at ell = 100, Lee22 -0.5 to -2.3% (D)",
     "picasso: percent-level bias and ~20% scatter at 0.1-1 R500c (non-radiative training)"],
    ["Battaglia et al. 2012 (arXiv:1109.3711)", "Battaglia 2016 (arXiv:1607.02442)", "Lee et al. 2022 (arXiv:2205.01710)", "Mead et al. 2020 HMx (arXiv:2005.00009)", "Keruzore et al. 2024 picasso (arXiv:2408.17445)", "Kugel et al. 2023 (arXiv:2306.05492)"],
    "Fallback below the calibrated mass range (currently B12/B16).", status="B12/B16 fallback in use")

add("e_shortcut", "Baryon pasting and baryonification",
    "Analytic gas in hydrostatic equilibrium pasted on N-body haloes (halo- or particle-based), or particle displacement (baryonification) applied to maps.",
    ["N-body haloes (M, c), particles for particle-based pasting, feedback parameters to calibrate"],
    "Osato & Nagai 2023: all-sky halo-based map ~170 min on 320 cores; lightcone particle-based ~62 min on 224 cores.",
    ["physical model; handles asphericity (particle-based) and diffuse gas for kSZ"],
    ["itself needs calibration (e.g. against the stacks above); halo-based tSZ within ~10% of the halo model, particle-based ~20% lower"],
    ["Osato & Nagai 2023 (arXiv:2201.02632)", "Shaw et al. 2010 (arXiv:1006.1945)", "Bode et al. 2009 (arXiv:0905.3748)", "Schneider & Teyssier 2015 (arXiv:1510.06034)", "Anbajagane et al. 2024 (arXiv:2409.03822)"],
    "Alternative painter; would consume the same FLAMINGO stacks as calibration targets.", status="not done")

(OUT / "methods_catalogue.json").write_text(json.dumps(M, indent=1))
print(len(M), "methods")

L = [
    dict(paper="Nagai, Kravtsov & Vikhlinin 2007", arxiv="astro-ph/0703661", sim_or_data="ART AMR, 16 clusters (6 relaxed for profiles), 5e13-2e15 h^-1 Msun; Chandra",
         estimator="gas cells in shells (weighting not stated)", statistic="mean (and scatter) of the log", radii="r/r500; fits 0.5-2 r500 (sims), <~ r500 (Chandra)",
         normalisation="P500 = n_g,500 k T500 (mu 0.59, mu_e 1.14)", selection="relaxed subsample", mass_bins="none (one stack)", resolution="peak 5-7 h^-1 kpc"),
    dict(paper="Arnaud et al. 2010 (A10)", arxiv="0910.1234", sim_or_data="REXCESS 33 (31 used) XMM clusters + 3 simulation sets for 1-4 R500",
         estimator="deprojected n_e x deprojected, PSF-corrected T; sims with mass-weighted T", statistic="median (log dispersion)", radii="0.03-1 R500 (X-ray), 1-4 R500 (sims); Y to 5 R500",
         normalisation="P500 = 1.65e-3 E(z)^(8/3) (M500/3e14 h70^-1)^(2/3) h70^2 keV cm^-3, x M^0.12", selection="none (representative sample)", mass_bins="one universal stack, M500 1e14-1e15 Msun",
         resolution=">= 15 clusters per radius; sims renormalised +10% at r > 0.4 R500"),
    dict(paper="Battaglia et al. 2012 (B12)", arxiv="1109.3711", sim_or_data="GADGET-2 SPH, 165 h^-1 Mpc, 256^3, AGN feedback",
         estimator="thermal pressure in spherical shells (estimator not stated)", statistic="Y_Delta-weighted average of P/P_Delta", radii="x = r/R200; fit range not stated, >~ 2 R200 down-weighted; maps all gas to 6 R500 with Gaussian taper",
         normalisation="P_Delta = G M_Delta Delta rho_cr f_b/(2 R_Delta)", selection="substructure kept; FoF b = 0.2, shrinking-sphere COM centre",
         mass_bins="M200 power law (P0 18.1, xc 0.497, beta 4.35 at 1e14); M500 > 4.2e13 for power spectra", resolution="eps 20 h^-1 kpc, 32 SPH neighbours"),
    dict(paper="Battaglia 2016 (B16)", arxiv="1607.02442", sim_or_data="same simulations as B12",
         estimator="gas density in shells", statistic="stacked average (mean/median and weighting not stated)", radii="x = r/R200 (range not stated)",
         normalisation="rho_gas / rho_crit(z)", selection="not stated", mass_bins="M200 power laws (rho0 4.0e3 M^0.29, alpha 0.88, beta 3.83 for AGN feedback)", resolution="as B12"),
    dict(paper="Le Brun et al. 2014 (cosmo-OWLS)", arxiv="1312.5462", sim_or_data="cosmo-OWLS SPH",
         estimator="synthetic X-ray spectra fitted in 3D bins (10-20 bins within r500)", statistic="median, 68% range", radii="r/r500,hse",
         normalisation="rho_crit; S500,hse", selection="T < 1e5 K and gas bound to substructures excluded; centre = most-bound particle of the main subhalo",
         mass_bins="groups 5.75e13-1.54e14 and clusters 2.5e14-1e15 M500, resampled to observed medians", resolution="M500 >= 1e13"),
    dict(paper="Le Brun et al. 2015", arxiv="1501.05666", sim_or_data="cosmo-OWLS synthetic tSZ skies",
         estimator="Planck-like matched-filter aperture fluxes", statistic="stack of LBG-selected objects", radii="Y(<r500) and Y(<5 r500)",
         normalisation="Y vs M500", selection="locally brightest galaxies", mass_bins="stellar/halo mass bins", resolution="-"),
    dict(paper="McCarthy et al. 2014", arxiv="1312.5341", sim_or_data="cosmo-OWLS",
         estimator="pressure from synthetic X-ray density and temperature; tSZ maps by SPH-kernel smoothing", statistic="median, 68%", radii="r/r500,hse",
         normalisation="P500,hse = n_e,500 k T500,hse", selection="(maps: all gas)", mass_bins="groups 5.8e13-1.5e14, clusters 2.5e14-1e15 M500", resolution="-"),
    dict(paper="McCarthy et al. 2017 (BAHAMAS)", arxiv="1603.02702", sim_or_data="BAHAMAS 400 h^-1 Mpc",
         estimator="synthetic X-ray profiles; calibration on hot gas fractions (synthetic X-ray) and GSMF (30 kpc)", statistic="median (profiles); mean vs median for Y(<5 r500)",
         radii="10-20 log bins within r500; Y to 5 r500", normalisation="r500,hse scaling", selection="T < 1e5 K and satellite-bound gas excluded",
         mass_bins="groups 5.25e13-2e14 (526), clusters > 2e14 (166) M500,X", resolution="-"),
    dict(paper="Lee et al. 2022 (Lee22)", arxiv="2205.01710", sim_or_data="IllustrisTNG TNG300 (and SIMBA)",
         estimator="cell sums V^-1(bin) sum V_i A_i (geometric)", statistic="mean <P/P200>, lognormal likelihood with covariance (Hartlap)",
         radii="27 log bins 0.04-1.34 R200 (1.01 and 1.76 tested)", normalisation="P200; n200 = 200 rho_cr f_b/(X_H m_p)",
         selection="not stated", mass_bins="0.2 dex in h^-1 Msun from 1e13 h^-1; concentration quartiles; z <= 2", resolution="> 29 haloes per bin"),
    dict(paper="Braspenning et al. 2024 (B24)", arxiv="2312.08277", sim_or_data="FLAMINGO L2p8_m9 (L1_m9 for model variants)",
         estimator="Q_V (V = m/rho), Q_X (ROSAT; fiducial), Q_M", statistic="median over haloes, 16-84%", radii="30 log bins 0.01-3 R500c",
         normalisation="P500 with f_B 0.159, mu 0.59, mu_e 1.14", selection="no SF gas, T > 1e5 K, no gas AGN-heated in the last 15 Myr; VELOCIraptor + SOAP",
         mass_bins="0.5 dex in M500c, 1e13-1e15.5", resolution="shown where >= 16% of haloes have >= 1 particle"),
    dict(paper="Kugel et al. 2023", arxiv="2306.05492", sim_or_data="FLAMINGO calibration runs (32 per resolution)",
         estimator="aperture gas fraction f_gas(<R500c) (no profiles)", statistic="median, 16-84%", radii="R500c only",
         normalisation="f_gas; X-ray masses with b_HSE ~ N(0.74, 0.10)", selection="VELOCIraptor haloes", mass_bins="adaptive bins 1e13-1e15, >= 10 objects", resolution="M500c >= 10^13.5"),
    dict(paper="Schaye et al. 2023 (FLAMINGO)", arxiv="2306.04024", sim_or_data="FLAMINGO suite",
         estimator="SOAP aperture properties; on-the-fly HEALPix lightcone maps (tSZ, kSZ, DM, X-ray, lensing)", statistic="-", radii="SO and fixed apertures",
         normalisation="-", selection="-", mass_bins="-", resolution="m9: ~1e9 Msun baryons"),
    dict(paper="Kay et al. 2024", arxiv="2404.08539", sim_or_data="FLAMINGO massive clusters",
         estimator="pressure profiles", statistic="median scaled profile, GNFW fit with gamma fixed at 0.31", radii="(scaled radius)", normalisation="P500", selection="massive clusters", mass_bins="-", resolution="-"),
    dict(paper="Planck Int. V 2013", arxiv="1207.4061", sim_or_data="62 Planck ESZ clusters",
         estimator="y annuli (0.25 R500) deconvolved (10' beam) and deprojected (Croston 2006)", statistic="mean (median within 5%)", radii="to ~3 R500",
         normalisation="A10 P500 with M^0.12", selection="ESZ clusters with XMM data", mass_bins="one stack", resolution="10' beam"),
    dict(paper="Amodeo et al. 2021", arxiv="2009.05558", sim_or_data="ACT DR5 + Planck around CMASS",
         estimator="CAP filter on beam-convolved projected GNFW/OBB models", statistic="stack", radii="1-6 arcmin (~1-4 R_vir)",
         normalisation="-", selection="CMASS galaxies", mass_bins="mass-weighted over CMASS halo masses", resolution="beam 2.1' (f090), 1.3' (f150)"),
    dict(paper="McCarthy et al. 2025 (FLAMINGO kSZ)", arxiv="2410.19905", sim_or_data="FLAMINGO lightcone kSZ maps",
         estimator="Doppler-b HEALPix maps (Delta z 0.05 shells, Nside 16384 -> 8192), CAP filter, ACT DR5 150 GHz beam", statistic="velocity-weighted stack",
         radii="CAP radii", normalisation="-", selection="stellar-mass cut fitted to lensing; satellites included (+20% kSZ)", mass_bins="-", resolution="-"),
    dict(paper="Diemer & Kravtsov 2014", arxiv="1401.1216", sim_or_data="N-body",
         estimator="DM density profiles", statistic="median (mean checked)", radii="r/R200m, steepening at ~1-1.2 R200m",
         normalisation="rho_m", selection="isolated, >= 1000 particles in Rvir", mass_bins="peak height nu and accretion rate Gamma", resolution=">= 1000 particles"),
    dict(paper="This project (L1_m9 z = 0, 2026-10)", arxiv="-", sim_or_data="FLAMINGO L1_m9 HYDRO_FIDUCIAL 0077, 4,588 centrals",
         estimator="geometric shell sums (all 16 quantities stored)", statistic="inclusion-weighted mean + lognormal scatter", radii="67 shells to 4 R200c (40 to 6 R500c)",
         normalisation="P200; XGPaint electron units", selection="all gas, own satellites kept, recent-AGN removed, A2-ii partition",
         mass_bins="0.2 dex M200c 12.5-15.5, 300 random per bin below 10^14.5, all above", resolution="x >= max(0.04, 4 eps/R), >= 20 particles, >= 2 h"),
]
(OUT / "literature_table.json").write_text(json.dumps(L, indent=1))
print(len(L), "papers")
