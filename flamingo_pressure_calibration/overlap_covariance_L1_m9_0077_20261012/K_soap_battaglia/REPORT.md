## Battaglia parameters refitted to FLAMINGO (SOAP), Test K run 3

### In short
- **Data.** All 71 fiducial SOAP-HBT outputs (0007–0077, $z$ = 6.04 to 0; 2.16 TB) were read on STRW maas: 8,663,524 central-halo entries with $M_{200c}\ge10^{12.7}\,M_\odot$ (265,402 at $z=0$). In 0.2-dex mass bins per output they form 654 cells; the 597 cells with at least 10 haloes are used, at 70 outputs from $z$ = 0 to 5.5 (`results/k1_status.json`, `report/report_numbers.json` cells_used).
- **The published laws fail for FLAMINGO groups.** Over all cells they need a model-discrepancy floor $\sigma_d$ = 0.354–0.456 (B12) and 0.394–0.452 (B16) in ln per aperture point (`results/k3_fits.json` published). For $z<1$ and $M_{200c}<10^{13.5}$ the mean ln(law/SOAP) is +0.917 for $Y(<R_{500c})$ (a factor 2.5) and +0.609 for $M_{\rm gas}(<R_{200c})$ (a factor 1.8); for clusters ($z<1$, $\ge10^{13.5}$) the published B12 has ln rms 0.147 in $Y(<R_{500c})$; at $z\ge1$ the published B16 underpredicts $M_{\rm gas}(<R_{200c})$ (mean −0.239) (`report/report_numbers.json` aperture_ln_law_over_soap).
- **Refitted published form ('orig').** $\sigma_d$ = 0.0886, 0.0852, 0.0852 (pressure, R1/R2/R3) and 0.1476, 0.1269, 0.1258 (gas). The converged mass-break and free-shape variants reach 0.0540–0.0801 (pressure) and 0.0608–0.1055 (gas); G|R1|free did not converge and is not usable (see 'Fit quality'). No law of the Battaglia family reaches the few-per-cent level over the whole mass and redshift range: the per-cell core radius depends non-monotonically on mass and the group gas evolves quickly at $z<0.5$ (per-cell fits, residual maps).
- **Recommended for tSZ: pressure R3 brk_amp** (XGPaint BreakModel, 11 parameters, $\sigma_d$ = 0.0735). Over all 597 cells the ln rms of law/SOAP is 0.062 for $Y(<R_{500c})$, 0.047 for $Y(<R_{200c})$ and 0.083 for $Y(<5R_{500c})$; the median law/SOAP is 0.985 at $R_{500c}$ and 0.989 at $R_{200c}$.
- **Recommended for kSZ and DM: gas R3 free** (XGPaint BattagliaTauProfilePhysical, 15 parameters, $\sigma_d$ = 0.0652): ln rms 0.047 for $M_{\rm gas}(<R_{200c})$, 0.065 for $M_{\rm gas}(<5R_{500c})$, 0.047 for $M_{\rm gas}(<1000\,{\rm kpc})$ and 0.121 at $R_{2500c}$. Caveat: its $\alpha$ reaches 15 at $10^{15}M_\odot$, $z=0$ (a kink at $x_c$; see 'Reading the parameters'); gas R3 brk_all ($\sigma_d$ = 0.0865, smooth, not an XGPaint type) and gas R3 orig ($\sigma_d$ = 0.1258, published form) are the smooth alternatives. For electrons multiply the amplitude by $1/(0.9r)$ = 1.065–1.093.
- **Profile change at $z=0$, recommended law / published**, at exactly $x=r/R_{200c}$ = 0.1, 0.5, 1, 2 (`report/report_numbers.json` profile_ratio_refit_over_published):

  | field | $M_{200c}$ | $x$ = 0.1 | 0.5 | 1 | 2 |
  |---|---|---|---|---|---|
  | pressure | $10^{13}$ | 0.088 | 0.350 | 0.852 | 1.852 |
  | pressure | $10^{14}$ | 0.418 | 1.010 | 1.568 | 2.013 |
  | pressure | $10^{15}$ | 1.044 | 1.094 | 1.196 | 1.391 |
  | gas | $10^{13}$ | 0.279 | 0.281 | 0.442 | 0.870 |
  | gas | $10^{14}$ | 0.361 | 0.777 | 1.127 | 1.602 |
  | gas | $10^{15}$ | 0.664 | 0.965 | 0.866 | 1.114 |
- **Fit ranges.** Laws fitted within $R_{200c}$ (R1) do not extrapolate: pressure orig R1 overpredicts $Y(<5R_{500c})$ by $e^{0.53}$ = 1.70 on average. R2 and R3 agree inside $R_{200c}$; R3 is needed for 2–3.3 $R_{200c}$ (pressure brk_amp: $Y(<5R_{500c})$ ln rms 0.123 fitted to R2, 0.083 fitted to R3). R3 includes neighbour gas (inclusive apertures).
- **Checks** (`verification.json`): the aperture integral agrees with 40-digit quadrature to 4.2e-15 over $\alpha$ = 0.1–30 and $x/x_c$ up to 400 (1245 cases), and with adaptive quadrature at the nodes of every fitted law to 3.0e-14 (largest $\ln[(x/x_c)^\alpha]$ = 292.7); Python and XGPaint agree to 1.4e-09; the XGPaint snippet runs with the validated XGPaint and reproduces the fitted profiles to 1.4e-09; SOAP per-halo values equal the particle pass at $z=0$ to 0.16% in the median ($\ge10^{13.5}$); a re-read of 3 outputs with the final code reproduces every array of the fitted cell files bit for bit (except the stored quadrature-error diagnostics, ≤ 1.1e-15). The 32-node forward model reproduces the exact mean over every halo of a cell with a 99th-percentile error of 0.20% (pressure) and 0.21% (gas) at the R3 fit points and at most 2.1%; the 4 points per field above 1% are the inner apertures of 3 small cells ($N$ = 36, 80, 151). Propagated through the fits, this error shifts no law parameter by more than 0.19σ (`results/k7_fit_audit.json`).
- **This version corrects the first report** (section 'Changes'): the closed-form integral was wrong for $(x/x_c)^\alpha\gtrsim10^{12}$ in its $b<0.02$ branch, which corrupted the free gas fits; the gas sphere cuts are now $\bar x\le1.8$ and 3.3 as decided (1.9 and 3.6 before); all fits were rerun on maas.

### Data, selection and definitions
- **Files.** `/net/hypernova/data2/FLAMINGO/L1000N1800/HYDRO_FIDUCIAL/SOAP-HBT/halo_properties_00{07..77}.hdf5`, 71 files, 2.16 TB, read-only on maas (`results/k0_inventory.json`); every file has every column used.
- **Haloes.** Central subhaloes (InputHalos/IsCentral = 1) with SO/200_crit TotalMass $\ge10^{12.7}\,M_\odot$ at every output. The same halo appears at many outputs, so 8,663,524 counts halo entries, not distinct haloes.
- **Apertures.** SO 2500c, 1000c, 500c, 200c, 100c, BN98, 200m, 50c and 5xR500c (ComptonY, ComptonYWithoutRecentAGNHeating, GasMass, NumberOfGasParticles, SORadius) and InclusiveSphere 10–3000 physical kpc (GasMass, NumberOfGasParticles). SORadius is stored comoving and converted with its 'including cosmological corrections' factor (times the scale factor $1/(1+z)$).
- **Units check.** The SO radii are physical: $M_{200c}/(\tfrac43\pi R_{200c}^3)$ equals $200\rho_c(z)$ from the header $H(z)$ to within 5.1e-04 (median per output, every output).
- **Normalisation.** $f_b=\Omega_b/\Omega_m$ = 0.0486/0.306 = 0.15882 (header, $\Omega_m$ with massive neutrinos, as in Tests B–J); $P_{200}$ per halo from its own $M_{200c}$, $R_{200c}$ (the equations below).
- **Cells.** 0.2-dex bins of $\log_{10}M_{200c}$ from 12.7, one set per output; a cell is used with at least 10 haloes (597 of 654). The data point is the inclusion-free mean over all valid centrals of the cell; the forward model represents the haloes by 32 strata (below).
- **Aperture masks.** Per halo, an aperture is valid if it is filled (SORadius > 0; spheres need ≥ 100 bound particles). Per cell, it is used if ≥ 90% of the haloes have it, the median gas-particle count is ≥ 50 and its mean radius is ≥ 0.025 $R_{200c}$. The particle cut acts on the cell median (a cut on each halo would select gas-rich groups); every valid halo enters the mean. Spheres need a radius ≥ 22.8 kpc (4 softening lengths; drops 10 kpc). The gas fit uses spheres only from 100 kpc and mean $x\ge0.1$ (inside, the central galaxy's cold gas dominates; the 'Gcore' check keeps them). Within a cell an aperture whose mean $x$ is within 5% of one already kept is dropped (priority 200c, 500c, 2500c, 1000c, 5xR500c, 50c, 100c, 200m, BN98, then spheres).
- **Fit ranges.** R1: SO 2500c, 1000c, 500c, 200c and spheres with cell-mean $x\le1$; R2: adds 100c, BN98, 200m, 50c and spheres to $x\le1.8$; R3: adds 5xR500c and spheres to $x\le3.3$. The largest sphere mean $x$ in the gas datasets is 0.949 / 1.757 / 3.112 (R1/R2/R3; `report/report_numbers.json` sphere_points).
- **Points per fit** (`results/k3_fits.json` datasets): pressure 2,335 / 4,193 / 4,790 (R1/R2/R3), gas 3,313 / 5,430 / 6,382, over 597 cells. $R_{2500c}$ enters 554 of the 597 cells.
- **Per output** (table 'Haloes, cells and points per output' below; `report/report_numbers.json` per_output_counts): the 597 used cells hold 8,663,309 of the 8,663,524 selected halo entries; output 0007 ($z$ = 6.04, 6 haloes) has no cell with ≥ 10 haloes. Used cells per output range from 1 to 13.
- **Redshift coverage** (highest mass bin with ≥ 10 haloes at the output nearest $z$; `report/report_numbers.json` coverage_top_cell_by_z): $z=0$: 15.1-15.3 (21 haloes); $z=0.5$: 14.9-15.1 (26 haloes); $z=1$: 14.7-14.9 (16 haloes); $z=2$: 14.1-14.3 (22 haloes); $z=3$: 13.7-13.9 (10 haloes); $z=4$: 13.3-13.5 (14 haloes); $z=5$: 12.9-13.1 (16 haloes). The laws are extrapolations outside this domain.

#### Haloes, cells and points per output
| output | $z$ | haloes ($\ge10^{12.7}$) | cells: all / used | haloes in used cells | used $\log_{10}M_{200c}$ | pressure points R1 / R2 / R3 | gas points R1 / R2 / R3 |
|---|---|---|---|---|---|---|---|
| 0007 | 6.04 | 6 | 2 / 0 | 0 | – | 0 / 0 / 0 | 0 / 0 / 0 |
| 0008 | 5.5 | 38 | 3 / 1 | 33 | 12.7–12.9 | 4 / 6 / 7 | 4 / 7 / 8 |
| 0009 | 5 | 179 | 4 / 2 | 174 | 12.7–13.1 | 8 / 14 / 16 | 9 / 15 / 18 |
| 0010 | 4.75 | 348 | 4 / 2 | 341 | 12.7–13.1 | 8 / 14 / 16 | 9 / 15 / 19 |
| 0011 | 4.5 | 669 | 4 / 3 | 665 | 12.7–13.3 | 12 / 21 / 24 | 14 / 23 / 29 |
| 0012 | 4.25 | 1,273 | 5 / 3 | 1,268 | 12.7–13.3 | 12 / 21 / 24 | 15 / 24 / 30 |
| 0013 | 4 | 2,318 | 5 / 4 | 2,316 | 12.7–13.5 | 16 / 28 / 32 | 19 / 31 / 39 |
| 0014 | 3.75 | 4,090 | 6 / 4 | 4,087 | 12.7–13.5 | 16 / 28 / 32 | 20 / 33 / 41 |
| 0015 | 3.5 | 6,840 | 6 / 4 | 6,830 | 12.7–13.5 | 16 / 28 / 32 | 20 / 33 / 41 |
| 0016 | 3.25 | 11,208 | 6 / 5 | 11,206 | 12.7–13.7 | 20 / 35 / 40 | 24 / 42 / 52 |
| 0017 | 3 | 17,675 | 6 / 6 | 17,675 | 12.7–13.9 | 24 / 42 / 48 | 29 / 49 / 60 |
| 0018 | 2.95 | 19,383 | 6 / 6 | 19,383 | 12.7–13.9 | 24 / 42 / 48 | 29 / 50 / 61 |
| 0019 | 2.9 | 21,165 | 6 / 6 | 21,165 | 12.7–13.9 | 24 / 42 / 48 | 29 / 50 / 61 |
| 0020 | 2.85 | 23,013 | 6 / 6 | 23,013 | 12.7–13.9 | 24 / 42 / 48 | 29 / 50 / 60 |
| 0021 | 2.8 | 24,943 | 6 / 6 | 24,943 | 12.7–13.9 | 24 / 42 / 48 | 28 / 48 / 58 |
| 0022 | 2.75 | 27,110 | 6 / 6 | 27,110 | 12.7–13.9 | 24 / 42 / 48 | 30 / 51 / 61 |
| 0023 | 2.7 | 29,425 | 6 / 6 | 29,425 | 12.7–13.9 | 24 / 42 / 48 | 30 / 52 / 62 |
| 0024 | 2.65 | 31,936 | 7 / 6 | 31,935 | 12.7–13.9 | 24 / 42 / 48 | 30 / 52 / 63 |
| 0025 | 2.6 | 34,586 | 7 / 6 | 34,582 | 12.7–13.9 | 24 / 42 / 48 | 30 / 52 / 63 |
| 0026 | 2.55 | 37,439 | 7 / 6 | 37,431 | 12.7–13.9 | 24 / 42 / 48 | 29 / 50 / 60 |
| 0027 | 2.5 | 40,431 | 7 / 7 | 40,431 | 12.7–14.1 | 28 / 49 / 56 | 34 / 58 / 70 |
| 0028 | 2.45 | 43,493 | 7 / 7 | 43,493 | 12.7–14.1 | 28 / 49 / 56 | 36 / 61 / 74 |
| 0029 | 2.4 | 46,880 | 7 / 7 | 46,880 | 12.7–14.1 | 28 / 49 / 56 | 36 / 62 / 75 |
| 0030 | 2.35 | 50,247 | 7 / 7 | 50,247 | 12.7–14.1 | 28 / 49 / 56 | 36 / 62 / 75 |
| 0031 | 2.3 | 54,000 | 8 / 7 | 53,997 | 12.7–14.1 | 28 / 49 / 56 | 36 / 61 / 73 |
| 0032 | 2.25 | 57,819 | 8 / 7 | 57,815 | 12.7–14.1 | 28 / 49 / 56 | 35 / 59 / 71 |
| 0033 | 2.2 | 61,932 | 8 / 7 | 61,925 | 12.7–14.1 | 28 / 49 / 56 | 37 / 63 / 75 |
| 0034 | 2.15 | 66,114 | 8 / 8 | 66,114 | 12.7–14.3 | 32 / 56 / 64 | 42 / 71 / 85 |
| 0035 | 2.1 | 70,443 | 8 / 8 | 70,443 | 12.7–14.3 | 32 / 56 / 64 | 42 / 70 / 85 |
| 0036 | 2.05 | 75,107 | 8 / 8 | 75,107 | 12.7–14.3 | 32 / 56 / 64 | 41 / 68 / 83 |
| 0037 | 2 | 79,845 | 8 / 8 | 79,845 | 12.7–14.3 | 32 / 56 / 64 | 42 / 71 / 86 |
| 0038 | 1.95 | 84,652 | 9 / 8 | 84,650 | 12.7–14.3 | 32 / 56 / 64 | 43 / 72 / 87 |
| 0039 | 1.9 | 89,943 | 9 / 8 | 89,940 | 12.7–14.3 | 32 / 56 / 64 | 43 / 72 / 86 |
| 0040 | 1.85 | 95,244 | 9 / 8 | 95,239 | 12.7–14.3 | 32 / 56 / 64 | 44 / 72 / 85 |
| 0041 | 1.8 | 100,737 | 9 / 9 | 100,737 | 12.7–14.5 | 36 / 63 / 72 | 50 / 81 / 95 |
| 0042 | 1.75 | 106,239 | 9 / 9 | 106,239 | 12.7–14.5 | 36 / 63 / 72 | 51 / 83 / 97 |
| 0043 | 1.7 | 111,834 | 10 / 9 | 111,833 | 12.7–14.5 | 36 / 63 / 72 | 52 / 84 / 98 |
| 0044 | 1.65 | 117,646 | 10 / 9 | 117,644 | 12.7–14.5 | 36 / 63 / 72 | 51 / 84 / 98 |
| 0045 | 1.6 | 123,587 | 10 / 9 | 123,584 | 12.7–14.5 | 36 / 63 / 72 | 50 / 81 / 96 |
| 0046 | 1.55 | 129,582 | 10 / 9 | 129,578 | 12.7–14.5 | 36 / 63 / 72 | 51 / 83 / 98 |
| 0047 | 1.5 | 135,813 | 10 / 9 | 135,809 | 12.7–14.5 | 36 / 63 / 72 | 51 / 83 / 97 |
| 0048 | 1.45 | 142,076 | 10 / 9 | 142,072 | 12.7–14.5 | 36 / 63 / 72 | 50 / 81 / 95 |
| 0049 | 1.4 | 148,520 | 11 / 9 | 148,513 | 12.7–14.5 | 36 / 63 / 72 | 49 / 80 / 93 |
| 0050 | 1.35 | 155,165 | 11 / 9 | 155,157 | 12.7–14.5 | 36 / 63 / 72 | 51 / 81 / 94 |
| 0051 | 1.3 | 161,645 | 11 / 10 | 161,642 | 12.7–14.7 | 40 / 70 / 80 | 59 / 94 / 108 |
| 0052 | 1.25 | 168,303 | 11 / 10 | 168,300 | 12.7–14.7 | 40 / 70 / 80 | 60 / 95 / 109 |
| 0053 | 1.2 | 174,780 | 11 / 10 | 174,776 | 12.7–14.7 | 40 / 70 / 80 | 60 / 96 / 111 |
| 0054 | 1.15 | 181,039 | 11 / 10 | 181,034 | 12.7–14.7 | 40 / 70 / 80 | 62 / 98 / 112 |
| 0055 | 1.1 | 187,536 | 11 / 10 | 187,527 | 12.7–14.7 | 40 / 70 / 80 | 62 / 98 / 113 |
| 0056 | 1.05 | 193,721 | 11 / 11 | 193,721 | 12.7–14.9 | 43 / 76 / 87 | 63 / 100 / 116 |
| 0057 | 1 | 199,981 | 11 / 11 | 199,981 | 12.7–14.9 | 43 / 76 / 87 | 63 / 99 / 115 |
| 0058 | 0.95 | 206,115 | 11 / 11 | 206,115 | 12.7–14.9 | 43 / 76 / 87 | 65 / 100 / 116 |
| 0059 | 0.9 | 211,819 | 12 / 11 | 211,814 | 12.7–14.9 | 43 / 76 / 87 | 66 / 104 / 120 |
| 0060 | 0.85 | 217,328 | 12 / 11 | 217,321 | 12.7–14.9 | 43 / 76 / 87 | 66 / 105 / 121 |
| 0061 | 0.8 | 222,860 | 12 / 11 | 222,851 | 12.7–14.9 | 43 / 76 / 87 | 65 / 103 / 119 |
| 0062 | 0.75 | 228,254 | 12 / 12 | 228,254 | 12.7–15.1 | 46 / 82 / 94 | 71 / 112 / 131 |
| 0063 | 0.7 | 233,156 | 13 / 11 | 233,147 | 12.7–14.9 | 42 / 75 / 86 | 65 / 102 / 119 |
| 0064 | 0.65 | 238,097 | 13 / 12 | 238,096 | 12.7–15.1 | 46 / 82 / 94 | 70 / 109 / 128 |
| 0065 | 0.6 | 242,310 | 13 / 12 | 242,309 | 12.7–15.1 | 46 / 88 / 100 | 70 / 115 / 133 |
| 0066 | 0.55 | 246,402 | 13 / 12 | 246,400 | 12.7–15.1 | 46 / 94 / 106 | 71 / 123 / 141 |
| 0067 | 0.5 | 250,275 | 13 / 12 | 250,273 | 12.7–15.1 | 46 / 94 / 106 | 70 / 122 / 140 |
| 0068 | 0.45 | 253,473 | 13 / 12 | 253,470 | 12.7–15.1 | 45 / 93 / 105 | 70 / 121 / 139 |
| 0069 | 0.4 | 256,350 | 13 / 12 | 256,343 | 12.7–15.1 | 45 / 81 / 93 | 70 / 111 / 129 |
| 0070 | 0.35 | 258,959 | 13 / 12 | 258,950 | 12.7–15.1 | 45 / 81 / 93 | 68 / 108 / 127 |
| 0071 | 0.3 | 261,053 | 13 / 13 | 261,053 | 12.7–15.3 | 49 / 88 / 101 | 74 / 118 / 138 |
| 0072 | 0.25 | 262,815 | 13 / 13 | 262,815 | 12.7–15.3 | 49 / 88 / 101 | 74 / 116 / 135 |
| 0073 | 0.2 | 264,194 | 13 / 13 | 264,194 | 12.7–15.3 | 48 / 100 / 113 | 75 / 131 / 150 |
| 0074 | 0.15 | 265,329 | 14 / 13 | 265,326 | 12.7–15.3 | 48 / 100 / 113 | 73 / 128 / 147 |
| 0075 | 0.1 | 265,822 | 14 / 13 | 265,819 | 12.7–15.3 | 48 / 87 / 100 | 73 / 118 / 137 |
| 0076 | 0.05 | 265,513 | 14 / 13 | 265,508 | 12.7–15.3 | 48 / 87 / 100 | 75 / 118 / 137 |
| 0077 | 0 | 265,402 | 14 / 13 | 265,396 | 12.7–15.3 | 48 / 87 / 100 | 73 / 116 / 134 |
| **all** | 0–6.04 | 8,663,524 | 654 / 597 | 8,663,309 | | 2,335 / 4,193 / 4,790 | 3,313 / 5,430 / 6,382 |

### Equations (every symbol defined)
1. **B12 pressure profile** (Battaglia et al. 2012, eq. 10; XGPaint convention):
   P_e(x)/P200 = 0.5176 · P0 · (x/x_c)^γ · [1 + (x/x_c)^α]^(−β),  P200 = G M200c · 200ρ_c(z) · f_b / (2 R200c).
   - x = r/R200c, with r the physical distance from the halo centre [Mpc] (x dimensionless); P_e = n_e k_B T_e, the electron pressure [Pa]; P200 = the halo's self-similar thermal-pressure scale [Pa].
   - 0.5176 = (2+2X_H)/(3+5X_H) = P_e/P_th for a fully ionised gas with hydrogen mass fraction X_H = 0.76 (dimensionless).
   - P0 = amplitude (thermal pressure in units of P200; dimensionless); x_c = core radius [R200c]; α = transition sharpness (1 in B12) and γ = inner slope (−0.3 in B12) (dimensionless); β = outer exponent, P_e ∝ x^(γ−αβ) for x ≫ x_c (dimensionless).
   - G = gravitational constant [m³ kg⁻¹ s⁻²]; M200c = mass inside R200c (SOAP SO/200_crit TotalMass) [Msun]; R200c = radius of the sphere of mean density 200ρ_c(z) (SOAP SO/200_crit SORadius, converted to physical) [Mpc]; z = redshift of the halo's SOAP output (dimensionless); ρ_c(z) = 3H(z)²/(8πG), the critical density at z [kg m⁻³]; f_b = Ω_b/Ω_m = 0.0486/0.306 = 0.15882 (FLAMINGO, Ω_m incl. massive neutrinos; dimensionless).
   - XGPaint's Battaglia16ThermalSZProfile takes this β and converts it internally to the asymptotic slope αβ − γ.
2. **B16 gas-density profile** (Battaglia 2016, eq. A1, as implemented in XGPaint's BattagliaTauProfile):
   ρ_gas(x)/(f_b ρ_c(z)) = ρ0 · (x/x_c)^γ · [1 + (x/x_c)^α]^(−(β+γ)/α).
   - x = r/R200c (dimensionless); ρ_gas = gas mass density (all gas; no electron conversion) [Msun Mpc⁻³]; ρ0 = amplitude in units of f_b ρ_c(z) (dimensionless); x_c = scale radius (0.5 published) [R200c]; α = transition sharpness; β = outer slope (ρ_gas ∝ x^(−β) for x ≫ x_c); γ = inner slope (ρ_gas ∝ x^γ for x ≪ x_c; −0.2 published) (all dimensionless).
   - f_b = 0.15882 (dimensionless); ρ_c(z) = critical density at the redshift z of the halo's SOAP output [Msun Mpc⁻³]; R200c as in 1 [Mpc].
   - The paper's eq. A1 prints −(β−γ)/α and no f_b; the convention fitted here is XGPaint's (profiles_tau.jl). With it the published law gives M̃(<R200c) = 0.701 at 10^14 Msun, z = 0 (`report/report_numbers.json` published_integral_R200c_1e14_z0|G).
3. **Mass and redshift laws** for each free profile parameter Q (P0, x_c, β for B12; ρ0, α, β for B16; plus γ (B12) or x_c and γ (B16) in 'free'):
   - power law (orig, free): Q = A · (M200c / 10^14 Msun)^α_m · (1+z)^α_z, with M200c the halo mass [Msun], z the redshift of its SOAP output, A = value of Q at 10^14 Msun (no h) and z = 0 [units of Q], α_m, α_z = mass and redshift exponents (dimensionless);
   - brk_all (Lee22 BPLall style): ln Q = ln Q_b + α_m^low|high · ln(M200c/M_b) + α_z ln(1+z), with α_m^low for M200c < M_b and α_m^high for M200c ≥ M_b, M_b = the break mass shared by the three parameters [Msun], Q_b = value of Q at M_b and z = 0 [units of Q];
   - brk_amp (XGPaint BreakModel): the orig laws, with the amplitude Q_amp (P0 or ρ0) multiplied by (M200c/M_b)^α_break for M200c < M_b (α_break dimensionless; M_b = XGPaint M_break [Msun]);
   - the sign of each Q is fixed (γ < 0, all others > 0).
4. **Data** (per halo, from SOAP aperture integrals inside physical radius r = x R200c):
   Ỹ(<x) = Y(<r) (m_e c²/σ_T) / (P200 R200c³) = 4π ∫₀ˣ (P_e(x')/P200) x'² dx',   M̃(<x) = M_gas(<r)/(f_b M200c) = (3/200) ∫₀ˣ ρ_gas(x')/(f_b ρ_c(z)) x'² dx'.
   - r = aperture radius: SORadius/(1+z) (SORadius is stored comoving; z = redshift of the output) or 30–3000 physical kpc [Mpc]; x' = integration variable (radius in units of R200c; dimensionless).
   - Y(<r) = (σ_T/m_e c²) ∫_{<r} P_e dV, SOAP ComptonYWithoutRecentAGNHeating (ComptonY in the check) [physical Mpc²]; m_e = electron mass [kg]; c = speed of light [m s⁻¹]; σ_T = Thomson cross-section [m²]; M_gas(<r) = SOAP GasMass of the SO aperture or InclusiveSphere (all gas inside r) [Msun]; Ỹ, M̃ = normalised aperture integrals (dimensionless); P_e, P200, ρ_gas, M200c, R200c, f_b, ρ_c(z) as in 1 and 2. The right-hand equalities use M200c = (4π/3)·200ρ_c(z)·R200c³.
   - The integral I(x) = ∫₀ˣ F(x') x'² dx' of either profile F is computed in closed form: with q = x'/x_c, s = q^α, t = s/(1+s), I = pref · amp · x_c³/α · B(t; a_B, b_B), a_B = (3+γ)/α, b_B = e − a_B, e = the outer exponent (β for B12, (β+γ)/α for B16), pref = 0.5176 (B12) or 1 (B16), amp = P0 or ρ0, B(t; a_B, b_B) = ∫₀ᵗ u^(a_B−1)(1−u)^(b_B−1) du the incomplete beta function (u the integration variable; finite for t < 1 and any real b_B; b_B < 0 means an outer slope shallower than x⁻³); ln t and ln(1−t) = −ln(1+s) are passed separately so that the result stays exact for large s; I = +∞ where γ ≤ −3 (non-integrable core).
5. **Forward model of a cell mean**: d_{c,a} = ln[(1/N_{c,a}) Σ_{h∈c} F̃_{h,a}],  m_{c,a}(θ) = ln[Σ_{k=1}^{32} w_k · norm · I(x_{k,a}; Q(M_k, z_c; θ))].
   - c = cell index (one 0.2-dex M200c bin at one SOAP output); a = aperture index; h = halo index, the sum running over the N_{c,a} central haloes of cell c with aperture a valid; k = quadrature-node index (1–32; one node per halo when the cell has ≤ 32 haloes).
   - d_{c,a}, m_{c,a} = data and model (natural log; dimensionless); F̃_{h,a} = Ỹ (pressure) or M̃ (gas) of halo h in aperture a; N_{c,a} = count of those haloes.
   - Nodes = 4 mass-quantile strata × 8 ln-x-quantile strata of those haloes: w_k = stratum share (dimensionless), M_k = exp(stratum mean of ln M200c) [Msun], x_{k,a} = exp(stratum mean of ln x) [R200c], so every halo enters at its own aperture radius; z_c = redshift of the output of cell c; Q(M, z; θ) = the profile parameters given by the laws of 3; θ = the law parameters (9 to 15); norm = 4π (Ỹ) or 3/200 (M̃).
   - Accuracy (`results/k7_fit_audit.json`): Against the exact mean over every halo, the median error of the nodes at the R3 fit points is 3e-05 (pressure) and 5e-06 (gas) for the published law. Taking the larger of the published-law and perturbed-law errors, the 99th percentile is 0.20% (pressure) and 0.21% (gas) and the maximum 2.06%: 48 of 4,790 pressure and 87 of 6,382 gas points exceed 0.2%, and 4 of each exceed 1%. The points above 1% are the $R_{2500c}$ and $R_{1000c}$ apertures of 3 cells with 36–151 haloes (0055_14.5-14.7, 0026_13.7-13.9, 0070_14.7-14.9), where sd($\ln x$) = 0.35–0.53. For the fitted laws the error is similar (99th percentile ≤ 0.28%, maximum 2.1%). Propagated linearly through each fit, it shifts no law parameter by more than 0.19 Hessian σ (P|R2|brk_amp, amp:lnA) (`results/k7_fit_audit.json`).
6. **Likelihood**: χ²(θ) = Σ_c r_cᵀ [C_c + (σ_d² + σ_0²) 𝟏]⁻¹ r_c,  r_c = d_c − m_c(θ);  σ_d is solved so that χ²_min/(N_pts − n_θ) = 1.
   - d_c, m_c(θ), r_c = vectors of the cell's p data points, model values and residuals (ln; dimensionless); ᵀ = transpose; C_c = p × p bootstrap covariance of the cell's log means (200 halo resamples), its correlation shrunk towards the redshift-pooled correlation with weight λ = clip(p/N, 0.05, 1) (p = points in the cell, N = haloes in the cell); 𝟏 = the p × p identity matrix; σ_d = model-discrepancy floor [ln units]; σ_0 = 0.002 numerical floor [ln units]; χ²_min = χ² at the best-fitting θ; N_pts = number of data points of the fit; n_θ = number of law parameters.
   - BIC = χ²(θ; σ_ref) + n_θ ln N_pts, with σ_ref = the orig fit's σ_d of the same field and range.
   - Convergence: Every converged fit has $\chi^2/\nu=1$ to within 1.4e-06 at its quoted $\sigma_d$ (`results/k7_fit_audit.json` convergence); G|R1|free did not converge. Its least-squares solve stopped at the iteration limit of 600 evaluations in all 3 $\sigma_d$ iterations (scipy status 0) and again in the final solve, so $\chi^2/\nu$ = 0.9978 at its quoted $\sigma_d$ = 0.0971; the floor that gives $\chi^2/\nu=1$ at its final $\theta$ is 0.0970.

### Fit quality
| field | range | points / cells | published $\sigma_d$ | orig | brk_amp | brk_all | free | ΔBIC vs orig (brk_amp / brk_all / free) |
|---|---|---|---|---|---|---|---|---|
| P | R1 | 2335 / 597 | 0.4556 | 0.0886 | 0.0801 | 0.0754 | 0.0540 | -377 / -560 / -1364 |
| P | R2 | 4193 / 597 | 0.3718 | 0.0852 | 0.0705 | 0.0699 | 0.0587 | -1211 / -1246 / -2067 |
| P | R3 | 4790 / 597 | 0.3542 | 0.0852 | 0.0735 | 0.0727 | 0.0683 | -1127 / -1187 / -1577 |
| G | R1 | 3313 / 597 | 0.4521 | 0.1476 | 0.0961 | 0.0696 | 0.0971 (n.c.) | -1838 / -2476 / -1798 (n.c.) |
| G | R2 | 5430 / 597 | 0.3985 | 0.1269 | 0.0991 | 0.0714 | 0.0608 | -2042 / -3597 / -4026 |
| G | R3 | 6382 / 597 | 0.3945 | 0.1258 | 0.1055 | 0.0865 | 0.0652 | -1833 / -3272 / -4466 |

n.c. = not converged: the least-squares solve stopped at its iteration limit, so the value is not the χ²/ν = 1 floor.

- $\sigma_d$ is the rms model discrepancy per aperture point in ln (the floor that makes $\chi^2/\nu=1$). The statistical (bootstrap) errors of the cell means are much smaller: median 0.41% (pressure) and 0.28% (gas), 90th percentile 2.4% and 1.8%, up to 18.1% and 12.6% in the smallest high-mass cells (R3 datasets; `report/report_numbers.json` sd_boot). ΔBIC is evaluated at the orig fit's $\sigma_d$ of the same field and range.
- **Ranking by BIC** (best first; converged fits only): P R1: free < brk_all < brk_amp < orig; P R2: free < brk_all < brk_amp < orig; P R3: free < brk_all < brk_amp < orig; G R1: brk_all < brk_amp < orig; G R2: free < brk_all < brk_amp < orig; G R3: free < brk_all < brk_amp < orig. G|R1|free is left out: it did not converge (below).
- **Pressure free fits.** $x_c$ at $10^{14}$ = 8.58, 8.80, 5.17 (R1/R2/R3) and $\gamma$ = -0.708, -0.730, -1.039; at a bound: R1: beta:lnA; R2: beta:lnA; R3: none. With $x_c$ beyond most of the data they act as a near power law in $x$ inside $R_{200c}$; they interpolate well but should not be extrapolated.
- **Parameters at a fit bound** (within 1e-3 in the fitted coordinate; flagged in the tables): P|R1|free: beta:lnA; P|R2|free: beta:lnA; G|R1|orig: alpha:lnA; G|R1|free: xc:lnA; Gcore|R1|orig: amp:lnA, alpha:lnA. Gas R1 free is degenerate: $x_c$ at its upper bound 10 at $10^{14}$ and $\gamma$ down to -2.99 at the nodes; it is not usable.
- **Not converged: G|R1|free.** Its least-squares solve stopped at the iteration limit of 600 evaluations in all 3 $\sigma_d$ iterations (scipy status 0) and again in the final solve, so $\chi^2/\nu$ = 0.9978 at its quoted $\sigma_d$ = 0.0971; the floor that gives $\chi^2/\nu=1$ at its final $\theta$ is 0.0970. Every other fit reaches $\chi^2/\nu=1$ to within 1.4e-06. Its 20 bootstrap refits also stopped at their iteration limit (status 0 in 20 of 20). Its $\sigma_d$ and ΔBIC in the table (marked n.c.) are therefore not converged values, and `results/k_params.json` marks it `usable: false`. Being degenerate as well, it was not refitted.
- **Information per range** (orig, $\tfrac12\Delta\ln\det$ of the Hessian covariance, `results/k4_information.json`): pressure R1→R2 3.66 bits, R2→R3 2.80 bits; gas 6.54 and 2.64 bits. The pressure orig parameters shift by up to 13.3σ from R1 to R2 but by at most 1.1σ from R2 to R3.
- **Uncertainties.** Hessian errors include $\sigma_d$ as if the discrepancies were independent; bootstrap errors (halo resamples refitted at fixed $\sigma_d$: 100 for orig, 20 for brk_amp / brk_all / free, 10 for the checks) measure halo sampling only and are 0.07–0.98 times the Hessian errors (median ratio per fit). The tables quote the larger; both are in `results/k3_fits.json` and `results/k_params.json`.
- **Multiple starts.** Every fit was started from the published law or the orig fit (three break masses for the brk laws) and from the run-2 optimum; the free gas fits also from the verifier's corrected refit. The lowest $\sigma_d$ wins; every start's result is in `results/k3_fits.json` (starts). Different starts reached different optima (σ_d spread > 1e-4) in 2 of 29 fits (largest: P|R1|brk_amp 0.0049, P|R2|brk_all 0.0001); 1 start(s) were infeasible (a start vector whose $\gamma$ law reaches $\le-3$ at some node gives an infinite model and is rejected).

### Reading the parameters
**Reading the parameters.** $P_0$, $x_c$ and $\beta$ are strongly degenerate on aperture integrals: in the pressure orig R3 fit corr($\ln x_c$, $\ln\beta$) = 0.992 and corr($\ln P_0$, $\ln x_c$) = -0.957 (`results/k3_fits.json` corr). The refitted laws move along the degeneracy ($P_0$ = 3.71 at $10^{14}M_\odot$ instead of 18.1, $x_c$ = 1.254 instead of 0.497, $\beta$ = 5.32 instead of 4.35) while the profiles inside the data change much less. Compare profiles or aperture integrals, not single parameters.

**Mass dependence.** The refitted amplitude laws are steep: $P_0\propto M^{0.953}$ (orig R3; published $M^{0.154}$); brk_amp R3 has $M^{0.873}$ above and $M^{1.044}$ below $M_b=10^{14.03}M_\odot$. This is the gas depletion of FLAMINGO groups. The per-cell fits (published shape, three free parameters) give $x_c$ = 0.222 at $10^{12.79}$, a maximum 0.976 at $10^{13.59}$ and 0.737 at $10^{15.17}$ at $z=0$ (R2, cells not at a bound; `report/report_numbers.json` k2_z0_xc_vs_mass), a trend no single power law follows.

**Gas shape (gas R3 free).** At $10^{14}M_\odot$ and $z=0$: $\rho_0$ = 169.6, $x_c$ = 0.631, $\alpha$ = 3.652, $\beta$ = 2.373, $\gamma$ = -0.851 (published 4000, 0.5, 0.88, 3.83, −0.2): FLAMINGO gas is cuspier inside and much more extended outside than B16. The fitted transition sharpness grows steeply with mass and redshift, $\alpha$ = 3.65 (M/10^14)^0.622 (1+z)^1.126: 0.87 at $10^{13}$, 3.65 at $10^{14}$, 7.5 at $10^{14.5}$, 15.3 at $10^{15}$ and 23.5 at $10^{15.3}M_\odot$ ($z=0$), and 16.3 at $10^{14.5}$, $z=1$; over the fitted cells it spans 0.65–26.9 (`report/report_numbers.json` law_values, law_domain). For $\alpha\gtrsim10$ the profile is in effect a broken power law, $\rho\propto x^{\gamma}$ inside $x_c$ and $x^{-\beta}$ outside (at $10^{15}$, $z=0$: $x_c$ = 0.286, $\gamma$ = -0.669, $\beta$ = 2.637), with a sharp kink at $x_c$ in 3D (smoothed in projection). The aperture integrals cannot resolve how sharp the transition is, so treat the kink as a fitting artefact of the power-law form, not as a physical feature. The orig gas fits reach a similar shape with a very broad transition instead ($\alpha$ = 0.425 and 0.452 at $10^{14}$ in R2 / R3; at its lower bound in R1).

**Recently AGN-heated gas** (check, ComptonY instead of ComptonYWithoutRecentAGNHeating): the orig fits change $P_0$ by −0.7%, +2.3%, +2.4% (R1/R2/R3) and $\sigma_d$ by at most 0.0036.

**Central-galaxy gas** (check 'Gcore', spheres from 30 kpc and $x\ge0.025$): $\sigma_d$ = 0.1567, 0.1644, 0.1685 (R1/R2/R3); in R1 the amplitude is at its bound. The B16 form cannot describe the cold central gas and the hot halo together.

### Aperture integrals
The aperture-integral figure compares $Y(<R_{500c})$, $Y(<R_{200c})$ and $M_{\rm gas}(<R_{200c})$ predicted by each law (at every halo's own radius and mass) with the SOAP cell mean; the grey band is the cells' bootstrap error. ln(law/SOAP), rms / mean over the cells, R3 datasets (`report/report_numbers.json` aperture_ln_law_over_soap):

| quantity | law | all cells: rms / mean | $z<1$, $\ge10^{13.5}$ | $z<1$, $<10^{13.5}$ | $z\ge1$ |
|---|---|---|---|---|---|
| $Y(<R_{500c})$ | published | 0.437 / +0.172 | 0.147 / −0.035 | 0.978 / +0.917 | 0.309 / +0.099 |
| $Y(<R_{500c})$ | orig R3 | 0.084 / −0.023 | 0.126 / −0.017 | 0.073 / +0.027 | 0.058 / −0.038 |
| $Y(<R_{500c})$ | R3 brk_amp | 0.062 / −0.026 | 0.074 / −0.047 | 0.056 / +0.022 | 0.057 / −0.027 |
| $Y(<R_{200c})$ | published | 0.313 / +0.079 | 0.129 / −0.097 | 0.719 / +0.645 | 0.201 / +0.032 |
| $Y(<R_{200c})$ | orig R3 | 0.064 / −0.009 | 0.081 / −0.019 | 0.111 / +0.071 | 0.035 / −0.023 |
| $Y(<R_{200c})$ | R3 brk_amp | 0.047 / −0.006 | 0.039 / −0.031 | 0.103 / +0.074 | 0.026 / −0.012 |
| $Y(<5R_{500c})$ | published | 0.190 / −0.152 | 0.188 / −0.178 | 0.212 / −0.135 | 0.186 / −0.143 |
| $Y(<5R_{500c})$ | orig R3 | 0.086 / −0.000 | 0.042 / −0.001 | 0.108 / −0.008 | 0.095 / +0.001 |
| $Y(<5R_{500c})$ | R3 brk_amp | 0.083 / −0.011 | 0.051 / +0.026 | 0.122 / −0.038 | 0.083 / −0.021 |
| $M_{\rm gas}(<R_{200c})$ | published | 0.351 / −0.050 | 0.104 / +0.037 | 0.674 / +0.609 | 0.315 / −0.239 |
| $M_{\rm gas}(<R_{200c})$ | orig R3 | 0.083 / −0.007 | 0.131 / −0.103 | 0.079 / +0.045 | 0.049 / +0.026 |
| $M_{\rm gas}(<R_{200c})$ | R3 free | 0.047 / −0.010 | 0.042 / −0.034 | 0.070 / +0.056 | 0.042 / −0.013 |
| $M_{\rm gas}(<R_{200c})$ | R3 brk_all | 0.070 / −0.006 | 0.090 / −0.068 | 0.074 / +0.022 | 0.058 / +0.016 |
| $M_{\rm gas}(<5R_{500c})$ | published | 0.383 / −0.308 | 0.122 / −0.097 | 0.213 / −0.104 | 0.480 / −0.450 |
| $M_{\rm gas}(<5R_{500c})$ | orig R3 | 0.098 / −0.022 | 0.108 / +0.084 | 0.027 / +0.009 | 0.104 / −0.078 |
| $M_{\rm gas}(<5R_{500c})$ | R3 free | 0.065 / −0.003 | 0.088 / +0.055 | 0.089 / −0.059 | 0.043 / −0.016 |
| $M_{\rm gas}(<5R_{500c})$ | R3 brk_all | 0.124 / −0.022 | 0.159 / +0.082 | 0.107 / −0.007 | 0.108 / −0.073 |
| $M_{\rm gas}(<R_{2500c})$ | published | 0.626 / −0.178 | 0.378 / +0.303 | 0.763 / +0.707 | 0.697 / −0.496 |
| $M_{\rm gas}(<R_{2500c})$ | orig R3 | 0.273 / −0.052 | 0.178 / +0.015 | 0.236 / +0.210 | 0.311 / −0.112 |
| $M_{\rm gas}(<R_{2500c})$ | R3 free | 0.121 / −0.004 | 0.143 / +0.009 | 0.147 / +0.139 | 0.106 / −0.026 |
| $M_{\rm gas}(<R_{2500c})$ | R3 brk_all | 0.165 / −0.042 | 0.124 / +0.038 | 0.086 / +0.076 | 0.187 / −0.091 |

The core is the weakest part: at $R_{2500c}$ the recommended pressure law has ln rms 0.150 and the recommended gas law 0.121. The residual maps show coherent redshift structure in the inner apertures that a $(1+z)^{\alpha_z}$ law cannot follow.

### Profiles
The figures show $x^3P_e/P_{200}$ and $x^3\rho/(f_b\rho_c)$ (the $x^3$ weighting shows where the volume-integrated signal sits) at $M_{200c}=10^{12.8}$–$10^{15}M_\odot$ and $z$ = 0, 0.5, 1, 2, 3. Lines: the published law (grey dashed), the refitted orig laws of the three ranges, the recommended law and one alternative. Points: SOAP per-halo shells between consecutive apertures (each halo's own radii, mean over the cell, plotted at the mean radii; the innermost cumulative is not shown) from the cell containing the mass at the output nearest $z$; the label gives the cell's $N$ and $z$. At $z=0$: the study's particle stacks (all gas solid black, A2-ii dotted) and Test I's per-bin fits to A2-ii ($R_{\rm fit}=4$; gas = Test I's electron fit divided by $\kappa_E=1.094$). Where no SOAP cell exists the curves are extrapolations.

At $z=0$, in the 0.2-dex bins from $10^{12.9}$ to $10^{14.7}$, the A2-ii partition removes 5.6%–9.8% of the all-gas pressure at 2–3 $R_{200c}$ and 26.4%–46.3% at 3–4 $R_{200c}$; for the gas density 2.4%–9.2% and 10.9%–31.0% (`report/report_numbers.json` particles_z0_A2ii_removed_range). The inclusive SOAP apertures, and therefore the R3 laws, contain that gas.

The first figure shows all five profile parameters of the published, orig R3, recommended and alternative laws vs mass at $z$ = 0, 1, 2 (thick inside the SOAP data, thin where extrapolated). The others show the per-cell free fits (stage K2: three parameters per cell, published fixed shape, bootstrap covariance only; open symbols = a parameter at its bound) and the refitted orig law of each range. With three parameters on 3–4 apertures, R1 cell fits are often degenerate (341 of 597 pressure cells and 207 of 597 gas cells at a bound); in R3, 1 pressure and 131 gas cells are at a bound (`report/report_numbers.json` k2_cells).

### Electrons for DM and kSZ
SOAP stores gas mass, not electrons. The factors below are the particle-measured ratio $r$ = [$M_{\rm gas}/\mu_e$, fully ionised, $X=0.76$] / $N_e$(particles) at $z=0$ at the same radius (median and 16–84% over the study haloes; SOAP audit, `results/k4_electron_factors.json`). Electrons = (gas law / $\mu_e$) / $r$. XGPaint's BattagliaTauProfile paints $n_e=0.9\rho_{\rm gas}/\mu_e$, so to paint the particle electrons with a refitted gas law multiply its $\rho_0$ amplitude by $1/(0.9r)$: 1.065 ($10^{12.5}$–$10^{13.5}$), 1.091 ($10^{13.5}$–$10^{14.5}$), 1.093 ($\ge10^{14.5}$) at $R_{200c}$. The factors are measured at $z=0$ only; neutral and star-forming gas dominate the difference in cores below 0.1 $R_{200c}$.

### Using the parameters
- XGPaint: `report/xgpaint_refit_snippet.jl` (Float64 literals; runs with the validated XGPaint and reproduces the fitted profiles, `report/xgpaint_snippet_check.json`). Python: `report/k_params_usage.py`. Every fit with theta, Hessian covariance, bootstrap sd, sigma_d, data range, conventions and fit_status (converged, usable; G|R1|free is not usable): `results/k_params.json`.
- Full-particle run: pass per-halo aperture integrals (kind 0) or shell profiles (kind 1) for each cell to `k_fit_core.summarise_cell`, build the dataset with `build_dataset`, fit with `fit_global(D, Law(form, kind))`, then `bootstrap_global` and `law_table`; with the same Law kinds and cells the parameters are directly comparable. Compare profiles and aperture integrals (results/k4_profiles.npz, k4_profile_ratios.json), since P0, x_c and beta are degenerate.

### Known limits
- **Inclusive apertures.** SOAP SO apertures and InclusiveSpheres count all gas in the sphere, including satellites and neighbouring haloes. Beyond about 2 $R_{200c}$ the R3 laws contain neighbour gas that the study's A2-ii painting partition assigns to other haloes (at $z=0$, $10^{12.9}$–$10^{14.7}$: pressure 5.6%–9.8% at 2–3 $R_{200c}$ and 26.4%–46.3% at 3–4 $R_{200c}$; gas 2.4%–9.2% and 10.9%–31.0%). Painting every halo with an R3 law double-counts that gas.
- **The core.** Pressure inside about 0.27 $R_{200c}$ is constrained only by the $R_{2500c}$ integral (every halo forward-modelled at its own radius, so the +2–3% bias of stacking at the mean radius does not enter); the gas core only by spheres ≥ 100 kpc with $x\ge0.1$ and $R_{2500c}$. Inner slopes of the free fits below about 0.1 $R_{200c}$ are extrapolations. The recommended gas law has $\gamma$ = -2.36 to -0.64 over the fitted cells (-1.08 at $10^{13}$, -0.85 at $10^{14}$, -1.65 at $10^{13}$, $z=2$): $\gamma<-1$ gives a projected cusp.
- **Few gas particles at low mass.** $R_{2500c}$ is unusable (mostly the median-50-gas-particle cut) in 43 of 597 cells, all at low mass; those cells use only the outer apertures.
- **Physical-kpc spheres at high $z$.** $R_{200c}$ shrinks with $z$ at fixed mass, so a sphere of fixed physical radius moves outward in $x$; the 30–50 kpc spheres reach $x\ge0.1$ at high $z$ but are still dominated by central-galaxy gas, hence the 100-kpc floor for the gas fit.
- **Domain.** Cells reach 15.1-15.3 at $z=0$, 14.7-14.9 at $z=1$, 14.1-14.3 at $z=2$ and 13.7-13.9 at $z=3$. Clusters at high $z$ are extrapolated, and the laws differ from each other there; the recommended gas law's $\alpha$ grows to 33 at $10^{15}$, $z=1$ and 53 at $z=2$ outside the data.
- **Form.** Power laws in $M$ and $(1+z)$ cannot follow the non-monotonic $x_c(M)$ or the late-time evolution of group gas: the best $\sigma_d$ is 0.0540 (pressure) and 0.0608 (gas). Equal weight per aperture point means the fits are dominated by the many group and $z>1$ cells, not by the clusters that dominate the tSZ power.
- **Uncertainties.** The quoted errors treat the model discrepancy as independent per point; the bootstrap measures halo sampling only (20 refits for the brk and free variants).
- **Single simulation.** Only L1_m9 HYDRO_FIDUCIAL; the feedback variants have SOAP-HBT at the same outputs and can be fitted with the same code.

### Recommendations
- **tSZ:** paint with **pressure R3 brk_amp** (XGPaint BreakModel; `report/xgpaint_refit_snippet.jl`; $\sigma_d$ = 0.0735). If only Battaglia16ThermalSZProfile can be used, take **pressure R3 orig** ($\sigma_d$ = 0.0852; $Y(<R_{500c})$ ln rms 0.084 over all cells, 0.126 for clusters at $z<1$). Pressure R3 free has a lower $\sigma_d$ (0.0683) but puts $x_c$ beyond the data at low mass (24.1 $R_{200c}$ at $10^{13}$), so it is a near power law that should not be extrapolated. Do not use R1 laws beyond $R_{200c}$; R2 laws are as good inside $R_{200c}$ but extrapolate worse to 3.3 $R_{200c}$.
- **kSZ and FRB DM:** the best-fitting law is **gas R3 free** (BattagliaTauProfilePhysical with power laws for all five parameters; $\sigma_d$ = 0.0652), with the amplitude multiplied by $1/(0.9r)$ for electrons (1.065 below $10^{13.5}$, 1.091 for $10^{13.5}$–$10^{14.5}$, 1.093 above). Its transition sharpness $\alpha$ exceeds 10 above about $10^{14.5}M_\odot$ at $z=0$ (7.5 at $10^{14.5}$, 15.3 at $10^{15}$) and at lower masses at higher $z$, i.e. a kink at $x_c$; if a smooth profile is required, use **gas R3 brk_all** ($\sigma_d$ = 0.0865; not an XGPaint type, evaluate with `k_fit_core.Law`) or **gas R3 orig** ($\sigma_d$ = 0.1258, published form, XGPaint-ready). A free-shape variant with a constant $\alpha$ would remove the kink; it was not part of this run.
- **Change from the published parameters:** for groups the published laws do not describe FLAMINGO (at $z<1$, $<10^{13.5}$: factor 2.5 too much $Y(<R_{500c})$ and 1.8 too much $M_{\rm gas}(<R_{200c})$); for clusters at $z<1$ the published B12 is within 15% rms in $Y(<R_{500c})$, but the recommended pressure law puts 1.20–2.01 times the published pressure at 1–2 $R_{200c}$ for $10^{14}$–$10^{15}M_\odot$ ($z=0$), and the recommended gas law 0.87–1.60 times the published gas density there. Expect the tSZ and kSZ one-halo terms to drop where groups dominate and to change by tens of per cent at intermediate masses; quantify with the HalfDome painter before adopting.
- **Neighbour gas:** if the painting target is the A2-ii partition (Tests H–J), use the R3 laws inside 2 $R_{200c}$ and correct 2–4 $R_{200c}$ with the particle A2-ii / all-gas ratios, or refit A2-ii shells from the full-particle run with the same code.
- **Full-particle run:** pass per-halo shell profiles (kind 1) or aperture integrals (kind 0) to `k_fit_core.summarise_cell`, build the `Dataset` with the same cells and `Law` kinds and fit with `fit_global`; compare profiles and aperture integrals (`results/k4_profile_ratios.json`, `results/k4_profiles.npz`) and the per-cell fits rather than single parameters ($P_0$, $x_c$ and $\beta$ are degenerate: correlation up to 0.992 in pressure R3 orig).
- **Website:** the page source `K_soap_battaglia/report/fit_soap_battaglia.txt` (site content syntax) builds with `build_site.py --check` in a scratch copy of the site; it is installed into `stacking_choices_20261014/site/content/` by the workflow's site step (this test may write only in its own directory).

### Run record
- **maas runs** (`CLUSTER_RUN.json` maas_runs; nice 19, ionice 7, the wrapper's load and memory guards; read-only on /net/hypernova):
  - attempt 0 (smoke; 2 workers): returncode 0, wall 206 s, peak tree RSS 0.39 GiB, min machine available 178.4 GiB (attempts/run1_k0k1_20261015/test_K_done_0.json); fetched with k_maas.py fetchdir --dir results_smoke (14 files, 2.61 MiB)
  - attempt 1 (full (k0, k1, k2, k3 nominal fits); 3 workers): stopped by us (cluster_tests.py stop K) at 2026-10-15 16:19 EEST during the K3 bootstrap (estimated 4-5 h at 3 workers with 100/40/20 refits): returncode -15, wall 8544 s, peak tree RSS 2.50 GiB, min machine available 165.2 GiB (attempts/run1_k0k1_20261015/test_K_failed_3.json, test_K_3.log). K0/K1/K2 products and the nominal fits were kept; its run_manifest entry was not written (killed) - stage times above are from the log.
  - attempt 2 (full (k2, k3 resumed); 8 workers): returncode 0, wall 4952 s, peak tree RSS 3.26 GiB, min machine available 177.4 GiB (results/test_K_done.json, results/run_manifest.json); fetched 2026-10-15 18:30 EEST with cluster_tests.py fetch K (152 files, 56.4 MiB, SHA256 checked; results/SHA256SUMS OK)
  - attempt 3a (K1 provenance check (k1 on outputs 0077, 0037, 0008 with the run-3 code, --keep-halo 0, results_k1check/); 3 workers): returncode 0, wall 283 s, peak tree RSS 1.05 GiB, min machine available 173.5 GiB. all arrays of the three cell files identical to results/cells except the comp_err_* diagnostics (<= 1.1e-15); superseded by 3e (same check with the final k_fit_core.py)
  - attempt 3b (k2, k3 (first launch of the corrected fits); 8 workers): returncode -15, wall 322 s, peak tree RSS 1.94 GiB, min machine available 175.3 GiB. stopped by us after 5 min: the free laws can reach gamma <= -3 at extreme nodes (run-2 G|R1|free: -3.22), where the aperture integral diverges; k_fit_core.cumulative now returns +inf there
  - attempt 3c (k2, k3 (relaunch with the gamma guard); 3 workers): returncode -15, wall 783 s, peak tree RSS 1.45 GiB, min machine available 175.3 GiB. stopped by us after 13 min (k2 done): the wrapper had capped it to 3 workers while Test J ran; relaunched with 8 workers once J had finished, with corrected docstrings in k_fit_core.py (no functional change)
  - attempt 3d (k2, k3 (final fits); 8 workers): returncode 0, wall 6163 s, peak tree RSS 3.41 GiB, min machine available 177.0 GiB. final fits of this report (k2 440 s, k3 5721 s: nominal fits 2659 s, bootstrap 100/20/10 refits); fetched 2026-10-15 22:24 EEST with cluster_tests.py fetch K (152 files, 56.4 MiB, SHA256 checked; results/SHA256SUMS OK). STRW gateway unreachable 21:16-22:23 EEST (polls failed), the job was unaffected. No per-halo scratch was written in run 3 (k2/k3 only; the K1 checks used --keep-halo 0): /tmp/nagainis_halfdome_testK did not exist at 22:30 EEST (k_maas.py scratch: exists_after false)
  - attempt 3e (K1 provenance check with the final code (k1 on 0077, 0037, 0008, --keep-halo 0); 3 workers): returncode 0, wall 264 s, peak tree RSS 1.06 GiB, min machine available 176.2 GiB. compared by k1_check_compare.py -> results/k1_provenance_check.json
- **K1 provenance:** [77, 37, 8] re-read with the final code (k_read.py de4df13e1977…, k_common.py 5c97e9d0a9e5…): every array of the cell files used by the fits is bit-identical except the stored quadrature-error diagnostics comp_err_* (≤ 1.1e-15 absolute, from the corrected integral) (`results/k1_provenance_check.json`).
- **Local audit (this revision):** `k_fit_audit.py` → `results/k7_fit_audit.json`. It covers convergence and the compression error at the fit points, exact in the cells with ≤ 256 haloes and a 256-halo proxy above, with its linear effect on every fit. It reproduces the stored published-law diagnostic to 3e-08. `k_params_status.py` adds `fit_status` and `electron_factors` to `results/k_params.json` (`results/k7_params_provenance.json`). No maas job was started for this revision.
- **Local:** K4 comparisons (`k_compare.py`), figures (`make_figures.py`), K5 parity (`k_parity.py`, XGPaint through the validated Julia 1.12.2 checkout), snippet check (`k_snippet_check.py`), run-2 comparison (`k_run_compare.py`), verification (`verify_k.py`), report (`k_numbers.py`, `report/write_narrative.py`, `report_k.py`).

### Changes from earlier versions
- **Revision after the second verification (local only; no new maas run, no fitted number changed).** (1) The forward-model accuracy is now quoted correctly. The page used to say the 32 nodes reproduce the all-halo mean 'to within 0.2 per cent'; that bound is wrong at the fit points, where the 99th percentile is 0.20% / 0.21% and the maximum 2.06%. The new `k_fit_audit.py` also propagates the error through every fit: at most 0.19σ, and 0.17σ / 0.13σ for the recommended fits. (2) G|R1|free is marked as not converged (n.c.) in the tables and figure, left out of the BIC ranking and flagged `usable: false` in `results/k_params.json`. (3) A table of haloes, cells and points per output was added. (4) Every equation now defines all its symbols. (5) `results/k_params.json` gained a `fit_status` entry per fit and the `electron_factors` block its B16 convention refers to. All maas fields are unchanged; the maas file is kept as `attempts/run3_20261015/k_params_maas.json`, and the results/SHA256SUMS line was updated (`results/k7_params_provenance.json`).
- **Aperture integral.** `k_fit_core.cumulative` formed $1-t$ from $t=s/(1+s)$, $s=(x/x_c)^\alpha$, in its $b<0.02$ branch (outer slope shallower than $x^{-3}$), which loses all precision for $s\gtrsim10^{12}$: at $x/x_c$ = 13.75 (B16, $\gamma$ = −0.55) the old value was off by +120.5%, +616.3%, inf and inf for $\beta$ = 2.8, 2.95, 3.01, 3.2 at $\alpha$ = 11.6, but only 3e-07–5e-07 at $\alpha$ = 10. At the run-2 free gas laws it was wrong by more than 1e-6 at 69 of 196,823 quadrature nodes (R3; up to 198%) and 148 of 167,329 (R2; up to 105%), and the run-2 gas R1 free law had $\gamma\le-3$ (an infinite core integral) at 106 nodes (minimum -3.22) (`results/k6_run2_vs_run3.json`). It now passes $\ln t$ and $\ln(1-t)$ separately (incomplete beta / its complement at the exact $1-t$; for $b<0.02$ a series in $1-u$ with exact $\int w^{p-1}dw$ terms, Gauss–Legendre above $w=1/(a-1)$ for $a>4$) and returns $+\infty$ where $\gamma\le-3$ (non-integrable core, which a free $\gamma$ law can reach at extreme nodes). The first report's statement '1.9e-9 against quadrature' was valid only for $\alpha\le5$.
- **Sphere cuts.** The gas datasets now use spheres with cell-mean $x\le1.8$ (R2) and $\le3.3$ (R3), as decided; run 2 used 1.9 and 3.6 (undocumented). Pressure datasets have no spheres and are unchanged.
- **Fits rerun on maas** (k2, k3 from scratch; 8 workers) with extra start vectors and 20 instead of 12 bootstrap refits for the brk and free variants. Changes (`results/k6_run2_vs_run3.json`): G|R2|free: $\sigma_d$ 0.0632 → 0.0608, largest parameter shift 21.1σ (gamma:lnA; run-3 Hessian σ); G|R3|free: $\sigma_d$ 0.0665 → 0.0652, largest parameter shift 27.1σ (amp:lnA; run-3 Hessian σ); G|R1|free: $\sigma_d$ 0.0955 → 0.0971, largest parameter shift 2.6σ (amp:am; run-3 Hessian σ); P|R3|brk_amp: $\sigma_d$ 0.0735 → 0.0735, largest parameter shift 0.0σ (beta:az; run-3 Hessian σ); P|R3|orig: $\sigma_d$ 0.0852 → 0.0852, largest parameter shift 0.0σ (beta:am; run-3 Hessian σ); G|R3|orig: $\sigma_d$ 0.1257 → 0.1258, largest parameter shift 0.5σ (beta:az; run-3 Hessian σ).
- **Report fixes.** Profile ratios are now evaluated at exactly $x$ = 0.1, 0.5, 1, 2 (they were taken at the nearest grid points 0.0996, 0.5035, 0.992, 2.030); the redshift-coverage line was one mass bin too low for $z\ge1$; the A2-ii neighbour-gas ranges are now quoted per field from the particle stacks; parameters at a fit bound are flagged in the tables; the XGPaint snippet uses Float64 literals (with `alpha_amp=1` it raised a MethodError) and is checked by running it.

## Parameter tables

Units follow XGPaint: mass M200c in Msun (no h), pivot 1e14 Msun, laws Q = A (M200c/1e14 Msun)^alpha_m (1+z)^alpha_z; x_c in R200c; P0 in units of P200 (thermal; XGPaint multiplies by 0.5176); rho0 in units of f_b rho_c(z) (XGPaint BattagliaTauProfile convention, exponent -(beta+gamma)/alpha).

Errors are 1 sigma, taken as the larger of the Hessian error (which includes sigma_d) and the bootstrap error (100 refits for orig, 20 for brk_amp, brk_all and free, 10 for the Pagn check). "(at bound)" means the parameter sits within 1e-3 of its fit bound in the fitted coordinate.

The fits were not changed in this revision. G|R1|free did not converge and is not usable; results/k_params.json marks it fit_status.usable = false.

Sources:
- report/param_tables.md.txt, written by report_k.py from results/k3_fits.json.
- results/k_params.json, which holds theta, the Hessian covariance and boot_sd for every fit.

**Pressure (B12), published form ('orig'):**

| fit | parameter | A (at 1e14 Msun, z = 0) | alpha_m | alpha_z | sigma_d |
|---|---|---|---|---|---|
| published | P0 | 18.100 | 0.1540 | -0.7580 | – |
| published | x_c | 0.4970 | -8.65e-03 | 0.7310 | – |
| published | beta | 4.350 | 0.0393 | 0.4150 | – |
| orig R1 | P0 | 6.494 ± 0.3714 | 1.226 ± 0.0159 | 1.651 ± 0.0499 | 0.0886 |
| orig R1 | x_c | 0.5015 ± 0.0423 | -0.5980 ± 0.0193 | -1.048 ± 0.0732 | 0.0886 |
| orig R1 | beta | 2.912 ± 0.1266 | -0.0274 ± 0.0102 | -0.1516 ± 0.0369 | 0.0886 |
| orig R2 | P0 | 3.901 ± 0.1464 | 0.9478 ± 0.0135 | 1.170 ± 0.0425 | 0.0852 |
| orig R2 | x_c | 1.177 ± 0.0789 | -0.3399 ± 0.0224 | -0.3517 ± 0.0774 | 0.0852 |
| orig R2 | beta | 5.147 ± 0.2358 | 0.0158 ± 0.0151 | 0.1451 ± 0.0534 | 0.0852 |
| orig R3 | P0 | 3.707 ± 0.1046 | 0.9528 ± 0.0107 | 1.222 ± 0.0331 | 0.0852 |
| orig R3 | x_c | 1.254 ± 0.0568 | -0.3581 ± 0.0162 | -0.4145 ± 0.0548 | 0.0852 |
| orig R3 | beta | 5.321 ± 0.1577 | 8.19e-04 ± 0.0105 | 0.1143 ± 0.0363 | 0.0852 |

**Gas density (B16), published form ('orig'):**

| fit | parameter | A (at 1e14 Msun, z = 0) | alpha_m | alpha_z | sigma_d |
|---|---|---|---|---|---|
| published | rho0 | 4000 | 0.2900 | -0.6600 | – |
| published | alpha | 0.8800 | -0.0300 | 0.1900 | – |
| published | beta | 3.830 | 0.0400 | -0.0250 | – |
| orig R1 | rho0 | 3.524e+07 ± 3.494e+07 | -0.6047 ± 0.0387 | -2.605 ± 0.1597 | 0.1476 |
| orig R1 | alpha | 0.1000 ± 8.06e-03 (at bound) | 0.0290 ± 5.73e-03 | 0.7611 ± 0.0264 | 0.1476 |
| orig R1 | beta | 1.973 ± 0.0191 | -0.0488 ± 3.46e-03 | 0.4438 ± 0.0149 | 0.1476 |
| orig R2 | rho0 | 7894 ± 419.5 | -0.2993 ± 8.02e-03 | -0.8884 ± 0.0247 | 0.1269 |
| orig R2 | alpha | 0.4247 ± 6.73e-03 | 0.2502 ± 3.28e-03 | 0.8917 ± 0.0105 | 0.1269 |
| orig R2 | beta | 2.691 ± 0.0142 | 0.0872 ± 2.05e-03 | 0.3970 ± 6.84e-03 | 0.1269 |
| orig R3 | rho0 | 8380 ± 370.1 | -0.2912 ± 7.12e-03 | -0.8046 ± 0.0212 | 0.1258 |
| orig R3 | alpha | 0.4521 ± 5.48e-03 | 0.2359 ± 2.39e-03 | 0.7295 ± 7.45e-03 | 0.1258 |
| orig R3 | beta | 2.906 ± 0.0101 | 0.0823 ± 1.32e-03 | 0.2786 ± 4.41e-03 | 0.1258 |

**Pressure (B12): mass-break and free-shape variants, plus the check with recently AGN-heated gas (Pagn).** For brk_all, the A column gives Q_b, the value at M_b and z = 0.

| fit | parameter | A (at 1e14 Msun, z = 0) | alpha_m | alpha_z | sigma_d |
|---|---|---|---|---|---|
| brk_amp R1 | P0 | 4.940 ± 0.2608 | 0.7643 ± 0.0144 | 0.8862 ± 0.0373 | 0.0801 |
| brk_amp R1 | x_c | 0.8554 ± 0.0890 | -0.4627 ± 0.0260 | 0.1620 ± 0.0771 | 0.0801 |
| brk_amp R1 | beta | 3.895 ± 0.2807 | -0.0954 ± 0.0219 | 0.5371 ± 0.0595 | 0.0801 |
| brk_amp R1 | break | log10 M_b = 14.099 ± 0.0150, alpha_break = 0.2320 ± 0.0106 | | | |
| brk_all R1 | P0 | Q_b = 3.774 ± 0.1986 | 1.292 ± 0.0360 (low) / 0.6145 ± 0.0188 (high) | 0.4982 ± 0.0275 | 0.0754 |
| brk_all R1 | x_c | Q_b = 0.3964 ± 0.0297 | -1.398 ± 0.1149 (low) / 0.0767 ± 0.0324 (high) | 0.9690 ± 0.0639 | 0.0754 |
| brk_all R1 | beta | Q_b = 1.759 ± 0.0735 | -0.9808 ± 0.1159 (low) / 0.3094 ± 0.0240 (high) | 1.141 ± 0.0618 | 0.0754 |
| brk_all R1 | break | log10 M_b = 13.686 ± 0.0121 | | | |
| free R1 | P0 | 0.4201 ± 0.0357 | 0.7379 ± 0.0127 | 2.117 ± 0.0353 | 0.0540 |
| free R1 | x_c | 8.576 ± 0.7363 | -0.9988 ± 0.0211 | -1.277 ± 0.0531 | 0.0540 |
| free R1 | beta | 20.000 ± 1.803 (at bound) | -0.7491 ± 0.0237 | -0.6067 ± 0.0582 | 0.0540 |
| free R1 | gamma | -0.7079 ± 0.0193 | 0.2467 ± 6.64e-03 | -0.3637 ± 0.0164 | 0.0540 |
| brk_amp R2 | P0 | 3.619 ± 0.0944 | 0.7929 ± 7.88e-03 | 1.086 ± 0.0224 | 0.0705 |
| brk_amp R2 | x_c | 1.574 ± 0.0760 | -0.4835 ± 0.0139 | -0.2236 ± 0.0412 | 0.0705 |
| brk_amp R2 | beta | 6.254 ± 0.2228 | -0.1340 ± 0.0113 | 0.2380 ± 0.0313 | 0.0705 |
| brk_amp R2 | break | log10 M_b = 14.029 ± 0.0125, alpha_break = 0.2066 ± 6.55e-03 | | | |
| brk_all R2 | P0 | Q_b = 3.347 ± 0.1284 | 1.089 ± 0.0158 (low) / 0.6422 ± 0.0262 (high) | 0.9969 ± 0.0222 | 0.0699 |
| brk_all R2 | x_c | Q_b = 1.264 ± 0.0698 | -0.6668 ± 0.0363 (low) / -0.1925 ± 0.0404 (high) | -0.0411 ± 0.0418 | 0.0699 |
| brk_all R2 | beta | Q_b = 5.005 ± 0.1947 | -0.2834 ± 0.0318 (low) / 0.0619 ± 0.0272 (high) | 0.3777 ± 0.0322 | 0.0699 |
| brk_all R2 | break | log10 M_b = 13.889 ± 0.0122 | | | |
| free R2 | P0 | 0.3425 ± 0.0209 | 0.7576 ± 0.0102 | 1.972 ± 0.0301 | 0.0587 |
| free R2 | x_c | 8.802 ± 0.4916 | -0.9237 ± 0.0129 | -1.204 ± 0.0370 | 0.0587 |
| free R2 | beta | 20.000 ± 1.099 (at bound) | -0.6630 ± 0.0139 | -0.5628 ± 0.0401 | 0.0587 |
| free R2 | gamma | -0.7302 ± 0.0129 | 0.2207 ± 5.17e-03 | -0.1975 ± 0.0140 | 0.0587 |
| brk_amp R3 (recommended tSZ) | P0 | 3.769 ± 0.0779 | 0.8735 ± 7.23e-03 | 1.273 ± 0.0206 | 0.0735 |
| brk_amp R3 (recommended tSZ) | x_c | 1.376 ± 0.0420 | -0.5005 ± 9.88e-03 | -0.5069 ± 0.0320 | 0.0735 |
| brk_amp R3 (recommended tSZ) | beta | 5.625 ± 0.1144 | -0.1122 ± 6.84e-03 | 0.0561 ± 0.0216 | 0.0735 |
| brk_amp R3 (recommended tSZ) | break | log10 M_b = 14.028 ± 0.0144, alpha_break = 0.1705 ± 5.49e-03 | | | |
| brk_all R3 | P0 | Q_b = 4.670 ± 0.1916 | 1.029 ± 0.0125 (low) / 0.6052 ± 0.0331 (high) | 1.237 ± 0.0259 | 0.0727 |
| brk_all R3 | x_c | Q_b = 1.219 ± 0.0572 | -0.4380 ± 0.0209 (low) / -0.2577 ± 0.0409 (high) | -0.4449 ± 0.0432 | 0.0727 |
| brk_all R3 | beta | Q_b = 5.474 ± 0.1672 | -0.0568 ± 0.0146 (low) / -5.18e-03 ± 0.0242 (high) | 0.0976 ± 0.0295 | 0.0727 |
| brk_all R3 | break | log10 M_b = 14.071 ± 0.0143 | | | |
| free R3 | P0 | 0.1646 ± 0.0170 | 0.8009 ± 0.0134 | 2.321 ± 0.0325 | 0.0683 |
| free R3 | x_c | 5.167 ± 0.3326 | -0.6692 ± 9.35e-03 | -0.6764 ± 0.0328 | 0.0683 |
| free R3 | beta | 8.921 ± 0.4297 | -0.3718 ± 9.68e-03 | 0.1552 ± 0.0401 | 0.0683 |
| free R3 | gamma | -1.039 ± 0.0145 | 0.1531 ± 3.94e-03 | -0.3615 ± 0.0150 | 0.0683 |
| orig R1 (Y incl. recent AGN) | P0 | 6.450 ± 0.3924 | 1.200 ± 0.0175 | 1.897 ± 0.0543 | 0.0921 |
| orig R1 (Y incl. recent AGN) | x_c | 0.4867 ± 0.0424 | -0.5635 ± 0.0206 | -1.266 ± 0.0749 | 0.0921 |
| orig R1 (Y incl. recent AGN) | beta | 2.835 ± 0.1222 | -0.0113 ± 0.0101 | -0.2060 ± 0.0359 | 0.0921 |
| orig R2 (Y incl. recent AGN) | P0 | 3.993 ± 0.1611 | 0.9151 ± 0.0147 | 1.235 ± 0.0466 | 0.0864 |
| orig R2 (Y incl. recent AGN) | x_c | 1.085 ± 0.0767 | -0.3088 ± 0.0243 | -0.3752 ± 0.0830 | 0.0864 |
| orig R2 (Y incl. recent AGN) | beta | 4.851 ± 0.2286 | 0.0307 ± 0.0161 | 0.1445 ± 0.0557 | 0.0864 |
| orig R3 (Y incl. recent AGN) | P0 | 3.796 ± 0.1121 | 0.9147 ± 0.0114 | 1.263 ± 0.0349 | 0.0858 |
| orig R3 (Y incl. recent AGN) | x_c | 1.156 ± 0.0537 | -0.3181 ± 0.0172 | -0.4003 ± 0.0568 | 0.0858 |
| orig R3 (Y incl. recent AGN) | beta | 5.013 ± 0.1485 | 0.0218 ± 0.0109 | 0.1377 ± 0.0367 | 0.0858 |

**Gas density (B16): mass-break and free-shape variants:**

| fit | parameter | A (at 1e14 Msun, z = 0) | alpha_m | alpha_z | sigma_d |
|---|---|---|---|---|---|
| brk_amp R1 | rho0 | 1.354e+04 ± 1076 | -0.4495 ± 0.0110 | -1.010 ± 0.0310 | 0.0961 |
| brk_amp R1 | alpha | 0.3342 ± 7.69e-03 | 0.2579 ± 5.24e-03 | 0.9106 ± 0.0150 | 0.0961 |
| brk_amp R1 | beta | 2.320 ± 0.0190 | 0.0471 ± 3.33e-03 | 0.4961 ± 0.0105 | 0.0961 |
| brk_amp R1 | break | log10 M_b = 13.299 ± 5.81e-03, alpha_break = -0.6068 ± 0.0123 | | | |
| brk_all R1 | rho0 | Q_b = 3.045e+04 ± 2034 | -0.5066 ± 0.0196 (low) / -0.5811 ± 0.0116 (high) | -0.9697 ± 0.0235 | 0.0696 |
| brk_all R1 | alpha | Q_b = 0.1857 ± 2.85e-03 | -0.1148 ± 7.80e-03 (low) / 0.4214 ± 5.51e-03 (high) | 0.8892 ± 0.0112 | 0.0696 |
| brk_all R1 | beta | Q_b = 1.765 ± 0.0144 | -0.2521 ± 5.71e-03 (low) / 0.1808 ± 3.83e-03 (high) | 0.4955 ± 7.81e-03 | 0.0696 |
| brk_all R1 | break | log10 M_b = 13.473 ± 5.01e-03 | | | |
| free R1 (not converged, not usable) | rho0 | 81.028 ± 13.181 | -0.3148 ± 0.0414 | 3.767 ± 0.1414 | 0.0971 (n.c.) |
| free R1 (not converged, not usable) | x_c | 10.000 ± 1.805 (at bound) | -1.627 ± 0.0785 | -0.8170 ± 0.1002 | 0.0971 (n.c.) |
| free R1 (not converged, not usable) | alpha | 0.3441 ± 0.0155 | 7.61e-03 ± 1.87e-03 | 0.1040 ± 8.19e-03 | 0.0971 (n.c.) |
| free R1 (not converged, not usable) | beta | 2.716 ± 0.1569 | -0.6037 ± 0.0191 | 0.4882 ± 0.0422 | 0.0971 (n.c.) |
| free R1 (not converged, not usable) | gamma | -0.8046 ± 0.0289 | 0.4474 ± 0.0140 | -0.5285 ± 0.0301 | 0.0971 (n.c.) |
| brk_amp R2 | rho0 | 7172 ± 288.6 | -0.4125 ± 7.38e-03 | -0.7424 ± 0.0184 | 0.0991 |
| brk_amp R2 | alpha | 0.4491 ± 5.68e-03 | 0.3047 ± 2.90e-03 | 0.8065 ± 8.26e-03 | 0.0991 |
| brk_amp R2 | beta | 2.697 ± 0.0112 | 0.0865 ± 1.62e-03 | 0.3912 ± 5.33e-03 | 0.0991 |
| brk_amp R2 | break | log10 M_b = 13.293 ± 6.31e-03, alpha_break = -0.4371 ± 9.77e-03 | | | |
| brk_all R2 | rho0 | Q_b = 1.39e+04 ± 527.8 | -0.7549 ± 0.0168 (low) / -0.4223 ± 7.17e-03 (high) | -0.7220 ± 0.0144 | 0.0714 |
| brk_all R2 | alpha | Q_b = 0.2437 ± 2.30e-03 | 0.0435 ± 5.72e-03 (low) / 0.3878 ± 3.10e-03 (high) | 0.8001 ± 6.25e-03 | 0.0714 |
| brk_all R2 | beta | Q_b = 2.050 ± 9.10e-03 | -0.1465 ± 4.19e-03 (low) / 0.1637 ± 1.82e-03 (high) | 0.4027 ± 3.94e-03 | 0.0714 |
| brk_all R2 | break | log10 M_b = 13.379 ± 5.78e-03 | | | |
| free R2 | rho0 | 247.2 ± 5.195 | 0.6780 ± 7.95e-03 | 0.4641 ± 0.0189 | 0.0608 |
| free R2 | x_c | 0.4635 ± 5.92e-03 | -0.2484 ± 4.32e-03 | -0.1020 ± 0.0107 | 0.0608 |
| free R2 | alpha | 3.777 ± 0.1881 | 1.100 ± 0.0275 | 1.390 ± 0.0403 | 0.0608 |
| free R2 | beta | 1.901 ± 8.86e-03 | 0.0997 ± 1.48e-03 | 0.2288 ± 4.99e-03 | 0.0608 |
| free R2 | gamma | -0.8342 ± 6.29e-03 | -0.1366 ± 2.00e-03 | 0.3476 ± 4.14e-03 | 0.0608 |
| brk_amp R3 | rho0 | 7227 ± 256.9 | -0.3989 ± 6.98e-03 | -0.6348 ± 0.0173 | 0.1055 |
| brk_amp R3 | alpha | 0.4803 ± 4.99e-03 | 0.2863 ± 2.44e-03 | 0.6589 ± 6.43e-03 | 0.1055 |
| brk_amp R3 | beta | 2.898 ± 8.53e-03 | 0.0793 ± 1.12e-03 | 0.2827 ± 3.71e-03 | 0.1055 |
| brk_amp R3 | break | log10 M_b = 13.331 ± 8.33e-03, alpha_break = -0.3673 ± 9.31e-03 | | | |
| brk_all R3 | rho0 | Q_b = 1.28e+04 ± 482.5 | -0.9947 ± 0.0257 (low) / -0.3652 ± 6.95e-03 (high) | -0.5859 ± 0.0153 | 0.0865 |
| brk_all R3 | alpha | Q_b = 0.2773 ± 2.46e-03 | 0.1331 ± 6.07e-03 (low) / 0.3261 ± 2.58e-03 (high) | 0.6531 ± 5.49e-03 | 0.0865 |
| brk_all R3 | beta | Q_b = 2.310 ± 8.62e-03 | -0.0947 ± 3.94e-03 (low) / 0.1253 ± 1.32e-03 (high) | 0.2944 ± 3.09e-03 | 0.0865 |
| brk_all R3 | break | log10 M_b = 13.293 ± 5.25e-03 | | | |
| free R3 (recommended kSZ/DM) | rho0 | 169.6 ± 4.667 | 0.8190 ± 0.0137 | 0.6503 ± 0.0192 | 0.0652 |
| free R3 (recommended kSZ/DM) | x_c | 0.6305 ± 0.0104 | -0.3427 ± 8.48e-03 | -0.3094 ± 0.0132 | 0.0652 |
| free R3 (recommended kSZ/DM) | alpha | 3.652 ± 0.1564 | 0.6216 ± 0.0187 | 1.126 ± 0.0492 | 0.0652 |
| free R3 (recommended kSZ/DM) | beta | 2.373 ± 0.0180 | 0.0458 ± 3.74e-03 | 0.0386 ± 8.17e-03 | 0.0652 |
| free R3 (recommended kSZ/DM) | gamma | -0.8509 ± 7.00e-03 | -0.1042 ± 1.90e-03 | 0.3870 ± 5.45e-03 | 0.0652 |

Fixed (not fitted) parameters are the published values: B12 alpha = 1 and gamma = -0.3 (gamma is freed in 'free'); B16 x_c = 0.5 and gamma = -0.2 (both freed in 'free'). The Gcore check fits (central-galaxy gas included; sigma_d 0.1567 / 0.1644 / 0.1685) are in results/k3_fits.json and k_params.json.

**Fit quality** (sigma_d, ln per aperture point; ΔBIC at the orig fit's sigma_d; n.c. = not converged):

| field | range | points / cells | published sigma_d | orig | brk_amp | brk_all | free | ΔBIC vs orig (brk_amp / brk_all / free) |
|---|---|---|---|---|---|---|---|---|
| P | R1 | 2335 / 597 | 0.4556 | 0.0886 | 0.0801 | 0.0754 | 0.0540 | -377 / -560 / -1364 |
| P | R2 | 4193 / 597 | 0.3718 | 0.0852 | 0.0705 | 0.0699 | 0.0587 | -1211 / -1246 / -2067 |
| P | R3 | 4790 / 597 | 0.3542 | 0.0852 | 0.0735 | 0.0727 | 0.0683 | -1127 / -1187 / -1577 |
| G | R1 | 3313 / 597 | 0.4521 | 0.1476 | 0.0961 | 0.0696 | 0.0971 (n.c.) | -1838 / -2476 / -1798 (n.c.) |
| G | R2 | 5430 / 597 | 0.3985 | 0.1269 | 0.0991 | 0.0714 | 0.0608 | -2042 / -3597 / -4026 |
| G | R3 | 6382 / 597 | 0.3945 | 0.1258 | 0.1055 | 0.0865 | 0.0652 | -1833 / -3272 / -4466 |
## Correction (2026-10-16, after the final verification)

For the pressure mass-break law (brk_amp), the tabulated P0 coefficient A is XGPaint's `P0_amp`, the amplitude law above the break. It is not P0 at 10^14 Msun: because M_b = 10^14.099 / 10^14.029 / 10^14.028 (R1/R2/R3) lies above the 10^14 Msun pivot, P0(10^14 Msun, z = 0) = A (10^14 Msun / M_b)^alpha_break = 4.686 (R1), 3.569 (R2) and 3.728 (R3, the recommended tSZ law). The tabulated values, the XGPaint snippet and results/k_params.json are correct as P0_amp. The refitted parameters require the validated XGPaint checkout (branch cluster, commit 1660e08); the older copy julia_env/dev/XGPaint.jl hard-codes the published laws and normalises the B16 density differently.
