# Test A: overlapping structures, contamination vs selection (FLAMINGO L1_m9, z = 0)

Plan: `/root/.claude/plans/there-has-been-multiple-indexed-lantern.md`, section "Test A".

**Inputs.** The step-1.2 particle-pass products of `overlap_covariance_L1_m9_0077_20261012`:
- 4,588 centrals: the 4,375 radii234 haloes plus 213 added annulus-only haloes;
- 35 analysis shells to 4 R200c;
- owner x phase accumulators, excision zones, the scaled-distance partition and per-overlapper bound gas.

**Run.** The full run was on STRW maas (7,618 s, 4 workers, nice 19, peak 7.3 GiB). Figures and checks were made locally.
All 21 cluster checks and 5 local checks pass (`verification.json`).

**Conventions.**
- Pressure is the geometric estimator sum(y)/V_shell with recently AGN-heated gas removed; all gas is the paired control.
- Electrons (B16) feed DM and kSZ.
- Stacks are inclusion-weighted means.
- Intervals are 16-84% percentiles of 1,000 stratified halo bootstraps (500 refitted).
- The reference fit is GLS objective (c) to 2 R200c; objective (a) is shown for continuity.
- Observables use a hard cut at 4 R200c.

**Overlay definition.** A satellite or neighbouring central with mu = M_j/M200c >= 0.1 whose sphere reaches inside 2 R200c.
- CONT-1R: the sphere reaches inside 1 R200c.
- CONT-annulus: it reaches only the 1-2R annulus.

## 1. Plain answers

### Inner profile (< 1 R200c): different, but because of selection and mass normalisation, not contamination

- **The deficit.** CONT is low compared with CLEAN:
  - pressure: Y(<0.5R) -9 to -13%, Y(<1R) -1.5 to -5.6%;
  - electrons: Y(<0.5R) -6 to -11%;
  - B12 inner amplitude P0 xc^0.3: -17 to -27%;
  - CONT-1R: Y(<0.5R) -11 to -18%.
- **It is not contamination.** Contamination can only add gas. Removing the overlappers' bound gas moves Y(<0.5R) by 0 to
  -1.5%, the wrong direction.
- **Localisation.** Inside 0.71R the whole deficit sits in the shells that span a flagged overlapper (near -8.9%
  [-10.2, -7.8]). The other shells match CLEAN (far 0.0% [-1.1, +1.0]).
- **Built-in control.** CONT-annulus haloes have no overlapper inside 1R. Inside 0.71R they match CLEAN within 2%, consistent
  with zero, for 13.7-14.9. The edge units differ: +5.4% at 13.5-13.7 and -5.1% at >= 14.9.
- **Host-only mass.** 749 haloes host a flagged overlapper centred inside R200c (median M_host/M200c = 0.84). In host units
  the deficit disappears: Y(<0.5R) is -0.3 to +3.4% of CLEAN.
- **Interpretation.** These are merging systems whose M200c includes a massive satellite. P200 and R200c are set by the
  combined mass, so in M200c units the core looks under-pressured.

### 1-2 R200c: real, localised contamination, plus a smaller selection part

- **Near vs far shells.** Near shells are +49% [44, 53] in pressure and +21% [19, 22] in electrons; far shells are -2.1% and
  -4.5%.
- **CONT-annulus** Y(<2R) is +10 to +17% (+1% at >= 14.9).
- **Injection.** Overlapper excess injected into CLEAN haloes (bound gas plus unbound gas within 2 R_j above the local mean)
  accounts for +19% of the measured +25% in pressure and +11% of +12% in electrons.
- **Beyond 2R.**
  - Near shells at 2-4R: +118% (P) and +57% (E).
  - Far shells at 2-4R: pressure is consistent with zero, but electrons show +9% [7, 12], an environment (selection)
    effect.
  - In the 0.71-1R band there is a non-localised +7% (far shells, pressure).

### Does subtraction recover CLEAN? No method does, in either flavour (part of the difference is selection)

| Method | Flavour (a): remove all flagged overlappers | Flavour (b): keep own satellites |
|---|---|---|
| Bound gas | removes 2-7% of Y(<4R); the inner deficit stays; the 1-2R near excess goes from +49% to +38% | <= 0.3% change in Y(<4R) (only 92 of 1,650 CONT haloes have a flagged neighbour) |
| Template | same as bound (a) to within 0.3%; truncating neighbours at R or 2R makes no difference | <= 0.5% |
| Excision kappa = 1 (2) | removes 6-14% (10-15%) of Y(<4R) relative to raw on the same haloes; overshoots CLEAN, Y(<1R) 6-16% low | <= 0.6% |
| Host-only units + bound (a) | inner part recovered (Y(<0.5R) within +-3%); Y(<1R) +4 to +12% and Y(<2R) +8 to +22% above CLEAN | - |

### Method validity

- **Placebo excision on CLEAN haloes.** Isotropic spheres bias the profile by <= 0.45%. Spheres pointed at the nearest
  massive neighbour bias it by -0.3 to -0.7% at kappa = 1, and by -2.1% (1-2R) and -3.9% (2-4R) at kappa = 2.
- **Injection-recovery.**
  - Excision at kappa = 1 leaves <= 0.9% (1-2R) and 2.7% (2-4R); kappa = 2 leaves <= 0.4%.
  - Bound-gas removal and templates leave +3.5 to +5% (P) and +1.3 to +3% (E).
  - kappa = 0.5 leaves 3-14%.

### Observables, relative to painting with ALL (2R fits)

- **CLEAN:**
  - yy +0.7 / +2.1 / +4.1 / +7.9% at ell = 100 / 1000 / 3000 / 8000;
  - kSZ 1h -4.4 / -2.6 / +0.7% at ell = 100 / 1000 / 3000;
  - y x DM +0.1% at ell = 1000;
  - mean DM -2.2%; DM Q16 / Q50 / Q99.9 -3.3 / -2.8 / +1.1%.
- **CONT-raw:** yy -1.5 / -5.7 / -10.3 / -17.2%; mean DM +5.0%.
- **Subtracted CONT:** bound (a) yy -8 to -17%; excised kappa = 1 yy about -20%.
- **A2 with 2R fits:** <= 0.2%.
- **A2 with the measured stacks to 4R:**
  - A2-ii: mean DM -3.3%, Q16 -5.8%, Q50 -4.0%, yy -1.5% at ell = 100, kSZ 1h -6.4% at ell = 100;
  - A2-i: about a third of that.
- **Checks:** the mixture CLEAN + CONT reproduces ALL to <= 0.6%; keeping or removing recent-AGN gas changes observables by
  <= 2e-5.

### Painting target: ALL haloes with A2-ii ownership

- **Rule.** Satellites stay with their host; gas that is closer in r/R200c to another painted distinct halo goes to that halo.
- **Inside 2R it equals ALL:** |dP0| <= 0.7%, |dxc| <= 1.4%, |dbeta| <= 1.0% below 14.9 (<= 2.2% at >= 14.9), so the 2R
  calibration is unchanged.
- **Beyond 2R** it removes the 2.4-7.4% of Y(<4R) and 5.1-7.6% of N_e(<4R) that belong to neighbouring painted haloes and
  would otherwise be painted twice.
- **4R fits become adequate.**
  - Pressure sigma_d drops from 8.4-14.3% to 0.2-3.9%.
  - Electrons drop from 8.2-16.3% to 3.4-8.1% for >= 13.9 (worse at 13.5-13.9: 8.2-11.2% vs 6.4-7.7%).
- **Do not calibrate on CLEAN or on subtracted CONT.** HalfDome never paints satellites separately, and CLEAN is a selected,
  more relaxed subset.

## 2. Sample

| unit (log M200c) | N | CLEAN | CONT-1R | CONT-annulus | population CONT [%] | CONT with flagged neighbour |
|---|---|---|---|---|---|---|
| 13.5-13.7 | 326 | 204 | 72 | 50 | 34.1 | 5 |
| 13.7-13.9 | 324 | 211 | 63 | 50 | 32.6 | 2 |
| 13.9-14.1 | 319 | 191 | 78 | 50 | 33.8 | 5 |
| 14.1-14.3 | 327 | 201 | 76 | 50 | 32.5 | 5 |
| 14.3-14.5 | 317 | 197 | 70 | 50 | 32.6 | 1 |
| 14.5-14.7 | 895 | 631 | 172 | 92 | 29.5 | 17 |
| 14.7-14.9 | 326 | 237 | 61 | 28 | 27.3 | 1 |
| >= 14.9 (pooled, log M trend) | 154 | 104 | 33 | 17 | 32.5 | 2 |
| 15.1-15.5 (exploratory) | 27 | 19 | 5 | 3 | 29.6 | 0 |

- Main definition: 1,650 CONT (1,013 reach inside 1R, 637 annulus only). mu >= 0.03 gives 3,465 CONT; mu >= 0.3 gives 419.
- Flagged overlappers: 1,884 own satellites, 88 neighbouring centrals, 7 satellites of neighbours. Every overlapper that
  reaches inside 1R is an own satellite.
- Inclusion weights reproduce the census population counts to 7e-16.

## 3. Method

- **Grid and units.**
  - 35-shell sub-grid of x200, with the fit radii 1, 1.34, 1.5, 1.75, 2, 3 and 4 as exact edges.
  - 0.2-dex units from 13.5 to 14.9; >= 14.9 is pooled with a log M trend (published mass slopes, parameters quoted at
    10^15 Msun).
  - Resolution masks are per unit and shared by every sample and method.
- **Fits.** B12 (pressure) and B16 (electrons, gas) on ln(mean).
  - Objective (c): bootstrap Cov[ln mean], shrunk towards the leave-one-out pooled correlation of ALL, with sigma_d solved
    so that chi2/nu = 1.
  - Objective (a): diagonal max(SD/mean, 5%).
- **Bootstrap.** Draws are made within the (bin, class) design strata, so methods on the same haloes are paired and CLEAN vs
  CONT are independent.
- **Subtraction methods.**
  - Bound: gas bound to the flagged overlappers is removed; a neighbouring central's satellites count with it.
  - Excision: zones with s < kappa are removed and the shell is divided by its retained volume; shells with < 50% retained
    are dropped. A fixed halo set S(kappa) is used per comparison.
  - Template: the exact offset shell average of the CLEAN 2R fit. Neighbouring centrals are truncated at R200c (2R200c as a
    variant); satellites are scaled to their measured bound gas.
  - Host-only renormalisation: profiles are remapped to r/R_host and P/P200(M_host) and binned by M_host.
- **A2.**
  - A2-i: gas bound to other painted haloes (M200c >= 6.6e12) is removed.
  - A2-ii: scaled-distance partition.
- **Observables.** `kernels/engine.py` with three profile conventions: fit2 (reference), fit4, and the measured stacks.
  - Population mixtures: 1h terms and the DM-PDF intensity are summed over lightcones scaled by the census CONT fraction; 2h
    terms use the mixture mean profile.
  - Observable intervals are linearised through the kernels Jacobians (good to about 10-30%).

## 4. Contamination vs selection (figs 1, 4, 5, 12)

Stacked integrals relative to CLEAN, pressure, in % ("cylinder" = projected aperture b < 2R through the 4R sphere):

| sample / CLEAN | unit | Y(<0.5R) | Y(<1R) | Y(<2R) | Y(<4R) | cylinder b<2R |
|---|---|---|---|---|---|---|
| CONT-raw | 13.9-14.1 | -12.6 [-14.7, -10.3] | -2.9 [-4.2, -1.6] | +6.2 [+4.3, +8.3] | +11.2 [+7.8, +14.6] | +8.7 [+6.3, +11.2] |
| CONT-raw | 14.5-14.7 | -10.4 [-11.7, -9.2] | -3.1 [-3.7, -2.4] | +2.9 [+2.1, +3.8] | +11.8 [+9.9, +13.8] | +6.2 [+5.1, +7.4] |
| CONT-raw | >= 14.9 | -9.6 [-13.1, -6.2] | -5.6 [-7.3, -3.8] | -2.8 [-4.3, -0.9] | +2.3 [-0.3, +5.2] | -0.6 [-2.6, +1.6] |
| CONT-1R | 14.5-14.7 | -15.3 [-17.0, -13.8] | -5.8 [-6.6, -5.0] | -0.7 [-1.8, +0.5] | +3.7 [+1.6, +5.7] | +0.7 [-0.5, +2.0] |
| CONT-annulus | 14.5-14.7 | -1.3 [-3.1, +0.6] | +2.0 [+1.0, +2.9] | +9.7 [+8.4, +11.0] | +27.1 [+23.4, +31.0] | +16.5 [+14.6, +18.5] |
| CONT, host units | 14.5-14.7 | +0.5 [-1.5, +2.2] | +8.4 [+7.3, +9.5] | +16.0 [+15.0, +17.1] | +25.4 [+23.2, +27.6] | +19.6 [+18.3, +20.9] |
| ALL | 14.5-14.7 | -3.1 [-3.5, -2.7] | -0.9 [-1.1, -0.7] | +0.9 [+0.6, +1.1] | +3.5 [+2.9, +4.1] | +1.8 [+1.5, +2.2] |

Localisation test (pressure, supported units combined, Y-weighted per band), in %:

| | < 0.71R | 0.71-1R | 1-2R | 2-4R |
|---|---|---|---|---|
| CONT-raw, near | -8.9 [-10.2, -7.8] | +10.4 [+8.3, +12.3] | +48.6 [+44.4, +52.7] | +117.7 [+101.9, +133.2] |
| CONT-raw, far | -0.0 [-1.1, +1.0] | +7.0 [+5.4, +8.7] | -2.1 [-4.8, +0.4] | -4.7 [-13.9, +5.8] |
| bound (a), near | -9.6 | +8.5 | +38.3 | +76.3 |
| excised (a) kappa1, near | -22.5 | -2.6 | +28.2 | +69.9 |
| CLEAN null, near / far | +0.3 / -0.2 | -0.0 / +0.0 | -1.3 / +1.5 | +5.9 / -2.2 |

- Electrons, near / far: < 0.71R -9.2 / -0.7; 1-2R +20.8 / -4.5; 2-4R +57.5 / +9.4.
- CLEAN null: |z| <= 1.4.

**Threshold sensitivity** (CONT / CLEAN, pressure).
- mu >= 0.03: Y(<0.5R) -12 to -15%. With 3,465 CONT, CLEAN@0.03 is a relaxed minority; painting with it would raise yy by
  +8 to +26%.
- mu >= 0.3: Y(<2R) +4 to +33%, Y(<4R) +26 to +46%.

**B12 parameter shifts at 2R, GLS** (fig. 3):
- ALL vs CLEAN: P0 -4 to -9% (+0.3% at >= 14.9); inner amplitude -2 to -9%.
- CONT-raw: P0 -15 to -28%; inner amplitude -17 to -27%.
- CONT-1R: P0 -34 to -46%, xc +25 to +55%; inner amplitude -30 to -38%.
- CONT-annulus: inner amplitude -8 to +13%, with P0 up while xc and beta fall along the degeneracy.
- Bound (a) in host units: inner amplitude -6 to -21%.
- Objective (a): ALL vs CLEAN P0 -5 to -8%; CONT -20 to -29%.

## 5. Subtraction methods and validation (figs 2, 6)

Each method relative to raw CONT on the same haloes. Each cell gives Y(<0.5R) / Y(<2R) / Y(<4R) in %.

| Method | 13.9-14.1 | 14.5-14.7 | >= 14.9 |
|---|---|---|---|
| bound (a) | -0.0 / -2.9 / -4.2 | -0.4 / -3.5 / -5.4 | -1.5 / -3.3 / -5.3 |
| template (a) | -0.1 / -2.9 / -4.2 | -0.6 / -3.5 / -5.4 | -1.7 / -3.3 / -5.6 |
| excised (a) kappa1, vs raw on S(1) | -4.0 / -8.7 / -8.9 | -5.3 / -9.2 / -10.0 | -6.7 / -9.2 / -10.9 |
| excised (a) kappa2, vs raw on S(2) | -0.8 / -11.6 / -13.0 | -0.6 / -9.6 / -12.0 | +0.2 / -7.8 / -10.9 |
| any flavour (b) method | <= 0.6 | <= 0.6 | <= 0.6 |

- S(kappa) keeps 92% (kappa 0.5), 75% (kappa 1) and 49% (kappa 2) of CONT halo-shells inside 0.2R.
- Excised kappa1 vs CLEAN: Y(<1R) -6 to -16%; Y(<2R) -13 to +3%.

Placebo excision on CLEAN haloes (%):

| | < 0.71R | 0.71-1R | 1-2R | 2-4R |
|---|---|---|---|---|
| isotropic kappa 0.5 / 1 / 2 | 0.0 / -0.1 / -0.0 | 0.0 / -0.1 / -0.3 | 0.0 / -0.0 / -0.4 | 0.0 / 0.0 / 0.0 |
| neighbour direction kappa 0.5 / 1 / 2 | -0.1 / -0.3 / -0.1 | -0.1 / -0.4 / -0.9 | -0.1 / -0.4 / -2.1 | -0.1 / -0.7 / -3.9 |

Injection-recovery, pressure (%):

| | < 0.71R | 0.71-1R | 1-2R | 2-4R |
|---|---|---|---|---|
| measured CONT-raw / CLEAN | -5.8 | +9.0 | +24.8 | +39.7 |
| injected excess, 1 R_j / 2 R_j | +4.7 / +5.9 | +5.9 / +9.7 | +12.1 / +18.8 | +21.9 / +31.8 |
| bound (a) residual | +4.0 | +3.5 | +3.9 | +5.0 |
| template (a) residual | +3.9 | +3.6 | +4.0 | +5.0 |
| excision kappa 0.5 / 1 / 2 residual | +2.6 / -0.1 / -0.0 | +3.9 / -0.0 / -0.3 | +7.4 / +0.9 / -0.4 | +13.5 / +2.7 / +0.2 |

Electrons, in the same four bands:
- measured: -4.1 / +3.5 / +11.9 / +23.0;
- injected (2 R_j): +5.0 / +6.6 / +10.7 / +14.5;
- bound residual: +3.0 / +1.8 / +1.4 / +1.3;
- excision kappa 2 residual: <= 0.3.

## 6. Observables (fig. 7; relative to ALL, %; brackets: linearised bootstrap half-range)

| variant | yy 100 | yy 1000 | yy 3000 | yy 8000 | kSZ 1h 1000 | y x DM 1000 | mean DM | DM var | Q16 | Q50 | Q99.9 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| CLEAN (fit2) | +0.7 (0.6) | +2.1 (0.5) | +4.1 (0.5) | +7.9 (0.6) | -2.6 | +0.1 | -2.2 (0.2) | -0.3 | -3.3 | -2.8 | +1.1 |
| CONT-raw | -1.5 | -5.7 | -10.3 | -17.2 | +6.8 | -0.1 | +5.0 | +2.1 | +7.3 | +6.2 | -1.6 |
| CONT bound (a) | -8.3 | -10.2 | -12.1 | -16.7 | +0.2 | -5.4 | +1.6 | -2.5 | +3.2 | +2.5 | -3.4 |
| CONT template (a) | -8.5 | -10.4 | -12.3 | -16.8 | -0.3 | -5.9 | +1.5 | -2.9 | +3.1 | +2.3 | -3.8 |
| CONT excised (a) kappa1 | -19.5 | -21.1 | -21.9 | -21.1 | -4.2 | -14.2 | +0.7 | -6.7 | +2.9 | +1.9 | -6.2 |
| A2-i / A2-ii (fit2) | <= 0.2 | <= 0.1 | <= 0.1 | 0.0 | <= 0.1 | <= 0.1 | <= 0.1 | <= 0.1 | <= 0.1 | <= 0.1 | 0.0 |
| mix CLEAN + CONT-raw | -0.6 | -0.3 | -0.2 | -0.0 | +0.5 | +0.1 | +0.2 | +0.5 | +0.2 | +0.2 | +0.4 |
| mix CLEAN + bound (a) | -2.9 | -1.6 | -0.8 | +0.1 | -1.6 | -1.5 | -0.9 | -1.0 | -1.2 | -1.0 | -0.2 |
| A2-i (measured stack) | -0.6 | -0.1 | +0.0 | -0.0 | -0.7 | -0.2 | -1.0 | -0.5 | -1.7 | -1.2 | -0.3 |
| A2-ii (measured stack) | -1.5 | -0.3 | +0.0 | -0.0 | -2.6 | -0.6 | -3.3 | -1.5 | -5.8 | -4.0 | -0.8 |
| ALL, AGN gas kept | <= 2e-5 relative | | | | | | | | | | |

- ALL measured stack vs ALL 2R fit: yy 0.0 / -1.4 / -1.7 / -1.6%; mean DM -1.1%.
- ALL 4R fit vs ALL 2R fit: yy -2.3 / -4.1 / -4.6 / +3.6%.
- The extra 1h power from splitting the population into two mean profiles is 0.07-0.5%.

## 7. A2 (fig. 8)

Removed fraction of the ALL integral (%):

| unit | P A2-i <4R | P A2-ii <2R | P A2-ii <4R | E A2-i <4R | E A2-ii <4R |
|---|---|---|---|---|---|
| 13.5-13.7 | 1.42 | 0.08 | 7.39 | 1.07 | 5.08 |
| 13.9-14.1 | 1.91 | 0.03 | 4.74 | 1.74 | 5.67 |
| 14.5-14.7 | 1.69 | 0.04 | 3.45 | 2.46 | 6.87 |
| >= 14.9 | 1.14 | 0.05 | 2.36 | 2.85 | 7.60 |

A2-i removes nothing inside 2R (<= 0.01%). The A2-i owner codes are unambiguous (0 mixed cells).

## 8. AGN paired table (fig. 9)

Recently AGN-heated gas removed minus all gas, relative, in %:

| field | sample | unit | dY(<0.1R) | dY(<1R) | dY(<4R) | d ln P0 2R | max resolved shell | shell 0.025-0.029R |
|---|---|---|---|---|---|---|---|---|
| P | ALL | 13.5-13.7 | -3.6 [-4.4, -2.8] | -0.04 | -0.03 | +0.02 | 0.05 | -7.5 |
| P | ALL | 14.5-14.7 | -0.09 | -0.00 | -0.00 | -0.00 | 0.01 | -0.1 |
| P | CONT | 13.5-13.7 | -3.1 | -0.05 | -0.03 | -0.01 | 0.16 | 0.0 |
| E | ALL | 13.5-13.7 | -0.6 | -0.01 | -0.00 | +0.00 | 0.00 | -1.4 |

All 2R parameter shifts are <= 0.03%; observables change by <= 2e-5; only unresolved cores change, by up to -9%.

## 9. Fit radius (fig. 10; 14.5-14.7, pressure, GLS)

| | P0 | xc | beta | chi2/nu at sigma_d = 0 | sigma_d |
|---|---|---|---|---|---|
| ALL, 1R to 2R | 15.5 to 16.1 | 0.75 to 0.80 | 5.6 to 5.8 | 1.3-3.8 | <= 0.7% |
| ALL, 3R / 4R | 18.7 / 25.7 | 0.61 / 0.42 | 5.0 / 4.2 | 11 / 18 | 5.8 / 14.3% |
| A2-ii, 4R | 18.2 | 0.63 | 5.1 | - | 3.9% |
| CLEAN, 2R | 17.3 | 0.75 | 5.7 | - | - |

## 10. Verification (all passed)

**Cluster checks:**
- census classes; exact sub-grid;
- excision zone sums (5e-16); finite-iff-retained >= 50%;
- host remap identity and conservation (6e-16);
- weights vs population (7e-16); bootstrap strata;
- moments vs library (4e-16);
- mixture identity (4e-15);
- trend family (Jacobian 8e-10, grouping 4e-5);
- general fitter vs legacy objectives (1e-10);
- sigma_d gives chi2/nu = 1 (5e-10, 5,573 fits);
- 1,604/1,604 reference fits accepted;
- template vs Monte Carlo (|z| <= 2.7); template vs library (0);
- placebo volume fractions (0.0025);
- CLEAN null (|z| <= 1.4);
- engine mixture identity (2e-15); engine fits accepted;
- linearisation (informational).

**Local checks:**
- `results/SHA256SUMS` reproduced.
- Inputs match `CLUSTER_RUN.json`, except the fixed `geometry.py` (`c3a68811...`, equal to the local file).
- Templates recomputed: 0 non-finite; the 48 zero-amplitude satellites contribute exactly zero.
- AGN row <= 2e-5; mixture identity holds.
- 16 figures written as PNG and SVG.

## 11. Caveats

- **Injection inner bands are upper limits.** In 27% of (CLEAN halo, shell) entries the partner's s < 2 zones cover the whole
  shell, so no local mean exists and all "rest" gas there counts as excess. These are mostly inner shells, so the < 0.71R
  injected excess and the bound/template residuals there are upper limits. Excision at kappa >= 1 and the 1-4R results are
  unaffected.
- **Excision geometry.** Spheres scale with the overlapper's pseudo-R200c and cover the cores around central satellites;
  S(2) keeps only 49% of CONT halo-shells inside 0.2R.
- **Templates are model-limited.** Stripped satellites are more compact than the template; amplitude factors span 6e-4 to 1.3.
- **Host-only mass** is the HBT bound mass of overlappers centred inside R200c, without partial-sphere corrections.
- **Pooled >= 14.9 unit.** Uses the published mass slopes; for excised variants the trend model averages over all haloes.
- **Engine.** Intervals are linearised (good to about 10-30%). The engine applies the frozen z-response, support
  13.5-15.388 and the published fallback.
- **Sample size.** mu >= 0.3 sub-samples are small; 15.1-15.5 is exploratory.
- **Environment.** maas ran scipy 1.15.2 (local 1.17.1).

## 12. Files and runtimes

- **Code:** `run_test_a.py` (steps s1-s8, `common.py`), `make_figures.py`, `report_tables.py`, `local_checks.py`,
  `CLUSTER_RUN.json`.
- **Outputs:** `results/` (maas tables and checks), `work/logs/`, `figures/` (16 PNG + SVG), `results_smoke/` and
  `work_smoke/logs/` (local smoke test), `verification.json`, `SHA256SUMS`.
- **maas step times:** s1 33 s, s2a 5,647 s, s3 63 s, s2b 816 s, s4 27 s, s5 967 s, s6 57 s, s8 4 s.
- **Local times:** smoke 130 s, figures 16 s, checks 19 s.
- **Reproduce:**
  1. `STUDY_ROOT=<study> python -B run_test_a.py --full --workers 4`
  2. `python -B make_figures.py`
  3. `python -B local_checks.py`

(Written by the coordinator from the agent's hand-back; the harness blocked the agent from writing `.md` files.)
