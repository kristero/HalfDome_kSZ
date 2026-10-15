# Lee22 boundary bug: verdict (2026-10-08)

Question: "There is another boundary bug: halo radius/L_box^2. Check if that makes sense for Lee22. Check if that changes anything."

## Short answer

1. **The "R/L_box^2" reading is wrong.** The only place where a radius meets L_box^2 is in the cell search: `Rsq_normalized = Rsq/(acell*acell)`, with `acell = L/64`. That is the squared halo radius in cell units, (64 R/L)^2. It is compared with squared closest-point distances, which are also in cell units, so it is correct. No radius is divided by L_box^2 anywhere in group_particles. I grepped every use of `Bsize` and `acell` at c3fe9b8.
2. **There is a real boundary bug in the same function, and Lee22's code path had it.** `#define PER(x) ((x>=Ncells_side) ? 1 : (x<0) ? -1 : 0)` (detail/workspace_sorting.hpp:246) compares an `int` with a `size_t`. A negative cell index is converted to 2^64+x, so PER returns +1 instead of -1. Gas reached across one of the three low box faces (x, y or z = 0) is then placed about 2 L_box away and dropped. Nothing is double counted or put in the wrong bin. Gas across a high face is handled correctly. The default build used at c3fe9b8 (2021-06-25, compile.sh) has this bug. The 2024-04-08 hotfix 5791098 changed the default distance to the plain periodic one. The macro is still unchanged at HEAD.
3. **The bias scales linearly in R/L, not as R/L^2.** For one halo at distance d from a low face, the shell at radius r > d loses the fraction (1 - d/r)/2. Averaged over halo positions, a stacked mean profile is low by (3/4) x R200c/L_box. Edges and corners add only (R/L)^2 and (R/L)^3 terms.
4. **It changes none of our Lee22 corrections and no conclusion in sections 1 and 2.** Every painted quantity moves by 0.1-0.5 % at most, and the C_ell by at most 0.9 %. That is below the quoted precision or the statistical errors. I recommend no correction. The only edits I propose are documentary (see below).

## What I re-derived myself (measured)

| check | result | file |
|---|---|---|
| PER macro, standalone, GCC 11.4, compile.sh flags | xx = -3..-1 gives +1 (should be -1); 0..63 gives 0; 64, 65 gives +1. 3 `-Wsign-compare` warnings. Group at 0.3 acell, particle in cell 63: hint distance 407,437 ckpc/h, true distance -2,563 ckpc/h | verdict/per_test_run.log, per_test_compile.log (code: boundary_check/verdict/per_test.cpp) |
| Code read: c3fe9b8 and HEAD | Default build (no NAIVE, no EARLY_RETURN in compile.sh or compile_parttype.sh) uses the hint distance at c3fe9b8. At HEAD the default build uses the plain distance, and the EARLY_RETURN branch still uses the hint. PER is identical at HEAD | git show c3fe9b8/HEAD:detail/{prt_loop,workspace_sorting}.hpp |
| Re-read of the kept C++ outputs (real c3fe9b8 y_prof/ne_prof on CAMELS CV boxes, actual faces), with my own binning and stacking code | z=0, logM 13.0-14.2 (21 halos, 5 affected): buggy/plain n_e 0.9968 / 0.9835 / 0.9682 / 0.9468 and P_e 0.9957 / 0.9834 / 0.9686 / 0.9603 at x = 0.5 / 1 / 1.34 / 2.5. z=2, logM 12.5-13: n_e 0.889, P_e 0.910 at 1.34. Identical to impact_summary.txt | verdict/recheck_cpp_stacks.{log,json} |
| Monte Carlo of the drop rule for uniform halo centres (4e6 draws per radius) | dropped fraction 0.001117±0.000017, 0.003663±0.000030, 0.010908±0.000052 at r = 300, 1000, 3000 ckpc/h, against 3r/(4L) = 0.001098, 0.003659, 0.010976 | verdict/propagate_fit_bias.txt |
| y x DM chi^2 with the stored annulus models and the real Takahashi+25 jackknife covariance | Stored chi^2 reproduced exactly (e.g. ACT, full corrected pair: 23.041). Scaling the model up by 0.6 %, the bound on undoing the bug, changes chi^2 by +0.228 (ACT, full corrected pair), +0.105 (ACT, hybrid), +0.063 (ACT, B12 y x corrected DM), +0.042 (ACT, calibrated range), and +0.004 to +0.058 (Planck) | verdict/cross_chi2_scaling.json (code: boundary_check/verdict/cross_chi2_scaling.py) |

The three investigations measured the rest of the chain, and I checked their files. The real 2021 y_prof.cpp drops exactly the pairs that cross a low face: 1,183,670 synthetic pairs (e2e) and 2,006,196 pairs on CAMELS. The CAMELS losses are identical per group to the Python replica, over 52 runs. The plain, signed-PER and NAIVE builds are correct. The CAMELS gas, placed in a 205 Mpc/h box, follows the isotropic law to within a ratio of 1.000-1.023. Our own CAMELS and FLAMINGO codes wrap with the minimum image: `np.mod` + `cKDTree(boxsize=L)` in camels_direct_profiles.py:86-87,142, and `delta -= box*np.rint(delta/box)` in stack_products.py:202, particle_stage.py and density_products.py:172.

## Size for Lee22 (derived from the validated law; TNG300 L = 205,000 ckpc/h, comoving R200c)

The bins below are the ones Lee22 actually populated. The report quotes upper edges of 10^14.0 at z <= 0.2, about 10^13.6 at z = 1 and 10^13.2 at z = 2. That matches more than 29 halos per concentration quartile, i.e. N >= ~120 per 0.2-dex bin, with Tinker08 counts. Over 36 bins:

| quantity | expected bias of the printed TNG300 profile |
|---|---|
| x = 0.04 R200c | -0.0055 % to -0.0115 % |
| x = 1.0 | -0.14 % to -0.29 % |
| x = 1.34 (edge of the fit) | -0.185 % to -0.384 % |
| integral 0.04-1.34 (n_e, P_e) | about -0.10 % to -0.20 % |
| halos with a low face within 1.34 R200c | 0.74 % to 1.54 % |

Uncertainty:
- **Realised bias.** One-sided (the bug can only remove gas). The 1-sigma realisation scatter at 1.34 R200c is 0.07-0.27 % per bin, for N = 1225 down to 155 (impact_summary.txt). The positions of the actual TNG300 halos are not available here.
- **Shape of the law.** The isotropic law has a model uncertainty of about 2 % (CAMELS gas ratio 1.000-1.023).
- **Build.** If Lee22 used a NAIVE build, the bias is 0. The values above are therefore the expected bias for the default build, and an upper envelope otherwise.

**Propagated to the fit** (refit of the no-c GNFW on 27 points in 0.04-1.34, then a regression of the shifts on ln M and ln(1+z), standing in for Lee22's global fit):
- **Pressure:** d ln P0 -0.019 % to -0.028 %, d ln xc +0.06 % to +0.09 %, dbeta/beta +0.10 % to +0.14 %.
- **Density:** d ln n0 about -0.03 %, d ln xc' +0.10 % to +0.12 %, dbeta'/beta' +0.17 % to +0.21 %.
- **Exponents:** the mass and (1+z) exponents shift by at most 3e-4.

**Painted quantities** (biased fit / unbiased fit, with the production shape clip at 10^14.8):
- **Pressure:** Y_cyl(<4 R200c), the tSZ aperture, is 0.9973-0.9983. Y_sph(<R200c) is 0.9977-0.9992.
- **Density:** N_e in the R200c sphere (mean DM per hit) is 0.9979-0.9991 for 10^13-10^14, and 0.9967-0.9972 for extrapolated 10^15 halos. N_e in the 4 R200c sphere (kSZ) is 0.9955-0.9975.

Source: verdict/propagate_fit_bias.txt.

## Does it change anything?

**(i) Lee22 corrections: no.**
- **The (1+z)^2 and (1+z)^3 factors** are code bookkeeping: comoving shell volume and comoving R200 against physical rho_crit. The bug's own redshift trend is |dk| <= 0.0005. The measured slopes are 1.84±0.06 (pressure) and 3.0-3.1 (density).
- **F_n = 1/(h^3 X_H^2) = 5.57** is unit arithmetic. The measured F_n (5.1-5.9) and F_P (0.99-1.07, and 1.21-1.25 against FLAMINGO) would rise by at most 0.2 % without the bug.
- **The beta' convention test** compares the printed tables with the paper's own drawn fit curves, which carry the same loss, so the test cannot see the bug at all. The fitted beta' moves by +0.1-0.2 %, against the 6-19 % that separates the conventions.

**(ii) FRB DM PDFs and tSZ x FRB: no.**
- **DM per hit:** without the bug, the corrected density's mean DM per hit would rise by 0.08-0.21 % (10^13-10^14 h^-1 Msun). The quoted ratios to TNG (0.978-1.024) would then move by at most +0.002, against statistical errors of 0.4-1.3 %.
- **Low-DM deficit:** the bug cannot explain the 27-41 % deficit. It has the wrong sign (removing it raises the outer density and gives fewer low-DM hits), and it is about 100 times too small.
- **y x DM chi^2:** at most +0.23 (ACT) and +0.06 (Planck). The ACT Delta chi^2 of the full corrected pair against Battaglia goes from 14.0 to at most 14.2. No ranking changes.

**(iii) y power and FLAMINGO: no.**
- **Lee22 x (1+z)^2 map:** relative to FLAMINGO it would rise by at most 0.55 % (C_ell ~ Y^2, Y_cyl(<4 R200c) +0.17-0.27 %), e.g. 3.03 to at most 3.05 at ell = 300 and 2.29 to 2.30 at ell = 5000.
- **kSZ:** at most +0.9 % (1.37-1.49 times WebSky becomes at most 1.38-1.50).
- **Maps without Lee22:** the B12 map and the FLAMINGO-calibrated HalfDome comparison (0.95-0.99) contain no Lee22 and do not change.
- **FLAMINGO stacks:** these wrap correctly. The 14 halo-snapshots that cross a face match SOAP to within 2.7e-3 (flamingo_face_check.txt).

**(iv) Report sections 1-2:** no number or conclusion is wrong. Every quoted value measures the painted model as it stands. I propose adding the documentation below.

## Corrections to the investigations

1. **Investigation 1** (bug mechanism) quotes up to -0.62 % at 1.34 R200c, -0.02 % at 0.04 R200c and 2.5 % of halos touched. All three come from a 10^14.2 h^-1 Msun halo at z = 2 (R200c = 1268 ckpc/h), a bin with no TNG300 halos. For the populated bins the figures are -0.384 %, -0.0115 % and 1.54 %.
2. **Investigation 2** (impact) uses N >= 30 halos per bin as "Lee22's criterion". That admits z = 0.5 10^14.0-14.2 (N = 44) and z = 1 10^13.8-14.0 (N = 37), which lie outside the populated bins quoted in the report. Its maxima should therefore be about -0.38 % at 1.34 R200c (not -0.503 %) and about -0.20 % for the 0.04-1.34 integral (not -0.249 %). Its other numbers agree with mine: refit shifts, DM per hit -0.089 % to -0.339 %, and (1+z)-exponent shifts.
3. **Rockstar units.** The task context worried that Rockstar positions are in Mpc/h and radii in kpc/h. The public 2021 examples use FoF GroupPos and Group_R_Crit200, both in ckpc/h. The RockstarFields were added on 2021-06-27, after c3fe9b8, and get no unit conversion: convert_coords only casts float to double. A Mpc/h vs kpc/h mix would have put every halo centre within 205 ckpc/h of the origin, or made the shells 1000 times too small. That would have produced profiles of random gas, which is excluded because Lee22's own z = 0 TNG300 curves agree with the CAMELS-TNG gas to about 10 % at 0.3 < x < 1 (report Section 1.6: CAMELS/fit 0.98 and 0.88 for the pressure; the density agrees to 6 % after the h^3 X_H^2 conversion). So the custom file must have been consistent. This is inferred; the file is not public.

## Proposed report edits (not applied)

**A. s1_lee22.html, §1.5:** insert after the paragraph at line 268 ("For the pressure this is an identity of y_prof.cpp ..."):

```html
<aside class="note"><h4>A periodic-boundary bug in the extraction code (checked 2026-10-08; no correction)</h4>In <code>group_particles</code> as Lee22 used it (c3fe9b8, default build), the periodicity hint of the cell search, <code>#define PER(x) ((x&gt;=Ncells_side) ? 1 : (x&lt;0) ? -1 : 0)</code> (<code>detail/workspace_sorting.hpp:246</code>), compares an <code>int</code> cell index with the <code>size_t</code> constant <code>Ncells_side</code>. A negative index becomes a huge unsigned number, so cells reached across the three low box faces (x, y or z = 0) get the hint +1 instead of −1. Their gas is placed about 2 L<sub>box</sub> away and dropped; nothing is added or put in a wrong bin. The upstream hotfix 5791098 (2024-04-08) made the plain periodic distance the default. The real 2021 <code>y_prof.cpp</code> and <code>ne_prof.cpp</code>, run on CAMELS-TNG snapshots, drop exactly these pairs. A halo at distance d from a low face loses the fraction (1 − d/r)/2 of each shell at r &gt; d. Averaged over positions, the stacked mean profile at x = r/R<sub>200c</sub> is low by (3/4) x R<sub>200c</sub>/L<sub>box</sub>, which is linear in R/L. The <code>Rsq/(acell*acell)</code> in the same function is the squared radius in cell units and is correct. For TNG300 and the bins Lee22 populated, the loss is −0.006 to −0.012 % at 0.04 R<sub>200c</sub> and −0.19 to −0.38 % at 1.34 R<sub>200c</sub> (realisation scatter 0.07–0.27 % per bin). It does not affect C1–C9. The a<sup>2</sup> and a<sup>3</sup> are code bookkeeping (the bug's own redshift trend is |Δk| ≤ 5 × 10<sup>−4</sup>), F<sub>n</sub> = 5.57 is unit arithmetic, and the β′ test compares the tables with the paper's own curves, which carry the same loss. <span class="src">outputs/lee22_boundary_check_20261007/VERDICT.md</span></aside>
```

**B. s1_lee22.html, §1.12 "Open issues"** (after the "Low-DM tail" item, line ~502):

```html
<li><strong>Checked, closed: the periodicity bug in Lee22's extraction code</strong> (box in Section 1.5). Correcting Lee22's profiles for the dropped gas would raise the corrected density's DM per hit by 0.08–0.21 % (10<sup>13</sup>–10<sup>14</sup> h<sup>−1</sup> M<sub>☉</sub>; ≤ 0.33 % for extrapolated clusters), Y in the 4 R<sub>200c</sub> aperture by 0.17–0.27 %, C<sub>ℓ</sub><sup>yy</sup> by ≤ 0.55 % and F<sub>P</sub>, F<sub>n</sub> by ≤ 0.2 %. No correction is applied.</li>
```

**C. s2_frb.html, §2.10 "Open items"** (line ~465, new first or last item):

```html
<li><strong>Closed (2026-10-08): the periodicity bug in Lee22's extraction code</strong> (Section 1.5). It makes the printed profiles 0.2–0.4 % low at 1.34 R<sub>200c</sub>. Undoing it would move the mean DM per hit relative to TNG by ≤ +0.002, inside the 0.4–1.3 % statistical errors. The y × DM χ² would change by ≤ +0.23 (ACT, full corrected pair) and ≤ +0.06 (Planck), and the y power by ≤ +0.55 %. No ranking or conclusion changes.</li>
```

**D. Optional wording, s1 §1.7** (line 394): change "No boundary or electron convention could explain that excess" to "No halo-boundary (aperture) or electron convention could explain that excess". This keeps it distinct from the periodic box boundary.

## Open questions

- **Lee22's actual build.** Which build and commit Lee22 compiled (default or NAIVE; c3fe9b8 or later, before 2024-04-08) is not recorded. Only L. Thiele or B. K. K. Lee can confirm. The bias above is the default-build expectation and is 0 for NAIVE.
- **Realised TNG300 bias.** The realised bias for Lee22's actual halos could be computed from the TNG300 (or TNG300-1-Dark Rockstar) halo positions with the cap formula. TNG300 data was not accessible here.
- **Upstream fix.** PER is still wrong at HEAD 6c6d486 for EARLY_RETURN builds. The one-token fix `x>=(int)Ncells_side` would make the fast hint path correct. Reporting it upstream is the user's call.

## Files

- **Code:** frb_map_generation/lee22_tsz_corrected/boundary_check/verdict/{per_test.cpp, recheck_cpp_stacks.py, propagate_fit_bias.py, cross_chi2_scaling.py}
- **Outputs:** frb_map_generation/outputs/lee22_boundary_check_20261007/verdict/
- **Investigation outputs used:** impact/impact_summary.txt, estimate/, unit/, e2e/, flamingo_face_check.txt, particle_test_summary.txt
