# Step 0.1: profile-space kernels and the propagation engine

## Engine (`engine.py`)

It propagates any z = 0 profile F0(M, x), with x = r/R200c, through the frozen lightcone halo model: 421 x 238
(z, M) cells holding 6.99 M HalfDome haloes, CLASS distances, Tinker bias and P_lin.

- **Redshift response (assumed):** F(M, z, x) = F0 · Fpub(M, z, x) / Fpub(M, 0, x).
- **Profile inputs:**
  - `FitProfile`: per-0.2-dex-bin B12 or B16 fits, exactly as `evaluate_shapes`. The common support is
    13.5-15.388, with the published fallback at the actual mass outside it.
  - `TableProfile`: a tabulated F(bin, x).
  - `ScenarioProfile`: the old 0.02-dex table.
  - Any window W(M, x) can multiply the profile.
- **Transforms:** `lightcone_analytic.linear_radial_kernel`, the exact integral of a piecewise-linear F.
  - Grids are split into segments at break radii. This makes hard cuts exact, lets windows extend past the
    cut, and gives exact per-shell pieces.
  - One guard was added: the frozen 8-term moment series (|q| < 0.2) is exact only while q·x <= 1.5, so on
    grids extended beyond 7.5R the closed form is used past that point (2e-14 vs 4.5e-9 error without it).
- **Default fine grid:** 1024 log nodes over 1e-5..4R (the original had 256), plus the breaks.
- **Observables per variant**, on the 83-ell grid or an extended 106-ell grid (20..2e4):
  - yy or tau-tau C_ell, 1h and linear 2h (cross-mass terms kept), by-z and by-mass breakdowns; mean y and
    mean tau.
  - kSZ 1h = sum (sum v_LOS^2/c^2) |tau_ell|^2 / 4 pi, with the catalogue's matched velocities (the existing
    convention). A variant replaces them with N sigma_v,lin(z)^2.
  - Approximate kSZ 2h = sigma_v,lin^2/c^2 · P_lin · W_tau^2 / V: the high-k limit P_q_perp ~ 2 sigma^2
    P_lin, which over-predicts at ell <~ 300. The overall velocity normalisation cancels in the importance
    kernels.
  - sigma_v,lin(z) = f a H D sqrt(int P0 dk / 6 pi^2), with P0 rebuilt from the saved CLASS table:
    - the table rows collapse to 0.17% rms, and sigma8 = 0.81594 vs CLASS 0.81590;
    - this gives 309 / 301 / 278 km/s at z = 0.1 / 0.5 / 1, against catalogue rms 302 / 308 / 281 km/s.
  - DM terms (z_s = 1, cells with z <= 1, observer factor 1/(1+z)):
    - mean DM;
    - the Campbell variance sum N/4 pi int DM_h^2 d^2 theta;
    - DM-field C_ell 1h and 2h;
    - y x DM C_ell 1h and 2h;
    - the exact compound-Poisson per-sightline PDF via FFT, pmf = IFFT[exp(Lambda_hat - Lambda_tot)], on a
      0.125 pc cm^-3 grid up to 8192 with overflow tracked. Quantiles: 16 / 50 / 84 / 97.5 / 99.9%.
  - These DM definitions are the analytic twins of `frb_poisson_control.py` and are gated against it;
    `analytic_DM_moments.json` has no quadratic DM term.
  - Y(<x) and N_e(<x).
- **Jacobians:** dO/d ln F per segment and per 0.2-dex mass bin (15 calibration bins plus >= 15.5). DM
  quantiles use a second pass with the Mecke formula, dQ/d ln F = sum lambda·g·f(Q - x)/f(Q).
- **Parallelism:** fork workers over cost-balanced redshift chunks; results are summed in a fixed order.
- **Usage:**
  `E.Engine().run([E.Variant(...)], ell=E.ELL, workers=8, pairs=[("P4", "E4")])`; new fits enter via
  `E.FitProfile(field, params[15, 3], label, s["mass_edges"], s["support"], s["max_logmass"])`.

## Gates

All pass; recorded in `verification.json`, which embeds `verification_base_interpreter.json` and
`jacobians_verification.json`.

| Gate | Result |
|---|---|
| Parity with `tsz_halo_terms.npz`: 16 cases x 83 ell, 1h/2h/total, by-z, taper and matched controls | 1.1e-15 under scipy 1.11.4. In scipy 1.17 the raw offset is 1.084e-8 = the CODATA 2018->2022 sigma_T/m_e change; corrected, 4.8e-15 (by-z 9e-13). |
| Known numbers | -32.24 / -5.87 / -1.59%; taper -0.220 / -0.025%; Y-matched taper +0.611 / +0.665% |
| Fine grid vs original grid | 0.447% (existing node check: 0.441%) |
| Fine grid vs existing 1024-node refinement | 3.7e-5 |
| Fine grid vs its own doubling | 2.2e-4 |
| DM definitions (mean DM, lambda, yy self and 2h, mean y) vs `analytic_spectra.npz` | 2.2e-16 |
| kSZ self term | 1.0e-8 (float32 limit of the existing product) |
| Campbell variance vs 120k-ray Poisson control | +0.66 / +0.45 / +0.29% (0.2-0.45 sigma) |
| FFT quantiles vs Poisson control | pulls <= 1.6 sigma |
| FFT PDF mean and variance vs exact | -1.5e-8 and +2.2e-6 |
| 2-seed Monte Carlo (200k rays per seed) | pulls <= 1.6 sigma |
| Analytic transforms (top-hat / Gaussian) | 6.3e-14 / 1.7e-7 |
| Analytic columns | 2e-12 |
| Parseval | 4.9e-9 |
| Jacobian sum rules | 3.7e-15 (quantiles 4e-6) |
| Jacobian vs central finite differences (2 shell x bin cases, all observables) | <= 1.5e-5 (quantiles <= 5.8e-5) |

The finite-difference gate first failed because `make_grid` skipped single-node zones; the bug was fixed
and the gate re-run.

## Which radii and haloes carry the signal

**Normalisation.** Normalised importance = J/(k O), with k = 2 for quadratic observables (auto C_ell 1h and
2h, kSZ, DM variance) and k = 1 for means, quantiles and each side of a cross spectrum. Every map sums to 100%.

The auto 2h term P·W^2 is quadratic in F, so it also sums to 2. Only W itself, or a cross with a fixed
tracer, sums to 1. Raw and normalised values are both saved.

**Pressure (P4 = 4R fit, hard cut 4R): yy total, % by radial band.** Columns are ell = 100 / 300 / 1000 /
3000 / 8000.

| Band | ell 100 | 300 | 1000 | 3000 | 8000 |
|---|---|---|---|---|---|
| < 0.1R | 3 | 5 | 6 | 7 | 11 |
| 0.1-0.5R | 45 | 56 | 56 | 66 | 77 |
| 0.5-1R | 33 | 26 | 28 | 23 | 12 |
| 1-2R | 15 | 11 | 9 | 4 | 0 |
| 2-4R | 3 | 2 | 1 | 0 | 0 |

- The linear 2h term is more extended: 65% of it comes from inside 1R at ell = 100.
- P2 is within 1 point of P4. Mean y: 60% from inside 1R.

**Electrons (E4), % by band (< 0.1 / 0.1-0.5 / 0.5-1 / 1-2 / 2-4R).** Spectra are given at ell = 100 / 300 /
1000 / 3000 / 8000.

| Observable | Share |
|---|---|
| mean DM | 0 / 11 / 20 / 31 / 37 |
| DM variance | 2 / 37 / 32 / 21 / 8 |
| Q16 | 0 / 0 / 5 / 31 / 64 |
| Q50 | 0 / 3 / 15 / 37 / 45 |
| Q99.9 | 3 / 56 / 20 / 12 / 9 |
| kSZ 1h, inside 1R (by ell) | 46 / 48 / 53 / 67 / 91 |
| kSZ 1h, 2-4R (by ell) | 23 / 22 / 17 / 7 / -3 |
| DM-field 1h, inside 1R (by ell) | 54 / 55 / 62 / 85 / 108 |
| y x DM 1h electron side, inside 1R (by ell) | 64 / 67 / 75 / 96 / 107 |
| y x DM 1h pressure side, inside 1R (by ell) | 79 / 81 / 85 / 95 / 103 |

Inside-1R shares above 100% occur where outer shells have negative sensitivity at high ell (phase).

**Mass dependence and calibration coverage.**
- **yy:** calibrated bins carry 99-97% (90% at ell = 8000). The dominant bin moves with ell: 14.9-15.1 at
  ell = 100 (30%), 14.7-14.9 at 1000, 14.5-14.7 at 3000, 14.3-14.5 at 8000.
- **DM:** calibrated bins carry 59% of the mean, 51-89% of the quantiles (Q16 -> Q99.9) and 75% of the
  variance; Q99.9 is driven by the 14.3-14.7 bins (46%).
- **kSZ 1h:** calibrated share 91% at ell = 100 but only 30% at ell = 8000.

**Conclusion.**
- tSZ depends on 0.1-1R and is not core-dominated.
- DM depends on 1-4R and on haloes <= 10^13.9, where the published fallback and the shallow fitted beta apply.
- The current fit weights (peaking at 0.4-0.7R) suit tSZ but under-weight what DM needs.

![importance](figures/importance_heatmap.png)

## Jacobians (`jacobians.npz`)

1093 nodes and 70 segments. Breaks are the requested x200 edges (63 shells) plus the 8 coarse edges, so
the coarse aggregation is exact. The ell axis has 106 values; `ell_original83_index` selects the original 83.

| Array | Shape and meaning |
|---|---|
| `J_<obs>__<P2,P4,E2,E4>` | [shell, bin(, ell)] for `cl_1h`, `cl_2h`, `cl_total`, `mean` |
| `J_<obs>__<E2,E4>` | electrons only: `ksz_cl_1h`, `ksz_cl_1h_linear_sigma_v`, `ksz_cl_2h_approx`, `dm_cl_1h`, `dm_cl_2h`, `dm_mean`, `dm_var` |
| `J_dm_quantiles__E*` | [5, shell, bin] |
| `J_cross_{1h,2h}_{P,E}__P*xE*` | cross-spectrum, pressure side and electron side |
| `O_*` | observable values |
| `I_shell_z0__*` | [bin centre, shell], for Y(<x), N_e(<x) and their Jacobians |

The schema is stored in the file. Intended metric: K = sum_O lambda_O J_O^T J_O / O^2. Per-segment raw arrays
are in `jacobians_engine_output.npz`.

## Timing (8 workers)

| Run | Wall time |
|---|---|
| 50 variants (25 pressure + 25 electron with PDF/kSZ/DM), 1032 nodes, 83 ell | 119 s (892 CPU-s, 1.1 GB per worker) |
| Jacobian run | 281 + 57 s |
| Gate suite | 982 s |

## Caveats

- **Model:**
  - Deterministic mean profiles; the redshift response is assumed, not measured.
  - Flat-sky profiles with a Limber linear 2h term; all-gas profiles can double-count matter once a 2h
    term is added.
  - Cell-centre masses and redshifts.
- **Coverage:** the electron support is 13.5, as in `evaluate_shapes`. Mean DM is therefore 225.6 here
  versus 233.4 in the old moments file, which used 13.3 plus ramps.
- **Approximations:**
  - The kSZ 2h term is approximate.
  - The DM PDF assumes independent halo centres (no clustering, no IGM).
  - Importance values are local derivatives at the fitted profiles, not error budgets.

## Files

- Code: `engine.py`, `velocity.py`, `dm_pdf_mc.py`, `verify_engine.py`, `jacobians.py`, `importance.py`,
  `timing.py`, `plot_kernels.py`, `plot_style.py`.
- Data: `jacobians*.npz`, `importance.npz`, `importance.csv`, `importance_summary.json`, `timing.json`,
  `timing_bootstrap_outputs.npz`, `gate_spectra.npz`, `verification*.json`, `SHA256SUMS`.
- Figures: `figures/importance_{heatmap,mass_shell,flux_mass}`.

(Written by the coordinator from the agent's hand-back; the harness blocked the agent from writing `.md` files.)
