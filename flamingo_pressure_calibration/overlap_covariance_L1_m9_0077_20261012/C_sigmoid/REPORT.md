# Test C: sigmoid vs hard truncation

**Recommendation:** keep the hard spherical cut, integrated exactly, for Tests A, B and D, and treat x_t as
the physical choice.
- At 4R the edge shape moves pressure C_ell by <= 0.2%, below the plan's 1% trigger, so no windowed XGPaint
  profile type is needed.
- If a smooth edge is wanted for Fourier work, use a 0.05-dex logistic with a flux-matched edge.
- Electrons are about 5x more edge-sensitive (DM-field C_ell -1.1% at ell = 1000): avoid x_t <= 2R for DM.
- The tapers are not flux-neutral, and the 0.2-dex window reshapes the profile out to about 47R.

## Method

- **Profiles:** P4 and P2 (pressure, 4R and 2R fits), E4 and E2 (electrons), through `../kernels/engine.py`.
- **Windows:**
  - hard cut at x_t;
  - logistic in ln r, W = [1 + (x/x_t)^(1/w_ln)]^-1, with w = 0.05 and 0.2 dex (0.01 and 0.005 dex for
    convergence);
  - flux-matched logistic, with x_t' chosen per catalogue mass at z = 0 so that Y (or N_e) equals the
    hard-cut value: one value per calibrated bin, x_t'/x_t = 0.98-1.05 at 0.05 dex and 0.69-2.09 at 0.2 dex;
  - cosine taper over 0.9-1 x_t, plus its per-cell amplitude-matched control;
  - Gaussian taper exp[-((x - x_t)/s)^2] beyond x_t, with s = 0.1 and 0.2 x_t (B12's printed width
    "80 R500" taken as a typo).
- **Grids:**
  - one per x_t, extending to max(x_t, x_t')·(1 + 10 w_ln(0.2 dex)) = 6.7 / 17.6 / 46.8 R, which is
    >= x_t + 10 w_lin for every window;
  - 81 linear nodes across each edge zone (1757 / 1834 / 1913 nodes).
- **Lightcone spectra:** 106 ell values, 20 to 2e4. Linear P above ell = 8124 is clamped at the end of the
  CLASS table; the 2h term is <= 0.1% there.

## Gates (all pass)

| Gate | Result |
|---|---|
| Hard cut vs `tsz_halo_terms.npz` | <= 0.45% (the known 256-node bias) |
| Hard cut vs the existing 1024-node refinement | <= 3.1e-5 |
| E4/E2 at 4R vs the kernels Jacobian run | <= 4e-5 |
| w -> 0, pressure at x_t = 1 | max abs(Delta C) = 2.5 / 0.11 / 0.028% at 0.05 / 0.01 / 0.005 dex (x4.0 per halving, as w^2) |
| w -> 0, electrons | <= 0.13% at 0.005 dex |
| Analytic transforms | 6e-14 / 1.7e-7 |
| Node doubling (P4/E4, main windows) | <= 2.0e-4 |
| Flux-matched edge residual | 6.4e-8 at z = 0; <= 0.29% over all lightcone cells for P4 at 4R (assumed z-response) |
| Projection vs the frozen chord rule | 1.3e-6 |
| Box P(k) | reproduces `snapshot_grid128.npz` to 0 / 9e-16 |

## Results

**Total C_ell, P4 (P2 within 0.1-0.5 points):** variant/hard - 1 [%] at ell = 100 / 1000 / 3000 / 8000 / 2e4.

| Variant | x_t = 1 | x_t = 2 | x_t = 4 |
|---|---|---|---|
| Sigmoid 0.05 | -2.25 / -2.16 / -1.90 / -1.14 / -0.75 | -1.14 / -0.68 / -0.29 / +0.04 / +0.22 | -0.21 / -0.07 / 0.00 / +0.01 / -0.01 |
| Flux-matched 0.05 | -1.48 / -1.78 / -1.82 / -1.24 / -0.69 | -0.69 / -0.49 / -0.27 / +0.01 / +0.21 | -0.07 / -0.05 / 0.00 / +0.01 / -0.01 |
| Sigmoid 0.2 | -19.4 / -19.3 / -19.2 / -14.9 / -9.1 | - | -5.8 / -3.7 / -2.2 / -0.8 / -0.07 |
| Flux-matched 0.2 | - | - | -1.7 / -1.4 / -0.9 / -0.4 / -0.07 |
| Cosine | -4.6 / -2.8 / -1.5 / -0.2 / +0.6 | - | -0.23 / -0.03 / 0 / 0 / 0 |
| Cosine, amplitude-matched | - | - | +0.64 / +0.70 / +0.68 / +0.68 / +0.75 |
| Gaussian 0.1 | +7.0 / +4.0 / +2.0 / +0.1 / -1.1 | - | +0.31 / +0.03 / 0 / 0 / 0 |

The existing controls gave -0.22 / -0.03 (cosine) and +0.61 / +0.67 (amplitude-matched).

**DM-field C_ell, E4:** same columns and ell values.

| Variant | x_t = 1 | x_t = 2 | x_t = 4 |
|---|---|---|---|
| Sigmoid 0.05 | - | -1.2 / -1.8 / -1.8 / +0.6 / -2.2 | -0.55 / -0.99 / +0.15 / -0.14 / -0.20 |
| Flux-matched 0.05 | - | - | -0.62 / -1.09 / +0.21 / -0.12 / -0.19 |
| Cosine | -8.4 / -6.4 / -4.3 / -0.3 / +4.8 | - | -3.1 / -0.9 / +0.4 / -0.1 / 0.0 |
| Gaussian 0.1 | - | - | +5.0 / +1.2 / -0.6 / +0.1 / -0.2 |

**Where the hard edge adds high-ell power:**
- 4R: not identifiable (<= 0.01% at ell >= 3000);
- 2R: <= 0.2% at 2e4;
- 1R: <~ 0.5% after removing the uniform redistribution offset.

**Delta Y and Delta N_e [%]**, z = 0, at log M = 13.5 / 14.5 / 15:

| Variant | Quantity | x_t = 1 | x_t = 2 | x_t = 4 |
|---|---|---|---|---|
| Sigmoid 0.05 | Delta Y (P4) | -0.03 / -0.41 / -0.63 | -0.52 / -0.68 / -0.64 | -0.41 / -0.37 / -0.29 |
| Cosine | Delta Y (P4) | -4.0 / -3.6 / -3.0 | - | -1.0 / -0.6 / -0.4 |
| Gaussian 0.1 | Delta Y (P4) | +6.5 / +5.7 / +4.7 | - | +1.5 / +0.8 / +0.6 |
| Sigmoid 0.05 | Delta N_e (E4) | +1.4 / +1.0 / +0.6 | - | -0.1 / -0.3 / -0.4 |
| Cosine | Delta N_e (E4) | - | - | -2.5 / -2.1 / -1.7 |
| Gaussian 0.1 | Delta N_e (E4) | - | - | +4.1 / +3.3 / +2.7 |

The 0.2-dex window changes Y by -4 to -8% and N_e by up to +17% at 1R.

**DM statistics at x_t = 4R**, Delta [%] for mean / variance / Q16 / Q50 / Q84 / Q97.5 / Q99.9:

| Variant | Change |
|---|---|
| Sigmoid 0.05 | +0.13 / -0.56 / +0.74 / +0.09 / -0.06 / -0.07 / -0.03 |
| Flux-matched 0.05 | -0.02 / -0.60 / +0.45 / -0.10 / -0.17 / -0.13 / -0.06 |
| Cosine | -2.9 / -0.6 / -5.6 / -3.5 / -2.1 / -1.3 / -0.6 |
| Gaussian 0.1 | +4.7 / +0.9 / +9.2 / +5.7 / +3.4 / +2.0 / +1.0 |

- kSZ 1h with the 0.05-dex sigmoid at 4R: -0.8 / -1.1 / -0.8 / +0.3 / -0.3%.
- At 1R and 2R, 35% and 1.5% of sightlines hit no halo, so relative low-quantile changes are not meaningful.

**Single-halo transforms** (10^13.5, 10^14.5, 10^15 Msun at z = 0.1 / 0.5 / 1):
- The hard cut shows sinc sidelobes. The edge term (prop. to q^-2) grows relative to the cusp term
  (prop. to q^-2.7) toward high k: for 10^14.5 at z = 0.5 the ringing is about +-2% at k ~ 10/Mpc for 2R
  pressure, and about +-0.2% at 4R.
- Sigmoids show none; the cosine taper still rings; the Gaussian rings weakly.

**Box P(k)** (z = 0, 256^3, k up to 0.76/Mpc):
- P4 at 4R:
  - flux-matched 0.05-dex sigmoid: 0.00 / -0.08 / -0.35 / +0.01 / +0.08% at k = 0.02 / 0.1 / 0.3 / 0.6 / 0.8;
  - centred sigmoid: -0.7% (flux change);
  - cosine: -1.1%;
  - Gaussian: +1.6%.
- E4 at 4R, flux-matched: -0.2 / -1.3 / -1.6 / -1.2%.
- No ringing in any case.

![C_ell ratio, pressure](figures/cl_ratio_P4.png)

![C_ell ratio, electrons](figures/cl_ratio_E4.png)

![convergence](figures/sigmoid_w_to_zero.png)

## Caveats

- All halo-model caveats of `../kernels/REPORT.md` apply.
- Profiles beyond 4R are extrapolations of the fits.
- Flux matching is per mass at z = 0 only.
- Amplitude-matched variants are numerical controls, not physical profiles.
- The node-doubling check used a subset (P4 and E4, main windows).
- The box is model-painted at z = 0 only.

## Runtimes and files

| Run | Wall time |
|---|---|
| Lightcone, nominal (156 variants) | 1080 s |
| Lightcone, doubled subset | 1461 s |
| Box P(k) | 1512 s |
| Single-halo transforms | 111 s |
| Projections | ~60 s |

- **Code:** `setup_sigmoid.py`, `run_sigmoid.py` (phases nominal / doubled / tables), `single_halo.py`,
  `projected_profiles.py`, `box_power.py`, `verify_sigmoid.py`, `plot_sigmoid.py`.
- **Data:** `sigmoid_spectra*.npz`, `matched_edges.npz`, `sigmoid_cl_table.csv`, `sigmoid_flux_table.csv`,
  `single_halo_transforms.npz`, `projected_profiles.npz`, `box_power.npz`, `run_summary.json`,
  `box_gate.json`, `projection_gate.json`, `verification.json`, `SHA256SUMS`.
- **Figures (13):** `cl_ratio_{P4,P2,E4}`, `cl_ratio_zoom`, `sigmoid_w_to_zero`, `flux_change`,
  `single_halo_ell_{P4,E4}_xt{1,4}`, `single_halo_uk_ratio`, `projected_edge`, `box_power_ratio`.

(Written by the coordinator from the agent's hand-back; the harness blocked the agent from writing `.md` files.)
