# Cosmology-corrected FLAMINGO spectral fits

These are local, approximate spectral matches at FLAMINGO D3A cosmology. They are not direct halo-pressure measurements, posterior estimates, or a validated conversion between N-body realizations.

The original 8,192-row production configuration and prior were not changed.

| Parameter | L1_m9 | fgas-8sigma | Mstar-1sigma |
|---|---:|---:|---:|
| P0 | 27.8958 | 18.2359 | 33.7653 |
| xc | 1.37428 | 1.46717 | 1.37269 |
| beta | 8.09651 | 7.37547 | 8.37623 |
| alpha_m_P0 | 0.240687 | 0.273828 | 0.132032 |
| alpha_m_xc | -0.370872 | -0.29276 | -0.383487 |
| alpha_m_beta | -0.0271318 | 0.0614758 | -0.0432107 |
| alpha_z_P0 | -3.29505 | -3.48171 | -3.87061 |
| alpha_z_xc | 1.32128 | 1.56047 | 1.6022 |
| alpha_z_beta | 0.712305 | 0.933832 | 0.886075 |

| Variant | Previous RMS / maximum | New RMS / maximum | Selected candidate |
|---|---:|---:|---|
| L1_m9 | 2.31% / 7.58% | 1.17% / 4.27% | L1_m9_iter1 |
| fgas-8sigma | 1.90% / 5.88% | 0.60% / 1.64% | fgas-8sigma_iter1 |
| Mstar-1sigma | 1.75% / 6.59% | 0.77% / 2.54% | Mstar-1sigma_iter1 |

RMS and maximum residuals refer to the 40 clean, beam-smoothed, masked D_ell bins relative to the corresponding FLAMINGO spectrum. The new prediction is a full HalfDome map spectrum multiplied by a parameter-dependent one-plus-two-halo cosmology ratio.

| Variant | P0 with other eight coefficients fixed | Final proposal converged | Maximum beta on interpolation domain |
|---|---:|---|---:|
| L1_m9 | 20.3297 | False | 32.8732 |
| fgas-8sigma | 16.1933 | True | 49.999 |
| Mstar-1sigma | 35.0944 | False | 50 |

The full refits can move along parameter degeneracies, and use the current beta ceiling of 50; the older fgas-8sigma and Mstar-1sigma fits used 40, whereas the retained older L1_m9 fit used 64. Their parameter changes and fit improvements therefore cannot all be attributed to cosmology. An optimizer iteration limit is not convergence; the reported candidate is selected by its checked full-map residual.

See ../README.md for the equations, numerical checks, assumptions, sources and implications for SBI training.
