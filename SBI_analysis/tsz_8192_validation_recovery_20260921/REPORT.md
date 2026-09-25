# Resolution preflight

Completed 86/86 controls.

Raw NSIDE 8192 is the candidate; 4096 is an alternative and 16384 is reference only. All output maps are 4096, with the same 2 arcmin beam and sky mask.

The primary comparison is the resolution shift in a fixed unbinned compressed space relative to the scatter of a single observation. It is not the uncertainty of the mean of 64 mocks, and it is not a measured nine-parameter posterior bias.

Noise covariance is conditional on each fixed sky. It excludes cosmic variance, foreground residuals and model discrepancy. Split noise uses the historical multiplier 1 per split; half-depth survey splits would require a separate noise convention.

The existing frozen compressor and spherical local MOPED answer different questions. The former measures what the old transform notices; the latter tests sensitivity of the new model. Derivative-step and rank-threshold sensitivity are retained.

There is no universal NSIDE requirement. Compare compressed shift, retained information, cache convergence and extreme profiles at the intended multipole cut before selecting 4096 or 8192. Neither is automatically certified by this report.

## Battaglia12

| ell max | Frozen MOPED shift | Spherical MOPED shift | Information trace fraction |
|---:|---:|---:|---:|

| 1000 | 0.175 | 0.229 | 0.17 |

| 1500 | 0.282 | 0.378 | 0.222 |

| 2000 | 0.472 | 0.824 | 0.278 |

| 3000 | 1.85 | 2.84 | 0.529 |

| 4000 | 3.34 | 6.32 | 0.894 |

| 5000 | 4.23 | 8.65 | 0.988 |

| 6000 | 4.25 | 9.27 | 0.999 |

| 7979 | 4.16 | 9.21 | 1 |

## FL_L1_m9

| ell max | Frozen MOPED shift | Spherical MOPED shift | Information trace fraction |
|---:|---:|---:|---:|

| 1000 | 0.0294 | 0.0321 | 0.126 |

| 1500 | 0.0381 | 0.0645 | 0.173 |

| 2000 | 0.0725 | 0.11 | 0.228 |

| 3000 | 0.353 | 0.528 | 0.488 |

| 4000 | 0.641 | 1.28 | 0.883 |

| 5000 | 0.72 | 1.5 | 0.987 |

| 6000 | 0.841 | 1.58 | 0.997 |

| 7979 | 0.83 | 1.58 | 1 |

See `results/report.json` for failed or pending controls, timings, covariance ranks and sensitivity tests. The 256-row dataset is not submitted by these scripts.
