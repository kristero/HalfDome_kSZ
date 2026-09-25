# Four-child full-catalogue pixel test

Raw NSIDE8192 maps were averaged into NSIDE4096 parent pixels before the same beam and mask. All results use ell=80..7979. The comparison reference is the same centre-sampled 8192 map.

| Profile | 4096 centres | Four-child average | Average / parent window |
| --- | ---: | ---: | ---: |
| Battaglia12 | 9.512 | 22.242 | 8.1368 |
| FL_L1_m9 | 1.6322 | 26.678 | 9.3169 |
| compact | 5.1442e-06 | 3.0966e-09 | 1.8822e-07 |
| extended_shallow | 33.114 | 2000.9 | 692.96 |

Values are joint noise units in the five retained local MOPED directions, taking the larger result from the two anchor compressions. Each profile uses its own held-out split-noise covariance.

![Pixel quadrature](plots/pixel_quadrature_progress.png)

Accuracy experiment; finite child quadrature and isotropic window removal are both approximate. Not evidence that exact 4096 pixel integration fails or succeeds.

Dividing by the parent pixel window is the infinite-quadrature convention. With finitely many centre samples the response still differs. For a regular one-dimensional grid, the discrete averaging response equals parent sinc window divided by child sinc window; HEALPix windows are isotropic approximations. The pending 16384 child test checks convergence.

Pixel averaging changes the response even when total flux is conserved. [HEALPix documentation](https://healpix.sourceforge.io/html/intro_Pixel_window_functions.htm) defines this distinction. Parent-window correction alone must not be described as an exact correction for this four-point quadrature. The 16384 reference tests are still pending.
