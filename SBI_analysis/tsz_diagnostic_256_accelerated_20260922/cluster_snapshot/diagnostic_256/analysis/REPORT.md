# Completed 256-row diagnostic

Numerical pilot complete; inspect recovery and calibration before authorizing a larger run

Completed rows: 256. Numerical gate: True. Inference runs: 24.
Incomplete posterior-sampling runs: 0.

Primary MOPED RMSE / prior-mean RMSE by parameter:
{
  "P0": 0.9617604722121582,
  "xc": 0.8090336054080499,
  "beta": 0.9485540551719064,
  "alpha_m_P0": 0.9907787581266413,
  "alpha_m_xc": 0.9305667747041984,
  "alpha_m_beta": 0.9953026230141895,
  "alpha_z_P0": 0.9124574532945441,
  "alpha_z_xc": 0.8112997505668035,
  "alpha_z_beta": 0.9977892641287515
}

Numbers below one indicate improved held-out point prediction over the prior mean. They do not alone establish posterior calibration.

Remaining limits:

- Conditional SO noise on one fixed catalogue, not a full observational covariance
- 64 held-out Sobol rows provide a coarse coverage diagnostic, not a precise SBC certificate
- The NSIDE8192 point painter retains measured resolution systematics
- FLAMINGO posteriors are effective parameters at unmatched cosmology and gas physics
