# Corrected input transforms: 40 bins vs MOPED vs plain PCA vs Merlin vs no binning

r: Pearson r of truth and posterior mean. RMSE: of the posterior mean over the RMSE of the prior centre (1 = no better than the centre). width: median 68 % width over the flat-posterior width (1 = nothing learned). log p: mean log posterior density of the true parameters (joint, 9 parameters; higher is better). TARP: expected coverage of the 68 / 95 % regions. 3,000 test rows, 3-seed ensembles; inputs asinh-scaled at the noise level.

## 61,440 training simulations

| model | log p | TARP 68 | TARP 95 | mean r | mean RMSE | mean width |
|---|---|---|---|---|---|---|
| 40 bins | -2.39 | 0.72 | 0.96 | 0.584 | 0.792 | 0.778 |
| MOPED (9, all multipoles) | -4.20 | 0.73 | 0.97 | 0.599 | 0.781 | 0.786 |
| plain PCA (whitened), all multipoles | -4.67 | 0.69 | 0.96 | 0.677 | 0.718 | 0.673 |
| Merlin (whitened PCA -> learned head) | -3.38 | 0.67 | 0.95 | 0.720 | 0.675 | 0.561 |
| no binning (7,900 multipoles -> learned head) (N = 32,000) | -7.21 | 0.69 | 0.95 | 0.542 | 0.824 | 0.796 |

### r

| model | P0 | xc | beta | alpha_m_P0 | alpha_m_xc | alpha_m_beta | alpha_z_P0 | alpha_z_xc | alpha_z_beta | mean |
|---|---|---|---|---|---|---|---|---|---|---|
| 40 bins | 0.51 | 0.69 | 0.48 | 0.65 | 0.65 | 0.32 | 0.79 | 0.72 | 0.45 | 0.584 |
| MOPED (9, all multipoles) | 0.50 | 0.71 | 0.54 | 0.66 | 0.73 | 0.33 | 0.78 | 0.72 | 0.43 | 0.599 |
| plain PCA (whitened), all multipoles | 0.63 | 0.75 | 0.65 | 0.79 | 0.79 | 0.45 | 0.80 | 0.75 | 0.48 | 0.677 |
| Merlin (whitened PCA -> learned head) | 0.71 | 0.78 | 0.70 | 0.83 | 0.81 | 0.52 | 0.84 | 0.76 | 0.52 | 0.720 |
| no binning (7,900 multipoles -> learned head) (N = 32,000) | 0.45 | 0.64 | 0.48 | 0.62 | 0.63 | 0.29 | 0.73 | 0.67 | 0.38 | 0.542 |

### rmse

| model | P0 | xc | beta | alpha_m_P0 | alpha_m_xc | alpha_m_beta | alpha_z_P0 | alpha_z_xc | alpha_z_beta | mean |
|---|---|---|---|---|---|---|---|---|---|---|
| 40 bins | 0.86 | 0.72 | 0.88 | 0.76 | 0.76 | 0.95 | 0.61 | 0.69 | 0.89 | 0.792 |
| MOPED (9, all multipoles) | 0.86 | 0.70 | 0.85 | 0.76 | 0.69 | 0.94 | 0.63 | 0.69 | 0.90 | 0.781 |
| plain PCA (whitened), all multipoles | 0.78 | 0.66 | 0.76 | 0.61 | 0.62 | 0.89 | 0.60 | 0.66 | 0.88 | 0.718 |
| Merlin (whitened PCA -> learned head) | 0.70 | 0.63 | 0.71 | 0.56 | 0.58 | 0.85 | 0.54 | 0.64 | 0.85 | 0.675 |
| no binning (7,900 multipoles -> learned head) (N = 32,000) | 0.89 | 0.77 | 0.88 | 0.79 | 0.78 | 0.96 | 0.68 | 0.75 | 0.92 | 0.824 |

### width

| model | P0 | xc | beta | alpha_m_P0 | alpha_m_xc | alpha_m_beta | alpha_z_P0 | alpha_z_xc | alpha_z_beta | mean |
|---|---|---|---|---|---|---|---|---|---|---|
| 40 bins | 0.88 | 0.72 | 0.86 | 0.73 | 0.72 | 1.00 | 0.50 | 0.65 | 0.95 | 0.778 |
| MOPED (9, all multipoles) | 0.89 | 0.74 | 0.85 | 0.71 | 0.66 | 1.00 | 0.59 | 0.68 | 0.97 | 0.786 |
| plain PCA (whitened), all multipoles | 0.76 | 0.62 | 0.78 | 0.40 | 0.52 | 0.94 | 0.46 | 0.63 | 0.94 | 0.673 |
| Merlin (whitened PCA -> learned head) | 0.49 | 0.53 | 0.67 | 0.23 | 0.46 | 0.88 | 0.30 | 0.58 | 0.90 | 0.561 |
| no binning (7,900 multipoles -> learned head) (N = 32,000) | 0.90 | 0.73 | 0.85 | 0.74 | 0.73 | 0.99 | 0.57 | 0.70 | 0.97 | 0.796 |

## 32,000 training simulations

| model | log p | TARP 68 | TARP 95 | mean r | mean RMSE | mean width |
|---|---|---|---|---|---|---|
| 40 bins | -3.92 | 0.73 | 0.96 | 0.560 | 0.812 | 0.816 |
| MOPED (9, all multipoles) | -5.52 | 0.73 | 0.97 | 0.564 | 0.809 | 0.826 |
| plain PCA (whitened), all multipoles | -6.26 | 0.70 | 0.96 | 0.656 | 0.738 | 0.700 |
| Merlin (whitened PCA -> learned head) | -4.60 | 0.67 | 0.95 | 0.709 | 0.687 | 0.567 |
| no binning (7,900 multipoles -> learned head) | -7.21 | 0.69 | 0.95 | 0.542 | 0.824 | 0.796 |

### r

| model | P0 | xc | beta | alpha_m_P0 | alpha_m_xc | alpha_m_beta | alpha_z_P0 | alpha_z_xc | alpha_z_beta | mean |
|---|---|---|---|---|---|---|---|---|---|---|
| 40 bins | 0.49 | 0.68 | 0.45 | 0.60 | 0.61 | 0.30 | 0.78 | 0.70 | 0.43 | 0.560 |
| MOPED (9, all multipoles) | 0.47 | 0.68 | 0.50 | 0.63 | 0.66 | 0.31 | 0.73 | 0.69 | 0.40 | 0.564 |
| plain PCA (whitened), all multipoles | 0.58 | 0.74 | 0.63 | 0.77 | 0.78 | 0.44 | 0.77 | 0.73 | 0.47 | 0.656 |
| Merlin (whitened PCA -> learned head) | 0.69 | 0.77 | 0.69 | 0.82 | 0.81 | 0.50 | 0.83 | 0.76 | 0.51 | 0.709 |
| no binning (7,900 multipoles -> learned head) | 0.45 | 0.64 | 0.48 | 0.62 | 0.63 | 0.29 | 0.73 | 0.67 | 0.38 | 0.542 |

### rmse

| model | P0 | xc | beta | alpha_m_P0 | alpha_m_xc | alpha_m_beta | alpha_z_P0 | alpha_z_xc | alpha_z_beta | mean |
|---|---|---|---|---|---|---|---|---|---|---|
| 40 bins | 0.87 | 0.73 | 0.89 | 0.80 | 0.81 | 0.95 | 0.63 | 0.71 | 0.90 | 0.812 |
| MOPED (9, all multipoles) | 0.88 | 0.73 | 0.87 | 0.78 | 0.75 | 0.95 | 0.69 | 0.72 | 0.92 | 0.809 |
| plain PCA (whitened), all multipoles | 0.82 | 0.67 | 0.78 | 0.64 | 0.63 | 0.90 | 0.64 | 0.68 | 0.88 | 0.738 |
| Merlin (whitened PCA -> learned head) | 0.72 | 0.64 | 0.73 | 0.57 | 0.59 | 0.87 | 0.55 | 0.65 | 0.86 | 0.687 |
| no binning (7,900 multipoles -> learned head) | 0.89 | 0.77 | 0.88 | 0.79 | 0.78 | 0.96 | 0.68 | 0.75 | 0.92 | 0.824 |

### width

| model | P0 | xc | beta | alpha_m_P0 | alpha_m_xc | alpha_m_beta | alpha_z_P0 | alpha_z_xc | alpha_z_beta | mean |
|---|---|---|---|---|---|---|---|---|---|---|
| 40 bins | 0.91 | 0.77 | 0.89 | 0.76 | 0.84 | 1.00 | 0.55 | 0.67 | 0.95 | 0.816 |
| MOPED (9, all multipoles) | 0.90 | 0.77 | 0.87 | 0.77 | 0.75 | 1.00 | 0.70 | 0.71 | 0.96 | 0.826 |
| plain PCA (whitened), all multipoles | 0.83 | 0.64 | 0.80 | 0.46 | 0.53 | 0.93 | 0.53 | 0.64 | 0.95 | 0.700 |
| Merlin (whitened PCA -> learned head) | 0.51 | 0.54 | 0.68 | 0.24 | 0.46 | 0.88 | 0.32 | 0.58 | 0.89 | 0.567 |
| no binning (7,900 multipoles -> learned head) | 0.90 | 0.73 | 0.85 | 0.74 | 0.73 | 0.99 | 0.57 | 0.70 | 0.97 | 0.796 |

## Per-row coverage of the 68 % / 95 % intervals and fraction of rows narrower than half the flat width

| model | coverage68 (mean) | coverage95 (mean) | narrower than half flat (mean) |
|---|---|---|---|
| Merlin (whitened PCA -> learned head) | 0.678 | 0.943 | 0.422 |
| 40 bins | 0.717 | 0.959 | 0.152 |
| MOPED (9, all multipoles) | 0.734 | 0.963 | 0.119 |
| plain PCA (whitened), all multipoles | 0.721 | 0.956 | 0.272 |
| no binning: 7,900 multipoles -> learned head (N = 32,000) | 0.703 | 0.950 | 0.110 |
