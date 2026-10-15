# Emulator for the complete 64k tSZ dataset

This independent pipeline maps the nine pressure parameters to the **40 clean,
masked, beam-smoothed Compton-y power-spectrum bands** used by the 64k SBI
analysis. It preserves the existing emulator and SBI outputs. It does not
emulate the unbinned spectrum or the unbinned MOPED compression.

The input is `SBI_analysis/tsz64k_sbi_local/data/rows_65536`, checked against the
frozen Sobol design and the separately archived 64,440 and 1,096 source pieces.
The regenerated 1,096 rows differ between collectors only at approximately
6e-13 relative floating-point binning precision. The original nine-parameter
order and wide prior bounds come directly from this dataset's metadata.

## Model and statistical separation

- Exact amplitude dependence: `D_ell = P0**2 * shape(other_eight_parameters)`.
- Network targets: standardized `log10(D_ell / P0**2)`.
- Inputs: `log10(xc)`, `log10(beta)`, and the six linear evolution exponents.
- Residual networks: four blocks, widths 256 and 384, independently initialized.
- Split seed: 20260929; 49,152 training / 6,554 validation / 9,830 test rows.
- Normalizers use training rows only. Validation selects checkpoints and the
  single network or geometric-mean ensemble. The test set is opened for error
  metrics only after selection. There is no final refit using validation data.

Factoring the amplitude is exact because pressure, Compton-y, beam smoothing,
and masking are linear in P0, while the power spectrum is quadratic. It does
not impose an approximation on the other eight pressure parameters. Positive
linear D_ell predictions are returned even though training uses logarithms.

## Reproduce

From the HalfDome repository, with a Python environment containing NumPy,
PyTorch, and Matplotlib (Python 3.11 or later):

```bash
python emulator_tSZ/tsz64k_20260929/train.py \
  --output emulator_tSZ/tsz64k_20260929/runs/new_run --device cuda
python emulator_tSZ/tsz64k_20260929/evaluate.py \
  --output emulator_tSZ/tsz64k_20260929/runs/new_run --device cuda
```

Use a fresh output directory. CPU execution is supported with `--device cpu`.
The completed run is in `runs/main/`; `RESULTS.md` contains the measured
metrics, `COMPLETE.json` records completion, and `SHA256SUMS` verifies artifacts.

## Predict

```python
import sys
import numpy as np
sys.path.insert(0, "emulator_tSZ/tsz64k_20260929")
from model import Emulator

emulator = Emulator("emulator_tSZ/tsz64k_20260929/runs/main/emulator.pt")
theta = np.array([18.1, .497, 4.35, .154, -.00865, .0393, -.758, .731, .415])
dl = emulator.predict(theta)  # shape (1, 40), dimensionless linear D_ell
```

The parameter order is `P0, xc, beta, alpha_m_P0, alpha_m_xc, alpha_m_beta,
alpha_z_P0, alpha_z_xc, alpha_z_beta`. Out-of-prior inputs are rejected. No
noise is added by `predict`. For MCMC, compare the predictions with the data
using an independently specified likelihood.

## Diagnostics and accuracy budgets

Loss histories, per-bin error percentiles/bias/RMSE, profile error distributions,
parameter-slice diagnostics, prediction parity, representative spectra, and
the six worst profiles by two metrics are saved as PNG and vector PDF figures.
CSV files retain all profile parameters, source row IDs, and per-bin metrics.

Fractional RMSE weights fractional errors equally; linear RMSE is dominated by
the brightest spectra. Log RMSE summarizes errors across the prior's very wide
dynamic range. All three are reported. Independent Battaglia12 and 16 nearby
direct-painter spectra provide external checks unused during training.

The reference covariance distance is
`epsilon_B12 = sqrt((prediction-truth).T @ C_B12^-1 @ (prediction-truth))`.
The 64-realization OAS covariance is used without dividing by 64. It varies
instrumental split noise at a fixed older B12 sky, excludes cosmic variance,
and is only a **reference precision scale** across the wide prior and updated
painter. No covariance-adjusted training objective is used.

The declared percentage gate requires median absolute error <= 1%, 95th
percentile <= 5%, maximum per-bin mean bias <= 1%, and all parameter-quartile
95th percentiles <= 7.5%. The separate provisional MCMC gate requires epsilon
< 0.1 at B12, < 0.3 at every local external point, and a held-out 95th-percentile
reference epsilon < 0.3. These diagnostic thresholds do not establish posterior
coverage or accuracy under a parameter-dependent covariance. `quality_gate.json`
reports failures as well as passes; completion is not synonymous with passing.

`evaluate.py` reuses the existing numerical metric functions in
`emulator_tSZ/train_so_noiseless_emulator.py`; it does not reuse the older
emulator's narrower prior or weights. Source snapshots are saved with results.
