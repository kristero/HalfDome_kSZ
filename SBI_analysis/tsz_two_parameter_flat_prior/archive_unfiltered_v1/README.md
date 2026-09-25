# Flat two-parameter tSZ prior

The user requested the full extended ranges for only P0 and beta, uniformly in
physical values, and confirmed that the other seven parameters remain at
Battaglia12. This is a separate two-dimensional rectangular prior:

| Parameter | Lower | Upper | Fiducial |
|---|---:|---:|---:|
| P0 | 1 | 60 | 18.1 |
| beta | 2.8 | 16 | 4.35 |

The joint density is 1/(59*13.2) inside the rectangle and zero outside.
There are no logarithmic coordinates, Gaussian weights, or rejection cuts.
Both analytic marginal PDFs are flat. The seven fixed values and full original
parameter order are recorded in `prior.json`.

## Physics and reference interpretation

For A=P0, xc, beta, A(M,z)=A0*(M200c/1e14 Msun)^alpha_m*(1+z)^alpha_z.
The thermal pressure is Pth/P200=P0*q^(-0.3)*(1+q)^(-beta), with
q=r/(xc*R200c), xc amplitude 0.497, and every evolution exponent fixed to B12.
Electron pressure is 0.5176 times thermal pressure. Changing P0 multiplies
pressure and Compton y; changing beta changes radial decline and spectral shape.
Beta is the raw Battaglia exponent; the asymptotic outer slope is -(beta+0.3).

FLAMINGO markers use P0 and beta from the saved cosmology-corrected nine-parameter
clean-spectrum fits. Their seven other fit coordinates differed from B12.
These are projections for comparison, not new two-parameter best fits, posterior
constraints, or a claim that the fixed-seven-parameter model fits FLAMINGO.
Old purple boundaries are the projected support of the same verified nine-
parameter, 40-bin SBI bundle used in the preceding plots.

The full flat rectangle includes points rejected by the preceding joint guards.
With fixed B12 evolution, that earlier slope rule requires beta0 >= about 3.3541,
and the relative-size rule requires beta0 <= 10.875; finite-Y200 may restrict
additional low-amplitude combinations. The separate `previous_guards_diagnostic`
figure shows their intersection. These diagnostics do not filter the prior.
The rectangle is not certified for full-map generation: existing worker guards
would reject some rows. No cluster run is changed or submitted by this code.

## Usage

Run from this directory using a Python environment with NumPy, SciPy and
Matplotlib:

```bash
python make_plots.py
```

This writes exact joint/marginal comparison figures as PNG and vector PDF,
the separate guard diagnostic, an 8192-row scrambled Sobol parameter design,
the expanded nine-column design, reference values, validation and input hashes.
The arrays are parameter designs only; no tSZ maps or spectra are generated.

The count is configurable, with prefix-stable sampling:

```bash
python make_plots.py --count 524288 --output outputs_524288
```

For independent inference draws rather than a Sobol design:

```python
from flat_prior import FlatPrior
prior = FlatPrior()
theta = prior.sample(1000, seed=12345)       # columns: P0, beta
full_theta = prior.expand(theta)            # original nine-parameter order
log_density = prior.log_prob(theta)
# Optional, when PyTorch is installed:
torch_prior = prior.as_torch_distribution()
```

## Files added

- `prior.json`: editable bounds, fixed parameters and sampling contract.
- `flat_prior.py`: normalized prior, IID/Sobol sampling and parameter expansion.
- `make_plots.py`: reference verification, exact plots, design and guard audit.
- `README.md`: physics, reference limitations and reproduction commands.
- `outputs/`: generated plots, arrays, CSV files and verification records.

The previous nine-parameter source files, plots and active cluster run are
preserved. Full-map validation and any new two-parameter FLAMINGO fitting are
separate work.
