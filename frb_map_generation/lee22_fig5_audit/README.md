# Lee22 Figure 5 density audit

This is a separate, read-only evaluation of the existing HalfDome Lee22 profile.
It does not repaint a lightcone or change the density code, profile caches, or
the previous TNG/Ralf PDFs. Outputs are in
`../outputs/lee22_fig5_audit_20260922/`; read `REPORT.md` there for the result.

## Reproduce

From the HalfDome repository, in WSL:

```bash
JULIA_DEPOT_PATH=/home/kn18001/.julia OPENBLAS_NUM_THREADS=1 \
  /home/kn18001/.juliaup/bin/julia --threads=1 \
  frb_map_generation/lee22_fig5_audit/export_profiles.jl
```

Then run with Python that has NumPy, SciPy, Matplotlib and PyMuPDF:

```bash
python frb_map_generation/lee22_fig5_audit/compare_fig5.py --fetch
```

Omit `--fetch` to reuse the retained reference PDF. In this environment the
Windows conda Python supplies these dependencies and working internet access.
`--fetch` obtains only the public arXiv source and extracts the density figure
and TeX; it does not require a TNG key or fetch a simulation catalogue.

## What the programs do

- `export_profiles.jl` evaluates the actual XGPaint wrapper used by HalfDome
  with both `:xgpaint_ne2d` and `:literal`, keeping the common `:mcut` pivot,
  mass-shape clip and physical-redshift scaling. It checks density equivalence
  and the normalization of 48 finite, spherical `1R200c` chords. Both native
  cosmologies, three redshifts, both figure mass bins, and 181 radii are sampled.
- `compare_fig5.py` extracts the original vector paths, validates the log-axis
  transformation against ticks, independently checks the Julia profile against
  Table A2, and makes the PDF/PNG/SVG and numerical comparison. It asserts the
  expected production profile hash, preventing an unnoticed source change.

The comparison uses geometric mass-bin centres. Shading spans 101 mass values
inside each bin; it is not an uncertainty band. The optional fitted mass and
amplitude in `audit.json` are explicitly a shape diagnostic and are never used
in the plotted prediction or primary errors. The reference points are extracted
from the authors' figure, not the original hydro data or their covariance.

The arXiv source calls the density asset `figures/fig_4.pdf`; in the compiled
paper it is Figure 5. The reference used here is arXiv v1. The journal site
blocked direct access during this audit; no claim is made to a separate
digitization of the final journal PDF.
