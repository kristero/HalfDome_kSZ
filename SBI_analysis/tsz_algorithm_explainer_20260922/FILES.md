# Files created for the illustrated explanation

- `make_figures.py`: analytic boundary curves, HEALPix child geometry, cache counts, illustrative scheduling and beam diagrams; checks and source hashes.
- `build_note.py`: generates the illustrated PDF and `EXPLANATION.md`.
- `publish.py`: copies this explanatory bundle to idark and verifies file hashes; submits no jobs.
- `illustration_data.json`: numerical curve checks, pixel coordinates, resource counts and measured comparison values.
- `boundary_curves.csv`: all plotted pressure-shape, derivative and physical-projection curves.
- `plots/`: seven PNG figures and seven editable SVG figures.
- `qa.json`: PDF validation and illustration checks.
- `manifest.json`, `publication.json`: delivered checksums and remote verification.
- `../../output/pdf/tsz_algorithms_illustrated_20260922.pdf`: the final local PDF; published as `report.pdf` on the cluster.

Existing simulator, prior, cache implementation and prepared dataset files were read, not edited. No full-sky job or dataset was launched. Boundary curves and scheduling illustrations are identified separately from measured cluster results. Temporary rendered PDF pages and the verified delivery tarball are disposable.

Reproduce locally: run `make_figures.py` in the HalfDome scientific Python environment, then `build_note.py` with system Python and reportlab. Publishing uses Windows Python and the existing idark SSH alias. The source paths and hashes refer to the preserved sibling experiment `tsz_reuse_cache_20260921`.
