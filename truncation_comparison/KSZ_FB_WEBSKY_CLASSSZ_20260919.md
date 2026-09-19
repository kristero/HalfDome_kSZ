# Halo kSZ with/without f_b, checked against CLASS-SZ and WebSky (Stein+2020)

Date: 2026-09-19. Direct test of whether removing the f_b factor from Battaglia16 (the reading
that matches Lee+2022's own Fig. 5 curve, see `frb_map_generation/LEE22_FIG5_SHAPE_CHECK_20260919.md`)
is a legitimate general fix, or specific to reproducing that one figure. Computed the actual halo
kSZ power spectrum both ways, on the cluster (job 598208, idark), and compared against two
independent external references.

## 1. What was computed

`paint_ksz_fb_comparison_maps.jl`: full 85,224,251-halo HalfDome lightcone, NSIDE 4096, the same
validated spherical (4 R200c) truncation as `TSZ_KSZ_TRUNCATION_COMPARISON_20260917.md`, same
halos/geometry/catalogue velocities in one pass. "Without f_b" is obtained by dividing the
per-pixel, already-truncated, already-velocity-weighted tau accumulation by f_b in the same loop
-- exact, since f_b is a flat multiplicative constant on Battaglia16's gas density with no
interaction with mass, redshift, radius or the truncation geometry (confirmed: the two maps' rms
ratio is 6.356, matching 1/f_b = 6.356 to 4 significant figures). `compute_ksz_fb_spectra.py`:
anafast, pixel-window corrected, to ell=8192.

## 2. External references

- **CLASS-SZ** B16 halo kSZ (1h+2h), already computed 2026-09-17 (`class_sz_ksz_x4_reference.npz`,
  same 4 R200c truncation, f_free=0.9 matching XGPaint's ne2d convention) -- reused as-is; classy_sz
  has no "remove f_b" mode since its B16 gas density is a physical convention by construction.
- **Stein et al. 2020** (WebSky, arXiv:2001.08787) Figure 6: digitized directly from the PDF
  (pixel-color extraction, axis-tick calibrated) -- the "WebSky Halo" curve and the "Battaglia et
  al. (2010)" point, which the paper explicitly says was measured from the same hydrodynamical
  simulations used to fit the free-electron profile applied there (Sec. 4.4.2) -- i.e. real,
  published, simulation-based ground truth for this exact quantity.

## 3. Result

| ell | with f_b (uK^2) | without f_b (uK^2) | CLASS-SZ 1h+2h | WebSky Halo (Stein+20) |
|---|---:|---:|---:|---:|
| 300 | 0.127 | 5.12 | 0.182 | 0.109 |
| 1000 | 0.370 | 14.96 | 0.429 | 0.199 |
| 2000 | 0.568 | 22.95 | 0.616 | 0.381 |
| 3000 | 0.670 | 27.06 | 0.689 | 0.507 |
| 5000 | 0.764 | 30.85 | 0.680 | 0.606 |

**With f_b** tracks CLASS-SZ's 1h term almost exactly below ell ~ 2000 and sits within the same
15-90% band above the real WebSky Halo curve that CLASS-SZ itself sits above it (e.g. at ell=3000,
CLASS-SZ/WebSky = 1.36, ours/WebSky = 1.32 -- the same known offset, already attributed in
`TSZ_KSZ_TRUNCATION_COMPARISON_20260917.md` to the one-halo abundance/velocity-convention
differences, not a new effect here).

**Without f_b** is 27-40x higher than with f_b at every ell tested (matching 1/f_b^2 = 40.1, since
kSZ power is quadratic in tau) and is excluded by both external references by the same factor.
Figure: `spectra/ksz_fb_websky_classsz_comparison.{png,pdf,svg}`.

## 4. Conclusion

Removing f_b from Battaglia16 is not a general fix: it is specific to reproducing Lee+2022's own
Fig. 5 rendering of "Battaglia (2016)", and is decisively ruled out as a physical convention by an
independent theory code (CLASS-SZ) and an independent, real, published simulation (WebSky). The
production convention (with f_b) is confirmed correct at the power-spectrum level, consistent with
`TSZ_KSZ_TRUNCATION_COMPARISON_20260917.md`'s existing CLASS-SZ comparison and with
Stein et al. 2020 eq. (3.15)'s explicit use of the mean baryon (not matter) density.

## 5. Files

- `paint_ksz_fb_comparison_maps.jl`, `compute_ksz_fb_spectra.py` (new)
- `plot_ksz_fb_websky_classsz_comparison.py` (new; digitized WebSky data reproduced by re-running
  the pixel-extraction against `tmp/pdfs/stein2020_websky_arxiv_2001.08787.pdf` page 23 if needed --
  not committed, derived from a third party's published figure)
- `spectra/binned_spectra_fb.csv`, `spectra/ksz_fb_websky_classsz_comparison.{png,pdf,svg}`
- Cluster job 598208 (idark), `/lustre/work/kristero10/frb_data/ksz_fb_comparison_20260919/`
