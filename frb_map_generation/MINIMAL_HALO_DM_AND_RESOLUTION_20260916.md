# Why HalfDome's halo-DM PDF has no low tail: minimal grazing DM and ray resolution

Date: 2026-09-16. Question: TNG's within-R200 halo DM extends to 0.1-1 pc cm^-3 in every mass window while the HalfDome Battaglia16 (B16) products stop abruptly (about 22 pc cm^-3 at 1R200c, 3 at 3R200c). Is this the HEALPix ray pixelisation averaging out grazing rays, or something else?

## 1. Smallest DM one halo can add to a ray

`compute_minimal_halo_dm.jl` evaluates each model at the aperture edge, `b = N R200c`, for M200c from the HalfDome floor (7.327e12) to 3.8e15 Msun and z = 0.05-1 (`outputs/minimal_dm_20260916/minimal_halo_dm_grid.csv`; figure `minimal_halo_dm.png`).

| Convention | Grazing DM of a floor-mass halo [pc cm^-3] | Where the PDF starts |
|---|---:|---:|
| B16, projected aperture 1R200c, long LOS | 22.3 (z=0.5) to 24.7 (z=0.02) | 23.1 (first populated bin) |
| B16, projected aperture 3R200c | 3.0 to 3.4 | 3.5 |
| Lee22 no-c corrected, 1R200c | 4.2 (z=1) to 24.2 (z=0.02) | about 4 |
| Lee22 best corrected, 1R200c | 3.5 (z=1) to 22.0 (z=0.02) | about 4 |
| B16, gas only inside the R200c sphere | 2.83 at b=0.99R, 0.88 at 0.999R, 0.28 at 0.9999R, -> 0 | no floor |

Annotated values at z = 0.5 for B16 at the 1R200c edge: 22.3 (7.3e12), 25.5 (1e13), 59.4 (1e14), 128 (1e15). The projected convention integrates the profile along the whole line of sight even for a ray at the edge of the disc, so a ray that touches the aperture still picks up the full column at impact parameter R200c. The grazing DM is nearly redshift independent because the column scales with n200 R200c proportional to rho_cr(z) R200c and the 1/(1+z) factor compensates. Hence every HalfDome positive-DM ray carries at least the floor-mass edge column, and the PDF cannot extend below it.

If instead the halo gas is counted only inside the sphere of radius R200c (the natural definition for a simulation catalogue that assigns gas cells to halos), the DM of a ray at impact parameter b goes to zero continuously with the chord length, `DM proportional to sqrt(1 - b/R200c)` near the edge. This produces a power-law low tail down to arbitrarily small DM, which is exactly what TNG shows in every window (its total curve rises from 0.1 pc cm^-3 as a rough power law up to the peak). TNG's tail is therefore expected from a spherical membership definition, independent of resolution, while a projected aperture with the profile's long LOS has a hard floor by construction. Konietzka's exact within-r200 definition is not documented in the files we hold; the shape of the tail is consistent with a spherical cut.

## 2. Resolution test: NSIDE 2048, 4096, 8192

`run_battaglia16_z1_1r200c_nside_series_local.sh` produced B16 1R200c products at NSIDE 2048 and 8192 with the same 120k-ray count, seed, catalogue, cache and windows as the NSIDE 4096 product (`outputs/zsrc1p0_nside{2048,8192}_nrays120000_allhalos_r200cx1p0_m200cprofile_seed42`). `plot_nside_low_dm_tail.py` compares them (`nside_low_dm_tail_b16_1r200c.png`, `nside_low_dm_tail_counts.csv`).

| NSIDE | pixel [arcmin] | rays with DM>0 (total) | first populated bin | rays with DM<30 | rays with DM<50 |
|---:|---:|---:|---:|---:|---:|
| 2048 | 1.72 | 64.72% | 23.07 | 2756 | 14715 |
| 4096 | 0.86 | 64.73% | 23.07 | 2592 | 14686 |
| 8192 | 0.43 | 65.12% | 23.07 | 2786 | 14931 |

The three PDFs agree bin by bin within the Poisson scatter of 120k rays; the ratio panels are flat at 1 with no trend towards low DM. The first populated bin is identical, and it coincides with the grazing floor. The reason is in how the generator works: rays are pixel centres, and each ray receives the exact profile value at its exact angular separation from the halo centre; no profile is averaged over a pixel, so the "grazing" geometry is already exact at any NSIDE. NSIDE only changes which 120k directions are drawn. The aperture radius theta200c of a floor-mass halo is 1 to 7 arcmin over z = 1 to 0.05, larger than the pixel size at every NSIDE tested (panel c of the minimal-DM figure), so even the disc-search geometry is well resolved. A 16384 run would not change this conclusion; it was not run.

## 3. Factors behind TNG's lower DM at fixed window, ranked

1. Truncation convention at the halo boundary (dominant for the low tail): spherical membership gives DM -> 0 for grazing rays; the projected aperture with long LOS gives a floor of 22 pc cm^-3 (1R200c) or 3 (3R200c). This is a definition difference, not a resolution effect.
2. Mass floor: HalfDome resolves halos above 7.3e12 Msun only, so the windows starting at 1e10 lack all the low-mass halos that give TNG DMs of 0.1-10 pc cm^-3 in the 1e10-1e12 and 1e10-1e13 panels. Within 1e13-1e14 this does not apply, yet TNG still reaches 0.1, which again points to the boundary convention.
3. Gas outside R200c along the LOS: the projected convention adds the B16 column beyond the sphere (for a ray at b = R200c, the whole column comes from r > R200c). For 1R200c this raises HalfDome relative to a spherical cut by roughly 13% in mean DM (the earlier 3R200c sphere-versus-projected test gave 12.8%).
4. Ray-pixel resolution: no measurable effect (section 2).
5. TNG-side pixelisation: if Konietzka's rays are HEALPix pixel values that average many rays per pixel, pixel averaging would smooth the TNG PDF and fill in the low tail; we cannot check this from the arrays we hold. It would act in the same direction as the spherical cut but cannot create DMs below the per-ray minimum.

For a like-for-like comparison with TNG's within-R200 quantity, the HalfDome models should be painted with a spherical R200c truncation (the machinery exists for the map painter: `paint_halfdome_matched_profile_dm_map.jl` with `radius_scaled_dm_cache.jl`; the histogram generator would need a chord-limited profile option), or TNG should be recomputed with a projected aperture.

## Files

- `frb_map_generation/compute_minimal_halo_dm.jl`, `plot_minimal_halo_dm.py`, `plot_nside_low_dm_tail.py`, `run_battaglia16_z1_1r200c_nside_series_local.sh`.
- `frb_map_generation/outputs/minimal_dm_20260916/`: `minimal_halo_dm.{png,svg}`, `minimal_halo_dm_grid.csv`, `minimal_halo_dm_summary.txt`, `nside_low_dm_tail_b16_1r200c.{png,svg}`, `nside_low_dm_tail_counts.csv`.
