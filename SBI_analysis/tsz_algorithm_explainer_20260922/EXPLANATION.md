# What "smooth boundary; half all" means

This label combines two cache changes: a smooth mathematical continuation across the halo's sphere edge, and half as many interpolation nodes on each of three axes. It does not refer to smoothing the physical gas edge, changing the Gaussian beam, reducing map NSIDE or halving the prior. Greedy scheduling and child-pixel averaging are separate experiments.

A cross-section through the adopted spherical halo. The blue segment is the allowed LOS chord. Gas beyond the sphere does not contribute. X = 4 and all distances here are in R200c units.

![spherical_chord](plots/spherical_chord.png)

For projected radius x = R_perp/R200c, the physical Compton-y is a finite LOS integral. Let p(r) be the dimensionless gNFW pressure shape, A its physical tSZ normalization, and L the half-length of the chord:

```text
L(x) = sqrt(X^2 - x^2),  X = 4
y(x) = A * 2 * integral_0^L p(sqrt(x^2 + l^2)) dl
h(x) = integral_0^1 p(sqrt(x^2 + (X^2-x^2)u^2)) du
y(x) = 2L * [A h(x)]       for x < X; otherwise y = 0
```

The cache stores A h(x), not the final y(x). Dividing the LOS integral by the chord length gives a finite, positive quantity at the edge, suitable for log interpolation. Painting restores the exact 2L and enforces zero outside. The square-root edge remains in the physical projection; the cache alone is smooth.

The boundary is the moving surface theta = 4 theta200c(M,z) inside the rectangular interpolation domain. It is not the edge of the sky mask or the minimum/maximum cache coordinate. Its location changes with halo mass and redshift, so interpolation stencils on all three axes can cross it.

# Why the previous cache had a kink

Analytic-shape illustrations at fixed alpha = 1 and gamma = -0.3. Shaded regions are outside the physical sphere. The old and new interior functions coincide before interpolation; the final gas projection remains zero outside. The derivative panel exposes the removed kink.

![smooth_boundary](plots/smooth_boundary.png)

Inside the sphere, both versions use the same chord mean. The earlier cache used p(x) as a positive placeholder outside the sphere. This matched the value at x = X but not the derivative:

```text
h(X) = p(X)
h'(X-) = p'(X) * integral_0^1 (1-u^2) du = (2/3)p'(X)
previous exterior: h'(X+) = p'(X)
```

A cubic spline uses neighboring cache coefficients. The non-smooth placeholder can therefore influence the interpolated result just inside the sphere, although the painter never directly paints exterior gas. Taking logarithms does not eliminate this derivative discontinuity. A coarser grid can make its effect more visible.

The new exterior uses the same integral expression for h(x) when x &gt; X. Its radial argument stays real and positive: r squared = x squared times (1-u squared) + X squared times u squared. This is an auxiliary continuation of the analytic pressure shape; it is not the physical pressure outside the sphere. It matches the limiting value and slope at X.

The exterior is evaluated with a 16-point Gauss-Legendre rule on u in [0,1]. The code computes log-pressure, subtracts its largest value before summing, then restores that scale. This avoids underflow in unused steep cache tails. The physical interior LOS still uses the existing bounded adaptive quadrature. There is no new physical amplitude normalization.

# Exact code path and the half-sized grid

edge_extension/entrypoint.jl loads the frozen benchmark with automatic execution disabled, includes smooth_exterior.jl to replace the chord_mean method in that process, and then runs main_benchmark(). The experimental override does not edit the installed XGPaint package. The change to the branch is:

```text
# Interior: unchanged in the experiment
if x < X
    L = sqrt((X-x)*(X+x))
    return chord_quadrature(x, xc, alpha, beta, gamma, L)/(2L)
end
# Previous: generalized_nfw(x, xc, alpha, beta, gamma)
# New:
return smooth_exterior_mean(x, xc, alpha, beta, gamma, X)
```

Actual node counts and the Float64 values array only. Additional interpolation coefficients and temporary arrays are not included. The sky maps are separate allocations.

![cache_nodes](plots/cache_nodes.png)

The tested task sets nodes = [256,128,64], in the order ln(theta), ln(z), log10(M/Msun). build_cache passes these counts to LinRange, leaving each axis endpoint unchanged. The count changes from 16,777,216 to 2,097,152 values: an eightfold reduction. The grid spacings increase by about two, rather than exactly two, because an N-node axis has N-1 intervals.

The angular range still comes from RadialFourierTransform(n=512, pad=256); that n is not changed to 256. The redshift interval stays 0.001 to 5; the mass interval starts at log10(M/Msun) = 12 and retains the configured maximum. Cache values retain the absolute 1e-300 floor, log transformation and cubic B-spline interpolation. The exact halo chord is restored after interpolation.

Pressure parameters, halo selection, physical sphere radius, LOS tolerance, beam, mask, raw NSIDE8192 and ellmax7979 are unchanged by these two cache edits. The completed smooth-half task used shared-geometry painting with static scheduling and pixel_targets = []; its accuracy result does not already include the separate greedy or pixel-averaging changes.

# What was demonstrated, and where to inspect it

The boundary gate evaluated the original and replacement interior methods on four pressure shapes and six radii. Interior values were bitwise identical; the physical support check passed. Finite-difference edge derivatives agreed with the analytic limit to better than 1.7e-5 relative error. These are source-level checks, separate from the full-catalogue spectrum test.

| Smooth half vs smooth doubled | 5 directions | 9 directions | Relative spectrum norm |

| --- | --- | --- | --- |

| Battaglia12 | 0.0001193 | 0.0001211 | 2.92e-07 |

| FLAMINGO fit | 0.0002078 | 0.0002122 | 2.45e-07 |

| Compact / faint | 1.697e-12 | 1.786e-12 | 8.77e-07 |

| Bright / shallow | 0.006964 | 0.007488 | 7.46e-08 |

All models use 85,224,251 halos, the same 2 arcmin beam and mask, and all unbinned multipoles ell=80..7979. MOPED distances use each model's held-out conditional SO noise scale, taking the larger result from two local anchor compressions. They are not posterior biases. The doubled cache is a convergence reference, not an exact analytic sky.

The four-model smooth cache build fell from 54.45 to 12.12 seconds; total clean process time changed from 29.72 to 29.46 minutes. The maps and painting dominate cost. A larger error at the bright extreme is acceptable under the user's stated criterion; no pressure combination is removed on that basis. A good result on four models is not a global numerical-error guarantee over all nine parameters.

| Existing source (under tsz_reuse_cache_20260921) | Role |

| --- | --- |

| spherical_truncation_profiles.jl:69 | Bounded normalized physical LOS quadrature. |

| spherical_truncation_profiles.jl:113 | Exact chord factor and physical support. |

| edge_extension/smooth_exterior.jl:8 | 16-node positive exterior continuation. |

| edge_extension/smooth_exterior.jl:26 | Interior-preserving method override. |

| edge_extension/entrypoint.jl:1 | Load order that activates the experimental method. |

| edge_extension/controls/001/task.toml:1 | The actual smooth-half task: nodes and raw NSIDE. |

| benchmark.jl:26 | Grid construction and cubic log interpolation. |

| benchmark.jl:77 | Restore the chord during pixel painting. |

| benchmark.jl:154 | Signal transform, beam and output-map synthesis. |

This note adds explanation and figures, not a new simulator implementation. make_figures.py saves hashes of these sources. Its boundary curves are independent numerical illustrations of the same formulas, while the table above comes from the already completed cluster results.

# Child-centre averaging: the sample points

A parent pixel is one pixel of the lower-resolution map. HEALPix's nested hierarchy splits it into four equal-area children when NSIDE doubles, and sixteen descendants when NSIDE quadruples. A child centre is simply the sky direction at the centre of one such smaller pixel; it is not a halo centre.

Actual HEALPix boundaries and centres for one equatorial NSIDE4096 parent, drawn in a tangent projection. All three panels cover the same sky area. The dots mark the evaluation locations; geometry was checked by assigning every child centre back to its parent.

![child_centres](plots/child_centres.png)

The centre-painted fine map contains y(n_child) at each dot. average_children sums those map values and divides by the number of children. For the four-child experiment:

```text
y_parent ~= [y(n1)+y(n2)+y(n3)+y(n4)]/4
exact area average = (1/Omega_parent) * integral_pixel y(n) dOmega
```

The division matters: y is a surface-brightness-like field, not a total flux per pixel. Because the children have equal area, this arithmetic mean conserves the integrated flux of the discrete fine map and preserves a constant map. It does not guarantee the true continuous halo flux: a sufficiently narrow peak can fall between all sample points.

For 8192 to 4096 there are four samples; for 16384 to 4096 there are sixteen. Increasing their number is numerical quadrature refinement. The implementation changes RING indices to NESTED indices to locate descendants, reads the RING fine map and returns a RING parent map. This index conversion does not rotate or smooth the sky by itself.

The [HEALPix pixel-window documentation](https://healpix.sourceforge.io/html/intro_Pixel_window_functions.htm) defines area averaging and an isotropic approximation to its harmonic response. The finite child-centre quadrature has its own response, so dividing by the parent pixel window alone is not an exact correction. The completed four- and sixteen-point tests have not established a converged replacement painter.

This was an experimental averaging path. The ordinary 8192-to-4096 pipeline uses harmonic reconstruction after the beam, described on the final page. It does not call average_children.

# Greedy scheduling: keep available workers busy

Each halo is a piece of computational work. Nearby or large-angular-size halos touch more pixels and rings, so equal halo counts need not take equal time. Static scheduling assigns fixed contiguous chunks to workers. If expensive halos are concentrated near the end of the catalogue, one worker can remain busy after others finish.

Illustrative schedule for the SAME sixteen blocks on four workers, with identical block costs in both panels. Each color identifies a block. White gaps are idle time. This is an algorithm illustration, not a captured trace of the cluster threads.

![greedy_scheduling](plots/greedy_scheduling.png)

```text
# Previous shared painter
Threads.@threads :static for i in eachindex(masses)
    # paint halo i
end

# Separate balanced-painter experiment
block_size = 256
Threads.@threads :greedy for first in 1:block_size:length(masses)
    for i in first:min(first+block_size-1, length(masses))
        # same painting arithmetic and ring locks
    end
end
```

When a worker finishes one block, it takes the next available block. The block size avoids creating scheduling overhead for every individual halo. It is a performance setting, not a mass cut, halo selection or pressure parameter. All halos and all four maps are still processed; no block is subdivided midway through its work.

The completed four-map default-cache test reduced painting from 1,612 to 879 seconds, and the last catalogue chunk from about 865 to 215 seconds. The full spectra agree to about 8e-16 relative norm. Different addition order can change floating-point rounding, so the expected criterion is numerical equivalence, not bitwise-identical sums.

The implementation is work_balance/balanced_painter.jl:10. Julia describes [:greedy scheduling](https://docs.julialang.org/en/v1/base/multi-threading/) as workers taking further iterator values as they become available, suited to unequal workloads. The code keeps ring locks to protect shared map additions; it does not use thread-ID-indexed mutable buffers.

# Why paint at 8192 and form the output at 4096?

The signal path. Noise splits are added and masked consistently afterward in the diagnostic observation operator. Clean performance controls stop at the masked signal spectrum. The Gaussian beam is applied once.

![output_pipeline](plots/output_pipeline.png)

Here output means the map used for masking and noise; the stored spectrum still has 7,900 unbinned multipoles. Raw halo profiles contain sharp cores and an edge before the beam; a denser painting grid reduces errors in their harmonic coefficients. After retaining coefficients through ell=12287 and applying the 2 arcmin beam, the code evaluates the smoother harmonic field on a 4096 grid. It does not average four raw child pixels.

The left two panels are a 1D Gaussian-halo illustration, not a full-sky convergence test. The beam plot uses the actual 2 arcmin transfer function. More accurate initial sampling and a smaller final smooth-map grid can coexist.

![sampling_before_after_beam](plots/sampling_before_after_beam.png)

| Output NSIDE | Pixels | sqrt(pixel area) | One Float64 map |

| --- | --- | --- | --- |

| 4096 | 201,326,592 | 0.859 arcmin | 1.5 GiB |

| 8192 | 805,306,368 | 0.429 arcmin | 6 GiB |

Using 4096 therefore saves a factor of four in map storage for the output signal, mask and each noise split. Harmonic coefficient storage at fixed lmax is unchanged, and transform runtime does not necessarily improve by four. The science cutoff stays ell=7979. The beam leaves 2.05% of signal power there and about 0.00995% at ell=12287; these factors motivate the choice but do not prove sufficient accuracy.

Qualification to the earlier recommendation: the completed raw-resolution tests held output NSIDE at 4096. They measure the effect of raw painting resolution, but do not independently certify the final 4096 map. The current map2alm calls also use niter=0. A definitive output test must use the same beam-smoothed alms, synthesize 4096 and 8192 maps, apply the same continuous mask and matched harmonic noise splits, then compare unbinned spectra and MOPED through ell=7979. Mask sampling and final-transform error belong in that test.

Thus output4096 is a justified cost-saving candidate, conditional on that check. A value of lmax below 3*NSIDE-1 is a library convention, not an exact accuracy theorem. The [Healpix.jl harmonic-transform documentation](https://juliaastro.org/Healpix/stable/alm/) describes the approximation and optional iterations. No output-resolution validation or production change was performed for this explanatory note.
