# Why the exterior cache continuation matters

Let p(r) denote the dimensionless gNFW pressure shape and X=4. For a projected
impact parameter x<X, the cached chord mean is

    h(x) = integral_0^1 p(sqrt[x^2 + (X^2-x^2) u^2]) du.

The physical projected y is A * 2 sqrt(X^2-x^2) * h(x), and is exactly zero
outside X. A is the unchanged physical pressure-to-Compton-y normalization.
The repaired LOS quadrature computes the same h inside X.

At the edge, differentiation under the integral gives

    h(X) = p(X)
    h'(X-) = p'(X) integral_0^1 (1-u^2) du = (2/3) p'(X).

The old outside placeholder h(x)=p(x) has h'(X+)=p'(X). It matches values but
creates a derivative discontinuity at the moving edge. Cubic interpolation
uses nodes on both sides, so the unused exterior can affect values just inside
the painted sphere. This is a numerical source of interpolation error, not a
reason to exclude shallow pressure profiles.

For x>X the same integral expression remains real and positive: its radial
argument ranges from X to x. It supplies a smooth mathematical continuation
of the cache without assigning any physical gas to the exterior. This test
evaluates that continuation with a 16-point Gauss-Legendre rule on u=[0,1],
using log-sum-exp to avoid loss of tiny tail values. The quadrature integrates
the polynomial factors in the first two edge derivatives exactly; the direct
physical interior branch is bitwise unchanged. Far-out values are solely a
cache extension, and never enter painting directly.

The gate verifies unchanged interior values, unchanged zero exterior support,
the analytic first derivative from both sides, and finite extreme unused tails.
Full-catalogue tests compare default, half-all-axes and doubled grids at fixed
NSIDE8192, beam, mask and ellmax7979. They use all four original benchmark
profiles and the same conditional unbinned-MOPED error scale. No result is
automatically promoted into the held 256-row dataset.

Files: `smooth_exterior.jl`, `entrypoint.jl`, `gate.jl`, `run.py`, `analyze.py`,
`prepare_submit.py`, generated quadrature table/plan/manifests and PBS scripts.
