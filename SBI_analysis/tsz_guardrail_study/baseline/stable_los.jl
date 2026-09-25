# A numerical replacement for exactly the same finite LOS integral.
# Loaded only in this campaign's Julia process; archived XGPaint is untouched.
# The fixed alpha=1, gamma=-0.3 profile and endpoint=1e5 are asserted.
@eval XGPaint begin
    const extended_cleanup_stats = Ref((count=0, floor=0.0))

    function replace_nonpositive_with_floor!(profile_grid)
        # Native cleanup multiplies the smallest positive value by 1e-6.
        # That multiplication can itself underflow to zero, so log(profile)
        # contains -Inf and contaminates the cubic interpolator globally.
        # Preserve the native floor whenever representable; otherwise use
        # the smallest strictly positive value of the same floating type.
        T = eltype(profile_grid)
        minimum_positive = typemax(T)
        count_bad = 0
        for value in profile_grid
            if value > zero(T)
                minimum_positive = min(minimum_positive, value)
            else
                count_bad += 1
            end
        end
        if count_bad == 0
            extended_cleanup_stats[] = (count=0, floor=0.0)
            return 0, zero(T)
        end
        smallest_positive = nextfloat(zero(T))
        native_floor = isfinite(minimum_positive) ? minimum_positive*T(1e-6) : smallest_positive
        floor_value = max(native_floor, smallest_positive)
        @inbounds for i in eachindex(profile_grid)
            profile_grid[i] <= zero(T) && (profile_grid[i] = floor_value)
        end
        @assert all(v -> isfinite(v) && v > 0, profile_grid)
        extended_cleanup_stats[] = (count=count_bad, floor=Float64(floor_value))
        println("Strictly positive cache floor: count=", count_bad, " floor=", floor_value)
        flush(stdout)
        return count_bad, floor_value
    end

    function _nfw_profile_los_quadrature(x, xc, alpha, beta, gamma;
                                        zmax=1e5, rtol=1e-12, order=9)
        @assert alpha == 1.0 && gamma == -0.3 && zmax == 1e5
        raw_beta = beta + gamma
        @assert x > 0 && xc > 0 && raw_beta > 0.7
        # l=x sinh(u), r=x cosh(u). Normalize at the maximum of the
        # transformed integrand to avoid subnormal relative-error stalls.
        logx, logxc = log(x), log(xc)
        upper = asinh(zmax/x)
        peak = clamp(logxc + log(.7/(raw_beta-.7)), logx, log(hypot(x, zmax)))
        logshape(logr) = .7*logr + .3*logxc - raw_beta*log1p(exp(logr-logxc))
        normalization = logshape(peak)
        integrand(u) = exp(logshape(logx+log(cosh(u)))-normalization)
        value, error_estimate = quadgk(integrand, 0., upper; rtol=rtol,
                              order=order, maxevals=4096)
        isfinite(value) && value > 0 && error_estimate <= 5rtol*value ||
            error("Scaled LOS quadrature did not converge")
        # Underflow may still correctly yield zero at gratuitously distant
        # interpolation nodes. The guarded positive cache cleanup handles it.
        return exp(log(2.) + normalization + log(value))
    end
end
