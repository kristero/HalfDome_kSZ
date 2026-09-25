# Standalone comparison with the frozen production override and Julia QuadGK.
using XGPaint, DelimitedFiles, TOML
include(joinpath(@__DIR__, "inputs", "stable_los.jl"))

function finite_column(x, p0, xc, beta, endpoint)
    lower, upper = log(x), log(hypot(x, endpoint))
    peak = beta <= .7 ? upper : clamp(log(.7*xc/(beta-.7)), lower, upper)
    logshape(logr) = .7logr + .3log(xc) - beta*log1p(exp(logr-log(xc)))
    scale = logshape(peak)
    integrand(u) = exp(logshape(lower+log(cosh(u)))-scale)
    value, error = XGPaint.quadgk(integrand, 0., asinh(endpoint/x); rtol=2e-12, maxevals=4096)
    @assert isfinite(value) && value > 0 && error <= 1e-11value
    return exp(log(2p0)+scale+log(value))
end

function main()
    root = dirname(@__FILE__)
    probes = readdlm(joinpath(root, "results", "column_probes.csv"), ',', Float64; skipstart=1)
    differences, old_differences = Float64[], Float64[]
    rejected = 0
    open(joinpath(root, "results", "julia_columns.csv"), "w") do stream
        println(stream, "radius,beta,los,python,julia,relative_difference,production_asserted")
        for row in eachrow(probes)
            x, p0, xc, beta, endpoint, python = row
            value = finite_column(x, p0, xc, beta, endpoint)
            difference = abs(value/python-1)
            push!(differences, difference)
            asserted = false
            if endpoint == 1e5
                # XGPaint's argument uses an outer-slope parameter beta_raw-gamma.
                try
                    native = p0*XGPaint._nfw_profile_los_quadrature(x, xc, 1., beta+.3, -.3)
                    push!(old_differences, abs(native/value-1))
                catch exception
                    exception isa AssertionError || rethrow()
                    asserted = true
                    rejected += 1
                end
            end
            println(stream, join((x, beta, endpoint, python, value, difference, asserted), ','))
        end
    end
    @assert maximum(differences) < 1e-8
    @assert maximum(old_differences) < 1e-8
    result = Dict("probes"=>length(differences), "production_assertions"=>rejected,
        "max_python_julia_relative_difference"=>maximum(differences),
        "max_supported_production_difference"=>maximum(old_differences),
        "julia_version"=>string(VERSION), "passed"=>true)
    open(joinpath(root, "results", "julia_verification.toml"), "w") do stream
        TOML.print(stream, result)
    end
    println(result)
end

main()
