#!/usr/bin/env julia
# Per-halo electron counts clarify what extending a profile to 3R200c adds.
# These are volume integrals, not predicted ratios of kSZ power spectra.
include(joinpath(@__DIR__, "paint_ksz.jl"))

function aperture_diagnostics(config_path)
    cfg = TOML.parsefile(config_path)
    model_pair = models(cfg)
    destination = joinpath(cfg["output"], "aperture_electron_counts.csv")
    open(destination, "w") do stream
        println(stream, "model,mass_msun,z,electrons_inside_1r200c,electrons_inside_3r200c,Ne3_over_Ne1")
        for (name, model) in zip(("lee22_previous", "battaglia16"), model_pair)
            for mass in (1e13, 5e13, 1e14, 5e14, 1e15), z in (0., .5, 1., 2., 3.5)
                par, amplitude = electron_amplitude(model, mass, z)
                radius = Float64(ustrip(uconvert(u"m", XGPaint.R_Δ(model, mass*XGPaint.M_sun, z, 200))))
                integrand(x) = x^2 * XGPaint.generalized_nfw(x, par.xc, par.α, par.β, par.γ)
                inside_one, _ = lee2022_quadgk_function()(integrand, 0., 1.; rtol=1e-9)
                inside_three, _ = lee2022_quadgk_function()(integrand, 0., 1., 3.; rtol=1e-9)
                factor = 4pi * radius^2 * amplitude / SIGMA_T
                println(stream, join((name, mass, z, factor*inside_one,
                                      factor*inside_three, inside_three/inside_one), ","))
            end
        end
    end
    println("Saved $destination")
end

aperture_diagnostics(ARGS[1])
