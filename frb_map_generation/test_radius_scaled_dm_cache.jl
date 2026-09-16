#!/usr/bin/env julia
# Prototype validation only: compare an external radius-scaled shape table to
# direct profiles and the old angular cache. Do not install it into production.
include(joinpath(@__DIR__, "audit_lee22_profile_cache.jl"))
using HDF5
using Interpolations
using Random
using Base.Threads

function column_amplitude(name, model, mass, z)
    radius = XGPaint.R_Δ(model, mass*XGPaint.M_sun, z, 200)
    rho = XGPaint.ρ_crit(model, z)
    mp = XGPaint.constants.ProtonMass
    if name == "lee2022"
        parameters = Production.lee2022_no_concentration_parameters(mass, z, 0.68)
        return ustrip(u"pc*cm^-3", parameters.n0*200*rho/(0.76*mp)*(0.049/0.31)*radius)
    end
    parameters = XGPaint.get_params(model, mass*XGPaint.M_sun, z)
    particle_mass = XGPaint.constants.ElectronMass + (2*0.76/1.76)*mp + ((1-0.76)/(2*1.76))*4*mp
    return ustrip(u"pc*cm^-3", 0.9*parameters.P₀*(0.049/0.31)*rho*radius/particle_mass)
end

function direct_shape(name, physical_b16, mass, z, x, amplitude)
    if name == "lee2022"
        p = Production.lee2022_no_concentration_parameters(mass, z, 0.68)
        return Production.lee2022_dimensionless_los(x, p.x_c, p.alpha, p.beta_prime, p.gamma)
    end
    radius = XGPaint.R_Δ(physical_b16, mass*XGPaint.M_sun, z, 200)
    return ustrip(u"pc*cm^-3", XGPaint.ne2d(physical_b16, x*radius, mass*XGPaint.M_sun, z))/amplitude
end

function build_shape_grid(name, model, physical_b16, refinement)
    logx = collect(range(log(1e-7), log(5.1); length=160*refinement+1))
    redshifts = collect(range(0.0, 1.0; length=32*refinement+1))
    logm = sort!(unique(vcat(collect(range(12.0, 15.7; length=64*refinement+1)), [log10(10.0^13.61/0.68)])))
    values = Array{Float64}(undef, length(logx), length(redshifts), length(logm))
    pairs = collect(Iterators.product(eachindex(redshifts), eachindex(logm)))
    println("Building radius-scaled ", name, " shape grid ", size(values), "; threads=", Threads.nthreads())
    flush(stdout)
    Threads.@threads :static for k in eachindex(pairs)
        iz, im = pairs[k]
        mass, z = 10.0^logm[im], redshifts[iz]
        amplitude = column_amplitude(name, model, mass, z)
        for ix in eachindex(logx)
            value = direct_shape(name, physical_b16, mass, z, exp(logx[ix]), amplitude)
            isfinite(value) && value>0 || error("Invalid dimensionless shape")
            values[ix, iz, im] = log(value)
        end
    end
    return (interpolator=interpolate((logx, redshifts, logm), values, Gridded(Linear())),
            logx=logx, redshift=redshifts, logmass=logm, logshape=values)
end

function run_radius_test()
    options = Production.CLI_OPTIONS
    output_dir = get(options, "output_dir", joinpath(@__DIR__, "outputs", "radius_scaled_cache_test"))
    mkpath(output_dir)
    refinement = parse(Int, get(options, "refinement", "1"))
    refinement in (1, 2) || error("Supported refinements: 1 and 2")
    paths = Dict("battaglia16" => options["battaglia16_cache"], "lee2022" => options["lee22_cache"])
    physical_b16 = XGPaint.BattagliaTauProfilePhysical(Omega_c=0.261, Omega_b=0.049, h=0.68)
    # Test points use the old angular coordinate definition to isolate numerical
    # interpolation changes from any change to aperture/large-angle geometry.
    points = Tuple{Float64,Float64,Float64,String}[]
    for line in readlines(joinpath(output_dir, "nearby_catalogue_test_points.csv"))[2:end]
        row, mass, z = parse.(Float64, split(line, ','))
        for radius in (0.04, 0.3, 1.0, 3.0, 5.0)
            push!(points, (mass, z, radius, "nearby_catalogue"))
        end
    end
    rng = MersenneTwister(20260910)
    for k in 1:500
        mass = 10.0^(log10(7.327085124063481e12) + rand(rng)*(log10(3.785236255949654e15)-log10(7.327085124063481e12)))
        z = 10.0^(log10(0.00215)*(1-rand(rng)))
        radius = exp(log(0.001)+rand(rng)*(log(5.0)-log(0.001)))
        push!(points, (mass, z, radius, "held_out_random"))
    end
    output = joinpath(output_dir, "radius_scaled_validation_r$(refinement).csv")
    open(output, "w") do io
        println(io, "profile,population,mass_msun,redshift,r_perp_r200c,effective_x,direct_dm,old_cache_dm,radius_cache_dm,old_relative_error,new_relative_error")
        for name in ["battaglia16", "lee2022"]
            runtime = Production.dm_profile_runtime_configuration((dm_profile=name, lee2022_concentration_mode="none"))
            old = Production.build_dm_interpolator_compatible(runtime.model;
                cache_file=paths[name], overwrite=false, cleanup_nonpositive=true,
                generated_model_family=runtime.generated_model_family, cache_signature=runtime.cache_signature).profile
            grid = build_shape_grid(name, runtime.model, physical_b16, refinement)
            h5open(joinpath(output_dir, "$(name)_radius_shape_r$(refinement).h5"), "w") do h5
                h5["log_radius_r200c"] = grid.logx
                h5["redshift"] = grid.redshift
                h5["log10_mass_msun"] = grid.logmass
                h5["log_dimensionless_shape"] = grid.logshape
                attributes(h5)["axes_in_h5py"] = "logmass,redshift,logradius"
                attributes(h5)["status"] = "experimental numerical test; not installed in production"
                attributes(h5)["profile"] = name
                attributes(h5)["normalization_changed"] = false
            end
            for (mass, z, radius, population) in points
                r200 = XGPaint.R_Δ(runtime.model, mass*XGPaint.M_sun, z, 200)
                theta200 = XGPaint.angular_size(runtime.model, r200, z)
                theta = XGPaint.angular_size(runtime.model, radius*r200, z)
                effective_x = theta/theta200
                direct = runtime.model(theta, mass, z)
                old_value = old(theta, mass, z)
                fixed = column_amplitude(name, runtime.model, mass, z)/(1+z) *
                    exp(grid.interpolator(log(effective_x), z, log10(mass)))
                println(io, join((name, population, mass, z, radius, effective_x, direct, old_value, fixed,
                                  old_value/direct-1, fixed/direct-1), ','))
            end
            flush(io)
            println("Finished validation of ", name)
            flush(stdout)
        end
    end
    println("Saved ", output)
end

run_radius_test()
