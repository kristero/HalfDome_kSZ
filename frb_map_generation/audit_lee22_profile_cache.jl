#!/usr/bin/env julia
# Read-only production-cache audit. No halo catalogue is repainted and neither
# XGPaint nor its caches are changed. Arguments use --output-dir=/path syntax.
module Production
include(joinpath(@__DIR__, "generate_halfdome_z1_dm_mass_windows.jl"))
end

using XGPaint
using Unitful
using UnitfulAstro
using Printf
using SHA

function main()
    options = Production.CLI_OPTIONS
    output_dir = get(options, "output_dir", joinpath(@__DIR__, "outputs", "lee22_power_audit_20260909"))
    mkpath(output_dir)
    paths = Dict(
        "battaglia16" => get(options, "battaglia16_cache", joinpath(@__DIR__, "outputs", "shared_xgpaint_dm_cache.jld2")),
        "lee2022" => get(options, "lee22_cache", joinpath(@__DIR__, "outputs", "halfdome_frb_inputs", "lee2022_tablea2_noconcentration_m200c_profile_owned_los_v2_dm_cache.jld2")),
    )
    runtimes = Dict(name => Production.dm_profile_runtime_configuration(
        (dm_profile=name, lee2022_concentration_mode="none"),
    ) for name in keys(paths))
    interpolators = Dict(name => Production.build_dm_interpolator_compatible(
        runtimes[name].model; cache_file=paths[name], overwrite=false,
        cleanup_nonpositive=true,
        generated_model_family=runtimes[name].generated_model_family,
        cache_signature=runtimes[name].cache_signature,
    ).profile for name in keys(paths))
    physical_b16 = XGPaint.BattagliaTauProfilePhysical(Omega_c=0.261, Omega_b=0.049, h=0.68)
    # Dense sampling close to z=0 probes the coarsest relative redshift spacing
    # of the interpolation table, in addition to the requested mass/z grid.
    redshifts = [0.0022, 0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 4.0]
    logmasses = [12.5, log10(7.327085124063481e12), 13.0, 13.5, 14.0, 14.5, 15.0, 15.5]
    radii = [0.001, 0.01, 0.04, 0.1, 0.3, 0.5, 1.0, 1.34, 2.0, 3.0, 5.0]
    output = joinpath(output_dir, "production_cache_vs_direct_profiles.csv")
    open(output, "w") do io
        println(io, "profile,log10_mass_msun,redshift,r_perp_r200c,theta_rad,cached_dm_pc_cm3,direct_dm_pc_cm3,b16_physical_dm_pc_cm3,cached_over_direct")
        for name in ["battaglia16", "lee2022"]
            model = runtimes[name].model
            for z in redshifts, logmass in logmasses, radius in radii
                mass = 10.0^logmass
                r200 = XGPaint.R_Δ(model, mass * XGPaint.M_sun, z, 200)
                # This matches the actual generator: theta=angular_size(r,R,z),
                # then the production cached angular model is evaluated there.
                theta = XGPaint.angular_size(model, radius * r200, z)
                cached = interpolators[name](theta, mass, z)
                direct = model(theta, mass, z)
                physical = ustrip(u"pc*cm^-3", XGPaint.ne2d(
                    physical_b16, radius*r200, mass*XGPaint.M_sun, z,
                )) / (1+z)
                all(isfinite, (cached, direct, physical)) || error("Non-finite grid value")
                min(cached, direct, physical) > 0 || error("Non-positive grid value")
                println(io, join((name, logmass, z, radius, theta, cached, direct, physical, cached/direct), ','))
            end
            println("Completed ", name, " cache/direct grid")
            flush(stdout)
        end
    end
    open(joinpath(output_dir, "profile_audit_provenance.txt"), "w") do io
        println(io, "xgpaint_path=", pathof(XGPaint))
        println(io, "julia_version=", VERSION)
        println(io, "project=", Base.active_project())
        println(io, "scope=read-only cache and direct profile evaluation; no rebuild, no halo selection changes")
        for name in sort(collect(keys(paths)))
            println(io, name, "_cache=", paths[name])
            println(io, name, "_cache_sha256=", open(sha256, paths[name]) |> bytes2hex)
        end
    end
    nearby_path = joinpath(output_dir, "nearby_catalogue_test_points.csv")
    if isfile(nearby_path)
        open(joinpath(output_dir, "nearby_halo_cache_vs_direct.csv"), "w") do io
            println(io, "profile,catalogue_row_zero_based,mass_msun,redshift,r_perp_r200c,cached_dm_pc_cm3,direct_dm_pc_cm3,cached_over_direct")
            for line in readlines(nearby_path)[2:end]
                row, mass, z = parse.(Float64, split(line, ','))
                for name in ["battaglia16", "lee2022"], radius in [0.04, 1.0, 3.0]
                    model = runtimes[name].model
                    r200 = XGPaint.R_Δ(model, mass*XGPaint.M_sun, z, 200)
                    theta = XGPaint.angular_size(model, radius*r200, z)
                    direct = model(theta, mass, z)
                    cached = interpolators[name](theta, mass, z)
                    println(io, join((name, Int(row), mass, z, radius, cached, direct, cached/direct), ','))
                end
            end
        end
        println("Completed nearby catalogue halo checks")
    end
    # A physical consistency diagnostic, NOT a renormalization prescription.
    # Convert free-electron number to ionized-gas-equivalent mass with primordial
    # fully ionized H/He (mu_e=2/(1+X_H)), and divide by (Omega_b/Omega_m)*M200c.
    # Compare both the full sphere and only the fitted radial shell .04-1 R200c.
    open(joinpath(output_dir, "ionized_gas_budget_and_los_checks.csv"), "w") do io
        println(io, "log10_mass_msun,redshift,lee_gas_equivalent_over_fbM_r0_to_1,lee_gas_equivalent_over_fbM_r004_to_1,b16_gas_equivalent_over_fbM_r0_to_1,lee_los_split_over_default")
        quad = Production.lee2022_quadgk_function()
        for z in [0.0, 0.1, 0.5, 1.0], logmass in [12.5, 13.0, 13.5, 14.0, 14.5, 15.0, 15.5]
            mass = 10.0^logmass
            par = Production.lee2022_no_concentration_parameters(mass, z, 0.68)
            lee_shape(x) = Production.lee2022_dimensionless_density(x, par.x_c, par.alpha, par.beta_prime, par.gamma)
            density_factor = 3*(2/(1+0.76))/0.76 * par.n0
            lee_full = density_factor*quad(x -> x^2*lee_shape(x), 0.0, 1.0; rtol=1e-10)[1]
            lee_shell = density_factor*quad(x -> x^2*lee_shape(x), 0.04, 1.0; rtol=1e-10)[1]
            b = XGPaint.get_params(runtimes["battaglia16"].model, mass*XGPaint.M_sun, z)
            b16_full = 0.9*3*b.P₀/200 * quad(
                x -> x^2*XGPaint.generalized_nfw(x, b.xc, b.α, b.β, b.γ),
                0.0, 1.0; rtol=1e-10,
            )[1]
            # Independently segmented LOS integration resolves the core before
            # extending to the large numerical upper limit; checks missed peaks.
            xperp = 0.04
            split_los = 2*quad(y -> lee_shape(sqrt(xperp^2+y^2)),
                0.0, 0.01, 0.1, 1.0, 10.0, 100.0, 1e3, 1e5; rtol=1e-10)[1]
            default_los = Production.lee2022_dimensionless_los(xperp, par.x_c, par.alpha, par.beta_prime, par.gamma)
            println(io, join((logmass, z, lee_full, lee_shell, b16_full, split_los/default_los), ','))
        end
    end
    open(joinpath(output_dir, "three_dimensional_density_profiles.csv"), "w") do io
        println(io, "log10_mass_msun,redshift,r_r200c,b16_ne_cm3,lee22_ne_cm3")
        b16 = runtimes["battaglia16"].model
        lee = runtimes["lee2022"].model
        xh = 0.76
        mp = XGPaint.constants.ProtonMass
        # Same conversion as XGPaint.ne2d, but applied before projection.
        mass_per_electron = XGPaint.constants.ElectronMass +
            (2*xh/(1+xh))*mp + ((1-xh)/(2*(1+xh)))*(4*mp)
        for z in [0.0, 0.1, 0.5, 1.0, 2.0, 3.0, 4.0], logmass in [12.5, 13.0, 13.5, 14.0, 14.5, 15.0, 15.5]
            mass = 10.0^logmass
            b = XGPaint.get_params(b16, mass*XGPaint.M_sun, z)
            l = Production.lee2022_no_concentration_parameters(mass, z, 0.68)
            rho_critical = XGPaint.ρ_crit(lee, z)
            n200 = 200*rho_critical/(xh*mp)*(lee.omega_b/lee.omega_m)
            for radius in exp.(range(log(0.001), log(3.0); length=100))
                b16_rho = b.P₀*XGPaint.generalized_nfw(radius, b.xc, b.α, b.β, b.γ)*b16.f_b*rho_critical
                b16_ne = ustrip(u"cm^-3", 0.9*b16_rho/mass_per_electron)
                lee_ne = ustrip(u"cm^-3", l.n0*n200*Production.lee2022_dimensionless_density(
                    radius, l.x_c, l.alpha, l.beta_prime, l.gamma,
                ))
                println(io, join((logmass, z, radius, b16_ne, lee_ne), ','))
            end
        end
    end
    println("Saved ", output)
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
