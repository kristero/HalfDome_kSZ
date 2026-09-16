#!/usr/bin/env julia
# Tests of published density coefficients, units, concentration, and extrapolation.
# Finite-LOS sensitivity is measured explicitly, never hidden by rescaling gas.
include(joinpath(@__DIR__, "paint_halfdome_matched_profile_dm_map.jl"))

function preferred_profile_tests()
    options = parse_options(ARGS)
    out = abspath(option(options, "output_dir", joinpath(@__DIR__, "outputs", "lee22_preferred_tests")))
    mkpath(out)
    ProfileSupport.run_lee2022_no_concentration_profile_self_test()
    PS = ProfileSupport
    h = 0.68
    cut = 10.0^13.75 / h
    @assert isapprox(PS.duffy2008_c200c(2e12/h, 0.0, h), 5.71; rtol=1e-14)
    p = PS.lee2022_concentration_parameters(cut, 0.0, h; concentration=10.0)
    @assert isapprox(p.n0, 15.7; rtol=1e-14)
    @assert isapprox(p.x_c, 2.2; rtol=1e-14)
    @assert isapprox(p.beta_prime, 7.5; rtol=1e-14)
    for mass in (cut/2, cut*2), z in (0.0, 0.7)
        at10 = PS.lee2022_concentration_parameters(mass, z, h; concentration=10.0)
        at5 = PS.lee2022_concentration_parameters(mass, z, h; concentration=5.0)
        for (field, exponent) in ((:n0, 0.63), (:x_c, -1.37), (:beta_prime, -1.11))
            @assert isapprox(getproperty(at5, field)/getproperty(at10, field), 0.5^exponent; rtol=1e-14)
        end
    end
    left = PS.lee2022_concentration_parameters(cut*(1-1e-9), 0.3, h)
    right = PS.lee2022_concentration_parameters(cut*(1+1e-9), 0.3, h)
    for field in (:n0, :x_c, :beta_prime)
        @assert isapprox(getproperty(left, field), getproperty(right, field); rtol=1e-8)
    end
    best = PS.Lee2022ConcentrationDMProfile()
    legacy = PS.Lee2022NoConcentrationDMProfile()
    b16 = PS.dm_profile_runtime_configuration((dm_profile="battaglia16", lee2022_concentration_mode="none")).model
    physical_b16 = XGPaint.BattagliaTauProfilePhysical(Omega_c=0.261, Omega_b=0.049, h=0.68)
    quad = PS.lee2022_quadgk_function()
    rows = 0
    divergent = 0
    open(joinpath(out, "preferred_profile_grid.csv"), "w") do io
        println(io, "profile,log10_mass_msun,redshift,radius_r200c,ne_cm3,column_pc_cm3,spherical3_column_pc_cm3,outer_slope,column_converges,los_1e5_over_1e4,gas_equivalent_fraction_r200c")
        for logm in (12.5, 13.0, 13.5, 14.0, 14.5, 15.0, 15.5), z in (0.0, 0.1, 0.5, 1.0, 2.0, 3.0, 4.0)
            mass = 10.0^logm
            for (label, model) in (("battaglia16", b16), ("lee22_legacy", legacy), ("lee22_preferred_duffy", best))
                amplitude = radius_column_amplitude(model, mass, z)
                r200 = XGPaint.R_Δ(model, mass*XGPaint.M_sun, z, 200)
                radius_pc = Float64(ustrip(u"pc", r200))
                shape = radius_density_shape(model, mass, z)
                if model isa PS.AbstractLee2022DMProfile
                    p = PS.lee2022_parameters(model, mass, z)
                    outer = p.beta_prime - p.gamma
                    # Integrate gas-equivalent density using a primordial mass
                    # per free electron. This is a convention diagnostic, not
                    # an imposed upper bound or a renormalization instruction.
                    n200 = 200*XGPaint.ρ_crit(model,z)/(0.76*XGPaint.constants.ProtonMass)*(0.049/0.31)
                    mu_e = XGPaint.constants.ProtonMass/((1+0.76)/2)
                    integral = quad(x -> shape(x)*x*x, 0.0, 0.04, 1.0; rtol=1e-8)[1]
                    gas = Float64(ustrip(4pi*r200^3*n200*p.n0*mu_e*integral/(mass*XGPaint.M_sun*(0.049/0.31))))
                    divergent += label == "lee22_preferred_duffy" && outer <= 1
                else
                    p = XGPaint.get_params(model, mass*XGPaint.M_sun, z)
                    outer = p.β
                    gas = NaN
                end
                for x in exp.(range(log(0.001), log(5.0); length=90))
                    if model isa PS.AbstractLee2022DMProfile
                        # Deliberately evaluate both finite bounds even when the
                        # infinite integral does not exist, to diagnose it.
                        s1 = PS.lee2022_dimensionless_los(x,p.x_c,p.alpha,p.beta_prime,p.gamma; los_max_r200c=1e4)
                        s2 = PS.lee2022_dimensionless_los(x,p.x_c,p.alpha,p.beta_prime,p.gamma; los_max_r200c=1e5)
                        column = amplitude*s2/(1+z)
                        ratio = s2/s1
                        density = amplitude/radius_pc * shape(x)
                    else
                        column = amplitude*radius_direct_shape(model,physical_b16,mass,z,x,amplitude)/(1+z)
                        ratio = NaN
                        density = amplitude/radius_pc * shape(x)
                    end
                    sphere_column = x < 3 ? amplitude/(1+z)*2sqrt(9-x*x)*
                        spherical_chord_mean(model,mass,z,x,3.0) : 0.0
                    isfinite(column) && column > 0 || error("Invalid direct column")
                    isfinite(sphere_column) && sphere_column >= 0 || error("Invalid spherical column")
                    println(io, join((label,logm,z,x,density,column,sphere_column,outer,outer>1,ratio,gas), ','))
                    rows += 1
                end
            end
        end
    end
    open(joinpath(out, "preferred_profile_test_status.txt"), "w") do io
        println(io, "coefficient_and_regression_tests=PASS")
        println(io, "grid_rows=$(rows)")
        println(io, "preferred_divergent_mass_redshift_pairs=$(divergent)")
        println(io, "preferred_untruncated_full_mass_map=REJECTED_DIVERGENT_TAILS")
        println(io, "user_approved_spherical_cut_r200c=3")
        println(io, "legacy_n0_pivot=1e14_physical_Msun_unchanged")
        println(io, "preferred_n0_pivot=10^13.75_Msun_per_h_per_Eq12")
        println(io, "scatter=none; concentration=Duffy08 median proxy, not Klypin catalogue values")
    end
    println("PASS coefficient tests; diagnosed $(divergent) divergent preferred-profile mass/redshift pairs.")
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    preferred_profile_tests()
end
