#!/usr/bin/env julia
# Evaluate the installed HalfDome implementation; no catalogue, map, or cache is changed.
using XGPaint, Unitful, UnitfulAstro, SHA
include(joinpath(@__DIR__, "..", "lee2022_frb_dm_profile.jl"))

out = joinpath(@__DIR__, "..", "outputs", "lee22_fig5_audit_20260922")
mkpath(out)
max_density_error = 0.0
max_projection_error = 0.0

open(joinpath(out, "halfdome_profiles.csv"), "w") do io
    println(io, "cosmology,h,omega_m,omega_b,log10_mass_hinv,redshift,x,ne_native_m3,ne_paper_m3,n200_paper_m3,native_factor")
    for (name, h, om, ob) in [("HalfDome", 0.68, 0.31, 0.049), ("TNG", 0.6774, 0.3089, 0.0486)]
        native = Lee2022NoConcentrationDMProfile(; Omega_c=om-ob, Omega_b=ob, h=h,
            normalization=:xgpaint_ne2d, n0_pivot=:mcut,
            shape_clip_mass_msun=10.0^14.8/h, redshift_scaling=:physical)
        paper = Lee2022NoConcentrationDMProfile(; Omega_c=om-ob, Omega_b=ob, h=h,
            normalization=:literal, n0_pivot=:mcut,
            shape_clip_mass_msun=10.0^14.8/h, redshift_scaling=:physical)
        wrapped_native = Battaglia16DensityDMProfile(Lee2022XGPaintDMProfile(native))
        wrapped_paper = Battaglia16DensityDMProfile(Lee2022XGPaintDMProfile(paper))
        # Both figure bins, including their full mass span, at a matched snapshot.
        for logmass in sort(unique(vcat(collect(13.2:0.01:13.4), collect(13.8:0.01:14.0)))), z in (0.0, 0.5, 1.0)
            mass = 10.0^logmass/h
            rho = XGPaint.ρ_crit(paper, z)
            n200 = Float64(ustrip(uconvert(u"m^-3", 200*rho*(ob/om)/(0.76*XGPaint.constants.ProtonMass))))
            for x in exp10.(range(log10(0.04), log10(1.5), length=181))
                ne_native = halo_electron_density_m3(wrapped_native, x, mass, z)
                ne_paper = halo_electron_density_m3(wrapped_paper, x, mass, z)
                direct = halo_electron_density_m3(paper, x, mass, z)
                global max_density_error = max(max_density_error, abs(ne_paper/direct - 1))
                @assert isapprox(ne_native/ne_paper, lee2022_normalization_factor(native); rtol=2e-13)
                println(io, join((name,h,om,ob,logmass,z,x,ne_native,ne_paper,n200,lee2022_normalization_factor(native)), ','))
            end
        end
        # The factor also survives the exact finite 1R200c chord integration.
        for logmass in (13.3,13.9), z in (0.0,0.5,1.0), b in (0.01,0.1,0.5,0.99)
            mass = 10.0^logmass/h
            dn = chord_dm_pc_cm3(wrapped_native,b,sqrt(1-b*b),mass,z;rtol=1e-9)
            dp = chord_dm_pc_cm3(wrapped_paper,b,sqrt(1-b*b),mass,z;rtol=1e-9)
            global max_projection_error = max(max_projection_error,abs(dn/dp/lee2022_normalization_factor(native)-1))
        end
    end
end
@assert max_density_error < 2e-12
@assert max_projection_error < 2e-10
open(joinpath(out,"julia_validation.txt"),"w") do io
    println(io,"max_wrapper_vs_direct_density_relative_error=",max_density_error)
    println(io,"max_chord_normalization_relative_error=",max_projection_error)
    println(io,"profile_sha256=",bytes2hex(sha256(read(joinpath(@__DIR__,"..","lee2022_frb_dm_profile.jl")))))
    println(io,"XGPaint_source=",pathof(XGPaint))
end
println("PASS: density mapping and 48 finite-chord normalization checks")
