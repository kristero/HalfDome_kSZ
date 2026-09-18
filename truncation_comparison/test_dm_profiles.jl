# Lee22 no-c (XGPaint-native) and Battaglia16 DM profiles with the ChordMeanProfile wrapper,
# checked against the FRB code's own SphericalChordDMProfile at X = 4.
using XGPaint
module FRB
include(joinpath(@__DIR__, "..", "frb_map_generation", "generate_halfdome_z1_dm_mass_windows.jl"))
end
include(joinpath(@__DIR__, "spherical_truncation_profiles.jl"))
using .SphericalTruncation

const H = 0.6774; const OMEGA_B = 0.0486; const OMEGA_C = 0.2603; const X = 4.0
clip = 10.0^14.8 / H
lee_inner = FRB.Lee2022XGPaintDMProfile(FRB.Lee2022NoConcentrationDMProfile(
    Omega_c=OMEGA_C, Omega_b=OMEGA_B, h=H, normalization=:xgpaint_ne2d, n0_pivot=:mcut,
    shape_clip_mass_msun=clip, redshift_scaling=:physical))
b16_inner = FRB.HaloDMProfile(BattagliaTauProfile(Omega_c=OMEGA_C, Omega_b=OMEGA_B, h=H))
println("Lee22 provenance tokens: ", FRB.lee2022_variant_tokens(lee_inner.lee))
for (name, inner) in (("lee22_noc_xgpnative", lee_inner), ("battaglia16", b16_inner))
    mine = ChordMeanProfile(inner, X)
    frb = FRB.SphericalChordDMProfile(FRB.Battaglia16DensityDMProfile(inner), X)
    println("== ", name)
    for (M, z) in ((7e12, 0.1), (3e13, 0.5), (1e14, 0.5), (1e15, 0.3), (3e15, 1.5))
        θ200 = theta_r200c(inner, M, z); θmax = X * θ200
        for x in (0.01, 0.5, 1.0, 2.0, 3.5)
            θ = x * θ200
            proj = Float64(inner(θ, M, z))
            mine_sph = spherical_value_direct(mine, θ, M, z)
            frb_sph = FRB.spherical_dm_pc_cm3(frb, θ, M, z)
            rebuilt = mine(θ, M, z) * chord_factor(θ, θmax, X)
            isapprox(mine_sph, frb_sph; rtol=1e-6) || error("$name: mine $mine_sph vs FRB $frb_sph at M=$M z=$z x=$x")
            isapprox(mine_sph, rebuilt; rtol=1e-8) || error("rebuild mismatch")
            x in (0.01, 1.0, 3.5) && println("  M=$M z=$z x=$x  DM_proj=$(round(proj, sigdigits=5)) pc/cm3  sphere/proj=$(round(mine_sph/proj, digits=4))")
        end
    end
end
println("PASS: ChordMeanProfile matches the FRB SphericalChordDMProfile for both DM profiles at X=4")
