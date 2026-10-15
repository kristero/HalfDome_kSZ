# Numerical regression against the existing serial XGPaint implementation.
# Run with JULIA_NUM_THREADS=4 and the same Julia project as run_local.sh.
using Test, LinearAlgebra
include(joinpath(@__DIR__, "paint_maps.jl"))

@testset "Paired maps agree with direct serial XGPaint" begin
    cfg = TOML.parsefile(joinpath(@__DIR__, "settings.toml"))
    cfg["nside"] = 128
    cfg["interpolation"] = Dict("n_radius"=>96, "n_redshift"=>32, "n_mass"=>32,
                                "radius_min"=>1e-7, "validation_rtol"=>0.01)
    bounds = Dict("mass_min_msun"=>1e13, "mass_max_msun"=>1e15, "z_min"=>0.01, "z_max"=>0.2)
    c = cfg["cosmology"]
    model = BattagliaTauProfile(Omega_b=c["Omega_b"], Omega_c=c["Omega_c"], h=c["h"])
    # Include overlapping halos, an equatorial seam, and a polar cap.
    mass = [2e14, 4e14, 3e14, 8e14]
    z = [0.035, 0.05, 0.10, 0.02]
    theta, phi = [pi/2, pi/2+0.01, 0.015, pi-0.01], [0.002, 2pi-0.003, 2.1, 5.0]
    pos = hcat([collect(Healpix.ang2vec(theta[i],phi[i])) .* 100 for i in eachindex(mass)]...)
    vel = pos .* 3  # Each halo has +300 km/s radial velocity.
    mktempdir() do out
        itp, _ = make_table(model, cfg, bounds, out)
        tau = HealpixMap{Float64,RingOrder}(cfg["nside"])
        ksz = HealpixMap{Float64,RingOrder}(cfg["nside"])
        direct = HealpixMap{Float64,RingOrder}(cfg["nside"])
        fill!(tau.pixels,0); fill!(ksz.pixels,0); fill!(direct.pixels,0)
        workspace = XGPaint.HealpixRingProfileWorkspace(tau.resolution)
        paint_pair!(tau, ksz, workspace, itp, model, mass, z, pos, vel, cfg)
        # Serial path avoids the shared-pixel writes in threaded XGPaint.paint!.
        XGPaint.paintrange!(1:length(mass), direct, workspace, model,
                           mass, z, phi, pi/2 .- theta)
        @test sum(direct.pixels) > 0
        @test norm(tau.pixels-direct.pixels)/norm(direct.pixels) < 0.01
        @test maximum(abs.(ksz.pixels .+ tau.pixels .* (300/299792.458))) < 1e-15
        # Duplicate overlapping objects must double both maps, without lost writes.
        first_tau, first_ksz = copy(tau.pixels), copy(ksz.pixels)
        paint_pair!(tau, ksz, workspace, itp, model, mass, z, pos, vel, cfg)
        @test isapprox(tau.pixels, 2 .* first_tau; rtol=1e-14)
        @test isapprox(ksz.pixels, 2 .* first_ksz; rtol=1e-14)
    end
end
