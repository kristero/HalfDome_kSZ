using Test
include(joinpath(@__DIR__, "simulator/tSZ_visuals/run_halfdome_fullsky_so_noise.jl"))
include(joinpath(@__DIR__, "safe_paint.jl"))

struct OverlapTestProfile <: XGPaint.AbstractProfile{Float64} end
(::OverlapTestProfile)(theta, mass, z) = 1.0 + 0.25*cos(10*theta) + 0.1*z
XGPaint.compute_θmin(::OverlapTestProfile) = 1e-5
XGPaint.compute_θmax(::OverlapTestProfile, mass, z) = 0.15

function main()
    out = isempty(ARGS) || occursin('=', ARGS[1]) ? joinpath(@__DIR__, "validation_plots") : ARGS[1]
    mkpath(out)
    n = 1000
    model = OverlapTestProfile()
    ws = XGPaint.HealpixRingProfileWorkspace{Float64}(Healpix.Resolution(128))
    mass, redshift = fill(1e14, n), fill(0.5, n)
    ra, dec = fill(1.0, n), fill(0.1, n)
    serial = HealpixMap{Float64, RingOrder}(128)
    fill!(serial.pixels, 0.0)
    XGPaint.paintrange!(1:n, serial, ws, model, mass, redshift, ra, dec)
    legacy = HealpixMap{Float64, RingOrder}(128)
    fixed = HealpixMap{Float64, RingOrder}(128)
    @test Threads.nthreads() >= 2
    rows = zeros(5, 3)
    for trial in 1:5
        # Dispatch explicitly to the old unspecialized threaded implementation.
        invoke(XGPaint.paint!, Tuple{Any, Any, Any, Any, Any, Any, Any},
               legacy, ws, model, mass, redshift, ra, dec)
        XGPaint.paint!(fixed, ws, model, mass, redshift, ra, dec)
        denominator = sum(serial.pixels)
        rows[trial, :] = [trial, sum(abs.(legacy.pixels - serial.pixels))/denominator,
                          sum(abs.(fixed.pixels - serial.pixels))/denominator]
        @test fixed.pixels ≈ serial.pixels rtol=1e-12
    end
    writedlm(joinpath(out, "painter_overlap_errors.csv"), rows, ',')
    println("Legacy and ring-locked relative L1 map errors:")
    show(stdout, "text/plain", rows)
    println("\nPASSED: ring-locked painter matches serial reference for overlapping halos.")
end

main()
