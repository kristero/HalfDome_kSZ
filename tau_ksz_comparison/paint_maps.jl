#!/usr/bin/env julia
using TOML, SHA, Random, HDF5, Healpix, Interpolations, XGPaint

"""XGPaint's angular R200c, using physical M200c in solar masses."""
theta200(model, mass, z) = XGPaint.compute_θmax(model, mass * XGPaint.M_sun, z; mult=1)

function make_table(model, cfg, bounds, out)
    grid = cfg["interpolation"]
    lr = range(log(grid["radius_min"]), log(cfg["aperture_r200c"]);
               length=grid["n_radius"])
    lz = range(log1p(bounds["z_min"]), log1p(bounds["z_max"]);
               length=grid["n_redshift"])
    lm = range(log10(bounds["mass_min_msun"]), log10(bounds["mass_max_msun"]);
               length=grid["n_mass"])
    cache = joinpath(out, "profile_table.h5")
    if isfile(cache)
        values = h5open(cache, "r") do f
            read(f["log_tau"])
        end
    else
        println("Tabulating the existing XGPaint BattagliaTauProfile: ",
                (length(lr), length(lz), length(lm))); flush(stdout)
        values = Array{Float64}(undef, length(lr), length(lz), length(lm))
        Threads.@threads for im in eachindex(lm)
            mass = 10.0^lm[im]
            for iz in eachindex(lz)
                z = expm1(lz[iz])
                size = theta200(model, mass, z)
                for ir in eachindex(lr)
                    tau = Float64(model(exp(lr[ir]) * size, mass, z))
                    isfinite(tau) && tau > 0 || error("Nonpositive/nonfinite XGPaint tau")
                    values[ir, iz, im] = log(tau)
                end
            end
        end
        h5open(cache * ".partial", "w") do f
            f["log_tau"] = values
            f["log_radius"] = collect(lr)
            f["log1p_z"] = collect(lz)
            f["log10_mass_msun"] = collect(lm)
        end
        mv(cache * ".partial", cache)
    end
    itp = Interpolations.scale(
        Interpolations.interpolate(values, BSpline(Cubic(Line(OnGrid())))), lr, lz, lm)
    # Compare off-grid interpolation with the original profile, including edges.
    rng = MersenneTwister(100)
    worst = 0.0
    for i in 1:150
        t = i <= 8 ? [(i-1) & 1, ((i-1) >> 1) & 1, ((i-1) >> 2) & 1] : rand(rng, 3)
        rlog = clamp(first(lr) + t[1] * (last(lr)-first(lr)), first(lr), last(lr))
        zlog = clamp(first(lz) + t[2] * (last(lz)-first(lz)), first(lz), last(lz))
        mlog = clamp(first(lm) + t[3] * (last(lm)-first(lm)), first(lm), last(lm))
        mass, z = 10.0^mlog, expm1(zlog)
        direct = model(exp(rlog) * theta200(model, mass, z), mass, z)
        worst = max(worst, abs(exp(itp(rlog, zlog, mlog))/direct - 1))
    end
    println("Maximum interpolation error over 150 checks: ", worst); flush(stdout)
    worst <= grid["validation_rtol"] || error("Increase interpolation grid resolution")
    return itp, worst
end

"""Each worker owns a disjoint band of rings, so overlapping halos cannot race.

The angular geometry and profile are from XGPaint. A pixel's tau is evaluated
once, then added to tau and to kSZ with -v_los/c. No velocity randomization.
"""
function paint_pair!(tau_map, ksz_map, workspace, itp, model, mass, z, pos, vel, cfg)
    count = length(mass)
    theta = Vector{Float64}(undef, count)
    phi = similar(theta)
    sizes = similar(theta)
    v_over_c = similar(theta)
    logm, logz = log10.(mass), log1p.(z)
    starts, stops = zeros(Int, count), zeros(Int, count)
    Threads.@threads for i in 1:count
        r = sqrt(sum(abs2, @view pos[:, i]))
        theta[i], phi[i] = Healpix.vec2ang(pos[1,i]/r, pos[2,i]/r, pos[3,i]/r)
        sizes[i] = theta200(model, mass[i], z[i])
        aperture = cfg["aperture_r200c"] * sizes[i]
        aperture < pi || error("A halo aperture spans >= pi radians")
        starts[i], stops[i] = XGPaint.get_relevant_rings(workspace.res, theta[i], aperture)
        v_over_c[i] = -sum(pos[:,i] .* vel[:,i]) / r / 299792.458 * cfg["velocity_to_kms"]
    end
    nrings = length(workspace.ring_first_pixels)
    workers = Threads.nthreads()
    rmin = cfg["interpolation"]["radius_min"]
    Threads.@threads :static for worker in 1:workers
        first_ring = fld((worker-1)*nrings, workers) + 1
        last_ring = fld(worker*nrings, workers)
        for i in 1:count
            lo, hi = max(starts[i], first_ring), min(stops[i], last_ring)
            lo > hi && continue
            aperture = cfg["aperture_r200c"] * sizes[i]
            x0, y0, z0 = Healpix.ang2vec(theta[i], phi[i])
            for ring in lo:hi
                ranges = XGPaint.get_ring_disc_ranges(workspace, ring, theta[i], phi[i], aperture)
                first_pixel = workspace.ring_first_pixels[ring]
                for range in ranges, pixel in range
                    p = first_pixel + pixel - 1
                    x1, y1, z1 = Healpix.pix2vecRing(workspace.res, p)
                    d2 = (x1-x0)^2 + (y1-y0)^2 + (z1-z0)^2
                    angle = acos(clamp(1-d2/2, -1.0, 1.0))
                    angle < aperture || continue
                    logr = log(max(angle/sizes[i], rmin))
                    tau = exp(itp(logr, logz[i], logm[i]))
                    tau_map.pixels[p] += tau
                    ksz_map.pixels[p] += tau * v_over_c[i]
                end
            end
        end
    end
end

function main()
    length(ARGS) == 2 || error("Usage: paint_maps.jl effective_settings.toml catalog_bounds.toml")
    cfg, bounds = TOML.parsefile(ARGS[1]), TOML.parsefile(ARGS[2])
    realpath(dirname(dirname(pathof(XGPaint)))) == realpath(cfg["xgpaint"]) ||
        error("Julia loaded a different XGPaint checkout: $(pathof(XGPaint))")
    out = cfg["output"]
    c = cfg["cosmology"]
    model = BattagliaTauProfile(Omega_b=c["Omega_b"], Omega_c=c["Omega_c"], h=c["h"])
    println("XGPaint source: ", pathof(XGPaint)); flush(stdout)
    itp, interpolation_error = make_table(model, cfg, bounds, out)
    tau_map = HealpixMap{Float64,RingOrder}(cfg["nside"])
    ksz_map = HealpixMap{Float64,RingOrder}(cfg["nside"])
    fill!(tau_map.pixels, 0); fill!(ksz_map.pixels, 0)
    workspace = XGPaint.HealpixRingProfileWorkspace(tau_map.resolution)
    painted = 0
    h5open(cfg["catalog"], "r") do f
        total = bounds["rows_scanned"]
        for start in 1:cfg["chunk_size"]:total
            stop = min(start+cfg["chunk_size"]-1, total)
            rows = start:stop
            mass = Float64.(f["halo_mass_m200c"][rows])
            cfg["catalog_masses_msun_h"] && (mass ./= c["h"])
            z = Float64.(f["redshift"][rows])
            pos, vel = Float64.(f["Position"][:,rows]), Float64.(f["Velocity"][:,rows])
            keep = [isfinite(mass[i]) && mass[i]>0 && isfinite(z[i]) && z[i]>0 &&
                    all(isfinite, pos[:,i]) && sum(abs2, pos[:,i])>0 for i in eachindex(mass)]
            all(isfinite, vel[:,keep]) || error("Invalid velocity for a valid halo")
            paint_pair!(tau_map, ksz_map, workspace, itp, model,
                        mass[keep], z[keep], pos[:,keep], vel[:,keep], cfg)
            painted += sum(keep)
            println("Scanned $stop/$total; painted $painted halos"); flush(stdout)
        end
    end
    painted == bounds["valid_halos"] || error("Scan and paint halo counts differ")
    all(isfinite, tau_map.pixels) && all(isfinite, ksz_map.pixels) || error("Nonfinite map")
    dest = joinpath(out, "maps.h5")
    h5open(dest * ".partial", "w") do f
        f["tau"] = tau_map.pixels
        f["ksz_deltaT_over_T"] = ksz_map.pixels
        attributes(f)["ordering"] = "RING"
        attributes(f)["nside"] = cfg["nside"]
        attributes(f)["painted_halos"] = painted
        attributes(f)["complete_catalog"] = bounds["complete_catalog"]
        attributes(f)["interpolation_max_relative_error"] = interpolation_error
        attributes(f)["beam_noise_mask"] = "none"
        attributes(f)["aperture_geometry"] = "projected 4R200c; native XGPaint LOS"
    end
    mv(dest * ".partial", dest)
    println("Saved ", dest)
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
