#!/usr/bin/env julia
# verify_5 (adversarial check of claim C6): repaint the full-HalfDome 3R200c kSZ map with the LITERAL
# Lee22 no-c density (eq. 9 as printed) split into redshift bins, plus two TNG-calibrated variants
#   n_e -> C_TNG(z) n_e,literal,  C_TNG = 0.2841790493880425 (1+z)^2.483190184608004   (F_sz_budget fit)
#   (a) extrapolated to all z, (b) frozen at z = 2 above z = 2.
# Geometry, chord tables, amplitudes and velocities are exactly those of the published painted maps
# (lee22_pdf_ksz_comparison/paint_ksz.jl, whose functions are included read-only; its main() is guarded).
# Usage: julia -t 20 paint_ksz_literal_zsplit_ctng.jl <out_dir> [--nside=4096] [--max-rows=0] [--validate-only]
using XGPaint, Unitful, UnitfulAstro, Healpix, HDF5, Interpolations, TOML, Random
const REPO = "/home/cbllover/HalfDome/frb_map_generation"
include(joinpath(REPO, "lee22_pdf_ksz_comparison", "paint_ksz.jl"))

const KSZ_DIR = joinpath(REPO, "outputs", "lee22_pdf_ksz_comparison_20260922", "ksz")
const C0_TNG = 0.2841790493880425
const K_TNG = 2.483190184608004
const ZEDGES = (0.0, 0.1, 0.25, 0.5, 1.0, Inf)
const MAPNAMES = ("lit_z0p00_0p10", "lit_z0p10_0p25", "lit_z0p25_0p50", "lit_z0p50_1p00", "lit_z1p00_inf",
                  "litC_tng_extrap", "litC_tng_frozen_z2")
ctng(z) = C0_TNG * (1 + z)^K_TNG

function getarg(key, default)
    for a in ARGS
        startswith(a, "--$key=") && return split(a, "="; limit=2)[2]
    end
    return default
end

function zbin(z)
    for b in 1:5
        ZEDGES[b] <= z < ZEDGES[b+1] && return b
    end
    error("bad z $z")
end

"""7 maps; the Lee22 chord-shape table is evaluated once per pixel."""
function paint_chunk7!(maps, workspace, table, lee_model, lit_over_xgp, mass, z, pos, vel, sphere, rmin)
    n = length(mass); theta = zeros(n); phi = zeros(n); sizes = zeros(n); vc = zeros(n)
    amp = zeros(n); starts = zeros(Int, n); stops = zeros(Int, n); bins = zeros(Int, n)
    w6 = zeros(n); w7 = zeros(n)
    logm = log10.(mass); logz = log1p.(z)
    Threads.@threads for i in eachindex(mass)
        r = sqrt(sum(abs2, @view pos[:, i])); direction = pos[:, i] ./ r
        theta[i], phi[i] = Healpix.vec2ang(direction...)
        r200 = XGPaint.R_Δ(lee_model, mass[i] * XGPaint.M_sun, z[i], 200)
        sizes[i] = Float64(XGPaint.angular_size(lee_model, sphere * r200, z[i]))
        starts[i], stops[i] = XGPaint.get_relevant_rings(workspace.res, theta[i], sizes[i])
        vc[i] = -sum(direction .* vel[:, i]) / 299792.458
        _, a = electron_amplitude(lee_model, mass[i], z[i])
        amp[i] = a * lit_over_xgp                  # literal (eq. 9 printed) amplitude
        bins[i] = zbin(z[i])
        w6[i] = ctng(z[i]); w7[i] = ctng(min(z[i], 2.0))
    end
    nrings = length(workspace.ring_first_pixels); workers = Threads.nthreads()
    Threads.@threads :static for worker in 1:workers
        first_ring = fld((worker - 1) * nrings, workers) + 1; last_ring = fld(worker * nrings, workers)
        for i in eachindex(mass)
            lo = max(starts[i], first_ring); hi = min(stops[i], last_ring); lo > hi && continue
            x0, y0, z0 = Healpix.ang2vec(theta[i], phi[i])
            mb = maps[bins[i]].pixels; m6 = maps[6].pixels; m7 = maps[7].pixels
            for ring in lo:hi
                ranges = XGPaint.get_ring_disc_ranges(workspace, ring, theta[i], phi[i], sizes[i])
                first_pixel = workspace.ring_first_pixels[ring]
                for range in ranges, pixel in range
                    p = first_pixel + pixel - 1
                    x1, y1, z1 = Healpix.pix2vecRing(workspace.res, p)
                    d2 = (x1 - x0)^2 + (y1 - y0)^2 + (z1 - z0)^2
                    angle = 2asin(min(1.0, sqrt(d2) / 2)); u = angle / sizes[i]
                    u < 1 || continue
                    chord = 2sphere * sqrt(max(0.0, 1 - u * u))
                    logu = log(max(rmin, u))
                    s = chord * amp[i] * exp(table.itp(logu, logz[i], logm[i])) * vc[i]
                    mb[p] += s; m6[p] += s * w6[i]; m7[p] += s * w7[i]
                end
            end
        end
    end
    return nothing
end

function main7()
    out = ARGS[1]; mkpath(out)
    nside = parse(Int, getarg("nside", "4096")); max_rows = parse(Int, getarg("max-rows", "0"))
    cfg = TOML.parsefile(joinpath(KSZ_DIR, "settings.toml")); bounds = TOML.parsefile(joinpath(KSZ_DIR, "bounds.toml"))
    model_pair = models(cfg)
    # reads the existing chord_shape_tables.h5 (no write: the file exists) and re-validates 800 points per model
    tables = make_tables(cfg, bounds, model_pair, KSZ_DIR)
    lee = model_pair[1]
    lit_over_xgp = 1 / lee2022_normalization_factor(lee.lee)
    println("literal/xgpnative amplitude = $lit_over_xgp"); flush(stdout)
    sphere = cfg["sphere_r200c"]; rmin = cfg["interpolation"]["radius_fraction_min"]
    # spot values of the literal amplitude for the independent Python check
    open(joinpath(out, "spot_amplitudes.txt"), "w") do io
        println(io, "# mass_msun z R200c_m theta200c_rad literal_amp_(sigmaT*n_e0*R200c)_dimless chordmean_logu_-3 chordmean_logu_-1")
        for mass in (1e13, 1e14, 1e15), zz in (0.05, 0.5, 2.0)
            par, a = electron_amplitude(lee, mass, zz)
            r200 = Float64(ustrip(uconvert(u"m", XGPaint.R_Δ(lee, mass * XGPaint.M_sun, zz, 200))))
            th = Float64(XGPaint.angular_size(lee, XGPaint.R_Δ(lee, mass * XGPaint.M_sun, zz, 200), zz))
            println(io, "$mass $zz $r200 $th $(a*lit_over_xgp) $(exp(tables[1].itp(-3.0, log1p(zz), log10(mass)))) $(exp(tables[1].itp(-1.0, log1p(zz), log10(mass))))")
        end
    end
    "--validate-only" in ARGS && return
    dest = joinpath(out, "ksz_literal_zsplit_ctng_nside$(nside).h5"); isfile(dest) && error("Preserving completed maps $dest")
    maps = [HealpixMap{Float64,RingOrder}(nside) for _ in 1:7]
    foreach(m -> fill!(m.pixels, 0.0), maps); workspace = XGPaint.HealpixRingProfileWorkspace(maps[1].resolution)
    count = 0; t0 = time()
    total = max_rows == 0 ? bounds["total_catalog_rows"] : max_rows
    h5open(cfg["catalog"], "r") do f
        for start in 1:cfg["chunk_size"]:total
            stop = min(start + cfg["chunk_size"] - 1, total); rows = start:stop
            mass = Float64.(f["halo_mass_m200c"][rows]) ./ cfg["cosmology"]["h"]
            z = Float64.(f["redshift"][rows]); pos = Float64.(f["Position"][:, rows]); vel = Float64.(f["Velocity"][:, rows])
            @assert all(isfinite, mass) && all(>(0), mass) && all(isfinite, z) && all(>(0), z)
            paint_chunk7!(maps, workspace, tables[1], lee, lit_over_xgp, mass, z, pos, vel, sphere, rmin)
            count += length(mass)
            println("Painted $count; elapsed=$(round(time()-t0,digits=1)) s"); flush(stdout)
        end
    end
    h5open(dest * ".partial", "w") do f
        for (i, name) in enumerate(MAPNAMES)
            @assert all(isfinite, maps[i].pixels); f[name] = maps[i].pixels
        end
        attributes(f)["nside"] = nside; attributes(f)["sphere_r200c"] = sphere; attributes(f)["halos"] = count
        attributes(f)["literal_over_xgpnative"] = lit_over_xgp
        attributes(f)["C_TNG"] = "$(C0_TNG)*(1+z)^$(K_TNG)"; attributes(f)["zedges"] = collect(ZEDGES)
        attributes(f)["ordering"] = "RING"; attributes(f)["units"] = "Delta T / T (dimensionless)"
    end
    mv(dest * ".partial", dest); println("Saved $dest"); flush(stdout)
end

main7()
