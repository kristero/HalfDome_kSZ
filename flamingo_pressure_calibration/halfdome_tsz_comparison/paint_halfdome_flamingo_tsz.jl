# HalfDome Compton-y maps painted with the FLAMINGO L2p8_m9 Battaglia-form pressure calibration.
#
# Same painting as truncation_comparison/radius_series/paint_radius_series.jl (its y_sphere4 map is the
# Battaglia+12 reference): XGPaint Battaglia16ThermalSZProfile, gas inside the 4 R200c sphere only
# (ChordMeanProfile cache x exact chord factor), HalfDome cosmology, catalogue halo_mass_m200c / h in
# physical Msun, NSIDE 4096. One pass paints one map per coefficient file given in --profiles.
#
# Usage: julia -t 20 paint_halfdome_flamingo_tsz.jl --profiles=name1:file1.toml,name2:file2.toml
#        [--catalog=...] [--output-dir=...] [--cache-dir=...] [--nside=4096] [--lmax=8192] [--max-halos=0]
using Dates, HDF5, Healpix, Interpolations, TOML, XGPaint
using Base.Threads
include(joinpath(@__DIR__, "..", "..", "truncation_comparison", "spherical_truncation_profiles.jl"))
using .SphericalTruncation

const H = 0.6774; const OMEGA_B = 0.0486; const OMEGA_C = 0.2603       # HalfDome cosmology (radius series)
const RADIUS = 4.0                                                    # sphere radius in R200c

function getarg(key, default)
    for a in ARGS
        startswith(a, "--$key=") && return split(a, "="; limit=2)[2]
    end
    return default
end
catalog = getarg("catalog", "/home/cbllover/HalfDome/lightcone_100.hdf5")
nside = parse(Int, getarg("nside", "4096"))
max_halos = parse(Int, getarg("max-halos", "0"))
chunk_size = parse(Int, getarg("chunk-size", "1000000"))
lmax = min(parse(Int, getarg("lmax", "8192")), 3nside - 1)
output_dir = getarg("output-dir", joinpath(@__DIR__, "output")); mkpath(output_dir)
cache_dir = getarg("cache-dir", joinpath(@__DIR__, "caches")); mkpath(cache_dir)
profile_specs = [split(s, ":"; limit=2) for s in split(getarg("profiles", ""), ",") if !isempty(s)]
isempty(profile_specs) && error("give --profiles=name:file.toml[,name:file.toml]")

# ---------------------------------------------------------------- profiles from the calibration files
const KEYS = ("P0_amp", "P0_alpha_m", "P0_alpha_z", "x_c_amp", "x_c_alpha_m", "x_c_alpha_z", "beta_amp", "beta_alpha_m",
              "beta_alpha_z", "alpha_amp", "alpha_alpha_m", "alpha_alpha_z", "gamma_amp", "gamma_alpha_m", "gamma_alpha_z")
names = String[]; models = Battaglia16ThermalSZProfile[]; coefficient_sets = Dict{String,Any}()
for (name, file) in profile_specs
    c = TOML.parsefile(file)
    kw = Dict(Symbol(k) => Float64(c[k]) for k in KEYS)
    push!(names, String(name)); push!(models, Battaglia16ThermalSZProfile(; Omega_c=OMEGA_C, Omega_b=OMEGA_B, h=H, kw...))
    coefficient_sets[String(name)] = Dict("file" => abspath(file), (String(k) => v for (k, v) in kw)...)
end

# XGPaint's pressure must equal the fitted formula 0.5176 P0 (x/xc)^gamma [1 + (x/xc)^alpha]^(-beta)
println("pressure check, XGPaint Pe/P200 against the fitted formula:")
for (name, model) in zip(names, models)
    c = coefficient_sets[name]; worst = 0.0
    for M in (1e13, 1e14, 1e15), z in (0.0, 0.5, 2.0), x in (0.05, 0.3, 1.0, 3.0)
        par = XGPaint.get_params(model, M * XGPaint.M_sun, z)
        xgp = 0.5176 * par.P₀ * XGPaint.generalized_nfw(x, par.xc, par.α, par.β, par.γ)
        lm, lz = log(M / 1e14), log1p(z)
        p0 = c["P0_amp"] * exp(c["P0_alpha_m"] * lm + c["P0_alpha_z"] * lz)
        xc = c["x_c_amp"] * exp(c["x_c_alpha_m"] * lm + c["x_c_alpha_z"] * lz)
        be = c["beta_amp"] * exp(c["beta_alpha_m"] * lm + c["beta_alpha_z"] * lz)
        q = x / xc
        expected = 0.5176 * p0 * q^c["gamma_amp"] * (1 + q^c["alpha_amp"])^(-be)
        worst = max(worst, abs(xgp / expected - 1))
    end
    println("  $(name): max relative difference $(worst)")
    worst < 1e-10 || error("XGPaint pressure differs from the fitted formula for $(name)")
end
flush(stdout)

function load_or_build(name, model)
    file = joinpath(cache_dir, "$(name)_sphere4_chordmean_interpolator.jld2")
    t0 = time(); existed = isfile(file)
    itp = build_interpolator(model; cache_file=file, N_logtheta=512, pad=128, logM_max=15.7, overwrite=!existed, verbose=false)
    println("interpolator $(name): $(existed ? "loaded" : "built") in $(round(time() - t0; digits=1)) s"); flush(stdout)
    return itp
end
itps = Tuple(load_or_build(n, ChordMeanProfile(m, RADIUS)) for (n, m) in zip(names, models))
theta_min = maximum(exp(first(first(ip.itp.ranges))) for ip in itps)
logm_lo = maximum(first(ip.itp.ranges[3]) for ip in itps); logm_hi = minimum(last(ip.itp.ranges[3]) for ip in itps)
z_hi = minimum(last(ip.itp.ranges[2]) for ip in itps)
geometry = models[1]                                 # R200c and angles depend only on cosmology, not on the profile
for (M, z) in ((1e13, 0.5), (1e14, 0.5), (1e15, 0.3))
    θ = 0.7 * theta_r200c(geometry, M, z)
    println("  cache/direct M=$M z=$z: ", join(["$(n) $(round(itps[k](θ, M, z) / ChordMeanProfile(models[k], RADIUS)(θ, M, z); digits=7))"
                                               for (k, n) in enumerate(names)], "  "))
end
flush(stdout)

# ---------------------------------------------------------------- paint
resolution = Healpix.Resolution(nside)
workspace = XGPaint.HealpixRingProfileWorkspace{Float64}(resolution)
ring_locks = [ReentrantLock() for _ in eachindex(workspace.ring_thetas)]
maps = [HealpixMap{Float64,RingOrder}(nside) for _ in names]
for m in maps; fill!(m.pixels, 0.0); end
pix = Tuple(m.pixels for m in maps)
const NP = length(names)

function paint_batch!(masses, redshifts, positions)
    updates = zeros(Int, Threads.maxthreadid())
    Threads.@threads :static for i in eachindex(masses)
        tid = Threads.threadid()
        x, y, zc = Float64(positions[1, i]), Float64(positions[2, i]), Float64(positions[3, i]); d = sqrt(x^2 + y^2 + zc^2)
        ux, uy, uz = x / d, y / d, zc / d
        θc, φc = Healpix.vec2ang(ux, uy, uz); θc = Float64(θc); φc = mod(Float64(φc), 2pi)
        M = Float64(masses[i]); z = Float64(redshifts[i])
        θmax = min(RADIUS * theta_r200c(geometry, M, z), pi)
        ring_start, ring_stop = XGPaint.get_relevant_rings(workspace.res, θc, θmax); n = 0
        for ring in ring_start:ring_stop
            r1, r2 = XGPaint.get_ring_disc_ranges(workspace, ring, θc, φc, θmax); first_pixel = workspace.ring_first_pixels[ring]
            lock(ring_locks[ring]) do
                for lp in Iterators.flatten((r1, r2))
                    gp = first_pixel + lp - 1
                    px, py, pz = Healpix.pix2vecRing(workspace.res, gp)
                    θ = acos(clamp(ux * px + uy * py + uz * pz, -1.0, 1.0)); θ < θmax || continue
                    θe = max(θ, theta_min); f = chord_factor(θ, θmax, RADIUS); n += 1
                    for k in 1:NP
                        pix[k][gp] += itps[k](θe, M, z) * f
                    end
                end
            end
        end
        updates[tid] += n
    end
    return sum(updates)
end

t0 = time(); total_visits = 0; nsel = 0; mmin = Inf; mmax = -Inf; zmin = Inf; zmax = -Inf
h5open(catalog, "r") do f
    pos_ds = f["Position"]; mass_ds = f["halo_mass_m200c"]; z_ds = f["redshift"]
    N = size(pos_ds, 2); last = max_halos == 0 ? N : min(max_halos, N)
    for start in 1:chunk_size:last
        stop = min(start + chunk_size - 1, last); idx = start:stop
        pos = Float64.(pos_ds[:, idx]); masses = Float64.(mass_ds[idx]) ./ H; zs = Float64.(z_ds[idx])
        keep = isfinite.(masses) .& (masses .> 0) .& isfinite.(zs) .& (zs .> 0) .& vec(all(isfinite, pos; dims=1))
        pos = pos[:, keep]; masses = masses[keep]; zs = zs[keep]
        all(logm_lo .<= log10.(masses) .<= logm_hi) || error("mass outside cache range in chunk $start")
        all(zs .<= z_hi) || error("redshift outside cache range in chunk $start")
        global total_visits += paint_batch!(masses, zs, pos); global nsel += length(masses)
        global mmin = min(mmin, minimum(masses)); global mmax = max(mmax, maximum(masses))
        global zmin = min(zmin, minimum(zs)); global zmax = max(zmax, maximum(zs))
        if start == 1 || (start ÷ chunk_size) % 10 == 0 || stop == last
            println("  rows $start:$stop  halos=$nsel  pixel visits=$total_visits  elapsed=$(round(time() - t0; digits=0)) s"); flush(stdout)
        end
    end
end
paint_seconds = time() - t0
println("painting done in $(round(paint_seconds; digits=1)) s"); flush(stdout)

# ---------------------------------------------------------------- spectra and outputs
t1 = time(); cls = Vector{Vector{Float64}}(undef, NP)
for (k, m) in enumerate(maps)
    all(isfinite, m.pixels) || error("non-finite pixels in $(names[k])")
    cls[k] = anafast(m; lmax=lmax, niter=0)
    Healpix.saveToFITS(m, "!" * joinpath(output_dir, "halfdome_y_$(names[k])_sphere4_nside$(nside).fits"), typechar="D")
    println("  $(names[k]): mean y=$(sum(m.pixels) / length(m.pixels))  saved and anafast done ($(round(time() - t1; digits=0)) s)"); flush(stdout)
end
open(joinpath(output_dir, "raw_cl.csv"), "w") do io
    println(io, "ell," * join(names, ","))
    for l in 0:lmax
        println(io, string(l) * "," * join((string(cls[k][l + 1]) for k in 1:NP), ","))
    end
end
open(joinpath(output_dir, "provenance.toml"), "w") do io
    TOML.print(io, Dict{String,Any}(
        "created_utc" => Dates.format(now(UTC), dateformat"yyyy-mm-ddTHH:MM:SS"), "host" => gethostname(), "julia" => string(VERSION),
        "julia_threads" => Threads.nthreads(), "xgpaint" => string(pathof(XGPaint)), "catalog" => catalog, "nside" => nside, "lmax" => lmax,
        "halos_painted" => nsel, "max_halos_option" => max_halos, "pixel_visits" => total_visits,
        "paint_seconds" => round(paint_seconds; digits=1), "h" => H, "Omega_b" => OMEGA_B, "Omega_c" => OMEGA_C,
        "mass_definition" => "M200c, catalogue halo_mass_m200c / h -> physical Msun",
        "selected_mass_min_msun" => mmin, "selected_mass_max_msun" => mmax, "selected_redshift_min" => zmin, "selected_redshift_max" => zmax,
        "truncation" => "gas inside the 4 R200c sphere only (ChordMeanProfile cache x exact chord factor), as the y_sphere4 radius-series map",
        "spectra" => "Healpix.anafast(map; lmax, niter=0), raw C_ell, no pixel window applied", "theta_min_rad" => theta_min,
        "profiles" => coefficient_sets))
end
println("done: $(output_dir)")
