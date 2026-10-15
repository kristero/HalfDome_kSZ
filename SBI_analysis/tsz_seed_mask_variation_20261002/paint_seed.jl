# One full-fidelity B12 lightcone painting, reused for every requested mask.
# Bootstrap, cache audit and signal beam are copied from the frozen paint_mocks.jl.
# Run with the staged Julia 1.12.2 environment: paint_seed.jl task.toml.
using Printf, Random, Statistics, LinearAlgebra, TOML, SHA

const TASK = TOML.parsefile(ARGS[1])
const OUTDIR = TASK["out"]
const REPO = get(ENV, "HALFDOME_REPO", "/home/cbllover/HalfDome")
const BUNDLE = joinpath(REPO, "tSZ_64k_flat_prior_SO_baseline_deproj0")
const ENGINE = joinpath(BUNDLE, "engine")
const SOURCES = joinpath(BUNDLE, "halfdome_sources")
const SO_DIR = joinpath(SOURCES, "code", "halfdome", "other_sims", "SO")
const CATALOGUE = TASK["catalogue"]
const SCRATCH = mktempdir()
@assert isfile(CATALOGUE) "HalfDome catalogue not found at $CATALOGUE"
mkpath(OUTDIR)

# ---------------- the test's command line (reference/test256/base_command.json) with local paths ----------------
function test_argv()
    lines = readlines(joinpath(BUNDLE, "reference", "test256", "base_command.json"))
    start = findfirst(l -> occursin("\"argv\"", l), lines)
    argv = String[]
    for line in lines[start+1:end]
        m = match(r"^\s*\"(.*)\",?\s*$", line)
        m === nothing && break
        push!(argv, m.captures[1])
    end
    return argv
end
engine_args = String[]
for value in test_argv()[2:end]
    startswith(value, "--") && continue
    endswith(value, "/gate.jl") && continue
    if startswith(value, "halfdome_path=")
        value = "halfdome_path=" * CATALOGUE
    elseif startswith(value, "output_dir=")
        value = "output_dir=" * joinpath(SCRATCH, "raw")
    elseif startswith(value, "cache_dir=")
        value = "cache_dir=" * joinpath(SCRATCH, "cache")
    elseif startswith(value, "baseline_noise_path=") || startswith(value, "goal_noise_path=")
        key, original = split(value, "="; limit=2)
        value = key * "=" * joinpath(SO_DIR, basename(original))
    elseif occursin("/lustre/", value)      # an original test path that no rule above replaced (local paths may be on /lustre)
        error("unhandled test path in base_command.json: $value")
    end
    push!(engine_args, value)
end
empty!(ARGS); append!(ARGS, engine_args)

# ---------------- environment scrubbing (pipeline/common.py) ----------------
const ENV_PATTERN = r"env\s*=\s*\"([A-Z][A-Z0-9_]*)\"|ENV\[\"([A-Z][A-Z0-9_]*)\"\]|haskey\(ENV,\s*\"([A-Z][A-Z0-9_]*)\"\)|get\(ENV,\s*\"([A-Z][A-Z0-9_]*)\""
sources = vcat(filter(p -> endswith(p, ".jl"), readdir(ENGINE; join=true)),
               [joinpath(root, f) for (root, _, files) in walkdir(SOURCES) for f in files if endswith(f, ".jl")])
read_names = Set{String}()
for path in sources, m in eachmatch(ENV_PATTERN, read(path, String)), c in m.captures
    c === nothing || push!(read_names, c)
end
for key in collect(keys(ENV))
    if (key in read_names && !(key in ("SLURM_JOB_ID", "SLURM_ARRAY_TASK_ID"))) || key in ("JULIA_PROJECT", "JULIA_LOAD_PATH") ||
       startswith(key, "TSZ_") || startswith(key, "BATTAGLIA_") || startswith(key, "BARYON_SLICE_")
        delete!(ENV, key)
    end
end
ENV["JULIA_DEPOT_PATH"] = join(DEPOT_PATH, ":")
ENV["FLAMINGO_CAMPAIGN"] = SOURCES
ENV["HALFDOME_SOURCE_DIR"] = joinpath(SOURCES, "code", "halfdome")
ENV["PREFLIGHT_OUTPUT"] = SCRATCH
ENV["HALO_BLOCK_SIZE"] = "256"
ENV["OPENBLAS_NUM_THREADS"] = "1"; ENV["MKL_NUM_THREADS"] = "1"; ENV["OMP_NUM_THREADS"] = string(Threads.nthreads())
ENV["HDF5_USE_FILE_LOCKING"] = "FALSE"
ENV["BENCHMARK_LOAD_ONLY"] = "1"

# ---------------- the frozen engine, in engine.jl's include order ----------------
include(joinpath(ENGINE, "benchmark.jl"))
include(joinpath(ENGINE, "smooth_exterior.jl"))
include(joinpath(ENGINE, "balanced_painter.jl"))
println("engine loaded; XGPaint from ", pathof(XGPaint), "; ", Threads.nthreads(), " threads"); flush(stdout)

# probe_cache: verbatim from engine/engine.jl (the per-row accuracy audit)
function probe_cache(profile)
    max_relative = 0.; max_scaled = 0.; squared = 0.; denominator = 0.
    probes = Vector{Vector{Float64}}()
    for lm in [12.013,12.53,13.07,13.61,14.13,14.69,15.19,15.67],
        z in [.00113,.0037,.0127,.047,.173,.61,1.73,4.71]
        mass = 10.0^lm
        p = XGPaint.prepare_profile_slice(profile.model,mass,z)
        central = spherical_value_direct(profile.model,0.,mass,z)
        @assert isfinite(central) && central>0
        for x in vcat(10.0.^range(-7.,log10(3.5),length=36),[3.8,3.98,3.9998])
            theta = max(x*p.θ200,exp(first(profile.itp.ranges[1])))
            direct = spherical_value_direct(profile.model,theta,mass,z)
            estimated = profile(theta,mass,z)*chord_factor(theta,4p.θ200,4.)
            @assert isfinite(estimated) && estimated>=0 && isfinite(direct)
            scaled = abs(estimated-direct)/central
            max_scaled = max(max_scaled,scaled)
            if direct > central*1e-5
                max_relative = max(max_relative,abs(estimated/direct-1))
            end
            squared += (estimated-direct)^2; denominator += direct^2
            push!(probes,[lm,z,x,direct,estimated,central])
        end
    end
    return Dict{String,Any}("max_relative_visible"=>max_relative,"max_absolute_over_central"=>max_scaled,
        "relative_l2"=>sqrt(squared/denominator),"points"=>length(probes)),probes
end

# make_signal: verbatim from engine/engine.jl
function make_signal(raw, output_nside)
    alm = measure("raw_map2alm") do
        Healpix.map2alm(raw; lmax=12287, niter=0)
    end
    Healpix.almxfl!(alm, Healpix.gaussbeam(deg2rad(2/60), alm.lmax))
    return measure("signal_alm2map") do
        Healpix.alm2map(alm, output_nside)
    end
end

function atomic_toml(path, record)
    temporary = path * ".tmp"
    open(io -> TOML.print(io, record), temporary, "w")
    mv(temporary, path; force=true)
end

function save_spectrum(dir, name, values)
    @assert length(values) == 7980 && all(isfinite, values)
    path = joinpath(dir, name * ".npy")
    write_npy_float64_vector(path * ".tmp", values, name)
    mv(path * ".tmp", path; force=true)
end

function audited_profile(cfg)
    for nodes in [[256, 128, 64], [512, 256, 128], [1024, 512, 256]]
        started = time()
        profile = build_cache(cfg, nodes)
        audit, _ = probe_cache(profile)
        audit["nodes"] = nodes
        audit["seconds"] = time() - started
        println("CACHE ", audit); flush(stdout)
        if audit["max_relative_visible"] <= .004 && audit["max_absolute_over_central"] <= .004
            return profile, audit
        end
    end
    error("No pressure cache met the frozen engine's 0.4% audit threshold")
end

"""Measure one mask without altering the shared signal or noise maps.

The primary spectrum uses the SAME beam correction as the trained model.
Multiplication by B_ell after masking only approximately commutes with masking;
we also measure the exact map-level beamed-noise result as a separate diagnostic.
Neither spectrum is divided by f_sky or by the beam; both remain signed pseudo-C_ell.
"""
function observe_mask(signal, noises, beamed_noises, base, mask_seed, beam)
    dir = joinpath(OUTDIR, "mask_$(mask_seed)")
    mkpath(dir)
    started = time()
    info = random_apodized_cap_mask(4096, .4, 60., MersenneTwister(mask_seed))
    mask = info.mask
    mask_sha = pixel_sha(mask)
    if mask_seed == 12345
        @assert mask_sha == "a6c3d64d5ab83e79b6d9b76cccbaf66a708c3e68adce85b9a0d3f610d4100689"
    end
    masked_signal = deepcopy(signal)
    masked_signal.pixels .*= mask.pixels
    clean = compute_cl(base, masked_signal)
    splits = [deepcopy(n) for n in noises]
    for n in splits
        n.pixels .*= mask.pixels
    end
    nn = compute_cross_cl(base, splits[1], splits[2])
    for n in splits
        n.pixels .+= masked_signal.pixels
    end
    noisy = compute_cross_cl(base, splits[1], splits[2])
    splits = nothing; GC.gc()
    corrected = clean .+ beam .* (noisy .- clean .- nn) .+ beam.^2 .* nn

    splits = [deepcopy(n) for n in beamed_noises]
    for n in splits
        n.pixels .= n.pixels .* mask.pixels .+ masked_signal.pixels
    end
    exact = compute_cross_cl(base, splits[1], splits[2])
    for (name, values) in [("clean_cl", clean), ("noise_cl", nn), ("production_cl", noisy),
                           ("beam_corrected_cl", corrected), ("exact_beam_cl", exact)]
        save_spectrum(dir, name, values)
    end
    hashes = Dict(f => bytes2hex(open(sha256, joinpath(dir, f)))
                  for f in readdir(dir) if endswith(f, ".npy"))
    atomic_toml(joinpath(dir, "complete.toml"), Dict(
        "simulation_seed" => TASK["simulation_seed"], "mask_seed" => mask_seed,
        "noise_seeds" => TASK["noise_seeds"], "mask_sha256" => mask_sha,
        "center_xyz" => [info.center.x, info.center.y, info.center.z],
        "support_fsky" => info.support_fsky, "mean_weight" => info.mean_weight,
        "mean_weight2" => info.mean_weight2, "seconds" => time() - started,
        "spectrum_sha256" => hashes, "split_Nell_multiplier" => 1.0,
        "primary_beam_operator" => "Css+B*(Cproduction-Css-Cnn)+B^2*Cnn",
        "exact_beam_operator" => "cross(mask*(B*y+B*n1), mask*(B*y+B*n2))"))
    println("MASK DONE ", mask_seed, " seconds=", time()-started); flush(stdout)
end

function main()
    started = time()
    theta = Float64.(TASK["theta"])
    cfg = configure(theta)
    base = cfg.base_cfg
    @assert base.nside == 8192 && base.cl_lmax == 7979 && base.cl_niter == 0
    @assert base.gaussian_beam_fwhm_arcmin == 2.0 && base.cosmo_h == .68
    @assert cfg.fsky == .4 && cfg.mask_seed == 12345 && cfg.noise_deprojection == 0
    @assert cfg.mask_apodization_arcmin == 60. && base.mass_min == 1e12
    profile, audit = audited_profile(base)
    profiles = [profile]
    @assert isconcretetype(eltype(profiles))
    warm = [blank_state(32)]
    paint_shared!(warm, profiles, [1. 0.; 0. 1.; 0. 0.], [1e15, 1e15], [.01, .01])
    warm = nothing; GC.gc()
    states = [blank_state(8192)]
    count = catalogue_pass!(base, states, profiles; shared_geometry=true)
    # Different simulations have different halo counts; only the training seed has a fixed reference count.
    if TASK["simulation_seed"] == 100
        @assert count == 85224251
    end
    @assert 0 < count <= TASK["catalogue_rows"]
    signal = make_signal(states[1].m_hp, 4096)
    states = nothing; profiles = nothing; profile = nothing; GC.gc()
    save_spectrum(OUTDIR, "unmasked_clean_cl", compute_cl(base, signal))
    ell, native = read_so_noise_native_cl(cfg.baseline_noise_path, 7979; column=2, input_is_dl=false)
    noise_cl = so_noise_cl_vector_for_synalm(ell, native, 7979)
    beam = Healpix.gaussbeam(deg2rad(2/60), 7979)
    save_spectrum(OUTDIR, "beam_bl", beam)
    noises = HealpixMap{Float64,RingOrder}[]
    beamed_noises = HealpixMap{Float64,RingOrder}[]
    for seed in TASK["noise_seeds"]
        alm = Healpix.synalm(noise_cl, 7979, 7979, MersenneTwister(seed))
        push!(noises, Healpix.alm2map(alm, 4096))
        Healpix.almxfl!(alm, beam)
        push!(beamed_noises, Healpix.alm2map(alm, 4096))
    end
    noise_hashes = pixel_sha.(noises)
    @assert noise_hashes[1] != noise_hashes[2]
    for seed in TASK["mask_seeds"]
        observe_mask(signal, noises, beamed_noises, base, seed, beam)
        GC.gc()
    end
    atomic_toml(joinpath(OUTDIR, "complete.toml"), Dict(
        "simulation_seed" => TASK["simulation_seed"], "catalogue" => CATALOGUE,
        "catalogue_bytes" => filesize(CATALOGUE), "selected_halos" => count,
        "theta" => theta, "cache_audit" => audit, "mask_seeds" => TASK["mask_seeds"],
        "noise_seeds" => TASK["noise_seeds"], "noise_sha256" => noise_hashes,
        "raw_nside" => 8192, "output_nside" => 4096, "ell_max" => 7979,
        "beam_fwhm_arcmin" => 2., "sphere_R200c" => 4., "mass_min_Msun" => 1e12,
        "mass_convention" => "catalogue M200c in Msun/h divided by h=0.68",
        "redshift_selection" => "all finite nonnegative catalogue redshifts; training cache 0.001 to 5",
        "julia" => string(VERSION), "threads" => Threads.nthreads(),
        "seconds" => time()-started, "timings" => TIMINGS))
end

main()
