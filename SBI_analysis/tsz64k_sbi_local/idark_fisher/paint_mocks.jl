# SO mocks of chosen Battaglia+12 parameter sets with the frozen tSZ_64k engine, at production fidelity
# (raw NSIDE 8192 -> 2 arcmin beam -> NSIDE 4096, f_sky 0.4 mask, SO baseline Deproj-0 noise, lmax 7979).
# Only spectra are written; maps stay in memory.
#
#   julia --project=<tsz_local_profile_tests/julia_env> --threads=12 paint_mocks.jl task.toml
#
# task.toml:
#   out = "/abs/path/output"
#   [[cases]]
#   label = "B12"
#   theta = [18.1, 0.497, 4.35, 0.154, -0.00865, 0.0393, -0.758, 0.731, 0.415]
#   seeds = [[39001001, 39001002], [910001, 910002]]      # split noise seeds; one cross spectrum per pair
#
# Per case, in <out>/<label>/: unmasked_clean_cl.npy, masked_clean_cl.npy, noisy_cross_<s1>_<s2>.npy (C_ell, ell = 0..7979)
# and case.toml (theta, cache grid and audit, seeds, noise and mask hashes, timings). All cases share one catalogue pass
# (paint_shared!, as the dataset engine does with 4 rows); each keeps its own pressure field, map and noise.
using Printf, Random, Statistics, LinearAlgebra, TOML, SHA

const TASK = TOML.parsefile(ARGS[1])
const OUTDIR = TASK["out"]
const REPO = get(ENV, "HALFDOME_REPO", "/home/cbllover/HalfDome")
const BUNDLE = joinpath(REPO, "tSZ_64k_flat_prior_SO_baseline_deproj0")
const ENGINE = joinpath(BUNDLE, "engine")
const SOURCES = joinpath(BUNDLE, "halfdome_sources")
const SO_DIR = joinpath(SOURCES, "code", "halfdome", "other_sims", "SO")
const CATALOGUE = get(ENV, "HALFDOME_CATALOGUE", joinpath(REPO, "lightcone_100.hdf5"))
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

# ---------------- caches with the engine's accuracy audit ----------------
cases = TASK["cases"]
configs = [configure(Float64.(c["theta"])) for c in cases]
base = configs[1].base_cfg
@assert base.nside == 8192 && base.cl_lmax == 7979 && base.cl_niter == 0 && base.gaussian_beam_fwhm_arcmin == 2.0
@assert all(c.fsky == 0.4 && c.mask_seed == 12345 && c.noise_deprojection == 0 for c in configs)
profiles = Any[]; audits = Any[]
for (case, cfg) in zip(cases, configs)
    accepted = false
    for nodes in [[256, 128, 64], [512, 256, 128], [1024, 512, 256]]
        t0 = time()
        profile = build_cache(cfg.base_cfg, nodes)
        metrics, _ = probe_cache(profile)
        metrics["nodes"] = nodes; metrics["seconds"] = time() - t0
        @printf("%s: cache %s max relative %.2e, |error|/central %.2e\n", case["label"], join(nodes, "x"),
                metrics["max_relative_visible"], metrics["max_absolute_over_central"]); flush(stdout)
        if metrics["max_relative_visible"] <= .004 && metrics["max_absolute_over_central"] <= .004
            push!(profiles, profile); push!(audits, metrics); accepted = true; break
        end
    end
    accepted || error("$(case["label"]): no cache grid met the 0.4 % target")
end
profiles = [p for p in profiles]            # concrete element type, as engine.jl requires
@assert isconcretetype(eltype(profiles))

# ---------------- one catalogue pass for all cases ----------------
states = [blank_state(8192) for _ in profiles]
let warm = [blank_state(32) for _ in profiles]
    paint_shared!(warm, profiles, [1. 0.; 0. 1.; 0. 0.], [1e15, 1e15], [.01, .01])
end
GC.gc()
t_paint = time()
count = catalogue_pass!(base, states, profiles; shared_geometry=true)
@assert count == 85224251 "the engine stops for any other selected-halo count"
@printf("painted %d halos for %d case(s) in %.0f s\n", count, length(cases), time() - t_paint); flush(stdout)

# ---------------- observation operators per case ----------------
mask_info = random_apodized_cap_mask(4096, .4, 60., MersenneTwister(12345))
mask = mask_info.mask
mask_sha = pixel_sha(mask)
@assert mask_sha == "a6c3d64d5ab83e79b6d9b76cccbaf66a708c3e68adce85b9a0d3f610d4100689"
ell_native, native = read_so_noise_native_cl(configs[1].baseline_noise_path, 7979; column=2, input_is_dl=false)
noise_cl = so_noise_cl_vector_for_synalm(ell_native, native, 7979)
for (j, case) in enumerate(cases)
    label = case["label"]; dir = joinpath(OUTDIR, label); mkpath(dir)
    t0 = time()
    signal = make_signal(states[j].m_hp, 4096)
    states[j] = blank_state(1); GC.gc()
    write_npy_float64_vector(joinpath(dir, "unmasked_clean_cl.npy"), compute_cl(base, signal), "unmasked clean")
    signal.pixels .*= mask.pixels
    write_npy_float64_vector(joinpath(dir, "masked_clean_cl.npy"), compute_cl(base, signal), "masked clean")
    noise_hashes = Dict{String,Any}()
    for pair in get(case, "seeds", Any[])
        s1, s2 = Int(pair[1]), Int(pair[2]); @assert s1 != s2
        n1 = generate_gaussian_noise_map(noise_cl, 4096, 7979, MersenneTwister(s1))
        n2 = generate_gaussian_noise_map(noise_cl, 4096, 7979, MersenneTwister(s2))
        noise_hashes["$(s1)_$(s2)"] = [pixel_sha(n1), pixel_sha(n2)]
        n1.pixels .= n1.pixels .* mask.pixels .+ signal.pixels
        n2.pixels .= n2.pixels .* mask.pixels .+ signal.pixels
        cross = compute_cross_cl(base, n1, n2)
        @assert all(isfinite, cross)
        write_npy_float64_vector(joinpath(dir, "noisy_cross_$(s1)_$(s2).npy"), cross, "signed split cross")
        n1 = nothing; n2 = nothing; GC.gc()
        @printf("%s: noise pair %d %d done (%.0f s)\n", label, s1, s2, time() - t0); flush(stdout)
    end
    open(joinpath(dir, "case.toml"), "w") do io
        TOML.print(io, Dict("label" => label, "theta" => Float64.(case["theta"]), "cache" => audits[j],
                            "seeds" => [Int.(p) for p in get(case, "seeds", Any[])], "noise_sha256" => noise_hashes,
                            "mask_sha256" => mask_sha, "selected_halos" => count, "raw_nside" => 8192, "output_nside" => 4096,
                            "julia" => string(VERSION), "threads" => Threads.nthreads(), "seconds_observation" => time() - t0,
                            "timings" => TIMINGS))
    end
    signal = nothing; GC.gc()
end
println("done: ", OUTDIR)
