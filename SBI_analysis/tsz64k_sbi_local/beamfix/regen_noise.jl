# Regenerate the SO map noise of tSZ_64k rows from their two seeds and compute the masked noise x noise cross spectrum,
# exactly as the frozen engine does (engine/engine.jl, save_observation):
#     n_i = alm2map(synalm(N_ell, 7979, 7979, MersenneTwister(seed_i)), 4096)       (generate_gaussian_noise_map)
#     C_nn = anafast(mask n_1, mask n_2; lmax = 7979, niter = 0)                     (compute_cross_cl)
# The stored production spectrum is C_noisy = C_ss + C_sn + C_nn, and C_ss (masked clean) is stored too, so C_nn
# isolates the signal x noise term C_sn = C_noisy - C_ss - C_nn. beamfix.py then forms the beam-consistent spectrum
# C_ss + B_ell C_sn + B_ell^2 C_nn (B_ell: the 2 arcmin Gaussian beam that the engine applies to the signal only).
# For tasks flagged qa = 1 the beamed noise cross C(mask b_1, mask b_2), b_i = alm2map(B_ell alm_i), is computed as well,
# which tests the B_ell^2 C_nn approximation.
#
#   julia --project=<bundle julia_env> --threads=<n> regen_noise.jl <run root> <task list> [<task list> ...]
#
# A task list has one line per task, "<label> <seed1> <seed2> <qa> <hash>" ('#' starts a comment); hash = 1 records the
# sha256 of both noise maps (as the engine hashes them; ~6 s per map, single-threaded, so only for rows whose production
# hashes are compared). Workers on several nodes can
# share one run root: each task is claimed by creating the folder <run root>/claims/<label> (mkdir is atomic, also on
# NFS). Outputs in <run root>/nn/ (C_ell for ell = 0..7979, float64):
#     <label>.npy           C_nn
#     <label>_beamed.npy    C_bb (qa tasks only)
#     <label>.toml          seeds, noise map sha256 (hash tasks), timings; written last = task done
# and once per run root: beam_bl.npy (B_ell, ell = 0..7979) and noise_cl.npy (the SO N_ell given to synalm).
# A worker stops before starting a task if <run root>/STOP exists or if the walltime left (REGEN_WALL_SECONDS, from the
# job script) is less than 3 times its slowest task so far plus 10 minutes, so that no task is left half-done.
const START_TIME = time()
const MY_ARGS = copy(ARGS)
length(MY_ARGS) >= 2 || error("usage: regen_noise.jl <run root> <task list> [<task list> ...]")
const RUN = abspath(MY_ARGS[1])
const LISTS = [abspath(p) for p in MY_ARGS[2:end]]
const OWNER = string(get(ENV, "PBS_JOBID", "nojob"), "@", gethostname(), " pid", getpid())   # job id first: reclaim reads it
const MAX_FAILURES = 3                              # a task that failed this often (nn/<label>.failed) is not claimed again
const WALL_SECONDS = parse(Float64, get(ENV, "REGEN_WALL_SECONDS", "Inf"))
using Printf, Random, Statistics, LinearAlgebra, TOML, SHA, Dates

const REPO = get(ENV, "HALFDOME_REPO", "/home/cbllover/HalfDome")
const BUNDLE = joinpath(REPO, "tSZ_64k_flat_prior_SO_baseline_deproj0")
const ENGINE = joinpath(BUNDLE, "engine")
const SOURCES = joinpath(BUNDLE, "halfdome_sources")
const SO_DIR = joinpath(SOURCES, "code", "halfdome", "other_sims", "SO")
const CATALOGUE = get(ENV, "HALFDOME_CATALOGUE", joinpath(REPO, "lightcone_100.hdf5"))
const SCRATCH = mktempdir()
@assert isfile(CATALOGUE) "HalfDome catalogue not found at $CATALOGUE"

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

println("engine loaded; ", Threads.nthreads(), " threads; owner ", OWNER); flush(stdout)

# ---------------- the observation operators of the engine ----------------
# configure() is only used for the observation settings, which are the same for every theta (asserted below)
const B12_THETA = [18.1, 0.497, 4.35, 0.154, -0.00865, 0.0393, -0.758, 0.731, 0.415]
const MASK_SHA = "a6c3d64d5ab83e79b6d9b76cccbaf66a708c3e68adce85b9a0d3f610d4100689"
cfg = configure(B12_THETA)
base = cfg.base_cfg
@assert base.cl_lmax == 7979 && base.cl_niter == 0 && base.gaussian_beam_fwhm_arcmin == 2.0
@assert cfg.fsky == 0.4 && cfg.mask_seed == 12345 && cfg.noise_deprojection == 0
mask = random_apodized_cap_mask(4096, .4, 60., MersenneTwister(12345)).mask
@assert pixel_sha(mask) == MASK_SHA
ell_native, native = read_so_noise_native_cl(cfg.baseline_noise_path, 7979; column=2, input_is_dl=false)
noise_cl = so_noise_cl_vector_for_synalm(ell_native, native, 7979)
bl = Healpix.gaussbeam(deg2rad(2/60), 7979)                  # B_ell of the signal's 2 arcmin beam, ell = 0..7979

nn_dir = joinpath(RUN, "nn"); claims_dir = joinpath(RUN, "claims")
mkpath(nn_dir); mkpath(claims_dir)

"""Write `vector` to `path` through a temporary file in the same folder, so that readers never see a partial file."""
function atomic_npy(path, vector, label)
    temporary = path * ".tmp." * string(getpid())
    write_npy_float64_vector(temporary, vector, label)
    mv(temporary, path; force=true)
end

function atomic_toml(path, dict)
    temporary = path * ".tmp." * string(getpid())
    open(io -> TOML.print(io, dict), temporary, "w")
    mv(temporary, path; force=true)
end

isfile(joinpath(RUN, "beam_bl.npy")) || atomic_npy(joinpath(RUN, "beam_bl.npy"), bl, "beam B_ell")
isfile(joinpath(RUN, "noise_cl.npy")) || atomic_npy(joinpath(RUN, "noise_cl.npy"), noise_cl, "SO N_ell for synalm")

"""Task `label` is done when its .toml exists and, for a qa task, its beamed spectrum, and for a hash task, its hashes."""
function task_done(label, qa, want_hash)
    path = joinpath(nn_dir, label * ".toml")
    isfile(path) || return false
    qa && !isfile(joinpath(nn_dir, label * "_beamed.npy")) && return false
    want_hash && isempty(get(TOML.parsefile(path), "noise_sha256", String[])) && return false
    return true
end

failures(label) = (p = joinpath(nn_dir, label * ".failed"); isfile(p) ? countlines(p) : 0)

"""Claim task `label` for this worker; false if another worker claimed it first."""
function claim(label)
    folder = joinpath(claims_dir, label)
    try
        mkdir(folder)
    catch
        return false
    end
    write(joinpath(folder, "owner"), OWNER)
    return true
end

"""The engine's noise pair for seeds (s1, s2), C_nn and (qa) C_bb, written to nn/<label>*. Returns the task record."""
function run_task(label, s1, s2, qa, want_hash)
    t0 = time()
    seconds = Dict{String,Float64}()
    tick(name, t) = (seconds[name] = get(seconds, name, 0.0) + time() - t; time())
    t = time()
    alm1 = Healpix.synalm(noise_cl, 7979, 7979, MersenneTwister(s1))
    alm2 = Healpix.synalm(noise_cl, 7979, 7979, MersenneTwister(s2))
    t = tick("synalm", t)
    n1 = Healpix.alm2map(alm1, 4096)
    n2 = Healpix.alm2map(alm2, 4096)
    t = tick("alm2map", t)
    hashes = String[]
    if want_hash
        hashes = [pixel_sha(n1), pixel_sha(n2)]              # as the engine hashes them: before masking
        @assert hashes[1] != hashes[2]
        t = tick("sha", t)
    end
    if qa
        Healpix.almxfl!(alm1, bl); Healpix.almxfl!(alm2, bl)
        b1 = Healpix.alm2map(alm1, 4096)
        b2 = Healpix.alm2map(alm2, 4096)
        t = tick("alm2map", t)
        b1.pixels .*= mask.pixels; b2.pixels .*= mask.pixels
        cbb = compute_cross_cl(base, b1, b2)
        t = tick("anafast_cross", t)
        @assert all(isfinite, cbb)
        atomic_npy(joinpath(nn_dir, label * "_beamed.npy"), cbb, "beamed noise cross")
        b1 = nothing; b2 = nothing
    end
    alm1 = nothing; alm2 = nothing
    n1.pixels .*= mask.pixels; n2.pixels .*= mask.pixels
    cnn = compute_cross_cl(base, n1, n2)
    t = tick("anafast_cross", t)
    @assert all(isfinite, cnn)
    atomic_npy(joinpath(nn_dir, label * ".npy"), cnn, "noise cross")
    n1 = nothing; n2 = nothing
    record = Dict{String,Any}("label" => label, "seeds" => [s1, s2], "noise_sha256" => hashes, "qa" => qa,
                              "seconds" => time() - t0, "transform_seconds" => seconds, "owner" => OWNER,
                              "julia" => string(VERSION), "threads" => Threads.nthreads(), "mask_sha256" => MASK_SHA,
                              "utc" => string(Dates.now(Dates.UTC)))
    atomic_toml(joinpath(nn_dir, label * ".toml"), record)   # written last: marks the task as done
    return record
end

done_here = 0; failed_here = 0; slowest = 0.0; stop_reason = "task lists exhausted"
for list in LISTS
    global done_here, failed_here, slowest, stop_reason
    for line in eachline(list)
        text = strip(split(line, "#")[1])
        isempty(text) && continue
        fields = split(text)
        length(fields) == 5 || error("bad task line in $list: $line")
        label = String(fields[1]); s1 = parse(Int, fields[2]); s2 = parse(Int, fields[3])
        qa = fields[4] == "1"; want_hash = fields[5] == "1"
        task_done(label, qa, want_hash) && continue
        failures(label) >= MAX_FAILURES && continue
        if isfile(joinpath(RUN, "STOP"))
            stop_reason = "STOP file"; break
        end
        left = WALL_SECONDS - (time() - START_TIME)
        if left < 3 * slowest + 600
            stop_reason = @sprintf("walltime: %.0f s left, slowest task %.0f s", left, slowest); break
        end
        claim(label) || continue
        record = try
            run_task(label, s1, s2, qa, want_hash)
        catch err
            err isa InterruptException && rethrow()
            # count the failure, remove this process's temporary files and release the claim if it is still ours
            open(io -> println(io, OWNER, " ", Dates.now(Dates.UTC)), joinpath(nn_dir, label * ".failed"), "a")
            for f in readdir(nn_dir)
                startswith(f, label) && occursin(".tmp." * string(getpid()), f) && rm(joinpath(nn_dir, f); force=true)
            end
            owner_file = joinpath(claims_dir, label, "owner")
            if isfile(owner_file) && strip(read(owner_file, String)) == OWNER
                rm(joinpath(claims_dir, label); recursive=true, force=true)
            end
            @error "task $label failed (failure $(failures(label)))" exception=(err, catch_backtrace())
            failed_here += 1
            if failed_here >= MAX_FAILURES
                stop_reason = "$failed_here task failures in this process"; break
            end
            GC.gc(); continue
        end
        done_here += 1; slowest = max(slowest, record["seconds"])
        @printf("%s done in %.1f s (%s); %d tasks by this worker\n", label, record["seconds"],
                join(["$(k) $(round(v; digits=1))" for (k, v) in sort(collect(record["transform_seconds"]))], ", "),
                done_here); flush(stdout)
        GC.gc(false)
    end
    stop_reason == "task lists exhausted" || break
end
println("worker finished: $done_here tasks; ", stop_reason); flush(stdout)
