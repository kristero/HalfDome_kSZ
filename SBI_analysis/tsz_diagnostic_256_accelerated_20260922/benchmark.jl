# Isolated acceleration experiment. Keep the physical projector and observation
# operator frozen; change only data reuse, cache node counts and test rendering.
ENV["PREFLIGHT_LOAD_ONLY"] = "1"
include(joinpath(@__DIR__, "fullsky_test.jl"))
using DelimitedFiles

const PARAMETER_KEYS = ["P0_amp", "x_c_amp", "beta_amp", "P0_alpha_m",
    "x_c_alpha_m", "beta_alpha_m", "P0_alpha_z", "x_c_alpha_z", "beta_alpha_z"]
const TIMINGS = Dict{String,Float64}()

function measure(f, key)
    started = time()
    value = f()
    TIMINGS[key] = get(TIMINGS, key, 0.) + time() - started
    return value
end

function configure(theta)
    for (key, value) in zip(PARAMETER_KEYS, theta)
        replace_or_add_arg!(ARGS, "battaglia_" * key, value)
    end
    return load_halfdome_fullsky_so_noise_config()
end

"""Vary node counts, never the angular/mass/redshift domain or LOS tolerance."""
function build_cache(cfg, nodes)
    model = ChordMeanProfile(build_tsz_model(cfg), 4.)
    rft = XGPaint.RadialFourierTransform(n=512, pad=cfg.interpolator_pad)
    lt = LinRange(log(minimum(rft.r)), log(maximum(rft.r)), nodes[1])
    lz = LinRange(log(.001), log(5.), nodes[2])
    lm = LinRange(12., cfg.interpolator_logM_max, nodes[3])
    _, _, _, values = XGPaint.profile_grid(model, lt, exp.(lz), lm)
    @assert all(isfinite, values) && minimum(values) >= 0
    values .= max.(values, 1e-300)
    interpolator = XGPaint.Interpolations.interpolate(log.(values),
        XGPaint.BSpline(XGPaint.Cubic(XGPaint.Line(XGPaint.OnGrid()))))
    scaled = XGPaint.scale(interpolator, lt, lz, lm)
    return XGPaint.LogInterpolatorProfile(model, LogRedshiftGrid(scaled, (lt, lz, lm)))
end

function blank_state(nside)
    map = HealpixMap{Float64,RingOrder}(nside)
    fill!(map.pixels, 0.)
    workspace = XGPaint.HealpixRingProfileWorkspace{Float64}(Healpix.Resolution(nside))
    return (; m_hp=map, workspace)
end

"""Paint K pressure profiles using each geometric calculation only once.

This is valid only at fixed cosmology, mass definition and 4 R200c sphere.
M, z, sky position and intersected pixels are independent of gNFW parameters.
Do not share the resulting pressure values or noise between dataset rows.
"""
function paint_shared!(states, profiles, positions, masses, redshifts)
    workspace = states[1].workspace
    locks = [ReentrantLock() for _ in workspace.ring_thetas]
    theta_min = exp(first(profiles[1].itp.ranges[1]))
    Threads.@threads :static for i in eachindex(masses)
        x, y, zpos = positions[1,i], positions[2,i], positions[3,i]
        distance = sqrt(x*x + y*y + zpos*zpos)
        ux, uy, uz = x/distance, y/distance, zpos/distance
        tc, pc = Healpix.vec2ang(ux, uy, uz)
        pc = mod(pc, 2pi)
        mass, z = masses[i], redshifts[i]
        logmass, logz = log10(mass), log(z)
        radius = min(4theta_r200c(profiles[1].model, mass, z), pi)
        first_ring, last_ring = XGPaint.get_relevant_rings(workspace.res, tc, radius)
        for ring in first_ring:last_ring
            a, b = XGPaint.get_ring_disc_ranges(workspace, ring, tc, pc, radius)
            first_pixel = workspace.ring_first_pixels[ring]
            lock(locks[ring]) do
                for lp in Iterators.flatten((a, b))
                    pixel = first_pixel + lp - 1
                    px, py, pz = Healpix.pix2vecRing(workspace.res, pixel)
                    theta = acos(clamp(ux*px + uy*py + uz*pz, -1., 1.))
                    theta < radius || continue
                    chord = chord_factor(theta, radius, 4.)
                    logtheta = log(max(theta, theta_min))
                    for j in eachindex(profiles)
                        value = exp(profiles[j].itp.interpolator(logtheta, logz, logmass))
                        states[j].m_hp.pixels[pixel] += value * chord
                    end
                end
            end
        end
    end
    return length(masses)
end

"""Read only one original chunk at a time; no 85-million-halo RAM cache."""
function catalogue_pass!(cfg, states, profiles; shared_geometry=false)
    selected_count = 0
    h5open(cfg.halfdome_path, "r") do h5
        total = size(h5["Position"], 2)
        for first in 1:cfg.chunkN:total
            last = min(first + cfg.chunkN - 1, total)
            mass, redshift = measure("read_mass_redshift") do
                (Float64.(h5["halo_mass_m200c"][first:last]) ./ cfg.cosmo_h,
                 Float64.(h5["redshift"][first:last]))
            end
            indices, masses, redshifts = measure("selection") do
                keep = isfinite.(mass) .& isfinite.(redshift) .& (redshift .>= 0.)
                cfg.apply_mass_cut && (keep .&= mass .>= cfg.mass_min)
                local_indices = findall(keep)
                (collect(first:last)[local_indices], mass[local_indices], redshift[local_indices])
            end
            isempty(indices) && continue
            positions = measure("read_positions") do
                read_hdf5_columns(h5["Position"], 1:3, indices)
            end
            measure("painting") do
                if shared_geometry
                    paint_shared!(states, profiles, positions, masses, redshifts)
                else
                    for j in eachindex(profiles)
                        paint_visual_batch!(states[j], profiles[j], view(positions,1,:),
                            view(positions,2,:), view(positions,3,:), nothing, masses, redshifts)
                    end
                end
            end
            selected_count += length(masses)
            println("CHUNK ", first, ":", last, " selected=", selected_count,
                    " elapsed_read=", TIMINGS["read_mass_redshift"] + TIMINGS["read_positions"],
                    " elapsed_paint=", TIMINGS["painting"])
            flush(stdout)
        end
    end
    @assert selected_count > 0
    return selected_count
end

"""Equal-area child-centre quadrature for parent-pixel averages.

This is a convergence experiment, not an assertion of exact pixel integration.
Julia HEALPix indices are one-based; nested children are contiguous.
"""
function average_children(fine, coarse_nside)
    fine_nside = fine.resolution.nside
    @assert fine_nside % coarse_nside == 0
    count = (fine_nside ÷ coarse_nside)^2
    coarse = HealpixMap{Float64,RingOrder}(coarse_nside)
    Threads.@threads :static for pixel in eachindex(coarse.pixels)
        parent = Healpix.ring2nest(coarse.resolution, pixel) - 1
        total = 0.
        for child in 1:count
            index = Healpix.nest2ring(fine.resolution, count*parent + child)
            total += fine.pixels[index]
        end
        coarse.pixels[pixel] = total / count
    end
    return coarse
end

function observe(raw, mask, path; window_nside=0)
    mkpath(path)
    alms = measure("map2alm") do
        Healpix.map2alm(raw; lmax=healpix_default_lmax(4096), niter=0)
    end
    beam = Healpix.gaussbeam(deg2rad(2/60), alms.lmax)
    if window_nside > 0
        window = vec(readdlm(joinpath(@__DIR__, "inputs", "pixwin$(window_nside).txt")))
        @assert length(window) == length(beam) && all(window .> 0)
        beam ./= window
    end
    Healpix.almxfl!(alms, beam)
    signal = measure("output_alm2map") do; Healpix.alm2map(alms, 4096); end
    write_npy_float64_vector(joinpath(path,"unmasked_clean_cl.npy"),
        Healpix.alm2cl(Healpix.map2alm(signal; lmax=7979, niter=0)), "clean unmasked")
    signal.pixels .*= mask.mask.pixels
    cl = measure("masked_spectrum") do
        Healpix.alm2cl(Healpix.map2alm(signal; lmax=7979, niter=0))
    end
    @assert length(cl) == 7980 && all(isfinite,cl) && all(cl .>= 0)
    write_npy_float64_vector(joinpath(path,"masked_clean_cl.npy"),cl,"clean masked")
end

function main_benchmark()
    task = TOML.parsefile(ENV["BENCHMARK_TASK"])
    configs = [configure(case["theta"]) for case in task["cases"]]
    cfg = configs[1].base_cfg
    @assert cfg.batching_mode == "full" && cfg.cl_lmax == 7979 && cfg.cl_niter == 0
    @assert cfg.gaussian_beam_fwhm_arcmin == 2. && cfg.interpolator_pad == 256
    @assert all(c.fsky == .4 && c.mask_seed == 12345 && c.mask_apodization_arcmin == 60. for c in configs)
    # Warm compilation of both painting kernels using tiny disposable maps.
    profiles = measure("cache") do
        [build_cache(c.base_cfg, task["nodes"]) for c in configs]
    end
    positions = [1. 0.; 0. 1.; 0. 0.]
    warm = [blank_state(32) for _ in profiles]
    paint_shared!(warm, profiles, positions, [1e15,1e15], [.01,.01])
    paint_visual_batch!(warm[1], profiles[1], positions[1,:], positions[2,:], positions[3,:],
                        nothing, [1e15,1e15], [.01,.01])
    warm = nothing; GC.gc()
    states = measure("allocation") do; [blank_state(cfg.nside) for _ in profiles]; end
    start = time()
    counts = Int[]
    if task["mode"] == "reread"
        for j in eachindex(profiles)
            push!(counts, catalogue_pass!(cfg, states[j:j], profiles[j:j]))
        end
    else
        count = catalogue_pass!(cfg, states, profiles; shared_geometry=task["mode"] == "fused")
        counts = fill(count, length(profiles))
    end
    catalogue_seconds = time() - start
    mask = random_apodized_cap_mask(4096, .4, 60., MersenneTwister(12345))
    for j in eachindex(profiles)
        target = joinpath(OUT,task["cases"][j]["label"])
        observe(states[j].m_hp, mask, target)
        for coarse_nside in get(task,"pixel_targets",Int[])
            coarse = measure("pixel_average") do; average_children(states[j].m_hp, coarse_nside); end
            observe(coarse, mask, target * "_pixel$(coarse_nside)")
            # Save both conventions; the isotropic pixel window is approximate.
            observe(coarse, mask, target * "_pixel$(coarse_nside)_dewindow"; window_nside=coarse_nside)
            coarse = nothing; GC.gc()
        end
    end
    save_toml(joinpath(OUT,"benchmark.toml"), Dict("timings"=>TIMINGS,
        "catalogue_seconds"=>catalogue_seconds,"selected_halos_per_case"=>counts,
        "raw_nside"=>cfg.nside,"output_nside"=>4096,"nodes"=>task["nodes"],
        "threads"=>Threads.nthreads(),"mode"=>task["mode"],"cases"=>length(profiles),
        "mask_sha256"=>pixel_sha(mask.mask),"ell_max"=>7979,"noise_generated"=>false,
        "peak_raw_map_GiB"=>length(profiles)*12cfg.nside^2*8/2.0^30,
        "scope"=>"Fixed-cosmology controls, not production certification"))
end

get(ENV,"BENCHMARK_LOAD_ONLY","0") == "1" || main_benchmark()
