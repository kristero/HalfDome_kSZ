# The frozen physical projector is shared by tests and dataset production.
ENV["BENCHMARK_LOAD_ONLY"] = "1"
include(joinpath(@__DIR__, "benchmark.jl"))
include(joinpath(@__DIR__, "smooth_exterior.jl"))
include(joinpath(@__DIR__, "balanced_painter.jl"))
using LinearAlgebra

const TASK = TOML.parsefile(ENV["BENCHMARK_TASK"])

function make_signal(raw, output_nside)
    alm = measure("raw_map2alm") do
        Healpix.map2alm(raw; lmax=12287, niter=0)
    end
    Healpix.almxfl!(alm, Healpix.gaussbeam(deg2rad(2/60), alm.lmax))
    return measure("signal_alm2map") do
        Healpix.alm2map(alm, output_nside)
    end
end

function save_observation(signal, cfg, path, seeds; save_unmasked=true)
    mkpath(path)
    nside = signal.resolution.nside
    mask = random_apodized_cap_mask(nside, .4, 60., MersenneTwister(12345))
    if save_unmasked
        write_npy_float64_vector(joinpath(path,"unmasked_clean_cl.npy"),
            compute_cl(cfg.base_cfg,signal), "unmasked clean")
    end
    signal.pixels .*= mask.mask.pixels
    write_npy_float64_vector(joinpath(path,"masked_clean_cl.npy"),
        compute_cl(cfg.base_cfg,signal), "masked clean")
    noise_hashes = String[]
    if !isempty(seeds)
        @assert length(seeds) == 2 && seeds[1] != seeds[2]
        ell, native = read_so_noise_native_cl(cfg.baseline_noise_path,7979;
            column=2,input_is_dl=false)
        noise_cl = so_noise_cl_vector_for_synalm(ell,native,7979)
        splits = [generate_gaussian_noise_map(noise_cl,nside,7979,MersenneTwister(s)) for s in seeds]
        append!(noise_hashes,pixel_sha.(splits))
        @assert noise_hashes[1] != noise_hashes[2]
        for split in splits
            split.pixels .= split.pixels .* mask.mask.pixels .+ signal.pixels
        end
        cross = compute_cross_cl(cfg.base_cfg,splits[1],splits[2])
        @assert all(isfinite,cross)
        write_npy_float64_vector(joinpath(path,"masked_noisy_cross_cl.npy"),cross,"signed split cross")
    end
    save_toml(joinpath(path,"observation.toml"),Dict(
        "output_nside"=>nside,"ell_max"=>7979,"mask_sha256"=>pixel_sha(mask.mask),
        "split_seeds"=>seeds,"noise_sha256"=>noise_hashes,
        "noise_beam_applied"=>false,"split_Nell_multiplier"=>1.,
        "signal_beam_fwhm_arcmin"=>2.))
end

function run_maps()
    cases = TASK["cases"]
    configs = [configure(c["theta"]) for c in cases]
    cfg = configs[1].base_cfg
    @assert cfg.nside == 8192 && cfg.cl_lmax == 7979 && cfg.cl_niter == 0
    @assert cfg.gaussian_beam_fwhm_arcmin == 2. && cfg.interpolator_pad == 256
    @assert all(c.fsky == .4 && c.mask_seed == 12345 for c in configs)
    profiles = measure("cache") do
        [build_cache(c.base_cfg,case["nodes"]) for (c,case) in zip(configs,cases)]
    end
    warm = [blank_state(32) for _ in profiles]
    paint_shared!(warm,profiles,[1. 0.;0. 1.;0. 0.],[1e15,1e15],[.01,.01])
    warm = nothing; GC.gc()
    # The catalogue and geometry are reused; each pressure field remains separate.
    states = Any[blank_state(8192) for _ in profiles]
    count = catalogue_pass!(cfg,states,profiles;shared_geometry=true)
    @assert count == 85224251
    for j in eachindex(cases)
        case = cases[j]
        path = joinpath(OUT,case["label"])
        for output_nside in get(TASK,"output_nsides",[4096])
            signal = make_signal(states[j].m_hp,output_nside)
            target = length(get(TASK,"output_nsides",[4096])) == 1 ? path : joinpath(path,"output$(output_nside)")
            save_observation(signal,configs[j],target,get(case,"seeds",Int[]))
            signal = nothing; GC.gc()
        end
        states[j] = nothing; GC.gc()
    end
    save_toml(joinpath(OUT,"batch.toml"),Dict("timings"=>TIMINGS,
        "selected_halos"=>count,"cases"=>length(cases),"raw_nside"=>8192,
        "threads"=>Threads.nthreads(),"scheduler"=>"greedy blocks of 256"))
end

function numerical_gate()
    cases = TASK["cases"]
    profiles = [build_cache(configure(c["theta"]).base_cfg,[256,128,64]) for c in cases]
    rng = MersenneTwister(20260922)
    positions = randn(rng,3,1024)
    masses = exp10.(12.01 .+ 3.6rand(rng,1024))
    redshifts = exp.(log(.0011) .+ log(4.8/.0011)*rand(rng,1024))
    separate = [blank_state(128) for _ in profiles]
    shared = [blank_state(128) for _ in profiles]
    for j in eachindex(profiles)
        paint_visual_batch!(separate[j],profiles[j],positions[1,:],positions[2,:],positions[3,:],
            nothing,masses,redshifts)
    end
    paint_shared!(shared,profiles,positions,masses,redshifts)
    errors = [norm(a.m_hp.pixels-b.m_hp.pixels)/norm(a.m_hp.pixels) for (a,b) in zip(separate,shared)]
    @assert maximum(errors) < 1e-12
    @assert all(all(isfinite,s.m_hp.pixels) && maximum(s.m_hp.pixels)>0 for s in shared)
    for profile in profiles, (mass,z) in zip(masses,redshifts)
        @assert theta_r200c(profile.model,mass,z) == theta_r200c(profiles[1].model,mass,z)
    end
    # Same random harmonic realization at different synthesis resolutions.
    cl = zeros(49);cl[3:end] .= 1e-12
    first = generate_gaussian_noise_map(cl,32,48,MersenneTwister(123))
    repeat = generate_gaussian_noise_map(cl,32,48,MersenneTwister(123))
    other = generate_gaussian_noise_map(cl,32,48,MersenneTwister(124))
    @assert pixel_sha(first)==pixel_sha(repeat) && pixel_sha(first)!=pixel_sha(other)
    save_toml(joinpath(OUT,"gate.toml"),Dict("passed"=>true,"painting_relative_l2"=>errors,
        "geometry_identical"=>true,"seed_repeatability"=>true))
end

"""Probe positive spherical columns directly, avoiding unused cache corners.

Relative error is tested where the column exceeds 1e-5 of that halo's central
column. The maximum absolute error / central column is also retained, so a
faint-tail error cannot silently disappear. This is an empirical probe, not a
proof about all continuous M,z,theta values or a pixel-resolution validation.
"""
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
    return Dict("max_relative_visible"=>max_relative,"max_absolute_over_central"=>max_scaled,
        "relative_l2"=>sqrt(squared/denominator),"points"=>length(probes)),probes
end

function audit_profiles()
    for case in TASK["cases"]
        target = joinpath(OUT,case["label"]);mkpath(target)
        cfg = configure(case["theta"]).base_cfg
        records = Any[]; accepted = false
        for nodes in [[256,128,64],[512,256,128],[1024,512,256]]
            started = time();profile = build_cache(cfg,nodes)
            metrics,probes = probe_cache(profile)
            metrics["nodes"] = nodes;metrics["seconds"] = time()-started
            push!(records,metrics)
            writedlm(joinpath(target,"probes_$(nodes[1]).csv"),permutedims(hcat(probes...)),',')
            profile=nothing;GC.gc()
            if metrics["max_relative_visible"]<=.004 && metrics["max_absolute_over_central"]<=.004
                accepted=true;break
            end
        end
        save_toml(joinpath(target,"audit.toml"),Dict("accepted"=>accepted,"attempts"=>records))
        println("AUDIT ",case["label"]," ",accepted," ",records[end]);flush(stdout)
        # Keep every parameter draw; unresolved rows stop later production.
    end
end

mode = TASK["mode"]
mode=="gate" ? numerical_gate() : mode=="audit" ? audit_profiles() : run_maps()
