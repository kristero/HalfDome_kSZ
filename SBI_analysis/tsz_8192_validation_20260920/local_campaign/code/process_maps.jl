# Reuse the HalfDome map operators verbatim; replace only the signal source.
using TOML, SHA, Pkg

const SOURCE_DIR = ENV["HALFDOME_SOURCE_DIR"]
include(joinpath(SOURCE_DIR, "tSZ_visuals", "run_halfdome_fullsky_so_noise.jl"))

file_sha(path) = open(path, "r") do stream
    bytes2hex(sha256(stream))
end
pixel_sha(map) = bytes2hex(sha256(reinterpret(UInt8, map.pixels)))

function save_toml(path, values)
    open(path, "w") do stream
        TOML.print(stream, values)
    end
end

function runtime_info()
    dependencies = Pkg.dependencies()
    return Dict(
        "julia_version" => string(VERSION),
        "healpix_version" => string(dependencies[Base.PkgId(Healpix).uuid].version),
        "hdf5_version" => string(dependencies[Base.PkgId(HDF5).uuid].version),
        "xgpaint_source" => pathof(XGPaint),
        "xgpaint_sha256" => file_sha(pathof(XGPaint)),
        "threads" => Threads.nthreads(),
        "adapter_sha256" => file_sha(@__FILE__),
        "operator_sha256" => file_sha(joinpath(SOURCE_DIR, "tSZ_visuals", "run_halfdome_fullsky_so_noise.jl")),
        "beam_source_sha256" => file_sha(joinpath(SOURCE_DIR, "tSZ_visuals", "output.jl")),
    )
end

function probe(output)
    # A small regression fingerprint, not an astronomical map or science test.
    mask = random_apodized_cap_mask(8, 0.4, 60.0, MersenneTwister(12345))
    cl = [ell < 2 ? 0.0 : 1e-12 / (ell * (ell + 1)) for ell in 0:15]
    noise = generate_gaussian_noise_map(cl, 8, 15, MersenneTwister(22446))
    coefficients = Healpix.synalm(cl, 15, 15, MersenneTwister(22446))
    original = copy(noise.pixels)
    smooth_healpix_map_gaussian!(noise, 2.0; niter=0)
    @assert all(isfinite, noise.pixels)
    @assert mask.support_fsky > 0.35 && mask.support_fsky < 0.45
    reference_sum = masked_signal_plus_noise_map(noise, HealpixMap{Float64, RingOrder}(original), mask.mask)
    inplace_sum = copy(original)
    @. inplace_sum = (noise.pixels + inplace_sum) * mask.mask.pixels
    @assert inplace_sum == reference_sum.pixels
    result = runtime_info()
    result["rng_uniform"] = rand(MersenneTwister(12345), 12)
    result["rng_normal"] = randn(MersenneTwister(22446), 12)
    result["mask_pixel_sha256"] = pixel_sha(mask.mask)
    result["noise_pixel_sha256"] = bytes2hex(sha256(reinterpret(UInt8, original)))
    result["beam_pixel_sha256"] = pixel_sha(noise)
    result["noise_first_pixels"] = original[1:12]
    result["beam_first_pixels"] = noise.pixels[1:12]
    result["status"] = "small_operator_probe_passed"
    result["inplace_masked_sum_matches_original"] = true
    save_toml(output, result)
    println("Small operator probe passed: ", output)
end

function load_cache(campaign, cfg)
    path = joinpath(campaign, "cache", "shared_mask_noise.hdf5")
    meta = TOML.parsefile(joinpath(campaign, "cache", "complete.toml"))
    @assert file_sha(path) == meta["cache_file_sha256"]
    for (key, actual) in runtime_info()
        # The thread count can vary for probes, but production uses 26.
        key == "threads" && continue
        @assert meta["runtime"][key] == actual "Runtime/source differs from shared cache: $key"
    end
    @assert meta["noise_table_sha256"] == file_sha(cfg.baseline_noise_path)
    maps = h5open(path, "r") do handle
        [HealpixMap{Float64, RingOrder}(read(handle[name])) for name in ("mask", "noise1", "noise2")]
    end
    for (name, map) in zip(("mask", "noise1", "noise2"), maps)
        @assert pixel_sha(map) == meta[name * "_pixel_sha256"]
    end
    return maps, meta
end

function build_cache(campaign, cfg)
    directory = joinpath(campaign, "cache")
    mkpath(directory)
    marker = joinpath(directory, "complete.toml")
    if isfile(marker)
        load_cache(campaign, cfg)
        println("Verified existing shared mask and noise cache")
        return
    end
    path = joinpath(directory, "shared_mask_noise.hdf5")
    isfile(path) && error("Incomplete cache exists; inspect before overwriting")
    mask_info = random_apodized_cap_mask(4096, 0.4, 60.0, MersenneTwister(12345))
    ell, native = read_so_noise_native_cl(cfg.baseline_noise_path, 7979; column=2, input_is_dl=false)
    noise_cl = so_noise_cl_vector_for_synalm(ell, native, 7979)
    seeds = [noise_split_seed(cfg, "baseline", split) for split in 1:2]
    @assert seeds == [22446, 22447]
    noise1 = generate_gaussian_noise_map(noise_cl, 4096, 7979, MersenneTwister(seeds[1]))
    noise2 = generate_gaussian_noise_map(noise_cl, 4096, 7979, MersenneTwister(seeds[2]))
    mask = mask_info.mask
    h5open(path, "w") do handle
        handle["mask"] = mask.pixels
        handle["noise1"] = noise1.pixels
        handle["noise2"] = noise2.pixels
    end
    meta = Dict("runtime" => runtime_info(), "mask_seed" => 12345,
                "noise_root_seed" => 12345, "split_seeds" => seeds,
                "noise_table_sha256" => file_sha(cfg.baseline_noise_path),
                "mask_pixel_sha256" => pixel_sha(mask),
                "noise1_pixel_sha256" => pixel_sha(noise1),
                "noise2_pixel_sha256" => pixel_sha(noise2),
                "support_fsky" => mask_info.support_fsky,
                "mask_mean" => mask_info.mean_weight,
                "mask_mean_square" => mask_info.mean_weight2,
                "cache_file_sha256" => file_sha(path))
    # Retain the actual fixed noise-noise term for the diagnostic decomposition.
    noise1.pixels .*= mask.pixels
    noise2.pixels .*= mask.pixels
    cross = compute_cross_cl(cfg.base_cfg, noise1, noise2)
    write_npy_float64_vector(joinpath(directory, "masked_noise_cross_cl.npy"), cross, "fixed noise cross spectrum")
    save_toml(marker, meta)
    println("Shared mask/noise cache complete")
end

function process_signal(campaign, output, input, cfg)
    mkpath(output)
    isfile(joinpath(output, "complete.toml")) && error("Completed map output exists")
    if input == "halfdome"
        # Measure the actual selected catalogue redshift support in small chunks.
        # This is diagnostic only; keep the original halo selection unchanged.
        support = h5open(cfg.base_cfg.halfdome_path, "r") do handle
            zmin, zmax, selected = Inf, -Inf, 0
            count = length(handle["redshift"])
            for first in 1:1_000_000:count
                indices = first:min(first + 999_999, count)
                redshift = handle["redshift"][indices]
                mass = handle["halo_mass_m200c"][indices] ./ cfg.base_cfg.cosmo_h
                keep = isfinite.(redshift) .& isfinite.(mass) .& (redshift .>= 0) .& (mass .>= cfg.base_cfg.mass_min)
                if any(keep)
                    values = redshift[keep]
                    zmin, zmax = min(zmin, minimum(values)), max(zmax, maximum(values))
                    selected += length(values)
                end
            end
            Dict("selected_redshift_min" => Float64(zmin), "selected_redshift_max" => Float64(zmax),
                 "selected_halos" => selected, "mass_min_Msun" => cfg.base_cfg.mass_min,
                 "h" => cfg.base_cfg.cosmo_h, "Omega_b" => cfg.base_cfg.cosmo_omegab,
                 "Omega_c" => cfg.base_cfg.cosmo_omegac)
        end
        save_toml(joinpath(output, "halfdome_catalogue_support.toml"), support)
        println("Painting fresh HalfDome Battaglia12 control")
        signal = paint_halfdome_fullsky_signal_map(cfg.base_cfg)
    else
        println("Reading FLAMINGO y map: ", input)
        values = h5open(input, "r") do handle
            read(handle["data"])
        end
        @assert length(values) == 12 * 4096^2 && eltype(values) == Float64
        @assert all(isfinite, values)
        signal = HealpixMap{Float64, RingOrder}(values)
        # Same function, default harmonic limit and niter as HalfDome.
        smooth_healpix_map_gaussian!(signal, 2.0; niter=0)
    end
    GC.gc()
    signal_digest = pixel_sha(signal)
    unmasked_cl = compute_cl(cfg.base_cfg, signal)
    write_npy_float64_vector(joinpath(output, "unmasked_clean_cl.npy"), unmasked_cl, "unmasked beamed clean Cl")
    maps, cache_meta = load_cache(campaign, cfg)
    mask, noise1, noise2 = maps
    masked_clean = masked_signal_map(signal, mask)
    clean_cl = compute_cl(cfg.base_cfg, masked_clean)
    write_npy_float64_vector(joinpath(output, "masked_clean_cl.npy"), clean_cl, "masked clean Cl")
    masked_clean = nothing
    GC.gc()
    # Reuse the cache arrays in place to limit peak memory. Expressions match
    # masked_signal_plus_noise_map exactly: (signal + noise) * mask.
    @. noise1.pixels = (signal.pixels + noise1.pixels) * mask.pixels
    @. noise2.pixels = (signal.pixels + noise2.pixels) * mask.pixels
    noisy_cl = compute_cross_cl(cfg.base_cfg, noise1, noise2)
    write_npy_float64_vector(joinpath(output, "masked_noisy_cross_cl.npy"), noisy_cl, "masked noisy split cross Cl")
    result = Dict("runtime" => runtime_info(), "source" => input,
                  "signal_beamed_pixel_sha256" => signal_digest,
                  "cache_file_sha256" => cache_meta["cache_file_sha256"],
                  "mask_pixel_sha256" => cache_meta["mask_pixel_sha256"],
                  "noise1_pixel_sha256" => cache_meta["noise1_pixel_sha256"],
                  "noise2_pixel_sha256" => cache_meta["noise2_pixel_sha256"],
                  "nside" => 4096, "beam_fwhm_arcmin" => 2.0, "ell_max" => 7979,
                  "signal_mean" => mean(signal.pixels), "signal_min" => minimum(signal.pixels),
                  "signal_max" => maximum(signal.pixels),
                  "status" => "map_processing_completed")
    input != "halfdome" && (result["input_sha256"] = file_sha(input))
    save_toml(joinpath(output, "complete.toml"), result)
    println("Map processing complete: ", output)
end

function main()
    mode = ENV["FLAMINGO_MODE"]
    if mode == "probe"
        probe(ENV["FLAMINGO_OUTPUT"])
        return
    end
    cfg = load_halfdome_fullsky_so_noise_config()
    @assert cfg.base_cfg.nside == 4096 && cfg.base_cfg.cl_lmax == 7979
    @assert cfg.base_cfg.cl_niter == 0 && cfg.noise_lmax == 7979
    @assert cfg.noise_seed == cfg.mask_seed == 12345 && cfg.noise_deprojection == 0
    @assert cfg.base_cfg.gaussian_beam_fwhm_arcmin == 2.0
    @assert cfg.fsky == 0.4 && cfg.mask_apodization_arcmin == 60.0
    if mode == "cache"
        build_cache(ENV["FLAMINGO_CAMPAIGN"], cfg)
    elseif mode == "map"
        process_signal(ENV["FLAMINGO_CAMPAIGN"], ENV["FLAMINGO_OUTPUT"], ENV["FLAMINGO_INPUT"], cfg)
    else
        error("Unknown mode: $mode")
    end
end

main()
