# Calibration of the frozen 8k forward model. Repaint once per task and reuse
# the signal for a batch of independent split-noise draws.
using TOML
const OUTPUT = ENV["CAL_OUTPUT"]
mkpath(OUTPUT)
ENV["FLAMINGO_MODE"] = "probe"
ENV["FLAMINGO_OUTPUT"] = joinpath(OUTPUT, "operator_probe.toml")
include(joinpath(ENV["FLAMINGO_CAMPAIGN"], "code", "process_maps.jl"))
include(joinpath(ENV["CAL_FROZEN_CODE"], "stable_los.jl"))

function calibrate()
    cfg = load_halfdome_fullsky_so_noise_config()
    @assert cfg.base_cfg.nside == 4096 && cfg.base_cfg.cl_lmax == 7979
    @assert cfg.base_cfg.cl_niter == 0 && cfg.noise_lmax == 7979
    @assert cfg.base_cfg.gaussian_beam_fwhm_arcmin == 2.0
    @assert cfg.fsky == .4 && cfg.mask_seed == 12345
    save_toml(joinpath(OUTPUT, "parameters.toml"),
        Dict(string(k)=>Float64(v) for (k,v) in pairs(cfg.base_cfg.battaglia_params)))
    signal = paint_halfdome_fullsky_signal_map(cfg.base_cfg)
    mask = random_apodized_cap_mask(4096, cfg.fsky,
        cfg.mask_apodization_arcmin, MersenneTwister(cfg.mask_seed)).mask
    cache = TOML.parsefile(joinpath(ENV["FLAMINGO_CAMPAIGN"], "cache", "complete.toml"))
    @assert pixel_sha(mask) == cache["mask_pixel_sha256"]
    @assert file_sha(cfg.baseline_noise_path) == cache["noise_table_sha256"]
    for (key, value) in runtime_info()
        key == "threads" && continue
        @assert cache["runtime"][key] == value "Frozen runtime mismatch: $key"
    end
    clean_map = masked_signal_map(signal, mask)
    clean = compute_cl(cfg.base_cfg, clean_map)
    write_npy_float64_vector(joinpath(OUTPUT, "clean.npy"), clean, "calibration clean")
    clean_map = nothing
    GC.gc()
    ell, native = read_so_noise_native_cl(cfg.baseline_noise_path, 7979;
                                        column=2, input_is_dl=false)
    noise_cl = so_noise_cl_vector_for_synalm(ell, native, 7979)
    seeds = TOML.parsefile(ENV["CAL_SEEDS_FILE"])["pairs"]
    noise_hashes = String[]
    for (i, pair) in enumerate(seeds)
        noise1 = generate_gaussian_noise_map(noise_cl, 4096, 7979, MersenneTwister(pair[1]))
        noise2 = generate_gaussian_noise_map(noise_cl, 4096, 7979, MersenneTwister(pair[2]))
        push!(noise_hashes, pixel_sha(noise1), pixel_sha(noise2))
        @. noise1.pixels = (signal.pixels + noise1.pixels) * mask.pixels
        @. noise2.pixels = (signal.pixels + noise2.pixels) * mask.pixels
        cross = compute_cross_cl(cfg.base_cfg, noise1, noise2)
        write_npy_float64_vector(joinpath(OUTPUT, "noisy_$(i).npy"), cross, "independent calibration cross")
        noise1 = nothing
        noise2 = nothing
        GC.gc()
        println("Completed independent pair ", i, "/", length(seeds))
        flush(stdout)
    end
    save_toml(joinpath(OUTPUT, "complete.toml"), Dict(
        "runtime"=>runtime_info(), "mask_pixel_sha256"=>pixel_sha(mask),
        "signal_beamed_pixel_sha256"=>pixel_sha(signal),
        "noise_table_sha256"=>file_sha(cfg.baseline_noise_path),
        "stable_los_sha256"=>file_sha(joinpath(ENV["CAL_FROZEN_CODE"], "stable_los.jl")),
        "split_seeds"=>seeds, "noise_pixel_hashes"=>noise_hashes,
        "noise_Nell_multiplier"=>1.0, "status"=>"complete"))
end
calibrate()
