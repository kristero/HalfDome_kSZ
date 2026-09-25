# Experimental diagnostic, NOT a certified production renderer.
ENV["PREFLIGHT_LOAD_ONLY"]="1"
include(joinpath(@__DIR__,"fullsky_test.jl"))

function noisy_diagnostic()
    cfg=load_halfdome_fullsky_so_noise_config()
    @assert cfg.base_cfg.nside==4096 && cfg.base_cfg.cl_lmax==7979
    @assert cfg.fsky==.4 && cfg.mask_seed==12345
    seeds=parse.(Int,split(ENV["DIAGNOSTIC_SPLIT_SEEDS"],","))
    @assert length(seeds)==2 && seeds[1]!=seeds[2]
    start=time()
    signal=paint_halfdome_fullsky_signal_map(cfg.base_cfg)
    signal_seconds=time()-start
    clean=compute_cl(cfg.base_cfg,signal)
    write_npy_float64_vector(joinpath(OUT,"unmasked_clean_cl.npy"),clean,"diagnostic unmasked clean")
    mask=random_apodized_cap_mask(4096,.4,cfg.mask_apodization_arcmin,MersenneTwister(12345))
    signal.pixels .*= mask.mask.pixels
    write_npy_float64_vector(joinpath(OUT,"masked_clean_cl.npy"),compute_cl(cfg.base_cfg,signal),"diagnostic masked clean")
    ell,native=read_so_noise_native_cl(cfg.baseline_noise_path,7979;column=2,input_is_dl=false)
    noise_cl=so_noise_cl_vector_for_synalm(ell,native,7979)
    first=generate_gaussian_noise_map(noise_cl,4096,7979,MersenneTwister(seeds[1]))
    first_sha=pixel_sha(first)
    second=generate_gaussian_noise_map(noise_cl,4096,7979,MersenneTwister(seeds[2]))
    second_sha=pixel_sha(second)
    @assert first_sha!=second_sha
    first.pixels .= first.pixels .* mask.mask.pixels .+ signal.pixels
    second.pixels .= second.pixels .* mask.mask.pixels .+ signal.pixels
    cross=compute_cross_cl(cfg.base_cfg,first,second)
    @assert all(isfinite,cross)
    write_npy_float64_vector(joinpath(OUT,"masked_noisy_cross_cl.npy"),cross,"diagnostic signed cross")
    save_toml(joinpath(OUT,"diagnostic.toml"),Dict("seconds"=>time()-start,
        "signal_seconds"=>signal_seconds,"split_seeds"=>seeds,
        "noise1_sha256"=>first_sha,"noise2_sha256"=>second_sha,
        "mask_sha256"=>pixel_sha(mask.mask),"noise_beam_applied"=>false,
        "split_Nell_multiplier"=>1.,"cache"=>CACHE_STATS,
        "production_certified"=>false,
        "purpose"=>"Experimental flat-prior pipeline diagnostic; not production-certified"))
end
noisy_diagnostic()
