# Process-local replacement of cache loading. The archived signal painting,
# beam, mask multiplication and cross-spectrum implementation remain in use.
# Only the two noise arrays are synthesized afresh for this immutable row.

function noise_ensemble_check(noise_cl)
    lmax, samples = 127, 64
    cl = noise_cl[1:lmax+1]
    selected = [l for l in 20:lmax if cl[l+1] > 0]
    modes = sum(2l+1 for l in selected)
    auto1, auto2, cross = 0.0, 0.0, 0.0
    for index in 1:samples
        # This diagnostic stream is separate from the training streams.
        a = Healpix.synalm(cl, lmax, lmax, MersenneTwister(9000000+2index))
        b = Healpix.synalm(cl, lmax, lmax, MersenneTwister(9000001+2index))
        aa, bb, ab = Healpix.alm2cl(a), Healpix.alm2cl(b), Healpix.alm2cl(a,b)
        auto1 += sum((2l+1)*aa[l+1]/cl[l+1] for l in selected)
        auto2 += sum((2l+1)*bb[l+1]/cl[l+1] for l in selected)
        cross += sum((2l+1)*ab[l+1]/cl[l+1] for l in selected)
    end
    count = samples*modes
    values = [auto1/count, auto2/count, cross/count]
    zscores = [(values[1]-1)/sqrt(2/count), (values[2]-1)/sqrt(2/count),
               values[3]/sqrt(1/count)]
    @assert maximum(abs.(zscores)) < 8
    return Dict("draw_pairs"=>samples, "modes_per_draw"=>modes,
                "auto1_over_Nell"=>values[1], "auto2_over_Nell"=>values[2],
                "cross_over_Nell"=>values[3], "standardized_errors"=>zscores)
end

function load_cache(campaign, cfg)
    path = joinpath(campaign, "cache", "shared_mask_noise.hdf5")
    meta = TOML.parsefile(joinpath(campaign, "cache", "complete.toml"))
    @assert file_sha(path) == meta["cache_file_sha256"]
    for (key, actual) in runtime_info()
        key == "threads" && continue
        @assert meta["runtime"][key] == actual "Runtime/source differs: $key"
    end
    @assert meta["noise_table_sha256"] == file_sha(cfg.baseline_noise_path)
    mask = h5open(path, "r") do handle
        HealpixMap{Float64, RingOrder}(read(handle["mask"]))
    end
    @assert pixel_sha(mask) == meta["mask_pixel_sha256"]
    ell, native = read_so_noise_native_cl(cfg.baseline_noise_path, cfg.noise_lmax;
                                        column=2, input_is_dl=false)
    noise_cl = so_noise_cl_vector_for_synalm(ell, native, cfg.noise_lmax)
    seeds = parse.(Int, split(ENV["EXTENDED_NOISE_SPLIT_SEEDS"], ","))
    @assert length(seeds) == 2 && seeds[1] != seeds[2]
    noise1 = generate_gaussian_noise_map(noise_cl, 4096, 7979, MersenneTwister(seeds[1]))
    sha1 = pixel_sha(noise1)
    checks = Dict{String,Any}()
    if get(ENV, "EXTENDED_NOISE_VALIDATE", "0") == "1"
        replay = generate_gaussian_noise_map(noise_cl, 4096, 7979, MersenneTwister(seeds[1]))
        @assert pixel_sha(replay) == sha1
        replay = nothing
        GC.gc()
        checks["full_resolution_replay_passed"] = true
        checks["harmonic_ensemble"] = noise_ensemble_check(noise_cl)
    end
    noise2 = generate_gaussian_noise_map(noise_cl, 4096, 7979, MersenneTwister(seeds[2]))
    sha2 = pixel_sha(noise2)
    @assert sha1 != sha2
    meta["noise1_pixel_sha256"], meta["noise2_pixel_sha256"] = sha1, sha2
    provenance = Dict{String,Any}(
        "mode"=>"independent_per_row", "seed_algorithm"=>ENV["EXTENDED_NOISE_SEED_ALGORITHM"],
        "master_seed"=>parse(Int, ENV["EXTENDED_NOISE_MASTER_SEED"]),
        "seed_row_id"=>parse(Int, ENV["EXTENDED_NOISE_ROW_ID"]), "split_seeds"=>seeds,
        "noise1_pixel_sha256"=>sha1, "noise2_pixel_sha256"=>sha2,
        "mask_pixel_sha256"=>meta["mask_pixel_sha256"],
        "noise_table_sha256"=>meta["noise_table_sha256"],
        "noise_lmax"=>7979, "noise_beam_applied"=>false,
        "split_Nell_multiplier"=>1.0, "checks"=>checks)
    save_toml(joinpath(ROW_OUTPUT, "noise.toml"), provenance)
    return [mask, noise1, noise2], meta
end
