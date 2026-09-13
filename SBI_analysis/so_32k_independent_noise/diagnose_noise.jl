# Real HEALPix noise tests, without painting the expensive halo catalogue.
using Test
include(joinpath(@__DIR__, "simulator", "tSZ_visuals", "run_halfdome_fullsky_so_noise.jl"))
@test VERSION == v"1.12.2"
@test realpath(pathof(XGPaint)) == realpath(joinpath(@__DIR__, "vendor/XGPaint/src/XGPaint.jl"))

function main()
    out, plan_path = ARGS[1:2]
    nside, lmax = parse.(Int, ARGS[3:4])
    mask_seed = parse(Int, ARGS[5])
    append!(ARGS, ["baseline_noise_path=" * joinpath(@__DIR__, "noise", SO_NOISE_BASELINE_FILENAME),
                   "goal_noise_path=" * joinpath(@__DIR__, "noise", SO_NOISE_GOAL_FILENAME),
                   "nside=$(nside)",
                   "so_noise_deprojections=0,2", "sobol_csv_path=", "print_runtime_environment=false"])
    replace_or_add_arg!(ARGS, "cl_lmax", lmax)
    replace_or_add_arg!(ARGS, "so_noise_lmax", lmax)
    template = load_halfdome_fullsky_so_noise_config()
    template_fields = (; (name => getproperty(template, name) for name in fieldnames(typeof(template)))...)
    mask_info = random_apodized_cap_mask(nside, 0.4, 60.0, MersenneTwister(mask_seed))
    mask = mask_info.mask
    @test mask.pixels == random_apodized_cap_mask(nside, 0.4, 60.0, MersenneTwister(mask_seed)).mask.pixels
    # Exclude shared zero-mask pixels from the map-correlation calculation.
    pixels = findall(mask.pixels .> 0.99)[1:8:end]
    plan, _ = readdlm(plan_path, ',', Any; header=true)
    fields = Matrix{Float64}(undef, length(pixels), size(plan, 1))
    auto = Matrix{Float64}(undef, lmax + 1, size(plan, 1))
    cross = Matrix{Float64}(undef, lmax + 1, div(size(plan, 1), 2))
    image_indices = [Healpix.ang2pixRing(mask.resolution, pi/2 - lat, mod(lon, 2pi))
                     for lat in range(-pi/2+0.01, pi/2-0.01; length=144),
                         lon in range(-pi, pi; length=288)]
    writedlm(joinpath(out, "mask.csv"), mask.pixels[image_indices], ',')
    map1 = nothing
    for (i, row) in enumerate(eachrow(plan))
        scheme, product, mode, design_row, split, seed, root = row
        seed, root, split = Int(seed), Int(root), Int(split)
        case, deprojection = split_product(String(product))
        # Exercise the exact seed function used in production, not only Python's copy.
        cfg = HalfDomeFullSkySONoiseConfig(; merge(template_fields, (noise_seed=root, noise_deprojection=deprojection))...)
        @test noise_split_seed(cfg, case, split) == seed
        path = joinpath(@__DIR__, "noise", "SO_LAT_Nell_T_atmv1_$(case)_fsky0p4_ILC_tSZ.txt")
        ell, native = read_so_noise_native_cl(path, lmax; column=deprojection+2)
        cl = so_noise_cl_vector_for_synalm(ell, native, lmax)
        map = generate_gaussian_noise_map(cl, nside, lmax, MersenneTwister(seed))
        fields[:, i] = map.pixels[pixels]
        # Harmonic power, before pixelization/masking, has an exact N_ell expectation.
        alm = Healpix.synalm(cl, lmax, lmax, MersenneTwister(seed))
        auto[:, i] = Healpix.alm2cl(alm, alm)
        masked = masked_signal_map(map, mask)
        if split == 1
            map1 = masked
            if product == "baseline_deproj0" && mode == "two_param" && design_row <= 2
                writedlm(joinpath(out, "map_$(scheme)_row$(Int(design_row)).csv"),
                         masked.pixels[image_indices], ',')
            end
        else
            cross[:, div(i, 2)] = Healpix.anafast(map1, masked; lmax=lmax, niter=0)
        end
        println("Noise diagnostic ", i, "/", size(plan, 1), ": ", scheme, " ", product, " seed=", seed)
    end
    writedlm(joinpath(out, "map_correlations.csv"), cor(fields), ',')
    writedlm(joinpath(out, "auto_cl.csv"), auto, ',')
    writedlm(joinpath(out, "masked_cross_cl.csv"), cross, ',')
    writedlm(joinpath(out, "mask_moments.csv"),
             [mask_info.support_fsky mask_info.mean_weight mask_info.mean_weight2], ',')
end

function split_product(product)
    parts = split(product, "_deproj")
    return parts[1], parse(Int, parts[2])
end

main()
