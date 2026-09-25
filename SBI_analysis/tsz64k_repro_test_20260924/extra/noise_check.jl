# Noise-seed check for the 65,536-row bundle, run with the bundle's frozen code.
#
# Loaded exactly like engine.jl, so read_so_noise_native_cl, so_noise_cl_vector_for_synalm,
# generate_gaussian_noise_map (= Healpix.synalm with MersenneTwister(seed), then alm2map)
# and pixel_sha are the functions the production engine calls. Every realization is
# drawn with the engine's call, Healpix.synalm(noise_cl, 7979, 7979, MersenneTwister(seed)).
#
# For independent Gaussian alms with the SO spectrum C_l, the bin sum
# B = sum_{l in bin} (2l+1) Chat_l has mean S1 = sum (2l+1) C_l and variance 2*S2
# (S2 = sum (2l+1) C_l^2); a cross spectrum of two independent maps has mean 0 and
# variance S2. The z-scores below use exactly these expectations.
ENV["BENCHMARK_LOAD_ONLY"] = "1"
include(joinpath(ENV["TSZ64K_ENGINE"], "benchmark.jl"))
using LinearAlgebra, Random, Statistics

const SPEC = TOML.parsefile(joinpath(OUT, "noise_spec.toml"))
const LMAX = 7979
# The SO table covers l = 80..7979 (no noise power below 80): the pipeline's 40 bins.
const EDGES = vcat(collect(80:200:7880), [7980])         # bin k: EDGES[k] <= l < EDGES[k+1]
const NB = length(EDGES) - 1
const NLOW = 3                                           # all-pairs test on bins 1-3 (l = 80..679)

alm_index(l, m) = m * (2LMAX + 1 - m) ÷ 2 + l + 1

function coefficient_layout()
    n = (LMAX + 1) * (LMAX + 2) ÷ 2
    ls = Vector{Int32}(undef, n); ws = Vector{Float64}(undef, n)
    for m in 0:LMAX, l in m:LMAX
        i = alm_index(l, m); ls[i] = l; ws[i] = m == 0 ? 1.0 : 2.0
    end
    return ls, ws
end

"Per bin: sum over l in the bin of (2l+1) * cross power of a and b."
function binned_cross(a::Vector{ComplexF64}, b::Vector{ComplexF64}, ls, ws)
    per_l = zeros(LMAX + 1)
    @inbounds for i in eachindex(a)
        per_l[ls[i] + 1] += ws[i] * (real(a[i]) * real(b[i]) + imag(a[i]) * imag(b[i]))
    end
    return [sum(@view per_l[EDGES[k] + 1:EDGES[k + 1]]) for k in 1:NB]
end

function summary(z)
    v = filter(isfinite, vec(z))
    return Dict("count" => length(v), "mean" => mean(v), "std" => std(v), "max_abs" => maximum(abs, v),
                "n_abs_gt_4" => count(x -> abs(x) > 4, v), "n_abs_gt_5" => count(x -> abs(x) > 5, v))
end

function main()
    started = time()
    c = configure(SPEC["theta"])
    ell, native = read_so_noise_native_cl(c.baseline_noise_path, LMAX; column=2, input_is_dl=false)
    noise_cl = so_noise_cl_vector_for_synalm(ell, native, LMAX)
    @assert length(noise_cl) == LMAX + 1
    ls, ws = coefficient_layout()
    S1 = [sum((2l + 1) * noise_cl[l + 1] for l in EDGES[k]:EDGES[k + 1] - 1) for k in 1:NB]
    S2 = [sum((2l + 1) * noise_cl[l + 1]^2 for l in EDGES[k]:EDGES[k + 1] - 1) for k in 1:NB]
    @assert all(>(0), S1) && all(>(0), S2)
    draw(seed) = Healpix.synalm(noise_cl, LMAX, LMAX, MersenneTwister(seed)).alm
    first_alm = Healpix.synalm(noise_cl, LMAX, LMAX, MersenneTwister(1))
    @assert length(first_alm.alm) == length(ls) && first_alm.lmax == LMAX && first_alm.mmax == LMAX
    first_alm = nothing

    low = [findall(i -> EDGES[k] <= ls[i] < EDGES[k + 1], eachindex(ls)) for k in 1:NLOW]
    train_rows = Int.(SPEC["train_rows"]); train = [Int.(s) for s in SPEC["train_seeds"]]
    eng_rows = Int.(SPEC["eng_rows"]); eng = [Int.(s) for s in SPEC["eng_seeds"]]
    nreal = 2length(train_rows) + 2length(eng_rows)
    features = [zeros(2length(idx), nreal) for idx in low]
    labels = String[]; auto_ratio = zeros(nreal, NB); auto_z = zeros(nreal, NB)
    pair_z = Dict(k => Vector{Vector{Float64}}() for k in
                  ("same_row_splits", "consecutive_rows", "train_vs_test_seed_same_row"))
    pair_labels = Dict(k => String[] for k in keys(pair_z))

    function record!(label, a)
        push!(labels, label); j = length(labels)
        auto = binned_cross(a, a, ls, ws)
        auto_ratio[j, :] .= auto ./ S1
        auto_z[j, :] .= (auto .- S1) ./ sqrt.(2 .* S2)
        for k in 1:NLOW
            idx = low[k]; n = length(idx)
            @inbounds for (p, i) in enumerate(idx)
                s = sqrt(ws[i])
                features[k][p, j] = s * real(a[i]); features[k][n + p, j] = s * imag(a[i])
            end
        end
    end
    pair!(kind, label, a, b) = (push!(pair_z[kind], binned_cross(a, b, ls, ws) ./ sqrt.(S2));
                                push!(pair_labels[kind], label))

    previous = nothing; first_draw = nothing
    for (r, seeds) in zip(train_rows, train)
        a1, a2 = draw(seeds[1]), draw(seeds[2])
        r == train_rows[1] && (first_draw = copy(a1))
        record!("train_$(r)_1", a1); record!("train_$(r)_2", a2)
        pair!("same_row_splits", "train_$(r)", a1, a2)
        previous === nothing || pair!("consecutive_rows", "train_$(r-1)_1|train_$(r)_1", previous, a1)
        previous = a1
    end
    previous = nothing; GC.gc()
    train_of = Dict(zip(train_rows, train))
    for (r, seeds) in zip(eng_rows, eng)
        e1, e2 = draw(seeds[1]), draw(seeds[2])
        record!("test_$(r)_1", e1); record!("test_$(r)_2", e2)
        pair!("same_row_splits", "test_$(r)", e1, e2)
        t1, t2 = draw(train_of[r][1]), draw(train_of[r][2])
        pair!("train_vs_test_seed_same_row", "row_$(r)_split1", t1, e1)
        pair!("train_vs_test_seed_same_row", "row_$(r)_split2", t2, e2)
        GC.gc()
    end
    # Positive control: the same seed drawn twice is bitwise identical, and a repeated
    # realization would show up as an enormous cross-correlation.
    again = draw(train[1][1])
    repeat_bitwise = again == first_draw
    control_z = binned_cross(again, first_draw, ls, ws) ./ sqrt.(S2)
    again = nothing; first_draw = nothing; GC.gc()

    # All pairs of all realizations at l = 2..479, where the SO noise is largest.
    allpairs = Dict{String,Any}()
    zall = Float64[]
    for k in 1:NLOW
        G = Symmetric(features[k]' * features[k])
        z = Float64[]
        for j in 1:nreal, i in 1:j-1
            push!(z, G[i, j] / sqrt(S2[k]))
        end
        allpairs["bin_$(EDGES[k])_$(EDGES[k+1]-1)"] = summary(z)
        append!(zall, z)
    end
    allpairs["all_low_bins"] = summary(zall)
    writedlm(joinpath(OUT, "allpairs_lowl_z.csv"), zall, ',')

    # Pixel SHA-256 of full noise maps (NSIDE 4096), to match against production rows.
    hashes = Dict{String,String}()
    for entry in SPEC["hash_maps"]
        label, seed = String(entry[1]), parse(Int, String(entry[2]))
        hashes[label] = pixel_sha(generate_gaussian_noise_map(noise_cl, 4096, LMAX, MersenneTwister(seed)))
        GC.gc()
    end

    writedlm(joinpath(OUT, "bins.csv"), hcat(EDGES[1:end-1], EDGES[2:end] .- 1, S1, S2), ',')
    writedlm(joinpath(OUT, "auto_ratio.csv"), auto_ratio, ',')
    writedlm(joinpath(OUT, "auto_z.csv"), auto_z, ',')
    writedlm(joinpath(OUT, "labels.txt"), labels)
    for (kind, rows) in pair_z
        writedlm(joinpath(OUT, "pair_z_$(kind).csv"), permutedims(hcat(rows...)), ',')
        writedlm(joinpath(OUT, "pair_labels_$(kind).txt"), pair_labels[kind])
    end
    save_toml(joinpath(OUT, "noise_check.toml"), Dict(
        "realizations" => nreal, "train_rows" => length(train_rows), "test_seed_rows" => length(eng_rows),
        "noise_table" => c.baseline_noise_path, "lmax" => LMAX, "bins" => NB,
        "auto_power_z" => summary(auto_z), "auto_power_ratio_mean_per_bin" => vec(mean(auto_ratio; dims=1)),
        "pairs" => Dict(k => summary(permutedims(hcat(v...))) for (k, v) in pair_z),
        "all_pairs_low_ell" => allpairs, "same_seed_twice_bitwise_identical" => repeat_bitwise,
        "same_seed_twice_min_z" => minimum(control_z),
        "map_pixel_sha256" => hashes, "seconds" => time() - started))
    println("noise check done in $(round(time() - started; digits=1)) s")
end

main()
