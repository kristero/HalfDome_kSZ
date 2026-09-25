# Broad-prior off-grid cache test. No sky maps or training rows are generated.
ENV["BENCHMARK_LOAD_ONLY"] = "1"
include(joinpath(@__DIR__, "..", "benchmark.jl"))
using Statistics

design = readdlm(joinpath(@__DIR__, "theta_probe.csv"), ',', Float64)
results = Dict[]
for row in axes(design,1)
    cfg = configure(design[row,:]).base_cfg
    model = ChordMeanProfile(build_tsz_model(cfg),4.)
    rng = MersenneTwister(219260921)
    # Off-grid probes over the full cache M,z domain, including unused corners.
    masses = exp10.(12 .+ 3.7rand(rng,1024))
    redshifts = exp.(log(.001) .+ log(5000)*rand(rng,1024))
    impact = exp.(log(1e-8) .+ log(4/1e-8)*rand(rng,1024))
    angles = [impact[i]*theta_r200c(model,masses[i],redshifts[i]) for i in 1:1024]
    direct = [spherical_value_direct(model,angles[i],masses[i],redshifts[i]) for i in 1:1024]
    central = [spherical_value_direct(model,0.,masses[i],redshifts[i]) for i in 1:1024]
    @assert all(isfinite,direct) && all(direct .>= 0) && all(central .> 0)
    for nodes in ([512,256,128],[256,128,64],[128,64,32])
        start = time();profile = build_cache(cfg,nodes);cache_seconds=time()-start
        theta_min = exp(first(profile.itp.ranges[1]))
        estimate = [profile(max(angles[i],theta_min),masses[i],redshifts[i]) *
            chord_factor(angles[i],4theta_r200c(model,masses[i],redshifts[i]),4.) for i in 1:1024]
        @assert all(isfinite,estimate) && all(estimate .>= 0)
        scaled = abs.(estimate-direct)./central
        # Avoid calling relative errors in virtually zero tails a physical failure.
        visible = direct .> central .* 1e-12
        relative = abs.(estimate[visible]./direct[visible].-1)
        push!(results,Dict("row"=>row,"nodes"=>nodes,"cache_seconds"=>cache_seconds,
            "central_scaled_quantiles"=>quantile(scaled,[.5,.95,.99,1.]),
            "relative_visible_quantiles"=>quantile(relative,[.5,.95,.99,1.]),
            "visible_points"=>sum(visible),"floor_hits"=>sum(angles.<theta_min)))
        save_toml(joinpath(OUT,"cache_probe.toml"),Dict("cases"=>results,
            "scope"=>"Pointwise central-scaled errors, not full-spectrum certification"))
        println("PROBE row=",row," nodes=",nodes," max_central_scaled=",maximum(scaled));flush(stdout)
        profile=nothing;GC.gc()
    end
end
save_toml(joinpath(OUT,"cache_probe_complete.toml"),Dict("complete"=>true,"rows"=>size(design,1),
    "grids_per_row"=>3,"points_per_grid"=>1024))
