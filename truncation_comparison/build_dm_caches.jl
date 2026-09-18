using XGPaint
include(joinpath(@__DIR__, "dm_profiles_common.jl"))
cache_dir = joinpath(@__DIR__, "caches"); overwrite = "--overwrite" in ARGS
p = dm_inner_profiles()
for (M, z) in ((1e13, 0.5), (1e15, 0.1)); isapprox(theta_r200c(p.b16, M, z), theta_r200c(p.lee22, M, z); rtol=1e-6) || error("cosmology mismatch between DM profiles"); end
for (name, model) in zip(DM_CACHE_NAMES, dm_models())
    file = joinpath(cache_dir, "$(name)_interpolator.jld2")
    if isfile(file) && !overwrite; println("exists, skipping: $file"); continue; end
    println("Building $name with $(Threads.nthreads()) threads"); flush(stdout); t0 = time()
    build_interpolator(model; cache_file=file, N_logtheta=512, pad=128, logM_max=15.7, overwrite=true, verbose=true)
    println("done $name in $(round(time() - t0; digits=1)) s"); flush(stdout)
end
