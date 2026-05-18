using XGPaint, Healpix, HDF5, Interpolations

include(joinpath(@__DIR__, "config.jl"))
include(joinpath(@__DIR__, "instrumentation.jl"))
include(joinpath(@__DIR__, "output.jl"))
include(joinpath(@__DIR__, "model.jl"))

function run_interpolator_cache_build()
    t0 = time()
    cfg = try
        load_visual_config()
    catch err
        if err isa SkipVisualRun
            println("Skipping interpolator cache build: $(err.message)")
            println("Interpolator cache step elapsed time: $(round(time() - t0; digits=2)) s")
            return nothing
        end
        rethrow()
    end

    if get_bool_arg("print_run_summary", true; env="PRINT_RUN_SUMMARY")
        print_visual_config(cfg)
    end

    if cfg.skip_existing_outputs && visual_outputs_complete(cfg; allow_any_run_instance=cfg.skip_existing_any_run_instance)
        if existing_visual_interpolator_cache(cfg) !== nothing
            println("Skipping interpolator cache build because required outputs and cache already exist.")
            println("Interpolator cache step elapsed time: $(round(time() - t0; digits=2)) s")
            return nothing
        end
        println("Required outputs already exist, but the cache is missing; building cache for dependent lightcones.")
    end

    if cfg.model_exists
        println("Skipping interpolator cache build because model_exists=true.")
        println("Interpolator cache step elapsed time: $(round(time() - t0; digits=2)) s")
        return nothing
    end

    build_visual_interpolator(cfg)
    println("Interpolator cache step elapsed time: $(round(time() - t0; digits=2)) s")
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_interpolator_cache_build()
end
