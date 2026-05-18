using XGPaint, Healpix, HDF5, Interpolations
using Base.Threads

include(joinpath(@__DIR__, "config.jl"))
include(joinpath(@__DIR__, "binning.jl"))
include(joinpath(@__DIR__, "cosmology_helpers.jl"))
include(joinpath(@__DIR__, "instrumentation.jl"))
include(joinpath(@__DIR__, "output.jl"))
include(joinpath(@__DIR__, "painting.jl"))
include(joinpath(@__DIR__, "model.jl"))
include(joinpath(@__DIR__, "catalog_halfdome.jl"))
include(joinpath(@__DIR__, "catalog_websky.jl"))

if !isdefined(@__MODULE__, :visual_outputs_complete)
    function visual_outputs_complete(cfg::VisualConfig; allow_any_run_instance::Bool=false)
        println("Output resume helpers are unavailable; continuing without skip-existing-output detection.")
        return false
    end
end

if !isdefined(@__MODULE__, :existing_completed_outputs)
    function existing_completed_outputs(cfg::VisualConfig; allow_any_run_instance::Bool=false)
        return String[]
    end
end

if !isdefined(@__MODULE__, :compute_cl)
    function compute_cl(cfg::VisualConfig, m_hp)
        if cfg.cl_lmax < 0
            println("compute_cl helper is unavailable; using Healpix default anafast with niter=$(cfg.cl_niter).")
            return Healpix.anafast(m_hp; niter=cfg.cl_niter)
        end

        println("compute_cl helper is unavailable; using Healpix.anafast with lmax=$(cfg.cl_lmax), niter=$(cfg.cl_niter).")
        return Healpix.anafast(m_hp; lmax=cfg.cl_lmax, niter=cfg.cl_niter)
    end
end

function safe_print_runtime_environment()
    if !isdefined(@__MODULE__, :print_runtime_environment)
        println("Runtime environment logging unavailable: print_runtime_environment is not defined.")
        println("This usually means run_tSZ_visuals.jl was updated without the matching instrumentation.jl.")
        println("Julia version: $(VERSION)")
        println("Julia threads available: $(Base.Threads.nthreads())")
        return nothing
    end

    try
        getfield(@__MODULE__, :print_runtime_environment)()
    catch err
        println("Runtime environment logging failed ($(typeof(err))): $(err)")
        println("Continuing because runtime logging is diagnostic only.")
    end
    return nothing
end

function run_tsz_visual_fits()
    t0 = time()
    cfg = try
        load_visual_config()
    catch err
        if err isa SkipVisualRun
            println("Skipping tSZ visual run: $(err.message)")
            println("Total elapsed time: $(round(time() - t0; digits=2)) s")
            return nothing
        end
        rethrow()
    end

    ensure_output_dir(cfg)
    if cfg.skip_existing_outputs && visual_outputs_complete(cfg; allow_any_run_instance=cfg.skip_existing_any_run_instance)
        existing_outputs = existing_completed_outputs(cfg; allow_any_run_instance=cfg.skip_existing_any_run_instance)
        println("Skipping tSZ visual run because required outputs already exist:")
        for path in existing_outputs
            println("  $(abspath(path))")
        end
        println("Total elapsed time: $(round(time() - t0; digits=2)) s")
        return nothing
    end

    if get_bool_arg("print_run_summary", true; env="PRINT_RUN_SUMMARY")
        print_visual_config(cfg)
    end
    if get_bool_arg("print_runtime_environment", false; env="PRINT_RUNTIME_ENVIRONMENT")
        safe_print_runtime_environment()
    end

    y_model_interp = build_visual_interpolator(cfg)
    trim_process_memory()
    state = init_visual_maps(cfg)

    map_label = cfg.save_mass_map ? "y and mass FITS maps" : "y FITS maps"
    println("Painting $(cfg.binning_tag) $(map_label); nside=$(cfg.nside), threads=$(nthreads()).")

    paint_t0 = start_phase_timing()
    if cfg.catalog_source == "halfdome"
        run_halfdome_visuals!(cfg, state, y_model_interp)
    else
        itp_z_of_chi = make_z_of_chi_itp(omegam=cfg.cosmo_omegam, h_value=cfg.cosmo_h)
        run_websky_visuals!(cfg, state, y_model_interp, itp_z_of_chi)
    end
    print_phase_usage("Painting", paint_t0)

    output_y_map = state.m_hp
    if cfg.save_healpix_map || cfg.save_cl
        if cfg.apply_gaussian_beam
            println("Applying Gaussian beam to final tSZ map with FWHM=$(cfg.gaussian_beam_fwhm_arcmin) arcmin.")
        end
        output_y_map = prepare_tsz_map_for_output(cfg, state.m_hp)
    end
    if cfg.save_healpix_map
        save_final_maps!(cfg, output_y_map, state.mass_hp)
    end

    if cfg.save_cl
        cl = compute_cl(cfg, output_y_map)
        write_cl_fits_overwrite(cfg.cl_output_path, cl)
    end

    elapsed = time() - t0
    println("Finished $(cfg.simulation_tag) $(cfg.batching_mode) run.")
    println("Total elapsed time: $(round(elapsed; digits=2)) s")
    return state
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_tsz_visual_fits()
end
