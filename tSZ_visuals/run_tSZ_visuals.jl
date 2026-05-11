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

function run_tsz_visual_fits()
    t0 = time()
    cfg = try
        load_visual_config()
    catch err
        if err isa SkipVisualRun
            println("Skipping tSZ visual run: $(err.message)")
            return nothing
        end
        rethrow()
    end

    ensure_output_dir(cfg)
    print_visual_config(cfg)
    print_runtime_environment()

    y_model_interp = build_visual_interpolator(cfg)
    state = init_visual_maps(cfg)

    println("Initiating HealPix with NSide: $(cfg.nside)")
    println("Julia threads available: $(nthreads())")
    map_label = cfg.save_mass_map ? "y and mass FITS maps" : "y FITS maps"
    println("Creating $(cfg.bin_map_mode_tag) $(map_label) with $(cfg.batching_mode) batching.")

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
        cl = anafast(output_y_map, niter=0)
        write_cl_fits_overwrite(cfg.cl_output_path, cl)
    end

    elapsed = time() - t0
    println("Finished $(cfg.simulation_tag) Healpix tSZ visual FITS creation ($(cfg.batching_mode)).")
    println("Elapsed time: $(round(elapsed; digits=2)) s")
    return state
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_tsz_visual_fits()
end
