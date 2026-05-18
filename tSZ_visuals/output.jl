function ensure_output_dir(cfg::VisualConfig)
    isdir(cfg.output_dir) || mkpath(cfg.output_dir)
    return cfg.output_dir
end

function matching_output_with_optional_instance(output_dir::AbstractString, basename_prefix::AbstractString, basename_suffix::AbstractString)
    isdir(output_dir) || return nothing

    for entry in sort(readdir(output_dir; join=false))
        startswith(entry, basename_prefix) || continue
        endswith(entry, basename_suffix) || continue

        middle_start = lastindex(basename_prefix) + 1
        middle_stop = lastindex(entry) - lastindex(basename_suffix)
        middle = middle_start > middle_stop ? "" : entry[middle_start:middle_stop]
        if isempty(middle) || startswith(middle, "__")
            return joinpath(output_dir, entry)
        end
    end

    return nothing
end

function output_completion_specs(cfg::VisualConfig)
    specs = NamedTuple[]

    if cfg.save_healpix_map
        push!(specs, (
            label="final tSZ map",
            exact_path=cfg.fits_output_path,
            basename_prefixes=["$(cfg.simulation_tag)_tSZ_nside$(cfg.nside)_$(cfg.run_tag)"],
            basename_suffix="_m200c.fits"
        ))
    end

    if cfg.save_healpix_map && cfg.save_mass_map
        push!(specs, (
            label="final mass map",
            exact_path=cfg.mass_fits_output_path,
            basename_prefixes=["$(cfg.simulation_tag)_mass_nside$(cfg.nside)_$(cfg.run_tag)"],
            basename_suffix="_m200c.fits"
        ))
    end

    if cfg.save_cl
        cl_lmax_tag = build_cl_lmax_tag(cfg.cl_lmax)
        cl_tag_base = "$(cfg.beam_tag)_$(cfg.binning_tag)_$(cfg.bin_map_mode_tag)_$(cl_lmax_tag)"
        legacy_cl_tag_base = "$(cfg.beam_tag)_$(cfg.binning_tag)_$(cfg.bin_map_mode_tag)"
        push!(specs, (
            label="C_l FITS",
            exact_path=cfg.cl_output_path,
            basename_prefixes=unique([
                "$(cfg.simulation_tag)_tSZ_cl_m200c_$(cfg.param_tag)_$(cfg.cosmology_tag)_nside$(cfg.nside)_$(cl_tag_base)",
                "$(cfg.simulation_tag)_tSZ_cl_m200c_$(cfg.param_tag)_$(cfg.cosmology_tag)_nside$(cfg.nside)_$(legacy_cl_tag_base)"
            ]),
            basename_suffix=".fits"
        ))
    end

    return specs
end

function existing_completed_outputs(cfg::VisualConfig; allow_any_run_instance::Bool=false)
    found = String[]

    for spec in output_completion_specs(cfg)
        if isfile(spec.exact_path)
            push!(found, spec.exact_path)
        elseif allow_any_run_instance
            match_path = nothing
            for basename_prefix in spec.basename_prefixes
                match_path = matching_output_with_optional_instance(
                    cfg.output_dir,
                    basename_prefix,
                    spec.basename_suffix
                )
                match_path === nothing || break
            end
            match_path === nothing || push!(found, match_path)
        end
    end

    return found
end

function visual_outputs_complete(cfg::VisualConfig; allow_any_run_instance::Bool=false)
    specs = output_completion_specs(cfg)
    isempty(specs) && return false

    found = existing_completed_outputs(cfg; allow_any_run_instance=allow_any_run_instance)
    return length(found) == length(specs)
end

function duplicate_healpix_map(m::HealpixMap{<:Real, RingOrder})
    m_copy = HealpixMap{Float64, RingOrder}(m.resolution.nside)
    m_copy.pixels .= Float64.(m.pixels)
    return m_copy
end

function smooth_healpix_map_gaussian!(m::HealpixMap{<:Real, RingOrder}, fwhm_arcmin::Real; niter::Integer=0)
    fwhm_arcmin > 0 || error("fwhm_arcmin must be positive.")
    any(!iszero, m.pixels) || return m

    fwhm_rad = deg2rad(Float64(fwhm_arcmin) / 60.0)
    alm = Healpix.map2alm(m; niter=niter)
    beam = Healpix.gaussbeam(fwhm_rad, alm.lmax)
    Healpix.almxfl!(alm, beam)
    smoothed_map = Healpix.alm2map(alm, m.resolution.nside)
    m.pixels .= smoothed_map.pixels
    return m
end

function prepare_tsz_map_for_output(cfg::VisualConfig, m_hp; niter::Integer=0)
    if !cfg.apply_gaussian_beam
        return m_hp
    end

    m_hp_to_save = duplicate_healpix_map(m_hp)
    smooth_healpix_map_gaussian!(m_hp_to_save, cfg.gaussian_beam_fwhm_arcmin; niter=niter)
    return m_hp_to_save
end

function save_visual_bin_maps!(cfg::VisualConfig, m_hp, mass_hp, batch_y_path::AbstractString, batch_mass_path::AbstractString)
    isdir(dirname(batch_y_path)) || mkpath(dirname(batch_y_path))
    if cfg.save_mass_map
        isdir(dirname(batch_mass_path)) || mkpath(dirname(batch_mass_path))
    end

    m_hp_to_save = prepare_tsz_map_for_output(cfg, m_hp)
    Healpix.saveToFITS(
        m_hp_to_save,
        "!" * batch_y_path,
        typechar="D"
    )
    println("Saved FITS map to $(abspath(batch_y_path))")
    if cfg.save_mass_map
        Healpix.saveToFITS(
            mass_hp,
            "!" * batch_mass_path,
            typechar="D"
        )
        println("Saved FITS map to $(abspath(batch_mass_path))")
    end
    return nothing
end

function save_final_maps!(cfg::VisualConfig, m_hp, mass_hp)
    ensure_output_dir(cfg)
    Healpix.saveToFITS(
        m_hp,
        "!" * cfg.fits_output_path,
        typechar="D"
    )
    println("Saved final tSZ map to $(abspath(cfg.fits_output_path))")
    if cfg.save_mass_map
        Healpix.saveToFITS(
            mass_hp,
            "!" * cfg.mass_fits_output_path,
            typechar="D"
        )
        println("Saved final mass map to $(abspath(cfg.mass_fits_output_path))")
    end
    return nothing
end

function write_cl_fits_overwrite(path::AbstractString, cl_values)
    abs_path = abspath(path)
    parent_dir = dirname(abs_path)
    isdir(parent_dir) || mkpath(parent_dir)

    cl_array = Float64.(collect(cl_values))
    isfile(abs_path) && rm(abs_path; force=true)
    Healpix.writeClToFITS(abs_path, cl_array; overwrite=true)

    println("Saved Cl FITS to $(abs_path)")
    return abs_path
end

function compute_cl(cfg::VisualConfig, m_hp)
    if cfg.cl_lmax < 0
        println("Computing C_l with Healpix default lmax=$(healpix_default_lmax(cfg.nside)), niter=$(cfg.cl_niter).")
        println("This mode is memory-heavy at high NSIDE; set cl_lmax to cap memory if needed.")
        return Healpix.anafast(m_hp; niter=cfg.cl_niter)
    end

    println("Computing C_l with lmax=$(cfg.cl_lmax), niter=$(cfg.cl_niter).")
    return Healpix.anafast(m_hp; lmax=cfg.cl_lmax, niter=cfg.cl_niter)
end
