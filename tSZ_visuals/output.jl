function ensure_output_dir(cfg::VisualConfig)
    isdir(cfg.output_dir) || mkpath(cfg.output_dir)
    return cfg.output_dir
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
    Healpix.writeClToFITS(abs_path, cl_array; overwrite=true)

    println("Saved Cl FITS to $(abs_path)")
    return abs_path
end
