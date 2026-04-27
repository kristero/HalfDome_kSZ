function ensure_output_dir(cfg::VisualConfig)
    isdir(cfg.output_dir) || mkpath(cfg.output_dir)
    return cfg.output_dir
end

function save_visual_bin_maps!(cfg::VisualConfig, m_hp, mass_hp, batch_y_path::AbstractString, batch_mass_path::AbstractString)
    isdir(dirname(batch_y_path)) || mkpath(dirname(batch_y_path))
    if cfg.save_mass_map
        isdir(dirname(batch_mass_path)) || mkpath(dirname(batch_mass_path))
    end

    Healpix.saveToFITS(
        m_hp,
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
