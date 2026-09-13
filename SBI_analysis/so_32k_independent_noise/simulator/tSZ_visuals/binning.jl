function redshift_bin_edges(z_max::Real, mode::AbstractString, linear_dz::Real, log_dlog::Real)
    if mode == "linear"
        z_hi = (floor(Float64(z_max) / Float64(linear_dz)) + 1.0) * Float64(linear_dz)
        z_hi = max(z_hi, Float64(linear_dz))
        return collect(0.0:Float64(linear_dz):z_hi)
    end

    log_hi = (floor(log10(1.0 + Float64(z_max)) / Float64(log_dlog)) + 1.0) * Float64(log_dlog)
    log_hi = max(log_hi, Float64(log_dlog))
    log_edges = collect(0.0:Float64(log_dlog):log_hi)
    return 10.0 .^ log_edges .- 1.0
end

function mass_bin_edges(m_min::Real, m_max::Real, dlogm::Real)
    m_min > 0.0 || error("mass binning requires positive minimum mass.")
    m_max >= m_min || error("mass binning requires m_max >= m_min.")

    log_min = floor(log10(Float64(m_min)) / Float64(dlogm)) * Float64(dlogm)
    log_max = (floor(log10(Float64(m_max)) / Float64(dlogm)) + 1.0) * Float64(dlogm)
    log_edges = collect(log_min:Float64(dlogm):log_max)
    return 10.0 .^ log_edges
end

function z_range_tag(z_min::Real, z_max::Real; digits::Int=3)
    z_min_tag = fmt_param_value(round(Float64(z_min); digits=digits))
    z_max_tag = fmt_param_value(round(Float64(z_max); digits=digits))
    return "z_$(z_min_tag)_$(z_max_tag)"
end

function mass_range_tag(m_min::Real, m_max::Real)
    return "m_$(fmt_param_value(round(log10(Float64(m_min)); digits=3)))_$(fmt_param_value(round(log10(Float64(m_max)); digits=3)))"
end

function initial_range_tag(start_index::Integer, stop_index::Integer)
    return "idx_$(start_index)_$(stop_index)"
end

function output_bin_kind(cfg::VisualConfig)
    cfg.batching_mode == "redshift" && return "zbin"
    cfg.batching_mode == "mass" && return "massbin"
    return "chunk"
end

function bin_output_paths(cfg::VisualConfig, bin_number::Int, range_tag::AbstractString)
    bin_kind = output_bin_kind(cfg)
    y_path = joinpath(
        cfg.output_dir,
        "$(cfg.simulation_tag)_tSZ_$(bin_kind)$(bin_number)_nside$(cfg.nside)_$(cfg.run_tag)_$(range_tag).fits"
    )
    mass_path = joinpath(
        cfg.output_dir,
        "$(cfg.simulation_tag)_mass_$(bin_kind)$(bin_number)_nside$(cfg.nside)_$(cfg.run_tag)_$(range_tag).fits"
    )
    return y_path, mass_path
end
