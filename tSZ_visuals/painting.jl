function init_visual_maps(cfg::VisualConfig)
    m_hp = HealpixMap{Float64, RingOrder}(cfg.nside)
    fill!(m_hp.pixels, 0.0)

    mass_hp = HealpixMap{Float64, RingOrder}(cfg.nside)
    fill!(mass_hp.pixels, 0.0)

    tmp_hp = HealpixMap{Float64, RingOrder}(cfg.nside)
    fill!(tmp_hp.pixels, 0.0)

    batch_mass_hp = HealpixMap{Float64, RingOrder}(cfg.nside)
    fill!(batch_mass_hp.pixels, 0.0)

    bin_y_hp = HealpixMap{Float64, RingOrder}(cfg.nside)
    fill!(bin_y_hp.pixels, 0.0)

    bin_mass_hp = HealpixMap{Float64, RingOrder}(cfg.nside)
    fill!(bin_mass_hp.pixels, 0.0)

    res = Healpix.Resolution(cfg.nside)
    workspace = XGPaint.HealpixRingProfileWorkspace{Float64}(res)

    return (
        m_hp=m_hp,
        mass_hp=mass_hp,
        tmp_hp=tmp_hp,
        batch_mass_hp=batch_mass_hp,
        bin_y_hp=bin_y_hp,
        bin_mass_hp=bin_mass_hp,
        workspace=workspace
    )
end

function reset_bin_maps!(state)
    fill!(state.bin_y_hp.pixels, 0.0)
    fill!(state.bin_mass_hp.pixels, 0.0)
    return nothing
end

function bin_maps_to_save(cfg::VisualConfig, state)
    if cfg.cumulative_bin_maps
        return state.m_hp, state.mass_hp
    end
    return state.bin_y_hp, state.bin_mass_hp
end

function radius_to_angular_extent(radius_comoving::Real, chi_comoving::Real)
    if !isfinite(radius_comoving) || !isfinite(chi_comoving) || radius_comoving <= 0.0 || chi_comoving <= 0.0
        return 0.0
    end
    return min(Float64(radius_comoving / chi_comoving), pi)
end

function paint_mass_disc!(
    mass_hp::HealpixMap{Float64, RingOrder},
    workspace::XGPaint.HealpixRingProfileWorkspace{Float64},
    alpha::Real,
    delta::Real,
    radius_rad::Real,
    mass_value::Real
)
    if !isfinite(radius_rad) || radius_rad <= 0.0
        theta = pi / 2 - Float64(delta)
        phi = mod(Float64(alpha), 2pi)
        pix = Healpix.ang2pixRing(mass_hp.resolution, theta, phi)
        mass_hp.pixels[pix] += Float64(mass_value)
        return nothing
    end

    center_theta = Float64(pi / 2 - delta)
    center_phi = mod(Float64(alpha), 2pi)
    search_radius = min(Float64(radius_rad), pi)

    ring_start, ring_stop = XGPaint.get_relevant_rings(workspace.res, center_theta, search_radius)
    @inbounds for ring_idx in ring_start:ring_stop
        range1, range2 = XGPaint.get_ring_disc_ranges(workspace, ring_idx, center_theta, center_phi, search_radius)
        first_pixel = workspace.ring_first_pixels[ring_idx]
        for local_pix_idx in range1
            mass_hp.pixels[first_pixel + local_pix_idx - 1] += Float64(mass_value)
        end
        for local_pix_idx in range2
            mass_hp.pixels[first_pixel + local_pix_idx - 1] += Float64(mass_value)
        end
    end

    return nothing
end

function build_halo_mass_map!(
    mass_hp::HealpixMap{Float64, RingOrder},
    workspace::XGPaint.HealpixRingProfileWorkspace{Float64},
    ras::AbstractVector{<:Real},
    decs::AbstractVector{<:Real},
    masses::AbstractVector{<:Real},
    angular_radii::AbstractVector{<:Real}
)
    length(ras) == length(decs) == length(masses) == length(angular_radii) || error("Mass-map inputs must have the same length.")

    fill!(mass_hp.pixels, 0.0)
    @inbounds for i in eachindex(masses)
        paint_mass_disc!(
            mass_hp,
            workspace,
            ras[i],
            decs[i],
            angular_radii[i],
            masses[i]
        )
    end

    return nothing
end

function paint_visual_batch!(
    state,
    y_model_interp,
    x_batch,
    y_batch,
    z_batch,
    radius_batch::Vector{Float64},
    mass_batch::Vector{Float64},
    redshift_batch::Vector{Float64}
)
    isempty(redshift_batch) && return 0

    xs = Float64.(x_batch)
    ys = Float64.(y_batch)
    zs = Float64.(z_batch)
    chis = sqrt.(xs .^ 2 .+ ys .^ 2 .+ zs .^ 2)

    ra, dec = xyz_to_ra_dec_threaded(xs, ys, zs)
    perm = sortperm(dec)

    ra = ra[perm]
    dec = dec[perm]
    masses = mass_batch[perm]
    redshifts = redshift_batch[perm]
    angular_radii = radius_to_angular_extent.(radius_batch[perm], chis[perm])

    fill!(state.tmp_hp.pixels, 0.0)
    paint!(state.tmp_hp, state.workspace, y_model_interp, masses, redshifts, ra, dec)
    build_halo_mass_map!(state.batch_mass_hp, state.workspace, ra, dec, masses, angular_radii)

    state.m_hp.pixels .+= state.tmp_hp.pixels
    state.mass_hp.pixels .+= state.batch_mass_hp.pixels
    state.bin_y_hp.pixels .+= state.tmp_hp.pixels
    state.bin_mass_hp.pixels .+= state.batch_mass_hp.pixels
    return length(redshifts)
end
