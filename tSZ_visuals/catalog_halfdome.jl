function read_hdf5_columns(ds, rows, idx_batch::Vector{Int})
    isempty(idx_batch) && return Matrix{Float64}(undef, length(rows), 0)

    read_order = sortperm(idx_batch)
    idx_sorted = idx_batch[read_order]
    data_sorted = Matrix{Float64}(undef, length(rows), length(idx_sorted))

    block_start = 1
    while block_start <= length(idx_sorted)
        block_end = block_start
        while block_end < length(idx_sorted) && idx_sorted[block_end + 1] == idx_sorted[block_end] + 1
            block_end += 1
        end

        dataset_range = idx_sorted[block_start]:idx_sorted[block_end]
        data_sorted[:, block_start:block_end] .= Float64.(ds[rows, dataset_range])
        block_start = block_end + 1
    end

    return data_sorted[:, invperm(read_order)]
end

function halfdome_radius_from_rdisp(rdisp_spatial::AbstractMatrix{<:Real})
    size(rdisp_spatial, 1) == 3 || error("Expected 3 spatial Rdisp components per halo.")
    return vec(sqrt.(sum(abs2, rdisp_spatial; dims=1)))
end

function halfdome_selected_order(halo_mass::Vector{Float64}, redshift::Vector{Float64}, cfg::VisualConfig)
    keep = isfinite.(halo_mass) .& isfinite.(redshift) .& (redshift .>= 0.0)
    if cfg.apply_mass_cut
        keep .&= halo_mass .>= cfg.mass_min
    end

    selected_idx = findall(keep)
    isempty(selected_idx) && return Int[], Float64[], Float64[]

    selected_mass = halo_mass[selected_idx]
    selected_redshift = redshift[selected_idx]

    if cfg.batching_mode == "mass"
        order = sortperm(selected_mass)
    else
        order = sortperm(selected_redshift; rev=true)
    end

    return selected_idx[order], selected_mass[order], selected_redshift[order]
end

function paint_halfdome_indices!(
    h5,
    cfg::VisualConfig,
    state,
    y_model_interp,
    idx_batch::Vector{Int},
    mass_batch::Vector{Float64},
    redshift_batch::Vector{Float64}
)
    pos = read_hdf5_columns(h5["Position"], 1:3, idx_batch)
    rdisp_spatial = read_hdf5_columns(h5["Rdisp"], 1:3, idx_batch)
    radius_batch = halfdome_radius_from_rdisp(rdisp_spatial)

    return paint_visual_batch!(
        state,
        y_model_interp,
        view(pos, 1, :),
        view(pos, 2, :),
        view(pos, 3, :),
        radius_batch,
        mass_batch,
        redshift_batch
    )
end

function run_halfdome_redshift_bins!(cfg::VisualConfig, state, y_model_interp, h5, sorted_idx, sorted_mass, sorted_redshift)
    selected_z_max = maximum(sorted_redshift)
    selected_z_min = minimum(sorted_redshift)
    println("Selected HalfDome redshift range: [$(round(selected_z_min; digits=4)), $(round(selected_z_max; digits=4))].")

    edges = redshift_bin_edges(selected_z_max, cfg.redshift_binning_mode, cfg.redshift_bin_width, cfg.log_redshift_bin_width)
    nbins = length(edges) - 1
    zbin_ref = Ref(0)

    for bin_idx in nbins:-1:1
        z_min_bin = edges[bin_idx]
        z_max_bin = edges[bin_idx + 1]
        in_bin = findall(z -> z >= z_min_bin && z < z_max_bin, sorted_redshift)
        isempty(in_bin) && continue

        zbin_ref[] += 1
        zbin_number = zbin_ref[]
        reset_bin_maps!(state)
        idx_batch = sorted_idx[in_bin]
        mass_batch = sorted_mass[in_bin]
        redshift_batch = sorted_redshift[in_bin]

        painted_count = paint_halfdome_indices!(h5, cfg, state, y_model_interp, idx_batch, mass_batch, redshift_batch)
        actual_z_min = minimum(redshift_batch)
        actual_z_max = maximum(redshift_batch)
        println("Painted HalfDome redshift bin $(zbin_number) from bin $(bin_idx)/$(nbins) with $(painted_count) halos; z in [$(round(actual_z_min; digits=4)), $(round(actual_z_max; digits=4))].")

        if cfg.save_healpix_map && cfg.save_bin_maps
            y_path, mass_path = bin_output_paths(cfg, zbin_number, z_range_tag(actual_z_min, actual_z_max))
            y_map_to_save, mass_map_to_save = bin_maps_to_save(cfg, state)
            save_visual_bin_maps!(cfg, y_map_to_save, mass_map_to_save, y_path, mass_path)
        end
    end

    zbin_ref[] > 0 || error("HalfDome redshift binning did not paint any selected halos.")
end

function run_halfdome_mass_bins!(cfg::VisualConfig, state, y_model_interp, h5, sorted_idx, sorted_mass, sorted_redshift)
    selected_m_min = minimum(sorted_mass)
    selected_m_max = maximum(sorted_mass)
    println("Selected HalfDome mass range: [$(selected_m_min), $(selected_m_max)].")

    edges = mass_bin_edges(selected_m_min, selected_m_max, cfg.mass_bin_width_dex)
    nbins = length(edges) - 1
    mbin_ref = Ref(0)
    println("Painting HalfDome with $(nbins) mass bins using $(cfg.binning_tag), from low mass to high mass.")

    for bin_idx in 1:nbins
        m_min_bin = edges[bin_idx]
        m_max_bin = edges[bin_idx + 1]
        in_bin = findall(m -> m >= m_min_bin && m < m_max_bin, sorted_mass)
        isempty(in_bin) && continue

        mbin_ref[] += 1
        mbin_number = mbin_ref[]
        reset_bin_maps!(state)
        idx_batch = sorted_idx[in_bin]
        mass_batch = sorted_mass[in_bin]
        redshift_batch = sorted_redshift[in_bin]

        painted_count = paint_halfdome_indices!(h5, cfg, state, y_model_interp, idx_batch, mass_batch, redshift_batch)
        actual_m_min = minimum(mass_batch)
        actual_m_max = maximum(mass_batch)
        println("Painted HalfDome mass bin $(mbin_number) from bin $(bin_idx)/$(nbins) with $(painted_count) halos; log10(M) in [$(round(log10(actual_m_min); digits=3)), $(round(log10(actual_m_max); digits=3))].")

        if cfg.save_healpix_map && cfg.save_bin_maps
            y_path, mass_path = bin_output_paths(cfg, mbin_number, mass_range_tag(actual_m_min, actual_m_max))
            y_map_to_save, mass_map_to_save = bin_maps_to_save(cfg, state)
            save_visual_bin_maps!(cfg, y_map_to_save, mass_map_to_save, y_path, mass_path)
        end
    end

    mbin_ref[] > 0 || error("HalfDome mass binning did not paint any selected halos.")
end

function run_halfdome_initial_chunks!(cfg::VisualConfig, state, y_model_interp, h5)
    pos_ds = h5["Position"]
    mass_ds = h5["halo_mass_m200c"]
    redshift_ds = h5["redshift"]
    total_halo_count = size(pos_ds, 2)
    chunk_ref = Ref(0)

    for chunk_start in 1:cfg.chunkN:total_halo_count
        chunk_stop = min(chunk_start + cfg.chunkN - 1, total_halo_count)
        idx_range = chunk_start:chunk_stop
        mass = Float64.(mass_ds[idx_range]) ./ TSZ_H_VALUE
        redshift = Float64.(redshift_ds[idx_range])
        keep = isfinite.(mass) .& isfinite.(redshift) .& (redshift .>= 0.0)
        if cfg.apply_mass_cut
            keep .&= mass .>= cfg.mass_min
        end
        any(keep) || continue

        local_idx = findall(keep)
        reset_bin_maps!(state)
        idx_batch = collect(idx_range)[local_idx]
        mass_batch = mass[local_idx]
        redshift_batch = redshift[local_idx]
        painted_count = paint_halfdome_indices!(h5, cfg, state, y_model_interp, idx_batch, mass_batch, redshift_batch)

        chunk_ref[] += 1
        chunk_number = chunk_ref[]
        println("Painted HalfDome initial chunk $(chunk_number) for catalog indices $(chunk_start):$(chunk_stop) with $(painted_count) halos.")

        if cfg.save_healpix_map && cfg.save_bin_maps
            y_path, mass_path = bin_output_paths(cfg, chunk_number, initial_range_tag(chunk_start, chunk_stop))
            y_map_to_save, mass_map_to_save = bin_maps_to_save(cfg, state)
            save_visual_bin_maps!(cfg, y_map_to_save, mass_map_to_save, y_path, mass_path)
        end
    end

    chunk_ref[] > 0 || error("HalfDome initial chunking did not paint any selected halos.")
end

function run_halfdome_full_map!(cfg::VisualConfig, state, y_model_interp, h5)
    pos_ds = h5["Position"]
    mass_ds = h5["halo_mass_m200c"]
    redshift_ds = h5["redshift"]
    total_halo_count = size(pos_ds, 2)
    total_painted = 0
    chunk_counter = 0

    for chunk_start in 1:cfg.chunkN:total_halo_count
        chunk_stop = min(chunk_start + cfg.chunkN - 1, total_halo_count)
        idx_range = chunk_start:chunk_stop
        mass = Float64.(mass_ds[idx_range]) ./ TSZ_H_VALUE
        redshift = Float64.(redshift_ds[idx_range])
        keep = isfinite.(mass) .& isfinite.(redshift) .& (redshift .>= 0.0)
        if cfg.apply_mass_cut
            keep .&= mass .>= cfg.mass_min
        end
        any(keep) || continue

        local_idx = findall(keep)
        idx_batch = collect(idx_range)[local_idx]
        mass_batch = mass[local_idx]
        redshift_batch = redshift[local_idx]
        painted_count = paint_halfdome_indices!(h5, cfg, state, y_model_interp, idx_batch, mass_batch, redshift_batch)

        chunk_counter += 1
        total_painted += painted_count
        println("Accumulated HalfDome full-map chunk $(chunk_counter) for catalog indices $(chunk_start):$(chunk_stop) with $(painted_count) halos.")
    end

    total_painted > 0 || error("HalfDome full-map mode did not paint any selected halos.")
    println("Finished accumulating HalfDome full map with $(total_painted) halos.")
end

function run_halfdome_visuals!(cfg::VisualConfig, state, y_model_interp)
    h5open(cfg.halfdome_path, "r") do h5
        total_halo_count = size(h5["Position"], 2)
        @show total_halo_count

        if cfg.batching_mode == "full"
            return run_halfdome_full_map!(cfg, state, y_model_interp, h5)
        end

        if cfg.batching_mode == "initial"
            return run_halfdome_initial_chunks!(cfg, state, y_model_interp, h5)
        end

        halo_mass = Float64.(read(h5["halo_mass_m200c"])) ./ TSZ_H_VALUE
        redshift = Float64.(read(h5["redshift"]))
        sorted_idx, sorted_mass, sorted_redshift = halfdome_selected_order(halo_mass, redshift, cfg)

        selected_halo_count = length(sorted_idx)
        println("Selected halos after cuts: $(selected_halo_count)")
        selected_halo_count > 0 || error("No HalfDome halos passed the current selection.")

        if cfg.batching_mode == "mass"
            run_halfdome_mass_bins!(cfg, state, y_model_interp, h5, sorted_idx, sorted_mass, sorted_redshift)
        else
            run_halfdome_redshift_bins!(cfg, state, y_model_interp, h5, sorted_idx, sorted_mass, sorted_redshift)
        end
    end
end
