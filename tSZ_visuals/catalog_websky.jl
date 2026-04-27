function stream_websky_chunks(process_chunk!::F, cfg::VisualConfig, itp_z_of_chi) where {F}
    return open(cfg.websky_path, "r") do io
        total_halo_count = Int(read(io, Int32))
        RTHmax = read(io, Float32)
        redshiftbox = read(io, Float32)
        @show total_halo_count RTHmax redshiftbox

        buf = Matrix{Float32}(undef, 10, cfg.chunkN)
        nleft = total_halo_count
        chunk_start = 1

        while nleft > 0
            nthis = min(cfg.chunkN, nleft)
            chunk_stop = chunk_start + nthis - 1

            rawview = @view reinterpret(Float32, vec(buf))[1:10 * nthis]
            read!(io, rawview)

            cat = @view buf[:, 1:nthis]
            x = @view cat[1, :]
            y = @view cat[2, :]
            z = @view cat[3, :]
            radius = @view cat[7, :]
            redshift, halo_mass = compute_redshift_and_mass(x, y, z, radius, itp_z_of_chi, TSZ_RHO_M)

            process_chunk!(chunk_start, chunk_stop, x, y, z, radius, halo_mass, redshift)

            chunk_start += nthis
            nleft -= nthis
        end

        total_halo_count
    end
end

function selected_mask(cfg::VisualConfig, halo_mass, redshift)
    keep = isfinite.(redshift) .& isfinite.(halo_mass) .& (redshift .>= 0.0)
    if cfg.apply_mass_cut
        keep .&= halo_mass .>= cfg.mass_min
    end
    return keep
end

function summarize_websky_selection(cfg::VisualConfig, itp_z_of_chi)
    selected_count = Ref(0)
    z_min_ref = Ref(Inf)
    z_max_ref = Ref(-Inf)
    m_min_ref = Ref(Inf)
    m_max_ref = Ref(-Inf)

    total_halo_count = stream_websky_chunks(cfg, itp_z_of_chi) do _chunk_start, _chunk_stop, _x, _y, _z, _radius, halo_mass, redshift
        keep = selected_mask(cfg, halo_mass, redshift)
        if any(keep)
            kept_z = redshift[keep]
            kept_m = halo_mass[keep]
            selected_count[] += length(kept_z)
            z_min_ref[] = min(z_min_ref[], minimum(kept_z))
            z_max_ref[] = max(z_max_ref[], maximum(kept_z))
            m_min_ref[] = min(m_min_ref[], minimum(kept_m))
            m_max_ref[] = max(m_max_ref[], maximum(kept_m))
        end
    end

    return (
        total_halo_count=total_halo_count,
        selected_count=selected_count[],
        z_min=z_min_ref[],
        z_max=z_max_ref[],
        m_min=m_min_ref[],
        m_max=m_max_ref[]
    )
end

function paint_websky_filtered!(
    cfg::VisualConfig,
    state,
    y_model_interp,
    itp_z_of_chi,
    filter_fn::F
) where {F}
    painted_count = Ref(0)
    actual_z_min = Ref(Inf)
    actual_z_max = Ref(-Inf)
    actual_m_min = Ref(Inf)
    actual_m_max = Ref(-Inf)

    stream_websky_chunks(cfg, itp_z_of_chi) do _chunk_start, _chunk_stop, x, y, z, radius, halo_mass, redshift
        keep = selected_mask(cfg, halo_mass, redshift) .& filter_fn(halo_mass, redshift)
        if any(keep)
            x_batch = Float64.(x[keep])
            y_batch = Float64.(y[keep])
            z_batch = Float64.(z[keep])
            radius_batch = Float64.(radius[keep])
            mass_batch = halo_mass[keep]
            redshift_batch = redshift[keep]

            paint_visual_batch!(
                state,
                y_model_interp,
                x_batch,
                y_batch,
                z_batch,
                radius_batch,
                mass_batch,
                redshift_batch
            )

            painted_count[] += length(redshift_batch)
            actual_z_min[] = min(actual_z_min[], minimum(redshift_batch))
            actual_z_max[] = max(actual_z_max[], maximum(redshift_batch))
            actual_m_min[] = min(actual_m_min[], minimum(mass_batch))
            actual_m_max[] = max(actual_m_max[], maximum(mass_batch))
        end
    end

    return (
        painted_count=painted_count[],
        z_min=actual_z_min[],
        z_max=actual_z_max[],
        m_min=actual_m_min[],
        m_max=actual_m_max[]
    )
end

function run_websky_redshift_bins!(cfg::VisualConfig, state, y_model_interp, itp_z_of_chi, summary)
    edges = redshift_bin_edges(summary.z_max, cfg.redshift_binning_mode, cfg.redshift_bin_width, cfg.log_redshift_bin_width)
    nbins = length(edges) - 1
    println("Painting WebSky with $(nbins) redshift bins using $(cfg.binning_tag), streamed from z_max to z=0.")

    bin_ref = Ref(0)
    for bin_idx in nbins:-1:1
        z_min_bin = edges[bin_idx]
        z_max_bin = edges[bin_idx + 1]
        reset_bin_maps!(state)
        stats = paint_websky_filtered!(cfg, state, y_model_interp, itp_z_of_chi) do _mass, redshift
            (redshift .>= z_min_bin) .& (redshift .< z_max_bin)
        end

        stats.painted_count == 0 && continue
        bin_ref[] += 1
        bin_number = bin_ref[]
        println("Painted WebSky redshift bin $(bin_number) from bin $(bin_idx)/$(nbins) with $(stats.painted_count) halos; z in [$(round(stats.z_min; digits=4)), $(round(stats.z_max; digits=4))].")

        if cfg.save_healpix_map && cfg.save_bin_maps
            y_path, mass_path = bin_output_paths(cfg, bin_number, z_range_tag(stats.z_min, stats.z_max))
            y_map_to_save, mass_map_to_save = bin_maps_to_save(cfg, state)
            save_visual_bin_maps!(cfg, y_map_to_save, mass_map_to_save, y_path, mass_path)
        end
    end

    bin_ref[] > 0 || error("WebSky redshift bin streaming did not paint any selected halos.")
end

function run_websky_mass_bins!(cfg::VisualConfig, state, y_model_interp, itp_z_of_chi, summary)
    edges = mass_bin_edges(summary.m_min, summary.m_max, cfg.mass_bin_width_dex)
    nbins = length(edges) - 1
    println("Painting WebSky with $(nbins) mass bins using $(cfg.binning_tag), streamed from low mass to high mass.")

    bin_ref = Ref(0)
    for bin_idx in 1:nbins
        m_min_bin = edges[bin_idx]
        m_max_bin = edges[bin_idx + 1]
        reset_bin_maps!(state)
        stats = paint_websky_filtered!(cfg, state, y_model_interp, itp_z_of_chi) do halo_mass, _redshift
            (halo_mass .>= m_min_bin) .& (halo_mass .< m_max_bin)
        end

        stats.painted_count == 0 && continue
        bin_ref[] += 1
        bin_number = bin_ref[]
        println("Painted WebSky mass bin $(bin_number) from bin $(bin_idx)/$(nbins) with $(stats.painted_count) halos; log10(M) in [$(round(log10(stats.m_min); digits=3)), $(round(log10(stats.m_max); digits=3))].")

        if cfg.save_healpix_map && cfg.save_bin_maps
            y_path, mass_path = bin_output_paths(cfg, bin_number, mass_range_tag(stats.m_min, stats.m_max))
            y_map_to_save, mass_map_to_save = bin_maps_to_save(cfg, state)
            save_visual_bin_maps!(cfg, y_map_to_save, mass_map_to_save, y_path, mass_path)
        end
    end

    bin_ref[] > 0 || error("WebSky mass bin streaming did not paint any selected halos.")
end

function run_websky_initial_chunks!(cfg::VisualConfig, state, y_model_interp, itp_z_of_chi)
    chunk_ref = Ref(0)

    stream_websky_chunks(cfg, itp_z_of_chi) do chunk_start, chunk_stop, x, y, z, radius, halo_mass, redshift
        keep = selected_mask(cfg, halo_mass, redshift)
        if any(keep)
            reset_bin_maps!(state)
            painted_count = paint_visual_batch!(
                state,
                y_model_interp,
                Float64.(x[keep]),
                Float64.(y[keep]),
                Float64.(z[keep]),
                Float64.(radius[keep]),
                halo_mass[keep],
                redshift[keep]
            )

            chunk_ref[] += 1
            chunk_number = chunk_ref[]
            println("Painted WebSky initial chunk $(chunk_number) for catalog indices $(chunk_start):$(chunk_stop) with $(painted_count) halos.")

            if cfg.save_healpix_map && cfg.save_bin_maps
                y_path, mass_path = bin_output_paths(cfg, chunk_number, initial_range_tag(chunk_start, chunk_stop))
                y_map_to_save, mass_map_to_save = bin_maps_to_save(cfg, state)
                save_visual_bin_maps!(cfg, y_map_to_save, mass_map_to_save, y_path, mass_path)
            end
        end
    end

    chunk_ref[] > 0 || error("WebSky initial chunking did not paint any selected halos.")
end

function run_websky_full_map!(cfg::VisualConfig, state, y_model_interp, itp_z_of_chi)
    chunk_counter = Ref(0)
    total_painted = Ref(0)

    stream_websky_chunks(cfg, itp_z_of_chi) do chunk_start, chunk_stop, x, y, z, radius, halo_mass, redshift
        keep = selected_mask(cfg, halo_mass, redshift)
        if any(keep)
            painted_count = paint_visual_batch!(
                state,
                y_model_interp,
                Float64.(x[keep]),
                Float64.(y[keep]),
                Float64.(z[keep]),
                Float64.(radius[keep]),
                halo_mass[keep],
                redshift[keep]
            )

            chunk_counter[] += 1
            total_painted[] += painted_count
            println("Accumulated WebSky full-map chunk $(chunk_counter[]) for catalog indices $(chunk_start):$(chunk_stop) with $(painted_count) halos.")
        end
    end

    total_painted[] > 0 || error("WebSky full-map mode did not paint any selected halos.")
    println("Finished accumulating WebSky full map with $(total_painted[]) halos.")
end

function run_websky_visuals!(cfg::VisualConfig, state, y_model_interp, itp_z_of_chi)
    if cfg.batching_mode == "full"
        return run_websky_full_map!(cfg, state, y_model_interp, itp_z_of_chi)
    end

    if cfg.batching_mode == "initial"
        return run_websky_initial_chunks!(cfg, state, y_model_interp, itp_z_of_chi)
    end

    summary = summarize_websky_selection(cfg, itp_z_of_chi)
    println("Total halos in WebSky catalog: $(summary.total_halo_count)")
    println("Selected halos after cuts: $(summary.selected_count)")
    summary.selected_count > 0 || error("No WebSky halos passed the current selection.")

    if cfg.batching_mode == "mass"
        println("Selected WebSky mass range: [$(summary.m_min), $(summary.m_max)].")
        run_websky_mass_bins!(cfg, state, y_model_interp, itp_z_of_chi, summary)
    else
        println("Selected WebSky redshift range: [$(round(summary.z_min; digits=4)), $(round(summary.z_max; digits=4))].")
        run_websky_redshift_bins!(cfg, state, y_model_interp, itp_z_of_chi, summary)
    end
end
