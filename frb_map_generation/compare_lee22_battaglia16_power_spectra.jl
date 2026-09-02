#!/usr/bin/env julia

# Compare full-sky halo-DM angular power spectra for matched Lee22 and
# Battaglia16 HalfDome maps. Input maps must differ only in electron-density
# profile; their halo catalogue, M200c masses, 3R200c aperture, source plane,
# NSIDE, and foreground redshift interval must be identical.

using Dates
using HDF5
using Healpix
using Plots
using Statistics

function parse_options(args)
    options = Dict{String,String}()
    i = 1
    while i <= length(args)
        argument = String(args[i])
        startswith(argument, "--") || error("Unexpected positional argument: $(argument)")
        body = argument[3:end]
        if occursin('=', body)
            key, value = split(body, "="; limit=2)
        elseif i < length(args) && !startswith(String(args[i + 1]), "--")
            key, value = body, String(args[i + 1])
            i += 1
        else
            key, value = body, "true"
        end
        options[replace(lowercase(key), "-" => "_")] = value
        i += 1
    end
    return options
end

option(options, key, default) =
    get(options, replace(lowercase(String(key)), "-" => "_"), string(default))
int_option(options, key, default) = parse(Int, option(options, key, default))
float_option(options, key, default) = parse(Float64, option(options, key, default))

function bool_option(options, key, default)
    value = lowercase(strip(option(options, key, default)))
    value in ("1", "true", "yes", "on") && return true
    value in ("0", "false", "no", "off") && return false
    error("Cannot parse --$(key)=$(repr(value)) as a Boolean.")
end

function validate_and_center_map!(map, label)
    pixels = map.pixels
    isempty(pixels) && error("$(label) map has no pixels.")
    all(isfinite, pixels) || error("$(label) map contains NaN or Inf.")
    # Halo-DM maps are non-negative before monopole removal. This also rejects
    # finite HEALPix UNSEEN sentinels and prevents treating a cut sky as full sky.
    minimum(pixels) >= 0.0 || error(
        "$(label) is not a complete non-negative halo-DM map. " *
        "Cut-sky maps require a pseudo-C_ell mode-coupling correction.",
    )
    map_mean = mean(pixels)
    pixels .-= map_mean
    return map_mean
end

function write_full_spectrum_csv(path, ell, cl_lee22, cl_battaglia16, cl_cross)
    open(path, "w") do io
        println(
            io,
            "ell,C_ell_lee22,C_ell_battaglia16,C_ell_cross," *
            "D_ell_lee22,D_ell_battaglia16,D_ell_cross," *
            "lee22_minus_battaglia16_percent,cross_correlation_coefficient",
        )
        for index in eachindex(ell)
            l = ell[index]
            prefactor = l * (l + 1.0) / (2pi)
            lee = cl_lee22[index]
            battaglia = cl_battaglia16[index]
            cross = cl_cross[index]
            percent = battaglia == 0.0 ? NaN : 100.0 * (lee - battaglia) / battaglia
            denominator = sqrt(max(lee * battaglia, 0.0))
            correlation = denominator > 0.0 ? cross / denominator : NaN
            println(
                io,
                "$(l),$(lee),$(battaglia),$(cross)," *
                "$(prefactor * lee),$(prefactor * battaglia),$(prefactor * cross)," *
                "$(percent),$(correlation)",
            )
        end
    end
end

function logarithmic_bandpowers(
    ell,
    cl_lee22,
    cl_battaglia16,
    cl_cross;
    lmin::Int,
    number_of_bins::Int,
)
    lmax = last(ell)
    2 <= lmin < lmax || error("plot_lmin must be in [2, $(lmax - 1)].")
    number_of_bins > 1 || error("plot_bins must exceed one.")
    edges = unique(round.(
        Int,
        exp.(range(log(Float64(lmin)), log(Float64(lmax + 1)); length=number_of_bins + 1)),
    ))
    first(edges) > lmin && pushfirst!(edges, lmin)
    last(edges) <= lmax && push!(edges, lmax + 1)

    band_ell = Float64[]
    band_cl_lee22 = Float64[]
    band_cl_battaglia16 = Float64[]
    band_cl_cross = Float64[]
    for (lower, upper) in zip(edges[1:end-1], edges[2:end])
        indices = findall(l -> lower <= l < upper, ell)
        isempty(indices) && continue
        weights = 2.0 .* Float64.(ell[indices]) .+ 1.0
        push!(band_ell, sum(weights .* ell[indices]) / sum(weights))
        push!(band_cl_lee22, sum(weights .* cl_lee22[indices]) / sum(weights))
        push!(
            band_cl_battaglia16,
            sum(weights .* cl_battaglia16[indices]) / sum(weights),
        )
        push!(band_cl_cross, sum(weights .* cl_cross[indices]) / sum(weights))
    end
    return (
        ell=band_ell,
        cl_lee22=band_cl_lee22,
        cl_battaglia16=band_cl_battaglia16,
        cl_cross=band_cl_cross,
    )
end

function bandpower_columns(bands)
    prefactor = bands.ell .* (bands.ell .+ 1.0) ./ (2pi)
    dl_lee22 = prefactor .* bands.cl_lee22
    dl_battaglia16 = prefactor .* bands.cl_battaglia16
    dl_cross = prefactor .* bands.cl_cross
    percent = fill(NaN, length(bands.ell))
    correlation = fill(NaN, length(bands.ell))
    for index in eachindex(percent)
        battaglia = bands.cl_battaglia16[index]
        battaglia != 0.0 &&
            (percent[index] =
                100.0 * (bands.cl_lee22[index] - battaglia) / battaglia)
        denominator = sqrt(max(
            bands.cl_lee22[index] * bands.cl_battaglia16[index], 0.0,
        ))
        denominator > 0.0 &&
            (correlation[index] = bands.cl_cross[index] / denominator)
    end
    return (
        ell=bands.ell,
        cl_lee22=bands.cl_lee22,
        cl_battaglia16=bands.cl_battaglia16,
        cl_cross=bands.cl_cross,
        dl_lee22=dl_lee22,
        dl_battaglia16=dl_battaglia16,
        dl_cross=dl_cross,
        percent=percent,
        correlation=correlation,
    )
end

function write_bandpower_csv(path, columns)
    open(path, "w") do io
        println(
            io,
            "ell_effective,C_ell_lee22,C_ell_battaglia16,C_ell_cross," *
            "D_ell_lee22,D_ell_battaglia16,D_ell_cross," *
            "lee22_minus_battaglia16_percent,cross_correlation_coefficient",
        )
        for index in eachindex(columns.ell)
            println(
                io,
                "$(columns.ell[index]),$(columns.cl_lee22[index])," *
                "$(columns.cl_battaglia16[index]),$(columns.cl_cross[index])," *
                "$(columns.dl_lee22[index]),$(columns.dl_battaglia16[index])," *
                "$(columns.dl_cross[index]),$(columns.percent[index])," *
                "$(columns.correlation[index])",
            )
        end
    end
end

function save_comparison_plot(path, columns; nside, source_redshift, aperture, percent_limit)
    positive_lee = (columns.ell .>= 2) .& isfinite.(columns.dl_lee22) .&
                   (columns.dl_lee22 .> 0.0)
    positive_battaglia =
        (columns.ell .>= 2) .& isfinite.(columns.dl_battaglia16) .&
        (columns.dl_battaglia16 .> 0.0)
    any(positive_lee) || error("No positive Lee22 bandpowers can be plotted.")
    any(positive_battaglia) || error("No positive Battaglia16 bandpowers can be plotted.")

    upper = plot(
        columns.ell[positive_battaglia],
        columns.dl_battaglia16[positive_battaglia];
        label="Battaglia16",
        color=:mediumblue,
        linewidth=2.5,
        xscale=:log10,
        yscale=:log10,
        ylabel="D_ell^DM  [(pc cm^-3)^2]",
        title=(
            "HalfDome halo-DM angular power: " *
            "z_s=$(source_redshift), $(aperture)R200c, M200c, NSIDE=$(nside)"
        ),
        legend=:best,
        gridalpha=0.22,
        bottom_margin=0Plots.mm,
    )
    plot!(
        upper,
        columns.ell[positive_lee],
        columns.dl_lee22[positive_lee];
        label="Lee22 (no concentration)",
        color=:darkorange,
        linewidth=2.5,
    )

    finite_percent = isfinite.(columns.percent)
    any(finite_percent) || error("No finite percentage differences can be plotted.")
    lower = plot(
        columns.ell[finite_percent],
        columns.percent[finite_percent];
        label="",
        color=:black,
        linewidth=2.0,
        xscale=:log10,
        xlabel="Multipole ell",
        ylabel="(Lee22 - B16) / B16  [%]",
        gridalpha=0.22,
        top_margin=0Plots.mm,
    )
    hline!(lower, [0.0]; color=:gray, linestyle=:dot, linewidth=1.0, label="")
    if percent_limit > 0.0
        ylims!(lower, (-percent_limit, percent_limit))
    end

    figure = plot(
        upper,
        lower;
        layout=grid(2, 1; heights=[0.70, 0.30]),
        size=(980, 820),
        left_margin=8Plots.mm,
        right_margin=5Plots.mm,
        top_margin=5Plots.mm,
        bottom_margin=6Plots.mm,
    )
    savefig(figure, path)
    isfile(path) && filesize(path) > 0 ||
        error("Plot backend did not create a nonempty PDF: $(path)")
end

function write_hdf5(
    path,
    ell,
    cl_lee22,
    cl_battaglia16,
    cl_cross,
    columns;
    attributes,
)
    h5open(path, "w") do handle
        handle["ell"] = ell
        handle["cl_lee22"] = cl_lee22
        handle["cl_battaglia16"] = cl_battaglia16
        handle["cl_cross"] = cl_cross
        handle["band_ell_effective"] = columns.ell
        handle["band_cl_lee22"] = columns.cl_lee22
        handle["band_cl_battaglia16"] = columns.cl_battaglia16
        handle["band_cl_cross"] = columns.cl_cross
        handle["band_dl_lee22"] = columns.dl_lee22
        handle["band_dl_battaglia16"] = columns.dl_battaglia16
        handle["band_dl_cross"] = columns.dl_cross
        handle["band_lee22_minus_battaglia16_percent"] = columns.percent
        handle["band_cross_correlation_coefficient"] = columns.correlation
        for (key, value) in attributes
            attrs(handle)[String(key)] = value
        end
    end
end

function main()
    options = parse_options(ARGS)
    lee22_map_path = abspath(option(options, "lee22_map", ""))
    battaglia16_map_path = abspath(option(options, "battaglia16_map", ""))
    output_dir = abspath(option(
        options,
        "output_dir",
        "frb_map_generation/outputs/lee22_vs_battaglia16_power_spectra",
    ))
    lmax_requested = int_option(options, "lmax", 8192)
    niter = int_option(options, "niter", 0)
    plot_lmin = int_option(options, "plot_lmin", 2)
    plot_bins = int_option(options, "plot_bins", 55)
    source_redshift = float_option(options, "source_redshift", 1.0)
    aperture = float_option(options, "aperture_r200c", 3.0)
    deconvolve_pixel_window =
        bool_option(options, "deconvolve_pixel_window", false)
    percent_limit = float_option(options, "percent_limit", 0.0)
    overwrite = bool_option(options, "overwrite", false)

    isfile(lee22_map_path) || error("Lee22 FITS map not found: $(lee22_map_path)")
    isfile(battaglia16_map_path) ||
        error("Battaglia16 FITS map not found: $(battaglia16_map_path)")
    niter >= 0 || error("niter cannot be negative.")
    source_redshift > 0.0 || error("source_redshift must be positive.")
    aperture > 0.0 || error("aperture_r200c must be positive.")
    mkpath(output_dir)

    basename_root = "lee22_vs_battaglia16_full_halo_dm_power_spectra"
    full_csv = joinpath(output_dir, basename_root * ".csv")
    band_csv = joinpath(output_dir, basename_root * "_bandpowers.csv")
    hdf5_path = joinpath(output_dir, basename_root * ".h5")
    pdf_path = joinpath(output_dir, basename_root * ".pdf")
    provenance_path = joinpath(output_dir, basename_root * "_provenance.txt")
    outputs = (full_csv, band_csv, hdf5_path, pdf_path, provenance_path)
    existing = filter(isfile, outputs)
    isempty(existing) || overwrite || error(
        "Output exists: $(join(existing, ", ")). Pass --overwrite=true to replace it.",
    )

    println("Loading and transforming Lee22 map: $(lee22_map_path)")
    lee22_map = Healpix.readMapFromFITS(lee22_map_path, 1, Float64)
    nside = lee22_map.resolution.nside
    lee22_mean = validate_and_center_map!(lee22_map, "Lee22")
    lmax = min(lmax_requested, 3 * nside - 1)
    lmax >= 2 || error("lmax must be at least 2.")
    lee22_alm = Healpix.map2alm(lee22_map; lmax=lmax, mmax=lmax, niter=niter)
    lee22_map = nothing
    GC.gc()

    println("Loading and transforming Battaglia16 map: $(battaglia16_map_path)")
    battaglia16_map =
        Healpix.readMapFromFITS(battaglia16_map_path, 1, Float64)
    battaglia16_map.resolution.nside == nside || error(
        "Map NSIDE mismatch: Lee22=$(nside), " *
        "Battaglia16=$(battaglia16_map.resolution.nside).",
    )
    battaglia16_mean =
        validate_and_center_map!(battaglia16_map, "Battaglia16")
    battaglia16_alm =
        Healpix.map2alm(battaglia16_map; lmax=lmax, mmax=lmax, niter=niter)
    battaglia16_map = nothing
    GC.gc()

    println("Computing auto- and cross-spectra through ell=$(lmax)")
    cl_lee22 = Float64.(Healpix.alm2cl(lee22_alm))
    cl_battaglia16 = Float64.(Healpix.alm2cl(battaglia16_alm))
    cl_cross = Float64.(Healpix.alm2cl(lee22_alm, battaglia16_alm))
    lee22_alm = nothing
    battaglia16_alm = nothing
    GC.gc()

    if deconvolve_pixel_window
        pixel_window = Healpix.pixwin(nside)
        length(pixel_window) >= lmax + 1 || error(
            "HEALPix pixel window has only $(length(pixel_window)) multipoles.",
        )
        window_squared = pixel_window[1:lmax+1] .^ 2
        all(window_squared .> 0.0) ||
            error("HEALPix pixel window contains a zero in the requested range.")
        cl_lee22 ./= window_squared
        cl_battaglia16 ./= window_squared
        cl_cross ./= window_squared
    end

    ell = collect(0:lmax)
    bands = logarithmic_bandpowers(
        ell,
        cl_lee22,
        cl_battaglia16,
        cl_cross;
        lmin=plot_lmin,
        number_of_bins=plot_bins,
    )
    columns = bandpower_columns(bands)
    write_full_spectrum_csv(
        full_csv, ell, cl_lee22, cl_battaglia16, cl_cross,
    )
    write_bandpower_csv(band_csv, columns)
    save_comparison_plot(
        pdf_path,
        columns;
        nside=nside,
        source_redshift=source_redshift,
        aperture=aperture,
        percent_limit=percent_limit,
    )

    attributes = Dict{String,Any}(
        "created_utc" => Dates.format(now(UTC), dateformat"yyyy-mm-ddTHH:MM:SS"),
        "observable" => "observer-frame foreground halo DM fluctuation",
        "units_cl" => "(pc cm^-3)^2",
        "spectrum_definition" => "D_ell=ell(ell+1)C_ell/(2pi)",
        "percent_definition" =>
            "100*(C_ell_Lee22-C_ell_Battaglia16)/C_ell_Battaglia16",
        "lee22_map" => lee22_map_path,
        "battaglia16_map" => battaglia16_map_path,
        "lee22_removed_monopole_pc_cm3" => lee22_mean,
        "battaglia16_removed_monopole_pc_cm3" => battaglia16_mean,
        "nside" => nside,
        "ordering" => "RING",
        "lmax" => lmax,
        "niter" => niter,
        "source_redshift" => source_redshift,
        "mass_definition" => "M200c",
        "aperture_radius_definition" => "R200c",
        "aperture_r200c_multiplier" => aperture,
        "mass_range" => "complete resolved HalfDome catalogue range",
        "pixel_window_deconvolved" => deconvolve_pixel_window,
        "mask" => "none; complete full sky required",
        "monopole_removed" => true,
        "dipole_removed" => false,
        "input_is_120k_sparse_frb_map" => false,
    )
    write_hdf5(
        hdf5_path,
        ell,
        cl_lee22,
        cl_battaglia16,
        cl_cross,
        columns;
        attributes=attributes,
    )
    open(provenance_path, "w") do io
        for key in sort!(collect(keys(attributes)))
            println(io, "$(key)=$(attributes[key])")
        end
    end

    println("Saved full spectrum: $(full_csv)")
    println("Saved bandpowers: $(band_csv)")
    println("Saved HDF5: $(hdf5_path)")
    println("Saved comparison PDF: $(pdf_path)")
    println("Saved provenance: $(provenance_path)")
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
