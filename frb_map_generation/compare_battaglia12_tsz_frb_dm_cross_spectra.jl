#!/usr/bin/env julia

# Cross one matched Battaglia12 Compton-y map with two full-sky halo-DM maps:
# Lee22 no-concentration and Battaglia16.  All three input maps must use the
# same HalfDome lightcone, halo selection, M200c convention, NSIDE, and 3R200c
# angular support.

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
    minimum(pixels) >= 0.0 || error(
        "$(label) is not a complete non-negative full-sky map. " *
        "Masked maps require a pseudo-C_ell mode-coupling correction.",
    )
    map_mean = mean(pixels)
    pixels .-= map_mean
    return map_mean
end

function logarithmic_edges(lmin::Int, lmax::Int, number_of_bins::Int)
    2 <= lmin < lmax || error("plot_lmin must be in [2, $(lmax - 1)].")
    number_of_bins > 1 || error("plot_bins must exceed one.")
    edges = unique(round.(
        Int,
        exp.(range(log(Float64(lmin)), log(Float64(lmax + 1)); length=number_of_bins + 1)),
    ))
    first(edges) > lmin && pushfirst!(edges, lmin)
    last(edges) <= lmax && push!(edges, lmax + 1)
    return edges
end

function logarithmic_bandpowers(ell, arrays::NamedTuple; lmin::Int, number_of_bins::Int)
    edges = logarithmic_edges(lmin, last(ell), number_of_bins)
    band_ell = Float64[]
    output = Dict(name => Float64[] for name in keys(arrays))
    for (lower, upper) in zip(edges[1:end-1], edges[2:end])
        indices = findall(l -> lower <= l < upper, ell)
        isempty(indices) && continue
        weights = 2.0 .* Float64.(ell[indices]) .+ 1.0
        normalization = sum(weights)
        push!(band_ell, sum(weights .* ell[indices]) / normalization)
        for name in keys(arrays)
            values = getproperty(arrays, name)
            push!(output[name], sum(weights .* values[indices]) / normalization)
        end
    end
    return merge((ell=band_ell,), NamedTuple{keys(arrays)}(Tuple(output[name] for name in keys(arrays))))
end

function derived_columns(bands; percent_floor_fraction::Float64)
    prefactor = bands.ell .* (bands.ell .+ 1.0) ./ (2pi)
    dl_y_x_lee22 = prefactor .* bands.cl_y_x_lee22
    dl_y_x_battaglia16 = prefactor .* bands.cl_y_x_battaglia16

    finite_reference = abs.(bands.cl_y_x_battaglia16[isfinite.(bands.cl_y_x_battaglia16)])
    isempty(finite_reference) && error("Battaglia16 reference cross-spectrum is entirely non-finite.")
    reference_scale = maximum(finite_reference)
    denominator_floor = percent_floor_fraction * reference_scale

    percent = fill(NaN, length(bands.ell))
    r_y_lee22 = fill(NaN, length(bands.ell))
    r_y_battaglia16 = fill(NaN, length(bands.ell))
    for index in eachindex(percent)
        reference = bands.cl_y_x_battaglia16[index]
        if isfinite(reference) && abs(reference) > denominator_floor
            percent[index] = 100.0 * (bands.cl_y_x_lee22[index] - reference) / reference
        end

        denominator_lee = sqrt(max(bands.cl_yy[index] * bands.cl_dm_lee22[index], 0.0))
        denominator_battaglia =
            sqrt(max(bands.cl_yy[index] * bands.cl_dm_battaglia16[index], 0.0))
        denominator_lee > 0.0 &&
            (r_y_lee22[index] = bands.cl_y_x_lee22[index] / denominator_lee)
        denominator_battaglia > 0.0 &&
            (r_y_battaglia16[index] =
                bands.cl_y_x_battaglia16[index] / denominator_battaglia)
    end

    return merge(
        bands,
        (
            dl_y_x_lee22=dl_y_x_lee22,
            dl_y_x_battaglia16=dl_y_x_battaglia16,
            lee22_minus_battaglia16_percent=percent,
            r_y_lee22=r_y_lee22,
            r_y_battaglia16=r_y_battaglia16,
            percent_denominator_floor=denominator_floor,
        ),
    )
end

function write_full_csv(path, ell, spectra, percent_floor_fraction)
    reference_scale = maximum(abs.(spectra.cl_y_x_battaglia16[isfinite.(spectra.cl_y_x_battaglia16)]))
    denominator_floor = percent_floor_fraction * reference_scale
    open(path, "w") do io
        println(
            io,
            "ell,C_ell_yy,C_ell_dm_lee22,C_ell_dm_battaglia16," *
            "C_ell_y_x_dm_lee22,C_ell_y_x_dm_battaglia16," *
            "D_ell_y_x_dm_lee22,D_ell_y_x_dm_battaglia16," *
            "lee22_minus_battaglia16_cross_percent,r_ell_y_lee22,r_ell_y_battaglia16",
        )
        for index in eachindex(ell)
            l = ell[index]
            prefactor = l * (l + 1.0) / (2pi)
            ref = spectra.cl_y_x_battaglia16[index]
            percent = abs(ref) > denominator_floor ?
                100.0 * (spectra.cl_y_x_lee22[index] - ref) / ref : NaN
            denom_lee = sqrt(max(spectra.cl_yy[index] * spectra.cl_dm_lee22[index], 0.0))
            denom_b16 = sqrt(max(spectra.cl_yy[index] * spectra.cl_dm_battaglia16[index], 0.0))
            r_lee = denom_lee > 0.0 ? spectra.cl_y_x_lee22[index] / denom_lee : NaN
            r_b16 = denom_b16 > 0.0 ? spectra.cl_y_x_battaglia16[index] / denom_b16 : NaN
            println(
                io,
                "$(l),$(spectra.cl_yy[index]),$(spectra.cl_dm_lee22[index])," *
                "$(spectra.cl_dm_battaglia16[index]),$(spectra.cl_y_x_lee22[index])," *
                "$(ref),$(prefactor * spectra.cl_y_x_lee22[index])," *
                "$(prefactor * ref),$(percent),$(r_lee),$(r_b16)",
            )
        end
    end
end

function write_bandpower_csv(path, columns)
    open(path, "w") do io
        println(
            io,
            "ell_effective,C_ell_yy,C_ell_dm_lee22,C_ell_dm_battaglia16," *
            "C_ell_y_x_dm_lee22,C_ell_y_x_dm_battaglia16," *
            "D_ell_y_x_dm_lee22,D_ell_y_x_dm_battaglia16," *
            "lee22_minus_battaglia16_cross_percent,r_ell_y_lee22,r_ell_y_battaglia16",
        )
        for index in eachindex(columns.ell)
            println(
                io,
                "$(columns.ell[index]),$(columns.cl_yy[index])," *
                "$(columns.cl_dm_lee22[index]),$(columns.cl_dm_battaglia16[index])," *
                "$(columns.cl_y_x_lee22[index]),$(columns.cl_y_x_battaglia16[index])," *
                "$(columns.dl_y_x_lee22[index]),$(columns.dl_y_x_battaglia16[index])," *
                "$(columns.lee22_minus_battaglia16_percent[index])," *
                "$(columns.r_y_lee22[index]),$(columns.r_y_battaglia16[index])",
            )
        end
    end
end

function save_cross_plot(path, columns; nside, source_redshift, aperture, percent_limit)
    positive_lee = isfinite.(columns.dl_y_x_lee22) .& (columns.dl_y_x_lee22 .> 0.0)
    positive_b16 = isfinite.(columns.dl_y_x_battaglia16) .&
                   (columns.dl_y_x_battaglia16 .> 0.0)
    any(positive_lee) || error("No positive y x Lee22-DM bandpowers can be plotted.")
    any(positive_b16) || error("No positive y x Battaglia16-DM bandpowers can be plotted.")

    upper = plot(
        columns.ell[positive_b16],
        columns.dl_y_x_battaglia16[positive_b16];
        label="Battaglia12 y x Battaglia16 DM",
        color=:mediumblue,
        linewidth=2.7,
        xscale=:log10,
        yscale=:log10,
        ylabel="D_ell^{y x DM}  [pc cm^-3]",
        title="Matched HalfDome tSZ-FRB cross-spectra",
        legend=:best,
        gridalpha=0.22,
        bottom_margin=0Plots.mm,
    )
    plot!(
        upper,
        columns.ell[positive_lee],
        columns.dl_y_x_lee22[positive_lee];
        label="Battaglia12 y x Lee22 DM",
        color=:darkorange,
        linewidth=2.7,
    )
    finite_percent = isfinite.(columns.lee22_minus_battaglia16_percent)
    any(finite_percent) || error("No stable percentage differences can be plotted.")
    lower = plot(
        columns.ell[finite_percent],
        columns.lee22_minus_battaglia16_percent[finite_percent];
        label="",
        color=:black,
        linewidth=2.0,
        xscale=:log10,
        xlabel="Multipole ell",
        ylabel="(y x Lee22 - y x B16) / (y x B16)  [%]",
        gridalpha=0.22,
        top_margin=0Plots.mm,
    )
    hline!(lower, [0.0]; color=:gray, linestyle=:dot, linewidth=1.0, label="")
    percent_limit > 0.0 && ylims!(lower, (-percent_limit, percent_limit))

    figure = plot(
        upper,
        lower;
        layout=grid(2, 1; heights=[0.70, 0.30]),
        size=(1050, 840),
        left_margin=10Plots.mm,
        right_margin=7Plots.mm,
        top_margin=7Plots.mm,
        bottom_margin=7Plots.mm,
    )
    savefig(figure, path)
    isfile(path) && filesize(path) > 0 || error("Failed to create plot: $(path)")
end

function write_hdf5(path, ell, spectra, columns; attributes)
    h5open(path, "w") do handle
        handle["ell"] = ell
        for name in keys(spectra)
            handle[String(name)] = getproperty(spectra, name)
        end
        handle["band_ell_effective"] = columns.ell
        for name in (
            :cl_yy,
            :cl_dm_lee22,
            :cl_dm_battaglia16,
            :cl_y_x_lee22,
            :cl_y_x_battaglia16,
            :dl_y_x_lee22,
            :dl_y_x_battaglia16,
            :lee22_minus_battaglia16_percent,
            :r_y_lee22,
            :r_y_battaglia16,
        )
            handle["band_$(name)"] = getproperty(columns, name)
        end
        for (key, value) in attributes
            attrs(handle)[String(key)] = value
        end
    end
end

function main()
    options = parse_options(ARGS)
    y_map_path = abspath(option(options, "tsz_map", ""))
    lee22_map_path = abspath(option(options, "lee22_dm_map", ""))
    battaglia16_map_path = abspath(option(options, "battaglia16_dm_map", ""))
    output_dir = abspath(option(
        options,
        "output_dir",
        "frb_map_generation/outputs/power_spectra/tsz_frb_cross",
    ))
    lmax_requested = int_option(options, "lmax", 8192)
    niter = int_option(options, "niter", 0)
    plot_lmin = int_option(options, "plot_lmin", 2)
    plot_bins = int_option(options, "plot_bins", 55)
    source_redshift = float_option(options, "source_redshift", 1.0)
    aperture = float_option(options, "aperture_r200c", 3.0)
    deconvolve_pixel_window = bool_option(options, "deconvolve_pixel_window", false)
    percent_limit = float_option(options, "percent_limit", 0.0)
    percent_floor_fraction =
        float_option(options, "percent_denominator_floor_fraction", 1.0e-6)
    overwrite = bool_option(options, "overwrite", false)

    for (label, path) in (
        ("Battaglia12 Compton-y", y_map_path),
        ("Lee22 DM", lee22_map_path),
        ("Battaglia16 DM", battaglia16_map_path),
    )
        isfile(path) || error("$(label) FITS map not found: $(path)")
    end
    niter >= 0 || error("niter cannot be negative.")
    percent_floor_fraction >= 0.0 ||
        error("percent_denominator_floor_fraction cannot be negative.")
    mkpath(output_dir)

    root = "battaglia12_tsz_x_lee22_battaglia16_dm_cross_spectra"
    full_csv = joinpath(output_dir, root * ".csv")
    band_csv = joinpath(output_dir, root * "_bandpowers.csv")
    hdf5_path = joinpath(output_dir, root * ".h5")
    pdf_path = joinpath(output_dir, root * ".pdf")
    provenance_path = joinpath(output_dir, root * "_provenance.txt")
    outputs = (full_csv, band_csv, hdf5_path, pdf_path, provenance_path)
    existing = filter(isfile, outputs)
    isempty(existing) || overwrite || error(
        "Output exists: $(join(existing, ", ")). Pass --overwrite=true to replace it.",
    )

    println("Loading and transforming Battaglia12 Compton-y map: $(y_map_path)")
    y_map = Healpix.readMapFromFITS(y_map_path, 1, Float64)
    nside = y_map.resolution.nside
    lmax = min(lmax_requested, 3 * nside - 1)
    lmax >= 2 || error("lmax must be at least 2.")
    y_mean = validate_and_center_map!(y_map, "Battaglia12 Compton-y")
    y_alm = Healpix.map2alm(y_map; lmax=lmax, mmax=lmax, niter=niter)
    y_map = nothing
    GC.gc()
    cl_yy = Float64.(Healpix.alm2cl(y_alm))

    println("Loading and transforming Battaglia16 halo-DM map: $(battaglia16_map_path)")
    battaglia16_map = Healpix.readMapFromFITS(battaglia16_map_path, 1, Float64)
    battaglia16_map.resolution.nside == nside || error("Battaglia16 DM NSIDE mismatch.")
    battaglia16_mean = validate_and_center_map!(battaglia16_map, "Battaglia16 DM")
    battaglia16_alm =
        Healpix.map2alm(battaglia16_map; lmax=lmax, mmax=lmax, niter=niter)
    battaglia16_map = nothing
    GC.gc()
    cl_dm_battaglia16 = Float64.(Healpix.alm2cl(battaglia16_alm))
    cl_y_x_battaglia16 = Float64.(Healpix.alm2cl(y_alm, battaglia16_alm))
    battaglia16_alm = nothing
    GC.gc()

    println("Loading and transforming Lee22 halo-DM map: $(lee22_map_path)")
    lee22_map = Healpix.readMapFromFITS(lee22_map_path, 1, Float64)
    lee22_map.resolution.nside == nside || error("Lee22 DM NSIDE mismatch.")
    lee22_mean = validate_and_center_map!(lee22_map, "Lee22 DM")
    lee22_alm = Healpix.map2alm(lee22_map; lmax=lmax, mmax=lmax, niter=niter)
    lee22_map = nothing
    GC.gc()
    cl_dm_lee22 = Float64.(Healpix.alm2cl(lee22_alm))
    cl_y_x_lee22 = Float64.(Healpix.alm2cl(y_alm, lee22_alm))
    y_alm = nothing
    lee22_alm = nothing
    GC.gc()

    if deconvolve_pixel_window
        pixel_window = Healpix.pixwin(nside)
        length(pixel_window) >= lmax + 1 ||
            error("HEALPix pixel window is shorter than the requested lmax.")
        window_squared = pixel_window[1:lmax+1] .^ 2
        all(window_squared .> 0.0) || error("Pixel window contains a zero.")
        for values in (
            cl_yy,
            cl_dm_lee22,
            cl_dm_battaglia16,
            cl_y_x_lee22,
            cl_y_x_battaglia16,
        )
            values ./= window_squared
        end
    end

    ell = collect(0:lmax)
    spectra = (
        cl_yy=cl_yy,
        cl_dm_lee22=cl_dm_lee22,
        cl_dm_battaglia16=cl_dm_battaglia16,
        cl_y_x_lee22=cl_y_x_lee22,
        cl_y_x_battaglia16=cl_y_x_battaglia16,
    )
    bands = logarithmic_bandpowers(
        ell,
        spectra;
        lmin=plot_lmin,
        number_of_bins=plot_bins,
    )
    columns = derived_columns(
        bands;
        percent_floor_fraction=percent_floor_fraction,
    )

    write_full_csv(full_csv, ell, spectra, percent_floor_fraction)
    write_bandpower_csv(band_csv, columns)
    save_cross_plot(
        pdf_path,
        columns;
        nside=nside,
        source_redshift=source_redshift,
        aperture=aperture,
        percent_limit=percent_limit,
    )

    attributes = Dict{String,Any}(
        "created_utc" => Dates.format(now(UTC), dateformat"yyyy-mm-ddTHH:MM:SS"),
        "observable" => "Battaglia12 thermal-SZ Compton-y x foreground halo DM",
        "tsz_model" => "fiducial Battaglia12 pressure parameters via XGPaint Battaglia16ThermalSZProfile",
        "dm_models" => "Lee22 no-concentration and Battaglia16",
        "units_cl_y_x_dm" => "pc cm^-3",
        "spectrum_definition" => "D_ell=ell(ell+1)C_ell/(2pi)",
        "percent_definition" =>
            "100*(C_ell[y x Lee22 DM]-C_ell[y x Battaglia16 DM])/C_ell[y x Battaglia16 DM]",
        "percent_denominator_floor_fraction" => percent_floor_fraction,
        "band_percent_denominator_floor_absolute" => columns.percent_denominator_floor,
        "tsz_map" => y_map_path,
        "lee22_dm_map" => lee22_map_path,
        "battaglia16_dm_map" => battaglia16_map_path,
        "removed_tsz_monopole_y" => y_mean,
        "removed_lee22_dm_monopole_pc_cm3" => lee22_mean,
        "removed_battaglia16_dm_monopole_pc_cm3" => battaglia16_mean,
        "nside" => nside,
        "ordering" => "RING",
        "lmax" => lmax,
        "niter" => niter,
        "source_redshift" => source_redshift,
        "foreground_redshift_interval" => "(0, source_redshift]",
        "mass_definition" => "M200c",
        "aperture_radius_definition" => "R200c",
        "aperture_r200c_multiplier" => aperture,
        "mass_range" => "complete resolved HalfDome catalogue range",
        "pixel_window_deconvolved" => deconvolve_pixel_window,
        "mask" => "none; complete full sky required",
        "monopole_removed" => true,
        "dipole_removed" => false,
        "input_is_sparse_frb_ray_map" => false,
    )
    write_hdf5(hdf5_path, ell, spectra, columns; attributes=attributes)
    open(provenance_path, "w") do io
        for key in sort!(collect(keys(attributes)))
            println(io, "$(key)=$(attributes[key])")
        end
    end

    println("Saved full spectra: $(full_csv)")
    println("Saved bandpowers: $(band_csv)")
    println("Saved HDF5: $(hdf5_path)")
    println("Saved comparison PDF: $(pdf_path)")
    println("Saved provenance: $(provenance_path)")
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
