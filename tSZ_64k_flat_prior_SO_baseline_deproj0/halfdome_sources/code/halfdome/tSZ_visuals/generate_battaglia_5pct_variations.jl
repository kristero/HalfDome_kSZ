include(joinpath(@__DIR__, "config.jl"))

const DEFAULT_BATTAGLIA_FRACTIONAL_DEVIATION = 5.0e-2

function battaglia_parameter_specs()
    return (
        (field=:P0_amp, cli_key="battaglia_P0_amp"),
        (field=:P0_alpha_m, cli_key="battaglia_P0_alpha_m"),
        (field=:P0_alpha_z, cli_key="battaglia_P0_alpha_z"),
        (field=:x_c_amp, cli_key="battaglia_x_c_amp"),
        (field=:x_c_alpha_m, cli_key="battaglia_x_c_alpha_m"),
        (field=:x_c_alpha_z, cli_key="battaglia_x_c_alpha_z"),
        (field=:beta_amp, cli_key="battaglia_beta_amp"),
        (field=:beta_alpha_m, cli_key="battaglia_beta_alpha_m"),
        (field=:beta_alpha_z, cli_key="battaglia_beta_alpha_z"),
        (field=:alpha_amp, cli_key="battaglia_alpha_amp"),
        (field=:alpha_alpha_m, cli_key="battaglia_alpha_alpha_m"),
        (field=:alpha_alpha_z, cli_key="battaglia_alpha_alpha_z"),
        (field=:gamma_amp, cli_key="battaglia_gamma_amp"),
        (field=:gamma_alpha_m, cli_key="battaglia_gamma_alpha_m"),
        (field=:gamma_alpha_z, cli_key="battaglia_gamma_alpha_z"),
    )
end

function battaglia_csv_header()
    return [
        "label",
        "varied_parameter",
        "fractional_deviation",
        "cosmo_h",
        "cosmo_omegab",
        "cosmo_omegac",
        "cosmo_omegam",
        "battaglia_P0_amp",
        "battaglia_P0_alpha_m",
        "battaglia_P0_alpha_z",
        "battaglia_x_c_amp",
        "battaglia_x_c_alpha_m",
        "battaglia_x_c_alpha_z",
        "battaglia_beta_amp",
        "battaglia_beta_alpha_m",
        "battaglia_beta_alpha_z",
        "battaglia_alpha_amp",
        "battaglia_alpha_alpha_m",
        "battaglia_alpha_alpha_z",
        "battaglia_gamma_amp",
        "battaglia_gamma_alpha_m",
        "battaglia_gamma_alpha_z",
        "note",
    ]
end

function battaglia_row(label::AbstractString, varied_parameter::AbstractString, frac::Float64, p, note::AbstractString)
    return (
        label=String(label),
        varied_parameter=String(varied_parameter),
        fractional_deviation=Float64(frac),
        cosmo_h=TSZ_H_VALUE,
        cosmo_omegab=TSZ_OMEGAB,
        cosmo_omegac=TSZ_OMEGAC,
        cosmo_omegam=TSZ_OMEGAM,
        battaglia_P0_amp=p.P0_amp,
        battaglia_P0_alpha_m=p.P0_alpha_m,
        battaglia_P0_alpha_z=p.P0_alpha_z,
        battaglia_x_c_amp=p.x_c_amp,
        battaglia_x_c_alpha_m=p.x_c_alpha_m,
        battaglia_x_c_alpha_z=p.x_c_alpha_z,
        battaglia_beta_amp=p.beta_amp,
        battaglia_beta_alpha_m=p.beta_alpha_m,
        battaglia_beta_alpha_z=p.beta_alpha_z,
        battaglia_alpha_amp=p.alpha_amp,
        battaglia_alpha_alpha_m=p.alpha_alpha_m,
        battaglia_alpha_alpha_z=p.alpha_alpha_z,
        battaglia_gamma_amp=p.gamma_amp,
        battaglia_gamma_alpha_m=p.gamma_alpha_m,
        battaglia_gamma_alpha_z=p.gamma_alpha_z,
        note=String(note),
    )
end

function build_battaglia_variation_rows(frac::Float64)
    frac > 0.0 || error("fractional deviation must be positive.")

    defaults = default_battaglia_params()
    specs = battaglia_parameter_specs()
    rows = Any[
        battaglia_row("fiducial", "fiducial", 0.0, defaults, "fiducial cosmology, fiducial Battaglia parameters"),
    ]
    skipped_zero_fields = String[]

    for spec in specs
        fid_value = getfield(defaults, spec.field)
        if fid_value == 0.0
            push!(skipped_zero_fields, String(spec.field))
            continue
        end

        for (suffix, sign) in (("plus", 1.0), ("minus", -1.0))
            varied = merge(defaults, (spec.field => fid_value * (1.0 + sign * frac),))
            push!(
                rows,
                battaglia_row(
                    "$(String(spec.field))_$(suffix)_$(round(100 * frac; digits=4))pct",
                    String(spec.field),
                    sign * frac,
                    varied,
                    "fiducial cosmology; one-at-a-time Battaglia variation"
                )
            )
        end
    end

    return rows, skipped_zero_fields
end

function write_battaglia_variation_csv(path::AbstractString, rows)
    mkpath(dirname(path))
    header = battaglia_csv_header()
    open(path, "w") do io
        println(io, join(header, ","))
        for row in rows
            values = (
                row.label,
                row.varied_parameter,
                string(row.fractional_deviation),
                string(row.cosmo_h),
                string(row.cosmo_omegab),
                string(row.cosmo_omegac),
                string(row.cosmo_omegam),
                string(row.battaglia_P0_amp),
                string(row.battaglia_P0_alpha_m),
                string(row.battaglia_P0_alpha_z),
                string(row.battaglia_x_c_amp),
                string(row.battaglia_x_c_alpha_m),
                string(row.battaglia_x_c_alpha_z),
                string(row.battaglia_beta_amp),
                string(row.battaglia_beta_alpha_m),
                string(row.battaglia_beta_alpha_z),
                string(row.battaglia_alpha_amp),
                string(row.battaglia_alpha_alpha_m),
                string(row.battaglia_alpha_alpha_z),
                string(row.battaglia_gamma_amp),
                string(row.battaglia_gamma_alpha_m),
                string(row.battaglia_gamma_alpha_z),
                row.note,
            )
            println(io, join(values, ","))
        end
    end
end

function print_battaglia_example_commands(rows)
    println("\nExample run_full_map.jl commands:")
    for row in rows
        if row.varied_parameter == "fiducial"
            println("julia tSZ_visuals/run_full_map.jl")
            continue
        end

        println(
            "julia tSZ_visuals/run_full_map.jl ",
            "battaglia_P0_amp=", row.battaglia_P0_amp, " ",
            "battaglia_P0_alpha_m=", row.battaglia_P0_alpha_m, " ",
            "battaglia_P0_alpha_z=", row.battaglia_P0_alpha_z, " ",
            "battaglia_x_c_amp=", row.battaglia_x_c_amp, " ",
            "battaglia_x_c_alpha_m=", row.battaglia_x_c_alpha_m, " ",
            "battaglia_x_c_alpha_z=", row.battaglia_x_c_alpha_z, " ",
            "battaglia_beta_amp=", row.battaglia_beta_amp, " ",
            "battaglia_beta_alpha_m=", row.battaglia_beta_alpha_m, " ",
            "battaglia_beta_alpha_z=", row.battaglia_beta_alpha_z, " ",
            "battaglia_alpha_amp=", row.battaglia_alpha_amp, " ",
            "battaglia_alpha_alpha_m=", row.battaglia_alpha_alpha_m, " ",
            "battaglia_alpha_alpha_z=", row.battaglia_alpha_alpha_z, " ",
            "battaglia_gamma_amp=", row.battaglia_gamma_amp, " ",
            "battaglia_gamma_alpha_m=", row.battaglia_gamma_alpha_m, " ",
            "battaglia_gamma_alpha_z=", row.battaglia_gamma_alpha_z
        )
    end
end

function main()
    frac = get_float_arg(
        "fractional_deviation",
        DEFAULT_BATTAGLIA_FRACTIONAL_DEVIATION;
        env="TSZ_BATTAGLIA_FRACTIONAL_DEVIATION"
    )
    output_path = abspath(
        get_string_arg(
            "output_path",
            joinpath(repo_root(), "tSZ_visuals", "battaglia_5pct_variations.csv");
            env="TSZ_BATTAGLIA_VARIATION_OUTPUT_PATH"
        )
    )

    rows, skipped_zero_fields = build_battaglia_variation_rows(frac)
    write_battaglia_variation_csv(output_path, rows)

    println("Wrote $(length(rows)) Battaglia variations to $(output_path)")
    if !isempty(skipped_zero_fields)
        println(
            "Skipped zero-centered parameters for relative variations: ",
            join(skipped_zero_fields, ", "),
            "."
        )
    end
    print_battaglia_example_commands(rows)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
