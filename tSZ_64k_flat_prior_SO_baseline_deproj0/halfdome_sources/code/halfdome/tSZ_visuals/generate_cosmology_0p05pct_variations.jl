include(joinpath(@__DIR__, "config.jl"))

const DEFAULT_FRACTIONAL_DEVIATION = 5.0e-4

function build_variation_rows(frac::Float64)
    frac > 0.0 || error("fractional deviation must be positive.")

    fid_h = TSZ_H_VALUE
    fid_ob = TSZ_OMEGAB
    fid_oc = TSZ_OMEGAC

    rows = NamedTuple[
        (
            label="fiducial",
            cosmo_h=fid_h,
            cosmo_omegab=fid_ob,
            cosmo_omegac=fid_oc,
            cosmo_omegam=fid_ob + fid_oc,
            fractional_deviation=0.0
        ),
    ]

    for (name, field) in (("h", :cosmo_h), ("omegab", :cosmo_omegab), ("omegac", :cosmo_omegac))
        for (suffix, sign) in (("plus", 1.0), ("minus", -1.0))
            h = fid_h
            ob = fid_ob
            oc = fid_oc

            if field === :cosmo_h
                h *= 1.0 + sign * frac
            elseif field === :cosmo_omegab
                ob *= 1.0 + sign * frac
            else
                oc *= 1.0 + sign * frac
            end

            push!(
                rows,
                (
                    label="$(name)_$(suffix)_$(round(100 * frac; digits=4))pct",
                    cosmo_h=h,
                    cosmo_omegab=ob,
                    cosmo_omegac=oc,
                    cosmo_omegam=ob + oc,
                    fractional_deviation=sign * frac
                )
            )
        end
    end

    return rows
end

function write_variation_csv(path::AbstractString, rows)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "label,cosmo_h,cosmo_omegab,cosmo_omegac,cosmo_omegam,fractional_deviation")
        for row in rows
            println(
                io,
                join(
                    (
                        row.label,
                        string(row.cosmo_h),
                        string(row.cosmo_omegab),
                        string(row.cosmo_omegac),
                        string(row.cosmo_omegam),
                        string(row.fractional_deviation)
                    ),
                    ","
                )
            )
        end
    end
end

function print_example_commands(rows)
    println("\nExample run_full_map.jl commands:")
    for row in rows
        println(
            "julia tSZ_visuals/run_full_map.jl ",
            "cosmo_h=", row.cosmo_h, " ",
            "cosmo_omegab=", row.cosmo_omegab, " ",
            "cosmo_omegac=", row.cosmo_omegac
        )
    end
end

function main()
    frac = get_float_arg(
        "fractional_deviation",
        DEFAULT_FRACTIONAL_DEVIATION;
        env="TSZ_COSMO_FRACTIONAL_DEVIATION"
    )
    output_path = abspath(
        get_string_arg(
            "output_path",
            joinpath(repo_root(), "tSZ_visuals", "cosmology_0p05pct_variations.csv");
            env="TSZ_COSMO_VARIATION_OUTPUT_PATH"
        )
    )

    rows = build_variation_rows(frac)
    write_variation_csv(output_path, rows)

    println("Wrote $(length(rows)) cosmology variations to $(output_path)")
    print_example_commands(rows)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
