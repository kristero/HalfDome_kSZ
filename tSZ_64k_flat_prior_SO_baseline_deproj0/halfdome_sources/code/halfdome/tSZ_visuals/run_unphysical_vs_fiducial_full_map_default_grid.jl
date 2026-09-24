using Dates
using Printf

const COMPARISON_OUTPUT_DIR = "/home/cbllover/HalfDome/HPC_output/HalfDome/tests/unphysical_param_tests"

function set_env_values!(pairs)
    for (key, value) in pairs
        ENV[String(key)] = string(value)
    end
    return nothing
end

function local_halfdome_path()
    env_path = strip(get(ENV, "HALFDOME_PATH", ""))
    if !isempty(env_path) && ispath(env_path)
        return env_path
    end

    candidates = (
        "/home/cbllover/HalfDome/lightcone_100.hdf5",
        normpath(joinpath(@__DIR__, "..", "lightcone_100.hdf5")),
    )
    for path in candidates
        isfile(path) && return path
    end

    isempty(env_path) || @warn "HALFDOME_PATH is set but not accessible; keeping it so the HDF5 error shows the exact path." path=env_path
    return isempty(env_path) ? first(candidates) : env_path
end

# Full HalfDome catalog run. No N_logtheta/N_z/N_logM override is set here:
# XGPaint keeps its default 512 x 256 x 256 profile grid.
set_env_values!((
    "TSZ_VISUAL_OUTPUT_DIR" => COMPARISON_OUTPUT_DIR,
    "TSZ_VISUAL_CACHE_DIR" => joinpath(COMPARISON_OUTPUT_DIR, "cache"),
    "TSZ_CATALOG_SOURCE" => "halfdome",
    "HALFDOME_PATH" => local_halfdome_path(),
    "BATCHING_MODE" => "full",
    "SAVE_CL" => true,
    "SAVE_HEALPIX_MAP" => false,
    "SAVE_MASS_MAP" => false,
    "SAVE_BIN_MAPS" => false,
    "SKIP_EXISTING_OUTPUTS" => false,
    "APPLY_GAUSSIAN_BEAM" => false,
    "APPLY_MASS_CUT" => false,
    "ADD_STR_END" => "all_halos_unphysical_param_test",
    "BATTAGLIA_SOBOL_ROW" => 0,
    # Use existing caches when they match exactly; otherwise build the default grid.
    "MODEL_EXISTS" => false,
    "REUSE_EXISTING_CACHE" => true,
    "INTERPOLATOR_PAD" => 256,
    "INTERPOLATOR_LOGM_MAX" => 15.7,
))

include(joinpath(@__DIR__, "run_tSZ_visuals.jl"))

function patch_xgpaint_nonpositive_cleanup_thread_slots!()
    @eval XGPaint begin
        function replace_nonpositive_with_floor!(y_prof_grid)
            T = eltype(y_prof_grid)
            N_values = length(y_prof_grid)
            chunk_list = collect(chunks(1:N_values; n=Threads.nthreads()))
            local_mins = fill(typemax(T), length(chunk_list))
            local_positive_counts = zeros(Int, length(chunk_list))
            local_bad_counts = zeros(Int, length(chunk_list))

            Threads.@threads for chunk_idx in eachindex(chunk_list)
                local_min = local_mins[chunk_idx]
                positive_count = 0
                bad_count = 0

                @inbounds for idx in chunk_list[chunk_idx]
                    value = y_prof_grid[idx]
                    if value > zero(T)
                        positive_count += 1
                        if value < local_min
                            local_min = value
                        end
                    else
                        bad_count += 1
                    end
                end

                local_mins[chunk_idx] = local_min
                local_positive_counts[chunk_idx] = positive_count
                local_bad_counts[chunk_idx] = bad_count
            end

            replaced_count = sum(local_bad_counts)
            replaced_count == 0 && return replaced_count, zero(T)

            total_positive_count = sum(local_positive_counts)
            floor_val = if total_positive_count == 0
                nextfloat(zero(T))
            else
                minimum(local_mins[local_positive_counts .> 0]) * T(1e-6)
            end

            Threads.@threads for chunk_idx in eachindex(chunk_list)
                @inbounds for idx in chunk_list[chunk_idx]
                    if y_prof_grid[idx] <= zero(T)
                        y_prof_grid[idx] = floor_val
                    end
                end
            end

            return replaced_count, floor_val
        end
    end
    return nothing
end

patch_xgpaint_nonpositive_cleanup_thread_slots!()

const COMPARISON_PLOTS_AVAILABLE = try
    @eval using Plots
    true
catch err
    @warn "Plots.jl is unavailable; CSV and FITS outputs will still be saved." exception=(err, catch_backtrace())
    false
end

const BATTAGLIA_ENV_FIELDS = (
    ("BATTAGLIA_P0_AMP", :P0_amp),
    ("BATTAGLIA_P0_ALPHA_M", :P0_alpha_m),
    ("BATTAGLIA_P0_ALPHA_Z", :P0_alpha_z),
    ("BATTAGLIA_X_C_AMP", :x_c_amp),
    ("BATTAGLIA_X_C_ALPHA_M", :x_c_alpha_m),
    ("BATTAGLIA_X_C_ALPHA_Z", :x_c_alpha_z),
    ("BATTAGLIA_BETA_AMP", :beta_amp),
    ("BATTAGLIA_BETA_ALPHA_M", :beta_alpha_m),
    ("BATTAGLIA_BETA_ALPHA_Z", :beta_alpha_z),
    ("BATTAGLIA_ALPHA_AMP", :alpha_amp),
    ("BATTAGLIA_ALPHA_ALPHA_M", :alpha_alpha_m),
    ("BATTAGLIA_ALPHA_ALPHA_Z", :alpha_alpha_z),
    ("BATTAGLIA_GAMMA_AMP", :gamma_amp),
    ("BATTAGLIA_GAMMA_ALPHA_M", :gamma_alpha_m),
    ("BATTAGLIA_GAMMA_ALPHA_Z", :gamma_alpha_z),
)

function battaglia_env_pairs(params)
    return [(env_key, getproperty(params, field)) for (env_key, field) in BATTAGLIA_ENV_FIELDS]
end

const FIDUCIAL_BATTAGLIA_PARAMS = default_battaglia_params()

const UNPHYSICAL_BATTAGLIA_PARAMS = battaglia_namedtuple(
    P0_amp=40.0,
    P0_alpha_m=0.50,
    P0_alpha_z=-1.558,
    x_c_amp=1.0,
    x_c_alpha_m=-0.15,
    x_c_alpha_z=1.2,
    beta_amp=8,
    beta_alpha_m=0.2,
    beta_alpha_z=0.8,
    alpha_amp=1.0,
    alpha_alpha_m=0.0,
    alpha_alpha_z=0.0,
    gamma_amp=-0.3,
    gamma_alpha_m=0.0,
    gamma_alpha_z=0.0,
)

function dell_from_cl(cl)
    cl_vec = Float64.(collect(cl))
    ell = collect(0:(length(cl_vec) - 1))
    dell = ell .* (ell .+ 1.0) .* cl_vec ./ (2.0 * pi)
    return ell, cl_vec, dell
end

function write_dell_csv(label::AbstractString, cfg::VisualConfig, cl)
    ell, cl_vec, dell = dell_from_cl(cl)
    path = joinpath(cfg.output_dir, "$(label)_all_halos_D_ell.csv")
    isdir(dirname(path)) || mkpath(dirname(path))
    open(path, "w") do io
        println(io, "ell,C_ell,D_ell")
        for i in eachindex(ell)
            println(io, "$(ell[i]),$(@sprintf("%.17e", cl_vec[i])),$(@sprintf("%.17e", dell[i]))")
        end
    end
    println("Saved $(label) D_ell CSV to $(abspath(path))")
    return (path=path, ell=ell, cl=cl_vec, dell=dell)
end

function write_case_summary(label::AbstractString, cfg::VisualConfig)
    path = joinpath(cfg.output_dir, "$(label)_all_halos_summary.txt")
    p = cfg.battaglia_params
    slice = battaglia_slice_params(p, 1.0e14, 0.5)
    open(path, "w") do io
        println(io, "Full-catalog HalfDome tSZ run: $(label)")
        println(io, "timestamp=$(Dates.format(now(), dateformat"yyyy-mm-ddTHH:MM:SS"))")
        println(io, "catalog_path=$(cfg.catalog_path)")
        println(io, "apply_mass_cut=$(cfg.apply_mass_cut)")
        println(io, "mass_min=$(cfg.mass_min)")
        println(io, "nside=$(cfg.nside)")
        println(io, "chunkN=$(cfg.chunkN)")
        println(io, "output_cl_fits=$(abspath(cfg.cl_output_path))")
        println(io, "interpolator_grid=default_XGPaint_512x256x256")
        println(io, "P0_amp=$(p.P0_amp)")
        println(io, "P0_alpha_m=$(p.P0_alpha_m)")
        println(io, "P0_alpha_z=$(p.P0_alpha_z)")
        println(io, "x_c_amp=$(p.x_c_amp)")
        println(io, "x_c_alpha_m=$(p.x_c_alpha_m)")
        println(io, "x_c_alpha_z=$(p.x_c_alpha_z)")
        println(io, "beta_amp=$(p.beta_amp)")
        println(io, "beta_alpha_m=$(p.beta_alpha_m)")
        println(io, "beta_alpha_z=$(p.beta_alpha_z)")
        println(io, "alpha_amp=$(p.alpha_amp)")
        println(io, "alpha_alpha_m=$(p.alpha_alpha_m)")
        println(io, "alpha_alpha_z=$(p.alpha_alpha_z)")
        println(io, "gamma_amp=$(p.gamma_amp)")
        println(io, "gamma_alpha_m=$(p.gamma_alpha_m)")
        println(io, "gamma_alpha_z=$(p.gamma_alpha_z)")
        println(io, "slice_M1e14_z0p5_beta_outer=$(slice.beta_outer)")
        println(io, "slice_M1e14_z0p5_LOS_outer_converges_beta_outer_gt_1=$(slice.beta_outer > 1.0)")
        println(io, "slice_M1e14_z0p5_thermal_energy_converges_beta_outer_gt_3=$(slice.beta_outer > 3.0)")
    end
    println("Saved $(label) summary to $(abspath(path))")
    return path
end

function run_catalog_case(label::AbstractString, params; enforce_guardrails::Bool)
    println("")
    println("=== Starting full HalfDome all-halo run: $(label) ===")
    set_env_values!(battaglia_env_pairs(params))
    set_env_values!((
        "ENFORCE_BATTAGLIA_GUARDRAILS" => enforce_guardrails,
        "SKIP_INVALID_BATTAGLIA_ROWS" => false,
        "TSZ_RUN_INSTANCE_TAG" => "$(label)_all_halos_defaultgrid",
    ))

    result = run_tsz_visual_fits()
    result === nothing && error("$(label) run returned nothing.")
    result.cl === nothing && error("$(label) run did not compute C_l. SAVE_CL must be true.")

    spectra = write_dell_csv(label, result.cfg, result.cl)
    summary_path = write_case_summary(label, result.cfg)

    GC.gc()
    if isdefined(@__MODULE__, :trim_process_memory)
        trim_process_memory()
    end

    return (label=label, cfg=result.cfg, spectra=spectra, summary_path=summary_path)
end

function write_comparison_outputs(fid, unphys)
    ell_fid = fid.spectra.ell
    ell_unphys = unphys.spectra.ell
    n = min(length(ell_fid), length(ell_unphys))
    ell = ell_fid[1:n]
    all(ell .== ell_unphys[1:n]) || error("Fiducial and unphysical spectra have different ell grids.")

    cl_fid = fid.spectra.cl[1:n]
    cl_unphys = unphys.spectra.cl[1:n]
    dell_fid = fid.spectra.dell[1:n]
    dell_unphys = unphys.spectra.dell[1:n]
    cl_diff = cl_unphys .- cl_fid
    dell_diff = dell_unphys .- dell_fid
    cl_ratio = cl_unphys ./ cl_fid
    dell_ratio = dell_unphys ./ dell_fid
    cl_frac_diff = cl_diff ./ cl_fid
    dell_frac_diff = dell_diff ./ dell_fid

    outdir = fid.cfg.output_dir
    csv_path = joinpath(outdir, "fiducial_vs_unphysical_all_halos_D_ell_comparison.csv")
    open(csv_path, "w") do io
        println(io, "ell,C_ell_fiducial,C_ell_unphysical,C_ell_difference,C_ell_ratio,C_ell_fractional_difference,D_ell_fiducial,D_ell_unphysical,D_ell_difference,D_ell_ratio,D_ell_fractional_difference")
        for i in eachindex(ell)
            println(
                io,
                join((
                    ell[i],
                    @sprintf("%.17e", cl_fid[i]),
                    @sprintf("%.17e", cl_unphys[i]),
                    @sprintf("%.17e", cl_diff[i]),
                    @sprintf("%.17e", cl_ratio[i]),
                    @sprintf("%.17e", cl_frac_diff[i]),
                    @sprintf("%.17e", dell_fid[i]),
                    @sprintf("%.17e", dell_unphys[i]),
                    @sprintf("%.17e", dell_diff[i]),
                    @sprintf("%.17e", dell_ratio[i]),
                    @sprintf("%.17e", dell_frac_diff[i]),
                ), ",")
            )
        end
    end
    println("Saved comparison CSV to $(abspath(csv_path))")

    if COMPARISON_PLOTS_AVAILABLE
        valid = isfinite.(dell_fid) .& isfinite.(dell_unphys) .&
                (dell_fid .> 0.0) .& (dell_unphys .> 0.0) .& (ell .>= 2)
        if any(valid)
            spectra_png = joinpath(outdir, "fiducial_vs_unphysical_all_halos_D_ell_loglog.png")
            fig = Plots.plot(
                ell[valid],
                dell_fid[valid];
                xscale=:log10,
                yscale=:log10,
                linewidth=2,
                label="fiducial",
                xlabel="ell",
                ylabel="D_ell",
                title="Full HalfDome all-halo D_ell",
                legend=:best,
                size=(950, 620),
            )
            Plots.plot!(fig, ell[valid], dell_unphys[valid]; linewidth=2, label="unphysical")
            Plots.savefig(fig, spectra_png)
            println("Saved D_ell comparison plot to $(abspath(spectra_png))")
        else
            @warn "No positive finite overlapping D_ell values to plot on log-log axes."
        end

        frac_valid = isfinite.(dell_frac_diff) .& (ell .>= 2)
        if any(frac_valid)
            frac_png = joinpath(outdir, "fiducial_vs_unphysical_all_halos_D_ell_fractional_difference.png")
            fig = Plots.plot(
                ell[frac_valid],
                dell_frac_diff[frac_valid];
                xscale=:log10,
                linewidth=2,
                label="(unphysical - fiducial) / fiducial",
                xlabel="ell",
                ylabel="fractional D_ell difference",
                title="Full HalfDome all-halo D_ell fractional difference",
                legend=:best,
                size=(950, 620),
            )
            Plots.hline!(fig, [0.0]; linestyle=:dash, label="")
            Plots.savefig(fig, frac_png)
            println("Saved fractional-difference plot to $(abspath(frac_png))")
        end
    end

    return csv_path
end

fiducial = run_catalog_case("fiducial", FIDUCIAL_BATTAGLIA_PARAMS; enforce_guardrails=true)
unphysical = run_catalog_case("unphysical_requested", UNPHYSICAL_BATTAGLIA_PARAMS; enforce_guardrails=false)
write_comparison_outputs(fiducial, unphysical)
