if !haskey(ENV, "GKSwstype")
    ENV["GKSwstype"] = "png"
end
if !haskey(ENV, "GKS_WSTYPE")
    ENV["GKS_WSTYPE"] = "png"
end

# XGPaint isolated-halo FRB DM profile plots.
#
# This intentionally uses the same FRB-DM model path as the map code:
#     HaloDMProfile(BattagliaTauProfile(...))
#
# Important: do not pass tSZ/pressure Battaglia parameters here. HaloDMProfile
# already owns the electron-density/DM normalization in the installed XGPaint
# version. This script only overrides x_c_amp in the x_c-variation panel.
#
# Outputs:
#   1. XGPaint Battaglia 3D electron-density profile for one halo.
#   2. XGPaint HaloDMProfile DM(R_perp) for logM = 12, 13, 14, 15.
#   3. The same DM(R_perp), varying the Battaglia tau-profile x_c parameter.

using XGPaint
using Unitful
using UnitfulAstro
using Plots
using DelimitedFiles

const H_VALUE = 0.68
const OMEGAB = 0.049
const OMEGAC = 0.31 - OMEGAB
const OMEGAM = OMEGAB + OMEGAC
const ELECTRON_MASS = 9.1093837015e-28u"g"
const PROTON_MASS = 1.67262192369e-24u"g"
const GRAVITATIONAL_CONSTANT = 6.67430e-11u"m^3/kg/s^2"

const DEFAULT_X_C_AMP = 0.5
const DEFAULT_X_C_LOW = 0.150011
const DEFAULT_X_C_HIGH = 0.844503

const compute_theta_max_local =
    isdefined(XGPaint, Symbol("compute_", Char(0x03b8), "max")) ?
    getfield(XGPaint, Symbol("compute_", Char(0x03b8), "max")) :
    error("XGPaint does not define compute_theta_max.")

const xg_R_delta =
    isdefined(XGPaint, Symbol("R_", Char(0x0394))) ?
    getfield(XGPaint, Symbol("R_", Char(0x0394))) :
    error("XGPaint does not define R_delta.")

const SYM_ALPHA = Symbol(Char(0x03b1))
const SYM_BETA = Symbol(Char(0x03b2))
const SYM_GAMMA = Symbol(Char(0x03b3))
const SYM_P0 = Symbol("P", Char(0x2080))

function get_string_arg(key, default; env=nothing)
    if env !== nothing
        env_names = env isa AbstractString ? (env,) : env
        for env_name in env_names
            if haskey(ENV, env_name)
                return String(ENV[env_name])
            end
        end
    end
    prefix1 = "--" * key * "="
    prefix2 = key * "="
    for arg in ARGS
        if startswith(arg, prefix1)
            return String(split(arg, "=", limit=2)[2])
        elseif startswith(arg, prefix2)
            return String(split(arg, "=", limit=2)[2])
        end
    end
    return String(default)
end

function get_float_arg(key, default; env=nothing)
    value = get_string_arg(key, ""; env=env)
    isempty(value) || return parse(Float64, value)
    return Float64(default)
end

function get_int_arg(key, default; env=nothing)
    value = get_string_arg(key, ""; env=env)
    isempty(value) || return parse(Int, value)
    return Int(default)
end

function parse_float_list(text::AbstractString)
    values = Float64[]
    for token in split(text, ",")
        stripped = strip(token)
        isempty(stripped) && continue
        push!(values, parse(Float64, stripped))
    end
    isempty(values) && error("Need at least one value in $(repr(text)).")
    return values
end

function halo_dm_constructor()
    if isdefined(Main, :HaloDMProfile)
        return getfield(Main, :HaloDMProfile)
    elseif isdefined(XGPaint, :HaloDMProfile)
        return getfield(XGPaint, :HaloDMProfile)
    end
    error("HaloDMProfile is not available in this Julia/XGPaint environment.")
end

function make_dm_model(; x_c_amp=nothing)
    constructor = halo_dm_constructor()
    if x_c_amp === nothing
        tau_model = XGPaint.BattagliaTauProfile(Omega_c=OMEGAC, Omega_b=OMEGAB, h=H_VALUE)
        return constructor(tau_model)
    end
    return constructor(
        Omega_c=OMEGAC,
        Omega_b=OMEGAB,
        h=H_VALUE,
        x_c_amp=Float64(x_c_amp),
    )
end

function r200_kpc(tau_model, mass_msun::Real, z::Real)
    r200 = xg_R_delta(tau_model, Float64(mass_msun) * XGPaint.M_sun, Float64(z), 200)
    return Float64(ustrip(u"kpc", r200))
end

function rho_crit_hminus2_g_cm3(z::Real)
    h100 = 100.0u"km/s/Mpc"
    e2 = OMEGAM * (1.0 + Float64(z))^3 + (1.0 - OMEGAM)
    rho = 3.0 * h100^2 * e2 / (8.0 * pi * GRAVITATIONAL_CONSTANT)
    return uconvert(u"g/cm^3", rho)
end

function xgpaint_ne3d_cm3(tau_model, r_kpc::Real, mass_msun::Real, z::Real)
    mass_unitful = Float64(mass_msun) * XGPaint.M_sun
    r200 = xg_R_delta(tau_model, mass_unitful, Float64(z), 200)
    r200_kpc_value = Float64(ustrip(u"kpc", r200))
    x = max(Float64(r_kpc) / r200_kpc_value, eps(Float64))

    par = XGPaint.get_params(tau_model, mass_unitful, Float64(z))
    rho = getproperty(par, SYM_P0) * XGPaint.generalized_nfw(
        x,
        par.xc,
        getproperty(par, SYM_ALPHA),
        getproperty(par, SYM_BETA),
        getproperty(par, SYM_GAMMA),
    ) *
        rho_crit_hminus2_g_cm3(Float64(z))

    xH = 0.76
    nH_ne = 2 * xH / (xH + 1)
    nHe_ne = (1 - xH) / (2 * (1 + xH))
    factor = (ELECTRON_MASS + nH_ne * PROTON_MASS +
        nHe_ne * 4 * PROTON_MASS) / tau_model.cosmo.h^2

    ne = rho / factor
    return Float64(ustrip(u"cm^-3", ne))
end

function xgpaint_dm_pc_cm3(dm_model, tau_model, rperp_kpc::Real, mass_msun::Real, z::Real)
    theta200 = Float64(compute_theta_max_local(
        dm_model,
        Float64(mass_msun) * XGPaint.M_sun,
        Float64(z);
        mult=1,
    ))
    theta = max(Float64(rperp_kpc) / r200_kpc(tau_model, mass_msun, z) * theta200, eps(Float64))
    value = dm_model(theta, Float64(mass_msun), Float64(z))
    return Float64(value)
end

function save_table(path::AbstractString, header::Vector{String}, rows)
    parent = dirname(path)
    isempty(parent) || isdir(parent) || mkpath(parent)
    open(path, "w") do io
        println(io, join(header, ","))
        for row in rows
            println(io, join(row, ","))
        end
    end
    return path
end

function make_plots()
    output_dir = get_string_arg(
        "output_dir",
        joinpath(@__DIR__, "outputs", "xgpaint_isolated_halo_dm_profiles");
        env="XGPAINT_FRB_PROFILE_OUTPUT_DIR",
    )
    isdir(output_dir) || mkpath(output_dir)

    z = get_float_arg("redshift", 0.05)
    fiducial_mass = get_float_arg("fiducial_mass", 1.5e12)
    log_masses = parse_float_list(get_string_arg("log_masses", "12,13,14,15"))
    curve_points = get_int_arg("curve_points", 120)
    marker_points = get_int_arg("marker_points", 10)
    rmin_fraction = get_float_arg("rmin_fraction", 0.01)
    rmax_fraction = get_float_arg("rmax_fraction", 0.98)

    xc_low = get_float_arg("xc_low", DEFAULT_X_C_LOW)
    xc_fid = get_float_arg("xc_fid", DEFAULT_X_C_AMP)
    xc_high = get_float_arg("xc_high", DEFAULT_X_C_HIGH)
    xc_values = [xc_low, xc_fid, xc_high]

    default_dm = make_dm_model()
    default_tau = default_dm

    default(fontfamily="Computer Modern", linewidth=2, framestyle=:box)

    # Figure 1: XGPaint 3D electron density profile for one isolated halo.
    r200_fid = r200_kpc(default_tau, fiducial_mass, z)
    r_ne = 10 .^ range(log10(0.005 * r200_fid), log10(r200_fid); length=300)
    ne = [xgpaint_ne3d_cm3(default_tau, r, fiducial_mass, z) for r in r_ne]
    p1 = plot(
        r_ne,
        ne;
        xscale=:log10,
        yscale=:log10,
        xlabel="r [kpc]",
        ylabel="n_e(r) [cm^-3]",
        label="XGPaint HaloDMProfile",
        title="Isolated halo: log10(M200c)=$(round(log10(fiducial_mass); digits=2)), z=$(z)",
        grid=true,
        size=(760, 560),
    )
    vline!(p1, [r200_fid]; color=:black, linestyle=:dash, linewidth=1, label="r200c")
    savefig(p1, joinpath(output_dir, "figure1_xgpaint_ne_profile.png"))
    save_table(
        joinpath(output_dir, "figure1_xgpaint_ne_profile.csv"),
        ["r_kpc", "ne_cm3", "mass_msun", "redshift", "r200_kpc"],
        ([r_ne[i], ne[i], fiducial_mass, z, r200_fid] for i in eachindex(r_ne)),
    )

    # Figure 2: XGPaint DM(R_perp) for several halo masses.
    p2 = plot(
        xlabel="R_perp [kpc]",
        ylabel="DM(R_perp) [pc cm^-3]",
        xscale=:log10,
        yscale=:log10,
        title="Direct XGPaint HaloDMProfile projection",
        grid=true,
        size=(800, 580),
    )
    rows2 = Vector{Any}[]
    palette2 = palette(:viridis, length(log_masses))
    for (i, logM) in enumerate(log_masses)
        mass = 10.0^logM
        r200 = r200_kpc(default_tau, mass, z)
        r_curve = 10 .^ range(log10(rmin_fraction * r200), log10(rmax_fraction * r200); length=curve_points)
        dm_curve = [
            xgpaint_dm_pc_cm3(default_dm, default_tau, r, mass, z)
            for r in r_curve
        ]
        plot!(
            p2,
            r_curve,
            dm_curve;
            color=palette2[i],
            label="logM=$(logM), r200=$(round(r200; digits=1)) kpc",
        )

        r_mark = range(0.05 * r200, rmax_fraction * r200; length=marker_points)
        dm_mark = [
            xgpaint_dm_pc_cm3(default_dm, default_tau, r, mass, z)
            for r in r_mark
        ]
        scatter!(p2, r_mark, dm_mark; color=palette2[i], label=false, markersize=4)

        append!(
            rows2,
            ([logM, mass, r_curve[j], dm_curve[j], r200, z] for j in eachindex(r_curve)),
        )
    end
    savefig(p2, joinpath(output_dir, "figure2_xgpaint_dm_vs_impact_mass.png"))
    save_table(
        joinpath(output_dir, "figure2_xgpaint_dm_vs_impact_mass.csv"),
        ["logM", "mass_msun", "Rperp_kpc", "DM_pc_cm3", "r200_kpc", "redshift"],
        rows2,
    )

    # Figure 3: vary x_c for the same masses.
    layout_rows = ceil(Int, length(log_masses) / 2)
    p3 = plot(layout=(layout_rows, 2), size=(980, 780))
    colors = [:blue, :black, :red]
    rows3 = Vector{Any}[]
    for (panel, logM) in enumerate(log_masses)
        mass = 10.0^logM
        for (ixc, xc) in enumerate(xc_values)
            dm = make_dm_model(x_c_amp=xc)
            tau = dm
            r200 = r200_kpc(tau, mass, z)
            r_curve = 10 .^ range(log10(rmin_fraction * r200), log10(rmax_fraction * r200); length=curve_points)
            dm_curve = [
                xgpaint_dm_pc_cm3(dm, tau, r, mass, z)
                for r in r_curve
            ]
            println(
                "x_c sanity: logM=$(logM), x_c=$(xc), " *
                "DM_min=$(minimum(dm_curve)), DM_max=$(maximum(dm_curve))"
            )
            plot!(
                p3,
                r_curve,
                dm_curve;
                subplot=panel,
                xscale=:log10,
                yscale=:log10,
                xlabel="R_perp [kpc]",
                ylabel="DM [pc cm^-3]",
                title="logM=$(logM)",
                label="x_c=$(round(xc; digits=4))",
                color=colors[ixc],
                grid=true,
            )
            append!(
                rows3,
                ([logM, mass, xc, r_curve[j], dm_curve[j], r200, z] for j in eachindex(r_curve)),
            )
        end
    end
    savefig(p3, joinpath(output_dir, "figure3_xgpaint_dm_vs_impact_xc_variation.png"))
    save_table(
        joinpath(output_dir, "figure3_xgpaint_dm_vs_impact_xc_variation.csv"),
        ["logM", "mass_msun", "x_c", "Rperp_kpc", "DM_pc_cm3", "r200_kpc", "redshift"],
        rows3,
    )

    println("Wrote XGPaint isolated-halo plots to $(abspath(output_dir))")
    println("  figure1_xgpaint_ne_profile.png")
    println("  figure2_xgpaint_dm_vs_impact_mass.png")
    println("  figure3_xgpaint_dm_vs_impact_xc_variation.png")
    println("Also wrote matching CSV tables.")
end

make_plots()
