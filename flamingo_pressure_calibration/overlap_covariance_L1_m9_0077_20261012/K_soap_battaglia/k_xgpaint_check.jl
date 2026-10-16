# Test K, K5 parity: evaluate XGPaint's Battaglia profiles (validated checkout /home/kn18001/.julia/dev/XGPaint,
# branch 'cluster') at a few (M, z, x) with the published parameters and with test parameter sets of the refitted
# kinds (power laws for every parameter, free gamma, free x_c, BreakModel), and print everything as JSON for
# k_parity.py (which recomputes the same numbers with k_fit_core and compares them).
# Run (as the depot owner):  runuser -u kn18001 -- env JULIA_DEPOT_PATH=/home/kn18001/.julia \
#     /home/kn18001/.julia/juliaup/julia-1.12.2+0.x64.linux.gnu/bin/julia k_xgpaint_check.jl <cases.jl> <out.json>
# <cases.jl> (written by k_parity.py) defines `inp` = Dict("cases" => ..., "masses" => ..., "redshifts" => ...,
# "x3d" => ..., "xproj" => ...).  No JSON package is needed (the default environment has none as a direct dependency).
using XGPaint, Unitful, UnitfulAstro

const C = XGPaint.constants
include(ARGS[1])
out = Dict{String,Any}()

tojson(x::AbstractString) = "\"" * x * "\""
tojson(x::Real) = isfinite(x) ? repr(Float64(x)) : "null"
tojson(x::AbstractVector) = "[" * join(map(tojson, x), ",") * "]"
tojson(x::AbstractDict) = "{" * join(["\"$(k)\":" * tojson(v) for (k, v) in x], ",") * "}"

function y_model(pars)
    return XGPaint.Battaglia16ThermalSZProfile(; Omega_c=0.2589, Omega_b=0.0486, h=0.6774,
        P0_amp=pars["P0"][1], P0_alpha_m=pars["P0"][2], P0_alpha_z=pars["P0"][3],
        x_c_amp=pars["xc"][1], x_c_alpha_m=pars["xc"][2], x_c_alpha_z=pars["xc"][3],
        alpha_amp=pars["alpha"][1], alpha_alpha_m=pars["alpha"][2], alpha_alpha_z=pars["alpha"][3],
        beta_amp=pars["beta"][1], beta_alpha_m=pars["beta"][2], beta_alpha_z=pars["beta"][3],
        gamma_amp=pars["gamma"][1], gamma_alpha_m=pars["gamma"][2], gamma_alpha_z=pars["gamma"][3])
end
function break_model(pars, abrk, mbrk)
    return XGPaint.BreakModel(; Omega_c=0.2589, Omega_b=0.0486, h=0.6774, alpha_break=abrk, M_break=mbrk,
        P0_amp=pars["P0"][1], P0_alpha_m=pars["P0"][2], P0_alpha_z=pars["P0"][3],
        x_c_amp=pars["xc"][1], x_c_alpha_m=pars["xc"][2], x_c_alpha_z=pars["xc"][3],
        alpha_amp=pars["alpha"][1], alpha_alpha_m=pars["alpha"][2], alpha_alpha_z=pars["alpha"][3],
        beta_amp=pars["beta"][1], beta_alpha_m=pars["beta"][2], beta_alpha_z=pars["beta"][3],
        gamma_amp=pars["gamma"][1], gamma_alpha_m=pars["gamma"][2], gamma_alpha_z=pars["gamma"][3])
end
function tau_model(pars)
    return XGPaint.BattagliaTauProfilePhysical(; Omega_c=0.2589, Omega_b=0.0486, h=0.6774,
        P0_amp=pars["P0"][1], P0_alpha_m=pars["P0"][2], P0_alpha_z=pars["P0"][3],
        x_c_amp=pars["xc"][1], x_c_alpha_m=pars["xc"][2], x_c_alpha_z=pars["xc"][3],
        alpha_amp=pars["alpha"][1], alpha_alpha_m=pars["alpha"][2], alpha_alpha_z=pars["alpha"][3],
        beta_amp=pars["beta"][1], beta_alpha_m=pars["beta"][2], beta_alpha_z=pars["beta"][3],
        gamma_amp=pars["gamma"][1], gamma_alpha_m=pars["gamma"][2], gamma_alpha_z=pars["gamma"][3])
end

for (name, case) in pairs(inp["cases"])
    kind = String(case["kind"])
    pars = Dict(String(k) => Float64.(collect(v)) for (k, v) in pairs(case["pars"]))
    model = kind == "y" ? y_model(pars) : kind == "tau" ? tau_model(pars) : break_model(pars, Float64(case["abrk"]), Float64(case["mbrk"]))
    rows = Any[]
    for M in inp["masses"], z in inp["redshifts"]
        Mu = M * XGPaint.M_sun
        p = XGPaint.get_params(model, Mu, z)
        r200 = XGPaint.R_Δ(model, Mu, z, 200)                       # physical, with units
        rho_c = XGPaint.ρ_crit(model, z)                               # kg m^-3
        row = Dict{String,Any}("M" => M, "z" => z, "xc" => p.xc, "alpha" => p.α, "beta_xgp" => p.β, "gamma" => p.γ,
            "amp" => p.P₀, "R200_Mpc" => ustrip(uconvert(u"Mpc", r200)), "rho_crit_kg_m3" => ustrip(rho_c),
            "f_b" => model.f_b)
        xs = Float64.(collect(inp["x3d"]))
        row["gnfw3d"] = [XGPaint.generalized_nfw(x, p.xc, p.α, p.β, p.γ) for x in xs]
        xp = Float64.(collect(inp["xproj"]))
        row["los"] = [XGPaint._nfw_profile_los_quadrature(x, p.xc, p.α, p.β, p.γ) for x in xp]
        if kind == "tau"
            # rho_2d at physical radii r = x R200 (Msun-free: returns kg / m^2)
            row["rho2d_kg_m2"] = [ustrip(uconvert(u"kg/m^2", XGPaint.rho_2d(model, x * r200, Mu, z))) for x in xp]
        else
            # P_th(200) prefactor: G M 200 rho_c f_b / 2 (Pa m) and P_th_los / R200 at x (Pa)
            row["pth_prefactor_Pa_m"] = ustrip(uconvert(u"Pa*m", C.G * Mu * 200 * rho_c * model.f_b / 2))
            if kind == "break"
                row["break_factor"] = M < case["mbrk"] ? (M / case["mbrk"])^case["abrk"] : 1.0
            end
        end
        push!(rows, row)
    end
    out[String(name)] = rows
end
open(ARGS[2], "w") do io
    write(io, tojson(out))
end
println("wrote ", ARGS[2])
