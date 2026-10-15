# Load the fitted pressure family without altering existing maps or caches.
# Consult calibrated_model.json for support and residuals before painting.
using TOML
using XGPaint

function load_flamingo_pressure(parameter_file; h=0.68, Omega_b=0.049, Omega_c=0.261)
    coefficients = Dict(Symbol(key) => Float64(value)
                        for (key, value) in TOML.parsefile(parameter_file))
    return Battaglia16ThermalSZProfile(;
        h=Float64(h), Omega_b=Float64(Omega_b), Omega_c=Float64(Omega_c),
        coefficients...,
    )
end
