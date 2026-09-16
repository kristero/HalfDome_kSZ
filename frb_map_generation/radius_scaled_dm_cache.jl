# Numerical DM interpolation outside XGPaint. Included after ProfileSupport.
# Interpolate only the dimensionless projected shape in (log(R/R200c), z,
# log10(M/Msun)); evaluate its physical amplitude and angular radius per halo.
using Unitful
using UnitfulAstro
using SHA
using Random

struct RadiusScaledDMCache{M,I}
    model::M
    itp::I
    logradius::Vector{Float64}
    redshifts::Vector{Float64}
    logmasses::Vector{Float64}
    spherical_cut_r200c::Float64
end

function radius_density_shape(model, mass, z)
    if model isa ProfileSupport.AbstractLee2022DMProfile
        p = ProfileSupport.lee2022_parameters(model, mass, z)
        return x -> ProfileSupport.lee2022_dimensionless_density(x, p.x_c, p.alpha, p.beta_prime, p.gamma)
    end
    p = XGPaint.get_params(model, mass * XGPaint.M_sun, z)
    # Use the installed Battaglia convention (its exponent is not Lee22's
    # -beta_prime). This is the same integrand used by XGPaint.ne2d.
    return x -> XGPaint.generalized_nfw(x, p.xc, p.α, p.β, p.γ)
end

"""Mean dimensionless density along the finite spherical chord.

Factor out its shrinking length before interpolation. The remaining quantity
has a finite, positive edge limit, so log interpolation cannot create a rim
spike or smear a zero boundary into the halo interior.
"""
function spherical_chord_mean(model, mass, z, x, cut)
    0 <= x <= cut || error("Radius outside spherical profile")
    density = radius_density_shape(model, mass, z)
    x == cut && return density(cut)
    length_half = sqrt(max(0.0, (cut-x)*(cut+x)))
    integrand(t) = density(sqrt(x*x + length_half^2*t*t))
    return Float64(ProfileSupport.lee2022_quadgk_function()(integrand, 0.0, 1.0; rtol=1e-8, order=9)[1])
end

function radius_column_amplitude(model, mass, z)
    r200 = XGPaint.R_Δ(model, mass * XGPaint.M_sun, z, 200)
    rho = XGPaint.ρ_crit(model, z)
    mp = XGPaint.constants.ProtonMass
    if model isa ProfileSupport.AbstractLee2022DMProfile
        p = ProfileSupport.lee2022_parameters(model, mass, z)
        column = p.n0 * 200 * rho / (model.hydrogen_mass_fraction * mp) *
                 (model.omega_b / model.omega_m) * r200
    else
        # Match the installed Battaglia16 physical profile exactly. These
        # abundance factors belong to that model, not a Lee22 renormalization.
        p = XGPaint.get_params(model, mass * XGPaint.M_sun, z)
        electron_mass = XGPaint.constants.ElectronMass +
            (2 * 0.76 / 1.76) * mp + ((1 - 0.76) / (2 * 1.76)) * 4 * mp
        column = 0.9 * p.P₀ * (0.049 / 0.31) * rho * r200 / electron_mass
    end
    return Float64(ustrip(u"pc*cm^-3", column))
end

function radius_direct_shape(model, physical_b16, mass, z, x, amplitude)
    if model isa ProfileSupport.AbstractLee2022DMProfile
        p = ProfileSupport.lee2022_parameters(model, mass, z)
        if model isa ProfileSupport.Lee2022ConcentrationDMProfile
            p.beta_prime - p.gamma > 1 || error(
                "Lee22 preferred fit has a divergent untruncated column at M=$(mass), z=$(z). " *
                "Select a physically specified outer boundary before painting this extrapolation.",
            )
        end
        return ProfileSupport.lee2022_dimensionless_los(x, p.x_c, p.alpha, p.beta_prime, p.gamma)
    end
    r200 = XGPaint.R_Δ(physical_b16, mass * XGPaint.M_sun, z, 200)
    return Float64(ustrip(u"pc*cm^-3",
        XGPaint.ne2d(physical_b16, x * r200, mass * XGPaint.M_sun, z))) / amplitude
end

function radius_cache_signature(runtime, refinement, zmax, spherical_cut)
    # Plain arrays avoid Julia/JLD2 reconstructed-type portability failures.
    # A content signature prevents reuse after changing any local density
    # coefficients, numerical implementation, cosmology, or installed XGPaint.
    sources = [@__FILE__, joinpath(@__DIR__, "lee2022_frb_dm_profile.jl")]
    for (root, _, files) in walkdir(dirname(pathof(XGPaint)))
        append!(sources, [joinpath(root, f) for f in files if endswith(f, ".jl")])
    end
    hashes = [bytes2hex(sha256(read(path))) for path in sort(sources)]
    return join(vcat(["radius_shape_v1", runtime.generated_model_family,
        runtime.cache_signature, repr(runtime.model), string(refinement), string(zmax), string(spherical_cut)], hashes), "|")
end

function build_radius_scaled_cache(runtime, path; refinement=2, zmax=1.0, spherical_cut=0.0)
    refinement in (1, 2) || error("Use radius refinement 1 or 2")
    zmax > 0 || error("Radius cache zmax must be positive")
    isfinite(spherical_cut) && spherical_cut >= 0 || error("Invalid spherical cutoff")
    signature = radius_cache_signature(runtime, refinement, zmax, spherical_cut)
    model = runtime.model
    if isfile(path)
        logx, zs, logm, values = h5open(path, "r") do h5
            read(attributes(h5)["signature"]) == signature || error(
                "Radius cache signature mismatch: $(path). Use a new output directory.")
            (read(h5["log_radius_r200c"]), read(h5["redshift"]),
             read(h5["log10_mass_msun"]), read(h5["log_dimensionless_shape"]))
        end
    else
        xmax = spherical_cut > 0 ? spherical_cut : 5.1
        logx = collect(range(log(1e-7), log(xmax); length=160*refinement+1))
        zs = collect(range(0.0, zmax; length=32*refinement+1))
        logm = sort!(unique(vcat(collect(range(12.0, 15.7; length=64*refinement+1)),
            [log10(10.0^13.61 / model.cosmo.h), log10(10.0^13.75 / model.cosmo.h)])))
        values = Array{Float64}(undef, length(logx), length(zs), length(logm))
        physical_b16 = XGPaint.BattagliaTauProfilePhysical(Omega_c=0.261, Omega_b=0.049, h=0.68)
        pairs = collect(Iterators.product(eachindex(zs), eachindex(logm)))
        println("Building dimensionless radius cache ", size(values)); flush(stdout)
        Threads.@threads :static for k in eachindex(pairs)
            iz, im = pairs[k]
            mass, z = 10.0^logm[im], zs[iz]
            amplitude = radius_column_amplitude(model, mass, z)
            for ix in eachindex(logx)
                x = min(exp(logx[ix]), xmax)
                shape = spherical_cut > 0 ? spherical_chord_mean(model, mass, z, x, spherical_cut) :
                    radius_direct_shape(model, physical_b16, mass, z, x, amplitude)
                isfinite(shape) && shape > 0 || error("Invalid dimensionless column")
                values[ix, iz, im] = log(shape)
            end
        end
        mkpath(dirname(path))
        h5open(path, "w") do h5
            h5["log_radius_r200c"] = logx
            h5["redshift"] = zs
            h5["log10_mass_msun"] = logm
            h5["log_dimensionless_shape"] = values
            attributes(h5)["signature"] = signature
            attributes(h5)["axes_in_h5py"] = "logmass,redshift,logradius"
            attributes(h5)["normalization_changed"] = false
            attributes(h5)["spherical_cut_r200c"] = spherical_cut
            attributes(h5)["aperture_kind"] = spherical_cut > 0 ?
                "finite spherical LOS; chord length factored out" : "projected footprint; original LOS unchanged"
        end
    end
    size(values) == (length(logx), length(zs), length(logm)) || error("Invalid radius table dimensions")
    all(isfinite, values) || error("Non-finite radius table")
    for axis in (logx, zs, logm)
        all(diff(axis) .> 0) || error("Radius cache axes must strictly increase")
    end
    itp = Interpolations.interpolate((logx, zs, logm), values,
                                    Interpolations.Gridded(Interpolations.Linear()))
    return RadiusScaledDMCache(model, itp, logx, zs, logm, Float64(spherical_cut))
end

"""Prepare dimensional factors once per halo, not once per painted pixel."""
function prepare_painted_halo(cache::RadiusScaledDMCache, mass, z, aperture)
    first(cache.redshifts) <= z <= last(cache.redshifts) || error("Redshift outside radius cache")
    logmass = log10(mass)
    first(cache.logmasses) <= logmass <= last(cache.logmasses) || error("Mass outside radius cache")
    r200 = XGPaint.R_Δ(cache.model, mass * XGPaint.M_sun, z, 200)
    theta200 = Float64(XGPaint.angular_size(cache.model, r200, z))
    theta_max = ProfileSupport.compute_theta_max_r200c_external(cache.model, mass, z, aperture)
    cut = cache.spherical_cut_r200c
    cut > 0 && aperture != cut && error("Spherical cutoff must equal the painted aperture in this test")
    cut == 0 && log(theta_max / theta200) > last(cache.logradius) && error("Aperture outside radius cache")
    amplitude = radius_column_amplitude(cache.model, mass, z) / (1 + z)
    function at_angle(theta)
        # Spherical tests use the screen-plane impact radius, consistent with
        # XGPaint's atan(R/DA) footprint. The projected baseline deliberately
        # keeps theta/theta200 so that its old geometry is unchanged.
        x = cut > 0 ? tan(theta) / tan(theta200) : theta / theta200
        cut > 0 && x >= cut && return 0.0
        # The projected central column is finite. Only its tiny unresolved
        # central-radius limit uses x=1e-7; this is not a minimum DM cutoff.
        lx = log(max(x, exp(first(cache.logradius))))
        lx <= last(cache.logradius) || error("Radius outside cache; extrapolation disabled")
        chord = cut > 0 ? 2sqrt(max(0.0, (cut-x)*(cut+x))) : 1.0
        return amplitude * chord * exp(cache.itp(lx, z, logmass))
    end
    return (theta_max=theta_max, value=at_angle)
end

function validate_radius_cache(cache; output_path, sample_count=600, tolerance=0.01)
    rng = MersenneTwister(20260913)
    worst = 0.0
    open(output_path, "w") do io
        println(io, "mass_msun,redshift,radius_r200c,direct_dm,cached_dm,relative_error")
        for i in 1:sample_count
            mass = 10.0^(12.86 + rand(rng)*(15.59-12.86))
            z = 10.0^(log10(0.00215)*(1-rand(rng))) * last(cache.redshifts)
            cut = cache.spherical_cut_r200c
            aperture = cut > 0 ? cut : 5.0
            x = exp(log(1e-6)+rand(rng)*(log(aperture)-log(1e-6)))
            # Include near-edge points where a naive log-column grid fails.
            cut > 0 && i % 5 == 0 && (x = cut*(1-10.0^(-rand(rng)*7-1)))
            prepared = prepare_painted_halo(cache, mass, z, aperture)
            r200 = XGPaint.R_Δ(cache.model, mass * XGPaint.M_sun, z, 200)
            theta200 = XGPaint.angular_size(cache.model, r200, z)
            theta = cut > 0 ? atan(x*tan(theta200)) : x*theta200
            if cut > 0
                # Independent LOS integration in the physical chord coordinate.
                density = radius_density_shape(cache.model, mass, z)
                half_chord = sqrt((cut-x)*(cut+x))
                los = 2*ProfileSupport.lee2022_quadgk_function()(
                    l -> density(sqrt(x*x+l*l)), 0.0, half_chord; rtol=1e-9, order=9)[1]
                direct = radius_column_amplitude(cache.model,mass,z)/(1+z)*los
            else
                direct = Float64(cache.model(theta, mass, z))
            end
            cached = prepared.value(theta)
            error_value = cached/direct - 1
            isfinite(error_value) || error("Non-finite cache validation error")
            worst = max(worst, abs(error_value))
            println(io, join((mass, z, x, direct, cached, error_value), ','))
        end
    end
    worst <= tolerance || error("Radius-cache validation failed: maximum relative error $(worst)")
    println("PASS radius cache: $(sample_count) independent points, maximum relative error=$(worst)")
    return worst
end
