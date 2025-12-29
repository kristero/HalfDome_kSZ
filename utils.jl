# utils.jl

function xyz_to_ra_dec(x::AbstractVector{T}, y::AbstractVector{T}, z::AbstractVector{T}) where T
    @assert length(x) == length(y) == length(z)

    N   = length(x)
    ra  = Array{T}(undef, N)
    dec = Array{T}(undef, N)

    for i in 1:N
        r = sqrt(x[i]^2 + y[i]^2 + z[i]^2)
        vx, vy, vz = x[i]/r, y[i]/r, z[i]/r

        θ, ϕ = Healpix.vec2ang(vx, vy, vz)   # θ: colatitude, ϕ: longitude
        dec[i] = T(π)/2 - θ
        ra[i]  = ϕ
    end

    return ra, dec
end


function halo_field_weights(halo_mass; logM0=13, k=4.0)
    logM = log10.(halo_mass)
    σ(x; x0, k) = 1 / (1 + exp(-k*(x - x0)))
    # halo weight
    w_halo  = σ.(logM; x0=logM0, k=k)
    # complementary field weight
    w_field = 1 .- w_halo                  
    return w_halo, w_field
end


# ---------- radial (r200c, θ200c) helpers ----------

"""
    rho_crit_z(z; Ωm=0.3, ΩΛ=0.7)

Critical density ρ_c(z) in Msun/h / (Mpc/h)^3.
"""
rho_crit_z(z; Ωm=0.3, ΩΛ=0.7) = 2.775e11 * (Ωm*(1+z)^3 + ΩΛ)


"""
    r200c_mph(M200c, z; Ωm=0.3, ΩΛ=0.7)

r_200c in (Mpc/h) given M_200c in Msun/h.
"""
function r200c_mph(M200c, z; Ωm=0.3, ΩΛ=0.7)
    ρc = rho_crit_z(z; Ωm=Ωm, ΩΛ=ΩΛ)
    return (3M200c / (4π*200*ρc))^(1/3)
end


"""
    theta200c_all(halo_mass, redshift, z2r; Ωm=0.3, ΩΛ=0.7)

Vectorised θ_200c for each halo in radians.

- halo_mass :: M200c in Msun/h
- redshift  :: z
- z2r       :: comoving distance interpolator (Mpc/h),
              e.g. XGPaint.build_z2r_interpolator(...)
"""
function theta200c_all(halo_mass::AbstractVector,
                       redshift::AbstractVector,
                       z2r;
                       Ωm=0.3, ΩΛ=0.7)
    @assert length(halo_mass) == length(redshift)

    N      = length(halo_mass)
    θ200   = Array{Float64}(undef, N)

    for i in 1:N
        M  = halo_mass[i]
        z  = redshift[i]
        r200  = r200c_mph(M, z; Ωm=Ωm, ΩΛ=ΩΛ)  # Mpc/h
        r_com = z2r(z)                          # Mpc/h
        D_A   = r_com / (1 + z)
        θ200[i] = r200 / D_A                    # radians
    end

    return θ200
end


"""
    split_halo_field_mass_theta(halo_mass, θ200_arcmin;
                                mass_min=1e13, θmin_arcmin=0.5)

Return (sel_halo, sel_field) Boolean masks using BOTH:

- M_200c >= mass_min
- θ_200c >= θmin_arcmin

This mimics the WebSky-style selection.
"""
function split_halo_field_mass_theta(halo_mass,
                                     θ200_arcmin;
                                     mass_min=1e13,
                                     θmin_arcmin=0.5)

    @assert length(halo_mass) == length(θ200_arcmin)

    sel_halo  = (halo_mass .>= mass_min) .& (θ200_arcmin .>= θmin_arcmin)
    sel_field = .!sel_halo
    return sel_halo, sel_field
end


# --------------------------------------------------
# Sigmoid + smooth radial mixing helpers
# --------------------------------------------------

"""
    sigmoid(x; x0=0.0, k=1.0)

Standard logistic sigmoid:
    σ(x) = 1 / (1 + exp(-k * (x - x0)))
"""
sigmoid(x; x0=0.0, k=1.0) = 1.0 / (1.0 + exp(-k * (x - x0)))


"""
    radial_weight(θ; θ0, Δθ_smooth)

Radial sigmoid in angle (radians). Returns a weight in [0,1]:

- w ≈ 1  for θ << θ0   (inner / "halo" dominated)
- w ≈ 0  for θ >> θ0   (outer / "field" dominated)
"""
function radial_weight(θ; θ0, Δθ_smooth)
    # sign chosen so that small θ -> w ~ 1, large θ -> w ~ 0
    return 1.0 / (1.0 + exp((θ - θ0)/Δθ_smooth))
end


"""
    mix_radial_profiles(θ, prof_halo, prof_field; θ0, Δθ_smooth)

Given:
- θ           :: vector of angular radii [radians]
- prof_halo   :: 1-halo profile evaluated on θ
- prof_field  :: 2-halo/field profile evaluated on θ

return mixed profile:
    prof_mix(θ) = w(θ) * prof_halo(θ) + (1 - w(θ)) * prof_field(θ)
with w(θ) a radial sigmoid.
"""
function mix_radial_profiles(θ::AbstractVector,
                             prof_halo::AbstractVector,
                             prof_field::AbstractVector;
                             θ0,
                             Δθ_smooth)

    @assert length(θ) == length(prof_halo) == length(prof_field)

    prof_mix = similar(prof_halo)
    @inbounds for i in eachindex(θ)
        w = radial_weight(θ[i]; θ0=θ0, Δθ_smooth=Δθ_smooth)
        prof_mix[i] = w * prof_halo[i] + (1.0 - w) * prof_field[i]
    end
    return prof_mix
end