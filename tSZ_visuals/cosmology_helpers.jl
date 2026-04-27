function make_z_of_chi_itp(; omegam, h_value, z1=0.0, z2=6.0, nz=100_000)
    H0_local = 100.0 * h_value
    H(z) = H0_local * sqrt(omegam * (1 + z)^3 + 1 - omegam)
    dchidz(z) = TSZ_C_KMS / H(z)

    za = collect(range(z1, z2; length=nz))
    dz = za[2] - za[1]
    chia = similar(za)

    chia[1] = 0.0
    s = 0.0
    @inbounds for i in 2:length(za)
        s += 0.5 * (dchidz(za[i - 1]) + dchidz(za[i])) * dz
        chia[i] = s
    end

    return linear_interpolation(chia, za; extrapolation_bc=Line())
end

function make_chi_of_z_itp(; omegam, h_value, z1=0.0, z2=6.0, nz=100_000)
    H0_local = 100.0 * h_value
    H(z) = H0_local * sqrt(omegam * (1 + z)^3 + 1 - omegam)
    dchidz(z) = TSZ_C_KMS / H(z)

    za = collect(range(z1, z2; length=nz))
    dz = za[2] - za[1]
    chia = similar(za)

    chia[1] = 0.0
    s = 0.0
    @inbounds for i in 2:length(za)
        s += 0.5 * (dchidz(za[i - 1]) + dchidz(za[i])) * dz
        chia[i] = s
    end

    return linear_interpolation(za, chia; extrapolation_bc=Line())
end

@inline function m200m_to_m200c_scalar(m200m::Float64, z::Float64)
    one_plus_z = 1.0 + z
    ez_num = TSZ_OMEGAM * one_plus_z^3
    omegamz = ez_num / (ez_num + 1.0 - TSZ_OMEGAM)
    return m200m * omegamz^0.35
end

function compute_redshift_and_mass(x, y, z, radius, itp_z_of_chi, rho_m)
    n = length(x)
    redshift = Vector{Float64}(undef, n)
    halo_mass = Vector{Float64}(undef, n)
    mass_prefactor = (4.0 * pi / 3.0) * rho_m

    @threads for i in 1:n
        xi = Float64(x[i])
        yi = Float64(y[i])
        zi = Float64(z[i])
        ri = Float64(radius[i])

        chi = sqrt(xi * xi + yi * yi + zi * zi)
        zi_redshift = itp_z_of_chi(chi)

        redshift[i] = zi_redshift
        halo_mass[i] = m200m_to_m200c_scalar(mass_prefactor * ri^3, zi_redshift)
    end

    return redshift, halo_mass
end

function xyz_to_ra_dec_threaded(x::AbstractVector{T}, y::AbstractVector{T}, z::AbstractVector{T}) where {T}
    length(x) == length(y) == length(z) || error("x, y, z must have the same length.")

    n = length(x)
    ra = Vector{T}(undef, n)
    dec = Vector{T}(undef, n)

    @threads for i in 1:n
        r = sqrt(x[i]^2 + y[i]^2 + z[i]^2)
        vx = x[i] / r
        vy = y[i] / r
        vz = z[i] / r

        theta, phi = Healpix.vec2ang(vx, vy, vz)
        dec[i] = T(pi) / 2 - theta
        ra[i] = phi
    end

    return ra, dec
end
