using XGPaint
using HDF5
using Healpix
using Unitful: ustrip
using Base.Threads

const h_value = 0.68
const omegab = 0.049
const omegac = 0.31 - omegab

function get_string_arg(key, default)
    prefix1 = "--" * key * "="
    prefix2 = key * "="
    for a in ARGS
        if startswith(a, prefix1)
            return String(split(a, "=", limit=2)[2])
        elseif startswith(a, prefix2)
            return String(split(a, "=", limit=2)[2])
        end
    end
    return String(default)
end

function get_float_arg(key, default)
    prefix1 = "--" * key * "="
    prefix2 = key * "="
    for a in ARGS
        if startswith(a, prefix1)
            return parse(Float64, split(a, "=", limit=2)[2])
        elseif startswith(a, prefix2)
            return parse(Float64, split(a, "=", limit=2)[2])
        end
    end
    return Float64(default)
end

function parse_bool_arg(value)
    value_norm = lowercase(strip(String(value)))
    if value_norm in ("1", "true", "t", "yes", "y", "on")
        return true
    elseif value_norm in ("0", "false", "f", "no", "n", "off")
        return false
    end
    error("Could not parse boolean value $(repr(value)).")
end

function get_bool_arg(key, default)
    prefix1 = "--" * key * "="
    prefix2 = key * "="
    for a in ARGS
        if startswith(a, prefix1)
            return parse_bool_arg(split(a, "=", limit=2)[2])
        elseif startswith(a, prefix2)
            return parse_bool_arg(split(a, "=", limit=2)[2])
        end
    end
    return Bool(default)
end

function compute_theta_min_local(model)
    if hasproperty(model, :itp)
        itp = getproperty(model, :itp)
        if hasproperty(itp, :ranges)
            return exp(Float64(first(first(getproperty(itp, :ranges)))))
        end
    end
    return eps(Float64)
end

function theta200c_rad(profile_model, mass_msun::Float64, redshift::Float64)
    r200 = XGPaint.r200c_comoving(profile_model, mass_msun, redshift)
    chi_ang = XGPaint.angular_diameter_dist(profile_model.cosmo, redshift)
    if !isfinite(ustrip(r200)) || !isfinite(ustrip(chi_ang)) || ustrip(chi_ang) <= 0.0
        return NaN
    end
    return Float64(ustrip(r200 / chi_ang))
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

function sample_healpix_map_at_points(m::HealpixMap{<:Real, RingOrder}, ras, decs)
    length(ras) == length(decs) || error("ras and decs must have the same length.")
    sampled_values = Vector{Float64}(undef, length(ras))
    res = m.resolution

    @inbounds for i in eachindex(ras)
        theta = pi / 2 - decs[i]
        phi = ras[i]
        pix = Healpix.ang2pixRing(res, theta, phi)
        sampled_values[i] = Float64(m.pixels[pix])
    end

    return sampled_values
end

function build_frb_overdensity_map(sample_x, sample_y, sample_z, nside::Integer)
    length(sample_x) == length(sample_y) == length(sample_z) || error(
        "sample_x, sample_y, and sample_z must have the same length."
    )

    frb_map = HealpixMap{Float64, RingOrder}(nside)
    fill!(frb_map.pixels, 0.0)
    res = Healpix.Resolution(nside)

    @inbounds for i in eachindex(sample_x)
        r = sqrt(sample_x[i]^2 + sample_y[i]^2 + sample_z[i]^2)
        vx = sample_x[i] / r
        vy = sample_y[i] / r
        vz = sample_z[i] / r
        theta, phi = Healpix.vec2ang(vx, vy, vz)
        pix = Healpix.ang2pixRing(res, theta, phi)
        frb_map.pixels[pix] += 1.0
    end

    mean_counts = length(sample_x) / length(frb_map.pixels)
    mean_counts > 0.0 || error("Cannot build an FRB overdensity map with zero samples.")
    frb_map.pixels ./= mean_counts
    frb_map.pixels .-= 1.0

    return frb_map
end

function offset_ra_dec_along_meridian(ra::Float64, dec::Float64, theta_offset::Float64)
    if !isfinite(theta_offset) || theta_offset <= 0.0
        return ra, dec
    end

    cosdec = cos(dec)
    ux = cosdec * cos(ra)
    uy = cosdec * sin(ra)
    uz = sin(dec)

    tx = -sin(dec) * cos(ra)
    ty = -sin(dec) * sin(ra)
    tz = cos(dec)

    vx = cos(theta_offset) * ux + sin(theta_offset) * tx
    vy = cos(theta_offset) * uy + sin(theta_offset) * ty
    vz = cos(theta_offset) * uz + sin(theta_offset) * tz
    vnorm = sqrt(vx^2 + vy^2 + vz^2)
    vx /= vnorm
    vy /= vnorm
    vz /= vnorm

    theta, phi = Healpix.vec2ang(vx, vy, vz)
    return Float64(phi), Float64(pi / 2 - theta)
end

function offset_ra_dec_arrays(ras, decs, theta_offsets)
    length(ras) == length(decs) == length(theta_offsets) || error("ras, decs, and theta_offsets must have the same length.")
    offset_ras = Vector{Float64}(undef, length(ras))
    offset_decs = Vector{Float64}(undef, length(ras))

    @threads for i in eachindex(ras)
        offset_ras[i], offset_decs[i] = offset_ra_dec_along_meridian(Float64(ras[i]), Float64(decs[i]), Float64(theta_offsets[i]))
    end

    return offset_ras, offset_decs
end

function write_csv_table(
    output_path::AbstractString,
    sample_mass,
    sample_redshift,
    theta200c,
    theta_eval,
    dm_xgpaint
)
    output_dir = dirname(output_path)
    isdir(output_dir) || mkpath(output_dir)

    open(output_path, "w") do io
        println(io, "sample_index,sample_mass,sample_redshift,theta200c_rad,theta_eval_rad,dm_xgpaint")
        @inbounds for i in eachindex(sample_mass)
            println(
                io,
                string(
                    i, ",",
                    sample_mass[i], ",",
                    sample_redshift[i], ",",
                    theta200c[i], ",",
                    theta_eval[i], ",",
                    dm_xgpaint[i]
                )
            )
        end
    end

    return output_path
end

catalog_path = get_string_arg("catalog_path", "")
output_path = get_string_arg("output_path", "")
dm_cache_file = get_string_arg("dm_cache_file", "cached_FRB_true_DM_Websky_cosmo.jld2")
frb_map_output_path = get_string_arg("frb_map_output_path", "")
exclude_host = get_bool_arg("exclude_host", false)
host_exclusion_rvir_factor = get_float_arg("host_exclusion_rvir_factor", 3.0)
generate_field_from_scratch = get_bool_arg("generate_field_from_scratch", true)
field_nside = Int(round(get_float_arg("field_nside", 4096.0)))
need_positions = generate_field_from_scratch || !isempty(frb_map_output_path)

isempty(catalog_path) && error("Pass catalog_path=... to the backend.")
isempty(output_path) && error("Pass output_path=... to the backend.")
host_exclusion_rvir_factor >= 0.0 || error("host_exclusion_rvir_factor must be nonnegative.")
field_nside > 0 || error("field_nside must be positive.")

println("XGPaint DM backend configuration:")
println("  catalog_path=$(catalog_path)")
println("  output_path=$(output_path)")
println("  dm_cache_file=$(dm_cache_file)")
println("  frb_map_output_path=$(isempty(frb_map_output_path) ? "<disabled>" : frb_map_output_path)")
println("  exclude_host=$(exclude_host)")
println("  host_exclusion_rvir_factor=$(host_exclusion_rvir_factor)")
println("  generate_field_from_scratch=$(generate_field_from_scratch)")
println("  field_nside=$(field_nside)")

sample_x, sample_y, sample_z, sample_mass, sample_redshift = h5open(catalog_path, "r") do h5
    haskey(h5, "sample_mass") || error("Catalog does not contain sample_mass.")
    haskey(h5, "sample_redshift") || error("Catalog does not contain sample_redshift.")
    masses = Float64.(read(h5["sample_mass"]))
    redshifts = Float64.(read(h5["sample_redshift"]))
    if need_positions
        position_requirement = generate_field_from_scratch ?
            "generate_field_from_scratch=true" :
            "frb_map_output_path was requested"
        haskey(h5, "sample_x") || error("Catalog does not contain sample_x, which is required because $(position_requirement).")
        haskey(h5, "sample_y") || error("Catalog does not contain sample_y, which is required because $(position_requirement).")
        haskey(h5, "sample_z") || error("Catalog does not contain sample_z, which is required because $(position_requirement).")
        sx = Float64.(read(h5["sample_x"]))
        sy = Float64.(read(h5["sample_y"]))
        sz = Float64.(read(h5["sample_z"]))
        return sx, sy, sz, masses, redshifts
    end
    return Float64[], Float64[], Float64[], masses, redshifts
end

length(sample_mass) == length(sample_redshift) || error("sample_mass and sample_redshift lengths do not match.")
sample_count = length(sample_mass)
sample_count > 0 || error("Catalog contains no FRB samples.")
if need_positions
    length(sample_x) == length(sample_y) == length(sample_z) == sample_count || error("sample_x/sample_y/sample_z lengths must match sample_mass.")
end

if !isempty(frb_map_output_path)
    frb_map_output_dir = dirname(frb_map_output_path)
    isdir(frb_map_output_dir) || mkpath(frb_map_output_dir)
    frb_map = build_frb_overdensity_map(sample_x, sample_y, sample_z, field_nside)
    Healpix.saveToFITS(frb_map, "!" * frb_map_output_path, typechar="D")
    println("Saved FRB overdensity map to $(frb_map_output_path)")
end

dm_model = HaloDMProfile(BattagliaTauProfile(Omega_c=omegac, Omega_b=omegab, h=h_value))
dm_model_interp = build_interpolator(dm_model, cache_file=dm_cache_file, overwrite=false)
theta_min = compute_theta_min_local(dm_model_interp)

theta200c = Vector{Float64}(undef, sample_count)
theta_eval = Vector{Float64}(undef, sample_count)
dm_xgpaint = Vector{Float64}(undef, sample_count)

@threads for i in 1:sample_count
    mass_i = sample_mass[i]
    z_i = sample_redshift[i]

    if !isfinite(mass_i) || !isfinite(z_i) || mass_i <= 0.0 || z_i < 0.0
        theta200c[i] = NaN
        theta_eval[i] = NaN
        dm_xgpaint[i] = NaN
        continue
    end

    theta200_i = theta200c_rad(dm_model, mass_i, z_i)
    theta200c[i] = theta200_i

    theta_offset = exclude_host && isfinite(theta200_i) ? host_exclusion_rvir_factor * theta200_i : 0.0
    theta_eval_i = max(theta_min, theta_offset)
    theta_eval[i] = theta_eval_i
    if !generate_field_from_scratch
        dm_xgpaint[i] = Float64(dm_model_interp(theta_eval_i, mass_i, z_i))
    else
        dm_xgpaint[i] = NaN
    end
end

if generate_field_from_scratch
    sample_ra_unsorted, sample_dec_unsorted = xyz_to_ra_dec_threaded(sample_x, sample_y, sample_z)
    sample_ra_eval, sample_dec_eval =
        exclude_host ?
        offset_ra_dec_arrays(sample_ra_unsorted, sample_dec_unsorted, theta_eval) :
        (Float64.(sample_ra_unsorted), Float64.(sample_dec_unsorted))

    sample_perm = sortperm(sample_dec_unsorted)
    sample_ra = Float64.(sample_ra_unsorted[sample_perm])
    sample_dec = Float64.(sample_dec_unsorted[sample_perm])
    sample_mass_sorted = sample_mass[sample_perm]
    sample_redshift_sorted = sample_redshift[sample_perm]

    dm_hp = HealpixMap{Float64, RingOrder}(field_nside)
    fill!(dm_hp.pixels, 0.0)
    res = Healpix.Resolution(field_nside)
    workspace = XGPaint.HealpixRingProfileWorkspace{Float64}(res)

    paint!(dm_hp, workspace, dm_model_interp, sample_mass_sorted, sample_redshift_sorted, sample_ra, sample_dec)
    dm_xgpaint .= sample_healpix_map_at_points(dm_hp, sample_ra_eval, sample_dec_eval)
end

saved_path = write_csv_table(output_path, sample_mass, sample_redshift, theta200c, theta_eval, dm_xgpaint)
println("Saved XGPaint DM backend table to $(saved_path)")
