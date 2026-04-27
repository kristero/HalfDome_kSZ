using XGPaint, Healpix, HDF5, Interpolations
using Base.Threads

const h_value = 0.68
const c_kms = 299_792.458
const omegab = 0.049
const omegac = 0.31 - omegab
const omegam = omegab + omegac
const omegal = 1.0 - omegam
const H0 = 100.0 * h_value
const rho_m = 2.775e11 * omegam * h_value^2

# -------------------------
# options
# -------------------------
model_exists = true         # set to false to (re)build the model interpolator
save_healpix_map = true     # save Healpix map FITS
save_cl = true              # compute and save power spectrum
apply_mass_cut = true       # apply mass cut

t0 = time()

path = "lightcone_100.hdf5"
websky_path_default = "other_sims/sims/halos.pksc"
nside = 1024
chunkN = 2_000_000          # tune to your RAM

add_str_end = "13Msol_cutoff_HALO"
mass_min = 1.0e12

# -------------------------
# Battaglia16 model parameters (editable)
# -------------------------
function get_float_arg(key, default; env=nothing)
    if env !== nothing && haskey(ENV, env)
        return parse(Float64, ENV[env])
    end
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

function get_string_arg(key, default; env=nothing)
    if env !== nothing && haskey(ENV, env)
        return ENV[env]
    end
    prefix1 = "--" * key * "="
    prefix2 = key * "="
    for a in ARGS
        if startswith(a, prefix1)
            return split(a, "=", limit=2)[2]
        elseif startswith(a, prefix2)
            return split(a, "=", limit=2)[2]
        end
    end
    return String(default)
end

catalog_source = lowercase(get_string_arg("catalog_source", "halfdome"; env="TSZ_CATALOG_SOURCE"))
catalog_source in ("halfdome", "websky") || error("Unsupported catalog_source=$(repr(catalog_source)). Use \"halfdome\" or \"websky\".")
halfdome_path = get_string_arg("halfdome_path", path; env="HALFDOME_PATH")
websky_path = get_string_arg("websky_path", websky_path_default; env="WEBSKY_PATH")
catalog_path = catalog_source == "halfdome" ? halfdome_path : websky_path
simulation_name = get_string_arg("simulation_name", catalog_source; env="SIMULATION_NAME")
redshift_binning_mode_raw = lowercase(get_string_arg("redshift_binning_mode", "linear"; env="REDSHIFT_BINNING_MODE"))
redshift_binning_mode = redshift_binning_mode_raw in ("log", "logz", "log1p") ? "log1p" : redshift_binning_mode_raw
redshift_binning_mode in ("linear", "log1p") || error("Unsupported redshift_binning_mode=$(repr(redshift_binning_mode_raw)). Use \"linear\" or \"log1p\".")
redshift_bin_width = get_float_arg(
    "redshift_bin_width",
    get_float_arg("websky_redshift_bin_width", 1.0; env="WEBSKY_REDSHIFT_BIN_WIDTH");
    env="REDSHIFT_BIN_WIDTH"
)
log_redshift_bin_width = get_float_arg("log_redshift_bin_width", 0.2; env="LOG_REDSHIFT_BIN_WIDTH")
redshift_bin_width > 0.0 || error("redshift_bin_width must be positive.")
log_redshift_bin_width > 0.0 || error("log_redshift_bin_width must be positive.")

const BATTAGLIA_P0_AMP_DEFAULT = 18.1
const BATTAGLIA_P0_ALPHA_M_DEFAULT = 0.154
const BATTAGLIA_P0_ALPHA_Z_DEFAULT = -0.758
const BATTAGLIA_X_C_AMP_DEFAULT = 0.497
const BATTAGLIA_X_C_ALPHA_M_DEFAULT = -0.00865
const BATTAGLIA_X_C_ALPHA_Z_DEFAULT = 0.731
const BATTAGLIA_BETA_AMP_DEFAULT = 4.35
const BATTAGLIA_BETA_ALPHA_M_DEFAULT = 0.0393
const BATTAGLIA_BETA_ALPHA_Z_DEFAULT = 0.415
const BATTAGLIA_ALPHA_AMP_DEFAULT = 1.0
const BATTAGLIA_ALPHA_ALPHA_M_DEFAULT = 0.0
const BATTAGLIA_ALPHA_ALPHA_Z_DEFAULT = 0.0
const BATTAGLIA_GAMMA_AMP_DEFAULT = -0.3
const BATTAGLIA_GAMMA_ALPHA_M_DEFAULT = 0.0
const BATTAGLIA_GAMMA_ALPHA_Z_DEFAULT = 0.0

battaglia_P0_amp = get_float_arg("battaglia_P0_amp", BATTAGLIA_P0_AMP_DEFAULT; env="BATTAGLIA_P0_AMP")
battaglia_P0_alpha_m = get_float_arg("battaglia_P0_alpha_m", BATTAGLIA_P0_ALPHA_M_DEFAULT; env="BATTAGLIA_P0_ALPHA_M")
battaglia_P0_alpha_z = get_float_arg("battaglia_P0_alpha_z", BATTAGLIA_P0_ALPHA_Z_DEFAULT; env="BATTAGLIA_P0_ALPHA_Z")

battaglia_x_c_amp = get_float_arg("battaglia_x_c_amp", BATTAGLIA_X_C_AMP_DEFAULT; env="BATTAGLIA_X_C_AMP")
battaglia_x_c_alpha_m = get_float_arg("battaglia_x_c_alpha_m", BATTAGLIA_X_C_ALPHA_M_DEFAULT; env="BATTAGLIA_X_C_ALPHA_M")
battaglia_x_c_alpha_z = get_float_arg("battaglia_x_c_alpha_z", BATTAGLIA_X_C_ALPHA_Z_DEFAULT; env="BATTAGLIA_X_C_ALPHA_Z")

battaglia_beta_amp = get_float_arg("battaglia_beta_amp", BATTAGLIA_BETA_AMP_DEFAULT; env="BATTAGLIA_BETA_AMP")
battaglia_beta_alpha_m = get_float_arg("battaglia_beta_alpha_m", BATTAGLIA_BETA_ALPHA_M_DEFAULT; env="BATTAGLIA_BETA_ALPHA_M")
battaglia_beta_alpha_z = get_float_arg("battaglia_beta_alpha_z", BATTAGLIA_BETA_ALPHA_Z_DEFAULT; env="BATTAGLIA_BETA_ALPHA_Z")

battaglia_alpha_amp = get_float_arg("battaglia_alpha_amp", BATTAGLIA_ALPHA_AMP_DEFAULT; env="BATTAGLIA_ALPHA_AMP")
battaglia_alpha_alpha_m = get_float_arg("battaglia_alpha_alpha_m", BATTAGLIA_ALPHA_ALPHA_M_DEFAULT; env="BATTAGLIA_ALPHA_ALPHA_M")
battaglia_alpha_alpha_z = get_float_arg("battaglia_alpha_alpha_z", BATTAGLIA_ALPHA_ALPHA_Z_DEFAULT; env="BATTAGLIA_ALPHA_ALPHA_Z")

battaglia_gamma_amp = get_float_arg("battaglia_gamma_amp", BATTAGLIA_GAMMA_AMP_DEFAULT; env="BATTAGLIA_GAMMA_AMP")
battaglia_gamma_alpha_m = get_float_arg("battaglia_gamma_alpha_m", BATTAGLIA_GAMMA_ALPHA_M_DEFAULT; env="BATTAGLIA_GAMMA_ALPHA_M")
battaglia_gamma_alpha_z = get_float_arg("battaglia_gamma_alpha_z", BATTAGLIA_GAMMA_ALPHA_Z_DEFAULT; env="BATTAGLIA_GAMMA_ALPHA_Z")

function fmt_param_value(x)
    s = string(x)
    s = replace(s, "-" => "m")
    s = replace(s, "." => "p")
    s = replace(s, "+" => "")
    return s
end

function safe_filename_tag(s::AbstractString)
    tag = lowercase(strip(String(s)))
    tag = replace(tag, r"[^A-Za-z0-9_+\-.]+" => "_")
    tag = replace(tag, r"_+" => "_")
    return isempty(tag) ? "simulation" : tag
end

function build_redshift_binning_tag(mode::AbstractString, linear_dz::Real, log_dlog::Real)
    if mode == "linear"
        return "zlin_dz$(fmt_param_value(Float64(linear_dz)))"
    end
    return "zlog1p_dlog$(fmt_param_value(Float64(log_dlog)))"
end

function build_param_tag()
    parts = String[]
    if battaglia_P0_amp != BATTAGLIA_P0_AMP_DEFAULT
        push!(parts, "battaglia_P0_amp_" * fmt_param_value(battaglia_P0_amp))
    end
    if battaglia_P0_alpha_m != BATTAGLIA_P0_ALPHA_M_DEFAULT
        push!(parts, "battaglia_P0_alpha_m_" * fmt_param_value(battaglia_P0_alpha_m))
    end
    if battaglia_P0_alpha_z != BATTAGLIA_P0_ALPHA_Z_DEFAULT
        push!(parts, "battaglia_P0_alpha_z_" * fmt_param_value(battaglia_P0_alpha_z))
    end
    if battaglia_x_c_amp != BATTAGLIA_X_C_AMP_DEFAULT
        push!(parts, "battaglia_x_c_amp_" * fmt_param_value(battaglia_x_c_amp))
    end
    if battaglia_x_c_alpha_m != BATTAGLIA_X_C_ALPHA_M_DEFAULT
        push!(parts, "battaglia_x_c_alpha_m_" * fmt_param_value(battaglia_x_c_alpha_m))
    end
    if battaglia_x_c_alpha_z != BATTAGLIA_X_C_ALPHA_Z_DEFAULT
        push!(parts, "battaglia_x_c_alpha_z_" * fmt_param_value(battaglia_x_c_alpha_z))
    end
    if battaglia_beta_amp != BATTAGLIA_BETA_AMP_DEFAULT
        push!(parts, "battaglia_beta_amp_" * fmt_param_value(battaglia_beta_amp))
    end
    if battaglia_beta_alpha_m != BATTAGLIA_BETA_ALPHA_M_DEFAULT
        push!(parts, "battaglia_beta_alpha_m_" * fmt_param_value(battaglia_beta_alpha_m))
    end
    if battaglia_beta_alpha_z != BATTAGLIA_BETA_ALPHA_Z_DEFAULT
        push!(parts, "battaglia_beta_alpha_z_" * fmt_param_value(battaglia_beta_alpha_z))
    end
    if battaglia_alpha_amp != BATTAGLIA_ALPHA_AMP_DEFAULT
        push!(parts, "battaglia_alpha_amp_" * fmt_param_value(battaglia_alpha_amp))
    end
    if battaglia_alpha_alpha_m != BATTAGLIA_ALPHA_ALPHA_M_DEFAULT
        push!(parts, "battaglia_alpha_alpha_m_" * fmt_param_value(battaglia_alpha_alpha_m))
    end
    if battaglia_alpha_alpha_z != BATTAGLIA_ALPHA_ALPHA_Z_DEFAULT
        push!(parts, "battaglia_alpha_alpha_z_" * fmt_param_value(battaglia_alpha_alpha_z))
    end
    if battaglia_gamma_amp != BATTAGLIA_GAMMA_AMP_DEFAULT
        push!(parts, "battaglia_gamma_amp_" * fmt_param_value(battaglia_gamma_amp))
    end
    if battaglia_gamma_alpha_m != BATTAGLIA_GAMMA_ALPHA_M_DEFAULT
        push!(parts, "battaglia_gamma_alpha_m_" * fmt_param_value(battaglia_gamma_alpha_m))
    end
    if battaglia_gamma_alpha_z != BATTAGLIA_GAMMA_ALPHA_Z_DEFAULT
        push!(parts, "battaglia_gamma_alpha_z_" * fmt_param_value(battaglia_gamma_alpha_z))
    end
    if isempty(parts)
        return "base"
    end
    return "base_plus_" * join(parts, "__")
end

function ensure_writable_output_dir(preferred_dir::AbstractString, fallback_dir::AbstractString)
    candidates = (abspath(preferred_dir), abspath(fallback_dir))

    for candidate in candidates
        try
            isdir(candidate) || mkpath(candidate)
            probe_path = joinpath(candidate, ".codex_write_probe")
            open(probe_path, "w") do io
                write(io, "ok")
            end
            rm(probe_path; force=true)
            return candidate
        catch err
            @warn "Output directory is not writable, trying fallback." candidate exception=(err, catch_backtrace())
        end
    end

    error("Could not find a writable output directory. Tried $(collect(candidates)).")
end

function fallback_output_path(path::AbstractString)
    return joinpath(homedir(), "HalfDome_outputs", "visuals", basename(path))
end

param_tag = build_param_tag()
redshift_binning_tag = build_redshift_binning_tag(redshift_binning_mode, redshift_bin_width, log_redshift_bin_width)
run_tag = "$(add_str_end)_$(param_tag)_$(redshift_binning_tag)"
simulation_tag = safe_filename_tag(simulation_name)

output_dir = abspath(joinpath(homedir(), "HalfDome_outputs", "visuals"))
fits_output_path = joinpath(output_dir, "$(simulation_tag)_tSZ_nside$(nside)_$(run_tag)_m200c.fits")
mass_fits_output_path = joinpath(output_dir, "$(simulation_tag)_mass_nside$(nside)_$(run_tag)_m200c.fits")
cl_output_path = joinpath(output_dir, "$(simulation_tag)_tSZ_cl_m200c_$(param_tag)_nside$(nside)_zsort.fits")

println("Using output directory: $(output_dir)")
println("Using catalog source: $(catalog_source)")
println("Using simulation tag: $(simulation_tag)")
println("Using catalog path: $(catalog_path)")
println("Redshift binning mode: $(redshift_binning_mode)")
println("Linear redshift bin width dz: $(redshift_bin_width)")
println("Log redshift bin width dlog10(1+z): $(log_redshift_bin_width)")

println("Battaglia16 physical parameters:")
println("  P0_amp=$(battaglia_P0_amp), P0_alpha_m=$(battaglia_P0_alpha_m), P0_alpha_z=$(battaglia_P0_alpha_z)")
println("  x_c_amp=$(battaglia_x_c_amp), x_c_alpha_m=$(battaglia_x_c_alpha_m), x_c_alpha_z=$(battaglia_x_c_alpha_z)")
println("  beta_amp=$(battaglia_beta_amp), beta_alpha_m=$(battaglia_beta_alpha_m), beta_alpha_z=$(battaglia_beta_alpha_z)")
println("  alpha_amp=$(battaglia_alpha_amp), alpha_alpha_m=$(battaglia_alpha_alpha_m), alpha_alpha_z=$(battaglia_alpha_alpha_z)")
println("  gamma_amp=$(battaglia_gamma_amp), gamma_alpha_m=$(battaglia_gamma_alpha_m), gamma_alpha_z=$(battaglia_gamma_alpha_z)")
println("Mass map extent: HalfDome uses Rdisp spatial components; WebSky uses the catalog R column.")

function xyz_to_ra_dec_threaded(x::AbstractVector{T}, y::AbstractVector{T}, z::AbstractVector{T}) where T
    @assert length(x) == length(y) == length(z)

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

function make_z_of_chi_itp(; omegam, h_value, z1=0.0, z2=6.0, nz=100_000)
    H0_local = 100.0 * h_value
    H(z) = H0_local * sqrt(omegam * (1 + z)^3 + 1 - omegam)
    dchidz(z) = c_kms / H(z)

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

itp_z_of_chi = make_z_of_chi_itp(omegam=omegam, h_value=h_value)

@inline function m200m_to_m200c_scalar(m200m::Float64, z::Float64)
    one_plus_z = 1.0 + z
    ez_num = omegam * one_plus_z^3
    omegamz = ez_num / (ez_num + 1.0 - omegam)
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

function build_z_range_tag(z_min::Real, z_max::Real; digits::Int=3)
    z_min_tag = fmt_param_value(round(Float64(z_min); digits=digits))
    z_max_tag = fmt_param_value(round(Float64(z_max); digits=digits))
    return "z_$(z_min_tag)_$(z_max_tag)"
end

function batch_y_output_path(output_dir::AbstractString, simulation_tag::AbstractString, nside::Int, run_tag::AbstractString, bin_number::Int, z_min::Real, z_max::Real)
    z_tag = build_z_range_tag(z_min, z_max)
    return joinpath(
        output_dir,
        "$(simulation_tag)_tSZ_zbin$(bin_number)_nside$(nside)_$(run_tag)_$(z_tag).fits"
    )
end

function batch_mass_output_path(output_dir::AbstractString, simulation_tag::AbstractString, nside::Int, run_tag::AbstractString, bin_number::Int, z_min::Real, z_max::Real)
    z_tag = build_z_range_tag(z_min, z_max)
    return joinpath(
        output_dir,
        "$(simulation_tag)_mass_zbin$(bin_number)_nside$(nside)_$(run_tag)_$(z_tag).fits"
    )
end

function build_sorted_halo_order(halo_mass::Vector{Float64}, redshift::Vector{Float64}; selection::Bool, mass_min::Float64)
    keep = redshift .>= 0.0
    if selection
        keep .&= halo_mass .>= mass_min
    end

    selected_idx = findall(keep)
    selected_redshift = redshift[selected_idx]
    selected_mass = halo_mass[selected_idx]

    if isempty(selected_idx)
        return Int[], Float64[], Float64[]
    end

    order = sortperm(selected_redshift; rev=true)
    return selected_idx[order], selected_mass[order], selected_redshift[order]
end

function read_position_batch(pos_ds, idx_batch::Vector{Int})
    isempty(idx_batch) && return Matrix{Float64}(undef, 3, 0)

    read_order = sortperm(idx_batch)
    idx_batch_sorted = idx_batch[read_order]
    pos_sorted = Matrix{Float64}(undef, 3, length(idx_batch_sorted))

    block_start = 1
    while block_start <= length(idx_batch_sorted)
        block_end = block_start
        while block_end < length(idx_batch_sorted) &&
              idx_batch_sorted[block_end + 1] == idx_batch_sorted[block_end] + 1
            block_end += 1
        end

        dataset_range = idx_batch_sorted[block_start]:idx_batch_sorted[block_end]
        pos_sorted[:, block_start:block_end] .= Float64.(pos_ds[:, dataset_range])
        block_start = block_end + 1
    end

    undo_read_order = invperm(read_order)
    return pos_sorted[:, undo_read_order]
end

function read_rdisp_spatial_batch(rdisp_ds, idx_batch::Vector{Int})
    isempty(idx_batch) && return Matrix{Float64}(undef, 3, 0)

    read_order = sortperm(idx_batch)
    idx_batch_sorted = idx_batch[read_order]
    rdisp_sorted = Matrix{Float64}(undef, 3, length(idx_batch_sorted))

    block_start = 1
    while block_start <= length(idx_batch_sorted)
        block_end = block_start
        while block_end < length(idx_batch_sorted) &&
              idx_batch_sorted[block_end + 1] == idx_batch_sorted[block_end] + 1
            block_end += 1
        end

        dataset_range = idx_batch_sorted[block_start]:idx_batch_sorted[block_end]
        rdisp_sorted[:, block_start:block_end] .= Float64.(rdisp_ds[1:3, dataset_range])
        block_start = block_end + 1
    end

    undo_read_order = invperm(read_order)
    return rdisp_sorted[:, undo_read_order]
end

function effective_rdisp_radius_batch(rdisp_spatial::AbstractMatrix{<:Real})
    size(rdisp_spatial, 1) == 3 || error("Expected 3 spatial Rdisp components per halo.")
    return vec(sqrt.(sum(abs2, rdisp_spatial; dims=1)))
end

function radius_to_angular_extent(radius_comoving::Real, chi_comoving::Real)
    if !isfinite(radius_comoving) || !isfinite(chi_comoving) || radius_comoving <= 0.0 || chi_comoving <= 0.0
        return 0.0
    end
    return min(Float64(radius_comoving / chi_comoving), pi)
end

function paint_mass_disc!(
    mass_hp::HealpixMap{Float64, RingOrder},
    workspace::XGPaint.HealpixRingProfileWorkspace{Float64},
    alpha::Real,
    delta::Real,
    radius_rad::Real,
    mass_value::Real
)
    if !isfinite(radius_rad) || radius_rad <= 0.0
        theta = pi / 2 - Float64(delta)
        phi = mod(Float64(alpha), 2pi)
        pix = Healpix.ang2pixRing(mass_hp.resolution, theta, phi)
        mass_hp.pixels[pix] += Float64(mass_value)
        return nothing
    end

    center_theta = Float64(pi / 2 - delta)
    center_phi = mod(Float64(alpha), 2pi)
    search_radius = min(Float64(radius_rad), pi)

    ring_start, ring_stop = XGPaint.get_relevant_rings(workspace.res, center_theta, search_radius)
    @inbounds for ring_idx in ring_start:ring_stop
        range1, range2 = XGPaint.get_ring_disc_ranges(workspace, ring_idx, center_theta, center_phi, search_radius)
        first_pixel = workspace.ring_first_pixels[ring_idx]
        for local_pix_idx in range1
            mass_hp.pixels[first_pixel + local_pix_idx - 1] += Float64(mass_value)
        end
        for local_pix_idx in range2
            mass_hp.pixels[first_pixel + local_pix_idx - 1] += Float64(mass_value)
        end
    end

    return nothing
end

function build_halo_mass_map!(
    mass_hp::HealpixMap{Float64, RingOrder},
    workspace::XGPaint.HealpixRingProfileWorkspace{Float64},
    ras::AbstractVector{<:Real},
    decs::AbstractVector{<:Real},
    masses::AbstractVector{<:Real},
    angular_radii::AbstractVector{<:Real}
)
    length(ras) == length(decs) == length(masses) == length(angular_radii) || error("Mass-map inputs must have the same length.")

    fill!(mass_hp.pixels, 0.0)

    @inbounds for i in eachindex(masses)
        paint_mass_disc!(
            mass_hp,
            workspace,
            ras[i],
            decs[i],
            angular_radii[i],
            masses[i]
        )
    end

    return nothing
end

function save_healpix_fits_overwrite(
    hp_map::HealpixMap{Float64, RingOrder},
    path::AbstractString;
    typechar::AbstractString="D"
)
    abs_path = abspath(path)
    parent_dir = dirname(abs_path)
    isdir(parent_dir) || mkpath(parent_dir)

    if isfile(abs_path)
        rm(abs_path; force=true)
    end

    tmp_pixels_path = tempname(parent_dir) * ".bin"
    open(tmp_pixels_path, "w") do io
        write(io, hp_map.pixels)
    end

    python_code = """
import numpy as np
import healpy as hp
import sys

pixels_path = sys.argv[1]
output_path = sys.argv[2]
nside = int(sys.argv[3])

pixels = np.fromfile(pixels_path, dtype=np.float64)
expected = 12 * nside * nside
if pixels.size != expected:
    raise RuntimeError(f"Expected {expected} pixels for nside={nside}, got {pixels.size}")

hp.write_map(output_path, pixels, overwrite=True, nest=False, dtype=np.float64)
"""

    try
        run(`/usr/bin/python3 -c $python_code $tmp_pixels_path $abs_path $(string(hp_map.resolution.nside))`)
    finally
        rm(tmp_pixels_path; force=true)
    end

    println("Saved FITS map to $(abs_path)")
    return abs_path
end

function write_cl_fits_overwrite(path::AbstractString, cl_values)
    abs_path = abspath(path)
    parent_dir = dirname(abs_path)
    isdir(parent_dir) || mkpath(parent_dir)

    if isfile(abs_path)
        rm(abs_path; force=true)
    end

    cl_array = Float64.(collect(cl_values))
    tmp_cl_path = tempname(parent_dir) * ".bin"
    open(tmp_cl_path, "w") do io
        write(io, cl_array)
    end

    python_code = """
import numpy as np
from astropy.io import fits
import sys

cl_path = sys.argv[1]
output_path = sys.argv[2]

cl = np.fromfile(cl_path, dtype=np.float64)
fits.PrimaryHDU(cl).writeto(output_path, overwrite=True)
"""

    try
        run(`/usr/bin/python3 -c $python_code $tmp_cl_path $abs_path`)
    finally
        rm(tmp_cl_path; force=true)
    end

    println("Saved Cl FITS to $(abs_path)")
    return abs_path
end


function paint_redshift_sorted_batch!(
    m_hp,
    mass_hp,
    tmp_hp,
    batch_mass_hp,
    workspace,
    y_model_interp,
    x_batch,
    y_batch,
    z_batch,
    radius_batch::Vector{Float64},
    mass_batch::Vector{Float64},
    redshift_batch::Vector{Float64}
)
    xs = Float64.(x_batch)
    ys = Float64.(y_batch)
    zs = Float64.(z_batch)
    chis = sqrt.(xs .^ 2 .+ ys .^ 2 .+ zs .^ 2)

    ra, dec = xyz_to_ra_dec_threaded(xs, ys, zs)

    # Batches are defined by descending redshift; within each batch keep the
    # established dec ordering before paint!.
    perm = sortperm(dec)
    ra = ra[perm]
    dec = dec[perm]
    ms = mass_batch[perm]
    zsft = redshift_batch[perm]
    angular_radii = radius_to_angular_extent.(radius_batch[perm], chis[perm])

    fill!(tmp_hp.pixels, 0.0)
    paint!(tmp_hp, workspace, y_model_interp, ms, zsft, ra, dec)
    build_halo_mass_map!(batch_mass_hp, workspace, ra, dec, ms, angular_radii)
    m_hp.pixels .+= tmp_hp.pixels
    mass_hp.pixels .+= batch_mass_hp.pixels

    return nothing
end

function summarize_websky_selected_redshifts(
    websky_path::AbstractString,
    chunkN::Int,
    itp_z_of_chi,
    rho_m;
    selection::Bool,
    mass_min::Float64
)
    return open(websky_path, "r") do io
        total_halo_count = Int(read(io, Int32))
        RTHmax = read(io, Float32)
        redshiftbox = read(io, Float32)
        @show total_halo_count RTHmax redshiftbox

        buf = Matrix{Float32}(undef, 10, chunkN)
        nleft = total_halo_count
        selected_count = 0
        selected_z_min = Inf
        selected_z_max = -Inf

        while nleft > 0
            nthis = min(chunkN, nleft)

            rawview = @view reinterpret(Float32, vec(buf))[1:10 * nthis]
            read!(io, rawview)

            cat = @view buf[:, 1:nthis]
            x = @view cat[1, :]
            y = @view cat[2, :]
            z = @view cat[3, :]
            radius = @view cat[7, :]

            redshift, halo_mass = compute_redshift_and_mass(x, y, z, radius, itp_z_of_chi, rho_m)

            keep = isfinite.(redshift) .& isfinite.(halo_mass) .& (redshift .>= 0.0)
            if selection
                keep .&= halo_mass .>= mass_min
            end

            if any(keep)
                kept_redshift = redshift[keep]
                selected_count += length(kept_redshift)
                selected_z_min = min(selected_z_min, minimum(kept_redshift))
                selected_z_max = max(selected_z_max, maximum(kept_redshift))
            end

            nleft -= nthis
        end

        return total_halo_count, selected_count, selected_z_min, selected_z_max
    end
end

function redshift_bin_edges(z_max::Real, mode::AbstractString, linear_dz::Real, log_dlog::Real)
    if mode == "linear"
        z_hi = (floor(Float64(z_max) / Float64(linear_dz)) + 1.0) * Float64(linear_dz)
        z_hi = max(z_hi, Float64(linear_dz))
        return collect(0.0:Float64(linear_dz):z_hi)
    end

    log_hi = (floor(log10(1.0 + Float64(z_max)) / Float64(log_dlog)) + 1.0) * Float64(log_dlog)
    log_hi = max(log_hi, Float64(log_dlog))
    log_edges = collect(0.0:Float64(log_dlog):log_hi)
    return 10.0 .^ log_edges .- 1.0
end

function paint_websky_redshift_bin!(
    m_hp,
    mass_hp,
    tmp_hp,
    batch_mass_hp,
    workspace,
    y_model_interp,
    websky_path::AbstractString,
    chunkN::Int,
    itp_z_of_chi,
    rho_m,
    z_min::Float64,
    z_max::Float64;
    selection::Bool,
    mass_min::Float64
)
    painted_count = 0
    actual_z_min = Inf
    actual_z_max = -Inf

    open(websky_path, "r") do io
        total_halo_count = Int(read(io, Int32))
        read(io, Float32)
        read(io, Float32)

        buf = Matrix{Float32}(undef, 10, chunkN)
        nleft = total_halo_count

        while nleft > 0
            nthis = min(chunkN, nleft)

            rawview = @view reinterpret(Float32, vec(buf))[1:10 * nthis]
            read!(io, rawview)

            cat = @view buf[:, 1:nthis]
            x = @view cat[1, :]
            y = @view cat[2, :]
            z = @view cat[3, :]
            radius = @view cat[7, :]

            redshift, halo_mass = compute_redshift_and_mass(x, y, z, radius, itp_z_of_chi, rho_m)

            keep = isfinite.(redshift) .& isfinite.(halo_mass) .& (redshift .>= z_min) .& (redshift .< z_max)
            if selection
                keep .&= halo_mass .>= mass_min
            end

            if any(keep)
                x_batch = Float64.(x[keep])
                y_batch = Float64.(y[keep])
                z_batch = Float64.(z[keep])
                radius_batch = Float64.(radius[keep])
                mass_batch = halo_mass[keep]
                redshift_batch = redshift[keep]

                paint_redshift_sorted_batch!(
                    m_hp,
                    mass_hp,
                    tmp_hp,
                    batch_mass_hp,
                    workspace,
                    y_model_interp,
                    x_batch,
                    y_batch,
                    z_batch,
                    radius_batch,
                    mass_batch,
                    redshift_batch
                )

                painted_count += length(redshift_batch)
                actual_z_min = min(actual_z_min, minimum(redshift_batch))
                actual_z_max = max(actual_z_max, maximum(redshift_batch))
            end

            nleft -= nthis
        end
    end

    return painted_count, actual_z_min, actual_z_max
end

selection = apply_mass_cut

model = Battaglia16ThermalSZProfile(
    Omega_c=omegac,
    Omega_b=omegab,
    h=h_value,
    P0_amp=battaglia_P0_amp,
    P0_alpha_m=battaglia_P0_alpha_m,
    P0_alpha_z=battaglia_P0_alpha_z,
    x_c_amp=battaglia_x_c_amp,
    x_c_alpha_m=battaglia_x_c_alpha_m,
    x_c_alpha_z=battaglia_x_c_alpha_z,
    beta_amp=battaglia_beta_amp,
    beta_alpha_m=battaglia_beta_alpha_m,
    beta_alpha_z=battaglia_beta_alpha_z,
    alpha_amp=battaglia_alpha_amp,
    alpha_alpha_m=battaglia_alpha_alpha_m,
    alpha_alpha_z=battaglia_alpha_alpha_z,
    gamma_amp=battaglia_gamma_amp,
    gamma_alpha_m=battaglia_gamma_alpha_m,
    gamma_alpha_z=battaglia_gamma_alpha_z
)

cache_sim_tag = catalog_source == "halfdome" ? "HalfDome" : "Websky"
y_cache_file = "cached_tSZ_$(cache_sim_tag)_cosmo_$(param_tag).jld2"
if model_exists
    y_model_interp = build_interpolator(
        model,
        cache_file=y_cache_file,
        overwrite=false
    )
else
    y_model_interp = build_interpolator(
        model;
        cache_file=y_cache_file,
        pad=256,
        logM_max=15.7,
        overwrite=true,
        verbose=true
    )
end

m_hp = HealpixMap{Float64, RingOrder}(nside)
fill!(m_hp.pixels, 0.0)
mass_hp = HealpixMap{Float64, RingOrder}(nside)
fill!(mass_hp.pixels, 0.0)
res = Healpix.Resolution(nside)
w = XGPaint.HealpixRingProfileWorkspace{Float64}(res)
tmp_hp = HealpixMap{Float64, RingOrder}(nside)
fill!(tmp_hp.pixels, 0.0)
batch_mass_hp = HealpixMap{Float64, RingOrder}(nside)
fill!(batch_mass_hp.pixels, 0.0)

isdir(output_dir) || mkpath(output_dir)

println("Initiating HealPix with NSide: $nside")
println("Julia threads available: $(nthreads())")
println("Preparing $(simulation_tag) halos for high-z to low-z batching.")

if catalog_source == "halfdome"
    h5open(halfdome_path, "r") do h5
        pos_ds = h5["Position"]
        rdisp_ds = h5["Rdisp"]
        mass_ds = h5["halo_mass_m200c"]
        redshift_ds = h5["redshift"]

        total_halo_count = size(pos_ds, 2)
        @show total_halo_count

        halo_mass = Float64.(read(mass_ds)) ./ h_value
        redshift = Float64.(read(redshift_ds))

        sorted_idx, sorted_mass, sorted_redshift = build_sorted_halo_order(
            halo_mass,
            redshift;
            selection=selection,
            mass_min=mass_min
        )

        selected_halo_count = length(sorted_idx)
        println("Selected halos after cuts: $(selected_halo_count)")
        selected_halo_count > 0 || error("No HalfDome halos passed the current selection.")

        selected_z_min = minimum(sorted_redshift)
        selected_z_max = maximum(sorted_redshift)
        println(
            "Selected HalfDome redshift range: " *
            "[$(round(selected_z_min; digits=4)), $(round(selected_z_max; digits=4))]."
        )

        redshift_edges = redshift_bin_edges(selected_z_max, redshift_binning_mode, redshift_bin_width, log_redshift_bin_width)
        nbins = length(redshift_edges) - 1
        println(
            "Painting HalfDome with $(nbins) redshift bins using $(redshift_binning_tag), " *
            "from z_max to z=0."
        )

        zbin_number_ref = Ref(0)
        for bin_idx in nbins:-1:1
            z_min_bin = redshift_edges[bin_idx]
            z_max_bin = redshift_edges[bin_idx + 1]

            in_bin = findall(z -> z >= z_min_bin && z < z_max_bin, sorted_redshift)
            isempty(in_bin) && continue

            zbin_number_ref[] += 1
            zbin_number = zbin_number_ref[]

            idx_batch = sorted_idx[in_bin]
            mass_batch = sorted_mass[in_bin]
            redshift_batch = sorted_redshift[in_bin]

            pos = read_position_batch(pos_ds, idx_batch)
            rdisp_spatial = read_rdisp_spatial_batch(rdisp_ds, idx_batch)
            radius_batch = effective_rdisp_radius_batch(rdisp_spatial)

            paint_redshift_sorted_batch!(
                m_hp,
                mass_hp,
                tmp_hp,
                batch_mass_hp,
                w,
                y_model_interp,
                view(pos, 1, :),
                view(pos, 2, :),
                view(pos, 3, :),
                radius_batch,
                mass_batch,
                redshift_batch
            )

            z_max_batch = redshift_batch[1]
            z_min_batch = redshift_batch[end]
            println(
                "Painted HalfDome redshift bin $(zbin_number) from bin $(bin_idx)/$(nbins) " *
                "with $(length(in_bin)) halos; z in " *
                "[$(round(z_min_batch; digits=4)), $(round(z_max_batch; digits=4))]."
            )

            if save_healpix_map
                batch_y_path = batch_y_output_path(output_dir, simulation_tag, nside, run_tag, zbin_number, z_min_batch, z_max_batch)
                batch_mass_path = batch_mass_output_path(output_dir, simulation_tag, nside, run_tag, zbin_number, z_min_batch, z_max_batch)
                save_healpix_fits_overwrite(m_hp, batch_y_path, typechar="D")
                save_healpix_fits_overwrite(mass_hp, batch_mass_path, typechar="D")
            end
        end

        zbin_number_ref[] > 0 || error("HalfDome redshift binning did not paint any selected halos.")
    end
else
    total_halo_count, selected_halo_count, selected_z_min, selected_z_max = summarize_websky_selected_redshifts(
        websky_path,
        chunkN,
        itp_z_of_chi,
        rho_m;
        selection=selection,
        mass_min=mass_min
    )

    println("Total halos in WebSky catalog: $(total_halo_count)")
    println("Selected halos after cuts: $(selected_halo_count)")
    selected_halo_count > 0 || error("No WebSky halos passed the current selection.")
    println(
        "Selected WebSky redshift range: " *
        "[$(round(selected_z_min; digits=4)), $(round(selected_z_max; digits=4))]."
    )

    redshift_edges = redshift_bin_edges(selected_z_max, redshift_binning_mode, redshift_bin_width, log_redshift_bin_width)
    nbins = length(redshift_edges) - 1
    println(
        "Painting WebSky with $(nbins) redshift bins using $(redshift_binning_tag), " *
        "streamed from z_max to z=0."
    )

    batch_number_ref = Ref(0)
    for bin_idx in nbins:-1:1
        z_min_bin = redshift_edges[bin_idx]
        z_max_bin = redshift_edges[bin_idx + 1]

        painted_count, actual_z_min, actual_z_max = paint_websky_redshift_bin!(
            m_hp,
            mass_hp,
            tmp_hp,
            batch_mass_hp,
            w,
            y_model_interp,
            websky_path,
            chunkN,
            itp_z_of_chi,
            rho_m,
            z_min_bin,
            z_max_bin;
            selection=selection,
            mass_min=mass_min
        )

        painted_count == 0 && continue
        batch_number_ref[] += 1
        batch_number = batch_number_ref[]

        println(
            "Painted WebSky redshift bin $(batch_number) from bin $(bin_idx)/$(nbins) " *
            "with $(painted_count) halos; z in " *
            "[$(round(actual_z_min; digits=4)), $(round(actual_z_max; digits=4))]."
        )

        if save_healpix_map
            batch_y_path = batch_y_output_path(output_dir, simulation_tag, nside, run_tag, batch_number, actual_z_min, actual_z_max)
            batch_mass_path = batch_mass_output_path(output_dir, simulation_tag, nside, run_tag, batch_number, actual_z_min, actual_z_max)
            Healpix.saveToFITS(
                m_hp,
                "!" * batch_y_path,
                typechar="D"
            )
            Healpix.saveToFITS(
                mass_hp,
                "!" * batch_mass_path,
                typechar="D"
            )
        end
    end

    batch_number_ref[] > 0 || error("WebSky redshift bin streaming did not paint any selected halos.")
end

if save_healpix_map
    save_healpix_fits_overwrite(m_hp, fits_output_path, typechar="D")
    save_healpix_fits_overwrite(mass_hp, mass_fits_output_path, typechar="D")
end

if save_cl
    cl = anafast(m_hp, niter=0)
    write_cl_fits_overwrite(cl_output_path, cl)
end

println("Finished $(simulation_tag) Healpix tSZ total (redshift-sorted batched).")
elapsed = time() - t0
println("Elapsed time: $(round(elapsed; digits=2)) s")
