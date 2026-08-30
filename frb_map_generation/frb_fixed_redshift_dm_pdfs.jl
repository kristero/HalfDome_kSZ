if !haskey(ENV, "GKSwstype")
    ENV["GKSwstype"] = "png"
end
if !haskey(ENV, "GKS_WSTYPE")
    ENV["GKS_WSTYPE"] = "png"
end

# Fixed-redshift, hostless FRB DM PDF generator.
#
# For each requested FRB redshift, this places N random FRB sightlines on the
# sky, accumulates XGPaint HaloDMProfile contributions from foreground halos
# with z_halo <= z_frb, and plots all p(DM | z_frb) curves on one figure.

using XGPaint
using Healpix
using Interpolations
using HDF5
using Random
using Statistics
using Plots
using Base.Threads

const H_VALUE = 0.68
const OMEGAB = 0.049
const OMEGAC = 0.31 - OMEGAB
const OMEGAM = OMEGAB + OMEGAC
const C_KMS = 299_792.458

const compute_theta_max_local =
    isdefined(XGPaint, Symbol("compute_", Char(0x03b8), "max")) ?
    getfield(XGPaint, Symbol("compute_", Char(0x03b8), "max")) :
    error("XGPaint does not define compute_theta_max.")

thread_capacity() = isdefined(Base.Threads, :maxthreadid) ? Base.Threads.maxthreadid() : Base.Threads.nthreads()

code_root() = @__DIR__
project_root() = basename(code_root()) == "frb_map_generation" ? dirname(code_root()) : code_root()

function resolve_project_path(path::AbstractString)
    isempty(path) && return String(path)
    return isabspath(path) ? String(path) : normpath(joinpath(project_root(), path))
end

function resolve_halfdome_catalog_path(path::AbstractString)
    resolved = resolve_project_path(path)
    isdir(resolved) || return resolved

    candidates = filter(readdir(resolved; join=true)) do entry
        isfile(entry) && lowercase(splitext(entry)[2]) in (".h5", ".hdf5")
    end
    isempty(candidates) && error("halfdome_path=$(resolved) is a directory, but it contains no HDF5 catalog.")

    for preferred in ("lightcone_100.hdf5", "lightcone_100.h5", "halos.hdf5", "halos.h5")
        matches = filter(entry -> lowercase(basename(entry)) == preferred, candidates)
        length(matches) == 1 && return only(matches)
    end

    length(candidates) == 1 && return only(candidates)
    error("halfdome_path=$(resolved) contains multiple HDF5 files. Pass the exact catalog file.")
end

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

function get_int_arg(key, default; env=nothing)
    value = get_string_arg(key, ""; env=env)
    isempty(value) || return parse(Int, value)
    return Int(default)
end

function get_float_arg(key, default; env=nothing)
    value = get_string_arg(key, ""; env=env)
    isempty(value) || return parse(Float64, value)
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

function get_bool_arg(key, default; env=nothing)
    value = get_string_arg(key, ""; env=env)
    isempty(value) || return parse_bool_arg(value)
    return Bool(default)
end

function parse_float_list(text::AbstractString)
    values = Float64[]
    for token in split(text, ",")
        stripped = strip(token)
        isempty(stripped) && continue
        push!(values, parse(Float64, stripped))
    end
    isempty(values) && error("Need at least one redshift value.")
    return values
end

function fmt_param_value(x)
    s = string(x)
    s = replace(s, "-" => "m")
    s = replace(s, "." => "p")
    s = replace(s, "+" => "")
    return s
end

function make_chi_and_z_of_chi_itp(; omegam=OMEGAM, h_value=H_VALUE, z1=0.0, z2=6.0, nz=100_000)
    local_h0 = 100.0 * h_value
    h_of_z(z) = local_h0 * sqrt(omegam * (1.0 + z)^3 + 1.0 - omegam)
    dchidz(z) = C_KMS / h_of_z(z)

    z_grid = collect(range(z1, z2; length=nz))
    dz = z_grid[2] - z_grid[1]
    chi_grid = similar(z_grid)
    chi_grid[1] = 0.0

    chi_sum = 0.0
    @inbounds for i in 2:length(z_grid)
        chi_sum += 0.5 * (dchidz(z_grid[i - 1]) + dchidz(z_grid[i])) * dz
        chi_grid[i] = chi_sum
    end

    chi_of_z_itp = linear_interpolation(z_grid, chi_grid; extrapolation_bc=Line())
    z_of_chi_itp = linear_interpolation(chi_grid, z_grid; extrapolation_bc=Line())
    return chi_of_z_itp, z_of_chi_itp
end

function m200m_to_m200c_scalar(m200m::Float64, z::Float64)
    one_plus_z = 1.0 + z
    ez_num = OMEGAM * one_plus_z^3
    omegam_z = ez_num / (ez_num + 1.0 - OMEGAM)
    return m200m * omegam_z^0.35
end

function compute_redshift_and_mass(x, y, z, radius, z_of_chi_itp, rho_m)
    n = length(x)
    redshift = Vector{Float64}(undef, n)
    halo_mass = Vector{Float64}(undef, n)
    mass_prefactor = (4.0 * pi / 3.0) * rho_m

    Threads.@threads :static for i in 1:n
        xi = Float64(x[i])
        yi = Float64(y[i])
        zi = Float64(z[i])
        ri = Float64(radius[i])
        chi = sqrt(xi * xi + yi * yi + zi * zi)
        z_halo = z_of_chi_itp(chi)
        redshift[i] = z_halo
        halo_mass[i] = m200m_to_m200c_scalar(mass_prefactor * ri^3, z_halo)
    end

    return redshift, halo_mass
end

function stream_catalog_batches(
    process_batch!::F,
    catalog_source::AbstractString,
    halfdome_path::AbstractString,
    websky_path::AbstractString,
    chunkN::Int,
    z_of_chi_itp,
    rho_m
) where {F}
    if catalog_source == "halfdome"
        return h5open(halfdome_path, "r") do h5
            pos_ds = h5["Position"]
            mass_ds = h5["halo_mass_m200c"]
            redshift_ds = h5["redshift"]
            total_halo_count = size(pos_ds, 2)

            for batch_start in 1:chunkN:total_halo_count
                batch_stop = min(batch_start + chunkN - 1, total_halo_count)
                idx = batch_start:batch_stop
                pos = pos_ds[:, idx]
                x = @view pos[1, :]
                y = @view pos[2, :]
                z = @view pos[3, :]
                halo_mass = Float64.(mass_ds[idx]) ./ H_VALUE
                redshift = Float64.(redshift_ds[idx])
                process_batch!(batch_start, x, y, z, halo_mass, redshift)
            end

            return total_halo_count
        end
    end

    return open(websky_path, "r") do io
        total_halo_count = Int(read(io, Int32))
        rthmax = read(io, Float32)
        redshiftbox = read(io, Float32)
        println("WebSky header: total_halo_count=$(total_halo_count), RTHmax=$(rthmax), redshiftbox=$(redshiftbox)")

        buf = Matrix{Float32}(undef, 10, chunkN)
        batch_start = 1
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
            redshift, halo_mass = compute_redshift_and_mass(x, y, z, radius, z_of_chi_itp, rho_m)

            process_batch!(batch_start, x, y, z, halo_mass, redshift)
            batch_start += nthis
            nleft -= nthis
        end

        return total_halo_count
    end
end

function foreground_halo_mask(halo_mass, redshift; z_max::Float64, mass_min::Float64, mass_max::Float64)
    keep = isfinite.(redshift) .& isfinite.(halo_mass) .& (redshift .>= 0.0) .& (redshift .<= z_max)
    keep .&= halo_mass .> 0.0
    mass_min > 0.0 && (keep .&= halo_mass .>= mass_min)
    isfinite(mass_max) && (keep .&= halo_mass .< mass_max)
    return keep
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

function pixel_centers_to_ra_dec(res, pixels)
    ras = Vector{Float64}(undef, length(pixels))
    decs = Vector{Float64}(undef, length(pixels))

    Threads.@threads :static for i in eachindex(pixels)
        vx, vy, vz = Healpix.pix2vecRing(res, pixels[i])
        theta, phi = Healpix.vec2ang(vx, vy, vz)
        ras[i] = Float64(phi)
        decs[i] = Float64(pi / 2 - theta)
    end

    return ras, decs
end

function ra_dec_to_unit_vectors(ras, decs)
    ux = Vector{Float64}(undef, length(ras))
    uy = Vector{Float64}(undef, length(ras))
    uz = Vector{Float64}(undef, length(ras))

    Threads.@threads :static for i in eachindex(ras)
        ra = Float64(ras[i])
        dec = Float64(decs[i])
        cosdec = cos(dec)
        ux[i] = cosdec * cos(ra)
        uy[i] = cosdec * sin(ra)
        uz[i] = sin(dec)
    end

    return ux, uy, uz
end

function build_frb_pixel_lookup(frb_pixels)
    order = sortperm(frb_pixels)
    return Int.(frb_pixels[order]), Int.(order)
end

function add_if_frb_pixel!(
    local_dm,
    local_hits::Base.RefValue{Int},
    sorted_frb_pixels,
    sorted_frb_indices,
    global_pix::Int,
    halo_ux::Float64,
    halo_uy::Float64,
    halo_uz::Float64,
    frb_ux,
    frb_uy,
    frb_uz,
    frb_redshifts,
    theta_min::Float64,
    theta_max::Float64,
    mass::Float64,
    redshift::Float64,
    dm_model_interp
)
    frb_range = searchsorted(sorted_frb_pixels, global_pix)
    isempty(frb_range) && return nothing

    @inbounds for lookup_idx in frb_range
        frb_idx = sorted_frb_indices[lookup_idx]
        redshift <= frb_redshifts[frb_idx] || continue
        cosang = clamp(halo_ux * frb_ux[frb_idx] + halo_uy * frb_uy[frb_idx] + halo_uz * frb_uz[frb_idx], -1.0, 1.0)
        theta = acos(cosang)
        if theta <= theta_max
            local_dm[frb_idx] += Float64(dm_model_interp(max(theta, theta_min), mass, redshift))
            local_hits[] += 1
        end
    end
    return nothing
end

function accumulate_frb_dm_from_halo_batch!(
    frb_dm,
    workspace,
    dm_model_interp,
    sorted_frb_pixels,
    sorted_frb_indices,
    frb_ux,
    frb_uy,
    frb_uz,
    frb_redshifts,
    x,
    y,
    z,
    masses,
    redshifts
)
    isempty(masses) && return 0

    theta_min = compute_theta_min_local(dm_model_interp)
    nfrb_total = length(frb_dm)
    nthreads_capacity = thread_capacity()
    thread_dm = [zeros(Float64, nfrb_total) for _ in 1:nthreads_capacity]
    thread_hits = zeros(Int, nthreads_capacity)

    Threads.@threads :static for i in eachindex(masses)
        tid = Threads.threadid()
        local_dm = thread_dm[tid]
        local_hits = Ref(0)

        xi = Float64(x[i])
        yi = Float64(y[i])
        zi = Float64(z[i])
        mass_i = Float64(masses[i])
        redshift_i = Float64(redshifts[i])
        radius = sqrt(xi * xi + yi * yi + zi * zi)
        halo_ux = xi / radius
        halo_uy = yi / radius
        halo_uz = zi / radius
        center_theta, center_phi = Healpix.vec2ang(halo_ux, halo_uy, halo_uz)

        theta_max = Float64(compute_theta_max_local(dm_model_interp, mass_i * XGPaint.M_sun, redshift_i))
        if !isfinite(theta_max) || theta_max <= 0.0
            continue
        end

        ring_start, ring_stop = XGPaint.get_relevant_rings(workspace.res, center_theta, theta_max)
        for ring_idx in ring_start:ring_stop
            range1, range2 = XGPaint.get_ring_disc_ranges(workspace, ring_idx, center_theta, center_phi, theta_max)
            first_pixel = workspace.ring_first_pixels[ring_idx]
            for local_pix_idx in range1
                add_if_frb_pixel!(
                    local_dm, local_hits, sorted_frb_pixels, sorted_frb_indices,
                    first_pixel + local_pix_idx - 1,
                    halo_ux, halo_uy, halo_uz, frb_ux, frb_uy, frb_uz, frb_redshifts,
                    theta_min, theta_max, mass_i, redshift_i, dm_model_interp
                )
            end
            for local_pix_idx in range2
                add_if_frb_pixel!(
                    local_dm, local_hits, sorted_frb_pixels, sorted_frb_indices,
                    first_pixel + local_pix_idx - 1,
                    halo_ux, halo_uy, halo_uz, frb_ux, frb_uy, frb_uz, frb_redshifts,
                    theta_min, theta_max, mass_i, redshift_i, dm_model_interp
                )
            end
        end

        thread_hits[tid] += local_hits[]
    end

    for local_dm in thread_dm
        frb_dm .+= local_dm
    end
    return sum(thread_hits)
end

function histogram_pdf(values, edges)
    counts = zeros(Int, length(edges) - 1)
    @inbounds for value in values
        if !isfinite(value) || value < first(edges) || value > last(edges)
            continue
        end
        bin_idx = searchsortedlast(edges, value)
        if bin_idx == 0
            continue
        elseif bin_idx >= length(edges)
            bin_idx = length(edges) - 1
        end
        counts[bin_idx] += 1
    end

    total_count = sum(counts)
    densities = zeros(Float64, length(counts))
    if total_count > 0
        @inbounds for i in eachindex(counts)
            widths = edges[i + 1] - edges[i]
            densities[i] = counts[i] / (total_count * widths)
        end
    end

    centers = 0.5 .* (edges[1:end-1] .+ edges[2:end])
    return centers, densities, counts
end

function write_dm_table(path::AbstractString, redshift_grid, frb_count, frb_pixels_by_redshift, dm_by_redshift)
    output_dir = dirname(path)
    isdir(output_dir) || mkpath(output_dir)
    open(path, "w") do io
        println(io, "redshift,sample_index,pixel,dm")
        for iz in eachindex(redshift_grid)
            z_frb = redshift_grid[iz]
            pixels = frb_pixels_by_redshift[iz]
            dm_values = dm_by_redshift[iz]
            @inbounds for i in 1:frb_count
                println(io, "$(z_frb),$(i),$(pixels[i]),$(dm_values[i])")
            end
        end
    end
    return path
end

function write_summary(path::AbstractString; config_lines, redshift_grid, dm_by_redshift)
    output_dir = dirname(path)
    isdir(output_dir) || mkpath(output_dir)
    open(path, "w") do io
        for line in config_lines
            println(io, line)
        end
        println(io)
        println(io, "redshift,count,mean_dm,std_dm,min_dm,max_dm")
        for iz in eachindex(redshift_grid)
            vals = dm_by_redshift[iz]
            println(
                io,
                "$(redshift_grid[iz]),$(length(vals)),$(mean(vals)),$(std(vals)),$(minimum(vals)),$(maximum(vals))"
            )
        end
    end
    return path
end

function make_pdf_plot(path::AbstractString, redshift_grid, dm_by_redshift; dm_bin_count::Int=60, dm_min=0.0, dm_max=nothing)
    all_dm = reduce(vcat, dm_by_redshift)
    finite_dm = all_dm[isfinite.(all_dm)]
    isempty(finite_dm) && error("No finite DM values available for plotting.")

    lower = Float64(dm_min)
    upper = dm_max === nothing ? maximum(finite_dm) : Float64(dm_max)
    upper > lower || (upper = lower + max(abs(lower), 1.0))
    edges = collect(range(lower, upper; length=dm_bin_count + 1))

    p = plot(
        xlabel="DM [pc cm^-3]",
        ylabel="Estimated PDF p(DM | z_FRB)",
        title="FRB DM PDFs for fixed source redshifts",
        size=(1000, 700),
        legend=:topright,
        grid=true
    )

    for iz in eachindex(redshift_grid)
        values = dm_by_redshift[iz]
        centers, density, _ = histogram_pdf(values, edges)
        label = "z=$(round(redshift_grid[iz]; digits=1)), std=$(round(std(values); digits=3))"
        plot!(p, centers, density; linewidth=2, label=label)
    end

    output_dir = dirname(path)
    isdir(output_dir) || mkpath(output_dir)
    savefig(p, path)
    return path
end

catalog_source = lowercase(get_string_arg("catalog_source", "halfdome"; env=("FRB_PDF_CATALOG_SOURCE", "FRB_CATALOG_SOURCE")))
catalog_source in ("halfdome", "websky") || error("catalog_source must be \"halfdome\" or \"websky\".")

halfdome_path = resolve_halfdome_catalog_path(get_string_arg("halfdome_path", "lightcone_100.hdf5"; env=("FRB_PDF_HALFDOME_PATH", "FRB_HALFDOME_PATH")))
websky_path = resolve_project_path(get_string_arg("websky_path", "other_sims/sims/halos.pksc"; env=("FRB_PDF_WEBSKY_PATH", "FRB_WEBSKY_PATH")))
output_dir = resolve_project_path(get_string_arg("output_dir", joinpath("batched_data", "frb_fixed_redshift_pdfs"); env=("FRB_PDF_OUTPUT_DIR", "FRB_OUTPUT_DIR")))
default_dm_cache_file = joinpath(output_dir, "fixed_redshift_frb_xgpaint_dm_cache.jld2")
dm_cache_file = resolve_project_path(get_string_arg("dm_cache_file", default_dm_cache_file; env=("FRB_PDF_DM_CACHE_FILE", "FRB_DM_CACHE_FILE")))

redshift_grid = parse_float_list(get_string_arg("redshifts", "0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0"; env="FRB_PDF_REDSHIFTS"))
nside = get_int_arg("nside", 4096; env=("FRB_PDF_NSIDE", "FRB_NSIDE"))
chunkN = get_int_arg("chunkN", 1_000_000; env=("FRB_PDF_CHUNKN", "FRB_CHUNKN"))
frb_count = get_int_arg("frb_count", 1000; env=("FRB_PDF_COUNT", "FRB_COUNT"))
frb_seed = get_int_arg("frb_seed", 42; env=("FRB_PDF_SEED", "FRB_SEED"))
reuse_sky_positions = get_bool_arg("reuse_sky_positions", true; env="FRB_PDF_REUSE_SKY_POSITIONS")
halo_mass_min = get_float_arg("halo_mass_min", 0.0; env=("FRB_PDF_HALO_MASS_MIN", "FRB_HALO_MASS_MIN"))
halo_mass_max = get_float_arg("halo_mass_max", Inf; env=("FRB_PDF_HALO_MASS_MAX", "FRB_HALO_MASS_MAX"))
dm_cache_overwrite = get_bool_arg("dm_cache_overwrite", false; env=("FRB_PDF_DM_CACHE_OVERWRITE", "FRB_DM_CACHE_OVERWRITE"))
dm_cleanup_nonpositive = get_bool_arg("dm_cleanup_nonpositive", true; env=("FRB_PDF_DM_CLEANUP_NONPOSITIVE", "FRB_DM_CLEANUP_NONPOSITIVE"))
dm_value_sanity_max = get_float_arg("dm_value_sanity_max", 1.0e8; env=("FRB_PDF_DM_VALUE_SANITY_MAX", "FRB_DM_VALUE_SANITY_MAX"))
dm_bin_count = get_int_arg("dm_bin_count", 60; env="FRB_PDF_DM_BIN_COUNT")
dm_min = get_float_arg("dm_min", 0.0; env="FRB_PDF_DM_MIN")
dm_max_string = get_string_arg("dm_max", "auto"; env="FRB_PDF_DM_MAX")
dm_max = lowercase(strip(dm_max_string)) == "auto" ? nothing : parse(Float64, dm_max_string)

nside > 0 || error("nside must be positive.")
chunkN > 0 || error("chunkN must be positive.")
frb_count > 0 || error("frb_count must be positive.")
dm_bin_count >= 2 || error("dm_bin_count must be at least 2.")
all(>(0.0), redshift_grid) || error("All FRB redshifts must be positive.")
issorted(redshift_grid) || sort!(redshift_grid)
halo_mass_min >= 0.0 || error("halo_mass_min must be nonnegative.")
halo_mass_max > halo_mass_min || error("halo_mass_max must be greater than halo_mass_min.")
isdir(output_dir) || mkpath(output_dir)

catalog_path = catalog_source == "halfdome" ? halfdome_path : websky_path
z_tag = "z" * fmt_param_value(first(redshift_grid)) * "to" * fmt_param_value(last(redshift_grid))
base_tag = "$(catalog_source)_fixedredshift_frb_dm_pdf_$(z_tag)_nfrb$(frb_count)_seed$(frb_seed)_nside$(nside)"
plot_output_path = joinpath(output_dir, "$(base_tag).png")
table_output_path = joinpath(output_dir, "$(base_tag).csv")
summary_output_path = joinpath(output_dir, "$(base_tag)_summary.txt")

println("Fixed-redshift FRB DM PDF configuration:")
println("  catalog_source=$(catalog_source)")
println("  catalog_path=$(catalog_path)")
println("  output_dir=$(output_dir)")
println("  dm_cache_file=$(dm_cache_file)")
println("  dm_cache_overwrite=$(dm_cache_overwrite)")
println("  nside=$(nside), chunkN=$(chunkN)")
println("  redshifts=$(join(redshift_grid, ","))")
println("  frb_count_per_redshift=$(frb_count), frb_seed=$(frb_seed)")
println("  reuse_sky_positions=$(reuse_sky_positions)")
println("  foreground halo mass in [$(halo_mass_min), $(halo_mass_max))")

_, z_of_chi_itp = make_chi_and_z_of_chi_itp()
rho_m = 2.775e11 * OMEGAM * H_VALUE^2
res = Healpix.Resolution(nside)
rng = MersenneTwister(frb_seed)
npix = 12 * nside^2

frb_pixels_by_redshift = Vector{Vector{Int}}(undef, length(redshift_grid))
if reuse_sky_positions
    base_pixels = rand(rng, 1:npix, frb_count)
    for iz in eachindex(redshift_grid)
        frb_pixels_by_redshift[iz] = copy(base_pixels)
    end
else
    for iz in eachindex(redshift_grid)
        frb_pixels_by_redshift[iz] = rand(rng, 1:npix, frb_count)
    end
end

all_frb_pixels = Vector{Int}(undef, frb_count * length(redshift_grid))
all_frb_redshifts = Vector{Float64}(undef, frb_count * length(redshift_grid))
for iz in eachindex(redshift_grid)
    start_idx = (iz - 1) * frb_count + 1
    stop_idx = iz * frb_count
    all_frb_pixels[start_idx:stop_idx] .= frb_pixels_by_redshift[iz]
    all_frb_redshifts[start_idx:stop_idx] .= redshift_grid[iz]
end

frb_ra, frb_dec = pixel_centers_to_ra_dec(res, all_frb_pixels)
frb_ux, frb_uy, frb_uz = ra_dec_to_unit_vectors(frb_ra, frb_dec)
sorted_frb_pixels, sorted_frb_indices = build_frb_pixel_lookup(all_frb_pixels)
frb_dm_all = zeros(Float64, length(all_frb_pixels))

dm_model = HaloDMProfile(BattagliaTauProfile(Omega_c=OMEGAC, Omega_b=OMEGAB, h=H_VALUE))
ENV["XGPAINT_CLEANUP_NONPOSITIVE"] = dm_cleanup_nonpositive ? "true" : "false"
dm_model_interp = build_interpolator(dm_model, cache_file=dm_cache_file, overwrite=dm_cache_overwrite)
workspace = XGPaint.HealpixRingProfileWorkspace{Float64}(res)

processed_halo_count = Ref(0)
los_intersection_count = Ref(0)
max_frb_redshift = maximum(redshift_grid)

println("Accumulating DM for $(length(all_frb_pixels)) total FRB sightlines in one foreground-halo pass...")
total_halo_count = stream_catalog_batches(
    catalog_source,
    halfdome_path,
    websky_path,
    chunkN,
    z_of_chi_itp,
    rho_m
) do _batch_start, x, y, z, halo_mass, redshift
    keep = foreground_halo_mask(halo_mass, redshift; z_max=max_frb_redshift, mass_min=halo_mass_min, mass_max=halo_mass_max)
    any(keep) || return

    xs = Float64.(x[keep])
    ys = Float64.(y[keep])
    zs = Float64.(z[keep])
    masses = Float64.(halo_mass[keep])
    redshifts = Float64.(redshift[keep])

    los_intersection_count[] += accumulate_frb_dm_from_halo_batch!(
        frb_dm_all,
        workspace,
        dm_model_interp,
        sorted_frb_pixels,
        sorted_frb_indices,
        frb_ux,
        frb_uy,
        frb_uz,
        all_frb_redshifts,
        xs,
        ys,
        zs,
        masses,
        redshifts
    )
    processed_halo_count[] += length(masses)
end

max_sampled_dm = maximum(frb_dm_all)
if !isfinite(max_sampled_dm) || max_sampled_dm > dm_value_sanity_max
    error(
        "Sampled FRB DM maximum $(max_sampled_dm) is not physically plausible. " *
        "Check the XGPaint DM profile cache."
    )
end

dm_by_redshift = Vector{Vector{Float64}}(undef, length(redshift_grid))
for iz in eachindex(redshift_grid)
    start_idx = (iz - 1) * frb_count + 1
    stop_idx = iz * frb_count
    dm_by_redshift[iz] = copy(frb_dm_all[start_idx:stop_idx])
end

config_lines = [
    "catalog_source=$(catalog_source)",
    "catalog_path=$(catalog_path)",
    "total_halo_count=$(total_halo_count)",
    "processed_foreground_halo_count=$(processed_halo_count[])",
    "los_intersection_count=$(los_intersection_count[])",
    "nside=$(nside)",
    "frb_count_per_redshift=$(frb_count)",
    "frb_seed=$(frb_seed)",
    "redshifts=$(join(redshift_grid, ","))",
    "reuse_sky_positions=$(reuse_sky_positions)",
    "halo_mass_min=$(halo_mass_min)",
    "halo_mass_max=$(halo_mass_max)",
    "dm_bin_count=$(dm_bin_count)"
]

write_dm_table(table_output_path, redshift_grid, frb_count, frb_pixels_by_redshift, dm_by_redshift)
write_summary(summary_output_path; config_lines=config_lines, redshift_grid=redshift_grid, dm_by_redshift=dm_by_redshift)
make_pdf_plot(plot_output_path, redshift_grid, dm_by_redshift; dm_bin_count=dm_bin_count, dm_min=dm_min, dm_max=dm_max)

println("Processed $(processed_halo_count[]) foreground halos from $(total_halo_count) catalog halos.")
println("Found $(los_intersection_count[]) halo/FRB line-of-sight intersections.")
for iz in eachindex(redshift_grid)
    values = dm_by_redshift[iz]
    println(
        "z=$(redshift_grid[iz]): mean=$(mean(values)), std=$(std(values)), " *
        "min=$(minimum(values)), max=$(maximum(values))"
    )
end
println("Saved DM sample table: $(table_output_path)")
println("Saved summary: $(summary_output_path)")
println("Saved PDF plot: $(plot_output_path)")
