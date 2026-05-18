using XGPaint, Healpix, Interpolations
include("utils.jl")  # xyz_to_ra_dec
include("SOConvertNFW.jl")

using Unitful: ustrip
using LinearAlgebra
using Base.Threads
using .M200Convert

const c_kms = 299_792.458
const h_value = 0.6774

# -------------------------
# options
# -------------------------
model_exists = false         # set to false to (re)build the model interpolator
save_healpix_map = false     # save Healpix map FITS
save_cl = true               # compute and save power spectrum
apply_mass_cut = true        # apply mass cut
apply_ang_cut = false        # apply angular size cut

t0 = time()

path = "other_sims/sims/halos.pksc"
nside = 4096
chunkN = 20_000_000          # tune to your RAM

add_str_end = "13Msol_cutoff_HALO"
mass_min = 1.0e13

# output file names are finalized after parameter tag is built

# -------------------------
# Battaglia model parameters (editable)
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

const BATTAGLIA_P0_AMP_DEFAULT = 4.e3
const BATTAGLIA_P0_ALPHA_M_DEFAULT = 0.29
const BATTAGLIA_P0_ALPHA_Z_DEFAULT = -0.66
const BATTAGLIA_X_C_AMP_DEFAULT = 0.5
const BATTAGLIA_X_C_ALPHA_M_DEFAULT = 0
const BATTAGLIA_X_C_ALPHA_Z_DEFAULT = 0
const BATTAGLIA_BETA_AMP_DEFAULT = 3.83
const BATTAGLIA_BETA_ALPHA_M_DEFAULT = 0.04
const BATTAGLIA_BETA_ALPHA_Z_DEFAULT = -0.025
const BATTAGLIA_ALPHA_AMP_DEFAULT = 0.88
const BATTAGLIA_ALPHA_ALPHA_M_DEFAULT = -0.03
const BATTAGLIA_ALPHA_ALPHA_Z_DEFAULT = 0.19
const BATTAGLIA_GAMMA_AMP_DEFAULT = -0.2
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

param_tag = build_param_tag()
add_str_end = add_str_end * "_" * param_tag

fits_output_path = "batched_data/websky_kSZ_nside$(nside)_$(add_str_end)_m200c_htest_BATCHED_mtimesh_ADRIAN_FIX.fits"
cl_output_path = "batched_data/websky_kSZ_cl_test_phys_param_$(param_tag)_nside$(nside)_ADRIAN_FIX_f_e09.fits"
cache_file_path = "cached_kSZ_Websky_tau_$(param_tag)_f_.jld2"

println("Battaglia physical parameters:")
println("  P0_amp=$(battaglia_P0_amp), P0_alpha_m=$(battaglia_P0_alpha_m), P0_alpha_z=$(battaglia_P0_alpha_z)")
println("  x_c_amp=$(battaglia_x_c_amp), x_c_alpha_m=$(battaglia_x_c_alpha_m), x_c_alpha_z=$(battaglia_x_c_alpha_z)")
println("  beta_amp=$(battaglia_beta_amp), beta_alpha_m=$(battaglia_beta_alpha_m), beta_alpha_z=$(battaglia_beta_alpha_z)")
println("  alpha_amp=$(battaglia_alpha_amp), alpha_alpha_m=$(battaglia_alpha_alpha_m), alpha_alpha_z=$(battaglia_alpha_alpha_z)")
println("  gamma_amp=$(battaglia_gamma_amp), gamma_alpha_m=$(battaglia_gamma_alpha_m), gamma_alpha_z=$(battaglia_gamma_alpha_z)")

# -------------------------
# (optional) ratio approximation m200m(z)/m200c(z)
# y = a0 + a1 z + a2 z^2 + a3 z^3 + a4 z^4
# -------------------------
const a0 = 1.3595873806301997
const a1 = -0.49815455039058704
const a2 = 0.3014644154503205
const a3 = -0.08294138910919961
const a4 = 0.0083985355523884
ratio_m200m_over_m200c(z) = a0 + a1 * z + a2 * z^2 + a3 * z^3 + a4 * z^4

# -------------------------
# cosmology: chi(z) and z(chi)
# -------------------------
omegab = 0.049
omegac = 0.261
omegam = omegab + omegac
omegal = 1.0 - omegam
h = 0.68
H0 = 100 * h

s = 0.0
function make_z_of_chi_itp(; omegam, h_value, z1=0.0, z2=6.0, nz=100_000)
    H0 = 100 * h_value
    H(z) = H0 * sqrt(omegam * (1 + z)^3 + 1 - omegam)
    dchidz(z) = c_kms / H(z)

    za = collect(range(z1, z2; length=nz))
    dz = za[2] - za[1]
    chia = similar(za)

    s = 0.0
    @inbounds for i in eachindex(za)
        s += dchidz(za[i]) * dz
        chia[i] = s
    end

    return linear_interpolation(chia, za; extrapolation_bc=Line())
end

itp_z_of_chi = make_z_of_chi_itp(omegam=omegam, h_value=h_value)

function m200m_to_m200c(m200m, z)
    omegamz = omegam .* (1 .+ z) .^ 3 ./ (omegam .* (1 .+ z) .^ 3 .+ 1 .- omegam)
    return m200m .* omegamz .^ 0.35
end

@inline function m200m_to_m200c_scalar(m200m::Float64, z::Float64)
    one_plus_z = 1.0 + z
    ez_num = omegam * one_plus_z^3
    omegamz = ez_num / (ez_num + 1.0 - omegam)
    return m200m * omegamz^0.35
end

function compute_redshift_and_mass(x, y, z, R, itp_z_of_chi, rho_m)
    n = length(x)
    redshift = Vector{Float64}(undef, n)
    halo_mass = Vector{Float64}(undef, n)
    mass_prefactor = (4.0 * pi / 3.0) * rho_m

    @threads for i in 1:n
        xi = Float64(x[i])
        yi = Float64(y[i])
        zi = Float64(z[i])
        ri = Float64(R[i])

        chi = sqrt(xi * xi + yi * yi + zi * zi)
        zi_redshift = itp_z_of_chi(chi)

        redshift[i] = ifelse(isfinite(zi_redshift), zi_redshift, NaN)
        halo_mass[i] = m200m_to_m200c_scalar(mass_prefactor * ri^3, redshift[i])
    end

    return redshift, halo_mass
end

function xyz_to_ra_dec_threaded(x::AbstractVector{T}, y::AbstractVector{T}, z::AbstractVector{T}) where {T}
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

function filter_valid_kinematic_data(xs, ys, zs, vxs, vys, vzs, ms, zsft)
    r = sqrt.(xs .* xs .+ ys .* ys .+ zs .* zs)
    good = (r .> 0) .& isfinite.(r) .& isfinite.(zsft) .& isfinite.(ms)

    if !all(good)
        xs = xs[good]
        ys = ys[good]
        zs = zs[good]
        vxs = vxs[good]
        vys = vys[good]
        vzs = vzs[good]
        ms = ms[good]
        zsft = zsft[good]
        r = r[good]
    end

    return xs, ys, zs, vxs, vys, vzs, ms, zsft, r
end

function compute_proj_v_over_c(xs, ys, zs, vxs, vys, vzs, r, zsft)
    proj_v_over_c = (xs .* vxs .+ ys .* vys .+ zs .* vzs) ./ r ./ c_kms
    proj_v_over_c .= ifelse.(isfinite.(proj_v_over_c), proj_v_over_c, 0.0)
    return proj_v_over_c
end

# -------------------------
# density + selection
# -------------------------
rho_m = 2.775e11 * omegam * h^2
selection = apply_mass_cut || apply_ang_cut

# -------------------------
# model + map init
# -------------------------
model = BattagliaTauProfile(
    Omega_c=omegac, Omega_b=omegab, h=h_value,
    P0_amp=battaglia_P0_amp, P0_alpha_m=battaglia_P0_alpha_m, P0_alpha_z=battaglia_P0_alpha_z,
    x_c_amp=battaglia_x_c_amp, x_c_alpha_m=battaglia_x_c_alpha_m, x_c_alpha_z=battaglia_x_c_alpha_z,
    beta_amp=battaglia_beta_amp, beta_alpha_m=battaglia_beta_alpha_m, beta_alpha_z=battaglia_beta_alpha_z,
    alpha_amp=battaglia_alpha_amp, alpha_alpha_m=battaglia_alpha_alpha_m, alpha_alpha_z=battaglia_alpha_alpha_z,
    gamma_amp=battaglia_gamma_amp, gamma_alpha_m=battaglia_gamma_alpha_m, gamma_alpha_z=battaglia_gamma_alpha_z
)

# model = BattagliaTauProfile(Omega_c=0.2603, Omega_b=0.0486, h=0.6774)

# y_model_interp = build_interpolator(
#     model;
#     cache_file="battaglia_tau_interpolator_adrian_fix.jld2",
#     overwrite=false,
#     verbose=true,
# )

if model_exists
    y_model_interp = build_interpolator(
        model,
        cache_file=cache_file_path,
        overwrite=false
    )
else
    y_model_interp = build_interpolator(
        model;
        cache_file=cache_file_path,
        N_logθ     = 512,
        pad        = 256,
        logM_max   = 15.7,
        overwrite  = true,
        verbose    = true,
    )
end

m_hp = HealpixMap{Float64,RingOrder}(nside)
res = Healpix.Resolution(nside)
w = XGPaint.HealpixRingProfileWorkspace{Float64}(res)

println("Initiating HealPix with NSide: $nside")
println("Julia threads available: $(nthreads())")

# If you previously observed "only last batch remains", use temp accumulation:
tmp_hp = HealpixMap{Float64,RingOrder}(nside)

# -------------------------
# stream halos.pksc in batches
# -------------------------
open(path, "r") do io
    N = Int(read(io, Int32))
    RTHmax = read(io, Float32)
    redshiftbox = read(io, Float32)
    @show N RTHmax redshiftbox

    buf = Matrix{Float32}(undef, 10, chunkN)

    nleft = N
    i0 = 1
    while nleft > 0
        nthis = min(chunkN, nleft)
        i1 = i0 + nthis - 1

        # read exactly 10*nthis Float32 into the buffer
        rawview = @view reinterpret(Float32, vec(buf))[1:10 * nthis]
        read!(io, rawview)

        cat = @view buf[:, 1:nthis]

        x = @view cat[1, :]
        y = @view cat[2, :]
        z = @view cat[3, :]
        vx = @view cat[4, :]
        vy = @view cat[5, :]
        vz = @view cat[6, :]
        R = @view cat[7, :]

        redshift, halo_mass = compute_redshift_and_mass(x, y, z, R, itp_z_of_chi, rho_m)

        if selection
            finite_mask = isfinite.(halo_mass) .& isfinite.(redshift)
            sel_mass = apply_mass_cut ? ((halo_mass .>= mass_min) .& finite_mask) : finite_mask

            sel_ang = trues(length(halo_mass))
            if apply_ang_cut
                r200 = XGPaint.r200c_comoving.(Ref(model), halo_mass, redshift)
                chi_ang = XGPaint.angular_diameter_dist.(Ref(model.cosmo), redshift)

                ang_ok = isfinite.(r200) .& isfinite.(chi_ang) .& (chi_ang .> zero.(chi_ang))
                theta200 = ustrip.(r200 ./ chi_ang)
                theta_cut = 0.5 * (pi / 180) / 60

                sel_ang = ang_ok .& isfinite.(theta200) .& (theta200 .> theta_cut)
            end

            sel = sel_mass .& sel_ang
            if !any(sel)
                nleft -= nthis
                i0 += nthis
                continue
            end

            xs = Float64.(x[sel])
            ys = Float64.(y[sel])
            zs = Float64.(z[sel])
            vxs = Float64.(vx[sel])
            vys = Float64.(vy[sel])
            vzs = Float64.(vz[sel])
            ms = halo_mass[sel]
            zsft = redshift[sel]
        else
            xs = Float64.(x)
            ys = Float64.(y)
            zs = Float64.(z)
            vxs = Float64.(vx)
            vys = Float64.(vy)
            vzs = Float64.(vz)
            ms = halo_mass
            zsft = redshift
        end

        xs, ys, zs, vxs, vys, vzs, ms, zsft, r = filter_valid_kinematic_data(
            xs, ys, zs, vxs, vys, vzs, ms, zsft
        )

        if isempty(ms)
            nleft -= nthis
            i0 += nthis
            continue
        end

        proj_v_over_c = compute_proj_v_over_c(xs, ys, zs, vxs, vys, vzs, r, zsft)
        ra, dec = xyz_to_ra_dec_threaded(xs, ys, zs)

        # sort by dec (like your original)
        perm = sortperm(dec)
        ra = ra[perm]
        dec = dec[perm]
        zsft = zsft[perm]
        ms = ms[perm]
        proj_v_over_c = proj_v_over_c[perm]
        # proj_v_over_c .= proj_v_over_c ./ (1 .+ zsft)
        # robust accumulation: paint chunk into tmp, then add to global
        fill!(tmp_hp.pixels, 0.0)
        paint!(tmp_hp, w, y_model_interp, ms, zsft, ra, dec, proj_v_over_c)
        m_hp.pixels .+= tmp_hp.pixels

        # ratio_done = round(100 * i1 / N; digits=2)
        # print("Painted halos $ratio_done % (halos $i0 to $i1)\n")

        nleft -= nthis
        i0 += nthis
    end
end

isdir("batched_data") || mkpath("batched_data")

if save_healpix_map
    Healpix.saveToFITS(
        m_hp,
        "!" * fits_output_path,
        typechar="D"
    )
end

if save_cl
    cl = anafast(m_hp, niter=0)
    writeClToFITS(cl_output_path, collect(cl); overwrite=true)
end

println("Finished Healpix kSZ total (BATCHED)")
elapsed = time() - t0
println("Elapsed time: $(round(elapsed; digits=2)) s")
