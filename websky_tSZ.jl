using XGPaint, Healpix, Interpolations
include("utils.jl")  # xyz_to_ra_dec
include("SOConvertNFW.jl")

using .M200Convert

const h_value = 0.68
const c_kms = 299_792.458

# -------------------------
# options
# -------------------------
model_exists = false         # set to false to (re)build the model interpolator
save_healpix_map = false    # save Healpix map FITS
save_cl = true              # compute and save power spectrum
apply_mass_cut = true       # apply mass cut

path = "other_sims/sims/halos.pksc"
nside = 4096
chunkN = 2_000_000          # tune to your RAM

add_str_end = "13Msol_cutoff_HALO"
mass_min = 1.0e+13

# output file names (set these to whatever you want)
fits_output_path = "batched_data/websky_tSZ_nside4096_$(add_str_end)_m200m.fits"
cl_output_path = "batched_data/websky_tSZ_cl_m200c_test_phys_param.fits"

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
    return default
end

battaglia_P0_amp = get_float_arg("battaglia_P0_amp", 18.1; env="BATTAGLIA_P0_AMP")
battaglia_P0_alpha_m = get_float_arg("battaglia_P0_alpha_m", 0.154; env="BATTAGLIA_P0_ALPHA_M")
battaglia_P0_alpha_z = get_float_arg("battaglia_P0_alpha_z", -0.758; env="BATTAGLIA_P0_ALPHA_Z")

battaglia_x_c_amp = get_float_arg("battaglia_x_c_amp", 0.497; env="BATTAGLIA_X_C_AMP")
battaglia_x_c_alpha_m = get_float_arg("battaglia_x_c_alpha_m", -0.00865; env="BATTAGLIA_X_C_ALPHA_M")
battaglia_x_c_alpha_z = get_float_arg("battaglia_x_c_alpha_z", 0.731; env="BATTAGLIA_X_C_ALPHA_Z")

battaglia_beta_amp = get_float_arg("battaglia_beta_amp", 4.35; env="BATTAGLIA_BETA_AMP")
battaglia_beta_alpha_m = get_float_arg("battaglia_beta_alpha_m", 0.0393; env="BATTAGLIA_BETA_ALPHA_M")
battaglia_beta_alpha_z = get_float_arg("battaglia_beta_alpha_z", 0.415; env="BATTAGLIA_BETA_ALPHA_Z")

battaglia_alpha_amp = get_float_arg("battaglia_alpha_amp", 1.0; env="BATTAGLIA_ALPHA_AMP")
battaglia_alpha_alpha_m = get_float_arg("battaglia_alpha_alpha_m", 0.0; env="BATTAGLIA_ALPHA_ALPHA_M")
battaglia_alpha_alpha_z = get_float_arg("battaglia_alpha_alpha_z", 0.0; env="BATTAGLIA_ALPHA_ALPHA_Z")

battaglia_gamma_amp = get_float_arg("battaglia_gamma_amp", -0.3; env="BATTAGLIA_GAMMA_AMP")
battaglia_gamma_alpha_m = get_float_arg("battaglia_gamma_alpha_m", 0.0; env="BATTAGLIA_GAMMA_ALPHA_M")
battaglia_gamma_alpha_z = get_float_arg("battaglia_gamma_alpha_z", 0.0; env="BATTAGLIA_GAMMA_ALPHA_Z")

println("Battaglia16 physical parameters:")
println("  P0_amp=$(battaglia_P0_amp), P0_alpha_m=$(battaglia_P0_alpha_m), P0_alpha_z=$(battaglia_P0_alpha_z)")
println("  x_c_amp=$(battaglia_x_c_amp), x_c_alpha_m=$(battaglia_x_c_alpha_m), x_c_alpha_z=$(battaglia_x_c_alpha_z)")
println("  beta_amp=$(battaglia_beta_amp), beta_alpha_m=$(battaglia_beta_alpha_m), beta_alpha_z=$(battaglia_beta_alpha_z)")
println("  alpha_amp=$(battaglia_alpha_amp), alpha_alpha_m=$(battaglia_alpha_alpha_m), alpha_alpha_z=$(battaglia_alpha_alpha_z)")
println("  gamma_amp=$(battaglia_gamma_amp), gamma_alpha_m=$(battaglia_gamma_alpha_m), gamma_alpha_z=$(battaglia_gamma_alpha_z)")

using LinearAlgebra

# -------------------------
# (optional) ratio approximation m200m(z)/m200c(z)
# y = a0 + a1 z + a2 z^2 + a3 z^3 + a4 z^4
# -------------------------
const a0 = 1.3595873806301997
const a1 = -0.49815455039058704
const a2 = 0.3014644154503205
const a3 = -0.08294138910919961
const a4 = 0.0083985355523884
ratio_m200m_over_m200c(z) = a0 + a1*z + a2*z^2 + a3*z^3 + a4*z^4

# -------------------------
# cosmology: chi(z) and z(chi)
# -------------------------
omegab = 0.049
omegac = 0.31 - omegab
omegam = omegab + omegac
omegal =  1 - omegam
h      = 0.68
H0 = 100*h

s = 0.0
function make_z_of_chi_itp(; omegam, h_value, z1=0.0, z2=6.0, nz=100_000)
    H0 = 100*h_value
    H(z) = H0 * sqrt(omegam*(1+z)^3 + 1 - omegam)
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


# -------------------------
# density + selection
# -------------------------
ρ = 2.775e11 * omegam * h^2  # Msun / Mpc^3
selection = apply_mass_cut

# -------------------------
# model + map init
# -------------------------
model = Battaglia16ThermalSZProfile(Omega_c=omegac, Omega_b=omegab, h=h_value,
P0_amp=battaglia_P0_amp, P0_alpha_m=battaglia_P0_alpha_m, P0_alpha_z=battaglia_P0_alpha_z,
    x_c_amp=battaglia_x_c_amp, x_c_alpha_m=battaglia_x_c_alpha_m, x_c_alpha_z=battaglia_x_c_alpha_z,
    beta_amp=battaglia_beta_amp, beta_alpha_m=battaglia_beta_alpha_m, beta_alpha_z=battaglia_beta_alpha_z,
    alpha_amp=battaglia_alpha_amp, alpha_alpha_m=battaglia_alpha_alpha_m, alpha_alpha_z=battaglia_alpha_alpha_z,
    gamma_amp=battaglia_gamma_amp, gamma_alpha_m=battaglia_gamma_alpha_m, gamma_alpha_z=battaglia_gamma_alpha_z
)

if model_exists
    y_model_interp = build_interpolator(
        model,
        cache_file="cached_tSZ_Websky_cosmo_default.jld2",
        overwrite=false
    )
else
    y_model_interp = build_interpolator(
    model;
    cache_file = "cached_tSZ_Websky_cosmo_default.jld2",
    N_logθ     = 512,
    pad        = 256,
    logM_max   = 15.7,
    overwrite  = true,
    verbose    = true,
)
end

m_hp = HealpixMap{Float64,RingOrder}(nside)
res  = Healpix.Resolution(nside)
w    = XGPaint.HealpixRingProfileWorkspace{Float64}(res)

println("Initiating HealPix with NSide: $nside")

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
        i1 = i0 + nthis - 1   # <-- define i1


        # read exactly 10*nthis Float32 into the buffer
        rawview = @view reinterpret(Float32, vec(buf))[1:10*nthis]
        read!(io, rawview)

        cat = @view buf[:, 1:nthis]

        x  = @view cat[1, :];  y  = @view cat[2, :];  z  = @view cat[3, :]
        vx = @view cat[4, :];  vy = @view cat[5, :];  vz = @view cat[6, :]
        R  = @view cat[7, :]

        # chi and redshift
        chi = sqrt.(Float64.(x).^2 .+ Float64.(y).^2 .+ Float64.(z).^2)
        redshift = itp_z_of_chi.(chi)

        # ρ = 2.775e11 * h^2 .* (omegam .* (1 .+ redshift).^3)  # Msun / Mpc^3


        halo_mass = (4π/3) * ρ .* (Float64.(R) .^ 3)   # Float64 for XGPaint


        # Test with h units

        # x .= x .* (1 .+ redshift)
        # y .= y .* (1 .+ redshift)
        # z .= z .* (1 .+ redshift)


        halo_mass = M200Convert.m200m_to_m200c_arrays(halo_mass, redshift)

        # OPTIONAL: if your mass is actually M200m and you want to convert to M200c:
        # halo_mass .= halo_mass ./ ratio_m200m_over_m200c.(redshift)   # M200m to M200c
        
        # Msun are the right coordinates
        # halo_mass .= halo_mass .* h_value    # Msun to Msun/h 

        # mass cut (do it now to reduce work)
        if selection
            sel = halo_mass .>= mass_min
            if !any(sel)
                nleft -= nthis
                i0 += nthis
                continue
            end

            # materialize selected subset as Float64 vectors (avoid holding full chunk)
            xs  = Float64.(x[sel]);  ys  = Float64.(y[sel]);  zs  = Float64.(z[sel])
            vxs = Float64.(vx[sel]); vys = Float64.(vy[sel]); vzs = Float64.(vz[sel])
            ms  = halo_mass[sel]
            zsft = redshift[sel]

            ra, dec = xyz_to_ra_dec(xs, ys, zs)


            # sort by dec (like your original)
            perm = sortperm(dec)
            ra  = ra[perm]
            dec = dec[perm]
            zsft = zsft[perm]
            ms   = ms[perm]

            # robust accumulation: paint chunk into tmp, then add to global
            fill!(tmp_hp.pixels, 0.0)
            paint!(tmp_hp, w, y_model_interp, ms, zsft, ra, dec)
            m_hp.pixels .+= tmp_hp.pixels
        else
            # no selection: still avoid global arrays, but need Float64 vectors
            xs  = Float64.(x);  ys  = Float64.(y);  zs = Float64.(z)
            vxs = Float64.(vx); vys = Float64.(vy); vzs = Float64.(vz)
            ms  = halo_mass
            zsft = redshift


            ra, dec = xyz_to_ra_dec(xs, ys, zs)
    
            perm = sortperm(dec)
            ra  = ra[perm]
            dec = dec[perm]
            zsft = zsft[perm]
            ms   = ms[perm]


            fill!(tmp_hp.pixels, 0.0)
            paint!(tmp_hp, w, y_model_interp, ms, zsft, ra, dec)
            m_hp.pixels .+= tmp_hp.pixels

            # ratio_done = round(100 * i1 / N; digits=2)
            # print("Painted halos $ratio_done % (halos $i0 to $i1)\n")
        end

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


println("Finished Healpix tSZ total (BATCHED)")
