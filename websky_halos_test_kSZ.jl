using XGPaint, Healpix, Interpolations
include("utils.jl")  # xyz_to_ra_dec
include("SOConvertNFW.jl")
using Unitful
using LinearAlgebra
using .M200Convert

# -------------------------
# options
# -------------------------
model_exists = false         # set to false to (re)build the model interpolator
save_healpix_map = false    # save Healpix map FITS
save_cl = true              # compute and save power spectrum
apply_mass_cut = true       # apply mass cut
apply_ang_cut = true        # apply angular size cut


const c_kms = 299_792.458
const h_value = 0.6774

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
omegac = 0.261
omegam = omegab + omegac
omegal = 1.0 - omegam
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
# selection
# -------------------------



selection = apply_mass_cut || apply_ang_cut
add_str_end = "13Msol_cutoff_HALO"
mass_min = 1e13

# -------------------------
# model + map init
# -------------------------
model = BattagliaTauProfile(Omega_c=omegac, Omega_b=omegab, h=h_value) 
if model_exists
    y_model_interp = build_interpolator(
        model,
        cache_file="cached_model_Ntheta512_pad256_logM_max.jld2",
        overwrite=false
    )
else
    y_model_interp = build_interpolator(
    model;
    cache_file = "cached_model_Ntheta512_pad256_logM_max.jld2",
    N_logθ     = 512,
    pad        = 256,
    logM_max   = 15.7,
    overwrite  = true,
    verbose    = true,
)
end

nside = 4096
m_hp = HealpixMap{Float64,RingOrder}(nside)
res  = Healpix.Resolution(nside)
w    = XGPaint.HealpixRingProfileWorkspace{Float64}(res)

println("Initiating HealPix with NSide: $nside")

# If you previously observed "only last batch remains", use temp accumulation:
tmp_hp = HealpixMap{Float64,RingOrder}(nside)

# -------------------------
# stream halos.pksc in batches
# -------------------------
path = "other_sims/sims/halos.pksc"

open(path, "r") do io
    N = Int(read(io, Int32))
    RTHmax = read(io, Float32)
    redshiftbox = read(io, Float32)
    @show N RTHmax redshiftbox

    chunkN = 10_000_000                     # tune to your RAM
    buf = Matrix{Float32}(undef, 10, chunkN)

    nleft = N
    i0 = 1
    while nleft > 0
        nthis = min(chunkN, nleft)

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
        # Guard against any non-finite redshifts before using them
        redshift .= ifelse.(isfinite.(redshift), redshift, NaN)
        # E = sqrt.(omegam .* (1 .+ redshift).^3 .+ omegal)
        #ρ = 2.775e11 * h^2 .* (E.^2)  # Msun / Mpc^3
        ρ = 2.775e11*omegam*h^2 # Msun/Mpc^3

        # M200m 
        halo_mass = (4π/3) * ρ .* (Float64.(R) .^ 3)   # Float64 for XGPaint
        # To m200c in Msol units
        halo_mass = M200Convert.m200m_to_m200c_arrays(halo_mass, redshift)

        # OPTIONAL: if the mass is M200m and you want to convert to M200c:
        # halo_mass .= halo_mass ./ ratio_m200m_over_m200c.(redshift)   # M200m to M200c


        # halo_mass .= halo_mass .* h_value    # Msun to Msun/h

        # mass/angle cuts
        if selection
            # basic finite mask first
            finite_mask = isfinite.(halo_mass) .& isfinite.(redshift)
            sel_mass = apply_mass_cut ? ((halo_mass .>= mass_min) .& finite_mask) : finite_mask
            
            # This is the angular size cut
            θcut = 0.5 * (π / 180) / 60  # radians

            E2(z) = omegam*(1+z)^3 + omegal
            ρcrit(z) = ρcrit0 * E2(z) / omegam

            sel_ang = trues(length(halo_mass))
            if apply_ang_cut
                # --- apply selection ---
                r200 = XGPaint.r200c_comoving.(Ref(model), halo_mass, redshift)
                χ    = XGPaint.angular_diameter_dist.(Ref(model.cosmo), redshift)

                # Guard angular-size calculation against non-finite values or zero distance
                ang_ok = isfinite.(r200) .& isfinite.(χ) .& (χ .> zero.(χ))
                θ200 = ustrip.(r200 ./ χ)   # unitless
                θcut = 0.5 * (π/180)/60

                sel_ang = ang_ok .& isfinite.(θ200) .& (θ200 .> θcut)
            end

            sel = sel_mass .& sel_ang   # combined mask on full chunk


            if !any(sel)
                nleft -= nthis
                i0 += nthis
                continue
            end

            # materialize selected subset as spl_z_of_chiFloat64 vectors (avoid holding full chunk)
            xs  = Float64.(x[sel]);  ys  = Float64.(y[sel]);  zs  = Float64.(z[sel])
            vxs = Float64.(vx[sel]); vys = Float64.(vy[sel]); vzs = Float64.(vz[sel])
            ms  = halo_mass[sel]
            zsft = redshift[sel]

            # projected v/c (vectorized) with safety against r==0 or non-finite
            r = sqrt.(xs .* xs .+ ys .* ys .+ zs .* zs)
            good = (r .> 0) .& isfinite.(r) .& isfinite.(zsft)
            if !all(good)
                xs = xs[good]; ys = ys[good]; zs = zs[good]
                vxs = vxs[good]; vys = vys[good]; vzs = vzs[good]
                ms = ms[good]; zsft = zsft[good]
                r = r[good]
            end
            proj_v_over_c = (xs .* vxs .+ ys .* vys .+ zs .* vzs) ./ r ./ c_kms
            proj_v_over_c .= ifelse.(isfinite.(proj_v_over_c), proj_v_over_c, 0.0)

            ra, dec = xyz_to_ra_dec(xs, ys, zs)

            # sort by dec (like your original)
            perm = sortperm(dec)
            ra  = ra[perm]
            dec = dec[perm]
            zsft = zsft[perm]
            ms   = ms[perm]
            proj_v_over_c = proj_v_over_c[perm]
            den = 1 .+ zsft
            proj_v_over_c .= proj_v_over_c ./ den  # comoving units
            proj_v_over_c .= ifelse.(isfinite.(proj_v_over_c), proj_v_over_c, 0.0)


            # robust accumulation: paint chunk into tmp, then add to global
            fill!(tmp_hp.pixels, 0.0)
            paint!(tmp_hp, w, y_model_interp, ms, zsft, ra, dec, proj_v_over_c)
            m_hp.pixels .+= tmp_hp.pixels
        else
            # no selection: still avoid global arrays, but need Float64 vectors
            xs  = Float64.(x);  ys  = Float64.(y);  zs = Float64.(z)
            vxs = Float64.(vx); vys = Float64.(vy); vzs = Float64.(vz)
            ms  = halo_mass
            zsft = redshift

            r = sqrt.(xs .* xs .+ ys .* ys .+ zs .* zs)
            good = (r .> 0) .& isfinite.(r) .& isfinite.(zsft) .& isfinite.(ms)
            if !all(good)
                xs = xs[good]; ys = ys[good]; zs = zs[good]
                vxs = vxs[good]; vys = vys[good]; vzs = vzs[good]
                ms = ms[good]; zsft = zsft[good]
                r = r[good]
            end
            proj_v_over_c = (xs .* vxs .+ ys .* vys .+ zs .* vzs) ./ r ./ c_kms
            proj_v_over_c .= ifelse.(isfinite.(proj_v_over_c), proj_v_over_c, 0.0)
            ra, dec = xyz_to_ra_dec(xs, ys, zs)

            perm = sortperm(dec)
            ra  = ra[perm]
            dec = dec[perm]
            zsft = zsft[perm]
            ms   = ms[perm]
            proj_v_over_c = proj_v_over_c[perm]
            den = 1 .+ zsft
            proj_v_over_c .= proj_v_over_c ./ den  # comoving units
            proj_v_over_c .= ifelse.(isfinite.(proj_v_over_c), proj_v_over_c, 0.0)

            fill!(tmp_hp.pixels, 0.0)
            paint!(tmp_hp, w, y_model_interp, ms, zsft, ra, dec, proj_v_over_c)
            m_hp.pixels .+= tmp_hp.pixels

            ratio_done = round(100 * i1 / Ntot; digits=2)
            print("Painted halos $ratio_done % (halos $i0 to $i1)\n")
        end

        nleft -= nthis
        i0 += nthis
    end
end

isdir("batched_data") || mkpath("batched_data")

if save_healpix_map
    Healpix.saveToFITS(
        m_hp,
        "!batched_data/websky_kSZ_nside4096_sigmoid_$(add_str_end)_m200c_htest_BATCHED_mtimesh.fits",
        typechar="D"
    )
end

if save_cl
    cl = anafast(m_hp, niter=0)
    writeClToFITS("batched_data/websky_kSZ_cl_velocity_div_1pz.fits",
        collect(cl); overwrite=true)
end


println("Finished Healpix kSZ total with sigmoid (BATCHED)")
