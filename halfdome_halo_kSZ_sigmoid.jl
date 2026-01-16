using XGPaint, Plots, Pixell, HDF5, Healpix, Unitful, UnitfulAstro
include("utils.jl")  # brings xyz_to_ra_dec into Main

const c_kms = 299_792.458  
const h_value = 0.6774

h5 = h5open("lightcone_100.hdf5", "r")

pos = read(h5["Position"])

halo_mass = read(h5["halo_mass_m200c"]) / h_value # Correction for just Msol units for websky comparison 

halo_vel = read(h5["Velocity"])

redshift = read(h5["redshift"])

# Hard cut-off for mass 
selection = true
add_str_end = "13Msol_cutoff_HALO"
mass_min = 1e13        # Msun, WebSky threshold
sel = halo_mass .>= mass_min

if selection == true
	pos = pos[:, sel]
	halo_mass = halo_mass[sel]
	halo_vel = halo_vel[:, sel]
	redshift = redshift[sel]
	end

x = pos[1,:]
y = pos[2,:]
z = pos[3,:]

vx = halo_vel[1, :]
vy = halo_vel[2, :]
vz = halo_vel[3, :]

a = @. 1 / (1 + redshift)


# projected v/c (vectorized)
r = sqrt.(x .* x .+ y .* y .+ z .* z 
proj_v_over_c = (x .* vx .+ y .* vy .+ z .* vz) ./ r ./ c_kms

#proj_v_over_c = @. proj_v_over_c / a   # include scale factor
print("Finished calculating projected velocities with scale factor \n")
ra, dec = xyz_to_ra_dec(x, y, z)

# sorting all the arrays by dec 
perm = sortperm(dec)  # sortperm(dec, alg=ThreadsX.MergeSort)
ra = ra[perm]
dec = dec[perm]
redshift = redshift[perm]
halo_mass = halo_mass[perm]
proj_v_over_c = proj_v_over_c[perm]


# For kSZ no sigmoid
# model = BattagliaTauProfile(Omega_c=0.2603, Omega_b=0.0486,  h=0.6774)

# with Sigmoid
model = SigmoidBattagliaTauProfile(Omega_c=0.2603, Omega_b=0.0486,  h=h_value)

# y_model_interp = XGPaint.load_precomputed_battaglia_tau()
# Sigmoid file name: cached_model_sigmoid_r0_3_Ntheta512_pad256_integral_increased_acc
# Battaglia model:cached_model_Ntheta512_pad256_integral_increased_acc
y_model_interp = build_interpolator(model, cache_file="cached_model_sigmoid_r0_3_Ntheta512_pad256_integral_increased_acc.jld2", overwrite=false)
# y_model_interp = build_interpolator(model, cache_file="cached_test_no_sigmoid.jld2", overwrite=true)

nside = 4096
m_hp = HealpixMap{Float64,RingOrder}(nside)
res = Healpix.Resolution(nside)
print("Initiating the HealPix with NSide: $nside \n")

w = XGPaint.HealpixRingProfileWorkspace{Float64}(res)

@time paint!(m_hp, w, y_model_interp, halo_mass, redshift, ra, dec, proj_v_over_c)
Healpix.saveToFITS(m_hp, "!HalfDome_kSZ_nside4096_sigmoid_r0_3_position_hcorr_$(add_str_end).fits", typechar="D")  
print("Finished Healpix kSZ total with sigmoid \n")
