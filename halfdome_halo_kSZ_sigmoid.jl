using XGPaint, Plots, Pixell, HDF5, Healpix, Unitful, UnitfulAstro
include("utils.jl")  # brings xyz_to_ra_dec into Main

const c_kms = 299_792.458  
const h_value = 0.6774

h5 = h5open("lightcone_100.hdf5", "r")

pos = read(h5["Position"])

halo_mass = read(h5["halo_mass_m200c"])*h_value # Correction for just Msol units for websky comparison 

halo_vel = read(h5["Velocity"])

redshift = read(h5["redshift"])


# Hard cut-off for mass 

selection = true
add_str_end = "13Msol_cutoff_FIELD"
mass_min = 1e13        # Msun, WebSky threshold
sel = halo_mass .>= mass_min

if selection == true
	pos = pos[:, sel]
	halo_mass = halo_mass[sel]
	halo_vel = halo_vel[:, sel]
	redshift = redshift[sel]
	end

b = 10.0 .^ (11:0.25:17)

x = pos[1,:]
y = pos[2,:]
z = pos[3,:]

vx = halo_vel[1, :]
vy = halo_vel[2, :]
vz = halo_vel[3, :]

### For projected velocity
N = length(x)
proj_v_over_c = Array{eltype(x)}(undef, N)

for i in 1:N
    r = sqrt(x[i]^2 + y[i]^2 + z[i]^2)
    nx, ny, nz = x[i]/r, y[i]/r, z[i]/r            # LOS unit vector
    v_los = vx[i]*nx + vy[i]*ny + vz[i]*nz        # projected velocity along LOS
    proj_v_over_c[i] = v_los / c_kms
end

ra, dec = xyz_to_ra_dec(x, y, z)

# radians to degrees
ra_deg  = rad2deg.(ra)
dec_deg = rad2deg.(dec)

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
y_model_interp = build_interpolator(model, cache_file="cached2_model_sigmoid_Ntheta512_pad256_integral_reduced_acc.jld2", overwrite=false)

# y_model_interp = build_interpolator(model, cache_file="cached_test_no_sigmoid.jld2", overwrite=true)

nside = 4096
m_hp = HealpixMap{Float64,RingOrder}(nside)
res = Healpix.Resolution(nside)
print("Initiating the HealPix with NSide: $nside \n")

w = XGPaint.HealpixRingProfileWorkspace{Float64}(res)

@time paint!(m_hp, w, y_model_interp, halo_mass, redshift, ra, dec, proj_v_over_c)
Healpix.saveToFITS(m_hp, "!y_tSZ_nside4096_sigmoid_m1e13_corr_units.fits", typechar="D")  

plot(m_hp)
savefig("kSZ_HalfDome_y100_sigmoid_m1e13.png")
print("Finished Healpix kSZ total with sigmoid \n")
