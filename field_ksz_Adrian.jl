# arg 1 is mass cut, arg 2 is whether to use sigmoid or not
if length(ARGS) >= 1
    mass_cut = parse(Float64, ARGS[1])
else
    mass_cut = 0
end
do_sigmoid = false   # default
if length(ARGS) >= 2 && ARGS[2] == "sigmoid"
    do_sigmoid = true
end

println("sigmoiding enabled? ", do_sigmoid)

h = 0.6774
res = 1
seed = 100
fname = "/mnt/ceph/users/abayer/fastpm/halfdome/stampede2_3750Mpch_6144cube/final_res/halos/lightcone_$(seed).hdf5"
outpath = "/mnt/ceph/users/abayer/fastpm/halfdome/oneweek/ksz/"
if do_sigmoid
    outfile_ksz = joinpath(outpath, "TkSZ_div_TCMB_halo_res$(res)_s$(seed)_sigmoid.fits")
    outfile_ksz_cl = joinpath(outpath, "Cl_TkSZ_div_TCMB_halo_res$(res)_s$(seed)_sigmoid.fits")
else
    outfile_ksz = joinpath(outpath, "TkSZ_div_TCMB_halo_res$(res)_s$(seed).fits")
    outfile_ksz_cl = joinpath(outpath, "Cl_TkSZ_div_TCMB_halo_res$(res)_s$(seed).fits")
end

# FIXME make option?
# sigmoid settings (a=1 in halo center, goes to 0 by ~k*R200c)
K_CUTOFF  = 4.0     # center at 4*R200c
W_WIDTH   = 0.5     # transition width in R200c units
MASK_KIND = :sigmoid  # or :tanh

using HDF5

hf = h5open(fname, "r")
keys(hf)

pos = Float64.(read(hf,"Position"))

function cartesian_to_sky_coords(x, y, z)
    ra = atan(y, x)  # RA is the arctangent of y/x
    dec = asin(z / sqrt(x^2 + y^2 + z^2))  # Dec is arcsine of z / magnitude
    return ra, dec
end

ra = Float64[]
dec = Float64[]

for i in 1:length(pos[1,:])
    x = pos[1,i]
    y = pos[2,i]
    z = pos[3,i]
    _ra, _dec = cartesian_to_sky_coords(x, y, z)
    push!(ra, _ra)
    push!(dec, _dec)
end

using XGPaint, Unitful, UnitfulAstro

# example , ra and dec in radians, halo mass in M200c (Msun)
redshift, halo_mass = Float64.(read(hf,"redshift")), Float64.(read(hf,"halo_mass_m200c"))
halo_mass = halo_mass / h   # xgpaint wants msun

print("Number of halos: ", length(halo_mass))

# velocities
# compute projected v along radial direciton in 3d
vel = Float64.(read(hf, "Velocity"))
c_kms = 299_792.458
r = sqrt.(sum(pos.^2; dims=1))
proj_v_over_c = vec(sum(vel .* pos; dims=1) ./ (r .* c_kms))

perm = sortperm(dec)
ra = ra[perm]
dec = dec[perm]
redshift = redshift[perm]
halo_mass = halo_mass[perm]
proj_v_over_c = proj_v_over_c[perm]

# Mass cut
mask = halo_mass .> mass_cut
ra = ra[mask]
dec = dec[mask]
redshift = redshift[mask]
halo_mass = halo_mass[mask]
proj_v_over_c = proj_v_over_c[mask]

if do_sigmoid
    model = SigmoidBattagliaTauProfile(Omega_c=0.3089-0.0486, Omega_b=0.0486, h=0.6774)
    interp = build_interpolator(model, cache_file="cached_btau_sigmoid.jld2", overwrite=false)
else
    model = BattagliaTauProfile(Omega_c=0.3089-0.0486, Omega_b=0.0486, h=0.6774)
    interp = build_interpolator(model, cache_file="cached_btau.jld2", overwrite=false)
end

using Healpix

nside = 8192

# KSZ
m_hp = HealpixMap{Float64,RingOrder}(nside)
res = Healpix.Resolution(nside)
w = XGPaint.HealpixRingProfileWorkspace{Float64}(res)
XGPaint.paint!(m_hp, w, interp, halo_mass, redshift, ra, dec, proj_v_over_c)
Healpix.saveToFITS(m_hp, "!" * outfile_ksz, typechar="D")

cl = anafast(m_hp, niter=0)
writeClToFITS(outfile_ksz_cl, cl)