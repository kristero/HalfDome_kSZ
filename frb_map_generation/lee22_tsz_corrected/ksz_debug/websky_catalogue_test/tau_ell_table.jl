# Per-(M, z)-cell Fourier amplitude of the B16 tau profile, for the catalogue one-halo kSZ term (2026-10-02, ksz_debug).
# Adapted from ksz/poisson_mass_split.jl: the same chord-mean caches, chord factor, 4 R200c sphere, flat-sky Hankel transform
#   a(ell; M, z) = 2 pi int tau(theta) J0(ell theta) theta dtheta,   C_ell^{1h} = (1/4pi) sum_cells sum_i (v_i/c)^2 a^2,
# evaluated once per cell of the (log10 M200c [Msun], z) grid used by the painter (12.0:0.025:15.7, 0:0.025:4.0), so any halo
# histogram (HalfDome, WebSky with any mass conversion/selection) can be combined with it afterwards in Python.
# Sphere radii X = 4 (production cache, reused), 2 and 1 R200c (new caches here).
# Usage: julia -t 4 tau_ell_table.jl --output-dir=...
using XGPaint, HDF5, Interpolations, Unitful, UnitfulAstro
using Base.Threads
include(joinpath(@__DIR__, "..", "..", "ksz", "ksz_models.jl"))      # FMG, H, OMEGA_B, OMEGA_C, SphericalTruncation, MODELS
const besselj0 = XGPaint.SpecialFunctions.besselj0
getarg(key, default) = (for a in ARGS; startswith(a, "--$key=") && return split(a, "="; limit=2)[2]; end; default)
output_dir = getarg("output-dir", joinpath(FMG, "outputs", "lee22_ksz_debug_20261002", "websky_catalogue_test"))
mkpath(joinpath(output_dir, "caches"))
prod_cache = joinpath(FMG, "outputs", "lee22_ksz_20261002", "caches", "tau_b16_sphere4p0_chordmean_interpolator.jld2")
b16 = MODELS[1][2]
isa(b16, BattagliaTauProfile) || error("MODELS[1] is not B16")
XS = (4.0, 2.0, 1.0)
itps = Any[]
for Xs in XS
    cm = ChordMeanProfile(b16, Xs)
    cf = Xs == 4.0 ? prod_cache : joinpath(output_dir, "caches", "tau_b16_sphere$(replace(string(Xs), "." => "p"))_chordmean_interpolator.jld2")
    ip = build_interpolator(cm; cache_file=cf, overwrite=false, verbose=false)
    dev = maximum(abs(ip(f * theta_r200c(b16, M, z), M, z) / cm(f * theta_r200c(b16, M, z), M, z) - 1)
                  for (M, z) in ((2e13, 0.1), (1e14, 0.5), (5e14, 1.0), (3e13, 3.0)), f in (0.05, 0.3, 0.7) if f < Xs)
    println("X=$Xs cache $cf: max |cache/direct - 1| = $dev"); flush(stdout)
    dev < 2e-3 || error("cache check failed for X=$Xs")
    push!(itps, ip)
end
const LOGM_EDGES = collect(12.0:0.025:15.7); const Z_EDGES = collect(0.0:0.025:4.0)
lmc = 0.5 .* (LOGM_EDGES[1:end-1] .+ LOGM_EDGES[2:end]); zc = 0.5 .* (Z_EDGES[1:end-1] .+ Z_EDGES[2:end])
ells = exp10.(range(log10(30.0), log10(8192.0); length=80))
NTHETA = 3000
NM, NZ, NL, NX = length(lmc), length(zc), length(ells), length(XS)
A = zeros(Float64, NL, NM, NZ, NX)        # a(ell) per cell (steradian-integrated tau)
A0 = zeros(Float64, NM, NZ, NX)           # ell -> 0 limit: integral of tau over solid angle
TH200 = zeros(Float64, NM, NZ)
cells = [(im, iz) for im in 1:NM, iz in 1:NZ]
t0 = time()
@threads :dynamic for ci in eachindex(cells)
    im, iz = cells[ci]
    M = exp10(lmc[im]); z = zc[iz]
    th200 = theta_r200c(b16, M, z); TH200[im, iz] = th200
    for (k, ip) in enumerate(itps)
        Xs = XS[k]
        theta_min = exp(first(first(ip.itp.ranges)))
        θmax = Xs * th200
        lnθ = range(log(max(theta_min, 1.0e-9)), log(θmax); length=NTHETA); dln = step(lnθ)
        f = Vector{Float64}(undef, NTHETA); θ = Vector{Float64}(undef, NTHETA)
        for j in 1:NTHETA
            θ[j] = exp(lnθ[j]); f[j] = Float64(ip(θ[j], M, z)) * chord_factor(θ[j], θmax, Xs) * θ[j]^2 * dln
        end
        A0[im, iz, k] = 2pi * sum(f)
        for (il, ℓ) in enumerate(ells)
            s = 0.0
            @inbounds for j in 1:NTHETA
                s += f[j] * besselj0(ℓ * θ[j])
            end
            A[il, im, iz, k] = 2pi * s
        end
    end
end
println("table done in $(round(time() - t0; digits=1)) s"); flush(stdout)
dA = [Float64(ustrip(u"Mpc", XGPaint.angular_diameter_dist(b16.cosmo, z))) for z in zc]
h5open(joinpath(output_dir, "tau_ell_table_b16.h5"), "w") do f
    f["ell"] = collect(ells); f["logm_edges"] = LOGM_EDGES; f["z_edges"] = Z_EDGES; f["sphere_X"] = collect(XS)
    for (k, Xs) in enumerate(XS)
        tag = replace(string(Xs), "." => "p")
        f["a_b16_x$(tag)"] = A[:, :, :, k]; f["a0_b16_x$(tag)"] = A0[:, :, k]
    end
    f["theta200c_rad"] = TH200; f["dA_mpc"] = dA
    attrs(f)["note"] = "a(ell, logM, z): 2 pi int tau J0(ell theta) theta dtheta for XGPaint BattagliaTauProfile(0.261, 0.049, 0.68) inside the X R200c sphere (chord-mean caches), cell centres; C_1h = sum (v/c)^2 a^2 / 4pi"
end
println("saved")
