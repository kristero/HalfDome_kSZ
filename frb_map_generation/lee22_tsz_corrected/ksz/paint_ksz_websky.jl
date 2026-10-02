# Halo kSZ (Delta T / T) maps of the HalfDome lightcone with every fix needed for a like-for-like comparison with the WebSky halo kSZ
# (Stein et al. 2020, Fig. 6), painted in one catalogue pass so all variants share halos, velocities and geometry (2026-10-02).
#
# Fixes applied to every map (findings of ksz_debug/, workflow wf_d12ccf00-b27):
#   * HalfDome cosmology (Omega_m 0.3089, Omega_b 0.0486, h 0.6774; ksz_models_halfdome.jl) for profiles and the Msun/h conversion
#   * halos at z < Z_EXACT get exact per-halo profile tables (XGPaint's cache grid, linear in z with dz = 0.0196, is wrong at z < 0.03)
# Variants:
#   b16_phys, lee22corr_phys, lee22pipe_phys  physical halo kSZ: catalogue M200c, gas inside the 4 R200c sphere, XGPaint electrons
#   b16_websky, lee22corr_websky              WebSky recipe (Stein+20 Sec. 3.2.1, 4.4.2; inferred truncation):
#       - M200c from the catalogue M200m with an NFW c200m = 7 conversion (as WebSky)
#       - only halos with M200c > 1e13 Msun and theta200c > 0.5 arcmin (radius)
#       - gas inside the 1.5 R200c sphere (best fit to the WebSky curve; Stein+20 does not state it)
#       - B16 electrons n_e = (1 - Y_p/2) rho_gas/m_p (WebSky eq. 3.15, no 0.9): x WS_NE/XGP_NE on the B16 tau; Lee22 n_e unchanged
#       - minus a uniform compensating sphere with the halo's f_b M200m of gas, radius where the mean density is 3 rho_m(z)
# Velocities are HalfDome's own (N-body; 3-13 % higher rms than WebSky's 2LPT ones: a real modelling difference, not changed).
# Usage: julia -t 20 paint_ksz_websky.jl [--catalog=...] [--nside=4096] [--max-halos=0] [--chunk-size=1000000] [--output-dir=...]
#        [--self-test-only]
using Dates, HDF5, Healpix, Interpolations, Random, XGPaint, Unitful, UnitfulAstro
using Base.Threads
include(joinpath(@__DIR__, "ksz_models_halfdome.jl"))

const X_PHYS = 4.0; const X_WS = 1.5; const Z_EXACT = 0.1
const WS_MCUT = 1e13; const WS_THETA_CUT = 0.5 * pi / (180 * 60)
const C200M = 7.0; const YP = 0.24
const SIGMA_T = 6.6524587321e-29; const M_P = 1.67262192369e-27; const MPC = 3.0856775814913673e22; const G_SI = 6.67430e-11
const MSUN = Float64(ustrip(u"kg", 1.0 * XGPaint.M_sun))
const XGP_NE = 0.9 * M_P / xgpaint_mass_per_electron_kg()        # XGPaint ne2d: n_e = 0.7916 rho/m_p
const WS_NE = 1 - YP / 2                                            # WebSky halo: n_e = 0.88 rho/m_p
const F_B = OMEGA_B / OMEGA_M
const RHO_CR0 = 3 * (100 * H * 1e3 / MPC)^2 / (8 * pi * G_SI)       # kg m^-3

getarg(key, default) = (for a in ARGS; startswith(a, "--$key=") && return split(a, "="; limit=2)[2]; end; default)
catalog = getarg("catalog", "/home/cbllover/HalfDome/lightcone_100.hdf5")
nside = parse(Int, getarg("nside", "4096"))
max_halos = parse(Int, getarg("max-halos", "0"))
chunk_size = parse(Int, getarg("chunk-size", "1000000"))
output_dir = getarg("output-dir", joinpath(FMG, "outputs", "lee22_ksz_websky_20261002"))
self_test_only = "--self-test-only" in ARGS
mkpath(joinpath(output_dir, "caches")); mkpath(joinpath(output_dir, "maps"))

# ---------------------------------------------------------------- M200m -> M200c for an NFW halo with c200m = 7
mnfw(x) = log1p(x) - x / (1 + x)
omega_m_z(z) = OMEGA_M * (1 + z)^3 / (OMEGA_M * (1 + z)^3 + 1 - OMEGA_M)
function ratio_200c_200m(z)
    target = mnfw(C200M) / (C200M^3 * omega_m_z(z))      # m(x)/x^3 at R200c (x in units of r_s)
    lo, hi = 1e-3, C200M                                  # m(x)/x^3 decreases with x
    for _ in 1:200
        mid = 0.5 * (lo + hi)
        mnfw(mid) / mid^3 > target ? (lo = mid) : (hi = mid)
    end
    return mnfw(0.5 * (lo + hi)) / mnfw(C200M)
end
const RATIO_Z = collect(range(0.0, 5.0; length=2001))
const RATIO_V = ratio_200c_200m.(RATIO_Z)
ratio_c7(z) = (i = clamp(searchsortedlast(RATIO_Z, z), 1, length(RATIO_Z) - 1); t = (z - RATIO_Z[i]) / (RATIO_Z[i+1] - RATIO_Z[i]);
               (1 - t) * RATIO_V[i] + t * RATIO_V[i+1])

# ---------------------------------------------------------------- compensating sphere
d_A_m(z) = Float64(ustrip(u"Mpc", XGPaint.angular_diameter_dist(B16.cosmo, z))) * MPC
"""(theta_comp, tau0, R, dA): uniform sphere holding f_b M200m of gas at mean density 3 rho_m(z); tau(theta) = tau0 * 2 sqrt(R^2 - (dA theta)^2)."""
function compensation(m200m, z)
    rhom = OMEGA_M * RHO_CR0 * (1 + z)^3
    R = (m200m * MSUN / (4pi * rhom))^(1 / 3)
    n = F_B * m200m * MSUN * WS_NE / M_P / (4pi / 3 * R^3)
    dA = d_A_m(z)
    return atan(R / dA), SIGMA_T * n, R, dA
end
@inline tau_comp(θ, tau0, R, dA) = (b = dA * θ; b < R ? tau0 * 2 * sqrt(R^2 - b^2) : 0.0)

# ---------------------------------------------------------------- variants and caches
struct Variant
    name::String; model::Any; X::Float64; websky::Bool; scale::Float64; cm::Any; itp::Any
end
function make_variant(name, model, X, websky, scale)
    cm = ChordMeanProfile(model, X)
    tag = replace(string(X), "." => "p")
    itp = build_interpolator(cm; cache_file=joinpath(output_dir, "caches", "tau_$(name)_sphere$(tag)_chordmean_interpolator.jld2"),
                             overwrite=false, verbose=false)
    return Variant(name, model, X, websky, scale, cm, itp)
end
const VARIANTS = [make_variant("b16_phys", B16, X_PHYS, false, 1.0), make_variant("lee22corr_phys", L22CORR, X_PHYS, false, 1.0),
                  make_variant("lee22pipe_phys", L22PIPE, X_PHYS, false, 1.0),
                  make_variant("b16_websky", B16, X_WS, true, WS_NE / XGP_NE), make_variant("lee22corr_websky", L22CORR, X_WS, true, 1.0)]
const PHYS = findall(v -> !v.websky, VARIANTS); const WSV = findall(v -> v.websky, VARIANTS)
theta_min = maximum(exp(first(first(v.itp.itp.ranges))) for v in VARIANTS)
logm_lo = maximum(first(v.itp.itp.ranges[3]) for v in VARIANTS); logm_hi = minimum(last(v.itp.itp.ranges[3]) for v in VARIANTS)

"""Exact per-halo table of the chord-mean value on a log theta grid (for z < Z_EXACT)."""
function exact_table(v::Variant, M, z)
    p = XGPaint.prepare_profile_slice(v.cm, M, z)
    θhi = v.X * p.θ200 * (1 - 1e-9); θlo = max(theta_min, 1e-5 * p.θ200)
    lθ = collect(range(log(θlo), log(θhi); length=400))
    lg = [log(max(XGPaint.evaluate_profile_slice(v.cm, p, exp(t), M, z), 1e-300)) for t in lθ]
    return lθ, lg
end
@inline function table_eval(tab, θ)
    lθ, lg = tab; t = log(θ)
    t <= lθ[1] && return exp(lg[1]); t >= lθ[end] && return exp(lg[end])
    d = lθ[2] - lθ[1]; i = min(Int(floor((t - lθ[1]) / d)) + 1, length(lθ) - 1); w = (t - lθ[i]) / d
    return exp((1 - w) * lg[i] + w * lg[i+1])
end

# ---------------------------------------------------------------- self-tests
println("XGP_NE = $(XGP_NE), WS_NE = $(WS_NE), B16 WebSky tau scale = $(WS_NE / XGP_NE), f_b = $(F_B)")
r7 = [ratio_c7(z) for z in (0.0, 0.5, 1.0, 2.0)]
poly(z) = 1.3595873806301997 - 0.49815455039058704z + 0.3014644154503205z^2 - 0.08294138910919961z^3 + 0.0083985355523884z^4
println("M200c/M200m (NFW c200m=7) at z=0/0.5/1/2: $(round.(r7; digits=4)); websky_halos_test polynomial: $(round.([1/poly(z) for z in (0.0,0.5,1.0,2.0)]; digits=4))")
x0 = let lo = 1e-3, hi = C200M, tg = mnfw(C200M) / (C200M^3 * omega_m_z(0.5))
    for _ in 1:200; mid = 0.5(lo + hi); mnfw(mid) / mid^3 > tg ? (lo = mid) : (hi = mid); end; 0.5(lo + hi) end
dens = 200 * omega_m_z(0.5) * (mnfw(x0) / x0^3) / (mnfw(C200M) / C200M^3)     # mean density at R200c in units of rho_crit(z): must be 200 * 1
println("c=7 check: mean density inside R200c / (200 rho_crit) = $(dens / 200)"); abs(dens / 200 - 1) < 1e-9 || error("c200m conversion failed")
θc, t0, R, dA = compensation(1e14, 0.3)
ne_int = sum(tau_comp(b / dA, t0, R, dA) * 2pi * b for b in range(0, R; length=200001)) * (R / 200000) / SIGMA_T
ne_true = F_B * 1e14 * MSUN * WS_NE / M_P
println("compensation: electrons from integrating tau over the disc / f_b M (1-Y/2)/m_p = $(ne_int / ne_true); R_comp/R200m = ",
        R / (3 * 1e14 * MSUN / (4pi * 200 * OMEGA_M * RHO_CR0 * 1.3^3))^(1/3))
abs(ne_int / ne_true - 1) < 1e-4 || error("compensation normalisation failed")
worst = 0.0
for (M, z) in ((2e13, 0.1), (1e14, 0.5), (5e14, 1.0)), f in (0.3, 1.0)
    θ = f * theta_r200c(B16, M, z)
    for (name, m) in (("lee22corr", L22CORR), ("lee22pipe", L22PIPE))
        τ = Float64(m(θ, M, z)); dm = Float64(DM_REF[name](θ, M, z))
        global worst = max(worst, abs(τ / (SIGMA_T * dm * (1 + z) / Float64(ustrip(u"pc*cm^-3", 1u"m^-2"))) - 1))
    end
end
println("Lee22 tau = sigma_T (1+z) DM: max rel dev $(worst)"); worst < 1e-8 || error("Lee22 tau self-test failed")
for v in VARIANTS
    dev = maximum(abs(v.itp(f * theta_r200c(v.model, M, z), M, z) / v.cm(f * theta_r200c(v.model, M, z), M, z) - 1)
                  for (M, z) in ((2e13, 0.1), (1e14, 0.5), (5e14, 1.0)), f in (0.1, 0.7, 1.2))
    devz = 0.0
    for (M, z) in ((3e13, 0.004), (2e14, 0.02), (1e15, 0.06)), f in (0.05, 0.5, 1.2)
        tab = exact_table(v, M, z); θ = f * theta_r200c(v.model, M, z) * 1.0001
        devz = max(devz, abs(table_eval(tab, θ) / v.cm(θ, M, z) - 1))
    end
    println("$(v.name): cache/direct (z >= 0.1) max dev $(dev); exact low-z table/direct max dev $(devz)")
    (dev < 2e-3 && devz < 1e-3) || error("profile check failed for $(v.name)")
end
self_test_only && (println("self-test only: done"); exit(0))

# ---------------------------------------------------------------- painting
resolution = Healpix.Resolution(nside)
workspace = XGPaint.HealpixRingProfileWorkspace{Float64}(resolution)
ring_locks = [ReentrantLock() for _ in eachindex(workspace.ring_thetas)]
maps = [HealpixMap{Float64,RingOrder}(nside) for _ in VARIANTS]
for m in maps; fill!(m.pixels, 0.0); end
pix = [m.pixels for m in maps]
const NV = length(VARIANTS)

function paint_batch!(m200c, m200m, redshifts, positions, vlos_over_c, counters, sel, mws)
    Threads.@threads :static for i in eachindex(m200c)
        tid = Threads.threadid()
        x, y, zc = positions[1, i], positions[2, i], positions[3, i]
        d = sqrt(x^2 + y^2 + zc^2); ux, uy, uz = x / d, y / d, zc / d
        θc, φc = Healpix.vec2ang(ux, uy, uz); θc = Float64(θc); φc = mod(Float64(φc), 2pi)
        Mc = m200c[i]; Mm = m200m[i]; z = redshifts[i]; v = vlos_over_c[i]
        Mws = Mm * ratio_c7(z); mws[i] = Mws
        θ200 = theta_r200c(B16, Mc, z); θmax_p = X_PHYS * θ200
        θ200w = theta_r200c(B16, Mws, z)
        cut = (Mws > WS_MCUT) && (θ200w > WS_THETA_CUT) && (logm_lo <= log10(Mws) <= logm_hi)
        θmax_w = 0.0; θcomp = 0.0; t0 = 0.0; Rc = 0.0; dA = 0.0
        if cut
            θcomp, t0, Rc, dA = compensation(Mm, z)
            θmax_w = max(X_WS * θ200w, θcomp)
        end
        θmax = min(max(θmax_p, θmax_w), pi)
        exact = z < Z_EXACT
        tabs = exact ? [(VARIANTS[k].websky ? (cut ? exact_table(VARIANTS[k], Mws, z) : nothing) : exact_table(VARIANTS[k], Mc, z)) for k in 1:NV] : nothing
        rs, re = XGPaint.get_relevant_rings(workspace.res, θc, θmax)
        n = 0
        for ring in rs:re
            r1, r2 = XGPaint.get_ring_disc_ranges(workspace, ring, θc, φc, θmax)
            first_pixel = workspace.ring_first_pixels[ring]
            lock(ring_locks[ring]) do
                for lp in Iterators.flatten((r1, r2))
                    gp = first_pixel + lp - 1
                    px, py, pz = Healpix.pix2vecRing(workspace.res, gp)
                    θ = acos(clamp(ux * px + uy * py + uz * pz, -1.0, 1.0))
                    θ < θmax || continue
                    θe = max(θ, theta_min)
                    if θ < θmax_p
                        cf = chord_factor(θ, θmax_p, X_PHYS)
                        for k in PHYS
                            g = exact ? table_eval(tabs[k], θe) : VARIANTS[k].itp(θe, Mc, z)
                            pix[k][gp] -= v * g * cf
                        end
                    end
                    if cut
                        θw = X_WS * θ200w
                        if θ < θw
                            cf = chord_factor(θ, θw, X_WS)
                            for k in WSV
                                g = exact ? table_eval(tabs[k], θe) : VARIANTS[k].itp(θe, Mws, z)
                                pix[k][gp] -= v * VARIANTS[k].scale * g * cf
                            end
                        end
                        tc = tau_comp(θ, t0, Rc, dA)
                        if tc > 0
                            for k in WSV
                                pix[k][gp] += v * tc        # subtract the compensating sphere's kSZ (-tau_comp v/c)
                            end
                        end
                    end
                    n += 1
                end
            end
        end
        counters[tid, 1] += n; counters[tid, 2] += cut; counters[tid, 3] += exact; sel[i] = cut
    end
end

const LOGM_EDGES = collect(12.0:0.025:15.7); const Z_EDGES = collect(0.0:0.025:4.0)
hist_n = zeros(Int, length(LOGM_EDGES) - 1, length(Z_EDGES) - 1); hist_v2 = zeros(Float64, size(hist_n))
hist_v2lm = zeros(Float64, size(hist_n)); hist_v2z = zeros(Float64, size(hist_n))       # (v/c)^2-weighted log10 M and z sums
ws_n = zeros(Int, size(hist_n)); ws_v2 = zeros(Float64, size(hist_n)); ws_v2lm = zeros(Float64, size(hist_n)); ws_v2z = zeros(Float64, size(hist_n))
counters = zeros(Int, Threads.nthreads() + 8, 3)
t0w = time(); nsel = 0; vrms_acc = 0.0
h5open(catalog, "r") do f
    pos_ds = f["Position"]; mc_ds = f["halo_mass_m200c"]; mm_ds = f["halo_mass_m200m"]; vel_ds = f["Velocity"]; z_ds = f["redshift"]
    N = size(pos_ds, 2); last_row = max_halos == 0 ? N : min(max_halos, N)
    for start in 1:chunk_size:last_row
        idx = start:min(start + chunk_size - 1, last_row)
        pos = Float64.(pos_ds[:, idx]); vel = Float64.(vel_ds[:, idx])
        mc = Float64.(mc_ds[idx]) ./ H; mm = Float64.(mm_ds[idx]) ./ H; zs = Float64.(z_ds[idx])
        keep = isfinite.(mc) .& (mc .> 0) .& isfinite.(mm) .& (mm .> 0) .& isfinite.(zs) .& (zs .> 0) .&
               vec(all(isfinite, pos; dims=1)) .& vec(all(isfinite, vel; dims=1))
        pos = pos[:, keep]; vel = vel[:, keep]; mc = mc[keep]; mm = mm[keep]; zs = zs[keep]
        r = sqrt.(vec(sum(pos .^ 2; dims=1))); vlos_c = vec(sum(pos .* vel; dims=1)) ./ r ./ C_KMS
        all(logm_lo .<= log10.(mc) .<= logm_hi) || error("catalogue M200c outside cache range in chunk $start")
        # the catalogue is sorted by decreasing z: shuffle each chunk so the large low-z discs spread over all threads (:static)
        perm = randperm(MersenneTwister(start), length(mc))
        mc = mc[perm]; mm = mm[perm]; zs = zs[perm]; pos = pos[:, perm]; vlos_c = vlos_c[perm]
        sel = falses(length(mc)); mws = zeros(length(mc))
        paint_batch!(mc, mm, zs, pos, vlos_c, counters, sel, mws)
        for j in eachindex(mc)
            w = vlos_c[j]^2; iz = searchsortedlast(Z_EDGES, zs[j])
            im = searchsortedlast(LOGM_EDGES, log10(mc[j]))
            hist_n[im, iz] += 1; hist_v2[im, iz] += w; hist_v2lm[im, iz] += w * log10(mc[j]); hist_v2z[im, iz] += w * zs[j]
            if sel[j]
                im = searchsortedlast(LOGM_EDGES, log10(mws[j]))
                ws_n[im, iz] += 1; ws_v2[im, iz] += w; ws_v2lm[im, iz] += w * log10(mws[j]); ws_v2z[im, iz] += w * zs[j]
            end
        end
        global nsel += length(mc); global vrms_acc += sum(abs2, vlos_c .* C_KMS)
        println("  rows $(first(idx)):$(last(idx))  halos=$nsel  pixel updates=$(sum(counters[:, 1]))  websky-selected=$(sum(counters[:, 2]))  exact-lowz=$(sum(counters[:, 3]))  elapsed=$(round(time() - t0w; digits=0)) s")
        flush(stdout)
    end
end
for (k, v) in enumerate(VARIANTS)
    all(isfinite, maps[k].pixels) || error("non-finite pixels in $(v.name)")
    path = joinpath(output_dir, "maps", "halfdome_ksz_$(v.name)_nside$(nside).fits")
    Healpix.saveToFITS(maps[k], "!" * path, typechar="D")
    println("saved $path  rms=$(sqrt(sum(abs2, maps[k].pixels) / length(maps[k].pixels)))")
end
h5open(joinpath(output_dir, "maps", "halo_mass_redshift_histogram.h5"), "w") do f
    f["logm_edges"] = LOGM_EDGES; f["z_edges"] = Z_EDGES; f["count"] = hist_n; f["sum_vlos_over_c_squared"] = hist_v2
    f["sum_v2_log10m"] = hist_v2lm; f["sum_v2_z"] = hist_v2z
    f["websky_count"] = ws_n; f["websky_sum_vlos_over_c_squared"] = ws_v2; f["websky_sum_v2_log10m"] = ws_v2lm; f["websky_sum_v2_z"] = ws_v2z
    attrs(f)["mass"] = "log10 catalogue M200c [Msun] = halo_mass_m200c / 0.6774"
    attrs(f)["websky_mass"] = "websky_*: WebSky-selected halos binned in log10 M200c_ws = M200m x NFW(c200m=7) ratio [Msun]"
end
open(joinpath(output_dir, "maps", "provenance.txt"), "w") do io
    for (k, val) in sort(collect(Dict{String,Any}(
        "created_utc" => Dates.format(now(UTC), dateformat"yyyy-mm-ddTHH:MM:SS"), "catalog" => catalog, "nside" => nside,
        "halos_read" => nsel, "websky_selected_halos" => sum(counters[:, 2]), "exact_lowz_halos" => sum(counters[:, 3]),
        "elapsed_seconds" => round(time() - t0w; digits=1), "julia_version" => string(VERSION), "threads" => Threads.nthreads(),
        "cosmology" => "h=$(H) Omega_b=$(OMEGA_B) Omega_c=$(OMEGA_C) (HalfDome lightcone: Omega_m 0.3089)",
        "variants" => join(["$(v.name): X=$(v.X) R200c websky=$(v.websky) tau_scale=$(v.scale)" for v in VARIANTS], "; "),
        "websky_recipe" => "M200c = M200m x NFW(c200m=7) ratio; cut M200c>1e13 Msun and theta200c>0.5 arcmin; 1.5 R200c sphere; B16 n_e (1-Yp/2) rho/m_p; minus uniform sphere f_b M200m at 3 rho_m(z)",
        "lowz" => "exact per-halo tables (400 log-theta points) for z < $(Z_EXACT)", "map" => "Delta T / T = -tau v_los/c",
        "vlos_rms_kms" => sqrt(vrms_acc / max(nsel, 1)), "theta_min_rad" => theta_min, "shape_clip_msun" => SHAPE_CLIP,
    )); by=first)
        println(io, "$k=$val")
    end
end
println("done in $(round(time() - t0w; digits=1)) s")
