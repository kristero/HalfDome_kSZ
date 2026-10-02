# Catalogue one-halo (Poisson) kSZ power of the paint_ksz_websky.jl maps, split by halo mass (2026-10-02):
#   C_ell^{1h}(bin) = (1/4pi) sum_{cells in bin} sum_i (v_los,i/c)^2 |a_ell(M, z)|^2,
# with the painter's own definitions (read verbatim from paint_ksz_websky.jl: variants, caches, NFW c200m = 7 conversion,
# compensating sphere, exact low-z tables) and the (log10 M, z) histograms it writes, evaluated at each cell's (v/c)^2-weighted
# centroid in log10 M and z. Physical variants: catalogue M200c, 4 R200c sphere. WebSky variants: WebSky-selected halos binned in
# the converted M200c_ws, a_ell = scale a_ell^{1.5 R200c sphere} - a_ell^{comp}, with the uniform sphere's flat-sky transform
# a_ell^{comp} = sigma_T N_e / d_A^2 * 3 (sin x - x cos x) / x^3, x = ell R / d_A. Flat-sky Hankel transforms; the velocity-
# correlation (two-halo) part of the maps is not included. If <output-dir>/velocity_ratio_fit.h5 exists (velocity_ratio_fit.py), the
# WebSky variants are also given with each cell's (v/c)^2 rescaled to WebSky's 2LPT velocity dispersion at the same M200m and z
# (cl_1h_<variant>_wsvel): HalfDome halos with WebSky's velocities. If <output-dir>/websky_cell_v2.h5 exists (websky_cell_v2.py:
# WebSky's own <(v/c)^2> per (log10 M200c_ws, z) cell of the painter grid), cl_1h_<variant>_wscellvel uses n_HD x <v^2>_WS in every
# cell with >= 10 halos in both catalogues (smooth fit elsewhere): the velocity / abundance split of HalfDome vs WebSky.
# --hist=<h5> paints the one-halo of another histogram in the same format (e.g. WebSky's catalogue), written to --out-name.
# Usage: julia -t 16 poisson_mass_split_websky.jl [--output-dir=...] [--hist=...] [--out-name=ksz_websky_mass_split.h5]
using Dates, HDF5, Healpix, Interpolations, Random, XGPaint, Unitful, UnitfulAstro
using Base.Threads
include(joinpath(@__DIR__, "ksz_models_halfdome.jl"))
const besselj0 = XGPaint.SpecialFunctions.besselj0
let src = read(joinpath(@__DIR__, "paint_ksz_websky.jl"), String)
    i0 = first(findfirst("const X_PHYS", src)); i1 = first(findfirst("# ---------------------------------------------------------------- self-tests", src))
    include_string(Main, src[i0:i1-1], "paint_ksz_websky.jl (definitions)")
end

const MASS_BINS = [12.0, 13.0, 13.5, 14.0, 14.5, 15.0, 15.7]       # log10 M200c [Msun] (M200c_ws for the WebSky variants)
const ELLS = exp10.(range(log10(30.0), log10(8192.0); length=80))
const NTHETA = 3000
hist_file = getarg("hist", joinpath(output_dir, "maps", "halo_mass_redshift_histogram.h5")); out_name = getarg("out-name", "ksz_websky_mass_split.h5")
println("histogram $(hist_file)")
hist = h5open(hist_file, "r") do f
    Dict(k => read(f[k]) for k in keys(f))
end
haskey(hist, "websky_count") || error("histogram has no websky_* fields (painter older than the shuffled version)")
nb = length(MASS_BINS) - 1; nl = length(ELLS)
vfit = joinpath(output_dir, "velocity_ratio_fit.h5")
const VCOEF = isfile(vfit) && getarg("hist", "") == "" ? h5open(f -> read(f["coef"]), vfit) : Float64[]     # only for the HalfDome histogram
println(isempty(VCOEF) ? "no velocity_ratio_fit.h5: skipping the WebSky-velocity variants" : "velocity ratio fit: $(VCOEF)")
"<v^2>_WebSky / <v^2>_HalfDome at M200m [Msun], z (velocity_ratio_fit.py)"
rv2(m200m, z) = (x = log10(m200m) - 13.5; c = VCOEF; exp(2 * (c[1] + c[2] * x + c[3] * x^2 + c[4] * z + c[5] * z^2 + c[6] * x * z)))
cfile = joinpath(output_dir, "websky_cell_v2.h5")
const WSCELL = isfile(cfile) && !isempty(VCOEF) ? h5open(f -> (read(f["count"]), read(f["sum_vlos_over_c_squared"])), cfile) : nothing
WSCELL === nothing || (size(WSCELL[1]) == size(hist["websky_count"]) || error("websky_cell_v2.h5 grid differs from the histogram"))
println(WSCELL === nothing ? "no websky_cell_v2.h5 (or no velocity fit): skipping the per-cell WebSky-velocity variants" :
        "per-cell WebSky velocities: $(sum(WSCELL[1])) WebSky halos")
@inline wcomp(x) = x < 1e-3 ? 1 - x^2 / 10 : 3 * (sin(x) - x * cos(x)) / x^3

"""(cell list) for a histogram: (im, iz, count, sum v^2, centroid log10 M, centroid z)."""
function cells_of(n, v2, v2lm, v2z)
    [(im, iz, n[im, iz], v2[im, iz], v2lm[im, iz] / v2[im, iz], v2z[im, iz] / v2[im, iz])
     for im in axes(n, 1), iz in axes(n, 2) if n[im, iz] > 0 && v2[im, iz] > 0]
end
cells_p = cells_of(hist["count"], hist["sum_vlos_over_c_squared"], hist["sum_v2_log10m"], hist["sum_v2_z"])
cells_w = cells_of(hist["websky_count"], hist["websky_sum_vlos_over_c_squared"], hist["websky_sum_v2_log10m"], hist["websky_sum_v2_z"])
println("cells: physical $(length(cells_p)) ($(sum(hist["count"])) halos); websky $(length(cells_w)) ($(sum(hist["websky_count"])) halos); threads $(nthreads())")

"""a_ell of one halo's gas inside the X R200c sphere (chord-mean value x chord factor), as painted."""
function a_halo(v::Variant, M, z)
    θmax = v.X * theta_r200c(B16, M, z)
    lnθ = range(log(theta_min), log(θmax * (1 - 1e-12)); length=NTHETA); dln = step(lnθ)
    tab = z < Z_EXACT ? exact_table(v, M, z) : nothing
    f = zeros(NTHETA); θ = exp.(lnθ)
    for j in 1:NTHETA
        g = tab === nothing ? v.itp(θ[j], M, z) : table_eval(tab, θ[j])
        f[j] = v.scale * Float64(g) * chord_factor(θ[j], θmax, v.X) * θ[j]^2 * dln
    end
    return [2pi * sum(f[j] * besselj0(ℓ * θ[j]) for j in 1:NTHETA) for ℓ in ELLS]
end

cl = zeros(nl, nb, length(VARIANTS), nthreads() + 8)        # total one-halo per mass bin
cl_halo = zeros(nl, nb, length(VARIANTS), nthreads() + 8)   # WebSky variants without the compensating sphere (diagnostic)
cl_wsv = zeros(nl, nb, length(VARIANTS), nthreads() + 8)    # WebSky variants with WebSky's velocity dispersion (smooth fit)
cl_wsc = zeros(nl, nb, length(VARIANTS), nthreads() + 8)    # ... with WebSky's per-cell <v^2>
t0 = time()
@threads :static for ci in eachindex(cells_p)
    im, iz, n, v2, lm, z = cells_p[ci]; tid = threadid(); b = searchsortedlast(MASS_BINS, lm); (1 <= b <= nb) || continue
    for k in PHYS
        a = a_halo(VARIANTS[k], exp10(lm), z)
        @. cl[:, b, k, tid] += v2 * a^2 / (4pi)
    end
end
println("physical cells done in $(round(time() - t0; digits=1)) s"); flush(stdout)
@threads :static for ci in eachindex(cells_w)
    im, iz, n, v2, lm, z = cells_w[ci]; tid = threadid(); b = searchsortedlast(MASS_BINS, lm); (1 <= b <= nb) || continue
    Mws = exp10(lm); _, tau0, R, dA = compensation(Mws / ratio_c7(z), z)
    wv = isempty(VCOEF) ? 0.0 : rv2(Mws / ratio_c7(z), z)
    wc = WSCELL === nothing ? 0.0 : (WSCELL[1][im, iz] >= 10 && n >= 10 ? (WSCELL[2][im, iz] / WSCELL[1][im, iz]) / (v2 / n) : wv)
    ac = [tau0 * (4pi / 3) * R^3 / dA^2 * wcomp(ℓ * R / dA) for ℓ in ELLS]
    for k in WSV
        a = a_halo(VARIANTS[k], Mws, z)
        @. cl[:, b, k, tid] += v2 * (a - ac)^2 / (4pi)
        @. cl_halo[:, b, k, tid] += v2 * a^2 / (4pi)
        @. cl_wsv[:, b, k, tid] += wv * v2 * (a - ac)^2 / (4pi)
        @. cl_wsc[:, b, k, tid] += wc * v2 * (a - ac)^2 / (4pi)
    end
end
println("all cells done in $(round(time() - t0; digits=1)) s")
clb = dropdims(sum(cl; dims=4); dims=4); clh = dropdims(sum(cl_halo; dims=4); dims=4); clv = dropdims(sum(cl_wsv; dims=4); dims=4)
clc = dropdims(sum(cl_wsc; dims=4); dims=4)
h5open(joinpath(output_dir, out_name), "w") do f
    f["ell"] = collect(ELLS); f["log10_mass_bin_edges_msun"] = MASS_BINS
    for (k, v) in enumerate(VARIANTS)
        f["cl_1h_$(v.name)"] = clb[:, :, k]                          # (ell, mass bin), C_ell of Delta T / T
        v.websky && (f["cl_1h_$(v.name)_nocomp"] = clh[:, :, k])
        v.websky && !isempty(VCOEF) && (f["cl_1h_$(v.name)_wsvel"] = clv[:, :, k])
        v.websky && WSCELL !== nothing && (f["cl_1h_$(v.name)_wscellvel"] = clc[:, :, k])
    end
    attrs(f)["note"] = "catalogue one-halo kSZ per mass bin (painter definitions, cell centroids, exact z<$(Z_EXACT) tables); " *
                       "physical: catalogue M200c, 4 R200c; websky: M200c_ws, 1.5 R200c, minus compensating sphere; flat-sky Hankel"
    isempty(VCOEF) || (attrs(f)["velocity_ratio_coef"] = VCOEF)
    attrs(f)["histogram"] = hist_file
    attrs(f)["created_utc"] = Dates.format(now(UTC), dateformat"yyyy-mm-ddTHH:MM:SS")
end
T2 = 2.7255e6^2
for (k, v) in enumerate(VARIANTS), L in (300.0, 1000.0, 3000.0, 5000.0)
    il = argmin(abs.(ELLS .- L)); tot = sum(clb[il, :, k]); pre = ELLS[il] * (ELLS[il] + 1) / (2pi) * T2
    println(rpad(v.name, 17), " ell~$(lpad(round(Int, ELLS[il]), 4)): D_ell 1h = $(round(pre * tot; digits=4)) uK^2",
            v.websky ? " (no comp $(round(pre * sum(clh[il, :, k]); digits=4))" * (isempty(VCOEF) ? ")" : "; WebSky velocities $(round(pre * sum(clv[il, :, k]); digits=4))" *
            (WSCELL === nothing ? ")" : ", per cell $(round(pre * sum(clc[il, :, k]); digits=4)))")) : "",
            "; fractions ",
            join([string(round(clb[il, b, k] / tot; digits=3)) for b in 1:nb], " "))
end
