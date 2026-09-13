

# Import ChunkSplitters for better threading
using ChunkSplitters: chunks

# RECTANGULAR WORKSPACES


struct CarClenshawCurtisProfileWorkspace{T,A<:AbstractArray{T,2}} <: AbstractProfileWorkspace{T}
    sin_α::A
    cos_α::A
    sin_δ::A
    cos_δ::A
end

function profileworkspace(shape, wcs::CarClenshawCurtis)
    α_map, δ_map = posmap(shape, wcs)
    return CarClenshawCurtisProfileWorkspace(
        sin.(α_map), cos.(α_map), sin.(δ_map), cos.(δ_map))
end


struct GnomonicProfileWorkspace{T,A<:AbstractArray{T,2}} <: AbstractProfileWorkspace{T}
    sin_α::A
    cos_α::A
    sin_δ::A
    cos_δ::A
end

function profileworkspace(shape, wcs::Gnomonic)
    α_map, δ_map = posmap(shape, wcs)
    return GnomonicProfileWorkspace(
        sin.(α_map), cos.(α_map), sin.(δ_map), cos.(δ_map))
end

Base.show(io::IO, w::AbstractProfileWorkspace) = print(io, "$(typeof(w))")

@inline function interpolator_stage_start(label::AbstractString; verbose::Bool=true)
    if verbose
        println("[interpolator] START ", label)
        flush(stdout)
    end
    return time_ns()
end

@inline function interpolator_stage_end(label::AbstractString, t0_ns; verbose::Bool=true, extra::AbstractString="")
    if verbose
        elapsed_s = (time_ns() - t0_ns) / 1.0e9
        suffix = isempty(extra) ? "" : " " * extra
        println("[interpolator] END   ", label, " wall=", round(elapsed_s; digits=3), " s", suffix)
        flush(stdout)
    end
    return nothing
end

@inline prepare_profile_slice(model, mass, redshift) = nothing

@inline evaluate_profile_slice(model, prepared, theta, mass, redshift) = model(theta, mass, redshift)

function fill_profile_slice!(dest, model::AbstractGNFW{T}, logthetas, mass, redshift) where T
    prepared = prepare_profile_slice(model, mass, redshift)
    @inbounds for itheta in eachindex(logthetas)
        theta = exp(logthetas[itheta])
        dest[itheta] = max(zero(T), evaluate_profile_slice(model, prepared, theta, mass, redshift))
    end
    return nothing
end


function profile_grid(model::AbstractGNFW{T}; N_z=256, N_logM=128, N_logtheta=256, z_min=1e-3,
        z_max=5.0, logM_min=12, logM_max=15.7, logtheta_min=-15.7, logtheta_max=2.5) where T

    logthetas = LinRange(logtheta_min, logtheta_max, N_logtheta)
    redshifts = LinRange(z_min, z_max, N_z)
    logMs = LinRange(logM_min, logM_max, N_logM)
    return profile_grid(model, logthetas, redshifts, logMs)
end

function profile_grid(model::AbstractGNFW{T}, logthetas, redshifts, logMs) where T

    N_logtheta, N_z, N_logM = length(logthetas), length(redshifts), length(logMs)
    println(
        "[interpolator] profile_grid dims: N_logtheta=", N_logtheta,
        " N_z=", N_z,
        " N_logM=", N_logM,
        " threads=", Threads.nthreads()
    )
    flush(stdout)

    alloc_t0 = interpolator_stage_start("profile_grid allocation")
    A = zeros(T, (N_logtheta, N_z, N_logM))
    interpolator_stage_end(
        "profile_grid allocation",
        alloc_t0;
        extra="size=$(size(A)) eltype=$(T)"
    )

    eval_t0 = interpolator_stage_start("profile_grid threaded evaluation")
    N_profiles = N_z * N_logM
    nchunks = min(N_profiles, max(Threads.nthreads(), 8 * Threads.nthreads()))
    Threads.@threads for chunk in chunks(1:N_profiles; n=nchunks)
        for idx in chunk
            iz = 1 + ((idx - 1) % N_z)
            im = 1 + div(idx - 1, N_z)
            mass = 10^(logMs[im])
            redshift = redshifts[iz]
            fill_profile_slice!(view(A, :, iz, im), model, logthetas, mass, redshift)
        end
    end
    interpolator_stage_end("profile_grid threaded evaluation", eval_t0)

    return logthetas, redshifts, logMs, A
end


"""
Computes a real-space beam interpolator and a maximum
"""
function realspacegaussbeam(::Type{T}, θ_FWHM::Ti; rtol=1e-24, N_θ::Int=2000) where {T,Ti}
    Nlmax = ceil(Int, log2(8π / θ_FWHM))
    lmax = 2^Nlmax

    b_l = gaussbeam(θ_FWHM, lmax)
    θs = LinRange(zero(θ_FWHM), 5θ_FWHM, N_θ)
    b_θ = XGPaint.bl2beam(b_l, θs)
    atol = b_θ[begin] * rtol
    i_max = findfirst(<(atol), b_θ)

    θs = convert(LinRange{T, Int}, θs[begin:i_max])
    beam_real_interp = cubic_spline_interpolation(
        θs, T.(b_θ[begin:i_max]), extrapolation_bc=zero(T))
    return beam_real_interp, θs
end


# function realspacebeampaint!(hp_map, w::HealpixSerialProfileWorkspace, realprofile, flux, θ₀, ϕ₀)
#     x₀, y₀, z₀ = ang2vec(θ₀, ϕ₀)
#     XGPaint.queryDiscRing!(w.disc_buffer, w.ringinfo, hp_map.resolution, θ₀, ϕ₀, w.θmax)

#     for ir in w.disc_buffer
#         x₁, y₁, z₁ = w.posmap.pixels[ir]
#         d² = (x₁ - x₀)^2 + (y₁ - y₀)^2 + (z₁ - z₀)^2
#         θ = acos(1 - d² / 2)
#         hp_map.pixels[ir] += flux * realprofile(θ)
#     end
# end


"""Apply a beam to a profile grid"""
_thread_storage_count() = isdefined(Threads, :maxthreadid) ? Threads.maxthreadid() : Threads.nthreads()

function transform_profile_grid!(y_prof_grid, rft, lbeam)
    N_z = size(y_prof_grid, 2)
    N_logM = size(y_prof_grid, 3)
    N_profiles = N_z * N_logM
    nthreads = Threads.nthreads()
    rfts = [deepcopy(rft) for _ in 1:_thread_storage_count()]

    Threads.@threads for chunk in chunks(1:N_profiles; n=nthreads)
        local_rft = rfts[Threads.threadid()]
        for idx in chunk
            i = 1 + ((idx - 1) % N_z)
            j = 1 + div(idx - 1, N_z)
            rprof = copy(@view y_prof_grid[:, i, j])
            lprof = real2harm(local_rft, rprof)
            lprof .*= lbeam
            reverse!(lprof)
            y_prof_grid[:, i, j] .= harm2real(local_rft, lprof)
        end
    end
    return nothing
end

"prune a profile grid for negative values, extrapolate instead"
function cleanup_negatives!(y_prof_grid)
    floor_value = nextfloat(0.0)
    N_z = size(y_prof_grid, 2)
    N_logM = size(y_prof_grid, 3)
    N_profiles = N_z * N_logM

    Threads.@threads for chunk in chunks(1:N_profiles; n=Threads.nthreads())
        for idx in chunk
            i = 1 + ((idx - 1) % N_z)
            j = 1 + div(idx - 1, N_z)
            profile = @view y_prof_grid[:, i, j]
            first_positive_idx = findfirst(>(0), profile)

            if isnothing(first_positive_idx)
                fill!(profile, floor_value)
                continue
            end

            if first_positive_idx > 1
                profile[1:first_positive_idx-1] .= profile[first_positive_idx]
            end

            extrapolating = false
            fact = 1.0
            for k in first_positive_idx:length(profile)
                if profile[k] <= 0
                    extrapolating = true
                    if k == 1
                        profile[k] = max(profile[first_positive_idx], floor_value)
                        continue
                    elseif k == 2
                        fact = 1.0
                    else
                        prev_value = max(profile[k - 1], floor_value)
                        prev_prev_value = max(profile[k - 2], floor_value)
                        fact = prev_value / prev_prev_value
                    end
                end
                if extrapolating
                    profile[k] = max(fact * profile[k - 1], floor_value)
                end
            end
        end
    end
    return nothing
end

function replace_nonpositive_with_floor!(y_prof_grid)
    T = eltype(y_prof_grid)
    N_values = length(y_prof_grid)
    nthreads = Threads.nthreads()
    nthread_slots = _thread_storage_count()
    local_mins = fill(typemax(T), nthread_slots)
    local_positive_counts = zeros(Int, nthread_slots)
    local_bad_counts = zeros(Int, nthread_slots)

    Threads.@threads for chunk in chunks(1:N_values; n=nthreads)
        tid = Threads.threadid()
        local_min = local_mins[tid]
        positive_count = 0
        bad_count = 0

        @inbounds for idx in chunk
            value = y_prof_grid[idx]
            if value > zero(T)
                positive_count += 1
                if value < local_min
                    local_min = value
                end
            else
                bad_count += 1
            end
        end

        local_mins[tid] = local_min
        local_positive_counts[tid] += positive_count
        local_bad_counts[tid] += bad_count
    end

    replaced_count = sum(local_bad_counts)
    replaced_count == 0 && return replaced_count, zero(T)

    total_positive_count = sum(local_positive_counts)
    floor_val = if total_positive_count == 0
        nextfloat(zero(T))
    else
        minimum(local_mins[local_positive_counts .> 0]) * T(1e-6)
    end

    Threads.@threads for chunk in chunks(1:N_values; n=nthreads)
        @inbounds for idx in chunk
            if y_prof_grid[idx] <= zero(T)
                y_prof_grid[idx] = floor_val
            end
        end
    end

    return replaced_count, floor_val
end



# get angular size in radians of radius to stop at
function compute_θmax(model::AbstractProfile{T}, M_Δ, z; mult=4) where T
    r = R_Δ(model, M_Δ, z)
    return T(mult * angular_size(model, r, z))
end

# prevent infinities at cusp
compute_θmin(model::AbstractInterpolatorProfile) = exp(first(first(model.itp.ranges)))
compute_θmin(::AbstractProfile{T}) where T = eps(T) 


# find maximum radius to integrate to
function build_max_paint_logradius(logθs, redshifts, logMs, 
                              A::AbstractArray{T}; rtol=1e-2) where T
    
    logRs = zeros(T, (size(A)[2:3]))
    N_logM = length(logMs)
    N_logθ = length(logθs)
    dF_r = zeros(N_logθ)
    
    for im in 1:N_logM
        for (iz, z) in enumerate(redshifts)
            s = zero(T)
            for iθ in 1:(N_logθ-1)
                θ₁ = exp(logθs[iθ])
                θ₂ = exp(logθs[iθ+1])
                f₁ = A[iθ, iz, im] * θ₁
                f₂ = A[iθ+1, iz, im] * θ₂
                s += (θ₂ - θ₁) * (f₁ + f₂) / 2

                dF_r[iθ] = s
            end

            threshold = (1-rtol) * s
            for iθ in (N_logθ-1):-1:1
                if dF_r[iθ] < threshold
                    logRs[iz, im] = min(logθs[iθ], log(π))
                    break
                end
            end
            
        end
    end

    return scale(
        Interpolations.interpolate(logRs, BSpline(Cubic(Line(OnGrid())))), 
        redshifts, logMs);
end



"""
    LogInterpolatorProfile{T, P, I1}

A profile that interpolates over a positive-definite function (θ, z, M_halo), but internally
interpolates over log(θ) and log10(M) using a given interpolator. Evaluation of this profile
is then done by exponentiating the result of the interpolator.

```
    f(θ, z, M) = exp(itp(log(θ), z, log10(M)))
```

This is useful for interpolating over a large range of scales and masses, where the profile
is expected to be smooth in log-log space. It wraps the original model and also the 
interpolator object itself.
"""
struct LogInterpolatorProfile{T, P <: AbstractProfile{T}, I1, C} <: AbstractInterpolatorProfile{T}
    model::P
    itp::I1
    cosmo::C
end


function LogInterpolatorProfile(model::AbstractProfile, itp)
    return LogInterpolatorProfile(model, itp, model.cosmo)  # use wrapped cosmology
end

# forward the interpolator calls to the wrapped interpolator
# IMPORTANT: for backwards compat, interpolator internal order is θ, z, mass
# which DIFFERS from the rest of the code which is (θ, mass, z, α, δ)
# should fix this at some point
@inline (ip::LogInterpolatorProfile)(θ, Mh_Msun, z) = exp(ip.itp(log(θ), z, log10(Mh_Msun)))

Base.show(io::IO, ip::LogInterpolatorProfile{T,P,I1}) where {T,P,I1} = print(
    io, "LogInterpolatorProfile{$(T),\n  $(P),\n  ...} interpolating over size ", size(ip.itp))


function cleanup_nonpositive_enabled()
    raw = lowercase(strip(get(ENV, "XGPAINT_CLEANUP_NONPOSITIVE", "true")))
    raw in ("1", "true", "t", "yes", "y", "on") && return true
    raw in ("0", "false", "f", "no", "n", "off") && return false
    error("Invalid XGPAINT_CLEANUP_NONPOSITIVE=$(repr(raw)).")
end

"""Helper function to build a (theta, z, Mh) interpolator"""
function build_interpolator(model::AbstractProfile; cache_file::String="",
                            N_logtheta=512, pad=128, logM_max=15.7, overwrite=true, verbose=true)

    cleanup_nonpositive = cleanup_nonpositive_enabled()
    if verbose
        println(
            "[interpolator] config overwrite=", overwrite,
            " cache_file=", isempty(cache_file) ? "<none>" : cache_file,
            " cleanup_nonpositive=", cleanup_nonpositive,
            " N_logtheta=", N_logtheta,
            " pad=", pad,
            " logM_max=", logM_max,
            " threads=", Threads.nthreads()
        )
        flush(stdout)
    end

    if overwrite || (isfile(cache_file) == false)
        verbose && (print("Building new interpolator from model.
"); flush(stdout))
        rft_t0 = interpolator_stage_start("RadialFourierTransform"; verbose=verbose)
        rft = RadialFourierTransform(n=N_logtheta, pad=pad)
        interpolator_stage_end("RadialFourierTransform", rft_t0; verbose=verbose)

        range_t0 = interpolator_stage_start("rft radius bounds"; verbose=verbose)
        logtheta_min, logtheta_max = log(minimum(rft.r)), log(maximum(rft.r))
        interpolator_stage_end(
            "rft radius bounds",
            range_t0;
            verbose=verbose,
            extra="logtheta_min=$(logtheta_min) logtheta_max=$(logtheta_max)"
        )

        grid_t0 = interpolator_stage_start("profile_grid"; verbose=verbose)
        prof_logthetas, prof_redshift, prof_logMs, prof_y = profile_grid(model;
            N_logtheta=N_logtheta, logtheta_min=logtheta_min, logtheta_max=logtheta_max, logM_max=logM_max)
        interpolator_stage_end("profile_grid", grid_t0; verbose=verbose)

        if length(cache_file) > 0
            verbose && (print("Saving new interpolator to $(cache_file).
"); flush(stdout))
            save_t0 = interpolator_stage_start("cache save"; verbose=verbose)
            save(cache_file, Dict(
                "prof_logthetas" => prof_logthetas,
                "prof_redshift" => prof_redshift,
                "prof_logMs" => prof_logMs,
                "prof_y" => prof_y,
            ))
            interpolator_stage_end("cache save", save_t0; verbose=verbose)
        end
    else
        print("Found cached Battaglia profile model. Loading from disk.
")
        flush(stdout)
        load_t0 = interpolator_stage_start("cache load"; verbose=verbose)
        model_grid = load(cache_file)
        interpolator_stage_end("cache load", load_t0; verbose=verbose)

        unpack_t0 = interpolator_stage_start("cache unpack"; verbose=verbose)
        logtheta_key = haskey(model_grid, "prof_logthetas") ? "prof_logthetas" :
            (haskey(model_grid, "prof_logθs") ? "prof_logθs" :
             error("Cache is missing log-theta key. Found keys: $(collect(keys(model_grid)))"))
        prof_logthetas, prof_redshift, prof_logMs, prof_y = model_grid[logtheta_key],
            model_grid["prof_redshift"], model_grid["prof_logMs"], model_grid["prof_y"]
        interpolator_stage_end(
            "cache unpack",
            unpack_t0;
            verbose=verbose,
            extra="size=$(size(prof_y))"
        )
    end

    nonfinite_t0 = interpolator_stage_start("nonfinite scan"; verbose=verbose)
    nonfinite_count = count(x -> !isfinite(x), prof_y)
    interpolator_stage_end(
        "nonfinite scan",
        nonfinite_t0;
        verbose=verbose,
        extra="count=$(nonfinite_count)"
    )
    nonfinite_count == 0 || error(
        "build_interpolator encountered $(nonfinite_count) non-finite prof_y values (NaN/Inf)."
    )

    if cleanup_nonpositive
        cleanup_t0 = interpolator_stage_start("nonpositive cleanup"; verbose=verbose)
        replaced_count, floor_val = replace_nonpositive_with_floor!(prof_y)
        interpolator_stage_end(
            "nonpositive cleanup",
            cleanup_t0;
            verbose=verbose,
            extra="replaced=$(replaced_count) floor=$(floor_val)"
        )
        if verbose && replaced_count > 0
            println("Replaced ", replaced_count, " <=0 entries in prof_y with floor = ", floor_val)
            flush(stdout)
        end
    else
        nonpositive_t0 = interpolator_stage_start("nonpositive check"; verbose=verbose)
        has_nonpositive = any(prof_y .<= 0)
        interpolator_stage_end(
            "nonpositive check",
            nonpositive_t0;
            verbose=verbose,
            extra="has_nonpositive=$(has_nonpositive)"
        )
        has_nonpositive && error(
            "build_interpolator encountered nonpositive prof_y values, but XGPAINT_CLEANUP_NONPOSITIVE=false. " *
            "Re-enable cleanup or ensure the profile grid is strictly positive."
        )
    end

    log_t0 = interpolator_stage_start("log transform"; verbose=verbose)
    log_prof_y = log.(prof_y)
    interpolator_stage_end("log transform", log_t0; verbose=verbose)

    interpolate_t0 = interpolator_stage_start("Interpolations.interpolate"; verbose=verbose)
    itp = Interpolations.interpolate(log_prof_y, BSpline(Cubic(Line(OnGrid()))))
    interpolator_stage_end("Interpolations.interpolate", interpolate_t0; verbose=verbose)

    scale_t0 = interpolator_stage_start("scale"; verbose=verbose)
    interp_model = scale(itp, prof_logthetas, prof_redshift, prof_logMs)
    interpolator_stage_end("scale", scale_t0; verbose=verbose)

    wrap_t0 = interpolator_stage_start("LogInterpolatorProfile wrapper"; verbose=verbose)
    wrapped_model = LogInterpolatorProfile(model, interp_model)
    interpolator_stage_end("LogInterpolatorProfile wrapper", wrap_t0; verbose=verbose)
    return wrapped_model
end


function profile_paint_generic!(m::Enmap{T, 2, Matrix{T}, CarClenshawCurtis{T}},
                        workspace::CarClenshawCurtisProfileWorkspace, model, Mh, z, α₀, δ₀, 
                        θmax, normalization=1) where T

    # get indices of the region to work on
    i1, j1 = sky2pix(m, α₀ - θmax, δ₀ - θmax)
    i2, j2 = sky2pix(m, α₀ + θmax, δ₀ + θmax)
    i_start = floor(Int, max(min(i1, i2), 1))
    i_stop = ceil(Int, min(max(i1, i2), size(m, 1)))
    j_start = floor(Int, max(min(j1, j2), 1))
    j_stop = ceil(Int, min(max(j1, j2), size(m, 2)))
    θmin = compute_θmin(model)

    x₀ = cos(δ₀) * cos(α₀)
    y₀ = cos(δ₀) * sin(α₀) 
    z₀ = sin(δ₀)

    @inbounds for j in j_start:j_stop
        for i in i_start:i_stop
            x₁ = workspace.cos_δ[i,j] * workspace.cos_α[i,j]
            y₁ = workspace.cos_δ[i,j] * workspace.sin_α[i,j]
            z₁ = workspace.sin_δ[i,j]
            d² = (x₁ - x₀)^2 + (y₁ - y₀)^2 + (z₁ - z₀)^2
            θ =  acos(clamp(1 - d² / 2, -one(T), one(T)))
            θ = max(θmin, θ)  # clamp to minimum θ
            m[i,j] += ifelse(θ < θmax, 
                             T(normalization * model(θ, Mh, z)),
                             zero(T))
        end
    end
end

# fall back to generic profile painter if no specialized painter is defined for the model
function profile_paint!(m::Enmap{T, 2, Matrix{T}, CarClenshawCurtis{T}}, 
                        workspace::CarClenshawCurtisProfileWorkspace, model, 
                        Mh, z, α₀, δ₀, θmax, normalization=1) where T
    profile_paint_generic!(m, workspace, model, Mh, z, α₀, δ₀, θmax, normalization)
end


function profile_paint_generic!(m::Enmap{T, 2, Matrix{T}, Gnomonic{T}}, 
                                workspace::GnomonicProfileWorkspace, model, 
                                Mh, z, α₀, δ₀, θmax, normalization=1) where T

    # get indices of the region to work on
    i1, j1 = sky2pix(m, α₀ - θmax, δ₀ - θmax)
    i2, j2 = sky2pix(m, α₀ + θmax, δ₀ + θmax)
    i_start = floor(Int, max(min(i1, i2), 1))
    i_stop = ceil(Int, min(max(i1, i2), size(m, 1)))
    j_start = floor(Int, max(min(j1, j2), 1))
    j_stop = ceil(Int, min(max(j1, j2), size(m, 2)))
    θmin = compute_θmin(model)

    x₀ = cos(δ₀) * cos(α₀)
    y₀ = cos(δ₀) * sin(α₀) 
    z₀ = sin(δ₀)

    @inbounds for j in j_start:j_stop
        for i in i_start:i_stop
            x₁ = workspace.cos_δ[i,j] * workspace.cos_α[i,j]
            y₁ = workspace.cos_δ[i,j] * workspace.sin_α[i,j]
            z₁ = workspace.sin_δ[i,j]
            d² = (x₁ - x₀)^2 + (y₁ - y₀)^2 + (z₁ - z₀)^2
            θ =  acos(clamp(1 - d² / 2, -one(T), one(T)))
            θ = max(θmin, θ)  # clamp to minimum θ
            m[i,j] += ifelse(θ < θmax, 
                             normalization * model(θ, Mh, z),
                             zero(T))
        end
    end
end

# fall back to generic profile painter if no specialized painter is defined for the model
function profile_paint!(m::Enmap{T, 2, Matrix{T}, Gnomonic{T}}, 
                        workspace::GnomonicProfileWorkspace, model, Mh, z, α₀, δ₀, 
                        θmax, normalization=1) where T
    profile_paint_generic!(m, workspace, model, Mh, z, α₀, δ₀, θmax, normalization)
end


# function profile_paint_generic!(m::HealpixMap{T, RingOrder}, w::HealpixSerialProfileWorkspace, 
#         model, Mh, z, α₀, δ₀, θmax, normalization=1) where T
#     ϕ₀ = α₀
#     θ₀ = T(π)/2 - δ₀
#     x₀, y₀, z₀ = ang2vec(θ₀, ϕ₀)
#     θmin = compute_θmin(model)
#     XGPaint.queryDiscRing!(w.disc_buffer, w.ringinfo, m.resolution, θ₀, ϕ₀, θmax)
#     for ir in w.disc_buffer
#         x₁, y₁, z₁ = w.posmap[ir]
#         d² = (x₁ - x₀)^2 + (y₁ - y₀)^2 + (z₁ - z₀)^2
#         θ = acos(clamp(1 - d² / 2, -one(T), one(T)))
#         θ = max(θmin, θ)  # clamp to minimum θ
#         m.pixels[ir] += ifelse(θ < θmax, 
#                                     normalization * model(θ, Mh, z),
#                                     zero(T))
#     end
# end

function profile_paint_generic!(m::HealpixMap{T, RingOrder}, workspace::HealpixRingProfileWorkspace{T}, 
        model, Mh, z, α₀, δ₀, θmax, normalization=1) where T
    ϕ₀ = mod(T(α₀), T(2π))  # Normalize RA to [0, 2π)
    θ₀ = T(π)/2 - δ₀
    x₀, y₀, z₀ = ang2vec(θ₀, ϕ₀)
    θmin = compute_θmin(model)
    
    # Get relevant rings for this disc
    ring_start, ring_end = get_relevant_rings(workspace.res, θ₀, θmax)
    
    for ring_idx in ring_start:ring_end
        # Get pixel ranges on this ring that intersect the disc
        range1, range2 = get_ring_disc_ranges(workspace, ring_idx, θ₀, ϕ₀, θmax)
        
        # Get precomputed ring info
        first_pixel = workspace.ring_first_pixels[ring_idx]
        
        # Process both ranges (range2 may be empty for no phi wraparound)
        for pixel_range in (range1, range2)
            for pix_idx in pixel_range
                # Convert ring pixel index to global healpix pixel index
                global_pix = first_pixel + pix_idx - 1
                
                # Get position of this pixel
                x₁, y₁, z₁ = pix2vecRing(workspace.res, global_pix)
                
                # Compute angular distance
                d² = (x₁ - x₀)^2 + (y₁ - y₀)^2 + (z₁ - z₀)^2
                θ = acos(clamp(1 - d² / 2, -one(T), one(T)))
                θ = max(θmin, θ)  # clamp to minimum θ
                
                # Add contribution to map
                m.pixels[global_pix] += ifelse(θ < θmax,
                                              normalization * model(θ, Mh, z),
                                              zero(T))
            end
        end
    end
end

# fall back to generic profile painter if no specialized painter is defined for the model
function profile_paint!(m::HealpixMap{T, RingOrder}, w::HealpixRingProfileWorkspace{T}, model, 
                        Mh, z, α₀, δ₀, θmax, normalization=1) where T
    profile_paint_generic!(m, w, model, Mh, z, α₀, δ₀, θmax, normalization)
end


# paint the the sources in the given range of indices
function paintrange!(irange::AbstractUnitRange, m, workspace, model, masses, redshifts, αs, δs)
    for i in irange
        α₀ = αs[i]
        δ₀ = δs[i]
        Mh = masses[i]
        z = redshifts[i]
        θmax_ = compute_θmax(model, Mh * XGPaint.M_sun, z)
        profile_paint!(m, workspace, model, Mh, z, α₀, δ₀, θmax_)
    end
end


_fillzero!(m) = fill!(m, zero(eltype(m)))
_fillzero!(m::HealpixMap) = fill!(m.pixels, zero(eltype(m)))

# paint! is threaded by default
function paint!(m, workspace, model, masses, redshifts, αs, δs; 
                zerobeforepainting=true)
    
    zerobeforepainting && _fillzero!(m)

    N_sources = length(masses)
    
    if N_sources < 2Threads.nthreads()  # don't thread if there are not many sources
        return paintrange!(1:N_sources, m, workspace, 
            model, masses, redshifts, αs, δs)
    end

    # Use ChunkSplitters for better load balancing
    Threads.@threads for chunk in chunks(1:N_sources; n=2*Threads.nthreads())
        paintrange!(chunk, m, workspace, 
            model, masses, redshifts, αs, δs)
    end
end


# for kSZ, we need to extend paintrange! and paint! to take in a velocity

# paint the the sources in the given range
function paintrange!(irange::AbstractUnitRange, m, workspace, model, 
                     masses, redshifts, αs, δs, proj_v_over_c)
    for i in irange
        θmax = compute_θmax(model, masses[i] * XGPaint.M_sun, redshifts[i])
        profile_paint!(m, workspace, model, 
            masses[i], redshifts[i], αs[i], δs[i], θmax, proj_v_over_c[i])
    end
end


# extend general paint! to take in a projected velocity
function paint!(m, workspace, model, masses, redshifts, αs, δs, proj_v_over_c; 
        zerobeforepainting=true)
    zerobeforepainting && _fillzero!(m)

    N_sources = length(masses)
    
    if N_sources < 2Threads.nthreads()  # don't thread if there are not many sources
        return paintrange!(1:N_sources, m, workspace, 
            model, masses, redshifts, αs, δs, proj_v_over_c)
    end

    # Use ChunkSplitters for better load balancing
    Threads.@threads for chunk in chunks(1:N_sources; n=2*Threads.nthreads())
        paintrange!(chunk, m, workspace, 
            model, masses, redshifts, αs, δs, proj_v_over_c)
    end
end


# # serial version of the paint function, mostly for debugging
# function paint!(m, workspace::HealpixSerialProfileWorkspace, model, masses, redshifts, αs, δs; 
#         zerobeforepainting=true)
#     zerobeforepainting && _fillzero!(m)
#     return paintrange!(1:length(masses), m, workspace, 
#         model, masses, redshifts, αs, δs)
# end
