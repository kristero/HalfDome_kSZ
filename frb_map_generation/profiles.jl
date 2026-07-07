

# Import ChunkSplitters for better threading
using ChunkSplitters: chunks

# RECTANGULAR WORKSPACES


struct CarClenshawCurtisProfileWorkspace{T,A<:AbstractArray{T,2}} <: AbstractProfileWorkspace{T}
    sin_Î±::A
    cos_Î±::A
    sin_Î´::A
    cos_Î´::A
end

function profileworkspace(shape, wcs::CarClenshawCurtis)
    Î±_map, Î´_map = posmap(shape, wcs)
    return CarClenshawCurtisProfileWorkspace(
        sin.(Î±_map), cos.(Î±_map), sin.(Î´_map), cos.(Î´_map))
end


struct GnomonicProfileWorkspace{T,A<:AbstractArray{T,2}} <: AbstractProfileWorkspace{T}
    sin_Î±::A
    cos_Î±::A
    sin_Î´::A
    cos_Î´::A
end

function profileworkspace(shape, wcs::Gnomonic)
    Î±_map, Î´_map = posmap(shape, wcs)
    return GnomonicProfileWorkspace(
        sin.(Î±_map), cos.(Î±_map), sin.(Î´_map), cos.(Î´_map))
end

Base.show(io::IO, w::AbstractProfileWorkspace) = print(io, "$(typeof(w))")

@inline function interpolator_stage_start(label::AbstractString; verbose::Bool=true)
    if verbose
        println("[interpolator] START ", label)
    end
    return time_ns()
end

@inline function interpolator_stage_end(label::AbstractString, t0_ns; verbose::Bool=true, extra::AbstractString="")
    if verbose
        elapsed_s = (time_ns() - t0_ns) / 1.0e9
        suffix = isempty(extra) ? "" : " " * extra
        println("[interpolator] END   ", label, " wall=", round(elapsed_s; digits=3), " s", suffix)
    end
    return nothing
end


function profile_grid(model::AbstractGNFW{T}; N_z=256, N_logM=256, N_logÎ¸=512, z_min=1e-3, 
        z_max=5.0, logM_min=11, logM_max=15.7, logÎ¸_min=-16.5, logÎ¸_max=2.5) where T

    logÎ¸s = LinRange(logÎ¸_min, logÎ¸_max, N_logÎ¸)
    redshifts = LinRange(z_min, z_max, N_z)
    logMs = LinRange(logM_min, logM_max, N_logM)
    return profile_grid(model, logÎ¸s, redshifts, logMs)
end

function profile_grid(model::AbstractGNFW{T}, logÎ¸s, redshifts, logMs) where T

    N_logÎ¸, N_z, N_logM = length(logÎ¸s), length(redshifts), length(logMs)
    println(
        "[interpolator] profile_grid dims: N_logθ=", N_logÎ¸,
        " N_z=", N_z,
        " N_logM=", N_logM,
        " threads=", Threads.nthreads()
    )
    alloc_t0 = interpolator_stage_start("profile_grid allocation")
    A = zeros(T, (N_logÎ¸, N_z, N_logM))
    interpolator_stage_end(
        "profile_grid allocation",
        alloc_t0;
        extra="size=$(size(A)) eltype=$(T)"
    )

    # Use ChunkSplitters for better load balancing
    eval_t0 = interpolator_stage_start("profile_grid threaded evaluation")
    Threads.@threads for chunk in chunks(1:N_logM; n=Threads.nthreads())
        for im in chunk
            logM = logMs[im]
            M = 10^(logM)
            for (iz, z) in enumerate(redshifts)
                for iÎ¸ in 1:N_logÎ¸
                    Î¸ = exp(logÎ¸s[iÎ¸])
                    A[iÎ¸, iz, im] = max(zero(T), model(Î¸, M, z))
                end
            end
        end
    end
    interpolator_stage_end("profile_grid threaded evaluation", eval_t0)

    return logÎ¸s, redshifts, logMs, A
end


"""
Computes a real-space beam interpolator and a maximum
"""
function realspacegaussbeam(::Type{T}, Î¸_FWHM::Ti; rtol=1e-24, N_Î¸::Int=2000) where {T,Ti}
    Nlmax = ceil(Int, log2(8Ï€ / Î¸_FWHM))
    lmax = 2^Nlmax

    b_l = gaussbeam(Î¸_FWHM, lmax)
    Î¸s = LinRange(zero(Î¸_FWHM), 5Î¸_FWHM, N_Î¸)
    b_Î¸ = XGPaint.bl2beam(b_l, Î¸s)
    atol = b_Î¸[begin] * rtol
    i_max = findfirst(<(atol), b_Î¸)

    Î¸s = convert(LinRange{T, Int}, Î¸s[begin:i_max])
    beam_real_interp = cubic_spline_interpolation(
        Î¸s, T.(b_Î¸[begin:i_max]), extrapolation_bc=zero(T))
    return beam_real_interp, Î¸s
end


# function realspacebeampaint!(hp_map, w::HealpixSerialProfileWorkspace, realprofile, flux, Î¸â‚€, Ï•â‚€)
#     xâ‚€, yâ‚€, zâ‚€ = ang2vec(Î¸â‚€, Ï•â‚€)
#     XGPaint.queryDiscRing!(w.disc_buffer, w.ringinfo, hp_map.resolution, Î¸â‚€, Ï•â‚€, w.Î¸max)

#     for ir in w.disc_buffer
#         xâ‚, yâ‚, zâ‚ = w.posmap.pixels[ir]
#         dÂ² = (xâ‚ - xâ‚€)^2 + (yâ‚ - yâ‚€)^2 + (zâ‚ - zâ‚€)^2
#         Î¸ = acos(1 - dÂ² / 2)
#         hp_map.pixels[ir] += flux * realprofile(Î¸)
#     end
# end


"""Apply a beam to a profile grid"""
function transform_profile_grid!(y_prof_grid, rft, lbeam)
    N_z = size(y_prof_grid, 2)
    N_logM = size(y_prof_grid, 3)
    N_profiles = N_z * N_logM
    rfts = [deepcopy(rft) for _ in 1:Threads.nthreads()]

    Threads.@threads for chunk in chunks(1:N_profiles; n=Threads.nthreads())
        local_rft = rfts[Threads.threadid()]
        for idx in chunk
            i = 1 + ((idx - 1) % N_z)
            j = 1 + ((idx - 1) Ã· N_z)
            rprof = copy(@view y_prof_grid[:, i, j])
            lprof = real2harm(local_rft, rprof)
            lprof .*= lbeam
            reverse!(lprof)
            y_prof_grid[:, i, j] .= harm2real(local_rft, lprof)
        end
    end
    return nothing
#=
    for i in axes(y_prof_grid, 2)
        for j in axes(y_prof_grid, 3)
            rprof .= y_prof_grid[:,i,j]
            lprof = real2harm(rft, rprof)
            lprof .*= lbeam
            reverse!(lprof)
            rprofâ€² = harm2real(rft, lprof)
            y_prof_grid[:,i,j] .= rprofâ€²
        end
    end
=#
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
            j = 1 + ((idx - 1) Ã· N_z)
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
#=
    for i in axes(y_prof_grid, 2)
        for j in axes(y_prof_grid, 3)
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
=#
end

function replace_nonpositive_with_floor!(y_prof_grid)
    T = eltype(y_prof_grid)
    N_values = length(y_prof_grid)
    nthreads = Threads.nthreads()
    local_mins = fill(typemax(T), nthreads)
    local_positive_counts = zeros(Int, nthreads)
    local_bad_counts = zeros(Int, nthreads)

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

function apply_gaussian_beam_to_profile_grid!(
    y_prof_grid,
    rft,
    gaussian_beam_fwhm_arcmin::Real
)
    gaussian_beam_fwhm_arcmin > 0 || error("gaussian_beam_fwhm_arcmin must be positive.")

    Î¸_FWHM = deg2rad(Float64(gaussian_beam_fwhm_arcmin) / 60.0)
    l_template = real2harm(rft, copy(y_prof_grid[:, 1, 1]))
    lbeam = gaussbeam(Î¸_FWHM, length(l_template) - 1)

    transform_profile_grid!(y_prof_grid, rft, lbeam)
    cleanup_negatives!(y_prof_grid)
    return nothing
end

function gaussian_beam_cache_matches(model_grid, apply_gaussian_beam::Bool, gaussian_beam_fwhm_arcmin::Real)
    if !haskey(model_grid, "apply_gaussian_beam")
        return !apply_gaussian_beam
    end

    cached_apply_gaussian_beam = Bool(model_grid["apply_gaussian_beam"])
    cached_gaussian_beam_fwhm_arcmin = Float64(get(model_grid, "gaussian_beam_fwhm_arcmin", 0.5))

    return cached_apply_gaussian_beam == apply_gaussian_beam &&
           (!apply_gaussian_beam ||
            isapprox(cached_gaussian_beam_fwhm_arcmin, Float64(gaussian_beam_fwhm_arcmin); atol=1e-12, rtol=0.0))
end



# get angular size in radians of radius to stop at
function compute_Î¸max(model::AbstractProfile{T}, M_Î”, z; mult=4) where T
    r = R_Î”(model, M_Î”, z)
    return T(mult * angular_size(model, r, z))
end

# prevent infinities at cusp
compute_Î¸min(model::AbstractInterpolatorProfile) = exp(first(first(model.itp.ranges)))
compute_Î¸min(::AbstractProfile{T}) where T = eps(T) 


# find maximum radius to integrate to
function build_max_paint_logradius(logÎ¸s, redshifts, logMs, 
                              A::AbstractArray{T}; rtol=1e-2) where T
    
    logRs = zeros(T, (size(A)[2:3]))
    N_logM = length(logMs)
    N_logÎ¸ = length(logÎ¸s)
    dF_r = zeros(N_logÎ¸)
    
    for im in 1:N_logM
        for (iz, z) in enumerate(redshifts)
            s = zero(T)
            for iÎ¸ in 1:(N_logÎ¸-1)
                Î¸â‚ = exp(logÎ¸s[iÎ¸])
                Î¸â‚‚ = exp(logÎ¸s[iÎ¸+1])
                fâ‚ = A[iÎ¸, iz, im] * Î¸â‚
                fâ‚‚ = A[iÎ¸+1, iz, im] * Î¸â‚‚
                s += (Î¸â‚‚ - Î¸â‚) * (fâ‚ + fâ‚‚) / 2

                dF_r[iÎ¸] = s
            end

            threshold = (1-rtol) * s
            for iÎ¸ in (N_logÎ¸-1):-1:1
                if dF_r[iÎ¸] < threshold
                    logRs[iz, im] = min(logÎ¸s[iÎ¸], log(Ï€))
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

A profile that interpolates over a positive-definite function (Î¸, z, M_halo), but internally
interpolates over log(Î¸) and log10(M) using a given interpolator. Evaluation of this profile
is then done by exponentiating the result of the interpolator.

```
    f(Î¸, z, M) = exp(itp(log(Î¸), z, log10(M)))
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
# IMPORTANT: for backwards compat, interpolator internal order is Î¸, z, mass
# which DIFFERS from the rest of the code which is (Î¸, mass, z, Î±, Î´)
# should fix this at some point
@inline (ip::LogInterpolatorProfile)(Î¸, Mh_Msun, z) = exp(ip.itp(log(Î¸), z, log10(Mh_Msun)))

Base.show(io::IO, ip::LogInterpolatorProfile{T,P,I1}) where {T,P,I1} = print(
    io, "LogInterpolatorProfile{$(T),\n  $(P),\n  ...} interpolating over size ", size(ip.itp))

function cleanup_nonpositive_enabled()
    raw = lowercase(strip(get(ENV, "XGPAINT_CLEANUP_NONPOSITIVE", "true")))
    raw in ("1", "true", "t", "yes", "y", "on") && return true
    raw in ("0", "false", "f", "no", "n", "off") && return false
    error("Invalid XGPAINT_CLEANUP_NONPOSITIVE=$(repr(raw)).")
end

"""Helper function to build a (θ, z, Mh) interpolator"""
function build_interpolator(model::AbstractProfile; cache_file::String="", 
                            N_logθ=512, pad=256, overwrite=true, verbose=true)

    cleanup_nonpositive = cleanup_nonpositive_enabled()
    verbose && println(
        "[interpolator] config overwrite=", overwrite,
        " cache_file=", isempty(cache_file) ? "<none>" : cache_file,
        " cleanup_nonpositive=", cleanup_nonpositive,
        " N_logθ=", N_logθ,
        " pad=", pad,
        " threads=", Threads.nthreads()
    )

    if overwrite || (isfile(cache_file) == false)
        verbose && print("Building new interpolator from model.\n")
        rft_t0 = interpolator_stage_start("RadialFourierTransform"; verbose=verbose)
        rft = RadialFourierTransform(n=N_logθ, pad=pad)
        interpolator_stage_end("RadialFourierTransform", rft_t0; verbose=verbose)

        range_t0 = interpolator_stage_start("rft radius bounds"; verbose=verbose)
        logθ_min, logθ_max = log(minimum(rft.r)), log(maximum(rft.r))
        interpolator_stage_end(
            "rft radius bounds",
            range_t0;
            verbose=verbose,
            extra="logθ_min=$(logθ_min) logθ_max=$(logθ_max)"
        )

        grid_t0 = interpolator_stage_start("profile_grid"; verbose=verbose)
        prof_logθs, prof_redshift, prof_logMs, prof_y = profile_grid(model; 
            N_logθ=N_logθ, logθ_min=logθ_min, logθ_max=logθ_max)
        interpolator_stage_end("profile_grid", grid_t0; verbose=verbose)
        if length(cache_file) > 0
            verbose && print("Saving new interpolator to $(cache_file).\n")
            save_t0 = interpolator_stage_start("cache save"; verbose=verbose)
            save(cache_file, Dict("prof_logθs"=>prof_logθs, 
                "prof_redshift"=>prof_redshift, "prof_logMs"=>prof_logMs, "prof_y"=>prof_y))
            interpolator_stage_end("cache save", save_t0; verbose=verbose)
        end
    else
        print("Found cached Battaglia profile model. Loading from disk.\n")
        load_t0 = interpolator_stage_start("cache load"; verbose=verbose)
        model_grid = load(cache_file)
        interpolator_stage_end("cache load", load_t0; verbose=verbose)

        unpack_t0 = interpolator_stage_start("cache unpack"; verbose=verbose)
        prof_logθs, prof_redshift, prof_logMs, prof_y = model_grid["prof_logθs"], 
            model_grid["prof_redshift"], model_grid["prof_logMs"], model_grid["prof_y"]
        interpolator_stage_end(
            "cache unpack",
            unpack_t0;
            verbose=verbose,
            extra="size=$(size(prof_y))"
        )
    end
    
    # --- avoid log(0) / negative ---
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
        if replaced_count > 0
            verbose && println("Replaced ", replaced_count,
                               " <=0 entries in prof_y with floor = ", floor_val)
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
    # --------------------------------
    log_t0 = interpolator_stage_start("log transform"; verbose=verbose)
    log_prof_y = log.(prof_y)
    interpolator_stage_end("log transform", log_t0; verbose=verbose)

    interpolate_t0 = interpolator_stage_start("Interpolations.interpolate"; verbose=verbose)
    itp = Interpolations.interpolate(log_prof_y, BSpline(Cubic(Line(OnGrid()))))
    interpolator_stage_end("Interpolations.interpolate", interpolate_t0; verbose=verbose)

    scale_t0 = interpolator_stage_start("scale"; verbose=verbose)
    interp_model = scale(itp, prof_logθs, prof_redshift, prof_logMs)
    interpolator_stage_end("scale", scale_t0; verbose=verbose)

    wrap_t0 = interpolator_stage_start("LogInterpolatorProfile wrapper"; verbose=verbose)
    wrapped_model = LogInterpolatorProfile(model, interp_model)
    interpolator_stage_end("LogInterpolatorProfile wrapper", wrap_t0; verbose=verbose)
    return wrapped_model
end


function profile_paint_generic!(m::Enmap{T, 2, Matrix{T}, CarClenshawCurtis{T}},
                        workspace::CarClenshawCurtisProfileWorkspace, model, Mh, z, Î±â‚€, Î´â‚€, 
                        Î¸max, normalization=1) where T

    # get indices of the region to work on
    i1, j1 = sky2pix(m, Î±â‚€ - Î¸max, Î´â‚€ - Î¸max)
    i2, j2 = sky2pix(m, Î±â‚€ + Î¸max, Î´â‚€ + Î¸max)
    i_start = floor(Int, max(min(i1, i2), 1))
    i_stop = ceil(Int, min(max(i1, i2), size(m, 1)))
    j_start = floor(Int, max(min(j1, j2), 1))
    j_stop = ceil(Int, min(max(j1, j2), size(m, 2)))
    Î¸min = compute_Î¸min(model)

    xâ‚€ = cos(Î´â‚€) * cos(Î±â‚€)
    yâ‚€ = cos(Î´â‚€) * sin(Î±â‚€) 
    zâ‚€ = sin(Î´â‚€)

    @inbounds for j in j_start:j_stop
        for i in i_start:i_stop
            xâ‚ = workspace.cos_Î´[i,j] * workspace.cos_Î±[i,j]
            yâ‚ = workspace.cos_Î´[i,j] * workspace.sin_Î±[i,j]
            zâ‚ = workspace.sin_Î´[i,j]
            dÂ² = (xâ‚ - xâ‚€)^2 + (yâ‚ - yâ‚€)^2 + (zâ‚ - zâ‚€)^2
            Î¸ =  acos(clamp(1 - dÂ² / 2, -one(T), one(T)))
            Î¸ = max(Î¸min, Î¸)  # clamp to minimum Î¸
            m[i,j] += ifelse(Î¸ < Î¸max, 
                             T(normalization * model(Î¸, Mh, z)),
                             zero(T))
        end
    end
end

# fall back to generic profile painter if no specialized painter is defined for the model
function profile_paint!(m::Enmap{T, 2, Matrix{T}, CarClenshawCurtis{T}}, 
                        workspace::CarClenshawCurtisProfileWorkspace, model, 
                        Mh, z, Î±â‚€, Î´â‚€, Î¸max, normalization=1) where T
    profile_paint_generic!(m, workspace, model, Mh, z, Î±â‚€, Î´â‚€, Î¸max, normalization)
end


function profile_paint_generic!(m::Enmap{T, 2, Matrix{T}, Gnomonic{T}}, 
                                workspace::GnomonicProfileWorkspace, model, 
                                Mh, z, Î±â‚€, Î´â‚€, Î¸max, normalization=1) where T

    # get indices of the region to work on
    i1, j1 = sky2pix(m, Î±â‚€ - Î¸max, Î´â‚€ - Î¸max)
    i2, j2 = sky2pix(m, Î±â‚€ + Î¸max, Î´â‚€ + Î¸max)
    i_start = floor(Int, max(min(i1, i2), 1))
    i_stop = ceil(Int, min(max(i1, i2), size(m, 1)))
    j_start = floor(Int, max(min(j1, j2), 1))
    j_stop = ceil(Int, min(max(j1, j2), size(m, 2)))
    Î¸min = compute_Î¸min(model)

    xâ‚€ = cos(Î´â‚€) * cos(Î±â‚€)
    yâ‚€ = cos(Î´â‚€) * sin(Î±â‚€) 
    zâ‚€ = sin(Î´â‚€)

    @inbounds for j in j_start:j_stop
        for i in i_start:i_stop
            xâ‚ = workspace.cos_Î´[i,j] * workspace.cos_Î±[i,j]
            yâ‚ = workspace.cos_Î´[i,j] * workspace.sin_Î±[i,j]
            zâ‚ = workspace.sin_Î´[i,j]
            dÂ² = (xâ‚ - xâ‚€)^2 + (yâ‚ - yâ‚€)^2 + (zâ‚ - zâ‚€)^2
            Î¸ =  acos(clamp(1 - dÂ² / 2, -one(T), one(T)))
            Î¸ = max(Î¸min, Î¸)  # clamp to minimum Î¸
            m[i,j] += ifelse(Î¸ < Î¸max, 
                             normalization * model(Î¸, Mh, z),
                             zero(T))
        end
    end
end

# fall back to generic profile painter if no specialized painter is defined for the model
function profile_paint!(m::Enmap{T, 2, Matrix{T}, Gnomonic{T}}, 
                        workspace::GnomonicProfileWorkspace, model, Mh, z, Î±â‚€, Î´â‚€, 
                        Î¸max, normalization=1) where T
    profile_paint_generic!(m, workspace, model, Mh, z, Î±â‚€, Î´â‚€, Î¸max, normalization)
end


# function profile_paint_generic!(m::HealpixMap{T, RingOrder}, w::HealpixSerialProfileWorkspace, 
#         model, Mh, z, Î±â‚€, Î´â‚€, Î¸max, normalization=1) where T
#     Ï•â‚€ = Î±â‚€
#     Î¸â‚€ = T(Ï€)/2 - Î´â‚€
#     xâ‚€, yâ‚€, zâ‚€ = ang2vec(Î¸â‚€, Ï•â‚€)
#     Î¸min = compute_Î¸min(model)
#     XGPaint.queryDiscRing!(w.disc_buffer, w.ringinfo, m.resolution, Î¸â‚€, Ï•â‚€, Î¸max)
#     for ir in w.disc_buffer
#         xâ‚, yâ‚, zâ‚ = w.posmap[ir]
#         dÂ² = (xâ‚ - xâ‚€)^2 + (yâ‚ - yâ‚€)^2 + (zâ‚ - zâ‚€)^2
#         Î¸ = acos(clamp(1 - dÂ² / 2, -one(T), one(T)))
#         Î¸ = max(Î¸min, Î¸)  # clamp to minimum Î¸
#         m.pixels[ir] += ifelse(Î¸ < Î¸max, 
#                                     normalization * model(Î¸, Mh, z),
#                                     zero(T))
#     end
# end

function profile_paint_generic!(m::HealpixMap{T, RingOrder}, workspace::HealpixRingProfileWorkspace{T}, 
        model, Mh, z, Î±â‚€, Î´â‚€, Î¸max, normalization=1) where T
    Ï•â‚€ = mod(T(Î±â‚€), T(2Ï€))  # Normalize RA to [0, 2Ï€)
    Î¸â‚€ = T(Ï€)/2 - Î´â‚€
    xâ‚€, yâ‚€, zâ‚€ = ang2vec(Î¸â‚€, Ï•â‚€)
    Î¸min = compute_Î¸min(model)
    
    # Get relevant rings for this disc
    ring_start, ring_end = get_relevant_rings(workspace.res, Î¸â‚€, Î¸max)
    
    for ring_idx in ring_start:ring_end
        # Get pixel ranges on this ring that intersect the disc
        range1, range2 = get_ring_disc_ranges(workspace, ring_idx, Î¸â‚€, Ï•â‚€, Î¸max)
        
        # Get precomputed ring info
        first_pixel = workspace.ring_first_pixels[ring_idx]
        
        # Process both ranges (range2 may be empty for no phi wraparound)
        for pixel_range in (range1, range2)
            for pix_idx in pixel_range
                # Convert ring pixel index to global healpix pixel index
                global_pix = first_pixel + pix_idx - 1
                
                # Get position of this pixel
                xâ‚, yâ‚, zâ‚ = pix2vecRing(workspace.res, global_pix)
                
                # Compute angular distance
                dÂ² = (xâ‚ - xâ‚€)^2 + (yâ‚ - yâ‚€)^2 + (zâ‚ - zâ‚€)^2
                Î¸ = acos(clamp(1 - dÂ² / 2, -one(T), one(T)))
                Î¸ = max(Î¸min, Î¸)  # clamp to minimum Î¸
                
                # Add contribution to map
                m.pixels[global_pix] += ifelse(Î¸ < Î¸max,
                                              normalization * model(Î¸, Mh, z),
                                              zero(T))
            end
        end
    end
end

# fall back to generic profile painter if no specialized painter is defined for the model
function profile_paint!(m::HealpixMap{T, RingOrder}, w::HealpixRingProfileWorkspace{T}, model, 
                        Mh, z, Î±â‚€, Î´â‚€, Î¸max, normalization=1) where T
    profile_paint_generic!(m, w, model, Mh, z, Î±â‚€, Î´â‚€, Î¸max, normalization)
end


# paint the the sources in the given range of indices
function paintrange!(irange::AbstractUnitRange, m, workspace, model, masses, redshifts, Î±s, Î´s)
    for i in irange
        Î±â‚€ = Î±s[i]
        Î´â‚€ = Î´s[i]
        Mh = masses[i]
        z = redshifts[i]
        Î¸max_ = compute_Î¸max(model, Mh * XGPaint.M_sun, z)
        profile_paint!(m, workspace, model, Mh, z, Î±â‚€, Î´â‚€, Î¸max_)
    end
end


_fillzero!(m) = fill!(m, zero(eltype(m)))
_fillzero!(m::HealpixMap) = fill!(m.pixels, zero(eltype(m)))

# paint! is threaded by default
function paint!(m, workspace, model, masses, redshifts, Î±s, Î´s; 
                zerobeforepainting=true)
    
    zerobeforepainting && _fillzero!(m)

    N_sources = length(masses)
    
    if N_sources < 2Threads.nthreads()  # don't thread if there are not many sources
        return paintrange!(1:N_sources, m, workspace, 
            model, masses, redshifts, Î±s, Î´s)
    end

    # Use ChunkSplitters for better load balancing
    Threads.@threads for chunk in chunks(1:N_sources; n=2*Threads.nthreads())
        paintrange!(chunk, m, workspace, 
            model, masses, redshifts, Î±s, Î´s)
    end
end


# for kSZ, we need to extend paintrange! and paint! to take in a velocity

# paint the the sources in the given range
function paintrange!(irange::AbstractUnitRange, m, workspace, model, 
                     masses, redshifts, Î±s, Î´s, proj_v_over_c)
    for i in irange
        Î¸max = compute_Î¸max(model, masses[i] * XGPaint.M_sun, redshifts[i])
        profile_paint!(m, workspace, model, 
            masses[i], redshifts[i], Î±s[i], Î´s[i], Î¸max, proj_v_over_c[i])
    end
end


# extend general paint! to take in a projected velocity
function paint!(m, workspace, model, masses, redshifts, Î±s, Î´s, proj_v_over_c; 
        zerobeforepainting=true)
    zerobeforepainting && _fillzero!(m)

    N_sources = length(masses)
    
    if N_sources < 2Threads.nthreads()  # don't thread if there are not many sources
        return paintrange!(1:N_sources, m, workspace, 
            model, masses, redshifts, Î±s, Î´s, proj_v_over_c)
    end

    # Use ChunkSplitters for better load balancing
    Threads.@threads for chunk in chunks(1:N_sources; n=2*Threads.nthreads())
        paintrange!(chunk, m, workspace, 
            model, masses, redshifts, Î±s, Î´s, proj_v_over_c)
    end
end


# # serial version of the paint function, mostly for debugging
# function paint!(m, workspace::HealpixSerialProfileWorkspace, model, masses, redshifts, Î±s, Î´s; 
#         zerobeforepainting=true)
#     zerobeforepainting && _fillzero!(m)
#     return paintrange!(1:length(masses), m, workspace, 
#         model, masses, redshifts, Î±s, Î´s)
# end
