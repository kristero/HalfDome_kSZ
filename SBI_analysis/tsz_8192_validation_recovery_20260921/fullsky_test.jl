# Isolated clean-map benchmark. The archived observation operators stay intact.
using TOML
const OUT = ENV["PREFLIGHT_OUTPUT"]
mkpath(OUT)
ENV["FLAMINGO_MODE"] = "probe"
ENV["FLAMINGO_OUTPUT"] = joinpath(OUT, "operator_probe.toml")
include(joinpath(ENV["FLAMINGO_CAMPAIGN"], "code", "process_maps.jl"))
include(joinpath(@__DIR__, "spherical_truncation_profiles.jl"))
using .SphericalTruncation

struct LogRedshiftGrid{I,R}
    interpolator::I
    ranges::R
end
@inline (a::LogRedshiftGrid)(lt,z,lm) = a.interpolator(lt,log(z),lm)
Base.size(a::LogRedshiftGrid) = size(a.interpolator)
const CACHE_STATS = Dict{String,Any}()

function build_visual_interpolator(cfg::VisualConfig)
    model = ChordMeanProfile(build_tsz_model(cfg), 4.0)
    factor = parse(Int,get(ENV,"PREFLIGHT_GRID_FACTOR","1"))
    rft = XGPaint.RadialFourierTransform(n=512,pad=cfg.interpolator_pad)
    lt = LinRange(log(minimum(rft.r)),log(maximum(rft.r)),512factor)
    lz = LinRange(log(.001),log(5.),256factor)
    lm = LinRange(12.,cfg.interpolator_logM_max,128factor)
    t = time()
    _,_,_,values = XGPaint.profile_grid(model,lt,exp.(lz),lm)
    @assert all(isfinite,values) && minimum(values)>=0
    CACHE_STATS["seconds"] = time()-t
    CACHE_STATS["underflow_count"] = count(<=(0),values)
    # Absolute floor, never a fraction of the brightest cell. Its contribution
    # is negligible in y units, but interpolation fidelity is still tested.
    values .= max.(values,1e-300)
    itp = XGPaint.Interpolations.interpolate(log.(values),
        XGPaint.BSpline(XGPaint.Cubic(XGPaint.Line(XGPaint.OnGrid()))))
    scaled = XGPaint.scale(itp,lt,lz,lm)
    return XGPaint.LogInterpolatorProfile(model,LogRedshiftGrid(scaled,(lt,lz,lm)))
end

function paint_visual_batch!(state,profile,xs,ys,zs,radii,
                             masses::Vector{Float64},redshifts::Vector{Float64})
    # Integrate the 3-D pressure first; sample the resulting 2-D column here.
    # The exact chord is applied AFTER interpolation to preserve the moving edge.
    locks = [ReentrantLock() for _ in state.workspace.ring_thetas]
    theta_min = exp(first(profile.itp.ranges[1]))
    updates = zeros(Int,Threads.maxthreadid())
    Threads.@threads :static for i in eachindex(masses)
        d = sqrt(xs[i]^2+ys[i]^2+zs[i]^2)
        ux,uy,uz = xs[i]/d,ys[i]/d,zs[i]/d
        tc,pc = Healpix.vec2ang(ux,uy,uz)
        pc = mod(pc,2pi)
        mass,z = masses[i],redshifts[i]
        t200 = theta_r200c(profile.model,mass,z)
        radius = min(4t200,pi)
        first_ring,last_ring = XGPaint.get_relevant_rings(state.workspace.res,tc,radius)
        n = 0
        for ring in first_ring:last_ring
            a,b = XGPaint.get_ring_disc_ranges(state.workspace,ring,tc,pc,radius)
            first_pixel = state.workspace.ring_first_pixels[ring]
            lock(locks[ring]) do
                for lp in Iterators.flatten((a,b))
                    gp = first_pixel+lp-1
                    px,py,pz = Healpix.pix2vecRing(state.workspace.res,gp)
                    theta = acos(clamp(ux*px+uy*py+uz*pz,-1.,1.))
                    theta < radius || continue
                    value = profile(max(theta,theta_min),mass,z)*chord_factor(theta,radius,4.)
                    state.m_hp.pixels[gp] += value
                    n += 1
                end
            end
        end
        updates[Threads.threadid()] += n
    end
    return length(masses)
end

function prepare_tsz_map_for_output(cfg::VisualConfig,raw;niter::Integer=0)
    alm = Healpix.map2alm(raw;lmax=healpix_default_lmax(4096),niter=niter)
    Healpix.almxfl!(alm,Healpix.gaussbeam(deg2rad(2/60),alm.lmax))
    return Healpix.alm2map(alm,4096)
end

function main()
    cfg = load_halfdome_fullsky_so_noise_config()
    @assert cfg.base_cfg.cl_lmax==7979 && cfg.fsky==.4 && cfg.mask_seed==12345
    started=time()
    signal=paint_halfdome_fullsky_signal_map(cfg.base_cfg)
    painted=time()
    write_npy_float64_vector(joinpath(OUT,"unmasked_clean_cl.npy"),compute_cl(cfg.base_cfg,signal),"sphere clean")
    mask=random_apodized_cap_mask(4096,.4,cfg.mask_apodization_arcmin,MersenneTwister(12345))
    signal.pixels .*= mask.mask.pixels
    write_npy_float64_vector(joinpath(OUT,"masked_clean_cl.npy"),compute_cl(cfg.base_cfg,signal),"sphere masked")
    save_toml(joinpath(OUT,"numerics.toml"),Dict("seconds"=>time()-started,
        "painting_and_beam_seconds"=>painted-started,"cache"=>CACHE_STATS,
        "raw_nside"=>cfg.base_cfg.nside,"output_nside"=>4096,
        "threads"=>Threads.nthreads(),"mask_sha256"=>pixel_sha(mask.mask),
        "model"=>"sphere4; point sampling; log-z chord-mean cache; fixed 2 arcmin beam"))
end
get(ENV,"PREFLIGHT_LOAD_ONLY","0")=="1" || main()
