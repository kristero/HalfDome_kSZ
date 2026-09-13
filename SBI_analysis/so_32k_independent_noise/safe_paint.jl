# The upstream painter threads over halos. Their footprints can share pixels.
# Synchronize one HEALPix ring at a time, without per-thread full-sky maps.
function locked_profile_paint!(m, workspace, locks, model, mass, redshift, ra, dec, radius)
    T = eltype(m.pixels)
    phi0 = mod(T(ra), T(2pi))
    theta0 = T(pi)/2 - dec
    x0, y0, z0 = Healpix.ang2vec(theta0, phi0)
    theta_min = XGPaint.compute_θmin(model)
    first_ring, last_ring = XGPaint.get_relevant_rings(workspace.res, theta0, radius)
    for ring in first_ring:last_ring
        range1, range2 = XGPaint.get_ring_disc_ranges(workspace, ring, theta0, phi0, radius)
        first_pixel = workspace.ring_first_pixels[ring]
        lock(locks[ring])
        try
            for pixel_range in (range1, range2), pixel in pixel_range
                index = first_pixel + pixel - 1
                x1, y1, z1 = Healpix.pix2vecRing(workspace.res, index)
                distance2 = (x1-x0)^2 + (y1-y0)^2 + (z1-z0)^2
                theta = max(theta_min, acos(clamp(1-distance2/2, -one(T), one(T))))
                m.pixels[index] += ifelse(theta < radius, model(theta, mass, redshift), zero(T))
            end
        finally
            unlock(locks[ring])
        end
    end
end

function XGPaint.paint!(m::HealpixMap{T, RingOrder}, workspace::XGPaint.HealpixRingProfileWorkspace{T},
                       model, masses, redshifts, ras, decs; zerobeforepainting=true) where T
    zerobeforepainting && fill!(m.pixels, zero(T))
    length(masses) == length(redshifts) == length(ras) == length(decs) || error("Unaligned halo vectors")
    locks = [ReentrantLock() for _ in workspace.ring_first_pixels]
    Threads.@threads for i in eachindex(masses)
        radius = XGPaint.compute_θmax(model, masses[i] * XGPaint.M_sun, redshifts[i])
        locked_profile_paint!(m, workspace, locks, model, masses[i], redshifts[i], ras[i], decs[i], radius)
    end
    return nothing
end
