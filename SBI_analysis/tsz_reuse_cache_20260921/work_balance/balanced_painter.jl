# Same arithmetic and locks as frozen paint_shared!, with small scheduled blocks.
function paint_shared!(states, profiles, positions, masses, redshifts)
    workspace = states[1].workspace
    locks = [ReentrantLock() for _ in workspace.ring_thetas]
    theta_min = exp(first(profiles[1].itp.ranges[1]))
    # Fine-grained greedy scheduling prevents nearby catalogue tails from
    # remaining on one thread. Block size is a performance knob only.
    block_size = parse(Int,get(ENV,"HALO_BLOCK_SIZE","256"))
    @assert block_size > 0
    Threads.@threads :greedy for first in 1:block_size:length(masses)
        for i in first:min(first+block_size-1,length(masses))
        x, y, zpos = positions[1,i], positions[2,i], positions[3,i]
        distance = sqrt(x*x + y*y + zpos*zpos)
        ux, uy, uz = x/distance, y/distance, zpos/distance
        tc, pc = Healpix.vec2ang(ux, uy, uz)
        pc = mod(pc, 2pi)
        mass, z = masses[i], redshifts[i]
        logmass, logz = log10(mass), log(z)
        radius = min(4theta_r200c(profiles[1].model, mass, z), pi)
        first_ring, last_ring = XGPaint.get_relevant_rings(workspace.res, tc, radius)
        for ring in first_ring:last_ring
            a, b = XGPaint.get_ring_disc_ranges(workspace, ring, tc, pc, radius)
            first_pixel = workspace.ring_first_pixels[ring]
            lock(locks[ring]) do
                for lp in Iterators.flatten((a, b))
                    pixel = first_pixel + lp - 1
                    px, py, pz = Healpix.pix2vecRing(workspace.res, pixel)
                    theta = acos(clamp(ux*px + uy*py + uz*pz, -1., 1.))
                    theta < radius || continue
                    chord = chord_factor(theta, radius, 4.)
                    logtheta = log(max(theta, theta_min))
                    for j in eachindex(profiles)
                        value = exp(profiles[j].itp.interpolator(logtheta, logz, logmass))
                        states[j].m_hp.pixels[pixel] += value * chord
                    end
                end
            end
        end
    end
    end # greedy block loop
    return length(masses)
end
