"""Physical shell sums and the precise Battaglia families used in this pilot.

All particle inputs are already converted to comoving Mpc, physical Msun,
physical Mpc^-3, physical Mpc^2, Pa and K. SWIFT's stored electron density
already includes chemistry/ionization: do not multiply it by X_H again.
"""
import hashlib
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.constants import G, c, electron_mass, proton_mass, Boltzmann, physical_constants

MPC_CM = 3.08567758e24
MPC_M = MPC_CM / 100
MSUN_G = 1.98841e33
MSUN_KG = MSUN_G / 1000
SIGMA_T = physical_constants["Thomson cross section"][0]
XH_XGP = .76
MU_XGP_KG = (electron_mass + proton_mass * 2/(1+XH_XGP)) / .9
MU_XGP_MSUN = MU_XGP_KG / MSUN_KG
B12 = np.array([18.1, .497, 4.35, .154, -.00865, .0393, -.758, .731, .415])
B16 = np.array([4.e3, .88, 3.83, .29, -.03, .04, -.66, .19, -.025])
RMAX = 4.
MAP_N = 64
EDGES = np.unique(np.r_[0., np.geomspace(.025, 3., 31), 1., 2., np.geomspace(3., RMAX, 4)])
SUM_NAMES = ("count", "mass", "y", "y_noagn", "electrons", "electron_kT",
             "volume", "gas_pressure_volume", "hydrogen_mass", "helium_mass",
             "fully_ionized_electrons")


def p200(mass_msun, radius_mpc, baryon_fraction):
    mass = np.asarray(mass_msun)*MSUN_KG
    radius = np.asarray(radius_mpc)*MPC_M
    rho200 = mass/(4*np.pi/3*radius**3)
    return G*mass*rho200*baryon_fraction/(2*radius)


def hash_ids(ids):
    return hashlib.sha256(np.sort(np.asarray(ids,dtype="<u8")).tobytes()).hexdigest()


def accumulate(data, centre, radius, scale_factor, box, mass, baryon_fraction, heating, rotations=None):
    if not all(np.isfinite(value).all() for value in data.values()):
        raise ValueError('Non-finite particle values after physical-unit conversion')
    if np.any(data['Masses']<=0) or np.any(data['Densities']<=0):
        raise ValueError('Mass and density must be positive for the particle volume')
    if np.any(data['ElectronNumberDensities']<0) or np.any(data['ComptonYParameters']<0):
        raise ValueError('Thermal pressure and free-electron density cannot be negative')
    delta = data["Coordinates"] - centre
    delta -= box*np.rint(delta/box)
    relative = delta*scale_factor/radius
    x = np.linalg.norm(relative,axis=1)
    inside = x <= RMAX
    x = x[inside]
    selected = {name:np.asarray(value)[inside] for name,value in data.items()}
    relative = relative[inside]
    volume = selected["Masses"]/selected["Densities"]
    ne = selected["ElectronNumberDensities"]*volume
    recently = ((selected["LastAGNFeedbackScaleFactors"] >= heating["a_limit"]) &
                (selected["Temperatures"] >= heating["tmin_k"]) &
                (selected["Temperatures"] <= heating["tmax_k"]))
    weights = {
        "count":np.ones(len(x),dtype=np.int64), "mass":selected["Masses"],
        "y":selected["ComptonYParameters"],
        "y_noagn":np.where(recently,0.,selected["ComptonYParameters"]),
        "electrons":ne, "electron_kT":ne*Boltzmann*selected["Temperatures"],
        "volume":volume, "gas_pressure_volume":selected["Pressures"]*volume,
        "hydrogen_mass":selected["Masses"]*selected["Hydrogen"],
        "helium_mass":selected["Masses"]*selected["Helium"],
        "fully_ionized_electrons":selected["Masses"]*MSUN_KG/proton_mass*(selected["Hydrogen"]+selected["Helium"]/2),
    }
    result = {name:np.histogram(x,bins=EDGES,weights=value)[0] for name,value in weights.items()}
    result["aperture"] = np.array([len(x[x<=1]),selected["Masses"][x<=1].sum(),
                                   selected["ComptonYParameters"][x<=1].sum()])
    result["hsml_median_x"] = np.array([
        np.median(selected["SmoothingLengths"][(x>=lo)&(x<hi)])*scale_factor/radius
        if np.any((x>=lo)&(x<hi)) else np.nan for lo,hi in zip(EDGES[:-1],EDGES[1:])])
    result["ids"] = selected["ParticleIDs"]
    result["radius_samples"] = x
    result["hsml_samples"] = selected["SmoothingLengths"]*scale_factor/radius
    result["ids_hash"] = hash_ids(result["ids"])
    pixel_area = (2*RMAX/MAP_N)**2
    result["gas_map"] = np.histogram2d(relative[:,0],relative[:,1],bins=MAP_N,range=[[-RMAX,RMAX],[-RMAX,RMAX]],
        weights=selected["Masses"]/(baryon_fraction*mass))[0]/pixel_area
    y_normalization = SIGMA_T/(electron_mass*c*c)*p200(mass,radius,baryon_fraction)*radius*MPC_M
    result["y_map"] = np.histogram2d(relative[:,0],relative[:,1],bins=MAP_N,range=[[-RMAX,RMAX],[-RMAX,RMAX]],
        weights=selected["ComptonYParameters"]/(radius**2*y_normalization))[0]/pixel_area
    if rotations is not None:
        # Projection is linear: rotating each halo then stacking equals rotating
        # the unbinned spatial stack. These ten views are correlated.
        particle_y=selected["ComptonYParameters"]/(radius**2*y_normalization)
        particle_mass=selected["Masses"]/(baryon_fraction*mass)
        rotated=[]
        for rotation in np.asarray(rotations):
            view=relative@rotation.T
            rotated.append(np.histogram2d(view[:,0],view[:,1],bins=MAP_N,
                range=[[-RMAX,RMAX],[-RMAX,RMAX]],weights=particle_y)[0]/pixel_area)
        result["rotated_y_maps"]=np.asarray(rotated)
        voxel_volume=(2*RMAX/48)**3
        result["pressure_voxels"]=np.histogramdd(relative,bins=48,
            range=[[-RMAX,RMAX]]*3,weights=particle_y)[0]/voxel_volume
        result["gas_voxels"]=np.histogramdd(relative,bins=48,
            range=[[-RMAX,RMAX]]*3,weights=particle_mass)[0]/voxel_volume
    return result, selected


def physical_profiles(sums, halo, baryon_fraction):
    radius = np.asarray(halo["r_200_crit_mpc"])
    mass = np.asarray(halo["m200c_msun"])
    volume = 4*np.pi/3*radius[:,None]**3*np.diff(EDGES**3)[None,:]
    pressure = sums["y"]*MPC_M**2*electron_mass*c*c/SIGMA_T/(volume*MPC_M**3)
    pressure_noagn = sums["y_noagn"]*MPC_M**2*electron_mass*c*c/SIGMA_T/(volume*MPC_M**3)
    rho200 = mass/(4*np.pi/3*radius**3)
    critical_density = rho200/200
    pressure_normalization = p200(mass,radius,baryon_fraction)
    gas_density = sums["mass"]/volume
    electron_density = sums["electrons"]/volume
    xh = np.divide(sums["hydrogen_mass"].sum(axis=1),sums["mass"].sum(axis=1))
    yhe = np.divide(sums["helium_mass"].sum(axis=1),sums["mass"].sum(axis=1))
    n200_h = 200*critical_density*baryon_fraction*MSUN_KG/proton_mass*xh
    n200_e = 200*critical_density*baryon_fraction*MSUN_KG/proton_mass*(xh+yhe/2)
    return {"pressure_pa":pressure, "pressure_ratio":pressure/pressure_normalization[:,None],
            "pressure_noagn_ratio":pressure_noagn/pressure_normalization[:,None],
            "gas_density_msun_mpc3":gas_density,
            "gas_density_ratio":gas_density/(critical_density[:,None]*baryon_fraction),
            "electron_density_mpc3":electron_density,
            "electron_hydrogen_ratio":electron_density/n200_h[:,None],
            "electron_ionized_ratio":electron_density/n200_e[:,None],
            "tau_density_ratio":electron_density*MU_XGP_MSUN/(critical_density[:,None]*baryon_fraction),
            "pressure_nekt_pa":sums["electron_kT"]/(volume*MPC_M**3),
            "xh_mass_weighted":xh,"yhe_mass_weighted":yhe,
            "critical_density_msun_mpc3":critical_density,"p200_pa":pressure_normalization,
            "n200_hydrogen_mpc3":n200_h,"n200_ionized_mpc3":n200_e}


def shell_model(family, parameters, masses, edges=EDGES, redshift=0.):
    """Average the model over each physical shell and each halo's own mass."""
    nodes, weights = leggauss(32)
    lo, hi = edges[:-1], edges[1:]
    x = lo[:,None]+(hi-lo)[:,None]*(nodes+1)/2
    w = 3*x*x*weights*(hi-lo)[:,None]/(2*(hi**3-lo**3)[:,None])
    logm = np.log(np.asarray(masses)/1e14)
    amplitudes = np.asarray(parameters[:3])[:,None]*np.exp(np.asarray(parameters[3:6])[:,None]*logm[None,:]
                    +np.asarray(parameters[6:9])[:,None]*np.log1p(redshift))
    if family == "b12":
        p0, core, beta = amplitudes
        q = x[None,:,:]/core[:,None,None]
        values = .5176*p0[:,None,None]*q**(-.3)*(1+q)**(-beta[:,None,None])
    elif family == "b16":
        p0, alpha, beta = amplitudes
        q = x[None,:,:]/.5
        values = p0[:,None,None]*q**(-.2)*(1+q**alpha[:,None,None])**(-(beta[:,None,None]-.2)/alpha[:,None,None])
    else:
        raise ValueError(family)
    return np.sum(values*w[None,:,:],axis=-1)
