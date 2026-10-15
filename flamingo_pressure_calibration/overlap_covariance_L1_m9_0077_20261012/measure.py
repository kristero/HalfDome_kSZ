"""Compact linear accumulators for four gas components and three spheres."""
import numpy as np
import pilot_physics as P

RADII=np.array([2.,3.,4.])
COMPONENTS=('central_bound','own_satellites_bound','other_halos_bound','unbound')
QUANTITIES=('pressure','electron_density','gas_density')
NPIX=64
EXTENT=4.
PIXEL_AREA=(2*EXTENT/NPIX)**2
HS_EDGES=np.geomspace(1e-6,1e3,1025)
NB=len(P.EDGES)-1


def pixel_annuli():
    xy=-EXTENT+(np.arange(NPIX)+.5)*2*EXTENT/NPIX
    xx,yy=np.meshgrid(xy,xy,indexing='ij')
    radii=np.hypot(xx,yy)
    index=np.searchsorted(P.EDGES,radii.ravel(),side='right')-1
    valid=(index>=0)&(index<NB)
    return index,valid,np.bincount(index[valid],minlength=NB)


ANNULUS,ANNULUS_VALID,ANNULUS_COUNT=pixel_annuli()


def annular_means(maps):
    result=np.zeros(maps.shape[:-2]+(NB,))
    for i,image in enumerate(maps.reshape(-1,NPIX*NPIX)):
        total=np.bincount(ANNULUS[ANNULUS_VALID],weights=image[ANNULUS_VALID],minlength=NB)
        result.reshape(-1,NB)[i]=np.divide(total,ANNULUS_COUNT,out=np.zeros(NB),where=ANNULUS_COUNT>0)
    return result


def components(membership, central_id, satellite_ids):
    """Exclusive components; IDs are original halo-finder IDs, never SOAP rows."""
    category=np.where(membership<0,3,2).astype(np.int64)
    category[np.isin(membership,satellite_ids)]=1
    category[membership==central_id]=0
    return category


def accumulate(data, membership, halo, fb, heating, rotations):
    """Input has already been clipped to the exact periodic 4R sphere."""
    rotations=np.asarray(rotations)
    delta=data['Coordinates']-halo['centre_cmpc']
    delta-=halo['box']*np.rint(delta/halo['box'])
    relative=delta*halo['a']/halo['radius']
    x=np.linalg.norm(relative,axis=1)
    assert np.all(x<=4.+1e-12)
    category=components(membership,halo['id'],halo['satellites'])
    shell=np.searchsorted(P.EDGES,x,side='right')-1
    shell=np.minimum(shell,NB-1)
    code=category*NB+shell
    volume=data['Masses']/data['Densities']
    ne=data['ElectronNumberDensities']*volume
    recently=(data['LastAGNFeedbackScaleFactors']>=heating['a_limit']) & (data['Temperatures']>=heating['tmin_k']) & (data['Temperatures']<=heating['tmax_k'])
    weights={
        'count':np.ones(len(x)),'mass':data['Masses'],'y':data['ComptonYParameters'],
        'y_noagn':np.where(recently,0.,data['ComptonYParameters']),
        'electrons':ne,'electron_kT':ne*P.Boltzmann*data['Temperatures'],
        'volume':volume,'gas_pressure_volume':data['Pressures']*volume,
        'hydrogen_mass':data['Masses']*data['Hydrogen'],
        'helium_mass':data['Masses']*data['Helium'],
        'fully_ionized_electrons':data['Masses']*P.MSUN_KG/P.proton_mass*(data['Hydrogen']+data['Helium']/2)}
    # NumPy returns an integer bincount for empty input, even with float weights.
    # Keep the accumulator contract identical for empty and populated cells.
    sums=np.stack([np.bincount(code,weights=weights[name],minlength=4*NB).reshape(4,NB) for name in P.SUM_NAMES],axis=1).astype(np.float64,copy=False)
    hs=data['SmoothingLengths']*halo['a']/halo['radius']
    hb=np.searchsorted(HS_EDGES,hs,side='right')-1
    assert np.all((hb>=0)&(hb<len(HS_EDGES)-1)), 'Smoothing histogram overflow'
    hist=np.bincount(shell*(len(HS_EDGES)-1)+hb,minlength=NB*(len(HS_EDGES)-1)).reshape(NB,-1).astype(np.uint32)
    rho_c=halo['mass']/(4*np.pi/3*halo['radius']**3)/200
    p200=P.p200(halo['mass'],halo['radius'],fb)
    ynorm=P.SIGMA_T/(P.electron_mass*P.c**2)*p200*halo['radius']*P.MPC_M
    # Each particle weight is a dimensionless volume integral in x=r/R200c.
    projection_weights=np.stack([data['ComptonYParameters']/(halo['radius']**2*ynorm),
        ne*P.MU_XGP_MSUN/(fb*rho_c*halo['radius']**3),
        data['Masses']/(fb*rho_c*halo['radius']**3)])
    # Disjoint radial bands are cumulatively summed to form exact nested spheres.
    band=np.searchsorted(RADII,x,side='left')
    band=np.minimum(band,2)
    maps=np.zeros((2,3,4,3,NPIX,NPIX))
    view_profiles=np.zeros((len(rotations),3,4,3,NB))
    flux=np.zeros((len(rotations),3,3))
    for v,rotation in enumerate(rotations):
        view=relative@rotation.T
        pix=np.floor((view[:,:2]+EXTENT)/(2*EXTENT)*NPIX).astype(np.int64)
        pix=np.clip(pix,0,NPIX-1)
        key=((band*4+category)*NPIX+pix[:,0])*NPIX+pix[:,1]
        image=np.stack([np.bincount(key,weights=w,minlength=3*4*NPIX*NPIX).reshape(3,4,NPIX,NPIX) for w in projection_weights],axis=2)
        image=np.cumsum(image,axis=0)/PIXEL_AREA
        view_profiles[v]=annular_means(image)
        flux[v]=image.sum(axis=(1,3,4))*PIXEL_AREA
        if v==0:maps[0]=image
        else:maps[1]+=image/(len(rotations)-1)
    assert np.allclose(flux,flux[0:1],rtol=5e-13,atol=1e-16)
    projected=np.stack([view_profiles[0],view_profiles[1:].mean(axis=0)])
    return {'sums':sums,'hs_hist':hist,'maps':maps,'projected':projected,
            'view_profiles':view_profiles,'ids':data['ParticleIDs'],'flux':flux[0]}


def smoothing_upper_median(hist):
    cumulative=np.cumsum(hist,axis=-1)
    count=cumulative[:,:,-1]
    # Upper middle order statistic, followed by the histogram bin's upper edge.
    target=count//2+1
    index=np.argmax(cumulative>=target[:,:,None],axis=-1)
    result=HS_EDGES[index+1]
    result[count==0]=np.nan
    return result
