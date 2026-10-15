"""Conservative finite-sphere painting in Cartesian snapshot coordinates.

Tabulated radial kernels are interpolated in physical halo mass. Resolved
halos use pixel/cell quadrature, normalized to their spherical integral.
Unresolved halos deposit that integral in the containing pixel/cell. Slab
boundaries clip the line-of-sight integral; they do not accept every halo's
full projected sphere. Whole-box maps use periodic boundaries.
"""
import numpy as np
from numba import njit
from numpy.polynomial.legendre import leggauss
import physics as P

MASS_GRID = np.linspace(13.,15.5,61)
RADIAL_GRID = np.r_[0.,np.geomspace(1e-6,4.,600)]
PROJECT_GRID = np.r_[0.,np.geomspace(1e-5,4.,600)]
LOS_NODES,LOS_WEIGHTS = leggauss(48)


def parameters(fits,field,cut,component,mass):
    rows = [r for r in fits['results'] if r['field']==field and r['cut']==cut
            and r['component']==component and r['valid']]
    rows.sort(key=lambda r:r['mass_pivot_msun'])
    pivot = np.log10([r['mass_pivot_msun'] for r in rows])
    pars = np.asarray([r['parameters'] for r in rows])
    return np.stack([np.exp(np.interp(np.log10(mass),pivot,np.log(pars[:,j]))) for j in range(3)],axis=-1)


def kernels(fits,field,fit_radius,cut,component='all',baseline=False,scatter=None):
    """Dimensionless F(x), projected F, and spherical volume integral."""
    mass = 10**MASS_GRID
    pars = P.battaglia(field,mass) if baseline else parameters(fits,field,fit_radius,component,mass)
    if scatter is not None:
        # Template slots preserve joint Pe/ne and within-profile parameter correlations.
        base=pars.copy();all_pars=[]
        for slot in range(8):
            ratios=scatter[field][slot]
            bins=np.clip(np.floor((MASS_GRID-12.5)/.2).astype(int),0,14)
            all_pars.append(base*ratios[bins])
        pars=np.concatenate(all_pars,axis=0)
    radial = np.stack([P.shape(field,RADIAL_GRID,p) for p in pars])
    nodes,w = leggauss(128)
    r = cut*(nodes+1)/2
    integrals = np.array([4*np.pi*np.sum(P.shape(field,r,p)*r*r*w*cut/2) for p in pars])
    chord = np.sqrt(np.maximum(cut**2-PROJECT_GRID**2,0))
    u=(nodes+1)/2
    # z=chord*u^4 resolves the integrable central cusp. A plain z quadrature
    # can incorrectly suppress the centre when increasing the outer cut.
    los = chord[:,None]*u[None,:]**4
    radius = np.sqrt(PROJECT_GRID[:,None]**2+los**2)
    projected = np.stack([np.sum(P.shape(field,radius,p)*w[None,:]*4*u[None,:]**3,axis=1)*chord for p in pars])
    projected[:,PROJECT_GRID>=cut]=0
    return radial,projected,integrals


@njit(cache=True)
def mass_value(table,mi,mf,grid,x):
    return (1-mf)*np.interp(x,grid,table[mi]) + mf*np.interp(x,grid,table[mi+1])


@njit(cache=True)
def partial_column(radial,mi,mf,b,zlo,zhi,cut):
    chord = np.sqrt(max(cut*cut-b*b,0.))
    lo=max(-chord,zlo);hi=min(chord,zhi)
    if hi<=lo:
        return 0.
    result=0.
    for endpoint,sign in [(hi,1.),(lo,-1.)]:
        value=0.
        end=abs(endpoint)
        for k in range(len(LOS_NODES)):
            u=(LOS_NODES[k]+1)/2
            z=end*u**4
            value+=LOS_WEIGHTS[k]*mass_value(radial,mi,mf,RADIAL_GRID,np.sqrt(b*b+z*z))*4*end*u**3/2
        result+=sign*(1. if endpoint>=0 else -1.)*value
    return max(0.,result)


@njit(cache=True)
def paint_plane(position,velocity,radius,amplitude,mi,mf,radial,projected,integral,
                cut,origin,side,n,periodic):
    out=np.zeros((2,n,n))  # scalar deposit, same deposit times v_z
    dx=side/n
    for h in range(len(radius)):
        r=radius[h]; support=r*cut
        x=position[h,0]-origin[0];y=position[h,1]-origin[1];z=position[h,2]-origin[2]
        if not periodic and (x+support<0 or x-support>=side or y+support<0 or y-support>=side or z+support<0 or z-support>=side):
            continue
        total=amplitude[h]*((1-mf[h])*integral[mi[h]]+mf[h]*integral[mi[h]+1])
        if support<dx/2:
            if periodic or (0<=x<side and 0<=y<side and 0<=z<side):
                i=int(np.floor(x/dx))%n;j=int(np.floor(y/dx))%n
                out[0,i,j]+=total;out[1,i,j]+=total*velocity[h]
            continue
        ilo=int(np.floor((x-support)/dx))-1;ihi=int(np.floor((x+support)/dx))+1
        jlo=int(np.floor((y-support)/dx))-1;jhi=int(np.floor((y+support)/dx))+1
        norm=0.
        for i in range(ilo,ihi+1):
            for j in range(jlo,jhi+1):
                for di in [-.25,.25]:
                    for dj in [-.25,.25]:
                        b=np.hypot((i+.5+di)*dx-x,(j+.5+dj)*dx-y)/r
                        if b<cut:
                            norm+=mass_value(projected,mi[h],mf[h],PROJECT_GRID,b)/4
        if norm==0:
            if periodic or (0<=x<side and 0<=y<side and 0<=z<side):
                i=int(np.floor(x/dx))%n;j=int(np.floor(y/dx))%n
                out[0,i,j]+=total;out[1,i,j]+=total*velocity[h]
            continue
        for i in range(ilo,ihi+1):
            for j in range(jlo,jhi+1):
                if not periodic and (i<0 or i>=n or j<0 or j>=n):
                    continue
                value=0.
                for di in [-.25,.25]:
                    for dj in [-.25,.25]:
                        b=np.hypot((i+.5+di)*dx-x,(j+.5+dj)*dx-y)/r
                        if b<cut:
                            if periodic or (z>=support and z+support<=side):
                                value+=mass_value(projected,mi[h],mf[h],PROJECT_GRID,b)/4
                            else:
                                value+=partial_column(radial,mi[h],mf[h],b,-z/r,(side-z)/r,cut)/4
                deposit=total*value/norm
                out[0,i%n,j%n]+=deposit
                out[1,i%n,j%n]+=deposit*velocity[h]
    return out


@njit(cache=True)
def paint_cube(position,velocity,radius,amplitude,mi,mf,radial,integral,cut,origin,side,n):
    out=np.zeros((2,n,n,n),np.float32)
    dx=side/n
    for h in range(len(radius)):
        r=radius[h];support=r*cut
        xyz=position[h]-origin
        if np.any(xyz+support<0) or np.any(xyz-support>=side):
            continue
        total=amplitude[h]*((1-mf[h])*integral[mi[h]]+mf[h]*integral[mi[h]+1])
        ilo=np.floor((xyz-support)/dx).astype(np.int64)-1
        ihi=np.floor((xyz+support)/dx).astype(np.int64)+1
        norm=0.
        if support>=dx/2:
            for i in range(ilo[0],ihi[0]+1):
                for j in range(ilo[1],ihi[1]+1):
                    for k in range(ilo[2],ihi[2]+1):
                        distance=np.sqrt(((i+.5)*dx-xyz[0])**2+((j+.5)*dx-xyz[1])**2+((k+.5)*dx-xyz[2])**2)/r
                        if distance<cut:
                            norm+=mass_value(radial,mi[h],mf[h],RADIAL_GRID,distance)
        if norm==0:
            if np.all(xyz>=0) and np.all(xyz<side):
                index=np.floor(xyz/dx).astype(np.int64)
                out[0,index[0],index[1],index[2]]+=total
                out[1,index[0],index[1],index[2]]+=total*velocity[h]
            continue
        for i in range(max(0,ilo[0]),min(n-1,ihi[0])+1):
            for j in range(max(0,ilo[1]),min(n-1,ihi[1])+1):
                for k in range(max(0,ilo[2]),min(n-1,ihi[2])+1):
                    distance=np.sqrt(((i+.5)*dx-xyz[0])**2+((j+.5)*dx-xyz[1])**2+((k+.5)*dx-xyz[2])**2)/r
                    if distance<cut:
                        value=total*mass_value(radial,mi[h],mf[h],RADIAL_GRID,distance)/norm
                        out[0,i,j,k]+=value
                        out[1,i,j,k]+=value*velocity[h]
    return out


def region_fields(catalogue,fits,case,origin,side=250.,map_n=512,grid_n=256,whole=False):
    mass=catalogue['mass']
    cosmology=P.SOURCE if case.get('source_cosmology') else P.TARGET
    radius,p200,ne0=P.scales(mass,cosmology)
    position=catalogue['position'];velocity=catalogue['velocity'][:,2]
    support=radius*max(case['p_cut'],case['e_cut'])
    take=np.ones(len(mass),bool) if whole else np.all((position+support[:,None]>=origin)&(position-support[:,None]<origin+side),axis=1)
    mass=mass[take];position=position[take];velocity=velocity[take]
    radius=radius[take];p200=p200[take];ne0=ne0[take]
    coordinate=(np.log10(mass)-MASS_GRID[0])/(MASS_GRID[1]-MASS_GRID[0])
    mi=np.clip(np.floor(coordinate).astype(np.int64),0,len(MASS_GRID)-2)
    mf=np.clip(coordinate-mi,0,1)
    scatter=None
    if case.get('scatter_seed') is not None:
        scatter=joint_scatter(fits,int(case['scatter_seed']))
        rows=catalogue['row'][take].astype(np.int64)
        slot=((rows*1103515245+int(case['scatter_seed']))%8).astype(np.int64)
        mi=mi+slot*len(MASS_GRID)
    plane=[];cubes=[];integrated=[]
    for field,fit,cut,amplitude in [
        ('pressure',case['p_fit'],case['p_cut'],p200*P.SIGMA_T/(P.electron_mass*P.c**2)*P.MPC_M*radius**3),
        ('electrons',case['e_fit'],case['e_cut'],ne0*radius**3)]:
        radial,projected,integral=kernels(fits,field,fit,cut,case.get('component','all'),case.get('baseline',False),scatter)
        image=paint_plane(position,velocity,radius,amplitude,mi,mf,radial,projected,integral,
            float(cut),np.asarray(origin,float),float(side),map_n,whole)
        plane.append(image)
        integrated.append(float(np.sum(amplitude*((1-mf)*integral[mi]+mf*integral[mi+1]))))
        if not whole:
            cubes.append(paint_cube(position,velocity,radius,amplitude,mi,mf,radial,integral,
                float(cut),np.asarray(origin,float),float(side),grid_n))
    sums=np.array([plane[0][0],plane[1][0],plane[1][1]])
    grid=None if whole else np.array([cubes[0][0],cubes[1][0],cubes[1][1]])
    return sums,grid,dict(halos=int(take.sum()),integrated_full_spheres=integrated)


def joint_scatter(fits,seed):
    """Eight paired empirical parameter templates per bin; a sensitivity control.

    These include individual-fit noise and degeneracy. They are not certified
    draws from a noise-deconvolved physical gas-parameter distribution.
    """
    rng=np.random.default_rng(seed)
    result={field:np.ones((8,15,3)) for field in ['pressure','electrons']}
    for b in range(15):
        selected={field:next((r for r in fits['results'] if r['field']==field and r['component']=='all'
            and r['bin']==b and r['cut']==cut and r['valid']),None)
            for field,cut in [('pressure',2),('electrons',3)]}
        if any(r is None for r in selected.values()):
            continue
        lookup={field:dict(zip(r['individual_rows'],r['individual_parameters'])) for field,r in selected.items()}
        common=sorted(set(lookup['pressure'])&set(lookup['electrons']))
        if len(common)<2:
            continue
        for slot,row in enumerate(rng.choice(common,8,replace=True)):
            for field,r in selected.items():
                result[field][slot,b]=np.asarray(lookup[field][row])/r['parameters']
    return result
