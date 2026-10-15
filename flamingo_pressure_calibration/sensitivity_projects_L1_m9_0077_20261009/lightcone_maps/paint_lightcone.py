"""Stream the full HalfDome catalogue into matched finite-sphere sky tests.

The map painter assigns disjoint HEALPix rings to workers. Halo columns are
renormalized to their tabulated integrated flux, including unresolved objects;
point-ray columns are evaluated independently without pixel renormalization.
Profiles inherit the explicitly assumed published redshift response. This is
halo-only painting with bulk velocities, not a uniquely partitioned gas field.
"""
from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "8"

import argparse
import hashlib
import json
import resource
import sys
import time
from pathlib import Path

import h5py
import healpy as hp
import numba
import numpy as np
import psutil
from astropy.cosmology import FlatLambdaCDM
from numba import njit, prange
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPO = ROOT.parents[1]
sys.path.insert(0, str(ROOT))
import analysis_common as A

CATALOGUE = REPO/"lightcone_100.hdf5"
TABLES = ROOT/"profile_column_tables.npz"
COUNTS = REPO/"frb_map_generation/outputs/lee22_tsz_verification_20260929/B_class_sz/catalogue/catalogue_hist.npz"
COSMOLOGY = FlatLambdaCDM(H0=68., Om0=.31, Ob0=.049, Tcmb0=0)
SIGMA_T = A.P.SIGMA_T
MPC_M = A.P.MPC_M
DM_TO_M2 = 3.08567758e22
WORKERS = 8
MEMORY_LIMIT = 16*2**30
WALL_LIMIT = 4*3600


@njit(cache=True)
def bracket(grid, value):
    index = min(max(np.searchsorted(grid, value)-1, 0), len(grid)-2)
    weight = min(max((value-grid[index])/(grid[index+1]-grid[index]), 0.), 1.)
    return index, weight


@njit(cache=True)
def mix_integral(values, task, field, im, iz, wm, wz):
    return ((1-wm)*((1-wz)*values[task,field,im,iz]+wz*values[task,field,im,iz+1])
            +wm*((1-wz)*values[task,field,im+1,iz]+wz*values[task,field,im+1,iz+1]))


@njit(cache=True,inline="always")
def column_coordinates(logu,u,cut):
    if u >= 1.:
        return -1,0.,0.
    if u<=np.exp(logu[1]):
        iu,wu=0,u/np.exp(logu[1])
    else:
        iu, wu = bracket(logu, np.log(u))
    chord = 2*cut*np.sqrt(max(1-u*u, 0.))
    return iu,wu,chord


@njit(cache=True,inline="always")
def column_preindexed(values,task,im,iz,wm,wz,coordinates):
    iu,wu,chord=coordinates
    if iu<0:return 0.,0.
    pe=0.;ne=0.
    for dm in range(2):
        for dz in range(2):
            weight=(wm if dm else 1-wm)*(wz if dz else 1-wz)
            pe+=weight*((1-wu)*values[task,0,im+dm,iz+dz,iu]+wu*values[task,0,im+dm,iz+dz,iu+1])
            ne+=weight*((1-wu)*values[task,1,im+dm,iz+dz,iu]+wu*values[task,1,im+dm,iz+dz,iu+1])
    return pe*chord,ne*chord


@njit(cache=True,inline="always")
def column_pair(values,logu,task,im,iz,wm,wz,u,cut):
    return column_preindexed(values,task,im,iz,wm,wz,column_coordinates(logu,u,cut))


def ring_geometry(nside):
    starts, count, cosine, sine, shifted = hp.ringinfo(nside, np.arange(1,4*nside,dtype=np.int64))
    return (np.asarray(starts,np.int64), np.asarray(count,np.int64), np.asarray(cosine),
            np.asarray(sine), np.asarray(shifted,float)*.5, np.arccos(cosine))


@njit(cache=True)
def ring_span(theta, phi, ang200, cut, ct, st, count, shift):
    alpha = np.arcsin(min(cut*ang200, .999999))
    denominator = st*np.sin(theta)
    threshold = (np.cos(alpha)-ct*np.cos(theta))/max(denominator,1.e-300)
    if threshold >= 1.:
        return 0, -1
    width = np.pi if threshold <= -1. else np.arccos(threshold)
    step = 2*np.pi/count
    centre = phi/step-shift
    lower, upper = int(np.ceil(centre-width/step)), int(np.floor(centre+width/step))
    if upper-lower+1 > count:
        upper = lower+count-1
    return lower, upper


@njit(cache=True, parallel=True)
def normalizations(meta, nearest, ring_theta, geometry, values, integrals,
                   logu, radii, cuts, baseline, pixel_area):
    """First pass measures each halo's discrete column sum for exact flux."""
    starts,count,ct,st,shift = geometry
    n,tasks = len(meta),len(radii)
    scale = np.zeros((n,tasks,2))
    expected = np.zeros((n,tasks,2))
    for h in prange(n):
        lm,z,theta,phi,ang,ay,at,v,imf,izf,wm,wz = meta[h]
        im,iz = int(imf),int(izf)
        for t in range(tasks):
            if lm >= cuts[t] or (lm < 13.3 and t not in baseline):
                continue
            for f in range(2):
                expected[h,t,f] = mix_integral(integrals,t,f,im,iz,wm,wz)*ang*ang/pixel_area
        alpha = np.arcsin(min(4*ang,.999999))
        lo = np.searchsorted(ring_theta,max(0.,theta-alpha),side="left")
        hi = np.searchsorted(ring_theta,min(np.pi,theta+alpha),side="right")
        for ring in range(lo,hi):
            p0,p1 = ring_span(theta,phi,ang,4.,ct[ring],st[ring],count[ring],shift[ring])
            for raw in range(p0,p1+1):
                p = raw%count[ring]
                angle_phi = 2*np.pi*(p+shift[ring])/count[ring]
                cosine = min(1.,max(-1.,ct[ring]*np.cos(theta)+st[ring]*np.sin(theta)*np.cos(angle_phi-phi)))
                x = np.sqrt(max(1-cosine*cosine,0.))/ang
                c2=column_coordinates(logu,x/2.,2.);c3=column_coordinates(logu,x/3.,3.);c4=column_coordinates(logu,x/4.,4.)
                for t in range(tasks):
                    if x>=radii[t] or lm>=cuts[t] or (lm<13.3 and t not in baseline):
                        continue
                    coordinates=c2 if radii[t]==2 else (c3 if radii[t]==3 else c4)
                    pe,ne = column_preindexed(values,t,im,iz,wm,wz,coordinates)
                    scale[h,t,0] += pe
                    scale[h,t,1] += ne
        for t in range(tasks):
            for f in range(2):
                if scale[h,t,f]>0:
                    scale[h,t,f] = expected[h,t,f]/scale[h,t,f]
    return scale, expected


@njit(cache=True, parallel=True)
def paint_rings(maps, meta, nearest, ring_theta, geometry, values, logu, radii, cuts,
                baseline, base4, scale, expected):
    """Every map pixel belongs to exactly one worker; no atomic additions."""
    starts,count,ct,st,shift = geometry
    for worker in prange(WORKERS):
        first = worker*len(count)//WORKERS
        last = (worker+1)*len(count)//WORKERS
        for h in range(len(meta)):
            lm,z,theta,phi,ang,ay,at,v,imf,izf,wm,wz = meta[h]
            im,iz = int(imf),int(izf)
            alpha = np.arcsin(min(4*ang,.999999))
            lo = max(first,np.searchsorted(ring_theta,max(0.,theta-alpha),side="left"))
            hi = min(last,np.searchsorted(ring_theta,min(np.pi,theta+alpha),side="right"))
            owner = first<=np.searchsorted(starts,nearest[h],side="right")-1<last
            # Fallback deposits integrated flux rather than losing tiny halos.
            if owner:
                for t in range(len(radii)):
                    if lm<13.3 and t not in baseline:
                        continue
                    fpe = expected[h,t,0] if scale[h,t,0]==0 else 0.
                    fne = expected[h,t,1] if scale[h,t,1]==0 else 0.
                    if t not in baseline:
                        fpe -= expected[h,base4,0] if scale[h,base4,0]==0 else 0.
                        fne -= expected[h,base4,1] if scale[h,base4,1]==0 else 0.
                    maps[t,0,nearest[h]] += ay*fpe
                    maps[t,1,nearest[h]] += at*v*fne
                    if z<=1.:
                        maps[t,2,nearest[h]] += at*fne/(SIGMA_T*DM_TO_M2*(1+z))
            if hi<=lo:
                continue
            for ring in range(lo,hi):
                p0,p1 = ring_span(theta,phi,ang,4.,ct[ring],st[ring],count[ring],shift[ring])
                for raw in range(p0,p1+1):
                    p = raw%count[ring];pixel = starts[ring]+p
                    angle_phi = 2*np.pi*(p+shift[ring])/count[ring]
                    cosine = min(1.,max(-1.,ct[ring]*np.cos(theta)+st[ring]*np.sin(theta)*np.cos(angle_phi-phi)))
                    x = np.sqrt(max(1-cosine*cosine,0.))/ang
                    c2=column_coordinates(logu,x/2.,2.);c3=column_coordinates(logu,x/3.,3.);c4=column_coordinates(logu,x/4.,4.)
                    bpe,bne = column_preindexed(values,base4,im,iz,wm,wz,c4)
                    bpe*=scale[h,base4,0];bne*=scale[h,base4,1]
                    for t in range(len(radii)):
                        if lm<13.3 and t not in baseline:
                            continue
                        pe=0.;ne=0.
                        if x<radii[t] and lm<cuts[t]:
                            coordinates=c2 if radii[t]==2 else (c3 if radii[t]==3 else c4)
                            pe,ne = column_preindexed(values,t,im,iz,wm,wz,coordinates)
                            pe*=scale[h,t,0];ne*=scale[h,t,1]
                        if t not in baseline:
                            pe-=bpe;ne-=bne
                        maps[t,0,pixel] += ay*pe
                        maps[t,1,pixel] += at*v*ne
                        if z<=1.:
                            maps[t,2,pixel] += at*ne/(SIGMA_T*DM_TO_M2*(1+z))


@njit(cache=True)
def paint_rays(output, meta, vectors, hit_halo, hit_ray, ray_vectors,
               values, logu, radii, cuts):
    """Evaluate matched infinitesimal rays; no pixel flux renormalization."""
    for pair in range(len(hit_ray)):
        h,ray = hit_halo[pair],hit_ray[pair]
        lm,z,theta,phi,ang,ay,at,v,imf,izf,wm,wz = meta[h]
        cosine = min(1.,max(-1.,np.dot(vectors[h],ray_vectors[ray])))
        x = np.sqrt(max(1-cosine*cosine,0.))/ang
        c2=column_coordinates(logu,x/2.,2.);c3=column_coordinates(logu,x/3.,3.);c4=column_coordinates(logu,x/4.,4.)
        for t in range(len(radii)):
            if lm>=cuts[t] or x>=radii[t]:
                continue
            coordinates=c2 if radii[t]==2 else (c3 if radii[t]==3 else c4)
            _,ne = column_preindexed(values,t,int(imf),int(izf),wm,wz,coordinates)
            output[t,ray] += at*ne/(SIGMA_T*DM_TO_M2*(1+z))


def load_tables():
    saved = dict(np.load(TABLES))
    saved["chord_mean"] = np.asarray(saved["chord_mean"],np.float32)
    saved["integrals"] = np.asarray(saved["integrals"],np.float64)
    # Tables retain an exact central ray at u=0, then a logarithmic grid.
    saved["logu"] = np.log(np.maximum(saved["u_grid"],saved["u_grid"][1]*.1))
    saved["log1pz"] = np.log1p(saved["z_grid"])
    saved["baseline"] = np.flatnonzero(saved["case_indices"]==0).astype(np.int64)
    saved["base4"] = int(np.flatnonzero((saved["case_indices"]==0)&(saved["radii"]==4))[0])
    assert len(saved["baseline"])==3
    return saved


def prepare(mass, z, position, velocity, table):
    mass=np.asarray(mass,np.float64);z=np.asarray(z,np.float64)
    position=np.asarray(position,np.float64);velocity=np.asarray(velocity,np.float64)
    norm = np.linalg.norm(position,axis=1)
    vectors = position/norm[:,None]
    theta = np.arccos(np.clip(vectors[:,2],-1,1));phi=np.mod(np.arctan2(vectors[:,1],vectors[:,0]),2*np.pi)
    logmass=np.log10(mass)
    if logmass.min()<table["logmass_grid"][0] or logmass.max()>table["logmass_grid"][-1]:
        raise ValueError("Actual catalogue mass lies outside table; extend published fallback")
    if z.min()<table["z_grid"][0] or z.max()>table["z_grid"][-1]:
        raise ValueError("Actual catalogue redshift lies outside table")
    rho=2.77536627e11*.68**2*(.31*(1+z)**3+.69)
    radius=(3*mass/(4*np.pi*200*rho))**(1/3)
    distance=np.interp(z,DISTANCE_Z,DISTANCE_DA)
    ang=radius/distance
    if np.any(4*ang>=.5):
        raise ValueError("Observer-near halo exceeds small-screen projection contract")
    p200=A.P.p200(mass,radius,.049/.31)
    ay=SIGMA_T/(A.P.electron_mass*A.P.c**2)*p200*radius*MPC_M
    ne_norm=(.049/.31)*rho/A.P.MU_XGP_MSUN/MPC_M**3
    at=SIGMA_T*ne_norm*radius*MPC_M
    v=-np.sum(vectors*velocity,axis=1)/299792.458
    im=np.clip(np.searchsorted(table["logmass_grid"],logmass)-1,0,len(table["logmass_grid"])-2)
    lz=np.log1p(z);iz=np.clip(np.searchsorted(table["log1pz"],lz)-1,0,len(table["log1pz"])-2)
    wm=(logmass-table["logmass_grid"][im])/np.diff(table["logmass_grid"])[im]
    wz=(lz-table["log1pz"][iz])/np.diff(table["log1pz"])[iz]
    meta=np.column_stack((logmass,z,theta,phi,ang,ay,at,v,im,iz,wm,wz))
    if not np.all(np.isfinite(meta)) or np.any(meta[:,4:7]<=0):
        raise ValueError("Nonfinite or nonpositive halo unit conversion")
    return np.ascontiguousarray(meta),np.ascontiguousarray(vectors)


DISTANCE_Z=np.r_[0.,np.geomspace(1.e-5,4.,30000)]
DISTANCE_DA=COSMOLOGY.angular_diameter_distance(DISTANCE_Z).value


def apply_paint(maps,meta,table,nside):
    starts,count,ct,st,shift,theta=ring_geometry(nside)
    geometry=(starts,count,ct,st,shift)
    nearest=hp.ang2pix(nside,meta[:,2],meta[:,3])
    scale,expected=normalizations(meta,nearest,theta,geometry,table["chord_mean"],table["integrals"],
                                  table["logu"],table["radii"],table["cut_logmass"],table["baseline"],hp.nside2pixarea(nside))
    paint_rings(maps,meta,nearest,theta,geometry,table["chord_mean"],table["logu"],table["radii"],
                table["cut_logmass"],table["baseline"],table["base4"],scale,expected)
    # Below13.3 all R4 alternatives are the identical published fallback.
    # Vectorizing this linear reduction avoids a Python loop over85M halos.
    low=meta[:,0]<13.3
    for t in range(len(table["radii"])):
        if t not in table["baseline"]:
            expected[low,t]=expected[low,table["base4"]]
    flux=np.empty((len(table["radii"]),3))
    area=hp.nside2pixarea(nside)
    flux[:,0]=np.einsum("ht,h->t",expected[:,:,0],meta[:,5])*area
    flux[:,1]=np.einsum("ht,h->t",expected[:,:,1],meta[:,6]*meta[:,7])*area
    flux[:,2]=np.einsum("ht,h->t",expected[:,:,1],
                       meta[:,6]*(meta[:,1]<=1)/(SIGMA_T*DM_TO_M2*(1+meta[:,1])))*area
    return flux


def independent_maps(stored,table):
    for t in range(len(table["radii"])):
        value=np.asarray(stored[t],np.float64)
        if t not in table["baseline"]:
            value+=stored[table["base4"]]
        yield t,value


def guard(started):
    process=psutil.Process()
    if process.memory_info().rss>MEMORY_LIMIT:
        raise MemoryError("16GiB process RSS limit exceeded")
    if time.perf_counter()-started>WALL_LIMIT:
        raise TimeoutError("Four-hour wall limit exceeded")


def gates(table):
    """Isolated off-centre halos must conserve flux even below pixel scale."""
    started=time.perf_counter();tasks=len(table["radii"])
    # At these redshifts, tests include sub-pixel and well-resolved objects.
    mass=10**np.array([12.9,13.8,15.35]);z=np.array([2.,.3,.02])
    direction=np.array([[1.,.02,.03],[-.2,.9,.1],[.2,.1,-1.]])
    velocity=np.array([[150.,-30.,20.],[-200.,50.,40.],[80.,20.,-100.]])
    meta,vectors=prepare(mass,z,direction,velocity,table)
    checks={"isolated_flux":[],"ring_geometry":[],"synthetic_halos":3}
    for nside in [128,256]:
        maps=np.zeros((tasks,3,hp.nside2npix(nside)),np.float32)
        expected=apply_paint(maps,meta,table,nside)
        measured=np.array([value.sum(axis=1)*hp.nside2pixarea(nside) for _,value in independent_maps(maps,table)])
        error=np.max(np.abs(measured-expected)/np.maximum(np.abs(expected),1.e-30))
        checks["isolated_flux"].append({"nside":nside,"max_relative_error":float(error)})
        starts,count,ct,st,shift,theta=ring_geometry(nside)
        ring=np.array([0,nside-1,2*nside-1,len(count)-1]);pixel=starts[ring]+count[ring]//3
        vx,vy,vz=hp.pix2vec(nside,pixel);phi=2*np.pi*(count[ring]//3+shift[ring])/count[ring]
        geometry_error=np.max(np.abs(np.column_stack((vx,vy,vz))-np.column_stack((st[ring]*np.cos(phi),st[ring]*np.sin(phi),ct[ring]))))
        checks["ring_geometry"].append(float(geometry_error))
    # A uniformly spread catalogue sample times actual NSIDE2048 geometry.
    with h5py.File(CATALOGUE) as f:
        n=len(f["redshift"]);selected=np.sort(np.random.default_rng(43).choice(n,4000,replace=False))
        meta,vectors=prepare(f["halo_mass_m200c"][selected]/.68,f["redshift"][selected],f["Position"][selected],f["Velocity"][selected],table)
    maps=np.zeros((tasks,3,hp.nside2npix(2048)),np.float32)
    tick=time.perf_counter();apply_paint(maps,meta,table,2048);elapsed=time.perf_counter()-tick
    tick=time.perf_counter();apply_paint(maps,meta,table,2048);warm=time.perf_counter()-tick
    del maps
    checks.update(timed_halos=len(meta),timed_seconds=elapsed,
                  linear_paint_extrapolation_hours=elapsed/len(meta)*n/3600,
                  warm_timed_seconds=warm,warm_linear_paint_extrapolation_hours=warm/len(meta)*n/3600,
                  peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                  gate_wall_seconds=time.perf_counter()-started,
                  passed=max(c["max_relative_error"] for c in checks["isolated_flux"])<3.e-6 and max(checks["ring_geometry"])<1.e-12)
    A.write_json(HERE/"synthetic_and_timing_gates.json",checks)
    print(json.dumps(checks),flush=True)


def source_rays():
    """Reuse all120k historical source directions, not a different seed42 RNG."""
    path=REPO/"frb_map_generation/outputs/random_sky_frb_los_z1_n120k/halfdome_zfrb1p0_zhalomax1p0_nside4096_nfrb120000_seed42_frb_sources.csv"
    saved=np.genfromtxt(path,delimiter=",",names=True)
    phi=saved["ra_rad"];cosine=np.sin(saved["dec_rad"]);sine=np.sqrt(1-cosine*cosine)
    vectors=np.column_stack((sine*np.cos(phi),sine*np.sin(phi),cosine))
    old_pixels=np.asarray(saved["pixel_ring"],np.int64)-1
    expected=np.asarray(hp.pix2vec(4096,old_pixels)).T
    error=float(np.max(np.abs(vectors-expected)))
    assert len(vectors)==120000 and error<1.e-12 and np.all(saved["z_source"]==1)
    np.savez_compressed(HERE/"frb_directions_seed42.npz",unit_vectors=vectors,pixels_nside4096_ring_zero_based=old_pixels,
                        old_dm_pc_cm3=saved["dm_pc_cm3"],source_redshift=1.,seed=42,
                        sampling="exact historical uniform source directions; old4096pixel centres, continuous columns",
                        source_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    A.write_json(HERE/"frb_source_receipt.json",{"source":str(path),"sources":120000,
                 "source_SHA256":hashlib.sha256(path.read_bytes()).hexdigest(),
                 "max_unitvector_difference_from_declared_HEALPix4096_pixel":error,
                 "identical_historical_directions":True,"source_redshift":1.,"seed":42,
                 "new_columns_evaluated_without_NSIDE2048_pixel_sampling":True})
    return vectors


def main_run(table,chunk_size=100000):
    """Full execution; called only after gates and resource preflight."""
    gate=json.loads((HERE/"synthetic_and_timing_gates.json").read_text())
    if not gate["passed"]:raise RuntimeError("Synthetic map gates did not pass")
    if (HERE/"complete.json").exists():raise RuntimeError("Preserving completed products")
    started=time.perf_counter();cpu=time.process_time();nside=2048;tasks=len(table["radii"])
    maps=np.zeros((tasks,3,hp.nside2npix(nside)),np.float32)
    baseline=table["baseline"];coarse=dict(table)
    for key in ("task_names","case_indices","radii","cut_logmass","chord_mean","integrals"):
        coarse[key]=np.ascontiguousarray(table[key][baseline])
    coarse["baseline"]=np.arange(3,dtype=np.int64);coarse["base4"]=2
    coarse_maps=np.zeros((3,3,hp.nside2npix(1024)),np.float32)
    rays=source_rays();tree=cKDTree(rays);ray_dm=np.zeros((tasks,len(rays)))
    hist=dict(np.load(COUNTS));fine_counts=np.zeros_like(hist["N"],dtype=np.int64)
    fine_v2=np.zeros_like(hist["N"],dtype=np.float64)
    expected_flux=np.zeros((tasks,3));coarse_flux=np.zeros((3,3));pairs=0;scanned=0
    source_stat=CATALOGUE.stat()
    with h5py.File(CATALOGUE) as f:
        total=len(f["redshift"])
        for start in range(0,total,chunk_size):
            stop=min(start+chunk_size,total)
            mass=np.asarray(f["halo_mass_m200c"][start:stop],float)/.68
            z=np.asarray(f["redshift"][start:stop],float)
            meta,vectors=prepare(mass,z,f["Position"][start:stop],f["Velocity"][start:stop],table)
            expected_flux+=apply_paint(maps,meta,table,nside)
            coarse_flux+=apply_paint(coarse_maps,meta,coarse,1024)
            fine_counts+=np.histogram2d(z,np.log(mass*.68),bins=(hist["z_edges"],hist["lnm_edges"]))[0].astype(np.int64)
            fine_v2+=np.histogram2d(z,np.log(mass*.68),bins=(hist["z_edges"],hist["lnm_edges"]),
                                    weights=(meta[:,7]*299792.458)**2)[0]
            foreground=np.flatnonzero(z<=1.)
            if len(foreground):
                chord=2*np.sin(np.arcsin(4*meta[foreground,4])/2)
                lengths=tree.query_ball_point(vectors[foreground],chord,workers=WORKERS,return_length=True)
                active=foreground[lengths>0]
                if len(active):
                    hits=tree.query_ball_point(vectors[active],2*np.sin(np.arcsin(4*meta[active,4])/2),
                                               workers=WORKERS,return_sorted=False)
                    hit_ray=np.concatenate(hits).astype(np.int64)
                    hit_halo=np.repeat(active,lengths[lengths>0]).astype(np.int64)
                    paint_rays(ray_dm,meta,vectors,hit_halo,hit_ray,rays,table["chord_mean"],table["logu"],table["radii"],table["cut_logmass"])
                    pairs+=len(hit_ray)
            scanned=stop
            guard(started)
            status={"stage":"catalogue_painting","rows_scanned":scanned,"rows_total":total,
                    "fraction":scanned/total,"matched_ray_halo_pairs":pairs,
                    "wall_seconds":time.perf_counter()-started,"RSS_GiB":psutil.Process().memory_info().rss/2**30,
                    "CPU_seconds":time.process_time()-cpu,"threads":WORKERS}
            A.write_json(HERE/"progress.json",status)
            if start==0 or stop%1000000==0 or stop==total:print(json.dumps(status),flush=True)
    if CATALOGUE.stat().st_size!=source_stat.st_size or CATALOGUE.stat().st_mtime_ns!=source_stat.st_mtime_ns:
        raise RuntimeError("Catalogue changed during pass")
    np.savez_compressed(HERE/"fine_velocity_histogram.npz",sum_vlos2_km2s2=fine_v2,N=fine_counts,
                        z_edges=hist["z_edges"],lnm_edges=hist["lnm_edges"],rows=scanned,
                        velocity_definition="minus Position_unit dot Velocity; square in km^2/s^2")
    np.savez_compressed(HERE/"frb_matched_ray_dm.npz",task_names=table["task_names"],dm_pc_cm3=ray_dm,
                        source_redshift=1.,finite_sphere_radii=table["radii"],exact_historical_directions=True)
    closure=np.empty((tasks,3));ell=np.arange(3001);cl=np.zeros((tasks,2,len(ell)))
    cross_cut=np.full((2,2,len(ell)),np.nan);auto_removed=np.full_like(cross_cut,np.nan)
    # Common bins include overflow counters; tails are never discarded silently.
    pdf_edges=np.r_[0.,np.geomspace(.01,1.e5,300)]
    pixel_counts=np.zeros((tasks,len(pdf_edges)-1),np.int64);ray_counts=np.zeros_like(pixel_counts)
    pixel_summaries=[];base_alm={};overviews=[]
    with h5py.File(HERE/"baseline_maps_nside2048.h5","w") as archive:
        archive.attrs["nside"]=2048;archive.attrs["ordering"]="RING"
        archive.attrs["fields"]="y, bulk-kSZ DeltaT/T, halo-only observed DM pc cm^-3 for z<=1"
        archive.attrs["gas_partition"]="all-gas fitted profiles; overlaps not uniquely owned"
        archive.attrs["pixel_policy"]="centre-sampled columns renormalized per-halo to integrated flux; nearest-pixel finite-flux fallback"
        for t,value in independent_maps(maps,table):
            guard(started)
            closure[t]=value.sum(axis=1)*hp.nside2pixarea(2048)
            minimum=float(value[2].min())
            # Difference-map floating subtraction can leave tiny negative DM.
            if minimum < -1.e-3:
                raise ValueError(f"Substantial negative DM in {table['task_names'][t]}: {minimum}")
            value[2]=np.maximum(value[2],0.)
            pixel_counts[t]=np.histogram(value[2],bins=pdf_edges)[0]
            ray_counts[t]=np.histogram(ray_dm[t],bins=pdf_edges)[0]
            pixel_summaries.append({"task":str(table["task_names"][t]),"pixel_mean":float(value[2].mean()),
                "pixel_percentiles_16_50_84_95_99":np.percentile(value[2],[16,50,84,95,99]),
                "ray_mean":float(ray_dm[t].mean()),"ray_percentiles_16_50_84_95_99":np.percentile(ray_dm[t],[16,50,84,95,99]),
                "pixel_max":float(value[2].max()),"ray_max":float(ray_dm[t].max()),
                "pixel_overflow":int(np.sum(value[2]>=pdf_edges[-1])),"ray_overflow":int(np.sum(ray_dm[t]>=pdf_edges[-1])),
                "negative_roundoff_min_before_clipping":minimum})
            overviews.append(np.array([hp.ud_grade(v,64) for v in value],np.float32))
            if t in baseline:
                archive.create_dataset(str(table["task_names"][t]),data=np.asarray(value,np.float32),
                                       chunks=(1,262144),compression="gzip",compression_opts=1,shuffle=True)
            for field in range(2):
                sky=value[field]-value[field].mean()
                alm=hp.map2alm(sky,lmax=3000,iter=0,pol=False)
                cl[t,field]=hp.alm2cl(alm)
                if t==table["base4"]:base_alm[field]=alm.copy()
                for cut_i,name in enumerate(("cut_ge15p3_r4","cut_ge15p1_r4")):
                    if table["task_names"][t]==name:
                        cross_cut[cut_i,field]=hp.alm2cl(base_alm[field],alm)
                        auto_removed[cut_i,field]=hp.alm2cl(base_alm[field]-alm)
            A.write_json(HERE/"progress.json",{"stage":"spectra_PDF","task":t,"tasks":tasks,
                "task_name":str(table["task_names"][t]),"wall_seconds":time.perf_counter()-started,
                "RSS_GiB":psutil.Process().memory_info().rss/2**30})
            print(json.dumps({"spectra_task":str(table["task_names"][t]),"wall_seconds":time.perf_counter()-started}),flush=True)
        archive.attrs["complete"]=True
    np.savez_compressed(HERE/"sky_overviews_nside64.npz",task_names=table["task_names"],maps=np.array(overviews))
    np.savez_compressed(HERE/"fullsky_spectra.npz",task_names=table["task_names"],ell=ell,cl_y_ksz=cl,
                        pixel_window=hp.pixwin(2048,lmax=3000),cross_baseline_retained=cross_cut,
                        removed_high_mass_auto=auto_removed,
                        cut_names=np.array(["cut_ge15p3_r4","cut_ge15p1_r4"]),
                        fields=np.array(["dimensionless_y","bulk_ksz_DeltaT_over_T"]))
    np.savez_compressed(HERE/"frb_DM_PDF.npz",task_names=table["task_names"],edges_pc_cm3=pdf_edges,
                        counts_allpixels_nside2048=pixel_counts,counts_matched_120000_rays=ray_counts,
                        allpixels_count=hp.nside2npix(2048),ray_count=120000)
    A.write_json(HERE/"frb_DM_statistics.json",{"tasks":pixel_summaries,"pixel_PDF_continuum_converged":False,
                 "all_possible_meaning":"all50331648HEALPix2048pixel-averaged sky values; not all continuum sightlines"})
    coarse_cl=np.zeros((3,2,1501));coarse_pdf=np.zeros((3,len(pdf_edges)-1),np.int64)
    coarse_stats=[]
    for t,value in independent_maps(coarse_maps,coarse):
        for field in range(2):coarse_cl[t,field]=hp.anafast(value[field]-value[field].mean(),lmax=1500,iter=0,pol=False)
        coarse_pdf[t]=np.histogram(np.maximum(value[2],0),bins=pdf_edges)[0]
        coarse_stats.append({"task":str(coarse["task_names"][t]),"DM_mean":float(value[2].mean()),
                             "DM_percentiles_16_50_84_95_99":np.percentile(value[2],[16,50,84,95,99])})
    np.savez_compressed(HERE/"nside_convergence.npz",task_names=coarse["task_names"],ell=np.arange(1501),
                        cl_y_ksz_nside1024=coarse_cl,counts_allpixels_nside1024=coarse_pdf,
                        counts_allpixels_nside2048=pixel_counts[baseline],pdf_edges=pdf_edges,
                        pixel_window1024=hp.pixwin(1024,lmax=1500),pixel_window2048=hp.pixwin(2048,lmax=1500),
                        independent_paint=True)
    A.write_json(HERE/"nside_convergence_statistics.json",{"independent_NSIDE1024_baselines":coarse_stats,
                 "continuum_PDF_convergence_not_assumed":True})
    relative_flux=np.abs(closure-expected_flux)/np.maximum(abs(expected_flux),1.e-30)
    checks={"passed":bool(np.max(relative_flux[:,[0,2]])<5.e-4 and np.all(np.isfinite(cl)) and np.all(np.isfinite(ray_dm))),
            "map_flux_relative_error_by_task_and_field":relative_flux,
            "count_histogram_equals_existing_fine_catalogue":bool(np.array_equal(fine_counts,hist["N"])),
            "fine_count_total":int(fine_counts.sum()),"catalogue_rows_scanned":scanned,
            "all_catalogue_halos_used":True,"matched_ray_pairs":pairs,
            "exact_prior_120000_source_directions":True,"independent_NSIDE1024_render":True,
            "internal_gas_velocities_used":False,"no_gas_unique_ownership":True,
            "total_spectrum_minus_analytic_self_is_not_certified_twohalo":True,
            "wall_seconds":time.perf_counter()-started,"CPU_seconds":time.process_time()-cpu,
            "peak_RSS_GiB":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
            "threads":WORKERS,"PBS_jobs":0,"raw_snapshot_particles_read":0,
            "catalogue_logical_bytes_requested":scanned*(8+4+3*4+3*4),
            "catalogue_file_size":source_stat.st_size,"catalogue_mtime_ns":source_stat.st_mtime_ns,
            "profile_table_SHA256":hashlib.sha256(TABLES.read_bytes()).hexdigest()}
    A.write_json(HERE/"verification.json",checks)
    A.write_json(HERE/"complete.json",{"complete":True,"numerical_checks_passed":checks["passed"],
                 "status":"all catalogue maps and matched rays completed; scientific limitations in verification"})
    print(json.dumps(A.plain(checks)),flush=True)
    if not checks["passed"]:raise RuntimeError("Map numerical verification failed")


if __name__=="__main__":
    HERE.mkdir(parents=True,exist_ok=True)
    parser=argparse.ArgumentParser();parser.add_argument("--gates",action="store_true");parser.add_argument("--run",action="store_true")
    options=parser.parse_args();table=load_tables()
    gates(table) if options.gates else main_run(table)
