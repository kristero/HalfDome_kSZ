"""Conditional lightcone halo self-terms and linear tSZ two-halo response.

This module uses saved catalogue histograms and saved CLASS ingredients.
It never calls CLASS, streams the catalogue, submits work, or reads particles.
The new z=0 radial profiles inherit an explicitly assumed published redshift
response. kSZ here is the bulk-velocity one-halo/self term only. Its distinct-
halo term must come from the matched painted maps, not a density-bias shortcut.
"""
from __future__ import annotations

import os
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import csv
import hashlib
import json
import math
from pathlib import Path
import resource
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate, interpolate

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
REPO = PROJECT.parents[1]
sys.path.insert(0, str(PROJECT))
import analysis_common as C

INGREDIENTS = REPO / "frb_map_generation/outputs/lee22_tsz_verification_20260929/B_class_sz/class_sz/b12_paint_all.npz"
COUNTS = REPO / "frb_map_generation/outputs/lee22_tsz_verification_20260929/B_class_sz/catalogue/catalogue_hist.npz"
VELOCITIES = REPO / "frb_map_generation/outputs/lee22_ksz_debug_20261002/halo_catalogue/halfdome_catalogue_pass.npz"
ELL = np.r_[0., np.geomspace(20., 6000., 40)]
H = .68
FB = .049/.31
C_KMS = 299792.458
T_CMB_UK = 2.7255e6
# One pc cm^-3 is (one pc in metres)*(10^6 electrons per m^3).
M2_PER_PC_CM3 = 3.08567758e16*1e6


def published_profile(field, mass, z, x):
    """The exact published B12/B16 dimensionless conventions of this study."""
    coefficients = C.P.B12 if field == 0 else C.P.B16
    factors = coefficients[:3, None] * (np.asarray(mass)[None, :]/1e14)**coefficients[3:6, None]
    factors *= (1+z)**coefficients[6:9, None]
    amplitude, second, beta = factors
    if field == 0:
        q = x[None, :]/second[:, None]
        return .5176*amplitude[:, None]*q**(-.3)*(1+q)**(-beta[:, None])
    q = x[None, :]/.5
    return amplitude[:, None]*q**(-.2)*(1+q**second[:, None])**(-(beta[:, None]-.2)/second[:, None])


class Ingredients:
    """Saved cosmology, Tinker bias and linear power, with explicit h units."""
    def __init__(self, path=INGREDIENTS):
        arrays = np.load(path)
        self.arrays = arrays
        self.z = arrays["tab_z"]
        self.ell = arrays["ell"]
        self.mass = arrays["tab_lnm_msunh"]
        self.da_spline = interpolate.CubicSpline(self.z, arrays["tab_da_mpc"])
        self.rho_spline = interpolate.CubicSpline(self.z, np.log(arrays["tab_rho_crit_h2msun_mpc3"]))
        self.bias_spline = interpolate.RectBivariateSpline(self.z, self.mass, arrays["tab_bias"])
        self.number_spline = interpolate.RectBivariateSpline(self.z, self.mass, np.log(arrays["tab_dndlnm_h3mpc3"]))
        self.pk_spline = interpolate.RectBivariateSpline(self.z, np.log(self.ell+.5),
                                                        np.log(np.maximum(arrays["tab_pklin_mpch3"],1e-30)), kx=1, ky=1)
        self.pk_floored_cells = int(np.sum(arrays["tab_pklin_mpch3"]<=0))

    def geometry(self, z, mass):
        da = self.da_spline(z)
        rho = np.exp(self.rho_spline(z))*H**2  # physical Msun/Mpc^3
        radius = (3*np.asarray(mass)/(4*np.pi*200*rho))**(1/3)
        rho_si = rho*C.P.MSUN_KG/C.P.MPC_M**3
        p200 = C.P.G*np.asarray(mass)*C.P.MSUN_KG*200*rho_si*FB/(2*radius*C.P.MPC_M)
        y_amp = C.P.SIGMA_T/(C.P.electron_mass*C.P.c**2)*p200*radius**3*C.P.MPC_M/da**2
        tau_amp = C.P.SIGMA_T*FB*rho/C.P.MU_XGP_MSUN*radius**3/(da**2*C.P.MPC_M**2)
        return da, radius, y_amp, tau_amp

    def bias(self, z, mass):
        return self.bias_spline.ev(np.full(len(mass),z),np.log(np.asarray(mass)*H))

    def pk(self, z, ell):
        # Native CLASS P is (Mpc/h)^3. The catalogue shell volumes below are Mpc^3.
        return np.exp(self.pk_spline.ev(np.full(len(ell),z),np.log(np.maximum(ell,self.ell[0])+.5)))/H**3


def linear_radial_kernel(x, q):
    """Integration weights for 4pi int x^2 F(x) sinc(qx) dx.

    F is piecewise linear in physical scaled radius x. The antiderivatives
    of x sin(qx) and x^2 sin(qx) integrate high-frequency oscillations exactly.
    A moment expansion handles q<0.2 without cancellation. Returned weights
    multiply endpoint values F(x); no extra numerical radial quadrature is used.
    """
    x = np.asarray(x)
    q = np.asarray(q)
    lo, hi = x[:-1], x[1:]
    dx = hi-lo
    flat = q.ravel()
    weights = np.zeros((len(flat), len(x)))
    small = np.abs(flat)<.2
    if np.any(small):
        terms = np.zeros((small.sum(),len(x)))
        qs = flat[small]
        for order in range(8):
            power = 2*order+2
            a = (hi**(power+1)-lo**(power+1))/(power+1)
            b = (hi**(power+2)-lo**(power+2))/(power+2)
            left = (hi*a-b)/dx
            right = (b-lo*a)/dx
            coefficient = (-1.)**order*qs**(2*order)/float(math.factorial(2*order+1))
            terms[:,:-1] += coefficient[:,None]*left
            terms[:,1:] += coefficient[:,None]*right
        weights[small] = terms*4*np.pi
    if np.any(~small):
        ql = flat[~small,None]
        sl, sh = np.sin(ql*lo), np.sin(ql*hi)
        cl, ch = np.cos(ql*lo), np.cos(ql*hi)
        i1 = (sh-sl)/ql**2 - (hi*ch-lo*cl)/ql
        i2 = -(hi**2*ch-lo**2*cl)/ql + 2*(hi*sh-lo*sl)/ql**2 + 2*(ch-cl)/ql**3
        weights[~small,:-1] += 4*np.pi/ql*(hi*i1-i2)/dx
        weights[~small,1:] += 4*np.pi/ql*(i2-lo*i1)/dx
    return weights.reshape(q.shape+(len(x),))


def radial_grid(source_x, radius):
    return np.r_[source_x[source_x<radius],radius]


def profile_at_x(source_x, profiles, new_x):
    # Positive radial tables use log-log interpolation; cuts are applied separately.
    interpolator = interpolate.interp1d(np.log(source_x), np.log(np.maximum(profiles,1e-300)),
                                        axis=-1, bounds_error=False, fill_value="extrapolate")
    return np.exp(interpolator(np.log(new_x)))


def profile_mass_grid(scenarios, radius_index, masses, x):
    profiles = scenarios["profiles"][:,:,radius_index]
    interpolator = interpolate.interp1d(scenarios["grid_logmass"], np.log(np.maximum(profiles,1e-300)),
                                        axis=2, bounds_error=False, fill_value="extrapolate")
    values = np.exp(interpolator(np.log10(masses)))
    return profile_at_x(scenarios["x"],values,x)


def redistribution_velocity(counts, logmass, z, coarse):
    """Allocate coarse per-cell sum(vlos^2) by fine counts, preserving each sum.

    This conditional allocation is an explicit approximation until the parent
    map pass supplies velocities on the same fine histogram. It does not add
    a model for the velocity field or its distinct-halo correlation.
    """
    im = np.clip(np.searchsorted(coarse["logm_edges"],logmass,side="right")-1,0,len(coarse["logm_edges"])-2)
    iz = np.clip(np.searchsorted(coarse["z_edges"],z,side="right")-1,0,len(coarse["z_edges"])-2)
    keys = iz[:,None]+im[None,:]*(len(coarse["z_edges"])-1)
    normalization = np.bincount(keys.ravel(),weights=counts.ravel(),minlength=coarse["hist_v2_c"].size)
    source = coarse["hist_v2_c"].ravel()
    allocation = np.divide(source[keys]*counts,normalization[keys],out=np.zeros_like(counts),where=normalization[keys]>0)
    assigned = source[normalization>0].sum()
    assert np.isclose(allocation.sum(),assigned,rtol=1e-12)
    return allocation,dict(method="fixed coarse-cell sum(vlos^2) redistributed in proportion to fine count", 
                           source_sum_vlos2_km2s2=float(source.sum()),allocated_sum_vlos2_km2s2=float(allocation.sum()),
                           unrepresented_fraction=float(1-assigned/source.sum()),
                           note="No velocities reread. Replace with the parent's matched fine histogram when available.")


def transform_gate():
    x = np.geomspace(1e-5,4.,512)
    q = np.r_[0., .001, .02, .1, .199, .201, 1., 3., 10., 50., 100., 300.]
    weights = linear_radial_kernel(x,q)
    checks = []
    for field in (0,1):
        values = published_profile(field,np.array([1e14]),.5,x)[0]
        actual = weights@values
        exact = []
        for frequency in q:
            if frequency == 0:
                result=integrate.quad(lambda r:4*np.pi*r*r*published_profile(field,np.array([1e14]),.5,np.array([max(r,1e-30)]))[0,0],0,4,epsabs=1e-9,epsrel=1e-9)[0]
            else:
                result=4*np.pi/frequency*integrate.quad(lambda r:r*published_profile(field,np.array([1e14]),.5,np.array([max(r,1e-30)]))[0,0],0,4,weight="sin",wvar=frequency,epsabs=1e-9,epsrel=1e-9)[0]
            exact.append(result)
        exact=np.asarray(exact)
        error=float(np.max(np.abs(actual-exact))/exact[0])
        checks.append(dict(field=C.FIELDS[field],maximum_absolute_error_relative_to_zero_mode=error,
                           q=q,relative_difference=np.divide(actual-exact,exact,out=np.full_like(exact,np.nan),where=np.abs(exact)>1e-20)))
        assert error<3e-3,checks[-1]
    return checks


def replay_fine_velocity(path, output=HERE):
    """Reweight saved tau kernels after the parent streams fine velocities once."""
    output=Path(output)
    saved=np.load(output/"tau_kernel_cache.npz")
    velocity=np.load(path)
    key=next(k for k in ("sum_vlos2","sum_vlos2_km2s2","hist_v2_c","vlos2") if k in velocity)
    values=velocity[key]
    if values.shape!=tuple(saved["histogram_shape"]):
        raise ValueError("Fine velocity histogram shape differs from cached count grid")
    iz,im=saved["cell_z"],saved["cell_mass"]
    kernels=saved["tau_kernels"]
    cl=np.einsum("cpl,p->cl",kernels**2,values[iz,im]/C_KMS**2)/(4*np.pi)
    np.savez_compressed(output/"ksz_self_fine_velocity.npz",ell=saved["ell"],labels=saved["labels"],cl_ksz_self=cl,
                        dl_ksz_self_uK2=cl*saved["ell"]*(saved["ell"]+1)/(2*np.pi)*T_CMB_UK**2,
                        matched_fine_velocity=True)
    return cl


def run(profile_path=PROJECT/"profile_scenarios.npz", histogram_path=COUNTS,
        output=HERE, velocity_histogram_path=None):
    started=time.perf_counter()
    output=Path(output);output.mkdir(parents=True,exist_ok=True);(output/"figures").mkdir(exist_ok=True)
    profile_path=Path(profile_path);histogram_path=Path(histogram_path)
    scenarios=dict(np.load(profile_path));hist=dict(np.load(histogram_path));coarse=dict(np.load(VELOCITIES))
    source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (profile_path,histogram_path,INGREDIENTS,VELOCITIES)}
    ingredients=Ingredients()
    names=scenarios["case_names"].astype(str)
    cutmass=scenarios.get("cut_logmass",scenarios.get("case_cut_logmass",np.full(len(names),np.inf)))
    cutmass=np.where(np.isfinite(cutmass),cutmass,np.inf)
    counts=hist["N"]
    z=(hist["z_edges"][1:]+hist["z_edges"][:-1])/2
    mass=np.exp((hist["lnm_edges"][1:]+hist["lnm_edges"][:-1])/2)/H
    if velocity_histogram_path is None:
        v2,velocity_note=redistribution_velocity(counts,np.log10(mass),z,coarse)
    else:
        velocity=np.load(velocity_histogram_path)
        source_hashes[str(Path(velocity_histogram_path))]=hashlib.sha256(Path(velocity_histogram_path).read_bytes()).hexdigest()
        key=next(k for k in ("sum_vlos2","sum_vlos2_km2s2","hist_v2_c","vlos2") if k in velocity)
        v2=velocity[key]
        assert v2.shape==counts.shape
        velocity_note=dict(method="matched fine-grid sum(vlos^2)",source=str(velocity_histogram_path))
    # Baseline for each cutoff, plus every scenario at 4R. Labels determine mapping.
    cases=[(0,0),(0,1)]+[(ci,2) for ci in range(len(names))]
    labels=np.array([f"{names[ci]}_R{int(C.RADII[ri])}" for ci,ri in cases])
    cl_y_self=np.zeros((len(cases),len(ELL)));cl_y_linear2h=np.zeros_like(cl_y_self)
    cl_ksz_self=np.zeros_like(cl_y_self)
    dm_mean=np.zeros(len(cases));lambda_hit=np.zeros(len(cases));mean_y=np.zeros(len(cases))
    occupied=np.argwhere(counts>0);lookup=np.full(counts.shape,-1,int);lookup[tuple(occupied.T)]=np.arange(len(occupied))
    tau_cache=np.zeros((len(cases),len(occupied),len(ELL)),np.float32)
    cell_y_self=np.zeros((len(cases),len(occupied),len(ELL)),np.float32)
    cell_ksz_self=np.zeros_like(cell_y_self)
    chi=ingredients.da_spline(hist["z_edges"])*(1+hist["z_edges"])
    shell_volume=np.diff(chi**3)/3 # Mpc^3 per sr; count histogram describes full sky
    shell_volume=np.maximum(shell_volume,1e-30)
    max_angle=0.
    for j,zz in enumerate(z):
        occupied_mass=np.flatnonzero(counts[j]>0)
        if not len(occupied_mass):continue
        masses=mass[occupied_mass]
        da,radius,y_amp,tau_amp=ingredients.geometry(zz,masses)
        max_angle=max(max_angle,float(np.max(4*radius/da)))
        q=radius[:,None]/da*(ELL[None,:]+.5);q[:,0]=0.
        bias=ingredients.bias(zz,masses);pk=ingredients.pk(zz,ELL)
        rows=lookup[j,occupied_mass]
        for ri,r_cut in enumerate(C.RADII):
            local_cases=[(out_i,ci) for out_i,(ci,radial_i) in enumerate(cases) if radial_i==ri]
            x=radial_grid(scenarios["x"],r_cut)
            base=profile_mass_grid(scenarios,ri,masses,x)
            evolution=np.array([published_profile(fi,masses,zz,x)/published_profile(fi,masses,0.,x) for fi in (0,1)])
            evolved=base*evolution[None]
            # Physical mass cut is applied after table interpolation, exactly at the threshold.
            for ci in range(len(names)):
                evolved[ci,:,np.log10(masses)>=cutmass[ci]]=0.
            weights=linear_radial_kernel(x,q)
            selected=np.array([ci for _,ci in local_cases])
            transform=np.einsum("mlx,cfmx->cfml",weights,evolved[selected],optimize=True)
            for local_i,(out_i,ci) in enumerate(local_cases):
                y=transform[local_i,0]*y_amp[:,None]
                tau=transform[local_i,1]*tau_amp[:,None]
                d1=counts[j,occupied_mass,None]*y*y/(4*np.pi)
                dk=v2[j,occupied_mass,None]/C_KMS**2*tau*tau/(4*np.pi)
                cl_y_self[out_i]+=d1.sum(axis=0)
                cl_ksz_self[out_i]+=dk.sum(axis=0)
                A=np.sum(counts[j,occupied_mass,None]*bias[:,None]*y,axis=0)/(4*np.pi*shell_volume[j])
                cl_y_linear2h[out_i]+=shell_volume[j]*A*A*pk
                tau_cache[out_i,rows]=tau
                cell_y_self[out_i,rows]=d1
                cell_ksz_self[out_i,rows]=dk
                mean_y[out_i]+=np.sum(counts[j,occupied_mass]*y[:,0])/(4*np.pi)
                if zz<=1:
                    dm_mean[out_i]+=np.sum(counts[j,occupied_mass]*tau[:,0])/(4*np.pi*C.P.SIGMA_T*M2_PER_PC_CM3*(1+zz))
                    angle=np.minimum(np.pi,r_cut*radius/da)
                    active=np.log10(masses)<cutmass[ci]
                    lambda_hit[out_i]+=np.sum(counts[j,occupied_mass]*(1-np.cos(angle))/2*active)
        if j%75==0:
            print(f"Analytic lightcone shell {j+1}/{len(z)}",flush=True)
    gate=transform_gate()
    factor=ELL*(ELL+1)/(2*np.pi)
    np.savez_compressed(output/"analytic_spectra.npz",ell=ELL,labels=labels,case_indices=np.array(cases),
                        cl_y_self=cl_y_self,cl_y_linear2h=cl_y_linear2h,
                        cl_ksz_self=cl_ksz_self,dl_y_self=cl_y_self*factor,dl_y_linear2h=cl_y_linear2h*factor,
                        dl_ksz_self_uK2=cl_ksz_self*factor*T_CMB_UK**2,
                        frb_z1_mean_DM_pc_cm3=dm_mean,poisson_interception_lambda=lambda_hit,
                        poisson_zero_hit_probability=np.exp(-lambda_hit),mean_Compton_y=mean_y,
                        cell_y_self=cell_y_self,cell_ksz_self=cell_ksz_self,
                        cell_z=occupied[:,0],cell_mass=occupied[:,1],z=z,mass_msun=mass)
    np.savez_compressed(output/"tau_kernel_cache.npz",ell=ELL,labels=labels,tau_kernels=tau_cache,
                        cell_z=occupied[:,0],cell_mass=occupied[:,1],histogram_shape=counts.shape)
    records=[]
    for ci,label in enumerate(labels):
        for li,ell in enumerate(ELL[1:],start=1):
            records.append(dict(case=label,ell=ell,Cl_y_self=cl_y_self[ci,li],Cl_y_linear_2h=cl_y_linear2h[ci,li],
                                Dl_ksz_bulk_self_uK2=cl_ksz_self[ci,li]*factor[li]*T_CMB_UK**2))
    with (output/"analytic_spectra.csv").open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    measured=[]
    for ci,label in enumerate(labels):
        measured.append(dict(case=label,mean_y=mean_y[ci],mean_halo_DM_z1_pc_cm3=dm_mean[ci],
                             poisson_interception_lambda=lambda_hit[ci],poisson_zero_hit_probability=np.exp(-lambda_hit[ci])))
    C.write_json(output/"analytic_DM_moments.json",measured)
    metadata=dict(redshift_evolution="F(M,z,x)=F0(M,x)*Fpublished(M,z,x)/Fpublished(M,0,x), an assumed extrapolation",
                  cosmology=dict(h=H,Omega_m=.31,Omega_b=.049),radii=C.RADII,labels=labels,
                  source_SHA256=source_hashes,catalogue_halos=int(counts.sum()),occupied_cells=len(occupied),
                  count_grid=counts.shape,velocity_histogram=velocity_note,
                  pressure_normalization="P_e/P200; P200=GM200 (200rho_c) fb/(2R200)",
                  electron_normalization="rho_tau/(fb rho_c), rho_tau=mu_XGP n_e; mu_XGP=(m_e+2m_p/1.76)/.9",
                  radial_transform="Exact oscillatory integral of piecewise-linear radial interpolation on the exported 512-node grid; missing r<1e-5R is negligible for the fixed shallow inner slopes.",
                  mass_redshift_approximation="Catalogue cells evaluated at mass/redshift bin centres; tests compare the same cells. Radial profiles log-interpolate in mass between exported grid nodes.",
                  tsz_self="C_l self = sum N_cell |y_l|^2/(4pi)",
                  tsz_linear2h="C_l linear2h = sum_z V_shell/sr [sum_mass N_cell b y_l/(4pi V_shell/sr)]^2 P_lin; includes cross-mass terms",
                  ksz_self="C_l self = sum_cell sum(vlos^2)/c^2 |tau_l|^2/(4pi); halo bulk motion only",
                  ksz_distinct_halo_term="Not computed here. Parent must use matched map total minus catalogue self, with signed finite-realization residual and pixel/flat-sky closure caveats.",
                  frb="Mean halo-only observer DM for zsource=1; tau/(sigma_T (1+z) pc_cm3_column). Poisson zero-hit is an independent-centre model, not measured clustered-ray zero-hit probability; no full PDF computed here.",
                  geometry="Flat-sky spherical Fourier transform, q=(ell+.5)R/DA; integrated means use q=0. No beam, map pixel window, instrumental noise, host/MW/IGM/reionization model.",
                  maximum_4R_angular_extent_radians=max_angle,
                  ownership="All-gas profiles around centres are not uniquely owned; linear2h added to these profiles can double-count environmental gas. no_other scenario is a sensitivity control, not a formal decomposition.",
                  transform_gate=gate,elapsed_seconds=time.perf_counter()-started,
                  peak_RSS_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2)
    C.write_json(output/"analytic_metadata.json",metadata)
    plot_results(output)
    assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest for path,digest in source_hashes.items())
    C.write_json(output/"analytic_verification.json",dict(passed=True,source_hashes_unchanged=True,
                 all_spectra_finite=bool(np.isfinite(cl_y_self).all() and np.isfinite(cl_y_linear2h).all() and np.isfinite(cl_ksz_self).all()),
                 no_negative_self_or_linear_power=bool((cl_y_self>=0).all() and (cl_y_linear2h>=0).all() and (cl_ksz_self>=0).all()),
                 no_new_particle_or_catalogue_reads=True,no_CLASS_calls=True,no_jobs=True,transform_gate=gate))
    print(json.dumps(C.plain(dict(elapsed=metadata["elapsed_seconds"],peak_RSS_GiB=metadata["peak_RSS_GiB"],DM=measured)),indent=2))
    return metadata


def plot_results(output=HERE):
    output=Path(output);arrays=np.load(output/"analytic_spectra.npz")
    plt.rcParams.update({"font.size":13,"axes.labelsize":15,"axes.titlesize":16,
                         "legend.fontsize":11,"axes.spines.top":False,"axes.spines.right":False})
    ell=arrays["ell"][1:]
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for i,radius in enumerate((2,3,4)):
        line=axes[0].plot(ell,1e12*(arrays["dl_y_self"][i,1:]+arrays["dl_y_linear2h"][i,1:]),label=f"{radius}R: self + linear 2h")[0]
        axes[0].plot(ell,1e12*arrays["dl_y_linear2h"][i,1:],ls="--",alpha=.7,color=line.get_color())
        axes[1].plot(ell,arrays["dl_ksz_self_uK2"][i,1:],label=f"{radius}R: bulk self")
    axes[0].set(xscale="log",yscale="log",xlabel=r"$\ell$",ylabel=r"$10^{12}D_\ell^{yy}$",title="Conditional tSZ halo model")
    axes[1].set(xscale="log",yscale="log",xlabel=r"$\ell$",ylabel=r"$D_\ell^{\rm kSZ,self}\ [\mu{\rm K}^2]$",title="Conditional bulk kSZ self term")
    axes[0].plot([],[],ls="--",color=".4",label="Linear 2h only")
    for ax in axes:ax.legend()
    fig.tight_layout()
    for extension in ("png","svg"):fig.savefig(output/"figures"/("radius_spectra."+extension),dpi=180,bbox_inches="tight")
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    labels=arrays["labels"].astype(str)
    baseline=2
    for i,label in enumerate(labels):
        if not any(token in label for token in ("pooled","cut_ge","unseen_tail")):continue
        display={"pooled_0p5_R4":"Pooled 0.5 dex","pooled_0p7_R4":"Pooled 0.7 dex",
                 "cut_ge15p3_R4":r"Cut $\log_{10}M\geq15.3$","cut_ge15p1_R4":r"Cut $\log_{10}M\geq15.1$",
                 "unseen_tail_last_fit_R4":"Unseen tail: last fit"}.get(label,label)
        axes[0].plot(ell,100*((arrays["cl_y_self"][i,1:]+arrays["cl_y_linear2h"][i,1:])/(arrays["cl_y_self"][baseline,1:]+arrays["cl_y_linear2h"][baseline,1:])-1),label=display)
        axes[1].plot(ell,100*(arrays["cl_ksz_self"][i,1:]/arrays["cl_ksz_self"][baseline,1:]-1))
    for ax,title in zip(axes,("tSZ: self + linear 2h","kSZ: bulk self")):
        ax.set(xscale="log",xlabel=r"$\ell$",ylabel="Change from baseline [%]",title=title);ax.axhline(0,color=".5",lw=1)
    axes[0].legend(fontsize=10)
    fig.tight_layout()
    for extension in ("png","svg"):fig.savefig(output/"figures"/("high_mass_sky_sensitivity."+extension),dpi=180,bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    run()
