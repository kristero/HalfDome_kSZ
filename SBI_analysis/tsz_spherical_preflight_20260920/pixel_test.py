"""Sparse HEALPix experiments: sample before/after the beam and integrate pixels.

No full-sky arrays are allocated. Fourier tests are tangent-plane halo tests;
they do not replace a masked full-sky convergence test. Every profile has unit
integrated Y, solely to express sampling errors as relative errors.
"""
import json
from pathlib import Path
import time
import numpy as np
import healpy as hp
from scipy.interpolate import PchipInterpolator
from scipy.special import i0e,j0,roots_legendre
from scipy.integrate import quad
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from projection import column,volume,shape

ROOT=Path(__file__).resolve().parent
ARCMIN=np.pi/10800
SIGMA=2*ARCMIN/np.sqrt(8*np.log(2))
ELLS=np.array([1000,2000,4000,6000,8000])


def radial_profile(theta200,xc,beta):
    # Interpolate log column including an exact zero beyond the physical edge.
    xx=np.r_[0.,np.geomspace(1e-11,4,1000)]
    # Interpolate the smooth chord MEAN, then restore the exact edge factor.
    yy=np.array([column(x,xc,beta)/(2*np.sqrt((4-x)*(4+x))) if x<4
                 else shape(4.,xc,beta) for x in xx])
    curve=PchipInterpolator(np.log(np.maximum(xx,1e-13)),np.log(np.maximum(yy,1e-300)))
    norm=theta200**2*volume(xc,beta)
    def profile(theta):
        x=np.asarray(theta)/theta200
        inside=np.clip(x,0,4)
        return 2*np.sqrt((4-inside)*(4+inside))*np.exp(curve(np.log(np.clip(x,1e-13,4))))/norm
    # Midpoint quadrature in log projected radius; double it to test convolution.
    beams=[];integrals=[]
    support=4*theta200+9*SIGMA
    radial=np.r_[0.,np.geomspace(1e-7*SIGMA,support,1200)]
    for n in [512,1024]:
        nodes,w=roots_legendre(n)
        lo,hi=np.log(1e-11),np.log(4)
        x=np.exp(lo+(nodes+1)*(hi-lo)/2)
        t=x*theta200
        weights=2*np.pi*t*t*profile(t)*w*(hi-lo)/2
        integrals.append(float(weights.sum()))
        smoothed=np.empty(len(radial))
        for start in range(0,len(radial),100):
            r=radial[start:start+100,None]
            smoothed[start:start+100]=np.sum(weights/(2*np.pi*SIGMA**2)*
                np.exp(-.5*((r-t)/SIGMA)**2)*i0e(r*t/SIGMA**2),axis=1)
        beams.append(smoothed)
    conv_error=float(np.max(abs(beams[0]-beams[1]))/max(beams[1]))
    beam_curve=PchipInterpolator(radial,np.log(np.maximum(beams[1],1e-300)))
    def prebeam(theta):
        theta=np.asarray(theta)
        return np.where(theta<=support,np.exp(beam_curve(np.minimum(theta,support))),0.)
    reference=np.array([np.sum(weights*j0((ell+.5)*t))*np.exp(-.5*(ell+.5)**2*SIGMA**2) for ell in ELLS])
    beam_flux=quad(lambda r:2*np.pi*r*prebeam(r),0,support,epsabs=2e-7,limit=200)[0]
    return profile,prebeam,support,reference,dict(radial_flux=integrals,
        convolution_relative_change=conv_error,prebeam_integrated_flux=float(beam_flux))


def offsets(nside,pixels,centre):
    vec=np.array(hp.pix2vec(nside,pixels,nest=True)).T
    dot=vec@centre
    cross=np.linalg.norm(np.cross(vec,centre),axis=1)
    theta=np.arctan2(cross,dot)
    e1=np.cross([0.,0.,1.],centre);e1/=np.linalg.norm(e1)
    e2=np.cross(centre,e1)
    factor=np.divide(theta,cross,out=np.ones_like(theta),where=cross>0)
    return theta,np.c_[vec@e1,vec@e2]*factor[:,None]


def measure(nside,centre,profile,radius,beam_after,children=1):
    pixels=hp.query_disc(nside,centre,radius,inclusive=children>1,nest=True)
    theta,xy=offsets(nside,pixels,centre)
    if children==1:
        value=profile(theta)
    else:
        # Nested HEALPix children exactly partition the parent. This quadrature
        # converges with child resolution; it is not claimed to be exact.
        sub=(pixels[:,None]*children**2+np.arange(children**2)).ravel()
        subtheta,_=offsets(nside*children,sub,centre)
        value=profile(subtheta).reshape(len(pixels),children**2).mean(axis=1)
    weights=value*hp.nside2pixarea(nside)
    power=[]
    directions=np.c_[np.cos(np.arange(12)*np.pi/12),np.sin(np.arange(12)*np.pi/12)]
    phase=xy@directions.T
    for ell in ELLS:
        modes=np.sum(weights[:,None]*np.exp(-1j*(ell+.5)*phase),axis=0)
        if beam_after:modes*=np.exp(-.5*(ell+.5)**2*SIGMA**2)
        power.append(float(np.mean(abs(modes)**2)))
    return float(weights.sum()),power


def main():
    started=time.monotonic();rng=np.random.default_rng(20260920)
    # Angular sizes cover unresolved and resolved systems; xc and beta are the
    # evolved shape values at each halo, not necessarily pivot amplitudes.
    cases=[('B12 compact',.35,.497,4.35),('B12 resolved',2.,.497,4.35),
           ('Small core',1.,.025,4.35),('Steep beta',1.,.497,16.),
           ('Shallow beta',1.,1.13,.644)]
    records=[];profiles={}
    for label,t200,xc,beta in cases:
        raw,pre,support,reference,checks=radial_profile(t200*ARCMIN,xc,beta)
        profiles[label]=(raw,pre)
        for nside in [4096,8192,16384]:
            methods=[('point then beam',raw,4*t200*ARCMIN,True,1),
                     ('beam then point',pre,support,False,1)]
            if nside==4096:
                methods += [('pixel integral 4x',raw,4*t200*ARCMIN,True,4),
                            ('pixel integral 16x',raw,4*t200*ARCMIN,True,16)]
            for name,profile,radius,after,children in methods:
                flux=[];power=[]
                positions=32 if children==1 else 12
                # Identical centre sequence for all methods and resolutions.
                positions_rng=np.random.default_rng(20260920)
                for _ in range(positions):
                    centre=positions_rng.normal(size=3);centre/=np.linalg.norm(centre)
                    f,p=measure(nside,centre,profile,radius,after,children)
                    flux.append(f);power.append(p)
                records.append(dict(case=label,theta200_arcmin=t200,xc=xc,beta=beta,
                    nside=nside,method=name,flux=flux,power=power,
                    power_reference=(reference**2).tolist(),checks=checks))
        print(label,'completed',time.monotonic()-started,flush=True)
    result=dict(records=records,ell=ELLS.tolist(),seconds=time.monotonic()-started,
        scope='Sparse actual HEALPix sampling; local Fourier power, not masked full-sky C_ell',
        pixel_power_convention='Pixel-integrated Fourier curves include the parent pixel response; no pixel-window deconvolution')
    (ROOT/'results/pixel_test.json').write_text(json.dumps(result,indent=2)+'\n')
    plt.rcParams.update({'font.size':14,'axes.labelsize':16,'legend.fontsize':12})
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    styles={'point then beam':('o','-'),'beam then point':('s','--')}
    for case_index,(label,*_) in enumerate(cases):
        color=plt.get_cmap('tab10')(case_index)
        for method,(marker,ls) in styles.items():
            rows=[r for r in records if r['case']==label and r['method']==method]
            mean=[np.mean(r['flux'])-1 for r in rows]
            scatter=[np.std(r['flux']) for r in rows]
            axes[0].errorbar([r['nside'] for r in rows],mean,yerr=scatter,
                color=color,marker=marker,ls=ls,label=label if method=='point then beam' else None)
        row=next(r for r in records if r['case']==label and r['method']=='beam then point' and r['nside']==4096)
        relative=np.array(row['power'])/row['power_reference']-1
        axes[1].plot(ELLS,100*relative.mean(0),color=color,label=label)
        axes[1].fill_between(ELLS,100*np.min(relative,axis=0),100*np.max(relative,axis=0),color=color,alpha=.12)
    axes[0].set(xscale='log',yscale='symlog',xlabel='NSIDE',ylabel=r'$Y_{\rm pixels}/Y-1$')
    axes[0].set_xticks([4096,8192,16384],['4096','8192','16384']);axes[0].legend()
    axes[0].set_title('Solid: post-beam; dashed: pre-beam',fontsize=14)
    axes[1].set(xlabel=r'$\ell$',ylabel='Power error [%]',title='Pre-beam painting at NSIDE 4096')
    for ax in axes:ax.axhline(0,color='.5',lw=.8);ax.grid(alpha=.15)
    fig.tight_layout()
    for ext in ['pdf','png']:fig.savefig(ROOT/'plots'/('pixel_prebeam_test.'+ext),dpi=180,bbox_inches='tight')
    fig,ax=plt.subplots(figsize=(8,5));theta=np.geomspace(1e-4,12,700)
    for label,(raw,pre) in profiles.items():
        line,=ax.loglog(theta,raw(theta*ARCMIN)*ARCMIN**2,label=label)
        ax.loglog(theta,pre(theta*ARCMIN)*ARCMIN**2,'--',color=line.get_color())
    ax.set(xlabel=r'$\theta$ [arcmin]',ylabel=r'$y/Y$ [arcmin$^{-2}$]',ylim=(1e-7,1e5))
    ax.legend();ax.grid(alpha=.15);fig.tight_layout()
    for ext in ['pdf','png']:fig.savefig(ROOT/'plots'/('raw_and_prebeam_profiles.'+ext),dpi=180,bbox_inches='tight')


if __name__=='__main__':main()
