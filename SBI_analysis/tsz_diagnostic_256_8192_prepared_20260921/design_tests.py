"""Flat physical prior, stable dataset prefixes, seed isolation, noise controls."""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.stats import qmc

ROOT=Path(__file__).resolve().parent
LOW=np.array([1,.025,2.8,-.6,-1,-.2,-6,-1.5,-.5])
HIGH=np.array([60,4,16,1.5,.4,.4,.5,3,2])
NAMES=['P0','xc','beta','alpha_m_P0','alpha_m_xc','alpha_m_beta',
       'alpha_z_P0','alpha_z_xc','alpha_z_beta']


def design(count,seed=20260920):
    return LOW+(HIGH-LOW)*qmc.Sobol(9,scramble=True,seed=seed).random_base2(int(np.ceil(np.log2(count))))[:count]


def noise_seed(row,split,stream='train',master=20260920):
    key=f'sphere4-flat-v1|{master}|{stream}|{row}|{split}'.encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8],'big')&((1<<63)-1)


def log_prob(theta):
    theta=np.asarray(theta)
    return np.where(np.all((theta>=LOW)&(theta<=HIGH),axis=-1),-np.log(HIGH-LOW).sum(),-np.inf)


def noise_test():
    # Real independent modes are sufficient to verify the Gaussian split
    # estimator identity. Fixed signal -> no 2*C^2 cosmic-variance term.
    rng=np.random.default_rng(20260920);modes=1024;samples=8192
    records=[]
    for noise,shared in [(1.,0.),(2.,0.),(1.,.3)]:
        cross=[];auto=[]
        signal=np.ones(modes)
        for _ in range(samples//128):
            common=rng.normal(scale=np.sqrt(shared),size=(128,modes))
            a=signal+common+rng.normal(scale=np.sqrt(noise),size=(128,modes))
            b=signal+common+rng.normal(scale=np.sqrt(noise),size=(128,modes))
            cross.extend(np.mean(a*b,axis=1));auto.extend(np.mean(a*a,axis=1))
        expected=1+shared
        variance=(2*(noise+2*shared)+(noise+shared)**2+shared**2)/modes
        z=(np.mean(cross)-expected)/np.sqrt(variance/samples)
        records.append(dict(split_noise=noise,shared_foreground_power=shared,
            cross_mean=float(np.mean(cross)),expected_cross=expected,
            auto_mean=float(np.mean(auto)),expected_auto=1+shared+noise,
            variance_ratio=float(np.var(cross,ddof=1)/variance),mean_z=float(z)))
    assert all(abs(r['mean_z'])<6 and abs(r['variance_ratio']-1)<.08 for r in records)
    return records


def main():
    big=design(524288);small=design(8192);assert np.array_equal(big[:8192],small)
    assert np.all(np.isfinite(log_prob(big)))
    counts=np.stack([np.histogram((small[:,j]-LOW[j])/(HIGH[j]-LOW[j]),bins=32,range=(0,1))[0] for j in range(9)])
    assert np.all(counts==256)
    seeds=[noise_seed(row,split) for row in range(524288) for split in [1,2]]
    assert len(set(seeds))==len(seeds)
    assert len(set(seeds+[noise_seed(i,j,'test') for i in range(1024) for j in [1,2]]))==len(seeds)+2048
    manifest=dict(parameter_order=NAMES,lower=LOW.tolist(),upper=HIGH.tolist(),
        density='independent uniform in physical values; no rejection or FLAMINGO weighting',
        seed=20260920,sampler='scrambled Sobol; fixed seed and dimension preserve prefixes',
        physical_model='B12 alpha=1, gamma=-0.3, pressure truncated at 4 physical R200c',
        old_dataset_reuse='No clean spectrum can be relabelled across cylindrical/spherical geometry',
        production_status='Not certified: full-map rendering and inference gates outstanding')
    (ROOT/'inputs/flat_prior.json').write_text(json.dumps(manifest,indent=2)+'\n')
    np.save(ROOT/'inputs/diagnostic_theta_128.npy',small[:128])
    result=dict(prefix_8192_524288_exact=True,uniform_32_bin_counts=counts.tolist(),
        unique_training_split_seeds=len(seeds),noise_controls=noise_test())
    (ROOT/'results/design_tests.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='uniform_32_bin_counts'},indent=2))


if __name__=='__main__':main()
