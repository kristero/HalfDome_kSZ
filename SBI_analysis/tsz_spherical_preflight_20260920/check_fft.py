"""Evaluate the FFT pre-beam experiment against real-space quadrature."""
import json
from pathlib import Path
import numpy as np
from scipy.interpolate import PchipInterpolator
from pixel_test import radial_profile,ARCMIN
from projection import volume
ROOT=Path(__file__).resolve().parent
records=[]
for i,(t,xc,beta) in enumerate([(.35,.497,4.35),(2.,.497,4.35),(1.,.025,4.35),(1.,.497,16.),(1.,1.13,.644)]):
    _,reference,support,_,_=radial_profile(t*ARCMIN,xc,beta)
    theta=np.geomspace(1e-7*ARCMIN,support,2000)
    actual_reference=reference(theta)*volume(xc,beta)*(t*ARCMIN)**2
    for n in [512,1024,2048]:
        data=np.loadtxt(ROOT/'results'/f'fft_prebeam_{n}.csv',delimiter=',')
        actual=PchipInterpolator(np.log(data[:,0]),data[:,2*i+2])(np.log(theta))
        error=float(np.max(abs(actual-actual_reference))/max(actual_reference))
        safe=theta>=.001*ARCMIN
        records.append(dict(case=i,n=n,peak_normalized_max_error=error,
            error_above_0001_arcmin=float(np.max(abs(actual[safe]-actual_reference[safe]))/max(actual_reference)),
            min_over_peak=float(min(actual)/max(actual_reference))))
(ROOT/'results/fft_prebeam.json').write_text(json.dumps(records,indent=2)+'\n')
print(json.dumps(records,indent=2))
