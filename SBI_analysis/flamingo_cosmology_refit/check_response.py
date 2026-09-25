"""Independent normalization checks and resolution sensitivity of the response."""
import argparse
import gc
import json
from pathlib import Path
import time

import numpy as np
from colossus.cosmology import cosmology
from colossus.lss import mass_function, bias

from halo_response import CosmologyResponse, tinker08, tinker10, top_hat
from prior_model import FIDUCIAL, save_json


def main(root):
    start=time.monotonic()
    cosmology.setCosmology("planck15",persistence="")
    formula_errors=[]
    for delta in (200,667,1200):
        for z in (0.,.5,3.):
            sigma=np.array([.4,.8,1.,2.])
            direct=mass_function.modelTinker08(sigma,z,str(delta)+"m")
            formula_errors.append(float(np.max(abs(tinker08(sigma,z,delta)/direct-1))))
            direct=bias.modelTinker10(1.68647/sigma,z,str(delta)+"m")
            formula_errors.append(float(np.max(abs(tinker10(sigma,delta)/direct-1))))
    assert max(formula_errors)<1e-10
    sigma_errors={}
    for name in ("halfdome_native","flamingo_D3A"):
        data=np.load(root/"audit"/(name+".npz"))
        meta=json.loads((root/"audit"/(name+".json")).read_text())
        window=top_hat(data["k"]*8/meta["input"]["h"])**2
        direct=np.sqrt(np.trapz(data["P_total_Mpc3"][0]*data["k"]**3*window,
                               x=np.log(data["k"]))/ (2*np.pi**2))
        sigma_errors[name]=float(abs(direct/meta["sigma8"]-1))
    assert max(sigma_errors.values())<5e-4
    fits=json.loads((root/"audit/pilot_results.json").read_text())["fits"]
    points={"Battaglia12":FIDUCIAL}
    points.update({name:np.array(record["best"]["theta"]) for name,record in fits.items()})
    coarse=CosmologyResponse(root)
    low={name:coarse(theta) for name,theta in points.items()}
    one,two=coarse.denominator.components(FIDUCIAL)
    fractions=two/(one+two)
    scaled=FIDUCIAL.copy();scaled[0]*=1.5
    np.testing.assert_allclose(coarse(scaled),low["Battaglia12"],rtol=1e-12,atol=0)
    del coarse
    gc.collect()
    fine=CosmologyResponse(root,nmass=48,nz=36,nradial=256)
    high={name:fine(theta) for name,theta in points.items()}
    errors={name:float(np.max(abs(low[name]/high[name]-1))) for name in points}
    result=dict(passed=max(errors.values())<.005,
        maximum_formula_relative_error=max(formula_errors),sigma8_integral_errors=sigma_errors,
        response_relative_resolution_errors=errors,coarse=[32,24,128],fine=[48,36,256],
        radial_rule="order also increases with halo angular size to resolve Hankel oscillations",
        response_ranges={name:[float(v.min()),float(v.max())] for name,v in high.items()},
        fiducial_two_halo_fraction_range=[float(fractions.min()),float(fractions.max())],
        seconds=time.monotonic()-start,
        scope="numerical halo-model validation; not validation against independent N-body cosmologies")
    save_json(root/"audit/response_check.json",result)
    np.savez(root/"audit/reference_responses.npz",**high)
    print(json.dumps(result,indent=2),flush=True)
    assert result["passed"],"Refine the response quadrature before fitting"


if __name__=="__main__":
    p=argparse.ArgumentParser(__doc__);p.add_argument("--root",type=Path,required=True)
    main(p.parse_args().root)
