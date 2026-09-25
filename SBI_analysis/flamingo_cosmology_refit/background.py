"""Prepare linear spectra and distances with explicit units and provenance."""
import argparse
import json
from pathlib import Path

import camb
import numpy as np

from prior_model import save_json


def prepare(root):
    config = json.loads((root/"code/cosmologies.json").read_text())
    z = np.expm1(np.linspace(0, np.log(5), 100))
    k = np.geomspace(1e-5, 100, 2048)  # 1/Mpc, not h/Mpc
    for name in ("halfdome_native", "flamingo_D3A"):
        target = root/"audit"/(name+".npz")
        if target.exists():
            continue
        c = config[name]
        p = camb.CAMBparams()
        p.set_cosmology(H0=100*c["h"], ombh2=c["Omega_b"]*c["h"]**2,
                        omch2=(c["Omega_m"]-c["Omega_b"])*c["h"]**2,
                        mnu=c["mnu_eV"], num_massive_neutrinos=int(c["mnu_eV"]>0),
                        neutrino_hierarchy="degenerate", nnu=3.046, TCMB=2.7255)
        # Include the massive-neutrino density inside, not in addition to, Om.
        p.omch2 = c["Omega_m"]*c["h"]**2-p.ombh2-p.omnuh2
        p.InitPower.set_params(As=c.get("A_s",2.1e-9), ns=c["n_s"])
        p.WantCls = False
        p.set_matter_power(redshifts=z[::-1], kmax=100, nonlinear=False, silent=True)
        result = camb.get_results(p)
        sigma8_initial = float(result.get_sigma8_0())
        factor = (c["sigma8"]/sigma8_initial)**2 if "sigma8" in c else 1.0
        sigma8 = sigma8_initial*np.sqrt(factor)
        if "sigma8_published_rounded" in c:
            assert abs(sigma8-c["sigma8_published_rounded"]) < .002
        arrays = dict(z=z, k=k, H_km_s_Mpc=result.hubble_parameter(z),
                      chi_Mpc=result.comoving_radial_distance(z))
        for variable, label in (("delta_nonu","P_cb_Mpc3"),("delta_tot","P_total_Mpc3")):
            power = result.get_matter_power_interpolator(nonlinear=False,
                var1=variable,var2=variable,hubble_units=False,k_hunit=False)
            arrays[label] = factor*power.P(z,k)
            assert np.isfinite(arrays[label]).all() and np.all(arrays[label]>0)
        np.savez(target,**arrays)
        save_json(root/"audit"/(name+".json"),dict(input=c,camb_version=camb.__version__,
            sigma8=sigma8,A_s_effective=p.InitPower.As*factor,
            Omega_nu=float(p.omnuh2/c["h"]**2),Omega_cb=float((p.omch2+p.ombh2)/c["h"]**2),
            units="k in 1/Mpc, P in Mpc^3, distances in Mpc, H in km/s/Mpc",
            normalization="native HD normalized to published sigma8; FL uses published A_s"))
        print(name,"sigma8",sigma8,flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument("--root",type=Path,required=True)
    prepare(parser.parse_args().root)
