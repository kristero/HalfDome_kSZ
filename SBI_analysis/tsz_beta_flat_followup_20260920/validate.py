"""Independent high-precision integrals and read-only production provenance."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
import mpmath as mp
import numpy as np
from study import column, energy


def main(root):
    mp.mp.dps = 60
    results = json.loads((root/'results/summary.json').read_text())
    errors, energy_errors = [], []
    for case in results['profiles']['cases']:
        p0, xc, beta = case['evolved']
        p, c, b = [mp.mpf(float(v)) for v in (p0, xc, beta)]
        x = mp.mpf(1)
        for outer in (None, 4., 8., 16.):
            end = mp.mpf(100000) if outer is None else mp.sqrt(mp.mpf(outer)**2-x*x)
            # Independent integration variable: direct ell, with decade breaks.
            knots = [mp.mpf(0)] + [mp.mpf(v) for v in (.01, .1, 1, 10, 100, 1000, 10000) if v < end] + [end]
            def integrand(ell):
                q = mp.sqrt(x*x + ell*ell)/c
                return 2*p*q**mp.mpf('-.3')*(1+q)**(-b)
            reference = mp.quad(integrand, knots)
            actual = column(float(x), p0, xc, beta, outer=outer)
            errors.append(float(abs(mp.mpf(actual)/reference-1)))
        for outer in (4., 8., 16.):
            upper = mp.mpf(outer)/(mp.mpf(outer)+c)
            reference = p*c**3*mp.betainc(mp.mpf('2.7'), b-mp.mpf('2.7'), 0, upper)
            energy_errors.append(float(abs(mp.mpf(energy(outer,p0,xc,beta))/reference-1)))
    assert max(errors) < 1e-9 and max(energy_errors) < 1e-9
    audit = dict(precision_digits=60, direct_los_checks=len(errors), energy_special_function_checks=len(energy_errors),
                 max_los_relative_error=max(errors), max_energy_relative_error=max(energy_errors))
    production = Path('/lustre/work/kristero10/halfdome_flamingo_linear_8192_20260915')
    if production.exists():
        manifest_path = production/'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        expected_manifest = '994333174a0c5a1e636a5cc065d000464ea3b4ddee2bb0082ccb4cfdd70cc600'
        digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest(manifest_path) == expected_manifest
        checks = [(production/'code'/name, value) for name,value in manifest['code_sha256'].items()]
        checks += [(production/'design'/name, value) for name,value in manifest['design_sha256'].items()]
        checks += [(Path(name), value) for name,value in manifest['dependency_sha256'].items()]
        checks += [(production/'run_config.json', manifest['run_config_sha256'])]
        for path, expected in checks:
            assert digest(path) == expected, str(path)
        audit['production'] = dict(manifest_unchanged=True, checked_hashes=len(checks)+1,
                                  root=str(production), manifest_sha256=expected_manifest)
    (root/'results/independent_validation.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit,indent=2),flush=True)
    subprocess.run([sys.executable,str(root/'render.py')],check=True)
    subprocess.run([sys.executable,str(root/'continuous_flat.py')],check=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parent)
    main(parser.parse_args().root)
