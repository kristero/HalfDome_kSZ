"""Embed compact measurements and joint coefficient samples in an offline viewer."""
import json
from pathlib import Path
import numpy as np
from plotly.offline import get_plotlyjs
from fit_study import plain

HERE=Path(__file__).resolve().parent


def main():
    summary=json.loads((HERE/'products/study_summary.json').read_text())
    cases=[]
    for case in summary['bins']:
        p=dict(np.load(HERE/'products'/(case['label']+'_profiles.npz')))
        individual=dict(np.load(HERE/'products'/(case['label']+'_parameters.npz')))
        samples=[];ids=[]
        for ri in range(3):
            valid=individual['accepted'][:,ri].all(axis=1)
            samples.append(individual['parameters'][valid,ri].reshape(-1,6))
            ids.append(individual['halo_ids'][valid])
        cases.append({'label':'-'.join(f'{x:.1f}' for x in case['mass_edges']),
            'count':case['halos'],'supported':case['strict_resolution_supported'],
            'edges':p['edges'],'mask':p['fit_base_mask'],
            'pressure':{'mean':p['all_pressure_mean'],'std':p['all_pressure_std'],
                'p':p['all_pressure_percentiles'],'reference':p['pressure_published_reference'],
                'fits':case['selections']['all']['pressure']},
            'electrons':{'mean':p['all_electrons_mean'],'std':p['all_electrons_std'],
                'p':p['all_electrons_percentiles'],'reference':p['electrons_published_reference'],
                'fits':case['selections']['all']['electrons']},
            'samples':samples,'ids':ids})
    template=(HERE/'viewer_template.html').read_text()
    html=template.replace('__PLOTLY__',get_plotlyjs()).replace('__DATA__',json.dumps(plain(cases),allow_nan=False,separators=(',',':')))
    output=HERE/'profile_explorer.html';output.write_text(html)
    print(json.dumps({'path':str(output),'bytes':output.stat().st_size,'bins':len(cases)}))


if __name__=='__main__':main()
