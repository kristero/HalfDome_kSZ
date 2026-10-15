"""Prepare inspectable tables, accounting, a figure gallery and delivery hashes."""
from pathlib import Path
import csv
import datetime
import hashlib
import html
import json
import re

ROOT=Path(__file__).resolve().parent
JOB_LOGS={
    '602967.idark':'prepare.log','602969.idark':'refit.log',
    **{'602974[%d].idark'%i:'compare_%d.log'%i for i in range(5)},
    '602976.idark':'compare_0_retry.log','603000.idark':'aggregate_initial.log',
    '603002.idark':'aggregate_second_attempt.log','603003.idark':'aggregate.log',
    '603004.idark':'refine_figures.log','603006.idark':'report_build.log',
    '603014.idark':'report_build_retry_initial.log','603015.idark':'report_build_retry.log'}


def load(name):
    return json.loads((ROOT/name).read_text())


def duration(value):
    parts=[float(v) for v in value.split(':')]
    return sum(v*60**i for i,v in enumerate(reversed(parts)))


def accounting():
    rows=[]
    for job,data in load('pbs_accounting_raw.json').items():
        text=data['text']
        row={'job':job,'query_succeeded':data['returncode']==0}
        for key in ['queue','exec_host','Exit_status','resources_used.cput',
                    'resources_used.walltime','resources_used.mem','Resource_List.ncpus']:
            match=re.search(r'^\s*'+re.escape(key)+r'\s*=\s*(.+)$',text,re.M)
            if match:row[key]=match.group(1).strip()
        if 'resources_used.cput' in row:
            row['used_cpu_hours']=duration(row['resources_used.cput'])/3600
        if 'resources_used.walltime' in row:
            row['wall_seconds']=duration(row['resources_used.walltime'])
            row['allocated_core_hours']=row['wall_seconds']*26/3600
        row['accounting_available']='used_cpu_hours' in row or 'wall_seconds' in row
        rows.append(row)
    # GNU time includes child-process CPU and is useful for sub-minute PBS jobs.
    times=[]
    for job,filename in JOB_LOGS.items():
        path=ROOT/filename
        text=path.read_text(errors='replace')
        user=re.search(r'User time \(seconds\): ([\d.]+)',text)
        system=re.search(r'System time \(seconds\): ([\d.]+)',text)
        if user and system:
            record=dict(job=job,file=path.name,cpu_seconds=float(user[1])+float(system[1]))
            wall=re.search(r'Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): ([\d:.]+)',text)
            rss=re.search(r'Maximum resident set size \(kbytes\): (\d+)',text)
            if wall:record['application_wall_seconds']=duration(wall[1])
            if rss:record['max_parent_rss_mib']=int(rss[1])/1024
            times.append(record)
    cpu_values=[r['used_cpu_hours'] for r in rows if 'used_cpu_hours' in r]
    allocation_values=[r['allocated_core_hours'] for r in rows if 'allocated_core_hours' in r]
    result={'pbs_jobs':rows,'gnu_time_logs':times,
            'pbs_used_cpu_hours':sum(cpu_values) if cpu_values else None,
            'pbs_allocated_core_hours':sum(allocation_values) if allocation_values else None,
            'timed_idark_cpu_hours':sum(t['cpu_seconds'] for t in times)/3600,
            'timed_idark_wall_seconds_sum':sum(t.get('application_wall_seconds',0) for t in times),
            'allocation_core_hours_lower_bound':26*sum(t.get('application_wall_seconds',0) for t in times)/3600,
            'complete_unique_job_logs':len(times)==len(JOB_LOGS),
            'note':'One explicit GNU time log per submitted job. CPU includes children; RSS is parent-only. The allocation lower bound excludes untimed shell/render/packaging stages. Missing PBS usage is null, never zero.'}
    (ROOT/'resource_accounting_summary.json').write_text(json.dumps(result,indent=2)+'\n')


def tables():
    rows=load('comparison_summary.json')['rows']
    keys=sorted(set().union(*(r.keys() for r in rows)))
    with (ROOT/'comparison_summary.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader();writer.writerows(rows)
    controls=[]
    for r in rows:
        if r['population']!='halfdome' or r['case'].startswith('fit'):
            continue
        ref=next(x for x in rows if x['population']=='halfdome' and x['case']=='main_hybrid'
                 and x['observable']==r['observable'])
        controls.append(dict(case=r['case'],observable=r['observable'],
            rms_ratio=r['rms']/ref['rms'],mean_ratio=r['mean']/ref['mean'] if r['observable']!='deltaT' else None,
            zero_atom=r['zero_atom']))
    (ROOT/'all_control_ratios.json').write_text(json.dumps(controls,indent=2)+'\n')


def gallery():
    cards=[]
    for path in sorted((ROOT/'figures').glob('*.png')):
        name=path.stem
        svg=path.with_suffix('.svg')
        link=f'<a href="figures/{svg.name}">SVG</a>' if svg.exists() else ''
        cards.append(f'<article data-name="{html.escape(name)}"><h2>{html.escape(name.replace("_"," "))}</h2>'
                     f'<a href="figures/{path.name}"><img loading="lazy" src="figures/{path.name}" alt="{html.escape(name)}"></a>'
                     f'<p><a href="figures/{path.name}">Full PNG</a> {link}</p></article>')
    document='''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>FLAMINGO / HalfDome z=0 comparison</title>
<style>
body{font:17px system-ui,sans-serif;margin:0;background:#f3f6fa;color:#172f43}
header,main{max-width:1400px;margin:auto;padding:24px}header{background:white}
h1{font-size:30px}h2{font-size:19px}p{line-height:1.5}input{font:inherit;padding:10px;width:min(90%,650px)}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(480px,1fr));gap:20px}
article{background:white;border-radius:10px;padding:15px;box-shadow:0 1px 5px #ccd7df}img{width:100%;height:auto}
a{color:#08618a}article[hidden]{display:none}@media(max-width:550px){.grid{display:block}article{margin:15px 0}}
</style><header><h1>FLAMINGO / HalfDome at z = 0</h1>
<p>Eight equal 250 Mpc cubes. FLAMINGO gas uses native electrons and actual gas velocities;
painted maps use halo bulk velocities. Whole-box views have different depths. Spatial bands are region variation.</p>
<p><a href="output/pdf/FLAMINGO_HalfDome_z0_summary.pdf">Concise PDF</a> ·
<a href="output/pdf/FLAMINGO_HalfDome_z0_detailed_report.pdf">Detailed PDF</a> ·
<a href="comparison_summary.csv">All case statistics (CSV)</a> ·
<a href="parameter_distributions.json">Parameter distributions</a></p>
<label>Filter figures <input id="filter" placeholder="Try dm, deltaT, matrix, parameters, or profiles_bin_10"></label>
</header><main class="grid">__CARDS__</main>
<script>document.getElementById('filter').addEventListener('input',event=>{
const query=event.target.value.toLowerCase();
document.querySelectorAll('article').forEach(card=>card.hidden=!card.dataset.name.toLowerCase().includes(query));
});</script></html>'''.replace('__CARDS__','\n'.join(cards))
    (ROOT/'index.html').write_text(document)


def files_changed():
    source=sorted([p.name for pattern in ['*.py','*.ps1','*.pbs','*.sh'] for p in ROOT.glob(pattern)])
    text='# Files created or edited in this isolated study\n\n'
    text+='No existing calibration code, previous study outputs or raw simulation files were edited.\n\n'
    text+='Latest report-finalization changes:\n\n'
    roles={
        'make_reports.py':'Final report reflow, physical-variable explanations, resource limits and atomic PDF saves.',
        'build_delivery.py':'Missing-accounting handling, one log per submitted job, gallery links and delivery hashes.',
        'run_local_report.ps1':'Windows/WSL memory guard for report and QA stages; separate local logs.',
        'verify_final_reports.py':'Frozen scientific hashes, final PDF hashes/page counts and full-page render sheets.',
        'finish_delivery.py':'Final QA status, cluster-receipt preservation and study-only cleanup.',
        'README.md':'Final scope, accounting definitions and cluster-verification limitations.'}
    text+=''.join('- `'+name+'`: '+role+'\n' for name,role in roles.items())
    text+='\nFull source inventory for the isolated study:\n\n'
    text+='Source files:\n\n'+''.join('- `'+n+'`\n' for n in source)
    text+='\nDocumentation: `README.md`, `FILES_CHANGED.md`.\n'
    text+='\nGenerated deliverables: `index.html`, `figures/`, `output/pdf/`, statistics CSV/JSON and SHA-256 manifests.\n'
    text+='\nExecuted version-1 scientific source remains under `executed_code_v1/`. '
    text+='`aggregate_recovered.py` preserves the plotting source recovered after an interrupted local edit.\n'
    (ROOT/'FILES_CHANGED.md').write_text(text)


def main():
    accounting();tables();gallery();files_changed()
    selected=[]
    for pattern in ['*.py','*.ps1','*.pbs','*.sh','*.json','*.csv','*.html','*.md']:
        selected+=list(ROOT.glob(pattern))
    selected+=list((ROOT/'output/pdf').glob('*.pdf'))+list((ROOT/'figures').glob('*'))
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(selected) if p.name!='delivery_manifest.json'}
    manifest=dict(timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=hashes)
    (ROOT/'delivery_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Delivery manifest:',len(hashes),'files')


if __name__=='__main__':main()
