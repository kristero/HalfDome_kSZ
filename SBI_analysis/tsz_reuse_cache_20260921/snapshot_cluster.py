"""Read-only scheduler/thread snapshot; retain evidence of load imbalance."""
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parent
remote=r'''
import json,re,subprocess
jobs=['598554.idark','598555.idark','598556.idark','598557.idark']
records=[]
for job in jobs:
    result=subprocess.run(['qstat','-f',job],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=10)
    fields={}
    for line in result.stdout.splitlines():
        m=re.match(r'\s*(job_state|exec_host|resources_used\.[a-z]+)\s*=\s*(.*)',line)
        if m:fields[m[1]]=m[2]
    records.append(dict(job=job,fields=fields))
threads=[]
for host in sorted({r['fields'].get('exec_host','').split('/')[0] for r in records}-{''}):
    result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=5',host,
        'ps','-u','kristero10','-L','-o','pid,tid,pcpu,stat,etime,comm'],
        stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=15)
    rows=[]
    for line in result.stdout.splitlines()[1:]:
        pieces=line.split()
        if len(pieces)==6 and 'julia' in pieces[-1]:
            rows.append(dict(pid=int(pieces[0]),tid=int(pieces[1]),cpu_percent=float(pieces[2]),
                             state=pieces[3],elapsed=pieces[4]))
    threads.append(dict(host=host,returncode=result.returncode,threads=rows,error=result.stderr.strip()))
print(json.dumps(dict(jobs=records,nodes=threads)))
'''
result=subprocess.run(['ssh','idark','/home/anaconda3/bin/python3','-'],input=remote,
                      text=True,capture_output=True,check=True)
data=json.loads(result.stdout);data['utc']=datetime.now(timezone.utc).isoformat()
(ROOT/'results/cluster_snapshot.json').write_text(json.dumps(data,indent=2)+'\n')
for node in data['nodes']:
    threads=node['threads']
    print(node['host'],'Julia threads',len(threads),'running',sum(t['state'].startswith('R') for t in threads),
          'highest cumulative CPU',sorted([t['cpu_percent'] for t in threads],reverse=True)[:8],node['error'])
