"""Versioned original-code probes with per-process watchdogs."""
import hashlib,json,os,subprocess,time
from pathlib import Path

ROOT=Path(__file__).resolve().parent
XG=Path('/home/kn18001/.julia/dev/XGPaint')
JULIA='/home/kn18001/.julia/juliaup/julia-1.12.2+0.x64.linux.gnu/bin/julia'
ENV=dict(os.environ,JULIA_DEPOT_PATH='/home/kn18001/.julia',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')

def main():
    provenance={}
    for name in ['profiles.jl','profiles_y.jl']:
        code=subprocess.check_output(['git','show','5dd0b57:src/'+name],cwd=XG)
        (ROOT/'inputs'/('stock_'+name)).write_bytes(code)
        provenance[name]=dict(commit_sha256=hashlib.sha256(code).hexdigest(),
            installed_sha256=hashlib.sha256((XG/'src'/name).read_bytes()).hexdigest())
    # The actual LOS code being called must be byte-identical to the commit.
    assert provenance['profiles_y.jl']['commit_sha256']==provenance['profiles_y.jl']['installed_sha256']
    (ROOT/'results/stock_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    cases=[('parse',),('B12',.01,.497,4.35),('compact_pivot',1e-8,.025,16.),
           ('steep_cache',1e-6,.025*6**.731,16*6**2),
           ('steep_occupied',1e-6,.025*4**.731,16*4**2),
           ('shallow',.1,.497,2.8*10**(-.2)*4**(-.5)),
           ('tail_underflow',10.,.025,80.),('stress_slow',1e-8,.025,40.)]
    status=[]
    for case in cases:
        cmd=[JULIA,'--startup-file=no','--threads=1','--project=/home/kn18001/.julia/environments/v1.12',
             str(ROOT/'stock_probe.jl')]+[str(x) for x in case]
        start=time.monotonic()
        with (ROOT/'results'/('stock_'+case[0]+'.log')).open('w') as log:
            try:
                p=subprocess.run(cmd,env=ENV,stdout=log,stderr=subprocess.STDOUT,timeout=35)
                record=dict(case=case[0],returncode=p.returncode,seconds=time.monotonic()-start)
            except subprocess.TimeoutExpired:
                record=dict(case=case[0],timeout_seconds=35,seconds=time.monotonic()-start)
        status.append(record);print(record,flush=True)
        (ROOT/'results/stock_status.json').write_text(json.dumps(status,indent=2)+'\n')

if __name__=='__main__':main()
