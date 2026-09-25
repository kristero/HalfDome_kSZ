"""Fetch only study diagnostics and spectra, never the large map caches."""
import argparse
from pathlib import Path
import subprocess
import tarfile

HERE = Path(__file__).resolve().parent
REMOTE = '/lustre/work/kristero10/tsz_guardrail_study_20260915'
PACK = r'''
from pathlib import Path
import tarfile
root=Path('/lustre/work/kristero10/tsz_guardrail_study_20260915')
files=[]
compact=COMPACT_FETCH
for folder in (['audit','logs'] if compact else ['audit','inputs','plots','logs']):
    files.extend(p for p in (root/folder).rglob('*') if p.is_file()
                 and (not compact or p.suffix in ['.json','.log','.sha256','.csv','.txt']))
for p in (root/'maps').rglob('*'):
    if p.is_file() and p.name in ['status.json','numerics.toml','parameters.toml','columns.csv','operator_probe.toml',
        'masked_clean_cl.npy','unmasked_clean_cl.npy','time.txt','run.log']:
        files.append(p)
with tarfile.open(root/'review_results.tar.gz','w:gz') as archive:
    for p in files:
        archive.add(p,arcname=str(p.relative_to(root)))
print(len(files))
'''


def fetch(destination, compact=False):
    subprocess.run(['ssh','-o','BatchMode=yes','idark','/home/anaconda3/bin/python3','-'],
                   input=PACK.replace('COMPACT_FETCH',repr(compact)),text=True,check=True)
    archive_path=HERE/'review_results.tar.gz'
    subprocess.run(['scp','-q','idark:'+REMOTE+'/review_results.tar.gz',str(archive_path)],check=True)
    destination.mkdir(parents=True,exist_ok=True)
    with tarfile.open(archive_path) as archive:
        for item in archive.getmembers():
            (destination/item.name).resolve().relative_to(destination.resolve())
        archive.extractall(destination,filter='data')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination',type=Path,default=HERE)
    parser.add_argument('--compact',action='store_true',help='Refresh small evidence and spectra after a complete initial fetch; rebuild figures locally.')
    args=parser.parse_args()
    fetch(args.destination,args.compact)
