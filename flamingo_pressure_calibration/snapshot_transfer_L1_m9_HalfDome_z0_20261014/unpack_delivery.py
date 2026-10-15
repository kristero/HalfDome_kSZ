"""Stream-verify and safely unpack the compact cluster delivery."""
from pathlib import Path
import hashlib
import json
import tarfile

root=Path(__file__).resolve().parent


def sha256(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


receipt=json.loads((root/'delivery_bundle_receipt.json').read_text())
archive=root/receipt['file']
assert archive.stat().st_size==receipt['bytes']
assert sha256(archive)==receipt['sha256']
with tarfile.open(archive,'r:gz') as tar:
    for member in tar.getmembers():
        target=(root/member.name).resolve()
        assert str(target).startswith(str(root.resolve())+'/'),member.name
        assert member.isfile() and not member.issym(),member.name
    tar.extractall(root)
# Older receipts inside an archive must not replace its external checksum.
(root/'delivery_bundle_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
complete=json.loads((root/'cluster_analysis_complete.json').read_text())
for name,value in complete['figures'].items():assert sha256(root/name)==value,name
reports=root/'reports_complete.json'
if reports.exists():
    for record in json.loads(reports.read_text())['outputs']:
        assert sha256(root/record['file'])==record['sha256']
print('Verified bundle,',len(complete['figures']),'figure files and report hashes')
