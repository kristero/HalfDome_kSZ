"""Copy small STRW execution receipts to idark without moving cached grids."""
import json
from cluster_control import ROOT, IDARK_ROOT, remote

path = ROOT / 'strw_execution_metadata.json'
metadata = json.loads(path.read_text(encoding='utf-8-sig'))
path.write_text(json.dumps(metadata, indent=2) + '\n')
program = f'''
import pathlib, json
root = pathlib.Path({IDARK_ROOT!r})
metadata = {metadata!r}
(root/'compact_complete.json').write_text(json.dumps(metadata['measured_flamingo_compact_export.json'],indent=2))
(root/'halo_union_complete.json').write_text(json.dumps(metadata['measured_halo_union_compact_export.json'],indent=2))
(root/'strw_execution_metadata.json').write_text(json.dumps(metadata,indent=2))
print('STRW receipts copied')
'''
print(remote('idark', program).decode())
