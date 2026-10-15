"""Read-only metadata listing of the COSMA FLAMINGO data service (hdfstream): which products exist per run."""
import json, time, sys
import hdfstream
t0 = time.time()
root = hdfstream.open('cosma', '/')
out = {}
def ls(path):
    try:
        g = root[path]
        return sorted(list(g.keys()))
    except Exception as e:
        return 'ERR ' + repr(e)[:200]
out['FLAMINGO'] = ls('FLAMINGO')
for box in ['L1_m9', 'L1_m8', 'L2p8_m9']:
    out[box] = ls(f'FLAMINGO/{box}')
    out[box + '/' + box] = ls(f'FLAMINGO/{box}/{box}')
out['L1_m9/snapshots'] = ls('FLAMINGO/L1_m9/L1_m9/snapshots')
out['L1_m9/SOAP-HBT'] = ls('FLAMINGO/L1_m9/L1_m9/SOAP-HBT')
out['elapsed_s'] = time.time() - t0
json.dump(out, open('cosma_listing.json', 'w'), indent=1)
for k, v in out.items():
    s = json.dumps(v)
    print(k, len(v) if isinstance(v, list) else '', s[:600])
