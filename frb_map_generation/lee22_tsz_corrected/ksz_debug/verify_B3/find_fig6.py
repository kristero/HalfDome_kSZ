import re, zlib, sys
fn = sys.argv[1]
raw = open(fn, 'rb').read()
for m in re.finditer(rb'stream\r?\n', raw):
    s = m.end(); e = raw.find(b'endstream', s)
    if e < 0: continue
    try: d = zlib.decompressobj().decompress(raw[s:e])
    except Exception: continue
    if b'/M0 Do' not in d: continue
    cols = re.findall(rb'([0-9.]+ [0-9.]+ [0-9.]+) (rg|RG)', d)
    from collections import Counter
    c = Counter(x[0] for x in cols)
    print('stream', m.start(), 'len', len(d), 'nDo', d.count(b' Do'), 'colours', c.most_common(8))
