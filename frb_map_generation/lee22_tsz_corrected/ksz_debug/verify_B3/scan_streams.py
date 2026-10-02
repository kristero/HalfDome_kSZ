"""Independent (no pymupdf) scan of raw PDF content streams: zlib-decompress every stream,
report those that contain the orange 'WebSky Halo' colour operator."""
import re, zlib, sys
fn = sys.argv[1]
raw = open(fn, 'rb').read()
out = []
for m in re.finditer(rb'stream\r?\n', raw):
    s = m.end()
    e = raw.find(b'endstream', s)
    if e < 0: continue
    try:
        d = zlib.decompress(raw[s:e])
    except Exception:
        try:
            d = zlib.decompressobj().decompress(raw[s:e])
        except Exception:
            continue
    if b'0.271' in d or b'.27058' in d or b'1 0.270' in d:
        hdr = raw[max(0, m.start()-300):m.start()]
        out.append((m.start(), len(d), hdr[-200:], d))
for pos, n, hdr, d in out:
    print('stream at', pos, 'len', n, 'hdr', hdr.replace(b'\n', b' ')[-160:])
    i = d.find(b'0.271')
    print('   sample:', d[max(0, i-300):i+600])
