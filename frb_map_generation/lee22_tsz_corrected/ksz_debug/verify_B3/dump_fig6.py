import re, zlib, sys
fn, pos = sys.argv[1], int(sys.argv[2])
raw = open(fn, 'rb').read()
m = re.compile(rb'stream\r?\n').match(raw, pos)
s = m.end(); e = raw.find(b'endstream', s)
d = zlib.decompressobj().decompress(raw[s:e])
open(sys.argv[3], 'wb').write(d)
print(raw[pos-1500:pos].decode('latin1'))
