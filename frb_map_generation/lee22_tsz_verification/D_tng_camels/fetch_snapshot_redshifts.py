#!/usr/bin/env python
"""Exact header redshifts of the CAMELS IllustrisTNG CV group catalogues (new numbering 044-090 <-> old 010-033),
read over HTTP with Range requests (h5py on a file-like object; only a few kB per file are transferred).
The illstack red_dict redshifts are rounded / from another simulation (e.g. 1.04838 vs CV_0 header 1.04760).
Output: <OUTDIR>/snapshot_redshifts.csv (box, old, new, z_header)
"""
import io
import os
import sys
import urllib.request

import h5py

import camels_common as cc

BASE = "https://users.flatironinstitute.org/~camels/Sims/IllustrisTNG/CV"


class HTTPRangeFile(io.RawIOBase):
    def __init__(self, url, block=1 << 16):
        self.url, self.block, self.pos, self.cache = url, block, 0, {}
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=60) as r:
            self.size = int(r.headers["Content-Length"])

    def readable(self):
        return True

    def seekable(self):
        return True

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos

    def tell(self):
        return self.pos

    def _blk(self, i):
        if i not in self.cache:
            lo = i * self.block
            hi = min(self.size, lo + self.block) - 1
            req = urllib.request.Request(self.url, headers={"Range": f"bytes={lo}-{hi}"})
            with urllib.request.urlopen(req, timeout=60) as r:
                self.cache[i] = r.read()
        return self.cache[i]

    def readinto(self, b):
        n = min(len(b), self.size - self.pos)
        out = bytearray()
        while len(out) < n:
            i = (self.pos + len(out)) // self.block
            off = (self.pos + len(out)) - i * self.block
            out += self._blk(i)[off:off + n - len(out)]
        b[:n] = out
        self.pos += n
        return n


def header_z(box, new):
    f = HTTPRangeFile(f"{BASE}/CV_{box}/groups_{new:03d}.hdf5")
    with h5py.File(f, "r") as h:
        return float(h["Header"].attrs["Redshift"]), float(h["Header"].attrs["Time"])


if __name__ == "__main__":
    boxes = range(27) if len(sys.argv) < 2 else [int(b) for b in sys.argv[1:]]
    out = os.path.join(cc.OUTDIR, "snapshot_redshifts.csv")
    rows = ["box,old,new,z_header,a_header,z_illstack_reddict"]
    for b in boxes:
        for old in range(10, 34):
            new = cc.old_to_new(old)
            z, a = header_z(b, new)
            rows.append(f"{b},{old},{new},{z:.6f},{a:.6f},{cc.OLD_SNAP_Z[old]}")
        print(f"CV_{b} done", flush=True)
    with open(out, "w") as fh:
        fh.write("\n".join(rows) + "\n")
