"""Verifier: pure-Python CRC32c (Castagnoli) of XGPaint src files at several commits and the working tree."""
import subprocess, sys
POLY = 0x82F63B78
TABLE = []
for i in range(256):
    c = i
    for _ in range(8):
        c = (c >> 1) ^ POLY if c & 1 else c >> 1
    TABLE.append(c)
def crc32c(b):
    c = 0xFFFFFFFF
    for x in b:
        c = TABLE[(c ^ x) & 0xFF] ^ (c >> 8)
    return c ^ 0xFFFFFFFF
assert crc32c(b"123456789") == 0xE3069283
R = "/home/kn18001/.julia/dev/XGPaint"
for f in ["src/profiles.jl", "src/profiles_y.jl", "src/model.jl", "src/XGPaint.jl"]:
    for c in ["e67a8a1", "ce1d239", "5dd0b57", "07fc686", "35f79ad", "HEAD"]:
        b = subprocess.run(["git", "-C", R, "show", f"{c}:{f}"], capture_output=True).stdout
        print(f"{f:20s} {c:8s} {len(b):6d} B crc32c {crc32c(b):08x}")
    b = open(f"{R}/{f}", "rb").read()
    print(f"{f:20s} worktree {len(b):6d} B crc32c {crc32c(b):08x}")
