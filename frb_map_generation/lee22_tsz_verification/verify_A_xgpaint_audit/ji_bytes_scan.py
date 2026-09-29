"""Verifier: scan XGPaint precompile .ji files for (fsize UInt64, crc32c UInt32) records of profiles.jl / profiles_y.jl
versions, by raw byte search (independent of Julia's cache-header parser)."""
import glob, os, struct, datetime
pats = {"profiles.jl HEAD/ce1d239": (26851, 0x8cfcf4e7), "profiles.jl 5dd0b57": (26854, 0x9468862b),
        "profiles.jl 07fc686": (26822, 0xd880011d), "profiles.jl e67a8a1": (26574, 0x85abdfe0),
        "profiles_y.jl HEAD": (8708, 0x2dca2a56), "profiles_y.jl e67a8a1": (8708, 0x9180b078)}
files = []
for d in ["/home/kn18001/.julia", "/home/cbllover/HalfDome/frb_map_generation/outputs/tsz_frb_1m_z1_r200c_apertures/.julia_depot",
          "/root/.julia"]:
    files += glob.glob(f"{d}/compiled/*/XGPaint/*.ji")
for f in sorted(files):
    b = open(f, "rb").read()
    hits = [k for k, (n, c) in pats.items() if struct.pack("<QI", n, c) in b]
    mt = datetime.datetime.fromtimestamp(os.path.getmtime(f)).strftime("%Y-%m-%d %H:%M")
    print(f"{mt} {f}: {hits}")
