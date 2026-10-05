"""Julia 1.12 (tSZ runtime) vs Julia 1.6 runs of the NEW kSZ painter on the idark speed-test catalogues."""
import sys, numpy as np, healpy as hp
A = "/lustre/work/kristero10/frb_data/lee22_ksz_speedtest112_20261002/runs"; B = "/lustre/work/kristero10/frb_data/lee22_ksz_speedtest_20261002/runs"
V = ("b16_phys", "lee22corr_phys", "lee22pipe_phys", "b16_websky", "lee22corr_websky")
for tag, case, ns in (("new112", "lowz", 1024), ("new112", "mid", 4096), ("new16", "lowz", 1024)):
    for v in V:
        a = hp.read_map("%s/%s_%s/maps/halfdome_ksz_%s_nside%d.fits" % (A, tag, case, v, ns), dtype=np.float64)
        b = hp.read_map("%s/new_%s/maps/halfdome_ksz_%s_nside%d.fits" % (B, case, v, ns), dtype=np.float64)
        d = a - b
        print("%-6s %-5s %-17s max|d|/max|ref| %.3e  relL2 %.3e  sum ratio-1 %+.3e" % (tag, case, v, np.abs(d).max() / np.abs(b).max(),
              np.sqrt((d ** 2).sum() / (b ** 2).sum()), a.sum() / b.sum() - 1), flush=True)
