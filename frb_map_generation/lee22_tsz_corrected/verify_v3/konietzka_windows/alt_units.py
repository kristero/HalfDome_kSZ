"""Hit fractions of the decade [1e14,1e15) under alternative edge scalings a (window [a 1e14, a 1e15) in M200c Msun),
x 0.944 TNG300 abundance, vs TNG's H10 decade (10to14 - 10to13): 2.011 / 8.306 / 11.726 %."""
import numpy as np
import hmf_lib as H
obs = {0.1: 2.011, 0.5: 8.306, 1.0: 11.726}
for lab, a in (("nominal a=1", 1.0), ("Msun/h edges (a=1/h)", 1 / H.h), ("a=h", H.h), ("M200m edges (a~0.75)", 0.75), ("a=0.9 (c=9 both edges)", 0.9)):
    row = []
    for z_s in (0.1, 0.5, 1.0):
        nb = sum(H.window(z, a * 1e14, a * 1e15, 200)[1] * (b - aa) for z, aa, b in H.intervals(z_s, "z"))
        hit = 100 * (1 - np.exp(-0.944 * nb))
        row.append(f"{hit:.2f}% (obs/pred {obs[z_s] / hit:.2f})")
    print(f"{lab:28s}", "  ".join(row))
