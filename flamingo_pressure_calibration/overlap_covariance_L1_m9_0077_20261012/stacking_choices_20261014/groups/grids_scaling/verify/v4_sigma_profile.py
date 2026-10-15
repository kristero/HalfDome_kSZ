"""Round-4 check (read-only): full per-shell objective-(a) sigma = max(SD/mean, 5%) on the fit grid's strict shells
inside 4 R200c, via the core's stack_bin (same call as assemble_results.py).  Prints, per bin: the profile, the shell
of the minimum, the shell of the maximum, the innermost and outermost shell, the outward rise (max beyond min / min)
and whether the profile is monotone on each side of the minimum."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import gs_common as C
sc = C.sc
e = sc.edges("fit")
summ = {}
for field in ("P", "E"):
    rows = []
    for b in C.calibrated(C.bins()):
        st = sc.stack_bin(sc.make_key(field, grid="fit"), b, "mean", "strict", n_boot=C.N_BOOT)
        m = st["resolved"] & (e[1:] <= 4 + 1e-12) & (st["value"] > 0)
        ii = np.flatnonzero(m)
        s = np.maximum(st["sd"] / st["value"], 0.05)[m]
        i0 = int(np.argmin(s)); imax = int(np.argmax(s))
        inner, outer = s[:i0 + 1], s[i0:]
        mono_in = bool(np.all(np.diff(inner) <= 0)); mono_out = bool(np.all(np.diff(outer) >= 0))
        # number of local reversals on each side
        rev_in = int(np.sum(np.diff(inner) > 0)); rev_out = int(np.sum(np.diff(outer) < 0))
        sh = lambda j: f"{e[ii[j]]:.3f}-{e[ii[j] + 1]:.3f}"
        rows.append(dict(bin=b["label"], n=len(s), min=s[i0], min_shell=sh(i0), max=s[imax], max_shell=sh(imax),
                         inner=s[0], inner_shell=sh(0), outer_last=s[-1], outer_shell=sh(len(s) - 1),
                         outward_rise=outer.max() / s[i0], outer_max=outer.max(), outer_max_shell=sh(i0 + int(np.argmax(outer))),
                         inward_max=inner.max(), inward_max_shell=sh(int(np.argmax(inner))),
                         rev_in=rev_in, rev_out=rev_out, prof=np.round(s, 3)))
    summ[field] = rows
    print(f"== {field}")
    for r in rows:
        print(f"  {r['bin']}: n={r['n']} min {r['min']:.3f} @ {r['min_shell']} | max {r['max']:.3f} @ {r['max_shell']} | "
              f"innermost {r['inner']:.3f} @ {r['inner_shell']} | outermost {r['outer_last']:.3f} @ {r['outer_shell']} | "
              f"outward max {r['outer_max']:.3f} @ {r['outer_max_shell']} rise {r['outward_rise']:.2f} | "
              f"inward max {r['inward_max']:.3f} @ {r['inward_max_shell']} | reversals in/out {r['rev_in']}/{r['rev_out']}")
        print("     ", " ".join(f"{v:.3f}" for v in r["prof"]))
    print(f"  {field}: outward rise {min(r['outward_rise'] for r in rows):.2f}-{max(r['outward_rise'] for r in rows):.2f}; "
          f"innermost {min(r['inner'] for r in rows):.3f}-{max(r['inner'] for r in rows):.3f}; "
          f"min-shell lower edges {sorted(set(r['min_shell'] for r in rows))}")
