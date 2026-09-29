"""verify_3: Lee22's own TNG stacks (red 'simulation' curves) relative to the cyan Battaglia curves they drew."""
import numpy as np, csv, json
CSV = "/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_normalization_forensics_20260928/B_paper_code_forensics/lee22_fig2_fig5_vector_curves.csv"
d = {}
for r in csv.DictReader(open(CSV)):
    d.setdefault((r["figure"], int(r["panel"]), r["curve"]), []).append((float(r["x_r_over_R200"]), float(r["x3_times_ratio"])))
d = {k: np.array(sorted(v)) for k, v in d.items()}
out = {}
for fig in ["fig2_pressure", "fig5_density"]:
    for p in [0, 1]:
        s, c = d[(fig, p, "simulation")], d[(fig, p, "battaglia")]
        x = s[:, 0]
        ratio = s[:, 1] / np.interp(np.log(x), np.log(c[:, 0]), c[:, 1])
        for lo, hi in [(0.5, 1.0), (0.3, 1.34), (0.04, 0.2)]:
            m = (x >= lo) & (x <= hi)
            out[f"{fig}|panel{p}|sim/cyan median x in [{lo},{hi}]"] = float(np.median(ratio[m]))
for k, v in out.items():
    print(k, round(v, 3))
json.dump(out, open("/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_normalization_forensics_20260928/verify_3/sim_vs_cyan.json", "w"), indent=1)
