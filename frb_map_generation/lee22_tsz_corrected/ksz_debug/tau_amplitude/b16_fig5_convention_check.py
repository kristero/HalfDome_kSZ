"""Which B16 density convention reproduces Battaglia (2016) Fig. 5 (rho x^2 / (f_b rho_crit), AGN feedback)?
Vector curves are read from the arXiv source figures (density_fit_m.pdf / density_fit_z.pdf converted to SVG with pdftocairo)
and compared with the Table 2 global fit evaluated with exponent -(beta+gamma)/alpha (XGPaint, CLASS-SZ) and with the printed
eq. (A1) exponent -(beta-gamma)/alpha. Prints the median ratio formula/figure over 0.1 < x < 1.5 for each curve."""
import re, sys
import numpy as np

SVG_DIR = sys.argv[1] if len(sys.argv) > 1 else "/tmp/claude-0/-home-cbllover-HalfDome/015746ae-94b7-42cb-8dc5-0e8c33dc8490/scratchpad/tauamp"
# axis calibration in the path coordinate frame (tick paths): x=0.1 at 136.556607, x=1 at 390.632058; y=10 at 73.199702, y=100 at 519.59868
X0, X1 = 136.556607, 390.632058
Y0, Y1 = 73.199702, 519.59868


def curves(svg):
    txt = open(svg).read()
    out = []
    for m in re.finditer(r'<path style="([^"]*)" d="([^"]*)"', txt):
        st, d = m.group(1), m.group(2)
        col = re.search(r"stroke:rgb\(([^)]*)\)", st)
        if not col or col.group(1) == "0%,0%,0%" or "stroke-width:3" not in st:
            continue
        nums = np.array([float(v) for v in re.findall(r"-?\d+\.?\d*", d)])
        pts = nums.reshape(-1, 2)
        x = 10 ** (-1 + (pts[:, 0] - X0) / (X1 - X0))
        y = 10 ** (1 + (pts[:, 1] - Y0) / (Y1 - Y0))
        out.append((col.group(1), "dash" in st, x, y))
    return out


def formula(x, M, z, sign):
    m, zp = M / 1e14, 1 + z
    rho0 = 4000 * m ** 0.29 * zp ** -0.66
    al = 0.88 * m ** -0.03 * zp ** 0.19
    be = 3.83 * m ** 0.04 * zp ** -0.025
    xc, ga = 0.5, -0.2
    u = x / xc
    return rho0 * u ** ga * (1 + u ** al) ** (-(be + sign * ga) / al) * x * x


def compare(svg, labels):
    cs = curves(svg)
    print(f"\n{svg.split('/')[-1]}: {len(cs)} coloured curves")
    for (col, dashed, x, y), (lab, M, z) in zip(cs, labels):
        sel = (x > 0.1) & (x < 1.5)
        r_plus = np.median(formula(x[sel], M, z, +1) / y[sel])
        r_minus = np.median(formula(x[sel], M, z, -1) / y[sel])
        pk = y[sel].max()
        print(f"  {lab:28s} {'fit ' if dashed else 'data'}  peak {pk:5.1f} at x={x[sel][np.argmax(y[sel])]:.2f}  "
              f"Table2/(fig): (beta+gamma) {r_plus:.3f}   (beta-gamma, printed) {r_minus:.3f}")


# z panel: legend order red z=0, green z=0.5, blue z=1 (solid data then dashed fit for each); mass of the stack ~ 1.4e14 assumed
zl = []
for lab, z in (("z=0.0", 0.0), ("z=0.5", 0.5), ("z=1.0", 1.0)):
    zl += [(lab + " M=1.4e14", 1.4e14, z)] * 2
compare(f"{SVG_DIR}/z.svg", zl)
# mass panel (z = 0 assumed): bins in legend order, geometric-mean mass
ml = []
for lo, hi in ((7e13, 8e13), (8e13, 9e13), (9e13, 1e14), (1e14, 2e14), (2e14, 3e14), (3e14, 2e15)):
    M = np.sqrt(lo * hi) if hi < 1e15 else 4.5e14
    ml += [(f"M={lo:.0e}-{hi:.0e}", M, 0.0)] * 2
compare(f"{SVG_DIR}/m.svg", ml)
