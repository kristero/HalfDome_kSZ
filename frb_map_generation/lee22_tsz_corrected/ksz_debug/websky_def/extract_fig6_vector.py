"""Extract the exact vector paths of Stein et al. 2020 (JCAP version) Fig. 6 curves.
Calibrates axes from the tick-mark line segments in the PDF itself."""
import numpy as np, pymupdf as fitz, sys
OUT = "/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_ksz_debug_20261002/websky_def/"
doc = fitz.open('/home/cbllover/HalfDome/tmp/pdfs/stein2020_iop.pdf')
p = doc[25]
dr = p.get_drawings()
# axis calibration from tick marks (major x ticks length 3.28, y ticks at x=134.19)
xt = {}; yt = []
for d in dr:
    if d.get('color') is None: continue
    if tuple(round(c, 3) for c in d['color']) != (0.0, 0.0, 0.0): continue
    if len(d['items']) != 1 or d['items'][0][0] != 'l': continue
    a, b = d['items'][0][1], d['items'][0][2]
    if abs(a.x - b.x) < 1e-3 and abs(a.y - 264.328) < 0.01 and abs(abs(b.y - a.y) - 3.283) < 0.01:
        xt[round(a.x, 3)] = 1
    if abs(a.y - b.y) < 1e-3 and abs(a.x - 134.187) < 0.01 and abs(abs(b.x - a.x) - 3.283) < 0.01:
        yt.append(a.y)
xt = sorted(xt); yt = sorted(set(round(y, 4) for y in yt))
print("major x ticks", xt, "y ticks", yt)
lx = np.log10([100, 300, 1000, 3000]); A = np.vstack([lx, np.ones(4)]).T
sx, x0 = np.linalg.lstsq(A, np.array(xt), rcond=None)[0]
yv = np.array([2.5, 2.0, 1.5, 1.0, 0.5])  # y ticks sorted by increasing page-y
sy, y0 = np.polyfit(yv, np.array(yt), 1)
print("x: page = %.4f*log10(l) + %.4f ; resid" % (sx, x0), np.array(xt) - (sx*lx + x0))
print("y: page = %.4f*D + %.4f ; resid" % (sy, y0), np.array(yt) - (sy*yv + y0))
print("axis ell range", 10**((134.187 - x0)/sx), 10**((500.579 - x0)/sx),
      "D range", (264.328 - y0)/sy, (85.86 - y0)/sy)
cols = {'halo': (1.0, 0.271, 0.0), 'field': (0.122, 0.467, 0.706), 'total': (0.184, 0.31, 0.31)}
res = {}
for name, col in cols.items():
    pts = []
    for d in dr:
        c = d.get('color')
        if c is None or tuple(round(x, 3) for x in c) != col: continue
        r = d['rect']; pts.append(((r.x0 + r.x1)/2, (r.y0 + r.y1)/2))
    pts = np.array(pts)
    ell = 10**((pts[:, 0] - x0)/sx); D = (pts[:, 1] - y0)/sy
    o = np.argsort(ell); ell, D = ell[o], D[o]
    # keep inside axes
    m = np.ones(len(ell), bool)  # matplotlib wrote every marker (clipped only on display): keep all
    print(name, "n markers", len(ell), "first ells", np.round(ell[:6], 3), "last", np.round(ell[-4:], 2),
          "max |ell-round(ell)| for ell<200:", np.abs(ell - np.round(ell))[ell < 200].max())
    res[name] = (ell[m], D[m])
    print(name, len(ell), "inside", m.sum(), "ell range", ell[m].min(), ell[m].max())
    np.savetxt(OUT + f"stein2020_fig6_vector_{name}.csv", np.c_[ell[m], D[m]], delimiter=",",
               header=f"ell,D_ell_uK2  Stein+2020 JCAP Fig.6 '{name}' exact PDF vector vertices (clipped to axes)")
for name in res:
    ell, D = res[name]
    q = [100, 200, 300, 500, 700, 1000, 2000, 3000, 4000, 5000, 5400]
    print(name, " ".join("l=%d:%.4f" % (l, np.interp(np.log(l), np.log(ell), D, left=np.nan, right=np.nan)) for l in q))
# markers: Battaglia et al. 2010 red triangle (marker centre)
for d in dr:
    c = d.get('color')
    if c is not None and tuple(round(x, 3) for x in c) == (0.839, 0.153, 0.157) and d['rect'].x0 > 0 and d['rect'].y1 < 264:
        r = d['rect']; cx = (r.x0 + r.x1)/2; cy = (r.y0 + r.y1)/2
        print("red triangle rect", r, "centre ell=%.1f D=%.4f" % (10**((cx - x0)/sx), (cy - y0)/sy),
              "(triangle centroid y=%.4f)" % ((r.y0 + 2*(r.y1-r.y0)/3*0 + (r.y0+r.y0+r.y1)/3 - r.y0 - y0)/sy))
for d in dr:  # black markers
    c = d.get('color'); r = d['rect']
    if c is not None and tuple(round(x, 3) for x in c) == (0.0, 0.0, 0.0) and 140 < r.x0 < 500 and 85 < r.y0 < 262 and r.width < 10 and r.height < 10 and r.width > 3:
        cx = (r.x0 + r.x1)/2; cy = (r.y0 + r.y1)/2
        print("black marker", len(d['items']), "ell=%.1f D=%.4f" % (10**((cx - x0)/sx), (cy - y0)/sy))
