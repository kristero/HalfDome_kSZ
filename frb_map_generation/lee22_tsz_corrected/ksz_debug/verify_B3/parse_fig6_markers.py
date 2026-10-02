"""Independent extraction of Stein+2020 Fig. 6 marker positions from the raw (zlib-decoded) PDF
content stream, without pymupdf: accumulate the relative 'cm' translations of each marker block.
Axis calibration from tick-mark positions and their text labels in the same stream."""
import re, sys, numpy as np
d = open(sys.argv[1], 'rb').read().decode('latin1')
outcsv = sys.argv[2]
lines = d.split('\n')
# --- locate marker blocks: a colour-setting 'Q q ... re W n /A6 gs <rgb> rg' preamble followed by 'q' and 'cm /Mk Do' lines
blocks = []
i = 0
cur_col = None
txt = d
# find every 'q\n' that starts a marker block and record the colour set just before it
for m in re.finditer(r'((?:[0-9.]+\s+){3})RG\s+((?:[0-9.]+\s+){3})rg\s*\nq\n((?:1 0 0 1 [-0-9.]+ [-0-9.]+ cm /M\d+ Do\n)+)Q', txt):
    col = tuple(round(float(v), 3) for v in m.group(2).split())
    tr = re.findall(r'1 0 0 1 ([-0-9.]+) ([-0-9.]+) cm /M\d+ Do', m.group(3))
    xy = np.cumsum(np.array(tr, float), axis=0)
    blocks.append((col, xy))
for col, xy in blocks:
    print('block colour', col, 'n markers', len(xy), 'x range', xy[:, 0].min(), xy[:, 0].max())
# --- calibration
xt = {100: 109.649063, 300: 278.216757, 1000: 462.950645, 3000: 631.51834}   # major tick 'm' x with labels (100),(300),(1000),(3000)
yt = {0.5: 98.365421, 1.0: 163.597421, 1.5: 228.829421, 2.0: 294.061421, 2.5: 359.293421}
lx = np.log10(list(xt)); px = np.array(list(xt.values()))
ax_, bx_ = np.polyfit(lx, px, 1)
print('x calib: slope %.6f  intercept %.6f  resid' % (ax_, bx_), px - (ax_*lx + bx_))
yv = np.array(list(yt)); py = np.array(list(yt.values()))
ay_, by_ = np.polyfit(yv, py, 1)
print('y calib: slope %.6f  intercept %.6f  resid' % (ay_, by_), py - (ay_*yv + by_))
# minor ticks check: 200 at 216.003437, 2000 at 569.305019, 5000 at 709.897854
for l, p in ((200, 216.003437), (400, 322.35781), (2000, 569.305019), (5000, 709.897854), (70, 54.921955*0 + 75.410602)):
    print('  minor tick ell=%d predicted page x %.4f actual %.4f' % (l, ax_*np.log10(l) + bx_, p))
axis_bottom = 46.179820546; axis_top = 46.179820546 + 326.16
print('axis D range: %.5f .. %.5f' % ((axis_bottom - by_)/ay_, (axis_top - by_)/ay_))
print('axis ell range: %.3f .. %.3f' % (10**((54.9219551438 - bx_)/ax_), 10**((54.9219551438 + 669.6 - bx_)/ax_)))
names = {(1.0, 0.271, 0.0): 'halo', (0.122, 0.467, 0.706): 'field', (0.184, 0.31, 0.31): 'total'}
res = {}
for col, xy in blocks:
    nm = names.get(col, str(col))
    ell = 10**((xy[:, 0] - bx_)/ax_); D = (xy[:, 1] - by_)/ay_
    res[nm] = (ell, D)
    print(nm, 'ell first', np.round(ell[:5], 3), 'last', np.round(ell[-3:], 2), 'n', len(ell),
          'max|ell-round| (ell<300)', np.abs(ell - np.round(ell))[ell < 300].max())
ell, D = res['halo']
np.savetxt(outcsv, np.c_[ell, D], delimiter=',', header='ell,D_ell_uK2 (independent raw-stream extraction, arXiv PDF p23)')
o = np.argsort(ell); ell, D = ell[o], D[o]
for l in (100, 200, 300, 400, 500, 503, 518, 551, 560, 600, 700, 1000, 2000, 3000, 5000):
    k = np.argmin(abs(ell - l))
    # local mean over +-5% in ell to beat the marker scatter
    w = (ell > l/1.05) & (ell < l*1.05)
    print('ell=%5d  nearest-marker D=%.4f (ell %.1f)   mean D in +-5%%: %.4f (n=%d)   median %.4f' % (l, D[k], ell[k], D[w].mean(), w.sum(), np.median(D[w])))
# where does the halo curve cross the axis floor?
lo = D < (axis_bottom - by_)/ay_
print('last ell with D below axis floor:', ell[lo].max() if lo.any() else None, ' n below floor', lo.sum())
# smooth (running mean in log-ell) crossing
from numpy.lib.stride_tricks import sliding_window_view
sm = np.convolve(D, np.ones(21)/21, mode='same')
fl = (axis_bottom - by_)/ay_
idx = np.where((sm[:-1] < fl) & (sm[1:] >= fl))[0]
print('smoothed curve crosses floor at ell', ell[idx])
