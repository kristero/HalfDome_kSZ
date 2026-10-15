"""Refine axes/ticks only, using completed cached analysis products."""
import functools
import hashlib
import json
import time
import numpy as np
from matplotlib.colors import SymLogNorm
from matplotlib.ticker import FuncFormatter
import aggregate_and_plot as A

original_save=A.save
A.SymLogNorm=functools.partial(SymLogNorm,base=10)


def scientific_tick(value,position):
    if value==0:return '$0$'
    exponent=int(round(np.log10(abs(value))))
    return ('$-' if value<0 else '$')+'10^{%d}$'%exponent


def save_refined(fig,name,vector=True):
    if name.startswith('profiles_bin_'):
        # Matplotlib autoscaling includes points hidden by xlim. Scale only
        # to residuals within the range the reader can actually see.
        for ax in fig.axes[2:4]:
            values=[]
            for line in ax.lines:
                x=np.asarray(line.get_xdata(),float)
                y=np.asarray(line.get_ydata(),float)
                good=(x>=.035)&(x<=4.05)&np.isfinite(y)
                values.extend(y[good])
            if values:
                low=min(0.,min(values));high=max(0.,max(values))
                pad=max(.07*(high-low),1e-12)
                ax.set_ylim(low-pad,high+pad)
    for ax in fig.axes:
        bar=getattr(ax,'_colorbar',None)
        if bar is None or not isinstance(bar.norm,SymLogNorm):continue
        limit=max(abs(bar.norm.vmin),abs(bar.norm.vmax))
        threshold=bar.norm.linthresh
        powers=10.**np.arange(int(np.ceil(np.log10(threshold*5))),
                             int(np.floor(np.log10(limit)))+1)
        if len(powers)>3:powers=powers[-3:]
        bar.set_ticks(np.r_[-powers[::-1],0.,powers])
        bar.formatter=FuncFormatter(scientific_tick)
        bar.update_ticks()
        bar.ax.minorticks_off()
    original_save(fig,name,vector)


def main():
    start=time.time()
    complete=json.loads((A.ROOT/'cluster_analysis_complete.json').read_text())
    assert complete['completed']
    A.save=save_refined
    A.map_figures()
    A.profile_figures(json.loads((A.ROOT/'fits.json').read_text()))
    A.overview_figures()
    complete['figures']={str(p.relative_to(A.ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in A.FIG.glob('*')}
    A.P.save_json(A.ROOT/'cluster_analysis_complete.json',complete)
    A.P.save_json(A.ROOT/'figure_refinement_complete.json',dict(completed=True,
        wall_seconds=time.time()-start,science_data_changed=False,
        changes=['Residual axis limits use visible radii','Base-10 symmetric-log colour scales with sparse ticks'],
        source_sha256=hashlib.sha256(__import__('pathlib').Path(__file__).read_bytes()).hexdigest()))


if __name__=='__main__':main()
