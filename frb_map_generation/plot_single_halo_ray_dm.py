"""Two-panel visual of one halo: electron density in the plane of the line of sight, and
rays at several impact parameters coloured by their cumulative DM.

Left: previous projected convention (aperture selection at b <= R200c, LOS to 1e5 R200c).
Right: like-for-like spherical R200c boundary (only the chord inside the sphere counts).
Input: the CSV files written by compute_single_halo_ray_dm.jl.
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm, PowerNorm
from matplotlib.patches import Circle, Rectangle

plt.rcParams.update({"font.size": 12, "axes.titlesize": 13.5, "axes.labelsize": 12.5})

L_SHOWN = 2.5  # R200c units shown along the line of sight (the grid extends further)


def read_halo(path):
    out = {}
    for line in Path(path).read_text().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def draw_ray(ax, l, dm, b, norm, cmap):
    pts = np.column_stack([l, np.full_like(l, b)])
    segs = np.stack([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segs, cmap=cmap, norm=norm, linewidth=5.0, capstyle="butt", zorder=5)
    lc.set_array(0.5 * (dm[:-1] + dm[1:]))
    lc.set_path_effects([pe.Stroke(linewidth=7.2, foreground="white"), pe.Normal()])
    ax.add_collection(lc)
    return lc


def sci(x):
    e = int(np.floor(np.log10(x)))
    return r"{:.2f}\times10^{{{}}}".format(x / 10**e, e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="frb_map_generation/outputs/single_halo_visual_20260917")
    ap.add_argument("--stem", default="single_halo_rays_b16_floor_mass_z0p5")
    args = ap.parse_args()
    root = Path(args.input)
    halo = read_halo(root / "halo.txt")
    mass, z = float(halo["mass_msun"]), float(halo["z"])
    r200c_mpc, theta_arcmin = float(halo["r200c_mpc"]), float(halo["theta200c_arcmin"])
    x_max = float(halo["x_max_r200c"])
    ls = np.loadtxt(root / "grid_l_r200c.csv")
    xs = np.loadtxt(root / "grid_x_r200c.csv")
    grid = np.loadtxt(root / "density_grid_cm3.csv", delimiter=",")
    rays = np.genfromtxt(root / "rays.csv", delimiter=",", names=True)
    totals = {round(float(r["impact_r200c"]), 4): r for r in csv.DictReader(open(root / "ray_totals.csv"))}
    impacts = sorted(totals)

    keep_l = np.abs(ls) <= L_SHOWN + 1e-9
    ls, grid = ls[keep_l], grid[:, keep_l]
    ll, xx = np.meshgrid(ls, xs)
    inside = np.hypot(ll, xx) <= 1.0
    dens_norm = LogNorm(vmin=grid.min() * .7, vmax=np.percentile(grid[inside], 99.5))
    dens_cmap = plt.get_cmap("Greys")
    dm_max = max(float(totals[b]["dm_projected_full_los"]) for b in impacts if totals[b]["credited_projected"] == "true")
    dm_norm = PowerNorm(gamma=0.5, vmin=0.0, vmax=dm_max)
    dm_cmap = plt.get_cmap("plasma")

    # absolute layout in inches: [panel | label column] x 2, then two colour bars
    pw, ph = 6.0, 6.0 * (2 * x_max) / (2 * L_SHOWN)
    lab_w, gap, left, bottom, top = 2.95, 0.5, 0.95, 1.25, 1.25
    cb_w, cb_lab = 0.22, 1.05
    W = left + 2 * (pw + lab_w) + gap + 0.15 + cb_w + cb_lab + 0.3 + cb_w + cb_lab + 0.25
    H = bottom + ph + top
    fig = plt.figure(figsize=(W, H))
    def add(x0, w, sharey=None):
        return fig.add_axes([x0 / W, bottom / H, w / W, ph / H], sharey=sharey)
    ax1 = add(left, pw)
    lab1 = add(left + pw, lab_w, sharey=ax1)
    ax2 = add(left + pw + lab_w + gap, pw)
    lab2 = add(left + 2 * pw + lab_w + gap, lab_w, sharey=ax2)
    cax1 = add(left + 2 * (pw + lab_w) + gap + 0.15, cb_w)
    cax2 = add(left + 2 * (pw + lab_w) + gap + 0.15 + cb_w + cb_lab + 0.3, cb_w)

    titles = [
        "Previous: projected aperture, profile line of sight to $10^5\\,R_{200c}$",
        "Like-for-like with TNG (within $R_{200}$): gas inside the $R_{200c}$ sphere only",
    ]
    columns = ["dm_projected_credited", "dm_sphere"]
    for ax, lab, title, column in zip((ax1, ax2), (lab1, lab2), titles, columns):
        if column == "dm_sphere":
            ax.add_patch(Rectangle((-L_SHOWN, -x_max), 2 * L_SHOWN, 2 * x_max, fc="white", ec=".78", hatch="////",
                                   lw=0, zorder=0.5))
            ax.patch.set_alpha(0)
            shown = np.ma.masked_where(~inside, grid)
        else:
            shown = grid
        mesh = ax.pcolormesh(ls, xs, shown, norm=dens_norm, cmap=dens_cmap, shading="auto", rasterized=True, zorder=1)
        ax.add_patch(Circle((0, 0), 1.0, fill=False, ec="#1f4e79", lw=1.7, ls="--", zorder=4))
        ax.text(0.04, -0.98, r"$R_{200c}$", fontsize=11.5, color="#1f4e79", ha="left", va="top", zorder=6)
        box = dict(fc="white", ec="none", alpha=.85, pad=1.6)
        if column == "dm_projected_credited":
            for sgn in (1, -1):
                ax.axhline(sgn, color="#B2182B", lw=1.4, ls=(0, (5, 3)), zorder=4)
            ax.text(-L_SHOWN + .07, 1.04, "aperture edge $b = R_{200c}$: rays inside are credited with the whole column",
                    color="#B2182B", fontsize=10, ha="left", va="bottom", zorder=6, bbox=box)
            ax.text(-L_SHOWN + .07, -x_max + .06,
                    "gas everywhere along the ray counts, also outside the sphere;\n"
                    "a ray just outside the aperture gets 0 although its column is not (hard edge)",
                    fontsize=10, color=".2", ha="left", va="bottom", zorder=6, bbox=box)
        else:
            ax.text(-L_SHOWN + .07, -x_max + .06,
                    "hatched: gas outside the $R_{200c}$ sphere is not counted;\n"
                    "only the chord inside the sphere contributes, so DM $\\to$ 0 for a grazing ray",
                    fontsize=10, color=".2", ha="left", va="bottom", zorder=6, bbox=box)
        for b in impacts:
            sel = np.isclose(rays["impact_r200c"], b) & (np.abs(rays["l_r200c"]) <= L_SHOWN + 1e-9)
            l, dm = rays["l_r200c"][sel], rays[column][sel]
            draw_ray(ax, l, dm, b, dm_norm, dm_cmap)
            ax.plot(L_SHOWN, b, marker=(3, 0, -90), ms=10, color=dm_cmap(dm_norm(dm[-1])), mec="white", mew=.8,
                    clip_on=False, zorder=7)
            t = totals[b]
            if column == "dm_projected_credited":
                if t["credited_projected"] == "true":
                    label = "$b={:.2f}$:  DM = {:.1f}".format(b, float(t["dm_projected_full_los"]))
                    if b == 1.0:
                        label += "  (grazing: floor)"
                else:
                    label = "$b={:.2f}$:  DM = 0  (outside aperture)".format(b)
            else:
                label = "$b={:.2f}$:  DM = {:.1f}".format(b, float(t["dm_sphere"]))
                if b == 1.0:
                    label += "  (grazing: chord 0)"
                elif b > 1.0:
                    label += "  (misses the sphere)"
            lab.text(.05, b, label, fontsize=9.3, va="center", ha="left", zorder=7)
        lab.axis("off")
        lab.set_xlim(0, 1)
        ax.set_xlim(-L_SHOWN, L_SHOWN)
        ax.set_ylim(-x_max, x_max)
        ax.set_xlabel(r"distance along the line of sight  $l/R_{200c}$   (source $\longrightarrow$ observer)")
        ax.set_ylabel(r"impact parameter  $b/R_{200c}$")
        ax.set_title(title, pad=9)
    cb1 = fig.colorbar(mesh, cax=cax1)
    cb1.set_label(r"electron density $n_e$ in the plane of the rays  [cm$^{-3}$]")
    sm = plt.cm.ScalarMappable(norm=dm_norm, cmap=dm_cmap)
    ticks = [t for t in (0, 2, 5, 10, 20, 50, 100, 200, 400) if t < 0.97 * dm_max] + [dm_max]
    cb2 = fig.colorbar(sm, cax=cax2, ticks=ticks)
    cb2.ax.set_yticklabels(["{:g}".format(t) if t != dm_max else "{:.0f}".format(t) for t in ticks])
    cb2.set_label(r"cumulative DM along the ray, source $\to$ observer  [pc cm$^{-3}$]")
    fig.suptitle("One halo, Battaglia16 profile:  $M_{{200c}} = {}\\,M_\\odot$ (HalfDome catalogue floor),  $z = {:.1f}$,  "
                 "$R_{{200c}} = {:.3f}$ Mpc,  $\\theta_{{200c}} = {:.2f}$ arcmin".format(sci(mass), z, r200c_mpc, theta_arcmin),
                 y=(H - 0.35) / H, fontsize=15)
    fig.text(left / W, 0.12 / H,
             "Ray colour: DM accumulated from the source side up to that point (observer frame), same colour scale in both panels. "
             "Totals in the label columns are the full line-of-sight values;\nin the left panel the column keeps growing beyond the panel "
             "(1.4 pc cm$^{-3}$ of each total lies outside $|l| < 3\\,R_{200c}$). Same halo, same rays, same profile in both panels; "
             "only the gas that is counted differs.",
             fontsize=10.2, color=".3", ha="left", va="bottom", linespacing=1.35)
    for ext in ("png", "svg", "pdf"):
        fig.savefig(root / "{}.{}".format(args.stem, ext), dpi=200 if ext == "png" else None, bbox_inches="tight", pad_inches=.2)
    print("Saved", root / (args.stem + ".png"))


if __name__ == "__main__":
    main()
