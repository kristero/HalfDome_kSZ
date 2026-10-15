"""Print the per-bin numbers behind every table and range in SECTION.md.

Read-only: it loads results.json from this folder and prints per-bin values
for the 8 main bins (13.5-15.1) and the two top bins (15.1-15.3, 15.3-15.5),
so that every range quoted in the text can be checked against the bin that
produces it.  Usage:  python -B section_check.py [part ...]
parts: ratios boot diag impact all (default all)
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, "results.json")))
BANDS = R["bands_R200c"]
MAIN = ["13.5-13.7", "13.7-13.9", "13.9-14.1", "14.1-14.3", "14.3-14.5",
        "14.5-14.7", "14.7-14.9", "14.9-15.1"]
TOP = ["15.1-15.3", "15.3-15.5"]


def fmt(v):
    if v is None:
        return "   -   "
    if isinstance(v, bool):
        return str(v)
    a = abs(v)
    if a >= 100:
        return "%7.1f" % v
    if a >= 10:
        return "%7.2f" % v
    return "%7.4f" % v


def rng(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "-"
    return "%s..%s" % (fmt(min(vals)).strip(), fmt(max(vals)).strip())


def table(getter, title, bands=BANDS):
    print("\n## " + title)
    print("%-9s" % "band" + "".join("%9s" % b[:-2] for b in MAIN + TOP) + "   main-range   (argmin,argmax)")
    for band in bands:
        vals = [getter(b, band) for b in MAIN + TOP]
        main = [(v, b) for v, b in zip(vals[:8], MAIN) if v is not None]
        arg = ""
        if main:
            vmin = min(main)
            vmax = max(main)
            arg = "(%s,%s)" % (vmin[1], vmax[1])
        print("%-9s" % band + "".join("%9s" % fmt(v) for v in vals)
              + "   " + rng(vals[:8]) + "   " + arg)


def band_get(variant, field, key="ratio"):
    br = R["band_ratios"][variant][field]

    def g(b, band):
        for e in br[b]:
            if e["band"] == band:
                return e.get(key)
        return None
    return g


def boot_get(field, key):
    bp = R["bootstrap"]["per_field"][field]

    def g(b, band):
        for e in bp[b]:
            if e["band"] == band:
                return e.get(key)
        return None
    return g


def ratios():
    for variant, fields in R["band_ratios"].items():
        for field in fields:
            table(band_get(variant, field), "%s %s ratio" % (variant, field))
            table(band_get(variant, field, "err"), "%s %s err" % (variant, field))


def boot():
    for field in R["bootstrap"]["per_field"]:
        for key in ["boot_over_analytic", "unstrat_over_strat", "jack_over_boot",
                    "rel_err_mean", "err_median_over_err_mean",
                    "N_eff_for_3pct_mean"]:
            table(boot_get(field, key), "bootstrap %s %s" % (field, key))


def impact():
    for variant, fields in R["impact"].items():
        for field, obs in fields.items():
            print("\n## impact %s %s" % (variant, field))
            for k, v in obs.items():
                if isinstance(v, dict):
                    keys = [kk for kk in v if not isinstance(v[kk], (dict, list))]
                    print("  %-34s" % k + "  ".join("%s=%s" % (kk, v[kk]) for kk in keys))
                else:
                    print("  %-34s %s" % (k, v))


def diag():
    print(json.dumps(R["diagnostics"]["identities"], indent=1)[:4000])
    print(json.dumps(R["diagnostics"]["identity_noise"], indent=1)[:4000])


if __name__ == "__main__":
    parts = sys.argv[1:] or ["all"]
    if "all" in parts or "ratios" in parts:
        ratios()
    if "all" in parts or "boot" in parts:
        boot()
    if "all" in parts or "impact" in parts:
        impact()
    if "all" in parts or "diag" in parts:
        diag()


def summary(variants=None, fields=None):
    """One line per variant/field/band: main-bin range, top-bin values, max paired err."""
    for variant, fl in R["band_ratios"].items():
        if variants and variant.split(":")[1] not in variants:
            continue
        for field in fl:
            if fields and field not in fields:
                continue
            g = band_get(variant, field)
            ge = band_get(variant, field, "err")
            for band in BANDS:
                v = [g(b, band) for b in MAIN]
                e = [ge(b, band) for b in MAIN]
                vv = [x for x in v if x is not None]
                if not vv:
                    continue
                ee = [x for x in e if x is not None]
                print("%-28s %-4s %-9s main %s (n=%d)  top %s / %s   maxerr %s" % (
                    variant, field, band, rng(v), len(vv),
                    fmt(g(TOP[0], band)).strip(), fmt(g(TOP[1], band)).strip(),
                    fmt(max(ee)).strip() if ee else "-"))


if __name__ == "__main__" and "summary" in sys.argv[1:]:
    summary()
