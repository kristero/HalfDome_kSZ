#!/usr/bin/env python3
"""Assemble the HalfDome analysis report: sections/*.html fragments -> index.html with a generated table of contents and a
figure index. Usage: python3 scripts/assemble.py (from anywhere). Writes <root>/index.html and <root>/figure_files.json."""
import html
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORDER = ["s1_lee22.html", "s2_frb.html", "s3_sbi64k.html", "s4_seedmask.html", "s5_merlin.html", "s6_xgpaint.html"]


def slug(text):
    s = re.sub(r"<[^>]+>", "", text)
    s = re.sub(r"[^a-z0-9]+", "-", html.unescape(s).lower()).strip("-")
    return s[:60] or "h"


def add_ids(frag, sec_id, seen):
    """Give every h3 an id (sec-id + slug) unless it has one; return fragment and [(level, id, text)]."""
    toc = []

    def repl(m):
        tag, attrs, inner = m.group(1), m.group(2) or "", m.group(3)
        idm = re.search(r'id="([^"]+)"', attrs)
        hid = idm.group(1) if idm else "%s-%s" % (sec_id, slug(inner))
        while hid in seen:
            hid += "-x"
        seen.add(hid)
        toc.append((tag, hid, re.sub(r"<[^>]+>", "", inner).strip()))
        if not idm:
            attrs = (attrs + ' id="%s"' % hid)
        return "<%s%s>%s</%s>" % (tag, attrs, inner, tag)

    frag = re.sub(r"<(h3)(\s[^>]*)?>(.*?)</h3>", repl, frag, flags=re.S)
    return frag, toc


def main():
    shell = (ROOT / "scripts" / "shell.html").read_text()
    seen, parts, nav, figs = set(), [], [], []
    for name in ORDER:
        p = ROOT / "sections" / name
        if not p.exists():
            print("missing", name)
            continue
        frag = p.read_text()
        # one excerpt-label convention across sections: path:L1-L2
        frag = re.sub(r'(data-src="[^"]*?):(\d+)-(\d+)"', r'\1:L\2-L\3"', frag)
        sm = re.search(r'<section\s+id="([^"]+)"', frag)
        sec_id = sm.group(1) if sm else name.split(".")[0]
        h2 = re.search(r"<h2[^>]*>(.*?)</h2>", frag, flags=re.S)
        title = re.sub(r"<[^>]+>", "", h2.group(1)).strip() if h2 else sec_id
        frag, toc = add_ids(frag, sec_id, seen)
        sub = "".join('<li><a href="#%s">%s</a></li>' % (hid, html.escape(t)) for _, hid, t in toc)
        nav.append('<li><a href="#%s">%s</a>%s</li>' % (sec_id, html.escape(title), "<ol>%s</ol>" % sub if sub else ""))
        parts.append(frag)
        for m in re.finditer(r'<img[^>]*src="([^"]+)"', frag):
            if not (ROOT / m.group(1)).exists():
                print("MISSING FIGURE", name, m.group(1))
            figs.append({"section": title, "file": m.group(1)})
    idx_rows = "".join('<tr><td>%s</td><td><code>%s</code></td></tr>' % (html.escape(f["section"]), html.escape(f["file"])) for f in figs)
    page = (shell.replace("<!--NAV-->", "<ol>%s</ol>" % "".join(nav))
                 .replace("<!--SECTIONS-->", "\n".join(parts))
                 .replace("<!--FIGINDEX-->", idx_rows)
                 .replace("<!--NFIG-->", str(len(figs))))
    (ROOT / "index.html").write_text(page)
    (ROOT / "figure_files.json").write_text(json.dumps(sorted({f["file"] for f in figs}), indent=1))
    print("index.html: %d sections, %d figures, %.1f kB" % (len(parts), len(figs), len(page) / 1e3))


if __name__ == "__main__":
    main()
