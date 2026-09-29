"""Minimal vector-curve extractor for matplotlib PDFs converted with `pdftocairo -svg`.

Returns every stroked path with its colour, width and absolute (page) coordinates, after composing all
parent-group and element transforms. Only M/L/Z commands are parsed (matplotlib line plots); paths with
curves are returned with their M/L vertices only (flagged has_curves)."""
import re
import xml.etree.ElementTree as ET

import numpy as np

NS = "{http://www.w3.org/2000/svg}"


def _parse_transform(s):
    m = np.eye(3)
    if not s:
        return m
    for name, args in re.findall(r"(\w+)\(([^)]*)\)", s):
        v = [float(a) for a in re.split(r"[ ,]+", args.strip()) if a]
        if name == "matrix":
            t = np.array([[v[0], v[2], v[4]], [v[1], v[3], v[5]], [0, 0, 1]])
        elif name == "translate":
            t = np.array([[1, 0, v[0]], [0, 1, v[1] if len(v) > 1 else 0], [0, 0, 1]])
        elif name == "scale":
            t = np.array([[v[0], 0, 0], [0, v[1] if len(v) > 1 else v[0], 0], [0, 0, 1]])
        else:
            raise ValueError(name)
        m = m @ t
    return m


def _style(el):
    st = {}
    for kv in (el.get("style") or "").split(";"):
        if ":" in kv:
            k, v = kv.split(":", 1)
            st[k.strip()] = v.strip()
    return st


def stroked_paths(svg_file):
    tree = ET.parse(svg_file)
    out = []
    ids = {el.get("id"): el for el in tree.iter() if el.get("id")}
    XL = "{http://www.w3.org/1999/xlink}href"

    def walk(el, mat, in_defs=False, depth=0):
        tag = el.tag.replace(NS, "")
        if tag in ("defs", "symbol", "clipPath", "mask"):
            in_defs = True
        m = mat @ _parse_transform(el.get("transform"))
        if tag == "use" and not in_defs:
            ref = ids.get((el.get(XL) or el.get("href") or "").lstrip("#"))
            if ref is not None and ref.tag.replace(NS, "") not in ("symbol",) and depth < 20:
                mu = m @ _parse_transform("translate(%s,%s)" % (el.get("x", "0"), el.get("y", "0")))
                # walk the referenced element as content (its own transform is applied inside)
                walk_content(ref, mu, depth + 1)
            return
        if tag == "path" and not in_defs:
            st = _style(el)
            stroke = st.get("stroke", "none")
            d = el.get("d", "")
            toks = re.findall(r"[MLCZmlcz]|-?\d*\.?\d+(?:e-?\d+)?", d)
            pts, i, cmd, curves = [], 0, None, False
            while i < len(toks):
                t = toks[i]
                if t in "MLCZmlcz":
                    cmd = t
                    i += 1
                    if t in "Zz":
                        continue
                    continue
                if cmd in ("M", "L"):
                    pts.append((float(toks[i]), float(toks[i + 1])))
                    i += 2
                elif cmd == "C":
                    curves = True
                    pts.append((float(toks[i + 4]), float(toks[i + 5])))
                    i += 6
                else:
                    i += 1
            if pts:
                p = np.array(pts)
                ph = np.c_[p, np.ones(len(p))] @ m.T
                out.append(dict(stroke=stroke, width=float(st.get("stroke-width", "1")),
                                opacity=float(st.get("stroke-opacity", "1")), dash=st.get("stroke-dasharray"),
                                fill=st.get("fill", "none"), xy=ph[:, :2], has_curves=curves))
        for ch in el:
            walk(ch, m, in_defs, depth)

    def walk_content(el, mat, depth):
        tag = el.tag.replace(NS, "")
        if tag in ("clipPath", "mask", "symbol"):
            return
        walk(el, mat, False, depth)

    walk(tree.getroot(), np.eye(3))
    return out
