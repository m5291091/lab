#!/usr/bin/env python3
"""Export the conceptual figures (library F01--F08) from the editable PowerPoint
library into publication assets (SVG / PDF / PNG) for the thesis body.

Gate T1B2.

Source of truth
---------------
The single editable source is
``docs/thesis/figures/editable/thesis_figure_library.pptx``, produced by
``scripts/generate_editable_figure_library.py``. This exporter does NOT re-author
the figures: it parses the committed PPTX package (stdlib ``zipfile`` +
``xml.etree``) and renders the slide's own shapes, connectors and text runs.
Any drift between the editable library and the exported assets is therefore
impossible by construction -- re-running this script after editing the library
re-derives every asset.

Nothing here reads or writes ``raw_data/``, ``result/`` or any measured value:
the eight conceptual figures carry no measured performance numbers.

Namespace note
--------------
Library IDs ``F01``--``F15`` (editable library) and canonical result figure IDs
``F1``--``F7`` (``result/figures/thesis/``) are SEPARATE namespaces. ``F01`` is
not ``F1``. Only F01--F08 (the conceptual figures) are exported here; F09--F15
are chart slides whose published counterparts are the canonical result figures.

Output
------
``docs/thesis/figures/exported/``
  ``figure_<n>_<m>_<slug>.svg``  vector, Word / web, extractable text
  ``figure_<n>_<m>_<slug>.pdf``  vector, final typesetting, embedded font subset
  ``figure_<n>_<m>_<slug>.png``  300 dpi raster preview for review only
  ``EXPORT_MANIFEST.tsv``        per-figure provenance and geometry
  ``README.md``                  tool / version / command record

Determinism
-----------
The SVG is generated from fixed geometry with no timestamps. ``rsvg-convert``
writes a wall-clock ``/CreationDate`` into the PDF, so the PDF is normalised to
a fixed date (same byte length, so the xref offsets stay valid). Two consecutive
runs are byte-identical.

Run:  python3 scripts/export_conceptual_figures.py
"""

import os
import re
import shutil
import subprocess
import sys
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # thesis_bc_project/
EDITABLE = ROOT / "docs" / "thesis" / "figures" / "editable"
PPTX_PATH = EDITABLE / "thesis_figure_library.pptx"
OUT = ROOT / "docs" / "thesis" / "figures" / "exported"

NS = {
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
}
A = "{%s}" % NS["a"]
P = "{%s}" % NS["p"]

EMU_PER_PT = 12700.0
EMU_PER_IN = 914400.0

# Deterministic PDF timestamp (SOURCE_DATE_EPOCH 1700000000 = 2023-11-14 22:13:20 UTC).
# Must keep the exact byte length cairo writes, so xref offsets stay valid.
PDF_FIXED_DATE = b"D:20231114221320+00'00"

# Margin added around the ink bounding box, in points.
CROP_MARGIN_PT = 9.0

# Font stack: Arial is the authored typeface; Nimbus Sans / Helvetica are the
# metric-compatible substitutes present on Linux rendering hosts.
FONT_STACK = "Arial, 'Nimbus Sans', Helvetica, 'Liberation Sans', sans-serif"

# Helvetica/Arial vertical metrics (hhea), em fractions. Used only to place the
# baseline inside a line box; horizontal metrics come from the generator.
FONT_ASCENT = 0.905
FONT_DESCENT = 0.212

# DrawingML "sm" triangle arrowhead, expressed in line widths. PowerPoint does
# not publish exact multipliers; these reproduce the on-screen proportions and
# are an explicit renderer approximation (recorded in the exported README).
ARROW_LEN_MULT = 3.0
ARROW_HALFWIDTH_MULT = 1.5

# Library figure (slide) -> thesis figure number + exported basename.
# Verified against the slide's own visible text, the editable FIGURE_MANIFEST
# and the receiving chapter section; see EXPORT_MANIFEST.tsv.
CONCEPTUAL = [
    # (library_id, slide_no, thesis_figure, basename, title, chapter_section)
    ("F01", 1, "1.1", "figure_1_1_thesis_overview",
     "Thesis Overview", "1.3"),
    ("F02", 2, "2.1", "figure_2_1_brandes_algorithm",
     "Brandes Algorithm Flow", "2.2"),
    ("F03", 3, "2.2", "figure_2_2_gh200_memory_hierarchy",
     "GH200 Memory Hierarchy", "2.5"),
    ("F04", 4, "4.1", "figure_4_1_gpu_execution_framework",
     "Overall GPU Execution Framework", "4.1"),
    ("F05", 5, "4.2", "figure_4_2_batch_source_mapping",
     "Batch-to-Source Mapping", "4.2"),
    ("F06", 6, "4.3", "figure_4_3_hybrid_bfs",
     "Hybrid BFS State Transition", "4.4"),
    ("F07", 7, "4.4", "figure_4_4_dual_stream_timeline",
     "Dual-Stream Timeline", "4.6"),
    ("F08", 8, "4.5", "figure_4_5_memory_management_variants",
     "Memory Management Variants", "4.7"),
]

VALIDATION = []


def note(name, ok, detail):
    VALIDATION.append((name, bool(ok), detail))
    return ok


# --------------------------------------------------------------------------- #
# Text metrics: imported from the library generator so the exporter and the
# PPTX layout guard cannot disagree about advance widths.
# --------------------------------------------------------------------------- #
def load_metrics():
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        import generate_editable_figure_library as gen
    except ImportError as exc:      # python-pptx absent
        raise SystemExit(
            "ERROR: cannot import scripts/generate_editable_figure_library.py "
            f"({exc}). The exporter reuses its Arial advance-width table so the "
            "export and the PPTX layout guard share one source of truth. Run "
            "with PYTHONPATH pointing at the python-pptx dependency set.")
    return gen


GEN = load_metrics()
ADVANCE = GEN.ADVANCE
LINE_HEIGHT = GEN.LINE_HEIGHT


def text_width_pt(text, size_pt, bold):
    table = ADVANCE[bool(bold)]
    missing = [ch for ch in text if ch not in table]
    if missing:
        raise SystemExit(f"ERROR: no advance width for {missing!r} in {text!r}")
    return sum(table[ch] for ch in text) / 1000.0 * size_pt


# --------------------------------------------------------------------------- #
# PPTX parsing.
# --------------------------------------------------------------------------- #
def emu_pt(v):
    return float(v) / EMU_PER_PT


def srgb(el):
    """Return '#rrggbb' for the first a:srgbClr under ``el``, else None."""
    if el is None:
        return None
    c = el.find(f"{A}srgbClr")
    return "#" + c.get("val").lower() if c is not None else None


def parse_xfrm(sp_pr):
    xfrm = sp_pr.find(f"{A}xfrm")
    if xfrm is None:
        return None
    off = xfrm.find(f"{A}off")
    ext = xfrm.find(f"{A}ext")
    return dict(
        x=emu_pt(off.get("x")), y=emu_pt(off.get("y")),
        w=emu_pt(ext.get("cx")), h=emu_pt(ext.get("cy")),
        flipH=xfrm.get("flipH") == "1", flipV=xfrm.get("flipV") == "1",
    )


def parse_line(sp_pr):
    ln = sp_pr.find(f"{A}ln")
    if ln is None:
        return None
    return dict(
        width=emu_pt(ln.get("w")) if ln.get("w") else 1.0,
        color=srgb(ln.find(f"{A}solidFill")) or "#000000",
        dashed=ln.find(f"{A}prstDash") is not None,
        head=ln.find(f"{A}headEnd") is not None,
        tail=ln.find(f"{A}tailEnd") is not None,
    )


def parse_text(sp):
    """Return DrawingML text as visible lines with per-run properties.

    Both the legacy literal-LF representation and standard ``a:br`` elements
    are accepted.  A paragraph boundary also starts a visible line, while each
    line retains its paragraph alignment and every run retains its own font
    family, size, weight, style, underline and colour.
    """
    tx = sp.find(f"{P}txBody")
    if tx is None:
        return None
    body = tx.find(f"{A}bodyPr")
    paras = tx.findall(f"{A}p")

    def property_value(primary, fallback, name, default=None):
        if primary is not None and primary.get(name) is not None:
            return primary.get(name)
        if fallback is not None and fallback.get(name) is not None:
            return fallback.get(name)
        return default

    def run_style(run, default_rpr):
        rpr = run.find(f"{A}rPr")
        size = property_value(rpr, default_rpr, "sz", "1800")
        bold = property_value(rpr, default_rpr, "b", "0") in ("1", "true")
        italic = property_value(rpr, default_rpr, "i", "0") in ("1", "true")
        underline = property_value(rpr, default_rpr, "u", "none") not in (None, "none")
        fill = rpr.find(f"{A}solidFill") if rpr is not None else None
        if fill is None and default_rpr is not None:
            fill = default_rpr.find(f"{A}solidFill")
        latin = rpr.find(f"{A}latin") if rpr is not None else None
        if latin is None and default_rpr is not None:
            latin = default_rpr.find(f"{A}latin")
        return dict(
            size=float(size) / 100.0,
            bold=bold,
            italic=italic,
            underline=underline,
            color=srgb(fill) or "#000000",
            font=(latin.get("typeface") if latin is not None else "Arial"),
        )

    lines = []
    for pa in paras:
        ppr = pa.find(f"{A}pPr")
        align = ppr.get("algn") if ppr is not None and ppr.get("algn") else "ctr"
        default_rpr = ppr.find(f"{A}defRPr") if ppr is not None else None
        current = []

        def finish_line():
            lines.append(dict(align=align, runs=list(current)))
            current.clear()

        for child in pa:
            if child.tag == f"{A}br":
                finish_line()
                continue
            if child.tag not in (f"{A}r", f"{A}fld"):
                continue
            style = run_style(child, default_rpr)
            text_nodes = child.findall(f"{A}t")
            text = "".join((node.text or "") for node in text_nodes)
            # Backward compatibility for Gate T1B2 PPTX files, whose runs put
            # authored newlines directly inside a:t.
            parts = re.split(r"(\r\n?|\n)", text)
            for part in parts:
                if not part:
                    continue
                if re.fullmatch(r"\r\n?|\n", part):
                    finish_line()
                else:
                    current.append(dict(text=part, **style))
        finish_line()

    if not any(run["text"] for line in lines for run in line["runs"]):
        return None
    return dict(
        lines=lines,
        anchor=(body.get("anchor") if body is not None else None) or "t",
        lIns=emu_pt(body.get("lIns")) if body is not None and body.get("lIns") else 7.2,
        rIns=emu_pt(body.get("rIns")) if body is not None and body.get("rIns") else 7.2,
    )


def parse_slide(zf, slide_no):
    """Parse one slide into a flat, renderer-ready shape list (z-order)."""
    root = ET.fromstring(zf.read(f"ppt/slides/slide{slide_no}.xml"))
    tree = root.find(f"{P}cSld").find(f"{P}spTree")
    shapes = []
    for el in tree:
        if el.tag == f"{P}sp":
            sp_pr = el.find(f"{P}spPr")
            geom = sp_pr.find(f"{A}prstGeom")
            prst = geom.get("prst") if geom is not None else None
            if prst not in ("roundRect", "rect", "ellipse"):
                raise SystemExit(f"ERROR: unsupported shape geometry {prst!r} "
                                 f"on slide {slide_no}")
            shapes.append(dict(
                kind="shape", prst=prst,
                name=el.find(f"{P}nvSpPr").find(f"{P}cNvPr").get("name"),
                geom=parse_xfrm(sp_pr),
                fill=srgb(sp_pr.find(f"{A}solidFill")),
                line=parse_line(sp_pr),
                text=parse_text(el),
            ))
        elif el.tag == f"{P}cxnSp":
            sp_pr = el.find(f"{P}spPr")
            geom = sp_pr.find(f"{A}prstGeom")
            if geom is None or geom.get("prst") != "line":
                raise SystemExit(f"ERROR: unsupported connector geometry on "
                                 f"slide {slide_no}")
            shapes.append(dict(
                kind="connector",
                name=el.find(f"{P}nvCxnSpPr").find(f"{P}cNvPr").get("name"),
                geom=parse_xfrm(sp_pr),
                line=parse_line(sp_pr),
            ))
        elif el.tag in (f"{P}nvGrpSpPr", f"{P}grpSpPr"):
            continue
        else:
            raise SystemExit(f"ERROR: unsupported element {el.tag} on slide {slide_no}")
    return shapes


def connector_points(geom):
    """Map a normalised connector xfrm (off/ext + flip flags) back to endpoints."""
    x1 = geom["x"] + (geom["w"] if geom["flipH"] else 0.0)
    x2 = geom["x"] + (0.0 if geom["flipH"] else geom["w"])
    y1 = geom["y"] + (geom["h"] if geom["flipV"] else 0.0)
    y2 = geom["y"] + (0.0 if geom["flipV"] else geom["h"])
    return x1, y1, x2, y2


# --------------------------------------------------------------------------- #
# Layout: text lines and ink extents.
# --------------------------------------------------------------------------- #
def text_lines(shape):
    """Return renderer-ready visible lines plus the text block's ink box."""
    t = shape["text"]
    g = shape["geom"]
    lines = t["lines"]
    line_sizes = [max((run["size"] for run in line["runs"]), default=18.0)
                  for line in lines]
    line_heights = [LINE_HEIGHT * size for size in line_sizes]
    block_h = sum(line_heights)
    if t["anchor"] == "ctr":
        top = g["y"] + g["h"] / 2.0 - block_h / 2.0
    elif t["anchor"] == "b":
        top = g["y"] + g["h"] - block_h
    else:
        top = g["y"]
    out, x0, x1 = [], None, None
    uniform_line_height = len(set(line_heights)) <= 1
    for index, (line, size, lh) in enumerate(zip(lines, line_sizes, line_heights)):
        # Preserve the legacy renderer's multiplication order for uniform text
        # blocks, avoiding a 0.001 pt fixed-format drift from repeated addition.
        offset = index * lh if uniform_line_height else sum(line_heights[:index])
        line_top = top + offset
        # Centre the largest em box within this line box. Smaller mixed-format
        # runs share that baseline, as they do in PowerPoint.
        base_in_line = ((lh - (FONT_ASCENT + FONT_DESCENT) * size) / 2.0
                        + FONT_ASCENT * size)
        by = line_top + base_in_line
        widths = [text_width_pt(run["text"], run["size"], run["bold"])
                  for run in line["runs"]]
        width = sum(widths)
        align = line["align"]
        if align == "ctr":
            ax = g["x"] + t["lIns"] + (g["w"] - t["lIns"] - t["rIns"]) / 2.0
            lx0, lx1 = ax - width / 2.0, ax + width / 2.0
        elif align == "r":
            ax = g["x"] + g["w"] - t["rIns"]
            lx0, lx1 = ax - width, ax
        else:
            ax = g["x"] + t["lIns"]
            lx0, lx1 = ax, ax + width
        x0 = lx0 if x0 is None else min(x0, lx0)
        x1 = lx1 if x1 is None else max(x1, lx1)
        out.append(dict(x=ax, y=by, align=align, runs=line["runs"]))
    ink = (x0, top, x1, top + block_h)
    return out, ink


def shape_ink(shape):
    """Ink bounding box (x0, y0, x1, y1) in points, stroke and arrowheads included."""
    g = shape["geom"]
    ln = shape.get("line")
    half = (ln["width"] / 2.0) if ln else 0.0
    if shape["kind"] == "connector":
        x1, y1, x2, y2 = connector_points(g)
        pad = half
        if ln and (ln["head"] or ln["tail"]):
            pad = max(pad, ARROW_HALFWIDTH_MULT * ln["width"])
        box = (min(x1, x2) - pad, min(y1, y2) - pad,
               max(x1, x2) + pad, max(y1, y2) + pad)
    else:
        box = (g["x"] - half, g["y"] - half, g["x"] + g["w"] + half, g["y"] + g["h"] + half)
        if shape["text"]:
            _, tink = text_lines(shape)
            box = (min(box[0], tink[0]), min(box[1], tink[1]),
                   max(box[2], tink[2]), max(box[3], tink[3]))
    return box


# --------------------------------------------------------------------------- #
# SVG emission.
# --------------------------------------------------------------------------- #
def esc(s):
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def fmt(v):
    """Fixed 3-decimal formatting so the SVG bytes are run-independent."""
    s = f"{v:.3f}"
    return "0.000" if s == "-0.000" else s


def arrow_marker(x_from, y_from, x_to, y_to, lw, color):
    """Filled triangle at (x_to, y_to) pointing away from (x_from, y_from).

    Returns (polygon_svg, shortened_endpoint) so the stroked line stops at the
    arrowhead base instead of protruding through the tip.
    """
    dx, dy = x_to - x_from, y_to - y_from
    d = (dx * dx + dy * dy) ** 0.5
    if d == 0:
        return "", (x_to, y_to)
    ux, uy = dx / d, dy / d
    ln = ARROW_LEN_MULT * lw
    hw = ARROW_HALFWIDTH_MULT * lw
    bx, by = x_to - ux * ln, y_to - uy * ln
    px, py = -uy, ux
    pts = (f"{fmt(x_to)},{fmt(y_to)} "
           f"{fmt(bx + px * hw)},{fmt(by + py * hw)} "
           f"{fmt(bx - px * hw)},{fmt(by - py * hw)}")
    return f'<polygon points="{pts}" fill="{color}"/>', (bx, by)


def render_svg(shapes, title, desc):
    ink = [shape_ink(s) for s in shapes]
    x0 = min(b[0] for b in ink) - CROP_MARGIN_PT
    y0 = min(b[1] for b in ink) - CROP_MARGIN_PT
    x1 = max(b[2] for b in ink) + CROP_MARGIN_PT
    y1 = max(b[3] for b in ink) + CROP_MARGIN_PT
    w, h = x1 - x0, y1 - y0

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{fmt(w / 72.0)}in" height="{fmt(h / 72.0)}in" '
        f'viewBox="{fmt(x0)} {fmt(y0)} {fmt(w)} {fmt(h)}" '
        f'version="1.1">',
        f"<title>{esc(title)}</title>",
        f"<desc>{esc(desc)}</desc>",
        f'<rect x="{fmt(x0)}" y="{fmt(y0)}" width="{fmt(w)}" height="{fmt(h)}" '
        f'fill="#ffffff"/>',
    ]

    for s in shapes:
        g = s["geom"]
        ln = s.get("line")
        if s["kind"] == "connector":
            ax, ay, bx, by = connector_points(g)
            color = ln["color"]
            lw = ln["width"]
            heads = []
            if ln["tail"]:
                poly, (bx, by) = arrow_marker(ax, ay, bx, by, lw, color)
                heads.append(poly)
            if ln["head"]:
                poly, (ax, ay) = arrow_marker(bx, by, ax, ay, lw, color)
                heads.append(poly)
            dash = f' stroke-dasharray="{fmt(4 * lw)},{fmt(3 * lw)}"' if ln["dashed"] else ""
            out.append(f'<line x1="{fmt(ax)}" y1="{fmt(ay)}" x2="{fmt(bx)}" '
                       f'y2="{fmt(by)}" stroke="{color}" stroke-width="{fmt(lw)}" '
                       f'stroke-linecap="butt"{dash}/>')
            out.extend(heads)
            continue

        fill = s["fill"] or "none"
        stroke = f' stroke="{ln["color"]}" stroke-width="{fmt(ln["width"])}"' if ln else ""
        if s["prst"] == "ellipse":
            out.append(f'<ellipse cx="{fmt(g["x"] + g["w"] / 2)}" '
                       f'cy="{fmt(g["y"] + g["h"] / 2)}" rx="{fmt(g["w"] / 2)}" '
                       f'ry="{fmt(g["h"] / 2)}" fill="{fill}"{stroke}/>')
        else:
            # DrawingML roundRect default adjust: radius = 16.667% of min(w, h).
            rx = f' rx="{fmt(0.16667 * min(g["w"], g["h"]))}"' if s["prst"] == "roundRect" else ""
            out.append(f'<rect x="{fmt(g["x"])}" y="{fmt(g["y"])}" '
                       f'width="{fmt(g["w"])}" height="{fmt(g["h"])}" '
                       f'fill="{fill}"{rx}{stroke}/>')

        if s["text"]:
            lines, _ = text_lines(s)
            for line in lines:
                anchor = {"ctr": "middle", "r": "end", "l": "start"}[line["align"]]
                runs = line["runs"]
                if len(runs) == 1:
                    run = runs[0]
                    weight = ' font-weight="bold"' if run["bold"] else ""
                    italic = ' font-style="italic"' if run["italic"] else ""
                    underline = ' text-decoration="underline"' if run["underline"] else ""
                    family = (FONT_STACK if run["font"].lower() == "arial"
                              else f"'{esc(run['font'])}', {FONT_STACK}")
                    out.append(
                        f'<text x="{fmt(line["x"])}" y="{fmt(line["y"])}" '
                        f'font-family="{family}" font-size="{fmt(run["size"])}" '
                        f'text-anchor="{anchor}"{weight}{italic}{underline} fill="{run["color"]}" '
                        f'xml:space="preserve">{esc(run["text"])}</text>')
                else:
                    spans = []
                    for run in runs:
                        weight = ' font-weight="bold"' if run["bold"] else ""
                        italic = ' font-style="italic"' if run["italic"] else ""
                        underline = ' text-decoration="underline"' if run["underline"] else ""
                        family = (FONT_STACK if run["font"].lower() == "arial"
                                  else f"'{esc(run['font'])}', {FONT_STACK}")
                        spans.append(
                            f'<tspan font-family="{family}" font-size="{fmt(run["size"])}"'
                            f'{weight}{italic}{underline} fill="{run["color"]}">'
                            f'{esc(run["text"])}</tspan>')
                    out.append(
                        f'<text x="{fmt(line["x"])}" y="{fmt(line["y"])}" '
                        f'text-anchor="{anchor}" xml:space="preserve">'
                        f'{"".join(spans)}</text>')

    out.append("</svg>")
    return "\n".join(out) + "\n", (w, h)


# --------------------------------------------------------------------------- #
# Raster / PDF conversion.
# --------------------------------------------------------------------------- #
def rsvg_version():
    v = subprocess.run(["rsvg-convert", "--version"], capture_output=True, text=True)
    return v.stdout.strip()


def normalise_pdf(path):
    """Replace cairo's wall-clock /CreationDate with a fixed, same-length value."""
    data = path.read_bytes()
    m = re.search(rb"/CreationDate\s*\((D:[^)]*)\)", data)
    if not m:
        return False
    old = m.group(1)
    if len(old) != len(PDF_FIXED_DATE):
        raise SystemExit(f"ERROR: unexpected /CreationDate length in {path.name}: "
                         f"{old!r} ({len(old)} bytes, expected {len(PDF_FIXED_DATE)})")
    path.write_bytes(data[:m.start(1)] + PDF_FIXED_DATE + data[m.end(1):])
    return True


def convert(svg_path, pdf_path, png_path):
    # 72 dpi for PDF so 1 SVG inch maps to exactly 72 PostScript points.
    subprocess.run(["rsvg-convert", "-f", "pdf", "-d", "72", "-p", "72",
                    "-o", str(pdf_path), str(svg_path)], check=True)
    normalise_pdf(pdf_path)
    subprocess.run(["rsvg-convert", "-f", "png", "-d", "300", "-p", "300",
                    "-o", str(png_path), str(svg_path)], check=True)


def png_size(path):
    import struct
    d = path.read_bytes()
    return struct.unpack(">II", d[16:24])


def png_ink_margins(path):
    """Non-white bounding box of the rendered PNG, as (l, t, r, b) pixel margins.

    This inspects what the renderer actually produced, so it catches clipping or
    a blank page that a purely geometric check on our own model would miss.
    """
    try:
        import numpy as np
        from PIL import Image
    except ImportError as exc:
        raise SystemExit(
            f"ERROR: Pillow/numpy required for the rendered-output check ({exc}).")
    a = np.array(Image.open(path).convert("RGB"))
    h, w = a.shape[:2]
    ink = (a < 250).any(axis=2)
    if not ink.any():
        return None, (w, h)
    rows = np.where(ink.any(axis=1))[0]
    cols = np.where(ink.any(axis=0))[0]
    return (int(cols.min()), int(rows.min()),
            int(w - 1 - cols.max()), int(h - 1 - rows.max())), (w, h)


# --------------------------------------------------------------------------- #
# Validation of the exported assets.
# --------------------------------------------------------------------------- #
def validate_asset(fig, svg_path, pdf_path, png_path, size_pt, shapes):
    tag = fig["thesis_figure"]

    svg = svg_path.read_text(encoding="utf-8")
    ET.fromstring(svg)                       # parse error => exception
    note(f"svg_parses[{tag}]", True, f"{svg_path.name} is well-formed XML")

    texts = re.findall(r"<text[^>]*>(.*?)</text>", svg, re.S)
    joined = "".join(texts)
    non_ascii = sorted({c for c in joined if ord(c) > 127})
    note(f"svg_ascii_only[{tag}]", not non_ascii,
         f"{len(texts)} text nodes, non-ASCII={non_ascii}")

    # No figure number and no internal artifact ID burned into the image.
    bad = []
    for t in texts:
        plain = re.sub(r"&[a-z]+;", "", t)
        if re.search(r"\bFigure\s+\d", plain) or re.search(r"\bF\d{1,2}\b", plain):
            bad.append(plain)
        if re.search(r"\b(job|Gate|checkpoint)\b", plain, re.I):
            bad.append(plain)
        if re.search(r"(raw_data/|result/|docs/|scripts/)", plain):
            bad.append(plain)
    note(f"no_internal_ids[{tag}]", not bad, f"offending in-figure text: {bad}")

    note(f"no_raster_in_svg[{tag}]", "<image" not in svg,
         "SVG contains no embedded raster image")

    pdf = pdf_path.read_bytes()
    note(f"pdf_vector[{tag}]", pdf.startswith(b"%PDF") and b"/FontFile" in pdf,
         "PDF has an embedded font program (text is vector, not traced raster)")
    note(f"pdf_no_raster[{tag}]", b"/Subtype /Image" not in pdf and b"/Image" not in pdf,
         "PDF contains no image XObject")
    mb = re.search(rb"/MediaBox\s*\[([^\]]*)\]", pdf)
    mw, mh = (float(v) for v in mb.group(1).split()[2:4])
    note(f"pdf_size_matches[{tag}]",
         abs(mw - size_pt[0]) < 1.0 and abs(mh - size_pt[1]) < 1.0,
         f"MediaBox {mw:.1f}x{mh:.1f} pt vs SVG {size_pt[0]:.1f}x{size_pt[1]:.1f} pt")

    pw, ph = png_size(png_path)
    exp_w = size_pt[0] / 72.0 * 300.0
    note(f"png_300dpi[{tag}]", abs(pw - exp_w) <= 2 and pw > 0 and ph > 0,
         f"{pw}x{ph} px = {pw / (size_pt[0] / 72.0):.0f} dpi")
    note(f"png_not_blank[{tag}]", png_path.stat().st_size > 5000,
         f"{png_path.stat().st_size} bytes")

    # Rendered-output check: real ink present, and none of it touching an edge.
    margins, (pxw, pxh) = png_ink_margins(png_path)
    note(f"rendered_not_blank[{tag}]", margins is not None,
         "rendered PNG contains non-white pixels")
    note(f"rendered_not_clipped[{tag}]", margins is not None and min(margins) >= 2,
         f"ink margins (l,t,r,b) = {margins} px in {pxw}x{pxh}")

    # Every shape must lie inside the exported canvas (no clipped/off-canvas ink).
    vb = re.search(r'viewBox="([^"]+)"', svg).group(1).split()
    vx0, vy0, vw, vh = (float(v) for v in vb)
    outside = []
    for s in shapes:
        b = shape_ink(s)
        if (b[0] < vx0 - 0.01 or b[1] < vy0 - 0.01
                or b[2] > vx0 + vw + 0.01 or b[3] > vy0 + vh + 0.01):
            outside.append(s["name"])
    note(f"no_object_outside_canvas[{tag}]", not outside, f"outside={outside}")

    # Authored line breaks must be honoured: one <text> per authored line.
    authored = sum(len(s["text"]["lines"]) for s in shapes if s.get("text"))
    note(f"text_lines_preserved[{tag}]", authored == len(texts),
         f"authored lines={authored} rendered text nodes={len(texts)}")

    # Minimum on-page font size at final print width.
    min_pt = min((run["size"] for s in shapes if s.get("text")
                  for line in s["text"]["lines"] for run in line["runs"]), default=0)
    note(f"min_font_size[{tag}]", min_pt >= 12.0,
         f"smallest in-figure type = {min_pt:g} pt at {size_pt[0] / 72.0:.2f} in wide")
    return min_pt


def check_text_fits(shapes, tag):
    """Every authored line must fit inside its own shape (no clipping/overflow)."""
    bad = []
    for s in shapes:
        t = s.get("text")
        if not t or s["kind"] != "shape":
            continue
        avail = s["geom"]["w"] - t["lIns"] - t["rIns"]
        for line in t["lines"]:
            line_text = "".join(run["text"] for run in line["runs"])
            w = sum(text_width_pt(run["text"], run["size"], run["bold"])
                    for run in line["runs"])
            if w > avail + 0.01:
                bad.append((s["name"], line_text, round(w, 2), round(avail, 2)))
    note(f"text_fits_in_shape[{tag}]", not bad, f"overflowing lines: {bad}")


def check_overlap(shapes, tag):
    """Filled shapes must not overlap each other's ink (containers excepted).

    A container is a shape that fully encloses another; that is deliberate
    (panels, group frames). Partial overlaps are the unintended kind.
    """
    boxes = [(s["name"], shape_ink(s)) for s in shapes if s["kind"] == "shape" and s["fill"]]
    bad = []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            (n1, a), (n2, b) = boxes[i], boxes[j]
            ox = min(a[2], b[2]) - max(a[0], b[0])
            oy = min(a[3], b[3]) - max(a[1], b[1])
            if ox <= 0.5 or oy <= 0.5:
                continue
            contains = ((a[0] <= b[0] + .5 and a[1] <= b[1] + .5
                         and a[2] >= b[2] - .5 and a[3] >= b[3] - .5)
                        or (b[0] <= a[0] + .5 and b[1] <= a[1] + .5
                            and b[2] >= a[2] - .5 and b[3] >= a[3] - .5))
            if not contains:
                bad.append((n1, n2, round(ox, 2), round(oy, 2)))
    note(f"no_unintended_overlap[{tag}]", not bad, f"partial overlaps: {bad}")


# --------------------------------------------------------------------------- #
# Manifest and README.
# --------------------------------------------------------------------------- #
def write_manifest(rows):
    import csv
    header = ["LibraryFigureID", "Namespace", "ThesisFigureNumber", "ChapterSection",
              "Title", "EditableSource", "EditableSlide", "SVG", "PDF", "PNG",
              "WidthIn", "HeightIn", "PNGPixels", "MinFontPt", "Objects",
              "CanonicalResultFigureID", "Status"]
    path = OUT / "EXPORT_MANIFEST.tsv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)
    return path


README = """\
# Exported Thesis Figures (conceptual figures F01--F08)

論文本体へ掲載する概念図の組版用 asset。**編集正本は
`../editable/thesis_figure_library.pptx`** であり、このディレクトリの
SVG/PDF/PNG はすべてその PPTX から機械的に導出される。ここのファイルを直接
編集してはならない。図を変更する場合は
`scripts/generate_editable_figure_library.py` を修正して PPTX を再生成し、
続いて `scripts/export_conceptual_figures.py` を実行する。

## Figure ID namespace

`F01`--`F15` は **editable library** の ID であり、
`result/figures/thesis/` の canonical result figure `F1`--`F7` とは
**別の namespace** である。`F01` は `F1` ではない。論文本文で参照する番号は
`ThesisFigureNumber`（Figure 1.1 など）であり、library ID は本文に出さない。
対応表は `../editable/FIGURE_MANIFEST.tsv` と `EXPORT_MANIFEST.tsv` にある。

## Formats

| 形式 | 用途 | 性質 |
|---|---|---|
| `.svg` | Word / Web 貼り込み | vector、テキストは `<text>` のまま（抽出・再検索可能） |
| `.pdf` | 最終組版 | vector、font subset 埋め込み（`/FontFile3`）、raster image なし |
| `.png` | review 用 preview | 300 dpi 相当。**組版には使わない** |

caption と Figure 番号は画像へ焼き込まない。論文側の caption 機構が担当する。
背景は全図とも白で統一し、ink bounding box に {margin} pt の余白を付けて
crop している（slide 全面の余白帯は含めない）。

## Export tool and command

```text
{tools}
```

```bash
cd thesis_bc_project
PYTHONPATH={deps} python3.12 scripts/export_conceptual_figures.py
```

interpreter は editable library generator と同じ Python 3.12 系を用いる
（`generate_editable_figure_library.py` の `figure_data.xlsx` は
`statistics.stdev` の実装差で Python 3.9 系だと sample SD 2 セルが 1 ULP
ずれるため、両 script とも 3.12 で実行する）。

`PYTHONPATH` は `python-pptx` を含む依存 set を指す。exporter 自身は PPTX を
`zipfile` + `xml.etree` で読むため python-pptx を必要としないが、Arial の
advance width table を `generate_editable_figure_library.py` から import して
PPTX 側の layout guard と同一の値を使うため、同 module が import 可能である
必要がある。

## Determinism

SVG は固定小数点で書き出し timestamp を持たない。`rsvg-convert` が PDF へ
書き込む `/CreationDate` は固定値へ正規化する（同一 byte 長のため xref offset
は不変）。連続 2 回の実行は全 asset が byte-identical になる。

## Renderer approximation (要人間確認)

- DrawingML の `type="triangle" w="sm" len="sm"` arrowhead は、線幅の
  {alen}x（長さ）/ {awid}x（底辺幅）の三角形として描画している。PowerPoint の
  内部係数は公開されていないため、これは見た目を合わせた近似である。
- PPTX の複数行 label は標準DrawingMLの `<a:br/>` で表現される。本 exporter は
  `<a:t>`、`<a:br/>`、複数 `<a:p>`、paragraph alignment、およびrun別font
  propertiesを解釈する。旧PPTXとの比較用に `<a:t>` 内literal LFも読み取れるが、
  正式生成物はliteral LFへ依存しない。
- font は Arial を第一候補とし、Linux 上では metric 互換の Nimbus Sans /
  Liberation Sans へ fallback する。

## Files

{files}
"""


def write_readme(rows, tools):
    files = "\n".join(
        f"- `{r[7]}` / `{r[8]}` / `{r[9]}` -- Figure {r[2]} {r[4]} "
        f"(library {r[0]}, slide {r[6]}, {r[10]} x {r[11]} in)" for r in rows)
    path = OUT / "README.md"
    path.write_text(README.format(
        tools=tools, files=files, deps="/tmp/gate_v0_editable_deps",
        margin=f"{CROP_MARGIN_PT:g}", alen=f"{ARROW_LEN_MULT:g}",
        awid=f"{ARROW_HALFWIDTH_MULT:g}"), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
def main():
    if not PPTX_PATH.exists():
        raise SystemExit(f"ERROR: editable source not found: {PPTX_PATH}")
    if not shutil.which("rsvg-convert"):
        raise SystemExit("ERROR: rsvg-convert not found; cannot export PDF/PNG.")
    OUT.mkdir(parents=True, exist_ok=True)

    tools = (f"rsvg-convert : {rsvg_version()}\n"
             f"python       : {sys.version.split()[0]}\n"
             f"source pptx  : docs/thesis/figures/editable/thesis_figure_library.pptx")

    rows = []
    with zipfile.ZipFile(PPTX_PATH) as zf:
        for lib_id, slide_no, thesis_no, base, title, section in CONCEPTUAL:
            shapes = parse_slide(zf, slide_no)
            desc = (f"Conceptual figure for thesis Figure {thesis_no} "
                    f"(section {section}); exported from the editable "
                    f"PowerPoint library slide {slide_no}.")
            svg, (w_pt, h_pt) = render_svg(shapes, title, desc)
            svg_path = OUT / f"{base}.svg"
            pdf_path = OUT / f"{base}.pdf"
            png_path = OUT / f"{base}.png"
            svg_path.write_text(svg, encoding="utf-8")
            convert(svg_path, pdf_path, png_path)

            fig = dict(thesis_figure=thesis_no)
            min_pt = validate_asset(fig, svg_path, pdf_path, png_path, (w_pt, h_pt), shapes)
            check_text_fits(shapes, thesis_no)
            check_overlap(shapes, thesis_no)

            pw, ph = png_size(png_path)
            rows.append([
                lib_id, "editable_library", thesis_no, section, title,
                "docs/thesis/figures/editable/thesis_figure_library.pptx", slide_no,
                svg_path.name, pdf_path.name, png_path.name,
                f"{w_pt / 72.0:.2f}", f"{h_pt / 72.0:.2f}", f"{pw}x{ph}",
                f"{min_pt:g}", len(shapes), "not_applicable", "Exported",
            ])

    manifest = write_manifest(rows)
    readme = write_readme(rows, tools)

    note("all_eight_conceptual_exported", len(rows) == 8,
         f"{len(rows)} conceptual figures exported")
    note("thesis_numbers_unique",
         len({r[2] for r in rows}) == 8, sorted(r[2] for r in rows))

    print("Gate T1B2 conceptual figure export")
    for name, ok, detail in VALIDATION:
        print(f"  [{'OK ' if ok else 'FAIL'}] {name}: {detail}")
    failed = [n for n, ok, _ in VALIDATION if not ok]
    for r in rows:
        print(f"{r[0]}\tFigure {r[2]}\t{r[7]}\t{r[10]}x{r[11]} in\t{r[12]} px")
    print(f"manifest: {manifest.relative_to(ROOT)}")
    print(f"readme:   {readme.relative_to(ROOT)}")
    if failed:
        raise SystemExit(f"VALIDATION FAILED: {failed}")
    print("ALL_EXPORT_VALIDATION_OK")


if __name__ == "__main__":
    main()
