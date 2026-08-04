"""PDF -> figure images + captions + the paragraphs that discuss them.

Figures are obtained by *rendering page regions*, not by pulling out embedded image
objects. Publishers routinely slice one figure into hundreds of image fragments (one
paper in this corpus carries 5086 of them) and draw others as vector paths with no image
object at all, so object extraction yields confetti for some papers and nothing for
others. Rendering the region a figure occupies works for both.
"""
from __future__ import annotations

import os
import re
import unicodedata
from dataclasses import dataclass, field

import fitz  # PyMuPDF

# Journals in this corpus label figures in English, German ("Abb."), Chinese ("图"),
# and French ("Figure"), sometimes bilingually in one caption block.
CAPTION_RE = re.compile(
    r"^\s*(?:(Figure|Fig|Abbildung|Abb|Scheme|Chart)\.?\s*(S?\d+[a-z]?)"
    r"|(FIGURE|FIG)\.?\s*(S?\d+[a-z]?)"
    r"|\u56fe\s*(\d+[a-z]?))\b", re.I)
REF_RE = re.compile(
    r"(?:\b(?:Figure|Fig|FIGURE|FIG|Abb|Abbildung)\.?|图)\s*(S?\d+)", re.I)


@dataclass
class Figure:
    paper_id: str
    label: str                 # "Figure 3"
    number: str                # "3" or "S2"
    page: int                  # 0-based
    bbox: tuple                # rendered region, PDF points
    caption: str
    image_path: str = ""
    context: list = field(default_factory=list)   # paragraphs that mention this figure
    width_px: int = 0
    height_px: int = 0


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    return re.sub(r"[ \t]+", " ", s).strip()


def _graphic_rects(page: fitz.Page, min_area: float = 400.0,
                   scanned: bool = False) -> list[fitz.Rect]:
    """Every drawn thing on the page: image placements and vector paths.

    On a scanned page the whole sheet is one image, which the full-page filter would
    normally discard; there the page itself is the figure source and panel detection
    does the rest."""
    rects = []
    for img in page.get_images(full=True):
        try:
            rects += [fitz.Rect(r) for r in page.get_image_rects(img[0])]
        except Exception:
            pass
    for d in page.get_drawings():
        r = fitz.Rect(d["rect"])
        if r.width > 1 and r.height > 1:
            rects.append(r)
    page_area = abs(page.rect.get_area())
    out = []
    for r in rects:
        r = r & page.rect
        if r.is_empty or r.get_area() < min_area:
            continue
        if r.get_area() > 0.98 * page_area and not scanned:   # full-page background
            continue
        out.append(r)
    return out


def _cluster(rects: list[fitz.Rect], gap: float = 12.0) -> list[fitz.Rect]:
    """Merge graphics that touch or nearly touch into one figure region."""
    boxes = [fitz.Rect(r) for r in rects]
    changed = True
    while changed:
        changed = False
        out: list[fitz.Rect] = []
        for b in boxes:
            for i, o in enumerate(out):
                grown = fitz.Rect(o.x0 - gap, o.y0 - gap, o.x1 + gap, o.y1 + gap)
                if grown.intersects(b):
                    out[i] = o | b
                    changed = True
                    break
            else:
                out.append(b)
        boxes = out
    return boxes


def _text_blocks(page: fitz.Page):
    blocks = []
    for b in page.get_text("blocks"):
        x0, y0, x1, y1, text = b[0], b[1], b[2], b[3], b[4]
        t = _norm(text)
        if t:
            blocks.append((fitz.Rect(x0, y0, x1, y1), t))
    return blocks


def find_captions(page: fitz.Page):
    """Caption blocks on the page: (rect, label, number, caption text)."""
    out = []
    for rect, text in _text_blocks(page):
        m = CAPTION_RE.match(text)
        if not m:
            continue
        raw = (m.group(1) or m.group(3) or ("\u56fe" if m.group(5) else "Figure"))
        kind = raw.title()
        if kind.lower().startswith(("fig", "abb")) or kind == "\u56fe":
            kind = "Figure"
        num = (m.group(2) or m.group(4) or m.group(5) or "").upper()
        if kind != "Figure":                      # schemes/charts are not data plots
            continue
        out.append((rect, f"{kind} {num}", num, text))
    return out


def figure_regions(page: fitz.Page, max_gap: float = 90.0, min_area: float = 12000.0):
    """Pair each caption with the graphics block it belongs to.

    A caption sits below its figure in almost every journal layout, but the two boxes
    often overlap: a wide figure's cluster can extend past the caption's top, and in
    two-column layouts a caption may sit beside part of the artwork. So the test is
    "the figure is mostly above the caption", not "strictly above it".

    Graphics that no caption claims are still returned, unlabelled — some PDFs have a
    scrambled text layer where captions cannot be read at all, and dropping their
    figures would lose the data entirely.
    """
    caps = find_captions(page)
    scanned = len(page.get_text().strip()) < 50      # no text layer: a scanned page
    clusters = [g for g in _cluster(_graphic_rects(page, scanned=scanned))
                if g.get_area() >= min_area]
    used, out = set(), []

    for crect, label, num, text in caps:
        best, best_d = None, 1e9
        for i, g in enumerate(clusters):
            if i in used:
                continue
            overlap = min(g.x1, crect.x1) - max(g.x0, crect.x0)
            if overlap <= 0.25 * min(g.width, crect.width):
                continue
            mid = (g.y0 + g.y1) / 2.0
            below = crect.y0 - g.y1               # caption under the figure
            above = g.y0 - crect.y1               # caption over the figure
            if crect.y0 > mid and below < max_gap:
                d = max(below, 0.0)
            elif 0 <= above < max_gap:
                d = above + 5.0                   # prefer the caption-below reading
            else:
                continue
            if d < best_d:
                best, best_d = i, d
        if best is None:
            continue
        used.add(best)
        out.append((clusters[best], crect, label, num, text))

    for i, g in enumerate(clusters):
        if i in used or g.get_area() < 3 * min_area:
            continue
        near = [t for r, t in _text_blocks(page)
                if 0 <= r.y0 - g.y1 < max_gap
                and min(r.x1, g.x1) - max(r.x0, g.x0) > 0.25 * min(r.width, g.width)]
        out.append((g, None, None, None, near[0] if near else ""))
    return out


def render(page: fitz.Page, rect: fitz.Rect, path: str, dpi: int = 300,
           pad: float = 4.0) -> tuple[int, int]:
    r = fitz.Rect(rect.x0 - pad, rect.y0 - pad, rect.x1 + pad, rect.y1 + pad) & page.rect
    pix = page.get_pixmap(clip=r, dpi=dpi, alpha=False)
    pix.save(path)
    return pix.width, pix.height


def _paragraphs(doc: fitz.Document) -> list[str]:
    paras = []
    for page in doc:
        for _, t in _text_blocks(page):
            if len(t) > 80:                       # skip headers, page numbers, labels
                paras.append(t)
    return paras


def context_for(paras: list[str], number: str, max_paras: int = 6) -> list[str]:
    """Paragraphs that reference this figure number, plus their immediate neighbour."""
    n = number.lstrip("S")
    is_si = number.upper().startswith("S")
    hits = []
    for i, p in enumerate(paras):
        for m in REF_RE.finditer(p):
            num = m.group(1).upper()
            if num.lstrip("S") == n and (num.startswith("S") == is_si):
                hits.append(i)
                break
    seen, out = set(), []
    for i in hits[:max_paras]:
        for j in (i, i + 1):
            if j < len(paras) and j not in seen:
                seen.add(j)
                out.append(paras[j])
    return out[:max_paras]


def extract(pdf_path: str, outdir: str, dpi: int = 300,
            min_px: int = 200) -> list[Figure]:
    """All figures of one paper, rendered, with caption and referencing context."""
    paper_id = os.path.splitext(os.path.basename(pdf_path))[0]
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", paper_id)
    os.makedirs(outdir, exist_ok=True)
    doc = fitz.open(pdf_path)
    paras = _paragraphs(doc)
    figs: list[Figure] = []
    unlabelled = 0
    for pno, page in enumerate(doc):
        for g, crect, label, num, ctext in figure_regions(page):
            if label is None:                     # scrambled text layer: keep the figure
                unlabelled += 1
                label, num = f"Unlabelled {unlabelled}", f"U{unlabelled}"
            name = f"{safe}__p{pno + 1}_{re.sub(r'[^A-Za-z0-9]', '', label)}.png"
            path = os.path.join(outdir, name)
            try:
                w, h = render(page, g, path, dpi=dpi)
            except Exception:
                continue
            if w < min_px or h < min_px:
                os.remove(path)
                continue
            figs.append(Figure(paper_id=paper_id, label=label, number=num, page=pno,
                               bbox=(g.x0, g.y0, g.x1, g.y1), caption=ctext,
                               image_path=path, width_px=w, height_px=h,
                               context=context_for(paras, num) if not num.startswith("U")
                               else []))
    doc.close()
    return figs
