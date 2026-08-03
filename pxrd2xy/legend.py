"""Legend recognition: attribute figure legend entries to separated curves.

Two independent routes are used and their agreement is recorded, following the
ensemble idea in He et al. (a legend assignment is high-confidence only when
independent inferences agree):

  * colour route   - the colour of the legend key line (or of the legend text itself)
                     is matched against the colour of the separated curve;
  * geometry route - the vertical position of the text is matched against the curve
                     that runs closest to it (handles 'a / b / c' side labels).
"""
from __future__ import annotations

import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment

from .axes import parse_number
from .core import rgb_to_lab


def _text_color(rgb, removed_mask, bbox):
    x0, y0, x1, y1 = [int(round(v)) for v in bbox]
    sub = rgb[max(y0, 0):y1 + 1, max(x0, 0):x1 + 1]
    msk = removed_mask[max(y0, 0):y1 + 1, max(x0, 0):x1 + 1]
    if sub.size == 0 or msk.sum() < 5:
        return None
    px = sub[msk]
    lab = rgb_to_lab(px)
    core = px[lab[:, 0] <= np.percentile(lab[:, 0], 40)]      # darkest 40% = glyph core
    return tuple(core.mean(0).round().astype(int)) if len(core) else None


def _key_color(rgb, removed_mask, ink, bbox, fr, lw_guess=2.0):
    """Colour of the short flat stroke immediately left of a legend text box."""
    x0, y0, x1, y1 = [int(round(v)) for v in bbox]
    band_y0, band_y1 = max(fr.top, y0 - 2), min(fr.bottom, y1 + 2)
    band_x1 = max(fr.left, x0 - 1)
    band_x0 = max(fr.left, int(x0 - 0.25 * fr.width))
    if band_x1 - band_x0 < 4 or band_y1 <= band_y0:
        return None
    sub_ink = ink[band_y0:band_y1 + 1, band_x0:band_x1 + 1]
    if sub_ink.sum() < 6:
        return None
    sub_rgb = rgb[band_y0:band_y1 + 1, band_x0:band_x1 + 1]
    cols = sub_ink.sum(0)
    if (cols > 0).sum() < 5:
        return None
    px = sub_rgb[sub_ink]
    lab = rgb_to_lab(px)
    core = px[lab[:, 0] <= np.percentile(lab[:, 0], 60)]
    return tuple(core.mean(0).round().astype(int)) if len(core) else None


def _de(c1, c2) -> float:
    if c1 is None or c2 is None:
        return 1e3
    a = rgb_to_lab(np.array([c1], np.uint8))[0]
    b = rgb_to_lab(np.array([c2], np.uint8))[0]
    return float(np.linalg.norm(a - b))


def candidate_entries(ocr_items, fr, rgb, removed_mask, ink, lw_guess=2.0):
    """Text boxes inside the axes box that could be legend entries."""
    out = []
    for it in ocr_items:
        x0, y0, x1, y1 = it["bbox"]
        cx, cy = it["cx"], it["cy"]
        if not (fr.left - 2 <= cx <= fr.right + 2 and fr.top - 2 <= cy <= fr.bottom + 2):
            continue
        if parse_number(it["text"]) is not None and len(it["text"].strip()) <= 2:
            continue
        txt = it["text"].strip()
        if len(txt) == 0:
            continue
        w, h = x1 - x0, y1 - y0
        if w < h and h > 0.3 * fr.height:            # rotated axis title
            continue
        if h > 0.25 * fr.height:
            continue
        out.append(dict(text=txt, bbox=it["bbox"], cx=cx, cy=cy, score=it["score"],
                        key_color=_key_color(rgb, removed_mask, ink, it["bbox"], fr, lw_guess),
                        text_color=_text_color(rgb, removed_mask, it["bbox"])))
    return out


def assign_legends(curves, entries, fr, max_color_de=22.0, max_dy_frac=0.12):
    """Hungarian matching of legend entries to curves on a colour+geometry cost."""
    if not curves or not entries:
        return
    n, m = len(entries), len(curves)
    cost = np.zeros((n, m))
    col_de = np.zeros((n, m))
    dy = np.zeros((n, m))
    for i, e in enumerate(entries):
        for j, c in enumerate(curves):
            de = min(_de(e["key_color"], c.rgb), _de(e["text_color"], c.rgb))
            col_de[i, j] = de
            if c.xs.min() <= e["cx"] <= c.xs.max():
                ycur = float(np.interp(e["cx"], c.xs, c.ys))
            else:
                ycur = float(c.ys[0] if e["cx"] < c.xs.min() else c.ys[-1])
            dy[i, j] = abs(ycur - e["cy"]) / max(fr.height, 1)
            cost[i, j] = min(de, 60.0) / 60.0 + 0.7 * min(dy[i, j], 0.5) / 0.5
    ri, ci = linear_sum_assignment(cost)
    for i, j in zip(ri, ci):
        c = curves[j]
        ok_color = col_de[i, j] <= max_color_de
        ok_geom = dy[i, j] <= max_dy_frac
        if not (ok_color or ok_geom):
            continue
        c.legend = entries[i]["text"]
        c.legend_source = ("colour+geometry" if (ok_color and ok_geom)
                           else "colour" if ok_color else "geometry")
        c.legend_color_de = float(col_de[i, j])
        c.legend_dy = float(dy[i, j])


def highlight_visualization(rgb, curve, all_mask, fr, mode="avg"):
    """Paper-style side-by-side: original | curve emphasised, everything else faded."""
    faded = (rgb.astype(np.float32) * 0.25 + 255 * 0.75).astype(np.uint8)
    right = faded.copy()
    m = np.zeros(rgb.shape[:2], np.uint8)
    pts = np.stack([curve.xs, np.round(curve.ys)], 1).astype(np.int32)
    cv2.polylines(m, [pts], False, 1, max(1, int(round(curve.linewidth))))
    sel = (m > 0) & all_mask
    if mode == "avg":
        right[sel] = np.array(curve.rgb, np.uint8)
    else:
        right[sel] = rgb[sel]
    gap = np.full((rgb.shape[0], 8, 3), 255, np.uint8)
    return np.concatenate([rgb, gap, right], axis=1)
