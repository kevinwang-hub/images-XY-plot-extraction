"""Find the individual plot panels inside a figure image.

A published figure is usually a composite: a crystal structure beside a diffractogram,
or a 2x2 grid of isotherms. Digitising it as one image is hopeless, and asking a vision
model for panel boxes gives coordinates that are roughly right and precisely wrong.

Axes, though, are the most geometrically distinctive thing on the page: long straight
runs of ink, at right angles, meeting at a corner. Detecting those directly costs
nothing, splits the composite, and doubles as a filter — a figure in which no axis
system is found is not an xy plot at all, so it never reaches a model.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Panel:
    x0: int
    y0: int
    x1: int
    y1: int              # generous crop, includes tick labels and axis titles
    ax_x0: int
    ax_y0: int
    ax_x1: int
    ax_y1: int           # the axes box itself
    kind: str            # "box" (framed) or "L" (two spines)

    @property
    def box(self):
        return (self.x0, self.y0, self.x1, self.y1)

    @property
    def area(self):
        return (self.x1 - self.x0) * (self.y1 - self.y0)


def _ink(gray: np.ndarray) -> np.ndarray:
    bg = np.median(gray)
    thr = max(40, int(bg) - 45)
    return (gray < thr).astype(np.uint8)


def _segments(ink: np.ndarray, axis: int, min_len: int, thickness: int = 3):
    """Long straight runs of ink along `axis` (0 = horizontal, 1 = vertical)."""
    k = (min_len, 1) if axis == 0 else (1, min_len)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, k)
    opened = cv2.morphologyEx(ink, cv2.MORPH_OPEN, kernel)
    opened = cv2.dilate(opened, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(opened, 8)
    out = []
    for i in range(1, n):
        x, y, w, h = (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
                      stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT])
        if axis == 0 and w >= min_len and h <= max(thickness * 4, 12):
            out.append((y + h // 2, x, x + w))
        elif axis == 1 and h >= min_len and w <= max(thickness * 4, 12):
            out.append((x + w // 2, y, y + h))
    return out


def _graduated(ink: np.ndarray, pos: int, lo: int, hi: int, horizontal: bool,
               depth: int) -> bool:
    """Does this line carry the marks of an axis — tick strokes or a row of numbers?

    This is what separates an axis from any other long straight line. Molecular
    structures are full of long straight bonds that pair up into convincing right
    angles; none of them have ticks or a row of tick labels running alongside.
    """
    span = hi - lo
    if span < 20:
        return False
    for sign in (1, -1):
        a, b = (pos + 2, pos + depth) if sign > 0 else (pos - depth, pos - 2)
        if horizontal:
            band = ink[max(a, 0):max(b, 1), lo:hi]
            prof = band.sum(axis=0)
        else:
            band = ink[lo:hi, max(a, 0):max(b, 1)]
            prof = band.sum(axis=1)
        if band.size == 0 or prof.max() < 1:
            continue
        strong = prof >= max(2, 0.45 * prof.max())
        groups, run = [], None
        for i, v in enumerate(strong):
            if v and run is None:
                run = i
            elif not v and run is not None:
                groups.append((run + i - 1) / 2.0)
                run = None
        if run is not None:
            groups.append((run + len(strong) - 1) / 2.0)
        if len(groups) < 3:
            continue
        if (groups[-1] - groups[0]) < 0.35 * span:
            continue
        d = np.diff(groups)
        if len(d) >= 2 and np.std(d) / max(np.mean(d), 1e-6) > 0.9:
            continue                              # irregular: not a tick/label row
        return True
    return False


def find_panels(rgb: np.ndarray, min_frac: float = 0.16, tol_frac: float = 0.06,
                min_panel_frac: float = 0.025) -> list[Panel]:
    """Axis systems in the image, as generous crops ready for digitisation."""
    H, W = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    ink = _ink(gray)
    hs = _segments(ink, 0, max(30, int(min_frac * W)))
    vs = _segments(ink, 1, max(30, int(min_frac * H)))
    if not hs and not vs:
        return []

    tol_x, tol_y = max(6, int(tol_frac * W)), max(6, int(tol_frac * H))
    cands, paired_h, paired_v = [], set(), set()
    for vi, (vy, vy0, vy1) in enumerate(vs):      # vertical: x=vy, spans vy0..vy1
        for hi, (hy, hx0, hx1) in enumerate(hs):  # horizontal: y=hy, spans hx0..hx1
            # the y axis must end near the x axis, and the x axis start near the y axis
            if abs(hy - vy1) > tol_y or not (vy0 - tol_y <= hy <= vy1 + tol_y):
                continue
            if abs(vy - hx0) > tol_x or not (hx0 - tol_x <= vy <= hx1 + tol_x):
                continue
            ax = (min(vy, hx0), vy0, hx1, max(hy, vy1))
            if (ax[2] - ax[0]) * (ax[3] - ax[1]) < min_panel_frac * W * H:
                continue
            cands.append(ax)
            paired_h.add(hi)
            paired_v.add(vi)

    # A plot may show only the x axis — very common for diffractograms, where the
    # intensity axis is arbitrary units and drawn as no line at all.
    for hi, (hy, hx0, hx1) in enumerate(hs):
        if hi in paired_h or (hx1 - hx0) < 0.25 * W:
            continue
        if not _graduated(ink, hy, hx0, hx1, True, max(8, int(0.05 * H))):
            continue
        top = _ceiling(ink, hy, hx0, hx1, H)
        if hy - top < 0.10 * H:
            continue
        cands.append((hx0, top, hx1, hy))
    for vi, (vx, vy0, vy1) in enumerate(vs):
        if vi in paired_v or (vy1 - vy0) < 0.25 * H:
            continue
        if not _graduated(ink, vx, vy0, vy1, False, max(8, int(0.05 * W))):
            continue
        right = _wall(ink, vx, vy0, vy1, W)
        if right - vx < 0.10 * W:
            continue
        cands.append((vx, vy0, right, vy1))

    if not cands:
        return []
    cands = [ax for ax in cands
             if _graduated(ink, ax[3], ax[0], ax[2], True, max(8, int(0.05 * H)))
             or _graduated(ink, ax[0], ax[1], ax[3], False, max(8, int(0.05 * W)))]
    if not cands:
        return []

    merged: list[list] = []
    for ax in sorted(cands, key=lambda a: -(a[2] - a[0]) * (a[3] - a[1])):
        for m in merged:
            if _iou(ax, m) > 0.35 or _contains(m, ax):
                m[0], m[1] = min(m[0], ax[0]), min(m[1], ax[1])
                m[2], m[3] = max(m[2], ax[2]), max(m[3], ax[3])
                break
        else:
            merged.append(list(ax))

    panels = []
    for ax in merged:
        kind = _kind(ax, hs, vs, tol_x, tol_y)
        crop = _grow(ax, merged, W, H)
        panels.append(Panel(*crop, *ax, kind))
    panels.sort(key=lambda p: (p.ax_y0 // max(1, H // 12), p.ax_x0))
    return panels


def _iou(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(ua, 1)


def _contains(outer, inner) -> bool:
    return (outer[0] - 4 <= inner[0] and outer[1] - 4 <= inner[1]
            and outer[2] + 4 >= inner[2] and outer[3] + 4 >= inner[3])


def _kind(ax, hs, vs, tol_x, tol_y) -> str:
    top = any(abs(hy - ax[1]) <= tol_y and hx0 <= ax[0] + tol_x and hx1 >= ax[2] - tol_x
              for hy, hx0, hx1 in hs)
    right = any(abs(vx - ax[2]) <= tol_x and vy0 <= ax[1] + tol_y and vy1 >= ax[3] - tol_y
                for vx, vy0, vy1 in vs)
    return "box" if (top and right) else "L"


def _grow(ax, others, W, H):
    """Pad the axes box out to include tick labels and axis titles, without eating a
    neighbouring panel."""
    w, h = ax[2] - ax[0], ax[3] - ax[1]
    x0, y0 = ax[0] - 0.30 * w, ax[1] - 0.12 * h
    x1, y1 = ax[2] + 0.08 * w, ax[3] + 0.28 * h
    for o in others:
        if o is ax or list(o) == list(ax):
            continue
        if o[3] < ax[1] and o[2] > ax[0] and o[0] < ax[2]:      # panel above
            y0 = max(y0, o[3] + 2)
        if o[1] > ax[3] and o[2] > ax[0] and o[0] < ax[2]:      # panel below
            y1 = min(y1, o[1] - 2)
        if o[2] < ax[0] and o[3] > ax[1] and o[1] < ax[3]:      # panel left
            x0 = max(x0, o[2] + 2)
        if o[0] > ax[2] and o[3] > ax[1] and o[1] < ax[3]:      # panel right
            x1 = min(x1, o[0] - 2)
    return (int(max(0, x0)), int(max(0, y0)), int(min(W, x1)), int(min(H, y1)))


def crop(rgb: np.ndarray, p: Panel) -> np.ndarray:
    return rgb[p.y0:p.y1, p.x0:p.x1].copy()


def _ceiling(ink: np.ndarray, axis_y: int, x0: int, x1: int, H: int) -> int:
    """Top of a plot that has no top spine: walk up from the x axis until the ink stops
    for a sustained stretch of blank rows."""
    col = ink[:axis_y, x0:x1]
    if col.size == 0:
        return max(0, axis_y - H // 3)
    rows = col.sum(axis=1)
    gap_need = max(6, int(0.045 * H))
    blank = 0
    for y in range(axis_y - 1, -1, -1):
        if rows[y] <= 1:
            blank += 1
            if blank >= gap_need:
                return min(y + blank, axis_y - 1)
        else:
            blank = 0
    return 0


def _wall(ink: np.ndarray, axis_x: int, y0: int, y1: int, W: int) -> int:
    row = ink[y0:y1, axis_x:]
    if row.size == 0:
        return min(W, axis_x + W // 3)
    cols = row.sum(axis=0)
    gap_need = max(6, int(0.045 * W))
    blank = 0
    for i, v in enumerate(cols):
        if v <= 1:
            blank += 1
            if blank >= gap_need:
                return axis_x + max(1, i - blank)
        else:
            blank = 0
    return W
