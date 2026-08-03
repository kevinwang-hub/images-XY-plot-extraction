"""Axis understanding: plot-frame detection, tick detection, OCR-based calibration.

Follows the strategy of He et al. (arXiv 2607.23886): a text/OCR model supplies the
*values and positions* of tick labels, and an independent geometric detector supplies
the *tick mark* positions; only labels that survive the cross-check are kept, and the
surviving pairs define the linear pixel -> data transformation.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field, asdict

import numpy as np

from .core import ink_mask, longest_run, group_consecutive

_OCR = None


def get_ocr():
    global _OCR
    if _OCR is None:
        from rapidocr_onnxruntime import RapidOCR

        _OCR = RapidOCR()
    return _OCR


def run_ocr(rgb: np.ndarray, cache_path: str | None = None) -> list[dict]:
    """PP-OCR (ONNX) text detection+recognition. Returns [{text, score, bbox, cx, cy}]."""
    if cache_path and os.path.exists(cache_path):
        with open(cache_path) as fh:
            return json.load(fh)
    import cv2

    res, _ = get_ocr()(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    items = []
    for box, txt, score in res or []:
        b = np.asarray(box, dtype=np.float32)
        items.append(
            dict(
                text=txt,
                score=float(score),
                bbox=[float(b[:, 0].min()), float(b[:, 1].min()),
                      float(b[:, 0].max()), float(b[:, 1].max())],
                cx=float(b[:, 0].mean()),
                cy=float(b[:, 1].mean()),
            )
        )
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w") as fh:
            json.dump(items, fh)
    return items


# ---------------------------------------------------------------- plot frame

@dataclass
class Frame:
    left: int
    right: int
    top: int
    bottom: int
    has_left: bool = True
    has_bottom: bool = True
    has_right: bool = False
    has_top: bool = False
    # inclusive pixel bands occupied by the frame lines themselves
    left_band: tuple = None
    right_band: tuple = None
    top_band: tuple = None
    bottom_band: tuple = None

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top

    def interior(self, shape) -> tuple[int, int, int, int]:
        """(x0, x1, y0, y1) inclusive box strictly inside the frame lines."""
        H, W = shape
        x0 = (self.left_band[1] + 1) if self.left_band else self.left
        x1 = (self.right_band[0] - 1) if (self.has_right and self.right_band) else self.right
        y0 = (self.top_band[1] + 1) if (self.has_top and self.top_band) else self.top
        y1 = (self.bottom_band[0] - 1) if self.bottom_band else self.bottom
        return max(0, x0), min(W - 1, x1), max(0, y0), min(H - 1, y1)


def _is_axis_line(ink: np.ndarray, band: tuple, vertical: bool) -> bool:
    """Reject look-alikes: an axis line is solid and of constant thickness, whereas a
    tall XRD peak (also a long run) tapers from base to tip."""
    a, b = band
    sub = ink[:, a:b + 1] if vertical else ink[a:b + 1, :]
    along = sub.any(axis=1 if vertical else 0)
    if along.sum() < 8:
        return False
    # the axis line is the longest *contiguous* stretch in the band; a panel label or a
    # tick label sharing the band shows up as a separate stretch and is ignored
    bands = group_consecutive(np.flatnonzero(along), gap=1)
    lo, hi = max(bands, key=lambda b: b[1] - b[0])
    if (hi - lo + 1) < 0.35 * len(along):
        return False
    width = (sub[lo:hi + 1] if vertical else sub[:, lo:hi + 1].T)
    w = width.sum(axis=1).astype(float)
    if w.size == 0 or w.max() == 0:
        return False
    # robust uniformity: curves that touch the axis add outliers, so require most rows to
    # sit at the same thickness rather than a small overall spread. A tall XRD spike
    # tapers from base to tip and fails this.
    wm = float(np.median(w))
    return bool(wm > 0 and np.mean(np.abs(w - wm) <= max(1.0, 0.5 * wm)) >= 0.75)


def _bounds_data(inner: np.ndarray, band: tuple, side: str, textless: np.ndarray,
                 frac: float = 0.08) -> bool:
    """A frame line must actually *bound* the data: little non-text ink beyond it.

    Without this, a clipped flat-top XRD peak - solid and of constant width - passes as
    an axis line and truncates the plot area.
    """
    a, b = band
    total = max(textless.sum(), 1)
    outside = {"left": textless[:, :a], "right": textless[:, b + 1:],
               "top": textless[:a], "bottom": textless[b + 1:]}[side]
    return bool(outside.sum() <= frac * total)


def detect_frame(ink: np.ndarray, ocr_items=None) -> Frame:
    """Locate the axes box from long straight ink lines (robust to L-shaped axes)."""
    H, W = ink.shape
    inner = ink.copy()                 # ignore lines that are image-border artefacts
    m = 3
    inner[:m], inner[-m:], inner[:, :m], inner[:, -m:] = False, False, False, False
    if not inner.any():
        inner = ink

    v = longest_run(inner, axis=0)      # per-column longest vertical run
    h = longest_run(inner, axis=1)      # per-row longest horizontal run

    vcols = np.flatnonzero(v >= max(0.35 * H, 0.75 * v.max()))
    hrows = np.flatnonzero(h >= max(0.35 * W, 0.75 * h.max()))
    vb = [b for b in group_consecutive(vcols, gap=3) if _is_axis_line(inner, b, True)]
    hb = [b for b in group_consecutive(hrows, gap=3) if _is_axis_line(inner, b, False)]

    textless = inner.copy()
    for it in (ocr_items or []):
        x0, y0, x1, y1 = [int(round(v)) for v in it["bbox"]]
        textless[max(y0 - 1, 0):y1 + 2, max(x0 - 1, 0):x1 + 2] = False

    ys, xs = np.nonzero(ink)
    ix0, ix1, iy0, iy1 = xs.min(), xs.max(), ys.min(), ys.max()

    lcand = [b for b in vb if _bounds_data(inner, b, "left", textless)]
    rcand = [b for b in vb if _bounds_data(inner, b, "right", textless)]
    tcand = [b for b in hb if _bounds_data(inner, b, "top", textless)]
    bcand = [b for b in hb if _bounds_data(inner, b, "bottom", textless)]

    lb = lcand[0] if lcand else None
    rb = rcand[-1] if rcand else None
    tb = tcand[0] if tcand else None
    bb = bcand[-1] if bcand else None

    left = int(np.mean(lb)) if lb else int(ix0)
    right = int(np.mean(rb)) if rb else int(ix1)
    top = int(np.mean(tb)) if tb else int(iy0)
    bottom = int(np.mean(bb)) if bb else int(iy1)
    if rb and (right - left) <= 0.3 * W:
        right, rb = int(ix1), None
    if tb and (bottom - top) <= 0.3 * H:
        top, tb = int(iy0), None

    return Frame(left, right, top, bottom, lb is not None, bb is not None,
                 rb is not None, tb is not None,
                 left_band=lb, right_band=rb, top_band=tb, bottom_band=bb)


# ---------------------------------------------------------------- tick marks

def detect_ticks_x(ink: np.ndarray, fr: Frame) -> np.ndarray:
    """Tick x-positions on the bottom axis (checks inside- and outside-pointing ticks)."""
    H, W = ink.shape
    tl = max(3, int(round(0.018 * fr.height)))
    cands = []
    for band in (slice(max(fr.top, fr.bottom - tl - 1), max(fr.top + 1, fr.bottom - 1)),
                 slice(min(H - 1, fr.bottom + 2), min(H, fr.bottom + tl + 2))):
        sub = ink[band, fr.left:fr.right + 1]
        if sub.size == 0:
            continue
        prof = sub.sum(axis=0).astype(float)
        if prof.max() < 2:
            continue
        strong = prof >= max(2.0, 0.6 * prof.max())
        pos = [fr.left + int(np.mean(b)) for b in group_consecutive(np.flatnonzero(strong), gap=2)]
        cands.append(np.array(pos, dtype=float))
    if not cands:
        return np.array([])
    # prefer the band whose positions are the most regularly spaced
    def regularity(p):
        if len(p) < 3:
            return -1.0
        d = np.diff(np.sort(p))
        return -float(np.std(d) / max(np.mean(d), 1e-6)) + 0.01 * len(p)

    return np.sort(max(cands, key=regularity))


def detect_ticks_y(ink: np.ndarray, fr: Frame) -> np.ndarray:
    H, W = ink.shape
    tl = max(3, int(round(0.018 * fr.width)))
    cands = []
    for band in (slice(min(fr.right, fr.left + 2), min(fr.right, fr.left + tl + 2)),
                 slice(max(0, fr.left - tl - 2), max(1, fr.left - 1))):
        sub = ink[fr.top:fr.bottom + 1, band]
        if sub.size == 0:
            continue
        prof = sub.sum(axis=1).astype(float)
        if prof.max() < 2:
            continue
        strong = prof >= max(2.0, 0.6 * prof.max())
        pos = [fr.top + int(np.mean(b)) for b in group_consecutive(np.flatnonzero(strong), gap=2)]
        cands.append(np.array(pos, dtype=float))
    if not cands:
        return np.array([])

    def regularity(p):
        if len(p) < 3:
            return -1.0
        d = np.diff(np.sort(p))
        return -float(np.std(d) / max(np.mean(d), 1e-6)) + 0.01 * len(p)

    return np.sort(max(cands, key=regularity))


# ---------------------------------------------------------------- OCR parsing

_NUM = re.compile(r"^[\s]*[-−–]?\s*\d{1,4}(?:[.,]\d{1,3})?[\s]*$")


def parse_number(txt: str):
    t = txt.strip().replace("−", "-").replace("–", "-").replace(" ", "")
    t = t.replace(",", ".") if t.count(",") == 1 and "." not in t else t.replace(",", "")
    if not re.fullmatch(r"-?\d{1,4}(?:\.\d{1,3})?", t):
        return None
    try:
        return float(t)
    except ValueError:
        return None


def theilsen(px: np.ndarray, val: np.ndarray) -> tuple[float, float]:
    """Median-of-pairwise-slopes fit (robust to a mis-OCR'd label)."""
    n = len(px)
    slopes = [(val[j] - val[i]) / (px[j] - px[i])
              for i in range(n) for j in range(i + 1, n) if abs(px[j] - px[i]) > 1e-6]
    a = float(np.median(slopes))
    b = float(np.median(val - a * px))
    return a, b


@dataclass
class AxisCalib:
    label: str = ""
    ticks: list = field(default_factory=list)      # [(pixel, value)] used for the fit
    a: float = 1.0                                  # value = a*pixel + b
    b: float = 0.0
    calibrated: bool = False
    residual: float = 0.0                            # max |fit - label| in data units
    n_labels_seen: int = 0
    n_labels_used: int = 0
    note: str = ""

    def to_data(self, pixel):
        return self.a * np.asarray(pixel, dtype=float) + self.b

    def to_pixel(self, value):
        return (np.asarray(value, dtype=float) - self.b) / self.a


def _calibrate(labels: list[tuple[float, float]], ticks: np.ndarray,
               span_px: float) -> AxisCalib:
    """labels: [(pixel_of_label_centre, value)]; snapped to detected ticks when close."""
    cal = AxisCalib(n_labels_seen=len(labels))
    if len(labels) < 2:
        cal.note = "fewer than 2 numeric tick labels found"
        return cal

    labels = sorted(labels)
    px = np.array([p for p, _ in labels], dtype=float)
    val = np.array([v for _, v in labels], dtype=float)

    # cross-check: snap each label to the nearest detected tick mark
    tol = max(4.0, 0.02 * span_px)
    if len(ticks):
        snapped = []
        for p in px:
            j = int(np.argmin(np.abs(ticks - p)))
            snapped.append(ticks[j] if abs(ticks[j] - p) <= tol else p)
        px = np.array(snapped, dtype=float)

    a, b = theilsen(px, val)
    if abs(a) < 1e-12:
        cal.note = "degenerate fit"
        return cal
    resid = np.abs(a * px + b - val)
    scale = max(abs(a) * span_px, 1e-9)
    keep = resid <= max(0.02 * scale, 1e-9)
    if keep.sum() >= 2 and keep.sum() < len(px):
        a, b = theilsen(px[keep], val[keep])
        resid = np.abs(a * px + b - val)
        keep = resid <= max(0.02 * scale, 1e-9)
    if keep.sum() < 2:
        cal.note = "tick labels inconsistent with a linear axis"
        return cal

    cal.a, cal.b = float(a), float(b)
    cal.ticks = [[float(p), float(v)] for p, v in zip(px[keep], val[keep])]
    cal.n_labels_used = int(keep.sum())
    cal.residual = float(resid[keep].max())
    cal.calibrated = True
    return cal


def calibrate_axes(rgb: np.ndarray, ink: np.ndarray, fr: Frame,
                   ocr_items: list[dict]) -> tuple[AxisCalib, AxisCalib, dict]:
    """Return (x_calib, y_calib, extras) where extras holds axis-label text etc."""
    H, W = ink.shape
    tx = detect_ticks_x(ink, fr)
    ty = detect_ticks_y(ink, fr)

    band = max(10.0, 0.10 * H)
    xlab, ylab = [], []
    for it in ocr_items:
        x0, y0, x1, y1 = it["bbox"]
        v = parse_number(it["text"])
        if v is None:
            continue
        below = fr.bottom - 2 <= it["cy"] <= fr.bottom + band
        left_of = (fr.left - band <= it["cx"] <= fr.left + 2) and (fr.top - 5 <= it["cy"] <= fr.bottom + 5)
        if below and fr.left - band <= it["cx"] <= fr.right + band:
            xlab.append((it["cx"], v))
        elif left_of:
            ylab.append((it["cy"], v))

    xcal = _calibrate(xlab, tx, fr.width)
    ycal = _calibrate(ylab, ty, fr.height)

    xcal.label, ycal.label = _axis_label_text(ocr_items, fr, H, W)
    return xcal, ycal, dict(x_ticks_detected=tx.tolist(), y_ticks_detected=ty.tolist(),
                            x_label_candidates=xlab, y_label_candidates=ylab)


_GREEK_FIX = [
    (re.compile(r"^2\s*[0Oo0θΘ]\s*", re.I), "2θ "),
    (re.compile(r"2\s*theta", re.I), "2θ"),
]


def normalize_axis_text(t: str) -> str:
    t = t.strip()
    for pat, rep in _GREEK_FIX:
        if pat.search(t):
            t = pat.sub(rep, t, count=1)
            break
    return re.sub(r"\s+", " ", t).strip()


def _axis_label_text(ocr_items, fr: Frame, H: int, W: int) -> tuple[str, str]:
    """Pick the x/y axis label: the non-numeric text furthest outside the frame."""
    xcand, ycand = [], []
    for it in ocr_items:
        x0, y0, x1, y1 = it["bbox"]
        w, h = x1 - x0, y1 - y0
        if parse_number(it["text"]) is not None:
            continue
        if len(it["text"].strip()) < 2:
            continue
        if it["cy"] > fr.bottom and w >= h:           # under the x axis, horizontal text
            xcand.append((it["cy"], it["text"]))
        if it["cx"] < fr.left and h > w:              # left of the y axis, vertical text
            ycand.append((-it["cx"], it["text"]))
    xl = normalize_axis_text(max(xcand)[1]) if xcand else ""
    # y label may be split into several vertical boxes ("Intensity", "a.u.")
    if ycand:
        ycand.sort(reverse=True)
        parts, x_ref = [], None
        for key, t in ycand:
            if x_ref is None or abs(key - x_ref) < 12:
                x_ref = key if x_ref is None else x_ref
                parts.append(t.strip())
        yl = normalize_axis_text(" ".join(parts))
    else:
        yl = ""
    return xl, yl
