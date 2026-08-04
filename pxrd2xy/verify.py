"""Visual verifier: re-plot the digitised curve and measure how well it overlaps
the ink actually drawn in the source figure.

Metrics (all in [0,1] unless stated):

  point_on_ink   fraction of digitised points that land on ink of the curve's own
                 colour - the primary "is every point on the right line" test
  colour_match   fraction of digitised points whose pixel colour belongs to the
                 curve's own colour cluster (catches a trace that hopped onto a
                 neighbouring curve)
  overlap_iou    round-trip overlap: the extracted xy data are re-plotted with
                 matplotlib in the geometry of the source axes and re-rasterised, then
                 compared with the original ink (IoU / precision / recall). This is the
                 "how well do the lines lie on top of each other" number, and it is not
                 circular - it exercises the axis transform and the stroke width too.
  coverage       fraction of the plot width spanned by the trace
  mean_dev_pct   mean distance from the trace to the nearest same-colour ink, as a
  p99_dev_pct    percentage of the plot height (the paper's <1% mean / <3% p99 test)
  explained      figure level: fraction of all data ink explained by the curves
"""
from __future__ import annotations

import numpy as np
import cv2


def render_trace(shape, xs, ys, lw: float) -> np.ndarray:
    """Perpendicular-stroke rendering (used for the visual overlay)."""
    m = np.zeros(shape, np.uint8)
    pts = np.stack([np.asarray(xs), np.round(np.asarray(ys))], 1).astype(np.int32)
    if len(pts) >= 2:
        cv2.polylines(m, [pts], False, 1, max(1, int(round(lw))))
    return m.astype(bool)


def render_trace_columns(shape, xs, ys, lw: float, reach: str = "mid") -> np.ndarray:
    """Re-rasterise the curve as a function plot: for every column, the vertical span
    the pen covers.

    reach="mid"  span to the midpoints of the neighbouring samples - how a rasteriser
                 actually fills a column, so this is the *precision* reference.
    reach="full" span all the way to the neighbouring samples. A steep flank is drawn
                 in both of the columns it passes through, so this is the *recall*
                 reference; it is what a sharp XRD peak legitimately occupies.
    """
    H, W = shape
    m = np.zeros(shape, bool)
    xs = np.asarray(xs, dtype=int)
    ys = np.asarray(ys, dtype=float)
    n = len(xs)
    half = max(lw, 1.0) / 2.0
    f = 1.0 if reach == "full" else 0.5
    for i in range(n):
        lo = hi = ys[i]
        if i > 0 and xs[i] - xs[i - 1] <= 2:
            y = ys[i] + f * (ys[i - 1] - ys[i])
            lo, hi = min(lo, y), max(hi, y)
        if i < n - 1 and xs[i + 1] - xs[i] <= 2:
            y = ys[i] + f * (ys[i + 1] - ys[i])
            lo, hi = min(lo, y), max(hi, y)
        a = int(np.clip(np.floor(lo - half), 0, H - 1))
        b = int(np.clip(np.ceil(hi + half), 0, H - 1))
        m[a:b + 1, xs[i]] = True
    return m


def envelope_band(mask: np.ndarray, lw: float) -> np.ndarray:
    """The pen stroke along the upper edge of a curve's ink.

    A single-valued trace can only reproduce the *upper edge* of the plotted line:
    inside a sharp peak the pen fills a whole column, and no y(x) can cover that area.
    Comparing the re-plot against this band therefore compares like with like.
    """
    H, W = mask.shape
    out = np.zeros_like(mask)
    t = max(1, int(round(lw)))
    for x in range(W):
        col = np.flatnonzero(mask[:, x])
        if col.size == 0:
            continue
        for a, b in _runs(col):
            out[a:min(b, a + t - 1) + 1, x] = True
    return out


def _runs(idx):
    from .core import group_consecutive
    return group_consecutive(idx, gap=1)


def stroke_width(mask: np.ndarray) -> float:
    """Pen width of a stroke mask via the distance transform (stroke-width transform)."""
    if not mask.any():
        return 2.0
    dt = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    v = dt[mask]
    return float(max(1.0, round(2.0 * np.percentile(v, 92) - 1.0)))


def _nearest_ink_dev(mask_cluster: np.ndarray, xs, ys) -> np.ndarray:
    """Per-column distance (px) from the trace to the nearest ink of its own colour."""
    dev = []
    for x, y in zip(xs, ys):
        col = np.flatnonzero(mask_cluster[:, int(x)])
        dev.append(float(np.min(np.abs(col - y))) if col.size else np.nan)
    return np.array(dev)


def roundtrip_render(curves, fr, xcal, ycal, shape, dpi: int = 100):
    """Re-plot the *data* (not the pixels) with matplotlib in the geometry of the source
    axes box, and rasterise it. Returns (per-curve masks, combined mask).

    This closes the loop: pixel -> data -> pixel. Every stage the digitiser applied
    (axis transform, trace, stroke width) has to be right for the re-rendered line to
    fall back onto the ink of the original figure.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    x0, x1, y0, y1 = fr.interior(shape)
    W, H = x1 - x0 + 1, y1 - y0 + 1
    xlo, xhi = float(xcal.to_data(x0 - 0.5)), float(xcal.to_data(x1 + 0.5))
    ylo, yhi = float(ycal.to_data(y1 + 0.5)), float(ycal.to_data(y0 - 0.5))

    masks = []
    for c in curves:
        fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi)
        axp = fig.add_axes([0, 0, 1, 1])
        axp.set_xlim(xlo, xhi)
        axp.set_ylim(ylo, yhi)
        axp.axis("off")
        if getattr(c, "style", "line") == "markers":
            axp.plot(c.data_x, c.data_y, linestyle="none", marker="o", color="k",
                     markersize=max(getattr(c, "marker_px", 4.0), 2.0) * 72.0 / dpi,
                     markeredgewidth=0, antialiased=True)
        else:
            axp.plot(c.data_x, c.data_y, lw=max(c.linewidth, 1.0) * 72.0 / dpi, color="k",
                     solid_capstyle="projecting", solid_joinstyle="miter",
                     antialiased=True)
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        buf = np.asarray(canvas.buffer_rgba())[:, :, :3]
        plt.close(fig)
        m = np.zeros(shape, bool)
        sub = buf.min(axis=2) < 215
        if sub.shape != (H, W):
            sub = cv2.resize(sub.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
        m[y0:y1 + 1, x0:x1 + 1] = sub
        masks.append(m)
    comb = np.zeros(shape, bool)
    for m in masks:
        comb |= m
    return masks, comb


def roundtrip_metrics(synth: np.ndarray, assigned: np.ndarray, cluster_ink: np.ndarray,
                      plot_ink: np.ndarray, fr, lw: float = 3.0) -> dict:
    """Overlap of the re-rendered line with the ink of the original figure."""
    inter = (synth & assigned).sum()
    prec = inter / max(synth.sum(), 1)
    rec = inter / max(assigned.sum(), 1)
    iou = inter / max((synth | assigned).sum(), 1)
    on_ink = (synth & plot_ink).sum() / max(synth.sum(), 1)
    on_same_colour = (synth & cluster_ink).sum() / max(synth.sum(), 1)

    # Column-wise upper-edge residual, in % of the plot height (paper's <1% / <3% test).
    # Measured only where the value is *resolvable*: in a column where the pen fills a
    # whole peak the plotted line has no single height, so such columns are excluded and
    # reported separately instead of being charged as error.
    Hp = max(fr.height, 1)
    lw_ref = max(3.0 * max(lw, 1.0), 4.0)
    dev, n_steep, n_tot = [], 0, 0
    W = synth.shape[1]
    tops = np.full(W, -1, np.int32)
    for x in range(W):
        b = np.flatnonzero(assigned[:, x])
        if b.size:
            tops[x] = b[0]
    for x in range(W):
        a = np.flatnonzero(synth[:, x])
        b = np.flatnonzero(assigned[:, x])
        if not (a.size and b.size):
            continue
        n_tot += 1
        run = next((r for r in _runs(b) if r[0] <= b[0] <= r[1]), (b[0], b[-1]))
        if (run[1] - run[0] + 1) > lw_ref:
            n_steep += 1
            continue
        # a one-column tolerance: the digitised x carries +-0.5 px of quantisation, and on a
        # steep flank half a column of x is a large amount of y
        cands = [tops[xx] for xx in (x - 1, x, x + 1) if 0 <= xx < W and tops[xx] >= 0]
        dev.append(min(abs(int(a[0]) - int(t)) for t in cands))
    dev = np.array(dev, dtype=float)
    return dict(overlap_iou=float(iou), overlap_precision=float(prec),
                overlap_recall=float(rec), on_ink=float(on_ink),
                on_same_colour=float(on_same_colour),
                mean_dev_pct=float(dev.mean() / Hp * 100) if dev.size else 100.0,
                p99_dev_pct=float(np.percentile(dev, 99) / Hp * 100) if dev.size else 100.0,
                n_cols_compared=int(dev.size),
                steep_frac=float(n_steep / max(n_tot, 1)))


def _edge_deviation(assigned: np.ndarray, xs, ys, lw: float) -> np.ndarray:
    """Signed-magnitude distance from each digitised point to the upper edge of the
    plotted stroke in its own column (the quantity the trace is meant to reproduce)."""
    out = []
    ref = lw / 2.0
    for x, y in zip(np.asarray(xs, int), np.asarray(ys, float)):
        col = np.flatnonzero(assigned[:, x])
        if col.size == 0:
            out.append(np.nan)
            continue
        best = None
        for a, b in _runs(col):
            d = 0.0 if a - 1 <= y <= b + 1 else min(abs(y - a), abs(y - b))
            if best is None or d < best[0]:
                best = (d, a)
        out.append(abs(y - (best[1] + ref)))
    return np.array(out)


def verify_curve(curve, plot_ink: np.ndarray, labimg: np.ndarray, assigned: np.ndarray,
                 fr) -> dict:
    H, W = plot_ink.shape
    Hp = max(fr.height, 1)
    cluster_mask = (labimg == curve.cluster)
    xi = np.asarray(curve.xs, dtype=int)
    yi = np.clip(np.round(curve.ys).astype(int), 0, H - 1)

    point_on_own = float(cluster_mask[yi, xi].mean())
    point_on_any = float(plot_ink[yi, xi].mean())
    colour_match = float((labimg[yi, xi] == curve.cluster).mean())

    return dict(point_on_ink=point_on_own, point_on_any_ink=point_on_any,
                colour_match=colour_match, coverage=float(curve.coverage),
                n_points=int(len(curve.xs)))


def combine_metrics(pix: dict, rt: dict, style: str = "line") -> dict:
    """Merge the pixel-level checks with the round-trip overlap into one score."""
    m = dict(pix)
    m.update(rt)
    m["style"] = style
    m["overlap_score"] = float(
        0.25 * m["point_on_ink"] + 0.15 * m["colour_match"] + 0.20 * m["on_ink"]
        + 0.15 * min(m["overlap_iou"] / 0.85, 1.0)
        + 0.15 * min(m["coverage"] / 0.95, 1.0)
        + 0.10 * (1.0 if m["mean_dev_pct"] <= 1.0 else 0.0))
    return m


def curve_status(m: dict) -> str:
    """Pass criteria mirror the paper's calibration of "visual agreement": mean deviation
    < 1% of the intensity range and < 3% for 99% of points, plus every digitised point
    sitting on ink of the curve's own colour and near-full width coverage.

    Area IoU is reported but only carries a weak floor: where the pen fills a whole peak
    column, the overlap of a line with that filled area is limited by pen geometry rather
    than by digitisation error, so gating on it would penalise spike-dense figures for
    being spike-dense.

    For a symbol-drawn series the floor is dropped entirely. The round trip re-plots every
    series as circles, so a correctly read series of triangles or open squares is bounded
    by the overlap of a circle with a triangle no matter how exact the coordinates are --
    that number measures the symbol shape, not the reading. What still has to hold for
    symbols is the part that does mean something: every point on its own ink."""
    # For a symbol series the round trip re-plots every point as a disc, so both
    # area-overlap and lands-on-ink measure the shape of the symbol rather than the
    # accuracy of the reading: the centre of an *open* circle is background, and a
    # perfectly read ring scores as landing off ink. What still has to hold, and does
    # all the work for symbols, is point_on_ink -- every digitised point sitting on ink
    # of its own colour, which is exact and shape-independent.
    markers = m.get("style") == "markers"
    iou_floor = 0.0 if markers else 0.45
    ink_floor = 0.35 if markers else 0.65
    if (m["point_on_ink"] >= 0.97 and m["colour_match"] >= 0.95 and m["on_ink"] >= ink_floor
            and m["mean_dev_pct"] <= 1.0 and m["p99_dev_pct"] <= 3.0
            and m["coverage"] >= 0.85 and m["overlap_iou"] >= iou_floor):
        return "pass"
    if (m["point_on_ink"] >= 0.88 and m["on_ink"] >= (0.25 if markers else 0.50)
            and m["mean_dev_pct"] <= 2.5 and m["coverage"] >= 0.60):
        return "warn"
    return "fail"


GATES = [
    ("point_on_ink", 0.97, "ge", "points off own-colour ink"),
    ("colour_match", 0.95, "ge", "colour mismatch (may have hopped curves)"),
    ("on_ink", 0.65, "ge", "re-plot lands off ink"),
    ("mean_dev_pct", 1.0, "le", "mean deviation > 1% of plot height"),
    ("p99_dev_pct", 3.0, "le", "peak-limited: worst-1% deviation > 3%"),
    ("coverage", 0.85, "ge", "does not span the plot width"),
    ("overlap_iou", 0.45, "ge", "low area overlap"),
]


def status_reason(m: dict) -> str:
    """Which pass gate(s) a curve missed, in plain words."""
    bad = []
    for key, thr, op, why in GATES:
        if key in ("overlap_iou", "on_ink") and m.get("style") == "markers":
            continue
        v = m.get(key)
        if v is None:
            continue
        if (op == "ge" and v < thr) or (op == "le" and v > thr):
            bad.append(f"{why} ({v:.2f})")
    return "; ".join(bad) if bad else "all checks passed"


def render_points(shape, xs, ys, size: float) -> np.ndarray:
    """A scatter re-rasterised as discs at its data points, and nowhere else."""
    m = np.zeros(shape, np.uint8)
    r = max(1, int(round(size * 0.5)))
    for x, y in zip(np.asarray(xs), np.asarray(ys)):
        cv2.circle(m, (int(round(x)), int(round(y))), r, 1, -1)
    return m.astype(bool)


def _render_curve(c, shape, grow: float = 0.0) -> np.ndarray:
    if getattr(c, "style", "line") == "markers":
        return render_points(shape, c.xs, c.ys,
                             getattr(c, "marker_px", c.linewidth) + grow)
    return render_trace_columns(shape, c.xs, c.ys, c.linewidth + grow, "full")


def _union_loose(curves, shape, grow: float = 2.0) -> np.ndarray:
    acc = np.zeros(shape, bool)
    for c in curves:
        acc |= _render_curve(c, shape, grow)
    return acc


def explained_fraction(curves, plot_ink, shape, grow: float = 2.0,
                       ignore: np.ndarray | None = None) -> float:
    """Fraction of the figure's ink *runs* claimed by some extracted curve.

    Counted per (column, run) rather than per pixel: a sharp peak is one tall run and
    is explained as soon as a trace passes through it, while a curve that was missed
    entirely leaves all of its runs unclaimed. That makes this a completeness measure
    rather than a restatement of the spike-area artefact.
    """
    H, W = shape
    tol = 1 + int(round(grow))
    ypos = {}
    for c in curves:
        for x, y in zip(np.asarray(c.xs, int), np.asarray(c.ys, float)):
            ypos.setdefault(int(x), []).append(y)
    tot = hit = 0
    for x in range(W):
        col = np.flatnonzero(plot_ink[:, x])
        if col.size == 0:
            continue
        for a, b in _runs(col):
            if ignore is not None and ignore[(a + b) // 2, x]:
                continue                       # figure text is not data
            tot += 1
            for y in ypos.get(x, ()):
                if a - tol <= y <= b + tol:
                    hit += 1
                    break
    return float(hit / max(tot, 1))


def unexplained_mask(curves, plot_ink, shape, grow: float = 2.0,
                     ignore: np.ndarray | None = None) -> np.ndarray:
    """Ink runs that no trace passes through - the same criterion as explained_fraction.

    Marking un-*covered* pixels instead would paint the inside of every sharp peak, which
    a single-valued y(x) can never cover; that is a property of the representation, not a
    missed curve.
    """
    H, W = shape
    tol = 1 + int(round(grow))
    ypos: dict[int, list] = {}
    for c in curves:
        for x, y in zip(np.asarray(c.xs, int), np.asarray(c.ys, float)):
            ypos.setdefault(int(x), []).append(y)
    out = np.zeros(shape, bool)
    for x in range(W):
        col = np.flatnonzero(plot_ink[:, x])
        if col.size == 0:
            continue
        for a, b in _runs(col):
            if ignore is not None and ignore[(a + b) // 2, x]:
                continue
            if not any(a - tol <= y <= b + tol for y in ypos.get(x, ())):
                out[a:b + 1, x] = True
    return out


# --------------------------------------------------------------------- visuals

def overlay_image(rgb, curves, fr, xcal, ycal, show_ticks: bool = True) -> np.ndarray:
    """Faded original + re-drawn traces + the tick calibration that was applied."""
    out = (rgb.astype(np.float32) * 0.22 + 255 * 0.78).astype(np.uint8)
    cv2.rectangle(out, (fr.left, fr.top), (fr.right, fr.bottom), (170, 170, 165), 1)
    for c in curves:
        pts = np.stack([c.xs, np.round(c.ys)], 1).astype(np.int32)
        col = tuple(int(v) for v in c.rgb)
        if getattr(c, "style", "line") == "markers":
            # A scatter is drawn as a scatter. Joining its points with a line would show
            # values between the measurements, which is exactly what was not read.
            r = max(2, int(round(getattr(c, "marker_px", 4.0) * 0.45)))
            for x, y in pts:
                cv2.circle(out, (int(x), int(y)), r, col, -1, cv2.LINE_AA)
            continue
        cv2.polylines(out, [pts], False, col, max(1, int(round(c.linewidth * 0.5))),
                      cv2.LINE_AA)
    if show_ticks and xcal.calibrated:
        for px, val in xcal.ticks:
            px = int(round(px))
            cv2.line(out, (px, fr.bottom - 7), (px, fr.bottom + 7), (235, 104, 52), 1)
            cv2.putText(out, f"{val:g}", (px - 8, fr.bottom + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 80, 30), 1, cv2.LINE_AA)
    return out


def diff_image(rgb, curves, plot_ink, fr, ignore=None) -> np.ndarray:
    """Green = digitised points on the right ink, red = off-ink, blue = ink nobody claimed.

    Recognised figure text is excluded - it is annotation, not data."""
    out = (rgb.astype(np.float32) * 0.18 + 255 * 0.82).astype(np.uint8)
    miss = unexplained_mask(curves, plot_ink, plot_ink.shape, ignore=ignore)
    out[miss] = [42, 122, 214]
    H = plot_ink.shape[0]
    for c in curves:
        xi = np.asarray(c.xs, dtype=int)
        yi = np.clip(np.round(c.ys).astype(int), 0, H - 1)
        ok = plot_ink[yi, xi]
        out[yi[ok], xi[ok]] = [27, 175, 122]
        out[yi[~ok], xi[~ok]] = [227, 73, 72]
    return out


def roundtrip_image(rgb, synth: np.ndarray, plot_ink: np.ndarray, fr) -> np.ndarray:
    """Overlap map of the re-plotted data against the original ink.

    grey = both, red = re-plot only, blue = original ink only. Inside a sharp peak the
    original ink fills the whole column while the re-plot draws a line, so some blue there
    is expected and is not a digitisation error."""
    out = np.full(rgb.shape, 255, np.uint8)
    both = synth & plot_ink
    only_s = synth & ~plot_ink
    only_o = plot_ink & ~synth
    out[both] = [110, 110, 108]
    out[only_s] = [227, 73, 72]
    out[only_o] = [42, 122, 214]
    cv2.rectangle(out, (fr.left, fr.top), (fr.right, fr.bottom), (200, 200, 196), 1)
    return out


def sidebyside(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    h = max(a.shape[0], b.shape[0])

    def pad(im):
        out = np.full((h, im.shape[1], 3), 255, np.uint8)
        out[:im.shape[0]] = im
        return out

    gap = np.full((h, 10, 3), 235, np.uint8)
    return np.concatenate([pad(a), gap, pad(b)], axis=1)
