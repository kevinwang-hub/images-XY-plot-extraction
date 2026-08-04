"""Curve separation and digitisation.

Implements the two-step scheme of He et al. (arXiv 2607.23886), adapted to powder XRD:

  1. colour decomposition of the plot area with BIRCH, giving clusters of
     similarly-coloured pixels;
  2. a line-segment connectivity graph, where a *segment* is a chain of
     column-wise ink runs between branch points, and each candidate curve is the
     combination of segments that maximises

         reward(l) = |l ∩ C_colour| / (W * w_hat)  -  (lambda / N) * sum |f''(x_i)|

     i.e. colour consistency minus a roughness penalty that suppresses the sharp
     kinks produced when segments of *different* curves are joined.

The maximisation is a dynamic program over the DAG of segments ordered along x.
Pixels of a non-matching colour contribute negatively, so a path is only extended
through a segment when that segment really belongs to the curve's colour.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import cv2

from .core import group_consecutive, rgb_to_lab

LAMBDA_ROUGH = 0.01          # lambda in Eq. 1 (paper's empirical value)
import os
JUMP_PENALTY = float(os.environ.get("PXRD_JUMP", 1.5))   # vertical-discontinuity weight
KINK_PENALTY = float(os.environ.get("PXRD_KINK", 0.05))  # junction-curvature weight


# ------------------------------------------------------------------ plot area

def plot_ink(ink: np.ndarray, fr, ticks_x=None, ticks_y=None) -> np.ndarray:
    """Ink strictly inside the frame lines, with inward-pointing tick marks removed."""
    x0, x1, y0, y1 = fr.interior(ink.shape)
    m = np.zeros_like(ink)
    m[y0:y1 + 1, x0:x1 + 1] = ink[y0:y1 + 1, x0:x1 + 1]

    tl = max(3, int(round(0.025 * fr.height)))
    for tx in (ticks_x if ticks_x is not None else []):
        tx = int(round(tx))
        for x in range(max(x0, tx - 2), min(x1, tx + 2) + 1):
            col = np.flatnonzero(m[:, x])
            if col.size == 0:
                continue
            for a, b in group_consecutive(col, gap=1):
                if b >= y1 - 1 and (b - a) <= tl:
                    m[a:b + 1, x] = False
    tlx = max(3, int(round(0.025 * fr.width)))
    for ty in (ticks_y if ticks_y is not None else []):
        ty = int(round(ty))
        for y in range(max(y0, ty - 2), min(y1, ty + 2) + 1):
            row = np.flatnonzero(m[y])
            if row.size == 0:
                continue
            for a, b in group_consecutive(row, gap=1):
                if a <= x0 + 1 and (b - a) <= tlx:
                    m[y, a:b + 1] = False
    return m


def remove_text_and_legend(mask: np.ndarray, ocr_items, fr, lw_guess: float = 2.0):
    """Drop connected components that are figure text, panel letters or legend keys.

    Text is identified by overlap with OCR boxes and by shape; legend key lines are
    short flat components sitting immediately left of an OCR box on the same row.
    """
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    Wp, Hp = max(fr.width, 1), max(fr.height, 1)
    total = mask.sum()
    boxes = [it["bbox"] for it in ocr_items]
    removed = np.zeros_like(mask)
    for i in range(1, n):
        x, y, w, h, area = (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
                            stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT],
                            stats[i, cv2.CC_STAT_AREA])
        if area > 0.02 * total and w > 0.25 * Wp:
            continue                                          # a real curve backbone
        cc_box = (x, y, x + w, y + h)
        drop = False
        # (a) sits inside / mostly inside a recognised text box
        for bx in boxes:
            ox = max(0, min(cc_box[2], bx[2]) - max(cc_box[0], bx[0]))
            oy = max(0, min(cc_box[3], bx[3]) - max(cc_box[1], bx[1]))
            if ox * oy > 0.55 * w * h:
                drop = True
                break
        # (b) short flat stroke just left of a text box -> legend key line
        if not drop and w < 0.25 * Wp and h <= max(6.0, 3.0 * lw_guess):
            cy = y + h / 2.0
            for bx in boxes:
                th = bx[3] - bx[1]
                if bx[1] - 2 <= cy <= bx[3] + 2 and 0 <= bx[0] - (x + w) < max(40, 3.0 * th):
                    drop = True
                    break
        # (c) small compact blob, dense fill -> glyph
        if not drop and w < 0.10 * Wp and h < 0.09 * Hp and area > 0.22 * w * h and area < 0.01 * total:
            drop = True
        if drop:
            removed |= (lab == i)
    return mask & ~removed, removed


# ------------------------------------------------------- colour decomposition

@dataclass
class ColorCluster:
    idx: int
    rgb: tuple
    lab: tuple
    n_pixels: int


def opaque_core(rgb: np.ndarray, mask: np.ndarray, bg: np.ndarray,
                rel: float = 0.82) -> np.ndarray:
    """Pixels at the opaque core of a stroke, i.e. not anti-aliasing halo.

    A halo pixel is a blend alpha*C + (1-alpha)*bg, so its contrast against the
    background is strictly lower than that of the stroke it belongs to. Keeping only
    pixels within `rel` of the local maximum contrast removes halos while keeping
    single-pixel-wide lines (which *are* their own local maximum).
    """
    contrast = np.abs(rgb.astype(np.int16) - bg.astype(np.int16)).max(axis=2).astype(np.float32)
    contrast[~mask] = 0
    local = cv2.dilate(contrast, np.ones((3, 3), np.uint8))
    return mask & (contrast >= rel * np.maximum(local, 1e-3))


def decompose_colors(rgb: np.ndarray, mask: np.ndarray, bg: np.ndarray,
                     birch_threshold: float = 7.0, merge_de: float = 20.0,
                     min_frac: float = 0.02):
    """BIRCH colour decomposition. Returns (label_image, [ColorCluster]).

    Only opaque stroke cores are clustered, so anti-aliased edge pixels cannot create
    spurious clusters. Halo pixels are then assigned to the *spatially* nearest core
    (not the nearest colour), which is what actually identifies which stroke they
    belong to.
    """
    from sklearn.cluster import Birch

    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return np.full(mask.shape, -1, np.int16), []

    rgb = cv2.medianBlur(rgb, 3)
    core = opaque_core(rgb, mask, bg)
    if core.sum() < 50:
        core = _run_cores(mask)
    cys, cxs = np.nonzero(core)
    if len(cys) < 50:
        cys, cxs = ys, xs
    sel = np.linspace(0, len(cys) - 1, min(20000, len(cys))).astype(int)
    samp = rgb[cys[sel], cxs[sel]]
    lab_s = rgb_to_lab(samp)

    br = Birch(threshold=birch_threshold, n_clusters=None).fit(lab_s)
    centers = br.subcluster_centers_
    counts = np.bincount(br.subcluster_labels_, minlength=len(centers))

    # merge centres that are perceptually indistinguishable
    order = np.argsort(-counts)
    merged, mcount = [], []
    for i in order:
        for j, c in enumerate(merged):
            if np.linalg.norm(centers[i] - c) < merge_de:
                w = mcount[j] + counts[i]
                merged[j] = (c * mcount[j] + centers[i] * counts[i]) / max(w, 1)
                mcount[j] = w
                break
        else:
            merged.append(centers[i].copy())
            mcount.append(counts[i])
    centers = np.array(merged)

    # label the cores by colour, keeping only clusters with enough core pixels
    core_lab = rgb_to_lab(rgb[cys, cxs])
    d = ((core_lab[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
    assign = d.argmin(1)
    keep = [k for k in range(len(centers)) if (assign == k).sum() >= min_frac * len(cys)]
    labimg = np.full(mask.shape, -1, np.int16)
    if not keep:
        return labimg, []
    kc = centers[keep]
    assign = ((core_lab[:, None, :] - kc[None, :, :]) ** 2).sum(-1).argmin(1)

    core_lab_img = np.full(mask.shape, -1, np.int16)
    core_lab_img[cys, cxs] = assign.astype(np.int16)

    # halo / remaining ink pixels: label of the spatially nearest core pixel
    dists = np.full((len(kc),) + mask.shape, np.inf, np.float32)
    for i in range(len(kc)):
        src = (core_lab_img == i)
        dt = cv2.distanceTransform((~src).astype(np.uint8), cv2.DIST_L2, 3)
        dists[i] = dt
    nearest = dists.argmin(0).astype(np.int16)
    labimg[mask] = nearest[mask]
    labimg[cys, cxs] = core_lab_img[cys, cxs]

    core_rgb = [rgb[(core_lab_img == i)].mean(0) if (core_lab_img == i).any() else bg
                for i in range(len(kc))]
    labimg = merge_interleaved(labimg, len(kc), rgbs=core_rgb, bg=bg)
    n = int(labimg.max()) + 1
    clusters = []
    for i in range(n):
        sel_i = (labimg == i)
        if not sel_i.any():
            continue
        px = rgb[sel_i]
        clusters.append(ColorCluster(i, tuple(px.mean(0).round().astype(int)),
                                     tuple(rgb_to_lab(px[::7] if len(px) > 7 else px)
                                           .mean(0).round(2)), int(sel_i.sum())))
    clusters.sort(key=lambda c: -c.n_pixels)
    return labimg, clusters


def _is_blend(ci, cj, bg, tol: float = 18.0) -> bool:
    """True when colour ci looks like colour cj blended with the background."""
    if ci is None or cj is None or bg is None:
        return False
    ci, cj, b = np.asarray(ci, float), np.asarray(cj, float), np.asarray(bg, float)
    d = cj - b
    n2 = float(d @ d)
    if n2 < 1e-6:
        return False
    a = float((ci - b) @ d) / n2
    if not 0.15 <= a <= 0.95:
        return False
    return float(np.linalg.norm(ci - (b + a * d))) <= tol


def merge_interleaved(labimg: np.ndarray, n: int, thresh: float = 0.40,
                      rgbs=None, bg=None) -> np.ndarray:
    """Merge colour clusters whose pixels are interleaved in the same strokes.

    JPEG artefacts and anti-aliasing smear a single plotted line across several colour
    clusters; those clusters are spatially *mixed inside the same stroke*, while two
    genuinely different curves only touch where they cross. Requiring the interleaving
    to be mutual therefore merges the former and keeps the latter apart.
    """
    if n <= 1:
        return labimg
    k = np.ones((3, 3), np.uint8)
    masks = [(labimg == i) for i in range(n)]
    sizes = np.array([m.sum() for m in masks], dtype=float)
    dil = [cv2.dilate(m.astype(np.uint8), k).astype(bool) for m in masks]
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in range(n):
        for j in range(i + 1, n):
            if sizes[i] == 0 or sizes[j] == 0:
                continue
            fij = (dil[i] & masks[j]).sum() / sizes[j]
            fji = (dil[j] & masks[i]).sum() / sizes[i]
            blend = (rgbs is not None and
                     (_is_blend(rgbs[i], rgbs[j], bg) or _is_blend(rgbs[j], rgbs[i], bg)))
            if min(fij, fji) >= thresh or (blend and max(fij, fji) >= 0.25):
                a, b = find(i), find(j)
                if a != b:
                    parent[b] = a
    roots = sorted({find(i) for i in range(n)})
    remap = {r: i for i, r in enumerate(roots)}
    out = labimg.copy()
    for i in range(n):
        out[labimg == i] = remap[find(i)]
    return out


def _run_cores(mask: np.ndarray) -> np.ndarray:
    """Middle pixel of every vertical ink run (least affected by anti-aliasing)."""
    out = np.zeros_like(mask)
    H, W = mask.shape
    for x in range(W):
        col = np.flatnonzero(mask[:, x])
        if col.size == 0:
            continue
        for a, b in group_consecutive(col, gap=1):
            out[(a + b) // 2, x] = True
    return out


# ------------------------------------------------- segment connectivity graph

@dataclass
class Node:
    x: int
    y0: int
    y1: int

    @property
    def mid(self) -> float:
        return (self.y0 + self.y1) / 2.0

    @property
    def h(self) -> int:
        return self.y1 - self.y0 + 1


@dataclass
class Segment:
    idx: int
    nodes: list
    x0: int
    x1: int
    counts: np.ndarray = field(default=None)   # pixels per colour cluster
    npix: int = 0

    def ys(self, lw: float) -> np.ndarray:
        """Trace y for each node: top edge + half a line width (peak-preserving)."""
        return np.array([min(n.y0 + lw / 2.0, n.y1) for n in self.nodes])


def build_graph(mask: np.ndarray, max_xgap: int = 2, ytol: int = 1):
    """Column-run graph -> maximal non-branching segments."""
    H, W = mask.shape
    nodes, per_col = [], []
    for x in range(W):
        col = np.flatnonzero(mask[:, x])
        ids = []
        if col.size:
            for a, b in group_consecutive(col, gap=1):
                ids.append(len(nodes))
                nodes.append(Node(x, int(a), int(b)))
        per_col.append(ids)

    succ = [[] for _ in nodes]
    pred = [[] for _ in nodes]
    for x in range(W):
        if not per_col[x]:
            continue
        for dx in range(1, max_xgap + 1):
            if x + dx >= W or not per_col[x + dx]:
                continue
            linked = False
            for i in per_col[x]:
                for j in per_col[x + dx]:
                    a, b = nodes[i], nodes[j]
                    if a.y0 - ytol <= b.y1 and b.y0 - ytol <= a.y1:
                        succ[i].append(j)
                        pred[j].append(i)
                        linked = True
            if linked:
                break

    # maximal chains: extend while single successor whose single predecessor is us
    seg_of = np.full(len(nodes), -1, np.int32)
    segments = []
    for i in range(len(nodes)):
        if seg_of[i] != -1:
            continue
        if len(pred[i]) == 1 and len(succ[pred[i][0]]) == 1:
            continue                                        # not a chain start
        chain, cur = [i], i
        seg_of[i] = len(segments)
        while len(succ[cur]) == 1 and len(pred[succ[cur][0]]) == 1:
            cur = succ[cur][0]
            if seg_of[cur] != -1:
                break
            seg_of[cur] = len(segments)
            chain.append(cur)
        segments.append(Segment(len(segments), [nodes[k] for k in chain],
                                nodes[chain[0]].x, nodes[chain[-1]].x))
    # any nodes missed by the chain walk become singletons
    for i in range(len(nodes)):
        if seg_of[i] == -1:
            seg_of[i] = len(segments)
            segments.append(Segment(len(segments), [nodes[i]], nodes[i].x, nodes[i].x))

    # segment-level adjacency
    sedge = [set() for _ in segments]
    for i, ss in enumerate(succ):
        for j in ss:
            a, b = seg_of[i], seg_of[j]
            if a != b:
                sedge[a].add(b)
    return nodes, segments, [sorted(s) for s in sedge], seg_of


def segment_counts(segments, labimg: np.ndarray, n_clusters: int):
    for s in segments:
        c = np.zeros(n_clusters, np.int64)
        tot = 0
        for nd in s.nodes:
            sl = labimg[nd.y0:nd.y1 + 1, nd.x]
            tot += sl.size
            v = sl[sl >= 0]
            if v.size:
                c += np.bincount(v, minlength=n_clusters)
        s.counts, s.npix = c, tot


# ------------------------------------------------------------ path extraction

def _slope_end(s: Segment, lw: float, k: int = 4) -> float:
    y = s.ys(lw)
    if len(y) < 2:
        return 0.0
    k = min(k, len(y) - 1)
    return (y[-1] - y[-1 - k]) / k


def _slope_start(s: Segment, lw: float, k: int = 4) -> float:
    y = s.ys(lw)
    if len(y) < 2:
        return 0.0
    k = min(k, len(y) - 1)
    return (y[k] - y[0]) / k


def _roughness(y: np.ndarray, Hp: float = 1.0) -> float:
    """Mean |f''| of the centreline, measured on y normalised by the plot height so the
    penalty is scale-free and only ever breaks ties between candidate combinations."""
    if len(y) < 3:
        return 0.0
    return float(np.abs(np.diff(y / max(Hp, 1.0), 2)).mean())


def best_path(segments, sedge, cluster: int, lw: float, Wp: int,
              used: np.ndarray, beta: float = 0.6, discount: float = 0.25,
              Hp: float = 1.0):
    """Dynamic program for the max-reward combination of segments (Eq. 1)."""
    norm = max(Wp * max(lw, 1.0), 1.0)
    gain = np.zeros(len(segments))
    for s in segments:
        match = float(s.counts[cluster])
        other = float(s.counts.sum() - s.counts[cluster])
        g = (match - beta * other) / norm
        g -= LAMBDA_ROUGH * _roughness(s.ys(lw), Hp) * len(s.nodes) / max(Wp, 1)
        if used[s.idx]:
            g *= discount if g > 0 else 1.0
        gain[s.idx] = g

    order = sorted(range(len(segments)), key=lambda i: (segments[i].x0, segments[i].x1))
    dp = gain.copy()
    prev = np.full(len(segments), -1, np.int32)
    for i in order:
        for j in sedge[i]:
            if segments[j].x0 <= segments[i].x0:
                continue
            # Two scale-free penalties on the join. The dominant one is vertical
            # continuity: crossing another curve keeps y continuous, whereas hopping onto
            # a neighbouring (same-coloured) curve or into an inseparable overlap blob
            # does not. The angle term only breaks remaining ties - it must not use raw
            # slopes, since a near-vertical XRD spike has a slope of hundreds of px/col.
            yi = segments[i].ys(lw)
            yj = segments[j].ys(lw)
            jump = abs(float(yj[0]) - float(yi[-1])) / max(Hp, 1.0)
            kink = abs(np.arctan(_slope_start(segments[j], lw))
                       - np.arctan(_slope_end(segments[i], lw))) / np.pi
            cand = dp[i] + gain[j] - JUMP_PENALTY * jump - KINK_PENALTY * kink
            if cand > dp[j]:
                dp[j] = cand
                prev[j] = i
    end = int(np.argmax(dp))
    path, cur = [], end
    while cur != -1:
        path.append(cur)
        cur = int(prev[cur])
    return path[::-1], float(dp[end])


@dataclass
class Curve:
    xs: np.ndarray
    ys: np.ndarray                 # pixel row (float), top-left origin
    cluster: int
    rgb: tuple
    linewidth: float
    reward: float
    coverage: float
    seg_ids: list
    mask: np.ndarray = None
    legend: str = ""
    legend_source: str = ""
    n_added_left: int = 0
    n_added_right: int = 0
    style: str = "line"          # "line" or "markers"
    marker_px: float = 0.0


def _trace_from_path(segments, path, lw: float, W: int):
    xs, ys = [], []
    for sid in path:
        s = segments[sid]
        yv = s.ys(lw)
        for nd, y in zip(s.nodes, yv):
            xs.append(nd.x)
            ys.append(y)
    xs = np.array(xs)
    ys = np.array(ys, dtype=float)
    o = np.argsort(xs, kind="stable")
    xs, ys = xs[o], ys[o]
    # one y per column (median of duplicates)
    ux = np.unique(xs)
    uy = np.array([np.median(ys[xs == x]) for x in ux])
    return ux, uy


def _linewidth_swt(mask) -> float:
    from .verify import stroke_width
    return stroke_width(mask)


def _linewidth(segments, path, labimg, cluster) -> float:
    """Stroke width = typical run height where the curve is locally flat.

    Steep flanks and peak apexes give tall runs that say nothing about the pen width,
    so only runs whose neighbours sit at almost the same height are counted.
    """
    flat, allh = [], []
    for sid in path:
        nds = segments[sid].nodes
        for i, nd in enumerate(nds):
            allh.append(nd.h)
            prev_ok = i == 0 or abs(nds[i - 1].y0 - nd.y0) <= 1
            next_ok = i == len(nds) - 1 or abs(nds[i + 1].y0 - nd.y0) <= 1
            if prev_ok and next_ok:
                flat.append(nd.h)
    if not allh:
        return 2.0
    lw = np.percentile(np.array(allh), 10)
    if len(flat) >= 20:
        lw = min(lw, np.median(np.array(flat)))
    return float(max(1.0, round(lw)))


def branch_traces(mask, labimg, cluster: int, fr, min_frac: float = 0.15):
    """Split one colour into an upper and a lower branch.

    An adsorption isotherm is not a function: the desorption branch runs back across the
    same pressures at a different uptake, so a single y(x) either skips those columns as
    ambiguous or zig-zags between the two. But the loop is two functions, and in the
    columns where both are present they are two clearly separated runs of the same
    colour. Taking the upper run for one series and the lower for the other recovers
    both, and where the loop is closed the two runs coincide — which is exactly what the
    physical curves do at the ends of the loop.
    """
    own = (labimg == cluster) & mask
    x0, x1, y0, y1 = fr.interior(mask.shape)
    Hp = max(fr.height, 1)
    cols, two = [], 0
    for x in range(x0, x1 + 1):
        rows = np.flatnonzero(own[:, x])
        if rows.size == 0:
            continue
        runs = group_consecutive(rows, gap=2)
        runs = [r for r in runs if (r[1] - r[0]) < 0.5 * Hp]
        if not runs:
            continue
        big = [r for r in runs if (r[1] - r[0] + 1) >= 1]
        sep = [r for r in big]
        cols.append((x, sep))
        if len(sep) >= 2 and (sep[-1][0] - sep[0][1]) > 0.04 * Hp:
            two += 1
    if not cols or two < min_frac * len(cols):
        return None
    up_x, up_y, lo_x, lo_y, sizes = [], [], [], [], []
    for x, runs in cols:
        a, b = runs[0]
        up_x.append(x)
        up_y.append((a + b) / 2.0)
        a2, b2 = runs[-1]
        lo_x.append(x)
        lo_y.append((a2 + b2) / 2.0)
        sizes.append(b - a + 1)
    span = max(up_x) - min(up_x)
    if span < 0.25 * fr.width:
        return None
    return ((np.asarray(up_x, float), np.asarray(up_y, float)),
            (np.asarray(lo_x, float), np.asarray(lo_y, float)),
            float(np.median(sizes)), own)


def column_trace(mask, labimg, cluster: int, fr, max_spread: float = 0.5):
    """Trace a series by the vertical centre of its own-colour ink, column by column.

    Symbol-drawn series (isotherms, magnetic data) defeat stroke tracing: each symbol is
    its own component, and where symbols crowd together they fuse into one blob, so the
    connectivity graph sees either confetti or a single lump. Neither needs solving —
    the series is still single-valued in x, so the centre of its ink in each column is
    the value. Columns where the ink spans most of the plot are skipped as ambiguous.
    """
    own = (labimg == cluster) & mask
    x0, x1, y0, y1 = fr.interior(mask.shape)
    xs, ys, sizes = [], [], []
    for x in range(x0, x1 + 1):
        rows = np.flatnonzero(own[:, x])
        if rows.size == 0:
            continue
        if (rows.max() - rows.min()) > max_spread * max(fr.height, 1):
            continue
        runs = group_consecutive(rows, gap=2)
        a, b = max(runs, key=lambda r: r[1] - r[0])
        xs.append(x)
        ys.append((a + b) / 2.0)
        sizes.append(b - a + 1)
    if len(xs) < 12:
        return None
    xs = np.asarray(xs, float)
    if (xs.max() - xs.min()) < 0.25 * fr.width:
        return None
    return xs, np.asarray(ys, float), float(np.median(sizes)), own


def scatter_series(mask, labimg, cluster: int, fr, min_points: int = 8):
    """A series drawn as separate markers rather than a continuous stroke.

    Isotherms, magnetic data and most property-vs-property plots are drawn as symbols.
    A stroke-tracing engine sees each symbol as its own fragment and returns a handful of
    disconnected stubs. Markers are easy to recognise instead — many small blobs of
    similar size, none of them long — and their centroids *are* the data points, so no
    tracing is needed at all.
    """
    own = (labimg == cluster) & mask
    n, lab, stats, cent = cv2.connectedComponentsWithStats(own.astype(np.uint8), 8)
    if n <= min_points:
        return None
    areas = stats[1:, cv2.CC_STAT_AREA].astype(float)
    ws = stats[1:, cv2.CC_STAT_WIDTH].astype(float)
    hs = stats[1:, cv2.CC_STAT_HEIGHT].astype(float)
    total = float(areas.sum())
    if total <= 0:
        return None
    med_a = float(np.median(areas))
    keep = ((areas >= 0.25 * med_a) & (areas <= 4.0 * med_a)
            & (ws <= 0.12 * max(fr.width, 1)) & (hs <= 0.12 * max(fr.height, 1)))
    if keep.sum() < min_points:
        return None
    # a dominant long component means this is a stroke with debris, not a marker series
    if areas.max() > 0.25 * total or ws.max() > 0.35 * fr.width:
        return None
    if keep.sum() < 0.55 * len(areas):
        return None
    pts = np.array([(cent[i + 1][0], cent[i + 1][1]) for i in range(len(areas)) if keep[i]])
    order = np.argsort(pts[:, 0])
    xs, ys = pts[order, 0], pts[order, 1]
    span = xs.max() - xs.min()
    if span < 0.25 * fr.width:
        return None
    size = float(np.median(np.sqrt(areas[keep])))
    m = np.zeros(mask.shape, bool)
    for i in range(len(areas)):
        if keep[i]:
            m |= (lab == i + 1)
    return xs, ys, size, m


def extract_curves(rgb, mask, fr, labimg, clusters, max_per_cluster: int = 8,
                   min_coverage: float = 0.25, verbose=False,
                   bg_color=None) -> list[Curve]:
    if not clusters:
        return []
    nodes, segments, sedge, seg_of = build_graph(mask)
    segment_counts(segments, labimg, len(clusters))
    Wp = max(fr.width, 1)
    used = np.zeros(len(segments), bool)
    out: list[Curve] = []

    for c in clusters:
        sc = scatter_series(mask, labimg, c.idx, fr)
        if sc is None:
            sc_cols = column_trace(mask, labimg, c.idx, fr)
        else:
            sc_cols = None
        if sc is not None:
            # A symbol-drawn series that occupies two separated runs per column is a
            # hysteresis loop: adsorption and desorption over the same pressures. Split
            # it here, where we already know the series is symbols and stroke tracing
            # was never going to work.
            br = branch_traces(mask, labimg, c.idx, fr)
            if br is not None:
                (ux, uy), (lx, ly), bsize, bmask = br
                for xs_, ys_ in ((ux, uy), (lx, ly)):
                    out.append(Curve(xs=xs_, ys=ys_, cluster=c.idx, rgb=c.rgb,
                                     linewidth=max(bsize, 2.0), reward=float(len(xs_)),
                                     coverage=len(xs_) / Wp, seg_ids=[], mask=bmask,
                                     style="markers", marker_px=bsize))
                continue
            xs, ys, size, m = sc
            out.append(Curve(xs=np.asarray(xs, float), ys=np.asarray(ys, float),
                             cluster=c.idx, rgb=c.rgb, linewidth=max(size, 2.0),
                             reward=float(len(xs)), coverage=(xs.max() - xs.min())
                             / max(fr.width, 1), seg_ids=[], mask=m,
                             style="markers", marker_px=size))
            continue
        cl_total = c.n_pixels
        lw = 2.0
        before = len(out)
        for it in range(max_per_cluster):
            path, reward = best_path(segments, sedge, c.idx, lw, Wp, used,
                                     Hp=max(fr.height, 1))
            if not path:
                break
            matched = sum(int(segments[s].counts[c.idx]) for s in path)
            amask0 = np.zeros(mask.shape, bool)
            for sid in path:
                for nd in segments[sid].nodes:
                    amask0[nd.y0:nd.y1 + 1, nd.x] = True
            lw = min(_linewidth(segments, path, labimg, c.idx), _linewidth_swt(amask0))
            xs, ys = _trace_from_path(segments, path, lw, rgb.shape[1])
            cov = len(xs) / Wp
            frac = matched / max(cl_total, 1)
            if verbose:
                print(f"   cluster{c.idx} it{it}: cov={cov:.2f} frac={frac:.2f} "
                      f"reward={reward:.3f} lw={lw:.1f} segs={len(path)}")
            if cov < min_coverage or reward <= 0 or frac < 0.12:
                break
            amask = np.zeros(mask.shape, bool)
            for s in path:
                for nd in segments[s].nodes:
                    amask[nd.y0:nd.y1 + 1, nd.x] = True
            out.append(Curve(xs=xs, ys=ys, cluster=c.idx, rgb=c.rgb, linewidth=lw,
                             reward=reward, coverage=cov, seg_ids=list(path), mask=amask))
            for s in path:
                used[s] = True
            remaining = cl_total - sum(
                int(segments[s].counts[c.idx]) for s in np.flatnonzero(used)
                if segments[s].counts[c.idx] > 0)
            if remaining < 0.18 * cl_total:
                break

        # Stroke tracing can only fail two ways on a symbol-drawn series: it returns
        # nothing, or it returns short fragments. Both are visible as poor coverage, so
        # the cloud tracer is tried whenever the graph did badly and kept if it does
        # better — no need to classify the drawing style up front.
        if sc_cols is not None:
            best = max((o.coverage for o in out[before:]), default=0.0)
            cx, cy, csize, cmask = sc_cols
            ccov = len(cx) / Wp
            br = branch_traces(mask, labimg, c.idx, fr)
            # Branches span the same columns as the single trace, so they can never win
            # on coverage — but where they exist they carry both curves instead of one,
            # which is strictly more of the figure's data.
            # Prefer branches only where stroke tracing already struggled: a stack of
            # three same-coloured curves is better served by the DP's repeated passes,
            # while a hysteresis loop defeats them entirely.
            if br is not None and best < 0.6:
                (ux, uy), (lx, ly), bsize, bmask = br
                del out[before:]
                for xs_, ys_ in ((ux, uy), (lx, ly)):
                    out.append(Curve(xs=xs_, ys=ys_, cluster=c.idx, rgb=c.rgb,
                                     linewidth=max(bsize, 2.0), reward=float(len(xs_)),
                                     coverage=len(xs_) / Wp, seg_ids=[], mask=bmask,
                                     style="markers", marker_px=bsize))
            elif ccov > max(best, 0.5) or (best < 0.5 and ccov > best):
                del out[before:]
                out.append(Curve(xs=cx, ys=cy, cluster=c.idx, rgb=c.rgb,
                                 linewidth=max(csize, 2.0), reward=float(len(cx)),
                                 coverage=ccov, seg_ids=[], mask=cmask,
                                 style="markers", marker_px=csize))
    # A symbol series has ink only where a symbol sits, so counting filled columns
    # measures how the authors drew it, not how completely it was read. What makes such
    # a series complete is spanning its axis, so report span for markers. (The decisions
    # above still compare column counts, which is the right basis for choosing between
    # two candidate tracings of the same pixels.)
    for c in out:
        if getattr(c, "style", "line") == "markers" and len(c.xs):
            c.coverage = float(np.max(c.xs) - np.min(c.xs)) / max(fr.width, 1)
    _relabel(out, rgb, labimg, len(clusters), mask, bg_color)
    return _dedupe(out, fr, bg_color)


def extend_traces(curves, labimg, fr, shape, text_cols=None, max_rise: float = 0.6):
    """Extend each trace to the plot edges, following its own colour.

    The reward DP walks the *connectivity* graph, so it stops where several curves merge
    into one blob - typically the strong low-angle peak, which is exactly the part of a
    PXRD pattern one least wants to lose. Colour still separates the curves inside such a
    blob, so the trace is continued column by column along runs of its own colour cluster,
    picking the run closest to where the curve already is.

    Columns covered by figure text are not crossed: text of the same colour as the curve
    would otherwise capture the trace.
    """
    x0, x1, y0, y1 = fr.interior(shape)
    Hp = max(fr.height, 1)
    text_cols = text_cols if text_cols is not None else set()
    for c in curves:
        if getattr(c, "style", "line") == "markers":
            continue                     # symbols are already the data points
        own = (labimg == c.cluster)
        for direction in (-1, 1):
            xs, ys = list(c.xs), list(c.ys)
            x = int(xs[0] if direction < 0 else xs[-1])
            y = float(ys[0] if direction < 0 else ys[-1])
            added = []
            gap = max(3, int(round(0.02 * fr.width)))
            while True:
                # step forward, jumping over columns with no own-colour ink (a dropout, or
                # a stretch hidden behind figure text) as long as the curve reappears close
                # to where it left off; a curve runs the full width of the plot, so a short
                # hole is an occlusion rather than the end of the line
                nxt, skipped = None, []
                probe = x
                for _ in range(gap + 1):
                    probe += direction
                    if probe < x0 or probe > x1:
                        break
                    col = np.flatnonzero(own[:, probe])
                    if col.size == 0 or probe in text_cols:
                        skipped.append(probe)
                        continue
                    runs = group_consecutive(col, gap=1)
                    best = min(runs, key=lambda r: 0.0 if r[0] - 1 <= y <= r[1] + 1
                               else min(abs(y - r[0]), abs(y - r[1])))
                    cand = min(best[0] + c.linewidth / 2.0, float(best[1]))
                    reach = max_rise * Hp if not skipped else max(3.0 * c.linewidth,
                                                                 0.05 * Hp)
                    if abs(cand - y) <= reach:
                        nxt = (probe, cand)
                    break
                if nxt is None:
                    break
                nx, ny = nxt
                for i, sx in enumerate(skipped, 1):      # linear bridge over the hole
                    added.append((sx, y + (ny - y) * i / (len(skipped) + 1)))
                added.append((nx, ny))
                x, y = nx, ny
            if added:
                added = [a for a in added if x0 <= a[0] <= x1]
            if added:
                if direction < 0:
                    added.reverse()
                    c.xs = np.array([a[0] for a in added] + xs)
                    c.ys = np.array([a[1] for a in added] + ys)
                    c.n_added_left = len(added)
                else:
                    c.xs = np.array(xs + [a[0] for a in added])
                    c.ys = np.array(ys + [a[1] for a in added])
                    c.n_added_right = len(added)
        c.coverage = len(c.xs) / max(fr.width, 1)
        m = np.zeros(shape, bool)
        for x, yv in zip(np.asarray(c.xs, int), np.asarray(c.ys, float)):
            col = np.flatnonzero(own[:, x])
            if col.size == 0:
                continue
            for a, b in group_consecutive(col, gap=1):
                if a - 1 <= yv <= b + 1:
                    m[a:b + 1, x] = True
                    break
        c.mask = c.mask | m
    return curves


def _row_crossings(mask: np.ndarray, y: int, lw: float, max_k: int = 4, min_k: int = 1):
    """Decompose one row of ink into the line crossings that drew it.

    The figure shows the data thickened by a pen of width `lw`, so a horizontal block of
    width W at row y was produced by k = round(W / lw) crossings of the pen. If those k
    crossings are evenly spread, the outermost pen centres sit half a pen width inside the
    block, giving centres at lw/2 + i*(W - lw)/(k - 1). For W = 1.5*lw this puts the two
    centres at 1/3 and 2/3 of the block - i.e. the *centre lines*, not the block edges.

    Only near-vertical ink is decomposed (W <= (max_k+0.5)*lw); a long horizontal run is a
    flat part of a curve and is handled by the column pass instead.
    """
    row = np.flatnonzero(mask[y])
    if row.size == 0:
        return []
    out = []
    for xa, xb in group_consecutive(row, gap=1):
        W = xb - xa + 1
        if W > (max_k + 0.5) * lw:
            continue
        k = max(1, int(round(W / lw)))
        if k < min_k:
            continue
        if k == 1:
            out.append((xa + (W - 1) / 2.0, y))
        else:
            step = (W - lw) / (k - 1)
            for i in range(k):
                out.append((xa + lw / 2.0 - 0.5 + i * step, y))
    return out


def refine_centerlines(curves, labimg, fr, shape, tall: float = 1.6):
    """Move each trace from the *edge* of its stroke onto the stroke's *centre line*.

    The figure is the data thickened by a pen, so the data is the centre of the stroke.
    Two situations let us say where that centre is:

      flat runs  - a column whose ink run is no taller than `tall` pen widths is a locally
                   flat piece of curve, and the centre of the run is the centre line;
      overlaps   - a row whose ink block is wide enough to hold k >= 2 pen crossings was
                   drawn by k separate passes of the pen (the flanks of a narrow peak, or
                   two peaks merged into one solid mass). Their centres follow from the
                   block width - see _row_crossings - and each is a real curve point, so
                   the trace is moved onto the nearest one.

    Everywhere else - a column filled by a peak narrower than the pen - the height is not
    recoverable from that column at all, and the existing peak-preserving estimate is kept
    rather than replaced by a guess. Deciding this per column, instead of re-deriving the
    whole trace, is what keeps the correction from wandering between a peak and its
    baseline in the columns where both are consistent with the ink.
    """
    x0, x1, y0, y1 = fr.interior(shape)
    for c in curves:
        if getattr(c, "style", "line") == "markers":
            continue                     # a marker centroid is already the centre
        own = (labimg == c.cluster)
        lw = max(float(c.linewidth), 1.0)
        xs = np.asarray(c.xs, int)
        ys = np.asarray(c.ys, float).copy()

        # resolvable overlaps: only blocks that really imply two or more pen passes
        multi: dict[int, list] = {}
        for y in range(y0, y1 + 1):
            for cx, cy in _row_crossings(own, y, lw, min_k=2):
                xi = int(round(cx))
                multi.setdefault(xi, []).append(float(cy))

        n_flat = n_multi = 0
        for i, x in enumerate(xs):
            col = np.flatnonzero(own[:, x])
            if col.size:
                run = None
                for a, b in group_consecutive(col, gap=1):
                    if a - 1 <= ys[i] <= b + 1:
                        run = (a, b)
                        break
                if run is not None and (run[1] - run[0] + 1) <= tall * lw:
                    ys[i] = (run[0] + run[1]) / 2.0        # flat: centre of the run
                    n_flat += 1
                    continue
            cand = multi.get(int(x))
            if cand:
                groups = _group_values(cand, 2.0 * lw)
                g = min(groups, key=lambda gg: abs(np.median(gg) - ys[i]))
                ys[i] = float(np.median(g))                # resolved overlap crossing
                n_multi += 1
        c.ys = ys
        c.n_centre_flat, c.n_centre_overlap = n_flat, n_multi
    return curves


def _group_values(vals, gap: float):
    """Split a list of y values into groups separated by more than `gap`."""
    if not vals:
        return []
    v = np.sort(np.asarray(vals, float))
    out, cur = [], [float(v[0])]
    for a in v[1:]:
        if a - cur[-1] > gap:
            out.append(cur)
            cur = [float(a)]
        else:
            cur.append(float(a))
    out.append(cur)
    return out


def resolve_extension_conflicts(curves, base, fr, frac: float = 0.4):
    """Undo extensions that cannot be trusted, i.e. that collapsed onto each other.

    When several curves share a colour cluster (near-identical hues), the colour-guided
    extension has nothing left to tell them apart and every one of them follows the same
    ink, so the added stretches coincide. Overlapping *added* stretches are therefore the
    signal that the extension was a guess, and the weaker curve is rolled back.

    Coincidence between the *original* (DP-assembled) parts is not touched: those were
    separated on connectivity and colour-consistency evidence.
    """
    order = sorted(range(len(curves)), key=lambda i: -curves[i].reward)
    reverted = []
    for rank, i in enumerate(order):
        ci = curves[i]
        added_i = _added_x(ci)
        if not added_i:
            continue
        clash = False
        for j in order[:rank]:
            cj = curves[j]
            common = added_i & set(np.asarray(cj.xs, int).tolist())
            if len(common) < frac * len(added_i):
                continue
            xs = np.array(sorted(common))
            d = np.abs(np.interp(xs, ci.xs, ci.ys) - np.interp(xs, cj.xs, cj.ys))
            lw = max(ci.linewidth, cj.linewidth)
            if float(np.mean(d < max(3.0, 1.5 * lw))) > 0.5:
                clash = True
                break
        if clash:
            xs, ys, m, cov = base[i]
            ci.xs, ci.ys, ci.mask, ci.coverage = xs.copy(), ys.copy(), m.copy(), cov
            ci.n_added_left = ci.n_added_right = 0
            reverted.append(i)
    return reverted


def _added_x(c) -> set:
    nl, nr = getattr(c, "n_added_left", 0), getattr(c, "n_added_right", 0)
    if not nl and not nr:
        return set()
    xs = np.asarray(c.xs, int)
    out = list(xs[:nl]) if nl else []
    if nr:
        out += list(xs[len(xs) - nr:])
    return set(int(v) for v in out)


def text_columns(ocr_items, fr, shape) -> set:
    """Plot-area columns occupied by recognised text (legend entries, annotations)."""
    x0, x1, y0, y1 = fr.interior(shape)
    cols = set()
    for it in ocr_items:
        bx0, by0, bx1, by1 = it["bbox"]
        if by1 < y0 or by0 > y1:
            continue
        cols.update(range(max(x0, int(bx0) - 1), min(x1, int(bx1) + 1) + 1))
    return cols


def text_mask(ocr_items, fr, shape, key_strip: float = 0.22) -> np.ndarray:
    """Annotation zone inside the plot: each recognised-text box plus the strip to its
    left, where its legend key line sits. Both are figure furniture, not data, so they
    are excluded from completeness accounting and from the "unclaimed ink" visual."""
    x0, x1, y0, y1 = fr.interior(shape)
    m = np.zeros(shape, bool)
    for it in ocr_items:
        bx0, by0, bx1, by1 = [int(round(v)) for v in it["bbox"]]
        if by1 < y0 or by0 > y1 or bx1 < x0 or bx0 > x1:
            continue
        ky0, ky1 = max(by0 - 1, y0), min(by1 + 2, y1 + 1)
        m[ky0:ky1, max(bx0 - 1, x0):min(bx1 + 2, x1 + 1)] = True
        left = max(x0, int(bx0 - key_strip * fr.width))
        m[ky0:ky1, left:max(bx0, left)] = True
    return m


def _relabel(curves, rgb, labimg, n_clusters: int, mask=None, bg=None):
    """Re-derive each curve's colour identity from the stroke its trace sits in.

    The trace runs along the upper edge of the stroke, where pixels are partly
    anti-aliased, so the colour is read from the most opaque pixel of the enclosing run
    rather than from the trace pixel itself.
    """
    H = labimg.shape[0]
    contrast = None
    if mask is not None and bg is not None:
        contrast = np.abs(rgb.astype(np.int16) - np.asarray(bg, np.int16)).max(axis=2)
        contrast = np.where(mask, contrast, -1)
    for c in curves:
        xi = np.asarray(c.xs, int)
        yi = np.clip(np.round(c.ys).astype(int), 0, H - 1)
        lab = labimg[yi, xi]
        lab = lab[lab >= 0]
        if lab.size:
            c.cluster = int(np.bincount(lab, minlength=n_clusters).argmax())
        picks = []
        lw = max(1, int(round(c.linewidth)))
        for x, y in zip(xi, yi):
            lo, hi = max(0, y - 1), min(H - 1, y + lw)
            if contrast is not None:
                col = contrast[lo:hi + 1, x]
                if col.max() < 0:
                    continue
                picks.append(rgb[lo + int(col.argmax()), x])
            else:
                picks.append(rgb[y, x])
        if picks:
            c.rgb = tuple(int(v) for v in np.median(np.array(picks), axis=0))


def _dedupe(curves: list[Curve], fr, bg_color=None) -> list[Curve]:
    """Drop traces that are near-duplicates of a stronger one."""
    curves = sorted(curves, key=lambda c: -c.reward)
    keep: list[Curve] = []
    for c in curves:
        dup = False
        for k in keep:
            lo, hi = max(c.xs.min(), k.xs.min()), min(c.xs.max(), k.xs.max())
            shortest = min(c.xs.max() - c.xs.min(), k.xs.max() - k.xs.min())
            if hi - lo < min(0.3 * fr.width, 0.8 * max(shortest, 1)):
                continue
            xi = np.arange(lo, hi + 1)
            d = np.abs(np.interp(xi, c.xs, c.ys) - np.interp(xi, k.xs, k.ys))
            lw = max(c.linewidth, k.linewidth)
            p70 = float(np.percentile(d, 70))
            halo = _is_blend(np.asarray(c.rgb, float), np.asarray(k.rgb, float), bg_color)
            if p70 < max(3.0, 1.3 * lw) or (halo and p70 < max(4.0, 2.5 * lw)):
                dup = True
                break
        if not dup:
            keep.append(c)
    return sorted(keep, key=lambda c: c.ys.mean())


def curve_pixel_mask(curve: Curve, segments_nodes, shape) -> np.ndarray:
    m = np.zeros(shape, bool)
    for nd in segments_nodes:
        m[nd.y0:nd.y1 + 1, nd.x] = True
    return m
