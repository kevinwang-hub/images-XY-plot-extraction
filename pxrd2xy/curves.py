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

    # A data marker and a stray glyph are both small compact blobs, and the glyph test
    # below cannot tell them apart one at a time -- which cost one isotherm 31 of its
    # points, more than half its ink. What separates them is the company they keep: a
    # glyph is one of a few, a marker is one of many identical. So the modal size of the
    # equant blobs is found first, and blobs of that size are exempt from the glyph test.
    # They can still be dropped for sitting inside a text box or beside a legend entry,
    # which is what a legend key does.
    ar = stats[1:, cv2.CC_STAT_AREA].astype(float)
    bw = stats[1:, cv2.CC_STAT_WIDTH].astype(float)
    bh = stats[1:, cv2.CC_STAT_HEIGHT].astype(float)
    equant = ((np.maximum(bw, bh) / np.maximum(np.minimum(bw, bh), 1.0) <= 2.2)
              & (ar >= 8) & (bw <= 0.10 * Wp) & (bh <= 0.09 * Hp))
    marker_band = None
    if equant.sum() >= 8:
        med = float(np.median(ar[equant]))
        band = equant & (ar >= 0.4 * med) & (ar <= 2.5 * med)
        if band.sum() >= 6:
            marker_band = (0.4 * med, 2.5 * med)

    def _in_band(w, h, area):
        return bool(marker_band and marker_band[0] <= area <= marker_band[1]
                    and max(w, h) / max(min(w, h), 1) <= 2.2)

    # A row of evenly spaced identical markers looks like a line of text to a character
    # recogniser, and on one magnetic panel it was duly read as a CJK glyph -- after
    # which the whole high-temperature tail of the series was deleted as text. A box
    # that covers several markers of the same size is a misread, not a caption, so it is
    # ignored. One marker beside a box is still a legend key and still goes.
    misread = set()
    if marker_band:
        for bi, bx in enumerate(boxes):
            hits = 0
            for i in range(1, n):
                x, y, w, h, area = (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
                                    stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT],
                                    stats[i, cv2.CC_STAT_AREA])
                if not _in_band(w, h, area):
                    continue
                ox = max(0, min(x + w, bx[2]) - max(x, bx[0]))
                oy = max(0, min(y + h, bx[3]) - max(y, bx[1]))
                if ox * oy > 0.55 * w * h:
                    hits += 1
            if hits >= 3:
                misread.add(bi)
    for i in range(1, n):
        x, y, w, h, area = (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
                            stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT],
                            stats[i, cv2.CC_STAT_AREA])
        if area > 0.02 * total and w > 0.25 * Wp:
            continue                                          # a real curve backbone
        cc_box = (x, y, x + w, y + h)
        drop = False
        # (a) sits inside / mostly inside a recognised text box
        for bi, bx in enumerate(boxes):
            if bi in misread:
                continue
            ox = max(0, min(cc_box[2], bx[2]) - max(cc_box[0], bx[0]))
            oy = max(0, min(cc_box[3], bx[3]) - max(cc_box[1], bx[1]))
            if ox * oy > 0.55 * w * h:
                drop = True
                break
        # (b) short stroke or symbol just left of a text box -> legend key.
        # A key drawn as a marker is as tall as it is wide, so a flat-stroke test misses
        # it; what identifies a key of either kind is that it is small and sits on the
        # text's own row, just before it. Left in, such a marker is data-coloured ink in
        # the middle of the plot and every tracer will try to account for it.
        if not drop and w < 0.25 * Wp and (h <= max(6.0, 3.0 * lw_guess)
                                           or (h < 0.05 * Hp and w < 0.15 * Wp)):
            cy = y + h / 2.0
            for bx in boxes:
                th = bx[3] - bx[1]
                if bx[1] - 2 <= cy <= bx[3] + 2 and 0 <= bx[0] - (x + w) < max(40, 3.0 * th):
                    drop = True
                    break
        # (c) small compact blob, dense fill -> glyph, unless it is one of a population
        # of identical blobs, in which case it is a data marker
        in_band = _in_band(w, h, area)
        if (not drop and not in_band and w < 0.10 * Wp and h < 0.09 * Hp
                and area > 0.22 * w * h and area < 0.01 * total):
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
    chained: bool = False
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


# ------------------------------------------------- what a colour is drawn as

def cluster_style(mask, labimg, cluster: int, fr) -> tuple[str, dict]:
    """How one colour is drawn: "points", "line", "mixed", or "unknown".

    Tried first, and it does not work well enough to be trusted on its own. Measured
    across this corpus, every statistic that ought to separate a scatter from a stroke
    overlaps between them:

        column coverage   points 0.45-1.00   lines 0.17-1.00
        median thickness  points 4-32 px     lines 4-14 px
        symbol ink share  points 0.35-0.79   lines 0.13-0.98

    The reason is structural: a stroke crossed by other curves is cut into many small
    equant fragments -- one line panel here yields 54 of them holding 98% of its ink --
    which is indistinguishable, component by component, from a row of symbols. So this
    is used only for the sub-question inside a scatter panel, where a *relative*
    comparison between the colours in one panel is reliable even though the absolute
    numbers are not: the guide line is the thin one and the data are the thick ones.
    Whether the panel is a scatter at all is asked of the classifier instead.
    """
    own = (labimg == cluster) & mask
    n, lab, stats, cent = cv2.connectedComponentsWithStats(own.astype(np.uint8), 8)
    if n <= 1:
        return "unknown", {}
    W, H = max(fr.width, 1), max(fr.height, 1)
    areas = stats[1:, cv2.CC_STAT_AREA].astype(float)
    ws = stats[1:, cv2.CC_STAT_WIDTH].astype(float)
    hs = stats[1:, cv2.CC_STAT_HEIGHT].astype(float)
    total = float(areas.sum())
    if total < 40:
        return "unknown", {}
    long_side = np.maximum(ws, hs)
    short_side = np.maximum(np.minimum(ws, hs), 1.0)
    aspect = long_side / short_side
    symbol = ((ws <= 0.06 * W) & (hs <= 0.06 * H) & (areas >= 4) & (aspect <= 2.5))
    stroke = (long_side >= 0.12 * max(W, H)) & (aspect >= 3.0)
    sym_ink = float(areas[symbol].sum()) / total
    stroke_ink = float(areas[stroke].sum()) / total
    n_sym = int(symbol.sum())
    # a symbol population is drawn with one pen: its members are all about one size
    uniform = 0.0
    if n_sym >= 4:
        a = areas[symbol]
        med = float(np.median(a))
        uniform = float(np.mean((a >= 0.4 * med) & (a <= 2.5 * med)))
    info = dict(n_components=int(len(areas)), n_symbols=n_sym, symbol_ink=sym_ink,
                stroke_ink=stroke_ink, uniform=uniform)
    if n_sym >= 6 and sym_ink >= 0.55 and stroke_ink <= 0.25 and uniform >= 0.6:
        return "points", info
    if n_sym >= 6 and sym_ink >= 0.15 and stroke_ink >= 0.25 and uniform >= 0.6:
        return "mixed", info
    if stroke_ink >= 0.5 or (len(areas) <= 3 and sym_ink < 0.5):
        return "line", info
    return "unknown", info


def point_series(mask, labimg, cluster: int, fr, style: str):
    """The data points of a colour drawn as symbols.

    One symbol, one data point, at its centre -- which is what the symbol means. No path
    is fitted through them and no value is interpolated between them: a scatter plot
    states its values at the pressures or temperatures that were measured and says
    nothing in between, so neither does this.

    Where symbols have run together the fused run is cut back into one point per column,
    because that is where the centres of the symbols under it lie.

    Returns (series, symbol_size, mask) where series is a list of (xs, ys, fill), fill
    being 1 for solid symbols, 0 for open ones and -1 where it could not be told.
    """
    own = (labimg == cluster) & mask
    n, lab, stats, cent = cv2.connectedComponentsWithStats(own.astype(np.uint8), 8)
    if n <= 1:
        return None
    W, H = max(fr.width, 1), max(fr.height, 1)
    areas = stats[1:, cv2.CC_STAT_AREA].astype(float)
    ws = stats[1:, cv2.CC_STAT_WIDTH].astype(float)
    hs = stats[1:, cv2.CC_STAT_HEIGHT].astype(float)
    aspect = np.maximum(ws, hs) / np.maximum(np.minimum(ws, hs), 1.0)
    symbol = (ws <= 0.06 * W) & (hs <= 0.06 * H) & (areas >= 4) & (aspect <= 2.5)
    if symbol.sum() >= 4:
        sym = max(float(np.median(np.sqrt(areas[symbol]))), 2.0)
    else:
        # Symbols packed tightly enough touch, and a whole series can arrive as a single
        # component. There is nothing to take a centroid of, but the centres are still
        # there: one per column, under the run. The symbol size is then the run's own
        # thickness, which is what the pen drew.
        th = own.sum(axis=0).astype(float)
        th = th[th > 0]
        if th.size < 12:
            return None
        sym = max(float(np.median(th)), 2.0)
    xs, ys, fill = [], [], []
    for i in range(len(areas)):
        w, h = ws[i], hs[i]
        if h <= 2 and w >= 4 * sym:
            continue                                   # a legend rule, not a symbol
        if symbol[i] or (w <= 2.5 * sym and h <= 2.5 * sym):
            xs.append(float(cent[i + 1][0]))
            ys.append(float(cent[i + 1][1]))
            fill.append(_symbol_filled(lab, stats, i + 1))
            continue
        if style != "mixed":
            continue                                   # stray ink, not this series
        x0 = int(stats[i + 1, cv2.CC_STAT_LEFT])
        y0 = int(stats[i + 1, cv2.CC_STAT_TOP])
        sub = lab[y0:y0 + int(h), x0:x0 + int(w)] == (i + 1)
        for c in range(sub.shape[1]):
            rows = np.flatnonzero(sub[:, c])
            if rows.size == 0:
                continue
            for a, b in group_consecutive(rows, gap=2):
                if (b - a + 1) > 0.5 * H:
                    continue
                xs.append(float(x0 + c))
                ys.append(float(y0 + (a + b) / 2.0))
                fill.append(-1)
    if len(xs) < 6:
        return None
    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)
    fill = np.asarray(fill, np.int8)
    o = np.argsort(xs)
    xs, ys, fill = xs[o], ys[o], fill[o]
    if (xs.max() - xs.min()) < 0.15 * W:
        return None
    # Solid and open symbols of one colour are two series -- adsorption and desorption,
    # cooling and warming. Splitting on the symbol is exact where chaining could only
    # guess, and it is how the figure's own legend distinguishes them.
    solid, open_ = fill == 1, fill == 0
    known = int(solid.sum()) + int(open_.sum())
    out = []
    if known >= 8 and solid.sum() >= 0.15 * known and open_.sum() >= 0.15 * known:
        for sel in (solid, open_):
            if sel.sum() >= 4 and (xs[sel].max() - xs[sel].min()) >= 0.15 * W:
                out.append((xs[sel], ys[sel], 1 if sel is solid else 0))
        unk = fill == -1
        if unk.any() and out:
            k = int(np.argmax([len(o_[0]) for o_ in out]))
            merged = np.argsort(np.concatenate([out[k][0], xs[unk]]))
            allx = np.concatenate([out[k][0], xs[unk]])[merged]
            ally = np.concatenate([out[k][1], ys[unk]])[merged]
            out[k] = (allx, ally, out[k][2])
    if not out:
        out = [(xs, ys, -1)]
    return out, sym, own



def symbol_series(rgb, mask, fr, bg, min_points: int = 6):
    """Every scatter series in a panel, found by grouping the symbols themselves.

    Colour decomposition works on pixels, and on a scatter that is the wrong unit. Every
    symbol carries a halo of blended edge pixels, so one series arrives as a saturated
    cluster plus a pale one, and a series that fades or is drawn over another can be
    split across both -- which is how a magnetic panel lost every point above 100 K to a
    grey "halo" cluster while its own colour kept the rest.

    A symbol is a better unit: it has one colour, which is the average over its whole
    body and therefore stable, and one size and one shape. Series are then groups of
    symbols that agree on all three. Solid and open symbols of the same colour are
    different series (adsorption and desorption); so are circles and triangles of the
    same colour, which pixel clustering cannot separate at all.

    Returns [(xs, ys, rgb, size, fill, mask)], one entry per series.
    """
    # Tick marks and the axis lines themselves are small dark blobs in a regular row,
    # which is exactly what a symbol population looks like. They are not data, and what
    # separates them is where they sit: on the frame. A margin of one symbol keeps real
    # points that merely come close to the axis.
    mask = mask.copy()
    ix0, ix1, iy0, iy1 = fr.interior(mask.shape)
    pad = 3
    mask[:iy0 + pad], mask[iy1 - pad + 1:] = False, False
    mask[:, :ix0 + pad], mask[:, ix1 - pad + 1:] = False, False

    n, lab, stats, cent = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if n <= 1:
        return []
    W, H = max(fr.width, 1), max(fr.height, 1)
    areas = stats[1:, cv2.CC_STAT_AREA].astype(float)
    ws = stats[1:, cv2.CC_STAT_WIDTH].astype(float)
    hs = stats[1:, cv2.CC_STAT_HEIGHT].astype(float)
    aspect = np.maximum(ws, hs) / np.maximum(np.minimum(ws, hs), 1.0)
    equant = (ws <= 0.06 * W) & (hs <= 0.06 * H) & (areas >= 6) & (aspect <= 2.2)
    if equant.sum() < min_points:
        return []
    med = float(np.median(areas[equant]))
    sym = max(float(np.sqrt(med)), 2.0)

    # tick marks are small, evenly spaced and equant, and they sit *on* the axis line --
    # which is the one thing a data point does not do
    bands = [b for b in (fr.left_band, fr.right_band) if b]
    rbands = [b for b in (fr.top_band, fr.bottom_band) if b]

    def _on_axis(x, y, w, h):
        return (any(x <= b[1] + 1 and x + w >= b[0] - 1 for b in bands)
                or any(y <= b[1] + 1 and y + h >= b[0] - 1 for b in rbands))

    pts = []            # (x, y, L, a, b, area, fill, component)
    for i in range(len(areas)):
        w, h, ar = ws[i], hs[i], areas[i]
        if h <= 2 and w >= 4 * sym:
            continue                                   # a legend rule
        if _on_axis(stats[i + 1, cv2.CC_STAT_LEFT], stats[i + 1, cv2.CC_STAT_TOP], w, h):
            continue                                   # a tick mark
        sel = lab == (i + 1)
        # Any equant blob is a symbol, whatever its size next to the others. Requiring it
        # to match the modal size cost one magnetic panel its whole high-temperature
        # tail: those squares are drawn larger than the rest of the series, and the
        # median was dragged down by anti-aliasing fragments besides.
        if equant[i] and ar >= 0.25 * med:
            col = rgb_to_lab(rgb[sel].reshape(-1, 3))[:, :3].mean(0)
            pts.append((float(cent[i + 1][0]), float(cent[i + 1][1]),
                        col[0], col[1], col[2], ar,
                        _symbol_filled(lab, stats, i + 1), i + 1))
            continue
        if ar < 3.0 * med:
            continue                                   # debris, or a fragment of one
        # symbols packed close enough to touch: their centres are the column centres
        x0 = int(stats[i + 1, cv2.CC_STAT_LEFT])
        y0 = int(stats[i + 1, cv2.CC_STAT_TOP])
        sub = sel[y0:y0 + int(h), x0:x0 + int(w)]
        for c in range(sub.shape[1]):
            rows = np.flatnonzero(sub[:, c])
            if rows.size == 0:
                continue
            for a_, b_ in group_consecutive(rows, gap=2):
                if (b_ - a_ + 1) > 0.5 * H:
                    continue
                px = rgb[y0 + a_:y0 + b_ + 1, x0 + c].reshape(-1, 3)
                col = rgb_to_lab(px)[:, :3].mean(0)
                pts.append((float(x0 + c), float(y0 + (a_ + b_) / 2.0),
                            col[0], col[1], col[2], med, -1, i + 1))
    if len(pts) < min_points:
        return []
    P = np.asarray(pts, float)

    # Group by colour, seeded from the densest colours first so that a series' own body
    # claims its halo rather than the halo becoming a series. Every symbol ends up in
    # some group: a leftover joins the nearest one instead of forming a series of its
    # own, because a handful of edge-shaded symbols is never a curve the paper drew.
    labcols = P[:, 2:5]
    # counted over separate symbols, not over the columns a fused run contributes: a
    # sparse series of twenty squares is a real series even beside a dense one of a
    # thousand, and scaling the floor to the total would erase it
    n_sym = int((P[:, 6] >= 0).sum())
    floor = max(min_points, int(0.03 * max(n_sym, min_points)))
    unassigned = np.ones(len(P), bool)
    centres = []
    while unassigned.any() and len(centres) < 8:
        idx = np.flatnonzero(unassigned)
        d = np.linalg.norm(labcols[idx][:, None, :] - labcols[idx][None, :, :], axis=2)
        seed = idx[int(np.argmax((d < 16).sum(1)))]
        near = unassigned & (np.linalg.norm(labcols - labcols[seed], axis=1) < 16)
        if near.sum() < floor:
            break
        centres.append(labcols[near].mean(0))
        unassigned[near] = False
    if not centres:
        return []
    # Overlapping symbols and anti-aliased edges shift a colour along one line only: the
    # one joining it to the background. So two centres that differ merely in how far
    # along that line they sit are the same series drawn over itself, and splitting on
    # them turned one blue isotherm into six. Merge them, keeping the most saturated as
    # the representative -- it is the colour the pen actually is.
    crgb = [_lab_to_rgb(c) for c in centres]
    keep = list(range(len(centres)))
    for i in range(len(centres)):
        for j in range(len(centres)):
            if i == j or keep[i] != i or keep[j] != j:
                continue
            if _is_blend(crgb[i], crgb[j], bg) or _is_blend(crgb[j], crgb[i], bg):
                di = float(np.linalg.norm(np.asarray(crgb[i], float) - np.asarray(bg, float)))
                dj = float(np.linalg.norm(np.asarray(crgb[j], float) - np.asarray(bg, float)))
                dull, bright = (i, j) if di < dj else (j, i)
                keep[dull] = bright
    centres = [centres[k] for k in range(len(centres)) if keep[k] == k]
    C = np.asarray(centres)
    owner = np.argmin(np.linalg.norm(labcols[:, None, :] - C[None, :, :], axis=2), axis=1)
    groups = [np.flatnonzero(owner == k) for k in range(len(C))]
    groups = [g for g in groups if len(g) >= min_points]

    out = []
    for g in groups:
        sub = P[g]
        # inside one colour, an open symbol and a solid one are different series, and so
        # are two clearly different sizes -- a circle and a triangle drawn the same colour
        keys = []
        fill = sub[:, 6]
        known = int((fill == 1).sum()) + int((fill == 0).sum())
        if known >= 8 and (fill == 1).sum() >= 0.15 * known and (fill == 0).sum() >= 0.15 * known:
            keys = [fill == 1, fill == 0]
            unk = fill == -1
            if unk.any():
                keys[int(np.argmax([k.sum() for k in keys]))] |= unk
        else:
            keys = [np.ones(len(sub), bool)]
        for k in keys:
            if k.sum() < min_points:
                continue
            q = sub[k]
            o = np.argsort(q[:, 0])
            q = q[o]
            if (q[:, 0].max() - q[:, 0].min()) < 0.1 * W:
                continue
            col = tuple(int(v) for v in _lab_to_rgb(q[:, 2:5].mean(0)))
            # its own ink, so every check downstream compares this series against the
            # symbols it came from rather than against everything in the panel
            comp = np.unique(q[:, 7].astype(int))
            m = np.isin(lab, comp) & mask
            out.append((q[:, 0].copy(), q[:, 1].copy(), col,
                        float(np.sqrt(np.median(q[:, 5]))), int(np.median(q[:, 6])), m))
    return out


def _lab_to_rgb(labv):
    arr = np.asarray(labv, np.float32).reshape(1, 1, 3)
    return cv2.cvtColor(arr, cv2.COLOR_LAB2RGB).reshape(3) * 255.0


def connector_lines(curves, fr) -> list[int]:
    """Indices of series that are a guide line threading a scatter, not data.

    Papers routinely join measured points with a line in another colour so the eye can
    follow them. That line states no value the points do not already state, and it draws
    values between measurements that were never made -- so the points are the data and
    the line is decoration.

    Two things identify it, and both are comparisons *within* one panel, which is what
    makes them safe: the guide is markedly thinner than the symbols it serves, and it
    runs along them. Neither absolute thickness nor closeness alone would do; a thin
    curve that goes its own way is data, and a thick series lying near another is two
    real series that happen to overlap.
    """
    if len(curves) < 2:
        return []
    thick = max((getattr(c, "marker_px", 0.0) or c.linewidth) for c in curves)
    if thick <= 0:
        return []
    drop = []
    for i, c in enumerate(curves):
        w = getattr(c, "marker_px", 0.0) or c.linewidth
        if w > 0.6 * thick or len(c.xs) < 12:
            continue
        for j, p in enumerate(curves):
            if j == i or tuple(p.rgb) == tuple(c.rgb) or len(p.xs) < 6:
                continue
            pw = getattr(p, "marker_px", 0.0) or p.linewidth
            if pw <= 0.6 * thick:
                continue
            o = np.argsort(p.xs)
            near = np.interp(c.xs, p.xs[o], p.ys[o], left=np.nan, right=np.nan)
            ok = np.abs(near - c.ys) <= max(pw, 0.02 * max(fr.height, 1))
            if np.isfinite(near).sum() >= 0.5 * len(c.xs) and \
                    float(np.nanmean(ok.astype(float))) >= 0.75:
                drop.append(i)
                break
    return drop


def marker_cloud(mask, labimg, cluster: int, fr, require_population: bool = True):
    """Every symbol of one colour, reduced to a point.

    Returns (xs, ys, symbol_size, filled) or None when the colour is not symbols.
    `filled` is 1 for a solid symbol, 0 for an open one and -1 where symbols have fused
    and the distinction cannot be made.

    A symbol series is a population of similar small blobs. The modal blob size is
    therefore the symbol size, and anything much wider than that is symbols that have
    fused — those are cut back into one point per column, which is where their centres
    lie. Blobs that are wide and only a pixel or two tall are legend rules, not data.
    """
    own = (labimg == cluster) & mask
    n, lab, stats, cent = cv2.connectedComponentsWithStats(own.astype(np.uint8), 8)
    if n <= 6:
        return None
    W, H = max(fr.width, 1), max(fr.height, 1)
    areas = stats[1:, cv2.CC_STAT_AREA].astype(float)
    ws = stats[1:, cv2.CC_STAT_WIDTH].astype(float)
    hs = stats[1:, cv2.CC_STAT_HEIGHT].astype(float)
    # A handful of anti-aliasing speckles beside a solid stroke is not a symbol series,
    # and chaining them yields a short curve that looks plausible and is not real. A
    # symbol series is a population by *ink*, not merely by component count: either its
    # symbols stand separately and hold most of the colour, or they have fused into runs
    # and no single run dominates. One long stroke with crumbs around it is neither.
    small = (ws <= 0.06 * W) & (hs <= 0.06 * H) & (areas >= 3)
    if small.sum() < 10:
        return None
    total = float(areas.sum())
    separate = float(areas[small].sum()) >= 0.25 * total
    fused = len(areas) >= 10 and areas.max() <= 0.60 * total
    # Without a hint this has to be inferred, and the test is deliberately strict so a
    # solid stroke with anti-aliasing crumbs around it is not mistaken for a population.
    # When the panel has been classified as symbol-drawn there is nothing to infer.
    if require_population and not (separate or fused):
        return None
    sym = float(np.median(np.sqrt(areas[small])))
    sym = max(sym, 2.0)
    xs, ys, fill = [], [], []
    for i in range(len(areas)):
        w, h = ws[i], hs[i]
        if h <= 2 and w >= 4 * sym:
            continue                                  # a legend rule, not a marker
        if w <= 2.5 * sym and h <= 2.5 * sym:
            xs.append(float(cent[i + 1][0]))
            ys.append(float(cent[i + 1][1]))
            fill.append(_symbol_filled(lab, stats, i + 1))
            continue
        x0 = int(stats[i + 1, cv2.CC_STAT_LEFT])
        y0 = int(stats[i + 1, cv2.CC_STAT_TOP])
        sub = lab[y0:y0 + int(h), x0:x0 + int(w)] == (i + 1)
        for c in range(sub.shape[1]):                 # fused symbols: one point per column
            rows = np.flatnonzero(sub[:, c])
            if rows.size == 0:
                continue
            for a, b in group_consecutive(rows, gap=2):
                if (b - a + 1) > 0.5 * H:
                    continue
                xs.append(float(x0 + c))
                ys.append(float(y0 + (a + b) / 2.0))
                fill.append(-1)
    if len(xs) < 40:
        return None
    o = np.argsort(xs)
    return (np.asarray(xs, float)[o], np.asarray(ys, float)[o], sym,
            np.asarray(fill, np.int8)[o])


def _symbol_filled(lab, stats, i: int) -> int:
    """1 if a symbol is solid, 0 if it is drawn open (a ring), -1 if it cannot be told.

    An open symbol encloses background: a region of non-symbol pixels inside its bounding
    box that cannot be reached from the box edge. A solid one does not.
    """
    x0 = int(stats[i, cv2.CC_STAT_LEFT])
    y0 = int(stats[i, cv2.CC_STAT_TOP])
    w = int(stats[i, cv2.CC_STAT_WIDTH])
    h = int(stats[i, cv2.CC_STAT_HEIGHT])
    if w < 4 or h < 4:
        return -1
    sub = (lab[y0:y0 + h, x0:x0 + w] == i)
    pad = np.zeros((h + 2, w + 2), np.uint8)
    pad[1:-1, 1:-1] = ~sub
    n, bl = cv2.connectedComponents(pad, 4)
    outside = bl[0, 0]
    for k in range(1, n):
        if k == outside:
            continue
        if (bl == k).sum() >= 0.12 * w * h:
            return 0
    return 1


def _chain(xs, ys, alive, fr, jump: float = 3.0, gap_pen: float = 2.0):
    """The most continuous single-valued path through a cloud of marker points.

    Each point taken is worth 1; each step costs the vertical distance travelled, as a
    fraction of the plot height, weighted heavily. A series that jumps down to a legend
    key and back pays that distance twice to gain one point, so it never pays -- while a
    genuinely steep curve pays it once and has no alternative, so it is still traced.
    Columns can be skipped for free, which is what lets the chain ignore a legend
    entirely rather than being dragged through it.
    """
    idx = np.flatnonzero(alive)
    if idx.size < 12:
        return None
    px, py = xs[idx], ys[idx]
    H, W = max(fr.height, 1), max(fr.width, 1)
    n = len(px)
    score = np.zeros(n)
    back = np.full(n, -1, int)
    starts = np.searchsorted(px, px, side="left")     # first point in this x column
    for i in range(n):
        s = starts[i]
        if s == 0:
            score[i] = 1.0
            continue
        cost = (score[:s] - jump * np.abs(py[:s] - py[i]) / H
                - gap_pen * (px[i] - px[:s]) / W)
        j = int(np.argmax(cost))
        if cost[j] > 0:
            score[i], back[i] = cost[j] + 1.0, j
        else:
            score[i] = 1.0
    end = int(np.argmax(score))
    path = []
    while end >= 0:
        path.append(end)
        end = back[end]
    path = np.asarray(path[::-1])
    # Skipping columns is free, which is what lets a chain ignore a legend; the cost is
    # that it will also step clean across a wide empty stretch to pick up whatever lies
    # on the far side, and the re-plot then draws a straight line through blank paper.
    # Nothing was read there, so nothing should be reported there: keep the longest run
    # that is actually continuous.
    gaps = np.flatnonzero(np.diff(px[path]) > max(0.08 * W, 20))
    if gaps.size:
        bounds = np.concatenate(([0], gaps + 1, [len(path)]))
        k = int(np.argmax(np.diff(bounds)))
        path = path[bounds[k]:bounds[k + 1]]
    # A curve need not cross the whole plot: an excitation spectrum plotted beside an
    # emission spectrum covers its own third of the axis and nothing more.
    if len(path) < 12 or (px[path].max() - px[path].min()) < 0.15 * W:
        return None
    return idx[path], px[path], py[path]


def chain_traces(mask, labimg, cluster: int, fr, max_chains: int = 2,
                 require_population: bool = True):
    """Trace a symbol-drawn series, and its second branch where the figure has one.

    Replaces column averaging, which cannot tell the curve's own ink from a legend key,
    an inset, or a neighbouring panel that the crop clipped in: it averages whatever
    shares the column. Chaining asks instead which points continue each other.
    """
    cloud = marker_cloud(mask, labimg, cluster, fr, require_population)
    if cloud is None:
        return None
    xs, ys, sym, fill = cloud
    own = (labimg == cluster) & mask
    # Adsorption and desorption are conventionally drawn as solid and open symbols of
    # the same colour, and that convention is the only local evidence that separates
    # them: the two arms are each perfectly continuous, so a most-continuous path is
    # free to run up one and back along the other, paying the crossing once and being
    # rewarded in every column after it. Where both symbol styles are present in
    # quantity, chain each style on its own and the arms come apart by construction.
    sides = _split_by_fill(xs, ys, fill, fr)
    if sides is not None:
        got = [_chain(sx, sy, np.ones(len(sx), bool), fr) for sx, sy in sides]
        got = [g for g in got if g is not None and _plausible(g[2], fr)]
        if len(got) == 2:
            return [(g[1], g[2]) for g in got], float(sym), own, _iso_frac(fill)
    # Two branches of one loop are both continuous, so a "most continuous path" is free
    # to run along one, step across, and run back along the other -- it pays the crossing
    # once and is rewarded in every column after it. Chaining cannot separate them; what
    # separates them is that in the columns where the loop is open, one branch is simply
    # above the other. Split there first, then chain each side to clean up outliers.
    sides = _split_branches(xs, ys, sym, fr)
    if sides is not None:
        out = []
        for sx, sy in sides:
            got = _chain(sx, sy, np.ones(len(sx), bool), fr)
            if got is not None and _plausible(got[2], fr):
                out.append((got[1], got[2]))
        if len(out) == 2:
            return out, float(sym), own, _iso_frac(fill)
    alive = np.ones(len(xs), bool)
    out = []
    for k in range(max_chains):
        got = _chain(xs, ys, alive, fr)
        if got is None:
            break
        sel, cx, cy = got
        if not _plausible(cy, fr):
            break
        if k and not _is_branch(out[0], (cx, cy), fr):
            break
        out.append((cx, cy))
        near = np.abs(ys[:, None] - cy[None, :])
        col = np.abs(xs[:, None] - cx[None, :])
        alive &= ~((near <= 1.5 * sym) & (col <= 1.5 * sym)).any(axis=1)
    # An empty result is not the same as "not symbol-drawn": the colour *was* a
    # population of symbols, and none of it was a series. Returning it, rather than
    # None, is what stops the caller falling back to a tracer that will happily
    # manufacture one out of the same debris.
    return out, float(sym), own, _iso_frac(fill)


def _iso_frac(fill) -> float:
    """How much of a cloud was symbols standing on their own.

    Chaining also handles curves drawn as continuous strokes -- their ink is cut into one
    point per column and chained just the same -- so succeeding at it is not evidence that
    a series is symbol-drawn. Isolated symbols are. This decides only how the series is
    labelled and re-plotted for verification, never how it was read.
    """
    return float(np.mean(fill != -1)) if len(fill) else 0.0


def _emit_chains(out, ch, cluster, Wp: int, style: str) -> None:
    chains, size, cmask, _ = ch
    for cx, cy in chains:
        out.append(Curve(xs=cx, ys=cy, cluster=cluster.idx, rgb=cluster.rgb,
                         linewidth=max(size, 2.0), reward=float(len(cx)),
                         coverage=(cx.max() - cx.min()) / Wp, seg_ids=[], mask=cmask,
                         style=style, chained=True,
                         marker_px=size if style == "markers" else 0.0))


def _plausible(cy, fr) -> bool:
    """Could this chain be a data series at all?

    A colour that survives clustering without being a series -- the blend along the
    boundary between two overlapping curves, say -- still yields a best chain, because
    the DP returns the best path through whatever it is given. What gives it away is
    that its best path is still wild: a real series moves by about a symbol from one
    symbol to the next, so it cannot keep crossing a large part of the plot. A steep
    curve makes such a step once or twice; blend debris makes one every few points.
    """
    if len(cy) <= 2:
        return True
    big = int((np.abs(np.diff(cy)) > 0.25 * max(fr.height, 1)).sum())
    return big <= max(3, 0.01 * len(cy))


def _split_by_fill(xs, ys, fill, fr, min_frac: float = 0.15):
    """Split a symbol cloud into its solid and open symbols, or None if it is not mixed.

    Fused symbols cannot be classified, so they are offered to both sides and the
    continuity of each chain decides which one keeps them.
    """
    W = max(fr.width, 1)
    solid, open_ = fill == 1, fill == 0
    n = max(int(solid.sum()) + int(open_.sum()), 1)
    if solid.sum() < min_frac * n or open_.sum() < min_frac * n:
        return None
    out = []
    for side in (solid, open_):
        sel = side | (fill == -1)
        if sel.sum() < 12 or (xs[sel].max() - xs[sel].min()) < 0.3 * W:
            return None
        out.append((xs[sel], ys[sel]))
    return out


def _split_branches(xs, ys, sym: float, fr, min_frac: float = 0.25):
    """Separate a hysteresis loop into its upper and lower arm, or None if it is not one.

    A loop is open where its two arms differ: those columns hold two groups of symbols
    with clear air between them. Where it is closed the arms coincide and both take the
    same point, which is what the physical curves do at the ends of the loop.

    The test is deliberately hard to pass. Any thick or noisy series has *some* columns
    that momentarily look two-valued, so a loop has to be two-valued over a sustained,
    contiguous stretch of the axis -- an isolated scatter of split columns is a series
    with texture, not two curves.
    """
    H, W = max(fr.height, 1), max(fr.width, 1)
    sep = max(1.5 * sym, 0.03 * H)
    slices = _column_slices(xs)
    if not slices:
        return None
    ux, uy, lx, ly = [], [], [], []
    split_at = []
    for a, b in slices:
        cy = np.sort(ys[a:b])
        runs = group_consecutive(np.round(cy).astype(int), gap=int(max(2, sym)))
        top, bot = runs[0], runs[-1]
        ux.append(xs[a])
        uy.append((top[0] + top[1]) / 2.0)
        lx.append(xs[a])
        ly.append((bot[0] + bot[1]) / 2.0)
        if (bot[0] - top[1]) >= sep:
            split_at.append(xs[a])
    if len(split_at) < min_frac * len(slices):
        return None
    # the open part of a loop is one stretch of the axis, not scattered columns
    span = max(split_at) - min(split_at)
    if span < 0.25 * W or (max(ux) - min(ux)) < 0.3 * W:
        return None
    # and the two arms have to be apart on average, not only where the series happened
    # to break: a single thick or gappy series also yields a "top" and a "bottom" run,
    # but they are the same curve and sit on top of each other almost everywhere.
    if float(np.mean(np.abs(np.asarray(uy) - np.asarray(ly)))) < 0.04 * H:
        return None
    return ((np.asarray(ux, float), np.asarray(uy, float)),
            (np.asarray(lx, float), np.asarray(ly, float)))


def _column_slices(xs):
    """(start, stop) index pairs of each distinct x in a cloud sorted by x."""
    if len(xs) == 0:
        return []
    edges = np.flatnonzero(np.diff(xs) > 0.5) + 1
    starts = np.concatenate(([0], edges))
    stops = np.concatenate((edges, [len(xs)]))
    return list(zip(starts, stops))


def _is_branch(first, second, fr) -> bool:
    """Is this second chain the other arm of the same curve, or a different object?

    A desorption branch runs back across the pressures the adsorption branch covered, so
    it overlaps it in x. An inset, or a neighbour clipped in by the crop, sits off to one
    side. Overlap in x is what separates the two.
    """
    ax, _ = first
    bx, _ = second
    lo, hi = max(ax.min(), bx.min()), min(ax.max(), bx.max())
    if hi <= lo:
        return False
    return (hi - lo) >= 0.6 * (bx.max() - bx.min()) and (bx.max() - bx.min()) >= 0.3 * fr.width


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
                   min_coverage: float = 0.15, verbose=False,
                   bg_color=None, style_hint: str | None = None) -> list[Curve]:
    """`style_hint` says how the panel is drawn -- "lines", "markers",
    "markers_joined_by_lines" or "mixed" -- and picks the tracer.

    The two cases need different algorithms and no amount of pixel statistics decides
    between them reliably: a series of symbols packed tightly enough looks like a stroke,
    and a stroke chopped by overlapping curves looks like symbols. But it is obvious at a
    glance, and a glance is what the classifier already gives -- it is looking at this
    panel anyway, and "how is this drawn" is a question about appearance, which is the
    kind vision models answer well. So it is asked there and dispatched here, instead of
    being guessed at from component-size histograms.
    """
    if not clusters:
        return []
    # A scatter is grouped by its symbols, not by its pixels, and that is a decision
    # about the whole panel rather than about one colour -- two series can share a
    # colour and be told apart by their symbol, which no per-colour loop could do.
    if style_hint in ("markers", "markers_joined_by_lines"):
        ss = symbol_series(rgb, mask, fr, bg_color)
        if ss:
            out = []
            for xs_, ys_, col, size, _f, smask in ss:
                # the colour cluster the points actually sit on, so the verification
                # step compares each series against its own ink and not an index
                yi = np.clip(np.round(ys_).astype(int), 0, labimg.shape[0] - 1)
                xi = np.clip(np.round(xs_).astype(int), 0, labimg.shape[1] - 1)
                lv = labimg[yi, xi]
                lv = lv[lv >= 0]
                cid = int(np.bincount(lv).argmax()) if lv.size else 0
                out.append(Curve(xs=xs_, ys=ys_, cluster=cid, rgb=col,
                                 linewidth=max(size, 2.0), reward=float(len(xs_)),
                                 coverage=(xs_.max() - xs_.min()) / max(fr.width, 1),
                                 seg_ids=[], mask=smask, style="markers", chained=True,
                                 marker_px=size))
            for i in sorted(connector_lines(out, fr), reverse=True):
                del out[i]
            return out

    nodes, segments, sedge, seg_of = build_graph(mask)
    segment_counts(segments, labimg, len(clusters))
    Wp = max(fr.width, 1)
    used = np.zeros(len(segments), bool)
    out: list[Curve] = []

    for c in clusters:
        # Lines, or points? The four cases in practice reduce to two treatments:
        #
        #   1 pure lines                     -> the connectivity graph
        #   2 pure points                    -> one data point per symbol centre
        #   3 points with a guide line in    -> the guide is decoration; the points are
        #     another colour                    the data, so case 2 after dropping it
        #   4 one colour, part symbols part  -> the fused stretch is symbols touching,
        #     fused run                         not a line; still case 2
        #
        # so the only question is which of the two, and it is asked of the classifier
        # because the pixels cannot answer it (see cluster_style for the measurements).
        if style_hint == "lines":
            cstyle = "line"
        elif style_hint in ("markers", "markers_joined_by_lines"):
            cstyle = "points"
        else:
            # "mixed" means the series in this panel are not all drawn the same way, so
            # there is no panel-level answer to apply and it falls to the geometry --
            # which is weak, but here it is choosing between two treatments that are
            # both defensible rather than deciding whether a series exists.
            cstyle, _ = cluster_style(mask, labimg, c.idx, fr)
            cstyle = "line" if cstyle in ("line", "unknown") else "points"

        if cstyle == "points":
            ps = point_series(mask, labimg, c.idx, fr, "mixed")
            if ps is not None:
                series, size, pmask = ps
                for px_, py_, _f in series:
                    out.append(Curve(xs=px_, ys=py_, cluster=c.idx, rgb=c.rgb,
                                     linewidth=max(size, 2.0), reward=float(len(px_)),
                                     coverage=(px_.max() - px_.min()) / Wp, seg_ids=[],
                                     mask=pmask, style="markers", chained=True,
                                     marker_px=size))
                continue

        ch = None
        sc_cols = None
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

        if ch is not None:
            chains, _, _, iso = ch
            best = max((o.coverage for o in out[before:]), default=0.0)
            cbest = max(((cx.max() - cx.min()) / Wp for cx, _ in chains), default=0.0)
            if chains and (cbest > best or (len(chains) == 2 and best < 0.6)):
                del out[before:]
                _emit_chains(out, ch, c, Wp, "markers" if (
                    style_hint not in ("lines", "markers_joined_by_lines")
                    and iso >= 0.20) else "line")
    # A guide line drawn through a scatter is decoration, not data: it interpolates
    # between measurements that were never made. The points already say everything it
    # says, so it goes.
    for i in sorted(connector_lines(out, fr), reverse=True):
        del out[i]
    _relabel(out, rgb, labimg, len(clusters), mask, bg_color)
    out = _dedupe(out, fr, bg_color)
    # Span, not filled-column count, for anything that came out of the point tracer: a
    # symbol series has ink only where a symbol sits, so counting filled columns measures
    # how the authors drew it rather than how completely it was read. Done here, once, on
    # the traces that actually survive, so the number always matches the curve it labels.
    for c in out:
        if (getattr(c, "chained", False)
                or getattr(c, "style", "line") == "markers") and len(c.xs):
            c.coverage = float(np.max(c.xs) - np.min(c.xs)) / max(fr.width, 1)
    return out


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
        if getattr(c, "chained", False):
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
        if getattr(c, "chained", False):
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
