"""Two alternative scatter readers, for comparison against the built-in one.

Both are adapted from WebPlotDigitizer (Ankit Rohatgi, AGPL-3.0), whose automatic
extraction has been solving this problem for a decade with a very different set of
assumptions from ours. Neither is a port; each takes the one idea that bears on the
failure we still have -- symbols too pale to survive colour clustering.

`colordist`  WPD asks the user for the series colour and keeps every pixel within a
             Euclidean distance of it in RGB, defaulting to 120, which is enormous. It
             never asks whether the colours *cluster*; a pale version of a colour is
             simply within the radius. Its blob detector then takes connected components
             between a minimum and maximum diameter and reports each centroid. Nothing
             requires a blob to resemble the others in size, which is what our own modal
             size band was quietly enforcing.

`template`   WPD's template matcher takes one symbol the user points at, and slides it
             over the *binary* image scoring the normalised overlap
             sum(T*I)/sqrt(sum(T^2)*sum(I^2)), keeping matches above a threshold and
             suppressing all but the best within one template. Because the score is
             computed on the mask and not on colour, a symbol that is barely darker than
             the paper matches exactly as well as a solid one -- shape carries the
             recognition, and colour is only consulted afterwards to say which series a
             match belongs to.
"""
from __future__ import annotations

import numpy as np
import cv2

from .core import group_consecutive


def _blob_points(binary, mask, min_dia, max_dia, H):
    """Connected components between two diameters, as centre points.

    Blobs far larger than one symbol are runs of symbols that touch, so they give one
    point per column rather than being discarded.
    """
    n, lab, stats, cent = cv2.connectedComponentsWithStats(
        (binary & mask).astype(np.uint8), 8)
    xs, ys, sizes = [], [], []
    for i in range(1, n):
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        a = stats[i, cv2.CC_STAT_AREA]
        dia = max(w, h)
        if dia < min_dia or a < 4:
            continue
        if dia <= max_dia:
            xs.append(float(cent[i][0]))
            ys.append(float(cent[i][1]))
            sizes.append(float(np.sqrt(a)))
            continue
        x0, y0 = int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP])
        sub = lab[y0:y0 + h, x0:x0 + w] == i
        for c in range(sub.shape[1]):
            rows = np.flatnonzero(sub[:, c])
            if rows.size == 0:
                continue
            for a0, b0 in group_consecutive(rows, gap=2):
                if (b0 - a0 + 1) > 0.5 * H:
                    continue
                xs.append(float(x0 + c))
                ys.append(float(y0 + (a0 + b0) / 2.0))
                sizes.append(float(b0 - a0 + 1))
    return np.asarray(xs), np.asarray(ys), sizes


def points_colordist(rgb, mask, fr, bg, distance: float = 100.0, min_points: int = 6):
    """WPD's model: one reference colour per series, a generous radius, blob centroids.

    Returns [(xs, ys, rgb, size, mask)].
    """
    W, H = max(fr.width, 1), max(fr.height, 1)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if n <= 1:
        return []
    areas = stats[1:, cv2.CC_STAT_AREA].astype(float)
    ws = stats[1:, cv2.CC_STAT_WIDTH].astype(float)
    hs = stats[1:, cv2.CC_STAT_HEIGHT].astype(float)
    aspect = np.maximum(ws, hs) / np.maximum(np.minimum(ws, hs), 1.0)
    equant = (ws <= 0.06 * W) & (hs <= 0.06 * H) & (areas >= 6) & (aspect <= 2.2)
    if equant.sum() < min_points:
        return []
    modal = float(np.median(np.sqrt(areas[equant])))

    # reference colours: the mean colour of each symbol, grouped the way WPD groups the
    # colours of an image -- greedily, in RGB, with a radius far wider than any halo
    cols, weights = [], []
    for i in np.flatnonzero(equant):
        px = rgb[lab == (i + 1)].reshape(-1, 3).astype(float)
        c = px.mean(0)
        for k, (cc, wt) in enumerate(zip(cols, weights)):
            if float(np.linalg.norm(cc - c)) <= distance:
                cols[k] = (cc * wt + c) / (wt + 1.0)
                weights[k] = wt + 1
                break
        else:
            cols.append(c)
            weights.append(1)
    # WPD's radius is 120 in RGB, which is enormous -- safe there because a human picks
    # the colour and means it. Automatically it merges a blue series into a magenta one.
    # The equivalent automatic choice is half the distance to the nearest other reference
    # colour in this panel, so the radius is as generous as it can be without reaching a
    # neighbour.
    order = np.argsort(-np.asarray(weights))
    radii = {}
    for k in order:
        others = [np.linalg.norm(cols[k] - cols[j]) for j in range(len(cols)) if j != k]
        radii[k] = float(np.clip(0.5 * min(others) if others else distance, 25.0, distance))
    out = []
    claimed = np.zeros(mask.shape, bool)
    img = rgb.astype(np.float32)
    for k in order:
        if weights[k] < min_points:
            continue
        d = np.linalg.norm(img - np.asarray(cols[k], np.float32), axis=2)
        near = (d <= radii[k]) & mask & ~claimed
        if near.sum() < 20:
            continue
        xs, ys, sizes = _blob_points(near, mask, max(3.0, 0.3 * modal),
                                     max(4.0 * modal, 12.0), H)
        if len(xs) < min_points or (xs.max() - xs.min()) < 0.1 * W:
            continue
        claimed |= near
        o = np.argsort(xs)
        out.append((xs[o], ys[o], tuple(int(v) for v in cols[k]),
                    float(np.median(sizes)) if sizes else modal, near.copy()))
    return out


def points_template(rgb, mask, fr, bg, threshold: float = 0.62, min_points: int = 6):
    """WPD's template matcher: recognise symbols by shape on the binary mask.

    Colour is consulted only after a match, to say which series it belongs to -- so a
    symbol too faint for colour clustering is still found, as long as it is the same
    shape as one that is not.
    """
    W, H = max(fr.width, 1), max(fr.height, 1)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if n <= 1:
        return []
    areas = stats[1:, cv2.CC_STAT_AREA].astype(float)
    ws = stats[1:, cv2.CC_STAT_WIDTH].astype(float)
    hs = stats[1:, cv2.CC_STAT_HEIGHT].astype(float)
    aspect = np.maximum(ws, hs) / np.maximum(np.minimum(ws, hs), 1.0)
    equant = (ws <= 0.06 * W) & (hs <= 0.06 * H) & (areas >= 8) & (aspect <= 2.2)
    if equant.sum() < min_points:
        return []
    # the template is the symbol closest to the modal size: the one most likely to be a
    # clean, unoverlapped example of what the series is drawn with
    med = float(np.median(areas[equant]))
    cand = np.flatnonzero(equant)
    seed = cand[int(np.argmin(np.abs(areas[cand] - med)))] + 1
    tx, ty = int(stats[seed, cv2.CC_STAT_LEFT]), int(stats[seed, cv2.CC_STAT_TOP])
    tw, th = int(stats[seed, cv2.CC_STAT_WIDTH]), int(stats[seed, cv2.CC_STAT_HEIGHT])
    if tw < 3 or th < 3:
        return []
    templ = (lab[ty:ty + th, tx:tx + tw] == seed).astype(np.float32)

    src = mask.astype(np.float32)
    score = cv2.matchTemplate(src, templ, cv2.TM_CCORR_NORMED)
    hits = np.argwhere(score >= threshold)
    if len(hits) == 0:
        return []
    # keep the best match within one template of any other, as WPD does
    vals = score[hits[:, 0], hits[:, 1]]
    keep, taken = [], np.zeros(score.shape, bool)
    for idx in np.argsort(-vals):
        y, x = hits[idx]
        if taken[max(0, y - th // 2):y + th // 2 + 1,
                 max(0, x - tw // 2):x + tw // 2 + 1].any():
            continue
        taken[y, x] = True
        keep.append((x + tw / 2.0, y + th / 2.0))
    if len(keep) < min_points:
        return []

    # colour each match by the ink under it, then group those colours generously
    pts, cols = [], []
    for cx, cy in keep:
        x0, y0 = int(cx - tw / 2), int(cy - th / 2)
        sub = mask[max(0, y0):y0 + th, max(0, x0):x0 + tw]
        pix = rgb[max(0, y0):y0 + th, max(0, x0):x0 + tw][sub]
        if pix.size == 0:
            continue
        pts.append((cx, cy))
        cols.append(pix.reshape(-1, 3).astype(float).mean(0))
    if len(pts) < min_points:
        return []
    P, C = np.asarray(pts), np.asarray(cols)
    groups, gc = [], []
    for i in range(len(P)):
        for k, cc in enumerate(gc):
            if float(np.linalg.norm(cc - C[i])) <= 90.0:
                groups[k].append(i)
                gc[k] = (cc * (len(groups[k]) - 1) + C[i]) / len(groups[k])
                break
        else:
            groups.append([i])
            gc.append(C[i].copy())
    out = []
    for g, cc in zip(groups, gc):
        if len(g) < min_points:
            continue
        q = P[g]
        o = np.argsort(q[:, 0])
        if (q[:, 0].max() - q[:, 0].min()) < 0.1 * W:
            continue
        m = np.zeros(mask.shape, bool)
        for cx, cy in q:
            y0, x0 = int(cy - th / 2), int(cx - tw / 2)
            m[max(0, y0):y0 + th, max(0, x0):x0 + tw] |= \
                mask[max(0, y0):y0 + th, max(0, x0):x0 + tw]
        out.append((q[o, 0], q[o, 1], tuple(int(v) for v in cc),
                    float(np.sqrt(med)), m))
    return out
