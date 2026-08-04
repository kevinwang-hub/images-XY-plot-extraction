#!/usr/bin/env python3
"""Compare the pixel reader and the model reader over every scatter panel in the corpus.

    python3 compare_scatter53.py -o out_scatter_cmp

The earlier comparison scored readers on how many of their points land on ink, which
flatters the pixel reader: its points are *derived* from ink, so they sit on it almost by
construction, while the model's points are produced independently and are held to a
harder standard. Point count was reported the same way, as though more were better -- but
a scatter's truth is the number of symbols the authors plotted, and cutting a fused run
into one point per column oversamples it.

So this compares them on two questions that don't presuppose an answer:

  accuracy   For each model point, how far is the nearest pixel point? Both are in axis
             units, normalised by the plot height, so this asks whether the model put its
             values where data actually is.

  trend      Interpolate the model's sparse points and measure the pixel reader's dense
             trace against that curve, over the range they share. This is the question a
             sparse reading has to answer: if a handful of points reproduce the curve, the
             handful is enough, and density was never the point.

`symbols` is what the geometry counts as distinct plotted symbols -- the closest thing to
a ground-truth point count, and the number both readers should be judged against.
"""
from __future__ import annotations

import argparse
import base64
import io as _io
import json
import os

import cv2
import numpy as np

from pxrd2xy.core import load_rgb, background_color, ink_mask, upscale_if_small
from pxrd2xy.report import CSS
import pxrd2xy.axes as ax_mod
import pxrd2xy.curves as cv_mod


def symbol_count(panel: str, ocr_cache: str | None = None) -> int:
    """Distinct plotted symbols, as the geometry sees them. Fused runs are counted by
    area rather than as one, since a blob of three touching circles is three points."""
    try:
        rgb0 = load_rgb(panel)
    except OSError:
        return 0
    rgb, sc = upscale_if_small(rgb0)
    bg = background_color(rgb)
    ink = ink_mask(rgb, bg, 40)
    oc = ax_mod.run_ocr(rgb, ocr_cache) if ocr_cache else []
    fr = ax_mod.detect_frame(ink, oc)
    pink = cv_mod.plot_ink(ink, fr, np.array([]), np.array([]))
    mask, _ = cv_mod.remove_text_and_legend(pink, oc, fr, 2.0 * max(sc, 1))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if n <= 1:
        return 0
    W, H = max(fr.width, 1), max(fr.height, 1)
    a = stats[1:, cv2.CC_STAT_AREA].astype(float)
    w = stats[1:, cv2.CC_STAT_WIDTH].astype(float)
    h = stats[1:, cv2.CC_STAT_HEIGHT].astype(float)
    asp = np.maximum(w, h) / np.maximum(np.minimum(w, h), 1.0)
    eq = (w <= 0.06 * W) & (h <= 0.06 * H) & (a >= 6) & (asp <= 2.2)
    if eq.sum() < 4:
        return 0
    med = float(np.median(a[eq]))
    total = 0
    for i in range(len(a)):
        if h[i] <= 2 and w[i] >= 4 * np.sqrt(med):
            continue
        total += max(1, int(round(a[i] / med))) if a[i] >= 0.5 * med else 0
    return total


def _norm(xs, ys, xr, yr):
    """Axis units -> fraction of the plot box, so panels can be pooled."""
    return ((np.asarray(xs) - xr[0]) / (xr[1] - xr[0]),
            (np.asarray(ys) - yr[0]) / (yr[1] - yr[0]))


def pixel_norm(xs, ys, panel):
    """The pixel reader's output as a fraction of the plot box.

    It does not always emit axis units. Where an axis was never calibrated -- 28 of the
    53 scatter panels here -- x comes back in pixels and y as a 0-1 fraction already.
    Normalising those by the *model's* axis range compares a fraction against a voltage
    and produces a number that means nothing, which is what an earlier version of this
    metric did.
    """
    ax = panel.get("axis") or {}
    fr = panel.get("frame") or {}
    sc = panel.get("work_scale") or 1.0
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    xr, yr = ax.get("x_range"), ax.get("y_range")
    if xr and xr[1] != xr[0]:
        xn = (xs - xr[0]) / (xr[1] - xr[0])
    elif fr and fr.get("right", 0) != fr.get("left", 0):
        xn = (xs * sc - fr["left"]) / (fr["right"] - fr["left"])
    else:
        return None, None
    yn = (ys - yr[0]) / (yr[1] - yr[0]) if (yr and yr[1] != yr[0]) else ys
    return xn, yn


def accuracy(lx, ly, pxn, pyn, xr, yr) -> float:
    """Median distance from each model point to the nearest pixel point, in % of the box."""
    if len(lx) == 0 or len(pxn) == 0:
        return float("nan")
    a = np.stack(_norm(lx, ly, xr, yr), 1)
    b = np.stack([np.asarray(pxn), np.asarray(pyn)], 1)
    step = max(1, len(b) // 4000)          # the dense trace does not need every point
    b = b[::step]
    d = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1))
    return float(np.median(d.min(1)) * 100)


def _one_trend(lxn, lyn, pxn, pyn) -> float:
    o = np.argsort(lxn)
    lxn, lyn = lxn[o], lyn[o]
    lxn, idx = np.unique(lxn, return_index=True)
    lyn = lyn[idx]
    if len(lxn) < 2:
        return float("nan")
    inside = (pxn >= lxn.min()) & (pxn <= lxn.max())
    if inside.sum() < 5:
        return float("nan")
    return float(np.median(np.abs(pyn[inside] - np.interp(pxn[inside], lxn, lyn))) * 100)


def trend(llm_series, pixel_series, xr, yr) -> float:
    """Median vertical gap between a pixel trace and the model's interpolated curve.

    Series must be matched before interpolating. Pooling every series in a panel and
    interpolating across them builds a curve that zigzags between two unrelated levels --
    on a panel carrying chi on the left axis and chi*T on the right, that measures the gap
    between the two axes rather than anything either reader got wrong. So each model
    series is matched to the pixel curve it sits closest to, and only that pair is
    compared.

    This is the measure that decides whether a sparse reading is sufficient: it asks what
    you lose by having only the model's points and joining them up.
    """
    out = []
    for lx, ly in llm_series:
        if len(lx) < 2:
            continue
        lxn, lyn = _norm(lx, ly, xr, yr)
        best = float("nan")
        for pxn, pyn in pixel_series:
            if len(pxn) < 5:
                continue
            v = _one_trend(lxn, lyn, np.asarray(pxn), np.asarray(pyn))
            if np.isfinite(v) and (not np.isfinite(best) or v < best):
                best = v
        if np.isfinite(best):
            out.append(best)
    return float(np.median(out)) if out else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pixels", default="out_scatter53/records.json")
    ap.add_argument("--llm", default="out_llm53.json")
    ap.add_argument("--deferred", default="out_papers/records.json")
    ap.add_argument("-o", "--out", default="out_scatter_cmp")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    prec = json.load(open(a.pixels))
    llm = json.load(open(a.llm))
    llm = llm.get("results", llm)
    want = {os.path.basename(d["panel_image"])[:-4]
            for x in json.load(open(a.deferred)) for d in (x.get("deferred") or [])}

    rows = []
    for x in prec:
        for p in x.get("panels", []):
            pid = os.path.basename(p.get("panel_image", ""))[:-4]
            if pid not in want:
                continue
            axis = p.get("axis") or {}
            xr, yr = axis.get("x_range"), axis.get("y_range")
            cs = p.get("curves", [])
            if not cs:
                continue
            pser_n = []
            for c in cs:
                a_, b_ = pixel_norm(c["x"], c["y"], p)
                if a_ is not None and len(a_) >= 5:
                    pser_n.append((a_, b_))
            if not pser_n:
                continue
            px = np.concatenate([a_ for a_, _b in pser_n])
            py = np.concatenate([b_ for _a, b_ in pser_n])
            d = llm.get(pid) or {}
            if not xr and d.get("x_min") is not None:
                xr = [d["x_min"], d["x_max"]]
            if not yr and d.get("y_min") is not None:
                yr = [d["y_min"], d["y_max"]]
            if not (xr and yr) or xr[1] == xr[0] or yr[1] == yr[0]:
                continue
            lx, ly, lser = [], [], []
            for s in d.get("series", []):
                sx, sy = [], []
                for q in s.get("points", []):
                    if isinstance(q, dict) and q.get("x") is not None:
                        sx.append(q["x"])
                        sy.append(q["y"])
                if sx:
                    lser.append((np.asarray(sx, float), np.asarray(sy, float)))
                    lx += sx
                    ly += sy

            rows.append(dict(
                panel=pid, category=p.get("category"), figure=p.get("figure"),
                paper=x.get("paper_id"), panel_image=p.get("panel_image"),
                symbols=symbol_count(p.get("panel_image", "")),
                n_pixel_curves=len(pser_n), n_pixel=int(len(px)),
                x_calibrated=bool((p.get("axis") or {}).get("x_calibrated")),
                y_calibrated=bool((p.get("axis") or {}).get("y_calibrated")),
                n_llm_series=len(d.get("series", [])), n_llm=int(len(lx)),
                accuracy_pct=accuracy(lx, ly, px, py, xr, yr),
                trend_pct=trend(lser, pser_n, xr, yr),
            ))
            print(f"  {pid[:44]:46s} sym={rows[-1]['symbols']:4d} "
                  f"px={rows[-1]['n_pixel']:5d} llm={rows[-1]['n_llm']:4d} "
                  f"acc={rows[-1]['accuracy_pct']:5.2f}% trend={rows[-1]['trend_pct']:5.2f}%")

    json.dump(rows, open(os.path.join(a.out, "scatter53.json"), "w"), indent=1)
    ok = [r for r in rows if np.isfinite(r["trend_pct"])]
    if ok:
        print(f"\n{len(rows)} panels compared")
        print(f"  symbols (geometry)  median {int(np.median([r['symbols'] for r in rows]))}")
        print(f"  points, pixels      median {int(np.median([r['n_pixel'] for r in rows]))}")
        print(f"  points, model       median {int(np.median([r['n_llm'] for r in rows]))}")
        print(f"  accuracy            median {np.median([r['accuracy_pct'] for r in ok]):.2f}% of box")
        print(f"  trend fidelity      median {np.median([r['trend_pct'] for r in ok]):.2f}% of box")
    print("wrote", os.path.join(a.out, "scatter53.json"))


if __name__ == "__main__":
    main()
