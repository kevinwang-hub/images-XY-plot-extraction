#!/usr/bin/env python3
"""Fold the scatter panels back into the corpus record, from both readers.

    python3 merge_scatter.py

The corpus run digitised the line plots and deferred the scatter. This merges the
deferred panels back in twice: once as the pixel reader read them, once as the model
read them. Both land in the same pool, each curve carrying a `method` that says which
produced it -- pooling them without that label would be the one thing a shared dataset
cannot survive, because the two readers fail in different ways and a consumer has to be
able to select on it.
"""
from __future__ import annotations

import argparse
import json
import os

import cv2
import numpy as np

from pxrd2xy.core import load_rgb


def draw_llm_overlay(panel: dict, entry: dict, figs_dir: str) -> str | None:
    """The model's claimed values, plotted back onto the faded panel.

    No score is attached to this view on purpose. The model's numbers were not derived
    from the pixels, so pixel agreement is not their test -- the person checking the
    panel is. What the drawing needs is only the plot box and an axis range: value ->
    fraction of the range -> position inside the box. Points the model places outside
    the box are clamped to its edge and ringed, so a wrong axis reading shows up as a
    rim of hollow markers instead of as nothing.
    """
    fr = panel.get("frame") or {}
    ax = panel.get("axis") or {}
    sc = panel.get("work_scale") or 1.0
    img_path = panel.get("panel_image", "")
    if not fr or not os.path.exists(img_path):
        return None
    xr, yr = ax.get("x_range"), ax.get("y_range")
    if not xr and entry.get("x_min") is not None:
        xr = [entry["x_min"], entry["x_max"]]
    if not yr and entry.get("y_min") is not None:
        yr = [entry["y_min"], entry["y_max"]]
    if not (xr and yr) or xr[1] == xr[0] or yr[1] == yr[0]:
        return None
    rgb = load_rgb(img_path)
    img = (rgb.astype(np.float32) * 0.25 + 255 * 0.75).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    L, R = fr["left"] / sc, fr["right"] / sc
    T, B = fr["top"] / sc, fr["bottom"] / sc
    pal = [(60, 60, 210), (200, 60, 60), (40, 150, 60), (170, 90, 190), (30, 140, 190)]
    H, W = img.shape[:2]
    clamped = 0
    for i, srs in enumerate(entry.get("series", [])):
        col = pal[i % len(pal)]
        for q in srs.get("points", []):
            if not isinstance(q, dict) or q.get("x") is None or q.get("y") is None:
                continue
            x = L + (q["x"] - xr[0]) / (xr[1] - xr[0]) * (R - L)
            y = B + (q["y"] - yr[0]) / (yr[1] - yr[0]) * (T - B)
            inside = (L - 4 <= x <= R + 4) and (T - 4 <= y <= B + 4)
            cx = int(round(min(max(x, L), R)))
            cy = int(round(min(max(y, T), B)))
            if 0 <= cx < W and 0 <= cy < H:
                if inside:
                    cv2.circle(img, (cx, cy), 4, col, -1, cv2.LINE_AA)
                else:
                    cv2.circle(img, (cx, cy), 5, col, 2, cv2.LINE_AA)
                    clamped += 1
    if clamped:
        cv2.putText(img, f"{clamped} points outside the plot box (rings, clamped)",
                    (int(L), max(14, int(T) - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (30, 30, 200), 1, cv2.LINE_AA)
    name = os.path.basename(img_path)[:-4] + "__llm.png"
    os.makedirs(figs_dir, exist_ok=True)
    cv2.imwrite(os.path.join(figs_dir, name), img)
    return name


def llm_curves(entry: dict, panel: dict, model: str) -> list:
    """The model's answer as curve records, in the same shape as a traced curve."""
    axis = panel.get("axis") or {}
    xr, yr = axis.get("x_range"), axis.get("y_range")
    if not xr and entry.get("x_min") is not None:
        xr = [entry["x_min"], entry["x_max"]]
    if not yr and entry.get("y_min") is not None:
        yr = [entry["y_min"], entry["y_max"]]
    out = []
    for i, s in enumerate(entry.get("series", [])):
        pts = [(q.get("x"), q.get("y")) for q in s.get("points", [])
               if isinstance(q, dict) and q.get("x") is not None and q.get("y") is not None]
        if len(pts) < 3:
            continue
        a = np.asarray(sorted(pts), float)
        out.append(dict(
            index=1000 + i, legend=s.get("label", ""), legend_source="model",
            style="markers", method="llm", model=model,
            marker=s.get("marker", ""), color_name=s.get("color", ""),
            x=[float(v) for v in a[:, 0]], y=[float(v) for v in a[:, 1]],
            status="unverified",
            quality=dict(n_points=len(a),
                         x_span=float(a[:, 0].max() - a[:, 0].min()),
                         axis_from="ocr" if (axis.get("x_range")) else "model"),
        ))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="out_papers/records.json")
    ap.add_argument("--pixels", default="out_scatter53/records.json")
    ap.add_argument("--llm", default="out_llm53.json")
    ap.add_argument("--scores", default="out_scatter_cmp/scatter53.json")
    ap.add_argument("--model", default="claude-opus-5")
    a = ap.parse_args()

    corpus = json.load(open(a.corpus))
    prec = json.load(open(a.pixels))
    raw = json.load(open(a.llm))
    llm = raw.get("results", raw)
    scores = {}
    if os.path.exists(a.scores):
        scores = {r["panel"]: r for r in json.load(open(a.scores))}

    want = {os.path.basename(d["panel_image"])[:-4]: (x["paper_id"], d)
            for x in corpus for d in (x.get("deferred") or [])}
    by_paper = {x["paper_id"]: x for x in corpus}

    added_px = added_llm = 0
    for x in prec:
        for p in x.get("panels", []):
            pid = os.path.basename(p.get("panel_image", ""))[:-4]
            if pid not in want:
                continue
            host = by_paper.get(x["paper_id"])
            if host is None:
                continue
            for c in p.get("curves", []):
                c["method"] = "pixels"
            entry = llm.get(pid) or {}
            lc = llm_curves(entry, p, a.model)
            ov = draw_llm_overlay(p, entry, os.path.join(os.path.dirname(a.corpus), "figs"))
            if ov:
                p["llm_overlay"] = ov
            sc = scores.get(pid)
            if sc:
                for c in lc:
                    c["quality"]["accuracy_pct_of_box"] = sc.get("accuracy_pct")
                    c["quality"]["trend_pct_of_box"] = sc.get("trend_pct")
                    c["quality"]["symbols_detected"] = sc.get("symbols")
            p["curves"] = list(p.get("curves", [])) + lc
            added_px += len([c for c in p["curves"] if c.get("method") == "pixels"])
            added_llm += len(lc)
            host.setdefault("panels", []).append(p)
        # the panels are no longer deferred once they are in
        host = by_paper.get(x["paper_id"])
        if host is not None:
            done = {os.path.basename(p.get("panel_image", ""))[:-4]
                    for p in x.get("panels", [])}
            host["deferred"] = [d for d in (host.get("deferred") or [])
                                if os.path.basename(d["panel_image"])[:-4] not in done]
            host["n_deferred"] = len(host["deferred"])

    json.dump(corpus, open(a.corpus, "w"))
    tot = sum(len(p.get("curves", [])) for x in corpus for p in x.get("panels", []))
    print(f"merged: {added_px} pixel curves + {added_llm} model curves over the scatter "
          f"panels; corpus now holds {tot} curves across "
          f"{sum(len(x.get('panels', [])) for x in corpus)} panels")
    left = sum(x.get("n_deferred", 0) for x in corpus)
    print(f"still deferred: {left}")


if __name__ == "__main__":
    main()
