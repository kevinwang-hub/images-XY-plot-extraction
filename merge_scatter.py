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

import numpy as np


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
