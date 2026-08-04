#!/usr/bin/env python3
"""Run the three scatter readers over the same panels and build a comparison page.

    python3 compare_scatter.py -o out_compare

The readers differ in what they trust. Ours groups whole symbols by colour, size and
fill. `colordist` keeps every pixel within a wide radius of one reference colour, the
way WebPlotDigitizer does, and takes blob centroids with no requirement that the blobs
resemble each other. `template` recognises symbols by matching one symbol's *shape*
against the binary mask, and only consults colour afterwards to say which series a match
belongs to.

The point of the page is to be looked at: each panel shows the original, then each
reader's points drawn over the faded original, with its series count.
"""
from __future__ import annotations

import argparse
import base64
import io as _io
import json
import os

import numpy as np

import cv2

from pxrd2xy.core import load_rgb, background_color, ink_mask
from pxrd2xy.pipeline import digitize
from pxrd2xy.report import CSS
from paper2xy.classify import classify
from paper2xy import llm_points

READERS = [("symbols", "ours — group whole symbols by colour, size and fill"),
           ("colordist", "colour distance — one reference colour, wide radius, blob centroids"),
           ("template", "template match — recognise the symbol's shape, colour only after"),
           ("llm", "the model reads the values straight off the image, in axis units")]


def on_ink(px, py, ink, tol: int = 2) -> float:
    """Fraction of points landing within `tol` px of ink — the same question of every
    reader, and the only one that does not depend on how the answer was produced."""
    if len(px) == 0:
        return 0.0
    H, W = ink.shape
    k = np.ones((2 * tol + 1, 2 * tol + 1), np.uint8)
    grown = cv2.dilate(ink.astype(np.uint8), k).astype(bool)
    xi = np.clip(np.round(np.asarray(px)).astype(int), 0, W - 1)
    yi = np.clip(np.round(np.asarray(py)).astype(int), 0, H - 1)
    return float(grown[yi, xi].mean())


def draw_points(rgb, series, path):
    """The same faded-original view the pixel readers produce, for the model's points."""
    out = (rgb.astype(np.float32) * 0.22 + 255 * 0.78).astype(np.uint8)
    out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    palette = [(200, 60, 60), (60, 60, 210), (40, 150, 60), (170, 90, 190), (30, 140, 190)]
    for i, (px, py) in enumerate(series):
        col = palette[i % len(palette)]
        for x, y in zip(px, py):
            cv2.circle(out, (int(round(x)), int(round(y))), 3, col, -1, cv2.LINE_AA)
    cv2.imwrite(path, out)


def llm_series_to_pixels(rec, data):
    """Map the model's answer, which is in axis units, back into the panel."""
    ax, fr, sc = rec.get("axis") or {}, rec.get("frame") or {}, rec.get("work_scale") or 1.0
    xr, yr = ax.get("x_range"), ax.get("y_range")
    # Where our own OCR did not calibrate an axis, use the ranges the model read off the
    # box itself. That makes this reader self-sufficient, which is the fairer test: it is
    # not being carried by the pixel pipeline it is being compared against.
    if not xr and data.get("x_min") is not None:
        xr = [data["x_min"], data["x_max"]]
    if not yr and data.get("y_min") is not None:
        yr = [data["y_min"], data["y_max"]]
    if not (xr and yr and fr) or xr[1] == xr[0] or yr[1] == yr[0]:
        return None
    L, R = fr["left"] / sc, fr["right"] / sc
    T, B = fr["top"] / sc, fr["bottom"] / sc
    out = []
    for srs in data.get("series", []):
        pts = [(p.get("x"), p.get("y")) for p in srs.get("points", [])
               if isinstance(p, dict) and p.get("x") is not None and p.get("y") is not None]
        if not pts:
            continue
        a = np.asarray(pts, float)
        px = L + (a[:, 0] - xr[0]) / (xr[1] - xr[0]) * (R - L)
        py = B + (a[:, 1] - yr[0]) / (yr[1] - yr[0]) * (T - B)
        out.append((px, py, srs.get("label", "")))
    return out


def _uri(path: str) -> str:
    import cv2
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return ""
    h, w = img.shape[:2]
    if w > 760:
        img = cv2.resize(img, (760, int(h * 760 / w)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".webp", img, [cv2.IMWRITE_WEBP_QUALITY, 88])
    return "data:image/webp;base64," + base64.b64encode(buf.tobytes()).decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--panels", default="out_papers/panels")
    ap.add_argument("-o", "--out", default="out_compare")
    ap.add_argument("--cache", default="out_papers/.ocr")
    ap.add_argument("-n", "--limit", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    names = sorted(n[:-4] for n in os.listdir(a.panels) if n.endswith(".png"))
    items = []
    for n in names:
        try:
            items.append({"id": n, "rgb": load_rgb(os.path.join(a.panels, n + ".png")),
                          "caption": "", "label": n})
        except OSError:
            continue                     # a panel left over from an earlier run
    names = [it["id"] for it in items]
    cls = classify(items)
    scatter = [n for n in names
               if cls.get(n, {}).get("render_style") in ("markers", "markers_joined_by_lines")]
    if a.limit:
        scatter = scatter[:a.limit]
    print(f"{len(scatter)} scatter panels of {len(names)}")

    rows = []
    for n in scatter:
        cell = []
        panel = os.path.join(a.panels, n + ".png")
        rgb0 = load_rgb(panel)
        ink0 = ink_mask(rgb0, background_color(rgb0), 40)
        rec0 = None
        for key, _desc in READERS:
            if key == "llm":
                sub = os.path.join(a.out, "llm")
                os.makedirs(sub, exist_ok=True)
                try:
                    data = llm_points.read_points(
                        rgb0, (rec0 or {}).get("axis"), cls.get(n, {}).get("caption", ""))
                    mapped = llm_series_to_pixels(rec0 or {}, data)
                except Exception as exc:
                    print(f"    llm error on {n}: {exc}")
                    cell.append((key, 0, 0, "", f"error: {exc}", 0.0))
                    continue
                if not mapped:
                    cell.append((key, 0, 0, "", "no calibration to map onto", 0.0))
                    continue
                ov = os.path.join(sub, n + "__overlay.png")
                draw_points(rgb0, [(p, q) for p, q, _l in mapped], ov)
                allx = np.concatenate([p for p, _q, _l in mapped])
                ally = np.concatenate([q for _p, q, _l in mapped])
                cell.append((key, len(mapped), int(len(allx)), _uri(ov),
                             ", ".join(l for _p, _q, l in mapped)[:60],
                             on_ink(allx, ally, ink0)))
                continue
            os.environ["PXRD_SCATTER"] = key
            sub = os.path.join(a.out, key)
            os.makedirs(sub, exist_ok=True)
            try:
                r = digitize(os.path.join(a.panels, n + ".png"), sub, cache_dir=a.cache,
                             save_visuals=True, name=n,
                             style_hint=cls.get(n, {}).get("render_style"))
            except Exception as exc:                       # a reader failing is a result
                cell.append((key, 0, 0, "", f"error: {exc}"))
                continue
            if rec0 is None:
                rec0 = r
            cs = [c for c in r["curves"] if c["style"] == "markers"]
            npts = sum(len(c["x"]) for c in cs)
            ov = os.path.join(sub, n + "__overlay.png")
            px = np.concatenate([np.asarray(c["px"]) for c in cs]) if cs else np.zeros(0)
            py = np.concatenate([np.asarray(c["py"]) for c in cs]) if cs else np.zeros(0)
            cell.append((key, len(cs), npts, _uri(ov) if os.path.exists(ov) else "",
                         ", ".join(c["status"] for c in cs), on_ink(px, py, ink0)))
        rows.append((n, cls.get(n, {}), cell))
        print(f"  {n[:44]:46s} " + "  ".join(
            f"{k}:{s}sr/{p}pt/ink{oi:.2f}" for k, s, p, _, _, oi in cell))

    say = lambda c: c.get("n_curves", "?")
    body = []
    for n, c, cell in rows:
        body.append(
            f'<div class="cmp"><div class="hd"><b>{n}</b> '
            f'<span class="m">{c.get("category","?")} · drawn as '
            f'{c.get("render_style","?")} · the model sees {say(c)} series</span></div>'
            '<div class="grid">' + "".join(
                f'<div class="col"><div class="lab">{k}</div>'
                f'<div class="n">{s} series · {p:,} points · '
                f'<b>{oi*100:.0f}%</b> on ink</div>'
                f'<div class="st">{st or "—"}</div>'
                + (f'<img src="{u}">' if u else '<div class="miss">no output</div>')
                + '</div>' for k, s, p, u, st, oi in cell) + '</div></div>')

    extra = """
.cmp{border:1px solid var(--line);border-radius:12px;margin:16px 0;background:var(--surf)}
.hd{padding:11px 15px;border-bottom:1px solid var(--line);font-size:13px}
.hd .m{color:var(--tx3);margin-left:8px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px;padding:14px}
.col{min-width:0}
.lab{font-size:12px;font-weight:600;text-transform:uppercase;letter-spacing:.05em}
.n{font-size:12.5px;color:var(--tx2);margin-top:2px}
.st{font-size:11.5px;color:var(--tx3);margin-bottom:6px}
.col img{width:100%;border:1px solid var(--line);border-radius:8px;display:block}
.miss{color:var(--tx3);font-size:12px;padding:20px;border:1px dashed var(--line);border-radius:8px}
.key{display:flex;gap:18px;flex-wrap:wrap;margin:6px 0 18px}
.key div{font-size:12.5px;color:var(--tx2);max-width:34ch}
.key b{color:var(--tx);display:block;font-size:12px;text-transform:uppercase;letter-spacing:.05em}
.note{color:var(--tx3);font-size:12.5px;max-width:70ch;margin-top:10px}
table td b{font-family:ui-monospace,Menlo,monospace}
"""
    key = "".join(f"<div><b>{k}</b>{d}</div>" for k, d in READERS)
    summary = ""
    for k, _d in READERS:
        oi = [c[5] for _n, _c, cell in rows for c in cell if c[0] == k]
        pt = [c[2] for _n, _c, cell in rows for c in cell if c[0] == k]
        if not oi:
            continue
        summary += (f"<tr><td><b>{k}</b></td><td>{np.median(oi)*100:.0f}%</td>"
                    f"<td>{np.mean(oi)*100:.0f}%</td>"
                    f"<td>{int(np.median(pt)):,}</td><td>{sum(pt):,}</td></tr>")
    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Scatter readers compared</title><style>{CSS}{extra}</style></head><body>
<div class="wrap"><h1>Scatter readers compared</h1>
<p class="sub">The same panels read four ways. Every view is the reader's points drawn
over the faded original — what you are judging is whether a dot sits on each symbol and
whether any symbol was missed. <b>On ink</b> is the fraction of a reader's points that
land within 2&nbsp;px of ink in the original figure — the same question asked of every
reader, and the only one that does not depend on how the answer was produced.</p>
<div class="key">{key}</div>
<h2>Across the {len(rows)} scatter panels</h2>
<table><thead><tr><th>reader</th><th>median on ink</th><th>mean on ink</th>
<th>median points</th><th>total points</th></tr></thead><tbody>{summary}</tbody></table>
<p class="note">Points landing on ink is a floor, not a grade — on a dense scatter almost
anything near the curve is ink. It is decisive only in one direction: a reader below it
is putting values where the figure has none.</p>
{''.join(body)}
</div></body></html>"""
    path = os.path.join(a.out, "compare.html")
    with _io.open(path, "w", encoding="utf-8") as fh:
        fh.write(html)
    json.dump([{"panel": n, "counts": {k: {"series": s, "points": p, "on_ink": oi}
                                       for k, s, p, _, _, oi in cell}} for n, _c, cell in rows],
              open(os.path.join(a.out, "compare.json"), "w"), indent=1)
    print("wrote", path)


if __name__ == "__main__":
    main()
