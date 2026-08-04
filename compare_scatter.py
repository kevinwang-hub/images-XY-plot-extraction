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

from pxrd2xy.core import load_rgb
from pxrd2xy.pipeline import digitize
from pxrd2xy.report import CSS
from paper2xy.classify import classify

READERS = [("symbols", "ours — group whole symbols by colour, size and fill"),
           ("colordist", "colour distance — one reference colour, wide radius, blob centroids"),
           ("template", "template match — recognise the symbol's shape, colour only after")]


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
        for key, _desc in READERS:
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
            cs = [c for c in r["curves"] if c["style"] == "markers"]
            npts = sum(len(c["x"]) for c in cs)
            ov = os.path.join(sub, n + "__overlay.png")
            cell.append((key, len(cs), npts, _uri(ov) if os.path.exists(ov) else "",
                         ", ".join(c["status"] for c in cs)))
        rows.append((n, cls.get(n, {}), cell))
        print(f"  {n[:46]:48s} " + "  ".join(f"{k}:{s}series/{p}pts" for k, s, p, _, _ in cell))

    say = lambda c: c.get("n_curves", "?")
    body = []
    for n, c, cell in rows:
        body.append(
            f'<div class="cmp"><div class="hd"><b>{n}</b> '
            f'<span class="m">{c.get("category","?")} · drawn as '
            f'{c.get("render_style","?")} · the model sees {say(c)} series</span></div>'
            '<div class="grid">' + "".join(
                f'<div class="col"><div class="lab">{k}</div>'
                f'<div class="n">{s} series · {p:,} points</div>'
                f'<div class="st">{st or "—"}</div>'
                + (f'<img src="{u}">' if u else '<div class="miss">no output</div>')
                + '</div>' for k, s, p, u, st in cell) + '</div></div>')

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
"""
    key = "".join(f"<div><b>{k}</b>{d}</div>" for k, d in READERS)
    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Scatter readers compared</title><style>{CSS}{extra}</style></head><body>
<div class="wrap"><h1>Scatter readers compared</h1>
<p class="sub">The same panels read three ways. Every view is the reader's points drawn
over the faded original — what you are judging is whether a dot sits on each symbol and
whether any symbol was missed.</p>
<div class="key">{key}</div>
{''.join(body)}
</div></body></html>"""
    path = os.path.join(a.out, "compare.html")
    with _io.open(path, "w", encoding="utf-8") as fh:
        fh.write(html)
    json.dump([{"panel": n, "counts": {k: {"series": s, "points": p}
                                       for k, s, p, _, _ in cell}} for n, _c, cell in rows],
              open(os.path.join(a.out, "compare.json"), "w"), indent=1)
    print("wrote", path)


if __name__ == "__main__":
    main()
