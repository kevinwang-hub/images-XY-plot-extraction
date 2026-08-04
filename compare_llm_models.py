#!/usr/bin/env python3
"""Read the same scatter panels with several model / effort settings and compare.

    python3 compare_llm_models.py -o out_llm

Each configuration is asked for the data points in axis units, its answer is mapped back
into the panel, and every configuration is scored the same way: how many of its points
land on ink, and how far each sits from the nearest symbol centre. The pixel reader runs
alongside as the reference, because the question is not which model is best at this but
whether any of them is close to reading the pixels.

Fable 5 has thinking permanently on -- the `thinking` parameter must be omitted, not set
to disabled -- which the reader already does, so the same call works for both families.
"""
from __future__ import annotations

import argparse
import base64
import io as _io
import json
import os

import cv2
import numpy as np

from pxrd2xy.core import load_rgb, background_color, ink_mask
from pxrd2xy.pipeline import digitize
from pxrd2xy.report import CSS
from paper2xy import llm_points, usage
from paper2xy.classify import classify

PANELS = [
    "10.1126_2Fsciadv.aaq1636__Figure_S4_0",
    "10.1126_2Fsciadv.aaq1636__Figure_S5_0",
    "10.1006_2Fjssc.2002.9584__Figure_7_0",
    "10.1039_2FC3CE41975D__Figure_9_0",
]

CONFIGS = [
    ("opus-5 · high", "claude-opus-5", "high"),
    ("opus-5 · max", "claude-opus-5", "max"),
    ("fable-5 · high", "claude-fable-5", "high"),
]


def _uri(path: str, max_w: int = 620) -> str:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return ""
    h, w = img.shape[:2]
    if w > max_w:
        img = cv2.resize(img, (max_w, int(h * max_w / w)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".webp", img, [cv2.IMWRITE_WEBP_QUALITY, 82])
    return "data:image/webp;base64," + base64.b64encode(buf.tobytes()).decode()


def scores(px, py, ink, tol: int = 2):
    """Fraction of points on ink, and the median distance to the nearest ink pixel.

    The second number is what separates "roughly right" from "right": a reading can put
    most of its points somewhere on the series and still sit consistently off the symbol
    centres, and only a distance shows that.
    """
    if len(px) == 0:
        return 0.0, float("nan")
    H, W = ink.shape
    k = np.ones((2 * tol + 1, 2 * tol + 1), np.uint8)
    grown = cv2.dilate(ink.astype(np.uint8), k).astype(bool)
    xi = np.clip(np.round(np.asarray(px)).astype(int), 0, W - 1)
    yi = np.clip(np.round(np.asarray(py)).astype(int), 0, H - 1)
    on = float(grown[yi, xi].mean())
    dt = cv2.distanceTransform((~ink).astype(np.uint8), cv2.DIST_L2, 3)
    return on, float(np.median(dt[yi, xi]))


def draw(rgb, series, path):
    out = (rgb.astype(np.float32) * 0.22 + 255 * 0.78).astype(np.uint8)
    out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    pal = [(200, 60, 60), (60, 60, 210), (40, 150, 60), (170, 90, 190), (30, 140, 190)]
    for i, (px, py) in enumerate(series):
        for x, y in zip(px, py):
            cv2.circle(out, (int(round(x)), int(round(y))), 3, pal[i % len(pal)], -1,
                       cv2.LINE_AA)
    cv2.imwrite(path, out)


def to_pixels(rec, data):
    ax, fr = rec.get("axis") or {}, rec.get("frame") or {}
    sc = rec.get("work_scale") or 1.0
    xr, yr = ax.get("x_range"), ax.get("y_range")
    if not xr and data.get("x_min") is not None:
        xr = [data["x_min"], data["x_max"]]
    if not yr and data.get("y_min") is not None:
        yr = [data["y_min"], data["y_max"]]
    if not (xr and yr and fr) or xr[1] == xr[0] or yr[1] == yr[0]:
        return None
    L, R = fr["left"] / sc, fr["right"] / sc
    T, B = fr["top"] / sc, fr["bottom"] / sc
    out = []
    for s in data.get("series", []):
        pts = [(p.get("x"), p.get("y")) for p in s.get("points", [])
               if isinstance(p, dict) and p.get("x") is not None and p.get("y") is not None]
        if not pts:
            continue
        a = np.asarray(pts, float)
        out.append((L + (a[:, 0] - xr[0]) / (xr[1] - xr[0]) * (R - L),
                    B + (a[:, 1] - yr[0]) / (yr[1] - yr[0]) * (T - B)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--panels", default="out_papers_scatter12/panels")
    ap.add_argument("-o", "--out", default="out_llm")
    ap.add_argument("--cache", default="out_papers_scatter12/.ocr")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    items = [{"id": n, "rgb": load_rgb(os.path.join(a.panels, n + ".png")),
              "caption": "", "label": n} for n in PANELS]
    cls = classify(items)

    rows = []
    for n in PANELS:
        panel = os.path.join(a.panels, n + ".png")
        rgb0 = load_rgb(panel)
        ink0 = ink_mask(rgb0, background_color(rgb0), 40)
        hint = cls.get(n, {}).get("render_style")

        sub = os.path.join(a.out, "pixels")
        os.makedirs(sub, exist_ok=True)
        rec = digitize(panel, sub, cache_dir=a.cache, save_visuals=True, name=n,
                       style_hint=hint)
        cs = [c for c in rec["curves"] if c["style"] == "markers"]
        px = np.concatenate([np.asarray(c["px"]) for c in cs]) if cs else np.zeros(0)
        py = np.concatenate([np.asarray(c["py"]) for c in cs]) if cs else np.zeros(0)
        on, dist = scores(px, py, ink0)
        cell = [("pixels (ours)", len(cs), int(len(px)), on, dist,
                 _uri(os.path.join(sub, n + "__overlay.png")))]

        for label, model, effort in CONFIGS:
            d = os.path.join(a.out, label.replace(" · ", "_").replace(" ", ""))
            os.makedirs(d, exist_ok=True)
            try:
                data = llm_points.read_points(rgb0, rec.get("axis"), model=model,
                                              effort=effort)
                mapped = to_pixels(rec, data)
            except Exception as exc:
                print(f"    {label} failed on {n}: {exc}")
                cell.append((label, 0, 0, 0.0, float("nan"), ""))
                continue
            if not mapped:
                cell.append((label, 0, 0, 0.0, float("nan"), ""))
                continue
            ov = os.path.join(d, n + "__overlay.png")
            draw(rgb0, mapped, ov)
            ax_ = np.concatenate([p for p, _q in mapped])
            ay_ = np.concatenate([q for _p, q in mapped])
            on_, dist_ = scores(ax_, ay_, ink0)
            cell.append((label, len(mapped), int(len(ax_)), on_, dist_, _uri(ov)))

        rows.append((n, cls.get(n, {}), cell))
        print(f"  {n[:42]:44s} " + "  ".join(
            f"{k}:{p}pt/ink{o:.2f}/d{dd:.1f}" for k, _s, p, o, dd, _u in cell))
        print("    spend so far: " + usage.summary())

    body = []
    for n, c, cell in rows:
        body.append(
            f'<div class="cmp"><div class="hd"><b>{n}</b> <span class="m">'
            f'{c.get("category","?")} · the model sees {c.get("n_curves","?")} series'
            '</span></div><div class="grid">' + "".join(
                f'<div class="col"><div class="lab">{k}</div>'
                f'<div class="n">{s} series · {p:,} points</div>'
                f'<div class="st"><b>{o*100:.0f}%</b> on ink · {dd:.1f} px to nearest ink</div>'
                + (f'<img src="{u}">' if u else '<div class="miss">no usable output</div>')
                + '</div>' for k, s, p, o, dd, u in cell) + '</div></div>')

    keys = ["pixels (ours)"] + [c[0] for c in CONFIGS]
    summary = ""
    for k in keys:
        got = [c for _n, _c, cell in rows for c in cell if c[0] == k]
        if not got:
            continue
        summary += (f"<tr><td><b>{k}</b></td>"
                    f"<td>{np.median([g[3] for g in got])*100:.0f}%</td>"
                    f"<td>{np.nanmedian([g[4] for g in got]):.1f} px</td>"
                    f"<td>{int(np.median([g[2] for g in got])):,}</td>"
                    f"<td>{sum(g[2] for g in got):,}</td></tr>")

    spend = "".join(f"<tr><td><b>{m}</b></td><td>{v['calls']}</td>"
                    f"<td>{v['input']:,}</td><td>{v['output']:,}</td>"
                    f"<td>${v['cost_usd']:.2f}</td></tr>"
                    for m, v in sorted(usage.by_model().items()))

    extra = """
.cmp{border:1px solid var(--line);border-radius:12px;margin:16px 0;background:var(--surf)}
.hd{padding:11px 15px;border-bottom:1px solid var(--line);font-size:13px}
.hd .m{color:var(--tx3);margin-left:8px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:12px;padding:14px}
.col{min-width:0}
.lab{font-size:12px;font-weight:600;text-transform:uppercase;letter-spacing:.05em}
.n{font-size:12.5px;color:var(--tx2);margin-top:2px}
.st{font-size:11.5px;color:var(--tx3);margin-bottom:6px}
.col img{width:100%;border:1px solid var(--line);border-radius:8px;display:block}
.miss{color:var(--tx3);font-size:12px;padding:20px;border:1px dashed var(--line);border-radius:8px}
.note{color:var(--tx3);font-size:12.5px;max-width:72ch;margin-top:10px}
"""
    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Reading a scatter: models compared</title><style>{CSS}{extra}</style></head><body>
<div class="wrap"><h1>Reading a scatter: models compared</h1>
<p class="sub">The same four panels, read by the pixel engine and by three model
configurations asked for the values directly. Every configuration gets the panel at full
resolution, the axis ranges our OCR recovered, and a schema that forces one point per
symbol.</p>
<h2>Across the {len(rows)} panels</h2>
<table><thead><tr><th>reader</th><th>median on ink</th><th>median distance to ink</th>
<th>median points</th><th>total points</th></tr></thead><tbody>{summary}</tbody></table>
<p class="note"><b>On ink</b> is the share of a reader's points landing within 2&nbsp;px of
ink. <b>Distance</b> is the median distance from a point to the nearest ink pixel — the
sharper measure, because a reading can put most of its points somewhere on the series and
still sit off the symbol centres.</p>
<h2>What this comparison cost</h2>
<table><thead><tr><th>model</th><th>calls</th><th>input</th><th>output</th><th>cost</th>
</tr></thead><tbody>{spend}</tbody></table>
{''.join(body)}
</div></body></html>"""
    path = os.path.join(a.out, "models.html")
    with _io.open(path, "w", encoding="utf-8") as fh:
        fh.write(html)
    json.dump({"panels": [{"panel": n,
                           "readers": {k: dict(series=s, points=p, on_ink=o, dist_px=dd)
                                       for k, s, p, o, dd, _u in cell}}
                          for n, _c, cell in rows],
               "usage": usage.by_model()},
              open(os.path.join(a.out, "models.json"), "w"), indent=1)
    print("wrote", path, "|", usage.summary())


if __name__ == "__main__":
    main()
