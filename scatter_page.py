#!/usr/bin/env python3
"""Build the page for the two scatter readers, over every scatter panel in the corpus."""
from __future__ import annotations

import base64
import html
import json
import os

import cv2
import numpy as np

from pxrd2xy.core import load_rgb as _load_rgb
from pxrd2xy.report import CSS


def load_rgb(path, tries: int = 4):
    """Per-file retry. This page is often built while another process is embedding
    hundreds of images from the same disk, and a single transient read failure should
    cost one retry, not the whole page."""
    import time
    for i in range(tries):
        try:
            return _load_rgb(path)
        except OSError:
            if i == tries - 1:
                raise
            time.sleep(2.0 * (i + 1))

OUT = "out_scatter_cmp"


def overlay(panel_png, pixel_curves, llm_series, path, mode, box=None):
    """Draw one reader's points over the faded panel.

    Model points that fall outside the plot box are clamped to its edge and drawn as
    rings rather than dropped: a reading with the wrong axis exponent then shows up as a
    rim of hollow markers -- visibly wrong -- instead of as an empty image, which reads
    as "produced nothing" when the model in fact produced values.
    """
    rgb = load_rgb(panel_png)
    img = (rgb.astype(np.float32) * 0.22 + 255 * 0.78).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    pal = [(200, 60, 60), (60, 60, 210), (40, 150, 60), (170, 90, 190), (30, 140, 190)]
    src = pixel_curves if mode == "pixels" else llm_series
    H, W = img.shape[:2]
    clamped = 0
    for i, (xs, ys) in enumerate(src):
        col = pal[i % len(pal)]
        for x, y in zip(xs, ys):
            inside = box is None or (box[0] - 4 <= x <= box[1] + 4
                                     and box[2] - 4 <= y <= box[3] + 4)
            if box is not None:
                x = min(max(x, box[0]), box[1])
                y = min(max(y, box[2]), box[3])
            cx, cy = int(round(x)), int(round(y))
            if not (0 <= cx < W and 0 <= cy < H):
                continue
            if inside:
                cv2.circle(img, (cx, cy), 2 if mode == "pixels" else 4, col, -1,
                           cv2.LINE_AA)
            else:
                cv2.circle(img, (cx, cy), 5, col, 2, cv2.LINE_AA)
                clamped += 1
    if clamped and box is not None:
        cv2.putText(img, f"{clamped} points outside the plot box (rings, clamped)",
                    (int(box[0]), max(14, int(box[2]) - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (30, 30, 200), 1, cv2.LINE_AA)
    cv2.imwrite(path, img)
    return clamped


def uri(path, max_w=460):
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    if im is None:
        return ""
    h, w = im.shape[:2]
    if w > max_w:
        im = cv2.resize(im, (max_w, int(h * max_w / w)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".webp", im, [cv2.IMWRITE_WEBP_QUALITY, 76])
    return "data:image/webp;base64," + base64.b64encode(buf.tobytes()).decode()


def main():
    rows = json.load(open(os.path.join(OUT, "scatter53.json")))
    prec = {os.path.basename(p.get("panel_image", ""))[:-4]: (x, p)
            for x in json.load(open("out_scatter53/records.json"))
            for p in x.get("panels", [])}
    raw = json.load(open("out_llm53.json"))
    llm = raw.get("results", raw)

    ok = [r for r in rows if np.isfinite(r["trend_pct"])]
    acc = np.array([r["accuracy_pct"] for r in rows if np.isfinite(r["accuracy_pct"])])
    tr = np.array([r["trend_pct"] for r in ok])
    band = lambda v, t: int((v <= t).sum())

    os.makedirs(os.path.join(OUT, "img"), exist_ok=True)
    cards = []
    for r in sorted(rows, key=lambda q: -(q["trend_pct"] if np.isfinite(q["trend_pct"]) else -1)):
        pid = r["panel"]
        if pid not in prec:
            continue
        _x, p = prec[pid]
        sc = p.get("work_scale") or 1.0
        pc = [(np.asarray(c["px"]), np.asarray(c["py"])) for c in p.get("curves", [])
              if "px" in c]
        d = llm.get(pid) or {}
        ax = p.get("axis") or {}
        fr = p.get("frame") or {}
        xr, yr = ax.get("x_range"), ax.get("y_range")
        if not xr and d.get("x_min") is not None:
            xr = [d["x_min"], d["x_max"]]
        if not yr and d.get("y_min") is not None:
            yr = [d["y_min"], d["y_max"]]
        ls = []
        if xr and yr and fr and xr[1] != xr[0] and yr[1] != yr[0]:
            L, R = fr["left"] / sc, fr["right"] / sc
            T, B = fr["top"] / sc, fr["bottom"] / sc
            for s in d.get("series", []):
                pts = [(q.get("x"), q.get("y")) for q in s.get("points", [])
                       if isinstance(q, dict) and q.get("x") is not None]
                if not pts:
                    continue
                a = np.asarray(pts, float)
                ls.append((L + (a[:, 0] - xr[0]) / (xr[1] - xr[0]) * (R - L),
                           B + (a[:, 1] - yr[0]) / (yr[1] - yr[0]) * (T - B)))
        box = None
        if fr:
            box = (fr["left"] / sc, fr["right"] / sc, fr["top"] / sc, fr["bottom"] / sc)
        imgs = []
        for mode, src in (("pixels", pc), ("model", ls)):
            if not src:
                imgs.append((mode, ""))
                continue
            fp = os.path.join(OUT, "img", f"{pid}__{mode}.png")
            n_out = overlay(p["panel_image"], pc, ls, fp, mode,
                            box if mode == "model" else None)
            lab = mode + (f" · {n_out} pts outside box" if n_out else "")
            imgs.append((lab, uri(fp)))
        t = r["trend_pct"]
        cls = "bad" if (np.isfinite(t) and t > 5) else ("mid" if np.isfinite(t) and t > 3 else "good")
        cards.append(
            f'<div class="cmp {cls}"><div class="hd"><b>{html.escape(pid)}</b>'
            f'<span class="m">{r.get("category","?")} · {r["symbols"]} symbols detected · '
            f'pixels {r["n_pixel"]:,} pts · model {r["n_llm"]} pts · '
            f'accuracy {r["accuracy_pct"]:.2f}% · trend '
            f'{("%.2f%%" % t) if np.isfinite(t) else "—"}</span></div>'
            '<div class="grid">' + "".join(
                f'<div class="col"><div class="lab">{m}</div>'
                + (f'<img src="{u}">' if u else '<div class="miss">no output</div>')
                + "</div>" for m, u in imgs) + "</div></div>")

    extra = """
.cmp{border:1px solid var(--line);border-radius:12px;margin:14px 0;background:var(--surf)}
.cmp.good{border-left:3px solid var(--good)}
.cmp.mid{border-left:3px solid var(--warn)}
.cmp.bad{border-left:3px solid var(--bad)}
.hd{padding:10px 14px;border-bottom:1px solid var(--line);font-size:13px}
.hd .m{color:var(--tx3);margin-left:8px;font-size:12px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:12px;padding:12px}
.lab{font-size:11.5px;font-weight:600;text-transform:uppercase;letter-spacing:.05em;margin-bottom:5px}
.col img{width:100%;border:1px solid var(--line);border-radius:8px;display:block}
.miss{color:var(--tx3);font-size:12px;padding:18px;border:1px dashed var(--line);border-radius:8px}
.note{color:var(--tx2);font-size:13px;max-width:74ch;margin:10px 0}
.note b{color:var(--tx)}
"""
    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Scatter: pixels vs model, over the whole corpus</title>
<style>{CSS}{extra}</style></head><body><div class="wrap">
<h1>Scatter: pixels vs model, over the whole corpus</h1>
<p class="sub">Every scatter panel the corpus contains, read twice — by the pixel engine and
by claude-opus-5 at high effort — and judged on two measures chosen not to presuppose that
denser is better.</p>

<h2>What is measured, and what was wrong before</h2>
<p class="note">The first comparison scored readers on <b>points landing on ink</b> and
reported <b>point count</b> as though more were better. Both were biased. Points-on-ink
flatters the pixel engine, whose points are <i>derived</i> from ink and sit on it almost by
construction, while the model's are produced independently. And a scatter's data is the
symbols the authors plotted: cutting a fused run into one point per column
<b>oversamples</b> it — 9× the symbol count here, against the model's 0.5×. Neither
matches; they miss in opposite directions.</p>
<p class="note"><b>accuracy</b> — distance from each model point to the nearest real data
point. <b>trend</b> — take the model's sparse points, join them up, and measure the dense
trace against that curve. The second is the question a sparse reading has to answer: if a
handful of points reproduces the curve, the handful is enough. Both are computed per
matched series and in fractions of the plot box, because the engine emits a 0–1 fraction
wherever an axis was never calibrated — 28 of these 53 panels.</p>

<h2>Result</h2>
<table><thead><tr><th>measure</th><th>median</th><th>75th pct</th><th>90th pct</th>
<th>within 1%</th><th>within 3%</th><th>within 5%</th></tr></thead><tbody>
<tr><td><b>accuracy</b></td><td>{np.median(acc):.2f}%</td><td>{np.percentile(acc,75):.2f}%</td>
<td>{np.percentile(acc,90):.2f}%</td><td>{band(acc,1)}/{len(acc)}</td>
<td>{band(acc,3)}/{len(acc)}</td><td>{band(acc,5)}/{len(acc)}</td></tr>
<tr><td><b>trend</b></td><td>{np.median(tr):.2f}%</td><td>{np.percentile(tr,75):.2f}%</td>
<td>{np.percentile(tr,90):.2f}%</td><td>{band(tr,1)}/{len(tr)}</td>
<td>{band(tr,3)}/{len(tr)}</td><td>{band(tr,5)}/{len(tr)}</td></tr>
</tbody></table>
<p class="note">On the typical panel the sparse reading is good: the model's points sit
{np.median(acc):.1f}% of the plot box from real data, and joining them up reproduces the
curve to within {np.median(tr):.1f}%. That is enough for a trend, and it is the case for
pooling the model's output rather than discarding it.
<b>The tail is the problem.</b> Roughly a quarter of panels are outside 5%, and a few are
wrong by orders of magnitude — all of them magnetic-susceptibility plots whose axes span
decades, where the model settled on the wrong exponent and every value went with it.
Nothing in its own output flags which reading that happened to, which is why every curve
in the pooled dataset carries a <code>method</code> field and, for model curves, its
measured accuracy and trend — so a consumer can filter rather than trust.</p>

<h2>Every panel, worst first</h2>
{''.join(cards)}
</div></body></html>"""
    path = os.path.join(OUT, "scatter.html")
    with open(path, "w") as fh:
        fh.write(doc)
    print("wrote", path, f"({os.path.getsize(path)/1e6:.1f} MB, {len(cards)} panels)")


if __name__ == "__main__":
    main()
