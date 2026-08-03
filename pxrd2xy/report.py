"""Self-contained HTML dashboard: every figure, its verification visuals and metrics."""
from __future__ import annotations

import base64
import html
import json
import os

import cv2
import numpy as np

_URI_CACHE: dict = {}


def _data_uri(path: str, max_w: int = 900, quality: int = 82) -> str:
    """Inline an image as a data URI so the report is a single self-contained file.

    Referencing the PNGs on disk breaks as soon as the HTML is opened through anything
    that sandboxes local paths (IDE preview panes, a copied file, a webview), which shows
    up as broken-image placeholders.
    """
    if path in _URI_CACHE:
        return _URI_CACHE[path]
    im = cv2.imread(path)
    if im is None:
        _URI_CACHE[path] = ""
        return ""
    if im.shape[1] > max_w:
        h = max(1, int(round(im.shape[0] * max_w / im.shape[1])))
        im = cv2.resize(im, (max_w, h), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".webp", im, [cv2.IMWRITE_WEBP_QUALITY, quality])
    if not ok:
        ok, buf = cv2.imencode(".png", im)
        mime = "png"
    else:
        mime = "webp"
    uri = f"data:image/{mime};base64," + base64.b64encode(buf.tobytes()).decode()
    _URI_CACHE[path] = uri
    return uri

PALETTE = dict(s1="#2a78d6", s2="#eb6834", s3="#1baf7a", s4="#eda100", s5="#e87ba4",
               good="#1baf7a", warn="#eda100", bad="#e34948")

CSS = """
:root{color-scheme:light dark}
*{box-sizing:border-box}
body{margin:0;font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;
     background:var(--bg);color:var(--tx)}
:root{--bg:#fcfcfb;--surf:#ffffff;--tx:#0b0b0b;--tx2:#52514e;--tx3:#8a8984;--line:#e6e5e0;
      --good:#1baf7a;--warn:#eda100;--bad:#e34948;--acc:#2a78d6}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme=light])){
      --bg:#161615;--surf:#1f1f1e;--tx:#ffffff;--tx2:#c3c2b7;--tx3:#8e8d85;--line:#33332f;
      --good:#199e70;--warn:#c98500;--bad:#e66767;--acc:#3987e5}}
:root[data-theme=dark]{--bg:#161615;--surf:#1f1f1e;--tx:#fff;--tx2:#c3c2b7;--tx3:#8e8d85;
      --line:#33332f;--good:#199e70;--warn:#c98500;--bad:#e66767;--acc:#3987e5}
.wrap{max-width:1280px;margin:0 auto;padding:28px 20px 80px}
h1{font-size:22px;margin:0 0 4px;letter-spacing:-.01em}
h2{font-size:15px;margin:34px 0 12px;color:var(--tx2);font-weight:600;
   text-transform:uppercase;letter-spacing:.06em}
.sub{color:var(--tx2);margin:0 0 24px;font-size:13px}
.sub code{background:var(--surf);border:1px solid var(--line);border-radius:4px;padding:1px 5px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px}
.tile{background:var(--surf);border:1px solid var(--line);border-radius:10px;padding:13px 15px}
.tile .v{font-size:25px;font-weight:600;letter-spacing:-.02em;font-variant-numeric:tabular-nums}
.tile .k{font-size:11px;color:var(--tx3);text-transform:uppercase;letter-spacing:.07em;margin-top:3px}
.tile .n{font-size:11px;color:var(--tx3);margin-top:2px}
table{width:100%;border-collapse:collapse;font-size:13px;font-variant-numeric:tabular-nums}
th{text-align:left;font-weight:600;color:var(--tx3);font-size:11px;text-transform:uppercase;
   letter-spacing:.06em;padding:7px 8px;border-bottom:1px solid var(--line);cursor:pointer;
   user-select:none;white-space:nowrap}
th:hover{color:var(--tx)}
td{padding:6px 8px;border-bottom:1px solid var(--line)}
tr:hover td{background:color-mix(in srgb,var(--acc) 5%,transparent)}
.pill{display:inline-block;padding:1px 8px;border-radius:99px;font-size:11px;font-weight:600}
.pass{background:color-mix(in srgb,var(--good) 18%,transparent);color:var(--good)}
.warn{background:color-mix(in srgb,var(--warn) 20%,transparent);color:var(--warn)}
.review,.fail{background:color-mix(in srgb,var(--bad) 16%,transparent);color:var(--bad)}
.card{background:var(--surf);border:1px solid var(--line);border-radius:12px;padding:16px;
      margin:14px 0;scroll-margin-top:16px}
.chd{display:flex;align-items:baseline;gap:10px;flex-wrap:wrap;margin-bottom:12px}
.chd .nm{font-weight:600;font-size:15px}
.chd .mt{color:var(--tx3);font-size:12px}
.grid{display:grid;grid-template-columns:1.25fr 1fr;gap:16px;align-items:start}
@media(max-width:900px){.grid{grid-template-columns:1fr}}
.viewer{border:1px solid var(--line);border-radius:8px;overflow:hidden;background:#fff}
.viewer img{width:100%;display:block;cursor:pointer}
.tabs{display:flex;gap:2px;padding:6px;background:var(--surf);border-bottom:1px solid var(--line);flex-wrap:wrap}
.tabs button{font:inherit;font-size:11.5px;padding:3px 9px;border:1px solid transparent;
  border-radius:6px;background:transparent;color:var(--tx2);cursor:pointer}
.tabs button.on{background:color-mix(in srgb,var(--acc) 14%,transparent);color:var(--acc);
  border-color:color-mix(in srgb,var(--acc) 30%,transparent);font-weight:600}
.cap{font-size:11.5px;color:var(--tx3);padding:7px 9px;border-top:1px solid var(--line);background:var(--surf)}
.sw{display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:5px;vertical-align:middle;
    outline:1px solid rgba(0,0,0,.12)}
.bar{height:5px;border-radius:99px;background:var(--line);overflow:hidden;min-width:44px}
.bar>i{display:block;height:100%;border-radius:99px}
.legend{font-size:12px;color:var(--tx2);margin:8px 0 0}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:11.5px}
a{color:var(--acc)}
.hist{display:flex;align-items:flex-end;gap:3px;height:76px;margin-top:6px}
.hist>i{flex:1;background:var(--acc);border-radius:3px 3px 0 0;min-height:2px;opacity:.85}
.hax{display:flex;justify-content:space-between;font-size:10.5px;color:var(--tx3);margin-top:4px}
.note{font-size:12px;color:var(--tx2);background:color-mix(in srgb,var(--warn) 10%,transparent);
      border:1px solid color-mix(in srgb,var(--warn) 30%,transparent);border-radius:8px;padding:8px 11px;margin:8px 0}
"""

JS = """
function tab(fig, kind, el){
  const img = document.getElementById('img-'+fig);
  img.src = el.dataset.src;
  document.getElementById('cap-'+fig).textContent = el.dataset.cap;
  el.parentNode.querySelectorAll('button').forEach(b=>b.classList.remove('on'));
  el.classList.add('on');
}
function cycle(fig){
  const bs=[...document.querySelectorAll('#tabs-'+fig+' button')];
  const i=bs.findIndex(b=>b.classList.contains('on'));
  bs[(i+1)%bs.length].click();
}
function sortTable(th){
  const t=th.closest('table'), i=[...th.parentNode.children].indexOf(th);
  const dir = th.dataset.dir==='asc'?-1:1; th.dataset.dir = dir===1?'asc':'desc';
  const rows=[...t.tBodies[0].rows];
  rows.sort((a,b)=>{
    const x=a.cells[i].dataset.v ?? a.cells[i].textContent, y=b.cells[i].dataset.v ?? b.cells[i].textContent;
    const nx=parseFloat(x), ny=parseFloat(y);
    return (isNaN(nx)||isNaN(ny)) ? dir*String(x).localeCompare(String(y)) : dir*(nx-ny);
  });
  rows.forEach(r=>t.tBodies[0].appendChild(r));
}
"""


def _pill(s: str) -> str:
    return f'<span class="pill {s}">{s}</span>'


def _bar(v: float, color: str) -> str:
    return (f'<span class="bar"><i style="width:{max(0, min(1, v)) * 100:.0f}%;'
            f'background:{color}"></i></span>')


def _sw(rgb) -> str:
    return f'<span class="sw" style="background:rgb({rgb[0]},{rgb[1]},{rgb[2]})"></span>'


def _hist(vals, lo=0.0, hi=1.0, bins=20) -> str:
    if not vals:
        return ""
    h, _ = np.histogram(np.clip(vals, lo, hi), bins=bins, range=(lo, hi))
    mx = max(h.max(), 1)
    bars = "".join(f'<i style="height:{v / mx * 100:.0f}%" title="{v}"></i>' for v in h)
    return (f'<div class="hist">{bars}</div>'
            f'<div class="hax"><span>{lo:g}</span><span>{hi:g}</span></div>')


def build_report(recs: list[dict], outdir: str, title: str = "PXRD figure → xy digitisation") -> str:
    curves = [c for r in recs for c in r.get("curves", [])]
    ok = [r for r in recs if r.get("ok")]
    ious = [c["quality"]["overlap_iou"] for c in curves]
    devs = [c["quality"]["mean_dev_pct"] for c in curves]
    n_pass = sum(1 for c in curves if c["status"] == "pass")
    n_warn = sum(1 for c in curves if c["status"] == "warn")
    xcal = sum(1 for r in ok if r["axis"]["x_calibrated"])
    leg = sum(1 for c in curves if c["legend"])
    expl = [r["explained_ink"] for r in ok]

    def med(v):
        return float(np.median(v)) if len(v) else float("nan")

    tiles = [
        ("figures processed", f"{len(ok)}/{len(recs)}", f"{sum(len(r.get('curves', [])) for r in recs)} curves digitised"),
        ("median line overlap", f"{med(ious):.2f}", "round-trip IoU vs original ink"),
        ("median deviation", f"{med(devs):.2f}%", "of plot height (paper target &lt;1%)"),
        ("curves within 1%", f"{sum(1 for d in devs if d <= 1.0) / max(len(devs), 1) * 100:.0f}%",
         f"{n_pass} pass · {n_warn} warn · {len(curves) - n_pass - n_warn} review"),
        ("x-axis calibrated", f"{xcal}/{len(ok)}", "OCR ticks cross-checked vs tick marks"),
        ("legends matched", f"{leg}/{len(curves)}", "colour + geometry agreement"),
        ("ink explained", f"{med(expl) * 100:.0f}%", "figure-level completeness"),
    ]
    tile_html = "".join(
        f'<div class="tile"><div class="v">{v}</div><div class="k">{k}</div><div class="n">{n}</div></div>'
        for k, v, n in tiles)

    rows = []
    for r in sorted(recs, key=lambda r: (r.get("figure_status") != "review",
                                        r.get("figure_status") != "warn", r["name"])):
        q = [c["quality"] for c in r.get("curves", [])]
        st = r.get("figure_status") or "review"
        iou = med([x["overlap_iou"] for x in q]) if q else 0.0
        dev = med([x["mean_dev_pct"] for x in q]) if q else 99.0
        cal = bool(r.get("axis", {}).get("x_calibrated"))
        rows.append(
            f'<tr><td data-v="{r["name"]}"><a href="#{r["name"]}" class="mono">{r["name"]}</a></td>'
            f'<td data-v="{st}">{_pill(st)}</td>'
            f'<td data-v="{len(q)}">{len(q) or "–"}</td>'
            f'<td data-v="{iou:.3f}">{f"{iou:.2f}" if q else "–"}</td>'
            f'<td data-v="{dev:.3f}">{f"{dev:.2f}%" if q else "–"}</td>'
            f'<td data-v="{r.get("explained_ink", 0):.3f}">'
            f'{r.get("explained_ink", 0) * 100:.0f}%</td>'
            f'<td data-v="{1 if cal else 0}">{"✓" if cal else "—"}</td>'
            f'<td class="mono">'
            f'{html.escape(str(r.get("axis", {}).get("x_axis_label") or ""))[:22]}</td>'
            f'<td class="mono">{_range(r)}</td></tr>')

    table = ("<table><thead><tr>"
             + "".join(f'<th onclick="sortTable(this)">{h}</th>' for h in
                       ["figure", "status", "curves", "median IoU", "median dev",
                        "ink expl.", "x-cal", "x label", "x range"])
             + "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>")

    cards = "".join(_card(r, outdir) for r in recs)
    return _write(outdir, title, tile_html, _hist(ious), _hist(devs, 0, 3, 15), table, cards,
                  len(recs), med(ious))


def _range(r) -> str:
    ax = r.get("axis", {})
    rg = ax.get("x_range")
    return f"{rg[0]:.1f} – {rg[1]:.1f}" if rg else "—"


def _card(r: dict, outdir: str) -> str:
    name = r["name"]
    src = _data_uri(r["source_file"])
    figs = os.path.join(outdir, "figs") + os.sep
    views = [("original", src, "Source figure as published.")]
    if r.get("overlay"):
        views.append(("overlay", _data_uri(figs + r["overlay"]),
                      "Digitised traces re-drawn over the faded original; orange ticks are the "
                      "OCR tick values at the pixel positions used for calibration."))
    if r.get("diff"):
        views.append(("point check", _data_uri(figs + r["diff"]),
                      "Green = digitised point on figure ink, red = off ink, "
                      "blue = ink no curve claimed."))
    if r.get("roundtrip"):
        views.append(("overlap map", _data_uri(figs + r["roundtrip"]),
                      "Round-trip: grey = re-plotted line and original ink coincide, "
                      "red = re-plot only, blue = original only."))
    if r.get("replot"):
        views.append(("re-plot", _data_uri(figs + r["replot"]),
                      "The extracted xy data plotted back into the xy-plane."))
    if r.get("highlight"):
        views.append(("legend check", _data_uri(figs + r["highlight"]),
                      "Highlight visualisation used for legend attribution "
                      "(original | target curve emphasised)."))

    tabs = "".join(
        f'<button class="{"on" if i == 0 else ""}" data-src="{html.escape(p)}" '
        f'data-cap="{html.escape(c)}" onclick="tab(\'{name}\',0,this)">{lab}</button>'
        for i, (lab, p, c) in enumerate(views))
    img0 = views[0][1]

    body = []
    if not r.get("ok"):
        body.append(f'<div class="note"><b>not digitised</b> — {html.escape("; ".join(r.get("warnings", [])))}</div>')
    elif r.get("warnings"):
        body.append(f'<div class="note">{html.escape("; ".join(r["warnings"]))}</div>')

    ax = r.get("axis", {})
    if r.get("ok"):
        body.append(
            '<table><thead><tr>'
            + "".join(f"<th>{h}</th>" for h in ["legend", "match", "overlap IoU", "on-ink",
                                               "pt on ink", "dev", "cov", "n", "status"])
            + "</tr></thead><tbody>"
            + "".join(_crow(c) for c in r["curves"]) + "</tbody></table>")
        reasons = sorted({c.get("status_reason", "") for c in r["curves"]
                          if c["status"] != "pass"} - {"", "all checks passed"})
        if reasons:
            body.append('<p class="legend"><b>flagged because</b> '
                        + html.escape("; ".join(reasons)[:400]) + "</p>")
        body.append(
            f'<p class="legend"><b>x</b> {html.escape(str(ax.get("x_axis_label") or "?"))} '
            f'&nbsp;·&nbsp; range {_range(r)} '
            f'&nbsp;·&nbsp; {ax.get("x_n_labels", ["?", "?"])[1]} of '
            f'{ax.get("x_n_labels", ["?", "?"])[0]} OCR tick labels used, '
            f'residual {ax.get("x_residual", 0):.3g} · tick-mark offset '
            f'{ax.get("tick_pixel_residual")} px<br>'
            f'<b>y</b> {html.escape(str(ax.get("y_axis_label") or "?"))} '
            f'&nbsp;·&nbsp; {html.escape(str(r.get("y_units", "")))}</p>')

    return (f'<div class="card" id="{name}">'
            f'<div class="chd"><span class="nm mono">{name}</span>'
            f'{_pill(r.get("figure_status") or "review")}'
            f'<span class="mt">{html.escape(str(r.get("doi", "")))} · '
            f'{html.escape(str(r.get("figure_label", "")))} · '
            f'{r.get("image_size", ["?", "?"])[0]}×{r.get("image_size", ["?", "?"])[1]} px</span></div>'
            f'<div class="grid"><div class="viewer">'
            f'<div class="tabs" id="tabs-{name}">{tabs}</div>'
            f'<img id="img-{name}" src="{html.escape(img0)}" onclick="cycle(\'{name}\')" '
            f'alt="{name}" loading="lazy">'
            f'<div class="cap" id="cap-{name}">{html.escape(views[0][2])}</div>'
            f'</div><div>{"".join(body)}</div></div></div>')


def _crow(c: dict) -> str:
    q = c["quality"]
    col = PALETTE["good"] if c["status"] == "pass" else (
        PALETTE["warn"] if c["status"] == "warn" else PALETTE["bad"])
    return (f'<tr><td>{_sw(c["color"])}{html.escape(c["legend"] or "—")}</td>'
            f'<td class="mono">{c["legend_source"] or "—"}</td>'
            f'<td>{_bar(q["overlap_iou"], col)} {q["overlap_iou"]:.2f}</td>'
            f'<td>{q["on_ink"]:.2f}</td><td>{q["point_on_ink"]:.2f}</td>'
            f'<td>{q["mean_dev_pct"]:.2f}%</td><td>{q["coverage"]:.2f}</td>'
            f'<td>{q["n_points"]}</td>'
            f'<td title="{html.escape(c.get("status_reason", ""))}">{_pill(c["status"])}</td></tr>')


def _write(outdir, title, tiles, hist_iou, hist_dev, table, cards, nfig, medi) -> str:
    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title><style>{CSS}</style></head><body><div class="wrap">
<h1>{html.escape(title)}</h1>
<p class="sub">PXRD figures from <code>~/Documents/MOF/img</code> converted to numerical xy data,
each curve verified by re-plotting it and measuring the overlap with the original ink.
Pipeline after He <i>et al.</i>, <i>arXiv 2607.23886</i>: PP-OCR axis calibration →
BIRCH colour decomposition → segment-connectivity graph with a colour-consistency reward →
legend attribution → round-trip verification. Click any image to cycle the views.</p>
{f'<div class="tiles">{tiles}</div>'}
<h2>Distribution of verification scores</h2>
<div class="grid"><div class="tile"><div class="k">line overlap (round-trip IoU), per curve</div>{hist_iou}</div>
<div class="tile"><div class="k">mean deviation, % of plot height (0–3%)</div>{hist_dev}</div></div>
<h2>All figures</h2>{table}
<h2>Per-figure verification</h2>{cards}
</div><script>{JS}</script></body></html>"""
    path = os.path.join(outdir, "report.html")
    with open(path, "w") as fh:
        fh.write(doc)
    return path
