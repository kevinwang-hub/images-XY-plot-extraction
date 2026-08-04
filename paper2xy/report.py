"""HTML dashboard for the paper pipeline: paper -> figure -> panel -> curves."""
from __future__ import annotations

import html
import os

import numpy as np

from pxrd2xy.report import CSS, JS, _data_uri, _hist, _pill, _sw


METHOD_HTML = """
<h2>What was tried, and what survived</h2>
<div class="method">
<table>
<thead><tr><th>Problem</th><th>Approaches tried</th><th>What it does now, and why</th></tr></thead>
<tbody>
<tr><td><b>Get figures out of a PDF</b></td>
<td>Extract embedded image objects (<code>page.get_images</code>); <b>MinerU</b> / layout-model extraction; render page regions</td>
<td><b>Render page regions.</b> Image-object extraction breaks two ways in this corpus: one paper stores a single figure as <b>5086 separate image fragments</b>, and other figures are pure vector with no image object at all. Rendering the region a figure occupies handles both. MinerU was not installed here, and once region rendering worked on every paper tested there was nothing left for a layout model to fix — so no extra dependency was added.</td></tr>

<tr><td><b>Split a composite figure into panels</b></td>
<td>Ask the vision model for panel boxes; detect axes geometrically</td>
<td><b>Geometry.</b> Vision models return panel coordinates that are roughly right and precisely wrong — the same failure the source paper documents for reading data points. Axes are long straight runs of ink meeting at a corner, which is trivial to detect exactly, costs nothing, and doubles as a filter: no axes, no plot, no model call.</td></tr>

<tr><td><b>Tell a plot from a molecular structure</b></td>
<td>Size/aspect heuristics; require ticks; vision classifier</td>
<td><b>Both, in that order.</b> Crystal structures are full of long straight bonds that pair into convincing right angles, so a naive axis detector reports them as plots. Requiring a row of tick strokes or tick <i>labels</i> alongside the line removes most; the vision classifier then rejected every one of the 9 that still got through in testing, and accepted all 12 real plots.</td></tr>

<tr><td><b>Say what kind of plot it is</b></td>
<td>Local <code>qwen3-vl:8b</code> / <code>gemma4:26b</code> via Ollama; <b>claude-opus-5</b></td>
<td><b>claude-opus-5</b>, with structured output so the answer is schema-checked rather than parsed out of prose. It was 21/21 correct on the first test set, which left no accuracy gap for a local model to close; the local VLMs remain the obvious swap for a cost-constrained or offline run. The model is <i>only</i> ever asked what something is — never where.</td></tr>

<tr><td><b>Turn curves into numbers</b></td>
<td>Ask a model to read values; trace pixels</td>
<td><b>Pixels.</b> Unchanged from the verified engine: OCR-calibrated axes, colour decomposition, segment graph, pen-width deconvolution, round-trip verification. Nothing in the numeric path depends on a model reading a coordinate.</td></tr>

<tr><td><b>Lines or points?</b></td>
<td>Column coverage; component shape; ink share; symbol uniformity; <b>ask the classifier</b></td>
<td><b>Measured first, and it does not separate.</b> Across this corpus every statistic that ought to tell a scatter from a stroke overlaps between them — column coverage 0.45–1.00 for points against 0.17–1.00 for lines, median thickness 4–32&nbsp;px against 4–14, symbol ink share 0.35–0.79 against 0.13–0.98. The reason is structural: a stroke crossed by other curves is cut into many small equant fragments — one line panel here yields 54 of them holding 98% of its ink — which is indistinguishable, component by component, from a row of symbols. So the question goes to the classifier, which is looking at the panel anyway and answered all six test panels correctly.</td></tr>

<tr><td><b>A plot of points</b></td>
<td>Trace a path through them; <b>one point per symbol</b></td>
<td><b>One symbol, one value, at its centre — and nothing in between.</b> A scatter states its values at the pressures that were measured and says nothing between them, so no path is fitted through the points and no line is drawn joining them, in the data or in any verification view. This is also what settles hysteresis: a loop is not a function and any single path through it must cross from one arm to the other, but as points there is nothing to cross.</td></tr>

<tr><td><b>Telling scatter series apart</b></td>
<td>Cluster the pixels; <b>group the symbols</b></td>
<td><b>Group the symbols.</b> Colour decomposition works on pixels, and on a scatter that is the wrong unit: every symbol carries a halo of blended edge pixels, so one series arrives as a saturated cluster plus a pale one — which is how a magnetic panel lost every point above 100 K to a grey "halo" cluster while its own colour kept the rest. A symbol is a better unit: it has one colour, averaged over its whole body and therefore stable, one size and one fill. Series are groups agreeing on all three, with solid and open symbols of one colour kept apart. Two centres that differ only in how far they sit along the line toward the background are the same series drawn over itself — merging those turned one blue isotherm back from six series into two.</td></tr>

<tr><td><b>Markers deleted before tracing</b></td>
<td>Text filters as written; <b>a marker is one of many</b></td>
<td><b>Three filters were eating the data.</b> The glyph filter could not tell a data marker from a stray character one at a time — but a glyph is one of a few and a marker is one of many identical, so the modal size of the equant blobs is found first and that band is exempt. OCR read a row of markers as a CJK glyph and the text filter deleted the whole high-temperature tail of a series; a text box covering three or more identical markers is now treated as a misread. And the symbol-size band was anchored to a median that anti-aliasing fragments drag down, so genuinely <i>large</i> symbols fell outside it and were discarded.</td></tr>

<tr><td><b>Judging a scatter reading</b></td>
<td>Points on ink, and point count; <b>accuracy and trend, per series</b></td>
<td><b>The first two measures were biased and I was wrong to lead with them.</b> Points-on-ink flatters the pixel reader — its points are <i>derived</i> from ink, so they sit on it almost by construction, while a model's are produced independently and judged against a harder bar. And reporting point count as though more were better inverts the truth: a scatter's data is the symbols the authors plotted, so cutting a fused run into one point per column <i>oversamples</i> it — measured here at 9× the symbol count, against the model's 0.5×. What replaced them: <b>accuracy</b>, the distance from each model point to the nearest real data point, and <b>trend</b>, the gap between the dense trace and the model's sparse points interpolated — each computed per matched series, and with both readers normalised to a fraction of the plot box, since the engine emits a 0–1 fraction wherever an axis was never calibrated (28 of these 53 panels).</td></tr>

<tr><td><b>Verifying a scatter</b></td>
<td>Match the rendered symbols; <b>a dot at each centre</b></td>
<td><b>Stop comparing shapes.</b> A scatter is re-drawn as a small dot at each centre — what was read — not as the symbols the paper used. Comparing that against the original ink measures the pen, not the reading: the centre of an open circle is background, so a perfectly read ring scored as landing off ink. Area overlap, lands-on-ink, coverage and deviation are all dropped from the gate for point series. What remains is exact and shape-independent — every point sitting on ink of its own colour — plus a deviation measured against the nearest symbol <i>centre</i> in its column, since a column can hold two symbols of one series, one on each arm of a loop, and there is no single right height.</td></tr>

<tr><td><b>Not-a-series colours</b></td>
<td>Trust the clustering; <b>test the best path</b></td>
<td><b>Test it.</b> The blend along the boundary between two overlapping curves survives colour clustering and still yields a best path, because the search returns the best path through whatever it is given. What gives it away is that its best path is <i>still</i> wild. Measured across this corpus, real series make at most one step longer than a quarter of the plot height; blend debris made thirteen.</td></tr>

<tr><td><b>Insets</b></td>
<td>Ignore them; <b>detect by their own axes</b></td>
<td><b>Detect, then drop after clustering.</b> An inset is a second plot sharing the host's colours, so nothing downstream can tell its curves from the host's — a trace walks out of the main plot into the inset with nothing looking wrong locally. What gives an inset away is that it brings its own axes: a long vertical run meeting a long horizontal run well inside the host frame, both solid and of constant thickness, enclosing a box far too small to be the host's. It must be removed <i>after</i> the colours are decided: clustering the smaller image redraws the cluster boundaries and cost a curve the inset had been helping to define.</td></tr>

<tr><td><b>Hysteresis loops</b></td>
<td>Single y(x) trace; split by symbol fill; upper/lower envelope</td>
<td><b>Both splits, in that order.</b> Two arms of a loop are each perfectly continuous, so chaining alone runs up one and back along the other — it pays the crossing once and is rewarded in every column after it. Adsorption and desorption are conventionally solid and open symbols of the same colour, so where both styles are present the arms come apart by construction; where the symbols have fused too much to tell, the arms are split by which is above the other, but only if they are apart over a sustained stretch of the axis rather than in scattered columns.</td></tr>

<tr><td><b>Name a curve labelled "1a"</b></td>
<td>Caption only; caption + citing paragraphs + paper opening</td>
<td><b>All three.</b> The compound behind "1" is defined once, in the abstract or synthesis section, and never repeated on the axes — so the caption alone cannot resolve it. Every resolution carries the quoted phrase that justifies it, and the resolver is instructed to repeat a label verbatim at low confidence rather than invent a formula.</td></tr>
</tbody></table>
<p class="note-line">Most figures in a paper are not plots — crystal structures, schemes, photographs — and a paper
whose figures are all structures correctly yields nothing. That is the expected outcome, not a failure:
the funnel below shows how many of each survive.</p>
</div>
"""

LIMITS_HTML = """
<h2>Where it still falls short</h2>
<div class="method"><table><tbody>
<tr><td><b>Crowded panel crops</b></td><td>Where two plots sit shoulder to shoulder with no
white gutter, the crop can carry a slice of its neighbour. The digitiser then reads real ink
that belongs to the other panel, and the curve is marked <b>fail</b> on coverage rather than
silently exported as good — which is the behaviour you want, but the crop is still wrong.</td></tr>
<tr><td><b>Very pale series</b></td><td>A series drawn close enough to the
background that its symbols sit near the ink threshold is still recovered only in part.</td></tr>
<tr><td><b>Legend text</b></td><td>OCR reads legend strings off the panel, and inside a busy
plot it sometimes returns a fragment (<code>200</code>) or two words fused
(<code>Openintermediate</code>). The context stage recovers the real identity from the paper
text anyway, so the resolved name is usually right even when the raw legend is not.</td></tr>
<tr><td><b>Peaks narrower than the pen</b></td><td>A reflection drawn thinner than the line
width comes back with its position and height right and its width inflated to the pen width.
Unrecoverable from the raster — the information is not in the image.</td></tr>
<tr><td><b>Dashed and dotted curves</b></td><td>Traced as the fragments they are drawn as,
not joined into one series.</td></tr>
<tr><td><b>Partial branches</b></td><td>A desorption branch that only exists over part of the
pressure range is flagged for not spanning the axis, even though that is what the figure shows.</td></tr>
</tbody></table>
<p class="note-line">Every curve below carries its own numbers and four verification views —
the crop, the trace overlaid on the original, a point-on-ink check, and the extracted data
re-plotted from scratch. Nothing is asserted that you cannot check by eye.</p></div>
"""

EXTRA_CSS2 = """
.method table{margin-top:8px}
.method td{vertical-align:top;font-size:12.5px;line-height:1.5}
.method td:first-child{white-space:nowrap;font-weight:600;color:var(--tx)}
.method td:nth-child(2){color:var(--tx3);font-size:12px}
.method code{background:var(--surf);border:1px solid var(--line);border-radius:4px;padding:0 4px}
.note-line{color:var(--tx2);font-size:12.5px;margin-top:12px}
.funnel{display:flex;gap:8px;align-items:stretch;flex-wrap:wrap;margin-top:8px}
.step{flex:1;min-width:130px;background:var(--surf);border:1px solid var(--line);
      border-radius:10px;padding:11px 13px}
.step .v{font-size:22px;font-weight:600}
.step .k{font-size:11px;color:var(--tx3);text-transform:uppercase;letter-spacing:.06em}
.step .n{font-size:11px;color:var(--tx3);margin-top:3px}
"""

EXTRA_CSS = """
.paper{border:1px solid var(--line);border-radius:12px;margin:16px 0;background:var(--surf)}
.phead{padding:12px 16px;border-bottom:1px solid var(--line);display:flex;gap:10px;
       align-items:baseline;flex-wrap:wrap}
.phead .pid{font-family:ui-monospace,Menlo,monospace;font-size:13px;font-weight:600}
.phead .meta{color:var(--tx3);font-size:12px}
.panel{padding:14px 16px;border-top:1px solid var(--line)}
.tag{display:inline-block;padding:1px 8px;border-radius:99px;font-size:11px;font-weight:600;
     background:color-mix(in srgb,var(--acc) 14%,transparent);color:var(--acc)}
.subj{color:var(--tx2);font-size:12.5px;margin:6px 0 0}
.ev{color:var(--tx3);font-size:11.5px;font-style:italic}
.cap{font-size:11.5px;color:var(--tx3);margin-top:6px}
.nores{color:var(--tx3);font-size:12.5px;padding:10px 16px}
"""


# A dozen panels can each carry five full-width views; two hundred papers cannot -- the
# page becomes tens of megabytes and stops being a web page. The views still all exist,
# just smaller, which is enough to see whether a trace sits on its curve.
MAX_W = int(os.environ.get("PAPER2XY_REPORT_W", "560"))
QUALITY = int(os.environ.get("PAPER2XY_REPORT_Q", "68"))


def _q(c, k, d=0.0):
    return (c.get("quality") or {}).get(k, d)


def build(recs: list, outdir: str, title: str = "Papers → xy data") -> str:
    papers = [r for r in recs]
    panels = [p for r in recs for p in r.get("panels", [])]
    curves = [c for p in panels for c in p.get("curves", [])]
    figs = sum(r.get("n_figures", 0) for r in recs)
    allpan = sum(r.get("n_panels", 0) for r in recs)
    ious = [_q(c, "overlap_iou") for c in curves]
    devs = [_q(c, "mean_dev_pct", 99) for c in curves]
    named = [c for c in curves if c.get("resolved_name")]
    hi = [c for c in named if c.get("name_confidence") == "high"]
    cats: dict = {}
    for p in panels:
        cats[p.get("category", "?")] = cats.get(p.get("category", "?"), 0) + 1

    def med(v):
        return float(np.median(v)) if len(v) else float("nan")

    tiles = [
        ("papers", f"{len(papers)}", f"{figs} figures found"),
        ("panels detected", f"{allpan}", "axis systems, geometric"),
        ("panels digitised", f"{len(panels)}", "classified as xy plots"),
        ("curves", f"{len(curves)}", f"{sum(_q(c,'n_points',0) for c in curves):,} points"),
        ("median line overlap", f"{med(ious):.2f}", "round-trip IoU vs original ink"),
        ("median deviation", f"{med(devs):.2f}%", "of plot height"),
        ("names resolved", f"{len(named)}/{len(curves)}",
         f"{len(hi)} high-confidence, from paper text"),
    ]
    tile_html = "".join(
        f'<div class="tile"><div class="v">{v}</div><div class="k">{k}</div>'
        f'<div class="n">{n}</div></div>' for k, v, n in tiles)
    cat_html = " ".join(
        f'<span class="tag">{html.escape(str(k))} · {v}</span>'
        for k, v in sorted(cats.items(), key=lambda kv: -kv[1]))

    # Panels classified as scatter and deliberately left undigitised. They are listed,
    # with what the classifier saw, so the run says plainly what it did not do -- and so
    # the list can be picked up later without repeating the classification.
    dfr = [d for r in recs for d in (r.get("deferred") or [])]
    deferred_html = ""
    if dfr:
        by_cat = {}
        for d in dfr:
            by_cat[d.get("category", "?")] = by_cat.get(d.get("category", "?"), 0) + 1
        rows_ = "".join(
            f'<tr><td>{html.escape(str(d.get("panel_id","")))}</td>'
            f'<td>{html.escape(str(d.get("figure","")))}</td>'
            f'<td>{html.escape(str(d.get("category","")))}</td>'
            f'<td>{html.escape(str(d.get("render_style","")))}</td>'
            f'<td>{d.get("n_curves_seen","?")}</td></tr>' for d in dfr[:400])
        deferred_html = (
            f'<h2>Deferred: {len(dfr)} scatter panels, classified but not digitised</h2>'
            '<p class="note-line">These were recognised as plots drawn with symbols and '
            'left alone on purpose. Everything cheap about them is already done — they '
            'are located, cropped, classified and counted — so the list below is a '
            'worklist, not a gap. '
            + " · ".join(f"{k} {v}" for k, v in sorted(by_cat.items(), key=lambda t: -t[1]))
            + '</p><div class="method"><table><thead><tr><th>panel</th><th>figure</th>'
            '<th>category</th><th>drawn as</th><th>series seen</th></tr></thead>'
            f'<tbody>{rows_}</tbody></table></div>')

    funnel = "".join(
        f'<div class="step"><div class="v">{v}</div><div class="k">{k}</div>'
        f'<div class="n">{n}</div></div>'
        for k, v, n in [
            ("PDFs", f"{len(papers)}", "papers read"),
            ("figures", f"{figs}", "regions rendered"),
            ("panels", f"{allpan}", "axis systems found"),
            ("plots", f"{len(panels)}", "classified digitisable"),
            ("curves", f"{len(curves)}", "traced and verified"),
            ("named", f"{len(named)}", "identity from paper text"),
        ])

    body = []
    for r in sorted(recs, key=lambda r: -len(r.get("panels", []))):
        pans = r.get("panels", [])
        head = (f'<div class="phead"><span class="pid">'
                f'{html.escape(r["paper_id"])}</span>'
                f'<span class="meta">{r.get("n_figures", 0)} figures · '
                f'{r.get("n_panels", 0)} panels · {len(pans)} digitised · '
                f'{r.get("seconds", "?")}s</span></div>')
        if not pans:
            body.append(f'<div class="paper">{head}'
                        f'<div class="nores">no digitisable xy plots found</div></div>')
            continue
        blocks = []
        for p in pans:
            views = []
            if os.path.exists(p.get("panel_image", "")):
                views.append(("panel", _data_uri(p["panel_image"], MAX_W, QUALITY),
                              "The cropped panel."))
            for key, lab, cap in [
                ("overlay", "overlay", "Digitised traces over the faded original."),
                ("diff", "point check", "Green = point on ink, red = off, blue = unclaimed."),
                ("roundtrip", "overlap map", "Re-plot vs original ink."),
                ("replot", "re-plot", "Extracted data plotted back."),
            ]:
                fp = os.path.join(outdir, "figs", p.get(key) or "")
                if p.get(key) and os.path.exists(fp):
                    views.append((lab, _data_uri(fp, MAX_W, QUALITY), cap))
            if not views:
                continue
            pid = html.escape(p["panel_id"] + r["paper_id"])[:60].replace("%", "_")
            tabs = "".join(
                f'<button class="{"on" if i == 0 else ""}" data-src="{v}" '
                f'data-cap="{html.escape(c)}" onclick="tab(\'{pid}\',0,this)">{l}</button>'
                for i, (l, v, c) in enumerate(views))
            rows = "".join(
                f'<tr><td>{_sw(c.get("color", [90, 90, 90]))}'
                f'{html.escape(c.get("legend") or "—")}</td>'
                f'<td><b>{html.escape(c.get("resolved_name") or "")}</b></td>'
                f'<td>{html.escape(c.get("role") or "")}</td>'
                f'<td>{html.escape(c.get("conditions") or "")}</td>'
                f'<td>{html.escape(c.get("name_confidence") or "")}</td>'
                f'<td>{_q(c, "overlap_iou"):.2f}</td>'
                f'<td>{_q(c, "mean_dev_pct", 0):.2f}%</td>'
                f'<td>{_q(c, "coverage", 0):.2f}</td>'
                f'<td>{_q(c, "n_points", 0)}</td>'
                f'<td>{_pill(c.get("status", "?"))}</td></tr>'
                for c in p.get("curves", []))
            ax = p.get("axis", {})
            ev = [c.get("name_evidence") for c in p.get("curves", [])
                  if c.get("name_evidence")]
            blocks.append(
                f'<div class="panel"><div class="chd">'
                f'<span class="nm mono">{html.escape(p["figure"])}'
                f'{(" (" + p["panel_letter"] + ")") if p.get("panel_letter") else ""}</span>'
                f'<span class="tag">{html.escape(str(p.get("category")))}</span>'
                f'{_pill(p.get("figure_status") or "review")}'
                f'<span class="mt">p{p.get("page")} · '
                f'{html.escape(p.get("material") or "")} '
                f'{html.escape(p.get("technique_detail") or "")}</span></div>'
                f'<div class="grid"><div class="viewer">'
                f'<div class="tabs" id="tabs-{pid}">{tabs}</div>'
                f'<img id="img-{pid}" src="{views[0][1]}" onclick="cycle(\'{pid}\')">'
                f'<div class="cap" id="cap-{pid}">{html.escape(views[0][2])}</div></div>'
                f'<div><table><thead><tr>'
                + "".join(f"<th>{h}</th>" for h in
                          ["legend", "resolved name", "role", "conditions", "conf",
                           "IoU", "dev", "cov", "n", "status"])
                + f'</tr></thead><tbody>{rows}</tbody></table>'
                f'<p class="subj">{html.escape(p.get("panel_subject") or "")}</p>'
                f'<p class="cap"><b>x</b> {html.escape(str(ax.get("x_axis_label") or "?"))}'
                f' · x-cal {"✓" if ax.get("x_calibrated") else "—"}'
                f' · <b>y</b> {html.escape(str(ax.get("y_axis_label") or "?"))}'
                f' · y-cal {"✓" if ax.get("y_calibrated") else "—"}'
                f' · {html.escape(str(p.get("y_units") or ""))}</p>'
                + (f'<p class="ev">evidence: {html.escape(ev[0][:200])}</p>' if ev else "")
                + f'<p class="cap">{html.escape((p.get("caption") or "")[:300])}</p>'
                f'</div></div></div>')
        body.append(f'<div class="paper">{head}{"".join(blocks)}</div>')

    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title><style>{CSS}{EXTRA_CSS}{EXTRA_CSS2}</style></head>
<body><div class="wrap">
<h1>{html.escape(title)}</h1>
<p class="sub">Every figure region rendered from the PDF, split into panels by axis
geometry, classified by a vision model, digitised by the pixel engine, and each curve's
identity resolved from the paper's own text. Models are asked what things <i>are</i>;
pixels are asked where things <i>are</i>. Click any image to cycle the views.</p>
<div class="tiles">{tile_html}</div>
{METHOD_HTML}
<h2>How many survive each stage</h2>
<div class="funnel">{funnel}</div>
{LIMITS_HTML}
{deferred_html}
<h2>What the panels turned out to be</h2><p>{cat_html}</p>
<h2>Verification scores</h2>
<div class="grid"><div class="tile"><div class="k">line overlap (round-trip IoU)</div>
{_hist(ious)}</div><div class="tile"><div class="k">mean deviation, % of plot height (0–3%)</div>
{_hist(devs, 0, 3, 15)}</div></div>
<h2>Papers</h2>{"".join(body)}
</div><script>{JS}</script></body></html>"""
    path = os.path.join(outdir, "report.html")
    with open(path, "w") as fh:
        fh.write(doc)
    return path
