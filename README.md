# pxrd2xy — PXRD figures → numerical xy data, with a visual verifier

Turns published powder-XRD figures into machine-readable
`(2θ, intensity)` data, and **verifies every curve** by re-plotting the extracted numbers
back into the xy-plane and measuring how well the re-plotted line overlaps the ink of the
original figure.

```bash
pip install numpy opencv-python scikit-learn scipy matplotlib rapidocr-onnxruntime

python3 run.py path/to/figures/ -o out --report   # a folder of figures + dashboard
python3 run.py path/to/one_figure.png -o out --report
open out/report.html                              # the dashboard
```

`out/` in this repository is a complete example run over 53 PXRD figures from the MOF
literature, so the dashboard and the extracted data can be inspected without re-running
anything.

## Method

The pipeline follows He *et al.*, *Harnessing X-ray Absorption Spectroscopy Data through
Multimodal Mining of Battery Literature* (arXiv 2607.23886), adapted from XANES to PXRD.
Where the paper uses a VLM for a step, this implementation uses the equivalent local model
or a geometric equivalent, so the whole thing runs offline.

| Stage | Paper | Here |
|---|---|---|
| Axis understanding | PP-OCRv5 tick labels + VLM cross-check | RapidOCR (PP-OCR ONNX) tick labels, cross-checked against **geometrically detected tick marks**; Theil–Sen fit with outlier rejection |
| Curve segmentation | BIRCH colour decomposition + line-segment connectivity graph, reward = colour consistency − λ·roughness (Eq. 1) | same, as a dynamic program over the segment DAG (`curves.py`) |
| Legend recognition | VLM on a highlight visualisation, 2 inpainting variants must agree | colour-of-key/text **and** geometric proximity, Hungarian matching; agreement of the two routes is recorded as the confidence; highlight visualisations are still generated |
| Metadata | VLM over caption + paper text | DOI/figure label from the file naming convention |
| Validation | 100 records checked by human experts | round-trip re-plot overlap per curve + the same <1 % mean / <3 % p99 deviation criterion |

After the reward DP, three more passes run — each added because it fixes a failure visible in
the overlays:

- **Colour-guided extension.** The DP walks the *connectivity* graph, so it stops wherever
  several curves merge into one blob — typically the strong low-angle peak, i.e. the part of
  a PXRD pattern you least want to lose. Colour still separates the curves inside such a
  blob, so each trace is continued column by column along runs of its own colour cluster,
  taking the run closest to where the curve already is. Columns covered by figure text are
  not crossed.
- **Conflict rollback.** Where several curves share a colour cluster (near-identical hues),
  the extension has nothing left to tell them apart and they all follow the same ink. Added
  stretches that coincide with another curve are therefore rolled back; the original
  DP-assembled parts, which were separated on real evidence, are never touched.
- **Centre-line correction.** The figure is the data *thickened by a pen*, so the data is the
  centre of the stroke, not its edge. Two situations pin that centre down: a column whose ink
  run is at most ~1.6 pen widths tall is locally flat, so the centre of the run is the centre
  line; and a *row* whose ink block is wide enough to hold k ≥ 2 pen crossings was drawn by k
  separate passes, whose centres follow from the block width — for a block of 1.5 pen widths
  the two centres sit at 1/3 and 2/3 of it. Both give real curve points, so the trace is moved
  onto them. Columns filled by a peak *narrower* than the pen carry no height information at
  all, and there the peak-preserving estimate is kept rather than replaced by a guess.
  Deciding this per column is what matters: re-deriving the whole trace from the same
  candidates (even with a continuity Viterbi) lets it wander between a peak and its baseline
  in the columns where both are consistent with the ink, which measurably destroys peak
  heights. Applied to 148 of 209 curves here; a per-curve round-trip check reverts the rest.

Details worth knowing:

- **Frame detection** accepts a long straight ink line as an axis only if it is *solid*, of
  *constant thickness*, and actually *bounds the data* (little non-text ink beyond it).
  Without those tests a tall or clipped XRD peak gets mistaken for an axis and truncates the
  plot area.
- **Tick–frame consistency** is then checked as a hard invariant: *a tick of an axis lies
  inside its own axes box*. If a tick label sits outside a detected boundary, that boundary
  cannot be an axis and is replaced by the extent of the non-text ink. This catches the case
  the shape tests cannot: several strong low-angle peaks, each clipped at the top and bottom
  of the plot, stack into a solid bar of constant width that looks exactly like a y-axis —
  and everything to its left, real data, would be silently discarded. It fires on 1 of the 53
  figures here and changes no aggregate metric, because it is a contradiction test rather
  than a threshold: the common layout, where the axis sits at the first tick, is within
  tolerance and untouched.
- **Legend candidates** must clear a minimum OCR confidence. Below it, a "legend" is usually
  the recogniser hallucinating words out of a noisy curve (a real legend entry in this set
  scores 1.00; one hallucination scored 0.52 and was being attributed to a curve).
- **Anti-aliasing** is handled by clustering only *opaque stroke cores* and then assigning
  halo pixels to the spatially nearest core. Colour clusters that are interleaved inside the
  same stroke, or that are a background-blend of another cluster, are merged — this is what
  keeps one plotted line from being split into several "curves".
- **Trace value** per column is `run_top + linewidth/2`: continuous with the run midpoint in
  smooth regions, and peak-preserving on spikes.
- **Reward penalties are scale-free** (a vertical-continuity term plus an *angle* term). Using
  raw slopes fails badly here: a near-vertical XRD spike has a slope of hundreds of
  px/column and would swamp the colour term.
- **Stacked same-colour curves** are recovered by re-running the DP after discounting already
  used segments — the common PXRD layout of several offset traces in one colour.
- **Figure furniture** (legend text, panel letters, and the key line to the left of each
  legend entry) is removed where it forms its own connected components, and excluded from
  completeness accounting where it touches a curve and cannot be removed safely.

## Verification

`out/report.html` shows, for every figure, five views (click to cycle): the original, the
traces re-drawn over it with the tick calibration used, a per-point check, the round-trip
overlap map, the re-plot in data coordinates, and the legend highlight visualisation.

Per curve:

| metric | meaning |
|---|---|
| `overlap_iou` / `precision` / `recall` | the extracted xy re-plotted through matplotlib in the source axes geometry, re-rasterised, and compared with the original ink. Not circular: it exercises the axis transform and stroke width. |
| `point_on_ink`, `colour_match` | fraction of digitised points that land on ink of the curve's own colour — catches a trace that hopped onto a neighbour |
| `mean_dev_pct`, `p99_dev_pct` | column-wise deviation as % of plot height (paper's criterion: <1 % mean, <3 % for 99 % of points), with a one-column tolerance for x-quantisation, measured only on columns where the value is *resolvable* — `steep_frac` reports how many were excluded |
| `status_reason` | which pass gate a curve missed, in plain words |
| `coverage` | fraction of the plot width spanned |
| `explained_ink` (per figure) | fraction of ink *runs* claimed by some curve — catches a missed curve |

`pass` requires point-on-ink ≥ 0.97, colour match ≥ 0.95, on-ink ≥ 0.65, mean deviation ≤ 1 %,
p99 deviation ≤ 3 %, coverage ≥ 0.85 and IoU ≥ 0.45.

Area IoU is reported but deliberately carries only a weak floor. Where the pen fills a whole
peak column, the overlap of a *line* with that filled area is capped by pen geometry, not by
digitisation error — verified by inspecting cases where a visually perfect extension scored
worse than a truncated one. Gating on it would penalise spike-dense figures for being
spike-dense, so the pass criteria rest on positional accuracy instead.

### Result on this collection

53 figures → **209 curves, 195 658 data points**, 67 pass / 126 warn / 16 review.
Median round-trip IoU **0.72**, median deviation **0.25 %** of plot height, 84 % of curves
within 1 %, 82 % spanning ≥ 90 % of the plot width. x-axis calibrated for 52/53 figures;
legends attributed for 189/209 curves; median **99.6 %** of figure data-ink explained.

**Absolute accuracy.** `tests/test_synthetic_accuracy.py` renders a figure from known xy data,
digitises it and compares against that data — the only check here that is not measured against
the (already lossy) pixels. Result on three stacked patterns with peaks from broad down to
sub-pixel: **y RMSE 1.1 %, 1.5 %, 2.0 % of the axis height**, peak positions within
**0.015–0.032°**, peak heights within **0.5 %** of the axis. The centre-line correction
accounts for roughly a fifth of that error being removed (1.40/1.93/2.49 % → 1.09/1.52/2.02 %).

Most `warn` labels are the peak-width limitation below rather than a mislocated trace: of the
flagged curves, 134 are flagged only for `p99 deviation > 3 %`, which on spike-dense figures
is dominated by flat-topped peaks. `status_reason` in the report says which gate each curve
missed.

**Known limitation.** A peak narrower than the plotted pen width is reconstructed as a
flat-topped plateau: position and height are right, the width is inflated by roughly the pen
width. This is intrinsic to a single-valued `y(x)` read off a thick stroke, and is the main
reason area-based IoU saturates near 0.7–0.9 on spike-dense figures rather than at 1.0.
Dashed and dotted curves are not traced (same limitation as the paper's algorithm).

## Outputs (`out/`)

```
report.html                  dashboard — start here (self-contained, images inlined)
pxrd_dataset.jsonl           one JSON record per curve (paper-style schema + quality block)
data/<figure>__<n>_<legend>.csv    per-curve xy data with a provenance header
records.json                 full machine-readable run log
figs/<figure>__overlay.png   traces over the faded original + tick calibration
figs/<figure>__diff.png      green = point on ink, red = off ink, blue = unclaimed run
figs/<figure>__roundtrip.png grey = re-plot and original coincide, red/blue = only one
figs/<figure>__replot.png    extracted data plotted back into the xy-plane
figs/<figure>__highlight0.png   legend highlight visualisation
```

`y` is in arbitrary units by construction (PXRD figures rarely carry numeric intensity
ticks): 0 = bottom axis, 1 = top of the plot box, increasing upward. When a figure *does*
have numeric y ticks they are used and `y_units` says so. Stacked curves keep their vertical
offsets, so a curve's own baseline is its offset.

## Source figures and rights

The figures this example run was built from are from journal articles (Wiley: *Angew. Chem.*,
*Adv. Funct. Mater.*); each record carries the `doi` and `figure_label` of its source. The raw
input figures are **not** in this repository, but `out/report.html` and
`out/figs/*__highlight0.png` embed them for verification purposes, and the numerical data in
`out/data/` and `out/pxrd_dataset.jsonl` is derived from them. Reuse of that material is
subject to the publishers' terms; cite the original papers via the recorded DOIs. The code is
independent of any particular figure set — point the CLI at your own.

## Layout

```
run.py                CLI
pxrd2xy/core.py       image loading, background/ink masks, run-length helpers
pxrd2xy/axes.py       frame detection, tick marks, OCR calibration
pxrd2xy/curves.py     colour decomposition, segment graph, reward DP, tracing
pxrd2xy/legend.py     legend attribution, highlight visualisations
pxrd2xy/verify.py     round-trip verification, metrics, overlay/diff renderers
pxrd2xy/pipeline.py   orchestration, dataset export
pxrd2xy/report.py     HTML dashboard
.ocrcache/            cached OCR per image (delete to re-run OCR)
```

Dependencies: `numpy`, `opencv-python`, `scikit-learn`, `scipy`, `matplotlib`,
`rapidocr-onnxruntime`.

Tunables via env var: `PXRD_JUMP` (vertical-continuity penalty, default 1.5),
`PXRD_KINK` (junction angle penalty, default 0.05). Raise `PXRD_JUMP` to make traces stop at
inseparable overlaps rather than chain through them; lower it to chain more aggressively.

---

# paper2xy — whole papers, end to end

`pxrd2xy` digitises a figure you already have. `paper2xy` starts from the PDF: it finds
the figures, works out which ones are data plots, digitises those, and recovers what each
curve actually *is* from the paper's own text.

```bash
set -a; source ~/Documents/api/anthropic.env; set +a     # ANTHROPIC_API_KEY
python3 run_papers.py /path/to/pdfs -n 10 -o out_papers --report
open out_papers/report.html
```

## Five stages, each narrowing the next

| Stage | What it does | Why this way |
|---|---|---|
| **ingest** | renders every figure *region* from the page, with its caption and the paragraphs citing it | Publishers slice one figure into hundreds of image fragments (5086 in one paper here) or draw it as pure vector with no image object at all. Extracting image *objects* gives confetti for some papers and nothing for others; rendering the region works for both. |
| **panels** | finds the axis systems inside each figure and crops them | Figures are composites — a crystal structure beside a diffractogram. Axes are the most geometrically distinctive thing on a page, so detecting them costs nothing and doubles as a filter: no axes, no plot, no model call. |
| **classify** | a vision model says what each panel is and whether it can be digitised | Geometry cannot tell a diffractogram from an NMR spectrum. The model is *only* ever a classifier. |
| **digitize** | the `pxrd2xy` engine turns curves into numbers and verifies them | Unchanged — round-trip verified, same metrics. |
| **context** | a text model resolves `1a` to a compound using the paper's words | Figures label curves the way authors think. The compound behind "1" is defined once, in the abstract or synthesis section, and never repeated on the axes. |

The division of labour is the point: **models are asked what things *are*, pixels are
asked where things *are*.** Nothing in the numeric path depends on a model reading a
coordinate off an image — the failure mode the source paper documents at length.

## Two filters that carry the pipeline

**An axis carries ticks.** Molecular structures are full of long straight bonds that pair
into convincing right angles, and a naive axis detector reports them as plots. Requiring a
row of tick strokes or tick *labels* alongside the line separates the two — and it costs
one array operation.

**A plot may have only one spine.** Diffractograms routinely draw the 2θ axis and nothing
else, because intensity is in arbitrary units. Panels are therefore accepted on an x axis
alone, with the plot's top found by walking up to the first sustained band of blank rows.

Adding both took panel yield on a test set from 8/48 figures to 17/48, while *removing*
the crystal-structure false positives — the vision classifier then rejected every one of
the 9 that remained, and accepted all 12 real plots.

## Tools and models actually used

| Job | What runs it | Where |
|---|---|---|
| PDF parsing, figure regions, captions, cross-references | **PyMuPDF** | local |
| Panel/axis detection, tick validation, all pixel work | **OpenCV + NumPy** | local |
| Tick labels and legend text | **RapidOCR** (PP-OCR ONNX) | local |
| Colour decomposition | **BIRCH** (scikit-learn) | local |
| "What kind of plot is this, can it be digitised" | **claude-opus-5**, structured output | API |
| "What compound is curve `1a`" | **claude-opus-5**, text only | API |

Both model calls are content-addressed on disk (`.llmcache/`, sha1 of the images and
prompt), so re-runs after an engine change cost nothing.

**MinerU** was considered for figure extraction and is not installed here; region
rendering worked on every paper in the corpus, including the scanned one, so no layout
model was added. **Local VLMs** (`qwen3-vl:8b`, `gemma4:26b` via Ollama) are the natural
substitute for the classifier if the run has to be offline or cost-bound — it is a
7-word answer against a fixed label set, not a hard vision task. They were not used here
because `claude-opus-5` was 21/21 correct on the test set, leaving no accuracy gap.

## What was tried, and what survived

| Problem | Tried | Kept, and why |
|---|---|---|
| get figures out of a PDF | embedded image objects; layout model; **render page regions** | Image objects are confetti in one paper (5086 fragments for one figure) and absent in vector-drawn ones. Region rendering handles both with no model. |
| split a composite figure | ask the vision model for boxes; **detect axes** | Vision-model coordinates are roughly right and precisely wrong. Axes are exactly detectable and double as a free filter. |
| plot vs. molecular structure | size heuristics; **ticks**; **vision classifier** | Bonds pair into convincing right angles. Ticks kill most; the classifier kills the rest. |
| lines or points? | column coverage; component shape; ink share; symbol uniformity; **ask the classifier** | Measured first, and no statistic separates them: column coverage 0.45–1.00 for points against 0.17–1.00 for lines, thickness 4–32 px against 4–14, symbol-ink share 0.35–0.79 against 0.13–0.98. A stroke crossed by other curves is cut into many small equant fragments — one line panel yields 54 of them holding 98 % of its ink — indistinguishable component-by-component from a row of symbols. So the classifier answers it. |
| which points are one series | colour-cluster the pixels; **group the symbols** | Pixels are the wrong unit: every symbol carries a halo, so one series arrives as a saturated cluster plus a pale one. A symbol has one stable colour, one size, one fill; a series is a group agreeing on all three. Shades differing only along the line to the background are merged — that turned one blue isotherm back from six series into two. |
| markers deleted before tracing | — | Three filters were eating them: the glyph filter (a glyph is one of a few, a marker one of many identical — the modal size band is now exempt), the OCR filter (a row of markers reads as text; one was recognised as a CJK glyph and the series' tail deleted), and a size band anchored to a median that fragments drag down, which discarded symbols for being too *large*. |
| a plot of points | trace a path through them; **one point per symbol** | One symbol, one value, at its centre, and nothing between. A scatter states its values where measurements were made; no path is fitted and no joining line is drawn, in the data or in any verification view. This is also what settles hysteresis — a loop is not a function, but as points there is no crossing to make. |
| symbols that have fused | treat the run as a line; **cut it back into points** | Where symbols touch, a whole series can arrive as one component; that stretch is symbols overlapping, not a line, and its values are the column centres under the run. |
| a guide line through a scatter | digitise it as a series; **drop it** | It states no value the points do not, and draws values between measurements never made. Identified by two within-panel comparisons: markedly thinner than the symbols it serves, and running along them. |
| colours that are not series | trust the clustering; **test the best chain** | The blend along the boundary between two overlapping curves still yields a best chain. Across this corpus real series make at most one step longer than a quarter of the plot height; blend debris made thirteen. |
| insets | ignore; **detect by their own axes** | An inset shares the host's colours, so nothing downstream can tell its curves from the host's. It is given away by bringing its own axes — two solid constant-thickness runs meeting well inside the host frame. Removed *after* clustering: clustering the smaller image redraws the cluster boundaries and cost a curve the inset had been helping to define. |
| hysteresis loops | single trace; **split by symbol fill**, then **by envelope** | Both arms are continuous, so chaining runs up one and back along the other. Adsorption and desorption are conventionally solid and open symbols, so where both styles are present the arms come apart by construction; otherwise they are split by which is above the other, and only where they are apart over a sustained stretch. |
| resolving `1a` | caption only; **caption + citing paragraphs + paper opening** | The compound is named once, far from the figure. Every resolution carries the quoted phrase that justifies it. |

## Awkward papers, handled

- **Scanned PDFs** (no text layer at all): the page itself becomes the figure and panel
  detection finds the plots inside. Captions are unavailable, so those curves carry no
  resolved name and are flagged low-confidence rather than guessed at.
- **Non-English captions**: `Abb.` (German) and `图` (Chinese) are recognised alongside
  `Figure` / `Fig.` / `FIG.`, as are the same forms in body-text cross-references.
- **Scrambled text layers**: some PDFs extract caption text as interleaved garbage. Those
  figures are still emitted, unlabelled, rather than dropped.

## Where it still falls short

- **Crowded panel crops.** With no white gutter between two plots the crop can carry a slice
  of its neighbour; the curve then fails on coverage rather than being exported as good.
- **Legend OCR** returns fragments inside busy plots (`200`, `Openintermediate`). The context
  stage recovers the real identity from the paper text anyway.
- **Dense fused symbols.** Where large circles overlap their neighbours, individual symbols
  cannot be recovered and neither branch split applies; the arms of such a loop are traced as
  one curve that crosses between them, and flagged `fail`.
- **Peaks narrower than the pen** come back with position and height right, width inflated.
- **Dashed curves** are traced as the fragments they are drawn as.
- **Partial branches** (a desorption arm covering half the pressure range) are flagged for not
  spanning the axis, which is what the figure actually shows.

## Outputs (`out_papers/`)

```
report.html          dashboard: paper -> figure -> panel -> curves, with every verification view
dataset.jsonl        one line per digitised curve: provenance, resolved identity, quality, xy_data
data/*.csv           per-curve xy with a provenance header
figures/  panels/    rendered figure regions and the panel crops fed to the digitiser
figs/                overlay / point-check / overlap-map / re-plot images per panel
records.json         full machine-readable run log
```

Every curve row carries `resolved_name`, `role` (experimental / simulated / calculated),
`conditions`, `name_confidence`, and `name_evidence` — the quoted phrase from the paper
that justifies the name. When the text does not define a label, the label is repeated
verbatim at low confidence; the resolver is instructed never to invent a formula.
