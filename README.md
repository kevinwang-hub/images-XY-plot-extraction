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
