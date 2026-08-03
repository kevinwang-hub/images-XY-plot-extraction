"""End-to-end PXRD figure -> xy data pipeline with visual verification."""
from __future__ import annotations

import json
import os
import re
import traceback
from dataclasses import asdict

import numpy as np
import cv2

from . import axes as ax
from . import curves as cv_mod
from . import legend as lg
from . import verify as vf
from .core import load_rgb, upscale_if_small, background_color, ink_mask

WILEY = re.compile(r"^(anie|adfm|adma|chem|smll|advs|aenm)\.", re.I)


def source_ids(path: str) -> tuple[str, str]:
    """(doi, figure_label) inferred from the folder/file naming convention."""
    folder = os.path.basename(os.path.dirname(path))
    stem = os.path.splitext(os.path.basename(path))[0]
    fig = stem[len(folder) + 1:] if stem.startswith(folder) else stem
    doi = f"10.1002/{folder}" if WILEY.match(folder) else folder
    fl = fig
    if re.fullmatch(r"s\d+[a-z]?", fig, re.I):
        fl = "Figure S" + fig[1:].upper()
    elif re.fullmatch(r"f\d+[a-z]?", fig, re.I):
        fl = "Figure " + fig[1:].upper()
    return doi, fl


def digitize(path: str, outdir: str, cache_dir: str | None = None,
             save_visuals: bool = True, verbose: bool = False) -> dict:
    doi, figure_label = source_ids(path)
    name = os.path.splitext(os.path.basename(path))[0]
    rec: dict = dict(source_file=path, name=name, doi=doi, figure_label=figure_label,
                     ok=False, curves=[], warnings=[])

    rgb0 = load_rgb(path)
    rgb, scale = upscale_if_small(rgb0)
    rec["image_size"] = [int(rgb0.shape[1]), int(rgb0.shape[0])]
    rec["work_scale"] = scale

    bg = background_color(rgb)
    ink = ink_mask(rgb, bg, thresh=40)
    cache = os.path.join(cache_dir, name + ".ocr.json") if cache_dir else None
    ocr_items = ax.run_ocr(rgb, cache)
    fr = ax.detect_frame(ink, ocr_items)
    rec["frame"] = dict(left=fr.left, right=fr.right, top=fr.top, bottom=fr.bottom,
                        box=[fr.has_left, fr.has_bottom, fr.has_right, fr.has_top])
    if fr.width < 40 or fr.height < 40:
        rec["warnings"].append("plot frame not found")
        return rec

    xcal, ycal, extras = ax.calibrate_axes(rgb, ink, fr, ocr_items)
    # a tick outside the box proves that boundary is not an axis; drop it and re-calibrate
    fixed_sides = ax.reconcile_frame_with_ticks(fr, xcal, ycal, ink, ocr_items)
    if fixed_sides:
        xcal, ycal, extras = ax.calibrate_axes(rgb, ink, fr, ocr_items)
        rec["frame"] = dict(left=fr.left, right=fr.right, top=fr.top, bottom=fr.bottom,
                            box=[fr.has_left, fr.has_bottom, fr.has_right, fr.has_top])
        rec["frame_fixed_sides"] = fixed_sides
        rec["warnings"].append("frame boundary rejected by tick position: "
                               + ", ".join(fixed_sides))
    # explicit fallbacks so every downstream stage shares one pixel->value transform:
    # x in original-image pixels, y in plot-box fractions increasing upward
    if not xcal.calibrated:
        xcal.a, xcal.b = 1.0 / scale, 0.0
    if not ycal.calibrated:
        ycal.a, ycal.b = -1.0 / max(fr.height, 1), fr.bottom / max(fr.height, 1)

    # axis verification: how well the fit reproduces the *detected* tick marks
    tickres = None
    if xcal.calibrated and extras["x_ticks_detected"]:
        pred = np.array([p for p, _ in xcal.ticks])
        det = np.array(extras["x_ticks_detected"])
        tickres = float(np.median([np.min(np.abs(det - p)) for p in pred]))
    rec["axis"] = dict(
        x_axis_label=xcal.label, y_axis_label=ycal.label,
        x_calibrated=xcal.calibrated, y_calibrated=ycal.calibrated,
        x_ticks=[[p / scale, v] for p, v in xcal.ticks], x_residual=xcal.residual,
        x_n_labels=[xcal.n_labels_seen, xcal.n_labels_used],
        x_range=[float(xcal.to_data(fr.left)), float(xcal.to_data(fr.right))]
        if xcal.calibrated else None,
        y_range=[float(ycal.to_data(fr.bottom)), float(ycal.to_data(fr.top))]
        if ycal.calibrated else None,
        tick_pixel_residual=tickres, note=xcal.note,
    )
    if not xcal.calibrated:
        rec["warnings"].append("x axis not calibrated: " + (xcal.note or "?"))

    # ---- curve separation
    pink = cv_mod.plot_ink(ink, fr, extras['x_ticks_detected'], extras['y_ticks_detected'])
    lw_guess = 2.0 * max(scale, 1.0)
    data_ink, removed = cv_mod.remove_text_and_legend(pink, ocr_items, fr, lw_guess)
    labimg, clusters = cv_mod.decompose_colors(rgb, data_ink, bg)
    rec["colors"] = [dict(rgb=[int(v) for v in c.rgb], n_pixels=c.n_pixels) for c in clusters]
    curves = cv_mod.extract_curves(rgb, data_ink, fr, labimg, clusters, verbose=verbose,
                                   bg_color=bg)
    if not curves:
        rec["warnings"].append("no curve extracted")
        return rec

    tcols = cv_mod.text_columns(ocr_items, fr, ink.shape)
    tmask = cv_mod.text_mask(ocr_items, fr, ink.shape)

    entries = lg.candidate_entries(ocr_items, fr, rgb, removed, ink, lw_guess)
    lg.assign_legends(curves, entries, fr)

    # ---- extend the traces to the plot edges along their own colour, then roll back the
    # extensions that merely duplicated another curve (see resolve_extension_conflicts)
    base_snap = [(c.xs.copy(), c.ys.copy(), c.mask.copy(), c.coverage) for c in curves]
    cv_mod.extend_traces(curves, labimg, fr, ink.shape, tcols)
    reverted = cv_mod.resolve_extension_conflicts(curves, base_snap, fr)
    rec["n_extended"] = sum(1 for c in curves
                            if getattr(c, "n_added_left", 0) or getattr(c, "n_added_right", 0))
    rec["n_extension_reverted"] = len(reverted)

    # ---- read the centre line of each stroke instead of its upper edge, keeping the
    # variant that actually re-renders closer to the figure. Both variants are drawn with
    # the same pen width, so this comparison is not affected by the pen-geometry bias that
    # makes area overlap unusable for judging the extension step.
    Hp = max(fr.height, 1)
    ycal_used = ycal.calibrated

    def evaluate(cs):
        for c in cs:
            c.data_x = xcal.to_data(c.xs)
            c.data_y = ycal.to_data(c.ys)
        sm, _ = vf.roundtrip_render(cs, fr, xcal, ycal, data_ink.shape)
        out = []
        for k, c in enumerate(cs):
            pix = vf.verify_curve(c, data_ink, labimg, c.mask, fr)
            rt = vf.roundtrip_metrics(sm[k], c.mask, labimg == c.cluster, data_ink, fr,
                                      c.linewidth)
            out.append(vf.combine_metrics(pix, rt))
        return out

    mode = os.environ.get("PXRD_CENTERLINE", "auto")
    edge_snap = [(c.xs.copy(), c.ys.copy(), c.coverage) for c in curves]
    edge_m = evaluate(curves)
    cv_mod.refine_centerlines(curves, labimg, fr, ink.shape)
    ctr_m = evaluate(curves)
    n_ctr = 0
    metrics = []
    for i, c in enumerate(curves):
        keep = (mode == "force" or
                (mode != "off" and
                 ctr_m[i]["overlap_iou"] >= edge_m[i]["overlap_iou"] - 0.01))
        if keep:
            n_ctr += 1
            metrics.append(ctr_m[i])
        else:
            xs, ys, cov = edge_snap[i]
            c.xs, c.ys, c.coverage = xs.copy(), ys.copy(), cov
            metrics.append(edge_m[i])
    rec["n_centerline"] = n_ctr
    for c in curves:
        c.data_x = xcal.to_data(c.xs)
        c.data_y = ycal.to_data(c.ys)
    synth_masks, synth_all = vf.roundtrip_render(curves, fr, xcal, ycal, data_ink.shape)

    for k, c in enumerate(curves):
        m = metrics[k]
        st = vf.curve_status(m)
        xd, yd = c.data_x, c.data_y
        rec["curves"].append(dict(
            index=k, legend=c.legend, legend_source=c.legend_source,
            legend_color_de=getattr(c, "legend_color_de", None),
            color=[int(v) for v in c.rgb], linewidth=float(c.linewidth / scale),
            reward=c.reward, status=st, status_reason=vf.status_reason(m), quality=m,
            x=[float(v) for v in xd], y=[float(v) for v in yd],
            px=[float(v / scale) for v in c.xs], py=[float(v / scale) for v in c.ys],
        ))
    rec["explained_ink"] = vf.explained_fraction(curves, data_ink, data_ink.shape,
                                                 ignore=tmask)
    rec["n_curves"] = len(curves)
    rec["y_units"] = ("data units from y ticks" if ycal_used
                      else "arb. units, 0 = bottom axis, 1 = top of plot box")
    rec["figure_status"] = _figure_status(rec)
    rec["ok"] = True

    if save_visuals:
        os.makedirs(outdir, exist_ok=True)
        ov = vf.overlay_image(rgb, curves, fr, xcal, ycal)
        df = vf.diff_image(rgb, curves, data_ink, fr, ignore=tmask)
        cv2.imwrite(os.path.join(outdir, f"{name}__diff.png"),
                    cv2.cvtColor(df, cv2.COLOR_RGB2BGR))
        rec["diff"] = f"{name}__diff.png"
        rt_img = vf.roundtrip_image(rgb, synth_all, data_ink, fr)
        cv2.imwrite(os.path.join(outdir, f"{name}__roundtrip.png"),
                    cv2.cvtColor(rt_img, cv2.COLOR_RGB2BGR))
        rec["roundtrip"] = f"{name}__roundtrip.png"
        cv2.imwrite(os.path.join(outdir, f"{name}__overlay.png"),
                    cv2.cvtColor(ov, cv2.COLOR_RGB2BGR))
        rec["overlay"] = f"{name}__overlay.png"
        rp = replot_png(rec, os.path.join(outdir, f"{name}__replot.png"))
        rec["replot"] = os.path.basename(rp)
        hl = lg.highlight_visualization(rgb, curves[0], data_ink, fr)
        cv2.imwrite(os.path.join(outdir, f"{name}__highlight0.png"),
                    cv2.cvtColor(hl, cv2.COLOR_RGB2BGR))
        rec["highlight"] = f"{name}__highlight0.png"
    return rec


def _figure_status(rec: dict) -> str:
    sts = [c["status"] for c in rec["curves"]]
    if not sts:
        return "fail"
    if all(s == "pass" for s in sts) and rec["explained_ink"] >= 0.85 and rec["axis"]["x_calibrated"]:
        return "pass"
    if any(s == "fail" for s in sts) or not rec["axis"]["x_calibrated"] or rec["explained_ink"] < 0.6:
        return "review"
    return "warn"


# ------------------------------------------------------------------- re-plot

def replot_png(rec: dict, path: str) -> str:
    """Re-plot the digitised xy data — the visual half of the verifier."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axp = plt.subplots(figsize=(6.0, 4.2), dpi=130)
    for c in rec["curves"]:
        col = np.array(c["color"]) / 255.0
        if col.max() > 0.93 and col.min() > 0.93:
            col = np.array([0.2, 0.2, 0.2])
        lab = c["legend"] or f"curve {c['index'] + 1}"
        axp.plot(c["x"], c["y"], lw=1.0, color=col, label=lab)
    axp.set_xlabel(rec["axis"]["x_axis_label"] or "2θ (°)", fontsize=10)
    axp.set_ylabel(rec["axis"]["y_axis_label"] or "Intensity (a.u.)", fontsize=10)
    axp.tick_params(labelsize=9, length=3, color="#9a9a95")
    for s in ("top", "right"):
        axp.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        axp.spines[s].set_color("#c9c9c4")
    if len(rec["curves"]) >= 2:
        axp.legend(fontsize=7.5, frameon=False, loc="best")
    axp.set_title(f"{rec['name']} — digitised", fontsize=10, color="#52514e")
    fig.tight_layout()
    fig.savefig(path, transparent=False)
    plt.close(fig)
    return path


# --------------------------------------------------------------------- batch

def run_batch(image_paths, outdir: str, cache_dir: str | None = None,
              verbose: bool = False) -> list[dict]:
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(os.path.join(outdir, "data"), exist_ok=True)
    recs = []
    for i, p in enumerate(image_paths, 1):
        try:
            rec = digitize(p, os.path.join(outdir, "figs"), cache_dir, verbose=verbose)
        except Exception as exc:
            rec = dict(source_file=p, name=os.path.splitext(os.path.basename(p))[0],
                       ok=False, curves=[], warnings=["exception: " + repr(exc)],
                       traceback=traceback.format_exc(), figure_status="review")
        recs.append(rec)
        print(f"[{i}/{len(image_paths)}] {rec['name']}: {len(rec.get('curves', []))} curves"
              f" status={rec.get('figure_status')} {' '.join(rec.get('warnings', []))}")
        _write_curve_files(rec, os.path.join(outdir, "data"))
    with open(os.path.join(outdir, "records.json"), "w") as fh:
        json.dump(recs, fh)
    _write_jsonl(recs, os.path.join(outdir, "pxrd_dataset.jsonl"))
    return recs


def _write_curve_files(rec: dict, ddir: str):
    for c in rec.get("curves", []):
        tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", c["legend"])[:28] or f"curve{c['index'] + 1}"
        fn = os.path.join(ddir, f"{rec['name']}__{c['index'] + 1}_{tag}.csv")
        with open(fn, "w") as fh:
            fh.write(f"# source: {rec['source_file']}\n")
            fh.write(f"# doi: {rec.get('doi')}  figure: {rec.get('figure_label')}\n")
            fh.write(f"# legend: {c['legend']}  (match: {c['legend_source']})\n")
            fh.write(f"# x: {rec['axis']['x_axis_label']}   y: {rec.get('y_units')}\n")
            fh.write("x,y\n")
            for x, y in zip(c["x"], c["y"]):
                fh.write(f"{x:.4f},{y:.6f}\n")


def _write_jsonl(recs, path):
    with open(path, "w") as fh:
        for r in recs:
            for c in r.get("curves", []):
                fh.write(json.dumps(dict(
                    doi=r.get("doi"), figure_label=r.get("figure_label"),
                    legend=c["legend"], x_axis_label=r["axis"]["x_axis_label"],
                    y_axis_label=r["axis"]["y_axis_label"], y_units=r.get("y_units"),
                    xy_data=[[round(x, 4), round(y, 6)] for x, y in zip(c["x"], c["y"])],
                    technique="PXRD", curve_color=c["color"], quality=c["quality"],
                    status=c["status"],
                )) + "\n")
