"""PDF paper -> classified, context-resolved xy datasets.

Five stages, each one narrowing what the next has to handle:

  ingest    render every figure region, with its caption and the paragraphs citing it
  panels    find the axis systems inside each figure (also the cheapest possible filter:
            no axes, no plot)
  classify  a vision model says what each panel is and whether it can be digitised
  digitize  the pixel-level engine turns curves into numbers and verifies them
  context   a text model resolves "1a" to a compound, using the paper's own words

The division of labour is deliberate: models are asked what things *are*, pixels are
asked where things *are*. Nothing in the numeric path depends on a model reading a
coordinate off an image.
"""
from __future__ import annotations

import json
import os
import re
import time
import traceback

import cv2
import fitz
import numpy as np

from . import classify as clf
from . import llmcache
from . import context as ctx
from .ingest import extract
from .panels import crop, find_panels
from pxrd2xy.core import load_rgb
from pxrd2xy.pipeline import digitize


def paper_head(pdf_path: str, chars: int = 3000) -> str:
    """Title / abstract / opening text — where compound numbers get defined."""
    try:
        doc = fitz.open(pdf_path)
        head = doc[0].get_text()
        if len(head) < chars and len(doc) > 1:
            head += "\n" + doc[1].get_text()
        doc.close()
        return re.sub(r"\s+", " ", head)[:chars]
    except Exception:
        return ""


def process_paper(pdf_path: str, outdir: str, client=None, dpi: int = 300,
                  only: set | None = None, verbose: bool = True) -> dict:
    t0 = time.time()
    figdir = os.path.join(outdir, "figures")
    paneldir = os.path.join(outdir, "panels")
    visdir = os.path.join(outdir, "figs")
    for d in (figdir, paneldir, visdir):
        os.makedirs(d, exist_ok=True)

    paper_id = os.path.splitext(os.path.basename(pdf_path))[0]
    rec = dict(paper_id=paper_id, pdf=pdf_path, figures=[], panels=[], warnings=[])

    figures = extract(pdf_path, figdir, dpi=dpi)
    rec["n_figures"] = len(figures)

    # ---- panels
    items, meta = [], {}
    for f in figures:
        try:
            rgb = load_rgb(f.image_path)
        except Exception:
            continue
        found = find_panels(rgb)
        rec["figures"].append(dict(label=f.label, number=f.number, page=f.page + 1,
                                   caption=f.caption[:600], image=f.image_path,
                                   n_panels=len(found), n_context=len(f.context)))
        for j, p in enumerate(found):
            pid = f"{re.sub(r'[^A-Za-z0-9]+', '_', f.label)}_{j}"
            sub = crop(rgb, p)
            path = os.path.join(paneldir, f"{re.sub(r'[^A-Za-z0-9._-]+', '_', paper_id)}"
                                          f"__{pid}.png")
            cv2.imwrite(path, cv2.cvtColor(sub, cv2.COLOR_RGB2BGR))
            items.append(dict(id=pid, rgb=sub, caption=f.caption, label=f.label))
            meta[pid] = dict(fig=f, panel=p, path=path)
    rec["n_panels"] = len(items)
    if not items:
        rec["seconds"] = round(time.time() - t0, 1)
        return rec

    # ---- classify
    try:
        cls = clf.classify(items, client=client)
    except Exception as exc:
        rec["warnings"].append(f"classify failed: {exc!r}")
        cls = {}

    head = paper_head(pdf_path)
    targets = [it for it in items if clf.is_target(cls.get(it["id"], {}), only)]
    rec["n_digitizable"] = len(targets)
    if verbose:
        print(f"    {paper_id[:34]:36s} figures={len(figures):2d} panels={len(items):2d} "
              f"digitizable={len(targets):2d}")

    # ---- digitize + resolve
    for it in targets:
        pid = it["id"]
        c = cls.get(pid, {})
        m = meta[pid]
        f = m["fig"]
        entry = dict(panel_id=pid, paper_id=paper_id, figure=f.label,
                     figure_number=f.number, page=f.page + 1, panel_image=m["path"],
                     category=c.get("category"), panel_letter=c.get("panel_letter", ""),
                     x_quantity=c.get("x_quantity", ""), y_quantity=c.get("y_quantity", ""),
                     n_curves_seen=c.get("n_curves", 0),
                     render_style=c.get("render_style", ""),
                     series_labels=c.get("series_labels", []),
                     caption=f.caption[:800], curves=[])
        try:
            d = digitize(m["path"], visdir, os.path.join(outdir, ".ocrcache"),
                         rgb_in=it["rgb"], style_hint=c.get("render_style"),
                         name=f"{re.sub(r'[^A-Za-z0-9]+', '_', paper_id)}__{pid}")
        except Exception as exc:
            entry["error"] = repr(exc)
            entry["traceback"] = traceback.format_exc()[-800:]
            rec["panels"].append(entry)
            continue

        entry.update(ok=d.get("ok", False), axis=d.get("axis", {}),
                     explained_ink=d.get("explained_ink"),
                     figure_status=d.get("figure_status"),
                     y_units=d.get("y_units"), warnings=d.get("warnings", []),
                     overlay=d.get("overlay"), diff=d.get("diff"),
                     replot=d.get("replot"), roundtrip=d.get("roundtrip"))
        entry["curves"] = d.get("curves", [])

        if entry["curves"]:
            try:
                res = ctx.resolve(f.caption, f.context,
                                  c.get("series_labels", []) or
                                  [cu.get("legend", "") for cu in entry["curves"]],
                                  len(entry["curves"]), c.get("category", ""),
                                  paper_head=head, client=client)
                ctx.attach(entry["curves"], res)
                entry["material"] = res.get("material", "")
                entry["technique_detail"] = res.get("technique_detail", "")
                entry["panel_subject"] = res.get("panel_subject", "")
            except Exception as exc:
                entry.setdefault("warnings", []).append(f"context failed: {exc!r}")
        rec["panels"].append(entry)

    rec["seconds"] = round(time.time() - t0, 1)
    return rec


def run(pdf_paths: list, outdir: str, only: set | None = None) -> list:
    import anthropic
    client = anthropic.Anthropic()
    os.makedirs(outdir, exist_ok=True)
    llmcache.DIR = llmcache.DIR or os.path.join(outdir, ".llmcache")
    recs = []
    for i, p in enumerate(pdf_paths, 1):
        print(f"[{i}/{len(pdf_paths)}] {os.path.basename(p)[:52]}")
        try:
            recs.append(process_paper(p, outdir, client=client, only=only))
        except Exception as exc:
            print(f"    FAILED: {exc!r}")
            recs.append(dict(paper_id=os.path.basename(p), error=repr(exc),
                             traceback=traceback.format_exc()[-1200:],
                             figures=[], panels=[], warnings=["exception"]))
        with open(os.path.join(outdir, "records.json"), "w") as fh:
            json.dump(recs, fh)
    export(recs, outdir)
    return recs


def export(recs: list, outdir: str) -> str:
    """One JSON line per digitised curve, with provenance and quality attached."""
    path = os.path.join(outdir, "dataset.jsonl")
    datadir = os.path.join(outdir, "data")
    os.makedirs(datadir, exist_ok=True)
    n = 0
    with open(path, "w") as fh:
        for r in recs:
            for p in r.get("panels", []):
                for c in p.get("curves", []):
                    row = dict(
                        paper_id=r["paper_id"], figure=p["figure"], page=p["page"],
                        panel=p["panel_id"], panel_letter=p.get("panel_letter", ""),
                        technique=p.get("category"),
                        material=p.get("material", ""),
                        technique_detail=p.get("technique_detail", ""),
                        legend=c.get("legend", ""),
                        resolved_name=c.get("resolved_name", ""),
                        role=c.get("role", ""), conditions=c.get("conditions", ""),
                        name_confidence=c.get("name_confidence", ""),
                        name_evidence=c.get("name_evidence", ""),
                        x_axis_label=p.get("axis", {}).get("x_axis_label", ""),
                        y_axis_label=p.get("axis", {}).get("y_axis_label", ""),
                        x_calibrated=p.get("axis", {}).get("x_calibrated"),
                        y_calibrated=p.get("axis", {}).get("y_calibrated"),
                        y_units=p.get("y_units", ""),
                        status=c.get("status"), quality=c.get("quality"),
                        xy_data=[[round(x, 5), round(y, 6)]
                                 for x, y in zip(c.get("x", []), c.get("y", []))],
                    )
                    fh.write(json.dumps(row) + "\n")
                    n += 1
                    tag = re.sub(r"[^A-Za-z0-9._-]+", "_",
                                 f"{r['paper_id']}__{p['panel_id']}__{c.get('legend') or c['index']}")[:120]
                    with open(os.path.join(datadir, tag + ".csv"), "w") as cf:
                        cf.write(f"# paper: {r['paper_id']}\n# figure: {p['figure']}"
                                 f"  panel: {p['panel_id']}\n")
                        cf.write(f"# technique: {p.get('category')}  "
                                 f"material: {p.get('material','')}\n")
                        cf.write(f"# series: {c.get('resolved_name') or c.get('legend','')}"
                                 f"  ({c.get('role','')})\n")
                        cf.write(f"# x: {p.get('axis',{}).get('x_axis_label','')}"
                                 f"   y: {p.get('y_units','')}\n")
                        cf.write("x,y\n")
                        for x, y in zip(c.get("x", []), c.get("y", [])):
                            cf.write(f"{x:.5f},{y:.6f}\n")
    print(f"exported {n} curves -> {path}")
    return path
